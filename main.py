"""Ask the Room: the whole program (spec V6). One process, five threads:

  capture     30 fps     core.capture.FrameBuffer runs its own thread
  perception  10-15 fps  wait_new -> detector.detect -> hands.update -> world.update (world logs events)
  voice       always     mic -> VAD -> whisper -> "was that for me?" -> Grok reads it -> ask -> speak + aim
                         (listen.mode: always | wake | click; the clicker means "listen now" in all three)
  net         every 5 s  NetMonitor probes; world.online follows it
  server      5 Hz       server.app.create_app(...) under uvicorn

    python main.py                  # the rig: camera, detector, servos, clicker
    python main.py --fake           # no hardware: sim camera + world, simulated laser rig, Enter to ask
    python main.py --no-voice       # dashboard and /ask only
    python main.py --camera 2

Questions are spoken. The mic is always on: Whisper transcribes what people near the table say,
voice/understand.py drops what isn't meant for the rig (never logged, the audio is only ever in
memory), and Grok (voice/understand.py, understand.backend: grok) works out what the rest meant;
offline the rule parser does (local Qwen is optional and not installed on the rig). The mic waits while the rig speaks, or it would answer itself. The
dashboard's /ask, the phone (over the BLE bridge) and texts (/sms) go through the same ask(); dashboard
and phone questions also speak and move the laser, texts only answer by text. Aiming never blocks speech: an uncalibrated laser
raises RuntimeError, which is logged, and the answer still plays.

--fake wiring (mirrors server/sim.py): server.sim.SimCamera plays the scripted tabletop story and
feeds the real World itself, so it stands in for capture + perception; the laser is act.sim.SimRig
(a FakeActuator plus its own simulated camera), calibrated at startup into a temp file. With
--video, a recording replaces the sim camera and goes through the real detector (needs ultralytics
and the model; the table falls back to frame == table when table_cal.json is missing).
"""
from __future__ import annotations

import argparse
import difflib
import logging
import os
import signal
import tempfile
import threading
import time
from collections import deque
from typing import Callable, Optional

import numpy as np
import requests

from core.config import load_config
from core.types import Answer, Intent

log = logging.getLogger("askroom.main")

OFF = {"on": False, "target": None, "err_cm": None}
NOT_HEARD = "Sorry, I didn't catch that."


class FlatTable:
    """Frame == table, for --video runs before the table is calibrated."""
    ok = True

    def __init__(self, cfg: dict):
        self.w, self.h = (cfg.get("table") or {}).get("size_cm", (90, 60))
        self.fw, self.fh = cfg.get("frame_size_px", (1280, 720))

    def px_to_cm(self, pts) -> np.ndarray:
        return np.asarray(pts, dtype=float).reshape(-1, 2) * [self.w / self.fw, self.h / self.fh]

    def cm_to_px(self, pts) -> np.ndarray:
        return np.asarray(pts, dtype=float).reshape(-1, 2) * [self.fw / self.w, self.fh / self.h]

    def calibrate(self, img) -> bool:
        return False


def laser_older_than_table(fit, table_cal_path: str) -> bool:
    """True if the table was calibrated after the laser was fitted (laser fits map table cm to pulses, so
    a new table frame shifts where they point)."""
    import json
    try:
        with open(table_cal_path) as f:
            t = float(json.load(f).get("t") or 0.0)
    except (OSError, ValueError):
        return False
    ts = float(getattr(fit, "timestamp", 0.0) or 0.0)
    return bool(t and ts and t > ts + 1.0)


def camera_source(s: str):
    """--camera: an index ('2') or a device path. Indices move when cameras are replugged; the
    /dev/v4l/by-id/ path names one camera for good (scripts/dock.sh passes /dev/v4l into the container)."""
    s = str(s).strip()
    return int(s) if s.isdigit() else s


def default_camera(cfg: dict):
    """No --camera: the rig's Brio by its stable path (config demo_check.camera) when that path exists,
    else index 0 (a laptop). An index is a guess on the rig: /dev/video2 is the Brio's infrared node."""
    path = str((cfg.get("demo_check") or {}).get("camera") or "").strip()
    if path and not path.isdigit() and os.path.exists(path):
        return path
    if path:
        log.info("camera %s not found; using camera 0", path)
    return camera_source(path) if path.isdigit() else 0


def warm_on_connect(netmon, warm: Callable[[], object]) -> None:
    """Warm the connection to Grok (core.xai.warm) now if online, and every time the network comes back:
    otherwise the first question after a start or a drop pays for the TLS handshake and can run past its
    time budget. On its own thread, so the network monitor is never held up."""
    def go() -> None:
        threading.Thread(target=warm, name="warm-grok", daemon=True).start()

    netmon.on_change(lambda online: go() if online else None)
    if netmon.online:
        go()


PHONE_ECHO_S = 3.0      # a voice question matching a phone question this recent is the same question
ANSWER_LATE_S = 10.0    # server.app.ASK_TIMEOUT_S: past it the asker was told "took too long", so stay quiet


def _norm(text: str) -> str:
    from voice.intents import normalize
    return normalize(text)


def _rules_intent(text: str, cfg: dict) -> Intent:
    from voice.intents import parse
    return parse(text, cfg)


def _rules_answer(text: str, world, events, cfg: dict) -> Answer:
    """The offline answer, from the rule parser and templates only (no model, no network): what the rig
    says when the full answer is stuck or failed. Never acts on the room (RESET, RECAL, TEACH) and
    never raises."""
    from voice.answers import answer
    from voice.llm import fallback
    try:
        intent = _rules_intent(text, cfg)
        if intent.kind in ("OTHER", "RESET", "RECAL", "TEACH"):
            return fallback()
        return answer(intent, world, events, cfg)
    except Exception:
        log.exception("rules answer failed")
        return fallback()


CUE_AFTER_VERDICT_S = 0.3      # overheard: after the model accepts, the answer gets this long before the cue


class Room:
    """Owns every part and thread. build() wires the rig or the fake; run() starts the threads."""

    def __init__(self, cfg: dict, world, events, table, frames, laser, ask: Callable[[str, str], Answer],
                 netmon=None, tts=None, stt=None, clicker=None, detector=None, hands=None,
                 interpret: Optional[Callable[[str], Intent]] = None):
        self.cfg, self.world, self.events, self.table = cfg, world, events, table
        self.frames, self.laser = frames, laser
        self.base_ask = self._router(ask)          # the care layer (attach_care) goes in front of this
        self.netmon, self.tts, self.stt, self.clicker = netmon, tts, stt, clicker
        self.detector, self.hands = detector, hands
        if interpret is None:
            from voice.understand import Understander
            interpret = Understander(dict(cfg, understand={"enabled": False}))     # rules only
        self.interpret = interpret
        m = cfg.get("main") or {}
        self.max_fps = float(m.get("perception_max_fps", 15))
        self.net_copy_s = float(m.get("net_copy_s", 5))
        self.laser_timeout_s = float(cfg.get("laser_timeout_s", 10))
        self.aim_join_s = float(cfg.get("laser_aim_join_s", 4.0))   # the mic waits this long for the aim, at most
        room = cfg.get("room") or {}
        self.room_enabled = bool(room.get("enabled", False))
        self.room_dwell_s = float(room.get("room_dwell_s", 5))
        self.room_require_zone = bool(room.get("require_zone", True))
        self.room_head_px = room.get("head_px")
        self.room_hand_s = float(room.get("hand_recent_s", 2))
        self.room_person_check = bool(room.get("person_check", True))
        self._hand_boxes: deque = deque(maxlen=64)   # (monotonic t, box_px): recent hands block room aims
        n8n = cfg.get("n8n") or {}
        self.webhook_url = str(n8n.get("webhook_url") or "")
        self.webhook_token = str(n8n.get("token") or "")
        li = cfg.get("listen") or {}
        self.listen_mode = str(li.get("mode", "always"))       # always | wake | click
        self.idle_s = float(li.get("idle_s", 8))
        self.echo_tail_s = float(li.get("echo_tail_s", 0.4))
        demo = cfg.get("demo") or {}
        self.cue_after_s = float(demo.get("thinking_cue_s", 1.0))       # 0 = off
        self.cue_phrases = list(demo.get("thinking_phrases") or ["Let me look.", "One moment.", "Hmm, let me check."])
        self._cues = 0
        vg = cfg.get("voice_guard") or {}
        self.answer_limit_s = float(vg.get("answer_limit_s", 10))   # then the rules / templates answer
        self.sorry_every_s = float(vg.get("sorry_every_s", 10))
        self.voice_restarts = 0
        self._sorry_t = float("-inf")
        self._ignored = 0
        self.stop_ev = threading.Event()
        self._acted = threading.local()            # .kind: RESET / RECAL if the router answered one
        self._turn = threading.local()             # .stale: set when the voice loop gave up on this ask thread
        self._clear_ev = threading.Event()         # RESET: the perception thread resets proposals + crops
        self._cal_warned = False
        self._aim_lock = threading.Lock()
        self._aim_gen = 0
        self._off_timer: Optional[threading.Timer] = None
        self.server = None
        self.record_answer: Optional[Callable[[str, Answer, str], None]] = None   # server log, for the phone
        self._answer_late = False                  # the last voice answer came from the rules fallback
        self._phone_qs: deque = deque(maxlen=5)    # (monotonic t, normalized text) of recent phone questions
        self.threads: list[threading.Thread] = []
        self.cleanup: list[Callable[[], None]] = []
        self.last_timing: dict = {}

    # -- asking and answering

    def _router(self, ask: Callable[[str, str], Answer]) -> Callable[[str, str], Answer]:
        """ask, noting the intent the router answered, so Room.ask acts only on a real RESET / RECAL:
        never on text the care layer handled ('remind me to reset the router') or rewrote."""
        def routed(text: str, source: str) -> Answer:
            if self._stale():                  # the voice loop already gave up on it: no TEACH binding either
                log.info("question %r timed out before the router got it; not acting on it", text)
                return _rules_answer(text, self.world, self.events, self.cfg)
            ans = ask(text, source)
            self._acted.kind = self.interpret(text).kind     # the same Intent the router used (cached)
            return ans
        return routed

    def ask(self, text: str, source: str) -> Answer:
        """The router, plus the two intents that act on the room (reset, recalibrate)."""
        self._acted.kind = None
        ans = self.base_ask(text, source)
        kind, self._acted.kind = self._acted.kind, None
        if kind in ("RESET", "RECAL") and self._stale():
            log.warning("%s for %r came after the voice loop gave up on it; not acting on it", kind, text)
            kind = None
        if kind == "RESET":
            self.world.reset()
            if self.hands is not None:
                self.hands.reset()
            self._clear_ev.set()
        elif kind == "RECAL":
            threading.Thread(target=self._recalibrate_and_tell, args=(source != "sms",), name="recal",
                             daemon=True).start()
        return ans

    def _stale(self) -> bool:
        """This thread is answering a voice question _ask_with_cue already answered from the rules."""
        ev = getattr(self._turn, "stale", None)
        return ev is not None and ev.is_set()

    def ask_and_act(self, text: str, source: str) -> Answer:
        """ask_fn for the server: dashboard questions are spoken and aimed, texts only answered. An
        answer that arrives after the server gave up (ANSWER_LATE_S) is dropped, not spoken: the asker
        already heard "that took too long", and a late laser would contradict it."""
        if source == "phone":
            self._phone_qs.append((time.monotonic(), _norm(text)))
        t0 = time.monotonic()
        ans = self.ask(text, source)
        late = time.monotonic() - t0
        if late > ANSWER_LATE_S:
            log.warning("answer to %r took %.1f s, past the server's %.0f s timeout; not speaking or aiming it",
                        text, late, ANSWER_LATE_S)
        elif source != "sms":
            self.respond(ans)
        return ans

    def _heard_from_phone(self, text: str) -> bool:
        """The mic heard a question the phone asked in the last PHONE_ECHO_S (a judge dictating next to
        the rig): answer it once, on the phone's request."""
        now, t = time.monotonic(), _norm(text)
        return any(now - t0 <= PHONE_ECHO_S and difflib.SequenceMatcher(None, t, q).ratio() >= 0.85
                   for t0, q in list(self._phone_qs))

    def respond(self, ans: Answer) -> tuple[threading.Thread, threading.Thread]:
        """Speak and aim at the same time, on two threads."""
        say = threading.Thread(target=self._speak, args=(ans.text,), name="speak", daemon=True)
        aim = threading.Thread(target=self.aim, args=(ans,), name="aim", daemon=True)
        say.start()
        aim.start()
        return say, aim

    def _speak(self, text: str) -> None:
        if self.tts is not None:
            self.tts.speak(text)
        else:
            log.info("answer (no speaker): %s", text)

    def aim(self, ans: Answer) -> Optional[float]:
        """Move the laser for an Answer. Returns the aim error in cm for a point, else None.
        Never raises: the spoken answer must play even when the laser can't."""
        action = ans.action or ("point" if ans.point_at or ans.target_cm else None)
        if action is None:
            return None
        err = None
        with self._aim_lock:
            try:
                if action.startswith("sweep:"):
                    self.laser.sweep_edge(action.split(":", 1)[1])
                elif action.startswith("room:"):
                    if not self._aim_room(action):
                        return None
                    self.world.laser = dict(self.laser.state)
                    self._schedule_off(min(self.laser_timeout_s, self.room_dwell_s))
                    return None
                else:
                    if ans.point_at is None and ans.target_cm is not None:   # visual Q&A: a raw table spot
                        pos, chain = tuple(ans.target_cm), ["table"]
                    else:
                        pos, chain = self.world.resolve(ans.point_at)
                    if pos is None:
                        log.info("no position for %s; not aiming", ans.point_at)
                        return None
                    if action == "circle":
                        self.laser.circle(pos)
                        self.laser.state["target"] = ans.point_at
                    else:
                        err = self.laser.aim_object(ans.point_at, pos)
                    log.info("laser -> %s at (%.1f, %.1f) via %s%s", ans.point_at, pos[0], pos[1],
                             ">".join(chain), f", err {err:.1f} cm" if err is not None else "")
            except (RuntimeError, ValueError) as ex:
                log.warning("laser not aimed: %s", ex)
                return None
            except Exception:
                log.exception("laser failed")
                return None
            self.world.laser = dict(self.laser.state)
            self._schedule_off()
        return err

    def _aim_room(self, action: str) -> bool:
        """'room:u,v' or 'room:u,v,x1,y1,x2,y2' (image px, optional object box): the gates, then
        Laser.aim_px (spec 0006). True if the dot is on target; otherwise the laser is off."""
        from act.room_map import beam_blocked, people_boxes_hog
        v = [float(x) for x in action.split(":", 1)[1].split(",")]
        uv, box = (v[0], v[1]), (tuple(v[2:6]) if len(v) >= 6 else None)
        rm = getattr(self.laser, "room_map", None)
        why = None
        if not self.room_enabled or rm is None:
            why = "room pointing is off or there is no room map"
        elif self.room_require_zone and rm.zone_at(uv) is None:
            why = f"({uv[0]:.0f}, {uv[1]:.0f}) is outside every room zone"
        else:
            now = time.monotonic()
            blockers = [b for t, b in list(self._hand_boxes) if now - t <= self.room_hand_s]
            f = self.frames.latest() if self.frames is not None else None
            if f is not None and f.img is not None and (f.img.shape[1], f.img.shape[0]) != rm.size_px:
                why = f"the room map was recorded at {rm.size_px[0]}x{rm.size_px[1]}; sweep again"
            elif self.room_person_check and f is not None and f.img is not None:
                people = people_boxes_hog(f.img)
                if people is None:
                    log.warning("no person detector in this OpenCV; room aim gated by hands and zones only")
                blockers += people or []
            if why is None and beam_blocked(uv, box, blockers, self.room_head_px):
                why = "a person or hand is in the way"
        if why is not None:
            log.info("room aim refused: %s", why)
            return False
        r = self.laser.aim_px(uv, box)
        log.info("laser -> room px (%.0f, %.0f): %s after %d tries, err %.1f px", uv[0], uv[1], r.reason,
                 r.tries, r.err_px)
        if not r.on_target:                   # never leave the dot somewhere it wasn't confirmed
            self.laser.off()
            self.world.laser = dict(OFF)
        return r.on_target

    def _schedule_off(self, timeout_s: Optional[float] = None) -> None:
        """Show the laser as off once the actuator's auto-off would have fired (and turn it off,
        which the fake actuators need; real ones have already done it). A shorter timeout_s (room
        aims' dwell cap) turns it off early."""
        self._aim_gen += 1
        gen = self._aim_gen
        if self._off_timer is not None:
            self._off_timer.cancel()

        def off() -> None:
            with self._aim_lock:
                if gen != self._aim_gen:
                    return
                try:
                    self.laser.off()
                except Exception:
                    log.exception("laser off failed")
                self.world.laser = dict(OFF)

        self._off_timer = threading.Timer(self.laser_timeout_s if timeout_s is None else timeout_s, off)
        self._off_timer.daemon = True
        self._off_timer.start()

    def recalibrate(self, timeout_s: Optional[float] = None) -> bool:
        """Refit the table frame from fresh frames. One-tag mode averages the tag over table_tag.frames
        consecutive frames, so this feeds new frames until it fits or timeout_s runs out."""
        f = self.frames.latest() if self.frames is not None else None
        if f is None or f.img is None:
            log.warning("recalibrate: no camera frame")
            return False
        tag = bool(getattr(self.table, "tag_mode", False))
        if timeout_s is None:
            timeout_s = max(2.0, getattr(self.table, "tag_frames", 1) / 10.0) if tag else 0.0
        before = self._frame_probe()
        deadline = time.monotonic() + timeout_s
        ok = self.table.calibrate(f.img)
        while not ok and time.monotonic() < deadline:
            f = self.frames.wait_new(f.idx, timeout=max(0.05, deadline - time.monotonic())) or f
            if f.img is not None:
                ok = self.table.calibrate(f.img)
        if not ok:
            log.warning("table recalibration failed (%s)", f"tag {self.table.tag_id} not held in view"
                        if tag else "markers 0-3 not all visible")
            return False
        after = self._frame_probe()
        moved = float(np.abs(after - before).max()) if before is not None and after is not None else 0.0
        log.info("table recalibration ok%s", f" (the table frame moved {moved:.1f} cm)" if moved >= 0.1 else "")
        fit = getattr(self.laser, "fit", None) if self.laser is not None else None
        if moved > 2.0 and fit is not None:
            if getattr(fit, "table_px_to_cm", None) is not None:
                log.warning("the table frame moved %.1f cm: laser aims are remapped through the camera, which is "
                            "right only if the camera did not move. If the camera was bumped, recalibrate the "
                            "laser (python -m act.calibrate --rig)", moved)
            else:
                log.warning("the table frame moved %.1f cm: recalibrate the laser (python -m act.calibrate --rig) "
                            "or it will point off", moved)
        return True

    def _recalibrate_and_tell(self, speak: bool) -> Optional[str]:
        """A spoken 'recalibrate': refit, then say what the person has to do about it, if anything. A
        one-tag refit that measures a new tracked area saves it to table_cal.json, but the world, laser,
        detector and answers read the size once at startup, so it only takes effect after a restart."""
        before = tuple(getattr(self.table, "size_cm", ()) or ())
        tag = bool(getattr(self.table, "tag_mode", False))
        msg = None
        if not self.recalibrate():
            msg = ("I couldn't recalibrate. Hold the table tag in view and ask again." if tag else
                   "I couldn't recalibrate. Make sure all four corner markers are in view and ask again.")
        elif tag and before and max(abs(a - b) for a, b in zip(self.table.size_cm, before)) >= 1.0:
            log.warning("tracked area changed from %.0f x %.0f to %.0f x %.0f cm; restart the app so every "
                        "part uses it", *before, *self.table.size_cm)
            msg = "Recalibrated, but the table area changed size. Restart me so I use the new size."
        if msg and speak:
            self._speak(msg)
        return msg

    def _frame_probe(self) -> Optional[np.ndarray]:
        """Where three fixed image points land on the table (cm), to tell how far a refit moved the frame."""
        if not getattr(self.table, "ok", True):
            return None
        try:
            w, h = self.cfg.get("frame_size_px", (1280, 720))
            return np.asarray(self.table.px_to_cm([[w / 4, h / 4], [3 * w / 4, h / 4], [w / 2, 3 * h / 4]]),
                              dtype=float)
        except Exception:
            return None

    # -- threads

    def perceive(self, frame):
        """One perception step: detect -> hand ids -> world.update. Returns (detections, world events),
        or None while the table is not calibrated (each frame is tried for the markers / tag until it is).
        perception_loop calls this for every new frame; eval.score_clip replays recordings through it."""
        if not self.table.ok:
            if not self._cal_warned:
                what = ("the table tag" if getattr(self.table, "tag_mode", False) else "markers 0-3")
                log.warning("table not calibrated; looking for %s every frame (python -m core.table)", what)
                self._cal_warned = True
            if self.table.calibrate(frame.img) and getattr(self.table, "tag_mode", False):
                log.warning("table calibrated: tracked area %.0f x %.0f cm; restart the app so every part "
                            "uses that size", *self.table.size_cm)
            return None
        if self._clear_ev.is_set():        # here, not in ask: the proposer isn't thread-safe
            self._clear_ev.clear()
            self.detector.reset_proposals()
        dets = self.detector.detect(frame)
        dets.hands = self.hands.update(dets.hands, dets.t)
        if self.room_enabled:
            self._hand_boxes.extend((time.monotonic(), h.box_px) for h in dets.hands)
        return dets, self.world.update(dets, frame)

    def perception_loop(self) -> None:
        last_idx, n, t_win = 0, 0, time.monotonic()
        period = 1.0 / self.max_fps if self.max_fps > 0 else 0.0
        while not self.stop_ev.is_set():
            t0 = time.monotonic()
            frame = self.frames.wait_new(last_idx, timeout=1.0)
            if frame is None:
                continue
            last_idx = frame.idx
            try:
                if self.perceive(frame) is None:
                    continue
            except Exception:
                log.exception("perception step failed")
                self.stop_ev.wait(0.1)
                continue
            n += 1
            now = time.monotonic()
            if now - t_win >= 2.0:
                self.world.fps = round(n / (now - t_win), 1)
                n, t_win = 0, now
            rest = period - (now - t0)
            if rest > 0:
                self.stop_ev.wait(rest)

    def voice_loop(self) -> None:
        """Listen and answer until stop. An error in one question is logged and met with a short
        sorry, and the loop listens again: one bad answer must not leave the rig deaf for the rest of
        the demo. (voice_watchdog restarts the loop if it dies anyway.)"""
        if self.stt is not None and self.listen_mode != "click":
            self.stt.log_text = False           # overheard chatter stays out of the logs
        while not self.stop_ev.is_set():
            try:
                self._voice_turn()
            except Exception:
                log.exception("answering failed; listening again")
                self._sorry()
                self.stop_ev.wait(0.2)

    def voice_watchdog(self, check_s: float = 1.0) -> None:
        """Runs voice_loop on its own thread and starts it again if it ever exits before stop."""
        while not self.stop_ev.is_set():
            t = threading.Thread(target=self.voice_loop, name="voice", daemon=True)
            t.start()
            while t.is_alive() and not self.stop_ev.is_set():
                t.join(check_s)
            if self.stop_ev.is_set():
                t.join(3)
                return
            self.voice_restarts += 1
            log.error("voice thread exited; restarting it (restart %d)", self.voice_restarts)
            self.stop_ev.wait(1.0)

    def _sorry(self) -> None:
        """Say a short sorry after a failed answer, at most once per sorry_every_s (never raises)."""
        now = time.monotonic()
        if now - self._sorry_t < self.sorry_every_s:
            return
        self._sorry_t = now
        try:
            self._speak("Sorry, something went wrong. Please ask again.")
        except Exception:
            log.exception("couldn't say sorry")

    def _voice_turn(self) -> None:
        """One pass of the voice loop: a clicker press, or one stretch of listening."""
        if self.listen_mode == "click":
            if self.clicker.wait_press(timeout=0.5):
                self._asked(time.monotonic())
            return
        if self.clicker is not None and self.clicker.pressed():
            self._asked(time.monotonic())       # "listen now": the next thing said is for the rig
            self._ignored = 0
            return
        if self._speaking():
            self.stop_ev.wait(0.05)
            return
        try:
            text = self.stt.hear(self.idle_s)
        except Exception:
            log.exception("listening failed")
            self.stop_ev.wait(1.0)
            return
        t_heard = time.monotonic()
        if self.stt.last_stop == "click":       # pressed while talking: that was for the rig
            if text:
                self._answer(text, t_heard, t_heard, {"mode": "asked", "ignored_since_last": self._ignored})
            else:
                self._asked(t_heard)
            self._ignored = 0
            return
        if not text:
            return
        if not self._for_rig(text):
            self._ignored += 1                  # dropped: not logged, not stored, not sent
            return
        if self._answer(text, t_heard, t_heard, {"mode": "overheard", "ignored_since_last": self._ignored}):
            self._ignored = 0
        else:
            self._ignored += 1

    def _certain(self, text: str) -> bool:
        """Overheard speech the model can't reject (Understander.certain); True without one."""
        certain = getattr(self.interpret, "certain", None)
        try:
            return True if certain is None else bool(certain(text))
        except Exception:
            log.exception("certain() failed")
            return False

    def _for_rig(self, text: str) -> bool:
        """Overheard speech that may be for the rig, by the checks that need no model (the model's
        reading, if any, runs behind the thinking cue in _ask_with_cue)."""
        screen = getattr(self.interpret, "screen", None)
        if screen is not None:
            return bool(screen(text))
        return self.interpret(text, overheard=True).kind != "IGNORE"

    def _speaking(self) -> bool:
        return bool(getattr(self.tts, "speaking", False))

    def _asked(self, t_press: float) -> None:
        """Clicker press: stop the current answer, listen for one question, answer it."""
        if self.tts is not None:
            self.tts.stop()                     # a click interrupts the previous answer
        self.clicker.clear()
        try:
            text = self.stt.listen()
        except Exception:
            log.exception("listening failed")
            self._speak("Sorry, the microphone isn't working.")
            return
        t_heard = time.monotonic()
        if not text:
            self._speak(NOT_HEARD)
            self.report({"heard": "", "answer": NOT_HEARD, "mode": "asked",
                         "record_transcribe_s": round(t_heard - t_press, 2)})
            return
        log.info("heard: %r", text)
        self._answer(text, t_press, t_heard, {"mode": "asked"})

    def _answer(self, text: str, t0: float, t_heard: float, extra: dict) -> bool:
        """Answer, speak and aim; report to n8n. Asked: t0 is the click. Overheard: the end of speech.
        Waits until the answer has been spoken plus echo_tail_s, so the mic doesn't hear the rig;
        a clicker press cuts the answer short (and is kept for the next question). False: overheard
        speech the model read as not for the rig (dropped unlogged)."""
        if self._heard_from_phone(text):
            log.info("heard %r: the phone just asked it; answered once", text)
            return True
        overheard = extra.get("mode") == "overheard"
        ans, cued = self._ask_with_cue(text, overheard=overheard)
        if ans is None:
            return False
        if overheard:
            log.info("heard: %r", text)
        if self.record_answer is not None:
            try:
                self.record_answer(text, ans, "voice")
            except Exception:
                log.exception("record_answer failed")
        t_ans = time.monotonic()
        intent = self.interpret(text) if not self._answer_late else _rules_intent(text, self.cfg)
        say, aim = self.respond(ans)
        aim.join(timeout=self.aim_join_s)
        if aim.is_alive():
            log.warning("laser still aiming after %.0f s; listening again without waiting for it", self.aim_join_s)
        self.last_timing = {"record_transcribe_s": round(t_heard - t0, 2),
                            "ask_s": round(t_ans - t_heard, 2),
                            ("speech_end_to_laser_s" if overheard else "click_to_laser_s"):
                                round(time.monotonic() - t0, 2),
                            "stt_ms": dict(getattr(self.stt, "last_ms", {}))}
        log.info("timing %s", self.last_timing)
        self.report({"heard": text, "intent": intent.kind, "object": intent.obj,
                     "understood_by": getattr(self.interpret, "last_by", "rules"),
                     "qwen_ms": round(getattr(self.interpret, "last_ms", 0.0)),
                     "answer": ans.text, "point_at": ans.point_at, "action": ans.action,
                     "laser_err_cm": (self.world.laser or {}).get("err_cm"),
                     "online": bool(self.world.online), "thinking_cue": cued, **extra, **self.last_timing})
        if self.listen_mode == "click":
            return True
        while say.is_alive() and not self.stop_ev.is_set():
            if self.clicker is not None and self.clicker.wait_press(timeout=0.05):
                self.tts.stop()
                self.clicker.press()            # handled by the loop as "listen now"
                break
            elif self.clicker is None:
                say.join(0.05)
        self.stop_ev.wait(self.echo_tail_s)
        return True

    def _ask_with_cue(self, text: str, overheard: bool = False) -> tuple[Optional[Answer], bool]:
        """ask(text, "voice") on its own thread; if no answer within demo.thinking_cue_s (a model is
        reading it, or Grok is answering), say a short "let me look" so the rig doesn't sit silent.
        overheard: the model's reading of overheard speech runs here too. The cue starts at once when the
        speech is surely for the rig (Understander.certain: wake word, an object named); otherwise only once
        the model accepted it, so chatter it rejects gets no "let me look" either. IGNORE returns
        (None, cued). No answer within voice_guard.answer_limit_s (Wi-Fi stalled) or an error: the rules
        and templates answer instead, and the late answer may no longer act on the room (Room.ask).
        Returns (answer, cued). The TTS lock queues the answer behind the cue."""
        box: dict = {}
        done, accepted, stale = threading.Event(), threading.Event(), threading.Event()

        def work() -> None:
            self._turn.stale = stale
            try:
                if overheard and self.interpret(text, overheard=True).kind == "IGNORE":
                    box["ignore"] = True
                    return
                accepted.set()
                box["ans"] = self.ask(text, "voice")
            except Exception as ex:
                box["err"] = ex
                log.exception("answering %r failed; the rules answer", text)
            finally:
                done.set()

        t0 = time.monotonic()
        left = lambda: max(0.0, self.answer_limit_s - (time.monotonic() - t0))   # noqa: E731
        self._answer_late = False
        threading.Thread(target=work, name="ask", daemon=True).start()
        cued = False
        if self.cue_after_s > 0 and self.tts is not None and self.cue_phrases:
            wait_s = self.cue_after_s
            if overheard and not self._certain(text):
                while not (accepted.is_set() or done.is_set()) and left() > 0:
                    done.wait(0.02)            # the model's verdict first
                wait_s = max(CUE_AFTER_VERDICT_S, self.cue_after_s - (time.monotonic() - t0))
            cued = not done.wait(min(wait_s, left()))
            if cued and not self.stop_ev.is_set():
                phrase = self.cue_phrases[self._cues % len(self.cue_phrases)]
                self._cues += 1
                threading.Thread(target=self._speak, args=(phrase,), name="cue", daemon=True).start()
        if not done.wait(left()):
            stale.set()                        # the ask thread must not act on it when it finally returns
            log.warning("no answer to %r after %.0f s; the rules answer", text, self.answer_limit_s)
            self._answer_late = True
            return _rules_answer(text, self.world, self.events, self.cfg), cued
        if box.get("ignore"):
            return None, cued
        if "err" in box:
            self._answer_late = True
            return _rules_answer(text, self.world, self.events, self.cfg), cued
        return box["ans"], cued

    def report(self, question: dict) -> None:
        """Send one spoken question to the n8n workflow (n8n.webhook_url), in the background so it
        never delays an answer. Empty URL: off."""
        if not self.webhook_url:
            return

        def post() -> None:
            try:
                requests.post(self.webhook_url, json=dict(question, t=time.time()), timeout=3,
                              headers={"x-askroom-token": self.webhook_token} if self.webhook_token else None)
            except requests.RequestException as ex:
                log.debug("n8n webhook: %s", ex)

        threading.Thread(target=post, name="report", daemon=True).start()

    def net_loop(self) -> None:
        while not self.stop_ev.is_set():
            self.world.online = bool(self.netmon.online)
            self.stop_ev.wait(self.net_copy_s)

    def _thread(self, target, name: str) -> None:
        t = threading.Thread(target=target, name=name, daemon=True)
        t.start()
        self.threads.append(t)

    def start_server(self, host: str, port: int) -> None:
        import uvicorn

        from server.app import create_app
        app = create_app(self.cfg, self.world, self.events, frames=self.frames,
                         ask_fn=self.ask_and_act, table=self.table, care=getattr(self, "care", None))
        self.record_answer = app.state.record_answer
        self.server = uvicorn.Server(uvicorn.Config(app, host=host, port=port, log_level="warning",
                                                    timeout_graceful_shutdown=2))
        self.server.install_signal_handlers = lambda: None     # the main thread handles Ctrl-C
        self._thread(self.server.run, "server")
        log.info("dashboard on http://localhost:%s", port)

    def run(self, host: str, port: int, voice: bool = True, perception: bool = True) -> None:
        if self.netmon is not None:
            self.netmon.on_change(lambda online: setattr(self.world, "online", online))
            self._thread(self.net_loop, "net")
        if perception:
            self._thread(self.perception_loop, "perception")
        if voice:
            self._thread(self.voice_watchdog, "voice-watchdog")   # runs and restarts voice_loop
            how = "press the clicker%s" % (" (or Enter)" if self.clicker.kind == "keyboard" else "")
            log.info("%s", {"click": f"{how} to ask a question",
                            "wake": f"listening for \"room, ...\"; or {how}",
                            }.get(self.listen_mode, f"listening; ask out loud, or {how}"))
        if getattr(self, "care", None) is not None:
            self.care.start()                                  # care scheduler: reminders, morning report
        self.start_server(host, port)

    def shutdown(self) -> None:
        self.stop_ev.set()
        if self._off_timer is not None:
            self._off_timer.cancel()
        if self.server is not None:
            self.server.should_exit = True
        for fn in [self._laser_off, *reversed(self.cleanup)]:
            try:
                fn()
            except Exception:
                log.exception("shutdown step failed")
        for t in self.threads:
            t.join(timeout=3)

    def _laser_off(self) -> None:
        self.laser.off()
        self.world.laser = dict(OFF)


# ---------------------------------------------------------------- construction

def clock_files(cfg: dict) -> list[str]:
    """Files whose mtimes the wall clock can't be behind: the last run's event DB, calibrations, config."""
    paths = cfg.get("paths") or {}
    return [paths.get("events_db", ""), paths.get("table_cal", ""), paths.get("laser_cal", ""),
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml")]


def warn_if_clock_behind(cfg: dict) -> Optional[float]:
    import net
    behind = net.clock_behind(clock_files(cfg))
    if behind is not None:
        log.warning("the clock is %.0f min behind the last saved file: spoken times and the n8n log will be "
                    "wrong until it is set. Join the phone hotspot so NTP sets it (timedatectl), or "
                    "`sudo date -s` on the Jetson", behind / 60)
    return behind


def make_laser(cfg: dict, frames, table):
    """The rig's Laser. An actuator that can't start (e.g. adafruit_servokit missing) doesn't stop the app:
    the laser is disabled, logged loudly, and answers are spoken only."""
    import act.actuator
    import act.laser
    actuator, why = act.actuator.make_actuator_or_fake(cfg)   # cfg["actuator"]: fake | pca9685 | serial
    if why is None and str(cfg.get("actuator", "fake")).lower() == "fake":
        log.warning("actuator is 'fake': the servos will not move. On the rig set `actuator: pca9685` "
                    "(or serial/bus) in config.local.yaml")
    laser = act.laser.Laser(actuator, frames, table, cfg["paths"]["laser_cal"], cfg=cfg)
    laser.disabled = why
    if why is not None:
        pass                                            # make_actuator_or_fake logged it
    elif laser.fit is None:
        log.warning("laser not calibrated (%s missing); answers will be spoken only. Run "
                    "python -m act.calibrate --rig", cfg["paths"]["laser_cal"])
    elif laser.fit.table_px_to_cm is None:
        if laser_older_than_table(laser.fit, getattr(table, "cal_path", "")):
            log.warning("laser_cal.json was fitted before the last table calibration: the laser may point "
                        "off; recalibrate it (python -m act.calibrate --rig)")
    else:
        moved = laser.refit_moved_cm()
        if moved is not None and moved > 0.5:
            log.warning("table refitted since the laser fit (frame moved %.1f cm): aims are remapped through the "
                        "camera, which is right only if the camera did not move. If it was bumped, recalibrate "
                        "the laser (python -m act.calibrate --rig)", moved)
    return laser


def build(cfg: dict, fake: bool = False, camera: int = 0, with_voice: bool = True,
          video: Optional[str] = None, keyboard: Optional[bool] = None) -> tuple[Room, bool]:
    """Construct everything. Returns (room, needs_perception_thread)."""
    import core.embed
    import core.events
    import core.world
    import net
    import voice.pipeline
    import voice.tts
    import voice.understand

    cleanup: list[Callable[[], None]] = []
    if not fake:
        import core.table
        core.table.apply_saved_size(cfg)            # one-tag mode: the saved tracked area, before anything reads it
        import core.table_area
        core.table_area.apply_saved_area(cfg)       # the tabletop outline (python -m core.table --outline), if still valid
        warn_if_clock_behind(cfg)
    if fake:
        snap = tempfile.mkdtemp(prefix="askroom_fake_snaps_")
        events = core.events.EventLog(":memory:", snap)
    else:
        events = core.events.EventLog(cfg["paths"]["events_db"], cfg["paths"]["snapshots"])
    cleanup.append(events.close)
    world = core.world.World(cfg, events, embed=core.embed.make_embedder(cfg))   # None unless reid.enabled

    detector = hands = None
    perception = True
    if fake and not video:
        from server.sim import SimCamera
        frames = SimCamera(cfg, world).start()          # plays the story and updates the world
        cleanup.append(frames.stop)
        table = frames.painter.table
        table.ok = True
        table.calibrate = lambda img: False
        perception = False
    else:
        import core.capture
        import core.detect
        import core.hands
        import core.table
        if video:
            frames = core.capture.VideoFileSource(video, loop=True)
        else:
            frames = core.capture.FrameBuffer(camera)
        cleanup.append(frames.stop)
        table = core.table.Table(cfg)                  # loads table_cal.json; table.ok says if calibrated
        # (one-tag mode: the saved tracked-area size went into cfg at the top of build())
        if video and not table.ok:
            log.warning("table not calibrated; using frame == table for the recording")
            table = FlatTable(cfg)
        detector = core.detect.Detector(cfg, table)
        hands = core.hands.HandTracker(frame_size=tuple(cfg["frame_size_px"]))

    if fake:
        from act.calibrate import calibrate
        from act.sim import SimRig
        rig = SimRig(cfg)
        cal = os.path.join(tempfile.mkdtemp(prefix="askroom_fake_"), "laser_cal.json")
        laser = rig.make_laser(cal)
        calibrate(laser)
        laser.off()
        log.info("fake laser: simulated rig calibrated (%s)", cal)
    else:
        laser = make_laser(cfg, frames, table)
        cleanup.append(laser.act.close)
        if (cfg.get("room") or {}).get("enabled"):
            from act.room_map import RoomMap
            path = cfg.get("room_map", "room_map.json")
            if os.path.exists(path):
                laser.room_map = RoomMap.load(path)
                log.info("room map: %d dots, zones: %s", laser.room_map.n_seen,
                         ", ".join(laser.room_map.zones) or "none (room aims refused)")
            else:
                log.warning("room.enabled but no %s; run python -m act.room_map --sweep", path)

    netmon = net.NetMonitor(cfg).start()
    cleanup.append(netmon.stop)
    interpret = voice.understand.Understander(cfg, online=lambda: netmon.online,
                                              aliases=getattr(world, "alias_phrases", None))
    interpret.warm()                                   # logs and falls back to the rules if the model is down
    import core.xai
    llm = cfg.get("llm") or {}
    warm_on_connect(netmon, lambda: core.xai.warm(llm.get("base_url") or core.xai.BASE_URL)
                    and log.info("Grok connection warm"))
    # Narration and visual memory (both off unless enabled in config): they attach to world.update, so
    # the perception loop and --fake's SimCamera feed them without a call here; stopped before the log closes.
    # The Grok settle check (off unless enabled) attaches the same way; visual Q&A reads its sightings.
    import core.grok_check
    import core.narration
    import voice.visual
    narrator = core.narration.from_config(cfg, events, world, online=lambda: netmon.online)
    visual = voice.visual.from_config(cfg, world, events, frames, table, online=lambda: netmon.online)
    # Automatic names for new things (off unless auto_name.enabled): attached after the visual layer, so
    # the crop store already knows which views are each thing's when a new one is queued for Grok.
    import core.auto_name
    namer = core.auto_name.from_config(cfg, world, online=lambda: netmon.online)
    checker = core.grok_check.from_config(cfg, events, world, table, online=lambda: netmon.online)
    if visual is not None:
        visual.grok_check = checker
    cleanup += [x.stop for x in (narrator, visual, namer, checker) if x is not None]
    ask = voice.pipeline.make_ask(cfg, world, events, net=netmon, interpret=interpret, visual=visual)
    tts = voice.tts.TTS(cfg, net=netmon)
    tts.warm()
    cleanup.append(tts.stop)

    stt = clicker = None
    if with_voice:
        from voice.stt import STT
        from voice.trigger import Clicker
        clicker = Clicker(cfg, keyboard=True if (fake and keyboard is None) else keyboard)
        cleanup.append(clicker.close)
        stt = STT(cfg, clicker=clicker, tts=tts)          # drops a clip the rig's own voice starts in
        stt.warm()

    room = Room(cfg, world, events, table, frames, laser, ask, netmon=netmon, tts=tts, stt=stt,
                clicker=clicker, detector=detector, hands=hands, interpret=interpret)
    room.cleanup = cleanup
    if (cfg.get("care") or {}).get("enabled", True):     # reminders, reports, follow-ups, profile (voice/care.py)
        from voice.care import attach_care
        cleanup.append(attach_care(room, cfg).stop)
    return room, perception


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Ask the Room")
    ap.add_argument("--fake", action="store_true", help="no hardware: sim camera, simulated laser, Enter to ask")
    ap.add_argument("--no-voice", action="store_true", help="dashboard and /ask only")
    ap.add_argument("--listen", choices=["always", "wake", "click"], help="override listen.mode")
    ap.add_argument("--camera", type=camera_source, default=None,
                    help="camera index, or a stable path like /dev/v4l/by-id/usb-046d_...-video-index0 "
                         "(default: config demo_check.camera if it exists, else 0)")
    ap.add_argument("--video", help="play this recording instead of the camera (through the real detector)")
    ap.add_argument("--host")
    ap.add_argument("--port", type=int)
    ap.add_argument("--config")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    for noisy in ("httpx", "pywhispercpp", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    cfg = load_config(args.config)
    if args.listen:
        cfg["listen"] = dict(cfg.get("listen") or {}, mode=args.listen)
    camera = args.camera if args.camera is not None else default_camera(cfg)
    room, perception = build(cfg, fake=args.fake, camera=camera, with_voice=not args.no_voice,
                             video=args.video)
    host = args.host or cfg["server"]["host"]
    port = args.port or cfg["server"]["port"]
    signal.signal(signal.SIGTERM, lambda *_: room.stop_ev.set())
    room.run(host, port, voice=not args.no_voice, perception=perception)
    try:
        while not room.stop_ev.wait(0.5):
            if room.server is not None and room.server.started is False and not room.threads[-1].is_alive():
                log.error("server exited (port %s in use?)", port)
                break
    except KeyboardInterrupt:
        pass
    log.info("shutting down")
    room.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
