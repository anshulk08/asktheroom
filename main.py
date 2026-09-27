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

Room memory (spec 0009 M0, `room_memory.enabled`, off by default) runs only on the live camera, never with
--fake or --video: the camera runs at 1920x1080 behind core.room_view.TableView, and the perception step
also hands the full frame to core.room.RoomMemory, one drawn zone every room_every_n frames.
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
WAKE_FILLER = {"hey", "hi", "ok", "okay", "yo", "please", "um", "uh"}   # words allowed around a bare wake word

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
        self.room_memory = None                    # core.room.RoomMemory when room_memory.enabled (build)
        self._room_err_t = float("-inf")
        if interpret is None:
            from voice.understand import Understander
            interpret = Understander(dict(cfg, understand={"enabled": False}))     # rules only
        self.interpret = interpret
        m = cfg.get("main") or {}
        self.max_fps = float(m.get("perception_max_fps", 15))
        self.net_copy_s = float(m.get("net_copy_s", 5))
        self.laser_timeout_s = float(cfg.get("laser_timeout_s", 10))
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
        self.stop_ev = threading.Event()
        self._acted = threading.local()            # .kind: RESET / RECAL if the router answered one
        self._clear_ev = threading.Event()         # RESET: the perception thread resets proposals, crops, room memory
        self._cal_warned = False
        self._aim_lock = threading.Lock()
        self._aim_gen = 0
        self._off_timer: Optional[threading.Timer] = None
        self.server = None
        self.record_answer: Optional[Callable[[str, Answer, str], None]] = None   # server log, for the phone
        self._phone_qs: deque = deque(maxlen=5)    # (monotonic t, normalized text) of recent phone questions
        self.threads: list[threading.Thread] = []
        self.cleanup: list[Callable[[], None]] = []
        self.last_timing: dict = {}

    # -- asking and answering

    def _router(self, ask: Callable[[str, str], Answer]) -> Callable[[str, str], Answer]:
        """ask, noting the intent the router answered, so Room.ask acts only on a real RESET / RECAL:
        never on text the care layer handled ('remind me to reset the router') or rewrote."""
        def routed(text: str, source: str) -> Answer:
            ans = ask(text, source)
            self._acted.kind = self.interpret(text).kind     # the same Intent the router used (cached)
            return ans
        return routed

    def ask(self, text: str, source: str) -> Answer:
        """The router, plus the two intents that act on the room (reset, recalibrate)."""
        self._acted.kind = None
        ans = self.base_ask(text, source)
        kind, self._acted.kind = self._acted.kind, None
        if kind == "RESET":
            self.world.reset()
            if self.hands is not None:
                self.hands.reset()
            self._clear_ev.set()
        elif kind == "RECAL":
            threading.Thread(target=self._recalibrate_and_tell, args=(source != "sms",), name="recal",
                             daemon=True).start()
        return ans

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
        if moved > 2.0 and self.laser is not None and getattr(self.laser, "fit", None) is not None:
            log.warning("the table frame moved %.1f cm: recalibrate the laser (python -m act.calibrate) "
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
            if self.room_memory is not None:   # same thread as its step: never concurrent with a visit
                try:
                    self.room_memory.reset()
                except Exception:              # the room must never cost the table
                    log.exception("room memory reset failed; table perception goes on")
        dets = self.detector.detect(frame)
        dets.hands = self.hands.update(dets.hands, dets.t)
        if self.room_enabled:
            self._hand_boxes.extend((time.monotonic(), h.box_px) for h in dets.hands)
        events = self.world.update(dets, frame)
        if self.room_memory is not None:
            events = events + self._room_step(frame)
        return dets, events

    def _room_step(self, frame) -> list:
        """Room memory (spec 0009 M0) on the full camera frame this table frame was cut from. Its errors are
        logged (at most every 10 s) and never cost table perception."""
        full_at = getattr(self.frames, "full_at", None)
        if full_at is None:
            return []
        try:
            full = full_at(frame.t)
            return list(self.room_memory.step(full) or []) if full is not None else []
        except Exception:                   # the room must never cost the table
            now = time.monotonic()
            if now - self._room_err_t > 10:
                log.exception("room memory step failed; table perception goes on")
                self._room_err_t = now
            return []

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
        if self.listen_mode == "click":
            while not self.stop_ev.is_set():
                if self.clicker.wait_press(timeout=0.5):
                    self._asked(time.monotonic())
            return
        if self.stt is not None:
            # Overheard chatter stays out of the logs (privacy, README). listen.log_overheard: true keeps
            # every transcript in the log while tuning the wake word on a rig; never for a deployment.
            self.stt.log_text = bool((self.cfg.get("listen") or {}).get("log_overheard", False))
        ignored = 0
        while not self.stop_ev.is_set():
            if self.clicker is not None and self.clicker.pressed():
                self._asked(time.monotonic())   # "listen now": the next thing said is for the rig
                ignored = 0
                continue
            if self._speaking():
                self.stop_ev.wait(0.05)
                continue
            try:
                text = self.stt.hear(self.idle_s)
            except Exception:
                log.exception("listening failed")
                self.stop_ev.wait(1.0)
                continue
            t_heard = time.monotonic()
            if self.stt.last_stop == "click":   # pressed while talking: that was for the rig
                if text:
                    self._answer(text, t_heard, t_heard, {"mode": "asked", "ignored_since_last": ignored})
                else:
                    self._asked(t_heard)
                ignored = 0
                continue
            if not text:
                continue
            if self._bare_wake(text):           # "Room!" ... pause ... the question: listen for it now
                log.info("heard the wake word alone; listening for the question")
                self._asked(t_heard)
                continue
            if self.interpret(text, overheard=True).kind == "IGNORE":
                ignored += 1                    # dropped: not logged, not stored, not sent
                continue
            log.info("heard: %r", text)
            self._answer(text, t_heard, t_heard, {"mode": "overheard", "ignored_since_last": ignored})
            ignored = 0

    def _speaking(self) -> bool:
        return bool(getattr(self.tts, "speaking", False))

    def _bare_wake(self, text: str) -> bool:
        """The wake word on its own ("Room!", "hey room"): people pause after it, so the VAD ends the
        utterance before the question. Treated like a clicker press: the next thing said is for the rig
        (rig run, Sat 26 Sep: "Room!" then "where is my wallet?" as two utterances, neither answered)."""
        from voice.intents import normalize
        from voice.understand import wake_words
        words = normalize(text).split()
        wake = set(wake_words(self.cfg))
        return bool(words) and any(w in wake for w in words) and all(w in wake or w in WAKE_FILLER for w in words)

    def _asked(self, t_press: float) -> None:
        """Clicker press, or the wake word alone: stop the current answer, listen for one question, answer it."""
        if self.tts is not None:
            self.tts.stop()                     # a click interrupts the previous answer
        if self.clicker is not None:
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

    def _answer(self, text: str, t0: float, t_heard: float, extra: dict) -> None:
        """Answer, speak and aim; report to n8n. Asked: t0 is the click. Overheard: the end of speech.
        Waits until the answer has been spoken plus echo_tail_s, so the mic doesn't hear the rig;
        a clicker press cuts the answer short (and is kept for the next question)."""
        if self._heard_from_phone(text):
            log.info("heard %r: the phone just asked it; answered once", text)
            return
        ans, cued = self._ask_with_cue(text)
        if self.record_answer is not None:
            try:
                self.record_answer(text, ans, "voice")
            except Exception:
                log.exception("record_answer failed")
        t_ans = time.monotonic()
        intent = self.interpret(text)
        say, aim = self.respond(ans)
        aim.join()
        overheard = extra.get("mode") == "overheard"
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
            return
        while say.is_alive() and not self.stop_ev.is_set():
            if self.clicker is not None and self.clicker.wait_press(timeout=0.05):
                self.tts.stop()
                self.clicker.press()            # handled by the loop as "listen now"
                break
            elif self.clicker is None:
                say.join(0.05)
        self.stop_ev.wait(self.echo_tail_s)

    def _ask_with_cue(self, text: str) -> tuple[Answer, bool]:
        """ask(text, "voice"); if no answer within demo.thinking_cue_s (a model is reading it, or Grok
        is answering), say a short "let me look" so the rig doesn't sit silent. Returns (answer, cued).
        The TTS lock queues the answer behind the cue."""
        if self.cue_after_s <= 0 or self.tts is None or not self.cue_phrases:
            return self.ask(text, "voice"), False
        box: dict = {}
        done = threading.Event()

        def work() -> None:
            try:
                box["ans"] = self.ask(text, "voice")
            except BaseException as ex:          # re-raised on the caller's thread
                box["err"] = ex
            finally:
                done.set()

        threading.Thread(target=work, name="ask", daemon=True).start()
        cued = not done.wait(self.cue_after_s)
        if cued:
            phrase = self.cue_phrases[self._cues % len(self.cue_phrases)]
            self._cues += 1
            threading.Thread(target=self._speak, args=(phrase,), name="cue", daemon=True).start()
            done.wait()
        if "err" in box:
            raise box["err"]
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
            self._thread(self.voice_loop, "voice")
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


def open_frames(cfg: dict, camera) -> tuple[object, Optional[tuple[int, int, int, int]]]:
    """The live camera, and the table view rect when room memory is on (else None).

    Room memory off: FrameBuffer(camera), as always. On (spec 0009 M0): the camera runs at
    room_memory.capture_size (1920x1080, zoom 100) in a short ring (ring_s: 1080p frames are big), and the
    table pipeline sees it through TableView, which cuts the zoom-160 region (table_view_rect) and resizes
    it to 1280x720, so everything that reads frames keeps getting table frames."""
    import core.capture
    from core.room_types import RoomConfig
    rc = RoomConfig.from_dict(cfg.get("room_memory"))
    decode_fps = (cfg.get("capture") or {}).get("decode_fps")      # grab every frame, decode only this many
    if not rc.enabled:
        return core.capture.FrameBuffer(camera, decode_fps=decode_fps), None
    from core.room_view import TableView, default_rect
    out = tuple(cfg.get("frame_size_px", (1280, 720)))
    rect = rc.table_view_rect
    if rect is None:
        rect = tuple(default_rect(rc.capture_size, rc.zoom, rc.ref_zoom, out))
        log.warning("room_memory: no table_view_rect, using the centred default %s; measure it "
                    "(python -m core.room --measure-rect) and put it in config.local.yaml", list(rect))
    w, h = rc.capture_size
    fb = core.capture.FrameBuffer(camera, ring_s=rc.ring_s,
                                  opener=lambda src: core.capture.open_camera(src, w, h), decode_fps=decode_fps)
    log.info("room memory: camera at %dx%d, table view %s", w, h, list(rect))
    return TableView(fb, rect, out), rect


def make_room_memory(cfg: dict, world, detector, rect):
    """core.room.RoomMemory on the detector's already-loaded prop model, or None (RoomMemory.from_config logs
    why: no zones file, no zones, zones drawn at another view). A failure to start is logged and the table
    runs without it. Unnamed things in zones need the detector's YOLOE proposer (see room_things)."""
    try:
        from core.room import RoomMemory
        return RoomMemory.from_config(cfg, world, detector.backend, rect, **room_things(cfg, world, detector))
    except Exception:
        log.exception("room memory failed to start; the table runs without it")
        return None


def room_things(cfg: dict, world, detector) -> dict:
    """RoomMemory.from_config's thing arguments: a zone YOLOE proposer on the detector's already-loaded
    YOLOE model (a second load does not fit the 8 GB Jetson), and a Grok name_fn when auto_name is enabled.
    {} (props only, logged) when room_memory is off or things are off, when the detector has no YOLOE
    proposer (proposals.kind is not yoloe), or when building either fails."""
    from core.room_types import RoomConfig
    rc = RoomConfig.from_dict(cfg.get("room_memory"))
    if not (rc.enabled and rc.things):
        return {}
    try:
        from core.proposals import YOLOEProposer
        shared = getattr(detector, "proposer", None)
        if not isinstance(shared, YOLOEProposer):
            log.info("room things off: the detector has no YOLOE proposer to share (proposals.kind: yoloe)")
            return {}
        ycfg = dict((cfg.get("proposals") or {}).get("yoloe") or {})
        ycfg["ignore_px"] = []                 # table-view px: meaningless on a zone crop
        proposer = YOLOEProposer(ycfg, model=shared.model)
        proposer.set_roi(None)                 # the zone polygon filters, not the table outline
        online = lambda: bool(getattr(world, "online", False))   # noqa: E731 (world.online follows NetMonitor)
        kw = {"proposer": proposer, "online": online}
        from core.auto_name import AutoNameConfig, AutoNamer
        if AutoNameConfig.from_dict(cfg.get("auto_name")).enabled:
            namer = AutoNamer(cfg, world, online=online, start=False)
            from core.room import make_verify_fn
            kw["name_fn"], kw["verify_fn"] = namer._ask, make_verify_fn(namer)
        else:
            log.info("room things are not named: auto_name.enabled is false")
        return kw
    except Exception:
        log.exception("room things failed to start; room memory tracks props only")
        return {}


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
    room_rect = None                                   # table view rect: room memory on (open_frames)
    perception = True
    if (fake or video) and (cfg.get("room_memory") or {}).get("enabled"):
        log.info("room_memory.enabled is ignored with --fake / --video: room memory needs the live camera")
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
            frames, room_rect = open_frames(cfg, camera)
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
        import act.actuator
        import act.laser
        actuator = act.actuator.make_actuator(cfg)     # cfg["actuator"]: fake | pca9685 | serial
        cleanup.append(actuator.close)
        if str(cfg.get("actuator", "fake")).lower() == "fake":
            log.warning("actuator is 'fake': the servos will not move. On the rig set `actuator: pca9685` "
                        "(or serial/bus) in config.local.yaml")
        laser = act.laser.Laser(actuator, frames, table, cfg["paths"]["laser_cal"], cfg=cfg)
        if laser.fit is None:
            log.warning("laser not calibrated (%s missing); answers will be spoken only",
                        cfg["paths"]["laser_cal"])
        elif laser_older_than_table(laser.fit, table.cal_path):
            log.warning("laser_cal.json was fitted before the last table calibration: the laser may point "
                        "off; recalibrate it (python -m act.calibrate)")
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
    interpret = voice.understand.Understander(cfg, online=lambda: netmon.online)
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
        stt = STT(cfg, clicker=clicker)
        stt.warm()

    room = Room(cfg, world, events, table, frames, laser, ask, netmon=netmon, tts=tts, stt=stt,
                clicker=clicker, detector=detector, hands=hands, interpret=interpret)
    room.cleanup = cleanup
    if room_rect is not None:
        room.room_memory = make_room_memory(cfg, world, detector, room_rect)
        if room.room_memory is not None:
            cleanup.append(room.room_memory.stop)          # its Grok naming worker
        if room.room_enabled:      # 0006 room pointing reads frames.latest(), now the table view (spec 0009 M5)
            log.error("room pointing (room.enabled) does not work with room_memory yet: room pointing is off")
            room.room_enabled = False
    if (cfg.get("care") or {}).get("enabled", True):     # reminders, reports, follow-ups, profile (voice/care.py)
        from voice.care import attach_care
        cleanup.append(attach_care(room, cfg).stop)
    return room, perception


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Ask the Room")
    ap.add_argument("--fake", action="store_true", help="no hardware: sim camera, simulated laser, Enter to ask")
    ap.add_argument("--no-voice", action="store_true", help="dashboard and /ask only")
    ap.add_argument("--listen", choices=["always", "wake", "click"], help="override listen.mode")
    ap.add_argument("--camera", type=camera_source, default=0,
                    help="camera index, or a stable path like /dev/v4l/by-id/usb-046d_0809_...-video-index0")
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
    room, perception = build(cfg, fake=args.fake, camera=args.camera, with_voice=not args.no_voice,
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
