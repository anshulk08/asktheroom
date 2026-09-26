"""Ask the Room: the whole program (spec V6). One process, five threads:

  capture     30 fps     core.capture.FrameBuffer runs its own thread
  perception  10-15 fps  wait_new -> detector.detect -> hands.update -> world.update (world logs events)
  voice       on click   clicker -> stt.record_until_silence -> stt.transcribe -> ask -> speak + aim
  net         every 5 s  NetMonitor probes; world.online follows it
  server      5 Hz       server.app.create_app(...) under uvicorn

    python main.py                  # the rig: camera, detector, servos, clicker
    python main.py --fake           # no hardware: sim camera + world, simulated laser rig, Enter to ask
    python main.py --no-voice       # dashboard and /ask only
    python main.py --camera 2

Typed questions (/ask) and texts (/sms) go through the same ask(); dashboard questions also speak
and move the laser, texts only answer by text. Aiming never blocks speech: an uncalibrated laser
raises RuntimeError, which is logged, and the answer still plays.

--fake wiring (mirrors server/sim.py): server.sim.SimCamera plays the scripted tabletop story and
feeds the real World itself, so it stands in for capture + perception; the laser is act.sim.SimRig
(a FakeActuator plus its own simulated camera), calibrated at startup into a temp file. With
--video, a recording replaces the sim camera and goes through the real detector (needs ultralytics
and the model; the table falls back to frame == table when table_cal.json is missing).
"""
from __future__ import annotations

import argparse
import logging
import os
import signal
import tempfile
import threading
import time
from typing import Callable, Optional

import numpy as np

from core.config import load_config
from core.types import Answer
from voice.intents import parse

log = logging.getLogger("askroom.main")

OFF = {"on": False, "target": None, "err_cm": None}
TEXT_ONLY = ("sms", "n8n")     # sources answered in text, without speaking or moving the laser


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


class Room:
    """Owns every part and thread. build() wires the rig or the fake; run() starts the threads."""

    def __init__(self, cfg: dict, world, events, table, frames, laser, ask: Callable[[str, str], Answer],
                 netmon=None, tts=None, stt=None, clicker=None, detector=None, hands=None):
        self.cfg, self.world, self.events, self.table = cfg, world, events, table
        self.frames, self.laser, self.base_ask = frames, laser, ask
        self.netmon, self.tts, self.stt, self.clicker = netmon, tts, stt, clicker
        self.detector, self.hands = detector, hands
        m = cfg.get("main") or {}
        self.max_fps = float(m.get("perception_max_fps", 15))
        self.net_copy_s = float(m.get("net_copy_s", 5))
        self.laser_timeout_s = float(cfg.get("laser_timeout_s", 10))
        self.stop_ev = threading.Event()
        self._aim_lock = threading.Lock()
        self._aim_gen = 0
        self._off_timer: Optional[threading.Timer] = None
        self.server = None
        self.threads: list[threading.Thread] = []
        self.cleanup: list[Callable[[], None]] = []
        self.last_timing: dict = {}

    # -- asking and answering

    def ask(self, text: str, source: str) -> Answer:
        """The router, plus the two intents that act on the room (reset, recalibrate)."""
        ans = self.base_ask(text, source)
        kind = parse(text, self.cfg).kind
        if kind == "RESET":
            self.world.reset()
            if self.hands is not None:
                self.hands.reset()
        elif kind == "RECAL":
            threading.Thread(target=self.recalibrate, name="recal", daemon=True).start()
        return ans

    def ask_and_act(self, text: str, source: str) -> Answer:
        """ask_fn for the server: dashboard questions are spoken and aimed; texts and n8n chat
        questions are only answered."""
        ans = self.ask(text, source)
        if source not in TEXT_ONLY:
            self.respond(ans)
        return ans

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
        action = ans.action or ("point" if ans.point_at else None)
        if action is None:
            return None
        err = None
        with self._aim_lock:
            try:
                if action.startswith("sweep:"):
                    self.laser.sweep_edge(action.split(":", 1)[1])
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

    def _schedule_off(self) -> None:
        """Show the laser as off once the actuator's auto-off would have fired (and turn it off,
        which the fake actuators need; real ones have already done it)."""
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

        self._off_timer = threading.Timer(self.laser_timeout_s, off)
        self._off_timer.daemon = True
        self._off_timer.start()

    def recalibrate(self) -> None:
        f = self.frames.latest() if self.frames is not None else None
        if f is None or f.img is None:
            log.warning("recalibrate: no camera frame")
            return
        ok = self.table.calibrate(f.img)
        log.info("table recalibration %s", "ok" if ok else "failed (markers 0-3 not all visible)")

    # -- threads

    def perception_loop(self) -> None:
        last_idx, n, t_win = 0, 0, time.monotonic()
        period = 1.0 / self.max_fps if self.max_fps > 0 else 0.0
        warned = False
        while not self.stop_ev.is_set():
            t0 = time.monotonic()
            frame = self.frames.wait_new(last_idx, timeout=1.0)
            if frame is None:
                continue
            last_idx = frame.idx
            if not self.table.ok:
                if not warned:
                    log.warning("table not calibrated; trying markers 0-3 every frame (python -m core.table)")
                    warned = True
                self.table.calibrate(frame.img)
                continue
            try:
                dets = self.detector.detect(frame)
                dets.hands = self.hands.update(dets.hands, dets.t)
                self.world.update(dets, frame)
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
        while not self.stop_ev.is_set():
            if not self.clicker.wait_press(timeout=0.5):
                continue
            t_press = time.monotonic()
            if self.tts is not None:
                self.tts.stop()                 # a click interrupts the previous answer
            self.clicker.clear()
            try:
                text = self.stt.listen()
            except Exception:
                log.exception("listening failed")
                self._speak("Sorry, the microphone isn't working.")
                continue
            t_heard = time.monotonic()
            if not text:
                self._speak("Sorry, I didn't catch that.")
                continue
            log.info("heard: %r", text)
            ans = self.ask(text, "voice")
            t_ans = time.monotonic()
            say, aim = self.respond(ans)
            aim.join()
            self.last_timing = {"record_transcribe_s": round(t_heard - t_press, 2),
                                "ask_s": round(t_ans - t_heard, 2),
                                "click_to_laser_s": round(time.monotonic() - t_press, 2),
                                "stt_ms": dict(self.stt.last_ms)}
            log.info("timing %s", self.last_timing)

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
                         ask_fn=self.ask_and_act, table=self.table)
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
            log.info("press the clicker%s to ask a question", " (or Enter)" if self.clicker.kind == "keyboard" else "")
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

def build(cfg: dict, fake: bool = False, camera: int = 0, with_voice: bool = True,
          video: Optional[str] = None, keyboard: Optional[bool] = None) -> tuple[Room, bool]:
    """Construct everything. Returns (room, needs_perception_thread)."""
    import core.events
    import core.world
    import net
    import voice.pipeline
    import voice.tts

    cleanup: list[Callable[[], None]] = []
    if fake:
        snap = tempfile.mkdtemp(prefix="askroom_fake_snaps_")
        events = core.events.EventLog(":memory:", snap)
    else:
        events = core.events.EventLog(cfg["paths"]["events_db"], cfg["paths"]["snapshots"])
    cleanup.append(events.close)
    world = core.world.World(cfg, events)

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
        laser = act.laser.Laser(actuator, frames, table, cfg["paths"]["laser_cal"], cfg=cfg)
        if laser.fit is None:
            log.warning("laser not calibrated (%s missing); answers will be spoken only",
                        cfg["paths"]["laser_cal"])

    netmon = net.NetMonitor(cfg).start()
    cleanup.append(netmon.stop)
    ask = voice.pipeline.make_ask(cfg, world, events, net=netmon)
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
                clicker=clicker, detector=detector, hands=hands)
    room.cleanup = cleanup
    return room, perception


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Ask the Room")
    ap.add_argument("--fake", action="store_true", help="no hardware: sim camera, simulated laser, Enter to ask")
    ap.add_argument("--no-voice", action="store_true", help="dashboard and /ask only")
    ap.add_argument("--camera", type=int, default=0, help="camera index")
    ap.add_argument("--video", help="play this recording instead of the camera (through the real detector)")
    ap.add_argument("--host")
    ap.add_argument("--port", type=int)
    ap.add_argument("--config")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    for noisy in ("httpx", "pywhispercpp", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    cfg = load_config(args.config)
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
