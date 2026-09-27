"""Pre-demo check (H8). Run before every judge: one green or red line per check, exit 1 if any fail.

    python demo_check.py                  # the rig
    python demo_check.py --skip-manual    # unattended: no "did you hear it" / kill-switch prompts
    python demo_check.py --fake           # no hardware: rendered home layout, simulated laser rig
    python demo_check.py --only 3 4       # just these checks

  1 camera      >= 25 fps, exposure locked (manual, as scripts/camera_setup.sh sets it)
  2 detector    >= 10 fps, all 8 objects seen on the home layout
  3 markers     ArUco 0-3 found; they land < 1 cm from where table_cal.json says they are
  4 laser       calibration error < 1.5 cm, a test point at the table centre lands < 3 cm away
  5 audio       mic level OK, speaker plays a test tone
  6 network     NetMonitor.check_once() agrees with a direct probe, and the dashboard shows it
  7 world       after reset, every object VISIBLE (at config demo_check.home_cm, if set)
  8 kill switch manual: laser on, press the kill switch, confirm the dot went out
  9 clock       not behind the last saved file (a Jetson offline with no RTC battery boots stale)
 10 room memory when room_memory.enabled: the zones file has zones drawn at this camera view (files only)
 11 room        when room.enabled: the room dot map matches the camera and mapped dots are hit again
 12 room app    the running app answers: perception fresh (fps), and /full.jpg (room memory's full frame)
 13 grok        api.x.ai answers GET /models with XAI_API_KEY (no tokens), with the round trip time
 14 devices     the mic (stt.input_device) and speaker (tts.output_device) named in config are present
 15 memory      MemAvailable >= room_check.min_ram_mb; no NvMapMemAlloc / Traceback in the app log's tail
 16 namer       the room namer's queue is bounded (a burst of 100 crops keeps <= 8)

    python demo_check.py --live           # the morning pass while the room app runs (it keeps the camera
                                          # and mic): checks 6, 9, 10, 12-16 only

Missing hardware fails that check with the reason, so this also runs on a laptop. Parts (camera,
table, detector, laser) are built on first use and shared; one that fails to build fails every
check that needs it, with the same reason. Every check has a deadline (DEADLINE_S: the detector and
world checks 180 s for a TensorRT load, prompts 300 s, the rest 30 s): a check that hangs (a speaker
that never takes the tone, a camera that never answers) fails "timed out" with its stack on stderr,
and the run goes on. Recording and the test tone have their own shorter deadlines.

--fake: the camera is a FrameBuffer over a rendered home layout (server.sim's painter and LAYOUT,
with ArUco 0-3 drawn in), the detector's backend reads boxes off that layout, the laser is an
act.sim.SimRig calibrated at startup, the mic is a synthetic tone and the speaker is silent. The
network check is real.
"""
from __future__ import annotations

import argparse
import faulthandler
import math
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
import traceback
import weakref
from collections import deque
from typing import Callable, Optional, Union
from urllib.parse import urlparse

import numpy as np

from core.config import load_config

GREEN, RED, YELLOW, RESET = "\033[32m", "\033[31m", "\033[33m", "\033[0m"

MIN_CAMERA_FPS = 25.0
MIN_DETECT_FPS = 10.0
MIN_DETECT_RATE = 0.5           # an object must be in at least half the frames
MAX_MARKER_CM = 1.0
MAX_LASER_FIT_CM = 1.5
MAX_LASER_AIM_CM = 3.0
SPEECH_DBFS = -35.0             # loudest 32 ms block while someone talks, at arm's length
SILENT_DBFS = -90.0             # below this the mic is sending digital zeros (muted or wrong device)

Result = tuple[Optional[bool], str]         # ok (None = skipped), message

DEADLINE_S = {"detector": 180.0, "world": 180.0}   # may load the TensorRT engine
MANUAL_DEADLINE_S = 300.0                          # checks that wait for a person to answer
DEFAULT_DEADLINE_S = 30.0
AUDIO_GRACE_S = 3.0                                # recording / tone may overrun their length by this


# ---------------------------------------------------------------- fake hardware

FAKE_MARKERS_CM = {0: (4, 4), 1: (86, 4), 2: (86, 54), 3: (4, 54)}
FAKE_MARKER_CM = 4.5


def fake_cfg(cfg: dict) -> dict:
    """The rendered frame is the table (server.sim), with markers inset so they stay in view."""
    t = dict(cfg.get("table") or {}, markers={k: list(v) for k, v in FAKE_MARKERS_CM.items()})
    return dict(cfg, table=t, table_tag=dict(cfg.get("table_tag") or {}, enabled=False))   # the fake draws 0-3


def fake_layout(cfg: dict) -> dict[str, tuple[float, float]]:
    from server.sim import LAYOUT
    return {o: LAYOUT[o] for o in cfg["objects"]}


def render_home(cfg: dict) -> np.ndarray:
    """The home layout as the overhead camera would see it, markers 0-3 included."""
    import cv2

    from server.sim import Painter
    p = Painter(cfg)
    img = p.draw({"pos": fake_layout(cfg), "hidden": set(), "carry": [], "hand": None}, "")
    d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    sx = img.shape[1] / p.table.W
    side = int(FAKE_MARKER_CM * sx)
    for mid, c in FAKE_MARKERS_CM.items():
        cx, cy = (int(v) for v in p.table.cm_to_px([c])[0])
        q = side // 5                                        # white quiet zone
        cv2.rectangle(img, (cx - side // 2 - q, cy - side // 2 - q), (cx + side // 2 + q, cy + side // 2 + q),
                      (255, 255, 255), -1)
        m = cv2.cvtColor(cv2.aruco.generateImageMarker(d, mid, side), cv2.COLOR_GRAY2BGR)
        img[cy - side // 2:cy - side // 2 + side, cx - side // 2:cx - side // 2 + side] = m
    return img


class FakeCameraSource:
    """read() -> the home layout with a little sensor noise, paced at fps (FrameBuffer's source API)."""

    def __init__(self, img: np.ndarray, fps: float = 30.0, seed: int = 0):
        self.img, self.period = img, 1.0 / fps
        self.rng = np.random.default_rng(seed)
        self.next_t = time.monotonic()

    def read(self):
        self.next_t += self.period
        time.sleep(max(0.0, self.next_t - time.monotonic()))
        noise = self.rng.integers(0, 3, self.img.shape, dtype=np.uint8)
        return True, np.maximum(self.img, noise) - noise       # darken by 0-2, without wrapping

    def release(self) -> None:
        pass


class LayoutBackend:
    """Detector backend that 'sees' the fake layout: one box per object, where the painter drew it."""

    def __init__(self, cfg: dict, infer_ms: float = 15.0):
        from eval.synth import SIZES
        from server.sim import Painter
        p = Painter(cfg)
        self.raw = [(o, 0.9, p._rect(c, SIZES.get(o, (8, 8)))) for o, c in fake_layout(cfg).items()]
        self.infer_s = infer_ms / 1000

    def infer(self, img: np.ndarray):
        time.sleep(self.infer_s)
        return list(self.raw)


# ---------------------------------------------------------------- prompts

class PromptReader:
    """input() for checks that can be abandoned: one thread reads stdin lines into a queue, and a check
    that timed out while waiting for an answer is marked so it can never take a later answer (a plain
    input() left blocked would swallow the next Enter)."""

    def __init__(self, stream=None):
        self.stream = stream if stream is not None else sys.stdin
        self.lines: deque = deque()
        self.cv = threading.Condition()
        self.eof = False
        self.dead: "weakref.WeakSet[threading.Thread]" = weakref.WeakSet()   # checks that timed out (idents get reused)
        self._thread: Optional[threading.Thread] = None

    def _run(self) -> None:
        while True:
            try:
                line = self.stream.readline()
            except (OSError, ValueError):
                line = ""
            with self.cv:
                if not line:
                    self.eof = True
                    self.cv.notify_all()
                    return
                self.lines.append(line.rstrip("\r\n"))
                self.cv.notify_all()

    def ask(self, prompt: str) -> str:
        print(prompt, end="", flush=True)
        with self.cv:
            if self._thread is None:
                self._thread = threading.Thread(target=self._run, name="stdin", daemon=True)
                self._thread.start()
            while True:
                if threading.current_thread() in self.dead:
                    raise TimeoutError("prompt abandoned: its check timed out")
                if self.lines:
                    return self.lines.popleft()
                if self.eof:
                    raise EOFError
                self.cv.wait(0.5)

    def abandon(self, thread: threading.Thread) -> None:
        with self.cv:
            self.dead.add(thread)
            self.cv.notify_all()


# ---------------------------------------------------------------- the parts, built on first use

class Rig:
    """Lazily builds and caches each part. A part that fails to build raises the same error again,
    so every check that needs a missing camera reports the camera, not a knock-on error."""

    def __init__(self, cfg: dict, fake: bool = False, camera: Union[int, str] = 0, manual: bool = True,
                 ask: Optional[Callable[[str], str]] = None):
        self.fake = fake
        self.cfg = fake_cfg(cfg) if fake else cfg
        self.prompts = PromptReader() if ask is None else None
        self.camera, self.manual = camera, manual
        self._ask = ask if ask is not None else self.prompts.ask
        self.live = False                          # --live: the room app is running and holds camera and audio
        self._parts: dict[str, object] = {}
        self.building: set[str] = set()             # parts being made right now (a hung check marks them)
        self._cleanup: list[Callable[[], None]] = []
        self.tmp = tempfile.mkdtemp(prefix="askroom_check_")
        self.home_cm = (fake_layout(self.cfg) if fake else
                        {o: tuple(v) for o, v in ((cfg.get("demo_check") or {}).get("home_cm") or {}).items()})
        self.home_tol_cm = float((cfg.get("demo_check") or {}).get("home_tol_cm", 5))

    def part(self, name: str):
        got = self._parts.get(name)
        if got is None:
            self.building.add(name)
            try:
                got = getattr(self, f"_make_{name}")()
            except Exception as e:                 # noqa: BLE001 - reported as the check's reason
                got = e
            finally:
                self.building.discard(name)
            self._parts.setdefault(name, got)      # a timeout may have marked it failed meanwhile
            got = self._parts[name]
        if isinstance(got, Exception):
            raise got
        return got

    def ask(self, prompt: str) -> Optional[str]:
        """A manual answer, or None when prompts are off (--skip-manual, or no terminal)."""
        if not self.manual:
            return None
        try:
            return self._ask(prompt).strip().lower()
        except EOFError:
            return None

    # -- makers

    def _make_frames(self):
        import core.capture
        src = FakeCameraSource(render_home(self.cfg)) if self.fake else self.camera
        fb = core.capture.FrameBuffer(src)
        self._cleanup.append(fb.stop)
        if fb.wait_new(0, timeout=3.0) is None:
            raise RuntimeError(f"camera {self.camera} opened but sent no frames in 3 s")
        return fb

    def _make_table(self):
        import core.table
        cal = os.path.join(self.tmp, "table_cal.json") if self.fake else None
        table = core.table.Table(self.cfg, cal_path=cal)
        if self.fake:
            f = self.part("frames").latest()
            if not table.calibrate(f.img):
                raise RuntimeError(f"fake markers not found (saw {table.found})")
        if not table.ok:
            raise RuntimeError(f"table not calibrated ({table.cal_path} missing): run python -m core.table")
        return table

    def _make_detector(self):
        import core.detect
        backend = LayoutBackend(self.cfg) if self.fake else None
        return core.detect.Detector(self.cfg, self.part("table"), backend=backend)

    def _make_laser(self):
        if self.fake:
            from act.calibrate import calibrate
            from act.sim import SimRig
            laser = SimRig(self.cfg).make_laser(os.path.join(self.tmp, "laser_cal.json"))
            calibrate(laser)
            laser.off()
            return laser
        import act.actuator
        import act.laser
        actuator = act.actuator.make_actuator(self.cfg)
        self._cleanup.append(actuator.close)
        laser = act.laser.Laser(actuator, self.part("frames"), self.part("table"),
                                self.cfg["paths"]["laser_cal"], cfg=self.cfg)
        self._cleanup.append(laser.off)
        return laser

    def _make_room(self):
        """(laser, frames) for room pointing: the laser with the room map loaded. --fake: the simulated
        room (act/sim.py RoomRig), swept here."""
        from act.room_map import RoomMap, sweep
        if self.fake:
            from act.sim import RoomRig
            rig = RoomRig(seed=0)
            laser = rig.make_laser()
            laser.room_map = sweep(laser, grid=(12, 9), n_pairs=1)
            return laser, rig.frames
        path = self.cfg.get("room_map", "room_map.json")
        if not os.path.exists(path):
            raise RuntimeError(f"no {path}: run python -m act.room_map --sweep (nobody in view)")
        laser = self.part("laser")
        laser.room_map = RoomMap.load(path)
        return laser, self.part("frames")

    def laser_off_now(self, timeout: float = 2.0) -> None:
        """Best effort before exiting: write laser-off straight to the hardware, without the actuator's
        lock (a timed-out check may still hold it) and with a deadline."""
        for name in ("laser", "room"):
            got = self._parts.get(name)
            laser = got[0] if isinstance(got, tuple) else got
            act = getattr(laser, "act", None)
            hw = getattr(act, "_hw_laser", None)
            if hw is None:
                continue
            t = threading.Thread(target=hw, args=(False, time.monotonic()), name="laser-off", daemon=True)
            t.start()
            t.join(timeout)

    def close(self) -> None:
        for fn in reversed(self._cleanup):
            try:
                fn()
            except Exception:                      # noqa: BLE001
                pass
        shutil.rmtree(self.tmp, ignore_errors=True)

    # -- audio seams (replaced in --fake and tests)

    def record(self, seconds: float) -> np.ndarray:
        if self.fake:
            t = np.arange(int(seconds * 16000)) / 16000
            return (0.2 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
        import sounddevice as sd
        dev = (self.cfg.get("stt") or {}).get("input_device")
        a = sd.rec(int(seconds * 16000), samplerate=16000, channels=1, dtype="float32", device=dev)
        deadline = time.monotonic() + seconds + AUDIO_GRACE_S
        while time.monotonic() < deadline and sd.get_stream().active:   # sd.wait() can wait forever
            time.sleep(0.05)
        done = not sd.get_stream().active
        sd.stop()
        if not done:
            raise TimeoutError(f"mic recording didn't finish in {seconds + AUDIO_GRACE_S:.0f} s "
                               f"(stt.input_device {dev!r})")
        return a[:, 0]

    def play(self, pcm: np.ndarray, rate: int) -> None:
        """The test tone on tts.output_device (what the voice uses; the default is HDMI in the container,
        which can block forever), on a thread with a deadline."""
        if self.fake:
            return
        from voice.tts import play_pcm, resolve_output_device
        spec = (self.cfg.get("tts") or {}).get("output_device")
        dev = resolve_output_device(spec)
        t = threading.Thread(target=play_pcm, args=(pcm, rate, dev), name="tone", daemon=True)
        t.start()
        t.join(len(pcm) / rate + AUDIO_GRACE_S)
        if t.is_alive():
            raise TimeoutError(f"speaker didn't finish the test tone (tts.output_device {spec!r}"
                               f"{' = default device' if dev is None else f' = {dev}'}): "
                               "check it with python -m voice.tts --devices")


def dbfs(x: np.ndarray) -> float:
    rms = float(np.sqrt(np.mean(np.square(x, dtype=np.float64)))) if len(x) else 0.0
    return 20 * math.log10(rms) if rms > 0 else -math.inf


def run_frames(rig: Rig, seconds: float, fn: Callable) -> tuple[int, float]:
    """Call fn(frame) on each new camera frame for `seconds`. Returns (frames handled, fps)."""
    frames = rig.part("frames")
    last, n = 0, 0
    t0 = time.monotonic()
    while time.monotonic() - t0 < seconds:
        f = frames.wait_new(last, timeout=1.0)
        if f is None:
            break
        last = f.idx
        fn(f)
        n += 1
    return n, n / max(time.monotonic() - t0, 1e-6)


# ---------------------------------------------------------------- checks

def exposure_mode(device: Union[int, str]) -> tuple[Optional[bool], str]:
    """(locked?, detail) from v4l2-ctl. auto_exposure 1 is manual on UVC cameras."""
    if not sys.platform.startswith("linux"):
        return None, "exposure lock only readable on Linux (v4l2)"
    if shutil.which("v4l2-ctl") is None:
        return None, "v4l2-ctl missing (apt install v4l-utils)"
    dev = device if isinstance(device, str) else f"/dev/video{device}"
    out = subprocess.run(["v4l2-ctl", "-d", dev, "-C", "auto_exposure",
                          "-C", "exposure_time_absolute"], capture_output=True, text=True, timeout=5).stdout
    vals = dict(ln.split(":", 1) for ln in out.splitlines() if ":" in ln)
    mode = vals.get("auto_exposure", "").strip()
    if not mode:
        return None, "v4l2-ctl gave no auto_exposure"
    exp = vals.get("exposure_time_absolute", "?").strip()
    if mode == "1":
        return True, f"exposure manual ({exp} x 100 us)"
    return False, f"auto exposure on (mode {mode}): run scripts/camera_setup.sh"


def check_camera(rig: Rig) -> Result:
    frames = rig.part("frames")
    time.sleep(2.5)                               # FrameBuffer.fps averages the last 60 reads
    fps = frames.fps
    locked, detail = (True, "sim camera, no exposure control") if rig.fake else exposure_mode(rig.camera)
    msg = f"{fps:.1f} fps, {detail}"
    return fps >= MIN_CAMERA_FPS and bool(locked), msg


def check_detector(rig: Rig) -> Result:
    det = rig.part("detector")
    seen: dict[str, int] = dict.fromkeys(rig.cfg["objects"], 0)

    def step(f):
        for d in det.detect(f).items:
            seen[d.cls] = seen.get(d.cls, 0) + 1

    run_frames(rig, 1.0, det.detect)              # warm up (TensorRT's first calls are slow)
    n, fps = run_frames(rig, 4.0, step)
    if n == 0:
        return False, "no frames from the camera"
    missing = [o for o, k in seen.items() if k < MIN_DETECT_RATE * n]
    msg = f"{fps:.1f} fps ({det.last_ms:.0f} ms/frame), {len(seen) - len(missing)}/{len(seen)} objects"
    if missing:
        msg += f"; missing: {', '.join(f'{o} {seen[o]}/{n}' for o in missing)}"
    return fps >= MIN_DETECT_FPS and not missing, msg


def check_markers(rig: Rig) -> Result:
    import core.table
    table = rig.part("table")
    img = rig.part("frames").latest().img
    if getattr(table, "tag_mode", False):
        return _check_tag(table, img)
    found = core.table.find_markers(img)
    ids = [i for i in core.table.TABLE_IDS if i in found]
    if len(ids) < 4:
        return False, f"found markers {ids}, need {list(core.table.TABLE_IDS)} (all others: {sorted(found)})"
    # A fresh 4-point fit is exact by construction; the useful error is drift against the saved
    # calibration (camera or table knocked since core.table ran).
    cm = table.px_to_cm(np.array([found[i] for i in ids]))
    err = np.linalg.norm(cm - np.array([table.markers_cm[i] for i in ids]), axis=1)
    msg = f"markers {ids}, max drift {err.max():.2f} cm (marker {ids[int(err.argmax())]})"
    if err.max() >= MAX_MARKER_CM:
        msg += ": recalibrate (python -m core.table, or say 'recalibrate')"
    return bool(err.max() < MAX_MARKER_CM), msg


def _check_tag(table, img) -> Result:
    """One-tag mode: the tag is usually picked up after calibrating, so it need not be in view. If it is,
    its corners must land where they did during calibration (else the camera or table moved)."""
    import json

    import core.table
    w, h = table.size_cm
    c = core.table.tag_corners(img, table.tag_id, table._det)
    try:
        saved = json.loads(open(table.cal_path).read()).get("markers_px", {}).get("tag")
    except (OSError, ValueError):
        saved = None
    if c is None or saved is None:
        return True, (f"one-tag calibration loaded ({w:g} x {h:g} cm); tag not in view, drift not checked "
                      f"(put it down and say 'recalibrate' if the camera moved)")
    err = float(np.linalg.norm(table.px_to_cm(c) - table.px_to_cm(np.array(saved)), axis=1).max())
    msg = f"tag {table.tag_id}, max drift {err:.2f} cm"
    if err >= MAX_MARKER_CM:
        msg += ": recalibrate (python -m core.table, or say 'recalibrate')"
    return err < MAX_MARKER_CM, msg


def check_laser(rig: Rig) -> Result:
    laser = rig.part("laser")
    if laser.fit is None:
        return False, f"not calibrated ({laser.cal_path} missing): run python -m act.calibrate --rig"
    fe = laser.fit.fit_error_cm or {}
    med, mx = float(fe.get("median", math.inf)), float(fe.get("max", math.inf))
    w, h = laser.table_size
    try:
        err = laser.aim((w / 2, h / 2))
    finally:
        laser.off()
    msg = f"fit {med:.2f} cm median ({mx:.2f} max, {laser.fit.n_points} pts); centre test "
    msg += "dot not seen" if math.isinf(err) else f"{err:.2f} cm"
    return med < MAX_LASER_FIT_CM and err < MAX_LASER_AIM_CM, msg


def check_audio(rig: Rig) -> Result:
    if rig.ask("Mic: press Enter, then say a sentence for 2 s ") is not None:
        a, speech = rig.record(2.0), True
    else:
        a, speech = rig.record(1.0), False
    blocks = [a[i:i + 512] for i in range(0, len(a) - 511, 512)] or [a]
    loud = max(dbfs(b) for b in blocks)
    peak = float(np.max(np.abs(a))) if len(a) else 0.0
    problems = []
    if loud < SILENT_DBFS:
        problems.append("mic sends silence (muted, or wrong stt.input_device)")
    elif speech and loud < SPEECH_DBFS:
        problems.append(f"speech too quiet (< {SPEECH_DBFS:.0f} dBFS): raise the mic gain")
    if peak >= 0.99:
        problems.append("clipping: lower the mic gain")
    mic = f"mic loudest {loud:.0f} dBFS" + ("" if speech else " (no speech test)")

    rate = 22050
    t = np.arange(int(0.5 * rate)) / rate
    tone = (0.3 * 32767 * np.sin(2 * np.pi * 660 * t) * np.minimum(1, (0.5 - t) * 20)).astype(np.int16)
    rig.play(tone, rate)
    heard = rig.ask("Speaker: did you hear the tone? [Y/n] ")
    if heard is not None and heard.startswith("n"):
        problems.append("tone not heard: check the speaker and the default output device")
    spk = "tone played" + ("" if heard is None else ", heard" if not heard.startswith("n") else ", not heard")
    return not problems, f"{mic}; {spk}" + (f"; {'; '.join(problems)}" if problems else "")


def direct_probe(url: str, timeout: float = 1.5) -> bool:
    """A plain TCP connect to the check host, independent of NetMonitor's HTTPS HEAD."""
    u = urlparse(url)
    try:
        socket.create_connection((u.hostname, u.port or (443 if u.scheme == "https" else 80)), timeout).close()
        return True
    except OSError:
        return False


def check_network(rig: Rig, netmon=None, probe: Callable[[str], bool] = direct_probe) -> Result:
    import core.world
    import net
    netmon = netmon or net.NetMonitor(rig.cfg)
    world = core.world.World(rig.cfg)
    netmon.on_change(lambda online: setattr(world, "online", online))    # as main.py wires it
    world.online = netmon.online                            # both start offline, as in main.py
    now = netmon.check_once()
    direct = probe(netmon.host)
    shown = world.state_json()["online"]
    mode = "online (Grok, ElevenLabs)" if now else "offline (templates, Piper)"
    if now != direct:
        return False, f"monitor says {'online' if now else 'offline'}, direct probe of {netmon.host} disagrees"
    if shown != now:
        return False, f"monitor says {mode} but the dashboard shows {'online' if shown else 'offline'}"
    return True, f"{mode}, dashboard agrees"


def check_world(rig: Rig) -> Result:
    import core.hands
    import core.world
    det = rig.part("detector")
    world = core.world.World(rig.cfg)
    hands = core.hands.HandTracker(frame_size=tuple(rig.cfg["frame_size_px"]))
    world.reset()

    def step(f):
        d = det.detect(f)
        d.hands = hands.update(d.hands, d.t)
        world.update(d, f)

    n, _ = run_frames(rig, 3.0, step)
    if n == 0:
        return False, "no frames from the camera"
    bad = []
    for o in rig.cfg["objects"]:
        e = world.get(o)
        if str(e.status) != "VISIBLE":
            bad.append(f"{o} {e.status}" + (f" ({e.parent})" if e.parent else ""))
        elif o in rig.home_cm and e.pos_cm is not None:
            off = math.dist(e.pos_cm, rig.home_cm[o])
            if off > rig.home_tol_cm:
                bad.append(f"{o} {off:.0f} cm from home")
    where = "at home" if rig.home_cm else "(no demo_check.home_cm set)"
    if bad:
        return False, f"reset, {n} frames; not ready: {', '.join(bad)}"
    return True, f"reset, all {len(rig.cfg['objects'])} VISIBLE {where}"


def check_kill_switch(rig: Rig) -> Result:
    if rig.fake:
        return None, "skipped: no kill switch in --fake"
    if not rig.manual:
        return None, "skipped: manual check (--skip-manual)"
    laser = rig.part("laser")
    laser.act.laser(True)
    try:
        rig.ask("Laser is ON. Press the kill switch, then Enter ")
        ans = rig.ask("Did the dot go out? [y/N] ")
    finally:
        laser.off()
    if ans and ans.startswith("y"):
        return True, "kill switch cuts the laser"
    return False, "dot stayed on: the kill switch must cut laser power before the demo"


def check_clock(rig: Rig) -> Result:
    import main
    import net
    behind = net.clock_behind(main.clock_files(rig.cfg))
    if behind is not None:
        return False, (f"{behind / 60:.0f} min behind the last saved file: spoken times will be wrong; "
                       "join the hotspot (NTP) or `sudo date -s`")
    return True, time.strftime("%a %b %d %H:%M %Z")


def check_room_memory(rig: Rig) -> Result:
    """Room memory (spec 0009 M0), when enabled: table_view_rect is measured, and the zones file exists, has
    zones and was drawn at this view (capture size, zoom, table_view_rect). Files only: no camera."""
    from core.room_types import RoomConfig
    rc = RoomConfig.from_dict(rig.cfg.get("room_memory"))
    if not rc.enabled:
        return None, "room memory off (room_memory.enabled: false)"
    from core.room_zones import Zones, view_version
    draw = "python -m core.room --zone NAME --say TEXT --poly x,y x,y x,y"
    if rc.table_view_rect is None:
        return False, ("no room_memory.table_view_rect: measure it (python -m core.room --measure-rect "
                       "--full FULL.jpg --ref REF.jpg) and put it in config.local.yaml")
    try:
        zones = Zones.load(rc.zones_path)
    except FileNotFoundError:
        return False, f"no {rc.zones_path}: draw the zones ({draw})"
    if not zones.zones:
        return False, f"{rc.zones_path} has no zones: draw them ({draw})"
    view = view_version(rc.capture_size, rc.zoom, rc.table_view_rect)
    if zones.view != view:
        return False, (f"zones drawn at view {zones.view}, the config gives {view} (capture size, zoom or "
                       f"table_view_rect changed): redraw them ({draw})")
    names = ", ".join(f"{n} ({z.say})" for n, z in zones.zones.items())
    return True, f"{len(zones.zones)} zones at view {view}: {names}"


def check_room(rig: Rig) -> Result:
    """Room pointing (spec 0006), when enabled: the map matches the camera, has zones, and three mapped
    dots spread over the room are hit again. The first look (the map's open-loop guess) must land within
    2 x tol_px: the loop would converge anyway, so that's what catches a moved camera or head."""
    if not (rig.cfg.get("room") or {}).get("enabled"):
        return None, "room pointing off (room.enabled: false)"
    laser, frames = rig.part("room")
    rm = laser.room_map
    f = frames.latest()
    if f is not None and f.img is not None and (f.img.shape[1], f.img.shape[0]) != rm.size_px:
        return False, f"map is {rm.size_px[0]}x{rm.size_px[1]}, camera is {f.img.shape[1]}x{f.img.shape[0]}: sweep again"
    idx = np.nonzero(rm.seen)[0]
    if len(idx) < 10:
        return False, f"only {len(idx)} dots in the map: sweep again with the room lit normally"
    picks = [rm.px[idx[int(q * (len(idx) - 1))]] for q in (0.2, 0.5, 0.8)]
    errs = []
    try:
        for uv in picks:
            r = laser.aim_px(uv)
            errs.append(r.first_err_px if r.on_target and r.first_err_px is not None else math.inf)
    finally:
        laser.off()
    shown = ", ".join("not hit" if math.isinf(e) else f"{e:.1f}" for e in errs)
    zones = ", ".join(rm.zones) or "none"
    if any(e > 2 * laser.tol_px for e in errs):
        return False, (f"map guess off by {shown} px: camera or head moved? sweep again "
                       "(python -m act.room_map --sweep)")
    if not rm.zones and not rig.fake:
        return False, f"re-hit {shown} px, but no zones: draw them (python -m act.room_map --zone NAME --poly ...)"
    return True, f"{rm.n_seen} dots, zones: {zones}; re-hit {shown} px"


# ---------------------------------------------------------------- the running room app (spec 0010 P1-3)

def _rc(rig: Rig) -> dict:
    return rig.cfg.get("room_check") or {}


def saved_port(path: str = "data/room/app.args") -> Optional[int]:
    """--port from the args scripts/room_app.sh started the app with (one per line), if any."""
    try:
        args = open(path).read().split("\n")
    except OSError:
        return None
    for i, a in enumerate(args):
        v = a[len("--port="):] if a.startswith("--port=") else (args[i + 1] if a == "--port" and i + 1 < len(args) else "")
        if v.strip().isdigit():
            return int(v)
    return None


def app_url(cfg: dict, args_path: str = "data/room/app.args") -> str:
    """room_check.app_url; else the --port scripts/room_app.sh saved; else server.port."""
    url = str((cfg.get("room_check") or {}).get("app_url") or "").strip()
    if url:
        return url.rstrip("/")
    port = saved_port(args_path) or (cfg.get("server") or {}).get("port", 8000)
    return f"http://127.0.0.1:{port}"


def check_room_app(rig: Rig, get: Optional[Callable] = None) -> Result:
    """The running app (not this process) answers: /state is fresh (perception alive, its fps), and with
    room memory on, /full.jpg serves the full camera frame (the room pipeline is up)."""
    if rig.fake and get is None:
        return None, "skipped: no running app in --fake"
    import requests
    get = get or requests.get
    url = app_url(rig.cfg)
    try:
        st = get(f"{url}/state", timeout=5).json().get("state") or {}
    except Exception as e:                                  # noqa: BLE001
        return False, (f"no room app at {url} ({type(e).__name__}): start it (scripts/room_app.sh start) "
                       "or check room_check.app_url")
    age = time.time() - float(st.get("t") or 0.0)
    fps = float(st.get("fps") or 0.0)
    max_age = float(_rc(rig).get("max_frame_age_s", 5))
    msg = f"app at {url}: {fps:.1f} fps, state {age:.1f} s old"
    if age > max_age or fps <= 0:
        return False, msg + ": perception has stalled (camera unplugged or busy?): restart the app"
    if not (rig.cfg.get("room_memory") or {}).get("enabled"):
        return True, msg + "; room memory off"
    try:
        r = get(f"{url}/full.jpg", timeout=10)
        ok = r.status_code == 200 and r.content[:2] == b"\xff\xd8"
    except Exception as e:                                  # noqa: BLE001
        return False, msg + f"; /full.jpg failed ({type(e).__name__})"
    if not ok:
        return False, msg + f"; no full camera frame (HTTP {r.status_code}): room memory isn't running"
    places = [n for n, v in (st.get("room") or {}).items() if isinstance(v, dict)]
    return True, msg + f"; room memory up ({len(places)} object(s) in zones)"


def _xai_key() -> str:
    """XAI_API_KEY from the environment, else from ./.env (never printed)."""
    key = os.environ.get("XAI_API_KEY", "").strip()
    if key or not os.path.isfile(".env"):
        return key
    for ln in open(".env", encoding="utf-8", errors="ignore"):
        k, _, v = ln.strip().partition("=")
        if k.strip().removeprefix("export ").strip() == "XAI_API_KEY":
            return v.strip().strip("'\"")
    return ""


def check_grok(rig: Rig, get: Optional[Callable] = None) -> Result:
    """api.x.ai reachable with our key: GET /models (no tokens), timed. Room naming stops without it."""
    import requests

    from core.xai import BASE_URL
    injected, get = get is not None, get or requests.get
    key = _xai_key()
    if not key:
        if rig.fake and not injected:
            return None, "skipped: no XAI_API_KEY in --fake"
        return False, "no XAI_API_KEY (environment or .env): Grok naming and answers are off"
    base = ((rig.cfg.get("llm") or {}).get("base_url") or BASE_URL).rstrip("/")
    t0 = time.monotonic()
    try:
        r = get(f"{base}/models", headers={"Authorization": f"Bearer {key}"},
                timeout=float(_rc(rig).get("grok_timeout_s", 10)))
    except Exception as e:                                  # noqa: BLE001
        return False, (f"{base} unreachable ({type(e).__name__}): offline, room naming stops. Switch to the "
                       "phone hotspot (docs/HOTSPOT.md)")
    ms = (time.monotonic() - t0) * 1000
    if r.status_code in (401, 403):
        return False, f"Grok refused the key (HTTP {r.status_code}) in {ms:.0f} ms: check XAI_API_KEY"
    if r.status_code >= 400:
        return False, f"Grok answered HTTP {r.status_code} in {ms:.0f} ms"
    slow = "; slow: answers will need the thinking cue" if ms > 1500 else ""
    return True, f"Grok reachable, {ms:.0f} ms round trip{slow}"


def _find_device(devices: list, spec, kind: str) -> tuple[Optional[dict], str]:
    """(device, why) for a config spec: an index, or part of a name (case-insensitive), among devices
    that have `kind` channels."""
    ch = "max_input_channels" if kind == "input" else "max_output_channels"
    cands = [dict(d, index=i) for i, d in enumerate(devices) if d.get(ch, 0) > 0]
    if isinstance(spec, int) or str(spec).strip().isdigit():
        hit = [d for d in cands if d["index"] == int(spec)]
        return (hit[0], "") if hit else (None, f"no {kind} device #{spec}")
    hit = [d for d in cands if str(spec).strip().lower() in str(d.get("name", "")).lower()]
    if not hit:
        return None, f"no {kind} device named like {spec!r}"
    return hit[0], (f" ({len(hit)} match, first used)" if len(hit) > 1 else "")


def asound_cards(root: str = "/proc/asound") -> list[dict]:
    """ALSA's own card list: [{'num', 'name' (id, driver, long names: matched), 'label' (shown), 'capture',
    'playback', 'busy_c', 'busy_p'}]. Unlike PortAudio, it lists a card the running app has open, and says so (a substream in
    state RUNNING)."""
    import glob
    import re
    try:
        lines = open(os.path.join(root, "cards")).read().splitlines()
    except OSError:
        return []
    cards = []
    for i, ln in enumerate(lines):
        m = re.match(r"\s*(\d+)\s+\[([^\]]*)\]:\s*(.*)", ln)
        if not m:
            continue
        num = int(m.group(1))
        more = lines[i + 1].strip() if i + 1 < len(lines) and not re.match(r"\s*\d+\s+\[", lines[i + 1]) else ""
        card = {"num": num, "name": f"{m.group(2).strip()} {m.group(3).strip()} {more}".strip(),
                "label": m.group(3).split(" - ", 1)[-1].strip()}
        for d in ("c", "p"):
            pcms = glob.glob(os.path.join(root, f"card{num}", f"pcm*{d}"))
            card["capture" if d == "c" else "playback"] = bool(pcms)
            busy = False
            for st in glob.glob(os.path.join(root, f"card{num}", f"pcm*{d}", "sub*", "status")):
                try:
                    busy |= "RUNNING" in open(st).read()
                except OSError:
                    pass
            card["busy_" + d] = busy
        cards.append(card)
    return cards


def _check_devices_alsa(rig: Rig, root: str) -> Result:
    """--live: the app holds the mic and speaker, and PortAudio hides a busy device, so look in ALSA's card
    list instead. A card in use counts as present (the app has it)."""
    cards = asound_cards(root)
    if not cards:
        return None, f"no {root}/cards (not Linux?): devices not checked"
    parts, problems = [], []
    for kind, section, key in (("input", "stt", "input_device"), ("output", "tts", "output_device")):
        spec = (rig.cfg.get(section) or {}).get(key)
        label = "mic" if kind == "input" else "speaker"
        d = "c" if kind == "input" else "p"
        if spec is None or (isinstance(spec, str) and not spec.strip()):
            problems.append(f"{section}.{key} not set: the {label} would be the default "
                            f"({'Brio in the corner' if kind == 'input' else 'HDMI, silent'})")
            continue
        if isinstance(spec, int) or str(spec).strip().isdigit():
            problems.append(f"{section}.{key} is an index ({spec}): indexes shift on replug, set part of the "
                            f"{label}'s name instead")
            continue
        hits = [c for c in cards if str(spec).strip().lower() in c["name"].lower()
                and c["capture" if kind == "input" else "playback"]]
        if not hits:
            problems.append(f"no {kind} card named like {spec!r}: plug the {label} in (cat {root}/cards)")
            continue
        c = hits[0]
        parts.append(f"{label} card {c['num']} {c['label']}"
                     + (" (in use by the app)" if c["busy_" + d] else ""))
    return not problems, "; ".join(parts + problems)


def check_devices(rig: Rig, query: Optional[Callable[[], list]] = None, asound: str = "/proc/asound") -> Result:
    """The mic and speaker named in config are plugged in. Null means the default device, which on the
    rig is the Brio's mic in the corner and HDMI (no speaker), so both must be named. --live reads ALSA's
    card list, since the running app has the devices open."""
    if rig.live and query is None and not rig.fake:
        return _check_devices_alsa(rig, asound)
    if query is None:
        if rig.fake:
            return None, "skipped: no audio devices in --fake"
        import sounddevice as sd
        query = lambda: list(sd.query_devices())            # noqa: E731
    devices = query()
    parts, problems = [], []
    for kind, section, key in (("input", "stt", "input_device"), ("output", "tts", "output_device")):
        spec = (rig.cfg.get(section) or {}).get(key)
        label = "mic" if kind == "input" else "speaker"
        if spec is None or (isinstance(spec, str) and not spec.strip()):
            problems.append(f"{section}.{key} not set: the {label} would be the default "
                            f"({'Brio in the corner' if kind == 'input' else 'HDMI, silent'})")
            continue
        d, why = _find_device(devices, spec, kind)
        if d is None:
            problems.append(f"{why}: plug the {label} in (python -m voice.tts --devices lists them)")
        else:
            parts.append(f"{label} #{d['index']} {d.get('name')}{why}")
    msg = "; ".join(parts + problems)
    return not problems, msg


def meminfo_available_mb(path: str = "/proc/meminfo") -> Optional[float]:
    try:
        for ln in open(path):
            if ln.startswith("MemAvailable:"):
                return int(ln.split()[1]) / 1024
    except OSError:
        return None
    return None


def log_tail(path: str, n: int) -> Optional[list[str]]:
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(max(0, size - 400 * n))                  # ~400 bytes a line is plenty
            return f.read().decode(errors="replace").splitlines()[-n:]
    except OSError:
        return None


def check_memory(rig: Rig, meminfo: str = "/proc/meminfo") -> Result:
    """RAM left with the app running, and the app log's tail free of GPU allocation failures
    (NvMapMemAlloc) and tracebacks. The Orin Nano's 8 GB are shared with the GPU."""
    if rig.fake and meminfo == "/proc/meminfo":
        return None, "skipped: no room app in --fake"
    rc = _rc(rig)
    avail = meminfo_available_mb(meminfo)
    need = float(rc.get("min_ram_mb", 1024))
    problems, parts = [], []
    if avail is None:
        parts.append("RAM not readable here (no /proc/meminfo)")
    else:
        parts.append(f"{avail / 1024:.1f} GB available")
        if avail < need:
            problems.append(f"under {need / 1024:.1f} GB: turn the visual-memory archive or narration off, "
                            "or restart the app")
    path, n = str(rc.get("app_log", "data/room/app.log")), int(rc.get("log_lines", 500))
    tail = log_tail(path, n)
    if tail is None:
        problems.append(f"no app log at {path}")
    else:
        nvmap = sum("NvMapMemAlloc" in ln for ln in tail)
        tb = sum(ln.startswith("Traceback") for ln in tail)
        parts.append(f"log tail ({len(tail)} lines): {nvmap} NvMapMemAlloc, {tb} tracebacks")
        if nvmap:
            problems.append("GPU memory allocation failures in the log: the app is short of memory")
        if tb:
            problems.append(f"tracebacks in {path}: read them")
    return not problems, "; ".join(parts + problems)


def check_namer(rig: Rig) -> Result:
    """The room namer (Grok names for room crops) keeps a bounded queue: a burst of 100 crops from a busy
    room must not grow memory or starve newer objects. Checked on the code the app runs, offline."""
    from core.room import RoomNamer
    namer = RoomNamer(lambda img: None, online=lambda: False, start=False)
    img = np.zeros((8, 8, 3), np.uint8)
    track = type("T", (), {"tid": "t", "zone": "z", "guess": None})()
    for _ in range(100):
        namer.submit(track, img)
    n = namer.pending()
    cap = (rig.cfg.get("room_memory") or {}).get("names_per_minute", "?")
    return n <= 8, f"100 crops queued offline -> {n} kept (cap 8); Grok calls <= {cap}/min"


LIVE_CHECKS = {"network", "clock", "room memory", "room app", "grok", "devices", "memory", "namer"}

CHECKS: list[tuple[str, Callable[[Rig], Result]]] = [
    ("camera", check_camera),
    ("detector", check_detector),
    ("markers", check_markers),
    ("laser", check_laser),
    ("audio", check_audio),
    ("network", check_network),
    ("world", check_world),
    ("kill switch", check_kill_switch),
    ("clock", check_clock),
    ("room memory", check_room_memory),
    ("room", check_room),
    ("room app", check_room_app),
    ("grok", check_grok),
    ("devices", check_devices),
    ("memory", check_memory),
    ("namer", check_namer),
]


def run_check(rig: Rig, fn: Callable[[Rig], Result]) -> Result:
    try:
        return fn(rig)
    except Exception as e:                         # noqa: BLE001 - the reason goes on the red line
        return False, f"{type(e).__name__}: {e}"


def deadline_for(name: str, rig: Rig) -> float:
    if rig.manual and name in ("audio", "kill switch"):
        return MANUAL_DEADLINE_S
    return DEADLINE_S.get(name, DEFAULT_DEADLINE_S)


def run_check_with_deadline(rig: Rig, name: str, fn: Callable[[Rig], Result],
                            timeout: Optional[float] = None) -> Result:
    """run_check on a daemon thread. Past the deadline the check fails "timed out", its stack goes to
    stderr, and any part it was still building is marked failed so later checks don't wait on it too."""
    timeout = deadline_for(name, rig) if timeout is None else timeout
    box: list[Result] = []
    t = threading.Thread(target=lambda: box.append(run_check(rig, fn)), name=f"check-{name}", daemon=True)
    try:                                                   # a long check shows where it is, every minute
        faulthandler.dump_traceback_later(60, repeat=True, file=sys.__stderr__)
    except (AttributeError, OSError, ValueError):          # no real stderr (e.g. under a test runner)
        pass
    try:
        t.start()
        t.join(timeout)
    finally:
        faulthandler.cancel_dump_traceback_later()
    if box:
        return box[0]
    frame = sys._current_frames().get(t.ident or -1)
    where = ""
    if frame is not None:
        stack = traceback.extract_stack(frame)
        print(f"--- check {name!r} stuck after {timeout:.0f} s:", file=sys.stderr)
        print("".join(traceback.format_list(stack)), file=sys.stderr, flush=True)
        where = f" in {stack[-1].name} ({os.path.basename(stack[-1].filename)}:{stack[-1].lineno})"
    if rig.prompts is not None:
        rig.prompts.abandon(t)                        # it can't take a later check's answer
    for part in list(rig.building):
        rig._parts[part] = TimeoutError(f"{part} hung while starting (check {name!r} timed out)")
    return False, f"timed out after {timeout:.0f} s{where}"


def line(i: int, name: str, ok: Optional[bool], msg: str, color: bool = True) -> str:
    tag, c = ("PASS", GREEN) if ok else ("SKIP", YELLOW) if ok is None else ("FAIL", RED)
    s = f"[{tag}] {i} {name:<12s} {msg}"
    return f"{c}{s}{RESET}" if color else s


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fake", action="store_true", help="no hardware: rendered layout, simulated laser")
    ap.add_argument("--skip-manual", action="store_true", help="no prompts; the kill switch is skipped")
    ap.add_argument("--camera", default=None,
                    help="camera index or /dev/v4l/by-id/ path (default: config demo_check.camera)")
    ap.add_argument("--only", type=int, nargs="*", help="run only these check numbers")
    ap.add_argument("--live", action="store_true",
                    help="the room app is running: only the checks that leave it the camera and mic")
    ap.add_argument("--config")
    a = ap.parse_args(argv)

    cfg = load_config(a.config)
    camera = str(a.camera if a.camera is not None else (cfg.get("demo_check") or {}).get("camera", 0)).strip()
    camera = int(camera) if camera.isdigit() else camera       # an index moves on replug; a by-id path does not
    manual = not a.skip_manual and sys.stdin.isatty()
    color = sys.stdout.isatty()
    rig = Rig(cfg, fake=a.fake, camera=camera, manual=manual)
    rig.live = a.live
    failed = 0
    try:
        for i, (name, fn) in enumerate(CHECKS, 1):
            if a.only and i not in a.only:
                continue
            if a.live and name not in LIVE_CHECKS:
                continue
            ok, msg = run_check_with_deadline(rig, name, fn)
            print(line(i, name, ok, msg, color), flush=True)
            failed += ok is False
    finally:
        rig.laser_off_now()                                 # even if a hung check holds the actuator
        closer = threading.Thread(target=rig.close, name="close", daemon=True)
        closer.start()
        closer.join(10.0)
        if closer.is_alive():
            print("(cleanup still running after 10 s; exiting anyway)", file=sys.stderr)
    print(f"{failed} failed" if failed else "all checks passed", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    code = main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)           # a hung check's thread or an audio library's exit hook can't hold the process
