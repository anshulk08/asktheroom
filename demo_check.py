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

Missing hardware fails that check with the reason, so this also runs on a laptop. Parts (camera,
table, detector, laser) are built on first use and shared; one that fails to build fails every
check that needs it, with the same reason.

--fake: the camera is a FrameBuffer over a rendered home layout (server.sim's painter and LAYOUT,
with ArUco 0-3 drawn in), the detector's backend reads boxes off that layout, the laser is an
act.sim.SimRig calibrated at startup, the mic is a synthetic tone and the speaker is silent. The
network check is real.
"""
from __future__ import annotations

import argparse
import math
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from typing import Callable, Optional
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


# ---------------------------------------------------------------- the parts, built on first use

class Rig:
    """Lazily builds and caches each part. A part that fails to build raises the same error again,
    so every check that needs a missing camera reports the camera, not a knock-on error."""

    def __init__(self, cfg: dict, fake: bool = False, camera: int = 0, manual: bool = True,
                 ask: Callable[[str], str] = input):
        self.fake = fake
        self.cfg = fake_cfg(cfg) if fake else cfg
        self.camera, self.manual, self._ask = camera, manual, ask
        self._parts: dict[str, object] = {}
        self._cleanup: list[Callable[[], None]] = []
        self.tmp = tempfile.mkdtemp(prefix="askroom_check_")
        self.home_cm = (fake_layout(self.cfg) if fake else
                        {o: tuple(v) for o, v in ((cfg.get("demo_check") or {}).get("home_cm") or {}).items()})
        self.home_tol_cm = float((cfg.get("demo_check") or {}).get("home_tol_cm", 5))

    def part(self, name: str):
        got = self._parts.get(name)
        if got is None:
            try:
                got = getattr(self, f"_make_{name}")()
            except Exception as e:                 # noqa: BLE001 - reported as the check's reason
                got = e
            self._parts[name] = got
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
        sd.wait()
        return a[:, 0]

    def play(self, pcm: np.ndarray, rate: int) -> None:
        if self.fake:
            return
        from voice.tts import play_pcm
        play_pcm(pcm, rate)


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

def exposure_mode(device: int) -> tuple[Optional[bool], str]:
    """(locked?, detail) from v4l2-ctl. auto_exposure 1 is manual on UVC cameras."""
    if not sys.platform.startswith("linux"):
        return None, "exposure lock only readable on Linux (v4l2)"
    if shutil.which("v4l2-ctl") is None:
        return None, "v4l2-ctl missing (apt install v4l-utils)"
    out = subprocess.run(["v4l2-ctl", "-d", f"/dev/video{device}", "-C", "auto_exposure",
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
        return False, f"not calibrated ({laser.cal_path} missing): run python -m act.calibrate"
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


CHECKS: list[tuple[str, Callable[[Rig], Result]]] = [
    ("camera", check_camera),
    ("detector", check_detector),
    ("markers", check_markers),
    ("laser", check_laser),
    ("audio", check_audio),
    ("network", check_network),
    ("world", check_world),
    ("kill switch", check_kill_switch),
]


def run_check(rig: Rig, fn: Callable[[Rig], Result]) -> Result:
    try:
        return fn(rig)
    except Exception as e:                         # noqa: BLE001 - the reason goes on the red line
        return False, f"{type(e).__name__}: {e}"


def line(i: int, name: str, ok: Optional[bool], msg: str, color: bool = True) -> str:
    tag, c = ("PASS", GREEN) if ok else ("SKIP", YELLOW) if ok is None else ("FAIL", RED)
    s = f"[{tag}] {i} {name:<12s} {msg}"
    return f"{c}{s}{RESET}" if color else s


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fake", action="store_true", help="no hardware: rendered layout, simulated laser")
    ap.add_argument("--skip-manual", action="store_true", help="no prompts; the kill switch is skipped")
    ap.add_argument("--camera", type=int, default=None, help="camera index (default: config demo_check.camera)")
    ap.add_argument("--only", type=int, nargs="*", help="run only these check numbers")
    ap.add_argument("--config")
    a = ap.parse_args(argv)

    cfg = load_config(a.config)
    camera = a.camera if a.camera is not None else int((cfg.get("demo_check") or {}).get("camera", 0))
    manual = not a.skip_manual and sys.stdin.isatty()
    color = sys.stdout.isatty()
    rig = Rig(cfg, fake=a.fake, camera=camera, manual=manual)
    failed = 0
    try:
        for i, (name, fn) in enumerate(CHECKS, 1):
            if a.only and i not in a.only:
                continue
            ok, msg = run_check(rig, fn)
            print(line(i, name, ok, msg, color), flush=True)
            failed += ok is False
    finally:
        rig.close()
    print(f"{failed} failed" if failed else "all checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
