"""Pan-tilt laser head bring-up (PCA9685 via act/actuator.py): I2C check, servo range sweep, keypress jog.

    scripts/dock.sh python3 scripts/laser_bringup.py check
    scripts/dock.sh python3 scripts/laser_bringup.py range                  # pan, then tilt, over servo_limits
    scripts/dock.sh python3 scripts/laser_bringup.py range --center         # just centre both servos
    scripts/dock.sh python3 scripts/laser_bringup.py jog                    # find the limits, laser off
    scripts/dock.sh python3 scripts/laser_bringup.py jog --laser --eye-safe-confirmed --max-on-s 4

No camera. Uses actuator: / servo_limits / *_channel from config.yaml + config.local.yaml, so set
actuator: pca9685 there first. The laser is off at start and on every exit; it can only be lit with
--laser AND --eye-safe-confirmed (the module is Class 2, < 1 mW), and each on period is cut at --max-on-s.
The live app may share the board: opening it re-inits the PCA9685, so its servos go limp until it moves again.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Callable, Iterable, Iterator, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from act.actuator import make_actuator_or_fake  # noqa: E402

STEPS_US = (5, 10, 25, 50, 100)            # jog step per key press
MOVES = {"a": (-1, 0), "left": (-1, 0), "d": (1, 0), "right": (1, 0),
         "w": (0, -1), "up": (0, -1), "s": (0, 1), "down": (0, 1)}
HELP = """  a / d or left / right  pan      w / s or up / down  tilt      + / -  bigger / smaller step
  c  centre   p  print pulses   [ or ]  mark this position as a limit   l  laser on / off   q  quit"""


# ---------------------------------------------------------------- check

def parse_i2cdetect(text: str, addr: int) -> bool:
    """True if `i2cdetect -y -r N` output shows `addr` (its hex, or UU: claimed by a kernel driver)."""
    for line in text.splitlines():
        row, sep, cells = line.partition(":")
        if sep and row.strip() and all(ch in "0123456789abcdefABCDEF" for ch in row.strip()):
            base = int(row.strip(), 16)
            if any(base + i == addr and c.lower() in (f"{addr:02x}", "uu") for i, c in enumerate(cells.split())):
                return True
    return False


def i2c_probe(bus: int = 7, addr: int = 0x40, exists: Callable[[str], bool] = os.path.exists,
              run: Callable = subprocess.run) -> tuple[bool, str]:
    """(found, message). smbus2 if importable, else i2cdetect; never raises."""
    dev = f"/dev/i2c-{bus}"
    if not exists(dev):
        return False, f"{dev} missing (not on this Jetson, or not passed into the container)"
    try:
        import smbus2
        with smbus2.SMBus(bus) as b:
            b.read_byte_data(addr, 0x00)                  # MODE1: a harmless read
        return True, f"0x{addr:02x} answers on {dev} (smbus2)"
    except ImportError:
        pass                                              # no smbus2: try i2cdetect
    except OSError as e:
        return False, f"no answer from 0x{addr:02x} on {dev} (smbus2: {e}): check SDA/SCL, VCC, GND"
    try:
        r = run(["i2cdetect", "-y", "-r", str(bus)], capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError) as e:
        return False, f"can't probe {dev}: no smbus2 and i2cdetect failed ({type(e).__name__}: {e})"
    if parse_i2cdetect(r.stdout or "", addr):
        return True, f"0x{addr:02x} answers on {dev} (i2cdetect)"
    return False, f"0x{addr:02x} not seen on {dev} (i2cdetect rc {r.returncode}): check wiring and power"


def cmd_check(cfg: dict, out=print, probe=i2c_probe) -> int:
    addr = int(cfg.get("pca9685_address", 0x40))
    found, msg = probe(7, addr)
    out(("OK   " if found else "FAIL ") + msg)
    act, why = make_actuator_or_fake(cfg)
    try:
        act.laser(False)
        out(f"driver: {type(act).__name__} (config actuator: {cfg.get('actuator')!r})")
        if why:
            out(f"fell back to FakeActuator: {why}")
        elif str(cfg.get("actuator", "fake")).lower() == "fake":
            out("config says actuator: fake, so nothing moves: set actuator: pca9685 in config.local.yaml")
        out(f"channels pan {cfg.get('pan_channel', 0)} tilt {cfg.get('tilt_channel', 1)} laser "
            f"{cfg.get('laser_channel', 15)}; limits {act.limits()}")
    finally:
        act.close()
    return 0 if found else 1


# ---------------------------------------------------------------- laser guard

class LaserGuard:
    """The only way this script lights the laser: needs `allowed`, and cuts each on period at max_on_s
    (a timer thread in real use; poll() applies it against `now` for tests and between keys)."""

    def __init__(self, act, allowed: bool, max_on_s: float = 4.0, now=time.monotonic,
                 timer: Optional[Callable] = threading.Timer, out=print):
        if max_on_s <= 0:
            raise ValueError("--max-on-s must be > 0")
        self.act, self.allowed, self.max_on_s, self.now, self.timer, self.out = act, allowed, max_on_s, now, timer, out
        self.on_since: Optional[float] = None               # clock time it was lit, None while off
        self._t = None
        if hasattr(act, "laser_timeout_s"):              # the actuator's own watchdog, as a second cut-off
            act.laser_timeout_s = min(float(act.laser_timeout_s), max_on_s)
        self.off()

    is_on = property(lambda self: self.on_since is not None)

    def on(self) -> bool:
        if not self.allowed:
            self.out("laser refused: needs --laser and --eye-safe-confirmed (Class 2, < 1 mW module)")
            return False
        self.act.laser(True)
        self.on_since = self.now()
        if self.timer is not None:
            self._t = self.timer(self.max_on_s, self._expire)
            self._t.daemon = True
            self._t.start()
        self.out(f"laser ON (off in {self.max_on_s:g} s)")
        return True

    def _expire(self) -> None:
        if self.is_on:
            self.off()
            self.out(f"laser auto-off after {self.max_on_s:g} s")

    def off(self) -> None:
        if self._t is not None:
            self._t.cancel()
            self._t = None
        self.on_since = None
        self.act.laser(False)                            # always written, even if we think it's off

    def toggle(self) -> None:
        self.off() if self.is_on else self.on()

    def poll(self) -> None:
        if self.is_on and self.now() - self.on_since >= self.max_on_s:
            self._expire()


# ---------------------------------------------------------------- range / jog

def clamp(act, pan: float, tilt: float) -> tuple[float, float]:
    (plo, phi), (tlo, thi) = act.limits()
    return min(phi, max(plo, pan)), min(thi, max(tlo, tilt))


def centre(act) -> tuple[float, float]:
    return tuple((lo + hi) / 2 for lo, hi in act.limits())


def sweep(act, steps: int = 7, pause_s: float = 0.6, sleep=time.sleep, out=print) -> None:
    """Pan over its limits, then tilt, in `steps` stops each (the other servo centred), laser off."""
    steps = max(2, steps)
    (plo, phi), (tlo, thi) = act.limits()
    pc, tc = centre(act)
    act.laser(False)
    for axis, lo, hi in (("pan", plo, phi), ("tilt", tlo, thi)):
        for i in range(steps):
            v = lo + (hi - lo) * i / (steps - 1)
            p, t = clamp(act, v, tc) if axis == "pan" else clamp(act, pc, v)
            out(f"  {axis}: pan {p:.0f} tilt {t:.0f}")
            act.move(p, t, duration_s=0.8)
            sleep(pause_s)
        act.move(pc, tc, duration_s=0.8)


def jog(act, keys: Iterable[str], guard: LaserGuard, out=print, step_us: float = 25):
    """Keypress jog. Returns ((pan, tilt) at the end, [marked (pan, tilt)]). Laser off on every exit."""
    pan, tilt = centre(act)
    si = STEPS_US.index(step_us) if step_us in STEPS_US else 2
    marks: list[tuple[float, float]] = []
    try:
        act.move(pan, tilt, duration_s=0.5)
        out(HELP)
        for k in keys:
            guard.poll()
            if k in ("q", "esc"):
                break
            if k in MOVES:
                dp, dt = MOVES[k]
                pan, tilt = clamp(act, pan + dp * STEPS_US[si], tilt + dt * STEPS_US[si])
                act.move(pan, tilt, duration_s=0.1)
            elif k in ("+", "="):
                si = min(len(STEPS_US) - 1, si + 1)
            elif k in ("-", "_"):
                si = max(0, si - 1)
            elif k == "c":
                pan, tilt = centre(act)
                act.move(pan, tilt, duration_s=0.5)
            elif k in ("[", "]"):
                marks.append((pan, tilt))
                out(f"  marked pan {pan:.0f} tilt {tilt:.0f}")
            elif k == "l":
                guard.toggle()
            elif k != "p":
                continue
            out(f"  pan {pan:.0f}  tilt {tilt:.0f}  step {STEPS_US[si]} us  laser {'ON' if guard.is_on else 'off'}")
    finally:
        guard.off()
    return (pan, tilt), marks


def limits_yaml(act, current: tuple[float, float], marks: list[tuple[float, float]]) -> str:
    """servo_limits snippet for config.local.yaml: the marks' min/max per axis (an axis with no spread
    keeps its configured limits), or just the current position if nothing was marked."""
    if not marks:
        return f"# no limits marked ([ or ]); current position: pan {current[0]:.0f}, tilt {current[1]:.0f}"
    lines = ["servo_limits:"]
    for i, (axis, cfg_lim) in enumerate(zip(("pan", "tilt"), act.limits())):
        lo, hi = min(m[i] for m in marks), max(m[i] for m in marks)
        lines.append(f"  {axis}: [{lo:.0f}, {hi:.0f}]" if hi > lo else
                     f"  {axis}: [{cfg_lim[0]:.0f}, {cfg_lim[1]:.0f}]   # not marked: configured limits")
    return "\n".join(lines)


def line_keys(stream=sys.stdin) -> Iterator[str]:
    """Non-tty fallback: one key name per line ('left', 'q') or each character of the line ('ddd')."""
    for line in stream:
        s = line.strip().lower()
        if s in MOVES or s == "esc":
            yield s
        else:
            yield from s


# ---------------------------------------------------------------- main

def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", help="config.yaml path (config.local.yaml next to it overrides)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("check", help="I2C 0x40 + which actuator driver loads")
    r = sub.add_parser("range", help="sweep pan then tilt over servo_limits, laser off")
    r.add_argument("--steps", type=int, default=7)
    r.add_argument("--pause", type=float, default=0.6, help="seconds at each stop")
    r.add_argument("--center", action="store_true", help="only centre both servos")
    r.add_argument("--yes", action="store_true", help="don't ask for Enter first")
    j = sub.add_parser("jog", help="keypress jog to find servo_limits")
    j.add_argument("--laser", action="store_true", help="allow l to light the laser ...")
    j.add_argument("--eye-safe-confirmed", action="store_true", help="... and confirm it is Class 2, < 1 mW")
    j.add_argument("--max-on-s", type=float, default=4.0, help="each laser-on period is cut at this")
    args = ap.parse_args(argv)

    from core.config import load_config
    cfg = load_config(args.config)
    if args.cmd == "check":
        return cmd_check(cfg)
    act, why = make_actuator_or_fake(cfg)
    act.laser(False)
    print(f"driver: {type(act).__name__}" + (f" (fell back: {why})" if why else ""))
    try:
        if args.cmd == "range":
            what = "centre both servos" if args.center else f"sweep pan then tilt over {act.limits()}"
            if not args.yes:
                input(f"Will {what}, laser off. Keep hands clear, Enter to go (Ctrl-C to stop) ")
            if args.center:
                act.move(*centre(act), duration_s=0.8)
                print(f"  centred: pan {act.pan:.0f} tilt {act.tilt:.0f}")
            else:
                sweep(act, args.steps, args.pause)
            return 0
        allowed = args.laser and args.eye_safe_confirmed
        if args.laser and not allowed:
            print("--laser without --eye-safe-confirmed: the laser stays off")
        guard = LaserGuard(act, allowed, args.max_on_s)
        if sys.stdin.isatty():
            from act.calibrate import terminal_keys
            with terminal_keys() as read_key:
                cur, marks = jog(act, iter(read_key, None), guard)
        else:
            print("stdin is not a terminal: type keys then Enter (e.g. 'ddd', 'left', 'q')")
            cur, marks = jog(act, line_keys(), guard)
        print("\n# config.local.yaml\n" + limits_yaml(act, cur, marks))
        return 0
    except (KeyboardInterrupt, EOFError):
        print("\nstopped")
        return 130
    finally:
        act.laser(False)
        act.close()


if __name__ == "__main__":
    sys.exit(main())
