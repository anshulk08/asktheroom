"""Slow raster over the whole servo range through the Actuator interface (H1/H3 check).

    python scripts/sweep.py                  # uses cfg 'actuator' (default fake), real time
    python scripts/sweep.py --actuator pca9685 --laser --rows 8 --row-s 2.5
    python scripts/sweep.py --fast           # fake actuator on a simulated clock, prints the path

Watch the dot: it should cover the table without hitting the servo end stops. If it leaves the
table, tighten servo_limits in config.yaml. Ctrl-C stops and switches the laser off.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from act.actuator import FakeActuator, SimClock, make_actuator  # noqa: E402
from core.config import load_config  # noqa: E402


def raster(act, rows: int, row_s: float, margin: float, laser: bool, verbose: bool) -> int:
    (plo, phi), (tlo, thi) = act.limits()
    mp, mt = margin * (phi - plo), margin * (thi - tlo)
    n = 0
    if laser:
        act.laser(True)
    for r in range(rows):
        tilt = tlo + mt + (thi - tlo - 2 * mt) * r / max(1, rows - 1)
        a, b = (plo + mp, phi - mp) if r % 2 == 0 else (phi - mp, plo + mp)
        act.move(a, tilt, duration_s=0.5)
        act.move(b, tilt, duration_s=row_s)
        if laser:
            act.laser(True)          # keep the auto-off timer from expiring mid-sweep
        n += 2
        if verbose:
            print(f"row {r + 1}/{rows}: tilt {tilt:.0f} us, pan {a:.0f} -> {b:.0f} us", flush=True)
    act.move((plo + phi) / 2, (tlo + thi) / 2, duration_s=0.5)
    return n + 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", help="config.yaml path")
    ap.add_argument("--actuator", help="override cfg actuator: pca9685|serial|bus|fake")
    ap.add_argument("--rows", type=int, default=6)
    ap.add_argument("--row-s", type=float, default=2.0, help="seconds per row")
    ap.add_argument("--margin", type=float, default=0.0, help="fraction of the range to stay inside")
    ap.add_argument("--laser", action="store_true", help="laser on during the sweep")
    ap.add_argument("--fast", action="store_true", help="fake actuator with a simulated clock")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    if args.actuator:
        cfg["actuator"] = args.actuator
    try:
        act = FakeActuator(cfg, clock=SimClock()) if args.fast else make_actuator(cfg)
    except ImportError as e:
        print(f"{cfg.get('actuator')} driver needs {e.name}: pip install "
              "adafruit-circuitpython-servokit (pca9685) or pyserial (serial)", file=sys.stderr)
        return 2
    print(f"{type(act).__name__} limits pan {act.limits()[0]} tilt {act.limits()[1]} us")
    try:
        moves = raster(act, args.rows, args.row_s, args.margin, args.laser, verbose=True)
    except KeyboardInterrupt:
        print("stopped")
        return 130
    finally:
        act.close()
    if isinstance(act, FakeActuator):
        (plo, phi), (tlo, thi) = act.limits()
        ok = all(plo <= p <= phi and tlo <= t <= thi for _, p, t in act.writes)
        print(f"{moves} moves, {len(act.writes)} pulse writes, all within limits: {ok}")
        return 0 if ok else 1
    print(f"{moves} moves done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
