"""Point the stepper turret at imaginary objects through TurretActuator and check it by eye.

    python scripts/turret_point_test.py /dev/ttyACM0        # Jetson; the Mac shows /dev/cu.usbmodem*

Line the mount up level and facing forward first: opening the port makes that 0,0. Each object
is held HOLD_S seconds; picture it at the stated spot. The board's reported position must match
the pointing math to within one microstep, and your eyes check the real angles (a steady error
on one axis means its gear ratio in firmware/turret/turret.ino is off). Ends parked at 0,0.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from act.actuator import TurretActuator  # noqa: E402
from act.pointing import solve_aim  # noqa: E402

HOLD_S = 4
STEP_DEG = (360 / (3200 * 4.0), 360 / (3200 * 3.2))  # one microstep at the pan / tilt axis

OBJECTS = [
    ("2 m straight ahead", (0.0, 0.0, 2.0)),
    ("1 m ahead, 1 m to the RIGHT (45 deg right)", (1.0, 0.0, 1.0)),
    ("1 m ahead, 1 m to the LEFT (45 deg left)", (-1.0, 0.0, 1.0)),
    ("directly to the RIGHT (90 deg)", (1.0, 0.0, 0.0)),
    ("1 m ahead, 1 m UP (45 deg up)", (0.0, 1.0, 1.0)),
    ("45 deg right and 45 deg up", (1.0, 2 ** 0.5, 1.0)),
    ("straight UP", (0.0, 1.0, 0.0)),
]


def main(port: str) -> int:
    cfg = {"servo_limits": {"pan": [1500 - 1700, 1500 + 1700], "tilt": [1500 - 450, 1500 + 900]},
           "turret": {"port": port}}   # the firmware's own limits, at 10 us/deg
    act = TurretActuator(cfg)
    failures = 0
    try:
        for name, xyz in OBJECTS:
            want = solve_aim(xyz)
            act.point_at(*xyz)
            pan, tilt, moving, _ = act._turret.position()
            ok = (abs(pan - want.pan) <= STEP_DEG[0] and abs(tilt - want.tilt) <= STEP_DEG[1]
                  and not moving)
            failures += not ok
            print(f"{'ok ' if ok else 'BAD'} {name:44s} want pan {want.pan:7.2f} tilt {want.tilt:6.2f}"
                  f"  board {pan:7.2f} {tilt:6.2f}")
            time.sleep(HOLD_S)
    finally:
        act.close()
        print("parked at 0,0")
    return failures


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    sys.exit(1 if main(sys.argv[1]) else 0)
