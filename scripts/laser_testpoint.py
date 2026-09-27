"""One lit test point: does the camera see the dot? Before any full sweep (docs/LASER_DEMO_RUNBOOK.md).

    scripts/dock.sh python3 scripts/laser_testpoint.py 0 -22 --eye-safe-confirmed            # pan, tilt (deg)
    scripts/dock.sh python3 scripts/laser_testpoint.py 0 -22 --blinks 3 --eye-safe-confirmed # + 3 slow blinks

No person check here: run it only with the room in front of the turret confirmed clear (--eye-safe-confirmed),
and never while the app runs (it holds the serial port and the camera).

Moves dark to (pan, tilt) (clamped to servo_limits and turret.tilt_max_deg), blinks --pairs off/on pairs and
prints the dot px (or None), the strongest score and where it is (a dot too faint for laser_diff_thr shows as
a high score under the threshold), and, with --blinks, the firmware's POS reply per blink (its last field is
the laser state as the Uno sees it). Writes data/testpoint.jpg with the dot circled. The laser ends off and
the head parks (close). Opening the port resets the Uno: line the head up level first.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main(argv=None) -> int:
    import cv2
    import numpy as np

    from act.laser import dot_score
    from act.room_map import _real_rig
    from core.config import load_config
    from main import camera_source, default_camera
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pan", type=float)
    ap.add_argument("tilt", type=float)
    ap.add_argument("--pairs", type=int, default=5)
    ap.add_argument("--blinks", type=int, default=0, help="slow visible blinks afterwards (1 s on)")
    ap.add_argument("--camera")
    ap.add_argument("--eye-safe-confirmed", action="store_true",
                    help="required: the operator confirms nobody is in front of the turret (this script has no "
                         "person check) and the module is the approved one")
    a = ap.parse_args(argv)
    if not a.eye_safe_confirmed:
        print("refused: this lights the laser with no person check; confirm the room in front of the turret is "
              "clear and rerun with --eye-safe-confirmed", file=sys.stderr)
        return 2
    cfg = load_config()
    laser, frames = _real_rig(cfg, camera_source(a.camera) if a.camera else default_camera(cfg))
    try:
        t0 = time.time()
        while frames.latest() is None and time.time() - t0 < 8:
            time.sleep(0.1)
        laser._move_dark(*laser._clamp(laser.act.deg_to_us(a.pan), laser.act.deg_to_us(a.tilt)))
        time.sleep(0.8)
        d = laser.find_dot_px(a.pairs, src=laser.px_source)
        laser.off()
        print("test point", a.pan, a.tilt, "-> dot px", d)
        off, on = laser.last_frames
        if off is not None and on is not None:
            sc = dot_score(off.img, on.img, laser.color)
            y, x = np.unravel_index(int(np.argmax(sc)), sc.shape)
            print("max score", int(sc.max()), "at", (int(x), int(y)), "threshold", laser.diff_thr)
            img = on.img.copy()
            if d:
                cv2.circle(img, (int(d[0]), int(d[1])), 40, (0, 255, 0), 6)
            Path("data").mkdir(exist_ok=True)
            cv2.imwrite("data/testpoint.jpg", cv2.resize(img, (img.shape[1] // 2, img.shape[0] // 2)))
        tur = getattr(laser.act, "_turret", None)
        for k in range(a.blinks):
            laser.act.laser(True)
            print("blink", k + 1, "firmware:", tur._command("P", "POS") if tur is not None else "n/a")
            time.sleep(1.0)
            laser.act.laser(False)
            time.sleep(0.7)
        return 0 if d is not None else 1
    finally:
        laser.off()
        laser.act.close()
        frames.stop()


if __name__ == "__main__":
    raise SystemExit(main())
