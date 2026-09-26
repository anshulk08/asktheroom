"""Laser calibration (spec H4): grid over pulse space -> dot positions -> poly2 fit -> laser_cal.json.

    python -m act.calibrate --sim            # run against act/sim.py and print the report
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile
from typing import Optional

import numpy as np

from act.laser import Laser, dot_px_diff, fit_poly, table_px_to_cm


class CalibrationError(RuntimeError):
    pass


def _grid(lo: float, hi: float, n: int, margin: float) -> np.ndarray:
    m = margin * (hi - lo)
    return np.linspace(lo + m, hi - m, n)


def calibrate(laser: Laser, grid: int = 4, margin: float = 0.05, settle_s: float = 0.3,
              min_points: int = 10, edge_tol_cm: float = 2.0, refine: bool = True) -> dict:
    """Visit a grid x grid raster over the pulse limits, find the dot at each, fit, save.

    Points where the dot is missing or lands off the table (it fell to the floor: wrong plane) are
    dropped. If fewer than `min_points` remain, a second, interleaved (grid-1)^2 pass is added. One
    round of outlier rejection (> max(2 cm, 3x median)) guards against reflections.
    refine: after the first fit, visit a 4x3 grid of table points (4 cm in from the edges) through
    the fit and refit with all points, so edges/corners are interpolated rather than extrapolated
    (the pulse-space grid leaves them bare because its corners fall off the table).
    Returns {'fit', 'fit_error_cm': {'median','max'}, 'n_points', 'misses', 'outliers', 'path'}.
    """
    (plo, phi), (tlo, thi) = laser.act.limits()
    w, h = laser.table_size
    xy: list[tuple[float, float]] = []
    pul: list[tuple[float, float]] = []
    misses = 0

    def visit(pans, tilts) -> None:
        visit_pts([(p, t) for t in tilts for p in pans])

    def visit_pts(pts) -> None:
        nonlocal misses
        for pan, tilt in pts:
            pan_c, tilt_c = laser._clamp(pan, tilt)
            laser.move_to(pan_c, tilt_c)
            laser.clock.sleep(settle_s)
            dot = laser.find_dot()
            if dot is None or not (-edge_tol_cm <= dot[0] <= w + edge_tol_cm
                                   and -edge_tol_cm <= dot[1] <= h + edge_tol_cm):
                misses += 1
                continue
            xy.append(dot)
            pul.append((pan_c, tilt_c))

    try:
        pans, tilts = _grid(plo, phi, grid, margin), _grid(tlo, thi, grid, margin)
        visit(pans, tilts)
        if len(xy) < min_points and grid > 1:
            visit((pans[:-1] + pans[1:]) / 2, (tilts[:-1] + tilts[1:]) / 2)
        if len(xy) < 6:
            raise CalibrationError(f"only {len(xy)} dots seen ({misses} misses): check the laser, "
                                   "servo_limits (must cover the table) and camera_latency_s")
        if refine:
            first = fit_poly(xy, pul, grid)
            inset = 4.0
            cm = [(x, y) for y in np.linspace(inset, h - inset, 3) for x in np.linspace(inset, w - inset, 4)]
            visit_pts([tuple(first.predict(c)) for c in cm])
        X, P = np.asarray(xy), np.asarray(pul)
        fit = fit_poly(X, P, grid)
        outliers = 0
        e = fit.residuals_cm(X, P)
        bad = e > max(2.0, 3 * float(np.median(e)))
        if bad.any() and (~bad).sum() >= 6:
            outliers = int(bad.sum())
            X, P = X[~bad], P[~bad]
            fit = fit_poly(X, P, grid)
    finally:
        laser.off()
    fit.table_px_to_cm = table_px_to_cm(laser.table, laser.table_size)
    fit.servo_limits = {"pan": [plo, phi], "tilt": [tlo, thi]}
    if laser.cal_path:
        fit.save(laser.cal_path)
    laser.fit = fit
    return {"fit": fit.to_dict(), "fit_error_cm": fit.fit_error_cm, "n_points": fit.n_points,
            "misses": misses, "outliers": outliers, "path": laser.cal_path}


def measure_latency(laser: Laser, trials: int = 5, wait_s: float = 0.5) -> dict:
    """Time from laser-on to the first frame (by Frame.t) that shows the dot. Aim at the table first.

    The true latency L satisfies first.t - t_cmd - period < L < first.t - t_cmd, so the max of
    (first.t - t_cmd) over trials is a safe value for cfg camera_latency_s.
    """
    seen = []
    for _ in range(trials):
        laser.act.laser(False)
        laser.clock.sleep(wait_s)
        ref = laser.frames.latest()
        laser.act.laser(True)
        t_cmd = laser.clock.now()
        last_idx = ref.idx if ref else -1
        while laser.clock.now() < t_cmd + wait_s:
            f = laser.frames.latest()
            if f is not None and f.idx != last_idx:
                last_idx = f.idx
                if ref is not None and dot_px_diff(ref.img, f.img) is not None:
                    seen.append(f.t - t_cmd)
                    break
            laser.clock.sleep(0.002)
    laser.off()
    if not seen:
        return {"trials": trials, "seen": 0, "recommended_s": None}
    return {"trials": trials, "seen": len(seen), "median_s": float(np.median(seen)),
            "recommended_s": round(float(max(seen)) + 0.005, 3)}


def aim_stats(laser: Laser, targets, truth=None) -> dict:
    """Aim at each target; report measured (camera) and, if `truth()` is given, true errors in cm."""
    meas, true, first, tries = [], [], [], []
    for tgt in targets:
        meas.append(laser.aim(tuple(tgt)))
        first.append(laser.last_aim.get("first_err_cm") or math.inf)
        tries.append(laser.last_aim.get("tries", 0))
        if truth is not None:
            true.append(float(np.linalg.norm(np.asarray(truth()) - np.asarray(tgt))))
    laser.off()

    def summ(v: list) -> dict:
        a = np.asarray(v, dtype=np.float64)
        f = a[np.isfinite(a)]
        return {"median": float(np.median(a)), "p90": float(np.percentile(f, 90)) if len(f) else math.inf,
                "max": float(a.max()), "n": len(a), "n_under_1cm": int((a < 1.0).sum())}
    out = {"measured_cm": summ(meas), "open_loop_cm": summ(first), "mean_tries": float(np.mean(tries))}
    if truth is not None:
        out["true_cm"] = summ(true)
    return out


def _sim_main(args: argparse.Namespace) -> int:
    from act.sim import SimRig
    from core.config import load_config

    rig = SimRig(load_config(), seed=args.seed)
    out = args.out or os.path.join(tempfile.mkdtemp(prefix="askroom_"), "laser_cal_sim.json")
    laser = rig.make_laser(out)
    rep = calibrate(laser, grid=args.grid)
    rep.pop("fit")
    print("sim limits      ", rig.cfg["servo_limits"])
    print("calibration     ", json.dumps(rep))
    laser.move_to(*laser.fit.predict((45, 30)))
    print("latency         ", json.dumps(measure_latency(laser)), f"(sim truth {rig.latency_s} s)")
    rng = np.random.default_rng(args.seed + 7)
    w, h = rig.table.size_cm
    tg = np.column_stack([rng.uniform(5, w - 5, args.aims), rng.uniform(5, h - 5, args.aims)])
    t0 = rig.clock.now()
    st = aim_stats(laser, tg, rig.true_dot_cm)
    st["sim_s_per_aim"] = round((rig.clock.now() - t0) / args.aims, 2)
    print("aim (%d targets)" % args.aims, json.dumps(st))
    ok = rep["fit_error_cm"]["max"] < 1.5 and st["true_cm"]["median"] <= 3.0
    print("H5 pass" if ok else "H5 FAIL")
    return 0 if ok else 1


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sim", action="store_true", help="calibrate the simulated rig (act/sim.py)")
    ap.add_argument("--grid", type=int, default=4)
    ap.add_argument("--aims", type=int, default=20, help="random aim targets to score after the fit")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", help="where to write the sim calibration json (default: a temp dir)")
    args = ap.parse_args(argv)
    if args.sim:
        return _sim_main(args)
    print("Real-rig calibration needs the camera FrameBuffer and Table from perception: from the app, "
          "call act.calibrate.calibrate(Laser(make_actuator(cfg), frames, table, cfg['paths']['laser_cal'], "
          "cfg=cfg)). Use --sim to run against the simulator.", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
