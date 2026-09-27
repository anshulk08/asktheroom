"""Laser calibration (spec H4): grid over pulse space -> dot positions -> poly2 fit -> laser_cal.json.

    python -m act.calibrate --sim            # run against act/sim.py and print the report
    python -m act.calibrate --rig            # the real rig: jog the limits, latency, fit, held-out aims
    python -m act.calibrate --rig --limits 1150 1800 1250 1700   # skip the jog (limits from a jog)
    python -m act.calibrate --rig --check    # sticky-note ruler check of a saved fit (10 spots)

--rig needs the table calibrated first (python -m core.table; its outline, python -m core.table
--outline, keeps the fit on the tabletop in one-tag mode), actuator: pca9685 (or serial) in
config.local.yaml, and the camera free (stop the app). The laser's auto-off timer stays on throughout:
press any key in the jog, or Enter in the check, to light it again. Gates (F6): fit median < 1.5 cm,
centre aim < 3 cm; the check: ruler median < 1.5 cm, max < 3 cm.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import sys
import tempfile
import time
from typing import Callable, Iterator, Optional

import cv2
import numpy as np

from act.laser import Laser, dot_px_diff, fit_poly, table_px_to_cm


class CalibrationError(RuntimeError):
    pass


MAX_FIT_CM = 1.5          # F6 gates (also demo_check's)
MAX_CENTRE_CM = 3.0


class Region:
    """Where the laser should land: the tabletop outline (table_area.polygon_cm) if one is set, else the
    table rectangle. In one-tag mode the rectangle is the camera's whole view of the table plane, floor
    included, so set the outline before calibrating there."""

    def __init__(self, size_cm, polygon=None):
        w, h = (float(v) for v in size_cm)
        self.poly = np.asarray(polygon, dtype=np.float32).reshape(-1, 2) if polygon is not None and len(polygon) >= 3 else None
        if self.poly is not None:
            (x0, y0), (x1, y1) = self.poly.min(axis=0), self.poly.max(axis=0)
            self.bbox = (float(x0), float(y0), float(x1), float(y1))
            m = cv2.moments(self.poly.reshape(-1, 1, 2))
            self.centre = (m["m10"] / m["m00"], m["m01"] / m["m00"]) if m["m00"] else \
                ((x0 + x1) / 2, (y0 + y1) / 2)
        else:
            self.bbox = (0.0, 0.0, w, h)
            self.centre = (w / 2, h / 2)

    @classmethod
    def from_cfg(cls, cfg: dict, size_cm) -> "Region":
        return cls(size_cm, (cfg.get("table_area") or {}).get("polygon_cm") or None)

    def inside(self, pt, margin: float = 0.0) -> bool:
        """At least `margin` cm inside (a negative margin allows that far outside)."""
        x, y = float(pt[0]), float(pt[1])
        if not (math.isfinite(x) and math.isfinite(y)):
            return False
        if self.poly is not None:
            return cv2.pointPolygonTest(self.poly.reshape(-1, 1, 2), (x, y), True) >= margin
        x0, y0, x1, y1 = self.bbox
        return x0 + margin <= x <= x1 - margin and y0 + margin <= y <= y1 - margin

    def grid(self, nx: int, ny: int, inset: float) -> list[tuple[float, float]]:
        x0, y0, x1, y1 = self.bbox
        pts = [(float(x), float(y)) for y in np.linspace(y0 + inset, y1 - inset, ny)
               for x in np.linspace(x0 + inset, x1 - inset, nx)]
        return [p for p in pts if self.inside(p, inset * 0.5)]

    def sample(self, n: int, inset: float, seed: int = 0) -> np.ndarray:
        rng = np.random.default_rng(seed)
        x0, y0, x1, y1 = self.bbox
        out: list[tuple[float, float]] = []
        for _ in range(200 * n):
            p = (rng.uniform(x0, x1), rng.uniform(y0, y1))
            if self.inside(p, inset):
                out.append(p)
                if len(out) == n:
                    break
        return np.asarray(out, dtype=np.float64).reshape(-1, 2)


def _grid(lo: float, hi: float, n: int, margin: float) -> np.ndarray:
    m = margin * (hi - lo)
    return np.linspace(lo + m, hi - m, n)


def calibrate(laser: Laser, grid: int = 4, margin: float = 0.05, settle_s: float = 0.3,
              min_points: int = 10, edge_tol_cm: float = 2.0, refine: bool = True,
              region: Optional[Region] = None) -> dict:
    """Visit a grid x grid raster over the pulse limits, find the dot at each, fit, save.

    Points where the dot is missing or lands off the table (it fell to the floor: wrong plane) are
    dropped. If fewer than `min_points` remain, a second, interleaved (grid-1)^2 pass is added. One
    round of outlier rejection (> max(2 cm, 3x median)) guards against reflections.
    refine: after the first fit, visit a 4x3 grid of table points (4 cm in from the edges) through
    the fit and refit with all points, so edges/corners are interpolated rather than extrapolated
    (the pulse-space grid leaves them bare because its corners fall off the table).
    region: where dots count (default the table rectangle; the tabletop outline on the rig).
    Returns {'fit', 'fit_error_cm': {'median','max'}, 'n_points', 'misses', 'outliers', 'path'}.
    """
    (plo, phi), (tlo, thi) = laser.act.limits()
    region = region or Region(laser.table_size)
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
            if dot is None or not region.inside(dot, -edge_tol_cm):
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
            visit_pts([tuple(first.predict(c)) for c in region.grid(4, 3, inset=4.0)])
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
    fit.actuator = str((laser.cfg or {}).get("actuator", "fake")).lower()
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
                if ref is not None and dot_px_diff(ref.img, f.img, color=getattr(laser, "color", "red")) is not None:
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


# ---------------------------------------------------------------- the rig (--rig)

JOG_STEPS_US = (2, 5, 10, 25, 50)       # per key press; 50 is the most a press can move a servo
JOG_HELP = """Jog the dot to 8 points on the TABLETOP's edge in turn, clockwise as the camera sees it: each
corner and each edge's middle (the prompt names the next one), and press Enter at each.
  a / d or left / right   pan        w / s or up / down   tilt
  [ / ]                   smaller / bigger step              u   undo the last point
  l                       laser on / off                     q   quit without saving
The laser turns itself off after laser_timeout_s idle; any key lights it again."""
# A pan/tilt head's view of a straight table edge bows outward between the corners (up to ~75 us for a head
# 40 cm up near an edge), so the corners alone would clip the edges' middles: record those too.
JOG_POINTS = ("top-left corner", "top edge middle", "top-right corner", "right edge middle",
              "bottom-right corner", "bottom edge middle", "bottom-left corner", "left edge middle")


@contextlib.contextmanager
def terminal_keys() -> Iterator[Callable[[], str]]:
    """read_key() -> one key name ('a', 'left', 'enter', ...) from a raw terminal; restored on exit."""
    import termios
    import tty
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    arrows = {"A": "up", "B": "down", "C": "right", "D": "left"}

    def read_key() -> str:
        c = os.read(fd, 1).decode(errors="ignore")
        if c == "\x1b":
            seq = os.read(fd, 2).decode(errors="ignore")
            return arrows.get(seq[-1:], "esc")
        if c in ("\r", "\n"):
            return "enter"
        if c == "\x03":
            raise KeyboardInterrupt
        return c.lower()

    try:
        tty.setcbreak(fd)
        yield read_key
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def jog_limits(act, read_key: Callable[[], str], out: Callable[[str], None] = print,
               step_us: float = 10, margin: float = 0.05, min_margin_us: float = 20.0):
    """Interactive jog: the person moves the dot to the tabletop's corners and edge middles (JOG_POINTS)
    and records each. Returns servo limits that cover them (their bounding box plus `margin` of the span,
    at least min_margin_us, on each side, inside the actuator's current limits), or None if they quit.
    The limits only need to be safe, not tight. Leaves the laser off."""
    (plo, phi), (tlo, thi) = act.limits()
    pan, tilt = (plo + phi) / 2, (tlo + thi) / 2
    steps = list(JOG_STEPS_US)
    si = steps.index(step_us) if step_us in steps else 2
    corners: list[tuple[float, float]] = []
    moves = {"a": (-1, 0), "left": (-1, 0), "d": (1, 0), "right": (1, 0),
             "w": (0, -1), "up": (0, -1), "s": (0, 1), "down": (0, 1)}
    out(JOG_HELP)
    act.move(pan, tilt, duration_s=0.5)
    act.laser(True)
    laser_on = True
    try:
        while len(corners) < len(JOG_POINTS):
            out(f"  point {len(corners) + 1}/{len(JOG_POINTS)} ({JOG_POINTS[len(corners)]}): pan {pan:.0f} "
                f"tilt {tilt:.0f} step {steps[si]} us")
            k = read_key()
            if k == "q":
                out("jog cancelled")
                return None
            if k in moves:
                dp, dt = moves[k]
                pan, tilt = act.clamp(pan + dp * steps[si], tilt + dt * steps[si])
                act.move(pan, tilt, duration_s=0.1)
            elif k == "[":
                si = max(0, si - 1)
            elif k == "]":
                si = min(len(steps) - 1, si + 1)
            elif k == "u" and corners:
                corners.pop()
            elif k == "enter":
                corners.append((pan, tilt))
                out(f"  {JOG_POINTS[len(corners) - 1]}: pan {pan:.0f}, tilt {tilt:.0f}")
            elif k == "l":
                laser_on = not laser_on
                act.laser(laser_on)
                continue
            if laser_on:
                act.laser(True)            # lights it again after an auto-off, and restarts the timer
    finally:
        act.laser(False)
    c = np.asarray(corners)
    lo, hi = c.min(axis=0), c.max(axis=0)
    m = np.maximum(margin * (hi - lo), min_margin_us)
    return ((max(plo, float(np.floor(lo[0] - m[0]))), min(phi, float(np.ceil(hi[0] + m[0])))),
            (max(tlo, float(np.floor(lo[1] - m[1]))), min(thi, float(np.ceil(hi[1] + m[1])))))


def rig_session(laser: Laser, region: Region, *, read_key: Optional[Callable[[], str]] = None,
                limits=None, grid: int = 4, aims: int = 10, seed: int = 0,
                out: Callable[[str], None] = print) -> dict:
    """The --rig procedure on a built Laser: servo limits (given, or jogged with read_key), camera latency,
    calibrate (saves laser_cal.json), then aims at `aims` held-out table points and the centre.
    Returns a report with 'pass' (fit median < 1.5 cm and centre < 3 cm)."""
    act = laser.act
    rep: dict = {"pass": False}
    if limits is None and read_key is not None:
        limits = jog_limits(act, read_key, out)
        if limits is None:
            rep["error"] = "jog cancelled"
            return rep
    if limits is not None:
        act.set_limits(limits)
    (plo, phi), (tlo, thi) = act.limits()
    rep["servo_limits"] = {"pan": [plo, phi], "tilt": [tlo, thi]}
    out(f"servo limits: pan {plo:.0f}-{phi:.0f}, tilt {tlo:.0f}-{thi:.0f} us")

    laser.move_to((plo + phi) / 2, (tlo + thi) / 2)
    lat = measure_latency(laser)
    rep["latency"] = lat
    if lat["seen"] >= max(1, lat["trials"] // 2) and lat["recommended_s"]:
        laser.latency_s = float(lat["recommended_s"])
        out(f"camera latency: {laser.latency_s:.3f} s ({lat['seen']}/{lat['trials']} seen)")
    else:
        out(f"camera latency: dot not seen at the middle of the limits ({lat['seen']}/{lat['trials']}); "
            f"keeping {laser.latency_s:.3f} s. Is the dot on the table and the room not too bright?")
    rep["camera_latency_s"] = laser.latency_s

    try:
        cal = calibrate(laser, grid=grid, region=region)
    except CalibrationError as e:
        rep["error"] = str(e)
        out(f"calibration FAILED: {e}")
        return rep
    fe = cal["fit_error_cm"]
    rep.update(fit_error_cm=fe, n_points=cal["n_points"], misses=cal["misses"], outliers=cal["outliers"],
               path=cal["path"])
    out(f"fit: {cal['n_points']} points ({cal['misses']} missed, {cal['outliers']} outliers), "
        f"error median {fe['median']:.2f} cm, max {fe['max']:.2f} cm -> {cal['path']}")

    targets = region.sample(aims, inset=6.0, seed=seed + 101)
    st = aim_stats(laser, targets) if len(targets) else {}
    rep["held_out"] = st
    if st:
        out(f"held-out aims ({len(targets)}): open-loop median {st['open_loop_cm']['median']:.2f} cm, "
            f"closed-loop median {st['measured_cm']['median']:.2f} cm (max {st['measured_cm']['max']:.2f}), "
            f"{st['mean_tries']:.1f} looks each")
    try:
        centre = laser.aim(region.centre)
        first = laser.last_aim.get("first_err_cm")
    finally:
        laser.off()
    rep["centre_cm"] = centre
    rep["centre_open_loop_cm"] = first
    fit_ok, centre_ok = fe["median"] < MAX_FIT_CM, centre < MAX_CENTRE_CM
    rep["pass"] = bool(fit_ok and centre_ok)
    shown = "dot not seen" if math.isinf(centre) else f"{centre:.2f} cm"
    out(f"F6 gate: fit median {fe['median']:.2f} < {MAX_FIT_CM} cm {'PASS' if fit_ok else 'FAIL'}; "
        f"centre ({region.centre[0]:.0f}, {region.centre[1]:.0f}) {shown} < {MAX_CENTRE_CM} cm "
        f"{'PASS' if centre_ok else 'FAIL'}")
    out("Put these in config.local.yaml (the jog's limits and the measured latency):\n"
        f"servo_limits:\n  pan: [{plo:.0f}, {phi:.0f}]\n  tilt: [{tlo:.0f}, {thi:.0f}]\n"
        f"camera_latency_s: {laser.latency_s:.3f}")
    return rep


def note_centre(before: np.ndarray, after: np.ndarray, table, min_cm2: float = 9.0,
                max_cm2: float = 400.0, thr: int = 35) -> Optional[tuple[float, float]]:
    """Table cm of the biggest new thing between two frames (a sticky note put down), or None."""
    d = cv2.absdiff(cv2.cvtColor(before, cv2.COLOR_BGR2GRAY), cv2.cvtColor(after, cv2.COLOR_BGR2GRAY))
    mask = cv2.morphologyEx((d > thr).astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, _, stats, cent = cv2.connectedComponentsWithStats(mask, connectivity=8)
    best, best_a = None, 0.0
    for i in range(1, n):
        x, y, w, h = (int(stats[i, k]) for k in (cv2.CC_STAT_LEFT, cv2.CC_STAT_TOP, cv2.CC_STAT_WIDTH,
                                                  cv2.CC_STAT_HEIGHT))
        box = table.px_to_cm(np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], dtype=np.float64))
        a = cv2.contourArea(box.astype(np.float32).reshape(-1, 1, 2)) * stats[i, cv2.CC_STAT_AREA] / max(1, w * h)
        if min_cm2 <= a <= max_cm2 and a > best_a:
            best, best_a = cent[i], a
    if best is None:
        return None
    c = table.px_to_cm(np.array([best], dtype=np.float64))[0]
    return float(c[0]), float(c[1])


SATURATED = 215     # red channel level (0-255) above which a surface leaves the dot too little headroom


def surface_red(laser: Laser, cm, img: Optional[np.ndarray], r_cm: float = 2.0) -> Optional[float]:
    """Median red level of the surface around cm with the laser off (the dot is found as a red rise, so a
    surface already near 255, white or yellow under bright light, can hide it)."""
    if img is None:
        return None
    c = np.asarray(cm, dtype=np.float64)
    box = laser.table.cm_to_px(c + np.array([[-r_cm, -r_cm], [r_cm, r_cm]]))
    (x0, y0), (x1, y1) = np.floor(box.min(axis=0)).astype(int), np.ceil(box.max(axis=0)).astype(int)
    patch = img[max(0, y0):max(0, y1), max(0, x0):max(0, x1), 2]
    return float(np.median(patch)) if patch.size else None


def _fresh(laser: Laser) -> Optional[np.ndarray]:
    """A frame taken after the laser's current state has reached the camera."""
    laser.clock.sleep(laser.latency_s + 0.2)
    f = laser.frames.latest()
    return None if f is None else f.img


def spot_check(laser: Laser, ask: Callable[[str], str], n: int = 10, out: Callable[[str], None] = print,
               region: Optional[Region] = None) -> dict:
    """Ruler check: the person puts a sticky note (a cross drawn at its centre) somewhere new, the camera
    finds it, the laser aims at it, and the person measures the dot to the cross with a ruler. That is
    the demo's question (does the dot land on what the camera sees?) answered without trusting the
    camera's own dot. Put at least one note ON the box's top face (and the notebook's cover if it is
    light): a surface whose red channel is near 255 leaves the dot no headroom, and the fix for that is
    lower exposure or gain (scripts/camera_setup.sh), not code. Each spot records that red level.
    Returns {'spots', 'ruler_cm', 'pass'} (ruler median < 1.5 cm and max < 3 cm)."""
    region = region or Region(laser.table_size)
    laser._need_fit()
    laser.off()
    before = _fresh(laser)
    spots: list[dict] = []
    while len(spots) < n:
        i = len(spots) + 1
        a = ask(f"Spot {i}/{n}: put a sticky note down somewhere new (spread them over the table; one on top of "
                "the box), take your hand out of view, press Enter (q to stop) ")
        if a.strip().lower() == "q":
            break
        after = _fresh(laser)
        c = None if before is None or after is None else note_centre(before, after, laser.table)
        if c is None or not region.inside(c, -2.0):
            out("  no new note found on the tabletop: is your hand out of view? Try again.")
            before = after
            continue
        red = surface_red(laser, c, after)
        cam = laser.aim(c)
        first = laser.last_aim.get("first_err_cm")
        out(f"  note at ({c[0]:.1f}, {c[1]:.1f}) cm; camera says the dot is "
            f"{'not seen' if math.isinf(cam) else f'{cam:.1f} cm'} from it"
            + ("" if red is None else f" (surface red {red:.0f}/255)"))
        if red is not None and red >= SATURATED:
            out(f"  the surface here is near saturation (red {red:.0f} >= {SATURATED}): if the camera misses the "
                "dot on the box or notebook, lower the gain (scripts/camera_setup.sh 166 <brio> 40 10 160 3200), "
                "then recalibrate the laser")
        ruler = None
        while ruler is None:
            r = ask("  Ruler: dot centre to the note's cross, in cm (Enter lights the dot again, x = no dot) ")
            r = r.strip().lower()
            if r == "":
                cam = laser.aim(c)
            elif r == "x":
                ruler = math.inf
            else:
                try:
                    ruler = float(r)
                except ValueError:
                    out("  a number in cm, please")
        laser.off()
        spots.append({"spot": i, "x_cm": round(c[0], 1), "y_cm": round(c[1], 1),
                      "camera_cm": None if math.isinf(cam) else round(cam, 2),
                      "open_loop_cm": None if first is None else round(first, 2),
                      "surface_red": None if red is None else round(red),
                      "ruler_cm": None if math.isinf(ruler) else ruler})
        before = _fresh(laser)                  # the new note is part of the scene now
    r = np.array([math.inf if s["ruler_cm"] is None else s["ruler_cm"] for s in spots], dtype=np.float64)
    rep: dict = {"spots": spots, "pass": False}
    if len(r):
        rep["ruler_cm"] = {"median": float(np.median(r)), "max": float(r.max()), "n": len(r)}
        rep["pass"] = bool(len(r) >= n and np.median(r) < MAX_FIT_CM and r.max() < MAX_CENTRE_CM)
    out(spot_table(rep))
    return rep


def spot_table(rep: dict) -> str:
    """Markdown results table (paste it into docs/RIG_RUNBOOK.md's evidence)."""
    def f(v):
        return "-" if v is None else f"{v:.1f}"
    rows = ["| spot | x cm | y cm | surface red | camera cm | open-loop cm | ruler cm | ok (< 3 cm) |",
            "|---|---|---|---|---|---|---|---|"]
    for s in rep["spots"]:
        ok = s["ruler_cm"] is not None and s["ruler_cm"] < MAX_CENTRE_CM
        red = s.get("surface_red")
        rows.append(f"| {s['spot']} | {s['x_cm']:.1f} | {s['y_cm']:.1f} | {'-' if red is None else red} | "
                    f"{f(s['camera_cm'])} | "
                    f"{f(s['open_loop_cm'])} | {f(s['ruler_cm'])} | {'yes' if ok else 'NO'} |")
    rc = rep.get("ruler_cm")
    if rc:
        rows.append(f"\nruler median {rc['median']:.1f} cm (< {MAX_FIT_CM}), max {rc['max']:.1f} cm "
                    f"(< {MAX_CENTRE_CM}) over {rc['n']} spots: {'PASS' if rep['pass'] else 'FAIL'}")
    return "\n".join(rows)


def _rig_main(args: argparse.Namespace) -> int:
    """Build the real rig's parts (as main.build does) and run rig_session or spot_check."""
    import act.actuator
    import core.capture
    import core.table
    import core.table_area
    import main as app
    from core.config import load_config

    cfg = load_config(args.config)
    core.table.apply_saved_size(cfg)
    core.table_area.apply_saved_area(cfg)
    if str(cfg.get("actuator", "fake")).lower() == "fake":
        print("actuator is 'fake' (nothing would move): set `actuator: pca9685` in config.local.yaml",
              file=sys.stderr)
        return 2
    table = core.table.Table(cfg)
    if not table.ok:
        print(f"table not calibrated ({table.cal_path} missing): run python -m core.table first", file=sys.stderr)
        return 2
    region = Region.from_cfg(cfg, cfg["table"]["size_cm"])
    if region.poly is None and (cfg.get("table_tag") or {}).get("enabled"):
        print("warning: no tabletop outline; in one-tag mode the fit then spans the whole camera view "
              "(floor included). Set it first: python -m core.table --outline", file=sys.stderr)
    camera = app.camera_source(args.camera) if args.camera else app.default_camera(cfg)
    actuator = act.actuator.make_actuator(cfg)
    frames = core.capture.FrameBuffer(camera)
    try:
        if frames.wait_new(0, timeout=5.0) is None:
            print(f"camera {camera} sent no frames in 5 s (is the app still running and holding it?)",
                  file=sys.stderr)
            return 2
        cal_path = args.out or cfg["paths"]["laser_cal"]
        if args.check:
            laser = Laser(actuator, frames, table, cal_path, cfg=cfg)
            if laser.fit is None:
                print(f"no {cal_path}: run python -m act.calibrate --rig first", file=sys.stderr)
                return 2
            rep = spot_check(laser, input, n=args.spots, region=region)
            os.makedirs("data/trials", exist_ok=True)
            log_path = time.strftime("data/trials/laser_spots_%Y%m%d_%H%M.json")
            with open(log_path, "w") as f:
                json.dump(rep, f, indent=1)
            print(f"saved {log_path}")
            return 0 if rep["pass"] else 1
        laser = Laser(actuator, frames, table, "", cfg=cfg)   # don't start from an old fit
        laser.cal_path = cal_path
        limits = None
        if args.limits:
            limits = ((args.limits[0], args.limits[1]), (args.limits[2], args.limits[3]))
        if limits is None and not args.no_jog:
            if not sys.stdin.isatty():
                print("the jog needs a terminal (scripts/dock.sh gives one), or pass --limits / --no-jog",
                      file=sys.stderr)
                return 2
            with terminal_keys() as read_key:
                rep = rig_session(laser, region, read_key=read_key, grid=args.grid, aims=args.aims,
                                  seed=args.seed)
        else:
            rep = rig_session(laser, region, limits=limits, grid=args.grid, aims=args.aims, seed=args.seed)
        return 0 if rep["pass"] else 1
    finally:
        try:
            actuator.close()
        finally:
            frames.stop()


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
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--sim", action="store_true", help="calibrate the simulated rig (act/sim.py)")
    mode.add_argument("--rig", action="store_true", help="calibrate the real rig (camera, table, servos)")
    ap.add_argument("--check", action="store_true", help="with --rig: sticky-note ruler check of the saved fit")
    ap.add_argument("--spots", type=int, default=10, help="--check: how many spots")
    ap.add_argument("--camera", help="--rig: camera index or /dev/v4l/by-id path (default: config demo_check.camera)")
    ap.add_argument("--limits", type=float, nargs=4, metavar=("PAN_LO", "PAN_HI", "TILT_LO", "TILT_HI"),
                    help="--rig: servo limits in us instead of the jog")
    ap.add_argument("--no-jog", action="store_true", help="--rig: use config servo_limits as they are")
    ap.add_argument("--config")
    ap.add_argument("--grid", type=int, default=4)
    ap.add_argument("--aims", type=int, default=None,
                    help="held-out aim targets to score after the fit (default: 20 sim, 10 rig)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", help="where to write the calibration json (sim default: a temp dir; rig: "
                                  "config paths.laser_cal)")
    args = ap.parse_args(argv)
    if args.sim:
        args.aims = 20 if args.aims is None else args.aims
        return _sim_main(args)
    if args.rig:
        args.aims = 10 if args.aims is None else args.aims
        return _rig_main(args)
    ap.print_help(sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
