"""Room dot map: pixel -> servo pulses anywhere the camera can see the dot (spec 0006). Owner: A.

A structured-light sweep: step the laser over a pulse grid covering the servo range, look for the dot
at each point (averaged on/off pairs) and record (pan, tilt) -> dot px, or a miss. A second pass
re-samples between neighbours whose dots jump in the image (depth edges) or where one was seen and the
other missed. No depth: with the pan-tilt pivot a few cm from the lens the map is nearly depth-invariant,
and Laser.aim_px's closed loop removes what's left.

pulses_for_px inverts the map locally: anchor on the nearest seen dot, fit an affine map on the samples
around it in pulse space, dropping the ones that don't fit (the map folds at a depth edge), and refuse
where nothing was mapped nearby.

Also here: the zones where room aims may fire (image polygons of surfaces below eye level) and the
safety gate (person or hand boxes on the beam path).

    python -m act.room_map --sim                     # sweep the simulated room, score aims
    python -m act.room_map --sweep                   # real rig: nobody in view, ~2-3 min
    python -m act.room_map --zone shelf --poly 400,60 490,60 490,80 400,80
    python -m act.room_map --list
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import time
from dataclasses import dataclass
from typing import Callable, Optional

import cv2
import numpy as np

log = logging.getLogger(__name__)


class SweepAborted(RuntimeError):
    """stop() returned True mid-sweep (a person came into view)."""


@dataclass
class Guess:
    pulses: tuple[float, float]
    J: np.ndarray          # 2x2 d(px)/d(pulses), px per µs
    fold: bool             # samples near the target disagreed (depth edge): the guess is rougher
    gap_px: float          # distance from the target to the nearest mapped dot
    n: int                 # samples in the local fit


class RoomMap:
    """Sweep samples: pulses (N,2) and dot px (N,2, NaN = not seen); step_us = grid spacing (pan, tilt)."""

    def __init__(self, pulses, px, step_us, size_px, zones: Optional[dict] = None, t: float = 0.0,
                 grid: Optional[tuple[int, int]] = None):
        self.pulses = np.asarray(pulses, dtype=np.float64).reshape(-1, 2)
        self.px = np.asarray(px, dtype=np.float64).reshape(-1, 2)
        self.step_us = (float(step_us[0]), float(step_us[1]))
        self.size_px = (int(size_px[0]), int(size_px[1]))
        self.zones: dict[str, list] = dict(zones or {})
        self.t, self.grid = t, grid
        self._index()

    def _index(self) -> None:
        self.seen = ~np.isnan(self.px[:, 0])
        self._p, self._x = self.pulses[self.seen], self.px[self.seen]
        if len(self._x) >= 2:
            d = np.linalg.norm(self._x[:, None] - self._x[None], axis=2)
            np.fill_diagonal(d, np.inf)
            self.spacing_px = float(np.median(d.min(axis=1)))
        else:
            self.spacing_px = math.nan
        self._J = None
        if len(self._x) >= 3:                     # global affine, the fallback Jacobian
            A = np.c_[self._p, np.ones(len(self._p))]
            coef, *_ = np.linalg.lstsq(A, self._x, rcond=None)
            self._J = coef[:2].T

    @property
    def n_seen(self) -> int:
        return int(self.seen.sum())

    def pulses_for_px(self, uv, max_gap_px: float = 60.0, radius_steps: float = 2.5,
                      fold_px: Optional[float] = None) -> Optional[Guess]:
        """Feedforward pulses and local Jacobian for image pixel uv; None when uv is farther than
        max_gap_px from every mapped dot."""
        if len(self._x) < 3:
            return None
        uv = np.asarray(uv, dtype=np.float64)
        d = np.linalg.norm(self._x - uv, axis=1)
        a = int(np.argmin(d))
        if d[a] > max_gap_px:
            return None
        step = np.asarray(self.step_us)
        q = (self._p - self._p[a]) / step
        near = np.nonzero(np.abs(q).max(axis=1) <= radius_steps)[0]
        thr = fold_px if fold_px is not None else max(4.0, 0.25 * self.spacing_px)
        w = np.exp(-(q[near] ** 2).sum(axis=1) / (2 * 1.2 ** 2))
        keep = np.ones(len(near), bool)
        fold, A = False, None
        while keep.sum() >= 3:
            i = near[keep]
            X = np.c_[self._p[i] - self._p[a], np.ones(len(i))] * np.sqrt(w[keep])[:, None]
            coef, *_ = np.linalg.lstsq(X, self._x[i] * np.sqrt(w[keep])[:, None], rcond=None)
            r = np.linalg.norm(np.c_[self._p[i] - self._p[a], np.ones(len(i))] @ coef - self._x[i], axis=1)
            r[i == a] = 0.0                                  # never drop the anchor
            j = int(np.argmax(r))
            if r[j] <= thr:
                A = coef
                break
            keep[np.nonzero(keep)[0][j]] = False
            fold = True
        if A is not None and abs(np.linalg.det(A[:2].T)) > 1e-9:
            J, c = A[:2].T, A[2]
        elif self._J is not None:
            J, c = self._J, self._x[a]
        else:
            return None
        dp = np.linalg.solve(J, uv - c)
        k = float(np.max(np.abs(dp) / (2.0 * step)))
        if k > 1:
            dp = dp / k                                      # don't extrapolate past 2 grid steps
        p = self._p[a] + dp
        return Guess((float(p[0]), float(p[1])), J, fold, float(d[a]), int(keep.sum()))

    # -- zones
    def zone_at(self, uv) -> Optional[str]:
        """Name of the zone polygon containing pixel uv (None if outside all)."""
        pt = (float(uv[0]), float(uv[1]))
        for name, poly in self.zones.items():
            if cv2.pointPolygonTest(np.asarray(poly, np.float32).reshape(-1, 1, 2), pt, False) >= 0:
                return name
        return None

    # -- persistence
    def to_dict(self) -> dict:
        px = [None if math.isnan(u) else [u, v] for u, v in self.px.tolist()]
        return {"version": 1, "pulses": self.pulses.tolist(), "px": px, "step_us": list(self.step_us),
                "size_px": list(self.size_px), "zones": self.zones, "t": self.t,
                "grid": list(self.grid) if self.grid else None}

    @classmethod
    def from_dict(cls, d: dict) -> "RoomMap":
        px = [[math.nan, math.nan] if p is None else p for p in d["px"]]
        g = d.get("grid")
        return cls(d["pulses"], px, d["step_us"], d["size_px"], d.get("zones"), float(d.get("t", 0.0)),
                   tuple(g) if g else None)

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(self.to_dict(), f)
        os.replace(tmp, path)

    @classmethod
    def load(cls, path: str) -> "RoomMap":
        with open(path) as f:
            return cls.from_dict(json.load(f))


# ---------------------------------------------------------------- sweep

def sweep(laser, grid: tuple[int, int] = (20, 15), n_pairs: int = 2, refine: bool = True,
          jump_k: float = 2.5, stop: Optional[Callable[[], bool]] = None,
          progress: Optional[Callable[[int, int], None]] = None) -> RoomMap:
    """Record the dot map over the laser's full servo range. stop() is checked before every point
    (return True when a person is in view) and raises SweepAborted. The laser ends off."""
    (plo, phi), (tlo, thi) = laser.act.limits()
    gx, gy = int(grid[0]), int(grid[1])
    pans, tilts = np.linspace(plo, phi, gx), np.linspace(tlo, thi, gy)
    step = (float(pans[1] - pans[0]) if gx > 1 else 1.0, float(tilts[1] - tilts[0]) if gy > 1 else 1.0)
    pts, seen = [], []
    size = None

    def visit(p: float, t: float) -> Optional[tuple[float, float]]:
        nonlocal size
        if stop is not None and stop():
            raise SweepAborted("person in view")
        laser.move_to(p, t, duration_s=0.1)
        laser.clock.sleep(laser.settle_s)
        d = laser.find_dot_px(n_pairs)
        f = laser.last_frames[1]
        if f is not None and f.img is not None:
            size = (f.img.shape[1], f.img.shape[0])
        return d

    try:
        order = [(p, t) for j, t in enumerate(tilts) for p in (pans if j % 2 == 0 else pans[::-1])]
        for k, (p, t) in enumerate(order):
            pts.append((p, t))
            seen.append(visit(p, t))
            if progress:
                progress(k + 1, len(order))
        if refine:
            px = {(round(p, 3), round(t, 3)): d for (p, t), d in zip(pts, seen)}
            G = [[px[(round(p, 3), round(t, 3))] for p in pans] for t in tilts]
            jumps = [float(np.hypot(*np.subtract(G[j][i], G[j2][i2])))
                     for j in range(gy) for i in range(gx) for j2, i2 in ((j, i + 1), (j + 1, i))
                     if j2 < gy and i2 < gx and G[j][i] is not None and G[j2][i2] is not None]
            med = float(np.median(jumps)) if jumps else math.inf
            extra = []
            for j in range(gy):
                for i in range(gx):
                    for j2, i2 in ((j, i + 1), (j + 1, i)):
                        if j2 >= gy or i2 >= gx:
                            continue
                        a, b = G[j][i], G[j2][i2]
                        if (a is None) != (b is None) or (a is not None and b is not None and
                                                          np.hypot(*np.subtract(a, b)) > jump_k * med):
                            extra.append(((pans[i] + pans[i2]) / 2, (tilts[j] + tilts[j2]) / 2))
            for p, t in sorted(set(extra), key=lambda x: (x[1], x[0])):
                pts.append((p, t))
                seen.append(visit(p, t))
    finally:
        laser.off()
    px = [(math.nan, math.nan) if d is None else d for d in seen]
    return RoomMap(pts, px, step, size or (0, 0), t=time.time(), grid=(gx, gy))


# ---------------------------------------------------------------- safety gate

_HOG = None


def people_boxes_hog(img: np.ndarray, width: int = 640) -> Optional[list[tuple[float, float, float, float]]]:
    """Standing people (OpenCV HOG) as (x1, y1, x2, y2) px. Misses seated or partly visible people,
    so it backs up the detector's hand boxes; it doesn't replace the zones or the dwell cap.
    None when this OpenCV has no HOG (5.x dropped it; the rig's 4.x has it): unknown, not "nobody"."""
    global _HOG
    if not hasattr(cv2, "HOGDescriptor"):
        return None
    if _HOG is None:
        _HOG = cv2.HOGDescriptor()
        _HOG.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    s = min(1.0, width / img.shape[1])
    small = cv2.resize(img, None, fx=s, fy=s) if s < 1.0 else img
    rects, _ = _HOG.detectMultiScale(small, winStride=(8, 8), padding=(8, 8), scale=1.05)
    return [(x / s, y / s, (x + w) / s, (y + h) / s) for x, y, w, h in rects]


def beam_blocked(target_px, box_px, blockers, head_px=None, margin_px: float = 20.0) -> bool:
    """A blocker box (person, hand), grown by margin_px, overlaps the target box (or point) or the
    image segment from the head to the target (the beam's path as the camera sees it)."""
    t = np.asarray(target_px, dtype=np.float64)
    tb = box_px if box_px is not None else (t[0], t[1], t[0], t[1])
    for b in blockers:
        x1, y1, x2, y2 = b[0] - margin_px, b[1] - margin_px, b[2] + margin_px, b[3] + margin_px
        if x1 <= tb[2] and tb[0] <= x2 and y1 <= tb[3] and tb[1] <= y2:
            return True
        if head_px is not None:
            ok, _, _ = cv2.clipLine((int(x1), int(y1), int(x2 - x1) + 1, int(y2 - y1) + 1),
                                    (int(head_px[0]), int(head_px[1])), (int(t[0]), int(t[1])))
            if ok:
                return True
    return False


# ---------------------------------------------------------------- CLI

def aim_room_stats(laser, rm: RoomMap, targets, truth=None, boxes=None) -> dict:
    """Aim at each target px; summarise tries, final px error and (with truth()) the true error."""
    rows = []
    for k, t in enumerate(targets):
        r = laser.aim_px(t, None if boxes is None else boxes[k], room_map=rm)
        true = None
        if truth is not None:
            d = truth()
            true = math.inf if d is None else float(np.hypot(d[0] - t[0], d[1] - t[1]))
        rows.append((r, true))
    laser.off()
    errs = [t for _, t in rows if t is not None]
    fin = [e for e in errs if math.isfinite(e)]
    return {"n": len(rows), "on_target": sum(r.on_target for r, _ in rows),
            "tries_median": float(np.median([r.tries for r, _ in rows])) if rows else None,
            "true_err_px_median": float(np.median(fin)) if fin else None,
            "true_err_px_p90": float(np.percentile(fin, 90)) if fin else None,
            "reasons": sorted({r.reason for r, _ in rows})}


def _sim_main(args: argparse.Namespace) -> int:
    from act.sim import RoomRig
    rig = RoomRig(b_cm=args.b_cm, seed=args.seed)
    laser = rig.make_laser()
    t0 = rig.clock.now()
    rm = sweep(laser, grid=tuple(args.grid), n_pairs=1)
    print(f"sweep: {len(rm.pulses)} points, {rm.n_seen} dots seen, {rig.clock.now() - t0:.0f} s simulated")
    rng = np.random.default_rng(args.seed + 1)
    lim = np.array(rm.size_px) - 1
    targets = [np.clip(rm._x[i] + rng.normal(0, 8, 2), 0, lim)
               for i in rng.choice(rm.n_seen, args.aims, replace=False)]
    print(json.dumps(aim_room_stats(laser, rm, targets, rig.true_dot_px), indent=1))
    if args.out:
        rm.save(args.out)
        print(f"wrote {args.out}")
    return 0


def _real_rig(cfg: dict, camera: int):
    """Actuator + camera for a sweep on the rig (no table needed)."""
    from act.actuator import make_actuator
    from act.laser import Laser
    from core.capture import FrameBuffer
    frames = FrameBuffer(camera)
    laser = Laser(make_actuator(cfg), frames, None, "", cfg=cfg)  # type: ignore[arg-type]
    return laser, frames


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sim", action="store_true", help="sweep the simulated room (act/sim.py)")
    ap.add_argument("--sweep", action="store_true", help="sweep the real room (nobody in view)")
    ap.add_argument("--zone", help="add or replace a zone with this name (needs --poly)")
    ap.add_argument("--poly", nargs="+", help="zone polygon as x,y image px pairs")
    ap.add_argument("--delete-zone", help="remove a zone")
    ap.add_argument("--list", action="store_true", help="print the map summary and zones")
    ap.add_argument("--grid", type=int, nargs=2, default=None, metavar=("NX", "NY"))
    ap.add_argument("--b-cm", type=float, default=3.0, help="--sim: pivot offset from the lens")
    ap.add_argument("--aims", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--camera", type=int, default=0, help="--sweep: camera index")
    ap.add_argument("--room-is-clear", action="store_true",
                    help="--sweep without a person detector (OpenCV 5): you checked nobody is in view")
    ap.add_argument("--out", help="where to write the map (default: config room_map)")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if args.sim:
        args.grid = args.grid or [16, 12]
        return _sim_main(args)
    from core.config import load_config
    cfg = load_config()
    room = cfg.get("room") or {}
    path = args.out or cfg.get("room_map", "room_map.json")
    if args.sweep:
        laser, frames = _real_rig(cfg, args.camera)
        clear_s = float(room.get("person_clear_s", 5))
        last = [time.monotonic() - clear_s]
        if people_boxes_hog(np.zeros((128, 64, 3), np.uint8)) is None and not args.room_is_clear:
            print("this OpenCV has no HOG person detector; check nobody is in the room and rerun "
                  "with --room-is-clear")
            return 1

        def stop() -> bool:
            f = frames.latest()
            if f is not None and f.img is not None and people_boxes_hog(f.img):
                last[0] = time.monotonic()
            return time.monotonic() - last[0] < clear_s

        try:
            old = RoomMap.load(path).zones if os.path.exists(path) else {}
            rm = sweep(laser, grid=tuple(args.grid or room.get("grid", (20, 15))),
                       n_pairs=int(room.get("n_pairs", 3)), stop=stop,
                       progress=lambda k, n: k % 20 == 0 and log.info("%d/%d", k, n))
            rm.zones = old
            rm.save(path)
            print(f"wrote {path}: {len(rm.pulses)} points, {rm.n_seen} dots seen, zones kept: {list(old)}")
        except SweepAborted:
            print("sweep aborted: a person came into view; clear the room and run it again")
            return 1
        finally:
            laser.act.close()
            frames.stop()
        return 0
    if not os.path.exists(path):
        print(f"no {path}; run python -m act.room_map --sweep first")
        return 1
    rm = RoomMap.load(path)
    if args.zone:
        if not args.poly or len(args.poly) < 3:
            ap.error("--zone needs --poly with at least 3 x,y points")
        rm.zones[args.zone] = [[float(v) for v in p.split(",")] for p in args.poly]
        rm.save(path)
    if args.delete_zone:
        rm.zones.pop(args.delete_zone, None)
        rm.save(path)
    age_h = (time.time() - rm.t) / 3600
    print(f"{path}: {len(rm.pulses)} points, {rm.n_seen} seen, spacing {rm.spacing_px:.1f} px, "
          f"{age_h:.1f} h old, zones: {', '.join(rm.zones) or 'none (room aims refused)'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
