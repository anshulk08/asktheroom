"""Closed-loop laser pointing (spec H4/H5). Owner: A.

Model: a 2nd-order polynomial per axis maps table cm (x, y) -> servo pulses (pan, tilt), fitted by
least squares on calibration points. aim() predicts, moves, looks for the dot and corrects with the
polynomial's local Jacobian until the dot is within laser_tol_cm (2 cm: SG90-class servos have
1.5-1.8 deg of deadband, so a 1 cm stop makes them hunt).
"""
from __future__ import annotations

import contextlib
import json
import logging
import math
import os
import time
from dataclasses import dataclass, field
from typing import Optional, Protocol

import cv2
import numpy as np

from act.actuator import Actuator, Clock
from core.types import Frame, Point

log = logging.getLogger(__name__)

DEFAULT_LATENCY_S = 0.15       # safe until act.calibrate --rig measures camera_latency_s (too short: stale frames)
FIRST_DOT_CM = 15.0            # a dot further than this from the target is something else (sleeve, reflection)
MAX_MISSES = 3                 # aim() gives up after this many looks in a row without the dot


class FrameSource(Protocol):
    def latest(self) -> Optional[Frame]: ...
    def at(self, t: float) -> Optional[Frame]: ...


class TableMap(Protocol):
    def px_to_cm(self, pts: np.ndarray) -> np.ndarray: ...
    def cm_to_px(self, pts: np.ndarray) -> np.ndarray: ...


# ---------------------------------------------------------------- dot detection

def _largest_blob(mask: np.ndarray, weight: np.ndarray, min_area: int, max_area: int
                  ) -> Optional[tuple[float, float]]:
    """Largest blob with min_area <= area <= max_area (a bigger one, e.g. a red sleeve, is skipped)."""
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n <= 1:
        return None
    areas = stats[1:, cv2.CC_STAT_AREA]
    ok = (areas >= min_area) & (areas <= max_area)
    if not ok.any():
        return None
    i = 1 + int(np.argmax(np.where(ok, areas, -1)))
    x, y, w, h = (int(stats[i, k]) for k in (cv2.CC_STAT_LEFT, cv2.CC_STAT_TOP,
                                              cv2.CC_STAT_WIDTH, cv2.CC_STAT_HEIGHT))
    m = (lab[y:y + h, x:x + w] == i)
    wt = np.where(m, np.maximum(weight[y:y + h, x:x + w].astype(np.float64), 1.0), 0.0)
    ys, xs = np.mgrid[y:y + h, x:x + w]
    s = wt.sum()
    return float((xs * wt).sum() / s), float((ys * wt).sum() / s)


def dot_score(off: np.ndarray, on: np.ndarray) -> np.ndarray:
    """Red rise minus half the green/blue rise (a saturated white core still scores; a hand or
    lighting change, which moves all channels, doesn't)."""
    d = on.astype(np.int16) - off.astype(np.int16)
    return d[..., 2] - (np.maximum(d[..., 0], d[..., 1]) >> 1)


def dot_px_diff(off: np.ndarray, on: np.ndarray, thr: int = 40, min_area: int = 2,
                max_area: int = 3000) -> Optional[tuple[float, float]]:
    """Laser dot from an off/on BGR pair (see dot_score)."""
    score = dot_score(off, on)
    mask = (score > thr).astype(np.uint8)
    return _largest_blob(mask, score, min_area, max_area)


def _dot_blobs(mask: np.ndarray, weight: np.ndarray, min_area: int, max_area: int,
               max_aspect: float = 4.0, min_fill: float = 0.3) -> list[tuple[float, float, float]]:
    """Every dot-shaped blob (area in range, not a streak by aspect, not a ring or edge by fill) as
    (x, y, peak): the centre of gravity of the score over the blob's bounding box, and its peak score."""
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    out = []
    for i in range(1, n):
        x, y, w, h, area = (int(stats[i, k]) for k in (cv2.CC_STAT_LEFT, cv2.CC_STAT_TOP,
                                                       cv2.CC_STAT_WIDTH, cv2.CC_STAT_HEIGHT,
                                                       cv2.CC_STAT_AREA))
        if area < min_area or area > max_area or max(w, h) > max_aspect * min(w, h) \
                or area < min_fill * w * h:
            continue
        m = lab[y:y + h, x:x + w] == i
        wt = np.where(m, np.maximum(weight[y:y + h, x:x + w].astype(np.float64), 1.0), 0.0)
        ys, xs = np.mgrid[y:y + h, x:x + w]
        s = wt.sum()
        out.append((float((xs * wt).sum() / s), float((ys * wt).sum() / s), float(weight[y:y + h, x:x + w][m].max())))
    return out


def _dot_blob(mask: np.ndarray, weight: np.ndarray, min_area: int, max_area: int,
              max_aspect: float = 4.0, min_fill: float = 0.3) -> Optional[tuple[float, float]]:
    """Brightest dot-shaped blob (see _dot_blobs), or None."""
    blobs = _dot_blobs(mask, weight, min_area, max_area, max_aspect, min_fill)
    if not blobs:
        return None
    x, y, _ = max(blobs, key=lambda b: b[2])
    return x, y


def dot_px_hsv(on: np.ndarray, min_area: int = 2, max_area: int = 1500) -> Optional[tuple[float, float]]:
    """Fallback on a single frame: bright saturated red (hue wraps at 0/180). Can false-fire on
    warm, brightly lit surfaces, so find_dot uses it only when no off frame arrived."""
    hsv = cv2.cvtColor(on, cv2.COLOR_BGR2HSV)
    lo = cv2.inRange(hsv, (0, 120, 230), (8, 255, 255))
    hi = cv2.inRange(hsv, (172, 120, 230), (180, 255, 255))
    mask = ((lo | hi) > 0).astype(np.uint8)
    return _largest_blob(mask, hsv[..., 2], min_area, max_area)


# ---------------------------------------------------------------- table frame

def homography(src, dst) -> np.ndarray:
    """3x3 H with dst ~ H @ src from 4 point pairs, in float64 (cv2.getPerspectiveTransform is float32)."""
    A, b = [], []
    for (x, y), (u, v) in zip(np.asarray(src, dtype=np.float64), np.asarray(dst, dtype=np.float64)):
        A += [[x, y, 1, 0, 0, 0, -u * x, -u * y], [0, 0, 0, x, y, 1, -v * x, -v * y]]
        b += [u, v]
    return np.append(np.linalg.solve(np.asarray(A), np.asarray(b)), 1.0).reshape(3, 3)


def apply_h(H: np.ndarray, pts) -> np.ndarray:
    p = np.asarray(pts, dtype=np.float64)
    q = p.reshape(-1, 2) @ H[:, :2].T + H[:, 2]
    return (q[:, :2] / q[:, 2:3]).reshape(p.shape)


def table_px_to_cm(table: "TableMap", size_cm) -> Optional[np.ndarray]:
    """The table's current px -> cm homography, read through its cm_to_px (so any TableMap works).
    None if the table isn't calibrated."""
    w, h = (float(v) for v in size_cm)
    cm = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float64)
    try:
        px = np.asarray(table.cm_to_px(cm), dtype=np.float64).reshape(4, 2)
        return homography(px, cm) if np.isfinite(px).all() else None
    except Exception:  # noqa: BLE001 - uncalibrated table, degenerate corners
        return None


# ---------------------------------------------------------------- calibration model

def _features(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    return np.stack([np.ones_like(u), u, v, u * u, u * v, v * v], axis=-1)


@dataclass
class LaserFit:
    """pulses = F(u, v) @ coef, with u = (x - cx)/s, v = (y - cy)/s (normalised for conditioning)."""
    coef: np.ndarray                       # 6x2: columns pan, tilt
    norm: tuple[float, float, float]       # cx, cy, s
    fit_error_cm: dict = field(default_factory=dict)   # {'median':, 'max':}
    n_points: int = 0
    grid: Optional[int] = None
    timestamp: float = 0.0
    # The table's px -> cm homography when the fit was made. The fit's cm are that table frame's cm, so
    # after a table refit Laser maps new cm -> px -> these cm (exact while the camera and head stay put).
    table_px_to_cm: Optional[np.ndarray] = None
    servo_limits: Optional[dict] = None    # the limits it was fitted within (for the record)

    def _uv(self, xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        cx, cy, s = self.norm
        return (xy[..., 0] - cx) / s, (xy[..., 1] - cy) / s

    def predict(self, xy) -> np.ndarray:
        """cm (2,) or (N,2) -> pulses (pan, tilt) of the same shape."""
        xy = np.asarray(xy, dtype=np.float64)
        u, v = self._uv(xy)
        return _features(u, v) @ self.coef

    def jacobian(self, xy) -> np.ndarray:
        """2x2 d(pan, tilt)/d(x, y) at one point, in µs per cm."""
        u, v = self._uv(np.asarray(xy, dtype=np.float64))
        s = self.norm[2]
        du = np.array([0.0, 1.0, 0.0, 2 * u, v, 0.0]) / s
        dv = np.array([0.0, 0.0, 1.0, 0.0, u, 2 * v]) / s
        return np.stack([du @ self.coef, dv @ self.coef], axis=1)   # rows: pan, tilt

    def residuals_cm(self, xy: np.ndarray, pulses: np.ndarray) -> np.ndarray:
        """Per-point fit error, pulse residual converted to cm through the local Jacobian."""
        r = self.predict(xy) - pulses
        return np.array([np.linalg.norm(np.linalg.solve(self.jacobian(p), ri))
                         for p, ri in zip(xy, r)])

    def to_dict(self) -> dict:
        return {"model": "poly2", "coef": self.coef.tolist(), "norm": list(self.norm),
                "fit_error_cm": self.fit_error_cm, "n_points": self.n_points, "grid": self.grid,
                "timestamp": self.timestamp, "servo_limits": self.servo_limits,
                "table_px_to_cm": None if self.table_px_to_cm is None else self.table_px_to_cm.tolist()}

    @classmethod
    def from_dict(cls, d: dict) -> "LaserFit":
        H = d.get("table_px_to_cm")
        return cls(np.asarray(d["coef"], dtype=np.float64), tuple(d["norm"]),  # type: ignore[arg-type]
                   d.get("fit_error_cm", {}), int(d.get("n_points", 0)), d.get("grid"),
                   float(d.get("timestamp", 0.0)),
                   None if H is None else np.asarray(H, dtype=np.float64), d.get("servo_limits"))

    def save(self, path: str) -> None:
        d = os.path.dirname(os.path.abspath(path))
        os.makedirs(d, exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(self.to_dict(), f, indent=1)
        os.replace(tmp, path)

    @classmethod
    def load(cls, path: str) -> "LaserFit":
        with open(path) as f:
            return cls.from_dict(json.load(f))


def fit_poly(xy_cm, pulses, grid: Optional[int] = None) -> LaserFit:
    """Least-squares fit of pulses = poly2(cm). Needs >= 6 points (use >= 9)."""
    xy = np.asarray(xy_cm, dtype=np.float64).reshape(-1, 2)
    p = np.asarray(pulses, dtype=np.float64).reshape(-1, 2)
    if len(xy) < 6:
        raise ValueError(f"need >= 6 calibration points, got {len(xy)}")
    c = xy.mean(axis=0)
    s = float(max(np.ptp(xy[:, 0]), np.ptp(xy[:, 1]), 1e-6) / 2)
    tmp = LaserFit(np.zeros((6, 2)), (float(c[0]), float(c[1]), s))
    u, v = tmp._uv(xy)
    coef, *_ = np.linalg.lstsq(_features(u, v), p, rcond=None)
    fit = LaserFit(coef, tmp.norm, n_points=len(xy), grid=grid, timestamp=time.time())
    e = fit.residuals_cm(xy, p)
    fit.fit_error_cm = {"median": float(np.median(e)), "max": float(e.max())}
    return fit


# ---------------------------------------------------------------- laser

EDGES = ("left", "right", "top", "bottom")


@dataclass
class PxAim:
    """Result of Laser.aim_px. on_target: the dot was last seen inside the (shrunk) box or within
    tol_px. seen: the dot was detected at least once. reason: in_box | within_tol | max_tries |
    not_seen | lost | jumped | unmapped."""
    err_px: float
    on_target: bool
    seen: bool
    tries: int
    dot_px: Optional[tuple[float, float]]
    reason: str
    first_err_px: Optional[float] = None


def _in_box(p, box, shrink: float = 0.2) -> bool:
    x1, y1, x2, y2 = box
    mx, my = (x2 - x1) * shrink / 2, (y2 - y1) * shrink / 2
    return x1 + mx <= p[0] <= x2 - mx and y1 + my <= p[1] <= y2 - my


class FullFrames:
    """The full camera frames behind a core.room_view.TableView (latest_full / full_at) as a FrameSource: room
    pointing (aim_px, the dot sweep) works in full-frame px, while table aims keep the table view."""

    def __init__(self, view):
        self.view = view

    def latest(self) -> Optional[Frame]:
        return self.view.latest_full()

    def at(self, t: float) -> Optional[Frame]:
        return self.view.full_at(t)

    def stop(self) -> None:
        stop = getattr(self.view, "stop", None)
        if stop is not None:
            stop()


class Laser:
    """Points the laser at table positions. Holds act.lock (if any) for each high-level action."""

    def __init__(self, act: Actuator, frames: FrameSource, table: TableMap, cal_path: str, *,
                 cfg: Optional[dict] = None, clock: Optional[Clock] = None,
                 latency_s: Optional[float] = None):
        cfg = cfg or {}
        self.act, self.frames, self.table, self.cal_path = act, frames, table, cal_path
        self.cfg = cfg
        self.clock: Clock = clock or getattr(act, "clock", None) or Clock()
        self.latency_s = float(latency_s if latency_s is not None
                               else cfg.get("camera_latency_s", DEFAULT_LATENCY_S))
        self.frame_timeout_s = float(cfg.get("laser_frame_timeout_s", 1.0))
        self.settle_s = 0.15            # after a move, before looking
        self.gain = 0.7
        self.tol_cm = float(cfg.get("laser_tol_cm", 2.0))
        self.max_tries = 8
        self.anti_backlash_us = float(cfg.get("laser_anti_backlash_us", 25))
        self.diff_thr = int(cfg.get("laser_diff_thr", 40))
        self.max_dot_px = int(cfg.get("laser_max_dot_px", 3000))
        self.first_dot_cm = float(cfg.get("laser_first_dot_cm", FIRST_DOT_CM))
        self.table_size = tuple((cfg.get("table") or {}).get("size_cm", (90, 60)))
        room = cfg.get("room") or {}
        self.room_map = None            # act.room_map.RoomMap, set by the caller (main.py) when enabled
        self.px_frames: Optional[FrameSource] = None   # room pointing's frames (FullFrames); None: self.frames
        self.room_n_pairs = int(room.get("n_pairs", 3))
        self.tol_px = float(room.get("tol_px", 12))
        self.deadband_us = float(room.get("deadband_us", 8))
        self.jump_px = float(room.get("jump_px", 30))
        self.max_map_gap_px = float(room.get("max_map_gap_px", 60))
        # Eye safety (laser_room:): an aim's laser is never lit longer than this, counted from its first light;
        # blinks and re-lights don't reset it (the actuator and the firmware timers alone would).
        self.max_on_s = float((cfg.get("laser_room") or {}).get("max_on_s", 4.0))
        self.fit: Optional[LaserFit] = None
        self.disabled: Optional[str] = None   # why the hardware isn't usable (set by the app); aims refuse
        self.state = {"on": False, "target": None, "err_cm": None}
        self.last_aim: dict = {}
        self.last_frames: tuple[Optional[Frame], Optional[Frame]] = (None, None)
        if cal_path and os.path.exists(cal_path):
            self.fit = LaserFit.load(cal_path)

    # -- helpers
    def _locked(self):
        lk = getattr(self.act, "lock", None)
        return lk if lk is not None else contextlib.nullcontext()

    def _need_fit(self) -> LaserFit:
        if self.disabled:
            raise RuntimeError(f"laser disabled ({self.disabled})")
        if self.fit is None:
            raise RuntimeError(f"laser not calibrated (no {self.cal_path}); run python -m act.calibrate --rig")
        return self.fit

    def to_fit_cm(self, cm) -> np.ndarray:
        """Current table cm -> the cm the fit was made in (through the camera pixel; the identity when the
        table hasn't been refitted, or for a fit saved before the homography was stored)."""
        cm = np.asarray(cm, dtype=np.float64)
        H = self.fit.table_px_to_cm if self.fit is not None else None
        if H is None:
            return cm
        try:
            px = np.asarray(self.table.cm_to_px(cm.reshape(-1, 2)), dtype=np.float64)
        except Exception:  # noqa: BLE001 - table not calibrated: nothing to map through
            return cm
        return apply_h(H, px).reshape(cm.shape)

    def refit_moved_cm(self) -> Optional[float]:
        """How far (cm, max over the table corners) the current table frame is from the fit's. None if
        the fit has no stored homography or the table isn't calibrated."""
        if self.fit is None or self.fit.table_px_to_cm is None or table_px_to_cm(self.table, self.table_size) is None:
            return None
        w, h = self.table_size
        c = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float64)
        return float(np.linalg.norm(self.to_fit_cm(c) - c, axis=1).max())

    def _clamp(self, pan: float, tilt: float) -> tuple[float, float]:
        (plo, phi), (tlo, thi) = self.act.limits()
        return min(phi, max(plo, float(pan))), min(thi, max(tlo, float(tilt)))

    def _move_dark(self, pan: float, tilt: float, duration_s: float = 0.3) -> None:
        """Every move is made with the laser off: a lit head slewing between spots sweeps the room unchecked."""
        self.act.laser(False)
        self.move_to(pan, tilt, duration_s=duration_s)

    def _over_budget(self, lit_at: Optional[float]) -> bool:
        return lit_at is not None and self.clock.now() - lit_at >= self.max_on_s

    def move_to(self, pan: float, tilt: float, duration_s: float = 0.3) -> None:
        """Move, always finishing from below on both axes so servo backlash is repeatable
        (calibration and aiming then see the same offset, which the fit absorbs)."""
        pan, tilt = self._clamp(pan, tilt)
        a = self.anti_backlash_us
        if a > 0:
            self.act.move(*self._clamp(pan - a, tilt - a), duration_s=duration_s)
            self.act.move(pan, tilt, duration_s=0.06)
        else:
            self.act.move(pan, tilt, duration_s=duration_s)

    @property
    def px_source(self) -> FrameSource:
        """Where room pointing looks for the dot: the full camera frame when the app runs a table view."""
        return self.px_frames if self.px_frames is not None else self.frames

    def _grab(self, on: bool, src: Optional[FrameSource] = None) -> Optional[Frame]:
        """Switch the laser, then return the first frame captured after the switch + camera latency.
        Frame.t is stamped when the frame leaves the pipeline, which lags the scene by ~2-3 frames,
        so a fixed short sleep would return a frame that still shows the old laser state."""
        self.act.laser(on)
        t_cmd = self.clock.now()
        deadline = t_cmd + self.latency_s + self.frame_timeout_s
        while True:
            f = (src or self.frames).latest()
            if f is not None and f.t > t_cmd + self.latency_s:
                return f
            if self.clock.now() > deadline:
                log.warning("no fresh frame %.2f s after laser %s", self.frame_timeout_s, on)
                return None
            self.clock.sleep(0.005)

    # -- spec API
    def find_dot_px(self, n_pairs: int = 1, gate: bool = True,
                    roi: Optional[tuple[int, int, int, int]] = None,
                    unique: bool = False, src: Optional[FrameSource] = None) -> Optional[tuple[float, float]]:
        """Blink the laser n_pairs times and return the dot in image px (None if not seen). Laser ends on.
        The pairs' scores are averaged, which lifts a dim far dot out of sensor noise (SNR ~ sqrt(n));
        the threshold drops by sqrt(n) to keep the false-alarm rate. gate: pick the brightest
        dot-shaped blob instead of the largest one. roi: (x0, y0, x1, y1) px; only look inside it.
        unique (with gate): None unless exactly one dot-shaped blob shows (faint specks under half the
        brightest one's peak don't count). src: the frames to look in (default self.frames, the table view;
        room pointing passes px_source, the full frame)."""
        with self._locked():
            acc, n, off, on = None, 0, None, None
            for _ in range(max(1, n_pairs)):
                off = self._grab(False) if src is None else self._grab(False, src)
                on = self._grab(True) if src is None else self._grab(True, src)
                if on is None:
                    break
                if off is not None:
                    sc = dot_score(off.img, on.img).astype(np.float32)
                    acc = sc if acc is None else acc + sc
                    n += 1
            self.last_frames = (off, on)
            if on is None:
                return None
            x0, y0, x1, y1 = 0, 0, on.img.shape[1], on.img.shape[0]
            if roi is not None:
                x0, y0 = max(0, int(roi[0])), max(0, int(roi[1]))
                x1, y1 = min(x1, int(math.ceil(roi[2]))), min(y1, int(math.ceil(roi[3])))
                if x0 >= x1 or y0 >= y1:
                    return None             # the region is off the picture
            if acc is None:
                # Fallback only without an off frame: when the diff is possible and empty, the dot
                # really isn't visible, and a single-frame red mask would fire on warm surfaces.
                d = dot_px_hsv(on.img[y0:y1, x0:x1], max_area=self.max_dot_px // 2)
                return None if d is None else (d[0] + x0, d[1] + y0)
            score = acc / n
            mask = (score > self.diff_thr / math.sqrt(n)).astype(np.uint8)
            if roi is not None:
                keep = np.zeros_like(mask)
                keep[y0:y1, x0:x1] = 1
                mask &= keep
            if gate and unique:        # a second blob half as bright or more (a glint) makes it ambiguous
                blobs = sorted(_dot_blobs(mask, score, 2, self.max_dot_px), key=lambda b: -b[2])
                if not blobs or (len(blobs) > 1 and blobs[1][2] >= 0.5 * blobs[0][2]):
                    return None
                return blobs[0][0], blobs[0][1]
            if gate:
                return _dot_blob(mask, score, 2, self.max_dot_px)
            return _largest_blob(mask, score, 2, self.max_dot_px)

    def find_dot(self, near_cm=None, radius_cm: Optional[float] = None) -> Optional[Point]:
        """Blink the laser and return the dot position in table cm (None if not seen). Laser ends on.
        Picks the brightest dot-shaped blob. near_cm: look only within radius_cm (default
        first_dot_cm) of it, and reject a dot further than that, so a red sleeve, skin or a
        reflection elsewhere on the table is never taken for the dot."""
        roi = None
        if near_cm is not None:
            r = self.first_dot_cm if radius_cm is None else float(radius_cm)
            c = np.asarray(near_cm, dtype=np.float64)
            box = self.table.cm_to_px(c + np.array([[-r, -r], [r, -r], [r, r], [-r, r]]))
            roi = (*np.floor(box.min(axis=0)), *np.ceil(box.max(axis=0)))
        px = self.find_dot_px(1, gate=True, roi=roi)
        if px is None:
            return None
        cm = self.table.px_to_cm(np.array([px], dtype=np.float64)).reshape(-1)
        if near_cm is not None and math.dist(cm, near_cm) > r:
            return None
        return float(cm[0]), float(cm[1])

    def find_dot_wide(self, n_pairs: int = 3) -> Optional[Point]:
        """The whole picture, averaged over n_pairs blinks, when the dot isn't near the target (the head or
        camera was knocked, or the fit is off). Only an unambiguous answer counts: one dot-shaped blob
        and no other half as bright, else None. Laser ends on."""
        px = self.find_dot_px(n_pairs, gate=True, unique=True)
        if px is None:
            return None
        cm = self.table.px_to_cm(np.array([px], dtype=np.float64)).reshape(-1)
        return float(cm[0]), float(cm[1])

    def aim(self, target_cm: tuple, mode: str = "point", check=None) -> float:
        """Point at target_cm. mode 'point' closes the loop on the seen dot (<= 8 tries, stop < tol_cm);
        'open' just moves to the prediction and measures once. Only a dot within first_dot_cm of the
        target counts; if the first look misses, one whole-picture search (find_dot_wide) may find a
        dot that is further off (knocked head), and the loop corrects from there. After MAX_MISSES
        looks in a row without it, stop and hold the pose. Returns the last measured error in cm (inf if
        the dot was never seen). Leaves the laser on, except when the dot was never seen: an unconfirmed
        dot could be anywhere, so the laser goes off. The actuator's auto-off timer restarts.
        Eye safety: every move is dark; check() (if given) runs before each look and a reason it returns
        stops the aim ('unsafe'); the aim stops at max_on_s after its first light ('budget'); the laser is
        left on only for 'within_tol'."""
        fit = self._need_fit()
        target = np.asarray(target_cm, dtype=np.float64)
        target_fit = self.to_fit_cm(target)
        tries = 1 if mode == "open" else self.max_tries
        err, first, n, misses, reason, wide = math.inf, None, 0, 0, "max_tries", False
        lit_at, unsafe = None, None
        with self._locked():
            cmd = np.array(self._clamp(*fit.predict(target_fit)))
            self._move_dark(*cmd)
            for i in range(tries):
                self.clock.sleep(self.settle_s)
                unsafe = check() if check is not None else None
                if unsafe is not None:
                    reason = "unsafe"
                    break
                if self._over_budget(lit_at):
                    reason = "budget"
                    break
                lit_at = self.clock.now() if lit_at is None else lit_at
                dot = self.find_dot(near_cm=target)
                n += 1
                if dot is None and first is None and not wide and mode != "open":
                    wide = True
                    dot = self.find_dot_wide()
                    if dot is not None:
                        log.warning("laser dot %.0f cm from the target: head or camera moved? (found by a "
                                    "whole-picture search)", math.dist(dot, target))
                if dot is None:         # occluded or missed: look again (counts as a try)
                    misses += 1
                    if misses >= MAX_MISSES:
                        reason = "lost" if first is not None else "not_seen"
                        break
                    continue
                misses = 0
                err = float(np.linalg.norm(target - np.asarray(dot)))
                first = err if first is None else first
                if err < self.tol_cm:
                    reason = "within_tol"
                    break
                if i == tries - 1:
                    break
                e = target_fit - self.to_fit_cm(np.asarray(dot))
                cmd = np.array(self._clamp(*(cmd + self.gain * fit.jacobian(target_fit) @ e)))
                self._move_dark(*cmd, duration_s=0.1)
            if first is None and reason not in ("unsafe", "budget"):
                reason = "not_seen"
            lit = reason == "within_tol"             # only a confirmed hit stays lit ('lost' is often a hand)
            self.act.laser(lit)
        self.last_aim = {"tries": n, "first_err_cm": first, "err_cm": err, "reason": reason, "wide": wide,
                         "unsafe": unsafe, "lit_s": 0.0 if lit_at is None else self.clock.now() - lit_at}
        self.state = {"on": lit, "target": None,
                      "err_cm": None if math.isinf(err) else round(err, 2)}
        return err

    def aim_object(self, name: str, target_cm: tuple) -> float:
        """aim(), but for shiny objects (cfg shiny_objects) aim shiny_offset_cm toward the table
        centre so the dot lands on the table next to the object instead of glinting off it."""
        tgt = np.asarray(target_cm, dtype=np.float64)
        if name in (self.cfg.get("shiny_objects") or []):
            off = float(self.cfg.get("shiny_offset_cm", 3))
            d = np.asarray(self.table_size, dtype=np.float64) / 2 - tgt
            n = float(np.linalg.norm(d))
            tgt = tgt + (d / n if n > 1e-6 else np.array([1.0, 0.0])) * off
        err = self.aim(tuple(tgt))
        self.state["target"] = name
        return err

    def aim_px(self, target_px, box_px=None, *, room_map=None, n_pairs: Optional[int] = None,
               tol_px: Optional[float] = None, check=None) -> PxAim:
        """Point at image pixel target_px anywhere in the room (spec 0006), with no depth: the dot seen
        inside the object's box is on the object. Feedforward from the room dot map, then a P step
        through the local pixel/pulse Jacobian, which a Broyden update corrects after every step
        (a Jacobian mapped on the floor is ~2x off on a near shelf). Stops when the dot is inside
        box_px shrunk 20%, or within tol_px. Leaves the laser on only when on_target; off otherwise (dot
        never seen, lost, or it jumped: something nearer is in the way). Eye safety: every move is dark;
        check() (if given) runs before each look, and a reason it returns stops the aim ('unsafe'); a jump
        stops it at once; the aim stops at max_on_s after its first light ('budget')."""
        rm = room_map if room_map is not None else self.room_map
        if rm is None:
            raise RuntimeError("no room map; run python -m act.room_map --sweep")
        target = np.asarray(target_px, dtype=np.float64)
        tol = self.tol_px if tol_px is None else float(tol_px)
        pairs = self.room_n_pairs if n_pairs is None else int(n_pairs)
        g = rm.pulses_for_px(target, max_gap_px=self.max_map_gap_px)
        if g is None:
            return PxAim(math.inf, False, False, 0, None, "unmapped")
        J = np.array(g.J, dtype=np.float64)                          # px per µs
        cap = 2.0 * np.asarray(rm.step_us, dtype=np.float64)
        n, misses, first, err = 0, 0, None, math.inf
        seen, jumped, dot, reason = False, False, None, "max_tries"
        prev_cmd = prev_dot = None
        lit_at, unsafe = None, None
        with self._locked():
            cmd = np.array(self._clamp(*g.pulses))
            self._move_dark(*cmd)
            for i in range(self.max_tries):
                self.clock.sleep(self.settle_s)
                unsafe = check() if check is not None else None
                if unsafe is not None:
                    reason = "unsafe"
                    break
                if self._over_budget(lit_at):
                    reason = "budget"
                    break
                lit_at = self.clock.now() if lit_at is None else lit_at
                d = self.find_dot_px(pairs, src=self.px_source)
                n += 1
                if d is None:
                    dot, misses = None, misses + 1
                    if not seen and misses >= 3:
                        reason = "not_seen"
                        break
                    continue
                dot, seen = np.asarray(d), True
                if prev_dot is not None:
                    du, ds = cmd - prev_cmd, dot - prev_dot
                    pred = J @ du
                    if np.linalg.norm(ds - pred) > self.jump_px + 1.5 * np.linalg.norm(pred):
                        jumped = True                   # discontinuity: landed on something nearer (a
                        reason = "jumped"               # person?): stop at once, dark
                        break
                    elif np.linalg.norm(du) > self.deadband_us:
                        J2 = J + np.outer(ds - pred, du) / float(du @ du)
                        if np.linalg.det(J2) * np.linalg.det(J) > 0 and np.linalg.cond(J2) < 1e3:
                            J = J2
                e = target - dot
                err = float(np.linalg.norm(e))
                first = err if first is None else first
                if box_px is not None and _in_box(dot, box_px):
                    reason = "in_box"
                    break
                if err < tol:
                    reason = "within_tol"
                    break
                if i == self.max_tries - 1:
                    break
                step = self.gain * np.linalg.solve(J, e)
                k = float(np.max(np.abs(step) / cap))
                prev_cmd, prev_dot = cmd, dot
                cmd = np.array(self._clamp(*(cmd + (step / k if k > 1 else step))))
                self._move_dark(*cmd, duration_s=0.1)
            if seen and dot is None and reason == "max_tries":
                reason = "lost"
            ok = reason in ("in_box", "within_tol") and not jumped
            self.act.laser(ok)
        res = PxAim(err, ok, seen, n, None if dot is None else (float(dot[0]), float(dot[1])),
                    reason, first)
        self.last_aim = {"tries": n, "first_err_px": first, "err_px": err, "reason": reason, "unsafe": unsafe,
                         "lit_s": 0.0 if lit_at is None else self.clock.now() - lit_at}
        self.state = {"on": ok, "target": None, "err_cm": None,
                      "err_px": None if math.isinf(err) else round(err, 1)}
        return res

    def _trace(self, pts_cm: np.ndarray, seg_s: float) -> None:
        """Trace a path lit: reach its start dark, light, follow it, and stop (dark) at max_on_s."""
        fit = self._need_fit()
        with self._locked():
            pul = fit.predict(self.to_fit_cm(pts_cm))
            self.act.laser(False)
            self.act.move(*self._clamp(*pul[0]), duration_s=0.3)
            self.act.laser(True)
            lit_at, lit = self.clock.now(), True
            for p in pul[1:]:
                if self._over_budget(lit_at):
                    self.act.laser(False)
                    lit = False
                    break
                self.act.move(*self._clamp(*p), duration_s=seg_s)
        self.last_aim = {"reason": "trace", "lit_s": self.clock.now() - lit_at}
        self.state["on"] = lit

    def sweep_edge(self, edge: str, inset_cm: float = 4.0, passes: int = 2) -> None:
        """Run the dot back and forth along one table edge ('carried off the left side')."""
        if edge not in EDGES:
            raise ValueError(f"edge must be one of {EDGES}, got {edge!r}")
        w, h = self.table_size
        m = 5.0
        a, b = {"left": ((inset_cm, m), (inset_cm, h - m)),
                "right": ((w - inset_cm, m), (w - inset_cm, h - m)),
                "top": ((m, inset_cm), (w - m, inset_cm)),
                "bottom": ((m, h - inset_cm), (w - m, h - inset_cm))}[edge]
        s = np.linspace(0, 1, 12)[:, None]
        line = np.asarray(a) + s * (np.asarray(b) - np.asarray(a))
        pts = [line if k % 2 == 0 else line[::-1] for k in range(passes)]
        self._trace(np.concatenate(pts), seg_s=0.06)
        self.state["target"] = f"edge:{edge}"

    def circle(self, center_cm: tuple, r_cm: float = 5, laps: int = 2) -> None:
        """Trace a circle around a last-seen position ('I lost track, last seen here')."""
        n = 24
        a = np.linspace(0, 2 * np.pi * laps, n * laps + 1)
        c = np.asarray(center_cm, dtype=np.float64)
        self._trace(c + r_cm * np.stack([np.cos(a), np.sin(a)], axis=1), seg_s=0.04)

    def off(self) -> None:
        self.act.laser(False)
        self.state = {"on": False, "target": None, "err_cm": None}
