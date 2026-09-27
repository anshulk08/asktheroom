"""The table view over the full camera frame (spec 0009 M0, Capture).

With room memory on, the Brio runs at 1920x1080, zoom 100. TableView wraps the FrameBuffer and gives every
existing latest()/at(t) caller the zoom-160 region (table_view_rect, full-frame px) resized to 1280x720,
so the table pipeline is unchanged; latest_full()/full_at(t) give the room pass the full frame.
core/capture.py does not change. measure_rect fits table_view_rect once per rig by ECC-aligning a zoom-160
reference frame into a zoom-100 full frame; the result goes in config.local.yaml.
Python 3.10 (JetPack 6).
"""
from __future__ import annotations

import threading
from collections import OrderedDict
from typing import Any, Optional, Sequence

import cv2
import numpy as np

from core.types import BoxPx, Frame

OUT_SIZE = (1280, 720)
_CACHE_N = 4                                   # table-view cuts kept, by source frame idx
_ECC_MAX_W = 960                               # ECC runs with the full frame at most this wide


def default_rect(size_px: Sequence[int], zoom: int = 100, ref_zoom: int = 160,
                 out_size: Sequence[int] = OUT_SIZE) -> BoxPx:
    """A centred rect covering 1/(ref_zoom/zoom) of the frame width, with out_size's aspect: a first guess
    at table_view_rect if the camera's zoom were a pure centre crop (measure_rect gives the real one)."""
    fw, fh = int(size_px[0]), int(size_px[1])
    w = int(round(fw * zoom / ref_zoom))
    h = int(round(w * out_size[1] / out_size[0]))
    if h > fh:                                 # a frame taller-aspect than out_size: height limits
        h = fh
        w = int(round(h * out_size[0] / out_size[1]))
    x1 = int(round((fw - w) / 2))
    y1 = int(round((fh - h) / 2))
    return (x1, y1, x1 + w, y1 + h)


def cut(img: np.ndarray, rect: Sequence[int], out_size: Sequence[int] = OUT_SIZE) -> np.ndarray:
    """img[y1:y2, x1:x2] resized to out_size: INTER_AREA shrinking, INTER_LINEAR enlarging."""
    x1, y1, x2, y2 = (int(v) for v in rect)
    h, w = img.shape[:2]
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"rect {tuple(rect)} is empty inside a {w}x{h} image")
    crop = img[y1:y2, x1:x2]
    ow, oh = int(out_size[0]), int(out_size[1])
    shrink = (x2 - x1) >= ow and (y2 - y1) >= oh
    return cv2.resize(crop, (ow, oh), interpolation=cv2.INTER_AREA if shrink else cv2.INTER_LINEAR)


class TableView:
    """The FrameSource API (latest, at, wait_new, fps, stop) over a full-frame source, returning table-view
    Frames: same t, wall and idx, img cut to `rect` and resized to out_size (None stays None).
    latest_full()/full_at(t) return the source's own frames. Anything else is the source's attribute."""

    def __init__(self, source: Any, rect: Sequence[int], out_size: Sequence[int] = OUT_SIZE):
        self.source = source
        self.rect: BoxPx = tuple(int(v) for v in rect)
        self.out_size = (int(out_size[0]), int(out_size[1]))
        self._lock = threading.Lock()
        # idx -> (source img, table-view Frame); the img identity check keeps a reused idx from matching
        self._cache: "OrderedDict[int, tuple[np.ndarray, Frame]]" = OrderedDict()

    def _view(self, f: Optional[Frame]) -> Optional[Frame]:
        if f is None:
            return None
        if f.img is None:
            return Frame(t=f.t, wall=f.wall, img=None, idx=f.idx)
        with self._lock:
            hit = self._cache.get(f.idx)
            if hit is not None and hit[0] is f.img:
                self._cache.move_to_end(f.idx)
                return hit[1]
        # cut outside the lock: two threads may cut the same frame once each, which is harmless
        v = Frame(t=f.t, wall=f.wall, img=cut(f.img, self.rect, self.out_size), idx=f.idx)
        with self._lock:
            self._cache[f.idx] = (f.img, v)
            self._cache.move_to_end(f.idx)
            while len(self._cache) > _CACHE_N:
                self._cache.popitem(last=False)
        return v

    def latest(self) -> Optional[Frame]:
        return self._view(self.source.latest())

    def at(self, t: float) -> Optional[Frame]:
        return self._view(self.source.at(t))

    def wait_new(self, after_idx: int, timeout: float = 1.0) -> Optional[Frame]:
        return self._view(self.source.wait_new(after_idx, timeout))

    def latest_full(self) -> Optional[Frame]:
        return self.source.latest()

    def full_at(self, t: float) -> Optional[Frame]:
        return self.source.at(t)

    @property
    def fps(self) -> float:
        return self.source.fps

    def stop(self) -> None:
        self.source.stop()

    def __getattr__(self, name: str) -> Any:
        # only reached for names TableView lacks; guard against lookups before __init__ set `source`
        src = self.__dict__.get("source")
        if src is None:
            raise AttributeError(name)
        return getattr(src, name)


def _grey(img: np.ndarray) -> np.ndarray:
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    return g.astype(np.float32)


def measure_rect(full_img: np.ndarray, ref_img: np.ndarray, init_rect: Sequence[int]) -> tuple[BoxPx, float]:
    """Fit table_view_rect: ECC (MOTION_AFFINE, grey) aligning ref_img (the zoom-160 table image) into
    full_img (the zoom-100 frame), starting from init_rect. Both are downscaled so the full frame is at most
    960 px wide; a coarse-to-fine pass (1/4, 1/2, then that scale) widens the capture range.
    Returns the rect (x0, y0, x0 + sx*ref_w, y0 + sy*ref_h, rounded) and the final ECC correlation.
    Raises RuntimeError when ECC does not converge."""
    fh, fw = full_img.shape[:2]
    rh, rw = ref_img.shape[:2]
    full_g, ref_g = _grey(full_img), _grey(ref_img)
    x0, y0, x1, y1 = (float(v) for v in init_rect)
    # M maps ref edge coords (pixel corners, 0..rw) to full-frame edge coords: the rect is M of the ref corners
    m = np.array([[(x1 - x0) / rw, 0.0, x0], [0.0, (y1 - y0) / rh, y0], [0.0, 0.0, 1.0]])
    s_fine = min(1.0, _ECC_MAX_W / fw)
    scales = [s for s in (s_fine / 4, s_fine / 2) if s * (x1 - x0) >= 160] + [s_fine]
    crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 200, 1e-6)
    cc = 0.0
    for s in scales:
        img = cv2.resize(full_g, (max(1, round(fw * s)), max(1, round(fh * s))), interpolation=cv2.INTER_AREA)
        tw, th = max(8, round(s * (x1 - x0))), max(8, round(s * (y1 - y0)))
        tmpl = cv2.resize(ref_g, (tw, th), interpolation=cv2.INTER_AREA)
        sx_img, sy_img = img.shape[1] / fw, img.shape[0] / fh       # exact per-axis after rounding
        rx, ry = tw / rw, th / rh
        # template index -> ref edge coords (R); full edge coords -> working image index (S)
        r = np.array([[1 / rx, 0, 0.5 / rx], [0, 1 / ry, 0.5 / ry], [0, 0, 1]])
        sm = np.array([[sx_img, 0, -0.5], [0, sy_img, -0.5], [0, 0, 1]])
        warp = (sm @ m @ r)[:2].astype(np.float32)
        try:
            cc, warp = cv2.findTransformECC(tmpl, img, warp, cv2.MOTION_AFFINE, crit, None, 5)
        except cv2.error as ex:
            raise RuntimeError(f"ECC did not converge at scale {s:.3f}: {ex}") from ex
        w3 = np.vstack([warp.astype(np.float64), [0, 0, 1]])
        m = np.linalg.inv(sm) @ w3 @ np.linalg.inv(r)
    x0f, y0f = m[0, 2], m[1, 2]
    rect = (int(round(x0f)), int(round(y0f)), int(round(x0f + m[0, 0] * rw)), int(round(y0f + m[1, 1] * rh)))
    return rect, float(cc)
