"""The band of table around each object, remembered in colour, for the world's unknown-cover rule
(core/world.py rule 4, unknown_cover: in config.yaml).

A cover the detectors have no class for (a blanket, a jacket, a napkin) is laid over objects by hand, so
the hand rule alone would call every object it touched picked up. What tells the two apart is the spot's
surroundings once the hand has moved off: a real pick-up leaves the band of table around the object's
spot as it was, a cover leaves it changed all round. The background model (relations.BackgroundModel)
cannot say this for objects that lay there since startup (it never saw the table under them), and in
grey a red blanket on brown wood differs too little from the table; a colour memory of the band itself
can.

Per object, the band ring_cm wide around its box (clipped to the frame) is stored downscaled, and only
after it looked the same for adopt_s while the object was detected with no hand box over the band: a
band caught while an arm or a cover is moving in is never remembered. A detection box that moves off
the remembered one (the object put down elsewhere) starts over. Compared later, each of the four sides
(left and right full height, top and bottom between them) counts as changed when side_changed of its
pixels differ by more than pixel_diff in some colour channel; pixels under a hand box are left out, and
a side with too few pixels left is not judged.
"""
from __future__ import annotations

from dataclasses import dataclass, fields

import cv2
import numpy as np

from core import geom
from core.types import BoxCm, BoxPx

MAX_DIM = 64            # the stored band is downscaled so its longer side is at most this many px
MIN_VISIBLE = 0.3       # a side with less than this share of its pixels clear of hand boxes is not judged
SAME_MAX = 0.1          # two views of the band with at most this share of pixels changed are the same
SAME_BOX_IOU = 0.7      # a detection box this much like the remembered one is the object in place


@dataclass
class UnknownCoverConfig:
    """The unknown_cover: section of config.yaml (every key optional; these are the defaults)."""
    enabled: bool = True
    ring_cm: float = 4.0        # width of the band of table around an object that is remembered
    pixel_diff: float = 30.0    # a pixel differs from memory by more than this in some channel (0-255)
    side_changed: float = 0.5   # a side of the band is changed when this share of its pixels differ
    sides_min: int = 3          # this many sides changed (none clear of change): something lies over the spot
    settle_s: float = 0.3       # ... at every look for this long: a cover at rest, not an arm passing
    adopt_s: float = 1.0        # the band is remembered once it looked the same this long
    wait_max_s: float = 2.0     # the longest an absence waits on an unclear or settling band

    @classmethod
    def from_config(cls, cfg) -> 'UnknownCoverConfig':
        raw = dict(getattr(cfg, 'unknown_cover', None) or {})
        return cls(**{k: v for k, v in raw.items() if k in {f.name for f in fields(cls)}})


@dataclass
class Band:
    """One view of the band around an object."""
    t: float
    outer: tuple[int, int, int, int]       # the grown box, px, clipped to the frame
    box_px: BoxPx                           # the object's box it was taken around
    box_cm: BoxCm
    crop: np.ndarray                        # BGR, downscaled; the object's own box is not used


class SurroundMemory:
    def __init__(self, cfg: UnknownCoverConfig):
        self.cfg = cfg
        self._mem: dict[str, Band] = {}
        self._pending: dict[str, Band] = {}

    # ----- remembering ---------------------------------------------------------------------------

    def memory(self, name: str) -> Band | None:
        return self._mem.get(name)

    def forget(self, name: str) -> None:
        self._mem.pop(name, None)
        self._pending.pop(name, None)

    def observe(self, name: str, img: np.ndarray, box_px: BoxPx, box_cm: BoxCm, t: float,
                hands_px: list[BoxPx]) -> None:
        """The object is detected at box_px now: remember the band around it once it has looked the same
        for adopt_s. Nothing is learned while a hand box touches the band."""
        outer = self._outer(img, box_px, box_cm)
        if outer is None or any(geom.intersection(outer, h) for h in hands_px):
            return
        p = self._pending.get(name)
        if p is not None and geom.iou(p.box_px, box_px) >= SAME_BOX_IOU:
            now = self._crop(img, p.outer)
            if now is not None and self._same(p, now):
                if t - p.t >= self.cfg.adopt_s:
                    self._mem[name] = Band(t, p.outer, p.box_px, p.box_cm, now)
                return
        crop = self._crop(img, outer)
        if crop is not None:
            self._pending[name] = Band(t, outer, tuple(box_px), tuple(box_cm), crop)

    # ----- looking -------------------------------------------------------------------------------

    def sides(self, name: str, img: np.ndarray, hands_px: list[BoxPx]) -> list[float | None] | None:
        """Share of changed pixels on each side of the remembered band (left, right, top, bottom); None
        for a side too hidden by hands, or for all of it without a memory."""
        band = self._mem.get(name)
        if band is None or img is None:
            return None
        cur = self._crop(img, band.outer)
        if cur is None or cur.shape != band.crop.shape:
            return None
        diff = np.abs(cur.astype(np.int16) - band.crop.astype(np.int16)).max(axis=2) > self.cfg.pixel_diff
        clear = self._clear_mask(band, hands_px, diff.shape)
        out: list[float | None] = []
        for sl in self._side_slices(band, diff.shape):
            if sl is None:
                out.append(None)
                continue
            vis = clear[sl]
            if vis.size == 0 or vis.mean() < MIN_VISIBLE:
                out.append(None)
                continue
            out.append(float(diff[sl][vis].mean()))
        return out

    def look(self, name: str, img: np.ndarray, hands_px: list[BoxPx]) -> bool | None:
        """True: changed on at least sides_min sides (something lies over the spot and around it).
        False: some side is clear of change (the table shows there). None: no memory, or every side that
        can be seen is changed but hands hide too many to tell."""
        s = self.sides(name, img, hands_px)
        if s is None:
            return None
        seen = [v for v in s if v is not None]
        changed = sum(v >= self.cfg.side_changed for v in seen)
        if changed >= self.cfg.sides_min:
            return True
        if changed < len(seen):
            return False
        return None

    def bare(self, name: str, img: np.ndarray, hands_px: list[BoxPx]) -> bool:
        """The band looks as remembered again: at least sides_min sides seen, at most one of them changed
        (a cover lifted off the spot)."""
        s = self.sides(name, img, hands_px)
        if s is None:
            return False
        seen = [v for v in s if v is not None]
        return len(seen) >= self.cfg.sides_min and sum(v >= self.cfg.side_changed for v in seen) <= 1

    # ----- geometry --------------------------------------------------------------------------------

    def _outer(self, img, box_px, box_cm):
        wcm, hcm = box_cm[2] - box_cm[0], box_cm[3] - box_cm[1]
        wpx, hpx = box_px[2] - box_px[0], box_px[3] - box_px[1]
        if wcm <= 0 or hcm <= 0 or wpx <= 0 or hpx <= 0:
            return None
        g = self.cfg.ring_cm * (wpx / wcm + hpx / hcm) / 2
        h, w = img.shape[:2]
        x1, y1 = max(0, int(box_px[0] - g)), max(0, int(box_px[1] - g))
        x2, y2 = min(w, int(round(box_px[2] + g))), min(h, int(round(box_px[3] + g)))
        return (x1, y1, x2, y2) if x2 - x1 >= 4 and y2 - y1 >= 4 else None

    def _scale(self, outer) -> float:
        return min(1.0, MAX_DIM / max(outer[2] - outer[0], outer[3] - outer[1]))

    def _crop(self, img, outer):
        x1, y1, x2, y2 = outer
        if img is None or img.ndim != 3 or y2 > img.shape[0] or x2 > img.shape[1]:
            return None
        crop = img[y1:y2, x1:x2]
        s = self._scale(outer)
        if s < 1.0:
            size = (max(1, int(round((x2 - x1) * s))), max(1, int(round((y2 - y1) * s))))
            crop = cv2.resize(crop, size, interpolation=cv2.INTER_AREA)
        return crop.copy()

    def _to_crop(self, band: Band, x: float, y: float, shape) -> tuple[int, int]:
        s = self._scale(band.outer)
        cx = int(round((x - band.outer[0]) * s))
        cy = int(round((y - band.outer[1]) * s))
        return min(max(cx, 0), shape[1]), min(max(cy, 0), shape[0])

    def _side_slices(self, band: Band, shape):
        bx1, by1 = self._to_crop(band, band.box_px[0], band.box_px[1], shape)
        bx2, by2 = self._to_crop(band, band.box_px[2], band.box_px[3], shape)
        h, w = shape
        out = [(slice(0, h), slice(0, bx1)), (slice(0, h), slice(bx2, w)),
               (slice(0, by1), slice(bx1, bx2)), (slice(by2, h), slice(bx1, bx2))]
        return [sl if sl[0].stop > sl[0].start and sl[1].stop > sl[1].start else None for sl in out]

    def _clear_mask(self, band: Band, hands_px, shape) -> np.ndarray:
        clear = np.ones(shape, bool)
        for hb in hands_px:
            x1, y1 = self._to_crop(band, hb[0], hb[1], shape)
            x2, y2 = self._to_crop(band, hb[2], hb[3], shape)
            clear[y1:y2, x1:x2] = False
        return clear

    def _same(self, a: Band, crop: np.ndarray) -> bool:
        if crop.shape != a.crop.shape:
            return False
        diff = np.abs(crop.astype(np.int16) - a.crop.astype(np.int16)).max(axis=2) > self.cfg.pixel_diff
        mask = np.ones(diff.shape, bool)
        bx1, by1 = self._to_crop(a, a.box_px[0], a.box_px[1], diff.shape)
        bx2, by2 = self._to_crop(a, a.box_px[2], a.box_px[3], diff.shape)
        mask[by1:by2, bx1:bx2] = False          # the object itself may change (a phone screen): the band may not
        return not mask.any() or diff[mask].mean() <= SAME_MAX
