"""Class-agnostic object proposals for the open world (Detection.cls == 'thing'). Owner: P.

The fixed-class detector only knows the configured objects; everything else on the table reaches the
world model through a proposer. Two proposers share one interface, so which one wins is a config
switch (proposals.kind):

  change  ChangeProposer: the table is fixed and the camera does not move, so anything that is not
          the empty table is a candidate object. Needs no model and knows no classes (~2 ms a frame).
  yoloe   YOLOEProposer: a prompt-free YOLOE checkpoint (.pt or .engine), its boxes used without
          their class names. A reduced-head export (*reduced*.engine / .onnx) skips ultralytics'
          slow postprocess: core/yoloe_fast.py.

Detector.detect() calls propose(img, known_px, hands_px), drops proposals that duplicate a known
object or a hand (dedupe(), thresholds in proposals.dedupe) and appends the rest to Detections.items
as cls 'thing'. The world (core/things.py) decides which entity each proposal is.

ChangeProposer, per frame, at a working resolution of work_px on the long side:
  1. difference from an empty-table reference in CIE LAB, after removing the frame's global
     lightness and colour cast (auto exposure, white balance);
  2. per-pixel threshold = k x (frame noise from a robust MAD, plus the pixel's own noise from the
     reference frames) + a floor, so grain never fires and the threshold follows the camera;
  3. shadows (darker by a bounded ratio, chroma unchanged) do not count;
  4. hand and known-object boxes are cut out BEFORE grouping, so an unknown object touching the
     keys or held next to a hand is its own region; open / close morphology; connected components;
  5. reject: tiny / huge regions, regions touching the table border next to a hand (arms), regions
     still moving (a hand or arm the detector missed: objects on a table do not move by themselves;
     motion = frame-to-frame change in the last still_frames frames), small
     red regions (the laser dot), ArUco markers (masked), and ghosts: a region whose pixels now look
     like the table around it but did not in the reference is a revealed table (something that was
     in the reference moved away): the reference is healed there and nothing is proposed.
The reference is the median of ref_frames frames at startup, reset() (RESET / recalibration) or a
saved image, taken after skipping warmup_frames frames (a webcam settles for a few seconds after the stream
opens), with hand and known-object boxes left out (those pixels are learned the first time they
are seen empty for adopt_frames frames). It is refreshed slowly where the frame matches it and there
is no known object, hand or proposal, so lighting drift is followed while a stationary new object is
never absorbed. If most of the table changes for rebuild_frames frames (camera bumped, lights
switched), the reference is recaptured.
"""
from __future__ import annotations

import dataclasses
import logging
import time
from dataclasses import dataclass, field, fields
from typing import Optional, Protocol, Sequence

import cv2
import numpy as np

from core import geom
from core.table_area import TableArea
from core.types import BoxPx
from core.yoloe_fast import is_reduced

log = logging.getLogger(__name__)

Polygon = Sequence[tuple[float, float]]
THING = 'thing'                 # Detection.cls of every proposal (core/things.py reads the same)


@dataclass
class Proposal:
    box_px: BoxPx
    conf: float                          # evidence strength 0-1, not a probability
    area_px: int = 0                     # changed pixels inside the box, full-resolution px
    mask: Optional[np.ndarray] = None    # bool, box-sized (YOLOE seg models only)
    occluded: bool = False               # mostly inside a person box: never a new thing (core/things.py)


class Proposer(Protocol):
    def propose(self, img: np.ndarray, known: list[BoxPx], hands: list[BoxPx]) -> list[Proposal]: ...

    def reset(self) -> None: ...

    def set_roi(self, polygon: Optional[Polygon]) -> None: ...


def _from_dict(cls, raw: Optional[dict]):
    known = {f.name for f in fields(cls)}
    return cls(**{k: v for k, v in (raw or {}).items() if k in known})


# ================================================================================ dedupe

@dataclass
class DedupeConfig:
    """proposals.dedupe. A proposal duplicates a known object if IoU >= known_iou, or it lies mostly
    inside the known box (known_inside of it covered), or it holds the known box (known_contain of the
    known box inside it) while being at most known_grow times its area: a loose box around the same
    object. A larger box that only partly overlaps (an unknown next to the keys) is kept. Hands are
    stricter (fingers and arms spill past the hand box). Among proposals, a box nested in a bigger one
    is a part of that object (a keycap of a laptop, one wire of a cable pile) whatever the scores, unless
    it is clearly its own object lying on it (under nest_small of its area, scoring nest_keep_conf or
    more). A part is kept flagged occluded: it may carry on a thing already tracked (keys in an open
    tub), never start one. A box narrower than min_side_px is a sliver, not an object."""
    known_iou: float = 0.5
    known_inside: float = 0.7
    known_contain: float = 0.8
    known_grow: float = 2.0
    hand_iou: float = 0.3
    hand_inside: float = 0.5
    self_iou: float = 0.5               # two proposals this similar: keep the more confident
    self_inside: float = 0.8            # a proposal nested this much in a bigger kept one: dropped
    min_side_px: int = 0                # full-resolution px; 0 = no minimum
    nest_small: float = 0.25            # a nested box under this share of its host's area ...
    nest_keep_conf: float = 0.7         # ... scoring this or more is its own object (on the rig's frames
                                        # laptop keys and cable parts scored under 0.7)

    @classmethod
    def from_dict(cls, raw: Optional[dict]) -> 'DedupeConfig':
        return _from_dict(cls, raw)


def dedupe(props: list[Proposal], known: list[BoxPx], hands: list[BoxPx], cfg: DedupeConfig) -> list[Proposal]:
    """Proposals that are not a known object, a hand, a sliver or another proposal, most confident
    first; parts of a bigger proposal come back flagged occluded. Nesting is decided biggest first, so of
    a chain of nested boxes the outermost is the object: by confidence, a keycap scoring above its laptop
    dropped the laptop and kept every key as an object (on the rig, dozens of things on one laptop)."""
    kept: list[Proposal] = []
    for p in sorted(props, key=lambda q: -q.conf):
        b = p.box_px
        if min(b[2] - b[0], b[3] - b[1]) < cfg.min_side_px:
            continue
        if any(geom.iou(b, k) >= cfg.known_iou or geom.overlap_frac(k, b) >= cfg.known_inside
               or (geom.overlap_frac(b, k) >= cfg.known_contain and geom.area(b) <= cfg.known_grow * geom.area(k))
               for k in known):
            continue
        if any(geom.iou(b, h) >= cfg.hand_iou or geom.overlap_frac(h, b) >= cfg.hand_inside for h in hands):
            continue
        if any(geom.iou(b, q.box_px) >= cfg.self_iou for q in kept):
            continue
        kept.append(p)
    objects: list[Proposal] = []
    parts: set[int] = set()
    for p in sorted(kept, key=lambda q: -geom.area(q.box_px)):
        hosts = [q for q in objects if geom.overlap_frac(q.box_px, p.box_px) >= cfg.self_inside]
        if hosts and not (geom.area(p.box_px) < cfg.nest_small * min(geom.area(q.box_px) for q in hosts)
                          and p.conf >= cfg.nest_keep_conf):
            parts.add(id(p))
        else:
            objects.append(p)
    return [dataclasses.replace(p, occluded=True) if id(p) in parts else p for p in kept]


# ================================================================================ change proposer

@dataclass
class ChangeConfig:
    """proposals.change. Pixel sizes are at the working resolution unless named _frac (of the frame)."""
    work_px: int = 640                  # long side of the working image (1280x720 -> 640x360)
    ref_frames: int = 10                # median of this many frames makes the reference
    warmup_frames: int = 45             # ... taken after skipping this many (a webcam's first frames after the
                                        # stream opens are dark and unstable; a reference from them made
                                        # every object on the table look changed: 71 flickering spots vs 8)
    adopt_frames: int = 5               # a never-seen pixel seen empty this many frames in a row is learned
    k_sigma: float = 4.0                # threshold = k_sigma x noise + floor
    noise_cap: float = 3.0              # frame noise is capped at this x the noise right after capture
    floor_l: float = 10.0               # 8-bit LAB units (L 0-255)
    floor_c: float = 7.0
    shadow_ratio: tuple[float, float] = (0.35, 0.95)   # current / reference lightness of a shadow
    shadow_chroma: float = 10.0         # ... whose chroma moved less than this
    open_px: int = 3
    close_px: int = 5
    min_area_frac: float = 0.0004       # of the frame: ~370 px at 1280x720, ~2.5 cm^2 at 12 px/cm
    max_area_frac: float = 0.15
    min_side_px: int = 3
    hand_gap_px: int = 6                # a region touching the table border this close to a hand: an arm
    still_frames: int = 2               # a pixel that changed frame to frame in the last this many frames ...
    moving_frac: float = 0.1            # ... is moving; a region with more than this share moving is no object
    motion_k: float = 4.0               # frame-to-frame change above motion_k x noise x sqrt(2) + floor
    laser_max_frac: float = 0.0015      # regions smaller than this ...
    laser_red_frac: float = 0.25        # ... with this share of red pixels are the laser dot
    marker_grow: float = 0.35           # ArUco boxes grown by this fraction before masking
    ghost_like: float = 0.6             # revealed table: >= this share now looks like the table around it
    ghost_ref_like: float = 0.35        # ... and <= this share did in the reference
    ghost_ring_px: int = 4
    refresh_alpha: float = 0.03         # per frame, where the frame matches the reference
    drift_alpha: float = 0.002          # per frame, changed pixels in no box (slow lighting shifts)
    protect_px: int = 4                 # boxes grown by this before protecting them from refresh
    rebuild_frac: float = 0.4           # this share of the table changed ...
    rebuild_frames: int = 45            # ... this many frames in a row: recapture the reference
    ignore_px: list = field(default_factory=list)      # extra [x1, y1, x2, y2] full-res boxes to mask

    @classmethod
    def from_dict(cls, raw: Optional[dict]) -> 'ChangeConfig':
        c = _from_dict(cls, raw)
        c.shadow_ratio = tuple(c.shadow_ratio)
        return c


def _odd_kernel(n: int) -> Optional[np.ndarray]:
    n = int(n)
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (n, n)) if n > 1 else None


class ChangeProposer:
    def __init__(self, cfg: Optional[dict | ChangeConfig] = None, roi: Optional[Polygon] = None):
        self.cfg = cfg if isinstance(cfg, ChangeConfig) else ChangeConfig.from_dict(cfg)
        self._roi_poly = None if roi is None else [tuple(map(float, p)) for p in roi]
        self._k_open, self._k_close = _odd_kernel(self.cfg.open_px), _odd_kernel(self.cfg.close_px)
        self.last_ms = 0.0
        self.debug: dict = {}           # last frame's masks and rejected regions, for visualisation
        self._shape: Optional[tuple[int, int]] = None
        self.reset()

    # -- reference lifecycle

    def reset(self) -> None:
        """Forget the reference; the next ref_frames frames capture a new one."""
        self._stack: list[np.ndarray] = []
        self._ref: Optional[np.ndarray] = None      # (h, w, 3) float32 LAB, lighting-compensated
        self._known: Optional[np.ndarray] = None    # (h, w) bool: pixel has a reference value
        self._sig2: tuple = (None, None)            # per-pixel reference noise^2 (L, chroma), or None
        self._run: Optional[np.ndarray] = None      # (h, w) uint8 empty-in-a-row counter for unknown px
        self._static: Optional[np.ndarray] = None   # (h, w) bool: markers and ignore_px, never proposed
        self._loaded: Optional[np.ndarray] = None   # load_reference() image, applied on the next frame
        self._changed_run = 0
        self._valid_mask, self._adopting = None, True
        self._noise_ref: Optional[tuple[float, float]] = None
        self._prev: Optional[np.ndarray] = None     # last frame's compensated LAB, for motion
        self._since: Optional[np.ndarray] = None    # (h, w) uint8 frames since the pixel last moved
        self._warm = max(0, int(self.cfg.warmup_frames))   # frames still to skip before collecting

    @property
    def ready(self) -> bool:
        return self._ref is not None or self._loaded is not None

    def set_roi(self, polygon: Optional[Polygon]) -> None:
        """The table outline in full-resolution px (grown by the caller's margin), or None for the
        whole frame. Changes outside it (people, the floor) are never proposals."""
        self._roi_poly = None if polygon is None else [tuple(map(float, p)) for p in polygon]
        if self._shape is not None:
            self._roi = self._roi_mask()
            self._valid_mask = None

    def load_reference(self, img: np.ndarray) -> None:
        """Use a saved empty-table image (any size, same camera pose) as the reference, from the next
        frame on (the frame fixes the working size). A later reset() captures a live one instead."""
        self.reset()
        self._loaded = img

    def _apply_loaded(self, frame: np.ndarray) -> None:
        img, self._loaded = self._loaded, None
        self._ref = self._lab(img)
        self._known = np.ones(self._small, bool)
        self._sig2 = (None, None)
        self._run = np.zeros(self._small, np.uint8)
        self._static = self._static_mask(frame)

    def save_reference(self, path: str) -> bool:
        """Write the reference (working resolution, BGR) for load_reference()."""
        if self._ref is None:
            return False
        lab = np.clip(self._ref, 0, 255).astype(np.uint8)
        return bool(cv2.imwrite(path, cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)))

    # -- geometry helpers

    def _setup(self, shape: tuple[int, int]) -> None:
        if self._shape == shape:
            return
        h, w = shape
        self._shape = shape
        self._scale = min(1.0, self.cfg.work_px / max(h, w))
        self._small = (max(1, round(h * self._scale)), max(1, round(w * self._scale)))
        self._stride = max(2, int(np.sqrt(self._small[0] * self._small[1] / 6000.0)))   # ~6k noise samples
        self._roi = self._roi_mask()
        loaded = self._loaded
        self.reset()
        self._loaded = loaded

    def _roi_mask(self) -> np.ndarray:
        sh, sw = self._small
        if self._roi_poly is None:
            m = np.ones((sh, sw), bool)
        else:
            m8 = np.zeros((sh, sw), np.uint8)
            pts = np.round(np.array(self._roi_poly, np.float64) * self._scale).astype(np.int32)
            cv2.fillPoly(m8, [pts], 1)
            m = m8.astype(bool)
        border = m & ~cv2.erode(m.astype(np.uint8), np.ones((3, 3), np.uint8), borderValue=0).astype(bool)
        self._border = border
        return m

    def _lab(self, img: np.ndarray) -> np.ndarray:
        """Working-size LAB float32. INTER_AREA is only fast at power-of-two factors (a 3x one costs
        ~4 ms at 720p), so: area by the largest power of two that stays above the working size (this
        averages the grain away), then linear to the exact size."""
        sh, sw = self._small
        small = img
        while small.shape[0] >= 2 * sh and small.shape[1] >= 2 * sw:
            small = cv2.resize(small, (small.shape[1] // 2, small.shape[0] // 2), interpolation=cv2.INTER_AREA)
        if small.shape[:2] != (sh, sw):
            small = cv2.resize(small, (sw, sh), interpolation=cv2.INTER_LINEAR)
        self._bgr = small
        return cv2.cvtColor(small, cv2.COLOR_BGR2LAB).astype(np.float32)

    def _small_box(self, b, grow: int = 0) -> tuple[int, int, int, int]:
        s = self._scale
        sh, sw = self._small
        return (max(0, int(b[0] * s) - grow), max(0, int(b[1] * s) - grow),
                min(sw, int(np.ceil(b[2] * s)) + grow), min(sh, int(np.ceil(b[3] * s)) + grow))

    def _box_mask(self, boxes, grow: int = 0) -> np.ndarray:
        m = np.zeros(self._small, bool)
        for b in boxes:
            x1, y1, x2, y2 = self._small_box(b, grow)
            m[y1:y2, x1:x2] = True
        return m

    def _static_mask(self, img: np.ndarray) -> np.ndarray:
        """ArUco markers (found in this full-resolution frame) and config ignore_px boxes."""
        boxes = [tuple(b) for b in self.cfg.ignore_px]
        try:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
            det = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50),
                                          cv2.aruco.DetectorParameters())
            corners, ids, _ = det.detectMarkers(gray)
            for c in corners if ids is not None else []:
                c = c.reshape(4, 2)
                (x1, y1), (x2, y2) = c.min(0), c.max(0)
                g = self.cfg.marker_grow * max(x2 - x1, y2 - y1)
                boxes.append((x1 - g, y1 - g, x2 + g, y2 + g))
        except Exception:                           # no aruco module: markers are just not masked
            log.debug("aruco marker masking unavailable", exc_info=True)
        self.markers_px = boxes
        return self._box_mask(boxes)

    # -- per frame

    def propose(self, img: Optional[np.ndarray], known: list[BoxPx], hands: list[BoxPx]) -> list[Proposal]:
        if img is None:
            return []
        t0 = time.perf_counter()
        self._setup(img.shape[:2])
        if self._loaded is not None:
            self._apply_loaded(img)
        if self._ref is None and self._warm > 0:
            self._warm -= 1
            self.last_ms = 1000 * (time.perf_counter() - t0)
            return []
        lab = self._lab(img)
        if self._ref is None:
            self._collect(img, lab, known, hands)
            self.last_ms = 1000 * (time.perf_counter() - t0)
            return []
        out = self._propose(lab, known, hands)
        self.last_ms = 1000 * (time.perf_counter() - t0)
        return out

    def _collect(self, img, lab, known, hands) -> None:
        frame = lab.copy()
        frame[self._box_mask(list(known) + list(hands), self.cfg.protect_px)] = np.nan
        self._stack.append(frame)
        if len(self._stack) < self.cfg.ref_frames:
            return
        st = np.stack(self._stack)
        self._stack = []
        with np.errstate(all='ignore'), _quiet_nan():
            ref = np.nanmedian(st, axis=0)
            sd = np.nanstd(st, axis=0)
        known_px = ~np.isnan(ref[..., 0])
        self._ref = np.nan_to_num(ref).astype(np.float32)
        sd = np.nan_to_num(sd).astype(np.float32)
        self._sig2 = (np.ascontiguousarray(sd[..., 0] ** 2), np.ascontiguousarray(sd[..., 1] ** 2 + sd[..., 2] ** 2))
        # The capture's own frame noise, in the units _propose measures (Gaussian sigma for L, median
        # chroma distance for a/b), so the noise cap holds from the first frame on.
        seen = known_px & self._roi
        if seen.sum() >= 50:
            self._noise_ref = (max(0.5, float(np.median(sd[..., 0][seen]))),
                               max(0.5, 1.177 * float(np.median(np.sqrt((self._sig2[1][seen]) / 2)))))
        self._known = known_px
        self._run = np.zeros(self._small, np.uint8)
        self._static = self._static_mask(img)
        self._changed_run = 0

    def _propose(self, lab, known, hands) -> list[Proposal]:
        c = self.cfg
        excl_hands = self._box_mask(hands, c.protect_px)
        excl_known = self._box_mask(known, c.protect_px)
        self._adopt(lab, excl_hands | excl_known)

        valid = self._valid()
        d = cv2.subtract(lab, self._ref)
        k = self._stride
        samp = valid[::k, ::k]
        ds = d[::k, ::k][samp]
        if len(ds) < 50:
            return []
        off = np.median(ds, axis=0)                          # global lightness / colour cast
        # Noise from the 30th percentile, not the median: it stays right with up to 70% of the table
        # changed (a MAD inflates near 50% and raises every threshold with it). Scaled to a Gaussian
        # sigma for L and to the median of the chroma distance (Rayleigh) for a/b.
        s_l = max(0.5, float(np.percentile(np.abs(ds[:, 0] - off[0]), 30)) / 0.385)
        s_c = max(0.5, 1.39 * float(np.percentile(np.hypot(ds[:, 1] - off[1], ds[:, 2] - off[2]), 30)))
        # Capped at noise_cap x the capture's noise (a loaded image: the first frame's): an uneven light
        # change makes everything look 'noisy'; capped, it reads as changed and triggers the rebuild.
        if self._noise_ref is None:
            self._noise_ref = (s_l, s_c)
        s_l = min(s_l, c.noise_cap * self._noise_ref[0])
        s_c = min(s_c, c.noise_cap * self._noise_ref[1])
        dl, da, db = cv2.split(d)
        dl -= off[0]
        da -= off[1]
        db -= off[2]
        dc = cv2.magnitude(da, db)
        e_l = np.abs(dl)
        e_l /= self._thr(0, s_l, c.floor_l)
        e_c = dc / self._thr(1, s_c, c.floor_c)
        ref_l = self._ref[..., 0] + 4.0
        cur_l = dl + ref_l                                   # compensated current lightness + 4
        shadow = ((dl < 0) & (cur_l >= c.shadow_ratio[0] * ref_l) & (cur_l <= c.shadow_ratio[1] * ref_l)
                  & (dc < c.shadow_chroma))
        changed = (cv2.max(e_l, e_c) > 1.0) & valid          # anything, shadows included
        np.copyto(e_l, 0.0, where=shadow)
        ev = cv2.max(e_l, e_c)
        fg = (ev > 1.0) & valid

        frac = float(changed[::k, ::k][samp].mean())
        self._changed_run = self._changed_run + 1 if frac > c.rebuild_frac else 0
        if self._changed_run >= c.rebuild_frames:
            log.info("proposals: %.0f%% of the table changed for %d frames; recapturing the reference",
                     100 * frac, self._changed_run)
            self.reset()
            return []

        m = fg.astype(np.uint8)
        if self._k_open is not None:
            m = cv2.morphologyEx(m, cv2.MORPH_OPEN, self._k_open)
        m[excl_hands | excl_known] = 0
        if self._k_close is not None:
            m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, self._k_close)
        m &= valid.astype(np.uint8)
        n, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)

        cur = cv2.subtract(lab, (float(off[0]), float(off[1]), float(off[2]), 0.0))   # comparable with the reference
        moving = self._motion(cur, s_l, s_c)
        props, rejected, heal = self._components(n, labels, stats, ev, fg, valid, cur, hands, moving)
        for comp, (x, y) in heal:
            h, w = comp.shape
            self._ref[y:y + h, x:x + w][comp] = cur[y:y + h, x:x + w][comp]
        self._refresh(cur, changed, known, hands, [p.box_px for p in props])
        self.debug = {'fg': fg, 'mask': m, 'shadow': shadow & valid, 'rejected': rejected, 'moving': moving,
                      'noise': (round(s_l, 2), round(s_c, 2)), 'offset': tuple(np.round(off, 1)),
                      'changed_frac': round(frac, 3)}
        return props

    def _motion(self, cur, s_l: float, s_c: float) -> np.ndarray:
        """Pixels that changed from one frame to the next within the last still_frames frames. Both
        frames are lighting-compensated, so auto exposure is not motion; grain is opened away."""
        c = self.cfg
        if self._prev is None or self._prev.shape != cur.shape:
            self._prev = cur.copy()
            self._since = np.full(cur.shape[:2], 255, np.uint8)
            return np.zeros(cur.shape[:2], bool)
        dl, da, db = cv2.split(cv2.subtract(cur, self._prev))
        np.copyto(self._prev, cur)
        k2 = c.motion_k * np.sqrt(2.0)
        m = ((np.abs(dl) > k2 * s_l + c.floor_l) | (cv2.magnitude(da, db) > k2 * s_c + c.floor_c)).astype(np.uint8)
        if self._k_open is not None:
            m = cv2.morphologyEx(m, cv2.MORPH_OPEN, self._k_open)
        np.copyto(self._since, np.minimum(self._since, 254) + 1)
        self._since[m.astype(bool)] = 0
        return self._since < c.still_frames

    def _valid(self) -> np.ndarray:
        """Pixels that have a reference, lie on the table and are not a marker; cached until one of
        those changes (adopt, set_roi, a new reference)."""
        if self._valid_mask is None:
            self._valid_mask = self._known & self._roi & ~self._static
        return self._valid_mask

    def _thr(self, ch: int, noise: float, floor: float):
        """k x (frame noise combined with the pixel's own reference noise) + floor, per pixel; a
        scalar when the reference has no per-pixel noise (a loaded image)."""
        sig2 = self._sig2[ch]
        if sig2 is None:
            return self.cfg.k_sigma * noise + floor
        return cv2.sqrt(sig2 + noise * noise) * self.cfg.k_sigma + floor

    def _adopt(self, lab, excluded) -> None:
        """Pixels with no reference yet (hidden by an object or hand at capture) are learned once they
        have been seen clear of every known box and hand for adopt_frames frames in a row."""
        if not self._adopting:
            return
        unknown = ~self._known
        if not unknown.any():
            self._adopting = False
            return
        clear = unknown & ~excluded & self._roi
        self._run[clear] = np.minimum(self._run[clear].astype(np.int16) + 1, 255).astype(np.uint8)
        self._run[unknown & ~clear] = 0
        adopt = clear & (self._run >= self.cfg.adopt_frames)
        if adopt.any():
            self._ref[adopt] = lab[adopt]
            self._known |= adopt
            self._valid_mask = None

    def _components(self, n, labels, stats, ev, fg, valid, cur, hands, moving=None):
        c = self.cfg
        s = self._scale
        fh, fw = self._shape
        frame_area = fh * fw
        min_a = c.min_area_frac * frame_area * s * s
        max_a = c.max_area_frac * frame_area * s * s
        laser_a = c.laser_max_frac * frame_area * s * s
        hand_boxes = [self._small_box(h) for h in hands]
        border_labels = set(np.unique(labels[self._border & (labels > 0)]).tolist()) if n > 1 else set()
        props, rejected, heal = [], [], []
        for i in range(1, n):
            x, y, w, h, area = (int(v) for v in stats[i])
            box = (int(x / s), int(y / s), int(np.ceil((x + w) / s)), int(np.ceil((y + h) / s)))
            reason = None
            if area < min_a or w < c.min_side_px or h < c.min_side_px:
                reason = 'small'
            elif area > max_a:
                reason = 'large'
            elif i in border_labels and any(
                    geom.intersection((x - c.hand_gap_px, y - c.hand_gap_px, x + w + c.hand_gap_px,
                                       y + h + c.hand_gap_px), hb) for hb in hand_boxes):
                reason = 'arm'
            comp = labels[y:y + h, x:x + w] == i
            if (reason is None and moving is not None
                    and float(moving[y:y + h, x:x + w][comp].mean()) > c.moving_frac):
                reason = 'moving'
            if reason is None and area < laser_a and self._is_laser(x, y, comp):
                reason = 'laser'
            if reason is None and self._is_ghost(x, y, w, h, comp, fg, valid, cur):
                reason = 'ghost'
                if not any(geom.intersection((x, y, x + w, y + h), hb) for hb in hand_boxes):
                    heal.append((comp, (x, y)))
            if reason is not None:
                rejected.append((box, reason))
                continue
            strength = float(np.minimum(ev[y:y + h, x:x + w][comp], 5.0).mean())
            fill = area / float(w * h)
            conf = (1.0 - np.exp(-(strength - 1.0) * 1.5)) * (0.6 + 0.4 * fill) * min(1.0, np.sqrt(area / (3 * min_a)))
            props.append(Proposal(box_px=box, conf=round(float(np.clip(conf, 0.05, 0.99)), 3),
                                  area_px=int(area / (s * s))))
        return props, rejected, heal

    def _is_laser(self, x, y, comp) -> bool:
        """A small region where a good share of pixels is saturated red (or blown out to white)."""
        px = self._bgr[y:y + comp.shape[0], x:x + comp.shape[1]][comp].astype(np.int16)
        b, g, r = px[:, 0], px[:, 1], px[:, 2]
        red = ((r > 150) & (r > g + 60) & (r > b + 60)) | ((r > 235) & (g > 200) & (b > 200) & (r >= g))
        return float(red.mean()) >= self.cfg.laser_red_frac

    def _is_ghost(self, x, y, w, h, comp, fg, valid, cur) -> bool:
        """The region now looks like the table around it but did not in the reference: an object that
        was captured in the reference has moved away."""
        g = self.cfg.ghost_ring_px
        sh, sw = self._small
        x1, y1, x2, y2 = max(0, x - g), max(0, y - g), min(sw, x + w + g), min(sh, y + h + g)
        inner = np.zeros((y2 - y1, x2 - x1), bool)
        inner[y - y1:y - y1 + h, x - x1:x - x1 + w] = comp
        ring = cv2.dilate(inner.astype(np.uint8), np.ones((2 * g + 1, 2 * g + 1), np.uint8)).astype(bool)
        ring &= ~inner & ~fg[y1:y2, x1:x2] & valid[y1:y2, x1:x2]
        if ring.sum() < 20:
            return False
        now = cur[y1:y2, x1:x2]
        ref = self._ref[y1:y2, x1:x2]
        return (self._like(now, ring, inner) >= self.cfg.ghost_like
                and self._like(ref, ring, inner) <= self.cfg.ghost_ref_like)

    def _like(self, lab, ring, inner) -> float:
        """Share of inner pixels within the colour spread of the ring pixels (the table nearby)."""
        rp = lab[ring]
        med = np.median(rp, axis=0)
        spread = float(np.median(np.linalg.norm(rp - med, axis=1)))
        tol = max(self.cfg.floor_l, 3.0 * spread)
        return float((np.linalg.norm(lab[inner] - med, axis=1) < tol).mean())

    def _refresh(self, cur, changed, known, hands, props) -> None:
        c = self.cfg
        boxes = self._box_mask(list(known) + list(hands) + list(props), c.protect_px)
        base = self._known & self._roi
        grown = cv2.dilate(changed.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
        fast = base & ~boxes & ~grown
        slow = base & ~boxes & grown
        cv2.accumulateWeighted(cur, self._ref, c.refresh_alpha, mask=fast.astype(np.uint8))
        cv2.accumulateWeighted(cur, self._ref, c.drift_alpha, mask=slow.astype(np.uint8))


class _quiet_nan:
    """np.nanmedian warns on all-NaN pixels (hidden by a hand in every reference frame): expected."""

    def __enter__(self):
        import warnings
        self._w = warnings.catch_warnings()
        self._w.__enter__()
        warnings.simplefilter('ignore', RuntimeWarning)

    def __exit__(self, *a):
        return self._w.__exit__(*a)


# ================================================================================ YOLOE adapter

# Prompt-free vocabularies name the table, people and hands too; none of them is a thing on the table.
PEOPLE = ['person', 'man', 'woman', 'child', 'boy', 'girl', 'patient', 'head', 'hair', 'hand', 'arm', 'finger',
          'foot', 'toe', 'leg', 'knee']   # a box mostly inside one: occluded
# What people wear is a person while worn: on the rig, feet up at the coffee table came as 'shoe', 'sock',
# 'jeans'. A worn-item box touching a person box (grown by YOLOEConfig.worn_near_px) is treated as one; a
# shoe alone on the table, away from people, is an object.
WORN = ['glove', 'shoe', 'footwear', 'leather shoe', 'running shoe', 'sneaker', 'boot', 'cowboy boot', 'sandal',
        'slipper', 'sock', 'air sock', 'jeans', 'pants', 'pant', 'sweat pant', 'shirt', 'polo shirt']
PERSON_INSIDE = 0.6                     # share of a box inside a person box that flags it occluded
DEFAULT_IGNORE = PEOPLE + ['table', 'dining table', 'desk', 'office desk', 'coffee table', 'tabletop', 'countertop',
                           'floor', 'wall', 'wood', 'wood floor', 'hardwood', 'plywood']


@dataclass
class YOLOEConfig:
    """proposals.yoloe."""
    model: str = 'models/yoloe-26s-seg-pf.pt'   # prompt-free .pt, or a .engine exported from it, or a
                                                # *reduced*.engine / .onnx (core/yoloe_fast.py fast path)
    conf: float = 0.15
    imgsz: int = 640
    half: bool = False                  # .pt on a GPU only; an .engine has its precision baked in
    iou: float = 0.5                    # class-agnostic NMS inside the model call
    max_det: int = 100
    ignore_px: list = field(default_factory=list)      # [x1, y1, x2, y2] full-res boxes never proposed in
    min_area_frac: float = 0.0004
    max_area_frac: float = 0.15
    masks: bool = False                 # keep box-sized masks (seg checkpoints)
    ignore_classes: list = field(default_factory=lambda: list(DEFAULT_IGNORE))
    worn_near_px: int = 40              # a worn item (WORN) this close to a person box is part of them

    @classmethod
    def from_dict(cls, raw: Optional[dict]) -> 'YOLOEConfig':
        return _from_dict(cls, raw)


def _np(x) -> np.ndarray:
    return x.cpu().numpy() if hasattr(x, 'cpu') else np.asarray(x)


class YOLOEProposer:
    """Prompt-free YOLOE, class-agnostic: every box above conf that is plausible in size, inside the
    table and not an ignored class (table, person, hand) is a proposal. Classes are only used to drop
    those and to flag a box lying mostly inside a person box as occluded (a finger or bracelet boxed on
    its own, or an object being carried): kept, so the world can follow an existing thing through it,
    but never the start of a new thing. The world never sees the class names."""

    def __init__(self, cfg: Optional[dict | YOLOEConfig] = None, model=None):
        self.cfg = cfg if isinstance(cfg, YOLOEConfig) else YOLOEConfig.from_dict(cfg)
        if model is None and is_reduced(self.cfg.model):     # reduced-head export: core/yoloe_fast.py
            from core import yoloe_fast
            model = yoloe_fast.ReducedYOLOE(self.cfg.model, imgsz=self.cfg.imgsz)
        if model is None:
            from ultralytics import YOLO
            model = YOLO(self.cfg.model)         # ultralytics picks the YOLOE class from the file name
        self.model = model
        self.ignore = {s.lower() for s in self.cfg.ignore_classes}
        self._roi: Optional[np.ndarray] = None
        self.last_ms = 0.0

    def set_roi(self, polygon: Optional[Polygon]) -> None:
        self._roi = None if polygon is None else np.array(polygon, np.float32).reshape(-1, 1, 2)

    def reset(self) -> None:
        pass

    def propose(self, img: Optional[np.ndarray], known: list[BoxPx], hands: list[BoxPx]) -> list[Proposal]:
        if img is None:
            return []
        c = self.cfg
        t0 = time.perf_counter()
        kw = dict(imgsz=c.imgsz, conf=c.conf, iou=c.iou, agnostic_nms=True, max_det=c.max_det,
                  retina_masks=c.masks, verbose=False)
        if c.half:                               # newer ultralytics warns on every call that passes it
            kw['half'] = True
        r = self.model.predict(img, **kw)[0]
        names = getattr(r, 'names', None) or getattr(self.model, 'names', {}) or {}
        xyxy, conf, cls = _np(r.boxes.xyxy), _np(r.boxes.conf), _np(r.boxes.cls)
        masks = _np(r.masks.data) > 0.5 if (c.masks and getattr(r, 'masks', None) is not None) else None
        fh, fw = img.shape[:2]
        lo, hi = c.min_area_frac * fh * fw, c.max_area_frac * fh * fw
        label = [str(names.get(int(k), '')).lower() for k in cls]
        people = [tuple(float(v) for v in b) for b, s, n in zip(xyxy, conf, label) if s >= c.conf and n in PEOPLE]
        g = c.worn_near_px
        worn = [tuple(float(v) for v in b) for b, s, n in zip(xyxy, conf, label) if s >= c.conf and n in WORN
                and any(geom.intersection((b[0] - g, b[1] - g, b[2] + g, b[3] + g), p) for p in people)]
        people += worn
        cands = []
        for j, (b, s, k) in enumerate(zip(xyxy, conf, cls)):
            if s < c.conf or label[j] in self.ignore or tuple(float(v) for v in b) in worn:
                continue
            occluded = any(geom.overlap_frac(p, tuple(float(v) for v in b)) >= PERSON_INSIDE for p in people)
            box = tuple(int(round(v)) for v in b)
            if not lo <= geom.area(box) <= hi:
                continue
            if self._roi is not None and cv2.pointPolygonTest(self._roi, geom.center(box), False) < 0 \
                    and cv2.pointPolygonTest(self._roi, ((box[0] + box[2]) / 2, box[3]), False) < 0:
                continue                         # neither its centre nor where it stands (a tall thing
                                                 # at the far edge) is on the table
            cx, cy = geom.center(box)
            if any(x1 <= cx <= x2 and y1 <= cy <= y2 for x1, y1, x2, y2 in c.ignore_px):
                continue
            m = masks[j][box[1]:box[3], box[0]:box[2]].copy() if masks is not None else None
            cands.append(Proposal(box_px=box, conf=round(float(s), 3), mask=m, occluded=occluded))
        out = dedupe(cands, [], [], DedupeConfig())     # agnostic NMS once more, across the classes
        self.last_ms = 1000 * (time.perf_counter() - t0)
        return out


# ================================================================================ construction

def make_proposer(cfg: dict) -> Optional[Proposer]:
    """The proposer the config asks for (proposals.enabled / proposals.kind), or None."""
    p = cfg.get('proposals') or {}
    if not p.get('enabled', False):
        return None
    kind = p.get('kind', 'change')
    if kind == 'change':
        prop = ChangeProposer(p.get('change'))
        ref = (p.get('change') or {}).get('reference')
        if ref:
            img = cv2.imread(ref)
            if img is None:
                log.warning("proposals: reference image %s not found; capturing one at startup", ref)
            else:
                prop.load_reference(img)
        return prop
    if kind == 'yoloe':
        return YOLOEProposer(p.get('yoloe'))
    raise ValueError(f"proposals.kind must be 'change' or 'yoloe', not {kind!r}")


def table_roi(table, cfg: dict) -> Optional[list[tuple[float, float]]]:
    """The proposers' ROI in frame px: the operator's tabletop outline (table_area.polygon_cm, see
    core/table_area.py) when one is set, else the calibrated area grown by proposals.roi_margin_cm; None
    without a calibrated table that can map cm to px (then the whole frame is used)."""
    if table is None or not getattr(table, 'ok', False) or not hasattr(table, 'cm_to_px'):
        return None
    area = TableArea.from_dict(cfg.get('table_area'))
    w, h = (cfg.get('table') or {}).get('size_cm', (90, 60))
    m = float((cfg.get('proposals') or {}).get('roi_margin_cm', 2.0))
    try:
        pts = table.cm_to_px(area.polygon_cm if area.defined else
                             [[-m, -m], [w + m, -m], [w + m, h + m], [-m, h + m]])
    except Exception:
        log.warning("proposals: table outline unavailable; using the whole frame", exc_info=True)
        return None
    return [tuple(float(v) for v in p) for p in np.asarray(pts).reshape(-1, 2)]
