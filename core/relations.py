"""Pure spatial-relation helpers used by world.py (spec 3.5): contacts, covers, container dwell,
parent chains, and the image-based background / appearance checks. No state about beliefs lives here;
world.py owns Entities. Everything is deterministic and unit-testable without hardware.

INTERFACE CONTRACT — signatures are fixed; implementations are filled in by the relations task.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from core import geom
from core.types import BoxCm, BoxPx, Detection, Entity, PointCm

_BG_MAX_DIM = 320      # BackgroundModel works on frames downscaled to at most this many px per side


def _gray(img: np.ndarray) -> np.ndarray:
    """uint8 grayscale view of a grayscale or BGR image."""
    if img.ndim == 3 and img.shape[2] == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if img.ndim == 3:
        img = img[:, :, 0]
    return img.astype(np.uint8, copy=False)


def _crop(img: np.ndarray, box_px: BoxPx) -> np.ndarray | None:
    """Copy of img inside box_px clipped to the image, or None if nothing is left."""
    h, w = img.shape[:2]
    x1, y1 = max(0, int(box_px[0])), max(0, int(box_px[1]))
    x2, y2 = min(w, int(box_px[2])), min(h, int(box_px[3]))
    if x2 <= x1 or y2 <= y1:
        return None
    return img[y1:y2, x1:x2].copy()


def touching_hands(obj_box: BoxCm, hands: list[Detection], min_overlap: float) -> list[str]:
    """Hand ids (Detection.cls, e.g. 'hand:3') whose box covers >= min_overlap of obj_box."""
    return [h.cls for h in hands if geom.overlap_frac(h.box_cm, obj_box) >= min_overlap]


@dataclass
class CoverState:
    box_cm: BoxCm
    moved_at: float | None      # last time the cover moved, or None if never seen moving


class CoverMotion:
    """Tracks each cover's box and when it last moved (centre shifted >= min_move_cm from its
    last stationary position)."""

    def __init__(self, min_move_cm: float):
        self.min_move_cm = min_move_cm
        self._anchor: dict[str, PointCm] = {}
        self._state: dict[str, CoverState] = {}

    def update(self, name: str, box: BoxCm, t: float) -> None:
        c = geom.center(box)
        st = self._state.get(name)
        if st is None:
            self._anchor[name] = c
            self._state[name] = CoverState(box_cm=box, moved_at=None)
            return
        st.box_cm = box
        if geom.dist(c, self._anchor[name]) >= self.min_move_cm:
            st.moved_at = t
            self._anchor[name] = c

    def state(self, name: str) -> CoverState | None:
        st = self._state.get(name)
        return None if st is None else CoverState(box_cm=st.box_cm, moved_at=st.moved_at)


def covering_candidates(last_box: BoxCm, covers: dict[str, CoverState], now: float,
                        min_overlap: float, moved_window_s: float) -> list[str]:
    """Covers whose current box overlaps >= min_overlap of last_box AND that moved within
    moved_window_s of now. Sorted by overlap, largest first."""
    scored = []
    for name, st in covers.items():
        if st.moved_at is None or now - st.moved_at > moved_window_s:
            continue
        ov = geom.overlap_frac(st.box_cm, last_box)
        if ov >= min_overlap:
            scored.append((ov, name))
    scored.sort(key=lambda x: -x[0])     # stable: ties keep dict order
    return [name for _, name in scored]


class DwellTracker:
    """Per hand track: remembers visits where the hand centre stayed inside a container box for
    >= min_dwell_s and then left."""

    HISTORY_S = 60.0    # completed visits older than this (relative to the latest update) are dropped

    def __init__(self, min_dwell_s: float):
        self.min_dwell_s = min_dwell_s
        self._entered: dict[tuple[str, str], float] = {}          # (hand, container) -> entry t
        self._visits: dict[str, list[tuple[str, float]]] = {}     # hand -> [(container, exit t)]

    def update(self, t: float, hands: list[Detection], containers: dict[str, BoxCm]) -> None:
        inside_now = set()
        for h in hands:
            c = geom.center(h.box_cm)
            for cname, cbox in containers.items():
                if geom.contains_point(cbox, c):
                    inside_now.add((h.cls, cname))
        for key in sorted(set(self._entered) - inside_now, key=lambda k: self._entered[k]):
            entry = self._entered.pop(key)
            if t - entry >= self.min_dwell_s:
                self._visits.setdefault(key[0], []).append((key[1], t))
        for key in inside_now:
            self._entered.setdefault(key, t)
        cutoff = t - self.HISTORY_S
        for hid in list(self._visits):
            kept = [v for v in self._visits[hid] if v[1] >= cutoff]
            if kept:
                self._visits[hid] = kept
            else:
                del self._visits[hid]

    def visits_since(self, hand_id: str, since: float) -> list[tuple[str, float]]:
        """(container, exit_time) for completed qualifying visits with exit_time >= since,
        oldest first. A visit still in progress is not returned."""
        return [v for v in self._visits.get(hand_id, []) if v[1] >= since]


def resolve_chain(name: str, entities: dict[str, Entity], max_depth: int) -> tuple[PointCm | None, list[str]]:
    """Follow parent links through entity names up to max_depth levels.
    Returns (position to point at, chain), e.g. keys INSIDE box UNDER notebook ->
    (notebook.pos_cm, ['keys', 'box', 'notebook']). Stops at a parent that is not an entity name
    ('hand:3', 'unknown', None); position is then the last entity's pos_cm (may be None).
    A name that is not an entity gives (None, []). Cycles stop at the first repeated name."""
    if name not in entities:
        return None, []
    chain = [name]
    cur = entities[name]
    for _ in range(max_depth):
        parent = cur.parent
        if parent is None or parent not in entities or parent in chain:
            break
        chain.append(parent)
        cur = entities[parent]
    return cur.pos_cm, chain


def descendants(name: str, entities: dict[str, Entity], max_depth: int) -> list[str]:
    """Entities whose parent chain reaches name within max_depth levels (children first)."""
    out: list[str] = []
    seen = {name}
    frontier = [name]
    for _ in range(max_depth):
        nxt = [n for n, e in entities.items() if e.parent in frontier and n not in seen]
        if not nxt:
            break
        seen.update(nxt)
        out.extend(nxt)
        frontier = nxt
    return out


class BackgroundModel:
    """Median of recent grayscale frames of the table, updated only outside excluded boxes
    (objects, hands) so stationary objects never become 'background'."""

    # Implementation notes:
    # - Frames are downscaled internally (INTER_AREA) so the longest side is <= _BG_MAX_DIM px; a
    #   full 1280x720 30-frame median costs ~0.4 s, the 320x180 one ~25 ms. Box arguments stay in
    #   full-resolution px and are rounded outward onto the small grid.
    # - Excluded pixels get the previous background value pushed instead of the frame's value.
    # - A pixel that has never been seen un-excluded (e.g. an object sitting there since the first
    #   frame) has no background yet. The first time it is seen, every stored frame is backfilled with
    #   that value; until then it is ignored by changed().
    # - ready = at least min(n_frames, 5) frames pushed: 5 frames give a usable median without
    #   waiting for the whole ring (30 frames) at startup.

    MIN_READY = 5

    def __init__(self, n_frames: int, threshold: float):
        self.n_frames = max(1, int(n_frames))
        self.threshold = threshold
        self._reset(None)

    def _reset(self, shape: tuple[int, int] | None) -> None:
        self._shape = shape                        # full-resolution (h, w)
        self._scale = 1
        self._stack: np.ndarray | None = None      # (n, h', w') uint8 ring of small frames
        self._count = 0
        self._idx = 0
        self._bg: np.ndarray | None = None         # (h', w') uint8 median
        self._known: np.ndarray | None = None      # (h', w') bool: pixel ever seen un-excluded
        if shape is not None:
            h, w = shape
            self._scale = max(1, -(-max(h, w) // _BG_MAX_DIM))
            sh, sw = -(-h // self._scale), -(-w // self._scale)
            self._stack = np.zeros((self.n_frames, sh, sw), np.uint8)
            self._bg = np.zeros((sh, sw), np.uint8)
            self._known = np.zeros((sh, sw), bool)

    @property
    def ready(self) -> bool:
        return self._count >= min(self.n_frames, self.MIN_READY)

    def _prep(self, img: np.ndarray) -> np.ndarray:
        g = _gray(img)
        if self._scale > 1:
            size = (self._stack.shape[2], self._stack.shape[1])     # cv2 wants (w, h)
            g = cv2.resize(g, size, interpolation=cv2.INTER_AREA)
        return g

    def _small_box(self, box_px: BoxPx) -> tuple[int, int, int, int] | None:
        s = self._scale
        _, sh, sw = self._stack.shape
        x1, y1 = max(0, int(box_px[0]) // s), max(0, int(box_px[1]) // s)
        x2, y2 = min(sw, -(-int(box_px[2]) // s)), min(sh, -(-int(box_px[3]) // s))
        if x2 <= x1 or y2 <= y1:
            return None
        return x1, y1, x2, y2

    def update(self, img: np.ndarray, exclude_px: list[BoxPx]) -> None:
        if self._shape != img.shape[:2]:
            self._reset(img.shape[:2])
        g = self._prep(img)
        excluded = np.zeros(g.shape, bool)
        for b in exclude_px:
            sb = self._small_box(b)
            if sb is not None:
                excluded[sb[1]:sb[3], sb[0]:sb[2]] = True
        seen = ~excluded
        first_seen = seen & ~self._known
        if self._count and first_seen.any():
            self._stack[:self._count, first_seen] = g[first_seen]
        self._known |= seen
        frame = g.copy()
        frame[excluded] = self._bg[excluded]
        self._stack[self._idx] = frame
        self._idx = (self._idx + 1) % self.n_frames
        self._count = min(self._count + 1, self.n_frames)
        med = np.median(self._stack[:self._count], axis=0)
        self._bg = np.rint(med).astype(np.uint8)

    KNOWN_MIN_FRAC = 0.5

    def known(self, box_px: BoxPx) -> bool:
        """True when the model has seen background (un-excluded) for >= half of the pixels in box_px.
        changed() is False both for 'unchanged' and 'never seen'; this tells the two apart, so an
        object sitting there since startup (only a jittery edge ever seen) is not read as covered."""
        if self._known is None:
            return False
        sb = self._small_box(box_px)
        if sb is None:
            return False
        x1, y1, x2, y2 = sb
        return float(self._known[y1:y2, x1:x2].mean()) >= self.KNOWN_MIN_FRAC

    def changed(self, img: np.ndarray, box_px: BoxPx) -> bool:
        """Mean absolute difference from background inside box_px exceeds threshold.
        False when there is no background yet for any pixel of the (clipped) box."""
        if self._stack is None or self._count == 0 or self._shape != img.shape[:2]:
            return False
        sb = self._small_box(box_px)
        if sb is None:
            return False
        x1, y1, x2, y2 = sb
        known = self._known[y1:y2, x1:x2]
        if not known.any():
            return False
        cur = self._prep(img)[y1:y2, x1:x2].astype(np.int16)
        diff = np.abs(cur - self._bg[y1:y2, x1:x2].astype(np.int16))
        return float(diff[known].mean()) > self.threshold


class AppearanceMemory:
    """Last-seen image patch per object. Guards against detector misses: if the patch at the
    object's last box still matches, the object is still there (review fix)."""

    # A patch whose grey-level std is below FLAT_STD is "flat": NCC is undefined there, so two flat
    # patches match when their means are within FLAT_TOL grey levels; flat vs textured never matches.
    FLAT_STD = 2.0
    FLAT_TOL = 15.0

    def __init__(self) -> None:
        self._patches: dict[str, np.ndarray] = {}

    def remember(self, name: str, img: np.ndarray, box_px: BoxPx) -> None:
        """Store the grayscale patch at box_px (clipped). An empty/off-image box is ignored."""
        patch = _crop(_gray(img), box_px)
        if patch is not None:
            self._patches[name] = patch.astype(np.float32)

    def still_there(self, name: str, img: np.ndarray, box_px: BoxPx, min_ncc: float) -> bool:
        stored = self._patches.get(name)
        if stored is None:
            return False
        cur = _crop(_gray(img), box_px)
        if cur is None:
            return False
        if cur.shape != stored.shape:
            cur = cv2.resize(cur, (stored.shape[1], stored.shape[0]), interpolation=cv2.INTER_AREA)
        return self._similarity(stored, cur.astype(np.float32)) >= min_ncc

    @classmethod
    def _similarity(cls, a: np.ndarray, b: np.ndarray) -> float:
        sa, sb = float(a.std()), float(b.std())
        if sa < cls.FLAT_STD or sb < cls.FLAT_STD:
            both_flat = sa < cls.FLAT_STD and sb < cls.FLAT_STD
            return 1.0 if both_flat and abs(float(a.mean()) - float(b.mean())) <= cls.FLAT_TOL else 0.0
        a0, b0 = a - a.mean(), b - b.mean()
        return float((a0 * b0).mean() / (sa * sb))
