"""Crops of what the detector saw, for open-world identity and naming. Owner: P.

Naming an unknown thing ("what is that?"), matching it by appearance, or showing it on the dashboard
needs a picture of it, and the frame an answer is asked about is rarely a good one (a hand is on it,
it is moving). The Detector feeds every frame's objects and proposals here; per object this keeps
  best    the cleanest view so far: sharp, confident, no hand on it, not overlapped by anything else
  recent  the latest view (at most every recent_every_s)
Configured objects are keyed by name. Proposals have no identity in perception, so they are tracked by
box position (a proposal continues the track whose last box it overlaps). Each run of views of one
object is a series with a stable id (sid); a track starts a new series when its spot was empty for
renew_after_s or the object's colour changes at once (swap_de: another object may sit there now).

Ownership. Position is not identity: a crop is returned for a world thing only when that thing was
seen as the crop's series. The world keeps the Detection's own box_cm object as an entity's last box
(core/world.py _debounce), so `entity.box_cm is track.box_cm` says the world's latest observation of
the entity IS the series' latest view: bind(entity) then makes the entity the series' owner up to that
view's time (owner_t). Views after owner_t are unconfirmed, and a series another thing claims is
closed at its last confirmed view; the claimant starts a series of its own. voice/visual.py binds every
thing after each world.update; for_entity(entity) binds lazily too. When nothing confirms ownership,
for_entity returns None: no close-up rather than a possibly wrong one. Every crop keeps its sid and
capture time t (the Frame.t it was cut from). for_box() is a position lookup with no identity.
Memory is bounded: at most max_tracks live tracks and max_tracks closed series, each of two crops of at
most size_px on the long side (~12 MB at the defaults).

The Detector registers its store as the process's active() store, so the world and the answer code
can reach it without new wiring through main.py. event_snapshot() fetches the JPEG that
core/events.py saved for an object's latest event.
"""
from __future__ import annotations

import itertools
import os
import threading
from collections import deque
from dataclasses import dataclass, replace
from typing import Optional

import cv2
import numpy as np

from core import geom
from core.types import BoxPx, Detection, Entity

THING = 'thing'
SHARP_REF = 60.0          # Laplacian variance at which a crop counts as half sharp (64 px grey crop)


@dataclass
class Crop:
    img: np.ndarray               # BGR, long side <= size_px; never modified after creation
    box_px: BoxPx
    box_cm: tuple
    t: float                      # Frame.t of the view it was cut from
    score: float
    sid: int = 0                  # the series it belongs to


@dataclass
class CropTrack:
    """One series of views of one object (a live track's current series, or a closed one)."""
    key: str                      # the object name, or 'p<N>' for a proposal track
    cls: str
    box_px: BoxPx
    box_cm: tuple                 # the latest view's box: the Detection's own object (see bind)
    last_t: float
    best: Optional[Crop] = None
    recent: Optional[Crop] = None
    sid: int = 0                  # stable id of this series; never reused
    owner: Optional[str] = None   # the world thing this series was confirmed as (things only)
    owner_t: Optional[float] = None   # time of the latest view confirmed as owner
    confirmed: tuple = (None, None)   # (best, recent) as they were at owner_t
    look: Optional[np.ndarray] = None  # running mean Lab colour of hand-free views (the swap check)


def _clip(img: np.ndarray, box, margin: float = 0.0) -> Optional[tuple[int, int, int, int]]:
    h, w = img.shape[:2]
    x1, y1, x2, y2 = (float(v) for v in box)
    mx, my = margin * (x2 - x1), margin * (y2 - y1)
    x1, y1 = max(0, int(x1 - mx)), max(0, int(y1 - my))
    x2, y2 = min(w, int(np.ceil(x2 + mx))), min(h, int(np.ceil(y2 + my)))
    return (x1, y1, x2, y2) if x2 - x1 >= 4 and y2 - y1 >= 4 else None


def _look(img: np.ndarray, box: BoxPx) -> Optional[np.ndarray]:
    """Mean Lab colour of the box (8 x 8 px), or None for a box too small to tell."""
    c = _clip(img, box)
    if c is None or img.ndim != 3:
        return None
    x1, y1, x2, y2 = c
    small = cv2.resize(img[y1:y2, x1:x2], (8, 8), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(small, cv2.COLOR_BGR2LAB).reshape(-1, 3).mean(axis=0)


def _look_dist(a: np.ndarray, b: np.ndarray) -> float:
    """Colour distance in 8-bit Lab with lightness at half weight (shadows and exposure move L most)."""
    d = np.asarray(a, float) - np.asarray(b, float)
    return float(np.hypot(0.5 * d[0], np.hypot(d[1], d[2])))


def clean_score(img: np.ndarray, box: BoxPx, conf: float, others: list, hands: list) -> float:
    """0-1: how good a view of the object in box this frame is. Product of detector confidence,
    sharpness (Laplacian variance of a 64 px grey crop), isolation (share of the box not covered by
    another box) and a strong penalty for any hand overlap; boxes cut by the frame edge count less."""
    c = _clip(img, box)
    if c is None:
        return 0.0
    x1, y1, x2, y2 = c
    g = cv2.cvtColor(img[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img[y1:y2, x1:x2]
    s = 64.0 / max(g.shape)
    g = cv2.resize(g, (max(4, round(g.shape[1] * s)), max(4, round(g.shape[0] * s))), interpolation=cv2.INTER_AREA)
    lap = float(cv2.Laplacian(g, cv2.CV_32F).var())
    sharp = lap / (lap + SHARP_REF)
    iso = 1.0 - max([geom.overlap_frac(o, box) for o in others] + [0.0])
    hand = max([geom.overlap_frac(h, box) for h in hands] + [0.0])
    hand_f = 1.0 if hand <= 0.0 else 0.25 * (1.0 - hand)
    h, w = img.shape[:2]
    edge = 0.6 if box[0] <= 0 or box[1] <= 0 or box[2] >= w or box[3] >= h else 1.0
    return float(conf) * sharp * iso * hand_f * edge


class CropStore:
    def __init__(self, size_px: int = 128, max_tracks: int = 48, margin: float = 0.15,
                 recent_every_s: float = 0.5, match_iou: float = 0.3, match_px: float = 30.0,
                 renew_after_s: float = 1.0, swap_de: float = 30.0):
        self.size_px, self.max_tracks, self.margin = int(size_px), int(max_tracks), float(margin)
        self.renew_after_s, self.swap_de = float(renew_after_s), float(swap_de)
        self.recent_every_s, self.match_iou, self.match_px = float(recent_every_s), float(match_iou), float(match_px)
        self._tracks: dict[str, CropTrack] = {}
        self._closed: deque = deque(maxlen=self.max_tracks)     # finished series, newest last
        self._next = 1
        self._sids = itertools.count(1)
        self._lock = threading.Lock()

    @classmethod
    def from_config(cls, cfg: Optional[dict]) -> 'CropStore':
        c = dict(cfg or {})
        c.pop('enabled', None)
        return cls(**c)

    def __len__(self) -> int:
        return len(self._tracks)

    def clear(self) -> None:
        with self._lock:
            self._tracks.clear()
            self._closed.clear()

    # -- writing (perception thread)

    def update(self, img: Optional[np.ndarray], items: list[Detection], hands: list[Detection], t: float) -> None:
        if img is None or not items:
            return
        hand_boxes = [h.box_px for h in hands]
        with self._lock:
            keys = self._assign(items)
            for d, key in zip(items, keys):
                others = [o.box_px for o in items if o is not d]
                score = clean_score(img, d.box_px, d.conf, others, hand_boxes)
                touched = any(geom.overlap_frac(h, d.box_px) > 0 for h in hand_boxes)
                look = _look(img, d.box_px) if d.cls == THING and not touched else None
                tr = self._tracks.get(key)
                if tr is None:
                    tr = self._start(key, d, t)
                elif d.cls == THING and (t - tr.last_t > self.renew_after_s or (
                        look is not None and tr.look is not None and _look_dist(look, tr.look) > self.swap_de)):
                    self._close(tr)                 # gone a while, or it looks different: maybe another object
                    tr = self._start(key, d, t)
                tr.box_px, tr.box_cm, tr.last_t = d.box_px, d.box_cm, t
                if look is not None:
                    tr.look = look if tr.look is None else 0.8 * tr.look + 0.2 * look
                better = tr.best is None or score > tr.best.score
                due = tr.recent is None or t - tr.recent.t >= self.recent_every_s
                if not (better or due):
                    continue
                crop = self._cut(img, d, t, score, tr.sid)
                if crop is None:
                    continue
                if better:
                    tr.best = crop
                if due:
                    tr.recent = crop
            self._evict()

    def _assign(self, items: list[Detection]) -> list[str]:
        """Names for configured objects; for proposals the thing track whose last box they overlap
        best (IoU >= match_iou, else centre within match_px), one proposal per track, else a new one."""
        keys: list[Optional[str]] = [None if d.cls == THING else d.cls for d in items]
        free = {k: tr for k, tr in self._tracks.items() if tr.cls == THING}
        pairs = []
        for i, d in enumerate(items):
            if d.cls != THING:
                continue
            for k, tr in free.items():
                ov = geom.iou(d.box_px, tr.box_px)
                dist = geom.dist(geom.center(d.box_px), geom.center(tr.box_px))
                if ov >= self.match_iou or dist <= self.match_px:
                    pairs.append((-ov, dist, -tr.last_t, i, k))
        used = set()
        for _, _, _, i, k in sorted(pairs):
            if keys[i] is None and k not in used:
                keys[i] = k
                used.add(k)
        for i, k in enumerate(keys):
            if k is None:
                keys[i] = f'p{self._next}'
                self._next += 1
        return keys

    def _start(self, key: str, d: Detection, t: float) -> CropTrack:
        tr = self._tracks[key] = CropTrack(key, d.cls, d.box_px, d.box_cm, t, sid=next(self._sids))
        return tr

    def _close(self, tr: CropTrack) -> None:
        """Keep a finished series (its crops may be some thing's close-up) among the closed ones."""
        if tr.best is not None or tr.recent is not None:
            self._closed.append(tr)

    def _cut(self, img, d: Detection, t: float, score: float, sid: int = 0) -> Optional[Crop]:
        c = _clip(img, d.box_px, self.margin)
        if c is None:
            return None
        x1, y1, x2, y2 = c
        crop = img[y1:y2, x1:x2]
        s = self.size_px / max(crop.shape[:2])
        crop = (cv2.resize(crop, (max(1, round(crop.shape[1] * s)), max(1, round(crop.shape[0] * s))),
                           interpolation=cv2.INTER_AREA) if s < 1 else crop.copy())
        return Crop(img=crop, box_px=tuple(d.box_px), box_cm=tuple(d.box_cm), t=t, score=round(score, 4), sid=sid)

    def _evict(self) -> None:
        while len(self._tracks) > self.max_tracks:
            self._close(self._tracks.pop(min(self._tracks, key=lambda k: self._tracks[k].last_t)))

    # -- ownership (the world's thread after each update, or any reader)

    def bind(self, ent: Entity) -> bool:
        """The world last observed thing ent as the view whose box object is ent.box_cm: make ent the
        owner of that view's series up to it. False when no series made that observation or another
        thing already owns it; then that series is closed at its last confirmed view and ent starts a
        new one (its crops come from later views)."""
        box = ent.box_cm
        if box is None:
            return False
        with self._lock:
            tr = next((tr for tr in itertools.chain(self._tracks.values(), self._closed)
                       if tr.cls == THING and tr.box_cm is box), None)
            if tr is None:
                return False
            if tr.owner is None or tr.owner == ent.name:
                tr.owner, tr.owner_t, tr.confirmed = ent.name, tr.last_t, (tr.best, tr.recent)
                return True
            if self._tracks.get(tr.key) is tr:
                tr.best, tr.recent = tr.confirmed          # views since owner_t were not the owner
                self._close(tr)
                self._tracks[tr.key] = CropTrack(tr.key, tr.cls, tr.box_px, tr.box_cm, tr.last_t,
                                                 sid=next(self._sids), owner=ent.name, owner_t=tr.last_t,
                                                 look=tr.look)
            return False

    # -- reading (any thread)

    def get(self, key: str) -> Optional[CropTrack]:
        with self._lock:
            return self._tracks.get(key)

    def for_box(self, box_cm=None, box_px=None, name: Optional[str] = None,
                min_iou: float = 0.2, gate_cm: float = 5.0) -> Optional[CropTrack]:
        """The track for a configured object by name, else the proposal track whose last box best
        overlaps box_cm (or box_px); failing that the nearest centre within gate_cm (box_cm only).
        Ties go to the most recently seen."""
        with self._lock:
            if name is not None and name in self._tracks:
                return self._tracks[name]
            use_cm = box_cm is not None
            box = box_cm if use_cm else box_px
            if box is None:
                return None
            best, key = None, None
            for tr in self._tracks.values():
                if tr.cls != THING:
                    continue
                other = tr.box_cm if use_cm else tr.box_px
                ov = geom.iou(box, other)
                d = geom.dist(geom.center(box), geom.center(other))
                ok = ov >= min_iou or (use_cm and d <= gate_cm)
                k = (ov, -d, tr.last_t)
                if ok and (key is None or k > key):
                    best, key = tr, k
            return best

    def for_entity(self, ent: Entity) -> Optional[CropTrack]:
        """Crops for a world entity, as a snapshot: a configured object's by its name; a thing's from
        the newest series confirmed as it (see bind), with only the crops taken up to its last
        confirmed view. None when no crop is known to be of this entity."""
        tr = self.get(ent.name)
        if tr is not None:
            with self._lock:
                return replace(tr)
        self.bind(ent)
        with self._lock:
            mine = [tr for tr in itertools.chain(self._tracks.values(), self._closed) if tr.owner == ent.name]
            for tr in sorted(mine, key=lambda tr: tr.owner_t, reverse=True):
                best, recent = (c if c is not None and c.t <= tr.owner_t else k
                                for c, k in zip((tr.best, tr.recent), tr.confirmed))
                if best is not None or recent is not None:
                    return replace(tr, best=best, recent=recent)
        return None


_active: Optional[CropStore] = None


def set_active(store: Optional[CropStore]) -> None:
    global _active
    _active = store


def active() -> Optional[CropStore]:
    """The crop store of the running Detector (one per process), or None."""
    return _active


def event_snapshot(events, obj: str, types=None, n: int = 5, table=None, half_cm: float = 8.0):
    """(event, image) for obj's newest event that has a snapshot on disk (types: only these event
    types), or None. With a table (cm_to_px), the image is cropped to a 2 x half_cm square around the
    event's to_cm (else from_cm); without a position the whole frame comes back. Snapshots are written
    by a background thread, so an event from the last few ms may have no file yet: it is skipped."""
    want = None if types is None else {str(t) for t in types}
    for ev in events.last(obj, n):
        if want is not None and str(ev.type) not in want:
            continue
        if not ev.snapshot or not os.path.exists(ev.snapshot):
            continue
        img = cv2.imread(ev.snapshot)
        if img is None:
            continue
        pos = ev.to_cm or ev.from_cm
        if table is not None and pos is not None:
            (cx, cy), = np.asarray(table.cm_to_px([pos]), float).reshape(-1, 2)
            (ex, _), = np.asarray(table.cm_to_px([(pos[0] + half_cm, pos[1])]), float).reshape(-1, 2)
            r = abs(ex - cx)
            c = _clip(img, (cx - r, cy - r, cx + r, cy + r))
            if c is not None:
                img = img[c[1]:c[3], c[0]:c[2]].copy()
        return ev, img
    return None
