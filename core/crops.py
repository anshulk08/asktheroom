"""Crops of what the detector saw, for open-world identity and naming. Owner: P.

Naming an unknown thing ("what is that?"), matching it by appearance, or showing it on the dashboard
needs a picture of it, and the frame an answer is asked about is rarely a good one (a hand is on it,
it is moving). The Detector feeds every frame's objects and proposals here; per object this keeps
  best    the cleanest view so far: sharp, confident, no hand on it, not overlapped by anything else
  recent  the latest view (at most every recent_every_s)
Configured objects are keyed by name. Proposals have no identity in perception, so they are keyed by
box position (a proposal continues the track whose last box it overlaps; after renew_after_s unseen its
crops start over, as another object may sit there now); the world asks by the box it
knows: for_box(box_cm=entity.box_cm) or for_entity(entity). Memory is bounded: at most max_tracks
tracks of two crops of at most size_px on the long side (~6 MB at the defaults).

The Detector registers its store as the process's active() store, so the world and the answer code
can reach it without new wiring through main.py. event_snapshot() fetches the JPEG that
core/events.py saved for an object's latest event.
"""
from __future__ import annotations

import os
import threading
from dataclasses import dataclass
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
    t: float
    score: float


@dataclass
class CropTrack:
    key: str                      # the object name, or 'p<N>' for a proposal track
    cls: str
    box_px: BoxPx
    box_cm: tuple
    last_t: float
    best: Optional[Crop] = None
    recent: Optional[Crop] = None


def _clip(img: np.ndarray, box, margin: float = 0.0) -> Optional[tuple[int, int, int, int]]:
    h, w = img.shape[:2]
    x1, y1, x2, y2 = (float(v) for v in box)
    mx, my = margin * (x2 - x1), margin * (y2 - y1)
    x1, y1 = max(0, int(x1 - mx)), max(0, int(y1 - my))
    x2, y2 = min(w, int(np.ceil(x2 + mx))), min(h, int(np.ceil(y2 + my)))
    return (x1, y1, x2, y2) if x2 - x1 >= 4 and y2 - y1 >= 4 else None


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
                 renew_after_s: float = 1.0):
        self.size_px, self.max_tracks, self.margin = int(size_px), int(max_tracks), float(margin)
        self.renew_after_s = float(renew_after_s)
        self.recent_every_s, self.match_iou, self.match_px = float(recent_every_s), float(match_iou), float(match_px)
        self._tracks: dict[str, CropTrack] = {}
        self._next = 1
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
                tr = self._tracks.get(key)
                if tr is None:
                    tr = self._tracks[key] = CropTrack(key, d.cls, d.box_px, d.box_cm, t)
                elif d.cls == THING and t - tr.last_t > self.renew_after_s:
                    tr.best = tr.recent = None      # gone a while: may be another object at the spot now
                tr.box_px, tr.box_cm, tr.last_t = d.box_px, d.box_cm, t
                better = tr.best is None or score > tr.best.score
                due = tr.recent is None or t - tr.recent.t >= self.recent_every_s
                if not (better or due):
                    continue
                crop = self._cut(img, d, t, score)
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

    def _cut(self, img, d: Detection, t: float, score: float) -> Optional[Crop]:
        c = _clip(img, d.box_px, self.margin)
        if c is None:
            return None
        x1, y1, x2, y2 = c
        crop = img[y1:y2, x1:x2]
        s = self.size_px / max(crop.shape[:2])
        crop = (cv2.resize(crop, (max(1, round(crop.shape[1] * s)), max(1, round(crop.shape[0] * s))),
                           interpolation=cv2.INTER_AREA) if s < 1 else crop.copy())
        return Crop(img=crop, box_px=tuple(d.box_px), box_cm=tuple(d.box_cm), t=t, score=round(score, 4))

    def _evict(self) -> None:
        while len(self._tracks) > self.max_tracks:
            del self._tracks[min(self._tracks, key=lambda k: self._tracks[k].last_t)]

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
        """Crops for a world entity: a configured object by its name, a thing by its last box."""
        tr = self.get(ent.name)
        if tr is not None:
            return tr
        return self.for_box(box_cm=ent.box_cm) if ent.box_cm is not None else None


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
