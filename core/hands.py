"""Hand tracking (spec 3.4). Owner: P.

Hands come from the detector's hand class (MediaPipe is dropped: CPU-only on Jetson, and its model
card excludes hands holding objects). This gives each hand a stable id 'hand:N' from frame to frame:
match by box overlap (IoU > iou_min) first, then by nearest centre within max_jump_px; a track
unseen for lost_s is dropped, and a later hand gets a new id.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Optional

from core.geom import iou
from core.types import Detection


@dataclass
class _Track:
    id: str
    det: Detection
    t: float


class HandTracker:
    def __init__(self, frame_size=(1280, 720), lost_s: float = 0.5, iou_min: float = 0.3,
                 edge_frac: float = 0.05, max_jump_px: float = 300.0):
        self.w, self.h = frame_size
        self.lost_s, self.iou_min, self.edge_frac, self.max_jump_px = lost_s, iou_min, edge_frac, max_jump_px
        self._tracks: dict[str, _Track] = {}
        self._gone: dict[str, _Track] = {}         # dropped tracks, for last() / exited_edge()
        self._next = 1

    def update(self, hands: list[Detection], t: float) -> list[Detection]:
        """Same boxes, in the same order, with cls = their track id."""
        for tid in [k for k, tr in self._tracks.items() if t - tr.t > self.lost_s]:
            self._gone[tid] = self._tracks.pop(tid)
        assigned: dict[int, str] = {}
        free = set(self._tracks)
        pairs = sorted(((iou(h.box_px, self._tracks[k].det.box_px), i, k)
                        for i, h in enumerate(hands) for k in free), reverse=True)
        for score, i, k in pairs:
            if score <= self.iou_min:
                break
            if i not in assigned and k in free:
                assigned[i] = k
                free.discard(k)
        for i, h in enumerate(hands):
            if i in assigned:
                continue
            near = [(self._dist(h, self._tracks[k].det), k) for k in free]
            near = [(d, k) for d, k in near if d <= self.max_jump_px]
            if near:
                k = min(near)[1]
                assigned[i] = k
                free.discard(k)
            else:
                assigned[i] = f"hand:{self._next}"
                self._next += 1
        out = []
        for i, h in enumerate(hands):
            d = replace(h, cls=assigned[i])
            self._tracks[assigned[i]] = _Track(assigned[i], d, t)
            out.append(d)
        return out

    @staticmethod
    def _dist(a: Detection, b: Detection) -> float:
        ax, ay = (a.box_px[0] + a.box_px[2]) / 2, (a.box_px[1] + a.box_px[3]) / 2
        bx, by = (b.box_px[0] + b.box_px[2]) / 2, (b.box_px[1] + b.box_px[3]) / 2
        return math.hypot(ax - bx, ay - by)

    def last(self, hand_id: str) -> Optional[Detection]:
        tr = self._tracks.get(hand_id) or self._gone.get(hand_id)
        return tr.det if tr else None

    def exited_edge(self, hand_id: str) -> Optional[str]:
        """The frame border the hand's last box was within edge_frac of (nearest if several), else None."""
        d = self.last(hand_id)
        if d is None:
            return None
        x1, y1, x2, y2 = d.box_px
        mx, my = self.edge_frac * self.w, self.edge_frac * self.h
        gaps = [(x1, mx, "left"), (self.w - x2, mx, "right"), (y1, my, "top"), (self.h - y2, my, "bottom")]
        near = [(g, side) for g, m, side in gaps if g <= m]
        return min(near)[1] if near else None

    def reset(self) -> None:
        """Forget tracks; ids keep counting up so old events never collide with new hands."""
        self._gone.update(self._tracks)
        self._tracks.clear()
