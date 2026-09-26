"""Baseline answerers for the evaluation (spec 3.13). Owner: eval.

Every system (these and the World adapter in eval/replay.py) has the same two methods:

    update(dets: Detections) -> None        called once per cached frame, in order
    predict(obj: str) -> dict               called once, after the last frame (the question)

and predict returns a Prediction dict:

    {"status": "VISIBLE"|"HELD"|"UNDER"|"INSIDE"|"GONE"|"UNKNOWN",
     "parent": "box"|"notebook"|"hand:3"|None,
     "pos_cm": [x, y] | None,          # where the object itself was last observed
     "resolved_cm": [x, y] | None,     # where the laser would point
     "edge": "left"|...|None}

Presence is debounced only lightly: an object is "present" at question time if it was detected
with conf >= conf_threshold in any of the last `present_window` frames (default 3 frames,
~0.3 s at 10 fps) so a single dropped detection in the final frame does not flip the answer.

Baselines:
  current_frame   answers only from the final frames: VISIBLE at its detection, else UNKNOWN.
  last_seen       VISIBLE if present, else UNKNOWN pointing at the last detected position.
  nearest_object  like last_seen, but if the object is not present it guesses the object is
                  inside/under whichever container or cover (config kinds 'container'/'cover')
                  is nearest to the object's last-seen spot, measured as point-to-rectangle
                  distance to that container's CURRENT (last detected) box, within
                  `near_cm` (15 cm). The laser then points at the container's current centre.
                  Otherwise it falls back to last_seen.

Design decision (documented for the pitch): nearest_object keeps no event memory. It gets plain
'inside' and 'covered' right, because the box/notebook is still next to where the object vanished.
For 'inside_box_moved' it compares against where the box is NOW, so once the box has moved more
than ~15 cm it answers the stale last-seen spot (no parent) and is scored wrong. If the box moved
only a little it gets the right parent and points at the box's new spot. Only the full world
model, which attaches the keys to the box and carries them along, gets the far move right.
"""
from __future__ import annotations

from collections import deque
from typing import Optional

from core.types import Detection, Detections, Point


def conf_threshold(cfg: dict, cls: str) -> float:
    ct = cfg.get("conf_threshold", 0.35)
    if isinstance(ct, dict):
        return float(ct.get(cls, ct.get("default", 0.35)))
    return float(ct)


def prediction(status: str, parent: Optional[str] = None, pos_cm: Optional[Point] = None,
               resolved_cm: Optional[Point] = None, edge: Optional[str] = None) -> dict:
    return {"status": status, "parent": parent,
            "pos_cm": [round(pos_cm[0], 2), round(pos_cm[1], 2)] if pos_cm is not None else None,
            "resolved_cm": ([round(resolved_cm[0], 2), round(resolved_cm[1], 2)]
                            if resolved_cm is not None else None),
            "edge": edge}


def rect_dist(p: Point, box: tuple[float, float, float, float]) -> float:
    """Distance from point p to an axis-aligned rectangle (0 when inside)."""
    x1, y1, x2, y2 = box
    dx = max(x1 - p[0], 0.0, p[0] - x2)
    dy = max(y1 - p[1], 0.0, p[1] - y2)
    return (dx * dx + dy * dy) ** 0.5


class _Base:
    name = "base"

    def __init__(self, cfg: dict, present_window: int = 3):
        self.cfg = cfg
        self.present_window = max(1, int(present_window))
        self.recent: deque[dict[str, Detection]] = deque(maxlen=self.present_window)
        self.last: dict[str, Detection] = {}      # last confident detection per class

    def _confident(self, dets: Detections) -> dict[str, Detection]:
        best: dict[str, Detection] = {}
        for d in dets.items:
            if d.conf >= conf_threshold(self.cfg, d.cls) and (
                    d.cls not in best or d.conf > best[d.cls].conf):
                best[d.cls] = d
        return best

    def update(self, dets: Detections) -> None:
        best = self._confident(dets)
        self.recent.append(best)
        self.last.update(best)

    def present(self, obj: str) -> Optional[Detection]:
        """Most recent confident detection of obj within the present window, else None."""
        for frame in reversed(self.recent):
            if obj in frame:
                return frame[obj]
        return None

    def predict(self, obj: str) -> dict:
        raise NotImplementedError


class CurrentFrame(_Base):
    name = "current_frame"

    def predict(self, obj: str) -> dict:
        d = self.present(obj)
        if d is not None:
            return prediction("VISIBLE", pos_cm=d.center_cm, resolved_cm=d.center_cm)
        return prediction("UNKNOWN")


class LastSeen(_Base):
    name = "last_seen"

    def predict(self, obj: str) -> dict:
        d = self.present(obj)
        if d is not None:
            return prediction("VISIBLE", pos_cm=d.center_cm, resolved_cm=d.center_cm)
        d = self.last.get(obj)
        if d is not None:
            return prediction("UNKNOWN", pos_cm=d.center_cm, resolved_cm=d.center_cm)
        return prediction("UNKNOWN")


class NearestObject(LastSeen):
    name = "nearest_object"

    def __init__(self, cfg: dict, present_window: int = 3, near_cm: float = 15.0):
        super().__init__(cfg, present_window)
        self.near_cm = near_cm
        objs = cfg.get("objects", {})
        self.holders = [n for n, k in objs.items() if k in ("container", "cover")]

    def predict(self, obj: str) -> dict:
        if self.present(obj) is not None or obj not in self.last:
            return super().predict(obj)
        spot = self.last[obj].center_cm
        best, best_d = None, None
        for h in self.holders:
            if h == obj or h not in self.last:
                continue
            dist = rect_dist(spot, self.last[h].box_cm)
            if dist <= self.near_cm and (best_d is None or dist < best_d):
                best, best_d = h, dist
        if best is None:
            return super().predict(obj)
        kind = self.cfg["objects"][best]
        return prediction("INSIDE" if kind == "container" else "UNDER", parent=best,
                          pos_cm=spot, resolved_cm=self.last[best].center_cm)


BASELINES = {c.name: c for c in (CurrentFrame, LastSeen, NearestObject)}
