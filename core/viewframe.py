"""The table from the user's seat (config viewer:, data/viewer.json).

Table cm (core/table.py) are the camera's frame: in one-tag mode x runs left to right across the table view
and y from its top (far from the camera) to its bottom, with the origin at the corner of all the camera sees
of the table plane. The world, the laser, relations and hand edges stay in that frame. What a person hears
or sees is turned into their frame here, at the outputs only:

  viewer frame   x runs from the user's left to their right, y from the far side to the side nearest them;
                 (0, 0) is the far left corner of the tabletop and size is (width, depth) from their seat.
  front          the camera-frame side of the table the user sits at: bottom (the camera's side), right,
                 top or left. The couch on the rig is at the table view's right: front right.
  tabletop       with an outline (core/table_area.py, table_area.json), the viewer frame is the outline's
                 smallest enclosing rectangle, turned square to the camera's axes (at most 45 deg) and
                 cropped to it, so the map and "your left" are the real table. Without one, the whole
                 calibrated area.

View.from_cfg(cfg) builds it from table.size_cm, table_area.polygon_cm and viewer.front (cfg is the app's
live dict: POST /orientation changes viewer.front and the next answer uses it). to_view() turns a point,
edge() and off_table() a camera edge ('left', from core/world.py and core/hands.py), area() says where a
point is. to_json() goes out in GET /state as "view"; the BLE bridge applies its affine "m" with the
standard library (mobile/bridge/bleproto.py), so the phone gets viewer-frame positions, sizes and edges.

The choice persists in data/viewer.json (paths.viewer), set by POST /orientation (server/app.py) from the
phone's "I sit here" (mobile/PROTOCOL.md 5b); apply_saved(cfg) puts it into cfg at start.
"""
from __future__ import annotations

import json
import logging
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

log = logging.getLogger(__name__)

FRONTS = ("bottom", "right", "top", "left")
DEFAULT_FRONT = "bottom"
FILE = "data/viewer.json"
Point = tuple[float, float]

# Camera edge -> viewer edge, per front: the side the user sits at becomes 'bottom' (nearest them).
_EDGE = {
    "bottom": {"left": "left", "right": "right", "top": "top", "bottom": "bottom"},
    "top": {"left": "right", "right": "left", "top": "bottom", "bottom": "top"},
    "right": {"right": "bottom", "left": "top", "bottom": "left", "top": "right"},
    "left": {"left": "bottom", "right": "top", "top": "left", "bottom": "right"},
}

# Viewer edge -> spoken side: 'carried off {OFF[edge]}'.
OFF = {"bottom": "the side of the table nearest you", "top": "the far side of the table",
       "left": "the table on your left", "right": "the table on your right"}
# Short words for the LLM's facts and the visual digest.
EDGE_WORD = {"bottom": "near side", "top": "far side", "left": "your left", "right": "your right"}


def norm_front(front) -> Optional[str]:
    """'Right' -> 'right'; None for anything that isn't one of FRONTS."""
    f = str(front or "").strip().lower()
    return f if f in FRONTS else None


def front_of(cfg: dict) -> str:
    return norm_front((cfg.get("viewer") or {}).get("front")) or DEFAULT_FRONT


def _min_rect(pts: Sequence[Point]) -> tuple[float, tuple[float, float, float, float]]:
    """(theta, (u0, v0, u1, v1)): the smallest-area rectangle around pts with one side parallel to a
    polygon edge (the minimum is always one), its angle folded into (-45, 45] deg from the x axis, and
    its bounds in the frame turned by -theta. Pure Python: the bridge's host has no numpy."""
    best = None
    n = len(pts)
    for i in range(n):
        (x0, y0), (x1, y1) = pts[i], pts[(i + 1) % n]
        if (x0, y0) == (x1, y1):
            continue
        th = math.atan2(y1 - y0, x1 - x0)
        th = (th + math.pi / 4) % (math.pi / 2) - math.pi / 4     # fold into [-45, 45) deg
        if th <= -math.pi / 4 + 1e-12:
            th += math.pi / 2
        b = _bounds(pts, th)
        area = (b[2] - b[0]) * (b[3] - b[1])
        if best is None or area < best[0] - 1e-9:
            best = (area, th, b)
    if best is None:
        return 0.0, _bounds(pts, 0.0)
    return best[1], best[2]


def _bounds(pts: Sequence[Point], th: float) -> tuple[float, float, float, float]:
    c, s = math.cos(th), math.sin(th)
    us = [x * c + y * s for x, y in pts]
    vs = [-x * s + y * c for x, y in pts]
    return min(us), min(vs), max(us), max(vs)


@dataclass(frozen=True)
class View:
    """Camera-frame table cm -> the user's frame. m is the 2x3 affine; size is (width, depth) from the seat."""
    front: str
    size: tuple[float, float]
    m: tuple[tuple[float, float, float], tuple[float, float, float]]
    outline: bool = False

    @staticmethod
    def from_cfg(cfg: dict) -> "View":
        t = cfg.get("table") or {}
        w, h = (float(v) for v in (t.get("size_cm") or (90, 60))[:2])
        poly = [(float(p[0]), float(p[1])) for p in ((cfg.get("table_area") or {}).get("polygon_cm") or [])]
        return View.make(front_of(cfg), (w, h), poly if len(poly) >= 3 else None)

    @staticmethod
    def make(front: str, size_cm: Sequence[float], outline: Optional[Sequence[Point]] = None) -> "View":
        front = norm_front(front) or DEFAULT_FRONT
        if outline:
            th, (u0, v0, u1, v1) = _min_rect(list(outline))
        else:
            th, (u0, v0, u1, v1) = 0.0, (0.0, 0.0, float(size_cm[0]), float(size_cm[1]))
        W, H = u1 - u0, v1 - v0
        c, s = math.cos(th), math.sin(th)
        # base frame: turned by -th and shifted so the rectangle is [0, W] x [0, H]
        base = ((c, s, -u0), (-s, c, -v0))
        # then the seat: (u, v) -> viewer (x, y)
        seat = {"bottom": ((1, 0, 0), (0, 1, 0)),
                "top": ((-1, 0, W), (0, -1, H)),
                "right": ((0, -1, H), (1, 0, 0)),
                "left": ((0, 1, 0), (-1, 0, W))}[front]
        m = tuple(tuple(round(seat[r][0] * base[0][k] + seat[r][1] * base[1][k] + (seat[r][2] if k == 2 else 0), 6)
                        for k in range(3)) for r in range(2))
        size = (W, H) if front in ("bottom", "top") else (H, W)
        return View(front, (round(size[0], 1), round(size[1], 1)), m, bool(outline))

    def to_view(self, p) -> Optional[Point]:
        """Camera-frame table cm -> viewer cm (None stays None)."""
        if p is None:
            return None
        x, y = float(p[0]), float(p[1])
        (a, b, c), (d, e, f) = self.m
        return a * x + b * y + c, d * x + e * y + f

    def edge(self, cam_edge: Optional[str]) -> Optional[str]:
        """A camera edge ('left', 'top', ...) as the user's: 'bottom' is the side nearest them."""
        return _EDGE[self.front].get(str(cam_edge)) if cam_edge else None

    def off_table(self, cam_edge: Optional[str]) -> str:
        """What follows 'carried off': 'the table on your left', 'the far side of the table', or 'the table'."""
        e = self.edge(cam_edge)
        return OFF[e] if e else "the table"

    def edge_word(self, cam_edge: Optional[str]) -> Optional[str]:
        e = self.edge(cam_edge)
        return EDGE_WORD[e] if e else None

    def cells(self, p) -> tuple[str, str]:
        """(row, col) thirds of the table from the seat: row 'far' / 'near' / '', col 'left' / 'right' / ''."""
        x, y = self.to_view(p)
        w, h = self.size
        col = "left" if x < w / 3 else "right" if x > 2 * w / 3 else ""
        row = "far" if y < h / 3 else "near" if y > 2 * h / 3 else ""
        return row, col

    def area(self, p) -> str:
        """Where a point is, as the user sees it: 'at the far left', 'on your right, near you', 'in the middle'."""
        if p is None:
            return "somewhere on the table"
        row, col = self.cells(p)
        if row and col:
            return f"at the far {col}" if row == "far" else f"on your {col}, near you"
        if row:
            return "on the far side" if row == "far" else "on the side nearest you"
        if col:
            return f"on your {col}"
        return "in the middle"

    def area_word(self, p) -> Optional[str]:
        """Short area for the LLM's facts: 'far left', 'near', 'your right', 'middle'; None for no point."""
        if not p:
            return None
        row, col = self.cells(p)
        if row and col:
            return f"{row} {col}"
        if row:
            return f"{row} side"
        return f"your {col}" if col else "middle"

    def to_json(self) -> dict:
        return {"front": self.front, "table": list(self.size), "m": [list(r) for r in self.m],
                "outline": self.outline}


# ----- the saved choice -------------------------------------------------------------------------------

def saved_path(cfg: dict) -> Path:
    return Path((cfg.get("paths") or {}).get("viewer", FILE))


def apply_saved(cfg: dict) -> dict:
    """Put a saved seat (data/viewer.json) into cfg['viewer']['front'] (in place); the configured one stays
    without a valid file."""
    p = saved_path(cfg)
    try:
        front = norm_front(json.loads(p.read_text()).get("front")) if p.exists() else None
    except (OSError, ValueError, AttributeError):
        log.warning("viewer: %s is unreadable; using viewer.front from the config", p)
        front = None
    v = dict(cfg.get("viewer") or {})
    v.setdefault("default_front", front_of(cfg))   # the configured seat, for a reset from the phone
    if front:
        v["front"] = front
    cfg["viewer"] = v
    return cfg


def set_front(cfg: dict, front) -> str:
    """The seat from the phone: into cfg (answers use it from the next one) and saved atomically. None (the
    phone's "use the rig's default") goes back to the configured seat and removes the saved one.
    ValueError for anything but bottom / right / top / left / None."""
    p = saved_path(cfg)
    if front is None:
        v = cfg.get("viewer") or {}
        f = norm_front(v.get("default_front")) or front_of(cfg)
        cfg["viewer"] = dict(v, front=f)
        try:
            p.unlink()
        except FileNotFoundError:
            pass
        except OSError as e:
            log.warning("viewer: could not remove %s: %s", p, e)
        return f
    f = norm_front(front)
    if f is None:
        raise ValueError(f"front must be one of {', '.join(FRONTS)}, not {front!r}")
    cfg["viewer"] = dict(cfg.get("viewer") or {}, front=f)
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_name(p.name + ".tmp")
        tmp.write_text(json.dumps({"front": f, "t": time.time()}))
        os.replace(tmp, p)
    except OSError as e:                     # the seat still applies until the app restarts
        log.warning("viewer: could not save %s: %s", p, e)
    return f
