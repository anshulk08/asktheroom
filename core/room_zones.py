"""Room zones (spec 0009 M0): drawn polygons in full-frame px with spoken names, the room_zones.json file
that holds them, and the view version that says which camera view they were drawn at.

Zone polygons are full-frame pixels of the 1920x1080 camera frame, never table cm. The view version hashes
only the capture size, zoom and table_view_rect: table calibration is not in it, so a spoken RECAL (which
rewrites table_cal.json) does not invalidate the zones. Python 3.10 (JetPack 6).
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence, Union

import cv2
import numpy as np

from core.types import BoxPx


@dataclass
class Zone:
    name: str
    say: str                                   # spoken name, e.g. 'the bookshelf'
    poly: list[tuple[float, float]]            # full-frame px

    def __post_init__(self) -> None:
        self.poly = [(float(x), float(y)) for x, y in self.poly]

    def contains(self, pt: tuple[float, float]) -> bool:
        """Inside the polygon or on its edge. A polygon with under 3 points contains nothing."""
        if len(self.poly) < 3:
            return False
        contour = np.asarray(self.poly, np.float32).reshape(-1, 1, 2)
        return cv2.pointPolygonTest(contour, (float(pt[0]), float(pt[1])), False) >= 0

    def bbox(self) -> BoxPx:
        """Integer pixel box x1, y1, x2, y2 (x2, y2 exclusive: img[y1:y2, x1:x2] holds every pixel the
        polygon touches), clipped at 0. The caller clips to the frame size."""
        xs = [p[0] for p in self.poly]
        ys = [p[1] for p in self.poly]
        x1, y1 = max(0, math.floor(min(xs))), max(0, math.floor(min(ys)))
        x2, y2 = max(0, math.floor(max(xs)) + 1), max(0, math.floor(max(ys)) + 1)
        return (int(x1), int(y1), int(x2), int(y2))


@dataclass
class Zones:
    view: str                                  # view_version() the polygons were drawn at
    size_px: tuple[int, int]                   # full frame (w, h) they were drawn on
    zones: dict[str, Zone] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.size_px = (int(self.size_px[0]), int(self.size_px[1]))

    def at(self, pt: tuple[float, float]) -> Optional[Zone]:
        """The first zone (file order) containing pt, or None."""
        for z in self.zones.values():
            if z.contains(pt):
                return z
        return None

    def to_dict(self) -> dict:
        return {
            "view": self.view,
            "size_px": list(self.size_px),
            "zones": {n: {"say": z.say, "poly": [list(p) for p in z.poly]} for n, z in self.zones.items()},
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Zones":
        zones = {n: Zone(n, str(z.get("say") or n), [tuple(p) for p in z.get("poly", [])])
                 for n, z in (d.get("zones") or {}).items()}
        return cls(view=str(d.get("view", "")), size_px=tuple(d["size_px"]), zones=zones)

    def save(self, path: Union[str, Path]) -> None:
        """Write JSON atomically (temp file, then rename), creating the parent directory."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_name(p.name + ".tmp")
        tmp.write_text(json.dumps(self.to_dict(), indent=2))
        os.replace(tmp, p)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "Zones":
        """Raises FileNotFoundError when the file is missing."""
        return cls.from_dict(json.loads(Path(path).read_text()))


def view_version(size_px: Sequence[int], zoom: int,
                 table_view_rect: Optional[Sequence[int]]) -> str:
    """First 12 hex chars of sha1 over the canonical JSON of capture size, zoom and table_view_rect.
    Lists and tuples hash the same. No table calibration in it (see the module docstring)."""
    canon = {
        "size_px": [int(v) for v in size_px],
        "zoom": int(zoom),
        "table_view_rect": None if table_view_rect is None else [int(v) for v in table_view_rect],
    }
    blob = json.dumps(canon, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:12]
