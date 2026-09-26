"""Tabletop outline: where objects may appear (config table_area:).

The calibration (core/table.py) maps the whole table PLANE, and in one-tag mode the tracked area is all
the camera sees of it: the floor, a chair or a knee beside the table map to table cm like the tabletop
does, and made phantom things on the rig. The outline is the tabletop itself, a polygon in table cm set
by the operator. It is used in two places:

  proposals   core/proposals.table_roi: the outline in image px is the proposers' ROI, so a proposal
              whose centre lies outside it is never made (ChangeProposer masks those pixels, YOLOE drops
              those boxes). proposals.*.ignore_px boxes still apply on top.
  admission   core/things.py: a NEW thing:N is created only in the interior, the outline shrunk by
              edge_cm. In the edge band existing things are still followed (visible, held, a hidden one
              coming back), so an object slid off the table keeps its identity until it leaves and the
              world reports how it went.

With no outline (polygon_cm empty, the default) the whole calibrated view counts, as before.

Setting it on the rig: python -m core.table --outline prints the steps; --outline-px takes the corners
clicked in a camera frame (converted with the calibration), --outline takes them in table cm. They are
saved to table_area.json next to table_cal.json with a hash of the calibration matrix: table cm move
when the table is recalibrated (one-tag mode puts the origin at the view's corner), so a saved outline
from another calibration is ignored with a warning. main.py applies a valid one at start
(apply_saved_area), before the world and the detector read the config.
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

import cv2
import numpy as np

log = logging.getLogger(__name__)

FILE = 'table_area.json'
EDGE_CM = 3.0
Point = tuple[float, float]


@dataclass
class TableArea:
    """The table_area: section: an outline in table cm (empty: the whole view) and its edge band."""
    polygon_cm: list[Point] = field(default_factory=list)
    edge_cm: float = EDGE_CM

    def __post_init__(self):
        self._contour = (np.array(self.polygon_cm, np.float32).reshape(-1, 1, 2)
                         if len(self.polygon_cm) >= 3 else None)

    @classmethod
    def from_dict(cls, raw: Optional[dict]) -> 'TableArea':
        raw = raw or {}
        pts = [(float(p[0]), float(p[1])) for p in raw.get('polygon_cm') or []]
        if pts and len(pts) < 3:
            log.warning("table_area.polygon_cm needs at least 3 corners, got %d; using the whole view", len(pts))
            pts = []
        return cls(pts, float(raw.get('edge_cm', EDGE_CM)))

    @property
    def defined(self) -> bool:
        return self._contour is not None

    def edge_dist(self, pt) -> float:
        """Distance in cm from pt to the outline, positive inside; +inf with no outline."""
        if self._contour is None:
            return math.inf
        return float(cv2.pointPolygonTest(self._contour, (float(pt[0]), float(pt[1])), True))

    def contains(self, pt) -> bool:
        """On the tabletop (the edge band included)."""
        return self.edge_dist(pt) >= 0.0

    def interior(self, pt) -> bool:
        """On the tabletop and at least edge_cm from its edge: where a new thing may be born."""
        return self.edge_dist(pt) >= self.edge_cm


# ----- saved with the calibration -------------------------------------------------------------------

def area_path(cfg: dict) -> Path:
    """table_area.json next to the calibration file (paths.table_cal)."""
    return Path((cfg.get('paths') or {}).get('table_cal', 'table_cal.json')).with_name(FILE)


def cal_hash(H) -> str:
    """Fingerprint of a px -> cm calibration matrix (JSON round-trips floats exactly)."""
    m = np.round(np.asarray(H, dtype=np.float64), 9).tolist()
    return hashlib.sha1(json.dumps(m).encode()).hexdigest()[:16]


def _saved_h(cfg: dict) -> Optional[np.ndarray]:
    p = Path((cfg.get('paths') or {}).get('table_cal', 'table_cal.json'))
    try:
        return np.array(json.loads(p.read_text())['H'], dtype=np.float64) if p.exists() else None
    except (OSError, ValueError, KeyError):
        return None


def load_saved(path: Path, H) -> Optional[list[Point]]:
    """The saved outline in table cm, or None (no file, unreadable, or set with another calibration)."""
    if not path.exists():
        return None
    try:
        d = json.loads(path.read_text())
        pts = [(float(x), float(y)) for x, y in d['polygon_cm']]
    except (OSError, ValueError, KeyError, TypeError):
        log.warning("table outline: %s is unreadable; ignoring it", path)
        return None
    if H is None or d.get('cal_hash') != cal_hash(H):
        log.warning("table outline: %s was set with another table calibration (table_cal.json changed); "
                    "ignoring it. Set it again: python -m core.table --outline", path)
        return None
    return pts if len(pts) >= 3 else None


def apply_saved_area(cfg: dict) -> dict:
    """Put a valid saved outline into cfg['table_area']['polygon_cm'] (in place), before the world and
    the detector read it. Without one the configured polygon_cm stays."""
    pts = load_saved(area_path(cfg), _saved_h(cfg)) if area_path(cfg).exists() else None
    if pts:
        cfg['table_area'] = dict(cfg.get('table_area') or {}, polygon_cm=[list(p) for p in pts])
        log.info("table outline: %d corners from %s", len(pts), area_path(cfg))
    return cfg


def set_outline(table, cfg: dict, points: Sequence, px: bool = False) -> list[Point]:
    """Save the outline (corners in table cm, or image px with px=True) for this calibration; returns
    it in table cm."""
    if not getattr(table, 'ok', False):
        raise RuntimeError("table is not calibrated: run python -m core.table first")
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    if len(pts) < 3:
        raise ValueError("an outline needs at least 3 corners")
    cm = table.px_to_cm(pts) if px else pts
    out = [(round(float(x), 2), round(float(y), 2)) for x, y in cm]
    path = area_path(cfg)
    path.write_text(json.dumps({'polygon_cm': [list(p) for p in out],
                                'polygon_px': pts.tolist() if px else None,
                                'cal_hash': cal_hash(table.H), 't': time.time()}, indent=1))
    return out


# ----- python -m core.table --outline ------------------------------------------------------------

HOW = """Tabletop outline: where objects may appear. The calibration covers all the camera sees of the
table plane (floor and chairs beside the table too); the outline is the tabletop itself.

  1. Take a frame from the running rig:   curl -o frame.jpg http://<jetson>:8000/frame.jpg
  2. Open frame.jpg in a viewer that shows pixel coordinates and note the tabletop's corners in
     order around the table (4, or more for an odd shape; stay a little inside the real edge).
  3. python -m core.table --outline-px 112,80 1190,64 1215,700 90,690
     (or --outline x,y ... in table cm; add --image frame.jpg to draw it into frame_outline.jpg)
  4. Restart the app. Recalibrating the table invalidates the outline: set it again after.

New things are only born at least table_area.edge_cm inside the outline; config.yaml table_area:."""


def parse_points(tokens: Sequence[str]) -> list[Point]:
    """['112,80', '1190,64', ...] -> [(112.0, 80.0), ...]; ValueError on a malformed corner."""
    out = []
    for tok in tokens:
        parts = tok.split(',')
        if len(parts) != 2:
            raise ValueError(f"corner {tok!r} is not x,y")
        out.append((float(parts[0]), float(parts[1])))
    return out


def outline_main(table, cfg: dict, cm_tokens: Optional[Sequence[str]], px_tokens: Optional[Sequence[str]],
                 image: Optional[str] = None) -> int:
    """The --outline / --outline-px CLI: 0 saved (or the steps printed), 1 uncalibrated, 2 bad corners."""
    px = bool(px_tokens)
    tokens = list(px_tokens or cm_tokens or [])
    if not tokens:
        print(HOW)
        saved = load_saved(area_path(cfg), getattr(table, 'H', None)) if area_path(cfg).exists() else None
        print(f"\ncurrent: {len(saved)} corners in {area_path(cfg)}" if saved else "\ncurrent: none (the whole view)")
        return 0
    try:
        pts = parse_points(tokens)
        if len(pts) < 3:
            raise ValueError("an outline needs at least 3 corners")
    except ValueError as e:
        print(f"outline not saved: {e}")
        return 2
    try:
        cm = set_outline(table, cfg, pts, px=px)
    except RuntimeError as e:
        print(f"outline not saved: {e}")
        return 1
    corners_px = pts if px else [tuple(float(v) for v in p) for p in table.cm_to_px(cm)]
    print("tabletop outline (table cm):", cm)
    print("               (image px):", [tuple(round(v) for v in p) for p in corners_px])
    print(f"saved {area_path(cfg)}; restart the app to use it")
    if image:
        img = cv2.imread(image)
        if img is None:
            print(f"could not read {image}")
        else:
            cv2.polylines(img, [np.round(np.array(corners_px)).astype(np.int32).reshape(-1, 1, 2)], True,
                          (0, 255, 0), 2)
            out = str(Path(image).with_name(Path(image).stem + '_outline.jpg'))
            cv2.imwrite(out, img)
            print(f"drawn into {out}")
    return 0
