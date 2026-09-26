"""Table calibration (spec 3.2). Owner: P.

ArUco markers 0-3 (DICT_4X4_50) sit at measured table positions (config table.markers, centres in cm,
marker 0 = origin, x right, y down). calibrate() fits the homography px -> cm from the four marker
centres and saves it to table_cal.json; px_to_cm / cm_to_px convert points. Print markers with
scripts/make_markers.py.

Everything is assumed to lie on the table plane: the top of a tall object maps a few cm off (spec
review: store per-object heights if that bites).
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

log = logging.getLogger(__name__)

TABLE_IDS = (0, 1, 2, 3)


def detector() -> cv2.aruco.ArucoDetector:
    """OpenCV 4.7+ API (Dictionary_get / DetectorParameters_create no longer exist)."""
    return cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50),
                                   cv2.aruco.DetectorParameters())


def find_markers(img: np.ndarray, det: Optional[cv2.aruco.ArucoDetector] = None) -> dict[int, np.ndarray]:
    """{marker id: centre px} for every marker found in a BGR or grey image."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    corners, ids, _ = (det or detector()).detectMarkers(gray)
    if ids is None:
        return {}
    return {int(i): c.reshape(4, 2).mean(axis=0) for c, i in zip(corners, ids.flatten())}


class Table:
    def __init__(self, cfg: dict, cal_path: Optional[str] = None):
        t = cfg.get("table") or {}
        self.size_cm = tuple(t.get("size_cm", (90, 60)))
        self.markers_cm = {int(k): tuple(v) for k, v in (t.get("markers") or {}).items()}
        self.cal_path = cal_path or (cfg.get("paths") or {}).get("table_cal", "table_cal.json")
        self.H: Optional[np.ndarray] = None       # px -> cm
        self.Hinv: Optional[np.ndarray] = None    # cm -> px
        self.found: list[int] = []
        self._det = detector()
        self.load()

    @property
    def ok(self) -> bool:
        return self.H is not None

    def calibrate(self, frame_img: np.ndarray) -> bool:
        """Fit from markers 0-3 in this image. On failure the previous calibration stays."""
        found = find_markers(frame_img, self._det)
        self.found = sorted(i for i in found if i in TABLE_IDS)
        if len(self.found) < 4:
            log.warning("table calibration: found markers %s, need %s", self.found, list(TABLE_IDS))
            return False
        px = np.float32([found[i] for i in TABLE_IDS])
        cm = np.float32([self.markers_cm[i] for i in TABLE_IDS])
        self._set(cv2.getPerspectiveTransform(px, cm))
        self.save({i: found[i].tolist() for i in TABLE_IDS})
        return True

    def _set(self, H: np.ndarray) -> None:
        self.H = np.asarray(H, dtype=np.float64)
        self.Hinv = np.linalg.inv(self.H)

    def px_to_cm(self, pts) -> np.ndarray:
        return self._apply(self.H, pts)

    def cm_to_px(self, pts) -> np.ndarray:
        return self._apply(self.Hinv, pts)

    def _apply(self, M: Optional[np.ndarray], pts) -> np.ndarray:
        if M is None:
            raise RuntimeError("table is not calibrated: run calibrate() with markers 0-3 in view")
        p = np.asarray(pts, dtype=np.float64).reshape(-1, 1, 2)
        return cv2.perspectiveTransform(p, M).reshape(-1, 2)

    def in_bounds(self, cm, margin: float = 0.0) -> bool:
        x, y = cm
        w, h = self.size_cm
        return -margin <= x <= w + margin and -margin <= y <= h + margin

    def save(self, markers_px: dict) -> None:
        Path(self.cal_path).write_text(json.dumps(
            {"H": self.H.tolist(), "markers_px": markers_px, "size_cm": list(self.size_cm), "t": time.time()},
            indent=1))

    def load(self) -> bool:
        p = Path(self.cal_path)
        if not p.exists():
            return False
        self._set(np.array(json.loads(p.read_text())["H"]))
        return True


def main(argv=None) -> int:
    """python -m core.table [--device 0 | --image frame.jpg]: calibrate from the live camera or an image,
    report which markers were found and where the frame corners land on the table."""
    import argparse

    from core.config import load_config
    ap = argparse.ArgumentParser(description=main.__doc__)
    ap.add_argument("--device", default="0")
    ap.add_argument("--image")
    ap.add_argument("--frames", type=int, default=15, help="live: try this many frames")
    a = ap.parse_args(argv)
    cfg = load_config()
    table = Table(cfg)
    if a.image:
        imgs = [cv2.imread(a.image)]
    else:
        from core.capture import FrameBuffer
        fb = FrameBuffer(int(a.device) if a.device.isdigit() else a.device)
        imgs, last = [], 0
        while len(imgs) < a.frames:
            f = fb.wait_new(last, 2.0)
            if f is None:
                break
            imgs.append(f.img)
            last = f.idx
        fb.stop()
    ok = any(table.calibrate(img) for img in imgs[::-1])
    seen = sorted({i for img in imgs[-3:] for i in find_markers(img)})
    print(f"markers seen: {seen}; table ids found: {table.found}; calibrated: {ok}")
    if ok:
        h, w = imgs[-1].shape[:2]
        corners = table.px_to_cm([[0, 0], [w, 0], [w, h], [0, h]])
        print("frame corners on the table (cm):", [tuple(round(float(v), 1) for v in c) for c in corners])
        print(f"saved {table.cal_path}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
