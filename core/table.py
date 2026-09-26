"""Table calibration (spec 3.2). Owner: P.

ArUco markers 0-3 (DICT_4X4_50) sit at measured table positions (config table.markers, centres in cm,
marker 0 = origin, x right, y down). calibrate() fits the homography px -> cm from the four marker
centres and saves it to table_cal.json; px_to_cm / cm_to_px convert points. Print markers with
scripts/make_markers.py.

One-tag mode (config table_tag.enabled): a single printed AprilTag (36h11, the family FTC uses) of known
size lying anywhere on the table replaces the four markers, so nothing is measured. Its four corners,
averaged over table_tag.frames frames, give the same px -> cm homography (the tag's printed size is the
scale; its tilt is in the homography, so no lens calibration is needed for one flat surface). The tracked
area is what the camera sees of the table plane: the frame's footprint in the tag's axes, shifted so its
top-left is (0, 0), becomes size_cm and is saved with the calibration; main.py applies it at startup
(apply_saved_size) because many modules read table.size_cm. Less accurate at the far edges than four
spread markers (the tag's corners are close together); fine for tracking, check laser error on the rig.

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


FAMILIES = {"aruco_4x4_50": cv2.aruco.DICT_4X4_50, "apriltag_36h11": cv2.aruco.DICT_APRILTAG_36h11}


def detector(family: str = "aruco_4x4_50") -> cv2.aruco.ArucoDetector:
    """OpenCV 4.7+ API (Dictionary_get / DetectorParameters_create no longer exist)."""
    return cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(FAMILIES[family]),
                                   cv2.aruco.DetectorParameters())


def tag_corners(img: np.ndarray, tag_id: int, det: cv2.aruco.ArucoDetector) -> Optional[np.ndarray]:
    """The 4 corner px of one tag (its top-left, top-right, bottom-right, bottom-left), or None."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    corners, ids, _ = det.detectMarkers(gray)
    if ids is None:
        return None
    for c, i in zip(corners, ids.flatten()):
        if int(i) == tag_id:
            return c.reshape(4, 2).astype(np.float64)
    return None


def apply_saved_size(cfg: dict) -> dict:
    """One-tag mode: put the saved tracked-area size into cfg['table']['size_cm'] (in place) before the
    world, laser, detector and answers read it. Four-marker mode keeps the configured size."""
    if not (cfg.get("table_tag") or {}).get("enabled"):
        return cfg
    p = Path((cfg.get("paths") or {}).get("table_cal", "table_cal.json"))
    try:
        size = json.loads(p.read_text()).get("size_cm") if p.exists() else None
    except (OSError, ValueError):
        size = None
    if size:
        cfg["table"] = dict(cfg.get("table") or {}, size_cm=[float(v) for v in size])
    return cfg


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
        tt = cfg.get("table_tag") or {}
        self.tag_mode = bool(tt.get("enabled"))
        self.tag_id = int(tt.get("id", 0))
        self.tag_cm = float(tt.get("size_cm", 16.0))
        self.tag_frames = max(1, int(tt.get("frames", 15)))
        self.max_cm = float(tt.get("max_cm", 250))
        self._tag_seen: list[np.ndarray] = []
        self._det = detector(str(tt.get("family", "apriltag_36h11"))) if self.tag_mode else detector()
        self._warned: tuple = (None, -1e9)        # (markers seen, time) of the last failure warning
        self.load()

    @property
    def ok(self) -> bool:
        return self.H is not None

    def calibrate(self, frame_img: np.ndarray) -> bool:
        """Fit from markers 0-3 in this image (or, in one-tag mode, from the tag averaged over several
        frames). On failure the previous calibration stays."""
        if self.tag_mode:
            return self._calibrate_tag(frame_img)
        found = find_markers(frame_img, self._det)
        self.found = sorted(i for i in found if i in TABLE_IDS)
        if len(self.found) < 4:
            # Callers retry every frame until the markers appear: warn when what's visible changes, or
            # every 10 s, not 30 times a second.
            now = time.monotonic()
            if self.found != self._warned[0] or now - self._warned[1] >= 10:
                log.warning("table calibration: found markers %s, need %s", self.found, list(TABLE_IDS))
                self._warned = (list(self.found), now)
            return False
        px = np.float32([found[i] for i in TABLE_IDS])
        cm = np.float32([self.markers_cm[i] for i in TABLE_IDS])
        self._set(cv2.getPerspectiveTransform(px, cm))
        self.save({i: found[i].tolist() for i in TABLE_IDS})
        return True

    def _calibrate_tag(self, img: np.ndarray) -> bool:
        c = tag_corners(img, self.tag_id, self._det)
        if c is None:
            self._tag_seen.clear()                # averaging needs consecutive sightings
            now = time.monotonic()
            if self._warned[0] != "tag" or now - self._warned[1] >= 10:
                log.warning("table calibration: tag %d not in view", self.tag_id)
                self._warned = ("tag", now)
            return False
        self._tag_seen.append(c)
        if len(self._tag_seen) < self.tag_frames:
            return False
        px = np.mean(self._tag_seen, axis=0)
        self._tag_seen.clear()
        s = self.tag_cm
        H_tag = cv2.getPerspectiveTransform(np.float32(px), np.float32([[0, 0], [s, 0], [s, s], [0, s]]))
        h, w = img.shape[:2]
        # Turn the tag's axes so +x runs left -> right across the image, whatever angle the tag lies at:
        # 'left of the table' and the phone's map then match the camera view, and the area stays tight.
        a, b = cv2.perspectiveTransform(np.float64([[[w * 0.25, h / 2]], [[w * 0.75, h / 2]]]), H_tag).reshape(-1, 2)
        ang = np.arctan2(b[1] - a[1], b[0] - a[0])
        c, sn = np.cos(-ang), np.sin(-ang)
        H_tag = np.array([[c, -sn, 0], [sn, c, 0], [0, 0, 1]]) @ H_tag
        foot = cv2.perspectiveTransform(np.float64([[[0, 0]], [[w - 1, 0]], [[w - 1, h - 1]], [[0, h - 1]]]),
                                        H_tag).reshape(-1, 2)
        lo = np.clip(foot.min(axis=0), -self.max_cm, self.max_cm)
        hi = np.clip(foot.max(axis=0), -self.max_cm, self.max_cm)
        shift = np.array([[1, 0, -lo[0]], [0, 1, -lo[1]], [0, 0, 1]], dtype=np.float64)
        self._set(shift @ H_tag)
        self.size_cm = (round(float(hi[0] - lo[0]), 1), round(float(hi[1] - lo[1]), 1))
        self.found = [self.tag_id]
        self.save({"tag": px.tolist()})
        log.info("table calibrated from tag %d: tracked area %.0f x %.0f cm", self.tag_id, *self.size_cm)
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
        d = json.loads(p.read_text())
        self._set(np.array(d["H"]))
        if self.tag_mode and d.get("size_cm"):
            self.size_cm = tuple(float(v) for v in d["size_cm"])
        return True


def main(argv=None) -> int:
    """python -m core.table [--device 0 | --image frame.jpg]: calibrate from the live camera or an image,
    report which markers were found and where the frame corners land on the table."""
    import argparse

    from core.config import load_config
    ap = argparse.ArgumentParser(description=main.__doc__)
    ap.add_argument("--device", default="0")
    ap.add_argument("--image")
    ap.add_argument("--frames", type=int, default=40, help="live: try this many frames")
    ap.add_argument("--outline", nargs="*", metavar="X,Y", help="tabletop corners in table cm (none: how to set it)")
    ap.add_argument("--outline-px", nargs="+", metavar="X,Y", help="tabletop corners in image px")
    a = ap.parse_args(argv)
    cfg = load_config()
    table = Table(cfg)
    if a.outline is not None or a.outline_px:        # core/table_area.py: where objects may appear
        from core.table_area import outline_main
        return outline_main(table, cfg, a.outline, a.outline_px, a.image)
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
    seen = sorted({i for img in imgs[-3:] for i in find_markers(img, table._det)})
    print(f"markers seen: {seen}; table ids found: {table.found}; calibrated: {ok}"
          + (f"; tracked area {table.size_cm[0]:g} x {table.size_cm[1]:g} cm" if ok and table.tag_mode else ""))
    if ok:
        h, w = imgs[-1].shape[:2]
        corners = table.px_to_cm([[0, 0], [w, 0], [w, h], [0, h]])
        print("frame corners on the table (cm):", [tuple(round(float(v), 1) for v in c) for c in corners])
        print(f"saved {table.cal_path}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
