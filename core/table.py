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

A calibration also records the camera view it was made at ("view": capture size, zoom, and with room memory
the table_view_rect as fractions of the capture, so 1080p and 1440p of the same crop are one view) and a
grey thumbnail of the calibration frame (table_cal_view.png, THUMB_W wide, beside table_cal.json) with the
box around the tag ("tag_patch"). demo_check compares them with the config and the camera now; the app
never reads them. A table_cal.json from before has neither and loads as always.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

log = logging.getLogger(__name__)

TABLE_IDS = (0, 1, 2, 3)
SIDECAR = "table_cal_view.png"                   # grey thumbnail of the calibration frame, beside table_cal.json
THUMB_W = 320
VIEW_TOL = 0.01                                  # view rects this close (fraction of the capture) are one view
PATCH_BOX = 3.5                                  # tag_patch box: the tag quad scaled this much about its centre
PATCH_MASK = 2.0                                 # ... minus this (the tag and its sheet): the table around the tag


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


def config_view(cfg: dict, frame_px=None) -> dict:
    """The camera view table frames come from, as a calibration records it. Room memory on: the capture
    size, zoom and table_view_rect as fractions of the capture (main.open_frames' default when unset), and
    the table view's size (frame_size_px). Off: the frame is the camera's (frame_px, else frame_size_px),
    with no rect."""
    fp = [int(v) for v in (frame_px or cfg.get("frame_size_px") or (1280, 720))]
    rm = cfg.get("room_memory") or {}
    if not rm.get("enabled"):
        return {"capture_size": fp, "zoom": None, "rect": None, "frame_px": fp}
    from core.room_types import RoomConfig
    from core.room_view import default_rect
    rc = RoomConfig.from_dict(rm)
    cw, ch = rc.capture_size
    rect = rc.table_view_rect or default_rect(rc.capture_size, rc.zoom, rc.ref_zoom, fp)
    frac = [round(rect[0] / cw, 5), round(rect[1] / ch, 5), round(rect[2] / cw, 5), round(rect[3] / ch, 5)]
    return {"capture_size": [int(cw), int(ch)], "zoom": int(rc.zoom), "rect": frac, "frame_px": fp}


def same_view(a: dict, b: dict, tol: float = VIEW_TOL) -> tuple[bool, str]:
    """(same, what differs) for two config_view()s. Capture px don't matter, only its aspect: the table
    view is resized to frame_px anyway. frame_px must match exactly (H is in those px)."""
    why = []
    if a.get("zoom") != b.get("zoom"):
        why.append(f"zoom {a.get('zoom')} vs {b.get('zoom')}")
    ra, rb = a.get("rect"), b.get("rect")
    if (ra is None) != (rb is None):
        why.append(f"table view {'rect' if ra else 'whole frame'} vs {'rect' if rb else 'whole frame'}")
    elif ra is not None and max(abs(float(x) - float(y)) for x, y in zip(ra, rb)) > tol:
        why.append(f"table_view_rect {[round(float(v), 3) for v in ra]} vs {[round(float(v), 3) for v in rb]} "
                   f"of the frame")
    (aw, ah), (bw, bh) = a.get("capture_size") or (1, 1), b.get("capture_size") or (1, 1)
    if abs((aw / ah) / (bw / bh) - 1) > tol:
        why.append(f"capture {aw}x{ah} vs {bw}x{bh}")
    if list(a.get("frame_px") or []) != list(b.get("frame_px") or []):
        why.append(f"frame {a.get('frame_px')} vs {b.get('frame_px')} px")
    return not why, "; ".join(why)


def thumb(img: np.ndarray, width: int = THUMB_W) -> np.ndarray:
    """Grey, `width` px wide (INTER_AREA): what table_cal_view.png holds."""
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    h = max(1, round(g.shape[0] * width / g.shape[1]))
    return cv2.resize(g, (width, h), interpolation=cv2.INTER_AREA)


def _scaled(quad: np.ndarray, k: float) -> np.ndarray:
    c = quad.mean(axis=0)
    return c + (quad - c) * k


def tag_patch(quad, size_px) -> dict:
    """tag_patch for a tag quad (px): the box around it (PATCH_BOX, clipped to the frame) and the part
    not to compare (PATCH_MASK: the tag and its paper, picked up after calibrating)."""
    q = np.asarray(quad, dtype=np.float64).reshape(-1, 2)
    w, h = int(size_px[0]), int(size_px[1])
    big = _scaled(q, PATCH_BOX)
    box = [int(max(0, np.floor(big[:, 0].min()))), int(max(0, np.floor(big[:, 1].min()))),
           int(min(w, np.ceil(big[:, 0].max()))), int(min(h, np.ceil(big[:, 1].max())))]
    return {"box": box, "mask": np.round(_scaled(q, PATCH_MASK), 1).tolist()}


def _write_atomic(path: Path, data: bytes) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


class Table:
    def __init__(self, cfg: dict, cal_path: Optional[str] = None):
        t = cfg.get("table") or {}
        self.size_cm = tuple(t.get("size_cm", (90, 60)))
        self.markers_cm = {int(k): tuple(v) for k, v in (t.get("markers") or {}).items()}
        self.cal_path = cal_path or (cfg.get("paths") or {}).get("table_cal", "table_cal.json")
        self._cal: tuple = (None, None)           # (H px -> cm, Hinv cm -> px), swapped as one (_set)
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
        self._view_cfg = {k: cfg.get(k) for k in ("frame_size_px", "room_memory")}   # config_view at save
        self.load()

    @property
    def ok(self) -> bool:
        return self.H is not None

    @property
    def H(self) -> Optional[np.ndarray]:          # px -> cm
        return self._cal[0]

    @H.setter
    def H(self, H: Optional[np.ndarray]) -> None:
        if H is None:
            self._cal = (None, None)
        else:
            self._set(H)

    @property
    def Hinv(self) -> Optional[np.ndarray]:       # cm -> px
        return self._cal[1]

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
        lo, hi = px.min(axis=0), px.max(axis=0)       # the table between the markers; they stay: no mask
        patch = {"box": [int(lo[0]), int(lo[1]), int(np.ceil(hi[0])), int(np.ceil(hi[1]))], "mask": None}
        self.save({i: found[i].tolist() for i in TABLE_IDS}, frame_img, patch)
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
        self.save({"tag": px.tolist()}, img, tag_patch(px, (w, h)))
        log.info("table calibrated from tag %d: tracked area %.0f x %.0f cm", self.tag_id, *self.size_cm)
        return True

    def _set(self, H: np.ndarray) -> None:
        """One assignment, so another thread (perception, the laser) never pairs a new H with the old
        Hinv while a spoken 'recalibrate' refits on its own thread."""
        H = np.asarray(H, dtype=np.float64)
        self._cal = (H, np.linalg.inv(H))

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

    def save(self, markers_px: dict, img: Optional[np.ndarray] = None, patch: Optional[dict] = None) -> None:
        """table_cal.json; with the calibration frame also its view and table_cal_view.png, whose sha1 goes
        in tag_patch (a thumbnail left from another fit is told apart). Both written atomically."""
        d = {"H": self.H.tolist(), "markers_px": markers_px, "size_cm": list(self.size_cm), "t": time.time()}
        p = Path(self.cal_path)
        if img is not None:
            ok, png = cv2.imencode(".png", thumb(img))
            if ok:
                _write_atomic(p.with_name(SIDECAR), png.tobytes())
                d["view"] = config_view(self._view_cfg, img.shape[1::-1])
                d["tag_patch"] = dict(patch or {}, file=SIDECAR, sha1=hashlib.sha1(png.tobytes()).hexdigest())
        _write_atomic(p, json.dumps(d, indent=1).encode())

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
    ap.add_argument("--outline-full", nargs="+", metavar="X,Y",
                    help="room memory: tabletop corners in full camera frame px (or --image's, e.g. /full.jpg)")
    a = ap.parse_args(argv)
    cfg = load_config()
    table = Table(cfg)
    if a.outline is not None or a.outline_px or a.outline_full:     # core/table_area.py: where objects may appear
        from core.table_area import outline_main
        return outline_main(table, cfg, a.outline, a.outline_px, a.image, a.outline_full)
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
