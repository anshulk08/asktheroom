"""Model-free labels by background subtraction, for guided capture (capture.py) under the fixed camera.

During capture one object at a time lies on an otherwise empty table, so the region that differs from
a median of empty-table frames is that object: no hand labelling, and no dependence on YOLO-World
(which scored the wallet 0.00 and the glasses under 0.2 from overhead). Pure numpy/OpenCV, so it runs
in the Jetson host venv (no torch) and in tests. Boxes are (x1, y1, x2, y2) pixels with exclusive
ends, like slices.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import cv2
import numpy as np

Box = tuple[int, int, int, int]


@dataclass
class Params:
    blur: int = 5                  # Gaussian kernel before differencing: the camera is grainy in dim light
    min_thresh: float = 18.0       # colour difference (0-255, max over channels) that counts as changed ...
    noise_mult: float = 1.5        # ... raised to this times the empty table's own noise (p99.5)
    open_px: int = 3               # removes speckle
    close_px: int = 9              # joins an object's own parts (glasses rims, key teeth)
    merge_px: int = 25             # pieces this close are one object (keys on a ring)
    border_px: int = 12            # objects must stay this far inside the frame; reaching it = arm in view
    min_blob_frac: float = 0.0002  # pieces smaller than this share of the frame are ignored
    min_area_frac: float = 0.0008  # smallest plausible object (a pill bottle is ~0.003)
    max_area_frac: float = 0.25    # larger means the light changed or the camera moved
    split_frac: float = 0.2        # a second piece this big (vs the object) means two things are there
    hand_min_frac: float = 0.004   # hand session: smaller changes are not a hand
    hand_len_px: int = 260         # an arm entering from an edge is cut to its last this-many px (the hand)
    roi_px: Optional[tuple] = None  # (x1, y1, x2, y2) the tabletop in camera px; changes outside it (the floor,
                                    # a chair, the person capturing) are ignored and its edges count as the frame's


@dataclass
class Label:
    ok: bool
    reason: str = ""
    box: Optional[Box] = None             # what the detector should learn
    mask: Optional[np.ndarray] = None     # bool, full frame: the pixels to cut out
    edge: Optional[str] = None            # hands: the frame edge the arm comes in from


def _kernel(px: int) -> np.ndarray:
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (max(1, px), max(1, px)))


def _bbox(mask: np.ndarray) -> Box:
    x, y, w, h = cv2.boundingRect(mask.astype(np.uint8))
    return (x, y, x + w, y + h)


class Background:
    """Per-pixel median of empty-table frames, with a change threshold set from those frames' own noise,
    so a grainier camera or dimmer room raises the threshold instead of labelling noise."""

    def __init__(self, frames: Sequence[np.ndarray], p: Optional[Params] = None):
        if not len(frames):
            raise ValueError("need at least one empty-table frame")
        self.p = p or Params()
        self.img = np.median(np.stack(frames), axis=0).round().astype(np.uint8)
        self._ref = self._smooth(self.img)
        self.noise = float(np.median([np.percentile(self.diff(f), 99.5) for f in frames[:10]]))
        self.thresh = max(self.p.min_thresh, self.p.noise_mult * self.noise)

    def _smooth(self, img: np.ndarray) -> np.ndarray:
        k = self.p.blur | 1
        return cv2.GaussianBlur(img, (k, k), 0)

    def diff(self, img: np.ndarray) -> np.ndarray:
        """Largest per-channel difference from the background: colour changes count, not just brightness."""
        return cv2.absdiff(self._smooth(img), self._ref).max(axis=2)

    def changed(self, img: np.ndarray) -> np.ndarray:
        m = (self.diff(img) > self.thresh).astype(np.uint8)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, _kernel(self.p.open_px))
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, _kernel(self.p.close_px))
        x1, y1, x2, y2 = region(self.p, *m.shape[:2])
        out = np.zeros_like(m)
        out[y1:y2, x1:x2] = m[y1:y2, x1:x2]
        return out


def region(p: Params, h: int, w: int) -> tuple[int, int, int, int]:
    """The part of the frame labels come from: the tabletop (roi_px) or the whole frame."""
    if not p.roi_px:
        return 0, 0, w, h
    x1, y1, x2, y2 = (int(v) for v in p.roi_px)
    return max(0, x1), max(0, y1), min(w, x2), min(h, y2)


def _groups(mask: np.ndarray, min_px: int, merge_px: int) -> list[np.ndarray]:
    """Pieces of at least min_px, grouped when within merge_px of each other; largest group first."""
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    big = [i for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= min_px]
    if not big:
        return []
    keep = np.isin(lab, big).astype(np.uint8)
    _, glab = cv2.connectedComponents(cv2.dilate(keep, _kernel(merge_px)), connectivity=8)
    glab = np.where(keep > 0, glab, 0)
    ids, counts = np.unique(glab[glab > 0], return_counts=True)
    return [glab == i for i in ids[np.argsort(-counts)]]


def _fill(mask: np.ndarray) -> np.ndarray:
    """Fill holes (a glasses lens, the gap in a key ring) so the cutout is one solid piece."""
    m = mask.astype(np.uint8)
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(m, contours, -1, 1, thickness=cv2.FILLED)
    return m > 0


def label_object(bg: Background, img: np.ndarray) -> Label:
    """The one new thing on the table, or a reason to retake the shot."""
    p = bg.p
    h, w = img.shape[:2]
    m = bg.changed(img)
    b = p.border_px
    rx1, ry1, rx2, ry2 = region(p, h, w)
    inner = np.zeros_like(m)
    inner[ry1 + b:ry2 - b, rx1 + b:rx2 - b] = 1
    groups = _groups(m & inner, int(p.min_blob_frac * h * w), p.merge_px)
    if not groups:
        return Label(False, "nothing new on the table (is the object there? is it darker/lighter than the table?)")
    main = _fill(groups[0])
    area = int(main.sum())
    others = [int(g.sum()) for g in groups[1:] if g.sum() > p.split_frac * area]
    if others:
        return Label(False, f"{1 + len(others)} separate things changed: leave only this object on the table")
    if area < p.min_area_frac * h * w:
        return Label(False, f"change too small ({area} px): is the object there?")
    if area > p.max_area_frac * h * w:
        return Label(False, f"change too large ({100 * area / (h * w):.0f}% of frame): light changed or camera moved?")
    box = _bbox(main)
    if box[0] <= rx1 + b or box[1] <= ry1 + b or box[2] >= rx2 - b or box[3] >= ry2 - b:
        return Label(False, "touches the frame edge: move the object inward and keep hands/arms out of view")
    return Label(True, box=box, mask=main)


def _entry_edge(mask: np.ndarray, band: int = 3) -> Optional[str]:
    """The frame edge the blob touches most (an arm comes in from there), or None if it floats inside."""
    touch = {"left": mask[:, :band].sum(), "right": mask[:, -band:].sum(),
             "top": mask[:band, :].sum(), "bottom": mask[-band:, :].sum()}
    edge = max(touch, key=touch.get)
    return edge if touch[edge] > 0 else None


def hand_part(mask: np.ndarray, edge: Optional[str], length: int) -> np.ndarray:
    """The last `length` px of an arm, measured back from its tip away from the entry edge."""
    if edge is None:
        return mask
    ys, xs = np.nonzero(mask)
    rows, cols = slice(None), slice(None)
    if edge == "bottom":
        rows = slice(ys.min(), ys.min() + length)
    elif edge == "top":
        rows = slice(max(0, ys.max() + 1 - length), ys.max() + 1)
    elif edge == "right":
        cols = slice(xs.min(), xs.min() + length)
    else:
        cols = slice(max(0, xs.max() + 1 - length), xs.max() + 1)
    out = np.zeros_like(mask)
    out[rows, cols] = mask[rows, cols]
    return out


def label_hands(bg: Background, img: np.ndarray, max_hands: int = 2) -> list[Label]:
    """Hands over the empty table. The mask keeps the whole arm (it occludes things when pasted); the
    box keeps only the hand end, which is what the detector's 'hand' class should mean."""
    p = bg.p
    h, w = img.shape[:2]
    out = []
    x1, y1, x2, y2 = region(p, h, w)
    for g in _groups(bg.changed(img), int(p.hand_min_frac * h * w), p.close_px)[:max_hands]:
        g = _fill(g)
        edge = _entry_edge(g[y1:y2, x1:x2])          # the arm comes in over the table region's edge
        out.append(Label(True, box=_bbox(hand_part(g, edge, p.hand_len_px)), mask=g, edge=edge))
    return out


class Stability:
    """True once the view has stopped changing for hold_s: the hand has left and the object has settled.
    Measured as the share of pixels that changed between consecutive downscaled frames, so grain (which
    averages out at low resolution) doesn't count but a hand anywhere in view does."""

    def __init__(self, hold_s: float = 0.7, pixel_delta: int = 10, max_changed: float = 0.002,
                 size: tuple[int, int] = (160, 90)):
        self.hold_s, self.pixel_delta, self.max_changed, self.size = hold_s, pixel_delta, max_changed, size
        self._prev: Optional[np.ndarray] = None
        self._since: Optional[float] = None

    def update(self, img: np.ndarray, t: float) -> bool:
        g = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), self.size, interpolation=cv2.INTER_AREA)
        if self._prev is None or (cv2.absdiff(g, self._prev) > self.pixel_delta).mean() > self.max_changed:
            self._since = t
        self._prev = g
        return t - self._since >= self.hold_s


def yolo_line(cls_id: int, box: Box, w: int, h: int) -> str:
    x1, y1, x2, y2 = box
    return f"{cls_id} {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} {(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}"


def read_yolo(path: Path, w: int, h: int) -> list[tuple[int, Box]]:
    out = []
    for line in Path(path).read_text().split("\n"):
        if line.strip():
            c, cx, cy, bw, bh = line.split()
            cx, cy, bw, bh = float(cx) * w, float(cy) * h, float(bw) * w, float(bh) * h
            out.append((int(c), (round(cx - bw / 2), round(cy - bh / 2), round(cx + bw / 2), round(cy + bh / 2))))
    return out


def cutout(img: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
    """BGRA crop of the mask's bounding box (alpha = mask) and the crop's top-left in the frame."""
    x1, y1, x2, y2 = _bbox(mask)
    bgra = cv2.cvtColor(img[y1:y2, x1:x2], cv2.COLOR_BGR2BGRA)
    bgra[:, :, 3] = mask[y1:y2, x1:x2].astype(np.uint8) * 255
    return bgra, (x1, y1)


def save_sample(data: Path, stem: str, img: np.ndarray, labels: Sequence[tuple[int, Box]],
                quality: int = 92) -> Path:
    """images/<stem>.jpg + labels/<stem>.txt, the flat layout the other fine-tuning scripts use.
    An empty label file marks a labelled frame with nothing in it (a negative)."""
    data = Path(data)
    (data / "images").mkdir(parents=True, exist_ok=True)
    (data / "labels").mkdir(parents=True, exist_ok=True)
    h, w = img.shape[:2]
    p = data / "images" / f"{stem}.jpg"
    cv2.imwrite(str(p), img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    (data / "labels" / f"{stem}.txt").write_text("".join(yolo_line(c, b, w, h) + "\n" for c, b in labels))
    return p


def contact_sheet(entries: Sequence[tuple[np.ndarray, Sequence[tuple[str, Box]], str]], cols: int = 6,
                  thumb_w: int = 320) -> np.ndarray:
    """Thumbnails with their boxes drawn, for eyeballing labels. entries: (image, [(name, box)], caption)."""
    if not entries:
        return np.zeros((10, 10, 3), np.uint8)
    h0, w0 = entries[0][0].shape[:2]
    s = thumb_w / w0
    th = int(round(h0 * s))
    rows = (len(entries) + cols - 1) // cols
    sheet = np.zeros((rows * th, cols * thumb_w, 3), np.uint8)
    for k, (img, boxes, caption) in enumerate(entries):
        t = cv2.resize(img, (thumb_w, th), interpolation=cv2.INTER_AREA)
        for name, (x1, y1, x2, y2) in boxes:
            p1, p2 = (int(x1 * s), int(y1 * s)), (int(x2 * s), int(y2 * s))
            cv2.rectangle(t, p1, p2, (0, 255, 0), 2)
            cv2.putText(t, name, (p1[0], max(12, p1[1] - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
        cv2.putText(t, caption, (4, th - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        r, c = divmod(k, cols)
        sheet[r * th:(r + 1) * th, c * thumb_w:(c + 1) * thumb_w] = t
    return sheet
