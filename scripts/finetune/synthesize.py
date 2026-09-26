"""Copy-paste composites from capture.py's cutouts: multi-object, occluded and hand-over-object
training images with exact boxes and no hand labelling (fast path, see README.md).

    python scripts/finetune/synthesize.py                   # 400 images, seed 0
    python scripts/finetune/synthesize.py --n 600 --seed 1
    python scripts/finetune/synthesize.py --p-distractor 0.6   # more scenes with unknown things in them

Backgrounds are the raw empty-table frames capture.py saved, not their median: pasted objects carry
the camera's grain, so the table must too or the grain itself would give them away. Objects get a
random position inside the area the real captures covered, rotation, mirror, +-10% scale and
brightness. Hands stay attached to the frame edge their arm came in from and often reach toward an
object, pasted last so they cover it. Later pastes cover earlier ones; every box is recomputed from
what is still visible, and objects less than 30% visible are dropped (unlabelled pixels a detector
can't be blamed for missing). Output is images/synth_<i>.jpg + labels/synth_<i>.txt: train.py always
trains on the "synth" trial, and cutouts from the held-out capture group are never pasted, so the
validation set stays real and unseen. Seeded: the same seed and inputs give the same images.

Distractors (capture.py --distractors: things that are none of the classes, cutout meta
"distractor": true) are pasted like objects, covering and covered like them, but never boxed: the
detector learns that an unknown thing on the table is not a phone. With probability --p-distractor an
image draws its items from objects and distractors together (some scenes end up distractors only);
otherwise from objects alone. With no distractor cutouts the images are exactly what they were before.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))      # run as a script or imported by tests
from bglabel import Box, save_sample  # noqa: E402
from capture import HOLDOUT, write_qa  # noqa: E402
from common import DEFAULT_DATA, class_names, trial_of, write_data_yaml  # noqa: E402

PREFIX = "synth"
NO_BOX = -1                          # Paste.cls of a distractor: it paints and covers, but gets no box


@dataclass
class Cut:
    name: str
    bgra: np.ndarray                 # alpha = the object's (or arm's) pixels
    box: Box                         # label box inside the cutout: all of it, or a hand's end of the arm
    origin: tuple[int, int]          # top-left in the frame it was cut from
    edge: Optional[str] = None       # hands: the frame edge the arm enters from
    stem: str = ""                   # the capture it came from
    distractor: bool = False         # none of the classes: pasted, never boxed


@dataclass
class Paste:
    cls: int
    bgra: np.ndarray                 # alpha 0-255 is coverage
    label: np.ndarray                # bool, same h x w: the pixels this paste's box is made from
    x: int                           # top-left in the canvas; may hang off any side
    y: int


def compose(bg: np.ndarray, pastes: Sequence[Paste], min_visible: float = 0.3) -> tuple[np.ndarray, list[tuple[int, Box]]]:
    """Alpha-blend pastes in order onto bg. Each box is the extent of the paste's label pixels still
    on top at the end; a paste with less than min_visible of its label pixels showing (covered, or off
    the frame) gets no box."""
    img = bg.astype(np.float32)
    H, W = bg.shape[:2]
    owner = np.full((H, W), -1, np.int32)
    clips = []
    for i, p in enumerate(pastes):
        h, w = p.bgra.shape[:2]
        x1, y1, x2, y2 = max(p.x, 0), max(p.y, 0), min(p.x + w, W), min(p.y + h, H)
        clips.append((x1, y1, x2, y2))
        if x2 <= x1 or y2 <= y1:
            continue
        sub = p.bgra[y1 - p.y:y2 - p.y, x1 - p.x:x2 - p.x]
        a = sub[:, :, 3:4].astype(np.float32) / 255
        img[y1:y2, x1:x2] = img[y1:y2, x1:x2] * (1 - a) + sub[:, :, :3] * a
        owner[y1:y2, x1:x2][sub[:, :, 3] >= 128] = i
    boxes = []
    for i, (p, (x1, y1, x2, y2)) in enumerate(zip(pastes, clips)):
        total = int(p.label.sum())
        if p.cls == NO_BOX or total == 0 or x2 <= x1 or y2 <= y1:
            continue
        vis = p.label[y1 - p.y:y2 - p.y, x1 - p.x:x2 - p.x] & (owner[y1:y2, x1:x2] == i)
        if vis.sum() < min_visible * total:
            continue
        ys, xs = np.nonzero(vis)
        boxes.append((p.cls, (x1 + int(xs.min()), y1 + int(ys.min()), x1 + int(xs.max()) + 1, y1 + int(ys.max()) + 1)))
    return np.clip(img, 0, 255).astype(np.uint8), boxes


def _label(bgra: np.ndarray, box: Box) -> np.ndarray:
    x1, y1, x2, y2 = box
    lab = np.zeros(bgra.shape[:2], bool)
    lab[y1:y2, x1:x2] = bgra[y1:y2, x1:x2, 3] >= 128
    return lab


def _gain(bgra: np.ndarray, g: float) -> np.ndarray:
    out = bgra.copy()
    out[:, :, :3] = np.clip(bgra[:, :, :3].astype(np.float32) * g, 0, 255).astype(np.uint8)
    return out


def _feather(bgra: np.ndarray) -> np.ndarray:
    """Soften the cut edge by a pixel so pastes don't show a hard seam."""
    out = bgra.copy()
    out[:, :, 3] = cv2.GaussianBlur(bgra[:, :, 3], (3, 3), 0)
    return out


def rotate(bgra: np.ndarray, label: np.ndarray, angle: float, scale: float = 1.0, mirror: bool = False):
    """Rotate (degrees) and scale about the centre into a canvas big enough for the whole result."""
    if mirror:
        bgra, label = bgra[:, ::-1], label[:, ::-1]
    h, w = bgra.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
    c, s = abs(m[0, 0]), abs(m[0, 1])
    nw, nh = int(math.ceil(h * s + w * c)), int(math.ceil(h * c + w * s))
    m[0, 2] += nw / 2 - w / 2
    m[1, 2] += nh / 2 - h / 2
    out = cv2.warpAffine(np.ascontiguousarray(bgra), m, (nw, nh), flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    lab = cv2.warpAffine(np.ascontiguousarray(label).astype(np.uint8), m, (nw, nh), flags=cv2.INTER_NEAREST) > 0
    return out, lab & (out[:, :, 3] >= 128)


def place_object(rng: np.random.Generator, cut: Cut, cls: int, region: Box) -> Paste:
    """A random pose inside region. A distractor's paste has cls NO_BOX and no label pixels."""
    bgra, lab = rotate(cut.bgra, _label(cut.bgra, cut.box), rng.uniform(0, 360), rng.uniform(0.9, 1.1),
                       bool(rng.random() < 0.5))
    h, w = bgra.shape[:2]
    rx1, ry1, rx2, ry2 = region
    x = int(rng.integers(rx1, max(rx1, rx2 - w) + 1))
    y = int(rng.integers(ry1, max(ry1, ry2 - h) + 1))
    if cut.distractor:
        cls, lab = NO_BOX, np.zeros_like(lab)
    return Paste(cls, _feather(_gain(bgra, rng.uniform(0.85, 1.15))), lab, x, y)


def place_hand(rng: np.random.Generator, cut: Cut, cls: int, frame: tuple[int, int],
               toward: Optional[tuple[float, float]] = None) -> Paste:
    """Keep the arm on its entry edge (mirrored to the opposite edge half the time), slide it along
    that edge (to line its hand up with `toward` if given) and sometimes pull it a little out of view."""
    W, H = frame
    if cut.edge is None:
        return place_object(rng, cut, cls, (0, 0, W, H))
    bgra, lab = cut.bgra, _label(cut.bgra, cut.box)
    h, w = bgra.shape[:2]
    x, y = cut.origin
    edge = cut.edge
    across = edge in ("top", "bottom")          # the arm lies along y; slide in x
    if rng.random() < 0.5:                      # mirror along the edge
        bgra, lab = (bgra[:, ::-1], lab[:, ::-1]) if across else (bgra[::-1], lab[::-1])
        x, y = (W - x - w, y) if across else (x, H - y - h)
    if rng.random() < 0.5:                      # to the opposite edge
        bgra, lab = (bgra[::-1], lab[::-1]) if across else (bgra[:, ::-1], lab[:, ::-1])
        x, y = (x, H - y - h) if across else (W - x - w, y)
        edge = {"top": "bottom", "bottom": "top", "left": "right", "right": "left"}[edge]
    ys, xs = np.nonzero(lab)
    hx, hy = x + xs.mean(), y + ys.mean()      # the hand's centre
    if across:
        x += int(toward[0] - hx) if toward else int(rng.uniform(-0.35, 0.35) * W)
        x = int(np.clip(x, -w // 3, W - 2 * w // 3))
    else:
        y += int(toward[1] - hy) if toward else int(rng.uniform(-0.35, 0.35) * H)
        y = int(np.clip(y, -h // 3, H - 2 * h // 3))
    out = int(rng.uniform(0, 0.15) * (h if across else w))
    x += {"left": -out, "right": out}.get(edge, 0)
    y += {"top": -out, "bottom": out}.get(edge, 0)
    return Paste(cls, _feather(_gain(np.ascontiguousarray(bgra), rng.uniform(0.85, 1.15))),
                 np.ascontiguousarray(lab), x, y)


def make_image(rng: np.random.Generator, backgrounds: Sequence[np.ndarray], cuts: dict[str, list[Cut]],
               names: list[str], region: Box, max_objects: int = 6, p_hand: float = 0.6,
               min_visible: float = 0.3, p_distractor: float = 0.5) -> tuple[np.ndarray, list[tuple[int, Box]]]:
    """One composite. max_objects bounds the pasted items (objects and distractors together)."""
    bg = backgrounds[int(rng.integers(len(backgrounds)))]
    bg = np.clip(bg.astype(np.float32) * rng.uniform(0.92, 1.08), 0, 255).astype(np.uint8)
    H, W = bg.shape[:2]
    dists = sorted(n for n, cs in cuts.items() if any(c.distractor for c in cs))
    objs = sorted(n for n in cuts if n != "hand" and n not in dists)
    if dists and (not objs or rng.random() < p_distractor):   # no draw without distractors: old images stay
        objs = sorted(objs + dists)
    k = int(rng.integers(1, min(max_objects, len(objs)) + 1)) if objs else 0
    pastes = []
    for name in rng.choice(objs, k, replace=False) if k else []:
        group = cuts[name]
        cls = NO_BOX if name in dists else names.index(name)
        pastes.append(place_object(rng, group[int(rng.integers(len(group)))], cls, region))
    if cuts.get("hand") and rng.random() < p_hand:
        for _ in range(2 if rng.random() < 0.25 else 1):
            toward = None
            if pastes and rng.random() < 0.6:
                t = pastes[int(rng.integers(len(pastes)))]
                toward = (t.x + t.bgra.shape[1] / 2, t.y + t.bgra.shape[0] / 2)
            hand = cuts["hand"][int(rng.integers(len(cuts["hand"])))]
            pastes.append(place_hand(rng, hand, names.index("hand"), (W, H), toward))
    return compose(bg, pastes, min_visible)


def load_cuts(data: Path, holdout: Optional[str] = HOLDOUT) -> dict[str, list[Cut]]:
    """Cutouts by name (classes and distractors), leaving out the held-out capture group so validation
    stays unseen. Cutouts without a "distractor" field (older captures) are class objects."""
    out: dict[str, list[Cut]] = {}
    for meta_p in sorted((Path(data) / "cutouts").glob("*.json")):
        meta = json.loads(meta_p.read_text())
        if holdout and trial_of(meta["stem"]) == holdout:
            continue
        bgra = cv2.imread(str(meta_p.with_suffix(".png")), cv2.IMREAD_UNCHANGED)
        if bgra is None or bgra.ndim != 3 or bgra.shape[2] != 4:
            continue
        out.setdefault(meta["name"], []).append(
            Cut(meta["name"], bgra, tuple(meta["box"]), tuple(meta["origin"]), meta.get("edge"), meta["stem"],
                bool(meta.get("distractor", False))))
    return out


def placement_region(cuts: dict[str, list[Cut]], frame: tuple[int, int], pad: float = 0.05) -> Box:
    """Where real objects were placed (the table in view), padded a little; the whole frame if unknown."""
    W, H = frame
    boxes = [(c.origin[0], c.origin[1], c.origin[0] + c.bgra.shape[1], c.origin[1] + c.bgra.shape[0])
             for n, cs in cuts.items() if n != "hand" for c in cs]
    if not boxes:
        return (0, 0, W, H)
    x1, y1 = min(b[0] for b in boxes), min(b[1] for b in boxes)
    x2, y2 = max(b[2] for b in boxes), max(b[3] for b in boxes)
    return (max(0, int(x1 - pad * W)), max(0, int(y1 - pad * H)), min(W, int(x2 + pad * W)), min(H, int(y2 + pad * H)))


def synthesize(data: Path, names: list[str], n: int = 400, seed: int = 0, holdout: Optional[str] = HOLDOUT,
               **kw) -> int:
    data = Path(data)
    backgrounds = [cv2.imread(str(p)) for p in sorted((data / "backgrounds").glob("*.jpg"))]
    backgrounds = [b for b in backgrounds if b is not None]
    if not backgrounds:
        raise FileNotFoundError(f"no empty-table frames in {data / 'backgrounds'}: run capture.py first")
    cuts = load_cuts(data, holdout)
    unknown = {n for n, cs in cuts.items() if not any(c.distractor for c in cs)} - set(names)
    if unknown:
        raise ValueError(f"cutouts for {sorted(unknown)}, which are not classes {names} (nor distractors)")
    clash = {n for n, cs in cuts.items() if any(c.distractor for c in cs)} & set(names)
    if clash:
        raise ValueError(f"distractor cutouts named like classes {sorted(clash)}: recapture them under another name")
    if not cuts:
        raise FileNotFoundError(f"no cutouts in {data / 'cutouts'}: run capture.py first")
    H, W = backgrounds[0].shape[:2]
    region = placement_region(cuts, (W, H))
    for sub, ext in (("images", "jpg"), ("labels", "txt")):
        for p in (data / sub).glob(f"{PREFIX}_*.{ext}"):
            p.unlink()
    for i in range(n):
        img, boxes = make_image(np.random.default_rng([seed, i]), backgrounds, cuts, names, region, **kw)
        save_sample(data, f"{PREFIX}_{i:05d}", img, boxes, quality=90)
    return n


def main(argv=None) -> int:
    from core.config import load_config
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=str(DEFAULT_DATA))
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-objects", type=int, default=6)
    ap.add_argument("--p-hand", type=float, default=0.6, help="share of images with a hand (a quarter of those get two)")
    ap.add_argument("--p-distractor", type=float, default=0.5,
                    help="share of images whose items are drawn from objects and distractors together "
                         "(only matters when capture.py --distractors saved some)")
    ap.add_argument("--holdout", default=HOLDOUT, help="capture group never pasted ('' pastes every cutout)")
    a = ap.parse_args(argv)
    names = class_names(load_config())
    data = Path(a.data)
    n = synthesize(data, names, a.n, a.seed, a.holdout or None, max_objects=a.max_objects, p_hand=a.p_hand,
                   p_distractor=a.p_distractor)
    write_data_yaml(data, names)
    print(f"wrote {n} composites to {data / 'images'}/{PREFIX}_*.jpg; "
          f"sample sheet: {write_qa(data, names, PREFIX, 'qa_synth.jpg', limit=48)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
