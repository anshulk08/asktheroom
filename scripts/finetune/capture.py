"""Guided capture with automatic labels: the fast path, no hand labelling (see README.md).

Run on the Jetson host venv (cv2 + numpy only) with the app stopped, since it holds the camera:

    python scripts/finetune/capture.py                          # empty table, 8 objects x 8 poses, hands
    python scripts/finetune/capture.py --only glasses keys      # redo some objects (same empty table)
    python scripts/finetune/capture.py --qa                     # rebuild the contact sheet only

Each object lies alone on the empty table, so bglabel.py's difference against the empty table is its
box. Writes into --data (default data/finetune), the layout the other scripts use:

    images/cap<g>_<object>-<k>.jpg, labels/...txt   g = k % 4 is the "trial" train.py splits on;
                                                    cap3 is held out (train.py --val-trials cap3)
    images/cap<g>_empty-<k>.jpg, empty labels       negatives
    cutouts/<stem>.png (BGRA) + <stem>.json         for synthesize.py
    backgrounds/empty-<k>.jpg                       raw empty-table frames for synthesize.py
    qa_capture.jpg                                  every capture with its box: look at it
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))      # run as a script or imported by tests
from bglabel import (Background, Params, Stability, contact_sheet, cutout, label_hands,  # noqa: E402
                     label_object, read_yolo, save_sample)
from common import DEFAULT_DATA, class_names, write_data_yaml  # noqa: E402

GROUPS = 4
HOLDOUT = f"cap{GROUPS - 1}"
POSE_HINTS = ["in the middle", "near a corner, not touching the edge", "rotated about 90 degrees",
              "flipped over or on its other side", "near another edge", "rotated about 45 degrees",
              "in another corner", "anywhere new, any angle"]


class Quit(Exception):
    pass


def stem(name: str, k: int) -> str:
    """cap<k % 4>_<name>-<k>. No '_' after the group, so common.trial_of() gives the group."""
    return f"cap{k % GROUPS}_{name.replace('_', '-')}-{k:02d}"


def _clear(data: Path, name: str) -> None:
    pat = f"cap*_{name.replace('_', '-')}-*"
    for sub, ext in (("images", "jpg"), ("labels", "txt"), ("cutouts", "png"), ("cutouts", "json")):
        for p in (data / sub).glob(f"{pat}.{ext}"):
            p.unlink()


def save_cutouts(data: Path, st: str, img: np.ndarray, name: str, labels) -> None:
    """One BGRA cutout per label, with its label box relative to the cutout (a hand's box is only the
    hand end of the arm) and the edge an arm comes in from, which synthesize.py keeps it attached to."""
    (data / "cutouts").mkdir(parents=True, exist_ok=True)
    h, w = img.shape[:2]
    for i, lab in enumerate(labels):
        s = st if len(labels) == 1 else f"{st}-{i}"
        bgra, (x0, y0) = cutout(img, lab.mask)
        cv2.imwrite(str(data / "cutouts" / f"{s}.png"), bgra)
        x1, y1, x2, y2 = lab.box
        meta = {"name": name, "stem": st, "origin": [x0, y0], "frame": [w, h], "edge": lab.edge,
                "box": [x1 - x0, y1 - y0, x2 - x0, y2 - y0]}
        (data / "cutouts" / f"{s}.json").write_text(json.dumps(meta))


class Capture:
    """The guided session. source: FrameBuffer-like (wait_new); ask/say: the terminal, or a test script."""

    def __init__(self, source, data: Path, names: list[str], *, poses: int = 8, params: Optional[Params] = None,
                 ask: Callable[[str], str] = input, say: Callable[[str], None] = print,
                 settle_timeout_s: float = 15.0, display: Optional[dict] = None):
        self.src, self.data, self.names, self.poses = source, Path(data), names, poses
        self.p = params or Params()
        self.ask, self.say = ask, say
        self.settle_timeout_s = settle_timeout_s
        self.display = display or {}
        self.bg: Optional[Background] = None
        self._idx = 0

    def _next(self):
        f = self.src.wait_new(self._idx, 3.0)
        if f is None:
            raise RuntimeError("the camera stopped delivering frames")
        self._idx = f.idx
        return f

    def _prompt(self, text: str) -> str:
        r = self.ask(text).strip().lower()
        if r == "q":
            raise Quit
        return r

    def background(self, n: int = 30, keep: int = 6) -> Background:
        self._prompt("Clear the table (markers can stay), hands and arms out of view, then press Enter: ")
        imgs = [self._next().img for _ in range(n)]
        self.bg = Background(imgs, self.p)
        (self.data / "backgrounds").mkdir(parents=True, exist_ok=True)
        for p in (self.data / "backgrounds").glob("empty-*.jpg"):
            p.unlink()
        for k, i in enumerate(np.linspace(0, n - 1, keep).astype(int)):
            cv2.imwrite(str(self.data / "backgrounds" / f"empty-{k:02d}.jpg"), imgs[i], [cv2.IMWRITE_JPEG_QUALITY, 95])
        for k in range(GROUPS):
            save_sample(self.data, f"cap{k}_empty-{k:02d}", imgs[k * (n // GROUPS)], [])
        self.say(f"  background from {n} frames: noise {self.bg.noise:.1f}, change threshold {self.bg.thresh:.1f}")
        return self.bg

    def settle(self, n: int = 5) -> Optional[list[np.ndarray]]:
        """The next n frames once the view has been still for a moment, or None if it never settles."""
        still = Stability()
        f = self._next()
        t0 = f.t
        while not still.update(f.img, f.t):
            if f.t - t0 > self.settle_timeout_s:
                return None
            f = self._next()
        return [self._next().img for _ in range(n)]

    def capture_object(self, cls_id: int, name: str) -> int:
        _clear(self.data, name)
        spoken = self.display.get(name, name.replace("_", " "))
        k = 0
        while k < self.poses:
            r = self._prompt(f"[{name} {k + 1}/{self.poses}] Place the {spoken} alone on the table, "
                             f"{POSE_HINTS[k % len(POSE_HINTS)]}; hands away; Enter (s = skip object, q = quit): ")
            if r == "s":
                break
            shots = self.settle()
            if shots is None:
                self.say("  retake: the view never settled (something moving? flickering light?)")
                continue
            lab = label_object(self.bg, np.median(np.stack(shots), axis=0).astype(np.uint8))
            if not lab.ok:
                self.say(f"  retake: {lab.reason}")
                continue
            img, st = shots[-1], stem(name, k)       # one real frame keeps real grain; the mask came from 5
            save_sample(self.data, st, img, [(cls_id, lab.box)])
            save_cutouts(self.data, st, img, name, [lab])
            x1, y1, x2, y2 = lab.box
            self.say(f"  ok: {st}  box {x2 - x1}x{y2 - y1} px at ({x1}, {y1})")
            k += 1
        return k

    def capture_hands(self, cls_id: int, seconds: float = 40.0, every_s: float = 0.3, max_frames: int = 120) -> int:
        _clear(self.data, "hand")
        self._prompt(f"Hands: for {seconds:.0f} s, move one hand, then both, slowly over the EMPTY table from "
                     f"different sides (open, fist, pointing, touching the table). Enter to start: ")
        f = self._next()
        t0, last, j = f.t, -1e9, 0
        while f.t - t0 < seconds and j < max_frames:
            if f.t - last >= every_s:
                last, labs = f.t, label_hands(self.bg, f.img)
                if labs:
                    st = stem("hand", j)
                    save_sample(self.data, st, f.img, [(cls_id, lab.box) for lab in labs])
                    save_cutouts(self.data, st, f.img, "hand", labs)
                    j += 1
            f = self._next()
        self.say(f"  {j} hand frames")
        return j

    def run(self, only: Optional[list[str]] = None, hands: bool = True, hand_seconds: float = 40.0) -> dict:
        got = {}
        try:
            self.background()
            for i, name in enumerate(self.names):
                if name == "hand" or (only and name not in only):
                    continue
                got[name] = self.capture_object(i, name)
            if hands and (not only or "hand" in only):
                got["hand"] = self.capture_hands(self.names.index("hand"), hand_seconds)
        except Quit:
            self.say("stopped; everything captured so far is saved")
        write_data_yaml(self.data, self.names)
        qa = write_qa(self.data, self.names)
        self.say(f"captured {got}; contact sheet: {qa}")
        return got


def write_qa(data: Path, names: list[str], prefix: str = "cap", out: str = "qa_capture.jpg",
             limit: Optional[int] = None) -> Path:
    """Contact sheet of the labelled images whose names start with prefix, boxes drawn."""
    data = Path(data)
    entries = []
    for p in sorted((data / "images").glob(f"{prefix}*.jpg"))[:limit]:
        lp = data / "labels" / f"{p.stem}.txt"
        img = cv2.imread(str(p))
        if img is None or not lp.exists():
            continue
        h, w = img.shape[:2]
        entries.append((img, [(names[c], b) for c, b in read_yolo(lp, w, h)], p.stem))
    path = data / out
    cv2.imwrite(str(path), contact_sheet(entries, cols=8, thumb_w=240), [cv2.IMWRITE_JPEG_QUALITY, 85])
    return path


def main(argv=None) -> int:
    from core.config import load_config
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=str(DEFAULT_DATA))
    ap.add_argument("--device", default="0", help="camera index or /dev/videoN")
    ap.add_argument("--poses", type=int, default=8)
    ap.add_argument("--only", nargs="*", help="object names (and/or 'hand') to (re)capture")
    ap.add_argument("--no-hands", action="store_true")
    ap.add_argument("--roi", help="x1,y1,x2,y2: the tabletop in camera px; changes outside it (floor, chair, "
                                  "the person capturing) are ignored")
    ap.add_argument("--hand-seconds", type=float, default=40.0)
    ap.add_argument("--hand-len", type=int, default=Params.hand_len_px,
                    help="px of an arm kept as the hand box (about 18 cm of arm at this camera height)")
    ap.add_argument("--qa", action="store_true", help="only rebuild qa_capture.jpg")
    a = ap.parse_args(argv)
    cfg = load_config()
    names = class_names(cfg)
    if a.qa:
        print(write_qa(Path(a.data), names))
        return 0
    bad = set(a.only or []) - set(names)
    if bad:
        print(f"unknown names {sorted(bad)}; choose from {names}", file=sys.stderr)
        return 2
    from core.capture import FrameBuffer
    fb = FrameBuffer(int(a.device) if a.device.isdigit() else a.device)
    try:
        cap = Capture(fb, Path(a.data), names, poses=a.poses, params=Params(hand_len_px=a.hand_len,
                                  roi_px=tuple(int(v) for v in a.roi.split(",")) if a.roi else None),
                      display=cfg.get("display_names") or {})
        cap.run(a.only, hands=not a.no_hands, hand_seconds=a.hand_seconds)
    finally:
        fb.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
