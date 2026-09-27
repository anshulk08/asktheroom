"""Guided capture with automatic labels: the fast path, no hand labelling (see README.md).

Run on the Jetson host venv (cv2 + numpy only) with the app stopped, since it holds the camera:

    python scripts/finetune/capture.py                          # empty table, 8 objects x 8 poses, hands
    python scripts/finetune/capture.py --only glasses keys      # redo some objects (same empty table)
    python scripts/finetune/capture.py --qa                     # rebuild the contact sheet only
    python scripts/finetune/capture.py --distractors airpods mug charger    # objects + hands + other things
    python scripts/finetune/capture.py --distractors airpods mug --only airpods   # redo one distractor
    python scripts/finetune/capture.py --room table --data data/ft-corner-table   # the room build's table view
    python scripts/finetune/capture.py --room couch --data data/ft-corner-couch   # one room zone's crop

Each object lies alone on the empty table, so bglabel.py's difference against the empty table is its
box. Writes into --data (default data/finetune), the layout the other scripts use:

    images/cap<g>_<object>-<k>.jpg, labels/...txt   g = k % 4 is the "trial" train.py splits on;
                                                    cap3 is held out (train.py --val-trials cap3)
    images/cap<g>_empty-<k>.jpg, empty labels       negatives
    images/cap<g>_<distractor>-<k>.jpg, empty labels  things that are none of the classes: negatives
    cutouts/<stem>.png (BGRA) + <stem>.json         for synthesize.py ("distractor": true on distractors)
    backgrounds/empty-<k>.jpg                       raw empty-table frames for synthesize.py
    qa_capture.jpg                                  every capture with its box: look at it
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Callable, Optional, Sequence

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
RESERVED = {"empty", "hand"}        # stems capture.py already uses for other things


class Quit(Exception):
    pass


def stem(name: str, k: int) -> str:
    """cap<k % 4>_<name>-<k>. No '_' after the group, so common.trial_of() gives the group."""
    return f"cap{k % GROUPS}_{name.replace('_', '-')}-{k:02d}"


def _clear(data: Path, name: str) -> None:
    """Delete one name's captures: cap<g>_<name>-<kk>[-<i>], not those of a longer name ('pill' keeps
    'pill_bottle')."""
    n = name.replace("_", "-")
    own = re.compile(rf"cap\d+_{re.escape(n)}-\d{{2,}}(-\d)?")
    for sub, ext in (("images", "jpg"), ("labels", "txt"), ("cutouts", "png"), ("cutouts", "json")):
        for p in (data / sub).glob(f"cap*_{n}-*.{ext}"):
            if own.fullmatch(p.stem):
                p.unlink()


def check_distractors(distractors: Sequence[str], names: Sequence[str]) -> list[str]:
    """Problems with distractor names: each must be one word (letters, digits, _) that is not a class,
    'hand' or 'empty', and not repeated."""
    bad = []
    for d in distractors:
        if not re.fullmatch(r"\w+", d):
            bad.append(f"'{d}' is not one word (use letters, digits and _)")
        elif d in names or d in RESERVED:
            bad.append(f"'{d}' is a class or reserved name, so it can't be a distractor")
    if len(set(distractors)) != len(distractors):
        bad.append("a distractor is listed twice")
    return bad


def save_cutouts(data: Path, st: str, img: np.ndarray, name: str, labels, distractor: bool = False) -> None:
    """One BGRA cutout per label, with its label box relative to the cutout (a hand's box is only the
    hand end of the arm) and the edge an arm comes in from, which synthesize.py keeps it attached to.
    A distractor's meta says "distractor": true: synthesize.py pastes it but never boxes it."""
    (data / "cutouts").mkdir(parents=True, exist_ok=True)
    h, w = img.shape[:2]
    for i, lab in enumerate(labels):
        s = st if len(labels) == 1 else f"{st}-{i}"
        bgra, (x0, y0) = cutout(img, lab.mask)
        cv2.imwrite(str(data / "cutouts" / f"{s}.png"), bgra)
        x1, y1, x2, y2 = lab.box
        meta = {"name": name, "stem": st, "origin": [x0, y0], "frame": [w, h], "edge": lab.edge,
                "box": [x1 - x0, y1 - y0, x2 - x0, y2 - y0]}
        if distractor:
            meta["distractor"] = True
        (data / "cutouts" / f"{s}.json").write_text(json.dumps(meta))


class Capture:
    """The guided session. source: FrameBuffer-like (wait_new); ask/say: the terminal, or a test script."""

    def __init__(self, source, data: Path, names: list[str], *, poses: int = 8, params: Optional[Params] = None,
                 ask: Callable[[str], str] = input, say: Callable[[str], None] = print,
                 settle_timeout_s: float = 15.0, display: Optional[dict] = None,
                 distractors: Sequence[str] = (), distractor_poses: int = 4):
        bad = check_distractors(distractors, names)
        if bad:
            raise ValueError("; ".join(bad))
        self.src, self.data, self.names, self.poses = source, Path(data), names, poses
        self.distractors, self.distractor_poses = list(distractors), distractor_poses
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

    def capture_object(self, cls_id: Optional[int], name: str, poses: Optional[int] = None) -> int:
        """Poses of one thing alone on the table. cls_id None: a distractor, saved with an empty label
        (a negative for every class); its cutout is still cut from the background difference."""
        _clear(self.data, name)
        spoken = self.display.get(name, name.replace("_", " "))
        poses = poses or self.poses
        k = 0
        while k < poses:
            r = self._prompt(f"[{name} {k + 1}/{poses}] Place the {spoken} alone on the table, "
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
            save_sample(self.data, st, img, [] if cls_id is None else [(cls_id, lab.box)])
            save_cutouts(self.data, st, img, name, [lab], distractor=cls_id is None)
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

    def capture_scene(self, seconds: float = 60.0, every_s: float = 0.3) -> int:
        """Unlabelled frames of the demo layout for the go/no-go sweep (eval/conf_sweep.py --images), into
        <data>/scene/: nothing here is trained on."""
        out = self.data / "scene"
        out.mkdir(parents=True, exist_ok=True)
        for p in out.glob("scene-*.jpg"):
            p.unlink()
        self._prompt(f"Scene: lay out the demo (notebook, box, keys, ...) as for a judge; for {seconds:.0f} s "
                     f"reach in now and then (a hand in view about half the time). Enter to start: ")
        f = self._next()
        t0, last, j = f.t, -1e9, 0
        while f.t - t0 < seconds:
            if f.t - last >= every_s:
                last = f.t
                cv2.imwrite(str(out / f"scene-{j:04d}.jpg"), f.img, [cv2.IMWRITE_JPEG_QUALITY, 95])
                j += 1
            f = self._next()
        self.say(f"  {j} scene frames in {out}")
        return j

    def run(self, only: Optional[list[str]] = None, hands: bool = True, hand_seconds: float = 40.0) -> dict:
        got = {}
        try:
            self.background()
            for i, name in enumerate(self.names):
                if name == "hand" or (only and name not in only):
                    continue
                got[name] = self.capture_object(i, name)
            for name in self.distractors:
                if not only or name in only:
                    got[name] = self.capture_object(None, name, self.distractor_poses)
            if hands and (not only or "hand" in only):
                got["hand"] = self.capture_hands(self.names.index("hand"), hand_seconds)
        except Quit:
            self.say("stopped; everything captured so far is saved")
        write_data_yaml(self.data, self.names)
        qa = write_qa(self.data, self.names)
        self.say(f"captured {got}; contact sheet: {qa}")
        return got


def room_rect(cfg: dict, view: str) -> tuple[tuple[int, int, int, int], tuple[int, int]]:
    """(full-frame rect, out size) of what the room build's detector sees (room_memory: in config.yaml plus
    config.local.yaml): 'table' is the table view (table_view_rect, or the centred default, resized to
    frame_size_px, as main.open_frames / core.room_view.TableView cut it); any other name is that zone of
    room_memory.zones_path (its bbox in the camera frame, shrunk to max_crop_px like core/room.py _visit)."""
    from core.room_types import RoomConfig
    from core.room_view import default_rect
    rc = RoomConfig.from_dict(cfg.get("room_memory"))
    fw, fh = rc.capture_size
    if view == "table":
        out = tuple(int(v) for v in cfg.get("frame_size_px", (1280, 720)))
        rect = rc.table_view_rect or default_rect(rc.capture_size, rc.zoom, rc.ref_zoom, out)
        return tuple(int(v) for v in rect), out
    from core.room_zones import Zones
    zones = Zones.load(rc.zones_path).zones
    if view not in zones:
        raise ValueError(f"no zone {view!r} in {rc.zones_path}; zones: {sorted(zones)} (or 'table')")
    bx1, by1, bx2, by2 = zones[view].bbox()
    x1, y1, x2, y2 = max(0, bx1), max(0, by1), min(fw, bx2), min(fh, by2)
    s = min(1.0, rc.max_crop_px / max(x2 - x1, y2 - y1))
    return (x1, y1, x2, y2), (max(1, round((x2 - x1) * s)), max(1, round((y2 - y1) * s)))


def room_source(cfg: dict, device, view: str):
    """The camera at room_memory.capture_size, seen through room_rect(view): a FrameBuffer-like source."""
    from core.capture import FrameBuffer, open_camera
    from core.room_types import RoomConfig
    from core.room_view import TableView
    rc = RoomConfig.from_dict(cfg.get("room_memory"))
    rect, out = room_rect(cfg, view)
    w, h = rc.capture_size
    fb = FrameBuffer(device, ring_s=rc.ring_s, opener=lambda src: open_camera(src, w, h))
    print(f"room view {view!r}: camera {w}x{h}, rect {list(rect)} -> {out[0]}x{out[1]}")
    return TableView(fb, rect, out)


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
    ap.add_argument("--only", nargs="*", help="object names (and/or 'hand', or --distractors names) to (re)capture")
    ap.add_argument("--distractors", nargs="*", default=[],
                    help="other things (one word each, not classes) to capture as negatives: pasted into "
                         "composites without a box, and their real images are labelled empty")
    ap.add_argument("--distractor-poses", type=int, default=4)
    ap.add_argument("--no-hands", action="store_true")
    ap.add_argument("--roi", help="x1,y1,x2,y2: the tabletop in camera px; changes outside it (floor, chair, "
                                  "the person capturing) are ignored")
    ap.add_argument("--hand-seconds", type=float, default=40.0)
    ap.add_argument("--hand-len", type=int, default=Params.hand_len_px,
                    help="px of an arm kept as the hand box (about 18 cm of arm at this camera height)")
    ap.add_argument("--qa", action="store_true", help="only rebuild qa_capture.jpg")
    ap.add_argument("--scene", type=float, metavar="SECONDS",
                    help="only record the demo layout's frames (unlabelled) into <data>/scene/ for the go/no-go")
    ap.add_argument("--room", metavar="VIEW",
                    help="capture what the room build's detector sees (room_memory: in config.local.yaml): "
                         "'table' = the table view cut from the full frame, or a zone name (its crop). "
                         "One --data dir per view; merge them with merge_sets.py")
    a = ap.parse_args(argv)
    cfg = load_config()
    names = class_names(cfg)
    if a.qa:
        print(write_qa(Path(a.data), names))
        return 0
    problems = check_distractors(a.distractors, names)
    if problems:
        print("; ".join(problems), file=sys.stderr)
        return 2
    bad = set(a.only or []) - set(names) - set(a.distractors)
    if bad:
        print(f"unknown names {sorted(bad)}; choose from {names} or --distractors", file=sys.stderr)
        return 2
    device = int(a.device) if a.device.isdigit() else a.device
    if a.room:
        fb = room_source(cfg, device, a.room)
    else:
        from core.capture import FrameBuffer
        fb = FrameBuffer(device)
    try:
        cap = Capture(fb, Path(a.data), names, poses=a.poses, params=Params(hand_len_px=a.hand_len,
                                  roi_px=tuple(int(v) for v in a.roi.split(",")) if a.roi else None),
                      display=cfg.get("display_names") or {}, distractors=a.distractors,
                      distractor_poses=a.distractor_poses)
        if a.scene:
            cap.capture_scene(a.scene)
        else:
            cap.run(a.only, hands=not a.no_hands, hand_seconds=a.hand_seconds)
    finally:
        fb.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
