"""Object and hand detection (spec 3.3). Owner: P.

Path A: YOLO-World v2 with the config prompts (zero-shot baseline and auto-labeller).
Path B (the plan): a YOLO11 model fine-tuned on overhead frames, classes = the 8 objects + hand.
Either way: keep the highest-confidence box per object, keep every hand box, convert to table cm.
Before that, known-object boxes that are really a hand or not a tabletop object are dropped
(detect_filter: in config.yaml): a box that is a hand box (zero-shot YOLO-World calls a hand 'phone'
or 'wallet'; per-class NMS keeps both labels), a box too big for any prop, and a box whose centre is
off the table (a phone in someone's pocket).
Open world (core/proposals.py): an optional class-agnostic proposer adds cls 'thing' items that are not
a known object or a hand (proposals: in config.yaml), and core/crops.py keeps crops of what was seen.

Runs inside the ultralytics/ultralytics:latest-jetson-jetpack6 container. Build TensorRT engines in
the same container that runs them (its TensorRT differs from the host's). Classes are fixed at
export time: changing prompts means re-exporting.

    python -m core.detect --export                       # YOLO-World + prompts -> .engine
    python -m core.detect --video trials/1/video.mp4     # fps + per-object detection rate
    python -m core.detect --device 0 --seconds 20        # same, live
"""
from __future__ import annotations

import datetime
import logging
import os
import time
from collections import Counter
from typing import Optional, Protocol

import numpy as np

from core import geom
from core.crops import CropStore, set_active
from core.proposals import THING, DedupeConfig, dedupe, make_proposer, table_roi
from core.types import Detection, Detections, Frame

log = logging.getLogger(__name__)

Raw = tuple[str, float, tuple[int, int, int, int]]      # (label, confidence, box_px)


class Backend(Protocol):
    def infer(self, img: np.ndarray) -> list[Raw]: ...


def class_list(cfg: dict) -> tuple[list[str], dict[str, str]]:
    """YOLO-World classes (every prompt, then 'hand') and label -> object name. Object names map to
    themselves too, so a fine-tuned model whose classes are the object names needs no other mapping."""
    prompts = cfg.get("prompts") or {}
    classes, to_obj = [], {}
    for obj in list(cfg.get("objects") or {}) + ["hand"]:
        to_obj[obj] = obj
        for p in prompts.get(obj, [obj]):
            if p not in to_obj or to_obj[p] == obj:
                to_obj[p] = obj
                if p not in classes:
                    classes.append(p)
    return classes, to_obj


def _label_names(names) -> list[str]:
    return [str(v) for v in (names.values() if isinstance(names, dict) else names)]


def missing_objects(cfg: dict, names) -> list[str]:
    """Configured objects (and 'hand') no model class maps to (by class_list): the detector can never
    report them, and before this they vanished without a word."""
    _, to_obj = class_list(cfg)
    have = {to_obj.get(n) for n in _label_names(names)}
    return [o for o in list(cfg.get("objects") or {}) + ["hand"] if o not in have]


def weights_info(path: str, names) -> dict:
    """Which weights are loaded, for the startup log and /state (state.perception.model)."""
    try:
        st = os.stat(path)
        size, mtime = st.st_size, datetime.datetime.fromtimestamp(st.st_mtime).isoformat(timespec="seconds")
    except OSError:                              # a hub name ultralytics downloads, or already gone
        size = mtime = None
    return {"path": str(path), "size": size, "mtime": mtime, "names": _label_names(names)}


class UltralyticsBackend:
    """A .pt / .engine through ultralytics. A YOLO-World .pt gets the prompts via set_classes."""

    def __init__(self, cfg: dict, model_path: Optional[str] = None):
        from ultralytics import YOLO
        d = cfg.get("detect") or {}
        self.path = model_path or d.get("model", "yolov8s-worldv2.pt")
        self.imgsz = int(d.get("imgsz", 640))
        self.half = bool(d.get("half", True))
        self.floor = float(d.get("min_conf", 0.05))
        self.model = YOLO(self.path)
        classes = class_list(cfg)[0]
        if "world" in self.path and self.path.endswith(".pt") and missing_objects(cfg, self.model.names):
            try:                                # a baked .pt (prompts in, CLIP removed) can't set_classes
                self.model.set_classes(classes)
            except Exception as ex:
                log.warning("%s: set_classes failed (%s); using the classes baked into it", self.path, ex)
        self.names = self.model.names
        self.info = weights_info(self.path, self.names)
        log.info("detector weights %s (%s bytes, modified %s), %d classes: %s", self.info["path"],
                 self.info["size"], self.info["mtime"], len(self.info["names"]), ", ".join(self.info["names"]))
        self.info["missing"] = missing_objects(cfg, self.names)
        if self.info["missing"]:
            log.warning("detector %s has no class for %s: never detected by name (conf_threshold / prompts "
                        "name them, the model does not)", self.path, ", ".join(self.info["missing"]))

    def infer(self, img: np.ndarray) -> list[Raw]:
        # An engine's precision is fixed at export; passing half= to it only prints a deprecation warning.
        extra = {"half": True} if self.half and not str(self.path).endswith(".engine") else {}
        r = self.model.predict(img, imgsz=self.imgsz, conf=self.floor, verbose=False, **extra)[0]
        b = r.boxes
        out = []
        for xyxy, conf, cls in zip(b.xyxy.cpu().numpy(), b.conf.cpu().numpy(), b.cls.cpu().numpy()):
            out.append((self.names[int(cls)], float(conf), tuple(int(round(v)) for v in xyxy)))
        return out


_FROM_CONFIG = object()


class Detector:
    def __init__(self, cfg: dict, table=None, backend: Optional[Backend] = None,
                 proposer=_FROM_CONFIG, crops=_FROM_CONFIG):
        """proposer / crops default to what config.yaml's proposals: section asks for; pass None to
        turn either off, or an instance (tests, experiments)."""
        self.cfg = cfg
        self.table = table
        self.backend = backend or UltralyticsBackend(cfg)
        _, self.to_obj = class_list(cfg)
        ct = cfg.get("conf_threshold", 0.35)
        self.thresholds = ct if isinstance(ct, dict) else {"default": ct}
        df = cfg.get("detect_filter") or {}
        self.hand_iou = float(df.get("hand_iou", 0.6))
        self.max_area_frac = float(df.get("max_area_frac", 0.25))
        self.table_margin_cm = df.get("table_margin_cm", 5.0)     # None: keep boxes off the table
        self.last_ms = 0.0
        self.last_proposal_ms = 0.0
        pc = cfg.get("proposals") or {}
        self.proposer = make_proposer(cfg) if proposer is _FROM_CONFIG else proposer
        self.dedupe_cfg = DedupeConfig.from_dict(pc.get("dedupe"))
        if crops is _FROM_CONFIG:
            cc = pc.get("crops") or {}
            crops = CropStore.from_config(cc) if pc.get("enabled") and cc.get("enabled", True) else None
        self.crops = crops
        if crops is not None:
            set_active(crops)
        self._roi_for = _FROM_CONFIG           # the table calibration the proposer's outline was made from
        self._open_err_t = float("-inf")

    def reset_proposals(self) -> None:
        """Recapture the empty-table reference and forget crops (RESET, or the table was cleared)."""
        if self.proposer is not None:
            self.proposer.reset()
        if self.crops is not None:
            self.crops.clear()

    def threshold(self, obj: str) -> float:
        return float(self.thresholds.get(obj, self.thresholds.get("default", 0.35)))

    def detect(self, frame: Frame) -> Detections:
        if self.table is None or not self.table.ok:
            raise RuntimeError("table is not calibrated: run core.table first")
        t0 = time.perf_counter()
        raw = self.backend.infer(frame.img)
        self.last_ms = 1000 * (time.perf_counter() - t0)
        known: list[Raw] = []
        hands: list[Raw] = []
        for label, conf, box in raw:
            obj = self.to_obj.get(label)
            if obj is None or conf < self.threshold(obj):
                continue
            (hands if obj == "hand" else known).append((obj, conf, box))
        max_area = self.max_area_frac * frame.img.shape[0] * frame.img.shape[1] if frame.img is not None else None
        best: dict[str, Detection] = {}
        for obj, conf, box in sorted(known, key=lambda r: -r[1]):
            if obj in best or any(geom.iou(box, h[2]) >= self.hand_iou for h in hands):
                continue
            if max_area is not None and geom.area(box) > max_area:
                continue
            d = self._to_det(obj, conf, box)
            if self._on_table(d):
                best[obj] = d
        items = list(best.values())
        hand_dets = [self._to_det(*r) for r in hands]
        if frame.img is not None and (self.proposer is not None or self.crops is not None):
            try:
                if self.proposer is not None:
                    items += self._proposals(frame.img, items, hand_dets)
                if self.crops is not None:
                    self.crops.update(frame.img, items, hand_dets, frame.t)
            except Exception:                   # the open world must never cost the known objects
                now = time.monotonic()
                if now - self._open_err_t > 10:
                    log.exception("open-world proposals / crops failed; known objects only this frame")
                    self._open_err_t = now
        return Detections(t=frame.t, frame_idx=frame.idx, items=items, hands=hand_dets)

    def _proposals(self, img: np.ndarray, items: list[Detection], hands: list[Detection]) -> list[Detection]:
        """Class-agnostic 'thing' detections that duplicate no known object or hand."""
        t0 = time.perf_counter()
        cal = getattr(self.table, "H", None)
        if cal is not self._roi_for:            # first frame, or the table was recalibrated
            if self._roi_for is not _FROM_CONFIG:
                self.proposer.reset()
            self._roi_for = cal
            self.proposer.set_roi(table_roi(self.table, self.cfg))
        known = [d.box_px for d in items]
        hand_boxes = [h.box_px for h in hands]
        props = dedupe(self.proposer.propose(img, known, hand_boxes), known, hand_boxes, self.dedupe_cfg)
        self.last_proposal_ms = 1000 * (time.perf_counter() - t0)
        return [self._to_det(THING, p.conf, p.box_px, p.occluded) for p in props]

    __call__ = detect

    def _on_table(self, d: Detection) -> bool:
        in_bounds = getattr(self.table, "in_bounds", None)
        if self.table_margin_cm is None or in_bounds is None:
            return True
        return bool(in_bounds(d.center_cm, float(self.table_margin_cm)))

    def _to_det(self, obj: str, conf: float, box, occluded: bool = False) -> Detection:
        x1, y1, x2, y2 = box
        pts = self.table.px_to_cm([[x1, y1], [x2, y2], [(x1 + x2) / 2, (y1 + y2) / 2]])
        (cx1, cy1), (cx2, cy2), (cx, cy) = (tuple(float(v) for v in p) for p in pts)
        return Detection(cls=obj, conf=round(conf, 3), box_px=(x1, y1, x2, y2),
                         center_cm=(cx, cy),
                         box_cm=(min(cx1, cx2), min(cy1, cy2), max(cx1, cx2), max(cy1, cy2)), occluded=occluded)


def export_engine(cfg: dict, pt: Optional[str] = None) -> str:
    """YOLO-World + the config prompts -> TensorRT FP16 engine. Run inside the Jetson container."""
    from ultralytics import YOLO
    d = cfg.get("detect") or {}
    model = YOLO(pt or d.get("world_pt", "yolov8s-worldv2.pt"))
    model.set_classes(class_list(cfg)[0])
    return model.export(format="engine", half=True, imgsz=int(d.get("imgsz", 640)))


def main(argv=None) -> int:
    import argparse

    import cv2

    from core.config import load_config
    from core.table import Table
    ap = argparse.ArgumentParser(description="detector: export, or report fps and detection rates")
    ap.add_argument("--export", action="store_true")
    ap.add_argument("--model")
    ap.add_argument("--video")
    ap.add_argument("--device")
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--save", help="write one annotated frame here")
    a = ap.parse_args(argv)
    cfg = load_config()
    if a.export:
        print("engine:", export_engine(cfg, a.model))
        return 0
    table = Table(cfg)
    if not table.ok:
        print("table not calibrated (run python -m core.table); using a flat frame->table mapping for this report")
        table = _FlatTable(cfg)
    det = Detector(cfg, table, UltralyticsBackend(cfg, a.model))
    if a.video:
        from core.capture import VideoFileSource
        frames = VideoFileSource(a.video, start=False).frames()
    else:
        from core.capture import FrameBuffer
        fb = FrameBuffer(int(a.device or 0))
        frames = _live(fb, a.seconds)
    seen, n, ms, last = Counter(), 0, [], None
    t0 = time.monotonic()
    for f in frames:
        d = det.detect(f)
        n += 1
        ms.append(det.last_ms)
        seen.update(i.cls for i in d.items)
        seen["hand"] += bool(d.hands)
        last = (f, d)
    wall = time.monotonic() - t0
    print(f"{n} frames, {n / wall:.1f} fps end to end, inference median {np.median(ms):.0f} ms")
    for obj in list(cfg.get("objects") or {}) + ["hand"]:
        print(f"  {obj:12s} detected in {100 * seen[obj] / max(n, 1):5.1f}% of frames")
    if a.save and last:
        img = last[0].img.copy()
        for x in last[1].items + last[1].hands:
            x1, y1, x2, y2 = x.box_px
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, f"{x.cls} {x.conf:.2f}", (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.imwrite(a.save, img)
    return 0


def _live(fb, seconds: float):
    end, last = time.monotonic() + seconds, 0
    try:
        while time.monotonic() < end:
            f = fb.wait_new(last, 1.0)
            if f is None:
                return
            last = f.idx
            yield f
    finally:
        fb.stop()


class _FlatTable:
    """Frame == table, for reports before calibration."""
    ok = True

    def __init__(self, cfg: dict):
        self.w, self.h = (cfg.get("table") or {}).get("size_cm", (90, 60))

    def px_to_cm(self, pts):
        p = np.asarray(pts, dtype=float).reshape(-1, 2)
        return p * [self.w / 1280, self.h / 720]


if __name__ == "__main__":
    raise SystemExit(main())
