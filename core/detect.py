"""Object and hand detection (spec 3.3). Owner: P.

Path A: YOLO-World v2 with the config prompts (zero-shot baseline and auto-labeller).
Path B (the plan): a YOLO11 model fine-tuned on overhead frames, classes = the 8 objects + hand.
Either way: keep the highest-confidence box per object, keep every hand box, convert to table cm.

Runs inside the ultralytics/ultralytics:latest-jetson-jetpack6 container. Build TensorRT engines in
the same container that runs them (its TensorRT differs from the host's). Classes are fixed at
export time: changing prompts means re-exporting.

    python -m core.detect --export                       # YOLO-World + prompts -> .engine
    python -m core.detect --video trials/1/video.mp4     # fps + per-object detection rate
    python -m core.detect --device 0 --seconds 20        # same, live
"""
from __future__ import annotations

import logging
import time
from collections import Counter
from typing import Optional, Protocol

import numpy as np

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
        if "world" in self.path and self.path.endswith(".pt"):
            self.model.set_classes(class_list(cfg)[0])
        self.names = self.model.names

    def infer(self, img: np.ndarray) -> list[Raw]:
        r = self.model.predict(img, imgsz=self.imgsz, conf=self.floor, half=self.half, verbose=False)[0]
        b = r.boxes
        out = []
        for xyxy, conf, cls in zip(b.xyxy.cpu().numpy(), b.conf.cpu().numpy(), b.cls.cpu().numpy()):
            out.append((self.names[int(cls)], float(conf), tuple(int(round(v)) for v in xyxy)))
        return out


class Detector:
    def __init__(self, cfg: dict, table=None, backend: Optional[Backend] = None):
        self.cfg = cfg
        self.table = table
        self.backend = backend or UltralyticsBackend(cfg)
        _, self.to_obj = class_list(cfg)
        ct = cfg.get("conf_threshold", 0.35)
        self.thresholds = ct if isinstance(ct, dict) else {"default": ct}
        self.last_ms = 0.0

    def threshold(self, obj: str) -> float:
        return float(self.thresholds.get(obj, self.thresholds.get("default", 0.35)))

    def detect(self, frame: Frame) -> Detections:
        if self.table is None or not self.table.ok:
            raise RuntimeError("table is not calibrated: run core.table first")
        t0 = time.perf_counter()
        raw = self.backend.infer(frame.img)
        self.last_ms = 1000 * (time.perf_counter() - t0)
        best: dict[str, Raw] = {}
        hands: list[Raw] = []
        for label, conf, box in raw:
            obj = self.to_obj.get(label)
            if obj is None or conf < self.threshold(obj):
                continue
            if obj == "hand":
                hands.append((obj, conf, box))
            elif obj not in best or conf > best[obj][1]:
                best[obj] = (obj, conf, box)
        return Detections(t=frame.t, frame_idx=frame.idx,
                          items=[self._to_det(*r) for r in best.values()],
                          hands=[self._to_det(*r) for r in hands])

    __call__ = detect

    def _to_det(self, obj: str, conf: float, box) -> Detection:
        x1, y1, x2, y2 = box
        pts = self.table.px_to_cm([[x1, y1], [x2, y2], [(x1 + x2) / 2, (y1 + y2) / 2]])
        (cx1, cy1), (cx2, cy2), (cx, cy) = (tuple(float(v) for v in p) for p in pts)
        return Detection(cls=obj, conf=round(conf, 3), box_px=(x1, y1, x2, y2),
                         center_cm=(cx, cy),
                         box_cm=(min(cx1, cx2), min(cy1, cy2), max(cx1, cx2), max(cy1, cy2)))


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
