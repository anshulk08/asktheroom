"""Fast path for the prompt-free YOLOE proposer: the reduced-head export plus a numpy postprocess.
Owner: P (open world). Used by core/proposals.py's YOLOEProposer when proposals.yoloe.model is a
reduced .engine / .onnx; every other model keeps the ultralytics path.

Why: the prompt-free head scores 4585 vocabulary classes per anchor, so the stock output is
[1, 4 + 4585 + 32, 8400] and ultralytics' postprocess spends ~22 ms on the Orin just reading it (the
engine itself runs in ~18 ms). experiments/openset/reduce_head.py does the class max / argmax inside
the graph, so the engine hands back [1, 38, 8400] and the host touches 120x less data:

    row 0-3    cx, cy, w, h of the box, in letterboxed input px (640 x 640)
    row 4      the best class score (already a sigmoid probability)
    row 5      that class's index, as a float (the vocabulary is in the export's metadata 'names')
    row 6-37   32 mask coefficients (seg exports; absent from a detect export: 6 rows)
    output1    [1, 32, 160, 160] mask prototypes over the same 640 x 640 input

This file then does what ultralytics does after a stock model (non_max_suppression with
agnostic_nms, multi_label off, then scale_boxes and process_mask): score > conf, class-agnostic greedy
NMS at iou, max_det, boxes back to frame px through the letterbox, and optionally box-sized masks.
One addition: boxes over max_area_frac of the frame (the table, the whole scene) are dropped before
NMS; they are never objects and would only cost NMS time. YOLOEProposer.people() (the laser's safety
gate) lifts that cap, since the nearest person fills the frame, and asks for per-class NMS
(agnostic_nms=False), so a shirt box cannot suppress the person wearing it.

ReducedYOLOE mimics the part of ultralytics' YOLO.predict() that YOLOEProposer reads (boxes.xyxy /
conf / cls, masks.data, names), so the proposer's filtering (ignored classes, size, table outline,
dedupe) is the same code for both paths.

Runners: a .engine through ultralytics' TensorRT backend (reads the metadata header onnx2engine writes,
replays a captured CUDA graph); a .onnx through onnxruntime (the Mac, tests, a fallback). Build the
engine in the Jetson container from the Mac-exported ONNX (a torch export there runs out of memory):

    python3 experiments/openset/jetson_onnx2engine.py yoloe-26s-seg-pf-reduced.onnx
    python -m core.yoloe_fast --model models/yoloe-26s-seg-pf-reduced.engine --image f.jpg --bench 100
"""
from __future__ import annotations

import ast
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Callable, Optional, Sequence

import cv2
import numpy as np

from core.config import ROOT

PAD_VALUE = 114                 # ultralytics' letterbox grey


def is_reduced(model_path: str) -> bool:
    """A reduced-head export (by file name, as reduce_head.py names it): .engine or .onnx only; a
    .pt always goes through ultralytics."""
    p = Path(str(model_path))
    return p.suffix in ('.engine', '.onnx') and 'reduced' in p.stem


def letterbox(img: np.ndarray, size: int = 640) -> tuple[np.ndarray, float, tuple[int, int]]:
    """Resize keeping the aspect ratio and pad to size x size, exactly as ultralytics'
    LetterBox(auto=False, center=True). Returns (image, gain, (pad_left, pad_top))."""
    h, w = img.shape[:2]
    r = min(size / h, size / w)
    nw, nh = round(w * r), round(h * r)
    dw, dh = (size - nw) / 2, (size - nh) / 2
    if (w, h) != (nw, nh):
        img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    top, bottom = round(dh - 0.1), round(dh + 0.1)
    left, right = round(dw - 0.1), round(dw + 0.1)
    out = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(PAD_VALUE,) * 3)
    return out, r, (left, top)


def nms(xyxy: np.ndarray, scores: np.ndarray, iou: float, max_det: int) -> np.ndarray:
    """Greedy class-agnostic NMS, highest score first (torchvision.ops.nms semantics). Indices."""
    order = np.argsort(-scores, kind='stable')
    x1, y1, x2, y2 = xyxy.T
    area = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    keep = []
    while order.size and len(keep) < max_det:
        i = order[0]
        keep.append(i)
        rest = order[1:]
        iw = np.clip(np.minimum(x2[i], x2[rest]) - np.maximum(x1[i], x1[rest]), 0, None)
        ih = np.clip(np.minimum(y2[i], y2[rest]) - np.maximum(y1[i], y1[rest]), 0, None)
        inter = iw * ih
        union = area[i] + area[rest] - inter
        ov = np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)
        order = rest[ov <= iou]
    return np.asarray(keep, np.int64)


def postprocess(out: np.ndarray, protos: Optional[np.ndarray], orig_shape: Sequence[int], gain: float,
                pad: Sequence[float], conf: float = 0.15, iou: float = 0.5, max_det: int = 100,
                max_area_frac: float = 0.35, masks: bool = False, max_nms: int = 30000,
                insz: int = 640, agnostic: bool = True) -> SimpleNamespace:
    """Reduced head output -> SimpleNamespace(xyxy (N, 4) float32 frame px, conf (N,), cls (N,) int,
    masks: list of box-sized bool arrays | None), most confident first. out: [1, 6 + nm, A] or
    [6 + nm, A]; protos: [1, nm, mh, mw] or None; gain / pad from letterbox() to insz x insz.
    agnostic=False: NMS within each class only (ultralytics' agnostic_nms=False)."""
    o = np.asarray(out, np.float32)
    o = o[0] if o.ndim == 3 else o
    fh, fw = int(orig_shape[0]), int(orig_shape[1])
    sel = np.flatnonzero(o[4] > conf)
    if sel.size > max_nms:
        sel = sel[np.argsort(-o[4, sel], kind='stable')[:max_nms]]
    c = o[:, sel]
    cx, cy, bw, bh = c[0], c[1], c[2], c[3]
    xyxy = np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], axis=1)
    xyxy[:, [0, 2]] -= pad[0]
    xyxy[:, [1, 3]] -= pad[1]
    xyxy /= gain
    area = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
    ok = area <= max_area_frac * fh * fw
    xyxy, c = xyxy[ok], c[:, ok]
    if agnostic:
        keep = nms(xyxy, c[4], iou, max_det)
    else:                               # per class: shift each class's boxes clear of the others (float64:
        span = float(np.ptp(xyxy)) + 1.0 if len(xyxy) else 0.0    # the offsets reach ~1e7 px)
        keep = nms(xyxy.astype(np.float64) + c[5].round()[:, None].astype(np.float64) * span, c[4], iou, max_det)
    xyxy, c = xyxy[keep], c[:, keep]
    # ultralytics clips after scaling, i.e. after NMS in frame units; clipping before would change IoUs
    xyxy[:, [0, 2]] = np.clip(xyxy[:, [0, 2]], 0, fw)
    xyxy[:, [1, 3]] = np.clip(xyxy[:, [1, 3]], 0, fh)
    res = SimpleNamespace(xyxy=xyxy.astype(np.float32), conf=c[4].astype(np.float32),
                          cls=c[5].round().astype(np.int64), masks=None)
    nm = o.shape[0] - 6
    if masks and nm > 0 and protos is not None:
        res.masks = _masks(c[6:].T, np.asarray(protos, np.float32).reshape(nm, *np.shape(protos)[-2:]),
                           res.xyxy, gain, pad, insz)
    return res


def _masks(coeffs: np.ndarray, protos: np.ndarray, xyxy: np.ndarray, gain: float, pad,
           insz: int = 640) -> list[np.ndarray]:
    """Box-sized bool masks: sigmoid(coeffs @ protos) > 0.5, i.e. logit > 0, sampled at each frame
    pixel of the box. A frame pixel maps to letterbox px (u + 0.5) * gain + pad, and the prototype
    grid to the letterbox like a bilinear upsample with align_corners=False, as ultralytics does."""
    nm, mh, mw = protos.shape
    logits = (coeffs @ protos.reshape(nm, -1)).reshape(-1, mh, mw)
    out = []
    for lg, (x1, y1, x2, y2) in zip(logits, xyxy):
        a, b, c, d = int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))
        if c <= a or d <= b:
            out.append(np.zeros((max(0, d - b), max(0, c - a)), bool))
            continue
        u = ((np.arange(a, c, dtype=np.float32) + 0.5) * gain + pad[0]) * (mw / insz) - 0.5
        v = ((np.arange(b, d, dtype=np.float32) + 0.5) * gain + pad[1]) * (mh / insz) - 0.5
        mx, my = np.meshgrid(u, v)
        m = cv2.remap(lg.astype(np.float32), mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        out.append(m > 0)
    return out


# ----- runners: input (1, 3, s, s) float32 in [0, 1] RGB -> (out, protos | None) -------------------

def _names(raw) -> dict:
    if isinstance(raw, dict):
        return {int(k): str(v) for k, v in raw.items()}
    try:
        return {int(k): str(v) for k, v in ast.literal_eval(str(raw)).items()}
    except Exception:
        return {}


class OrtRunner:
    """A reduced .onnx through onnxruntime."""

    def __init__(self, path: str, providers: Sequence[str] = ('cuda', 'cpu')):
        import onnxruntime as ort
        ep = {'tensorrt': 'TensorrtExecutionProvider', 'cuda': 'CUDAExecutionProvider',
              'cpu': 'CPUExecutionProvider'}
        avail = ort.get_available_providers()
        chosen = [ep[p] for p in providers if ep.get(p) in avail] or ['CPUExecutionProvider']
        self.session = ort.InferenceSession(path, providers=chosen)
        self.input = self.session.get_inputs()[0].name
        self.outputs = [o.name for o in self.session.get_outputs()]
        self.names = _names(self.session.get_modelmeta().custom_metadata_map.get('names', '{}'))
        self.provider = self.session.get_providers()[0]

    def __call__(self, x: np.ndarray, want_protos: bool):
        wanted = self.outputs if want_protos else self.outputs[:1]
        res = self.session.run(wanted, {self.input: x})
        return res[0], (res[1] if want_protos and len(res) > 1 else None)


class TrtRunner:
    """A reduced .engine through ultralytics' TensorRT backend (TensorRT 10 named tensors, the
    metadata header, a captured CUDA graph). The image goes up as uint8 and is converted on the GPU;
    only the small reduced output (and the prototypes when masks are wanted) comes back."""

    def __init__(self, path: str, device: str = 'cuda:0'):
        import torch
        from ultralytics.nn.backends.tensorrt import TensorRTBackend
        self.torch = torch
        self.backend = TensorRTBackend(path, torch.device(device))
        self.device = self.backend.device
        self.half = bool(self.backend.fp16)
        self.names = _names(self.backend.names)
        self.provider = 'TensorRT'

    def __call__(self, x, want_protos: bool):
        t = self.torch
        with t.no_grad():
            im = x if isinstance(x, t.Tensor) else t.from_numpy(x)
            im = im.to(self.device, non_blocking=True)
            if im.dtype == t.uint8:                     # (1, s, s, 3) BGR uint8 from ReducedYOLOE
                im = im[..., [2, 1, 0]].permute(0, 3, 1, 2).float().div_(255.0)
            im = im.half() if self.half else im.float()
            outs = self.backend.forward(im.contiguous())
            out = outs[0].float().cpu().numpy()
            protos = outs[1].float().cpu().numpy() if want_protos and len(outs) > 1 else None
        return out, protos


class ReducedYOLOE:
    """Stands in for ultralytics.YOLO(model) in YOLOEProposer: predict(img, **kw) -> [result]."""

    def __init__(self, model: Optional[str] = None, imgsz: int = 640, runner: Optional[Callable] = None,
                 providers: Sequence[str] = ('cuda', 'cpu'), max_area_frac: float = 0.35):
        if runner is None:
            p = Path(model)
            p = p if p.is_absolute() else ROOT / p
            runner = TrtRunner(str(p)) if p.suffix == '.engine' else OrtRunner(str(p), providers)
        self.runner = runner
        self.imgsz = int(imgsz)
        self.max_area_frac = float(max_area_frac)
        self.names = dict(getattr(runner, 'names', {}) or {})
        self.gpu_input = isinstance(runner, TrtRunner)

    def predict(self, img: np.ndarray, imgsz: Optional[int] = None, conf: float = 0.15, iou: float = 0.5,
                max_det: int = 100, retina_masks: bool = False, agnostic_nms: bool = True,
                max_area_frac: Optional[float] = None, **_ignored) -> list[SimpleNamespace]:
        """Same meaning as ultralytics' predict arguments; half / verbose are ignored (an engine's
        precision is baked in). max_area_frac (not an ultralytics argument) overrides the model's size
        cap for this call; float('inf') keeps boxes of any size."""
        t0 = time.perf_counter()
        s = int(imgsz or self.imgsz)
        lb, gain, pad = letterbox(img, s)
        if self.gpu_input:                  # convert on the GPU: 1.2 MB up instead of 4.9 MB
            x = np.ascontiguousarray(lb[None])
        else:
            x = np.ascontiguousarray(lb[..., ::-1].transpose(2, 0, 1)[None], dtype=np.float32) / 255.0
        t1 = time.perf_counter()
        out, protos = self.runner(x, bool(retina_masks))
        t2 = time.perf_counter()
        r = postprocess(out, protos, img.shape[:2], gain, pad, conf=conf, iou=iou, max_det=max_det,
                        max_area_frac=self.max_area_frac if max_area_frac is None else max_area_frac,
                        masks=bool(retina_masks), insz=s, agnostic=bool(agnostic_nms))
        t3 = time.perf_counter()
        masks = None
        if r.masks is not None:             # full-frame, like retina_masks=True; YOLOEProposer crops them
            full = np.zeros((len(r.masks), *img.shape[:2]), bool)
            for k, (m, b) in enumerate(zip(r.masks, r.xyxy)):
                a, bb = int(round(b[0])), int(round(b[1]))
                full[k, bb:bb + m.shape[0], a:a + m.shape[1]] = m
            masks = SimpleNamespace(data=full)
        boxes = SimpleNamespace(xyxy=r.xyxy, conf=r.conf, cls=r.cls.astype(np.float32))
        speed = {'preprocess': 1000 * (t1 - t0), 'inference': 1000 * (t2 - t1), 'postprocess': 1000 * (t3 - t2)}
        return [SimpleNamespace(boxes=boxes, masks=masks, names=self.names, speed=speed, orig_shape=img.shape[:2])]


# ----- command line: time it, compare with the ultralytics path -------------------------------------

def match_iou(a: np.ndarray, b: np.ndarray, thr: float = 0.5) -> tuple[list[tuple[float, int, int]], int, int]:
    """Greedy one-to-one matching of two box sets by IoU (best pairs first). Returns ((IoU, i, j)
    per matched pair, unmatched in a, unmatched in b)."""
    if not len(a) or not len(b):
        return [], len(a), len(b)
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area = lambda z: (z[:, 2] - z[:, 0]) * (z[:, 3] - z[:, 1])
    m = inter / (area(a)[:, None] + area(b)[None] - inter + 1e-9)
    pairs = sorted(((m[i, j], i, j) for i in range(len(a)) for j in range(len(b)) if m[i, j] >= thr), reverse=True)
    ua, ub, out = set(), set(), []
    for v, i, j in pairs:
        if i not in ua and j not in ub:
            ua.add(i)
            ub.add(j)
            out.append((float(v), i, j))
    return out, len(a) - len(ua), len(b) - len(ub)


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description='reduced-head YOLOE: time end to end, compare with ultralytics')
    ap.add_argument('--model', required=True, help='reduced .engine or .onnx')
    ap.add_argument('--image', action='append', required=True)
    ap.add_argument('--bench', type=int, default=0)
    ap.add_argument('--conf', type=float, default=0.15)
    ap.add_argument('--masks', action='store_true')
    ap.add_argument('--compare', help='a stock model for ultralytics (.engine / .onnx / .pt) on the same images')
    ap.add_argument('--providers', nargs='+', default=['cuda', 'cpu'], help='for a .onnx')
    a = ap.parse_args(argv)
    m = ReducedYOLOE(a.model, providers=a.providers)
    print(f'{a.model}: runner {getattr(m.runner, "provider", "?")}, {len(m.names)} names')
    ref = None
    if a.compare:
        from ultralytics import YOLO
        ref = YOLO(a.compare, task='segment')
    kw = dict(imgsz=640, conf=a.conf, iou=0.5, max_det=100, retina_masks=a.masks)
    for path in a.image:
        img = cv2.imread(path)
        for _ in range(5):
            m.predict(img, **kw)
        r = m.predict(img, **kw)[0]
        print(f'{path}: {len(r.boxes.conf)} boxes; ' + ', '.join(f'{k} {v:.1f} ms' for k, v in r.speed.items()))
        if a.bench:
            ts, parts = [], {'preprocess': [], 'inference': [], 'postprocess': []}
            for _ in range(a.bench):
                t = time.perf_counter()
                rr = m.predict(img, **kw)[0]
                ts.append(1000 * (time.perf_counter() - t))
                for k in parts:
                    parts[k].append(rr.speed[k])
            print(f'  end to end median {np.median(ts):.1f} ms (p90 {np.percentile(ts, 90):.1f}); '
                  + ', '.join(f'{k} {np.median(v):.1f}' for k, v in parts.items()))
        if ref is not None:
            for _ in range(3):
                ref.predict(img, verbose=False, **kw, agnostic_nms=True)
            rs, tr = [], []
            for _ in range(max(1, min(a.bench, 30))):
                t = time.perf_counter()
                q = ref.predict(img, verbose=False, **kw, agnostic_nms=True)[0]
                tr.append(1000 * (time.perf_counter() - t))
            bx, bc = q.boxes.xyxy.cpu().numpy(), q.boxes.conf.cpu().numpy()
            bk = q.boxes.cls.cpu().numpy().astype(int)
            fh, fw = img.shape[:2]
            big = (bx[:, 2] - bx[:, 0]) * (bx[:, 3] - bx[:, 1]) > m.max_area_frac * fh * fw
            keep = np.flatnonzero(~big)
            pairs, ua, ub = match_iou(r.boxes.xyxy, bx[keep])
            ious = [v for v, _, _ in pairs]
            print(f'  ultralytics {a.compare}: {len(bx)} boxes (+{int(big.sum())} over {m.max_area_frac:.0%} '
                  f'of the frame, not compared), end to end median {np.median(tr):.1f} ms, speed {q.speed}')
            if ious:
                same = np.mean([int(r.boxes.cls[i]) == bk[keep[j]] for _, i, j in pairs])
                dconf = max(abs(float(r.boxes.conf[i]) - float(bc[keep[j]])) for _, i, j in pairs)
                print(f'  matched {len(ious)} at IoU >= 0.5: mean IoU {np.mean(ious):.4f}, min {np.min(ious):.4f}; '
                      f'same class {same:.0%}, max |conf diff| {dconf:.4f}; unmatched fast {ua}, ultralytics {ub}')
                if a.masks and r.masks is not None and q.masks is not None:
                    qm = q.masks.data.cpu().numpy() > 0.5
                    mi = []
                    for _, i, j in pairs:
                        x = r.masks.data[i]
                        y = qm[keep[j]]
                        mi.append((x & y).sum() / max(1, (x | y).sum()))
                    print(f'  masks: mean IoU {np.mean(mi):.4f}, min {np.min(mi):.4f}')
            else:
                print(f'  no matches; unmatched fast {ua}, ultralytics {ub}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
