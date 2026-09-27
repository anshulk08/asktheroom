"""Appearance embeddings for open-world re-identification: the embed(frame_img, box_px) that
World(cfg, events, embed=...) uses to confirm which thing a proposal is (core/things.py).

DINOv2-S/14 won the measurement run (experiments/openset/embed_eval.py): best cross-frame rank-1 of
the candidates on overhead crops, 2.5-3.5 ms a crop on the Orin through TensorRT FP16, and FP16 vs
FP32 cosine >= 0.999. No global threshold separates same from different objects, so the world only
ranks with it under its causal / spatial rules; it never decides identity from looks alone.

    crop (box + margin, clipped to the frame) -> letterbox to input_size -> RGB, ImageNet mean /
    std -> onnxruntime -> L2-normalised float32 vector

Letterbox (pad with the mean) rather than stretching: on the proxy overhead frames with 90-degree
rotated copies (92 crops, CPU) it separated same / different objects a little better (pair AUC 0.990
vs 0.980, same cross-frame rank-1 0.957); a stretched crop of a rotated object is a different shape.

onnxruntime runs the ONNX with the TensorRT EP (FP16, engines cached under models/cache/reid; the
first build takes ~75 s on the Orin, later starts load the cache), falling back to CUDA, then CPU.
The session loads on a background thread by default: until it is ready every call returns None,
which the world reads as 'no appearance evidence', so a first-time engine build never stalls
startup or perception. Any failure is also None: appearance is optional evidence, never an error.

    python -m core.embed --image f.jpg --box 10 20 110 140 [--box ...] [--bench 50] [--save a.npz]
    python -m core.embed --compare a.npz b.npz        # cosine between two runs' vectors
"""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Callable, Optional, Sequence

import cv2
import numpy as np

from core.config import ROOT

log = logging.getLogger(__name__)

MEAN = np.array([0.485, 0.456, 0.406], np.float32)     # ImageNet, RGB (DINOv2's training stats)
STD = np.array([0.229, 0.224, 0.225], np.float32)
EP = {'tensorrt': 'TensorrtExecutionProvider', 'cuda': 'CUDAExecutionProvider',
      'cpu': 'CPUExecutionProvider'}


@dataclass
class ReidConfig:
    """The reid: section of config.yaml; every key is optional."""
    enabled: bool = False
    model: str = 'models/reid/dinov2_s14.onnx'
    providers: list = field(default_factory=lambda: ['tensorrt', 'cuda', 'cpu'])   # tried in order
    input_size: int = 224
    margin: float = 0.08            # of the box's width / height, each side (as evaluated)
    fit: str = 'letterbox'          # letterbox (keeps the aspect ratio) | stretch
    min_px: int = 8                 # a crop narrower than this (after clipping) is not embedded
    max_batch: int = 8              # the TensorRT profile's largest batch; bigger batches are split
    fp16: bool = True
    cache_dir: str = 'models/cache/reid'
    threads: int = 2                # CPU provider only: intra-op threads (perception shares the CPU)
    background: bool = True         # load (and build the engine) on a thread; None until ready

    @classmethod
    def from_dict(cls, raw: Optional[dict]) -> 'ReidConfig':
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in (raw or {}).items() if k in known})


def _path(p: str) -> Path:
    q = Path(p)
    return q if q.is_absolute() else ROOT / q


def provider_chain(cfg: ReidConfig, available: Sequence[str]) -> list[list]:
    """Provider lists to try, best first: each configured provider this onnxruntime has, followed
    by the ones after it (so [tensorrt, cuda, cpu] tries TRT+CUDA+CPU, then CUDA+CPU, then CPU).
    TensorRT gets FP16, the engine cache and a dynamic-batch profile 1..max_batch; without an
    explicit profile every new batch size would build another engine."""
    names = [EP[p] for p in (str(x).lower() for x in cfg.providers) if p in EP and EP[p] in available]
    s, b = int(cfg.input_size), int(cfg.max_batch)
    opts = {
        'trt_fp16_enable': bool(cfg.fp16),
        'trt_engine_cache_enable': True,
        'trt_engine_cache_path': str(_path(cfg.cache_dir)),
        'trt_timing_cache_enable': True,
        'trt_timing_cache_path': str(_path(cfg.cache_dir)),
        'trt_profile_min_shapes': f'images:1x3x{s}x{s}',
        'trt_profile_opt_shapes': f'images:{b}x3x{s}x{s}',
        'trt_profile_max_shapes': f'images:{b}x3x{s}x{s}',
    }
    cuda_opts = {'arena_extend_strategy': 'kSameAsRequested'}   # the detector shares the GPU
    entry = {EP['tensorrt']: (EP['tensorrt'], opts), EP['cuda']: (EP['cuda'], cuda_opts)}
    return [[entry.get(n, n) for n in names[i:]] for i in range(len(names))]


def _ort_session(cfg: ReidConfig):
    def make(path: str, providers: list):
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.intra_op_num_threads = int(cfg.threads)
        so.log_severity_level = 3              # TensorRT EP warnings are loud and harmless
        return ort.InferenceSession(path, sess_options=so, providers=providers)
    return make


class Embedder:
    """embed(frame_img, box_px) -> unit float32 vector | None, and batch(img, boxes) for several
    crops of one frame in a single model run. Thread-safe (onnxruntime sessions are)."""

    def __init__(self, cfg: ReidConfig | dict | None = None, session=None,
                 session_factory: Optional[Callable] = None, available: Optional[Sequence[str]] = None):
        self.cfg = cfg if isinstance(cfg, ReidConfig) else ReidConfig.from_dict(cfg)
        self._factory = session_factory or _ort_session(self.cfg)
        self._available = available
        self.session = session
        self.provider: Optional[str] = session.get_providers()[0] if session is not None else None
        self.error: Optional[str] = None
        self.load_s: Optional[float] = None
        self.last_ms = 0.0
        self._input = session.get_inputs()[0].name if session is not None else 'images'
        self._ready = threading.Event()
        if session is not None:
            self._ready.set()
        self._thread: Optional[threading.Thread] = None

    # -- loading

    @property
    def ready(self) -> bool:
        return self._ready.is_set()

    def wait_ready(self, timeout: Optional[float] = None) -> bool:
        return self._ready.wait(timeout)

    def start(self) -> 'Embedder':
        """Load on a daemon thread (see the module notes); returns self."""
        if self._thread is None and not self.ready:
            self._thread = threading.Thread(target=self.load, name='reid-load', daemon=True)
            self._thread.start()
        return self

    def load(self) -> bool:
        """Create the session down the provider chain; each attempt runs one warm-up batch of
        max_batch inside the try, because TensorRT builds (or loads) its engine on the first run and
        that is where it fails. False (and self.error) when nothing worked."""
        if self.ready:
            return True
        t0 = time.monotonic()
        avail = self._available
        if avail is None:
            import onnxruntime as ort
            avail = ort.get_available_providers()
        chains = provider_chain(self.cfg, avail)
        if not chains:
            self.error = f'none of {self.cfg.providers} is available (onnxruntime has {list(avail)})'
            log.warning('reid: %s; appearance off', self.error)
            return False
        path = str(_path(self.cfg.model))
        if any(isinstance(p, tuple) and p[0] == EP['tensorrt'] for p in chains[0]):
            _path(self.cfg.cache_dir).mkdir(parents=True, exist_ok=True)
        s = int(self.cfg.input_size)
        for chain in chains:
            first = chain[0] if isinstance(chain[0], str) else chain[0][0]
            try:
                sess = self._factory(path, chain)
                name = sess.get_inputs()[0].name
                for b in sorted({1, int(self.cfg.max_batch)}):
                    sess.run(None, {name: np.zeros((b, 3, s, s), np.float32)})
            except Exception as ex:
                log.warning('reid: %s failed (%s: %s); trying the next provider', first,
                            type(ex).__name__, str(ex)[:300])
                continue
            self.session, self._input = sess, name
            self.provider = sess.get_providers()[0]
            self.load_s = round(time.monotonic() - t0, 1)
            self.error = None
            self._ready.set()
            log.info('reid: %s on %s, ready in %.1f s', Path(path).name, self.provider, self.load_s)
            return True
        self.error = 'every provider failed'
        log.warning('reid: %s for %s; appearance off', self.error, path)
        return False

    # -- preprocessing

    def crop_box(self, shape, box_px) -> Optional[tuple[int, int, int, int]]:
        """box_px grown by margin and clipped to a frame of shape (h, w, ...); None when what is
        left is under min_px on a side (empty, off screen, or a sliver at the edge)."""
        h, w = shape[:2]
        try:
            x1, y1, x2, y2 = (float(v) for v in box_px[:4])
        except (TypeError, ValueError):
            return None
        if not all(np.isfinite((x1, y1, x2, y2))) or x2 <= x1 or y2 <= y1:
            return None
        mx, my = self.cfg.margin * (x2 - x1), self.cfg.margin * (y2 - y1)
        a, b = max(0, int(np.floor(x1 - mx))), max(0, int(np.floor(y1 - my)))
        c, d = min(w, int(np.ceil(x2 + mx))), min(h, int(np.ceil(y2 + my)))
        m = int(self.cfg.min_px)
        return (a, b, c, d) if c - a >= m and d - b >= m else None

    def preprocess(self, img: np.ndarray, box_px) -> Optional[np.ndarray]:
        """(3, s, s) float32 model input for one box, or None."""
        if img is None or img.ndim != 3:
            return None
        c = self.crop_box(img.shape, box_px)
        if c is None:
            return None
        crop = img[c[1]:c[3], c[0]:c[2]]
        s = int(self.cfg.input_size)
        ch, cw = crop.shape[:2]
        if self.cfg.fit == 'letterbox':
            k = s / max(ch, cw)
            nw, nh = max(1, round(cw * k)), max(1, round(ch * k))
        else:
            k, nw, nh = s / min(ch, cw), s, s
        interp = cv2.INTER_AREA if k < 1 else cv2.INTER_LINEAR
        small = cv2.resize(crop, (nw, nh), interpolation=interp)
        x = (small[..., ::-1].astype(np.float32) / 255.0 - MEAN) / STD
        if (nh, nw) != (s, s):                  # letterbox: pad with the mean, i.e. 0 after normalising
            canvas = np.zeros((s, s, 3), np.float32)
            top, left = (s - nh) // 2, (s - nw) // 2
            canvas[top:top + nh, left:left + nw] = x
            x = canvas
        return np.ascontiguousarray(x.transpose(2, 0, 1))

    # -- embedding

    def __call__(self, img, box_px) -> Optional[np.ndarray]:
        return self.batch(img, [box_px])[0]

    def batch(self, img, boxes: Sequence) -> list[Optional[np.ndarray]]:
        """One vector (or None) per box, in order; valid crops go through the model together, in
        chunks of max_batch."""
        out: list[Optional[np.ndarray]] = [None] * len(boxes)
        if not self.ready or img is None or not len(boxes):
            return out
        t0 = time.perf_counter()
        xs, idx = [], []
        for i, b in enumerate(boxes):
            x = self.preprocess(img, b)
            if x is not None:
                xs.append(x)
                idx.append(i)
        n = max(1, int(self.cfg.max_batch))
        try:
            for k in range(0, len(xs), n):
                emb = self.session.run(None, {self._input: np.stack(xs[k:k + n])})[0]
                emb = np.asarray(emb, np.float32).reshape(len(xs[k:k + n]), -1)
                norms = np.linalg.norm(emb, axis=1)
                for j, (v, nv) in enumerate(zip(emb, norms)):
                    if np.isfinite(nv) and nv > 0:
                        out[idx[k + j]] = (v / nv).astype(np.float32)
        except Exception:
            log.warning('reid: embedding failed; no appearance this call', exc_info=True)
            return [None] * len(boxes)
        self.last_ms = 1000 * (time.perf_counter() - t0)
        return out


def make_embedder(cfg) -> Optional[Embedder]:
    """The embedder config.yaml's reid: section asks for, or None (off, or the model file is
    missing): World(cfg, events, embed=make_embedder(cfg)) then behaves exactly as without one.
    Takes the plain config dict or core.config.Config."""
    raw = cfg.get('reid') if isinstance(cfg, dict) else getattr(cfg, 'reid', None)
    c = ReidConfig.from_dict(raw)
    if not c.enabled:
        return None
    if not _path(c.model).exists():
        log.warning('reid: %s not found (copy experiments/openset/weights/dinov2_s14.onnx there); '
                    'appearance off', _path(c.model))
        return None
    e = Embedder(c)
    if c.background:
        return e.start()
    e.load()
    return e if e.ready else None


# ----- command line: bench on the device, and check agreement between two runs --------------------

def main(argv=None) -> int:
    import argparse
    import json

    from core.config import load_config
    ap = argparse.ArgumentParser(description='re-ID embedder: time it, save vectors, compare runs')
    ap.add_argument('--image')
    ap.add_argument('--box', nargs=4, type=float, action='append', default=[])
    ap.add_argument('--boxes-json', help='{"name": [x1, y1, x2, y2], ...} for --image')
    ap.add_argument('--providers', nargs='+', help='override reid.providers, e.g. cpu')
    ap.add_argument('--bench', type=int, default=0, help='time this many runs at batch 1 and batch N')
    ap.add_argument('--save', help='write the vectors to this .npz')
    ap.add_argument('--compare', nargs=2, metavar=('A', 'B'))
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(name)s %(levelname)s %(message)s')
    if a.compare:
        x, y = np.load(a.compare[0]), np.load(a.compare[1])
        for k in x.files:
            if k in y.files:
                print(f'{k:16s} cosine {float(np.dot(x[k], y[k])):.5f}')
        return 0
    raw = dict(load_config().get('reid') or {})
    raw.update(enabled=True, background=False)
    if a.providers:
        raw['providers'] = a.providers
    e = Embedder(ReidConfig.from_dict(raw))
    if not e.load():
        print('FAILED:', e.error)
        return 1
    print(f'provider {e.provider}; session + warm-up {e.load_s} s')
    img = cv2.imread(a.image) if a.image else None
    if img is None:
        print('no --image; nothing to embed')
        return 0
    boxes = {f'box{i}': b for i, b in enumerate(a.box)}
    if a.boxes_json:
        boxes.update(json.load(open(a.boxes_json)))
    names, bxs = list(boxes), list(boxes.values())
    vecs = e.batch(img, bxs)
    if a.save:
        np.savez(a.save, **{n: v for n, v in zip(names, vecs) if v is not None})
        print('saved', a.save)
    if a.bench:
        for size in (1, len(bxs)):
            ts = []
            for _ in range(a.bench):
                t = time.perf_counter()
                e.batch(img, bxs[:size])
                ts.append(1000 * (time.perf_counter() - t))
            med = float(np.median(ts))
            print(f'batch {size}: median {med:.2f} ms/call, {med / size:.2f} ms/crop '
                  f'(p90 {np.percentile(ts, 90):.2f} ms), preprocessing included')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
