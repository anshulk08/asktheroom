# Inside askroom:latest: build FP16 TRT engine from a Mac-exported YOLOE ONNX (avoids torch CUDA alloc), then time it.
import sys, time, ast, os, traceback
import numpy as np, onnx
os.chdir("/askroom/experiments/openset/weights")
from ultralytics.utils.export.engine import onnx2engine
f = sys.argv[1]; out = f.replace(".onnx", ".engine")
meta = {p.key: p.value for p in onnx.load(f, load_external_data=False).metadata_props}
for k in ("stride", "batch", "imgsz", "names", "kpt_shape", "args", "channels", "end2end"):
    if k in meta:
        try: meta[k] = ast.literal_eval(meta[k])
        except Exception: pass
print("meta names:", len(meta.get("names", {})), "task:", meta.get("task"))
t0 = time.time()
if not os.path.exists(out):
    try:
        onnx2engine(f, out, workspace=2, quantize=16, dynamic=False, shape=(1, 3, 640, 640), metadata=meta)
        print("ENGINE_OK %s in %.0f s, %.1f MB" % (out, time.time() - t0, os.path.getsize(out) / 1e6), flush=True)
    except Exception as e:
        traceback.print_exc(); print("ENGINE_FAIL", repr(e)); sys.exit(1)
from ultralytics import YOLO
e = YOLO(out, task="segment")
img = (np.random.rand(540, 960, 3) * 255).astype(np.uint8)
if len(sys.argv) > 2:
    import cv2; img = cv2.imread(sys.argv[2])
for _ in range(10): e.predict(img, imgsz=640, verbose=False, conf=0.1)
ts, inf, pre, post = [], [], [], []
for _ in range(60):
    t = time.perf_counter(); r = e.predict(img, imgsz=640, verbose=False, conf=0.1)
    ts.append((time.perf_counter() - t) * 1000); s = r[0].speed
    inf.append(s["inference"]); pre.append(s["preprocess"]); post.append(s["postprocess"])
print("RESULT %s: end2end median %.1f ms (p90 %.1f) | inference median %.1f | pre %.1f | post %.1f | boxes %d" % (
    out, np.median(ts), np.percentile(ts, 90), np.median(inf), np.median(pre), np.median(post), len(r[0].boxes)))
