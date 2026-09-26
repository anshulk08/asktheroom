# Run inside askroom:latest container: export YOLOE prompt-free to TensorRT FP16 and time it.
import sys, time, os, traceback
os.chdir("/askroom/experiments/openset/weights")
from ultralytics import YOLOE, YOLO
import numpy as np
w = sys.argv[1]
t0 = time.time()
try:
    m = YOLOE(w)
    print("names:", len(m.names))
    eng = m.export(format="engine", half=True, imgsz=640, device=0, workspace=2)
    print("EXPORT_OK", eng, "in %.0f s" % (time.time() - t0), flush=True)
except Exception as e:
    traceback.print_exc(); print("EXPORT_FAIL", repr(e)); sys.exit(1)
e = YOLO(eng, task="segment")
img = (np.random.rand(540, 960, 3) * 255).astype(np.uint8)
for _ in range(10): e.predict(img, imgsz=640, verbose=False, conf=0.1)
ts = []; inf = []
for _ in range(50):
    t = time.perf_counter(); r = e.predict(img, imgsz=640, verbose=False, conf=0.1)
    ts.append((time.perf_counter() - t) * 1000); inf.append(r[0].speed["inference"])
print("END2END ms median %.1f p90 %.1f | inference-only median %.1f | pre %.1f post %.1f" % (
    np.median(ts), np.percentile(ts, 90), np.median(inf), r[0].speed["preprocess"], r[0].speed["postprocess"]))
