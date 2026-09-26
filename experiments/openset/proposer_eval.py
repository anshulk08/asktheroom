# YOLOE prompt-free as a class-agnostic proposer on live frames.
# usage: python proposer_eval.py frame.jpg [frame2.jpg ...]
# Masks the overlay legend (top-left 240x175) with the frame's median colour, runs both -pf checkpoints,
# dumps every detection >= 0.03 to JSON and saves annotated images at conf 0.10 / 0.25.
import sys, json, time, os, cv2, numpy as np
from ultralytics import YOLOE
OVL = (0, 0, 240, 175)
def mask_overlay(im):
    im = im.copy(); med = np.median(im[OVL[3]:, :].reshape(-1, 3), 0).astype(np.uint8)
    im[OVL[1]:OVL[3], OVL[0]:OVL[2]] = med; return im
models = {k: YOLOE(f"weights/{k}.pt") for k in ("yoloe-26s-seg-pf", "yoloe-11s-seg-pf")}
dev = os.environ.get("DEV", "mps")
res = {}
for f in sys.argv[1:]:
    raw = cv2.imread(f); im = mask_overlay(raw); stem = os.path.splitext(os.path.basename(f))[0]
    for k, m in models.items():
        for _ in range(2): m.predict(im, imgsz=640, conf=0.03, device=dev, verbose=False)  # warm
        ts = []
        for _ in range(5):
            t = time.perf_counter(); r = m.predict(im, imgsz=640, conf=0.03, device=dev, verbose=False, max_det=300)[0]; ts.append((time.perf_counter()-t)*1000)
        dets = [dict(cls=m.names[int(c)], conf=round(float(s), 3), xyxy=[round(float(v)) for v in b])
                for b, s, c in zip(r.boxes.xyxy, r.boxes.conf, r.boxes.cls)]
        res[f"{stem}/{k}"] = dict(ms_median=round(float(np.median(ts)), 1), dets=dets)
        for th in (0.10, 0.25):
            keep = r.boxes.conf >= th
            rr = r[keep.cpu().numpy()] if keep.any() else r[:0]
            out = rr.plot(line_width=1, font_size=9, img=im.copy())
            cv2.imwrite(f"out/{stem}_{k}_c{int(th*100):02d}.jpg", out)
        print(f"{stem} {k}: {np.median(ts):.0f} ms ({dev}); n>=.05 {sum(d['conf']>=.05 for d in dets)}, >=.10 {sum(d['conf']>=.10 for d in dets)}, >=.25 {sum(d['conf']>=.25 for d in dets)}", flush=True)
        for d in sorted(dets, key=lambda d: -d["conf"])[:25]:
            print("   %.2f %-22s %s" % (d["conf"], d["cls"][:22], d["xyxy"]))
json.dump(res, open(f"out/proposer_{time.strftime('%H%M%S')}.json", "w"), indent=1)
