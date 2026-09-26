# Spurious-box check on whatever the live camera shows (desk absent 21:48-22:30): overlay masked vs unmasked.
import cv2, glob, numpy as np
from ultralytics import YOLOE
m = YOLOE("weights/yoloe-26s-seg-pf.pt")
fs = sorted(glob.glob("frames/live_*.jpg")); pick = [fs[0], fs[5], fs[len(fs)//2], fs[-1]]
for f in pick:
    raw = cv2.imread(f); H, W = raw.shape[:2]
    for masked in (False, True):
        im = raw.copy()
        if masked: im[:175, :240] = np.median(im[175:].reshape(-1, 3), 0)
        r = m.predict(im, imgsz=640, conf=0.10, agnostic_nms=True, iou=0.5, device="mps", verbose=False)[0]
        B = r.boxes.xyxy.cpu().numpy(); S = r.boxes.conf.cpu().numpy()
        def in_ovl(b):
            iw, ih = min(b[2], 240) - b[0], min(b[3], 175) - b[1]
            return iw > 0 and ih > 0 and iw * ih > 0.5 * (b[2]-b[0]) * (b[3]-b[1])
        ovl = sum(in_ovl(b) for b in B); big = sum((b[2]-b[0])*(b[3]-b[1]) > 0.35*W*H for b in B)
        top = ", ".join(f"{m.names[int(c)]}:{s:.2f}" for c, s in sorted(zip(r.boxes.cls.tolist(), S), key=lambda t: -t[1])[:5])
        print(f"{f[-10:-4]} masked={masked}: boxes>=.10 {len(B)} (>=.25 {(S>=.25).sum()}), in-overlay {ovl}, scene-sized {big}; top: {top}")
        cv2.imwrite(f"out/live_{f[-10:-4]}_{'masked' if masked else 'unmasked'}_c10.jpg", r.plot(line_width=1, font_size=8))
