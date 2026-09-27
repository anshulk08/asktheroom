# Render YOLOE-pf masks for object-sized proposals only (drop scene-sized boxes > 35% of the frame), agnostic NMS.
import sys, cv2, numpy as np
from ultralytics import YOLOE
m = YOLOE("weights/yoloe-26s-seg-pf.pt"); th = float(sys.argv[1])
for f in sys.argv[2:]:
    im = cv2.imread(f); H, W = im.shape[:2]
    if "live" in f: im[:175, :240] = np.median(im[175:].reshape(-1, 3), 0)
    r = m.predict(im, imgsz=640, conf=th, agnostic_nms=True, iou=0.5, device="mps", verbose=False, retina_masks=True)[0]
    a = ((r.boxes.xyxy[:, 2]-r.boxes.xyxy[:, 0])*(r.boxes.xyxy[:, 3]-r.boxes.xyxy[:, 1])).cpu().numpy()
    rr = r[a < 0.35*W*H]
    out = im.copy(); rng = np.random.RandomState(0)
    if rr.masks is not None:
        for mk, b, s in zip(rr.masks.data.cpu().numpy(), rr.boxes.xyxy.cpu().numpy(), rr.boxes.conf.cpu().numpy()):
            col = rng.randint(60, 255, 3); out[mk > 0.5] = (0.5*out[mk > 0.5] + 0.5*col).astype(np.uint8)
            cv2.rectangle(out, tuple(map(int, b[:2])), tuple(map(int, b[2:])), col.tolist(), 1)
            cv2.putText(out, f"{s:.2f}", (int(b[0]), int(b[1])+10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
    o = f"out/masks_{f.split('/')[-1][:-4]}_c{int(th*100):02d}.jpg"; cv2.imwrite(o, out); print(o, len(rr), "object-sized proposals")
