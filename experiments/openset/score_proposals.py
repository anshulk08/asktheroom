# Score YOLOE-pf as a class-agnostic proposer against hand GT boxes.
# usage: python score_proposals.py gt.json [scale_suffix:factor ...]  e.g. _degraded:0.5 adds degraded variants
# A GT object is "found" at threshold t if some box (conf>=t) has IoU>=0.5. Other boxes are classified as
#   dup/part (>=70% of the box lies inside a GT object), scene (box area > 35% of image) or spurious (everything else).
import sys, json, cv2, numpy as np
from ultralytics import YOLOE
gt = json.load(open(sys.argv[1]))
for extra in sys.argv[2:]:
    suf, fac = extra.split(":"); fac = float(fac)
    for f, objs in list(gt.items()):
        if f.endswith(".jpg") and "_degraded" not in f:
            g = f.replace(".jpg", suf + ".jpg")
            gt[g] = {k: [v * fac for v in b] for k, b in objs.items()}
OVL = (240, 175)
def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    I = ix * iy; return I / ((a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - I + 1e-9)
def inside(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    return ix * iy / ((a[2]-a[0])*(a[3]-a[1]) + 1e-9)
rows = []
for k in ("yoloe-26s-seg-pf", "yoloe-11s-seg-pf"):
    m = YOLOE(f"weights/{k}.pt")
    for f, objs in gt.items():
        im = cv2.imread(f); H, W = im.shape[:2]
        if "live" in f: im[:OVL[1], :OVL[0]] = np.median(im[OVL[1]:].reshape(-1, 3), 0)
        for agn in (False, True):
            r = m.predict(im, imgsz=640, conf=0.03, device="mps", verbose=False, max_det=300, agnostic_nms=agn, iou=0.5)[0]
            B = r.boxes.xyxy.cpu().numpy().tolist(); S = r.boxes.conf.cpu().numpy().tolist()
            if agn:
                for th in (0.10, 0.25):
                    rr = r[(r.boxes.conf >= th).cpu().numpy()]
                    cv2.imwrite(f"out/score_{f.split('/')[-1][:-4]}_{k}_agn_c{int(th*100):02d}.jpg", rr.plot(line_width=1, font_size=8, labels=True))
            for th in (0.05, 0.10, 0.25, 0.40):
                bb = [b for b, s in zip(B, S) if s >= th]
                found = {o: max([iou(b, g) for b in bb], default=0) >= 0.5 for o, g in objs.items()}
                cat = {"match": 0, "dup/part": 0, "scene": 0, "spurious": 0}
                for b in bb:
                    if max(iou(b, g) for g in objs.values()) >= 0.5: cat["match"] += 1
                    elif (b[2]-b[0])*(b[3]-b[1]) > 0.35 * W * H: cat["scene"] += 1
                    elif max(inside(b, g) for g in objs.values()) >= 0.7: cat["dup/part"] += 1
                    else: cat["spurious"] += 1
                rows.append(dict(model=k, frame=f.split("/")[-1], agnostic_nms=agn, conf=th, recall=f"{sum(found.values())}/{len(found)}",
                                 missed=[o for o, v in found.items() if not v], boxes=len(bb), **cat))
                print(json.dumps(rows[-1]), flush=True)
json.dump(rows, open("out/score_" + sys.argv[1].replace(".json", "") + ".json", "w"), indent=1)
