# YOLOE visual prompts vs text prompts for finding a specific object in another frame.
# usage: python vp_eval.py gt.json REF_FRAME TARGET_FRAME [synthetic]
#   gt.json: {"frames/x.jpg": {"wallet": [x1,y1,x2,y2], ...}}   ids shared across frames
# For each object with a box in REF: visual prompt = that box on REF, predict on TARGET.
# 'synthetic': TARGET := REF with the object's crop rotated 90deg + pasted at a new desk location (original left in place,
#   so there are two copies; we report whether each copy is found).
import sys, json, cv2, numpy as np
from ultralytics import YOLOE
from ultralytics.models.yolo.yoloe import YOLOEVPSegPredictor
gt = json.load(open(sys.argv[1])); ref_f, tgt_f = sys.argv[2], sys.argv[3]; synth = len(sys.argv) > 4
OVL = (240, 175)
def clean(im, f=""):
    im = im.copy()
    if "live" in f: im[:OVL[1], :OVL[0]] = np.median(im[OVL[1]:].reshape(-1, 3), 0)
    return im
def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    I = ix * iy; return I / ((a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - I + 1e-9)
ref = clean(cv2.imread(ref_f), ref_f); tgt = clean(cv2.imread(tgt_f), tgt_f)
refgt, tgtgt = gt[ref_f], dict(gt.get(tgt_f, {}))
dev = "mps"
vp = YOLOE("weights/yoloe-26s-seg.pt")
for oid, b in refgt.items():
    t = tgt.copy(); tg = dict(tgtgt)
    if synth:
        x1, y1, x2, y2 = map(int, b); c = np.ascontiguousarray(np.rot90(ref[y1:y2, x1:x2]))
        h, w = c.shape[:2]; H, W = t.shape[:2]
        # paste into the emptiest desk region: pick position with lowest overlap to all gt boxes
        best = None
        for py in range((OVL[1] if "live" in ref_f else 0) + 5, H - h - 5, 10):
            for px in range(5, W - w - 5, 10):
                cand = [px, py, px + w, py + h]
                ov = sum(iou(cand, bb) for bb in refgt.values())
                if best is None or ov < best[0]: best = (ov, cand)
        px, py = best[1][:2]; t[py:py+h, px:px+w] = c; tg = {oid: b, oid + "_pasted": best[1]}
        cv2.imwrite(f"out/vp_synth_{oid}.jpg", t)
    r = vp.predict(t, refer_image=ref, visual_prompts=dict(bboxes=np.array([b], dtype=np.float32), cls=np.array([0])),
                   predictor=YOLOEVPSegPredictor, conf=0.05, device=dev, verbose=False)[0]
    dets = sorted([(float(s), [float(v) for v in bb]) for bb, s in zip(r.boxes.xyxy.cpu(), r.boxes.conf.cpu())], key=lambda d: -d[0])
    msg = []
    for gid, gb in tg.items():
        if not (gid == oid or gid == oid + "_pasted"): continue
        m = [(s, iou(bb, gb)) for s, bb in dets if iou(bb, gb) > 0.3]
        rank = next((i for i, (s, bb) in enumerate(dets) if iou(bb, gb) > 0.3), None)
        msg.append(f"{gid}: " + (f"found conf {m[0][0]:.2f} IoU {m[0][1]:.2f} rank {rank}" if m else "MISSED"))
    distract = [s for s, bb in dets if all(iou(bb, gb) <= 0.3 for gid, gb in tg.items() if gid.startswith(oid))]
    print(f"VP {oid}: {'; '.join(msg) or 'no target gt'} | n_dets {len(dets)}, best distractor {max(distract, default=0):.2f}", flush=True)
    out = r.plot(line_width=1, font_size=9); cv2.imwrite(f"out/vp_{'synth' if synth else 'xframe'}_{oid}_pred.jpg", out)
# text prompts on the same target (non-synthetic only)
if not synth:
    tp = YOLOE("weights/yoloe-26s-seg.pt"); names = list(refgt.keys()); tp.set_classes([n.replace("_", " ") for n in names])
    r = tp.predict(tgt, conf=0.01, device=dev, verbose=False)[0]
    for i, n in enumerate(names):
        dd = [(float(s), [float(v) for v in bb]) for bb, s, c in zip(r.boxes.xyxy.cpu(), r.boxes.conf.cpu(), r.boxes.cls.cpu()) if int(c) == i]
        g = tgtgt.get(n); hit = [s for s, bb in dd if g and iou(bb, g) > 0.3]
        print(f"TEXT '{n}': best conf on true object {max(hit, default=0):.2f}; best conf elsewhere {max([s for s, bb in dd if not g or iou(bb, g) <= 0.3], default=0):.2f}")
    cv2.imwrite("out/text_prompt_pred.jpg", r.plot(line_width=1, font_size=9))
