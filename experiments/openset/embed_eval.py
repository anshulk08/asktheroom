# Re-ID embedding comparison on overhead crops.
# usage: python embed_eval.py gt.json      gt.json = {"frames/x.jpg": {"wallet": [x1,y1,x2,y2], ...}, ...}
# Positives: same instance across frames + synthetic views (rot 90/180/35deg, +-12% box jitter, 0.7x brightness).
# Negatives: different instances. Reports min-pos, max-neg, margin, mean pos/neg, rank-1 (per query, best
# match among all other-instance crops vs same-instance crops), and ms/crop on the Mac.
import sys, json, time, itertools, cv2, numpy as np, torch, open_clip
from timm.utils.model import reparameterize_model
from PIL import Image
torch.set_grad_enabled(False)
dev = "mps" if torch.backends.mps.is_available() else "cpu"
gt = json.load(open(sys.argv[1]))
import os
EXCL = set(filter(None, os.environ.get("EXCLUDE", "").split(",")))
gt = {f: {k: b for k, b in o.items() if k not in EXCL} for f, o in gt.items()}
rng = np.random.RandomState(0)

def rot(img, ang):
    h, w = img.shape[:2]; M = cv2.getRotationMatrix2D((w/2, h/2), ang, 1.0)
    c, s = abs(M[0, 0]), abs(M[0, 1]); nw, nh = int(h*s + w*c), int(h*c + w*s)
    M[0, 2] += nw/2 - w/2; M[1, 2] += nh/2 - h/2
    return cv2.warpAffine(img, M, (nw, nh), borderMode=cv2.BORDER_REFLECT)
def crop(im, b, pad=0.08, jit=0.0):
    x1, y1, x2, y2 = b; w, h = x2-x1, y2-y1
    j = lambda: rng.uniform(-jit, jit)
    x1 += (j()-pad)*w; x2 += (j()+pad)*w; y1 += (j()-pad)*h; y2 += (j()+pad)*h
    H, W = im.shape[:2]
    return im[max(0, int(y1)):min(H, int(y2)), max(0, int(x1)):min(W, int(x2))].copy()

crops = []  # (instance, tag, bgr)
for f, objs in gt.items():
    im = cv2.imread(f)
    for oid, b in objs.items():
        base = crop(im, b)
        crops.append((oid, f"{f}:orig", base))
        crops.append((oid, f"{f}:rot90", np.ascontiguousarray(np.rot90(base))))
        crops.append((oid, f"{f}:rot180", np.ascontiguousarray(np.rot90(base, 2))))
        crops.append((oid, f"{f}:rot35", rot(base, 35)))
        crops.append((oid, f"{f}:jit", crop(im, b, jit=0.12)))
        crops.append((oid, f"{f}:dark", (base*0.7).astype(np.uint8)))
ids = [c[0] for c in crops]; print(len(crops), "crops,", len(set(ids)), "instances:", sorted(set(ids)))
for oid in set(ids):
    cv2.imwrite(f"out/crop_{oid}.jpg", [c[2] for c in crops if c[0] == oid][0])

def hsv_hist(bgr):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h = cv2.calcHist([hsv], [0, 1, 2], None, [16, 8, 4], [0, 180, 0, 256, 0, 256]).ravel()
    h = np.sqrt(h / h.sum()); return h / np.linalg.norm(h)   # cosine of sqrt-hist = Bhattacharyya coeff

embedders = {"hsv_hist": (lambda b: hsv_hist(b), None)}
def clipper(name, pt):
    m, _, pre = open_clip.create_model_and_transforms(name, pretrained=pt); m.eval()
    v = reparameterize_model(m.visual).eval().to(dev) if "Mobile" in name else m.visual.eval().to(dev)
    return v, pre
for key, (name, pt) in {"mobileclip2_s0": ("MobileCLIP2-S0", "dfndr2b"), "openclip_vitb32": ("ViT-B-32", "laion2b_s34b_b79k")}.items():
    embedders[key] = clipper(name, pt)
import torchvision.transforms as T
dino = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14", verbose=False).eval().to(dev)
embedders["dinov2_s14"] = (dino, T.Compose([T.Resize((224, 224)), T.ToTensor(), T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]))

report = {}
for key, (mod, pre) in embedders.items():
    t0 = time.perf_counter()
    if pre is None:
        E = np.stack([mod(c[2]) for c in crops]); ms = (time.perf_counter()-t0)*1000/len(crops); ms1 = ms
    else:
        X = torch.stack([pre(Image.fromarray(cv2.cvtColor(c[2], cv2.COLOR_BGR2RGB))) for c in crops]).to(dev)
        for _ in range(3): mod(X[:1]); mod(X[:8])
        if dev == "mps": torch.mps.synchronize()
        t = time.perf_counter()
        for _ in range(10): mod(X[:1])
        if dev == "mps": torch.mps.synchronize()
        ms1 = (time.perf_counter()-t)*100
        t = time.perf_counter(); E = mod(X)
        if dev == "mps": torch.mps.synchronize()
        ms = (time.perf_counter()-t)*1000/len(crops)
        E = torch.nn.functional.normalize(E.float(), dim=-1).cpu().numpy()
    S = E @ E.T; n = len(crops); same = np.array([[ids[i] == ids[j] for j in range(n)] for i in range(n)])
    off = ~np.eye(n, dtype=bool)
    # cross-frame positives only (same instance, different frame) - the realistic re-ID case
    fr = [c[1].split(":")[0] for c in crops]
    xf = np.array([[fr[i] != fr[j] for j in range(n)] for i in range(n)])
    pos, neg = S[same & off], S[~same]
    posx = S[same & xf] if (same & xf).any() else np.array([np.nan])
    r1 = np.mean([ids[np.argmax(np.where(off[i], S[i], -9))] == ids[i] for i in range(n)])
    # rank-1 using only originals from OTHER frames as the gallery (query = any view)
    gal = [j for j in range(n) if crops[j][1].endswith(":orig")]
    r1x = [ids[max((j for j in gal if fr[j] != fr[i]), key=lambda j: S[i, j])] == ids[i] for i in range(n)
           if any(fr[j] != fr[i] and ids[j] == ids[i] for j in gal)]
    from sklearn.metrics import roc_auc_score
    pxs = S[same & xf]; ngs = S[~same]
    auc = roc_auc_score(np.r_[np.ones(len(pxs)), np.zeros(len(ngs))], np.r_[pxs, ngs])
    report_extra = dict(auc_xframe=round(float(auc), 4), p05pos_minus_p95neg_xframe=round(float(np.percentile(pxs, 5) - np.percentile(ngs, 95)), 3),
                        p05_pos_xframe=round(float(np.percentile(pxs, 5)), 3), p95_neg=round(float(np.percentile(ngs, 95)), 3), p99_neg=round(float(np.percentile(ngs, 99)), 3))
    report[key] = dict(**report_extra, min_pos=float(pos.min()), mean_pos=float(pos.mean()), min_pos_xframe=float(np.nanmin(posx)),
                       mean_pos_xframe=float(np.nanmean(posx)), max_neg=float(neg.max()), mean_neg=float(neg.mean()),
                       margin=float(pos.min()-neg.max()), margin_xframe=float(np.nanmin(posx)-neg.max()),
                       rank1_all=float(r1), rank1_xframe_gallery=float(np.mean(r1x)) if r1x else None,
                       ms_per_crop_batched=round(ms, 2), ms_per_crop_b1=round(ms1, 2))
    # worst negative pair
    i, j = np.unravel_index(np.argmax(np.where(~same, S, -9)), S.shape)
    report[key]["hardest_neg"] = f"{crops[i][0]}~{crops[j][0]} ({crops[i][1].split(':')[1]}/{crops[j][1].split(':')[1]})"
    print(key, json.dumps(report[key]), flush=True)
json.dump(report, open(f"out/embed_{time.strftime('%H%M%S')}.json", "w"), indent=1)
