# Offline fallback check: rank the same 8 degraded crops by MobileCLIP2-S0 / ViT-B-32 text similarity ("a photo of a <q>").
import glob, os, torch, open_clip, cv2
from PIL import Image
from timm.utils.model import reparameterize_model
torch.set_grad_enabled(False)
files = sorted(glob.glob("crops_degr/*.jpg"), key=lambda p: int(os.path.basename(p).split("_")[0]))
truth = [os.path.basename(p).split("_", 1)[1][:-4] for p in files]
Q = {"wallet": "wallet", "keys": "set of house keys", "phone": "phone", "glasses": "reading glasses", "key_fob": "car key fob",
     "camera": "camera", "pen": "pen", "watch": "wristwatch"}
for name, pt in [("MobileCLIP2-S0", "dfndr2b"), ("ViT-B-32", "laion2b_s34b_b79k")]:
    m, _, pre = open_clip.create_model_and_transforms(name, pretrained=pt); m.eval()
    if "Mobile" in name: m = reparameterize_model(m)
    tok = open_clip.get_tokenizer(name)
    T = torch.nn.functional.normalize(m.encode_text(tok([f"a top-down photo of a {q} on a desk" for q in Q.values()])), dim=-1)
    I = torch.nn.functional.normalize(m.encode_image(torch.stack([pre(Image.open(p).convert("RGB")) for p in files])), dim=-1)
    S = T @ I.T
    top1 = sum(truth[int(S[i].argmax())] == o for i, o in enumerate(Q)); top2 = sum(truth.index(o) in S[i].topk(2).indices.tolist() for i, o in enumerate(Q))
    print(f"RESULT {name}: top1 {top1}/8, top2 {top2}/8")
