# Export candidate re-ID embedders (image tower only) to ONNX, batch-dynamic, opset 17.
import torch, open_clip, sys, time
torch.set_grad_enabled(False)
class Norm(torch.nn.Module):
    def __init__(s, m): super().__init__(); s.m = m
    def forward(s, x): y = s.m(x); return y / y.norm(dim=-1, keepdim=True)
models = {}
m, _, pre = open_clip.create_model_and_transforms("MobileCLIP2-S0", pretrained="dfndr2b"); m.eval()
print("MobileCLIP2-S0 preprocess:", pre)
models["mobileclip2_s0"] = (Norm(m.visual), pre.transforms[0].size)
m, _, pre = open_clip.create_model_and_transforms("ViT-B-32", pretrained="laion2b_s34b_b79k"); m.eval()
models["vitb32"] = (Norm(m.visual), 224)
d = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14").eval()
models["dinov2_s14"] = (Norm(d), 224)
for name, (mod, sz) in models.items():
    sz = sz if isinstance(sz, int) else sz[0]
    x = torch.randn(1, 3, sz, sz)
    f = f"weights/{name}.onnx"
    try:
        torch.onnx.export(mod, x, f, input_names=["images"], output_names=["emb"], opset_version=17,
                          dynamic_axes={"images": {0: "b"}, "emb": {0: "b"}}, dynamo=False)
        import onnxruntime as ort
        s = ort.InferenceSession(f); o = s.run(None, {"images": x.numpy()})[0]
        err = float((torch.from_numpy(o) - mod(x)).abs().max())
        print("OK", name, sz, o.shape, "max abs diff vs torch %.2e" % err)
    except Exception as e:
        print("FAIL", name, repr(e)[:300])
