"""Export MobileCLIP2-S0 (image and text towers) to ONNX for the visual archive (core/visual_memory.py).

    python scripts/export_mobileclip.py [--out models]      # needs torch, open_clip_torch, timm, onnxruntime

Writes models/mobileclip2_s0_image.onnx (images [b,3,256,256] RGB 0-1 -> unit embedding),
models/mobileclip2_s0_text.onnx (tokens [b,77] int64 -> unit embedding) and the BPE vocabulary
(models/bpe_simple_vocab_16e6.txt.gz) for core/clip_tokenizer.py, then checks ONNX against torch and our
tokenizer against open_clip's. MobileCLIP2 is reparameterized (timm) before export: its train-time
multi-branch blocks fold into single convolutions, which is what makes the ONNX / TensorRT graph small.
Batch is dynamic; opset 17. On the Jetson, build engines from these with trtexec as for the other models.
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="models")
    a = ap.parse_args(argv)
    import numpy as np
    import onnxruntime as ort
    import open_clip
    import torch
    from timm.utils.model import reparameterize_model

    torch.set_grad_enabled(False)
    torch.backends.mha.set_fastpath_enabled(False)   # the fused attention kernel has no ONNX export
    m, _, _ = open_clip.create_model_and_transforms("MobileCLIP2-S0", pretrained="dfndr2b")
    m = reparameterize_model(m.eval())

    class Image(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, images):
            return torch.nn.functional.normalize(self.m.encode_image(images), dim=-1)

    class Text(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, tokens):
            return torch.nn.functional.normalize(self.m.encode_text(tokens), dim=-1)

    os.makedirs(a.out, exist_ok=True)
    tok = open_clip.get_tokenizer("MobileCLIP2-S0")
    img_path = os.path.join(a.out, "mobileclip2_s0_image.onnx")
    txt_path = os.path.join(a.out, "mobileclip2_s0_text.onnx")
    x = torch.rand(2, 3, 256, 256)
    t = tok(["a photo of a red mug", "keys on a table"])
    torch.onnx.export(Image(m).eval(), x, img_path, input_names=["images"], output_names=["emb"],
                      opset_version=17, dynamic_axes={"images": {0: "b"}, "emb": {0: "b"}}, dynamo=False)
    torch.onnx.export(Text(m).eval(), t, txt_path, input_names=["tokens"], output_names=["emb"],
                      opset_version=17, dynamic_axes={"tokens": {0: "b"}, "emb": {0: "b"}}, dynamo=False)
    vocab = os.path.join(os.path.dirname(open_clip.__file__), "bpe_simple_vocab_16e6.txt.gz")
    shutil.copy(vocab, os.path.join(a.out, "bpe_simple_vocab_16e6.txt.gz"))

    for path, inp, name, ref in ((img_path, x, "images", Image(m)(x)), (txt_path, t, "tokens", Text(m)(t))):
        s = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        t0 = time.perf_counter()
        o = s.run(None, {name: inp.numpy()})[0]
        ms = (time.perf_counter() - t0) * 1000
        print(f"{os.path.basename(path)}: {o.shape}, max |onnx - torch| {np.abs(o - ref.numpy()).max():.2e}, "
              f"{ms:.0f} ms for a batch of {len(o)} on CPU, {os.path.getsize(path) / 1e6:.1f} MB")

    from core.clip_tokenizer import ClipTokenizer
    ours = ClipTokenizer(os.path.join(a.out, "bpe_simple_vocab_16e6.txt.gz"))
    probe = ["a photo of a red mug", "Grandma's ring & keys!", "what does the note say? 3:30 pm", "pill bottle"]
    same = (ours(probe) == tok(probe).numpy()).all()
    print("tokenizer matches open_clip:", bool(same))
    return 0 if same else 1


if __name__ == "__main__":
    raise SystemExit(main())
