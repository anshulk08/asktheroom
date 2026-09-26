# Inside askroom:latest: time re-ID embedders on the Orin with onnxruntime (TRT EP fp16, CUDA EP) at batch 1 and 8.
import sys, time, os, numpy as np, onnxruntime as ort
os.chdir("/askroom/experiments/openset/weights")
name, sz = sys.argv[1], int(sys.argv[2])
provs = {
  "trt_fp16": [("TensorrtExecutionProvider", {"trt_fp16_enable": True, "trt_engine_cache_enable": True,
               "trt_engine_cache_path": "/askroom/experiments/openset/weights/trtcache",
               "trt_profile_min_shapes": f"images:1x3x{sz}x{sz}", "trt_profile_max_shapes": f"images:8x3x{sz}x{sz}",
               "trt_profile_opt_shapes": f"images:8x3x{sz}x{sz}"}), "CUDAExecutionProvider"],
  "cuda_fp32": ["CUDAExecutionProvider"],
}
ref = None
for k, p in provs.items():
    try:
        t0 = time.time(); s = ort.InferenceSession(f"{name}.onnx", providers=p); tb = time.time() - t0
        for b in (1, 8):
            x = np.random.RandomState(0).rand(b, 3, sz, sz).astype(np.float32)
            for _ in range(5): o = s.run(None, {"images": x})[0]
            ts = []
            for _ in range(40):
                t = time.perf_counter(); o = s.run(None, {"images": x})[0]; ts.append((time.perf_counter() - t) * 1000)
            if b == 8:
                if ref is None: ref = o
                cos = float((o * ref).sum(1).min())
            print(f"RESULT {name} {k} batch{b}: median {np.median(ts):.1f} ms/batch, {np.median(ts)/b:.1f} ms/crop (p90 {np.percentile(ts,90):.1f}); session build {tb:.0f}s; providers {s.get_providers()[:1]}"
                  + (f"; min cos vs first provider {cos:.4f}" if b == 8 else ""), flush=True)
    except Exception as e:
        print("FAIL", name, k, repr(e)[:400], flush=True)
