# Graph surgery: replace YOLOE-pf output0 [1, 4+4585+32, 8400] by [1, 4+1+1+32, 8400] = boxes, max score, argmax class, mask coeffs.
# Makes the engine output ~120x smaller so host-side postprocess doesn't touch 4585x8400 scores.
import onnx, onnx.helper as h, numpy as np, sys
src, dst, nc = sys.argv[1], sys.argv[2], 4585
m = onnx.load(src); g = m.graph
out0 = g.output[0].name
C = lambda n, v: g.initializer.append(onnx.numpy_helper.from_array(np.array(v, dtype=np.int64), n))
C("s0", [0]); C("s4", [4]); C("s_c", [4 + nc]); C("s_end", [4 + nc + 32]); C("ax1", [1])
nodes = [
 h.make_node("Slice", [out0, "s0", "s4", "ax1"], ["boxes"]),
 h.make_node("Slice", [out0, "s4", "s_c", "ax1"], ["scores"]),
 h.make_node("Slice", [out0, "s_c", "s_end", "ax1"], ["mc"]),
 h.make_node("ReduceMax", ["scores"], ["smax"], axes=[1], keepdims=1),
 h.make_node("ArgMax", ["scores"], ["sarg_i"], axis=1, keepdims=1),
 h.make_node("Cast", ["sarg_i"], ["sarg"], to=onnx.TensorProto.FLOAT),
 h.make_node("Concat", ["boxes", "smax", "sarg", "mc"], ["output0_reduced"], axis=1)]
g.node.extend(nodes)
new_out = h.make_tensor_value_info("output0_reduced", onnx.TensorProto.FLOAT, [1, 38, 8400])
outs = [new_out] + list(g.output[1:]); del g.output[:]; g.output.extend(outs)
onnx.checker.check_model(m); onnx.save(m, dst)
import onnxruntime as ort
s = ort.InferenceSession(dst); x = np.random.rand(1, 3, 640, 640).astype(np.float32)
a = s.run(None, {"images": x}); f = ort.InferenceSession(src).run(None, {"images": x})[0]
print("shapes", [o.shape for o in a], "max-score match:", np.allclose(a[0][0, 4], f[0, 4:4+nc].max(0), atol=1e-5))
