"""core/embed.py: the re-ID crop embedder behind World(embed=...). No GPU needed: the session is
faked, or a tiny ONNX model runs on onnxruntime's CPU provider."""
import threading

import numpy as np
import pytest

from core import embed
from core.embed import Embedder, ReidConfig, make_embedder, provider_chain


class FakeSession:
    """Stands in for an onnxruntime session: returns one known vector per input crop."""

    def __init__(self, dim=6, providers=("CPUExecutionProvider",)):
        self.calls = []
        self.dim = dim
        self.providers = list(providers)

    def get_inputs(self):
        class I:
            name = "images"
        return [I()]

    def get_providers(self):
        return self.providers

    def run(self, outputs, feed):
        x = feed["images"]
        self.calls.append(x.copy())
        out = np.zeros((len(x), self.dim), np.float32)
        out[:, 0] = x.mean(axis=(1, 2, 3)) + 3.0     # depends on the crop, never zero
        out[:, 1] = 4.0
        return [out]


def cfg(**kw):
    c = dict(enabled=True, model="unused.onnx", input_size=32, margin=0.1, min_px=8, max_batch=4)
    c.update(kw)
    return ReidConfig.from_dict(c)


@pytest.fixture
def img():
    rng = np.random.RandomState(0)
    return rng.randint(0, 255, (120, 200, 3), np.uint8)


def test_returns_unit_float32_vector(img):
    e = Embedder(cfg(), session=FakeSession())
    v = e(img, (20, 30, 80, 90))
    assert v.dtype == np.float32 and v.shape == (6,)
    assert abs(float(np.linalg.norm(v)) - 1.0) < 1e-5


def test_empty_offscreen_tiny_or_missing_image_is_none(img):
    s = FakeSession()
    e = Embedder(cfg(), session=s)
    assert e(None, (0, 0, 50, 50)) is None
    assert e(img, (50, 50, 50, 80)) is None               # zero width
    assert e(img, (300, 10, 360, 60)) is None             # right of the frame
    assert e(img, (-80, -80, -10, -10)) is None           # above and left of it
    assert e(img, (10, 10, 14, 40)) is None               # under min_px even with the margin
    assert e(img, (np.nan, 0, 10, 10)) is None
    assert s.calls == []                                  # nothing reached the model


def test_preprocess_crops_with_margin_and_normalises(img):
    e = Embedder(cfg(input_size=16, margin=0.0, fit="stretch"), session=FakeSession())
    flat = np.zeros((40, 40, 3), np.uint8)
    flat[:] = (0, 128, 255)                                # BGR
    x = e.preprocess(flat, (10, 10, 30, 30))
    assert x.shape == (3, 16, 16) and x.dtype == np.float32
    rgb = np.array([255, 128, 0]) / 255.0
    want = (rgb - embed.MEAN) / embed.STD
    assert np.allclose(x.mean(axis=(1, 2)), want, atol=1e-5)   # RGB order, ImageNet mean / std


def test_margin_grows_the_crop_but_stays_in_the_frame():
    e = Embedder(cfg(margin=0.25), session=FakeSession())
    assert e.crop_box((100, 100), (40, 40, 60, 60)) == (35, 35, 65, 65)
    assert e.crop_box((100, 100), (0, 0, 20, 20)) == (0, 0, 25, 25)       # clipped at the edge


def test_letterbox_keeps_aspect_and_pads_with_the_mean():
    e = Embedder(cfg(input_size=16, margin=0.0, fit="letterbox"), session=FakeSession())
    tall = np.full((80, 40, 3), 255, np.uint8)
    x = e.preprocess(tall, (0, 0, 20, 80))                 # 20 x 80: 4 px wide at 16 tall
    assert np.allclose(x[:, :, 0], 0.0, atol=1e-6)         # left pad column is the mean -> 0
    assert np.all(x[:, :, 8] > 1.0)                        # the white object in the middle


def test_batch_one_run_for_all_valid_boxes(img):
    s = FakeSession()
    e = Embedder(cfg(), session=s)
    out = e.batch(img, [(20, 30, 80, 90), (500, 500, 600, 600), (100, 10, 180, 100)])
    assert out[1] is None
    assert out[0] is not None and out[2] is not None
    assert len(s.calls) == 1 and s.calls[0].shape[0] == 2
    single = e(img, (20, 30, 80, 90))
    assert np.allclose(single, out[0], atol=1e-6)          # batched == one at a time


def test_batch_is_split_at_max_batch(img):
    s = FakeSession()
    e = Embedder(cfg(max_batch=2), session=s)
    out = e.batch(img, [(10 + 10 * i, 10, 60 + 10 * i, 60) for i in range(5)])
    assert all(v is not None for v in out)
    assert [len(c) for c in s.calls] == [2, 2, 1]


def test_a_failing_model_gives_none_not_an_exception(img):
    class Broken(FakeSession):
        def run(self, outputs, feed):
            raise RuntimeError("cuda gone")
    e = Embedder(cfg(), session=Broken())
    assert e(img, (20, 30, 80, 90)) is None
    assert e.batch(img, [(20, 30, 80, 90)]) == [None]


def test_provider_chain_orders_and_configures_tensorrt(tmp_path):
    c = cfg(providers=["tensorrt", "cuda", "cpu"], cache_dir=str(tmp_path / "trt"), input_size=224, max_batch=8)
    avail = ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]
    chains = provider_chain(c, avail)
    assert [[p if isinstance(p, str) else p[0] for p in ch] for ch in chains] == [
        ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"],
        ["CUDAExecutionProvider", "CPUExecutionProvider"],
        ["CPUExecutionProvider"]]
    name, opts = chains[0][0]
    assert opts["trt_fp16_enable"] is True and opts["trt_engine_cache_enable"] is True
    assert opts["trt_engine_cache_path"] == str(tmp_path / "trt")
    assert opts["trt_profile_min_shapes"] == "images:1x3x224x224"
    assert opts["trt_profile_max_shapes"] == "images:8x3x224x224"


def test_provider_chain_skips_what_this_onnxruntime_lacks():
    chains = provider_chain(cfg(providers=["tensorrt", "cuda", "cpu"]), ["CPUExecutionProvider"])
    assert chains == [["CPUExecutionProvider"]]
    assert provider_chain(cfg(providers=["tensorrt"]), ["CPUExecutionProvider"]) == []


def test_load_falls_back_down_the_chain(monkeypatch, tmp_path):
    tried = []

    def factory(path, providers):
        first = providers[0] if isinstance(providers[0], str) else providers[0][0]
        tried.append(first)
        if first != "CPUExecutionProvider":
            raise RuntimeError(f"{first} failed")
        return FakeSession(dim=384, providers=[first])

    c = cfg(providers=["tensorrt", "cuda", "cpu"], input_size=16)
    avail = ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]
    e = Embedder(c, session_factory=factory, available=avail)
    e.load()
    assert tried == ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]
    assert e.provider == "CPUExecutionProvider" and e.ready


def test_background_load_returns_none_until_ready(img):
    gate = threading.Event()

    def factory(path, providers):
        gate.wait(5)
        return FakeSession()

    e = Embedder(cfg(providers=["cpu"]), session_factory=factory, available=["CPUExecutionProvider"])
    e.start()
    assert e(img, (20, 30, 80, 90)) is None                # still loading: no appearance, no wait
    gate.set()
    assert e.wait_ready(5)
    assert e(img, (20, 30, 80, 90)) is not None


def test_nothing_loads_when_no_provider_is_allowed(img):
    e = Embedder(cfg(providers=["tensorrt"]), available=["CPUExecutionProvider"])
    e.load()
    assert not e.ready and e.error
    assert e(img, (20, 30, 80, 90)) is None


# ----- make_embedder: the one line main.py calls

def _tiny_onnx(path, size=16, dim=8):
    """images [b, 3, s, s] -> global average pool -> linear -> [b, dim]."""
    onnx = pytest.importorskip("onnx")
    from onnx import TensorProto, helper, numpy_helper
    w = np.random.RandomState(1).randn(3, dim).astype(np.float32)
    g = helper.make_graph(
        [helper.make_node("ReduceMean", ["images"], ["pooled"], axes=[2, 3], keepdims=0),
         helper.make_node("MatMul", ["pooled", "w"], ["emb"])],
        "tiny", [helper.make_tensor_value_info("images", TensorProto.FLOAT, ["b", 3, size, size])],
        [helper.make_tensor_value_info("emb", TensorProto.FLOAT, ["b", dim])],
        [numpy_helper.from_array(w, "w")])
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)])
    m.ir_version = 8
    onnx.save(m, str(path))
    return path


def test_make_embedder_off_or_missing_is_none(tmp_path):
    assert make_embedder({}) is None
    assert make_embedder({"reid": {"enabled": False, "model": "x.onnx"}}) is None
    assert make_embedder({"reid": {"enabled": True, "model": str(tmp_path / "missing.onnx")}}) is None


def test_make_embedder_runs_a_real_onnx_on_cpu(tmp_path, img):
    pytest.importorskip("onnxruntime")
    path = _tiny_onnx(tmp_path / "tiny.onnx")
    e = make_embedder({"reid": {"enabled": True, "model": str(path), "providers": ["cpu"],
                                "input_size": 16, "background": False}})
    assert e is not None and e.ready and e.provider == "CPUExecutionProvider"
    v = e(img, (20, 30, 80, 90))
    assert v.shape == (8,) and abs(float(np.linalg.norm(v)) - 1) < 1e-5
    b = e.batch(img, [(20, 30, 80, 90), (100, 10, 180, 100)])
    assert np.allclose(b[0], v, atol=1e-5)


def test_world_uses_it_as_its_appearance_source(tmp_path):
    """The World calls embed(frame_img, box_px) and treats None as 'no appearance'."""
    from core.world import World
    from core.config import load_config
    e = Embedder(cfg(), session=FakeSession())
    w = World(load_config(), None, embed=e)
    w._frame = type("F", (), {"img": np.full((100, 100, 3), 90, np.uint8)})()
    v = w._embed((10, 10, 60, 60))
    assert v is not None and abs(float(np.linalg.norm(v)) - 1) < 1e-5
    assert w._embed((500, 500, 600, 600)) is None
