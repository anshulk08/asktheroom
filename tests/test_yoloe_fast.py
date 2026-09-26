"""core/yoloe_fast.py: postprocess for the reduced-head YOLOE export ([1, 4 + 1 + 1 + nm, 8400]: box
cx cy w h in letterboxed input px, max class score, argmax class, mask coefficients), on synthetic
tensors, plus the YOLOEProposer wiring with a fake runner."""
import numpy as np
import pytest

from core.proposals import YOLOEProposer
from core.yoloe_fast import ReducedYOLOE, is_reduced, letterbox, postprocess

N_ANCHORS = 50


def head(dets, nm=32, n=N_ANCHORS):
    """Raw reduced output [1, 6 + nm, n]: one anchor per (cx, cy, w, h, score, cls[, coeffs]) and
    the rest empty (score 0)."""
    out = np.zeros((1, 6 + nm, n), np.float32)
    for i, d in enumerate(dets):
        cx, cy, w, h, s, c = d[:6]
        out[0, :6, i] = (cx, cy, w, h, s, c)
        if len(d) > 6:
            out[0, 6:, i] = d[6]
    return out


# 1280 x 720 frame -> 640 x 360 inside 640 x 640: gain 0.5, 140 px of padding above and below
LB = dict(gain=0.5, pad=(0, 140))
SHAPE = (720, 1280)


def test_boxes_map_back_to_frame_pixels():
    r = postprocess(head([(320, 320, 100, 50, 0.9, 7)]), None, SHAPE, **LB)
    assert r.xyxy.shape == (1, 4)
    assert np.allclose(r.xyxy[0], [(270 - 0) / 0.5, (295 - 140) / 0.5, 370 / 0.5, (345 - 140) / 0.5])
    assert r.conf[0] == pytest.approx(0.9) and r.cls[0] == 7 and r.cls.dtype.kind == 'i'


def test_conf_threshold_and_class_agnostic_nms():
    dets = [(100, 200, 40, 40, 0.8, 1),       # kept
            (102, 201, 40, 40, 0.7, 2),       # same object, another class: suppressed (agnostic)
            (400, 300, 40, 40, 0.6, 1),       # elsewhere: kept
            (500, 300, 40, 40, 0.10, 3)]      # below conf
    r = postprocess(head(dets), None, SHAPE, conf=0.15, iou=0.5, **LB)
    assert list(np.round(r.conf.astype(float), 3)) == [0.8, 0.6]
    assert list(r.cls) == [1, 1]


def test_nms_keeps_boxes_that_only_touch():
    dets = [(100, 200, 40, 40, 0.8, 1), (125, 200, 40, 40, 0.7, 1)]   # IoU 15/65 ~ 0.23
    assert len(postprocess(head(dets), None, SHAPE, iou=0.5, **LB).conf) == 2
    assert len(postprocess(head(dets), None, SHAPE, iou=0.2, **LB).conf) == 1


def test_boxes_over_max_area_frac_of_the_frame_are_dropped():
    dets = [(320, 320, 640, 360, 0.95, 0),    # the whole table / frame
            (320, 320, 300, 200, 0.9, 1),     # 600 x 400 of 1280 x 720 = 26%: kept at 0.35
            (100, 200, 40, 40, 0.5, 2)]
    r = postprocess(head(dets), None, SHAPE, max_area_frac=0.35, **LB)
    assert list(r.cls) == [1, 2]
    r = postprocess(head(dets), None, SHAPE, max_area_frac=0.2, **LB)
    assert list(r.cls) == [2]


def test_max_det_keeps_the_most_confident():
    dets = [(40 + 60 * i, 300, 30, 30, 0.2 + 0.05 * i, i) for i in range(10)]
    r = postprocess(head(dets), None, SHAPE, max_det=3, **LB)
    assert list(r.cls) == [9, 8, 7]


def test_boxes_are_clipped_to_the_frame():
    r = postprocess(head([(10, 150, 40, 40, 0.9, 0)]), None, SHAPE, **LB)      # sticks out left and up
    x1, y1, x2, y2 = r.xyxy[0]
    assert x1 == 0 and y1 == 0 and x2 == pytest.approx(60) and y2 == pytest.approx(60)


def test_no_mask_rows_works():
    r = postprocess(head([(100, 200, 40, 40, 0.8, 1)], nm=0), None, SHAPE, **LB)
    assert len(r.conf) == 1 and r.masks is None


def test_empty_output():
    r = postprocess(head([]), None, SHAPE, **LB)
    assert r.xyxy.shape == (0, 4) and len(r.conf) == 0 and len(r.cls) == 0


def test_masks_from_coefficients_and_prototypes():
    """Prototype 0 is +10 over the left half of the 640 x 640 input and -10 on the right; a box
    across the middle whose coefficients pick prototype 0 gets a mask that is its left half."""
    protos = np.full((1, 32, 160, 160), -10.0, np.float32)
    protos[0, 0, :, :80] = 10.0
    coeff = np.zeros(32, np.float32)
    coeff[0] = 1.0
    r = postprocess(head([(320, 320, 200, 100, 0.9, 1, coeff)]), protos, SHAPE, masks=True, **LB)
    x1, y1, x2, y2 = (int(round(v)) for v in r.xyxy[0])
    m = r.masks[0]
    assert m.dtype == bool and m.shape == (y2 - y1, x2 - x1)          # box-sized
    w = m.shape[1]
    assert m[:, : w // 2 - 4].all() and not m[:, w // 2 + 4:].any()


def test_letterbox_matches_ultralytics():
    LetterBox = pytest.importorskip("ultralytics.data.augment").LetterBox
    rng = np.random.RandomState(0)
    for h, w in ((720, 1280), (459, 612), (480, 640), (1080, 1920)):
        img = rng.randint(0, 255, (h, w, 3), np.uint8)
        ours, gain, pad = letterbox(img, 640)
        ref = LetterBox((640, 640), auto=False, stride=32)(image=img)
        assert ours.shape == ref.shape == (640, 640, 3)
        assert np.array_equal(ours, ref)
        assert gain == pytest.approx(min(640 / h, 640 / w))


def test_is_reduced():
    assert is_reduced('models/yoloe-26s-seg-pf-reduced.engine')
    assert is_reduced('x/yoloe-26s-seg-pf-reduced.onnx')
    assert not is_reduced('models/yoloe-26s-seg-pf.engine')
    assert not is_reduced('models/yoloe-26s-seg-pf.pt')
    assert not is_reduced('models/reduced-ish.pt')                # .pt always goes through ultralytics


class FakeRunner:
    names = {0: 'cup', 1: 'person', 2: 'dining table', 3: 'charger'}

    def __init__(self, out, protos=None):
        self.out, self.protos, self.inputs = out, protos, []

    def __call__(self, x, want_protos):
        self.inputs.append(x)
        return self.out, (self.protos if want_protos else None)


def test_reduced_model_speaks_the_ultralytics_predict_api():
    run = FakeRunner(head([(320, 320, 100, 50, 0.9, 3), (100, 200, 40, 40, 0.5, 0)]))
    m = ReducedYOLOE(runner=run, imgsz=640)
    r = m.predict(np.zeros((720, 1280, 3), np.uint8), imgsz=640, conf=0.15, iou=0.5, agnostic_nms=True,
                  max_det=100, retina_masks=False, verbose=False)[0]
    assert run.inputs[0].shape == (1, 3, 640, 640) and run.inputs[0].dtype == np.float32
    assert r.names == run.names and r.masks is None
    assert r.boxes.xyxy.shape == (2, 4) and list(r.boxes.cls) == [3, 0]
    assert set(r.speed) == {'preprocess', 'inference', 'postprocess'}


def test_yoloe_proposer_uses_the_reduced_model():
    dets = [(345, 340, 45, 40, 0.6, 0),       # cup (1280x720 px ~ 600..690, 300..380): kept
            (345, 341, 43, 40, 0.5, 3),       # same object, another class: merged by NMS
            (100, 320, 200, 340, 0.9, 1),     # person: ignored class
            (320, 320, 640, 360, 0.8, 2)]     # the table: too big
    y = YOLOEProposer({'conf': 0.15}, model=ReducedYOLOE(runner=FakeRunner(head(dets))))
    props = y.propose(np.zeros((720, 1280, 3), np.uint8), [], [])
    assert len(props) == 1
    assert props[0].box_px == (645, 360, 735, 440) and props[0].conf == pytest.approx(0.6)


def test_reduced_onnx_runs_on_cpu_when_present():
    """The real reduced export through onnxruntime (skipped when the weights aren't on this machine)."""
    from pathlib import Path
    p = Path(__file__).resolve().parents[1] / 'experiments/openset/weights/yoloe-26s-seg-pf-reduced.onnx'
    if not p.exists():
        pytest.skip('reduced ONNX not present')
    pytest.importorskip('onnxruntime')
    m = ReducedYOLOE(str(p), providers=['cpu'])
    assert len(m.names) > 4000
    r = m.predict(np.full((720, 1280, 3), 114, np.uint8), conf=0.15)[0]
    assert r.boxes.xyxy.shape[1] == 4


def test_proposer_picks_the_fast_path_by_model_name(monkeypatch):
    import core.yoloe_fast
    made = []

    class Rec:
        names = {}

        def __init__(self, model, imgsz=640, **kw):
            made.append((model, imgsz))

    monkeypatch.setattr(core.yoloe_fast, 'ReducedYOLOE', Rec)
    y = YOLOEProposer({'model': 'models/yoloe-26s-seg-pf-reduced.engine', 'imgsz': 640})
    assert made == [('models/yoloe-26s-seg-pf-reduced.engine', 640)] and isinstance(y.model, Rec)
