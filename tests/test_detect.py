"""core/detect.py with a fake model backend (no ultralytics needed)."""
import numpy as np
import pytest

from core.config import load_config
from core.detect import Detector, class_list
from core.types import Frame

CFG = load_config()


class FakeBackend:
    """Returns the raw (label, conf, box_px) list it is given, whatever the image."""

    def __init__(self, raw):
        self.raw = raw

    def infer(self, img):
        return list(self.raw)


class TenPxPerCm:
    ok = True

    def px_to_cm(self, pts):
        return np.asarray(pts, dtype=float).reshape(-1, 2) / 10.0


def frame():
    return Frame(t=12.5, wall=1000.0, img=np.zeros((720, 1280, 3), np.uint8), idx=7)


def run(raw, cfg=CFG):
    return Detector(cfg, table=TenPxPerCm(), backend=FakeBackend(raw)).detect(frame())


def test_prompt_labels_map_back_to_object_names():
    d = run([("key ring", 0.8, (100, 100, 160, 140)), ("medicine bottle", 0.7, (300, 300, 340, 340))])
    assert sorted(i.cls for i in d.items) == ["keys", "pill_bottle"]


def test_only_the_highest_confidence_box_per_object_is_kept():
    d = run([("keys", 0.5, (0, 0, 10, 10)), ("key ring", 0.9, (200, 200, 260, 240)), ("keys", 0.6, (5, 5, 9, 9))])
    [k] = d.items
    assert k.conf == 0.9 and k.box_px == (200, 200, 260, 240)


def test_every_hand_box_is_kept_separately():
    d = run([("hand", 0.9, (0, 0, 100, 100)), ("hand", 0.6, (500, 500, 600, 600)), ("keys", 0.8, (1, 1, 5, 5))])
    assert [h.cls for h in d.hands] == ["hand", "hand"] and [i.cls for i in d.items] == ["keys"]


def test_boxes_below_the_threshold_and_unknown_labels_are_dropped():
    d = run([("keys", 0.2, (0, 0, 10, 10)), ("laptop", 0.95, (0, 0, 10, 10)), ("wallet", 0.5, (0, 0, 10, 10))])
    assert [i.cls for i in d.items] == ["wallet"]


def test_per_object_threshold_overrides_the_default():
    cfg = dict(CFG, conf_threshold={"default": 0.35, "keys": 0.15})
    d = run([("keys", 0.2, (0, 0, 10, 10)), ("wallet", 0.2, (0, 0, 10, 10))], cfg)
    assert [i.cls for i in d.items] == ["keys"]


def test_boxes_are_converted_to_table_cm_and_stamped_with_the_frame():
    d = run([("wallet", 0.9, (100, 200, 300, 400))])
    w = d.items[0]
    assert w.center_cm == pytest.approx((20.0, 30.0))
    assert w.box_cm == pytest.approx((10.0, 20.0, 30.0, 40.0))
    assert (d.t, d.frame_idx) == (12.5, 7)


def test_uncalibrated_table_is_refused():
    class NotCalibrated:
        ok = False
    with pytest.raises(RuntimeError):
        Detector(CFG, table=NotCalibrated(), backend=FakeBackend([])).detect(frame())


def test_class_list_is_every_prompt_plus_hand_with_a_label_map():
    classes, to_obj = class_list(CFG)
    assert "key ring" in classes and "hand" in classes and len(classes) == len(set(classes))
    assert to_obj["key ring"] == "keys" and to_obj["tv remote"] == "remote" and to_obj["hand"] == "hand"
    assert to_obj["keys"] == "keys"          # a fine-tuned model's class names are the object names
