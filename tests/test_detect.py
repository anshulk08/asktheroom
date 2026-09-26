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


def test_reset_proposals_recaptures_the_reference_and_forgets_crops():
    from core.crops import CropStore
    from core.types import Detection

    class Prop:
        resets = 0

        def reset(self):
            self.resets += 1

    store, prop = CropStore(), Prop()
    img = np.random.default_rng(0).integers(0, 255, (120, 160, 3), dtype=np.uint8)
    store.update(img, [Detection('thing', 0.9, (20, 20, 60, 60), (3.0, 3.0), (2, 2, 4, 4))], [], 1.0)
    det = Detector({}, table=None, backend=object(), proposer=prop, crops=store)
    assert len(store) == 1
    det.reset_proposals()
    assert prop.resets == 1 and len(store) == 0


# ----- detect_filter: known-object boxes that are a hand or not a tabletop object

class BoundedTable(TenPxPerCm):
    size_cm = (90, 60)

    def in_bounds(self, cm, margin=0.0):
        x, y = cm
        return -margin <= x <= 90 + margin and -margin <= y <= 60 + margin


def run_bounded(raw, cfg=CFG):
    return Detector(cfg, table=BoundedTable(), backend=FakeBackend(raw), proposer=None, crops=None).detect(frame())


def test_a_hand_also_labelled_as_an_object_stays_a_hand():
    d = run_bounded([("hand", 0.5, (400, 300, 520, 420)), ("phone", 0.7, (405, 305, 515, 425))])
    assert d.items == [] and [h.cls for h in d.hands] == ["hand"]


def test_the_real_object_wins_when_the_hand_lookalike_scored_higher():
    d = run_bounded([("hand", 0.5, (400, 300, 520, 420)), ("phone", 0.8, (405, 305, 515, 425)),
                     ("phone", 0.6, (100, 100, 160, 200))])
    [p] = d.items
    assert p.box_px == (100, 100, 160, 200)


def test_an_object_held_in_a_hand_is_kept():
    d = run_bounded([("hand", 0.8, (400, 300, 560, 460)), ("phone", 0.7, (450, 340, 510, 440))])
    assert [i.cls for i in d.items] == ["phone"]


def test_a_box_too_big_for_any_prop_is_dropped():
    d = run_bounded([("notebook", 0.6, (0, 0, 800, 600)), ("wallet", 0.6, (100, 100, 180, 160))])
    assert [i.cls for i in d.items] == ["wallet"]


def test_an_object_off_the_table_is_dropped():
    d = run_bounded([("phone", 0.9, (1000, 650, 1060, 700)), ("keys", 0.6, (100, 100, 150, 140))])
    assert [i.cls for i in d.items] == ["keys"]           # phone centre (103, 67.5) cm is off a 90 x 60 table
