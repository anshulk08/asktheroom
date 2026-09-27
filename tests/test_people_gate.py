"""YOLOEProposer.people(): the room laser's eye-safety gate. It must err towards seeing someone: any size
of person box, per-class NMS, worn items on their own, and a lower confidence than proposals. Fake
ultralytics-like models, and the reduced engine (core/yoloe_fast.py) with a fake runner."""
from types import SimpleNamespace

import numpy as np

from core.proposals import YOLOEProposer
from core.yoloe_fast import ReducedYOLOE
from tests.test_yoloe_fast import FakeRunner, head

W, H = 1280, 720
IMG = np.zeros((H, W, 3), np.uint8)
NAMES = {0: 'cup', 1: 'person', 2: 'dining table', 3: 'shirt', 4: 'jeans'}


class NmsYOLOE:
    """Emulates predict()'s conf filter and NMS by argument: agnostic_nms=True keeps only the most
    confident of two overlapping boxes whatever their class."""
    names = NAMES

    def __init__(self, dets):
        self.dets, self.calls = dets, []

    def predict(self, img, conf=0.25, agnostic_nms=False, **kw):
        self.calls.append(dict(conf=conf, agnostic_nms=agnostic_nms, **kw))
        d = sorted((x for x in self.dets if x[1] >= conf), key=lambda x: -x[1])
        keep = []
        for x in d:
            if not any((agnostic_nms or k[0] == x[0]) and _iou(k[2], x[2]) > 0.5 for k in keep):
                keep.append(x)
        boxes = SimpleNamespace(xyxy=np.array([x[2] for x in keep], np.float32).reshape(-1, 4),
                                conf=np.array([x[1] for x in keep], np.float32),
                                cls=np.array([x[0] for x in keep], np.float32))
        return [SimpleNamespace(boxes=boxes, masks=None, names=self.names)]


def _iou(a, b):
    iw = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    ih = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    i = iw * ih
    return i / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i)


def test_the_nearest_person_filling_the_frame_is_seen_through_the_reduced_engine():
    """The reduced engine drops boxes over 35% of the frame before NMS; the nearest person is one."""
    dets = [(320, 320, 480, 280, 0.8, 1),      # 960 x 560 of 1280 x 720 = 58% of the frame
            (100, 200, 40, 40, 0.6, 0)]        # a cup
    m = ReducedYOLOE(runner=FakeRunner(head(dets)), imgsz=640)
    m.names = NAMES
    y = YOLOEProposer({'conf': 0.15}, model=m)
    assert y.people(IMG) == [(160, 80, 1120, 640)]
    assert [p.box_px for p in y.propose(IMG, [], [])] == [(160, 80, 240, 160)]    # propose() still caps


def test_a_big_person_box_from_an_ultralytics_like_model_is_seen():
    m = NmsYOLOE([(1, 0.8, (0, 0, 1024, 540))])                                   # 60% of the frame
    assert YOLOEProposer({}, model=m).people(IMG) == [(0, 0, 1024, 540)]
    assert 'max_area_frac' not in m.calls[0]            # ultralytics rejects arguments it does not know


def test_a_shirt_box_does_not_suppress_the_person_wearing_it():
    m = NmsYOLOE([(3, 0.7, (400, 200, 800, 700)),          # shirt, more confident
                  (1, 0.5, (390, 150, 810, 719))])         # the person in it
    got = YOLOEProposer({}, model=m).people(IMG)
    assert m.calls[0]['agnostic_nms'] is False
    assert (390, 150, 810, 719) in got


def test_the_reduced_engine_does_per_class_nms_for_people():
    dets = [(320, 320, 200, 160, 0.7, 3),      # shirt
            (320, 320, 210, 180, 0.5, 1)]      # person, IoU ~0.85 with it
    m = ReducedYOLOE(runner=FakeRunner(head(dets)), imgsz=640)
    m.names = NAMES
    y = YOLOEProposer({}, model=m)
    assert len(y.people(IMG)) == 2
    r = m.predict(IMG, conf=0.15, iou=0.5, agnostic_nms=True)[0]        # propose()'s call: unchanged
    assert list(r.boxes.cls) == [3]


def test_a_lone_worn_item_is_a_person():
    """A leg under the table may be boxed only as 'jeans', with no person box anywhere near it."""
    m = NmsYOLOE([(4, 0.6, (300, 400, 500, 719)), (0, 0.6, (900, 100, 990, 180))])
    assert YOLOEProposer({}, model=m).people(IMG) == [(300, 400, 500, 719)]


def test_people_use_a_lower_confidence_than_proposals():
    m = NmsYOLOE([(1, 0.2, (300, 200, 500, 600)), (1, 0.1, (700, 200, 900, 600))])
    y = YOLOEProposer({'conf': 0.35}, model=m)
    assert y.people(IMG) == [(300, 200, 500, 600)]
    assert m.calls[0]['conf'] == 0.15
    assert YOLOEProposer({'conf': 0.35, 'people_conf': 0.05}, model=m).people(IMG) == \
        [(300, 200, 500, 600), (700, 200, 900, 600)]
