"""core/proposals.py: the change-detection proposer on synthetic tabletop frames, the YOLOE adapter with
a fake model, the perception-side dedupe, and the Detector hook."""
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from core.config import load_config
from core.detect import Detector
from core.proposals import (ChangeProposer, DedupeConfig, Proposal, YOLOEProposer, dedupe,
                            make_proposer)
from core.types import Frame
from tests.synth import skin, texture

W, H = 1280, 720
TABLE = np.array((200, 210, 220), np.float32)       # light grey-blue, as tests/synth.py
NOTEBOOK = (100, 100, 350, 280)
CFG = load_config()


class Table:
    """A textured table under a gentle light gradient, with fresh camera grain on every frame."""

    def __init__(self, seed=0, noise=3.0):
        rng = np.random.default_rng(seed)
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        light = 0.92 + 0.08 * (xx / W) + 0.04 * np.sin(yy / 90.0)
        grain = cv2.resize(rng.normal(0, 6, (H // 16, W // 16)).astype(np.float32), (W, H))
        self.base = TABLE[None, None, :] * light[..., None] + grain[..., None]
        self.grain = [rng.normal(0, noise, (H, W, 3)).astype(np.float32) for _ in range(7)]
        self.n = 0
        self.things: dict = {}          # key -> box_px, painted with texture(key)
        self.hands: list = []           # box_px, painted with skin
        self.arms: list = []
        self.shadows: list = []         # (box, factor)
        self.dots: list = []            # laser dots (x, y)
        self.coins: list = []           # small green discs the size of the dot's bloom
        self.gain = 1.0
        self.shift = (0, 0)

    def frame(self) -> np.ndarray:
        img = self.base.copy()
        for box, f in self.shadows:
            x1, y1, x2, y2 = box
            img[y1:y2, x1:x2] *= f
        for key, (x1, y1, x2, y2) in self.things.items():
            img[y1:y2, x1:x2] = texture(key, y2 - y1, x2 - x1)
        for i, (x1, y1, x2, y2) in enumerate(self.arms + self.hands):
            img[y1:y2, x1:x2] = skin(f'hand{i}', y2 - y1, x2 - x1)
        self.n += 1
        img = img * self.gain + self.grain[self.n % len(self.grain)]
        img = np.clip(img, 0, 255).astype(np.uint8)
        for x, y in self.dots:                          # red bloom around a blown-out centre
            cv2.circle(img, (x, y), 14, (40, 40, 255), -1)
            cv2.circle(img, (x, y), 5, (235, 225, 255), -1)
        for x, y in self.coins:
            cv2.circle(img, (x, y), 14, (60, 140, 90), -1)
        if self.shift != (0, 0):
            img = cv2.warpAffine(img, np.float32([[1, 0, self.shift[0]], [0, 1, self.shift[1]]]), (W, H),
                                 borderMode=cv2.BORDER_REFLECT)
        return img


def proposer(**kw):
    cfg = dict(ref_frames=5, adopt_frames=3)
    cfg.update(kw)
    return ChangeProposer(cfg)


def warm(p, tab, known=(), hands=()):
    for _ in range(p.cfg.ref_frames):
        assert p.propose(tab.frame(), list(known), list(hands)) == []
    assert p.ready


def run(p, tab, n=3, known=(), hands=()):
    out = []
    for _ in range(n):
        out = p.propose(tab.frame(), list(known), list(hands))
    return out


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    i = ix * iy
    return i / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i)


# ------------------------------------------------------------------ change proposer: scene cases

def test_a_new_object_is_one_proposal_with_its_box():
    tab, p = Table(), proposer()
    warm(p, tab)
    tab.things['mug'] = (600, 300, 690, 380)
    [pr] = run(p, tab)
    assert iou(pr.box_px, (600, 300, 690, 380)) > 0.8
    assert pr.conf >= 0.5


def test_an_empty_table_with_camera_grain_proposes_nothing():
    tab, p = Table(noise=5.0), proposer()
    warm(p, tab)
    for _ in range(40):
        assert p.propose(tab.frame(), [], []) == []


def test_a_global_brightness_change_proposes_nothing():
    tab, p = Table(), proposer()
    warm(p, tab)
    tab.gain = 1.12
    assert run(p, tab, 5) == []
    tab.gain = 0.85
    assert run(p, tab, 5) == []


def test_a_hand_and_its_arm_from_the_table_edge_are_not_things():
    tab, p = Table(), proposer()
    warm(p, tab)
    hand = (380, 300, 500, 420)
    tab.hands = [hand]
    tab.arms = [(0, 330, 385, 400)]                   # forearm reaching in from the left edge
    tab.shadows = [((420, 330, 560, 470), 0.6)]       # the hand's shadow
    assert run(p, tab, 3, hands=[hand]) == []


def test_a_shadow_is_not_a_thing():
    tab, p = Table(), proposer()
    warm(p, tab)
    tab.shadows = [((500, 250, 800, 450), 0.55)]
    assert run(p, tab, 3) == []


def test_a_known_object_is_subtracted_so_an_unknown_touching_it_is_kept_alone():
    tab, p = Table(), proposer()
    warm(p, tab)
    keys = (500, 300, 580, 360)
    tab.things['keys'] = keys
    tab.things['charger'] = (580, 290, 680, 370)      # shares the keys' right edge
    [pr] = run(p, tab, known=[keys])
    assert iou(pr.box_px, (580, 290, 680, 370)) > 0.7


def test_two_unknown_objects_are_two_proposals():
    tab, p = Table(), proposer()
    warm(p, tab)
    tab.things['a'] = (300, 400, 380, 470)
    tab.things['b'] = (800, 200, 900, 260)
    props = sorted(run(p, tab), key=lambda q: q.box_px[0])
    assert len(props) == 2
    assert iou(props[0].box_px, tab.things['a']) > 0.8 and iou(props[1].box_px, tab.things['b']) > 0.8


def test_the_laser_dot_is_ignored_but_a_small_object_of_its_size_is_not():
    tab, p = Table(), proposer()
    warm(p, tab)
    tab.dots = [(640, 360), (200, 600)]
    assert run(p, tab, 3) == []
    assert [r for _, r in p.debug['rejected']] == ['laser', 'laser']
    tab.dots, tab.coins = [], [(640, 360)]
    [pr] = run(p, tab, 3)
    assert iou(pr.box_px, (626, 346, 655, 375)) > 0.6


def test_a_notebook_known_at_startup_reveals_table_not_a_thing_when_moved():
    """Present while the reference was captured, but excluded as a known object: its old spot has no
    reference until it has been seen empty, and is then learned as table."""
    tab, p = Table(), proposer()
    tab.things['notebook'] = NOTEBOOK
    warm(p, tab, known=[NOTEBOOK])
    moved = (700, 350, 950, 530)
    tab.things['notebook'] = moved
    for _ in range(10):
        assert p.propose(tab.frame(), [moved], []) == []
    tab.things['mug'] = (180, 150, 270, 230)            # the revealed spot is table now
    [pr] = run(p, tab, known=[moved])
    assert iou(pr.box_px, (180, 150, 270, 230)) > 0.8


def test_an_undetected_object_in_the_reference_leaves_no_ghost_when_moved():
    """Captured INTO the reference (the detector missed it): the revealed table differs from the
    reference, but looks like the table around it, so it is a ghost: healed, never proposed."""
    tab, p = Table(), proposer()
    tab.things['notebook'] = NOTEBOOK
    warm(p, tab)
    moved = (700, 350, 950, 530)
    tab.things['notebook'] = moved
    for _ in range(5):
        props = p.propose(tab.frame(), [moved], [])
        assert not any(iou(q.box_px, NOTEBOOK) > 0.1 for q in props)
    assert run(p, tab, 3, known=[moved]) == []       # healed for good


def test_a_stationary_object_is_not_absorbed_into_the_reference_after_60_s():
    tab, p = Table(), proposer()
    warm(p, tab)
    tab.things['mug'] = (600, 300, 690, 380)
    [first] = run(p, tab, 1)
    for _ in range(60 * 15 - 1):                      # 60 s at 15 fps
        last = p.propose(tab.frame(), [], [])
    [pr] = last
    assert iou(pr.box_px, first.box_px) > 0.9 and pr.conf >= 0.8 * first.conf


def test_a_hand_shadow_that_lingers_does_not_leave_a_phantom():
    tab, p = Table(), proposer()
    warm(p, tab)
    hand = (380, 300, 500, 420)
    tab.hands, tab.shadows = [hand], [((420, 330, 560, 470), 0.6)]
    run(p, tab, 45, hands=[hand])                     # 3 s hovering
    tab.hands, tab.shadows = [], []
    assert run(p, tab, 3) == []


def test_a_lamp_switched_on_over_most_of_the_table_rebuilds_the_reference():
    """Uneven light the global offset cannot absorb: without a rebuild the whole right side is one huge
    changed region, and an object put there would vanish inside it."""
    tab, p = Table(), proposer(rebuild_frames=5)
    warm(p, tab)
    tab.base[:, 576:] *= np.float32((0.75, 1.0, 1.25))      # a warm lamp over 55% of the table
    for _ in range(5 + p.cfg.ref_frames):
        p.propose(tab.frame(), [], [])
    assert p.ready and run(p, tab, 3) == []
    tab.things['mug'] = (200, 300, 290, 380)
    [pr] = run(p, tab)
    assert iou(pr.box_px, (200, 300, 290, 380)) > 0.8


def test_regions_outside_the_table_roi_are_ignored():
    tab, p = Table(), proposer()
    p.set_roi([(200, 100), (1100, 100), (1100, 650), (200, 650)])
    warm(p, tab)
    tab.things['off'] = (20, 300, 120, 380)           # left of the table
    tab.things['on'] = (600, 300, 690, 380)
    [pr] = run(p, tab)
    assert iou(pr.box_px, (600, 300, 690, 380)) > 0.8


def test_a_saved_reference_can_be_loaded(tmp_path):
    tab, p = Table(), proposer()
    warm(p, tab)
    path = str(tmp_path / 'ref.png')
    p.save_reference(path)
    q = proposer()
    q.load_reference(cv2.imread(path))
    assert q.ready
    tab.things['mug'] = (600, 300, 690, 380)
    [pr] = run(q, tab)
    assert iou(pr.box_px, (600, 300, 690, 380)) > 0.7


def test_reset_recaptures_the_reference():
    tab, p = Table(), proposer()
    warm(p, tab)
    p.reset()
    assert not p.ready
    warm(p, tab)


def test_frames_without_an_image_are_skipped():
    assert proposer().propose(None, [], []) == []


# ------------------------------------------------------------------ dedupe

K = (100, 100, 200, 200)


def P(box, conf=0.6):
    return Proposal(box_px=box, conf=conf)


def test_a_proposal_matching_a_known_box_is_dropped():
    assert dedupe([P((105, 102, 205, 198))], [K], [], DedupeConfig()) == []


def test_a_proposal_inside_a_known_box_is_dropped():
    assert dedupe([P((120, 120, 160, 160))], [K], [], DedupeConfig()) == []


def test_a_proposal_slightly_larger_than_a_known_box_is_dropped():
    assert dedupe([P((85, 85, 225, 215))], [K], [], DedupeConfig()) == []


def test_an_unknown_that_partly_overlaps_a_known_box_is_kept():
    p = P((180, 120, 280, 200))                       # a charger lying across the keys' edge
    assert dedupe([p], [K], [], DedupeConfig()) == [p]


def test_a_proposal_mostly_covered_by_a_hand_is_dropped_but_one_beside_it_is_kept():
    near = P((190, 100, 260, 160))
    out = dedupe([P((110, 110, 190, 210)), near], [], [K], DedupeConfig())
    assert out == [near]


def test_overlapping_proposals_keep_the_most_confident():
    a, b = P((100, 100, 200, 200), 0.9), P((110, 105, 205, 200), 0.5)
    assert dedupe([b, a], [], [], DedupeConfig()) == [a]


def test_dedupe_thresholds_come_from_config():
    p = P((150, 100, 250, 200))                       # IoU 1/3 with K
    assert dedupe([p], [K], [], DedupeConfig()) == [p]
    assert dedupe([p], [K], [], DedupeConfig.from_dict({'known_iou': 0.3})) == []


# ------------------------------------------------------------------ YOLOE adapter (fake model)

class FakeYOLOE:
    names = {0: 'cup', 1: 'person', 2: 'dining table', 3: 'charger'}

    def __init__(self, dets):
        self.dets = dets
        self.calls = []

    def predict(self, img, **kw):
        self.calls.append(kw)
        xyxy = np.array([d[2] for d in self.dets], np.float32).reshape(-1, 4)
        boxes = SimpleNamespace(xyxy=xyxy, conf=np.array([d[1] for d in self.dets], np.float32),
                                cls=np.array([d[0] for d in self.dets], np.float32))
        return [SimpleNamespace(boxes=boxes, masks=None)]


def test_yoloe_boxes_are_class_agnostic_and_filtered():
    m = FakeYOLOE([(0, 0.6, (600, 300, 690, 380)),       # cup: kept
                   (3, 0.08, (100, 100, 150, 150)),       # below conf
                   (1, 0.9, (0, 0, 400, 700)),            # person: ignored class
                   (2, 0.8, (0, 0, 1280, 720)),           # the table itself: too big (and ignored)
                   (3, 0.5, (602, 302, 688, 382))])       # same object, another class: merged
    y = YOLOEProposer({'conf': 0.15}, model=m)
    props = y.propose(np.zeros((H, W, 3), np.uint8), [], [])
    assert [q.box_px for q in props] == [(600, 300, 690, 380)] and props[0].conf == pytest.approx(0.6)
    assert m.calls[0]['conf'] <= 0.15


def test_yoloe_respects_the_table_roi():
    m = FakeYOLOE([(0, 0.6, (20, 300, 100, 380)), (0, 0.6, (600, 300, 690, 380))])
    y = YOLOEProposer({}, model=m)
    y.set_roi([(200, 100), (1100, 100), (1100, 650), (200, 650)])
    assert [q.box_px for q in y.propose(np.zeros((H, W, 3), np.uint8), [], [])] == [(600, 300, 690, 380)]


# ------------------------------------------------------------------ Detector integration

class FakeBackend:
    def __init__(self, raw=()):
        self.raw = list(raw)

    def infer(self, img):
        return list(self.raw)


class TenPxPerCm:
    ok = True

    def px_to_cm(self, pts):
        return np.asarray(pts, dtype=float).reshape(-1, 2) / 10.0


def cfg_with(**section):
    cfg = dict(CFG)
    cfg['proposals'] = {**(CFG.get('proposals') or {}), **section}
    return cfg


def test_detector_builds_the_configured_proposer():
    det = Detector(cfg_with(enabled=True, kind='change'), table=TenPxPerCm(), backend=FakeBackend())
    assert isinstance(det.proposer, ChangeProposer)
    assert Detector(cfg_with(enabled=False), table=TenPxPerCm(), backend=FakeBackend()).proposer is None
    assert Detector(CFG, table=TenPxPerCm(), backend=FakeBackend(), proposer=None).proposer is None


def test_make_proposer_follows_kind():
    assert make_proposer(cfg_with(enabled=False)) is None
    assert isinstance(make_proposer(cfg_with(enabled=True, kind='change')), ChangeProposer)
    with pytest.raises(ValueError):
        make_proposer(cfg_with(enabled=True, kind='nope'))


def test_detector_adds_things_in_table_cm_and_dedupes_against_known_objects():
    tab = Table()
    backend = FakeBackend()
    det = Detector(CFG, table=TenPxPerCm(), backend=backend, proposer=proposer(), crops=None)
    for i in range(5):
        d = det.detect(Frame(t=i, wall=i, img=tab.frame(), idx=i))
        assert d.items == []
    tab.things['mug'] = (600, 300, 690, 380)
    tab.things['wallet'] = (900, 400, 1000, 470)
    backend.raw = [('wallet', 0.8, (898, 398, 1003, 472))]
    d = det.detect(Frame(t=9, wall=9, img=tab.frame(), idx=9))
    things = [x for x in d.items if x.cls == 'thing']
    assert [x.cls for x in d.items].count('wallet') == 1 and len(things) == 1
    assert things[0].box_cm == pytest.approx((60, 30, 69, 38), abs=1.0)
    assert det.last_proposal_ms > 0


def test_detector_without_an_image_skips_proposals():
    det = Detector(CFG, table=TenPxPerCm(), backend=FakeBackend([('keys', 0.9, (0, 0, 10, 10))]),
                   proposer=proposer())
    d = det.detect(Frame(t=0, wall=0, img=np.zeros((H, W, 3), np.uint8), idx=0))
    assert [x.cls for x in d.items] == ['keys']


def test_a_recalibrated_table_resets_the_reference():
    class Cal(TenPxPerCm):
        H = np.eye(3)

        def cm_to_px(self, pts):
            return np.asarray(pts, dtype=float).reshape(-1, 2) * 10.0

    tab, table = Table(), Cal()
    det = Detector(CFG, table=table, backend=FakeBackend(), proposer=proposer())
    for i in range(6):
        det.detect(Frame(t=i, wall=i, img=tab.frame(), idx=i))
    assert det.proposer.ready
    table.H = np.eye(3) * 1.0                          # calibrate() stores a new matrix
    det.detect(Frame(t=7, wall=7, img=tab.frame(), idx=7))
    assert not det.proposer.ready


def test_end_to_end_the_world_confirms_things_from_the_detector_and_crops_follow_them():
    """The contract with core/world.py + core/things.py: proposals become thing:N entities, an unknown
    touching a known object is its own thing, and the world's entity box finds its crops."""
    from core import crops
    from core.world import World
    tab, backend = Table(), FakeBackend()
    det = Detector(CFG, table=TenPxPerCm(), backend=backend, proposer=proposer(ref_frames=10))
    world, t, events = World(CFG), 100.0, []

    def step(n, raw=()):
        nonlocal t
        backend.raw = list(raw)
        for _ in range(n):
            t += 1 / 15
            f = Frame(t=t, wall=t, img=tab.frame(), idx=round(t * 15))
            events.extend(world.update(det.detect(f), f))

    step(15)
    tab.things['mug'] = (600, 300, 690, 380)
    step(30)
    keys = ('keys', 0.9, (200, 500, 280, 560))
    tab.things['keys'], tab.things['charger'] = keys[2], (280, 490, 380, 570)
    step(30, [keys])
    appeared = {e.obj: e.to_cm for e in events if str(e.type) == 'APPEARED'}
    assert sorted(appeared) == ['thing:1', 'thing:2']
    assert appeared['thing:1'] == pytest.approx((64.5, 34.0), abs=1.0)
    assert appeared['thing:2'] == pytest.approx((33.0, 53.0), abs=1.5)
    assert crops.active() is det.crops
    rec = det.crops.for_entity(world.get('thing:2'))
    assert rec is not None and rec.best is not None and rec.best.box_px[0] >= 270


def test_a_failing_proposer_never_costs_the_known_objects():
    class Broken:
        def set_roi(self, poly):
            pass

        def propose(self, img, known, hands):
            raise RuntimeError('boom')

    det = Detector(CFG, table=TenPxPerCm(), backend=FakeBackend([('keys', 0.9, (0, 0, 10, 10))]), proposer=Broken())
    d = det.detect(Frame(t=0, wall=0, img=np.zeros((H, W, 3), np.uint8), idx=0))
    assert [x.cls for x in d.items] == ['keys']
