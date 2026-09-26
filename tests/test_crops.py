"""core/crops.py: best-clean and most-recent crops per object / proposal, looked up by box."""
import os

import cv2
import numpy as np
import pytest

from core import crops as crops_mod
from core.crops import CropStore, clean_score, event_snapshot
from core.events import EventLog
from core.types import Detection, Entity, Event, EventType, Frame
from tests.synth import texture

W, H = 1280, 720


def det(cls, box, conf=0.8):
    x1, y1, x2, y2 = box
    return Detection(cls=cls, conf=conf, box_px=box, center_cm=((x1 + x2) / 20, (y1 + y2) / 20),
                     box_cm=(x1 / 10, y1 / 10, x2 / 10, y2 / 10))


def scene(things, hands=(), blur=0):
    img = np.full((H, W, 3), (200, 210, 220), np.uint8)
    for key, (x1, y1, x2, y2) in things.items():
        img[y1:y2, x1:x2] = texture(key, y2 - y1, x2 - x1)
    for x1, y1, x2, y2 in hands:
        img[y1:y2, x1:x2] = (140, 170, 215)
    if blur:
        img = cv2.GaussianBlur(img, (0, 0), blur)
    return img


MUG = (600, 300, 690, 380)


def test_a_blurred_or_hand_covered_view_scores_below_a_clean_one():
    clean = clean_score(scene({'mug': MUG}), MUG, 0.8, [], [])
    blurred = clean_score(scene({'mug': MUG}, blur=4), MUG, 0.8, [], [])
    hand = (640, 320, 760, 440)
    touched = clean_score(scene({'mug': MUG}, [hand]), MUG, 0.8, [], [hand])
    crowded = clean_score(scene({'mug': MUG}), MUG, 0.8, [(560, 280, 650, 360)], [])
    assert clean > blurred and clean > touched and clean > crowded
    assert touched < 0.3 * clean


def test_the_store_keeps_the_cleanest_crop_and_the_most_recent_one():
    store = CropStore(recent_every_s=0)
    hand = (640, 320, 760, 440)
    views = [(scene({'mug': MUG}, [hand]), [hand]),          # t=1 touched
             (scene({'mug': MUG}), []),                       # t=2 clean: the best
             (scene({'mug': MUG}, blur=4), [])]               # t=3 blurred: the most recent
    for t, (img, hands) in enumerate(views, 1):
        store.update(img, [det('thing', MUG)], [det('hand', h) for h in hands], float(t))
    rec = store.for_box(box_cm=(60, 30, 69, 38))
    assert rec.best.t == 2.0 and rec.recent.t == 3.0
    assert max(rec.best.img.shape[:2]) <= store.size_px


def test_known_objects_are_keyed_by_name_and_things_by_position():
    store = CropStore(recent_every_s=0)
    img = scene({'keys': (100, 100, 180, 160), 'a': MUG, 'b': (900, 500, 980, 560)})
    items = [det('keys', (100, 100, 180, 160)), det('thing', MUG), det('thing', (900, 500, 980, 560))]
    store.update(img, items, [], 1.0)
    store.update(img, [det('keys', (104, 100, 184, 160)), det('thing', (603, 301, 692, 381)),
                       det('thing', (900, 500, 980, 560))], [], 2.0)
    assert len(store) == 3
    assert store.for_entity(Entity(name='keys', kind='target')).key == 'keys'
    a = store.for_entity(Entity(name='thing:1', kind='target', box_cm=(60, 30, 69, 38)))
    b = store.for_box(box_cm=(90, 50, 98, 56))
    assert a is not None and b is not None and a.key != b.key
    assert store.for_box(box_cm=(20, 60, 25, 65)) is None           # nothing there


def test_the_store_is_bounded_by_evicting_the_least_recently_seen():
    store = CropStore(max_tracks=4, recent_every_s=0)
    for i in range(10):
        box = (50 + 110 * i, 300, 130 + 110 * i, 360)
        store.update(scene({f'k{i}': box}), [det('thing', box)], [], float(i))
    assert len(store) == 4
    assert store.for_box(box_px=(50 + 110 * 9, 300, 130 + 110 * 9, 360)) is not None
    assert store.for_box(box_px=(50, 300, 130, 360)) is None


def test_hands_get_no_crops_and_no_image_is_a_no_op():
    store = CropStore()
    store.update(None, [det('thing', MUG)], [], 1.0)
    store.update(scene({}), [], [det('hand:1', MUG)], 1.0)
    assert len(store) == 0


def test_the_last_store_created_by_a_detector_is_the_active_one():
    s = CropStore()
    crops_mod.set_active(s)
    assert crops_mod.active() is s


def test_event_snapshot_returns_the_newest_event_image(tmp_path):
    log = EventLog(':memory:', str(tmp_path))
    img = scene({'mug': MUG})
    ev = Event(t=1.0, wall=1000.0, obj='thing:1', type=EventType.APPEARED, to_cm=(64.5, 34.0))
    log.add(ev, Frame(t=1.0, wall=1000.0, img=img, idx=1))
    log.add(Event(t=2.0, wall=1001.0, obj='thing:1', type=EventType.PICKED_UP))     # no image
    log.flush()
    got = event_snapshot(log, 'thing:1')
    assert got is not None
    e, snap = got
    assert e.type == EventType.APPEARED and snap.shape == img.shape and os.path.exists(e.snapshot)

    class Tbl:
        def cm_to_px(self, pts):
            return np.asarray(pts, dtype=float).reshape(-1, 2) * 10

    e, crop = event_snapshot(log, 'thing:1', table=Tbl(), half_cm=6)
    assert crop.shape[:2] == (120, 120)
    assert event_snapshot(log, 'nothing') is None
    log.close()
