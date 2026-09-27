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
    mug = det('thing', (603, 301, 692, 381))
    store.update(img, [det('keys', (104, 100, 184, 160)), mug, det('thing', (900, 500, 980, 560))], [], 2.0)
    assert len(store) == 3
    assert store.for_entity(Entity(name='keys', kind='target')).key == 'keys'
    # a thing's crops come by the observation the world saw it as (its box object), not by position
    a = store.for_entity(Entity(name='thing:1', kind='target', box_cm=mug.box_cm))
    b = store.for_box(box_cm=(90, 50, 98, 56))
    assert a is not None and b is not None and a.key != b.key
    assert store.for_entity(Entity(name='thing:9', kind='target', box_cm=(60, 30, 69, 38))) is None
    assert store.for_box(box_cm=(20, 60, 25, 65)) is None           # nothing there


def test_a_thing_track_restarts_its_best_crop_after_a_gap():
    """A different object put where a proposal was must not inherit the old one's best crop: after the
    spot was empty for more than renew_after_s the track starts over; a short flicker keeps it."""
    store = CropStore(recent_every_s=0)
    store.update(scene({'red mug': MUG}), [det('thing', MUG)], [], 1.0)          # clean, sharp
    store.update(scene({'red mug': MUG}, blur=4), [det('thing', MUG)], [], 1.5)  # flicker gap: kept
    assert store.for_box(box_cm=(60, 30, 69, 38)).best.t == 1.0
    store.update(scene({'blue notebook': MUG}, blur=4), [det('thing', MUG)], [], 4.0)   # after 2.5 s away
    rec = store.for_box(box_cm=(60, 30, 69, 38))
    assert rec.best.t == 4.0 and rec.recent.t == 4.0


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


# ---------------------------------------------------------------- ownership: a crop belongs to the entity it was seen as

def tinted(key, colour):
    """A textured object whose colour is clearly `colour` ('red' or 'blue'): sharp, and its mean colour
    tells the two apart."""
    tex = texture(key, MUG[3] - MUG[1], MUG[2] - MUG[0]).copy()
    keep = 2 if colour == 'red' else 0
    for ch in range(3):
        if ch != keep:
            tex[..., ch] //= 5
    img = np.full((H, W, 3), (200, 210, 220), np.uint8)
    img[MUG[1]:MUG[3], MUG[0]:MUG[2]] = tex
    return img


def colour_of(crop):
    b, _, r = crop.img.reshape(-1, 3).mean(axis=0)
    return 'red' if r > b else 'blue'


def thing(name, d):
    """The world's entity for a thing last observed as detection d (the world keeps d.box_cm)."""
    return Entity(name=name, kind='target', box_cm=d.box_cm)


def test_a_replacement_after_a_gap_never_shows_as_the_old_thing():
    """Reviewer repro: a red thing's crop, the red thing replaced by a blue one at the same spot after
    a 2 s gap; asking for the red thing's close-up must not return the blue one."""
    store = CropStore(recent_every_s=0)
    red = det('thing', MUG)
    store.update(tinted('mug', 'red'), [red], [], 1.0)
    blue = det('thing', MUG)
    store.update(tinted('book', 'blue'), [blue], [], 3.0)
    old = store.for_entity(thing('thing:1', red))
    assert old is not None and old.best.t == 1.0 and colour_of(old.best) == 'red'
    assert old.recent is None or colour_of(old.recent) == 'red'
    new = store.for_entity(thing('thing:2', blue))
    assert new is not None and new.best.t == 3.0 and colour_of(new.best) == 'blue'
    # a box alone (same place, but not the observation the world saw) proves nothing
    lookalike = store.for_entity(Entity(name='thing:7', kind='target', box_cm=(60.0, 30.0, 69.0, 38.0)))
    assert lookalike is None
    again = store.for_entity(Entity(name='thing:1', kind='target', box_cm=(60.0, 30.0, 69.0, 38.0)))
    assert again is None or (colour_of(again.best) == 'red' and again.best.t == 1.0)


def test_an_instant_swap_keeps_each_things_own_close_up():
    """No gap: the red thing is swapped for a blue one between two frames. Whether the world calls the
    blue one a new thing or keeps the old identity, the red view is never shown for the blue one and
    the blue view never for the red one."""
    for same_identity in (False, True):
        store = CropStore(recent_every_s=0)
        reds = [det('thing', MUG) for _ in range(3)]
        for i, d in enumerate(reds):
            store.update(tinted('mug', 'red'), [d], [], 1.0 + i / 15)
            store.bind(thing('thing:1', d))                      # the world saw thing:1 in this frame
        blue = det('thing', MUG)
        t_blue = 1.0 + 3 / 15
        store.update(tinted('book', 'blue'), [blue], [], t_blue)
        name = 'thing:1' if same_identity else 'thing:2'
        store.bind(thing(name, blue))
        now = store.for_entity(thing(name, blue))
        assert now is not None and colour_of(now.best) == 'blue' and now.best.t == t_blue
        assert now.recent is None or colour_of(now.recent) == 'blue'
        if not same_identity:
            old = store.for_entity(thing('thing:1', reds[-1]))
            assert old is not None and colour_of(old.best) == 'red' and old.best.t < t_blue


def test_a_series_claimed_by_another_thing_stops_at_the_last_confirmed_view():
    """Two look-alike things swapped with no visible change: the store can't split them itself, so
    when the world says the view now belongs to another thing, the old thing keeps only what was
    confirmed as it and the new one starts from nothing."""
    store = CropStore(recent_every_s=0)
    first = det('thing', MUG)
    store.update(tinted('mug', 'red'), [first], [], 1.0)
    store.bind(thing('thing:1', first))
    swapped = det('thing', MUG)
    store.update(tinted('mug', 'red'), [swapped], [], 1.1)        # same look, different object
    store.bind(thing('thing:2', swapped))
    old = store.for_entity(thing('thing:1', first))
    assert old is not None and old.best.t == 1.0 and (old.recent is None or old.recent.t == 1.0)
    assert store.for_entity(thing('thing:2', swapped)) is None     # nothing confirmed as thing:2 yet
    later = det('thing', MUG)
    store.update(tinted('mug', 'red'), [later], [], 1.2)
    store.bind(thing('thing:2', later))
    got = store.for_entity(thing('thing:2', later))
    assert got is not None and {c.t for c in (got.best, got.recent) if c is not None} == {1.2}
    assert store.for_entity(thing('thing:1', first)).best.t == 1.0


def test_crops_carry_series_ids_and_times():
    store = CropStore(recent_every_s=0)
    a, b = det('thing', MUG), det('thing', MUG)
    store.update(tinted('mug', 'red'), [a], [], 1.0)
    store.update(tinted('book', 'blue'), [b], [], 3.0)
    ra, rb = store.for_entity(thing('thing:1', a)), store.for_entity(thing('thing:2', b))
    assert ra.sid != rb.sid and ra.best.sid == ra.sid and rb.best.sid == rb.sid
    assert ra.owner == 'thing:1' and ra.owner_t == 1.0 and rb.owner == 'thing:2' and rb.owner_t == 3.0
