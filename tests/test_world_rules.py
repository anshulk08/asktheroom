"""Cover / container / background rules of the world model (spec 3.5 and section 5 rule tests)."""
from pathlib import Path

import numpy as np
import pytest

from core.config import Config, ObjectSpec
from core.relations import BackgroundModel
from core.types import EventType, Status
from core.world import World
from tests.synth import Scene

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def cfg():
    return Config.load(ROOT / 'config.yaml')


@pytest.fixture
def scene(cfg):
    return Scene(cfg, fps=10, t0=1000.0)


@pytest.fixture
def world(cfg):
    return World(cfg)


def types(events):
    return [e.type for e in events]


def slide(scene, world, name, frm, to, seconds=0.5):
    """Move an already placed object from frm to to in straight steps, one per batch."""
    steps = max(1, round(seconds / scene.dt))
    events = []
    for i in range(1, steps + 1):
        f = i / steps
        scene.place(name, frm[0] + (to[0] - frm[0]) * f, frm[1] + (to[1] - frm[1]) * f)
        events += world.update(*scene.step())
    return events


def cover_keys(scene, world, keys_at=(40, 30), notebook_from=(90, 30)):
    """Keys and notebook settle apart; the notebook slides over the keys, which vanish under it."""
    scene.place('keys', *keys_at)
    scene.place('notebook', *notebook_from)
    scene.run(world, 1.0)
    events = slide(scene, world, 'notebook', notebook_from, keys_at)
    scene.remove('keys')
    return events + scene.run(world, 1.0)


# ----- cover rule ------------------------------------------------------------------------------

def test_notebook_slid_over_keys_is_covered_under_notebook(scene, world, cfg):
    events = cover_keys(scene, world)
    keys = world.get('keys')
    assert types(events) == [EventType.COVERED]
    assert events[0].parent == 'notebook'
    assert events[0].confidence == pytest.approx(cfg.conf_under)
    assert (keys.status, keys.parent) == (Status.UNDER, 'notebook')


def test_covered_keys_resolve_to_the_notebook(scene, world):
    cover_keys(scene, world)
    scene.place('notebook', 45, 32)                # nudged by less than the lifted threshold
    scene.run(world, 0.5)
    assert world.resolve('keys') == (pytest.approx((45, 32)), ['keys', 'notebook'])


def with_tray(cfg):
    """The shipped config has one cover; ambiguity between covers needs a second one."""
    cfg.objects.append(ObjectSpec(name='tray', kind='cover'))
    return cfg


def test_two_covers_qualifying_give_candidates_and_penalty(cfg):
    cfg = with_tray(cfg)
    scene, world = Scene(cfg, fps=10, t0=1000.0), World(cfg)
    scene.place('keys', 40, 30)
    scene.place('notebook', 90, 30)
    scene.place('tray', 40, 60)
    scene.run(world, 1.0)
    for i in range(1, 6):                          # both slide in together
        scene.place('notebook', 90 - 10 * i, 30)   # ends at (40, 30): covers all of the keys
        scene.place('tray', 40 + 2.2 * i, 60 - 6 * i)   # ends at (51, 30): covers 75% of them
        world.update(*scene.step())
    scene.remove('keys')
    events = scene.run(world, 1.0)
    keys = world.get('keys')
    assert types(events) == [EventType.COVERED]
    assert (keys.status, keys.parent, keys.candidates) == (Status.UNDER, 'notebook', ['tray'])
    assert events[0].confidence == pytest.approx(cfg.conf_under * cfg.ambiguity_penalty)


# ----- container rule --------------------------------------------------------------------------

BOX_AT = (80, 40)          # default 20 x 15 cm box: x 70-90, y 32.5-47.5
OUT_OF_BOX = (100, 60)


def carry_keys_into_box(scene, world, keys_at=(40, 30), hid=1):
    """Hand picks the keys up, carries them to the box, dwells 0.6 s and withdraws. Returns events
    up to the moment the hand's centre leaves the box."""
    scene.place('keys', *keys_at)
    scene.place('box', *BOX_AT)
    scene.run(world, 1.0)
    scene.hand(hid, *keys_at)
    events = scene.run(world, 0.3)
    scene.remove('keys')
    events += scene.run(world, 0.2)
    scene.hand(hid, 60, 35)
    events += scene.run(world, 0.3)
    scene.hand(hid, *BOX_AT)
    events += scene.run(world, 0.6)
    scene.hand(hid, *OUT_OF_BOX)
    return events + scene.run(world, 0.1)


def test_hand_into_box_and_out_with_keys_absent_is_put_inside(scene, world, cfg):
    events = carry_keys_into_box(scene, world)
    events += scene.run(world, cfg.reappear_wait_s + 0.2)
    keys = world.get('keys')
    assert types(events) == [EventType.PICKED_UP, EventType.PUT_INSIDE]
    assert events[1].parent == 'box'
    assert events[1].confidence == pytest.approx(cfg.conf_inside)
    assert (keys.status, keys.parent) == (Status.INSIDE, 'box')


def test_put_inside_waits_for_reappear_time_after_the_hand_leaves(scene, world, cfg):
    carry_keys_into_box(scene, world)
    assert scene.run(world, cfg.reappear_wait_s - 0.3) == []
    assert world.get('keys').status == Status.HELD


def test_hand_leaving_view_after_box_visit_still_gives_put_inside(scene, world, cfg):
    events = carry_keys_into_box(scene, world)
    scene.hand_off(1)                              # drops the keys in and walks away mid-table
    events += scene.run(world, cfg.reappear_wait_s + 0.2)
    assert types(events) == [EventType.PICKED_UP, EventType.PUT_INSIDE]
    assert world.get('keys').status == Status.INSIDE


def test_keys_reappearing_on_table_within_wait_are_visible_not_inside(scene, world, cfg):
    events = carry_keys_into_box(scene, world)
    scene.hand(1, 60, 55)
    scene.place('keys', 60, 55)                    # put down beside the box instead
    events += scene.run(world, cfg.reappear_wait_s + 1.0)
    assert EventType.PUT_INSIDE not in types(events)
    assert types(events)[-1] == EventType.MOVED
    assert world.get('keys').status == Status.VISIBLE


def test_cover_and_container_visited_by_a_hand_that_touched_the_keys_are_ambiguous(scene, world, cfg):
    scene.place('keys', 40, 30)
    scene.place('box', *BOX_AT)
    scene.place('notebook', 90, 10)
    scene.run(world, 1.0)
    scene.hand(1, 40, 30)                          # touches the keys (they stay on the table)
    scene.run(world, 0.3)
    scene.hand(1, *BOX_AT)                         # then goes into the box
    scene.run(world, 0.6)
    scene.hand_off(1)
    scene.run(world, 0.5)
    slide(scene, world, 'notebook', (90, 10), (40, 30))
    scene.remove('keys')
    events = scene.run(world, 1.0)
    keys = world.get('keys')
    assert types(events) == [EventType.COVERED]
    assert (keys.status, keys.parent, keys.candidates) == (Status.UNDER, 'notebook', ['box'])
    assert events[0].confidence == pytest.approx(cfg.conf_under * cfg.ambiguity_penalty)


# ----- propagation and nesting -----------------------------------------------------------------

def keys_in_box(scene, world, cfg):
    carry_keys_into_box(scene, world)
    scene.hand_off(1)
    scene.run(world, cfg.reappear_wait_s + 0.2)
    assert world.get('keys').status == Status.INSIDE


def pick_up_box(scene, world, hid=1):
    scene.hand(hid, *BOX_AT)
    scene.run(world, 0.3)
    scene.remove('box')
    return scene.run(world, 1.0)


def test_keys_inside_a_held_box_resolve_to_the_box_last_position(scene, world, cfg):
    keys_in_box(scene, world, cfg)
    events = pick_up_box(scene, world)
    assert types(events) == [EventType.PICKED_UP]
    assert world.get('box').status == Status.HELD
    assert world.get('keys').parent == 'box'
    assert world.resolve('keys') == (pytest.approx(BOX_AT), ['keys', 'box'])


def test_keys_inside_a_box_moved_30_cm_resolve_to_the_new_box_position(scene, world, cfg):
    keys_in_box(scene, world, cfg)
    pick_up_box(scene, world)
    scene.hand(1, 110, 40)
    scene.place('box', 110, 40)
    events = scene.run(world, 1.0)
    assert types(events) == [EventType.MOVED]
    assert world.get('keys').status == Status.INSIDE
    assert world.resolve('keys') == (pytest.approx((110, 40)), ['keys', 'box'])


def test_notebook_slid_over_box_with_keys_inside_nests_three_levels(scene, world, cfg):
    scene.place('notebook', 80, 10)
    keys_in_box(scene, world, cfg)
    slide(scene, world, 'notebook', (80, 10), BOX_AT)
    scene.remove('box')
    events = scene.run(world, 1.0)
    assert types(events) == [EventType.COVERED]
    assert world.get('box').parent == 'notebook'
    assert world.resolve('keys') == (pytest.approx(BOX_AT), ['keys', 'box', 'notebook'])


# ----- lifted cover ----------------------------------------------------------------------------

def test_new_rule_keys_load_from_config(cfg):
    assert cfg.lifted_overlap_max == pytest.approx(0.2)
    assert cfg.bg_update_every_s == pytest.approx(1.0)


def lift_notebook(scene, world):
    """Slide the notebook off the covered keys' spot (it stays on the table, detected)."""
    return slide(scene, world, 'notebook', (40, 30), (90, 30))


def test_notebook_lifted_with_keys_visible_is_uncovered(scene, world):
    cover_keys(scene, world)
    scene.place('keys', 40, 30)
    events = lift_notebook(scene, world) + scene.run(world, 1.0)
    assert types(events) == [EventType.UNCOVERED]
    assert world.get('keys').status == Status.VISIBLE


def test_notebook_lifted_with_nothing_under_is_lost_track_at_half_confidence(scene, world, cfg):
    cover_keys(scene, world)
    before = world.get('keys').confidence
    events = lift_notebook(scene, world) + scene.run(world, cfg.reappear_wait_s + 0.2)
    keys = world.get('keys')
    assert types(events) == [EventType.LOST_TRACK]
    assert (keys.status, keys.parent) == (Status.UNKNOWN, None)
    assert events[0].confidence == pytest.approx(before * cfg.lifted_cover_penalty, rel=1e-3)


def test_lifted_cover_waits_for_reappear_time_before_losing_track(scene, world, cfg):
    cover_keys(scene, world)
    events = lift_notebook(scene, world) + scene.run(world, cfg.reappear_wait_s - 0.8)
    assert events == []
    assert world.get('keys').status == Status.UNDER


def test_cover_taken_out_of_view_counts_as_lifted(scene, world, cfg):
    cover_keys(scene, world)
    scene.remove('notebook')                       # unexplained, so it is lost only after lost_grace_s
    events = scene.run(world, cfg.lost_grace_s + cfg.reappear_wait_s + 0.2)
    assert EventType.LOST_TRACK in types(events)
    assert world.get('keys').status == Status.UNKNOWN


def test_small_nudge_of_cover_keeps_keys_under(scene, world, cfg):
    cover_keys(scene, world)
    events = slide(scene, world, 'notebook', (40, 30), (48, 32)) + scene.run(world, cfg.reappear_wait_s + 0.5)
    assert events == []
    assert world.get('keys').status == Status.UNDER


# ----- grab and leave --------------------------------------------------------------------------

def test_hand_grabbing_keys_and_leaving_right_edge_at_once_is_gone_right(scene, world):
    scene.place('keys', 100, 30)
    scene.run(world, 1.0)
    scene.hand(1, 100, 30)
    scene.run(world, 0.3)
    scene.remove('keys')
    scene.hand(1, 113, 30)
    events = scene.run(world, 0.1)
    scene.hand(1, 125, 30)                         # hand box x 119-131: at the right edge
    events += scene.run(world, 0.1)
    scene.hand_off(1)                              # gone 0.2 s after the grab, before debounce
    events += scene.run(world, 1.5)
    keys = world.get('keys')
    assert types(events) == [EventType.PICKED_UP, EventType.EXITED_VIEW]
    assert events[1].edge == 'right'
    assert (keys.status, keys.edge, keys.parent) == (Status.GONE, 'right', None)


def test_hand_grabbing_keys_and_vanishing_mid_table_is_picked_up_then_lost(scene, world):
    scene.place('keys', 60, 30)
    scene.run(world, 1.0)
    scene.hand(1, 60, 30)
    scene.run(world, 0.3)
    scene.remove('keys')
    scene.hand_off(1)
    events = scene.run(world, 1.5)
    assert types(events) == [EventType.PICKED_UP, EventType.LOST_TRACK]
    assert world.get('keys').status == Status.UNKNOWN


# ----- BackgroundModel.known -------------------------------------------------------------------

def grey(h=200, w=200, value=100):
    return np.full((h, w), value, np.uint8)


def test_background_known_false_before_any_frame():
    assert not BackgroundModel(5, 25.0).known((50, 50, 90, 90))


def test_background_known_true_for_region_seen_unexcluded():
    bg = BackgroundModel(5, 25.0)
    bg.update(grey(), [])
    assert bg.known((50, 50, 90, 90))


def test_background_known_false_for_region_always_excluded():
    bg = BackgroundModel(5, 25.0)
    for _ in range(5):
        bg.update(grey(), [(50, 50, 90, 90)])
    assert not bg.known((50, 50, 90, 90))


def test_background_known_needs_half_of_the_box():
    bg = BackgroundModel(5, 25.0)
    bg.update(grey(), [(50, 50, 90, 90)])
    assert bg.known((10, 50, 90, 90))              # 40 of 80 px columns seen: 50%
    assert not bg.known((20, 50, 90, 90))          # 30 of 70 columns seen: 43%


def test_background_known_box_outside_image_is_false():
    bg = BackgroundModel(5, 25.0)
    bg.update(grey(), [])
    assert not bg.known((300, 300, 340, 340))


# ----- image rules (rendered synthetic frames) -------------------------------------------------

@pytest.fixture
def rscene(cfg):
    return Scene(cfg, fps=10, t0=1000.0, render=True)


def test_detector_missing_keys_whose_pixels_are_still_there_keeps_them_visible(rscene, world):
    rscene.place('keys', 40, 30)
    rscene.run(world, 2.0)
    rscene.miss('keys')
    events = rscene.run(world, 2.0)
    assert events == []
    assert world.get('keys').status == Status.VISIBLE
    rscene.miss('keys', False)
    assert rscene.run(world, 1.0) == []


def test_detector_miss_with_jittery_boxes_still_keeps_keys_visible(cfg, world):
    rscene = Scene(cfg, fps=10, t0=1000.0, render=True, jitter_px=3)
    rscene.place('keys', 40, 30)
    rscene.run(world, 2.0)
    rscene.miss('keys')
    assert rscene.run(world, 2.0) == []
    assert world.get('keys').status == Status.VISIBLE


def test_detector_miss_right_after_a_touch_is_not_a_pick_up(rscene, world):
    rscene.place('keys', 40, 30)
    rscene.run(world, 2.0)
    rscene.hand(1, 40, 30)
    rscene.run(world, 0.3)
    rscene.hand(1, 70, 50)                         # hand withdraws; the keys stay where they were
    rscene.miss('keys')
    events = rscene.run(world, 1.5)
    assert events == []
    assert world.get('keys').status == Status.VISIBLE


def let_background_settle(rscene, world, cfg):
    """Enough refreshes for the background model to be ready."""
    rscene.run(world, (BackgroundModel.MIN_READY + 0.5) * cfg.bg_update_every_s)


def test_keys_covered_by_something_undetected_are_under_unknown(rscene, world, cfg):
    let_background_settle(rscene, world, cfg)
    rscene.place('keys', 40, 30)
    rscene.run(world, 2.0)
    rscene.remove('keys')
    rscene.overlay('magazine', 41, 31, 16, 12)     # no detector class, no cover motion
    events = rscene.run(world, 1.5)
    keys = world.get('keys')
    assert types(events) == [EventType.COVERED]
    assert events[0].confidence == pytest.approx(cfg.conf_under_unknown)
    assert (keys.status, keys.parent) == (Status.UNDER, 'unknown')


def test_object_there_since_startup_then_removed_is_lost_track_not_under_unknown(cfg, world):
    # Detection boxes wobble a few px, so strips of the keys' own pixels get learned as background;
    # without known() those strips alone would read as 'something new covers the spot'.
    rscene = Scene(cfg, fps=10, t0=1000.0, render=True, jitter_px=6)
    rscene.place('keys', 40, 30)
    let_background_settle(rscene, world, cfg)
    rscene.remove('keys')
    events = rscene.run(world, cfg.lost_grace_s + 0.3)
    assert types(events) == [EventType.LOST_TRACK]
    assert world.get('keys').status == Status.UNKNOWN


# ----- headline scenario (spec checkpoint D9) --------------------------------------------------

def test_headline_keys_put_in_box_and_box_carried_30_cm(rscene, world, cfg):
    s, w = rscene, world
    s.place('keys', 40, 30)
    s.place('box', 80, 40)
    s.place('wallet', 20, 60)                      # bystanders must stay quiet throughout
    s.place('notebook', 100, 12)
    events = s.run(w, 2.0)
    s.hand(1, 40, 30)                              # grab the keys
    events += s.run(w, 0.4)
    s.remove('keys')
    for x, y in [(45, 31), (52, 33), (60, 35), (68, 37), (75, 39)]:    # carry them over
        s.hand(1, x, y)
        events += w.update(*s.step())
    s.hand(1, 80, 40)                              # into the box
    events += s.run(w, 0.8)
    s.hand(1, 95, 55)                              # hand withdraws and rests beside it
    events += s.run(w, cfg.reappear_wait_s + 0.5)
    s.hand(1, 80, 40)                              # grab the box
    events += s.run(w, 0.4)
    s.remove('box')
    for x in range(83, 111, 3):                    # carry it 30 cm to the right
        s.hand(1, x, 40)
        events += w.update(*s.step())
    s.place('box', 110, 40)                        # set it down, let go, leave
    events += s.run(w, 0.5)
    s.hand_off(1)
    events += s.run(w, 2.0)

    assert [(e.obj, e.type) for e in events] == [
        ('keys', EventType.PICKED_UP), ('keys', EventType.PUT_INSIDE),
        ('box', EventType.PICKED_UP), ('box', EventType.MOVED)]
    keys = w.get('keys')
    assert (keys.status, keys.parent) == (Status.INSIDE, 'box')
    assert w.resolve('keys') == (pytest.approx((110, 40)), ['keys', 'box'])
    assert w.state_json()['edges'][0] == ['keys', 'INSIDE', 'box']


# ----- gap fixes: a hand on the moving cover; objects moved in plain view ------------------------

def of(events, obj):
    return [e for e in events if e.obj == obj]


def test_notebook_slid_over_keys_by_a_hand_on_it_is_covered_not_picked_up(scene, world, cfg):
    """The hand holding the notebook passes right over the keys: that touch is the cover's, not a pick-up."""
    scene.place('keys', 40, 30)
    scene.place('notebook', 90, 30)
    scene.run(world, 1.0)
    events = []
    for i in range(1, 6):                          # hand and notebook slide in together
        scene.place('notebook', 90 - 10 * i, 30)
        scene.hand(1, 90 - 10 * i, 30)
        events += world.update(*scene.step())
    scene.remove('keys')
    events += scene.run(world, 0.3)
    scene.hand_off(1)
    events += scene.run(world, 1.0)
    keys = world.get('keys')
    assert types(of(events, 'keys')) == [EventType.COVERED]
    assert (keys.status, keys.parent) == (Status.UNDER, 'notebook')
    assert of(events, 'keys')[0].confidence == pytest.approx(cfg.conf_under)


def carry_in_view(scene, world, name, frm, to, hid=1, steps=10):
    """Hand grabs name and carries it (still detected) from frm to to, one step per batch."""
    events = []
    for i in range(1, steps + 1):
        f = i / steps
        p = (frm[0] + (to[0] - frm[0]) * f, frm[1] + (to[1] - frm[1]) * f)
        scene.place(name, *p)
        scene.hand(hid, *p)
        events += world.update(*scene.step())
    return events


def test_object_carried_in_view_and_set_down_elsewhere_is_picked_up_then_moved(scene, world):
    scene.place('keys', 40, 30)
    scene.run(world, 1.0)
    scene.hand(1, 40, 30)
    events = scene.run(world, 0.3)
    events += carry_in_view(scene, world, 'keys', (40, 30), (70, 30))
    scene.hand(1, 70, 55)                          # lets go and moves away
    events += scene.run(world, 0.3)
    scene.hand_off(1)
    events += scene.run(world, 1.0)
    assert types(events) == [EventType.PICKED_UP, EventType.MOVED]
    assert events[0].parent == 'hand:1' and events[0].from_cm == pytest.approx((40, 30))
    assert events[1].from_cm == pytest.approx((40, 30)) and events[1].to_cm == pytest.approx((70, 30))
    assert world.get('keys').status == Status.VISIBLE


def test_object_lifted_in_view_and_returned_is_picked_up_then_put_back(scene, world):
    scene.place('wallet', 40, 30)
    scene.run(world, 1.0)
    scene.hand(1, 40, 30)
    events = scene.run(world, 0.3)
    events += carry_in_view(scene, world, 'wallet', (40, 30), (40, 42), steps=5)
    events += carry_in_view(scene, world, 'wallet', (40, 42), (41, 30), steps=5)
    scene.hand(1, 40, 55)
    events += scene.run(world, 0.3)
    scene.hand_off(1)
    events += scene.run(world, 1.0)
    assert types(events) == [EventType.PICKED_UP, EventType.PUT_BACK]


def test_hand_resting_on_an_object_that_does_not_move_logs_nothing(scene, world):
    scene.place('keys', 40, 30)
    scene.run(world, 1.0)
    scene.hand(1, 40, 30)
    events = scene.run(world, 2.0)
    scene.hand_off(1)
    events += scene.run(world, 1.0)
    assert events == []
    assert world.get('keys').status == Status.VISIBLE


def test_object_carried_in_view_off_the_left_edge_is_one_pick_up_then_exited_view(scene, world):
    scene.place('keys', 40, 30)
    scene.run(world, 1.0)
    scene.hand(1, 40, 30)
    events = scene.run(world, 0.3)
    events += carry_in_view(scene, world, 'keys', (40, 30), (4, 30))
    scene.remove('keys')
    scene.hand_off(1)
    events += scene.run(world, 1.5)
    assert types(events) == [EventType.PICKED_UP, EventType.EXITED_VIEW]
    assert world.get('keys').status == Status.GONE and world.get('keys').edge == 'left'


# ----- a hand over an object hides it: decide when the hand has left the spot ------------------------

def wave(scene, world, xs, y=30, hid=1):
    """The hand sweeps through xs, one position per batch."""
    events = []
    for x in xs:
        scene.hand(hid, x, y)
        events += world.update(*scene.step())
    return events


def test_hand_waving_over_keys_that_the_detector_misses_logs_nothing(rscene, world):
    """A slow wave back and forth over the keys: the detector misses them the whole time (a moving arm
    nearby), long enough to count as absence. Uncovered, their pixels show they are there; covered,
    the spot cannot be seen until the hand moves on, and then they are there."""
    rscene.place('keys', 60, 30)
    rscene.run(world, 2.0)
    rscene.miss('keys')
    events = wave(rscene, world, [52, 56, 60, 64, 68, 64, 60, 56, 52, 56, 60, 64, 68, 64, 60])
    events += wave(rscene, world, [75, 85, 95])
    rscene.miss('keys', False)
    events += wave(rscene, world, [105, 115])
    rscene.hand_off(1)
    events += rscene.run(world, 1.0)
    assert events == []
    assert world.get('keys').status == Status.VISIBLE


def test_keys_gone_when_the_waving_hand_moves_on_are_picked_up_by_it(scene, world):
    scene.place('keys', 60, 30)
    scene.run(world, 1.0)
    scene.hand(1, 60, 30)
    scene.run(world, 0.2)
    scene.remove('keys')                            # grabbed, and the hand is already moving on
    events = wave(scene, world, [57, 63, 57, 63, 57, 63, 57, 63])
    assert events == []                             # the moving hand still covers the spot: undecided
    events += wave(scene, world, [75, 85, 90, 90, 90])
    assert types(events) == [EventType.PICKED_UP]
    assert (world.get('keys').status, world.get('keys').parent) == (Status.HELD, 'hand:1')


def test_a_hand_over_part_of_the_box_is_not_a_pick_up(scene, world):
    """The detector's box shrinks to the part of the box the hand leaves uncovered: its centre moves 6 cm
    with a hand on it, but the box has not left where it lay. Partly hidden, not carried."""
    scene.place('box', 40, 30)                      # 20 x 15 cm: x 30-50, y 22.5-37.5
    scene.run(world, 1.0)
    scene.hand(1, 40, 30)
    scene.place('box', 34, 30, w=8, h=15)           # only the left end shows
    events = scene.run(world, 1.0)
    scene.hand_off(1)
    scene.place('box', 40, 30)
    events += scene.run(world, 1.0)
    assert events == []
    assert world.get('box').status == Status.VISIBLE


# ----- briefly unseen: an undetected arm or a detector miss is not a loss (shell_1) -------------------

def test_keys_unseen_briefly_then_seen_in_place_log_nothing(scene, world, cfg):
    """No image, no hand: the detector drops the keys for most of the grace, then sees them where they were."""
    scene.place('keys', 40, 30)
    scene.run(world, 1.0)
    scene.miss('keys')
    events = scene.run(world, cfg.lost_grace_s - 0.3)
    assert world.get('keys').status == Status.VISIBLE
    assert world.resolve('keys') == (pytest.approx((40, 30)), ['keys'])     # 'where is it': its spot
    scene.miss('keys', False)
    events += scene.run(world, 1.0)
    assert events == []
    assert world.get('keys').status == Status.VISIBLE


def test_untouched_keys_half_hidden_by_an_undetected_arm_log_nothing(cfg, world):
    """shell_1: an arm reaching past (no hand detected) hides part of the wallet, which was there since
    start-up, and the detector drops it for ~1.8 s. Its patch does not match while the arm is over it and
    the background never saw that spot. Seen again where it was: no LOST_TRACK / CORRECTED churn."""
    s = Scene(cfg, fps=10, t0=1000.0, render=True)
    s.place('keys', 40, 30)
    let_background_settle(s, world, cfg)
    s.miss('keys')
    s.overlay('sleeve', 46, 30, 20, 10)            # x 36-56: over most of the keys (x 37-43)
    events = s.run(world, 1.8)
    assert world.get('keys').status == Status.VISIBLE
    del s.overlays['sleeve']
    s.miss('keys', False)
    events += s.run(world, 1.0)
    assert events == []
    assert world.get('keys').status == Status.VISIBLE


def test_keys_removed_without_a_hand_are_lost_when_the_grace_runs_out(scene, world, cfg):
    scene.place('keys', 40, 30)
    scene.run(world, 1.0)
    scene.remove('keys')
    assert scene.run(world, cfg.lost_grace_s - 0.2) == []
    events = scene.run(world, 0.4)
    assert types(events) == [EventType.LOST_TRACK]
    assert world.get('keys').status == Status.UNKNOWN


def test_keys_covered_while_unseen_are_covered_at_once(scene, world, cfg):
    """The grace only defers LOST_TRACK: a cover slid over keys the detector already missed is COVERED
    as soon as it lies over them."""
    scene.place('keys', 40, 30)
    scene.place('notebook', 90, 30)
    scene.run(world, 1.0)
    scene.miss('keys')
    scene.run(world, 1.0)                          # absence declared, grace running
    events = slide(scene, world, 'notebook', (90, 30), (40, 30))
    assert types(of(events, 'keys')) == [EventType.COVERED]
    assert world.get('keys').parent == 'notebook'


def test_a_hand_seen_at_keys_already_unseen_does_not_pick_them_up(scene, world, cfg):
    """shell_1 24.25 s: the arm reaching for the notebook hid the phone (no hand detected), then one frame
    of a hand over the phone. The phone's absence was already unexplained before that hand came: it did
    not take the phone, which is seen again where it was."""
    scene.place('keys', 40, 30)
    scene.run(world, 1.0)
    scene.miss('keys')
    events = scene.run(world, 1.0)                 # absent with nothing to explain it: waiting
    scene.hand(1, 40, 30)
    events += scene.run(world, 0.1)                # one frame of a hand at rest over the spot
    scene.hand_off(1)
    events += scene.run(world, 0.3)
    scene.miss('keys', False)
    events += scene.run(world, 1.0)
    assert events == []
    assert world.get('keys').status == Status.VISIBLE


def test_keys_taken_while_already_unseen_are_lost_when_the_grace_runs_out(scene, world, cfg):
    scene.place('keys', 40, 30)
    scene.run(world, 1.0)
    scene.remove('keys')
    events = scene.run(world, 1.0)
    scene.hand(1, 40, 30)
    events += scene.run(world, 0.3)
    scene.hand_off(1)
    events += scene.run(world, cfg.lost_grace_s)
    assert types(events) == [EventType.LOST_TRACK]
    assert world.get('keys').status == Status.UNKNOWN


# ----- a configured label jumping away while the object's pixels stay put ----------------------------

def test_a_label_jumping_away_from_keys_still_in_place_is_another_object(rscene, world, cfg):
    """The detector calls a neighbour 'keys' for a few frames (a label swap) while the real keys lie
    untouched where they were: their pixels still match there, so the keys do not move."""
    from tests.synth import DEFAULT_SIZE_CM
    w, h = DEFAULT_SIZE_CM['target']
    rscene.place('keys', 40, 30)
    rscene.run(world, 2.0)
    rscene.hand(1, 36, 30)                          # a hand brushes the keys just before
    rscene.run(world, 0.2)
    rscene.hand_off(1)
    rscene.overlay('keys', 40, 30, w, h)            # the real keys stay drawn at their spot ...
    rscene.place('keys', 48, 30)                    # ... while the 'keys' detection is 8 cm away
    events = rscene.run(world, 0.5)
    assert abs(world.get('keys').pos_cm[0] - 40) < 2
    rscene.place('keys', 40, 30)
    events += rscene.run(world, 1.0)
    assert events == []
    keys = world.get('keys')
    assert keys.status == Status.VISIBLE and abs(keys.pos_cm[0] - 40) < 2
