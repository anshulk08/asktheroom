import cv2
import numpy as np
import pytest

from core.relations import (AppearanceMemory, BackgroundModel, CoverMotion, CoverState, DwellTracker, covering_candidates,
                            descendants, resolve_chain, touching_hands)
from core.types import Detection, Entity


def hand(cls, box_cm):
    cx, cy = (box_cm[0] + box_cm[2]) / 2, (box_cm[1] + box_cm[3]) / 2
    return Detection(cls=cls, conf=0.9, box_px=(0, 0, 1, 1), center_cm=(cx, cy), box_cm=box_cm)


# --- touching_hands -------------------------------------------------------------------------

def test_touching_hands_returns_hand_covering_enough_of_object():
    obj = (0.0, 0.0, 10.0, 10.0)
    assert touching_hands(obj, [hand('hand:1', (5.0, 0.0, 20.0, 10.0))], 0.3) == ['hand:1']


def test_touching_hands_ignores_hand_below_min_overlap():
    obj = (0.0, 0.0, 10.0, 10.0)
    assert touching_hands(obj, [hand('hand:1', (8.0, 0.0, 20.0, 10.0))], 0.3) == []


def test_touching_hands_counts_overlap_exactly_at_threshold():
    obj = (0.0, 0.0, 10.0, 10.0)
    assert touching_hands(obj, [hand('hand:1', (7.0, 0.0, 20.0, 10.0))], 0.3) == ['hand:1']


def test_touching_hands_returns_all_touching_hands_in_input_order():
    obj = (0.0, 0.0, 10.0, 10.0)
    hands = [hand('hand:2', (0.0, 0.0, 5.0, 10.0)), hand('hand:7', (50.0, 50.0, 60.0, 60.0)),
             hand('hand:1', (5.0, 0.0, 10.0, 10.0))]
    assert touching_hands(obj, hands, 0.3) == ['hand:2', 'hand:1']


# --- CoverMotion ----------------------------------------------------------------------------

def test_cover_motion_state_is_none_for_unseen_cover():
    assert CoverMotion(2.0).state('notebook') is None


def test_cover_motion_first_update_has_no_movement():
    cm = CoverMotion(2.0)
    cm.update('notebook', (0.0, 0.0, 10.0, 10.0), 1.0)
    assert cm.state('notebook') == CoverState(box_cm=(0.0, 0.0, 10.0, 10.0), moved_at=None)


def test_cover_motion_small_jitter_is_not_movement():
    cm = CoverMotion(2.0)
    cm.update('notebook', (0.0, 0.0, 10.0, 10.0), 1.0)
    cm.update('notebook', (1.0, 0.0, 11.0, 10.0), 2.0)
    assert cm.state('notebook').moved_at is None


def test_cover_motion_always_stores_latest_box():
    cm = CoverMotion(2.0)
    cm.update('notebook', (0.0, 0.0, 10.0, 10.0), 1.0)
    cm.update('notebook', (1.0, 0.0, 11.0, 10.0), 2.0)
    assert cm.state('notebook').box_cm == (1.0, 0.0, 11.0, 10.0)


def test_cover_motion_shift_of_min_move_records_moved_at():
    cm = CoverMotion(2.0)
    cm.update('notebook', (0.0, 0.0, 10.0, 10.0), 1.0)
    cm.update('notebook', (2.0, 0.0, 12.0, 10.0), 2.5)
    assert cm.state('notebook').moved_at == 2.5


def test_cover_motion_slow_drift_accumulates_against_anchor():
    cm = CoverMotion(2.0)
    cm.update('notebook', (0.0, 0.0, 10.0, 10.0), 1.0)
    cm.update('notebook', (1.0, 0.0, 11.0, 10.0), 2.0)
    cm.update('notebook', (2.5, 0.0, 12.5, 10.0), 3.0)
    assert cm.state('notebook').moved_at == 3.0


def test_cover_motion_re_anchors_after_move():
    cm = CoverMotion(2.0)
    cm.update('notebook', (0.0, 0.0, 10.0, 10.0), 1.0)
    cm.update('notebook', (5.0, 0.0, 15.0, 10.0), 2.0)
    cm.update('notebook', (6.0, 0.0, 16.0, 10.0), 3.0)   # 1 cm from new anchor
    assert cm.state('notebook').moved_at == 2.0


def test_cover_motion_tracks_covers_independently():
    cm = CoverMotion(2.0)
    cm.update('notebook', (0.0, 0.0, 10.0, 10.0), 1.0)
    cm.update('tray', (50.0, 0.0, 60.0, 10.0), 1.0)
    cm.update('notebook', (5.0, 0.0, 15.0, 10.0), 2.0)
    assert cm.state('tray').moved_at is None


# --- covering_candidates --------------------------------------------------------------------

LAST = (10.0, 10.0, 20.0, 20.0)


def test_covering_candidates_includes_recently_moved_overlapping_cover():
    covers = {'notebook': CoverState((5.0, 5.0, 25.0, 25.0), moved_at=9.0)}
    assert covering_candidates(LAST, covers, 10.0, 0.6, 3.0) == ['notebook']


def test_covering_candidates_excludes_cover_that_never_moved():
    covers = {'notebook': CoverState((5.0, 5.0, 25.0, 25.0), moved_at=None)}
    assert covering_candidates(LAST, covers, 10.0, 0.6, 3.0) == []


def test_covering_candidates_excludes_cover_moved_too_long_ago():
    covers = {'notebook': CoverState((5.0, 5.0, 25.0, 25.0), moved_at=6.9)}
    assert covering_candidates(LAST, covers, 10.0, 0.6, 3.0) == []


def test_covering_candidates_includes_cover_moved_exactly_at_window_edge():
    covers = {'notebook': CoverState((5.0, 5.0, 25.0, 25.0), moved_at=7.0)}
    assert covering_candidates(LAST, covers, 10.0, 0.6, 3.0) == ['notebook']


def test_covering_candidates_excludes_cover_with_too_little_overlap():
    covers = {'notebook': CoverState((15.0, 5.0, 25.0, 25.0), moved_at=9.0)}   # covers 50%
    assert covering_candidates(LAST, covers, 10.0, 0.6, 3.0) == []


def test_covering_candidates_sorted_by_overlap_largest_first():
    covers = {'tray': CoverState((13.0, 5.0, 25.0, 25.0), moved_at=9.0),        # 70%
              'notebook': CoverState((5.0, 5.0, 25.0, 25.0), moved_at=9.5)}     # 100%
    assert covering_candidates(LAST, covers, 10.0, 0.6, 3.0) == ['notebook', 'tray']


# --- DwellTracker ---------------------------------------------------------------------------

BOX = {'box': (0.0, 0.0, 20.0, 20.0)}
IN = (8.0, 8.0, 12.0, 12.0)        # centre (10, 10): inside BOX
OUT = (48.0, 48.0, 52.0, 52.0)     # centre (50, 50): outside


def run_dwell(tracker, samples, containers=BOX):
    """samples: list of (t, [hand dets])"""
    for t, hands in samples:
        tracker.update(t, hands, containers)


def test_dwell_no_visits_for_unknown_hand():
    assert DwellTracker(0.3).visits_since('hand:1', 0.0) == []


def test_dwell_records_qualifying_visit_on_exit():
    dt = DwellTracker(0.3)
    run_dwell(dt, [(0.0, [hand('hand:1', IN)]), (0.2, [hand('hand:1', IN)]),
                   (0.4, [hand('hand:1', OUT)])])
    assert dt.visits_since('hand:1', 0.0) == [('box', 0.4)]


def test_dwell_visit_in_progress_is_not_returned():
    dt = DwellTracker(0.3)
    run_dwell(dt, [(0.0, [hand('hand:1', IN)]), (1.0, [hand('hand:1', IN)])])
    assert dt.visits_since('hand:1', 0.0) == []


def test_dwell_too_short_visit_does_not_qualify():
    dt = DwellTracker(0.3)
    run_dwell(dt, [(0.0, [hand('hand:1', IN)]), (0.2, [hand('hand:1', OUT)])])
    assert dt.visits_since('hand:1', 0.0) == []


def test_dwell_visit_of_exactly_min_dwell_qualifies():
    dt = DwellTracker(0.5)
    run_dwell(dt, [(1.0, [hand('hand:1', IN)]), (1.5, [hand('hand:1', OUT)])])
    assert dt.visits_since('hand:1', 0.0) == [('box', 1.5)]


def test_dwell_uses_hand_box_centre_not_overlap():
    dt = DwellTracker(0.3)
    edge = (15.0, 15.0, 35.0, 35.0)   # overlaps BOX but centre (25, 25) is outside
    run_dwell(dt, [(0.0, [hand('hand:1', edge)]), (1.0, [hand('hand:1', OUT)])])
    assert dt.visits_since('hand:1', 0.0) == []


def test_dwell_hand_disappearing_closes_visit():
    dt = DwellTracker(0.3)
    run_dwell(dt, [(0.0, [hand('hand:1', IN)]), (0.5, [hand('hand:1', IN)]), (0.7, [])])
    assert dt.visits_since('hand:1', 0.0) == [('box', 0.7)]


def test_dwell_container_disappearing_closes_visit():
    dt = DwellTracker(0.3)
    dt.update(0.0, [hand('hand:1', IN)], BOX)
    dt.update(0.5, [hand('hand:1', IN)], BOX)
    dt.update(0.8, [hand('hand:1', IN)], {})
    assert dt.visits_since('hand:1', 0.0) == [('box', 0.8)]


def test_dwell_visits_since_filters_by_exit_time():
    dt = DwellTracker(0.3)
    h = lambda b: [hand('hand:1', b)]
    run_dwell(dt, [(0.0, h(IN)), (1.0, h(OUT)), (2.0, h(IN)), (3.0, h(OUT))])
    assert dt.visits_since('hand:1', 1.5) == [('box', 3.0)]


def test_dwell_visits_since_returns_oldest_first():
    dt = DwellTracker(0.3)
    h = lambda b: [hand('hand:1', b)]
    containers = {'box': (0.0, 0.0, 20.0, 20.0), 'bin': (100.0, 0.0, 120.0, 20.0)}
    in_bin = (108.0, 8.0, 112.0, 12.0)
    run_dwell(dt, [(0.0, h(IN)), (1.0, h(OUT)), (2.0, h(in_bin)), (3.0, h(OUT))], containers)
    assert dt.visits_since('hand:1', 0.0) == [('box', 1.0), ('bin', 3.0)]


def test_dwell_tracks_hands_separately():
    dt = DwellTracker(0.3)
    run_dwell(dt, [(0.0, [hand('hand:1', IN), hand('hand:2', OUT)]),
                   (1.0, [hand('hand:1', OUT), hand('hand:2', OUT)])])
    assert dt.visits_since('hand:2', 0.0) == []


def test_dwell_forgets_visits_older_than_history_window():
    dt = DwellTracker(0.3)
    h = lambda b: [hand('hand:1', b)]
    run_dwell(dt, [(0.0, h(IN)), (1.0, h(OUT)), (200.0, h(OUT))])
    assert dt.visits_since('hand:1', 0.0) == []


# --- resolve_chain / descendants ------------------------------------------------------------

def ent(name, parent=None, pos=None, kind='target'):
    return Entity(name=name, kind=kind, parent=parent, pos_cm=pos)


def nested_world():
    return {'keys': ent('keys', 'box', (10.0, 10.0)),
            'box': ent('box', 'notebook', (60.0, 30.0), 'container'),
            'notebook': ent('notebook', None, (70.0, 38.0), 'cover'),
            'phone': ent('phone', None, (5.0, 5.0))}


def test_resolve_chain_follows_parents_to_outermost_entity():
    assert resolve_chain('keys', nested_world(), 3) == ((70.0, 38.0), ['keys', 'box', 'notebook'])


def test_resolve_chain_without_parent_is_just_itself():
    assert resolve_chain('phone', nested_world(), 3) == ((5.0, 5.0), ['phone'])


def test_resolve_chain_stops_at_hand_parent():
    world = {'keys': ent('keys', 'hand:2', (10.0, 10.0))}
    assert resolve_chain('keys', world, 3) == ((10.0, 10.0), ['keys'])


def test_resolve_chain_stops_at_unknown_parent():
    world = {'keys': ent('keys', 'unknown', (10.0, 10.0))}
    assert resolve_chain('keys', world, 3) == ((10.0, 10.0), ['keys'])


def test_resolve_chain_respects_max_depth():
    assert resolve_chain('keys', nested_world(), 1) == ((60.0, 30.0), ['keys', 'box'])


def test_resolve_chain_returns_last_entity_position_even_if_none():
    world = {'keys': ent('keys', 'box', (10.0, 10.0)), 'box': ent('box', None, None)}
    assert resolve_chain('keys', world, 3) == (None, ['keys', 'box'])


def test_resolve_chain_guards_against_cycles():
    world = {'a': ent('a', 'b', (1.0, 1.0)), 'b': ent('b', 'a', (2.0, 2.0))}
    assert resolve_chain('a', world, 10) == ((2.0, 2.0), ['a', 'b'])


def test_resolve_chain_unknown_name_is_empty():
    assert resolve_chain('ghost', nested_world(), 3) == (None, [])


def test_descendants_lists_direct_children_first():
    assert descendants('notebook', nested_world(), 3) == ['box', 'keys']


def test_descendants_of_leaf_is_empty():
    assert descendants('keys', nested_world(), 3) == []


def test_descendants_respects_max_depth():
    assert descendants('notebook', nested_world(), 1) == ['box']


def test_descendants_includes_all_siblings():
    world = nested_world()
    world['wallet'] = ent('wallet', 'box', (12.0, 12.0))
    assert descendants('box', world, 3) == ['keys', 'wallet']


def test_descendants_guards_against_cycles():
    world = {'a': ent('a', 'b'), 'b': ent('b', 'a')}
    assert descendants('a', world, 10) == ['b']


# --- BackgroundModel ------------------------------------------------------------------------

TABLE = 100
OBJ = (50, 50, 90, 90)          # x1, y1, x2, y2 px
ELSEWHERE = (120, 120, 160, 160)


def table_img(h=200, w=200):
    return np.full((h, w), TABLE, np.uint8)


def with_patch(img, box, value):
    out = img.copy()
    out[box[1]:box[3], box[0]:box[2]] = value
    return out


def test_background_not_ready_before_five_frames():
    bg = BackgroundModel(30, 25.0)
    for _ in range(4):
        bg.update(table_img(), [])
    assert not bg.ready


def test_background_ready_after_five_frames():
    bg = BackgroundModel(30, 25.0)
    for _ in range(5):
        bg.update(table_img(), [])
    assert bg.ready


def test_background_ready_after_n_frames_when_n_below_five():
    bg = BackgroundModel(3, 25.0)
    for _ in range(3):
        bg.update(table_img(), [])
    assert bg.ready


def test_background_reports_changed_for_painted_box():
    bg = BackgroundModel(10, 25.0)
    for _ in range(5):
        bg.update(table_img(), [])
    assert bg.changed(with_patch(table_img(), OBJ, 200), OBJ)


def test_background_reports_unchanged_for_untouched_region():
    bg = BackgroundModel(10, 25.0)
    for _ in range(5):
        bg.update(table_img(), [])
    assert not bg.changed(with_patch(table_img(), OBJ, 200), ELSEWHERE)


def test_background_small_difference_is_not_change():
    bg = BackgroundModel(10, 25.0)
    for _ in range(5):
        bg.update(table_img(), [])
    assert not bg.changed(with_patch(table_img(), OBJ, TABLE + 10), OBJ)


def test_background_unexcluded_change_is_absorbed_over_time():
    bg = BackgroundModel(5, 25.0)
    moved = with_patch(table_img(), OBJ, 200)
    for _ in range(5):
        bg.update(table_img(), [])
    for _ in range(5):
        bg.update(moved, [])
    assert not bg.changed(moved, OBJ)


def test_background_excluded_stationary_object_never_bleeds_in():
    bg = BackgroundModel(5, 25.0)
    with_obj = with_patch(table_img(), OBJ, 200)
    for _ in range(5):
        bg.update(table_img(), [])
    for _ in range(20):                      # 4x the ring length, always excluded
        bg.update(with_obj, [OBJ])
    assert bg.changed(with_obj, OBJ) and not bg.changed(table_img(), OBJ)


def test_background_object_present_from_first_frame_does_not_bleed_in():
    bg = BackgroundModel(5, 25.0)
    with_obj = with_patch(table_img(), OBJ, 200)
    for _ in range(10):
        bg.update(with_obj, [OBJ])
    bg.update(table_img(), [])               # object lifted: table seen once
    for _ in range(10):
        bg.update(with_obj, [OBJ])
    assert bg.changed(with_obj, OBJ) and not bg.changed(table_img(), OBJ)


def test_background_never_observed_region_reports_unchanged():
    bg = BackgroundModel(5, 25.0)
    with_obj = with_patch(table_img(), OBJ, 200)
    for _ in range(5):
        bg.update(with_obj, [OBJ])
    assert not bg.changed(with_obj, OBJ)


def test_background_accepts_bgr_frames():
    bg = BackgroundModel(10, 25.0)
    bgr = lambda g: np.repeat(g[:, :, None], 3, axis=2)
    for _ in range(5):
        bg.update(bgr(table_img()), [])
    assert bg.changed(bgr(with_patch(table_img(), OBJ, 200)), OBJ)


def test_background_clips_box_to_image_bounds():
    bg = BackgroundModel(10, 25.0)
    for _ in range(5):
        bg.update(table_img(), [])
    img = with_patch(table_img(), (180, 180, 200, 200), 200)
    assert bg.changed(img, (180, 180, 260, 260))


def test_background_box_fully_outside_image_is_unchanged():
    bg = BackgroundModel(10, 25.0)
    for _ in range(5):
        bg.update(table_img(), [])
    assert not bg.changed(table_img(), (300, 300, 340, 340))


def test_background_full_hd_frame_uses_full_resolution_box_coords():
    bg = BackgroundModel(10, 25.0)
    for _ in range(5):
        bg.update(table_img(720, 1280), [])
    box = (1000, 600, 1040, 640)
    img = with_patch(table_img(720, 1280), box, 200)
    assert bg.changed(img, box) and not bg.changed(img, (200, 200, 240, 240))


def test_background_full_hd_excluded_object_never_bleeds_in():
    bg = BackgroundModel(5, 25.0)
    box = (1000, 600, 1043, 641)             # not aligned to the internal downscale grid
    with_obj = with_patch(table_img(720, 1280), box, 200)
    for _ in range(5):
        bg.update(table_img(720, 1280), [])
    for _ in range(10):
        bg.update(with_obj, [box])
    assert bg.changed(with_obj, box)


# --- AppearanceMemory -----------------------------------------------------------------------

PATCH = (40, 40, 80, 80)


def texture(seed, size=40, block=8):
    rng = np.random.default_rng(seed)
    blocks = rng.integers(30, 220, (size // block, size // block)).astype(np.uint8)
    return np.kron(blocks, np.ones((block, block), np.uint8))


def scene(patch, box=PATCH):
    img = table_img()
    img[box[1]:box[3], box[0]:box[2]] = patch
    return img


def remembered(img=None):
    mem = AppearanceMemory()
    mem.remember('wallet', scene(texture(1)) if img is None else img, PATCH)
    return mem


def test_appearance_unknown_name_is_not_there():
    assert not AppearanceMemory().still_there('wallet', scene(texture(1)), PATCH, 0.7)


def test_appearance_same_patch_is_still_there():
    assert remembered().still_there('wallet', scene(texture(1)), PATCH, 0.7)


def test_appearance_different_texture_is_not_there():
    assert not remembered().still_there('wallet', scene(texture(2)), PATCH, 0.7)


def test_appearance_flat_table_is_not_there():
    assert not remembered().still_there('wallet', table_img(), PATCH, 0.7)


def test_appearance_small_brightness_change_is_still_there():
    brighter = scene(texture(1) + 10)
    assert remembered().still_there('wallet', brighter, PATCH, 0.7)


def test_appearance_flat_patches_of_same_intensity_match():
    flat = scene(np.full((40, 40), 180, np.uint8))
    assert remembered(flat).still_there('wallet', flat, PATCH, 0.7)


def test_appearance_flat_patches_of_different_intensity_do_not_match():
    flat = scene(np.full((40, 40), 180, np.uint8))
    assert not remembered(flat).still_there('wallet', table_img(), PATCH, 0.7)


def test_appearance_flat_memory_vs_textured_current_does_not_match():
    flat = scene(np.full((40, 40), 180, np.uint8))
    assert not remembered(flat).still_there('wallet', scene(texture(1)), PATCH, 0.7)


def test_appearance_resizes_current_patch_to_stored_shape():
    big_box = (30, 30, 90, 90)
    big = cv2.resize(texture(1), (60, 60), interpolation=cv2.INTER_NEAREST)
    assert remembered().still_there('wallet', scene(big, big_box), big_box, 0.7)


def test_appearance_accepts_bgr_images():
    bgr = np.repeat(scene(texture(1))[:, :, None], 3, axis=2)
    mem = AppearanceMemory()
    mem.remember('wallet', bgr, PATCH)
    assert mem.still_there('wallet', bgr, PATCH, 0.7)


def test_appearance_stores_a_copy_of_the_patch():
    img = scene(texture(1))
    mem = remembered(img)
    img[:] = TABLE
    assert mem.still_there('wallet', scene(texture(1)), PATCH, 0.7)


def test_appearance_remember_overwrites_previous_patch():
    mem = remembered()
    mem.remember('wallet', scene(texture(2)), PATCH)
    assert mem.still_there('wallet', scene(texture(2)), PATCH, 0.7)


def test_appearance_box_outside_image_is_not_there():
    assert not remembered().still_there('wallet', scene(texture(1)), (300, 300, 340, 340), 0.7)
