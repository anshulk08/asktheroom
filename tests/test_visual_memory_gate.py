"""The archive's change gate (core/visual_memory.py): on the corner rig people's legs at the table's edge and
the world's event churn made nearly every check a 'change' save (2420 of 2424 saves in an hour). settle_checks
keeps a change once the view holds still, change_min_gap_s spaces change saves, events_need_pixels ignores a
world event that changed nothing in view, change_on_table measures change on the tabletop outline only. The
code defaults keep the old policy; config.yaml turns the gate on."""
import numpy as np

from core.types import Event
from tests.test_visual import T0, archive, dets, frame, img, log  # noqa: F401  (log is a fixture)


def run(a, seq):
    """seq: [(t, image, hands, events)] fed one check each; returns the saved rows' (t - T0, reason)."""
    for t, im, hands, evs in seq:
        a.feed(frame(float(t), im), dets(float(t), hands), list(evs))
        a.drain()
    return [(round(r.t - T0), r.reason) for r in a.store.window(T0 - 1, T0 + 1e4)]


def wiggle(t):
    """A leg at the table's edge moving from check to check: a different block every second."""
    return img(block=(40, 40, 40), at=(40 * (t % 5), 600), size=100)


def test_by_default_a_moving_leg_is_saved_every_check(log):
    rows = run(archive(log), [(t, wiggle(t), 0, ()) for t in range(0, 10)])
    assert len([r for r in rows if r[1] == "change"]) == 9


def test_settle_waits_for_the_view_to_hold_still(log):
    a = archive(log, settle_checks=2)
    seq = [(t, wiggle(t), 0, ()) for t in range(0, 10)]                      # moving: nothing kept
    red = img(block=(0, 0, 255))
    seq += [(t, red, 0, ()) for t in range(10, 15)]                           # a mug put down, then still
    assert run(a, seq) == [(0, "first"), (12, "change")]                      # the settled view, once


def test_the_interval_floor_still_keeps_a_busy_view(log):
    a = archive(log, settle_checks=2)
    rows = run(a, [(t, wiggle(t), 0, ()) for t in range(0, 65)])
    assert rows == [(0, "first"), (30, "interval"), (60, "interval")]


def test_changes_are_spaced_by_the_minimum_gap(log):
    a = archive(log, change_min_gap_s=10)
    rows = run(a, [(t, wiggle(t), 0, ()) for t in range(0, 25)])
    assert rows == [(0, "first"), (10, "change"), (20, "change")]


EV = [Event(t=1.0, wall=T0 + 1, obj="thing:9", type="COVERED")]


def test_a_world_event_that_changed_nothing_in_view_is_not_a_change(log):
    same = img()
    assert run(archive(log, events_need_pixels=True), [(0, same, 0, ()), (1, same, 0, EV), (2, same, 0, ())]) \
        == [(0, "first")]


def test_by_default_any_world_event_is_a_change(log):
    same = img()
    assert run(archive(log), [(0, same, 0, ()), (1, same, 0, EV), (2, same, 0, ())]) == [(0, "first"), (1, "change")]


def test_an_event_under_a_hand_with_a_real_change_is_still_kept(log):
    a = archive(log, events_need_pixels=True, settle_checks=1)
    red = img(block=(0, 0, 255))
    ev = [Event(t=1.0, wall=T0 + 1, obj="keys", type="PUT_INSIDE")]
    rows = run(a, [(0, img(), 0, ()), (1, red, 1, ev), (2, red, 0, ()), (3, red, 0, ())])
    assert rows == [(0, "first"), (2, "change")]


class PxTable:
    """A calibrated table whose cm are frame px (so table_area.polygon_cm is an outline in px)."""
    ok = True

    @staticmethod
    def cm_to_px(pts):
        return pts


def outlined(log, **kw):
    from tests.test_visual import CFG
    a = archive(log, **kw)
    a.cfg = {**CFG, "table_area": {"polygon_cm": [[300, 150], [1000, 150], [1000, 500], [300, 500]]}}
    a.table = PxTable()
    return a


def test_with_a_tabletop_outline_a_leg_beside_the_table_is_no_change(log):
    rows = run(outlined(log), [(t, wiggle(t), 0, ()) for t in range(0, 10)])   # the leg is at y 600, off the table
    assert rows == [(0, "first")]


def test_with_a_tabletop_outline_a_mug_on_the_table_is_a_change(log):
    red = img(block=(0, 0, 255), at=(600, 300), size=60)
    rows = run(outlined(log), [(0, img(), 0, ()), (1, red, 0, ())])
    assert rows == [(0, "first"), (1, "change")]


def test_an_outline_off_the_view_or_no_table_counts_the_whole_view(log):
    from core.visual_memory import table_mask
    assert table_mask(None, {}, (1280, 720)) is None
    tiny = {"table_area": {"polygon_cm": [[0, 0], [5, 0], [5, 5]]}}
    assert table_mask(PxTable(), tiny, (1280, 720)) is None
    m = table_mask(PxTable(), {"table_area": {"polygon_cm": [[0, 0], [640, 0], [640, 720], [0, 720]]}}, (1280, 720))
    assert m.shape == (72, 128) and abs(m.mean() - 0.5) < 0.02


def test_config_yaml_turns_the_gate_on():
    from core.visual_memory import VisualConfig
    from tests.test_visual import CFG
    c = VisualConfig.from_dict(CFG.get("visual_memory"))
    assert c.events_need_pixels and c.change_min_gap_s == 5 and c.change_on_table and c.settle_checks == 0
