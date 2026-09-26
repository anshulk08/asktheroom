"""core/hands.py: stable hand ids across frames, track expiry, edge of exit."""
import pytest

from core.hands import HandTracker
from core.types import Detection


def hand(x, y, w=120, h=120, conf=0.9):
    """A detector hand box centred at pixel (x, y); cm fields are just px / 10."""
    b = (int(x - w / 2), int(y - h / 2), int(x + w / 2), int(y + h / 2))
    return Detection(cls="hand", conf=conf, box_px=b, center_cm=(x / 10, y / 10),
                     box_cm=tuple(v / 10 for v in b))


@pytest.fixture
def tr():
    return HandTracker(frame_size=(1280, 720), lost_s=0.5, iou_min=0.3, edge_frac=0.05)


def test_a_hand_crossing_the_table_keeps_one_id(tr):
    ids = set()
    for i in range(40):                                   # 30 px per frame: boxes overlap frame to frame
        [h] = tr.update([hand(100 + 30 * i, 360)], t=i / 10)
        ids.add(h.cls)
    assert ids == {"hand:1"}


def test_a_fast_hand_with_no_overlap_is_matched_by_nearest_centre(tr):
    tr.update([hand(200, 360)], t=0.0)
    [h] = tr.update([hand(340, 380)], t=0.1)              # 140 px jump: IoU 0, still the nearest
    assert h.cls == "hand:1"


def test_two_hands_get_two_ids_and_keep_them_when_listed_in_either_order(tr):
    a, b = tr.update([hand(200, 300), hand(900, 400)], t=0.0)
    assert {a.cls, b.cls} == {"hand:1", "hand:2"}
    out = tr.update([hand(910, 405), hand(205, 302)], t=0.1)
    assert [h.cls for h in out] == ["hand:2", "hand:1"]


def test_track_is_dropped_after_lost_s_and_a_new_hand_gets_a_new_id(tr):
    tr.update([hand(640, 360)], t=0.0)
    tr.update([], t=0.3)
    [h] = tr.update([hand(645, 360)], t=0.4)
    assert h.cls == "hand:1"                              # back within 0.5 s: same track
    tr.update([], t=0.5)
    [h] = tr.update([hand(640, 360)], t=1.0)
    assert h.cls == "hand:2"                              # gone 0.6 s: new track


def test_last_returns_the_hands_last_detection(tr):
    tr.update([hand(640, 360)], t=0.0)
    tr.update([hand(700, 360)], t=0.1)
    tr.update([], t=0.2)
    assert tr.last("hand:1").box_px == hand(700, 360).box_px
    assert tr.last("hand:9") is None


@pytest.mark.parametrize("x,y,edge", [(40, 360, "left"), (1250, 360, "right"), (640, 30, "top"),
                                      (640, 700, "bottom")])
def test_exited_edge_names_the_border_the_hand_was_last_seen_at(tr, x, y, edge):
    tr.update([hand(x, y, 100, 100)], t=0.0)
    assert tr.exited_edge("hand:1") == edge


def test_exited_edge_is_none_mid_table(tr):
    tr.update([hand(640, 360)], t=0.0)
    assert tr.exited_edge("hand:1") is None


def test_ids_are_not_reused_after_reset(tr):
    tr.update([hand(640, 360)], t=0.0)
    tr.reset()
    [h] = tr.update([hand(640, 360)], t=0.1)
    assert h.cls == "hand:2"
