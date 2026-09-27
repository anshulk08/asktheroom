"""room.point_looks: a WHERE the room look answers with a place in a zone ("the glasses are on the couch")
gets a laser aim when Moondream boxes the thing in that zone's close-up, the box lies inside the zone and is
small, and Grok says yes to a red-box close-up. Any other outcome leaves the spoken answer alone."""
import json

import pytest

from core.types import Answer
from tests.test_room_look import RoomFrames, T0, ask, zoned_qa, log  # noqa: F401  (log is a fixture)
from tests.test_visual import CFG

# ZONES in tests.test_room_look: the couch is (600, 700)-(1300, 1080) of a 1920x1080 frame


class FakeGrounder:
    """core.grounding's API: detect(img, phrase, deadline) -> [(x0, y0, x1, y1, conf)] in img's px."""
    def __init__(self, boxes=(), fail=False):
        self.boxes, self.fail, self.calls = list(boxes), fail, []

    def detect(self, img, phrase, deadline=None):
        self.calls.append((img.shape[:2], phrase))
        if self.fail:
            raise TimeoutError("moondream timed out")
        return list(self.boxes)


def look_qa(log, tmp_path, answer="Your mug is on the couch.", yes=True, boxes=((300, 500, 360, 550, 0.8),),
            on=True, fail=False, found=True):
    q, prov = zoned_qa(log, tmp_path, reply=json.dumps({"seen": "", "answer": answer, "confidence": 0.9}))
    q.cfg = {**q.cfg, "room": {**(q.cfg.get("room") or {}), "point_looks": on}}
    prov.replies = [json.dumps({"seen": "", "answer": answer, "confidence": 0.9, "found": found}),
                    json.dumps({"yes": yes, "confidence": 0.9})]
    g = FakeGrounder(boxes, fail)
    q._look_grounder = g if on else None
    return q, prov, g


def aim_box(a):
    assert a.action and a.action.startswith("room:"), a.action
    return [float(v) for v in a.action.split(":", 1)[1].split(",")]


def test_a_thing_placed_in_a_zone_gets_a_checked_aim_in_full_frame_px(log, tmp_path):
    q, prov, g = look_qa(log, tmp_path)
    a = q.look_room("where is my mug?", "mug")
    assert a.text == "Your mug is on the couch." and a.point_at is None       # never the table cm path
    # the couch close-up is cut at (495, 263)-(1405, 1080): Moondream's box is in its px
    [(shape, phrase)] = g.calls
    assert phrase == "mug" and shape[0] > 300
    u, v, x1, y1, x2, y2 = aim_box(a)
    assert (x1, y1, x2, y2) == (795, 763, 855, 813) and (u, v) == (825, 788)
    assert prov.calls[-1].system.startswith("You look at one object in a red box")    # Grok checked it
    assert a.evidence[0]["box_px"] == [795, 763, 855, 813]                             # the receipt is boxed


@pytest.mark.parametrize("answer", ["I don't see a mug.", "I can't tell from here.", "I see a mug somewhere.",
                                    # WS8's review: negatives that name a zone (normalize: "isn't" -> "isnt")
                                    "Your mug isn't on the couch.", "Nothing on the couch looks like a mug.",
                                    "I can't see it on the couch.", "There's no mug on the couch."])
def test_no_aim_without_a_placed_answer(log, tmp_path, answer):
    q, prov, g = look_qa(log, tmp_path, answer=answer)      # even if Grok's found said true
    a = q.look_room("where is my mug?", "mug")
    assert a.action is None and g.calls == []


def test_a_box_outside_the_zone_or_too_big_is_not_aimed_at(log, tmp_path):
    q, _, _ = look_qa(log, tmp_path, boxes=[(5, 5, 60, 40, 0.8)])            # up in the wall, outside the couch
    assert q.look_room("where is my mug?", "mug").action is None
    q, _, _ = look_qa(log, tmp_path, boxes=[(10, 10, 900, 700, 0.8)])        # a third of the frame: furniture
    assert q.look_room("where is my mug?", "mug").action is None


def test_grok_saying_no_stops_the_aim(log, tmp_path):
    q, _, _ = look_qa(log, tmp_path, yes=False)
    a = q.look_room("where is my mug?", "mug")
    assert a == Answer("Your mug is on the couch.") and a.action is None


def test_a_moondream_failure_or_miss_leaves_the_spoken_answer(log, tmp_path):
    for kw in ({"fail": True}, {"boxes": []}):
        q, _, _ = look_qa(log, tmp_path, **kw)
        a = q.look_room("where is my mug?", "mug")
        assert a.text == "Your mug is on the couch." and a.action is None


def test_off_by_default(log, tmp_path):
    q, _, _ = look_qa(log, tmp_path, on=False)
    assert q._grounder_for_looks() is None
    assert q.look_room("where is my mug?", "mug").action is None
    assert (CFG.get("room") or {}).get("point_looks") in (None, False)


def test_a_where_for_a_known_never_placed_object_passes_its_name(log, tmp_path):
    from core.fakeworld import FakeWorld
    from core.types import Entity
    q, prov, g = look_qa(log, tmp_path, answer="Your glasses are on the couch.")
    q.world = FakeWorld([Entity("glasses", "target")], log)
    a = ask(q, "where are my glasses?")
    assert g.calls and g.calls[0][1] == "glasses" and a.action.startswith("room:")


def test_a_thing_resting_above_the_drawn_edge_of_its_zone_still_counts():
    """Rig 27 Sep 04:42: glasses on the couch seat sat 40 px above the couch polygon's top edge."""
    from voice.visual import on_zone
    couch = [(973.3, 1053.3), (1320.0, 1053.3), (1320.0, 1440.0), (906.7, 1440.0)]
    assert on_zone(couch, (1097, 1008))                   # on the seat, above the drawn edge
    assert not on_zone(couch, (850, 1008))                # beside the couch
    assert not on_zone(couch, (1097, 500))                # far above it (the wall)


def test_found_false_or_missing_never_aims(log, tmp_path):
    for found in (False, None):
        q, _, g = look_qa(log, tmp_path, found=found)
        assert q.look_room("where is my mug?", "mug").action is None and g.calls == []
