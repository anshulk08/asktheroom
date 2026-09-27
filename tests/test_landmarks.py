"""WHERE on the table by landmarks (voice/landmarks.py): "where are the batteries?" answered "I think they are
on the table" because only taught names counted as landmarks and room mode names almost nothing. Now props,
taught things and confident Grok guesses count, big easy-to-see ones first, said from the user's seat
(viewer.front); with none close, Grok describes the spot within a short deadline, else the table area."""
import json
import time

import numpy as np
import pytest

from core.config import load_config
from core.events import EventLog
from core.fakeworld import FakeWorld
from core.types import Entity, Frame, Intent, Status
from voice.answers import answer

CFG = load_config()


class GuessWorld(FakeWorld):
    """FakeWorld plus the open-world names answers.py reads: taught labels and Grok's guesses."""
    def __init__(self, entities, events, labels=None, guesses=None):
        super().__init__(entities, events)
        self.labels, self.guesses = dict(labels or {}), dict(guesses or {})

    def thing_labels(self):
        return {n: self.labels.get(n) for n in self.entities if n.startswith("thing:")}

    def state_json(self):
        st = super().state_json()
        for e in st["entities"]:
            if e["name"] in self.guesses:
                e["guess"] = {"name": self.guesses[e["name"]], "confidence": 0.9}
            if e["name"] in self.labels:
                e["label"] = self.labels[e["name"]]
        return st

    def find_guess(self, words):
        return [(n, 1.0) for n, g in self.guesses.items() if g == words]


def thing(name, pos, size=(4.0, 4.0)):
    x, y = pos
    return Entity(name, "target", Status.VISIBLE, pos_cm=pos, last_seen=time.time(),
                  box_cm=(x - size[0] / 2, y - size[1] / 2, x + size[0] / 2, y + size[1] / 2))


@pytest.fixture
def log(tmp_path):
    lg = EventLog(":memory:", str(tmp_path / "snaps"))
    yield lg
    lg.close()


def cfg_front(front):
    return {**CFG, "viewer": {"front": front}, "table": {**(CFG.get("table") or {}), "size_cm": [90, 60]},
            "table_area": {"polygon_cm": []}}


def where(world, said="batteries", cfg=CFG, **kw):
    return answer(Intent("WHERE", None, f"where are my {said}", name=said), world, world.events, cfg, **kw)


def batteries_world(log, others, guesses=None, labels=None, pos=(40.0, 30.0)):
    ents = [thing("thing:1", pos, (5, 3))] + others
    return GuessWorld(ents, log, labels=labels, guesses={"thing:1": "batteries", **(guesses or {})})


# -- direction words from the user's seat

@pytest.mark.parametrize("front,expect", [("bottom", "just left of the laptop"), ("top", "just right of the laptop"),
                                          ("right", "just behind the laptop"),
                                          ("left", "just in front of the laptop, nearer you")])
def test_the_direction_is_said_from_the_users_seat(log, front, expect):
    # the batteries are 3 cm off the laptop's camera-left edge (smaller x)
    w = batteries_world(log, [thing("thing:2", (58.0, 30.0), (30, 20))], guesses={"thing:2": "laptop"})
    a = where(w, cfg=cfg_front(front))
    assert a.text == f"Your batteries, I think, are on the table, {expect}.", a.text


@pytest.mark.parametrize("front,expect", [("bottom", "in front of the notebook, nearer you"),
                                          ("top", "behind the notebook"),
                                          ("right", "left of the notebook"), ("left", "right of the notebook")])
def test_nearer_the_camera_is_turned_to_the_seat_too(log, front, expect):
    # 20 cm on the camera's side of the notebook (larger y)
    w = batteries_world(log, [Entity("notebook", "cover", Status.VISIBLE, pos_cm=(40.0, 10.0),
                                     box_cm=(30.0, 5.0, 50.0, 15.0), last_seen=time.time())])
    assert where(w, cfg=cfg_front(front)).text.endswith(f"on the table, {expect}.")


# -- which landmark

def test_a_big_guessed_thing_beats_a_small_named_one_nearby(log):
    # the pill bottle is 5.5 cm to one side, the laptop 14 cm further away: the laptop is the one to look for
    w = batteries_world(log, [thing("pill_bottle", (48.0, 30.0), (5, 5)), thing("thing:2", (40.0, 5.0), (32, 22))],
                        guesses={"thing:2": "laptop"})
    assert where(w).text.endswith("on the table, in front of the laptop, nearer you.")


def test_between_needs_opposite_sides(log):
    w = batteries_world(log, [thing("pill_bottle", (40.0, 38.0), (5, 5)), thing("thing:2", (40.0, 5.0), (32, 22))],
                        guesses={"thing:2": "laptop"})
    assert where(w).text.endswith("between the laptop and the pill bottle.")


def test_a_close_small_thing_is_used_when_nothing_big_is_near(log):
    w = batteries_world(log, [thing("pill_bottle", (47.0, 30.0), (5, 5))])
    assert where(w).text.endswith("just left of the pill bottle.")


def test_a_thing_with_no_name_and_no_guess_is_no_landmark(log):
    w = batteries_world(log, [thing("thing:2", (50.0, 30.0), (30, 20))])
    assert "thing" not in where(w).text.split("are on the table")[1]


def test_between_two_landmarks(log):
    w = batteries_world(log, [thing("thing:2", (20.0, 30.0), (20, 15)), thing("thing:3", (60.0, 30.0), (8, 8))],
                        guesses={"thing:2": "laptop", "thing:3": "water bottle"})
    assert where(w).text.endswith("between the laptop and the water bottle.")


def test_on_a_flat_landmark(log):
    w = batteries_world(log, [thing("notebook", (41.0, 31.0), (20, 14))])
    assert where(w).text.endswith("on the notebook.")


def test_too_far_from_everything_the_table_area_is_said(log):
    w = batteries_world(log, [thing("thing:2", (88.0, 58.0), (10, 10))], guesses={"thing:2": "laptop"},
                        pos=(5.0, 3.0))
    assert where(w).text == "Your batteries, I think, are on the table, at the far left."


def test_a_taught_label_is_a_landmark_by_its_name(log):
    w = batteries_world(log, [thing("thing:2", (52.0, 30.0), (10, 10))], labels={"thing:2": "charger"})
    assert where(w).text.endswith("just left of the charger.")


# -- Grok's description when no landmark says it

def test_with_no_landmark_grok_describes_the_spot(log):
    calls = []
    w = batteries_world(log, [], pos=(45.0, 30.0))

    def describe(obj, place):
        calls.append((obj, place))
        return "next to the TV remote"
    assert where(w, describe=describe).text == "Your batteries, I think, are on the table, next to the TV remote."
    assert calls == [("thing:1", None)]


def test_a_landmark_needs_no_grok_call(log):
    w = batteries_world(log, [thing("pill_bottle", (47.0, 30.0), (5, 5))])
    assert "pill bottle" in where(w, describe=lambda o, p: pytest.fail("called")).text


def test_a_late_or_failed_description_falls_back_to_the_area(log):
    w = batteries_world(log, [], pos=(45.0, 30.0))
    assert where(w, describe=lambda o, p: None).text == "Your batteries, I think, are on the table, in the middle."

    def boom(o, p):
        raise RuntimeError("down")
    assert where(w, describe=boom).text.endswith("in the middle.")


def test_a_room_place_gets_the_description(log):
    from core.room_types import Place
    w = GuessWorld([Entity("keys", "target", Status.VISIBLE, zone="couch", last_seen=time.time())], log)
    w.set_place("keys", Place(kind="room", zone="couch", say="the couch", status=Status.VISIBLE, chain=["keys"],
                              via="keys", box_px=(100, 100, 140, 130), observed_directly=True, fresh=True,
                              arrived_wall=time.time() - 30, arrival_observed=True))
    a = where(w, "keys", describe=lambda o, p: "by the armrest nearest the TV")
    assert a.text.startswith("Your keys are on the couch, by the armrest nearest the TV.")


# -- VisualQA.describe_where (the Grok call itself)

class TableFrames:
    def __init__(self):
        self.img = np.full((720, 1280, 3), 120, np.uint8)

    def latest(self):
        return Frame(t=0.0, wall=time.time(), img=self.img, idx=1)

    def latest_full(self):
        return Frame(t=0.0, wall=time.time(), img=np.full((1440, 2560, 3), 90, np.uint8), idx=1)


def describer(log, reply, online=True, delay=0.0):
    from core.narration import FakeProvider
    from server.sim import SimTable
    from tests.test_visual import vcfg
    from voice.visual import VisualQA
    w = batteries_world(log, [])

    def slow(job):
        time.sleep(delay)
        return reply
    prov = FakeProvider(slow if delay else reply)
    q = VisualQA(CFG, w, log, TableFrames(), SimTable(CFG), provider=prov, online=lambda: online, c=vcfg())
    return q, prov


def test_describe_where_boxes_the_thing_and_returns_a_short_phrase(log):
    q, prov = describer(log, json.dumps({"where": "It's next to the TV remote.", "confidence": 0.8}))
    assert q.describe_where("thing:1") == "next to the TV remote"
    [call] = prov.calls
    assert [p[0] for p in call.parts].count("image") == 2 and "left" in call.system.lower()


def test_describe_where_drops_camera_sided_words_and_unsure_replies(log):
    q, _ = describer(log, json.dumps({"where": "to the left of the laptop", "confidence": 0.9}))
    assert q.describe_where("thing:1") is None
    q, _ = describer(log, json.dumps({"where": "near the mug", "confidence": 0.2}))
    assert q.describe_where("thing:1") is None


def test_describe_where_is_skipped_offline_and_gives_up_at_the_deadline(log):
    q, prov = describer(log, json.dumps({"where": "near the mug", "confidence": 0.9}), online=False)
    assert q.describe_where("thing:1") is None and prov.calls == []
    import voice.visual as V
    q, _ = describer(log, json.dumps({"where": "near the mug", "confidence": 0.9}), delay=V.DESCRIBE_DEADLINE_S + 0.5)
    t0 = time.perf_counter()
    assert q.describe_where("thing:1") is None
    assert time.perf_counter() - t0 < V.DESCRIBE_DEADLINE_S + 0.3


def test_describe_where_for_a_room_place_uses_the_whole_view(log):
    from core.room_types import Place
    q, prov = describer(log, json.dumps({"where": "on the couch, by the armrest nearest the TV", "confidence": 0.9}))
    q.room_zones = [("couch", "the couch")]
    place = Place(kind="room", zone="couch", say="the couch", status=Status.VISIBLE, chain=["keys"], via="keys",
                  box_px=(1000, 1100, 1060, 1150))
    assert q.describe_where("keys", place) == "by the armrest nearest the TV"
    assert "on the couch" in prov.calls[0].parts[0][1]


def test_the_pipeline_passes_the_describer_only_online(log):
    from voice.pipeline import make_ask

    class Net:
        online = True

    class Visual:
        def __init__(self):
            self.calls = 0

        def route(self, intent, text, online):
            return None

        def describe_where(self, obj, place=None):
            self.calls += 1
            return "next to the TV remote"
    w = batteries_world(log, [], pos=(45.0, 30.0))
    v = Visual()
    ask = make_ask(CFG, w, log, net=Net(), visual=v, interpret=lambda t: Intent("WHERE", None, t, name="batteries"))
    assert ask("where are my batteries").text.endswith("next to the TV remote.") and v.calls == 1
    Net.online = False
    assert ask("where are my batteries").text.endswith("in the middle.") and v.calls == 1


@pytest.mark.parametrize("said,out", [("on the wooden table", None), ("on the table", None),
                                      ("on the wooden table, next to the mug", "next to the mug"),
                                      ("at the brown coffee table beside the notebook", "beside the notebook"),
                                      ("middle of the wooden table", "in the middle"),        # rig 05:50
                                      ("near the edge of the table, by the mug", "at the edge, by the mug")])
def test_describe_where_drops_the_place_said_again(log, said, out):
    """Rig 27 Sep 05:44: 'Your pill bottle, I think, is on the table, on the wooden table.'"""
    q, _ = describer(log, json.dumps({"where": said, "confidence": 0.9}))
    assert q.describe_where("thing:1") == out
