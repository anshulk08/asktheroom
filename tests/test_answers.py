import re
import time

import pytest

from core.config import load_config
from core.events import EventLog
from core.fakeworld import FakeWorld, demo_world
from core.types import Entity, Event, Status
from voice.answers import Answer, ago, answer, area, clock
from voice.intents import Intent, parse

CFG = load_config()


def ask(w, kind, obj=None, **kw):
    return answer(Intent(kind, obj, f"{kind} {obj}"), w, w.events, CFG, **kw)


def sentences(text):
    return [s for s in re.split(r"(?<=[.?!])\s+", text.strip()) if s]


def spoken_ok(a: Answer):
    assert a.text and a.text[0].isupper() and a.text.endswith((".", "?"))
    assert not re.search(r"[*#`_\[\]]", a.text), a.text  # no markdown, no raw 'pill_bottle'
    assert "None" not in a.text


@pytest.fixture
def w():
    return demo_world()


# ---------- helpers ----------

def test_ago():
    now = 10_000.0
    assert ago(now - 3, now) == "just now"
    assert ago(now - 40, now) == "40 seconds ago"
    assert ago(now - 70, now) == "a minute ago"
    assert ago(now - 120, now) == "2 minutes ago"
    assert ago(now - 3600, now) == "an hour ago"
    assert ago(now - 3 * 3600, now) == "3 hours ago"
    assert ago(None, now) == "a while ago"


def test_clock_format():
    assert re.fullmatch(r"\d{1,2}:\d\d (AM|PM)", clock(time.time()))


def test_area():
    assert area((5, 5), CFG) == "at the far left"
    assert area((45, 30), CFG) == "in the middle"
    assert area((85, 55), CFG) == "on your right, near you"
    assert area((5, 30), CFG) == "on your left"
    assert area((45, 55), CFG) == "on the side nearest you"
    assert area((45, 5), CFG) == "on the far side"
    assert area(None, CFG) == "somewhere on the table"


def seat(front):
    """CFG with a 100 x 60 cm camera table and the user at the camera-frame side front."""
    return {**CFG, "table": {**CFG["table"], "size_cm": [100, 60]}, "table_area": {"polygon_cm": []},
            "viewer": {"front": front}}


# Camera points: top left, top right, bottom left, bottom right, left middle, top middle, centre.
CAM = [(5, 5), (90, 5), (5, 55), (95, 55), (5, 30), (50, 5), (50, 30)]
AREAS = {
    "bottom": ["at the far left", "at the far right", "on your left, near you", "on your right, near you",
               "on your left", "on the far side", "in the middle"],
    "top": ["on your right, near you", "on your left, near you", "at the far right", "at the far left",
            "on your right", "on the side nearest you", "in the middle"],
    "right": ["at the far right", "on your right, near you", "at the far left", "on your left, near you",
              "on the far side", "on your right", "in the middle"],
    "left": ["on your left, near you", "at the far left", "on your right, near you", "at the far right",
             "on the side nearest you", "on your left", "in the middle"],
}
# Camera edge -> what follows 'carried off', per seat.
OFF = {
    "bottom": {"left": "the table on your left", "right": "the table on your right",
               "top": "the far side of the table", "bottom": "the side of the table nearest you"},
    "top": {"left": "the table on your right", "right": "the table on your left",
            "top": "the side of the table nearest you", "bottom": "the far side of the table"},
    "right": {"left": "the far side of the table", "right": "the side of the table nearest you",
              "top": "the table on your right", "bottom": "the table on your left"},
    "left": {"left": "the side of the table nearest you", "right": "the far side of the table",
             "top": "the table on your left", "bottom": "the table on your right"},
}


@pytest.mark.parametrize("front", AREAS)
def test_area_is_said_from_the_users_seat(front):
    assert [area(p, seat(front)) for p in CAM] == AREAS[front]


@pytest.mark.parametrize("front,edge", [(f, e) for f in OFF for e in OFF[f]])
def test_gone_says_the_side_from_the_seat_and_sweeps_the_camera_edge(w, front, edge):
    w.set("phone", edge=edge)
    a = answer(Intent("WHERE", "phone", "where is my phone"), w, w.events, seat(front))
    assert a.text == f"Your phone was carried off {OFF[front][edge]} 2 minutes ago."
    assert a.action == f"sweep:{edge}"


@pytest.mark.parametrize("front", OFF)
def test_history_and_lost_track_use_the_seat(front):
    now = time.time()
    fw = FakeWorld([Entity("phone", "target", Status.GONE, pos_cm=(3.0, 30.0), edge="top", last_seen=now - 60),
                    Entity("glasses", "target", Status.UNKNOWN, pos_cm=(90.0, 5.0), last_seen=now - 600,
                           confidence=0.2)])
    fw.events.add(Event(t=0.0, wall=now - 60, obj="phone", type="EXITED_VIEW", edge="top"))
    cfg = seat(front)
    h = answer(Intent("HISTORY", "phone", "what happened to my phone"), fw, fw.events, cfg, now=now).text
    assert f"carried off {OFF[front]['top']}" in h and "the top side" not in h
    at = AREAS[front][1]
    lost = answer(Intent("WHERE", "glasses", "where are my glasses"), fw, fw.events, cfg, now=now).text
    assert lost == f"I lost track of your glasses. I last saw them {at}{',' if ',' in at else ''} 10 minutes ago."


# ---------- WHERE, one per status (demo_world) ----------

def test_visible(w):
    a = ask(w, "WHERE", "wallet")
    spoken_ok(a)
    assert a.text.startswith("Your wallet is on the table")
    assert (a.point_at, a.action) == ("wallet", "point")


def test_visible_near_other_object(w):
    w.set("wallet", pos_cm=(65.0, 35.0))
    assert ask(w, "WHERE", "wallet").text == "Your wallet is on the table, near the box."


def test_inside_plural(w):
    a = ask(w, "WHERE", "keys")
    spoken_ok(a)
    assert a.text.startswith("Your keys are inside the box.")
    assert "They were put there 7 minutes ago." in a.text
    assert (a.point_at, a.action) == ("keys", "point")


def test_inside_chain_mentions_parent_location(w):
    w.set("box", status=Status.UNDER, parent="notebook")
    a = ask(w, "WHERE", "keys")
    assert "inside the box, which is under the notebook" in a.text
    assert (a.point_at, a.action) == ("keys", "point")


def test_under(w):
    a = ask(w, "WHERE", "pill_bottle")
    spoken_ok(a)
    assert a.text == "Your pill bottle is under the notebook."
    assert (a.point_at, a.action) == ("pill_bottle", "point")


def test_under_unknown_parent(w):
    w.set("pill_bottle", parent="unknown")
    a = ask(w, "WHERE", "pill_bottle")
    assert a.text == "Your pill bottle is under something near where it was."
    assert a.action == "point"


def test_held(w):
    a = ask(w, "WHERE", "remote")
    spoken_ok(a)
    assert a.text == "Someone is holding your remote right now."
    assert (a.point_at, a.action) == ("remote", "point")


def test_held_without_position():
    fw = FakeWorld([Entity("remote", "target", Status.HELD, parent="hand:1", confidence=0.9)])
    a = ask(fw, "WHERE", "remote")
    assert "holding your remote" in a.text
    assert a.action is None and a.point_at is None


def test_gone(w):
    a = ask(w, "WHERE", "phone")
    spoken_ok(a)
    assert a.text == "Your phone was carried off the table on your left 2 minutes ago."
    assert (a.point_at, a.action) == ("phone", "sweep:left")


def test_unknown_lost_track(w):
    a = ask(w, "WHERE", "glasses")
    spoken_ok(a)
    assert a.text == "I lost track of your glasses. I last saw them on your right, near you, 10 minutes ago."
    assert (a.point_at, a.action) == ("glasses", "circle")


def test_prop_uses_the(w):
    assert ask(w, "WHERE", "box").text.startswith("The box is on the table")


# ---------- confidence bands and ambiguity ----------

@pytest.mark.parametrize("conf,expect", [(0.9, "plain"), (0.7, "plain"), (0.6, "probably"),
                                         (0.5, "probably"), (0.49, "lost"), (0.1, "lost")])
def test_confidence_bands(w, conf, expect):
    w.set("keys", confidence=conf)
    a = ask(w, "WHERE", "keys")
    if expect == "plain":
        assert a.text.startswith("Your keys are inside the box.")
    elif expect == "probably":
        assert a.text.startswith("Your keys are probably inside the box.")
        assert a.action == "point"
    else:
        assert a.text.startswith("I lost track of your keys.") and a.action == "circle"


@pytest.mark.parametrize("name,status,parent", [("wallet", Status.VISIBLE, None),
                                                 ("pill_bottle", Status.UNDER, "notebook"),
                                                 ("remote", Status.HELD, "hand:2"),
                                                 ("phone", Status.GONE, None)])
def test_probably_for_every_status(w, name, status, parent):
    w.set(name, status=status, parent=parent, confidence=0.6)
    assert " probably " in ask(w, "WHERE", name).text


def test_ambiguity_mentions_both(w):
    w.set("keys", candidates=["notebook"], confidence=0.8)
    a = ask(w, "WHERE", "keys")
    spoken_ok(a)
    assert a.text == "Your keys are probably inside the box, or under the notebook."
    assert (a.point_at, a.action) == ("keys", "point")


def test_ambiguity_candidates_including_parent(w):
    w.set("pill_bottle", candidates=["notebook", "box"])
    a = ask(w, "WHERE", "pill_bottle")
    assert a.text == "Your pill bottle is probably under the notebook, or inside the box."


# ---------- HISTORY / HANDLED / CHANGES from demo_world events ----------

def test_history_oldest_first_with_parent_move(w):
    a = ask(w, "HISTORY", "keys")
    spoken_ok(a)
    assert a.text == ("Your keys were picked up 7 minutes ago, then put inside the box. "
                      "The box was moved 2 minutes ago.")
    assert len(sentences(a.text)) <= 2


def test_history_three_events(w):
    a = ask(w, "HISTORY", "pill_bottle")
    assert a.text.index("picked up") < a.text.index("put back down") < a.text.index("covered by the notebook")


def test_history_no_events(w):
    a = ask(w, "HISTORY", "wallet")
    assert a.text == "I haven't seen anything happen to your wallet since I started watching."


def test_handled_pill_bottle(w):
    a = ask(w, "HANDLED", "pill_bottle")
    spoken_ok(a)
    assert re.fullmatch(r"The pill bottle was picked up at \d{1,2}:\d\d [AP]M, 5 minutes ago, "
                        r"and put down at \d{1,2}:\d\d [AP]M\.", a.text), a.text


def test_handled_put_inside_and_plural(w):
    a = ask(w, "HANDLED", "keys")
    assert a.text.startswith("The keys were picked up at ") and "put inside the box at" in a.text


def test_handled_still_held(w):
    assert ask(w, "HANDLED", "remote").text.startswith("The remote was picked up at ")


def test_handled_moved_only(w):
    assert ask(w, "HANDLED", "box").text.startswith("The box was moved at ")


def test_handled_never_touched(w):
    a = ask(w, "HANDLED", "wallet")
    assert a.text == "I haven't seen anyone touch the wallet since I started watching."


def test_changes_summary(w):
    a = ask(w, "CHANGES")
    spoken_ok(a)
    assert len(sentences(a.text)) <= 3
    assert "remote" in a.text and "box" in a.text
    assert a.point_at is None and a.action is None


def test_changes_since_kwarg(w):
    a = ask(w, "CHANGES", since=time.time() - 60)
    assert a.text == "The remote was picked up just now."


def test_changes_nothing():
    fw = FakeWorld([Entity("keys", "target", Status.VISIBLE, pos_cm=(1.0, 1.0))])
    assert ask(fw, "CHANGES").text == "Nothing has changed in the last 10 minutes."


# ---------- pill bottle wording, every path ----------

def _pill_worlds():
    out = []
    for status, parent, extra in [
        (Status.VISIBLE, None, {}), (Status.INSIDE, "box", {}), (Status.UNDER, "notebook", {}),
        (Status.UNDER, "unknown", {}), (Status.HELD, "hand:1", {}), (Status.GONE, None, {"edge": "right"}),
        (Status.UNKNOWN, "unknown", {}), (Status.INSIDE, "box", {"confidence": 0.6}),
        (Status.INSIDE, "box", {"candidates": ["notebook"]}), (Status.VISIBLE, None, {"confidence": 0.2}),
    ]:
        w = demo_world()
        w.set("pill_bottle", status=status, parent=parent, **extra)
        out.append(w)
    return out


def test_pill_bottle_never_took_or_taken():
    for w in _pill_worlds():
        now = time.time()
        for typ, kw in [("TAKEN_OUT", {"parent": "box"}), ("PUT_INSIDE", {"parent": "box"}),
                        ("EXITED_VIEW", {"edge": "right"}), ("PICKED_UP", {}), ("LOST_TRACK", {}),
                        ("UNCOVERED", {}), ("CORRECTED", {}), ("FOUND", {})]:
            w.events.add(Event(t=0.0, wall=now - 1, obj="pill_bottle", type=typ, **kw))
        for kind in ["WHERE", "HISTORY", "HANDLED", "CHANGES"]:
            text = ask(w, kind, "pill_bottle").text.lower()
            assert not re.search(r"\b(took|taken|take|takes|swallow\w*)\b", text), (kind, text)


def test_pill_questions_end_to_end(w):
    for q in ["Did I take my pills?", "Have I taken my medicine?", "Where are my meds?",
              "What happened to my medication?"]:
        text = answer(parse(q, CFG), w, w.events, CFG).text.lower()
        assert "took" not in text and "taken" not in text, (q, text)


# ---------- misc intents ----------

@pytest.mark.parametrize("kind", ["WHERE", "HISTORY", "HANDLED"])
def test_no_object_asks_which(w, kind):
    a = ask(w, kind, None)
    assert a.text.startswith("Which object do you mean?")
    assert "keys, pill bottle, wallet" in a.text and a.point_at is None


def test_reset_recal_other(w):
    assert ask(w, "RESET") == Answer("Okay, resetting the table.")
    assert ask(w, "RECAL") == Answer("Recalibrating now.")
    a = ask(w, "OTHER")
    assert a.text == "I can tell you where things are, what happened to them, or what changed."
    assert a.point_at is None and a.action is None


def test_untracked_object():
    fw = FakeWorld([Entity("keys", "target", Status.VISIBLE, pos_cm=(1.0, 1.0))])
    assert "not tracking" in ask(fw, "WHERE", "wallet").text


def test_cfg_defaults_to_load_config(w):
    a = answer(Intent("WHERE", "keys", "where are my keys"), w, w.events)
    assert a.text.startswith("Your keys are inside the box.")


def test_file_backed_eventlog(tmp_path):
    log = EventLog(str(tmp_path / "e.db"), str(tmp_path / "s"))
    w = demo_world(log)
    assert ask(w, "HISTORY", "keys").text.startswith("Your keys were picked up")


@pytest.mark.parametrize("name", ["keys", "pill_bottle", "wallet", "glasses", "phone", "remote",
                                  "box", "notebook"])
@pytest.mark.parametrize("kind", ["WHERE", "HISTORY", "HANDLED"])
def test_every_object_short_and_spoken(w, name, kind):
    a = ask(w, kind, name)
    spoken_ok(a)
    assert len(sentences(a.text)) <= 3


def test_where_for_an_object_never_seen_says_so_and_does_not_point():
    fw = FakeWorld([Entity("keys", "target", Status.UNKNOWN, confidence=0.0)])
    ans = answer(parse("where are my keys?", CFG), fw, fw.events, CFG)
    assert "haven't seen" in ans.text.lower() and "lost track" not in ans.text.lower()
    assert ans.point_at is None


# ---------- open world ----------

def test_appeared_event_phrase_and_unnamed_things_in_changes():
    from core.types import Event
    fw = FakeWorld([Entity("keys", "target", Status.VISIBLE, pos_cm=(10.0, 10.0))])
    now = time.time()
    fw.events.add(Event(t=1.0, wall=now - 5, obj="thing:4", type="APPEARED", to_cm=(40.0, 30.0)))
    a = answer(Intent("CHANGES", None, "what changed"), fw, fw.events, CFG, now=now)
    spoken_ok(a)
    assert a.text == "The thing I haven't been told about was first seen just now."


def test_changes_in_a_busy_room_name_a_few_and_count_the_rest():
    # the rig's room had hundreds of unnamed things: the spoken answer ran to 21,000 characters
    fw = FakeWorld([Entity("keys", "target", Status.VISIBLE, pos_cm=(10.0, 10.0))])
    now = time.time()
    for i in range(400):
        fw.events.add(Event(t=float(i), wall=now - 300 + i * 0.5, obj=f"thing:{i}", type="APPEARED",
                            to_cm=(40.0, 30.0)))
    fw.events.add(Event(t=500.0, wall=now - 1, obj="keys", type="PICKED_UP"))
    a = answer(Intent("CHANGES", None, "what changed"), fw, fw.events, CFG, now=now)
    spoken_ok(a)
    assert len(a.text) < 300 and a.text.count("haven't been told about") == 1
    assert a.text.startswith("The keys were picked up just now.")
    assert a.text.endswith("399 other things also changed.")


def test_changes_name_other_known_things_before_counting():
    fw = FakeWorld([Entity(n, "target", Status.VISIBLE, pos_cm=(10.0, 10.0))
                    for n in ("keys", "wallet", "remote", "phone", "box")])
    now = time.time()
    for i, n in enumerate(("box", "phone", "remote", "wallet", "keys")):
        fw.events.add(Event(t=float(i), wall=now - 60 + i, obj=n, type="MOVED"))
    for i in range(5):
        fw.events.add(Event(t=10.0 + i, wall=now - 100 + i, obj=f"thing:{i}", type="APPEARED",
                            to_cm=(40.0, 30.0)))
    a = answer(Intent("CHANGES", None, "what changed"), fw, fw.events, CFG, now=now)
    assert a.text.endswith("The remote, phone, box and 5 other things also changed.")


def test_teach_intent_never_mentions_taking_pills(w):
    a = answer(Intent("TEACH", "pills", "this is my pills", name="pills"), w, w.events, CFG)
    assert "taken" not in a.text.lower() and "took" not in a.text.lower()
