"""Room answers (spec 0009 section 4, plan Task D): world.place() -> spoken room templates."""
import re
import time

import pytest

from core.config import load_config
from core.fakeworld import FakeWorld, demo_world
from core.room_types import Conflict, Place
from core.types import Answer, Entity, Status
from voice.answers import answer, clock
from voice.intents import Intent

CFG = load_config()
NOW = time.time()


def ask(w, obj, now=NOW):
    return answer(Intent("WHERE", obj, f"where is {obj}"), w, w.events, CFG, now=now)


def spoken_ok(a: Answer):
    assert a.text and a.text[0].isupper() and a.text.endswith((".", "?"))
    assert not re.search(r"[*#`_\[\]]", a.text), a.text
    assert "None" not in a.text
    assert "you put" not in a.text.lower()
    assert not re.search(r"\b(took|taken)\b", a.text.lower())


def room_place(obj, say="the bookshelf", via=None, fresh=True, absent=False, observed=True,
               arrived=NOW - 180, last_seen=NOW - 30, conflicts=None):
    via = via or obj
    chain = [obj] if via == obj else [obj, via]
    return Place(kind="room", zone=say.split()[-1], say=say,
                 status=Status.UNKNOWN if absent else Status.VISIBLE, chain=chain, via=via,
                 box_px=(100.0, 100.0, 140.0, 130.0), observed_directly=via == obj, fresh=fresh,
                 absent=absent, arrived_wall=arrived, last_seen_wall=last_seen, arrival_observed=observed,
                 conflicts=list(conflicts or []))


def conflict(obj, say="the bookshelf"):
    return Conflict(entity=obj, zone=say.split()[-1], say=say, track="r7", box_px=(1.0, 2.0, 3.0, 4.0),
                    seen_t=10.0, seen_wall=NOW - 5)


@pytest.fixture
def w():
    return demo_world()


def room(w, obj, **kw):
    w.set_place(obj, room_place(obj, **kw))
    a = ask(w, obj)
    spoken_ok(a)
    assert (a.point_at, a.action) == (None, None)   # room answers carry no laser action
    return a


class NoPlace:
    """A world with the read API but no place() (today's World before room memory)."""

    def __init__(self, w):
        self._w = w

    def __getattr__(self, k):
        if k == "place":
            raise AttributeError(k)
        return getattr(self._w, k)


# ---------- templates, one per row ----------

def test_direct_fresh_arrival_observed_plural(w):
    assert room(w, "keys").text == f"Your keys are on the bookshelf. They appeared there at {clock(NOW - 180)}."


def test_direct_fresh_arrival_observed_singular(w):
    assert room(w, "wallet").text == f"Your wallet is on the bookshelf. It appeared there at {clock(NOW - 180)}."


def test_direct_fresh_arrival_not_observed(w):
    t = NOW - 600
    assert room(w, "keys", observed=False, arrived=t).text == \
        f"Your keys are on the bookshelf. I've seen them there since {clock(t)}."
    assert room(w, "wallet", observed=False, arrived=t).text == \
        f"Your wallet is on the bookshelf. I've seen it there since {clock(t)}."


def test_direct_not_fresh(w):
    t = NOW - 900
    assert room(w, "keys", fresh=False, last_seen=t).text == f"I last saw your keys on the bookshelf at {clock(t)}."


def test_direct_absent(w):
    t = NOW - 900
    assert room(w, "keys", fresh=False, absent=True, last_seen=t).text == \
        f"I last saw your keys on the bookshelf at {clock(t)}. I can't see them there now."
    assert room(w, "wallet", fresh=False, absent=True, last_seen=t).text == \
        f"I last saw your wallet on the bookshelf at {clock(t)}. I can't see it there now."


def test_via_container_fresh(w):
    assert room(w, "keys", via="box").text == "Your keys are in the box. The box is on the bookshelf."
    assert room(w, "wallet", via="box").text == "Your wallet is in the box. The box is on the bookshelf."


def test_via_container_not_fresh(w):
    t = NOW - 900
    assert room(w, "keys", via="box", fresh=False, last_seen=t).text == \
        f"Your keys are in the box, which I last saw on the bookshelf at {clock(t)}."


def test_prop_uses_the(w):
    assert room(w, "box", say="the couch").text == f"The box is on the couch. It appeared there at {clock(NOW - 180)}."
    t = NOW - 900
    assert room(w, "box", fresh=False, last_seen=t).text == f"I last saw the box on the bookshelf at {clock(t)}."


def test_missing_times_still_speak(w):
    assert room(w, "keys", observed=False, arrived=None, last_seen=None).text == "Your keys are on the bookshelf."
    assert room(w, "keys", fresh=False, last_seen=None).text == "I last saw your keys on the bookshelf."


# ---------- pill bottle ----------

def test_pill_bottle_room_wording(w):
    a = room(w, "pill_bottle")
    assert a.text == f"Your pill bottle is on the bookshelf. It appeared there at {clock(NOW - 180)}."
    for kw in (dict(observed=False), dict(fresh=False), dict(fresh=False, absent=True),
               dict(via="box"), dict(via="box", fresh=False)):
        room(w, "pill_bottle", **kw)                         # spoken_ok checks no took / taken


def test_pill_bottle_table_wording_unaffected(w):
    assert ask(w, "pill_bottle").text == "Your pill bottle is under the notebook."


# ---------- conflicts ----------

def test_conflict_tail_on_table_answer(w):
    p = w.place("keys")
    p.conflicts = [conflict("keys"), conflict("keys", "the couch")]
    w.set_place("keys", p)
    a = ask(w, "keys")
    spoken_ok(a)
    assert a.text.startswith("Your keys are inside the box.")
    assert a.text.endswith(" I also see keys on the bookshelf.")
    assert "couch" not in a.text                          # only the first conflict
    assert (a.point_at, a.action) == ("keys", "point")    # the table answer keeps its laser action


def test_conflict_tail_on_room_answer(w):
    a = room(w, "keys", conflicts=[conflict("keys", "the couch")])
    assert a.text == f"Your keys are on the bookshelf. They appeared there at {clock(NOW - 180)}. " \
                     "I also see keys on the couch."


def test_conflict_tail_singular_takes_an_article(w):
    a = room(w, "wallet", conflicts=[conflict("wallet", "the couch")])
    assert a.text.endswith(" I also see a wallet on the couch.")


# ---------- FakeWorld.place and worlds without place ----------

def test_fakeworld_place_is_the_table_place(w):
    p = w.place("keys", NOW)
    assert (p.kind, p.zone, p.say) == ("table", "table", "the table")
    assert p.chain == ["keys", "box"] and p.via == "box" and not p.observed_directly
    assert p.pos_cm == w.get("box").pos_cm and p.status == Status.VISIBLE
    assert p.conflicts == []
    q = w.place("wallet")
    assert q.via == "wallet" and q.observed_directly and q.pos_cm == (60.0, 15.0)


def test_set_place_overrides_only_that_name(w):
    p = room_place("keys")
    w.set_place("keys", p)
    assert w.place("keys") is p
    assert w.place("wallet").kind == "table"


@pytest.mark.parametrize("name", ["keys", "pill_bottle", "wallet", "glasses", "phone", "remote", "box", "notebook"])
def test_table_answers_unchanged_by_place(w, name):
    with_place, without = ask(w, name), ask(NoPlace(w), name)
    assert (with_place.text, with_place.point_at, with_place.action) == \
        (without.text, without.point_at, without.action)


def test_world_without_place_works(w):
    nw = NoPlace(w)
    assert not hasattr(nw, "place")
    assert ask(nw, "wallet").text == "Your wallet is on the table."


def test_place_error_falls_back_to_table(w):
    def boom(name, now=None):
        raise RuntimeError("room memory broke")
    w.place = boom
    assert ask(w, "pill_bottle").text == "Your pill bottle is under the notebook."


def test_room_place_for_never_seen_on_table():
    fw = FakeWorld([Entity("keys", "target", Status.UNKNOWN, confidence=0.0)])
    fw.set_place("keys", room_place("keys"))
    assert ask(fw, "keys").text == f"Your keys are on the bookshelf. They appeared there at {clock(NOW - 180)}."


def test_absent_room_object_gets_no_look_alike_sentence(w):
    """_maybe_back hedges 'something similar came back' for lost table objects; a room object's answer is
    the room one, even when similar_to would offer a look-alike on the table."""
    w.set("wallet", status=Status.UNKNOWN, zone="shelf", pos_cm=None)
    w.set_place("wallet", room_place("wallet", fresh=False, absent=True))
    w.similar_to = lambda obj: [("remote", 0.9)]
    a = ask(w, "wallet")
    spoken_ok(a)
    assert a.text.endswith("I can't see it there now.") and "similar" not in a.text
    assert a.point_at is None


def test_under_a_cover_carried_to_a_room_zone_says_under(w):
    w.set("keys", status=Status.UNDER, parent="notebook")
    w.set_place("keys", room_place("keys", via="notebook"))
    a = ask(w, "keys")
    spoken_ok(a)
    assert a.text == "Your keys are under the notebook. The notebook is on the bookshelf."


def test_tentative_room_place_is_hedged_once(w):
    p = room_place("wallet")
    p.tentative = True
    w.set_place("wallet", p)
    a = ask(w, "wallet")
    spoken_ok(a)
    assert a.text.startswith("Your wallet, I think, is on the bookshelf.") and a.text.count("I think") == 1
    p2 = room_place("wallet", fresh=False, absent=True)
    p2.tentative = True
    w.set_place("wallet", p2)
    a = ask(w, "wallet")
    assert "what I think is your wallet" in a.text and a.text.count("I think") == 1
