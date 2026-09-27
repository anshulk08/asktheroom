"""A configured prop the detector never labelled is answered through a Grok-named thing, hedged (spec 0009
rig run: from the room camera the fine-tuned detector didn't label the remote; YOLOE + Grok called it
'remote control')."""
from __future__ import annotations

import time

from core.config import load_config
from core.fakeworld import FakeWorld
from core.types import Entity, Intent, Status
from voice.answers import answer

CFG = load_config()
NOW = time.time()


def world(remote_seen=False):
    ents = [Entity(n, k, Status.UNKNOWN, confidence=0.0) for n, k in (CFG.get("objects") or {}).items()]
    w = FakeWorld(ents)
    w.entities["thing:15"] = Entity("thing:15", "target", Status.VISIBLE, pos_cm=(80.0, 10.0), last_seen=NOW)
    if remote_seen:
        w.set("remote", status=Status.VISIBLE, pos_cm=(20.0, 20.0), last_seen=NOW, confidence=1.0)
    w.find_guess = lambda said: [("thing:15", 0.9)] if "remote" in said else []
    return w


def ask(w, obj="remote"):
    return answer(Intent("WHERE", obj, f"where is my {obj}"), w, w.events, CFG, now=NOW)


def test_never_seen_prop_is_answered_through_the_grok_named_thing_hedged():
    a = ask(world())
    assert "I think" in a.text and "remote" in a.text and "haven't seen" not in a.text
    assert a.point_at == "thing:15"


def test_a_seen_prop_keeps_its_own_answer():
    a = ask(world(remote_seen=True))
    assert "I think" not in a.text and a.point_at == "remote"


def test_no_guess_keeps_the_never_seen_answer():
    w = world()
    w.find_guess = lambda said: []
    assert ask(w).text.startswith("I haven't seen your remote yet.")


def test_the_prop_prompts_are_tried_too():
    w = world()
    w.find_guess = lambda said: [("thing:15", 0.8)] if said == "tv remote" else []
    a = ask(w)
    assert "I think" in a.text and a.point_at == "thing:15"


def test_guessed_thing_in_a_tentative_room_place_is_hedged_once():
    from core.room_types import Place
    w = world()
    w.set("thing:15", zone="couch", pos_cm=None)
    w.set_place("thing:15", Place(kind="room", zone="couch", say="the couch", status=Status.VISIBLE,
                                  chain=["thing:15"], via="thing:15", fresh=True, arrived_wall=NOW - 60,
                                  last_seen_wall=NOW - 1, arrival_observed=True, tentative=True))
    a = ask(w)
    assert a.text.startswith("Your remote, I think, is on the couch.") and a.text.count("I think") == 1


def test_the_freshest_of_several_guessed_things_answers():
    """Trial run: each trip back to the table starts a new thing, and a stale couch copy lingers; the one
    seen just now answers, not the one whose Grok phrase scores best."""
    w = world()
    w.entities["thing:20"] = Entity("thing:20", "target", Status.VISIBLE, pos_cm=(10.0, 10.0), last_seen=NOW + 5)
    w.find_guess = lambda said: [("thing:15", 3.0), ("thing:20", 1.5)] if "remote" in said else []
    assert ask(w).point_at == "thing:20"
    w.set("thing:20", status=Status.UNKNOWN)
    assert ask(w).point_at == "thing:15"


def test_a_lost_prop_defers_to_a_fresher_grok_named_thing():
    """Rig: the detector caught the real wallet once (prop 'wallet' seen, then LOST_TRACK), while the
    same wallet lived on as a Grok-named thing carried to the side table."""
    w = world()
    w.set("wallet", status=Status.UNKNOWN, pos_cm=(80.0, 60.0), last_seen=NOW - 120, confidence=0.4)
    w.entities["thing:15"].last_seen = NOW - 5
    w.find_guess = lambda said: [("thing:15", 3.0)] if "wallet" in said else []
    a = answer(Intent("WHERE", "wallet", "where is my wallet"), w, w.events, CFG, now=NOW)
    assert "I think" in a.text and a.point_at == "thing:15" and "lost track" not in a.text


def test_a_visible_or_hidden_prop_still_answers_itself():
    w = world()
    w.set("wallet", status=Status.UNDER, parent="notebook", pos_cm=(20.0, 20.0), last_seen=NOW - 30, confidence=0.85)
    w.entities["thing:15"].last_seen = NOW
    w.find_guess = lambda said: [("thing:15", 3.0)] if "wallet" in said else []
    a = answer(Intent("WHERE", "wallet", "where is my wallet"), w, w.events, CFG, now=NOW)
    assert a.point_at == "wallet" and "under the notebook" in a.text


def test_an_older_grok_thing_does_not_override_a_more_recently_seen_prop():
    w = world()
    w.set("wallet", status=Status.UNKNOWN, pos_cm=(80.0, 60.0), last_seen=NOW - 10, confidence=0.4)
    w.entities["thing:15"].last_seen = NOW - 300
    w.find_guess = lambda said: [("thing:15", 3.0)] if "wallet" in said else []
    a = answer(Intent("WHERE", "wallet", "where is my wallet"), w, w.events, CFG, now=NOW)
    assert a.point_at == "wallet"
