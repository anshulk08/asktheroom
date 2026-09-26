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
