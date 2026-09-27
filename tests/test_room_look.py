"""Grok looks at the whole room (spec 0009 M0 add-on): with room memory on, the camera's full frame is
available (TableView.latest_full). A question about the room or a drawn zone ("what's on the couch?")
sends that frame with the zone names; the answer is spoken only (no laser off the table)."""
import json

import numpy as np

from core.types import Answer, Frame
from tests.test_visual import T0, qa, log  # noqa: F401  (log is a fixture)
from voice.intents import parse
from tests.test_visual import CFG


class RoomFrames:
    """TableView's API: latest() is the table view, latest_full() the whole camera frame."""
    def __init__(self):
        self.table = np.zeros((720, 1280, 3), np.uint8)
        self.full = np.full((1080, 1920, 3), 60, np.uint8)

    def latest(self):
        return Frame(t=0.0, wall=T0, img=self.table, idx=1)

    def latest_full(self):
        return Frame(t=0.0, wall=T0, img=self.full, idx=1)


def room_reply(answer="There's a blue mug and a phone on the couch.", conf=0.9):
    return json.dumps({"answer": answer, "confidence": conf})


def room_qa(log, reply=None, zones=(("couch", "the couch"), ("bookshelf", "the bookshelf"))):
    q, prov = qa(log, reply or room_reply())
    q.frames = RoomFrames()
    q.room_zones = list(zones)
    return q, prov


def ask(q, text, online=True):
    return q.route(parse(text, CFG), text, online=online)


def test_a_question_about_a_zone_sends_the_whole_room_frame_with_the_zone_names(log):
    q, prov = room_qa(log)
    a = ask(q, "what's on the couch?")
    assert a == Answer("There's a blue mug and a phone on the couch.")      # spoken only: no point, no laser
    [call] = prov.calls
    images = [p for p in call.parts if p[0] == "image"]
    assert len(images) == 1
    texts = " ".join(p[1] for p in call.parts if p[0] == "text")
    assert "the couch" in texts and "the bookshelf" in texts
    s = call.system.lower()
    assert "room" in s and "medication" in s and "people" in s


def test_room_words_route_to_the_room_look_too(log):
    for text in ("is there anything on the floor?", "what's around the room?", "what's on the bookshelf?"):
        q, prov = room_qa(log)
        ask(q, text)
        assert len(prov.calls) == 1 and "room" in prov.calls[0].system.lower(), text


def test_a_table_question_still_looks_at_the_table(log):
    q, prov = room_qa(log, reply=json.dumps({"answer": "A set of keys.", "confidence": 0.9, "mark": None,
                                              "point": None}))
    ask(q, "what's on the table?")
    assert "tabletop" in prov.calls[0].system.lower()


def test_without_room_memory_room_questions_look_at_the_table_as_before(log):
    q, prov = qa(log, json.dumps({"answer": "I can only see the table.", "confidence": 0.9, "mark": None,
                                  "point": None}))
    ask(q, "what's on the couch?")
    assert "tabletop" in prov.calls[0].system.lower()


def test_offline_a_room_question_says_so(log):
    q, prov = room_qa(log)
    a = ask(q, "what's on the couch?", online=False)
    assert prov.calls == [] and "offline" in a.text.lower()


def test_an_unsure_room_answer_abstains(log):
    q, _ = room_qa(log, reply=room_reply("Maybe a cat?", conf=0.2))
    assert ask(q, "what's on the couch?").text == "I can't tell from here."


def test_the_disclosure_mentions_the_room_frame_when_room_memory_is_on(log):
    q, _ = room_qa(log)
    assert "room" in q.status()["disclosure"].lower()


# -- the whole room is the default view with room memory on (the room demo, spec 0010)

def pick_json(mark=None, conf=0.9):
    return json.dumps({"mark": mark, "label": None, "confidence": conf})


def test_what_do_you_see_looks_at_the_whole_room_by_default(log):
    for text in ("room, what do you see?", "what do you see?", "what's here?"):
        q, prov = room_qa(log, reply=room_reply("I see someone on the couch and a paper bag on the table."))
        a = ask(q, text)
        assert a.text.startswith("I see someone") and a.point_at is None, text
        assert "room" in prov.calls[0].system.lower() and "tabletop" not in prov.calls[0].system.lower(), text


def test_the_wake_word_does_not_make_a_table_question_a_room_question(log):
    q, prov = room_qa(log, reply=json.dumps({"answer": "A set of keys.", "confidence": 0.9, "mark": None,
                                              "point": None}))
    ask(q, "room, what's on the table?")
    assert "tabletop" in prov.calls[0].system.lower()


def test_a_zone_named_table_is_the_room(log):
    q, prov = room_qa(log, zones=(("side_table", "the side table"),))
    ask(q, "what's on the side table?")
    assert "tabletop" not in prov.calls[0].system.lower()


def test_where_is_an_unknown_thing_not_on_the_table_looks_in_the_whole_room(log):
    from tests.test_visual import pick_qa
    from voice.visual import PICK_SYSTEM
    q, prov = pick_qa(log, mark=None)
    q.frames, q.room_zones = RoomFrames(), [("couch", "the couch")]
    prov.replies = [pick_json(mark=None), room_reply("Your mug is on the couch.")]
    a = ask(q, "room, where is my mug?")
    assert [c.system == PICK_SYSTEM for c in prov.calls] == [True, False]      # the table first, then the room
    assert a == Answer("Your mug is on the couch.")                             # spoken only: no laser off the table


def test_an_unsure_table_pick_asks_the_room_too(log):
    from tests.test_visual import pick_qa
    q, prov = pick_qa(log, mark=2, conf=0.3)
    q.frames, q.room_zones = RoomFrames(), [("couch", "the couch")]
    prov.replies = [pick_json(mark=2, conf=0.3), room_reply("It's on the couch.")]
    assert ask(q, "where is my mug?").text == "It's on the couch."


def test_a_thing_picked_on_the_table_is_still_pointed_at(log):
    from tests.test_visual import pick_qa
    q, prov = pick_qa(log, mark=2)
    q.frames, q.room_zones = RoomFrames(), [("couch", "the couch")]
    a = ask(q, "where is my red mug?")
    assert len(prov.calls) == 1 and a.point_at == "thing:3"


def test_where_on_the_table_keeps_the_table_answer(log):
    from tests.test_visual import pick_qa
    q, prov = pick_qa(log, mark=None)
    q.frames, q.room_zones = RoomFrames(), [("couch", "the couch")]
    prov.replies = [pick_json(mark=None), room_reply("Your mug is on the couch.")]
    assert ask(q, "is my mug on the table?") == Answer("I can't see your mug on the table right now.")
    assert len(prov.calls) == 1                                                 # the table pick only
