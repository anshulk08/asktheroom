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
