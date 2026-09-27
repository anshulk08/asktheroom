"""Grok looks at the whole room (spec 0009 M0 add-on): with room memory on, the camera's full frame is
available (TableView.latest_full). A question about the room or a drawn zone ("what's on the couch?")
sends that frame with the zone names; the answer is spoken only (no laser off the table)."""
import json

import numpy as np
import pytest

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


# -- whole-room seeing (spec 0010): zone close-ups, honest absence, room frames for recall

ZONES = {"couch": {"say": "the couch", "poly": [[600, 700], [1300, 700], [1300, 1080], [600, 1080]]},
         "counter": {"say": "the kitchen counter", "poly": [[1700, 500], [1900, 500], [1900, 560], [1700, 560]]}}


def zoned_qa(log, tmp_path, reply=None, zones=ZONES, size=(1920, 1080), **kw):
    """room_qa with a zones file the look reads its polygons from (drawn on a frame of `size`)."""
    path = tmp_path / "room_zones.json"
    path.write_text(json.dumps({"view": "test", "size_px": list(size), "zones": zones}))
    q, prov = qa(log, reply or room_reply(), **kw)
    q.frames = RoomFrames()
    q.room_zones = [(n, z["say"]) for n, z in zones.items()]
    q.cfg = {**CFG, "room_memory": {**(CFG.get("room_memory") or {}), "zones_path": str(path)}}
    return q, prov


def images(call):
    return [p[1] for p in call.parts if p[0] == "image"]


def texts(call):
    return " ".join(p[1] for p in call.parts if p[0] == "text")


def decode(jpg):
    import cv2
    return cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)


def test_a_named_zone_gets_a_native_close_up_beside_the_whole_view(log, tmp_path):
    q, prov = zoned_qa(log, tmp_path)
    q.frames.full[500:560, 1700:1900] = (0, 128, 255)            # something orange on the far counter
    ask(q, "what's on the kitchen counter?")
    [call] = prov.calls
    whole, close = images(call)
    assert "Image 2: close-up of the kitchen counter" in texts(call)
    c = decode(close)
    assert max(c.shape[:2]) == q.c.room_crop_px                   # upscaled from the native crop
    assert (np.abs(c.astype(int) - (0, 128, 255)).sum(axis=2) < 60).mean() > 0.1   # the counter fills it
    assert "the kitchen counter (right side, middle height of the view)" in texts(call)


def test_an_unnamed_question_gets_close_ups_of_the_far_zones_only(log, tmp_path):
    q, prov = zoned_qa(log, tmp_path)
    ask(q, "what do you see?")
    t = texts(call := prov.calls[0])
    assert len(images(call)) == 2 and "close-up of the kitchen counter" in t and "close-up of the couch" not in t


def test_the_close_ups_are_capped(log, tmp_path):
    far = {f"shelf_{i}": {"say": f"shelf {i}", "poly": [[100 * i, 100], [100 * i + 50, 100], [100 * i + 50, 140],
                                                         [100 * i, 140]]} for i in range(5)}
    q, prov = zoned_qa(log, tmp_path, zones=far)
    ask(q, "what do you see?")
    assert len(images(prov.calls[0])) == 1 + q.c.room_crops


def test_zones_drawn_at_another_size_are_scaled_to_the_frame(log, tmp_path):
    half = {"counter": {"say": "the kitchen counter", "poly": [[850, 250], [950, 250], [950, 280], [850, 280]]}}
    q, prov = zoned_qa(log, tmp_path, zones=half, size=(960, 540))
    q.frames.full[500:560, 1700:1900] = (0, 128, 255)
    ask(q, "what's on the kitchen counter?")
    c = decode(images(prov.calls[0])[1])
    assert (np.abs(c.astype(int) - (0, 128, 255)).sum(axis=2) < 60).mean() > 0.1


def test_without_a_zones_file_the_room_look_still_answers(log, tmp_path):
    q, prov = room_qa(log)
    q.cfg = {**CFG, "room_memory": {"zones_path": str(tmp_path / "missing.json")}}
    assert ask(q, "what's on the couch?").text.startswith("There's a blue mug")
    assert len(images(prov.calls[0])) == 1


def test_zone_box_rises_above_the_surface_and_stays_in_the_frame():
    from voice.visual import zone_box
    x1, y1, x2, y2 = zone_box([(1700, 500), (1900, 500), (1900, 560), (1700, 560)], (1920, 1080))
    assert y1 < 500 - 60 and y2 > 560 and x1 < 1700 and x2 == 1920          # grown, upwards most, clipped
    assert x2 - x1 >= 200 and y2 - y1 >= 256                                # at least min_px tall
    assert zone_box([(0, 0), (1, 1)], (1920, 1080)) is None


def room_seen_reply(answer, conf=0.9, seen="a couch, a table, a kitchen counter with a cup"):
    return json.dumps({"seen": seen, "answer": answer, "confidence": conf})


def test_a_confident_i_dont_see_it_is_said_and_seen_is_never_spoken(log, tmp_path):
    q, prov = zoned_qa(log, tmp_path, reply=room_seen_reply("I don't see a mug anywhere.", 0.85))
    a = ask(q, "where is my mug?")
    assert a == Answer("I don't see a mug anywhere.")
    from voice.visual import ROOM_SCHEMA
    assert list(ROOM_SCHEMA["properties"])[0] == "seen"                     # listed before the answer
    s = prov.calls[-1].system                                                # after the table pick found no mark
    assert "don't see it" in s and "don't say it isn't in the room" in s


def test_a_known_object_the_tracker_never_saw_is_looked_for_in_the_room(log):
    from core.fakeworld import FakeWorld
    from core.types import Entity
    world = FakeWorld([Entity("glasses", "target")], log)                   # UNKNOWN, never placed
    q, prov = qa(log, room_seen_reply("I don't see your glasses."), world=world)
    q.frames, q.room_zones = RoomFrames(), [("couch", "the couch")]
    assert ask(q, "where are my glasses?").text == "I don't see your glasses."
    assert "room" in prov.calls[0].system.lower()
    assert "glasses" not in texts(prov.calls[0]).split("Question:")[0]      # no 'glasses: unknown' for Grok to echo
    assert ask(q, "where are my glasses?", online=False) is None             # offline: the world model answers


def test_a_known_object_the_tracker_has_placed_stays_with_the_world_model(log):
    from core.fakeworld import FakeWorld
    from core.types import Entity, Status
    world = FakeWorld([Entity("glasses", "target", Status.VISIBLE, pos_cm=(10.0, 10.0), last_seen=T0)], log)
    q, prov = qa(log, room_seen_reply("On the couch."), world=world)
    q.frames, q.room_zones = RoomFrames(), [("couch", "the couch")]
    assert ask(q, "where are my glasses?") is None and prov.calls == []


def test_the_table_prompts_no_longer_say_the_camera_looks_straight_down_only():
    from core.grok_check import SYSTEM as CHECK
    from voice.visual import LOOK_SYSTEM, PICK_SYSTEM, RECALL_SYSTEM, ROOM_RECALL_SYSTEM, ROOM_SYSTEM
    for s in (LOOK_SYSTEM, PICK_SYSTEM, RECALL_SYSTEM, CHECK):
        assert "at an angle" in s and "nothing beyond the table" not in s
    for s in (ROOM_SYSTEM, ROOM_RECALL_SYSTEM):
        assert "corner" in s and "someone" in s


# -- room frames in the archive, and recall from them

class FullSource(RoomFrames):
    """A TableView-like source: full_at(t) is the whole camera view (1920x1080, a colour per call)."""
    def __init__(self, color=(40, 160, 40)):
        super().__init__()
        self.full[:] = color

    def full_at(self, t):
        return Frame(t=t, wall=T0 + t, img=self.full, idx=1)


def room_archive(log, **kw):
    from tests.test_visual import archive
    a = archive(log, **kw)
    a.frames = FullSource()
    return a


def feed(a, t, im=None, events=()):
    from tests.test_visual import dets, frame, img
    a.feed(frame(t, img() if im is None else im), dets(t), list(events))
    a.drain()


def test_room_frames_are_kept_on_their_own_schedule(log):
    import cv2
    a = room_archive(log)
    for t in range(0, 70):
        feed(a, float(t))
    rows = a.store.window(T0 - 1, T0 + 1e4)
    room = [r for r in rows if r.view == "room"]
    assert [r.t - T0 for r in room] == [0.0, 30.0, 60.0]                     # every room_every_s
    assert all(r.view == "table" for r in rows if r not in room)
    im = cv2.imread(room[0].path)
    assert im.shape[1] == a.c.room_frame_px and tuple(im[5, 5]) == pytest.approx((40, 160, 40), abs=4)
    assert room[0].emb is not None and room[0].emb.shape[0] == 1 + 4 * 3      # the finer room tile grid
    assert a.store.stats()["room_frames"] == 3


def test_a_world_event_brings_the_next_room_frame_forward(log):
    from core.types import Event
    a = room_archive(log)
    feed(a, 0.0)
    feed(a, 5.0, events=[Event(t=5.0, wall=T0 + 5, obj="keys", type="MOVED")])      # too soon
    feed(a, 12.0, events=[Event(t=12.0, wall=T0 + 12, obj="keys", type="MOVED")])
    assert [r.t - T0 for r in a.store.window(T0 - 1, T0 + 1e4, "room")] == [0.0, 12.0]


def test_without_a_full_frame_source_no_room_frames(log):
    from tests.test_visual import archive
    a = archive(log)
    for t in range(0, 40):
        feed(a, float(t))
    assert a.store.window(T0 - 1, T0 + 1e4, "room") == []


def test_room_and_table_frames_have_separate_disk_caps(log):
    a = room_archive(log)
    for t in range(0, 400, 10):
        feed(a, float(t))
    table = sum(r.bytes for r in a.store.window(T0 - 1, T0 + 1e4, "table"))
    room = a.store.window(T0 - 1, T0 + 1e4, "room")
    cap = room[-1].bytes * 2.5 / 1e6
    a.store.prune(24, cap, now=T0 + 400, view="room")
    assert len(a.store.window(T0 - 1, T0 + 1e4, "room")) == 2                # oldest room frames went ...
    assert sum(r.bytes for r in a.store.window(T0 - 1, T0 + 1e4, "table")) == table   # ... the table's stayed


def test_an_archive_from_before_room_frames_gains_the_view_column(tmp_path):
    import sqlite3
    from core.events import EventLog
    from core.visual_memory import ArchiveStore
    db = str(tmp_path / "old.db")
    c = sqlite3.connect(db)
    c.execute("CREATE TABLE visual_frames (id INTEGER PRIMARY KEY, t REAL, path TEXT, bytes INTEGER, score REAL, "
              "hands INTEGER, reason TEXT, digest TEXT, emb BLOB, emb_n INTEGER, emb_model TEXT)")
    c.execute("INSERT INTO visual_frames (t, path, bytes) VALUES (?, 'old.jpg', 10)", (T0,))
    c.commit()
    c.close()
    lg = EventLog(db, str(tmp_path / "snaps"))
    try:
        st = ArchiveStore(lg)
        [old] = st.window(T0 - 1, T0 + 1, "table")                          # old rows are table frames
        assert old.view == "table" and st.window(T0 - 1, T0 + 1, "room") == []
    finally:
        lg.close()


def recall_room_qa(log, tmp_path, reply=None):
    q, prov = zoned_qa(log, tmp_path, reply=reply or json.dumps(
        {"seen": "8:00 a cup on the counter", "answer": "There was a cup on the kitchen counter at about 9:30.",
         "confidence": 0.9}))
    q.archive = room_archive(log)
    q.archive.frames.full[500:560, 1700:1900] = (0, 128, 255)
    for m in (0, 30, 60):
        feed(q.archive, m * 60.0)
    return q, prov


def test_a_past_room_question_is_answered_from_the_room_frames(log, tmp_path):
    from voice.visual import ROOM_RECALL_SYSTEM
    q, prov = recall_room_qa(log, tmp_path)
    a = ask(q, "what was on the kitchen counter earlier?")                    # was not routed at all before
    assert a.text == "There was a cup on the kitchen counter at about 9:30."
    [call] = prov.calls
    assert call.system == ROOM_RECALL_SYSTEM
    t = texts(call)
    assert "Picture 1 at 9:00 AM" in t and "Picture 3 at 10:00 AM" in t
    assert t.count("close-up of the kitchen counter") == 3                   # the named area, from each picture
    assert len(images(call)) == 6 and decode(images(call)[0]).shape[1] == 1280


def test_a_past_table_question_keeps_to_the_table_frames(log, tmp_path):
    from voice.visual import RECALL_SYSTEM
    q, prov = recall_room_qa(log, tmp_path, reply=json.dumps({"seen": "", "answer": "Your keys.", "confidence": 0.9}))
    ask(q, "what was on the table earlier?")
    assert prov.calls[0].system == RECALL_SYSTEM and "Frame 1 at" in texts(prov.calls[0])


def test_with_no_room_frames_yet_recall_uses_the_table_frames(log, tmp_path):
    from tests.test_visual import archive
    from voice.visual import RECALL_SYSTEM
    q, prov = zoned_qa(log, tmp_path, reply=json.dumps({"seen": "", "answer": "Keys.", "confidence": 0.9}))
    q.archive = archive(log)
    for m in (0, 30):
        feed(q.archive, m * 60.0)
    ask(q, "what did you see earlier?")
    assert prov.calls[0].system == RECALL_SYSTEM


def test_the_disclosure_mentions_saved_room_frames(log):
    q, _ = room_qa(log)
    assert "whole room" in q.status()["disclosure"]


def test_a_weak_text_match_on_room_frames_still_asks_grok(log, tmp_path):
    q, prov = recall_room_qa(log, tmp_path)
    q.c.min_sim = 0.99                                    # no room frame matches the words well enough
    a = ask(q, "was there a red mug here earlier?")
    assert a.text.startswith("There was a cup") and len(prov.calls) == 1   # frames over the window, not "I don't remember"
