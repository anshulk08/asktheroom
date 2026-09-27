"""The /live engineering view (server/live.py, server/app.py /live, main.Room.live_state). No network."""
import logging
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi.testclient import TestClient

from core.config import load_config
from core.events import EventLog
from core.fakeworld import demo_world
from core.types import Answer
from server.app import create_app
from server.live import LiveLog, summarize

T0 = 1_000_000.0


def rec(name, msg, t, live=None, level=logging.INFO):
    r = logging.LogRecord(name, level, __file__, 1, msg, None, None)
    r.created = t
    if live is not None:
        r.live = live
    return r


def feed(ring, rows):
    for name, msg, t, *live in rows:
        r = rec(name, msg, T0 + t, live[0] if live else None)
        ring.add(r, r.getMessage())


def asked(text, t, source="voice"):
    return ("voice.pipeline", f"asked ({source}): {text!r}", t, {"stage": "asked", "text": text, "source": source})


def answered(path, text, t, action=None):
    return ("voice.pipeline", f"answer via {path} in 20 ms: {text!r}", t,
            {"stage": "answer", "path": path, "answer": text, "action": action})


def test_a_mic_question_groups_heard_route_answer_and_the_laser_hit():
    ring = LiveLog()
    feed(ring, [
        ("askroom.main", "heard the wake word alone; listening for the question (speech 320 ms)", 0.0),
        ("voice.stt", "recorded 2.3 s, stopped by silence, speech=True", 1.0),
        ("askroom.main", "heard: 'Point to the couch.'", 3.0),
        asked("Point to the couch.", 3.01),
        ("voice.room_places", "zone couch answers the point question 'Point to the couch.'", 3.02),
        answered("zone point", "That's the couch.", 3.03, "room:1125,1250,906,1053,1320,1440"),
        ("askroom.main", "laser -> room px (1125, 1250): in_box after 2 tries, err 14.2 px", 6.0,
         {"laser": {"first_raw_px": 40.0}}),
        ("askroom.main", "timing {'record_transcribe_s': 2.8}", 6.1),
        ("core.room", "room track r:9 in couch looks like a laptop", 6.5),
    ])
    snap = ring.snapshot(now=T0 + 60)
    (q,) = snap["questions"]
    assert q["text"] == "Point to the couch." and q["source"] == "voice"
    assert q["route"] == "zone point" and q["answer"] == "That's the couch." and q["action"].startswith("room:")
    assert q["outcome"] == "hit"
    assert q["laser"] == {"kind": "room", "target": None, "px": [1125.0, 1250.0], "reason": "in_box", "tries": 2,
                          "err_px": 14.2, "first_raw_px": 40.0}
    assert [s["stage"] for s in q["stages"]] == ["wake", "heard", "asked", "route", "answer", "laser", "timing"]
    assert q["stages"][1]["dt"] == 0.0 and q["stages"][-2]["dt"] == 3.0
    assert "took" in q["stages"][0]
    # the mic's clip lines stay out; the naming line is the ambient feed's
    assert [r["msg"] for r in snap["ambient"]] == ["room track r:9 in couch looks like a laptop"]


@pytest.mark.parametrize("lines, outcome", [
    ([("askroom.main", "laser -> thing:147 at room px (365, 1264): budget after 3 tries, err 1147.0 px", 5)], "miss"),
    ([("askroom.main", "laser -> thing:9 at room px (1, 2): jumped after 3 tries, err 77.6 px", 5)], "miss"),
    ([("askroom.main", "room aim refused: (10, 10) is outside every room zone", 5)], "refused"),
    ([("askroom.main", "laser refused: the laser's zero moved (2 aims in a row)", 5)], "refused"),
    ([("askroom.main", "no point cue in the question: answered without the laser", 5)], "no_cue"),
    ([("askroom.main", "laser -> keys at (40.0, 30.0) via table, err 1.2 cm", 5)], "hit"),
    ([], "spoken"),
])
def test_outcomes(lines, outcome):
    ring = LiveLog()
    feed(ring, [("askroom.main", "heard: 'where are my keys'", 0), answered("world model templates", "On the table.", 1)]
         + lines)
    assert ring.snapshot(now=T0 + 60)["questions"][0]["outcome"] == outcome


def test_a_refusal_keeps_its_reason():
    ring = LiveLog()
    feed(ring, [("askroom.main", "heard: 'point to the pills'", 0),
                ("askroom.main", "laser refused: the laser's zero moved", 1)])
    q = ring.snapshot(now=T0 + 60)["questions"][0]
    assert q["laser"]["reason"] == "refused" and "zero moved" in q["laser"]["why"]


def test_a_phone_question_opens_its_own_entry_and_a_late_line_is_ambient():
    ring = LiveLog()
    feed(ring, [("askroom.main", "heard: 'where is the remote'", 0), answered("room track", "On the couch.", 1),
                asked("where are my keys", 10, source="phone"), answered("world model templates", "Here.", 10.5),
                ("askroom.main", "LASER LOCKED: the laser's zero moved", 200)])
    snap = ring.snapshot(now=T0 + 300)
    assert [q["text"] for q in snap["questions"]] == ["where are my keys", "where is the remote"]
    assert snap["questions"][0]["source"] == "phone"
    assert snap["ambient"][0]["stage"] == "lock"


def test_a_question_without_an_answer_yet_is_pending():
    ring = LiveLog()
    feed(ring, [("askroom.main", "heard: 'what color is my cup'", 0)])
    assert ring.snapshot(now=T0 + 2)["questions"][0]["outcome"] == "pending"


def test_the_ring_is_bounded():
    ring = LiveLog()
    for i in range(80):
        feed(ring, [("askroom.main", f"heard: 'q{i}'", i * 100.0)])
    for i in range(3000):
        feed(ring, [("core.room", f"room track r:{i} in couch looks like a cup", 9000 + i)])
    snap = ring.snapshot(now=T0 + 20000)
    assert len(snap["questions"]) == 50 and snap["questions"][0]["text"] == "q79"
    assert len(ring.records) == 2000 and len(ring.ambient) == 200


def test_a_bad_record_never_raises():
    ring = LiveLog()
    ring.emit(logging.LogRecord("askroom.main", logging.INFO, __file__, 1, "%d", ("not a number",), None))
    assert summarize({"id": 1, "t": T0, "text": "x", "source": "voice", "stages": []}, T0 + 60)["outcome"] == "spoken"


# ---------------------------------------------------------------- the routes

class Frames:
    def latest(self):
        return None


@pytest.fixture
def make(tmp_path):
    def build(live_fn=None, events=None):
        cfg = load_config()
        cfg["paths"] = dict(cfg["paths"], viewer=str(tmp_path / "viewer.json"))
        ev = events or EventLog(str(tmp_path / "e.db"), str(tmp_path / "snaps"))
        world = demo_world(ev)
        app = create_app(cfg, world, ev, frames=Frames(), ask_fn=lambda text, src: Answer("On the table."),
                         live_fn=live_fn)
        return TestClient(app), ev
    return build


def test_live_page_is_local_html(make):
    client, _ = make()
    r = client.get("/live")
    assert r.status_code == 200 and "text/html" in r.headers["content-type"]
    assert "/static/live.js" in r.text and "googleapis" not in r.text and "cdn" not in r.text.lower()
    assert client.get("/static/live.js").status_code == 200 and client.get("/static/live.css").status_code == 200


def test_live_state_shape(make, caplog):
    caplog.set_level(logging.INFO, logger="askroom.main")         # main.py logs INFO; pytest's root is WARNING
    rig = {"laser": {"locked": None, "drift_n": 0}, "online": True,
           "room": {"zones": [{"name": "couch", "say": "the couch"}], "tracks": []}}
    client, ev = make(live_fn=lambda: rig)
    ev.log_question("where are my keys", "WHERE", "keys", "On the table.", True, 12)
    logging.getLogger("askroom.main").info("heard: %r", "where are my keys")
    s = client.get("/live/state").json()
    assert {"server_t", "errors", "questions", "ambient", "rig", "history", "world"} <= set(s)
    assert s["rig"] == rig and s["errors"] == {}
    assert s["history"][0]["text"] == "where are my keys" and s["history"][0]["answer"] == "On the table."
    assert {"name", "status", "zone"} <= set(s["world"]["entities"][0])
    assert any(q["text"] == "where are my keys" for q in s["questions"])


def test_a_failing_source_leaves_the_others(make):
    def boom():
        raise RuntimeError("laser gone")
    client, _ = make(live_fn=boom)
    r = client.get("/live/state")
    assert r.status_code == 200
    s = r.json()
    assert "laser gone" in s["errors"]["rig"] and "rig" not in s
    assert "history" in s and "world" in s and "questions" in s


def test_live_is_read_only(make):
    client, _ = make()
    assert client.post("/live/state").status_code == 405
    assert client.post("/live").status_code == 405


# ---------------------------------------------------------------- main.Room.live_state

def test_room_live_state_is_plain_json_and_survives_a_broken_part():
    import json

    from main import Room
    room = Room.__new__(Room)
    tr = SimpleNamespace(tid="r:1", zone="couch", cls="thing", guess={"name": "laptop", "confidence": 0.8},
                         confirmed=True, role="pending", entity=None, hits=3, misses=0, last_wall=0.0,
                         box_px=(1, 2, 3, 4))
    zone = SimpleNamespace(name="couch", say="the couch")
    room.laser = SimpleNamespace(room_map=None, px_bias=np.array([3.0, -2.0]),
                                 state={"on": False, "target": None, "err": np.float64(np.nan)},
                                 last_aim={"tries": 2, "first_raw_px": np.float32(12.5), "unsafe": None})
    room.laser_locked, room._drift_n, room.drift_aims = None, 1, 2
    room.room_enabled, room.aim_cue = True, "point"
    room.netmon = SimpleNamespace(online=True)
    room.room_memory = SimpleNamespace(zones=SimpleNamespace(zones={"couch": zone}),
                                       tracker=SimpleNamespace(tracks=lambda: [tr]))
    s = room.live_state()
    json.dumps(s, allow_nan=False)
    assert s["laser"]["px_bias"] == [3.0, -2.0] and s["laser"]["state"]["err"] is None
    assert s["laser"]["last_aim"]["first_raw_px"] == 12.5 and s["online"] is True
    assert s["room"]["tracks"][0]["name"] == "laptop" and s["room"]["zones"] == [{"name": "couch", "say": "the couch"}]
    room.room_memory = SimpleNamespace(zones=None, tracker=None)          # broken: the laser part still comes back
    s = room.live_state()
    assert "error" in s["room"] and s["laser"]["drift_n"] == 1
