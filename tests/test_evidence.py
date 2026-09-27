"""Answer evidence (core/evidence.py): every answer about where something is or what happened carries its
proof picture. The WHERE / HISTORY / HANDLED templates cite the logged event's snapshot (a room arrival: the
whole camera view with the zone crop as its close-up), what-changed its first events, the room look the
frame it sent, recall the saved frames Grok says it used. /ask, last_answer and /state answers carry it,
and /snapshots serves archive frames too."""
import json
import os
import time

import cv2
import numpy as np
import pytest

from core import evidence
from core.config import load_config
from core.events import EventLog
from core.fakeworld import FakeWorld
from core.room_types import Place
from core.types import Answer, Entity, Event, Frame, Intent, Status
from voice.answers import answer, clock

CFG = load_config()
NOW = time.time()


@pytest.fixture
def log(tmp_path):
    lg = EventLog(str(tmp_path / "e.db"), str(tmp_path / "snaps"))
    yield lg
    lg.close()


def frame(wall, color=(60, 90, 120), size=(1280, 720)):
    return Frame(t=0.0, wall=wall, img=np.full((size[1], size[0], 3), color, np.uint8), idx=1)


def logged(log, obj, type_, wall, fr=None, context=None, **kw):
    ev = Event(t=0.0, wall=wall, obj=obj, type=type_, **kw)
    log.add(ev, fr if fr is not None else frame(wall), context=context)
    return ev


def ask(world, log, kind, obj):
    log.flush()                           # the snapshots are on disk, as seconds after an event on the rig
    return answer(Intent(kind, obj, ""), world, log, CFG, now=NOW)


class BoxWorld(FakeWorld):
    def box_px(self, name):
        return (100, 200, 180, 260)


def test_where_on_the_table_says_when_and_cites_the_put_down(log):
    w = BoxWorld([Entity("keys", "target", Status.VISIBLE, pos_cm=(20.0, 20.0), last_seen=NOW)], log)
    logged(log, "keys", "PICKED_UP", NOW - 400)
    down = logged(log, "keys", "PUT_BACK", NOW - 300, to_cm=(20.0, 20.0))
    a = ask(w, log, "WHERE", "keys")
    assert a.text.endswith(f"They were put there at {clock(NOW - 300)}.")
    [ev] = a.evidence
    assert ev["kind"] == "event" and ev["type"] == "PUT_BACK" and ev["obj"] == "keys"
    assert ev["snapshot_url"] == f"/snapshots/{os.path.basename(down.snapshot)}" and ev["t"] == round(NOW - 300, 3)
    assert ev["caption"] == f"Your keys, put back down at {clock(NOW - 300)}"
    assert ev["box"] == [100, 200, 180, 260] and ev["size"] == [1280, 720]    # nothing happened since: box now = then


def test_a_later_event_drops_the_box(log):
    w = BoxWorld([Entity("keys", "target", Status.VISIBLE, pos_cm=(20.0, 20.0), last_seen=NOW)], log)
    logged(log, "keys", "PUT_BACK", NOW - 300)
    logged(log, "keys", "LOST_TRACK", NOW - 100)
    a = ask(w, log, "WHERE", "keys")
    assert a.evidence[0]["type"] == "PUT_BACK" and a.evidence[0]["box"] is None


def test_just_now_for_a_fresh_put_down(log):
    w = FakeWorld([Entity("wallet", "target", Status.VISIBLE, pos_cm=(20.0, 20.0), last_seen=NOW)], log)
    logged(log, "wallet", "PUT_BACK", NOW - 20)
    assert ask(w, log, "WHERE", "wallet").text.endswith("It was put there just now.")


def test_where_in_a_room_zone_shows_the_whole_view_with_the_zone_close_up(log):
    full = np.full((1440, 2560, 3), 40, np.uint8)
    crop = np.full((300, 400, 3), 200, np.uint8)
    arrived = NOW - 200
    ev = logged(log, "keys", "FOUND", arrived, fr=Frame(0.0, arrived, crop, -1), context=full)
    log.flush()
    ctx = ev.snapshot[:-4] + "_room.jpg"
    assert cv2.imread(ctx).shape[:2] == (720, 1280)                          # the whole view, <= 1280 wide
    w = FakeWorld([Entity("keys", "target", Status.VISIBLE)], log)
    w.set_place("keys", Place(kind="room", zone="couch", say="the couch", status=Status.VISIBLE, chain=["keys"],
                              via="keys", box_px=(1000, 1100, 1100, 1180), observed_directly=True, fresh=True,
                              arrived_wall=arrived, last_seen_wall=NOW - 5, arrival_observed=True))
    a = answer(Intent("WHERE", "keys", ""), w, log, room_cfg(log), now=NOW)
    assert a.text == f"Your keys are on the couch. They appeared there at {clock(arrived)}."
    [e] = a.evidence
    assert e["snapshot_url"].endswith("_room.jpg") and e["closeup_url"].endswith("_keys_FOUND.jpg")
    assert e["box"] == [500, 550, 550, 590] and e["size"] == [1280, 720]      # camera px scaled to the saved view
    assert e["closeup_box"] == [100, 100, 200, 180] and e["closeup_size"] == [400, 300]   # and in the zone crop
    assert e["caption"] == f"Your keys, on the couch, confirmed at {clock(arrived)}"


def room_cfg(log):
    """capture 2560x1440, a couch zone whose crop is (900, 1000)-(1300, 1300) of the camera frame."""
    path = os.path.join(log.snap_dir, "..", "room_zones.json")
    with open(path, "w") as f:
        json.dump({"view": "t", "size_px": [2560, 1440], "zones": {"couch": {"say": "the couch", "poly": [
            [900, 1000], [1299, 1000], [1299, 1299], [900, 1299]]}}}, f)
    return {**CFG, "room_memory": {**(CFG.get("room_memory") or {}), "capture_size": [2560, 1440],
                                   "zones_path": path}}


def test_an_arrival_logged_before_whole_views_were_saved_shows_the_crop_boxed(log):
    crop = np.full((300, 400, 3), 200, np.uint8)
    logged(log, "keys", "FOUND", NOW - 200, fr=Frame(0.0, NOW - 200, crop, -1))          # no context
    log.flush()
    w = FakeWorld([Entity("keys", "target", Status.VISIBLE)], log)
    w.set_place("keys", Place(kind="room", zone="couch", say="the couch", status=Status.VISIBLE, chain=["keys"],
                              via="keys", box_px=(1000, 1100, 1100, 1180), observed_directly=True, fresh=True,
                              arrived_wall=NOW - 200, last_seen_wall=NOW - 5, arrival_observed=True))
    [e] = answer(Intent("WHERE", "keys", ""), w, log, room_cfg(log), now=NOW).evidence
    assert e["snapshot_url"].endswith("_keys_FOUND.jpg") and e["closeup_url"] is None
    assert e["box"] == [100, 100, 200, 180] and e["size"] == [400, 300] and e["box_px"] == [1000, 1100, 1100, 1180]


def test_handled_cites_the_put_down_then_the_pick_up(log):
    w = FakeWorld([Entity("keys", "target", Status.VISIBLE, pos_cm=(20.0, 20.0), last_seen=NOW)], log)
    logged(log, "keys", "PICKED_UP", NOW - 400, parent="hand:1")
    logged(log, "keys", "PUT_BACK", NOW - 300)
    a = ask(w, log, "HANDLED", "keys")
    assert [e["type"] for e in a.evidence] == ["PUT_BACK", "PICKED_UP"]


def test_history_cites_the_latest_events(log):
    w = FakeWorld([Entity("keys", "target", Status.VISIBLE, pos_cm=(20.0, 20.0), last_seen=NOW)], log)
    for i, t in enumerate(("PICKED_UP", "PUT_BACK", "MOVED")):
        logged(log, "keys", t, NOW - 500 + 100 * i)
    assert [e["type"] for e in ask(w, log, "HISTORY", "keys").evidence] == ["MOVED", "PUT_BACK"]


def test_what_changed_cites_its_first_events(log):
    w = FakeWorld([Entity(n, "target", Status.VISIBLE, pos_cm=(20.0, 20.0), last_seen=NOW)
                   for n in ("keys", "wallet", "phone")], log)
    logged(log, "keys", "PUT_BACK", NOW - 300)
    logged(log, "wallet", "MOVED", NOW - 200)
    logged(log, "phone", "PICKED_UP", NOW - 100)
    a = ask(w, log, "CHANGES", None)
    assert len(a.evidence) == 2 and all(e["kind"] == "change" for e in a.evidence)
    assert {e["obj"] for e in a.evidence} <= {"keys", "wallet", "phone"}


def test_no_snapshot_no_evidence_and_the_answer_is_unchanged(tmp_path):
    lg = EventLog(":memory:", str(tmp_path / "s"))
    w = FakeWorld([Entity("keys", "target", Status.VISIBLE, pos_cm=(20.0, 20.0), last_seen=NOW)], lg)
    lg.add(Event(t=0.0, wall=NOW - 300, obj="keys", type="PUT_BACK"))          # no frame: no snapshot
    a = answer(Intent("WHERE", "keys", ""), w, lg, CFG, now=NOW)
    assert a.evidence == [] and a == Answer(a.text, a.point_at, a.action, evidence=["ignored in =="])
    lg.close()


def test_snapshot_urls_only_for_plain_files_inside_the_snapshot_dir(tmp_path):
    snap = tmp_path / "snaps"
    (snap / "archive" / "20260926-23").mkdir(parents=True)
    assert evidence.snapshot_url(str(snap / "1_keys_FOUND.jpg"), str(snap)) == "/snapshots/1_keys_FOUND.jpg"
    assert evidence.snapshot_url(str(snap / "archive/20260926-23/17.jpg"), str(snap)) == \
        "/snapshots/archive/20260926-23/17.jpg"
    assert evidence.snapshot_url(str(tmp_path / "secret.jpg"), str(snap)) is None
    assert evidence.snapshot_url(str(snap / "sub" / "x.jpg"), str(snap)) is None
    assert evidence.snapshot_url(None, str(snap)) is None


# -- visual answers

def test_the_room_look_saves_and_cites_the_frame_it_sent(tmp_path):
    from tests.test_room_look import room_qa, room_reply
    lg = EventLog(":memory:", str(tmp_path / "snaps"))
    q, _ = room_qa(lg, reply=room_reply("Someone is on the couch."))
    a = q.look_room("what's on the couch?")
    [e] = a.evidence
    assert e["kind"] == "look" and e["snapshot_url"].endswith("_look.jpg")
    assert os.path.exists(tmp_path / "snaps" / e["snapshot_url"].split("/")[-1])
    lg.close()


def test_recall_cites_the_pictures_grok_says_it_used(tmp_path):
    from tests.test_visual import archive, fill, qa
    lg = EventLog(":memory:", str(tmp_path / "snaps"))
    q, prov = qa(lg, json.dumps({"seen": "", "answer": "Your keys were there at 9:10.", "confidence": 0.9,
                                 "pictures": [2, 9]}), recall_frames=3)
    q.archive = archive(lg)
    fill(q.archive, [(0, None), (10, (0, 0, 255)), (20, (0, 0, 255))])
    a = q.recall("What was on the table this morning?")
    [e] = a.evidence                                                        # 9 is not a picture it was sent
    assert e["kind"] == "recall" and e["snapshot_url"].startswith("/snapshots/archive/")
    assert e["caption"].startswith("Saved picture, ")
    lg.close()


# -- the server

def test_ask_last_answer_answers_and_archive_snapshots(tmp_path):
    from fastapi.testclient import TestClient
    from core.fakeworld import demo_world
    from server.app import create_app
    snaps = tmp_path / "snaps"
    lg = EventLog(str(tmp_path / "e.db"), str(snaps))
    (snaps / "archive" / "20260926-23").mkdir(parents=True)
    cv2.imwrite(str(snaps / "archive" / "20260926-23" / "17.jpg"), np.zeros((8, 8, 3), np.uint8))
    item = {"kind": "recall", "snapshot_url": "/snapshots/archive/20260926-23/17.jpg", "t": NOW,
            "caption": "Saved picture", "box": None, "size": None, "closeup_url": None, "obj": None, "type": None}
    cfg = {**CFG, "paths": dict(CFG["paths"], viewer=str(tmp_path / "viewer.json"))}
    app = create_app(cfg, demo_world(lg), lg, ask_fn=lambda text, src: Answer("There.", evidence=[item], obj="keys"))
    with TestClient(app) as c:
        r = c.post("/ask", json={"text": "where?"}).json()
        assert r["evidence"] == [item] and r["obj"] == "keys"
        st = c.get("/state").json()
        assert st["last_answer"]["evidence"] == [item] and st["answers"][-1]["evidence"] == [item]
        assert st["last_answer"]["obj"] == "keys" and st["answers"][-1]["obj"] == "keys"
        assert c.get(item["snapshot_url"]).status_code == 200
        assert c.get("/snapshots/archive/20260926-23/../../e.db").status_code in (400, 404)
        assert c.get("/snapshots/archive/x/17.jpg").status_code == 400
    lg.close()


def test_a_room_place_cites_the_arrival_nearest_its_arrival_time_with_camera_px(log):
    full = np.full((1440, 2560, 3), 40, np.uint8)
    crop = np.full((30, 40, 3), 200, np.uint8)
    first = logged(log, "keys", "FOUND", NOW - 900, fr=Frame(0.0, NOW - 900, crop, -1), context=full)
    logged(log, "keys", "FOUND", NOW - 60, fr=Frame(0.0, NOW - 60, crop, -1), context=full)   # a re-find later
    log.flush()
    w = FakeWorld([Entity("keys", "target", Status.VISIBLE)], log)
    w.set_place("keys", Place(kind="room", zone="couch", say="the couch", status=Status.VISIBLE, chain=["keys"],
                              via="keys", box_px=(1000, 1100, 1100, 1180), observed_directly=True, fresh=True,
                              arrived_wall=NOW - 890, last_seen_wall=NOW - 5, arrival_observed=True))
    a = answer(Intent("WHERE", "keys", ""), w, log, room_cfg(log), now=NOW)
    [e] = a.evidence
    assert e["t"] == round(first.wall, 3) and e["clock"] == clock(first.wall)
    assert e["box_px"] == [1000, 1100, 1100, 1180] and e["box"] == [500, 550, 550, 590]
    assert a.obj == "keys" and a.point_at is None                      # room answers never aim, but say what


def test_thing_snapshots_have_urls():
    assert evidence.snapshot_url("/s/1_thing:52_APPEARED.jpg", "/s") == "/snapshots/1_thing:52_APPEARED.jpg"
