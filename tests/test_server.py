"""Server tests (V7, V8, V11). FastAPI TestClient only; no network."""
import time
from urllib.parse import urlencode

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient
from twilio.request_validator import RequestValidator

from core.config import load_config
from core.events import EventLog
from core.fakeworld import demo_world
from core.types import Answer, Event, Frame
from server import overlay
from server.app import EventCursor, create_app

TOKEN = "test-auth-token-123"
GOOD = "+14045550123"


class Frames:
    def __init__(self):
        self.img = np.full((720, 1280, 3), 90, np.uint8)

    def latest(self):
        return Frame(t=time.monotonic(), wall=time.time(), img=self.img, idx=1)


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.delenv("ASKROOM_SMS_INSECURE", raising=False)
    monkeypatch.delenv("ASKROOM_PUBLIC_URL", raising=False)
    monkeypatch.setenv("TWILIO_AUTH_TOKEN", TOKEN)
    cfg = load_config()
    cfg["sms"] = {"whitelist": [GOOD]}
    cfg["paths"] = dict(cfg["paths"], viewer=str(tmp_path / "viewer.json"))   # never the repo's data/viewer.json
    cfg["server"] = dict(cfg["server"], push_hz=50, mjpeg_fps=50)
    snaps = tmp_path / "snaps"
    events = EventLog(str(tmp_path / "e.db"), str(snaps))
    world = demo_world(events)
    calls = []

    def ask_fn(text, source):
        calls.append((text, source))
        return Answer(text="The keys are inside the box <&>.", point_at="keys", action="point")

    (tmp_path / "secret.txt").write_text("secret")
    app = create_app(cfg, world, events, frames=Frames(), ask_fn=ask_fn)
    with TestClient(app) as client:
        yield dict(client=client, events=events, world=world, calls=calls, snaps=snaps, tmp=tmp_path)


def test_index_serves_html(env):
    r = env["client"].get("/")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]
    assert "Ask the Room" in r.text
    assert "vendor/vis-network.min.js" in r.text
    assert "googleapis" not in r.text and "cdn" not in r.text.lower()
    for path in ("/static/app.js", "/static/app.css", "/static/vendor/vis-network.min.js",
                 "/static/fonts/atkinson-next-latin-wght.woff2"):
        assert env["client"].get(path).status_code == 200, path


def test_ws_sends_state_with_entities_and_event_deltas(env):
    with env["client"].websocket_connect("/ws") as ws:
        first = ws.receive_json()
        assert first["initial"] is True
        names = {e["name"] for e in first["state"]["entities"]}
        assert {"keys", "box", "notebook"} <= names
        assert ["keys", "INSIDE", "box"] in first["state"]["edges"]
        assert len(first["events"]) == 10          # demo_world's history
        env["events"].add(Event(t=1.0, wall=time.time(), obj="wallet", type="PICKED_UP", parent="hand:1"))
        got = []
        for _ in range(20):
            msg = ws.receive_json()
            assert msg["initial"] is False
            got += msg["events"]
            if got:
                break
        assert [e["type"] for e in got] == ["PICKED_UP"]
        assert got[0]["obj"] == "wallet"


def test_event_cursor_no_duplicates(tmp_path):
    log = EventLog(str(tmp_path / "e.db"), str(tmp_path / "s"))
    now = time.time()
    log.add(Event(t=1, wall=now, obj="keys", type="MOVED"))
    c = EventCursor(0)
    assert len(c.poll(log)) == 1
    assert c.poll(log) == []
    log.add(Event(t=2, wall=now, obj="phone", type="MOVED"))   # same wall time, different event
    assert [e.obj for e in c.poll(log)] == ["phone"]
    assert c.poll(log) == []


def test_events_since_and_safe_snapshot_urls(env):
    events, snaps = env["events"], env["snaps"]
    now = time.time()
    f = Frames().latest()
    events.add(Event(t=1.0, wall=now + 1, obj="keys", type="MOVED", to_cm=(1.0, 2.0)), f)
    events.add(Event(t=2.0, wall=now + 2, obj="phone", type="MOVED", snapshot="/etc/passwd"))
    r = env["client"].get("/events", params={"since": now + 0.5})
    assert r.status_code == 200
    evs = r.json()
    assert [e["obj"] for e in evs] == ["keys", "phone"]
    assert evs[0]["to_cm"] == [1.0, 2.0]
    url = evs[0]["snapshot_url"]
    assert url.startswith("/snapshots/") and ".." not in url and url.count("/") == 2
    assert evs[1]["snapshot_url"] is None           # outside the snapshot dir: not exposed
    img = env["client"].get(url)
    assert img.status_code == 200 and img.headers["content-type"] == "image/jpeg"
    assert cv2.imdecode(np.frombuffer(img.content, np.uint8), cv2.IMREAD_COLOR) is not None
    # all demo events
    assert len(env["client"].get("/events", params={"since": 0}).json()) == 12


@pytest.mark.parametrize("path", [
    "/snapshots/..%2Fsecret.txt",
    "/snapshots/%2E%2E%2Fsecret.txt",
    "/snapshots/../secret.txt",
    "/snapshots/..%5Csecret.txt",
    "/snapshots/.hidden.jpg",
    "/snapshots/nope.jpg",
])
def test_snapshots_rejects_traversal(env, path):
    r = env["client"].get(path)
    assert r.status_code in (400, 404)
    assert "secret" not in r.text or r.status_code != 200


def test_a_thing_snapshot_with_a_colon_in_its_name_is_served(env):
    ev = Event(t=1.0, wall=time.time(), obj="thing:5210", type="APPEARED")
    env["events"].add(ev, Frame(t=1.0, wall=ev.wall, img=np.full((20, 30, 3), 90, np.uint8), idx=1))
    env["events"].flush()
    [e] = [e for e in env["client"].get("/events", params={"since": ev.wall - 1}).json() if e["obj"] == "thing:5210"]
    assert e["snapshot_url"].endswith("_thing:5210_APPEARED.jpg")
    r = env["client"].get(e["snapshot_url"])
    assert r.status_code == 200 and r.headers["content-type"] == "image/jpeg"


@pytest.mark.parametrize("path", ["/snapshots/thing:1%2F..%2Fsecret.txt", "/snapshots/a:..:b.jpg"])
def test_colons_open_no_way_out_of_the_snapshot_dir(env, path):
    r = env["client"].get(path)
    assert r.status_code in (400, 404)


def test_ask_returns_answer(env):
    r = env["client"].post("/ask", json={"text": "where are my keys?"})
    assert r.status_code == 200
    body = r.json()
    assert body["text"] == "The keys are inside the box <&>."
    assert body["point_at"] == "keys" and body["action"] == "point"
    assert isinstance(body["latency_ms"], int)
    assert env["calls"] == [("where are my keys?", "dashboard")]
    assert env["client"].post("/ask", json={"text": "  "}).status_code == 400


def test_ask_default_canned(tmp_path):
    cfg = load_config()
    events = EventLog(":memory:", str(tmp_path / "s"))
    app = create_app(cfg, demo_world(events), events)
    with TestClient(app) as c:
        r = c.post("/ask", json={"text": "Where are my car keys?"})
        assert r.status_code == 200
        assert "box" in r.json()["text"] and r.json()["point_at"] == "keys"


def _signed(params, url="http://testserver/sms"):
    return RequestValidator(TOKEN).compute_signature(url, params)


def _post_sms(client, params, sig):
    return client.post("/sms", content=urlencode(params),
                       headers={"Content-Type": "application/x-www-form-urlencoded",
                                "X-Twilio-Signature": sig})


def test_sms_rejects_bad_signature(env):
    params = {"From": GOOD, "Body": "where are my keys", "To": "+14045550000"}
    assert _post_sms(env["client"], params, "bogus").status_code == 403
    assert env["client"].post("/sms", data=params).status_code == 403    # no signature at all
    assert env["calls"] == []


def test_sms_ignores_non_whitelisted(env):
    params = {"From": "+19995550000", "Body": "where are my keys"}
    r = _post_sms(env["client"], params, _signed(params))
    assert r.status_code == 200
    assert "<Message>" not in r.text and "<Response></Response>" in r.text
    assert env["calls"] == []


def test_sms_answers_whitelisted_with_twiml(env):
    import xml.etree.ElementTree as ET
    params = {"From": GOOD, "Body": "where are my keys", "To": "+14045550000", "MessageSid": "SM1"}
    r = _post_sms(env["client"], params, _signed(params))
    assert r.status_code == 200
    assert "xml" in r.headers["content-type"]
    root = ET.fromstring(r.text)
    assert root.tag == "Response"
    assert root.find("Message").text == "The keys are inside the box <&>."   # escaped on the wire
    assert "&lt;&amp;&gt;" in r.text
    assert env["calls"] == [("where are my keys", "sms")]


def test_sms_insecure_mode(env, monkeypatch):
    monkeypatch.setenv("ASKROOM_SMS_INSECURE", "1")
    r = env["client"].post("/sms", data={"From": GOOD, "Body": "keys?"})
    assert r.status_code == 200 and "<Message>" in r.text


def test_video_streams_jpeg_frames(env):
    with env["client"].stream("GET", "/video", params={"frames_n": 2}) as r:
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("multipart/x-mixed-replace")
        data = b"".join(r.iter_bytes())
    assert data.count(b"Content-Type: image/jpeg") == 2
    jpg = data[data.index(b"\xff\xd8"):]
    jpg = jpg[:jpg.index(b"\xff\xd9") + 2]
    img = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
    assert img.shape[1] == 960


def test_video_placeholder_without_frames(tmp_path):
    cfg = load_config()
    events = EventLog(":memory:", str(tmp_path / "s"))
    app = create_app(cfg, demo_world(events), events, frames=None)
    with TestClient(app) as c:
        r = c.get("/frame.jpg")
        assert r.status_code == 200 and r.content[:2] == b"\xff\xd8"


def test_overlay_draw_with_and_without_table():
    w = demo_world()
    st = w.state_json()
    st["laser"] = {"on": True, "target": "keys", "err_cm": 0.8}
    img = np.zeros((720, 1280, 3), np.uint8)
    out = overlay.draw(img, st)
    assert out.shape == (540, 960, 3) and out.any()

    class Table:
        def cm_to_px(self, pts):
            return np.asarray(pts, float).reshape(-1, 2) * 10 + 100

    out2 = overlay.draw(img, st, table=Table())
    assert out2.shape == (540, 960, 3)
    assert not np.array_equal(out, out2)                   # positions + laser crosshair drawn
    assert overlay.draw(img, None).shape == (540, 960, 3)  # no state yet
    assert img.sum() == 0                                  # input untouched


def test_full_frame_view_with_zones_and_room_places(tmp_path):
    """Spec 0009: /full.jpg is the whole camera frame (room memory's TableView) with the zones drawn;
    404 when the frames source has no full frame."""
    from core.room_zones import Zone, Zones
    cfg = load_config()
    zp = tmp_path / "zones.json"
    Zones("v", (1920, 1080), {"couch": Zone("couch", "the couch", [(700, 800), (1000, 800), (1000, 1070)])}).save(zp)
    cfg = dict(cfg, room_memory=dict(cfg.get("room_memory") or {}, zones_path=str(zp)))
    events = EventLog(":memory:", str(tmp_path / "snaps"))

    class Full:
        rect = (0, 735, 613, 1080)

        def latest(self):
            return Frame(t=1.0, wall=1.0, img=np.zeros((720, 1280, 3), np.uint8), idx=1)

        def latest_full(self):
            return Frame(t=1.0, wall=1.0, img=np.zeros((1080, 1920, 3), np.uint8), idx=1)

    with TestClient(create_app(cfg, demo_world(events), events, frames=Full())) as c:
        r = c.get("/full.jpg")
        assert r.status_code == 200 and r.headers["content-type"] == "image/jpeg"
        img = cv2.imdecode(np.frombuffer(r.content, np.uint8), cv2.IMREAD_COLOR)
        assert img.shape[1] == 1280 and img.sum() > 0          # downscaled, zones drawn on black
        raw = cv2.imdecode(np.frombuffer(c.get("/full.jpg?raw=1").content, np.uint8), cv2.IMREAD_COLOR)
        assert raw.shape[:2] == (1080, 1920) and raw.max() < 8   # capture size, nothing drawn
        view = cv2.imdecode(np.frombuffer(c.get("/frame.jpg?raw=1").content, np.uint8), cv2.IMREAD_COLOR)
        assert view.shape[:2] == (720, 1280) and view.max() < 8
    with TestClient(create_app(cfg, demo_world(events), events, frames=None)) as c:
        assert c.get("/full.jpg").status_code == 404
        assert c.get("/full.jpg?raw=1").status_code == 404 and c.get("/frame.jpg?raw=1").status_code == 404
        assert c.get("/frame.jpg").status_code == 200           # the placeholder, drawn on


def test_voice_sets_the_rigs_voice_and_503_without_a_speaker(tmp_path):
    from voice.tts import VoiceChoice
    cfg = load_config()
    events = EventLog(str(tmp_path / "e.db"), str(tmp_path / "snaps"))
    got = []

    def voice_fn(engine, grok_voice, speed):
        got.append((engine, grok_voice, speed))
        return VoiceChoice.make(engine, grok_voice, speed)

    with TestClient(create_app(cfg, demo_world(events), events, voice_fn=voice_fn)) as client:
        r = client.post("/voice", json={"engine": "grok", "grok_voice": "Ara", "speed": 1.2})
        assert r.status_code == 200 and r.json() == {"engine": "grok", "grok_voice": "ara", "speed": 1.2}
        assert got == [("grok", "Ara", 1.2)]
        assert client.post("/voice", content=b"nope").status_code == 400
    with TestClient(create_app(cfg, demo_world(events), events)) as client:
        assert client.post("/voice", json={"engine": "grok"}).status_code == 503


# -- the user's seat (core/viewframe.py): GET /state "view", POST /orientation, the saved choice

def _seat_cfg(tmp_path, front="bottom"):
    cfg = load_config()
    cfg["paths"] = dict(cfg["paths"], viewer=str(tmp_path / "viewer.json"))
    cfg["table"] = dict(cfg["table"], size_cm=[100, 60])
    cfg["table_area"] = dict(cfg.get("table_area") or {}, polygon_cm=[])
    cfg["viewer"] = {"front": front, "sides": {}}
    return cfg


def test_state_carries_the_view_and_orientation_turns_and_saves_it(tmp_path):
    import json
    cfg = _seat_cfg(tmp_path)
    events = EventLog(":memory:", str(tmp_path / "s"))
    with TestClient(create_app(cfg, demo_world(events), events)) as c:
        body = c.get("/state").json()
        assert body["view"] == {"front": "bottom", "table": [100.0, 60.0],
                                "m": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], "outline": False}
        assert "sides" not in body
        with c.websocket_connect("/ws") as ws:
            assert ws.receive_json()["view"]["front"] == "bottom"
        r = c.post("/orientation", json={"front": "Right"})
        assert r.status_code == 200
        v = r.json()
        assert v["front"] == "right" and v["table"] == [60.0, 100.0] and v["outline"] is False
        (a, b, cc), (d, e, f) = v["m"]
        assert (a * 90 + b * 5 + cc, d * 90 + e * 5 + f) == pytest.approx((55, 90))   # the camera's top right
        assert cfg["viewer"]["front"] == "right" and c.get("/state").json()["view"] == v
        assert json.loads((tmp_path / "viewer.json").read_text())["front"] == "right"


@pytest.mark.parametrize("body", [{"front": "sideways"}, {"front": 3}, {}, [1], "right"])
def test_orientation_rejects_a_bad_front(tmp_path, body):
    cfg = _seat_cfg(tmp_path)
    events = EventLog(":memory:", str(tmp_path / "s"))
    with TestClient(create_app(cfg, demo_world(events), events)) as c:
        assert c.post("/orientation", json=body).status_code == 400
        assert c.post("/orientation", content=b"not json").status_code == 400
        assert cfg["viewer"]["front"] == "bottom" and not (tmp_path / "viewer.json").exists()


def test_orientation_null_goes_back_to_the_configured_seat(tmp_path):
    """The phone's "use the rig's default": front null restores config viewer.front and forgets the saved seat."""
    (tmp_path / "viewer.json").write_text('{"front": "top", "t": 1}')
    cfg = _seat_cfg(tmp_path, front="right")
    events = EventLog(":memory:", str(tmp_path / "s"))
    with TestClient(create_app(cfg, demo_world(events), events)) as c:
        assert c.get("/state").json()["view"]["front"] == "top"
        r = c.post("/orientation", json={"front": None})
        assert r.status_code == 200 and r.json()["front"] == "right"
        assert cfg["viewer"]["front"] == "right" and not (tmp_path / "viewer.json").exists()


def test_create_app_applies_a_saved_seat_and_state_carries_sides(tmp_path):
    (tmp_path / "viewer.json").write_text('{"front": "top", "t": 1}')
    cfg = _seat_cfg(tmp_path)
    cfg["viewer"]["sides"] = {"right": "couch"}
    events = EventLog(":memory:", str(tmp_path / "s"))
    with TestClient(create_app(cfg, demo_world(events), events)) as c:
        assert cfg["viewer"]["front"] == "top"
        body = c.get("/state").json()
        assert body["view"]["front"] == "top" and body["sides"] == {"right": "couch"}


def test_canned_gone_answer_is_worded_from_the_seat_and_sweeps_the_camera_edge(tmp_path):
    cfg = _seat_cfg(tmp_path)
    events = EventLog(":memory:", str(tmp_path / "s"))
    world = demo_world(events)
    assert world.get("phone").edge == "left"                     # demo_world: the phone left by the camera's left
    with TestClient(create_app(cfg, world, events)) as c:
        r = c.post("/ask", json={"text": "where is my phone?"}).json()
        assert r["text"] == "The phone was carried off the table on your left." and r["action"] == "sweep:left"
        c.post("/orientation", json={"front": "right"})
        r = c.post("/ask", json={"text": "where is my phone?"}).json()
        assert r["text"] == "The phone was carried off the far side of the table." and r["action"] == "sweep:left"


# -- the demo view (WS9): /demo, /demo/meta, /demo/boxes, /demo/evidence, /room_layout, ?w=, listening

class TableStub:
    Hinv = np.diag([10.0, 10.0, 1.0])            # 10 table-view px per cm, origin at the view's corner


class FullView:
    """room memory's TableView: the table view is the 817 x 460 region at (0, 980) of a 2560 x 1440 frame."""
    rect, out_size = (0, 980, 817, 1440), (1280, 720)

    def latest(self):
        return Frame(t=1.0, wall=1.0, img=np.zeros((720, 1280, 3), np.uint8), idx=1)

    def latest_full(self):
        return Frame(t=1.0, wall=1.0, img=np.full((1440, 2560, 3), 60, np.uint8), idx=1)


def _demo_app(tmp_path, **kw):
    cfg = load_config()
    cfg["paths"] = dict(cfg["paths"], viewer=str(tmp_path / "viewer.json"))
    cfg["room_memory"] = dict(cfg.get("room_memory") or {}, enabled=True, capture_size=[2560, 1440],
                              zones_path=str(tmp_path / "none.json"))
    events = EventLog(str(tmp_path / "e.db"), str(tmp_path / "snaps"))
    world = demo_world(events)
    return cfg, events, world, create_app(cfg, world, events, frames=FullView(), table=TableStub(), **kw)


def test_demo_page_is_served_offline(env):
    r = env["client"].get("/demo")
    assert r.status_code == 200 and "text/html" in r.headers["content-type"]
    for asset in ("/static/demo.js", "/static/demo.css"):
        assert asset in r.text and env["client"].get(asset).status_code == 200, asset
    assert "googleapis" not in r.text and "cdn" not in r.text.lower()
    assert 'src="http' not in r.text and 'href="http' not in r.text              # nothing from off the rig
    js = env["client"].get("/static/demo.js").text
    assert "http://" not in js and "https://" not in js


def test_demo_meta_and_boxes_put_table_cm_on_the_full_frame(tmp_path):
    cfg, events, world, app = _demo_app(tmp_path)
    world.get("wallet").box_cm = (10.0, 10.0, 20.0, 20.0)
    with TestClient(app) as c:
        meta = c.get("/demo/meta").json()
        assert meta["full"] is True and meta["image"] == "/full.jpg?raw=1" and meta["image_size"] == [2560, 1440]
        assert meta["table_view_rect"] == [0, 980, 817, 1440] and meta["view"]["front"] in ("bottom", "right", "top", "left")
        M = np.asarray(meta["cm_to_img"])
        p = M @ np.array([10.0, 10.0, 1.0])
        assert p[:2] / p[2] == pytest.approx([100 * 817 / 1280, 980 + 100 * 460 / 720])
        boxes = c.get("/demo/boxes").json()["boxes"]
        assert boxes["wallet"] == pytest.approx([63.8, 1043.9, 127.7, 1107.8], abs=0.1)
    with TestClient(create_app(cfg, world, events, frames=Frames(), table=TableStub())) as c:
        meta = c.get("/demo/meta").json()           # no full frame: the table view itself, Hinv as it is
        assert meta["full"] is False and meta["image"] == "/frame.jpg?raw=1" and meta["image_size"] == [1280, 720]
        assert c.get("/demo/boxes").json()["boxes"]["wallet"] == [100.0, 100.0, 200.0, 200.0]


def test_demo_evidence_is_the_newest_event_with_a_snapshot(tmp_path):
    cfg, events, world, app = _demo_app(tmp_path)
    snaps = tmp_path / "snaps"
    snaps.mkdir(exist_ok=True)
    (snaps / "111_wallet_MOVED.jpg").write_bytes(b"\xff\xd8jpeg")
    events.add(Event(t=1.0, wall=111.0, obj="wallet", type="MOVED", snapshot=str(snaps / "111_wallet_MOVED.jpg")))
    events.add(Event(t=2.0, wall=222.0, obj="wallet", type="LOST_TRACK"))              # no picture: skipped
    with TestClient(app) as c:
        got = c.get("/demo/evidence?obj=wallet").json()
        assert got == {"obj": "wallet", "type": "MOVED", "t": 111.0, "snapshot_url": "/snapshots/111_wallet_MOVED.jpg"}
        assert c.get("/demo/evidence?obj=nothing").json() == {} and c.get("/demo/evidence").json() == {}


def test_room_layout_is_turned_to_the_seat_and_404_without_room_memory(tmp_path):
    cfg, events, world, app = _demo_app(tmp_path)
    cfg["table"] = dict(cfg["table"], size_cm=[100, 70])
    cfg["table_area"] = dict(cfg.get("table_area") or {}, polygon_cm=[])
    with TestClient(app) as c:
        lay = {}
        for front in ("bottom", "right"):
            assert c.post("/orientation", json={"front": front}).status_code == 200
            lay[front] = c.get("/room_layout").json()
        for front, got in lay.items():
            couch = next(z for z in got["zones"] if z["id"] == "couch")
            x, y, w, h = couch["rect"]
            tx, ty, tw, th = got["table"]["rect"]
            assert got["front"] == front and got["v"] == 1
            assert all(v >= 0 for z in got["zones"] for v in z["rect"]) and got["table"]["origin"] == [tx, ty]
            if front == "right":                     # the couch is the seat's side: nearest the user, below the table
                assert y >= ty + th - 1 and x <= got["you"][0] <= x + w and y <= got["you"][1] <= y + h
                assert (tw, th) == (70, 100)
            else:                                    # from the camera's side the couch is on the table's right
                assert x >= tx + tw - 1 and (tw, th) == (100, 70)
    cfg["room_memory"] = dict(cfg["room_memory"], enabled=False)
    with TestClient(create_app(cfg, world, events, frames=FullView())) as c:
        assert c.get("/room_layout").status_code == 404


def test_full_raw_frame_shrinks_to_w_and_listening_is_in_state(tmp_path):
    lit = {"on": True}
    cfg, events, world, app = _demo_app(tmp_path, listening_fn=lambda: lit["on"])
    with TestClient(app) as c:
        img = cv2.imdecode(np.frombuffer(c.get("/full.jpg?raw=1&w=1600").content, np.uint8), cv2.IMREAD_COLOR)
        assert img.shape[:2] == (900, 1600)
        full = cv2.imdecode(np.frombuffer(c.get("/full.jpg?raw=1").content, np.uint8), cv2.IMREAD_COLOR)
        assert full.shape[:2] == (1440, 2560)
        assert c.get("/state").json()["listening"] is True
        lit["on"] = False
        assert c.get("/state").json()["listening"] is False
    with TestClient(create_app(cfg, world, events, frames=FullView())) as c:
        assert "listening" not in c.get("/state").json()
