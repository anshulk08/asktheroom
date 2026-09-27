"""core/grounding.py (Moondream 3.1 /detect and /point, flag-guarded) and the WHERE fallback in
voice/visual.py route(): box scaling, zone mapping, the deadline, no key -> no call, 429 backoff, retries,
caps, the Cloudflare fallback, the WS8 refind adapter, and which WHERE questions reach the grounder.
The HTTP layer is a fake session: nothing here touches the network."""
import base64
import json
import os
import time

import cv2
import numpy as np
import pytest
import requests

from core import grounding
from core.grounding import GroundingConfig, MoondreamGrounder, crop_box, find_anywhere, place_of
from core.room_types import Place
from core.types import Answer, Entity, Status
from tests.test_room_look import RoomFrames, room_seen_reply, zoned_qa
from tests.test_visual import CFG, T0, log  # noqa: F401  (log is a fixture)
from voice.intents import parse

KEY = {"MOONDREAM_API_KEY": "md-test-key"}


class Resp:
    def __init__(self, status=200, body=None, headers=None):
        self.status_code, self._body, self.headers = status, body if body is not None else {}, headers or {}
        self.text = json.dumps(self._body)

    def json(self):
        return self._body


class Session:
    """requests.Session's post, replying from a list in order (the last repeats); exceptions are raised."""
    def __init__(self, *replies):
        self.replies, self.posts = list(replies) or [Resp(body={"objects": []})], []

    def post(self, url, headers=None, json=None, timeout=None):
        self.posts.append({"url": url, "headers": headers, "json": json, "timeout": timeout})
        r = self.replies[min(len(self.posts) - 1, len(self.replies) - 1)]
        if isinstance(r, Exception):
            raise r
        return r


class Clock:
    def __init__(self, t=1000.0):
        self.t = t

    def __call__(self):
        return self.t


def grounder(*replies, env=KEY, clock=None, **kw):
    c = GroundingConfig(**{"enabled": True, "min_interval_s": 0.0, **kw})
    s = Session(*replies)
    clk = clock or Clock()
    slept = []

    def sleep(dt):
        slept.append(dt)
        clk.t += dt

    g = MoondreamGrounder(c, session=s, clock=clk, sleep=sleep, env=dict(env))
    return g, s, slept


def frame(w=2560, h=1440):
    return np.full((h, w, 3), 90, np.uint8)


def sent_size(post):
    url = post["json"].get("image_url") or post["json"].get("image")
    raw = base64.b64decode(url.split(",", 1)[1])
    im = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    return im.shape[1], im.shape[0]


# ---------------------------------------------------------------- the client

def test_detect_sends_the_model_and_scales_normalized_boxes_to_the_callers_image():
    g, s, _ = grounder(Resp(body={"request_id": "r1", "objects": [{"x_min": 0.25, "y_min": 0.5, "x_max": 0.3,
                                                                    "y_max": 0.6}]}))
    [box] = g.detect(frame(), "red coffee mug")
    assert box[:4] == pytest.approx((640.0, 720.0, 768.0, 864.0)) and box[4] == 0.8    # px of the 2560x1440 frame
    [p] = s.posts
    assert p["url"] == "https://api.moondream.ai/v1/detect"
    assert p["json"]["model"] == "moondream3.1-9B-A2B" and p["json"]["object"] == "red coffee mug"
    assert p["json"]["image_url"].startswith("data:image/jpeg;base64,")
    assert sent_size(p) == (1792, 1008)                     # shrunk before upload; the boxes don't care
    assert p["headers"]["X-Moondream-Auth"] == "md-test-key"
    assert "urllib" not in p["headers"]["User-Agent"] and p["timeout"] == 3.0


def test_point_returns_centres_in_image_px():
    g, s, _ = grounder(Resp(body={"points": [{"x": 0.5, "y": 0.25}, {"x": "bad"}]}))
    assert g.point(frame(1000, 800), "keys") == [(500.0, 200.0)]
    assert s.posts[0]["url"].endswith("/point")


def test_furniture_sized_and_malformed_boxes_are_dropped():
    g, _, _ = grounder(Resp(body={"objects": [{"x_min": 0, "y_min": 0, "x_max": 0.9, "y_max": 0.9},
                                              {"x_min": 0.5, "y_min": 0.5, "x_max": 0.4, "y_max": 0.6},
                                              {"x_min": 0.1, "y_min": 0.1, "x_max": 0.12, "y_max": 0.13}]}))
    assert [tuple(round(v) for v in b[:4]) for b in g.detect(frame(1000, 1000), "keys")] == [(100, 100, 120, 130)]


def test_no_key_means_no_call_and_no_exception(caplog):
    g, s, _ = grounder(env={})
    with caplog.at_level("INFO"):
        assert g.detect(frame(), "keys") == [] and g.point(frame(), "keys") == []
    assert s.posts == [] and "MOONDREAM_API_KEY" in caplog.text


def test_disabled_means_no_call():
    g, s, _ = grounder(enabled=False)
    assert g.detect(frame(), "keys") == [] and s.posts == []


def test_offline_is_nothing_found_never_an_exception():
    g, s, _ = grounder(requests.ConnectionError("no route"))
    assert g.detect(frame(), "keys") == []
    assert len(s.posts) == 2                                # one retry


def test_a_timeout_is_retried_once_with_backoff():
    g, s, slept = grounder(requests.Timeout("slow"), Resp(body={"objects": [{"x_min": 0.1, "y_min": 0.1,
                                                                              "x_max": 0.2, "y_max": 0.2}]}))
    assert len(g.detect(frame(), "keys")) == 1 and len(s.posts) == 2 and slept == [0.3]


def test_a_429_backs_off_every_call_until_it_passes():
    clk = Clock()
    g, s, _ = grounder(Resp(429, {"error": "Too Many Requests"}), Resp(body={"objects": []}), clock=clk,
                       backoff_429_s=10.0)
    assert g.detect(frame(), "keys") == [] and len(s.posts) == 1       # no retry into a 429
    assert g.detect(frame(), "keys") == [] and len(s.posts) == 1       # blocked: no request at all
    clk.t += 11
    g.detect(frame(), "keys")
    assert len(s.posts) == 2


def test_retry_after_longer_than_the_default_wins():
    clk = Clock()
    g, s, _ = grounder(Resp(429, {}, headers={"Retry-After": "30"}), Resp(body={"objects": []}), clock=clk)
    g.detect(frame(), "keys")
    clk.t += 15
    g.detect(frame(), "keys")
    assert len(s.posts) == 1


def test_a_bad_key_is_not_hammered():
    clk = Clock()
    g, s, _ = grounder(Resp(401, {"error": "Invalid API key"}), clock=clk)
    for _ in range(3):
        assert g.detect(frame(), "keys") == []
    assert len(s.posts) == 1


def test_the_per_minute_cap_stops_calls():
    g, s, _ = grounder(max_per_minute=2)
    for _ in range(4):
        g.detect(frame(), "keys")
    assert len(s.posts) == 2


def test_calls_are_spaced_to_the_rate_limit():
    clk = Clock()
    g, s, slept = grounder(clock=clk, min_interval_s=0.5)
    g.detect(frame(), "keys")
    g.detect(frame(), "keys")
    assert len(s.posts) == 2 and slept == [pytest.approx(0.5)]


def test_cloudflare_is_the_fallback_when_moondream_fails():
    env = {**KEY, "CF_ACCOUNT_ID": "acct", "CF_API_TOKEN": "cf-token"}
    g, s, _ = grounder(Resp(500, {}), Resp(500, {}),
                       Resp(body={"result": {"objects": [{"x_min": 0.5, "y_min": 0.5, "x_max": 0.6, "y_max": 0.6}],
                                             "points": None}, "success": True, "errors": []}), env=env)
    [b] = g.detect(frame(1000, 1000), "keys")
    assert b[:4] == pytest.approx((500, 500, 600, 600))
    cf = s.posts[-1]
    assert "api.cloudflare.com/client/v4/accounts/acct/ai/run/@cf/moondream/moondream3.1-9B-A2B" in cf["url"]
    assert cf["json"]["task"] == "detect" and cf["json"]["target"] == "keys"
    assert cf["headers"]["Authorization"] == "Bearer cf-token"


def test_an_empty_moondream_answer_is_final():
    env = {**KEY, "CF_ACCOUNT_ID": "acct", "CF_API_TOKEN": "cf-token"}
    g, s, _ = grounder(Resp(body={"objects": []}), env=env)
    assert g.detect(frame(), "keys") == [] and len(s.posts) == 1


def test_calls_from_many_threads_respect_the_cap():
    import threading
    g, s, _ = grounder(max_per_minute=5)
    ths = [threading.Thread(target=g.detect, args=(frame(200, 200), "keys")) for _ in range(20)]
    for t in ths:
        t.start()
    for t in ths:
        t.join()
    assert len(s.posts) == 5


# ---------------------------------------------------------------- places

Z = [("couch", "the couch", [(973, 1053), (1320, 1053), (1320, 1440), (906, 1440)]),
     ("side_table", "the side table", [(0, 740), (287, 740), (287, 867), (0, 867)]),
     ("doorway", "the doorway", [(1500, 300), (1700, 300), (1700, 900), (1500, 900)])]
RECT = (0, 980, 817, 1440)


def test_a_box_standing_in_a_zone_is_on_that_zone():
    assert place_of((1050, 1000, 1150, 1100), Z, RECT) == ("couch", "on the couch")       # footprint (1100, 1100)
    assert place_of((100, 700, 160, 800), Z, RECT) == ("side_table", "on the side table")
    assert place_of((1550, 600, 1600, 700), Z, RECT) == ("doorway", "by the doorway")


def test_a_box_on_the_table_view_is_on_the_table():
    assert place_of((300, 1100, 400, 1200), Z, RECT) == ("table", "on the table")


def test_elsewhere_is_near_a_zone_or_a_rough_region():
    assert place_of((1720, 800, 1780, 880), Z, RECT) == ("room", "near the doorway")
    assert place_of((2300, 300, 2350, 350), Z, RECT) == ("room", "somewhere on the right side of the room")
    assert place_of((1100, 100, 1150, 150), Z, RECT) == ("room", "somewhere on the far side of the room")


def test_zone_close_ups_are_padded_raised_and_at_least_a_tile():
    x1, y1, x2, y2 = crop_box([(0, 740), (287, 740), (287, 867), (0, 867)], (2560, 1440))
    assert x1 == 0 and x2 >= 287 * 1.15 - 1 and y1 < 740 - 0.5 * 127 and y2 > 867
    assert x2 - x1 >= 384 and y2 - y1 >= 384


class FakeGrounder:
    """detect(img, phrase) -> boxes by call number (image px); records the image sizes it was given."""
    def __init__(self, *answers, delay=0.0, **kw):
        self.c = GroundingConfig(enabled=True, **kw)
        self.answers, self.calls, self.delay = list(answers) or [[]], [], delay
        self.ok, self.confirmed = True, []

    def detect(self, img, phrase, deadline=None):
        self.calls.append((img.shape[1], img.shape[0], phrase))
        if self.delay:
            time.sleep(self.delay)
        return self.answers[min(len(self.calls) - 1, len(self.answers) - 1)]

    def point(self, img, phrase, deadline=None):
        return []

    def confirm(self, img, box, phrase, deadline=None):
        self.confirmed.append(tuple(round(v) for v in box[:4]))
        return self.ok


def test_find_anywhere_hits_the_full_frame_first():
    g = FakeGrounder([(1050, 1000, 1150, 1100, 0.8)])
    f = find_anywhere(g, frame(), "keys", Z, RECT)
    assert f.place == "couch" and f.where == "on the couch" and f.source == "full" and g.calls == [(2560, 1440, "keys")]


def test_a_miss_on_the_full_frame_searches_zone_close_ups_and_maps_back():
    g = FakeGrounder([], [(10, 150, 40, 170, 0.8)])      # found in the first (smallest) close-up
    f = find_anywhere(g, frame(), "keys", Z, RECT)
    x1, y1, _, _ = crop_box(Z[1][2], (2560, 1440))
    assert f.source == "zone:side_table" and f.box == (10 + x1, 150 + y1, 40 + x1, 170 + y1)
    assert f.place == "side_table" and g.calls[1][:2] != (2560, 1440)


def test_an_unconfirmed_box_is_never_said():
    g = FakeGrounder([(1050, 1000, 1150, 1100, 0.8), (100, 700, 160, 800, 0.8), (5, 5, 9, 9, 0.8)], [])
    g.ok = False
    assert find_anywhere(g, frame(), "umbrella", Z, RECT) is None
    assert g.confirmed[:2] == [(1050, 1000, 1150, 1100), (100, 700, 160, 800)]      # at most verify_boxes a search
    g2 = FakeGrounder([(1050, 1000, 1150, 1100, 0.8)], verify=False)
    g2.ok = False
    assert find_anywhere(g2, frame(), "keys", Z, RECT).place == "couch" and g2.confirmed == []


def test_confirm_asks_a_closed_query_on_the_boxs_close_up():
    g, s, _ = grounder(Resp(body={"answer": "No."}), Resp(body={"answer": "Yes"}))
    assert g.confirm(frame(), (1000, 600, 1100, 700), "umbrella") is False
    assert g.confirm(frame(), (1000, 600, 1100, 700), "keys") is True
    q = s.posts[0]
    assert q["url"].endswith("/query") and q["json"]["model"] == "moondream3.1-9B-A2B"
    assert q["json"]["question"] == "Is the object in the middle of this picture an umbrella? Answer yes or no."
    assert sent_size(q) == (300, 300)                                                # the box padded by its size
    assert "picture keys?" in s.posts[1]["json"]["question"]


def test_a_failed_confirm_is_a_no():
    g, s, _ = grounder(requests.ConnectionError("down"))
    assert g.confirm(frame(), (1000, 600, 1100, 700), "keys") is False


def test_find_anywhere_misses_after_every_close_up():
    g = FakeGrounder([])
    assert find_anywhere(g, frame(), "keys", Z, RECT) is None and len(g.calls) == 1 + 3


def test_find_anywhere_reads_a_table_view_source():
    g = FakeGrounder([(1050, 1000, 1150, 1100, 0.8)])
    assert find_anywhere(g, RoomFrames(), "keys", [], None).place == "room"
    assert g.calls[0][:2] == (1920, 1080)


# ---------------------------------------------------------------- the WS8 refind adapter

def test_refind_adapter_detects_on_the_view_and_returns_full_frame_boxes():
    g = FakeGrounder([(10, 20, 30, 40, 0.8)])
    grounding.configure(None, g)
    try:
        assert grounding.refind.needs_suspects is False
        out = grounding.refind("pill bottle", [], frame(640, 480), (1000, 500, 1640, 980), [])
        assert out == [((1010.0, 520.0, 1030.0, 540.0), 0.8, "moondream")]
        assert g.calls == [(640, 480, "pill bottle")]
    finally:
        grounding.configure(None, None)
        grounding._default = None


def test_from_config_is_off_by_default():
    assert grounding.from_config(CFG) is None
    assert grounding.from_config({"grounding": {"enabled": True}}) is not None


# ---------------------------------------------------------------- the WHERE fallback in route()

COUCH_BOX = (800, 800, 900, 900, 0.8)          # footprint (850, 900): inside ZONES' couch (1920x1080)


def ask(q, text, online=True):
    return q.route(parse(text, CFG), text, online=online)


def glasses_world(log, **fields):
    from core.fakeworld import FakeWorld
    return FakeWorld([Entity("glasses", "target", **fields),
                      Entity("keys", "target", Status.VISIBLE, pos_cm=(20.0, 15.0), box_cm=(17.0, 13.0, 23.0, 17.0),
                             last_seen=T0)], log)


def grounded(log, tmp_path, *answers, world=None, reply=None, **kw):
    q, prov = zoned_qa(log, tmp_path, reply or room_seen_reply("I don't see them."),
                       **({"world": world} if world is not None else {}))
    g = FakeGrounder(*answers, **kw)
    q.grounder = g
    return q, prov, g


def test_a_fresh_visible_thing_never_calls_the_grounder(log, tmp_path):
    q, prov, g = grounded(log, tmp_path, [COUCH_BOX])
    assert ask(q, "where are my keys?") is None                              # the WHERE template answers
    assert g.calls == [] and prov.calls == []


def test_a_thing_never_seen_is_grounded_and_said_hedged_on_its_zone(log, tmp_path):
    q, prov, g = grounded(log, tmp_path, [COUCH_BOX], world=glasses_world(log))
    a = ask(q, "where are my glasses?")
    assert a == Answer("Your glasses look like they're on the couch.")      # spoken only: no laser
    assert g.calls == [(1920, 1080, "glasses")] and prov.calls == []         # no Grok call after a hit


def test_a_lost_thing_is_grounded(log, tmp_path):
    q, prov, g = grounded(log, tmp_path, [COUCH_BOX],
                          world=glasses_world(log, pos_cm=(10.0, 10.0), last_seen=T0 - 600))    # UNKNOWN, seen before
    assert ask(q, "where are my glasses?").text == "Your glasses look like they're on the couch."


def test_a_miss_falls_through_to_the_grok_room_look(log, tmp_path):
    q, prov, g = grounded(log, tmp_path, [], world=glasses_world(log))
    a = ask(q, "where are my glasses?")
    assert len(g.calls) == 1 + 2 and a.text == "I don't see them."           # full frame + both zones, then Grok
    assert len(prov.calls) == 1 and "room" in prov.calls[0].system.lower()


def test_a_miss_on_a_placed_thing_falls_through_to_the_templates(log, tmp_path):
    q, prov, g = grounded(log, tmp_path, [],
                          world=glasses_world(log, status=Status.GONE, pos_cm=(10.0, 10.0), last_seen=T0, edge="left"))
    assert ask(q, "where are my glasses?") is None and len(g.calls) >= 1 and prov.calls == []


def test_a_name_the_world_doesnt_know_is_grounded(log, tmp_path):
    q, prov, g = grounded(log, tmp_path, [COUCH_BOX])
    a = ask(q, "where is my red mug?")
    assert a == Answer("Your red mug looks like it's on the couch.") and g.calls[0][2] == "red mug"


def test_an_unknown_name_found_on_the_table_goes_on_to_the_pick(log, tmp_path):
    q, prov, g = grounded(log, tmp_path, [(100, 900, 200, 1000, 0.8)],
                          reply=json.dumps({"mark": None, "label": None, "confidence": 0.9}))
    q.frames.rect = (0, 820, 600, 1080)
    ask(q, "where is my red mug?")
    assert len(g.calls) == 1 and len(prov.calls) >= 1                        # Grok's table pick, which can point


def test_the_deadline_bounds_a_slow_grounder(log, tmp_path):
    q, prov, g = grounded(log, tmp_path, [COUCH_BOX], world=glasses_world(log), delay=2.0, where_deadline_s=0.3)
    t0 = time.perf_counter()
    a = ask(q, "where are my glasses?")
    assert time.perf_counter() - t0 < 1.5
    assert a.text == "I don't see them."                                     # fell through to the room look


def test_offline_the_grounder_is_not_called(log, tmp_path):
    q, prov, g = grounded(log, tmp_path, [COUCH_BOX], world=glasses_world(log))
    ask(q, "where are my glasses?", online=False)
    assert g.calls == []


def test_a_table_question_is_not_grounded(log, tmp_path):
    q, prov, g = grounded(log, tmp_path, [COUCH_BOX], reply=json.dumps({"mark": None, "label": None, "confidence": 0.9}))
    ask(q, "is my mug on the table?")
    assert g.calls == []


def test_a_fresh_room_place_is_not_grounded_but_a_stale_one_is(log, tmp_path):
    def room(fresh, absent=False):
        return Place(kind="room", zone="couch", say="the couch", status=Status.VISIBLE, chain=["glasses"],
                     via="glasses", box_px=(1, 1, 2, 2), fresh=fresh, absent=absent, last_seen_wall=T0)
    w = glasses_world(log, status=Status.VISIBLE, zone="couch", last_seen=T0)
    q, prov, g = grounded(log, tmp_path, [COUCH_BOX], world=w)
    w.set_place("glasses", room(fresh=True))
    assert ask(q, "where are my glasses?") is None and g.calls == []
    w.set_place("glasses", room(fresh=False))
    assert ask(q, "where are my glasses?").text == "Your glasses look like they're on the couch."


def test_a_thing_hidden_in_a_known_container_is_not_grounded(log, tmp_path):
    w = glasses_world(log, status=Status.INSIDE, parent="keys", pos_cm=(20.0, 15.0), last_seen=T0)
    q, prov, g = grounded(log, tmp_path, [COUCH_BOX], world=w)
    ask(q, "where are my glasses?")
    assert g.calls == []


def test_a_prop_is_looked_for_by_its_detector_prompt(log, tmp_path):
    from core.fakeworld import FakeWorld
    q, prov, g = grounded(log, tmp_path, [COUCH_BOX], world=FakeWorld([Entity("remote", "target")], log))
    assert ask(q, "where is the remote?").text == "Your remote looks like it's on the couch."
    assert g.calls[0][2] == "remote control"


def test_grounding_on_is_disclosed(log, tmp_path):
    q, prov, g = grounded(log, tmp_path, [])
    assert "Moondream" in q.status()["disclosure"]
    q.grounder = None
    assert "Moondream" not in q.status()["disclosure"]


# ---------------------------------------------------------------- eval/grounding_bench.py

def test_bench_scoring_rules():
    from eval.grounding_bench import grok_score, is_hit
    assert is_hit((100, 100, 200, 200), (110, 110, 210, 210))                 # IoU ~0.68
    assert is_hit((0, 0, 1000, 1000), (490, 490, 520, 520))                   # centre in the true box
    assert not is_hit((0, 0, 10, 10), (500, 500, 520, 520)) and not is_hit(None, (0, 0, 1, 1))
    zs = [("couch", "the couch", []), ("side_table", "the side table", [])]
    assert grok_score("Your keys are on the couch.", (1, 1, 2, 2), "couch", zs) == (True, False)
    assert grok_score("I don't see your keys.", (1, 1, 2, 2), "couch", zs) == (False, False)
    assert grok_score("Your keys are on the side table.", (1, 1, 2, 2), "table", zs) == (False, False)
    assert grok_score("Your keys are on the couch.", None, None, zs) == (False, True)
    assert grok_score("I don't see any keys.", None, None, zs) == (False, False)


def test_bench_runs_end_to_end_on_a_fake_api(tmp_path, monkeypatch, capsys):
    from eval import grounding_bench
    monkeypatch.chdir(os.getcwd())                     # main() changes to the repo root; put it back after
    cv2.imwrite(str(tmp_path / "f1.jpg"), frame())
    (tmp_path / "gt.json").write_text(json.dumps({"f1.jpg": {"keys": [640, 720, 768, 864], "umbrella": None}}))
    (tmp_path / "room_zones.json").write_text(json.dumps({"view": "t", "size_px": [2560, 1440], "zones": {}}))
    (tmp_path / "md.env").write_text("MOONDREAM_API_KEY=md-test-key\n")
    monkeypatch.delenv("MOONDREAM_API_KEY", raising=False)
    monkeypatch.setenv("ASKROOM_NO_LOCAL_CONFIG", "1")
    replies = {"keys": {"objects": [{"x_min": 0.25, "y_min": 0.5, "x_max": 0.3, "y_max": 0.6}]},
               "umbrella": {"objects": [{"x_min": 0.1, "y_min": 0.1, "x_max": 0.12, "y_max": 0.12}]}}

    def ask(self, task, img, phrase, deadline):
        if task == "query":                    # the umbrella's box is something else
            return {"answer": "no" if "umbrella" in phrase else "yes"}
        return replies[phrase]
    monkeypatch.setattr(MoondreamGrounder, "_ask", ask)
    out = tmp_path / "r.json"
    assert grounding_bench.main(["--frames", str(tmp_path), "--env-file", str(tmp_path / "md.env"),
                                 "--zones", str(tmp_path / "room_zones.json"), "--out", str(out)]) == 0
    text = capsys.readouterr().out
    assert "md-test-key" not in text and "hit 1/1" in text and "false positives 0/1" in text
    rows = json.loads(out.read_text())
    assert [(r["name"], r["hit"], r["fp"]) for r in rows] == [("keys", True, False), ("umbrella", False, False)]
    assert rows[1]["full_fp"] is True                          # /detect alone boxed it; the /query check dropped it
