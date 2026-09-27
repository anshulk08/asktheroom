"""core/grok_trace.py: every Grok call through core/xai.Client is kept (purpose, model, latency, request
text, thumbnails of the images sent, the reply) for the demo's "Grok's eyes"; never the key or headers."""
import base64
import json

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from core import grok_trace, xai
from core.auto_name import NAME_SYSTEM
from core.config import load_config
from core.events import EventLog
from core.fakeworld import demo_world
from server.app import create_app

KEY = "xai-SECRET-key-123"


def data_url(w=1280, h=720):
    ok, buf = cv2.imencode(".jpg", np.full((h, w, 3), 120, np.uint8))
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()


class Session:
    """requests.Session stand-in: records the post, answers like xAI."""

    def __init__(self, status=200, content='{"object": true, "name": "tv remote", "also": [], "confidence": 0.9}'):
        self.status, self.content, self.posts = status, content, []

    def post(self, url, headers=None, json=None, timeout=None):
        self.posts.append((url, headers, json))
        body = {"choices": [{"message": {"content": self.content}}], "usage": {"prompt_tokens": 5}}

        class R:
            status_code = self.status
            text = "err"

            def json(self_inner):
                return body
        return R()


@pytest.fixture(autouse=True)
def fresh():
    grok_trace.clear()
    yield
    grok_trace.clear()


def naming_call(session):
    c = xai.Client("https://api.x.ai/v1", KEY, session=session)
    return c.chat.completions.create(model="grok-4.3", messages=[
        {"role": "system", "content": NAME_SYSTEM},
        {"role": "user", "content": [{"type": "text", "text": "Close-up of the object:"},
                                     {"type": "image_url", "image_url": {"url": data_url(), "detail": "high"}},
                                     {"type": "image_url", "image_url": {"url": data_url(300, 200)}},
                                     {"type": "text", "text": "What is it called?"}]}])


def test_a_call_is_kept_with_thumbnails_and_reply_but_never_the_key():
    naming_call(Session())
    [c] = grok_trace.calls()
    assert c["purpose"] == "naming" and c["model"] == "grok-4.3" and c["ok"] and c["ms"] >= 0
    assert "What is it called?" in c["request"] and "tv remote" in c["reply"]
    assert len(c["images"]) == 2
    big = cv2.imdecode(np.frombuffer(grok_trace.image(c["images"][0]), np.uint8), cv2.IMREAD_COLOR)
    small = cv2.imdecode(np.frombuffer(grok_trace.image(c["images"][1]), np.uint8), cv2.IMREAD_COLOR)
    assert big.shape[:2] == (270, 480) and small.shape[:2] == (200, 300)      # shrunk to 480 px, never enlarged
    assert KEY not in json.dumps(grok_trace.calls()) and "Bearer" not in json.dumps(grok_trace.calls())


def test_errors_are_kept_and_the_ring_drops_old_calls_with_their_images():
    with pytest.raises(xai.XAIError):
        naming_call(Session(status=429))
    assert grok_trace.calls()[0]["error"] == "HTTP 429" and not grok_trace.calls()[0]["ok"]
    first = grok_trace.calls()[0]["images"][0]
    for _ in range(grok_trace.KEEP):
        naming_call(Session())
    assert len(grok_trace.calls(100)) == grok_trace.KEEP and grok_trace.image(first) is None


def test_purpose_from_the_system_prompt():
    from core.grok_check import SYSTEM as CHECK
    from core.room import VERIFY_SYSTEM
    from voice.visual import LOOK_SYSTEM, RECALL_SYSTEM, ROOM_RECALL_SYSTEM, ROOM_SYSTEM
    assert [grok_trace.purpose(s) for s in (VERIFY_SYSTEM, CHECK, LOOK_SYSTEM, ROOM_SYSTEM, RECALL_SYSTEM,
                                            ROOM_RECALL_SYSTEM)] == \
        ["verify", "check", "look", "look_room", "recall", "recall_room"]
    assert grok_trace.purpose("Is the object in mark 2 the same object as the reference?") == "confirm"
    assert grok_trace.purpose("something else") == "other"


def test_server_serves_the_trace_and_thumbnails(tmp_path):
    naming_call(Session())
    events = EventLog(":memory:", str(tmp_path / "snaps"))
    with TestClient(create_app(load_config(), demo_world(events), events)) as c:
        [call] = c.get("/grok/trace?limit=5").json()
        r = c.get("/grok/img/" + call["images"][0])
        assert r.status_code == 200 and r.headers["content-type"] == "image/jpeg"
        assert c.get("/grok/img/99-0").status_code == 404 and c.get("/grok/img/..%2Fx").status_code == 404
