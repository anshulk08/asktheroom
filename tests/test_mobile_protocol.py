"""BLE bridge protocol (mobile/PROTOCOL.md): framing and reassembly, state compaction and change
detection, and the bridge logic with HTTP and the D-Bus layer faked."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import urllib.error

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "mobile", "bridge"))
import ble_bridge as B  # noqa: E402
import bleproto as P  # noqa: E402


# ---------------------------------------------------------------- framing

def roundtrip(payload: bytes, mtu: int, msg_id: int = 5) -> bytes:
    r = P.Reassembler()
    out = None
    chunks = P.frame(msg_id, payload, mtu)
    for i, c in enumerate(chunks):
        out = r.feed(c)
        if i < len(chunks) - 1:
            assert out is None
    return out


@pytest.mark.parametrize("mtu", [23, 100, 185, 247, 517])
@pytest.mark.parametrize("n", [0, 1, 178, 179, 180, 1000, 5000])
def test_frame_roundtrip_and_chunk_limits(mtu, n):
    payload = bytes((i * 7) & 0xFF for i in range(n))
    if n > P.chunk_payload_size(mtu) * P.MAX_CHUNKS:
        with pytest.raises(ValueError):
            P.frame(9, payload, mtu)
        return
    chunks = P.frame(9, payload, mtu)
    assert all(len(c) <= mtu - 3 for c in chunks)
    assert all(len(c) - P.HEADER_LEN <= P.chunk_payload_size(mtu) for c in chunks)
    assert [c[1] for c in chunks] == list(range(len(chunks)))
    assert all(c[0] == 9 for c in chunks)
    assert [c[2] for c in chunks] == [0] * (len(chunks) - 1) + [P.FLAG_FINAL]
    assert roundtrip(payload, mtu) == payload


def test_ios_mtu_gives_179_json_bytes_per_chunk():
    assert P.chunk_payload_size(185) == 179
    chunks = P.frame(0, b"x" * 179, 185)
    assert len(chunks) == 1 and len(chunks[0]) == 182
    assert len(P.frame(0, b"x" * 180, 185)) == 2


def test_empty_message_is_one_final_chunk():
    assert P.frame(3, b"", 185) == [bytes((3, 0, 1))]
    assert roundtrip(b"", 185) == b""


def test_max_chunks():
    size = P.chunk_payload_size(23)
    assert len(P.frame(0, b"a" * size * 256, 23)) == 256
    with pytest.raises(ValueError):
        P.frame(0, b"a" * (size * 256 + 1), 23)


def test_msg_id_wraps():
    c = P.MsgCounter()
    ids = [c.next("state") for _ in range(258)]
    assert ids[:2] == [0, 1] and ids[255] == 255 and ids[256] == 0 and ids[257] == 1
    assert c.next("answer") == 0          # independent per characteristic
    assert P.frame(256 + 4, b"x")[0][0] == 4


def test_lost_final_chunk_is_discarded_by_next_message():
    r = P.Reassembler()
    a = P.frame(1, b"A" * 400, 185)
    b = P.frame(2, b"B" * 50, 185)
    for c in a[:-1]:                      # final chunk of message 1 lost
        assert r.feed(c) is None
    assert r.feed(b[0]) == b"B" * 50
    assert r.dropped == 1 and r.completed == 1


def test_gap_drops_message_until_next_chunk_zero():
    r = P.Reassembler()
    a = P.frame(1, b"A" * 500, 185)      # 3 chunks
    assert r.feed(a[0]) is None
    assert r.feed(a[2]) is None           # chunk 1 missing
    assert r.dropped == 1
    assert r.feed(a[1]) is None           # late chunk 1: no message in progress, ignored
    b = P.frame(2, b"ok", 185)
    assert r.feed(b[0]) == b"ok"


def test_out_of_order_chunks_never_produce_a_corrupt_message():
    payload = bytes(range(256)) * 3
    chunks = P.frame(7, payload, 100)
    order = [0, 2, 1] + list(range(3, len(chunks)))
    r = P.Reassembler()
    results = [r.feed(chunks[i]) for i in order]
    assert all(x is None for x in results)   # reordered: dropped, not mis-assembled
    assert roundtrip(payload, 100, 8) == payload


def test_duplicate_chunk_and_repeated_chunk_zero():
    r = P.Reassembler()
    a = P.frame(1, b"A" * 400, 185)
    r.feed(a[0])
    assert r.feed(a[0]) is None           # restart on repeated chunk 0
    assert r.feed(a[1]) is None
    assert r.feed(a[2]) == b"A" * 400
    r2 = P.Reassembler()
    r2.feed(a[0])
    r2.feed(a[1])
    assert r2.feed(a[1]) is None          # duplicate middle chunk: dropped
    assert r2.feed(a[2]) is None


def test_stray_and_short_values_ignored():
    r = P.Reassembler()
    assert r.feed(b"") is None and r.feed(b"\x01\x00") is None
    assert r.feed(bytes((4, 3, 1)) + b"tail") is None    # chunk 3 with nothing in progress
    assert r.feed(bytes((4, 0, 1)) + b"{}") == b"{}"


def test_reassembler_size_cap():
    r = P.Reassembler(max_bytes=300)
    chunks = P.frame(1, b"z" * 1000, 185)
    assert all(r.feed(c) is None for c in chunks)
    assert r.dropped == 1


def test_same_id_after_completion_is_a_new_message():
    r = P.Reassembler()
    assert r.feed(P.frame(1, b"one")[0]) == b"one"
    assert r.feed(P.frame(1, b"two")[0]) == b"two"


# ---------------------------------------------------------------- question / answer

def test_parse_question_ok():
    assert P.parse_question(P.dumps({"id": 42, "q": "  where are my keys? "})) == (42, "where are my keys?", None)
    assert P.parse_question(P.dumps({"id": 70000, "q": "hi"}))[0] == 70000 & 0xFFFF


def test_parse_question_too_long_keeps_id():
    body = P.dumps({"id": 12, "q": "k" * 200})
    assert len(body) > 180
    assert P.parse_question(body) == (12, None, P.TOO_LONG_TEXT)
    exactly = P.dumps({"id": 1, "q": "k" * (180 - len(P.dumps({"id": 1, "q": ""})))})
    assert len(exactly) == 180 and P.parse_question(exactly)[2] is None
    multibyte = P.dumps({"id": 2, "q": "é" * 85})             # 170 chars, 180+ bytes
    assert len(multibyte) > 180 and P.parse_question(multibyte)[2] == P.TOO_LONG_TEXT


@pytest.mark.parametrize("raw", [b"where are my keys", b"\xff\xfe", b"[1,2]", P.dumps({"id": 3}),
                                 P.dumps({"id": 3, "q": "  "}), P.dumps({"id": True, "q": 5})])
def test_parse_question_rejects_bad_input(raw):
    qid, text, err = P.parse_question(raw)
    assert text is None and err and 0 <= qid <= 0xFFFF


def test_answer_msg_shape():
    a = P.answer_msg(5, True, "On the table.", "keys", "point", [41.2, 29.0], 812)
    assert list(a) == ["id", "ok", "text", "point_at", "action", "target", "ms"]
    assert json.loads(P.dumps(a)) == a


# ---------------------------------------------------------------- state compaction

ENT_KEYS = {
    "name": "keys", "kind": "target", "status": "INSIDE", "parent": "box", "pos_cm": [41.234, 29.01],
    "resolved_cm": [70.449, 38.15], "confidence": 0.8512, "candidates": [], "last_seen": 1727289980.44,
    "zone": "table", "edge": None,
}


def test_compact_entity_short_keys_rounding_and_omitted_nulls():
    e = P.compact_entity(ENT_KEYS)
    assert e == {"n": "keys", "k": "t", "s": "I", "p": "box", "xy": [41.2, 29.0], "r": [70.4, 38.1],
                 "c": 0.85, "ls": 1727289980.4}
    bare = P.compact_entity({"name": "box", "kind": "container", "status": "UNKNOWN", "parent": None,
                             "pos_cm": None, "resolved_cm": None, "confidence": 0.0, "last_seen": None,
                             "edge": None})
    assert bare == {"n": "box", "k": "c", "s": "X", "c": 0.0}


@pytest.mark.parametrize("status,letter", [("VISIBLE", "V"), ("HELD", "H"), ("UNDER", "U"), ("INSIDE", "I"),
                                           ("GONE", "G"), ("UNKNOWN", "X"), ("WEIRD", "X")])
def test_status_letters(status, letter):
    assert P.compact_entity({"name": "a", "kind": "target", "status": status})["s"] == letter


def test_kinds_and_gone_edge():
    assert P.compact_entity({"name": "n", "kind": "cover", "status": "VISIBLE"})["k"] == "v"
    g = P.compact_entity({"name": "wallet", "kind": "target", "status": "GONE", "edge": "left",
                          "pos_cm": [1.0, 20.0]})
    assert g["s"] == "G" and g["edge"] == "left"


def test_things_aliases_and_maybe_same_as_and_unknown_fields():
    t = P.compact_entity({"name": "thing:3", "kind": "target", "status": "VISIBLE", "pos_cm": [10, 12],
                          "aliases": ["charger", "cable"], "label": "charger",
                          "maybe_same_as": [["thing:1", 0.8123]], "future_field": {"x": 1}})
    assert t["a"] == ["charger", "cable"] and t["m"] == [["thing:1", 0.81]]
    assert "future_field" not in t and "label" not in t
    assert P.compact_entity({"name": "thing:4", "kind": "target", "status": "VISIBLE"})["a"] == []
    assert "a" not in P.compact_entity({"name": "keys", "kind": "target", "status": "VISIBLE", "aliases": []})


def test_compact_state_shape():
    st = {"t": 1.0, "online": True, "fps": 12.3, "entities": [ENT_KEYS, {"bad": 1}, "junk"],
          "edges": [["keys", "INSIDE", "box"]], "laser": {"on": True, "target": "box", "err_cm": 0.8},
          "aliases": {}, "merged": {}}
    c = P.compact_state(st, (90, 60), 1790000000.06)
    assert c["v"] == 1 and c["t"] == 1790000000.1 and c["table"] == [90.0, 60.0] and c["online"] is True
    assert c["laser"] == {"on": True, "target": "box"}
    assert [e["n"] for e in c["e"]] == ["keys"]
    empty = P.compact_state(None, (90, 60), 5.0)
    assert empty["e"] == [] and empty["laser"] == {"on": False, "target": None} and empty["online"] is False


def big_state(n_things: int = 12) -> dict:
    ents = []
    for i, name in enumerate(["keys", "pill_bottle", "wallet", "glasses", "phone", "remote", "box", "notebook"]):
        ents.append({"name": name, "kind": "container" if name == "box" else "cover" if name == "notebook" else "target",
                     "status": "INSIDE" if i % 2 else "VISIBLE", "parent": "box" if i % 2 else None,
                     "pos_cm": [12.345 + i, 45.678], "resolved_cm": [70.44, 38.11], "confidence": 0.8512,
                     "candidates": ["notebook"], "last_seen": 1790000000.123, "zone": "table", "edge": None})
    for i in range(n_things):
        ents.append({"name": f"thing:{i + 1}", "kind": "target", "status": "UNDER", "parent": "notebook",
                     "pos_cm": [33.3, 22.2], "resolved_cm": [33.3, 22.2], "confidence": 0.6, "candidates": [],
                     "last_seen": 1790000001.5, "zone": "table", "edge": None,
                     "aliases": ["phone charger", "cable"], "maybe_same_as": [["thing:2", 0.74]]})
    return {"t": 1.0, "online": True, "fps": 12.0, "entities": ents, "edges": [],
            "laser": {"on": True, "target": "keys", "err_cm": 0.5}}


def test_worst_realistic_state_fits():
    c = P.compact_state(big_state(12), (90, 60), 1790000000.0)
    payload = P.dumps(c)
    raw = len(json.dumps(big_state(12)))
    assert len(payload) < raw / 2                        # compaction pays for itself
    assert len(payload) < 4096
    assert len(P.frame(0, payload, 185)) <= 25          # ~25 notifications at iOS MTU
    assert len(P.frame(0, payload, 23)) <= P.MAX_CHUNKS  # still sendable at the minimum MTU


# ---------------------------------------------------------------- change detection

def test_state_changed_ignores_jitter_and_timestamps():
    a = P.compact_state(big_state(2), (90, 60), 1.0)
    b = json.loads(json.dumps(a))
    b["t"] = 99.0
    b["e"][0]["xy"] = [b["e"][0]["xy"][0] + 0.3, b["e"][0]["xy"][1]]    # visible object jitter
    b["e"][0]["ls"] = 123.0                                                  # visible: last_seen ticks
    b["e"][0]["c"] = round(b["e"][0]["c"] - 0.04, 2)
    assert not P.state_changed(a, b)
    assert P.state_changed(None, a)


@pytest.mark.parametrize("mutate", [
    lambda s: s["e"][0].update(s="H"),
    lambda s: s["e"][0].update(p="hand:1"),
    lambda s: s["e"][0].update(xy=[s["e"][0]["xy"][0] + 2.0, s["e"][0]["xy"][1]]),
    lambda s: s["e"][1].update(r=[1.0, 1.0]),
    lambda s: s["e"][1].update(ls=5.0),              # hidden object's last_seen changed
    lambda s: s["e"][0].update(c=0.2),
    lambda s: s["e"].pop(),
    lambda s: s["e"].append({"n": "thing:99", "k": "t", "s": "V", "a": []}),
    lambda s: s["e"][-1].update(a=["mug"]),
    lambda s: s["laser"].update(on=False),
    lambda s: s.update(online=False),
    lambda s: s["e"][0].pop("xy"),
])
def test_state_changed_detects_real_changes(mutate):
    a = P.compact_state(big_state(2), (90, 60), 1.0)
    b = json.loads(json.dumps(a))
    mutate(b)
    assert P.state_changed(a, b)


def test_status_msg_and_changes():
    s = P.status_msg(True, {"fps": 12.34, "online": True}, True, False)
    assert s == {"app": "up", "fps": 12.3, "online": True, "cal": True, "laser_cal": False}
    assert len(P.dumps(s)) < 180
    down = P.status_msg(False, {"fps": 12.34, "online": True}, False, False)
    assert down["app"] == "down" and down["fps"] == 0.0 and down["online"] is False
    assert not P.status_changed(s, dict(s, fps=12.9))
    assert P.status_changed(s, dict(s, fps=11.2))
    assert P.status_changed(s, dict(s, app="down"))
    assert P.status_changed(None, s)


# ---------------------------------------------------------------- outbox

def test_outbox_priority_and_latest_wins():
    o = P.Outbox()
    o.push("state", [b"s1a", b"s1b"])
    o.push("state", [b"s2a"])                     # s1 not started: replaced
    o.push("answer", [b"a1", b"a2"])
    o.push("status", [b"t1"])
    assert o.replaced == 1
    assert o.pop(10) == [("answer", b"a1"), ("answer", b"a2"), ("status", b"t1"), ("state", b"s2a")]
    assert o.pending() == 0


def test_outbox_never_tears_a_started_message():
    o = P.Outbox()
    o.push("state", [b"s1a", b"s1b", b"s1c"])
    assert o.pop(1) == [("state", b"s1a")]
    o.push("state", [b"s2a"])                     # s1 already started: keep it, queue s2 after
    o.push("state", [b"s3a"])                     # s2 not started: replaced by s3
    assert [v for _, v in o.pop(10)] == [b"s1b", b"s1c", b"s3a"]
    o.push("answer", [b"x"])
    o.push("answer", [b"y"])                      # answers are never dropped
    assert [v for _, v in o.pop(10)] == [b"x", b"y"]


def test_outbox_answer_jumps_ahead_of_long_state():
    o = P.Outbox()
    o.push("state", [b"s%d" % i for i in range(10)])
    assert len(o.pop(3)) == 3
    o.push("answer", [b"ans"])
    assert o.pop(1) == [("answer", b"ans")]


# ---------------------------------------------------------------- bridge logic, HTTP and BLE faked

WORLD = {"t": 1790000000.0, "online": True, "fps": 13.1, "entities": [
    dict(ENT_KEYS),
    {"name": "box", "kind": "container", "status": "VISIBLE", "parent": None, "pos_cm": [70.4, 38.1],
     "resolved_cm": [70.4, 38.1], "confidence": 1.0, "candidates": [], "last_seen": 1790000000.0,
     "zone": "table", "edge": None},
    {"name": "remote", "kind": "target", "status": "UNKNOWN", "parent": "unknown", "pos_cm": [50.0, 45.0],
     "resolved_cm": None, "confidence": 0.4, "candidates": [], "last_seen": 1789999000.0, "zone": "table",
     "edge": None}],
    "edges": [], "laser": {"on": False, "target": None, "err_cm": None}}


class FakeHTTP:
    def __init__(self):
        self.state = json.loads(json.dumps(WORLD))
        self.down = False
        self.ask_exc = None
        self.asked = []
        self.reply = {"text": "The keys are inside the box.", "point_at": "keys", "action": "point",
                      "latency_ms": 40}

    def get_json(self, path, timeout=1.0):
        if self.down:
            raise B.RoomDown("connection refused")
        assert path == "/state"
        return {"state": json.loads(json.dumps(self.state)), "last_answer": None, "server_t": 0}

    def post_json(self, path, body, timeout=12.0):
        if self.down:
            raise B.RoomDown("connection refused")
        if self.ask_exc:
            raise self.ask_exc
        assert path == "/ask"
        self.asked.append(body)
        return dict(self.reply)


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


class Phone:
    """What the BLE layer would deliver: per-characteristic reassembly of emitted chunks."""

    def __init__(self, mtu_limit=185):
        self.mtu_limit = mtu_limit
        self.asm = {c: P.Reassembler() for c in B.CHARS}
        self.msgs = {c: [] for c in B.CHARS}
        self.chunk_counts = {c: [] for c in B.CHARS}

    def emit(self, char, chunks):
        self.chunk_counts[char].append(len(chunks))
        for c in chunks:
            assert len(c) <= self.mtu_limit - 3
            m = self.asm[char].feed(c)
            if m is not None:
                self.msgs[char].append(json.loads(m))


def make(tmp_path=None, **kw):
    http, phone, mono = FakeHTTP(), Phone(), Clock()
    tcal = str(tmp_path / "table_cal.json") if tmp_path else None
    lcal = str(tmp_path / "laser_cal.json") if tmp_path else None
    core = B.BridgeCore(http, phone.emit, table_cm=(90, 60), table_cal=tcal, laser_cal=lcal,
                        clock=lambda: 1790000000.0 + mono.t, mono=mono, **kw)
    return core, http, phone, mono


def test_subscribe_sends_snapshot_then_rate_limits_and_heartbeats():
    core, http, phone, mono = make()
    core.poll_once()
    assert phone.msgs["state"] == []                 # nobody subscribed
    core.set_notifying("state", True)
    assert len(phone.msgs["state"]) == 1
    snap = phone.msgs["state"][0]
    assert snap["table"] == [90.0, 60.0] and {e["n"] for e in snap["e"]} == {"keys", "box", "remote"}
    keys = next(e for e in snap["e"] if e["n"] == "keys")
    assert keys == {"n": "keys", "k": "t", "s": "I", "p": "box", "xy": [41.2, 29.0], "r": [70.4, 38.1],
                    "c": 0.85, "ls": 1727289980.4}
    # unchanged: nothing until the 5 s heartbeat
    for _ in range(10):
        mono.t += 0.25
        core.poll_once()
    assert len(phone.msgs["state"]) == 1
    mono.t += 2.6
    core.poll_once()
    assert len(phone.msgs["state"]) == 2             # heartbeat at >= 5 s
    # a change 0.1 s after a send waits for the 0.5 s floor (2 Hz max)
    http.state["entities"][0]["status"] = "HELD"
    mono.t += 0.1
    core.poll_once()
    assert len(phone.msgs["state"]) == 2
    mono.t += 0.4
    core.poll_once()
    assert len(phone.msgs["state"]) == 3 and phone.msgs["state"][-1]["e"][0]["s"] == "H"


def test_state_rate_never_exceeds_2hz_under_constant_change():
    core, http, phone, mono = make()
    core.set_notifying("state", True)
    n0 = len(phone.msgs["state"])
    for i in range(40):                              # 10 s of 4 Hz polls, every poll a real change
        http.state["entities"][1]["pos_cm"] = [10.0 + 3 * i, 20.0]
        mono.t += 0.25
        core.poll_once()
    sent = len(phone.msgs["state"]) - n0
    assert 18 <= sent <= 21


def test_status_notify_read_and_app_down(tmp_path):
    core, http, phone, mono = make(tmp_path)
    core.set_notifying("status", True)
    core.poll_once()
    assert phone.msgs["status"][-1] == {"app": "up", "fps": 13.1, "online": True, "cal": True, "laser_cal": False}
    assert json.loads(core.status_read()) == phone.msgs["status"][-1]
    assert len(core.status_read()) < 180
    n = len(phone.msgs["status"])
    mono.t += 0.3
    http.state["fps"] = 13.4                          # small fps drift: no notify
    core.poll_once()
    assert len(phone.msgs["status"]) == n
    http.down = True                                  # app goes down: notified at once
    mono.t += 0.1
    core.poll_once()
    assert phone.msgs["status"][-1]["app"] == "down" and phone.msgs["status"][-1]["fps"] == 0.0
    (tmp_path / "laser_cal.json").write_text("{}")
    http.down = False
    mono.t += 1.1
    core.poll_once()
    assert phone.msgs["status"][-1] == {"app": "up", "fps": 13.4, "online": True, "cal": True, "laser_cal": True}


def test_cal_from_file_or_live_world(tmp_path):
    core, http, phone, mono = make(tmp_path)
    http.state["t"] = None
    core.poll_once()
    assert core.status["cal"] is False
    (tmp_path / "table_cal.json").write_text("{}")
    core.poll_once()
    assert core.status["cal"] is True


def test_question_answer_with_target_and_speaking_source():
    core, http, phone, mono = make()
    core.set_notifying("answer", True)
    core.on_question(P.dumps({"id": 321, "q": "where are my keys?"}))
    qid, text, t0 = core.asks.get_nowait()
    mono.t += 0.8
    msg = core.answer(qid, text, t0)
    assert http.asked == [{"text": "where are my keys?", "source": "dashboard"}]
    assert msg == {"id": 321, "ok": True, "text": "The keys are inside the box.", "point_at": "keys",
                   "action": "point", "target": [70.4, 38.1], "ms": 800}


def test_answer_target_falls_back_to_last_seen_and_null_without_point():
    core, http, phone, mono = make()
    http.reply = {"text": "I lost track of the remote.", "point_at": "remote", "action": "circle"}
    assert core.answer(1, "where is the remote?")["target"] == [50.0, 45.0]
    http.reply = {"text": "Someone is holding the wallet.", "point_at": None, "action": None}
    assert core.answer(2, "where is my wallet?")["target"] is None
    http.reply = {"text": "The wallet was carried off the left side.", "point_at": None, "action": "sweep:left"}
    a = core.answer(3, "where is my wallet?")
    assert a["action"] == "sweep:left" and a["ok"] is True


def test_answer_when_room_down_slow_or_broken():
    core, http, phone, mono = make()
    http.down = True
    a = core.answer(4, "where are my keys?")
    assert a["ok"] is False and a["text"] == "The room isn't running right now." and a["id"] == 4
    http.down = False
    http.ask_exc = TimeoutError("timed out")
    assert core.answer(5, "q")["text"] == B.SLOW_TEXT
    http.ask_exc = urllib.error.HTTPError("http://x/ask", 500, "boom", {}, None)
    b = core.answer(6, "q")
    assert b["ok"] is False and b["text"] == B.FAIL_TEXT


def test_rejections_are_answered_immediately():
    core, http, phone, mono = make()
    core.set_notifying("answer", True)
    core.on_question(P.dumps({"id": 77, "q": "k" * 300}))
    assert phone.msgs["answer"][-1] == {"id": 77, "ok": False, "text": "Question too long.", "point_at": None,
                                        "action": None, "target": None, "ms": 0}
    core.on_question(P.dumps({"id": 1, "q": "first"}))          # queued (worker not running)
    core.on_question(P.dumps({"id": 2, "q": "second"}))         # queue full: busy
    assert phone.msgs["answer"][-1]["id"] == 2 and phone.msgs["answer"][-1]["text"] == B.BUSY_TEXT
    assert http.asked == []


def test_worker_thread_answers_over_the_link():
    core, http, phone, mono = make()
    core.set_notifying("answer", True)
    core.start(poll_s=0.05)
    try:
        core.on_question(P.dumps({"id": 9, "q": "where are my keys?"}))
        import time
        end = time.time() + 3
        while time.time() < end and not phone.msgs["answer"]:
            time.sleep(0.02)
    finally:
        core.stop()
    assert phone.msgs["answer"] and phone.msgs["answer"][0]["id"] == 9 and phone.msgs["answer"][0]["ok"]


def test_mtu_learning_shrinks_chunks_and_resends_snapshot():
    core, http, phone, mono = make()
    http.state = big_state(6)
    core.poll_once()
    core.set_notifying("state", True)
    n = len(phone.msgs["state"])
    assert core.mtu() == 185
    phone.mtu_limit = 23
    core.note_mtu("/org/bluez/hci0/dev_AA", 23)       # a small-MTU central shows up
    assert core.force_state
    core.poll_once()
    assert len(phone.msgs["state"]) == n + 1 and phone.chunk_counts["state"][-1] > phone.chunk_counts["state"][0]
    core.note_mtu("/org/bluez/hci0/dev_BB", 517)      # the min wins
    assert core.mtu() == 23
    core.forget_device("/org/bluez/hci0/dev_AA")
    assert core.mtu() == 517
    core.note_mtu("x", "junk")
    core.note_mtu("y", 5)                             # below the ATT minimum: ignored
    assert core.mtu() == 517


def test_unsubscribe_stops_sending():
    core, http, phone, mono = make()
    core.set_notifying("state", True)
    core.set_notifying("state", False)
    n = len(phone.msgs["state"])
    mono.t += 10
    core.poll_once()
    assert len(phone.msgs["state"]) == n


def test_open_world_things_reach_the_phone():
    core, http, phone, mono = make()
    http.state["entities"].append({"name": "thing:1", "kind": "target", "status": "VISIBLE", "parent": None,
                                   "pos_cm": [20.0, 20.0], "resolved_cm": [20.0, 20.0], "confidence": 0.9,
                                   "candidates": [], "last_seen": 1.0, "zone": "table", "edge": None,
                                   "label": "charger", "aliases": ["charger"], "maybe_same_as": []})
    core.poll_once()
    core.set_notifying("state", True)
    t = next(e for e in phone.msgs["state"][-1]["e"] if e["n"] == "thing:1")
    assert t["a"] == ["charger"] and "m" not in t


# ---------------------------------------------------------------- misc

def test_le_event_mask_command_and_result_parsing():
    assert B.le_event_mask_cmd("hci0") == ["hcitool", "-i", "hci0", "cmd", "0x08", "0x0001",
                                           "DF", "1F", "0A", "00", "00", "00", "00", "00"]
    with pytest.raises(ValueError):
        B.le_event_mask_cmd("hci0", "DF1F")

    def ok(cmd, **kw):
        return subprocess.CompletedProcess(cmd, 0, "< HCI Command: ogf 0x08, ocf 0x0001, plen 8\n  DF 1F 0A 00 00 00 "
                                                   "00 00 \n> HCI Event: 0x0e plen 4\n  02 01 20 00 \n", "")

    def denied(cmd, **kw):
        return subprocess.CompletedProcess(cmd, 1, "< HCI Command: ogf 0x08, ocf 0x0001, plen 8\n  DF 1F 0A 00 00 "
                                                   "00 00 00 \nSend failed: Operation not permitted\n", "")

    def missing(cmd, **kw):
        raise FileNotFoundError("hcitool")

    assert B.apply_le_event_mask("hci0", run=ok) is True
    assert B.apply_le_event_mask("hci0", run=denied) is False
    assert B.apply_le_event_mask("hci0", run=missing) is False


def test_read_config_table_size(tmp_path):
    (tmp_path / "config.yaml").write_text("table:\n  size_cm: [120, 80]\npaths:\n  table_cal: t.json\n")
    size, tcal, lcal = B.read_config(str(tmp_path))
    assert size == (120.0, 80.0) and tcal == str(tmp_path / "t.json") and lcal.endswith("laser_cal.json")
    assert B.read_config(str(tmp_path / "missing"))[0] == (90.0, 60.0)


def test_uuids_follow_the_spec():
    uu = [P.SERVICE_UUID, P.QUESTION_UUID, P.ANSWER_UUID, P.STATE_UUID, P.STATUS_UUID]
    for i, u in enumerate(uu, start=1):
        assert u == f"8a1e000{i}-6b7f-4c2b-9e3a-2f5d7c1a000{i}"
    assert P.LOCAL_NAME == "AskTheRoom"


def test_bridge_questions_are_spoken_and_aimed():
    """A question the bridge POSTs to /ask reaches Room.ask_and_act with a source that is spoken and
    aimed (not a text-only source like sms). Checked by behaviour, so renaming server internals is fine."""
    from types import SimpleNamespace

    from fastapi.testclient import TestClient

    import main
    from core.config import load_config
    from core.fakeworld import demo_world
    from core.types import Answer
    from server.app import create_app

    seen = []
    world = demo_world()
    app = create_app(load_config(), world, world.events,
                     ask_fn=lambda text, source: seen.append(source) or Answer("ok"))
    r = TestClient(app).post("/ask", json={"text": "where are my keys?", "source": B.ASK_SOURCE})
    assert r.status_code == 200 and seen, r.text

    responded = []
    room = SimpleNamespace(ask=lambda text, source: Answer("ok", point_at="keys", action="point"),
                           respond=responded.append)
    main.Room.ask_and_act(room, "where are my keys?", seen[0])
    assert responded, f"source {seen[0]!r} is answered as text only"
