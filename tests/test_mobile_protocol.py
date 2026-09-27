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
    assert s == {"app": "up", "fps": 12.3, "online": True, "cal": True, "laser_cal": False, "gk": False,
                 "spk": False}
    assert len(P.dumps(s)) < 180
    down = P.status_msg(False, {"fps": 12.34, "online": True}, False, False)
    assert down["app"] == "down" and down["fps"] == 0.0 and down["online"] is False
    assert not P.status_changed(s, dict(s, fps=12.9))
    assert P.status_changed(s, dict(s, fps=11.2))
    assert P.status_changed(s, dict(s, app="down"))
    assert P.status_changed(None, s)


def test_status_gk_is_grok_check_on_and_online():
    gcs = {"enabled": True, "provider": "grok", "model": "grok-4", "calls_last_hour": 3, "last": None,
           "disclosure": "Grok check is on: ..."}
    s = P.status_msg(True, {"fps": 12.0, "online": True, "grok_check": gcs}, True, True)
    assert s["gk"] is True and len(P.dumps(s)) < 180
    assert P.status_msg(True, {"fps": 12.0, "online": False, "grok_check": gcs}, True, True)["gk"] is False
    assert P.status_msg(True, {"fps": 12.0, "online": True}, True, True)["gk"] is False     # older server / off
    assert P.status_msg(False, {"online": True, "grok_check": gcs}, True, True)["gk"] is False
    assert P.status_changed(s, dict(s, gk=False))


# ---------------------------------------------------------------- guesses, Grok names, stale things

def thing(n=1, **kw) -> dict:
    return dict({"name": f"thing:{n}", "kind": "target", "status": "VISIBLE", "pos_cm": [10, 10],
                 "confidence": 0.9, "last_seen": 1000.0, "aliases": []}, **kw)


def test_guess_preferred_over_belief_and_gc_rounded():
    t = P.compact_entity(thing(guess={"name": "mug", "confidence": 0.8765, "also": ["cup"]},
                               belief=[["cup", 0.6], ["bowl", 0.2]]))
    assert t["g"] == "mug" and t["gc"] == 0.88
    t = P.compact_entity(thing(guess={"name": "mug", "confidence": 0.5}, belief=[["cup", 0.7123]]))
    assert t["g"] == "cup" and t["gc"] == 0.71                    # higher-confidence belief wins
    t = P.compact_entity(thing(guess={"name": "mug", "confidence": 0.7}, belief=[["cup", 0.7]]))
    assert t["g"] == "mug"                                         # a tie keeps the guess
    t = P.compact_entity(thing(guess={"name": "mug"}))
    assert t["g"] == "mug" and "gc" not in t                       # no numeric confidence: no gc
    t = P.compact_entity(thing(guess={"name": "mug", "confidence": "high"}))
    assert t["g"] == "mug" and "gc" not in t


def test_belief_fallback_when_no_guess():
    t = P.compact_entity(thing(belief=[["water bottle", 0.456], ["thermos", 0.3]]))
    assert t["g"] == "water bottle" and t["gc"] == 0.46
    for b in ([], None, [[]], [["", 0.9]], "junk"):
        t = P.compact_entity(thing(belief=b))
        assert "g" not in t and "gc" not in t
    assert "belief" not in P.compact_entity(thing(belief=[["cup", 0.5]]))


def test_named_by_grok_sets_as():
    t = P.compact_entity(thing(aliases=["stapler"], label="stapler", named_by="grok"))
    assert t["a"] == ["stapler"] and t["as"] == "grok" and "named_by" not in t
    assert "as" not in P.compact_entity(thing(aliases=["stapler"], label="stapler"))          # taught
    assert "as" not in P.compact_entity(thing(aliases=["stapler"], named_by=None))
    assert "as" not in P.compact_entity({"name": "keys", "kind": "target", "status": "VISIBLE",
                                         "named_by": "grok"})                                  # things only


def test_stale_unnamed_things_are_dropped_named_and_configured_kept():
    now = 1000.0 + P.STALE_THING_S + 1
    ents = [thing(1, status="GONE"),                                   # stale unnamed: dropped
            thing(2, status="UNKNOWN"),                                # stale unnamed: dropped
            thing(3, status="GONE", aliases=["charger"]),              # named: kept
            thing(4, status="GONE", last_seen=now - 60),               # recent: kept
            thing(5, status="GONE", last_seen=None),                   # never seen: kept
            thing(6, status="UNDER"),                                  # hidden, not gone: kept
            thing(7, status="VISIBLE"),
            {"name": "wallet", "kind": "target", "status": "GONE", "last_seen": 1.0}]   # configured: kept
    c = P.compact_state({"online": True, "entities": ents}, (90, 60), now)
    assert [e["n"] for e in c["e"]] == ["thing:3", "thing:4", "thing:5", "thing:6", "thing:7", "wallet"]
    assert [e["n"] for e in P.compact_state({"entities": ents}, (90, 60), 1000.0)["e"]][:2] == \
        ["thing:1", "thing:2"]                                         # not stale yet


@pytest.mark.parametrize("mutate", [
    lambda s: s["e"][-1].update(g="cup"),
    lambda s: s["e"][-1].update(gc=0.2),
    lambda s: s["e"][-1].pop("gc"),
    lambda s: s["e"][-1].update(**{"as": "grok"}),
])
def test_state_changed_on_guess_and_grok_name(mutate):
    st = big_state(2)
    st["entities"][-1].update(guess={"name": "mug", "confidence": 0.6}, aliases=[])
    a = P.compact_state(st, (90, 60), 1790000000.0)
    assert a["e"][-1]["g"] == "mug" and a["e"][-1]["gc"] == 0.6
    b = json.loads(json.dumps(a))
    b["e"][-1]["gc"] = 0.64                                           # within the deadband
    assert not P.state_changed(a, b)
    mutate(b)
    assert P.state_changed(a, b)


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
        self.meta = {}
        self.reply = {"text": "The keys are inside the box.", "point_at": "keys", "action": "point",
                      "latency_ms": 40}

    def get_json(self, path, timeout=1.0):
        if self.down:
            raise B.RoomDown("connection refused")
        if path == "/room_layout":
            if getattr(self, "layout", None) is None:
                raise B.RoomDown("HTTP 404")
            return json.loads(json.dumps(self.layout))
        assert path == "/state"
        return {"state": json.loads(json.dumps(self.state)), "last_answer": None, "server_t": 0,
                **json.loads(json.dumps(self.meta))}

    def post_json(self, path, body, timeout=12.0):
        if self.down:
            raise B.RoomDown("connection refused")
        if self.ask_exc:
            raise self.ask_exc
        if path == "/voice":
            self.voiced = getattr(self, "voiced", []) + [body]
            return {"engine": body.get("engine") or "grok", "grok_voice": body.get("grok_voice") or "eve",
                    "speed": body.get("speed") or 1.0}
        if path == "/orientation":
            from core.viewframe import View
            self.oriented = getattr(self, "oriented", []) + [body]
            self.meta["view"] = View.make(body["front"], (100, 60)).to_json()
            return dict(self.meta["view"])
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
    assert phone.msgs["status"][-1] == {"app": "up", "fps": 13.1, "online": True, "cal": True, "laser_cal": False,
                                        "gk": False, "spk": False}
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
    assert phone.msgs["status"][-1] == {"app": "up", "fps": 13.4, "online": True, "cal": True, "laser_cal": True,
                                        "gk": False, "spk": False}


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
    assert http.asked == [{"text": "where are my keys?", "source": "phone"}]
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
    (tmp_path / "t.json").write_text('{"H": [[1,0,0],[0,1,0],[0,0,1]], "size_cm": [70.5, 48.0]}')
    assert B.read_config(str(tmp_path))[0] == (70.5, 48.0)       # one-tag calibration: the area it saved


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
                           respond=responded.append, _phone_qs=__import__("collections").deque())
    main.Room.ask_and_act(room, "where are my keys?", seen[0])
    assert responded, f"source {seen[0]!r} is answered as text only"


# ---------------------------------------------------------------- room answers and notices (P2)

def room_answer(seq, src="voice", q="where are my keys", text="Your keys are inside the box.", point_at="keys"):
    return {"seq": seq, "t": 1790000000.0 + seq, "src": src, "q": q, "text": text, "point_at": point_at,
            "action": "point" if point_at else None}


def notice(nid, text="It's 9 and the pill bottle hasn't been picked up.", point_at="pill_bottle"):
    return {"id": nid, "t": 1790000000.0, "kind": "reminder", "text": text, "point_at": point_at,
            "action": "point", "acknowledged": False}


def test_room_answers_and_notices_reach_the_phone_once_and_only_new_ones():
    core, http, phone, mono = make()
    http.meta = {"answers": [room_answer(1)], "notices": [notice(4)]}
    core.set_notifying("answer", True)
    core.poll_once()                                   # what happened before the phone connected: not replayed
    assert phone.msgs["answer"] == []
    http.meta = {"answers": [room_answer(1), room_answer(2, src="phone"), room_answer(3, q="what color is it",
                                                                                      text="Brown.", point_at=None)],
                 "notices": [notice(5), notice(4)]}
    core.poll_once()
    core.poll_once()                                   # again: nothing twice
    got = phone.msgs["answer"]
    assert got == [
        {"id": None, "src": "voice", "q": "what color is it", "ok": True, "text": "Brown.", "point_at": None,
         "action": None, "target": None, "ms": None},
        {"id": None, "src": "notice", "nid": 5, "kind": "reminder", "ok": True,
         "text": "It's 9 and the pill bottle hasn't been picked up.", "point_at": "pill_bottle",
         "action": "point", "target": None, "ms": None}]           # the phone's own question (src phone) is skipped


def test_room_answer_carries_the_target_position():
    core, http, phone, mono = make()
    http.meta = {"answers": []}
    core.set_notifying("answer", True)
    core.poll_once()
    http.meta = {"answers": [room_answer(1)]}
    core.poll_once()
    [m] = phone.msgs["answer"]
    assert m["src"] == "voice" and m["q"] == "where are my keys" and m["target"] == [70.4, 38.1]


def test_room_messages_wait_for_a_subscribed_phone_and_bad_rows_are_ignored():
    core, http, phone, mono = make()
    http.meta = {"answers": [], "notices": []}
    core.poll_once()
    http.meta = {"answers": [room_answer(1), "junk", {"seq": "x"}], "notices": [notice(1), {"id": None}]}
    core.poll_once()                                   # not subscribed: dropped, not queued
    core.set_notifying("answer", True)
    core.poll_once()
    assert phone.msgs["answer"] == []


def test_server_logs_recent_answers_with_their_source():
    from fastapi.testclient import TestClient

    from core.config import load_config
    from core.fakeworld import demo_world
    from core.types import Answer
    from server.app import create_app

    world = demo_world()
    seen = []
    app = create_app(load_config(), world, world.events,
                     ask_fn=lambda text, source: seen.append(source) or Answer("ok", point_at="keys", action="point"))
    c = TestClient(app)
    c.post("/ask", json={"text": "where are my keys?", "source": "phone"})
    c.post("/ask", json={"text": "and my wallet?", "source": "root"})            # unknown source -> dashboard
    app.state.record_answer("what color is it", Answer("Brown."), "voice")        # main.py's voice loop
    assert seen == ["phone", "dashboard"]
    rows = c.get("/state").json()["answers"]
    assert [(r["seq"], r["src"], r["q"], r["text"]) for r in rows] == [
        (1, "phone", "where are my keys?", "ok"), (2, "dashboard", "and my wallet?", "ok"),
        (3, "voice", "what color is it", "Brown.")]
    for i in range(30):
        app.state.record_answer(f"q{i}", Answer("a"), "voice")
    rows = c.get("/state").json()["answers"]
    assert len(rows) == 10 and rows[-1]["seq"] == 33


def test_clutter_belief_is_never_a_guess():
    e = {"name": "thing:4", "kind": "thing", "status": "VISIBLE", "aliases": [],
         "belief": [["not an object", 0.5], ["cable", 0.3]]}
    out = P.compact_entity(e)
    assert out["g"] == "cable" and out["gc"] == 0.3
    assert "g" not in P.compact_entity({**e, "belief": [["not an object", 0.9]]})


def test_read_config_applies_the_local_file(tmp_path):
    (tmp_path / "config.yaml").write_text("table:\n  size_cm: [90, 60]\npaths:\n  table_cal: table_cal.json\n")
    (tmp_path / "config.local.yaml").write_text("paths:\n  laser_cal: rig_laser.json\n")
    size, tcal, lcal = B.read_config(str(tmp_path))
    assert size == (90.0, 60.0) and tcal.endswith("table_cal.json") and lcal.endswith("rig_laser.json")


def test_the_bridge_follows_a_recalibrated_table_size(tmp_path):
    import os
    tcal = tmp_path / "table_cal.json"
    core = B.BridgeCore(None, lambda c, ch: None, table_cm=(90, 60), table_cal=str(tcal))
    assert core.current_table_cm() == (90.0, 60.0)                  # no calibration file yet
    tcal.write_text('{"H": [[1,0,0],[0,1,0],[0,0,1]], "size_cm": [70.5, 48.0]}')
    assert core.current_table_cm() == (70.5, 48.0)
    tcal.write_text('{"H": [[1,0,0],[0,1,0],[0,0,1]], "size_cm": [95, 55]}')
    os.utime(tcal, (1e9, 1e9))                                       # a new mtime, as a refit writes
    assert core.current_table_cm() == (95.0, 55.0)
    tcal.write_text('{"H": [[1,0,0],[0,1,0],[0,0,1]]}')               # four-marker file: keeps the size
    os.utime(tcal, (2e9, 2e9))
    assert core.current_table_cm() == (95.0, 55.0)


def test_a_room_object_carries_its_zone_and_no_table_position():
    """Room memory (spec 0010): a prop carried to the couch has no table cm; the phone lists it by zone."""
    room = P.compact_entity({"name": "wallet", "kind": "target", "status": "VISIBLE", "pos_cm": None,
                             "resolved_cm": None, "confidence": 0.8, "zone": "couch", "last_seen": 1790389800.0})
    assert room["z"] == "couch" and "xy" not in room and "r" not in room
    table = P.compact_entity({"name": "wallet", "kind": "target", "status": "VISIBLE", "pos_cm": [10, 20],
                              "zone": "table"})
    assert "z" not in table and table["xy"] == [10.0, 20.0]
    assert "z" not in P.compact_entity({"name": "keys", "kind": "target", "status": "VISIBLE"})   # older server


# -- the rig's speaker and the phone's voice (PROTOCOL.md 5a, status spk)

def test_status_spk_is_the_rigs_speaker_connected():
    on = P.status_msg(True, {"fps": 12.0, "online": True, "speaker": {"ok": True, "name": "bluez_sink.x"}}, True, True)
    assert on["spk"] is True and len(P.dumps(on)) < 180
    assert P.status_msg(True, {"speaker": {"ok": False}}, True, True)["spk"] is False
    assert P.status_msg(True, {"fps": 12.0}, True, True)["spk"] is False                 # older server
    assert P.status_msg(False, {"speaker": {"ok": True}}, True, True)["spk"] is False     # app down
    assert P.status_changed(on, dict(on, spk=False))


def test_parse_voice():
    assert P.parse_voice(P.dumps({"voice": {"e": "grok", "v": "ara", "s": 1.1}})) == \
        {"engine": "grok", "grok_voice": "ara", "speed": 1.1}
    assert P.parse_voice(P.dumps({"voice": {}})) == {"engine": None, "grok_voice": None, "speed": None}
    assert P.parse_voice(P.dumps({"id": 1, "q": "where are my keys?"})) is None
    for raw in (b"not json", b"[1]", P.dumps({"voice": "eve"}), P.dumps({"voice": {"v": "x" * 300}})):
        assert P.parse_voice(raw) is None


def test_a_voice_write_is_applied_and_never_answered():
    core, http, phone, mono = make()
    core.set_notifying("answer", True)
    before = len(phone.msgs["answer"])
    core.on_question(P.dumps({"voice": {"e": "rig", "v": "", "s": 0.9}}))
    for t in [t for t in B.threading.enumerate() if t.name == "voice"]:
        t.join(2)
    assert http.voiced == [{"engine": "rig", "grok_voice": "", "speed": 0.9}]
    assert core.asks.empty() and len(phone.msgs["answer"]) == before and http.asked == []


# -- the user's frame (PROTOCOL.md 5b, state "view"): a 100 x 60 cm camera table, worked out by hand

from core.viewframe import View  # noqa: E402

# camera point -> the user's, per front: A (90, 5) is the camera view's top right, B (10, 50) its bottom left
FRAMES = {
    "bottom": {"table": [100.0, 60.0], "A": [90.0, 5.0], "B": [10.0, 50.0],
               "edges": {"left": "left", "right": "right", "top": "top", "bottom": "bottom"}},
    "top": {"table": [100.0, 60.0], "A": [10.0, 55.0], "B": [90.0, 10.0],
            "edges": {"left": "right", "right": "left", "top": "bottom", "bottom": "top"}},
    "right": {"table": [60.0, 100.0], "A": [55.0, 90.0], "B": [10.0, 10.0],
              "edges": {"left": "top", "right": "bottom", "top": "right", "bottom": "left"}},
    "left": {"table": [60.0, 100.0], "A": [5.0, 10.0], "B": [50.0, 90.0],
             "edges": {"left": "bottom", "right": "top", "top": "left", "bottom": "right"}},
}


def view(front, size=(100, 60), outline=None):
    return View.make(front, size, outline).to_json()


def cam_state():
    return {"t": 1.0, "online": True, "laser": {"on": True, "target": "a"}, "entities": [
        {"name": "a", "kind": "target", "status": "VISIBLE", "pos_cm": [90, 5], "resolved_cm": [90, 5]},
        {"name": "b", "kind": "target", "status": "INSIDE", "parent": "box", "pos_cm": [10, 50],
         "resolved_cm": [90, 5]},
        {"name": "wallet", "kind": "target", "status": "GONE", "edge": "left", "pos_cm": [1, 30]},
        {"name": "phone", "kind": "target", "status": "GONE", "edge": "top", "pos_cm": [50, 1]},
        {"name": "remote", "kind": "target", "status": "VISIBLE", "zone": "couch"}]}


def test_parse_orient():
    assert P.parse_orient(P.dumps({"orient": {"front": "right"}})) == {"front": "right"}
    assert P.parse_orient(P.dumps({"orient": {"front": " Left "}})) == {"front": "left"}
    for f in P.FRONTS:
        assert P.parse_orient(P.dumps({"orient": {"front": f}})) == {"front": f}
    assert P.parse_orient(P.dumps({"id": 1, "q": "where are my keys?"})) is None
    assert P.parse_orient(P.dumps({"voice": {"e": "grok"}})) is None
    for raw in (b"not json", b"[1]", P.dumps({"orient": "right"}),
                P.dumps({"orient": {"front": "right", "pad": "x" * 200}})):
        assert P.parse_orient(raw) is None, raw
    assert P.parse_orient(P.dumps({"orient": {"front": None}})) == {"front": None}     # back to the rig's default
    for bad in ({}, {"front": "sideways"}, {"front": 3}):
        assert P.parse_orient(P.dumps({"orient": bad})) == {}, bad                     # an orient write, ignored


@pytest.mark.parametrize("front", list(FRAMES))
def test_compact_state_in_the_users_frame_for_every_front(front):
    want = FRAMES[front]
    c = P.compact_state(cam_state(), (100, 60), 5.0, view(front))
    assert c["table"] == want["table"] and c["view"] == {"f": front, "o": False}
    e = {x["n"]: x for x in c["e"]}
    assert e["a"]["xy"] == want["A"] and e["a"]["r"] == want["A"]
    assert e["b"]["xy"] == want["B"] and e["b"]["r"] == want["A"]
    assert e["wallet"]["edge"] == want["edges"]["left"] and e["phone"]["edge"] == want["edges"]["top"]
    assert "xy" not in e["remote"] and e["remote"]["z"] == "couch"           # a room object: no table position
    assert c["laser"] == {"on": True, "target": "a"} and c["online"] is True


@pytest.mark.parametrize("front", list(FRAMES))
def test_edges_targets_and_sweeps_for_every_front(front):
    want, v = FRAMES[front], view(front)
    for cam, user in want["edges"].items():
        assert P.VIEW_EDGE[front][cam] == user == View.make(front, (100, 60)).edge(cam)
        assert P.view_action(f"sweep:{cam}", v) == f"sweep:{user}"
    assert P.target_of(cam_state(), "a", v) == want["A"]
    assert P.target_of(cam_state(), "wallet", v) == P.compact_entity(cam_state()["entities"][2], P.view_of(v))["xy"]
    assert P.target_of(cam_state(), "nobody", v) is None
    for a in ("point", "circle", None, "sweep:sideways", "tour"):
        assert P.view_action(a, v) == a


def test_a_hand_written_view_is_applied_without_core():
    """The bridge's own affine, checked against literal numbers (not View.make)."""
    right = {"front": "right", "table": [60, 100], "m": [[0, -1, 60], [1, 0, 0]], "outline": False}
    c = P.compact_state(cam_state(), (100, 60), 5.0, right)
    assert c["table"] == [60.0, 100.0] and {x["n"]: x.get("xy") for x in c["e"]}["a"] == [55.0, 90.0]
    # an outline cropped 10 cm off the left and 5 off the top, seen from the camera's side
    crop = {"front": "bottom", "table": [80, 40], "m": [[1, 0, -10], [0, 1, -5]], "outline": True}
    c = P.compact_state(cam_state(), (100, 60), 5.0, crop, sides={"right": "couch", "bottom": "", "up": "x"})
    assert c["table"] == [80.0, 40.0] and c["view"] == {"f": "bottom", "o": True, "s": {"right": "couch"}}
    assert {x["n"]: x.get("xy") for x in c["e"]}["a"] == [80.0, 0.0]
    assert P.target_of(cam_state(), "b", crop) == [80.0, 0.0]
    assert [[round(x, 6) + 0.0 for x in r] for r in view("right")["m"]] == right["m"]   # the same numbers
    c = P.compact_state(cam_state(), (100, 60), 5.0, right, sides={"right": "couch"})
    assert c["view"]["s"] == {"right": "couch"}           # keyed by camera side, like "f"


def test_sides_are_labelled_by_camera_side_like_f():
    c = P.compact_state(cam_state(), (100, 60), 5.0, view("right"), sides={"right": "couch", "top": "window"})
    assert c["view"]["s"] == {"right": "couch", "top": "window"}


def test_no_view_or_a_bad_one_keeps_the_camera_frame():
    plain = P.compact_state(cam_state(), (100, 60), 5.0)
    assert "view" not in plain and plain["table"] == [100.0, 60.0]
    assert {x["n"]: x.get("xy") for x in plain["e"]}["a"] == [90.0, 5.0]
    assert {x["n"]: x.get("edge") for x in plain["e"]}["wallet"] == "left"
    for bad in (None, {}, {"front": "diagonal", "table": [1, 1], "m": [[1, 0, 0], [0, 1, 0]]},
                {"front": "right", "table": [60, 100], "m": [[0, -1], [1, 0]]},
                {"front": "right", "m": [[0, -1, 60], [1, 0, 0]]},
                {"front": "right", "table": [60, 100], "m": [[0, "x", 60], [1, 0, 0]]}):
        assert P.compact_state(cam_state(), (100, 60), 5.0, bad) == plain, bad
        assert P.target_of(cam_state(), "a", bad) == [90.0, 5.0]
        assert P.view_action("sweep:left", bad) == "sweep:left"


def test_state_changed_on_a_view_change():
    a = P.compact_state(cam_state(), (100, 60), 5.0, view("bottom"))
    assert not P.state_changed(a, P.compact_state(cam_state(), (100, 60), 6.0, view("bottom")))
    assert P.state_changed(a, P.compact_state(cam_state(), (100, 60), 5.0, view("right")))
    assert P.state_changed(a, P.compact_state(cam_state(), (100, 60), 5.0))              # view gone
    b = json.loads(json.dumps(a))
    b["view"]["o"] = True
    assert P.state_changed(a, b)
    b = json.loads(json.dumps(a))
    b["table"] = [99.0, 60.0]
    assert P.state_changed(a, b)


def test_the_bridge_sends_the_users_frame_when_the_app_publishes_a_view():
    core, http, phone, mono = make()
    http.meta = {"view": view("right", (90, 60)), "sides": {"right": "couch"}}
    core.poll_once()
    core.set_notifying("state", True)
    snap = phone.msgs["state"][-1]
    assert snap["table"] == [60.0, 90.0] and snap["view"] == {"f": "right", "o": False, "s": {"right": "couch"}}
    keys = next(e for e in snap["e"] if e["n"] == "keys")
    assert keys["xy"] == [31.0, 41.2] and keys["r"] == [21.9, 70.4]       # camera (41.234, 29.01), (70.449, 38.15)
    http.meta = {}                                                         # an older app: back to the camera frame
    mono.t += 1.0
    core.poll_once()
    snap = phone.msgs["state"][-1]
    assert "view" not in snap and snap["table"] == [90.0, 60.0]


def test_answers_and_room_answers_carry_the_users_target_and_sweep():
    core, http, phone, mono = make()
    http.meta = {"view": view("right", (90, 60)), "answers": []}
    core.set_notifying("answer", True)
    core.poll_once()
    msg = core.answer(1, "where are my keys?")
    assert msg["target"] == [21.9, 70.4] and msg["action"] == "point"
    http.reply = {"text": "The wallet was carried off the far side of the table.", "point_at": None,
                  "action": "sweep:top"}
    assert core.answer(2, "where is my wallet?")["action"] == "sweep:right"
    http.meta = dict(http.meta, answers=[dict(room_answer(1), action="sweep:left")])
    core.poll_once()
    [m] = phone.msgs["answer"]
    assert m["target"] == [21.9, 70.4] and m["action"] == "sweep:top"


def test_an_orient_write_turns_the_map_at_once_and_is_never_answered():
    core, http, phone, mono = make()
    core.poll_once()
    core.set_notifying("state", True)
    core.set_notifying("answer", True)
    assert "view" not in phone.msgs["state"][-1]
    sent = len(phone.msgs["state"])
    core.on_question(P.dumps({"orient": {"front": "right"}}))
    for t in [t for t in B.threading.enumerate() if t.name == "orient"]:
        t.join(2)
    assert http.oriented == [{"front": "right"}]
    assert len(phone.msgs["state"]) == sent + 1                            # right away, not at the next poll
    assert phone.msgs["state"][-1]["view"] == {"f": "right", "o": False}
    assert phone.msgs["state"][-1]["table"] == [60.0, 100.0]
    assert core.asks.empty() and phone.msgs["answer"] == [] and http.asked == []


# ---------------------------------------------------------------- the link fix (27 Sep): compression, pacing, cap

def room_rig_state(n_dupes: int = 150, now: float = 1790001000.0) -> dict:
    """The rig's /state at 00:42 on 27 Sep in shape: 100+ unnamed duplicates of one object piled on one
    spot, a few named props, most things long gone. It made 18-28 KB state messages."""
    st = big_state(0)
    for i in range(n_dupes):
        st["entities"].append({"name": f"thing:{3100 + i}", "kind": "target",
                               "status": ["VISIBLE", "UNKNOWN", "GONE"][i % 3], "pos_cm": [65.6, 27.0],
                               "resolved_cm": [65.6, 27.0], "confidence": 0.62, "last_seen": now - 300 + i,
                               "zone": "table", "aliases": [], "belief": [["charging cable", 0.61]]})
    return st


def test_deflate_round_trip_and_bad_input():
    raw = P.dumps(P.compact_state(room_rig_state(), (100, 60), 1790000100.0))
    z = P.deflate(raw)
    assert P.inflate(z) == raw and len(z) < len(raw) / 4
    with pytest.raises(ValueError):
        P.inflate(b"not deflate at all")
    with pytest.raises(ValueError):
        P.inflate(z[:len(z) // 2])                       # truncated
    with pytest.raises(ValueError):
        P.inflate(P.deflate(b"x" * 5000), max_bytes=1000)


def test_python_fixture_the_phone_test_uses():
    """ModelsTests/FramingTests on iOS inflate exactly these bytes."""
    import base64
    js = b'{"v":1,"t":1.0,"table":[90,60],"online":true,"laser":{"on":false,"target":null},"e":[]}'
    z = base64.b64decode("q1YqU7Iy1FEqAZJ6BkA6MSknVckq2tJAx8wgVkcpPy8nMw8oUFJUmqqjlJNYnFqkZFUNFFaySkvMKU4F6ShKTwVqzyvNyanVUQJpjq0FAA==")
    assert P.inflate(z) == js and P.deflate(js) == z


def test_compressed_frames_carry_the_flag_on_every_chunk_and_reassemble():
    raw = P.dumps(P.compact_state(room_rig_state(), (100, 60), 1790000100.0))
    chunks = P.frame(9, P.deflate(raw), 185, compressed=True)
    assert len(chunks) > 3 and all(c[2] & P.FLAG_Z for c in chunks)
    assert [c[2] & P.FLAG_FINAL for c in chunks] == [0] * (len(chunks) - 1) + [1]
    r = P.Reassembler()
    out = [r.feed(c) for c in chunks]
    assert out[-1] == raw and r.compressed == 1 and r.completed == 1 and r.chunks == len(chunks)
    assert P.frame(1, b"{}", 185) == [bytes((1, 0, 1)) + b"{}"]      # uncompressed: flags as before


def test_a_compressed_message_that_does_not_inflate_is_dropped():
    r = P.Reassembler()
    bad = P.frame(3, b"definitely not deflate", 185, compressed=True)
    assert [r.feed(c) for c in bad] == [None] and r.dropped == 1 and r.completed == 0
    assert r.feed(P.frame(4, b'{"ok":1}', 185)[0]) == b'{"ok":1}'


def test_parse_hello():
    assert P.parse_hello(P.dumps({"hello": {"z": 1}})) == {"z": True}
    assert P.parse_hello(P.dumps({"hello": {"z": True}})) == {"z": True}
    assert P.parse_hello(P.dumps({"hello": {}})) == {"z": False}
    for raw in (P.dumps({"q": "where are my keys?"}), P.dumps({"orient": {"front": "right"}}), b"nope",
                P.dumps({"hello": "hi"}), P.dumps({"hello": {"z": 1, "pad": "x" * 200}})):
        assert P.parse_hello(raw) is None, raw


def test_cap_state_cuts_the_least_useful_first():
    c = P.compact_state(room_rig_state(), (100, 60), 1790000100.0)
    assert len(P.dumps(c)) > 12_000                          # what the rig sent
    capped = P.cap_state(c, 12_000)
    assert len(P.dumps(capped)) <= 12_000 and capped["more"] == len(c["e"]) - len(capped["e"]) > 0
    kept = {e["n"] for e in capped["e"]}
    assert {"keys", "pill_bottle", "wallet", "box", "notebook"} <= kept      # named props always stay
    cut = [e for e in c["e"] if e["n"] not in kept]
    assert all(e["n"].startswith("thing:") for e in cut)
    # lost and gone duplicates go before visible ones
    assert sum(e["s"] == "V" for e in cut) == 0 or all(
        e["s"] == "V" for e in c["e"] if e["n"] in kept and e["n"].startswith("thing:"))
    small = P.compact_state(big_state(2), (90, 60), 1790000000.0)
    assert P.cap_state(small, 12_000) is small and "more" not in small


def test_pacer_limits_bytes_per_second():
    clock = Clock()
    pacer = P.Pacer(1000, 500, clock)
    assert pacer.take(500) and not pacer.take(1)             # the burst, then nothing
    clock.t += 0.2
    assert pacer.take(200) and not pacer.take(1)
    clock.t += 10
    assert pacer.take(500) and not pacer.take(1)             # never more than the burst saved up


def test_outbox_peek_is_what_pop_returns():
    ob = P.Outbox()
    assert ob.peek() is None
    ob.push("state", [b"s0", b"s1"])
    ob.push("answer", [b"a0"])
    assert ob.peek() == b"a0" and ob.pop(1) == [("answer", b"a0")]
    assert ob.peek() == b"s0"


def phone_with_z(core, phone, device="/dev_phone"):
    core.note_mtu(device, 185)
    core.on_question(P.dumps({"hello": {"z": 1}}), device)


def test_hello_turns_compression_on_for_that_phone_only():
    core, http, phone, mono = make()
    http.state = room_rig_state()
    core.poll_once()
    core.set_notifying("state", True)
    plain = phone.chunk_counts["state"][-1]
    assert not core.compress
    phone_with_z(core, phone)
    assert core.compress
    last = phone.msgs["state"][-1]
    assert phone.chunk_counts["state"][-1] < plain / 3             # the same snapshot, compressed
    assert phone.asm["state"].compressed >= 1 and last["e"] and "tx" in last
    assert not phone.msgs["answer"]                                # a hello is never answered
    core.note_mtu("/dev_mac", 185)                                 # a second central that never said hello
    assert not core.compress
    core.forget_device("/dev_mac")
    assert core.compress
    core.forget_device("/dev_phone")
    assert not core.compress


def test_the_rigs_28_kb_state_goes_out_small():
    """The regression: 150 duplicates made 28 KB states (about 55 chunks at MTU 517, 155 at 185) at 1.5 Hz.
    Capped and compressed, one state is a handful of chunks."""
    core, http, phone, mono = make()
    http.state = room_rig_state()
    phone_with_z(core, phone)
    core.poll_once()
    core.set_notifying("state", True)
    assert core.stats["state_bytes"] < 3000 and phone.chunk_counts["state"][-1] <= 17
    assert phone.msgs["state"][-1]["more"] > 0


def test_a_state_waits_while_the_last_is_still_going_out():
    core, http, phone, mono = make()
    core.poll_once()
    core.set_notifying("state", True)
    n = len(phone.msgs["state"])
    queued = {"state": 5}
    core.pending = lambda char: queued.get(char, 0)
    http.state["entities"][0]["status"] = "HELD"
    mono.t += 1.0
    core.poll_once()
    assert len(phone.msgs["state"]) == n and core.stats["deferred"] >= 1
    queued["state"] = 0
    core.poll_once()
    assert len(phone.msgs["state"]) == n + 1 and phone.msgs["state"][-1]["e"]


def test_tx_counts_the_state_chunks_sent_before_each_message():
    core, http, phone, mono = make()
    core.poll_once()
    core.set_notifying("state", True)
    for i in range(3):
        http.state["entities"][0]["pos_cm"] = [10.0 + 5 * i, 10.0]
        mono.t += 1.0
        core.poll_once()
    txs = [m["tx"] for m in phone.msgs["state"]]
    counts = phone.chunk_counts["state"]
    assert txs[0] == 0 and all(txs[i + 1] - txs[i] == counts[i] for i in range(len(txs) - 1))


def test_room_layout_goes_to_the_phone_when_it_changes():
    core, http, phone, mono = make()
    http.layout = {"v": 1, "size": [400, 300], "front": "right", "table": {"rect": [100, 100, 73, 100]},
                   "zones": [{"id": "couch", "say": "the couch", "rect": [0, 250, 200, 50], "kind": "seat"}],
                   "you": [136, 290]}
    core.poll_once()
    core.set_notifying("state", True)
    first = phone.msgs["state"][-1]
    assert first["lay"] == http.layout and first["lh"]
    mono.t += 6.0                                         # the next heartbeat carries only the hash
    core.poll_once()
    assert "lay" not in phone.msgs["state"][-1] and phone.msgs["state"][-1]["lh"] == first["lh"]
    http.layout["zones"][0]["say"] = "the sofa"
    mono.t += 31.0                                        # re-fetched every 30 s
    core.poll_once()
    assert phone.msgs["state"][-1]["lay"]["zones"][0]["say"] == "the sofa"
    assert phone.msgs["state"][-1]["lh"] != first["lh"]
    http.layout = None                                    # 404: no room map
    mono.t += 31.0
    core.poll_once()
    assert "lh" not in phone.msgs["state"][-1] and "lay" not in phone.msgs["state"][-1]


def test_pump_paces_chunks_and_keeps_priority():
    clock = Clock()
    ob, sent = P.Outbox(), []
    pacer = P.Pacer(1000, 300, clock)                     # 1 KB/s, a 300 B burst
    ob.push("state", [bytes(100)] * 10)
    assert P.pump(ob, pacer, lambda c, v: sent.append(c), 4) == 3     # the burst: 3 chunks of 100 B
    ob.push("answer", [bytes(50)])
    clock.t += 0.06
    assert P.pump(ob, pacer, lambda c, v: sent.append(c), 4) == 1 and sent[-1] == "answer"   # jumps the queue
    for _ in range(100):                                   # 10 ms ticks for a second: ~1000 B more
        clock.t += 0.01
        P.pump(ob, pacer, lambda c, v: sent.append(c), 4)
    assert sent.count("state") == 10 and ob.pending() == 0
