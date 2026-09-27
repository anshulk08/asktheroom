"""Ask the Room BLE protocol: UUIDs, notify framing, and the JSON messages (mobile/PROTOCOL.md).

Pure Python 3.10, standard library only: imported by the Jetson bridge (ble_bridge.py), the Mac
test client (test_client.py) and tests/test_mobile_protocol.py.

Framing (every NOTIFY on answer / state / status):
    byte 0  msg_id       u8, per characteristic, increments per message and wraps 255 -> 0
    byte 1  chunk_index  u8, 0, 1, 2, ... within the message
    byte 2  flags        u8, bit0 = FINAL (last chunk of the message); bit1 = COMPRESSED (set on every chunk of
                         a message whose bytes are raw DEFLATE of the JSON: only to a central that said
                         {"hello": {"z": 1}}); bits 2-7 are 0
    3..     payload      UTF-8 JSON bytes, at most MTU - 3 (ATT) - 3 (this header) per chunk
A message is at most 256 chunks. A receiver keeps one partial message per characteristic; a chunk
with a different msg_id discards any incomplete previous message.

The user's frame (PROTOCOL.md 5b, 7): GET /state carries "view" (core/viewframe.py View.to_json()), and
positions, sizes, edges and sweep actions sent to the phone are turned into it here with the plain affine
"m" (the bridge's host has no numpy and no repo package, so this does not import core.viewframe).
"""
from __future__ import annotations

import json
import math
import re
import zlib
from typing import Any, Callable, Optional

# ---------------------------------------------------------------- UUIDs and names

LOCAL_NAME = "AskTheRoom"
SERVICE_UUID = "8a1e0001-6b7f-4c2b-9e3a-2f5d7c1a0001"
QUESTION_UUID = "8a1e0002-6b7f-4c2b-9e3a-2f5d7c1a0002"
ANSWER_UUID = "8a1e0003-6b7f-4c2b-9e3a-2f5d7c1a0003"
STATE_UUID = "8a1e0004-6b7f-4c2b-9e3a-2f5d7c1a0004"
STATUS_UUID = "8a1e0005-6b7f-4c2b-9e3a-2f5d7c1a0005"

HEADER_LEN = 3
ATT_OVERHEAD = 3
FLAG_FINAL = 0x01
FLAG_Z = 0x02
MAX_INFLATED = 256 * 1024
MAX_CHUNKS = 256
DEFAULT_MTU = 185            # iOS negotiates ~185; used until a read/write tells us the real one
MIN_MTU = 23                 # the ATT minimum
MAX_QUESTION_BYTES = 180

STATUS_LETTER = {"VISIBLE": "V", "HELD": "H", "UNDER": "U", "INSIDE": "I", "GONE": "G", "UNKNOWN": "X"}
KIND_LETTER = {"target": "t", "container": "c", "cover": "v"}

FRONTS = ("bottom", "right", "top", "left")
# Camera edge -> viewer edge, per front (the same table as core/viewframe.py _EDGE): the side the user sits
# at becomes 'bottom', nearest them.
VIEW_EDGE = {
    "bottom": {"left": "left", "right": "right", "top": "top", "bottom": "bottom"},
    "top": {"left": "right", "right": "left", "top": "bottom", "bottom": "top"},
    "right": {"right": "bottom", "left": "top", "bottom": "left", "top": "right"},
    "left": {"left": "bottom", "right": "top", "top": "left", "bottom": "right"},
}

ROOM_DOWN_TEXT = "The room isn't running right now."
TOO_LONG_TEXT = "Question too long."


# ---------------------------------------------------------------- framing

def chunk_payload_size(mtu: int) -> int:
    """JSON bytes per notification for a negotiated ATT MTU."""
    return max(1, int(mtu) - ATT_OVERHEAD - HEADER_LEN)


def deflate(data: bytes) -> bytes:
    """Raw DEFLATE (RFC 1951, no zlib header): what the phone inflates with Apple's .zlib algorithm."""
    c = zlib.compressobj(9, zlib.DEFLATED, -15)
    return c.compress(data) + c.flush()


def inflate(data: bytes, max_bytes: int = MAX_INFLATED) -> bytes:
    """deflate()'s inverse; ValueError when it is not raw DEFLATE or inflates past max_bytes."""
    d = zlib.decompressobj(-15)
    try:
        out = d.decompress(data, max_bytes + 1)
    except zlib.error as ex:
        raise ValueError(f"not raw DEFLATE: {ex}") from None
    if len(out) > max_bytes or d.unconsumed_tail:
        raise ValueError("inflated message too big")
    if not d.eof:
        raise ValueError("truncated DEFLATE stream")
    return out


def frame(msg_id: int, payload: bytes, mtu: int = DEFAULT_MTU, compressed: bool = False) -> list[bytes]:
    """Split one message into notification values, each <= mtu - 3 bytes. compressed: payload is deflate()d
    (FLAG_Z on every chunk)."""
    size = chunk_payload_size(mtu)
    n = max(1, math.ceil(len(payload) / size))
    if n > MAX_CHUNKS:
        raise ValueError(f"message of {len(payload)} bytes needs {n} chunks (max {MAX_CHUNKS}) at MTU {mtu}")
    mid = msg_id & 0xFF
    out = []
    for i in range(n):
        flags = (FLAG_FINAL if i == n - 1 else 0) | (FLAG_Z if compressed else 0)
        out.append(bytes((mid, i, flags)) + payload[i * size:(i + 1) * size])
    return out


def parse_header(value: bytes) -> tuple[int, int, bool, bytes]:
    if len(value) < HEADER_LEN:
        raise ValueError("notification shorter than the 3-byte header")
    return value[0], value[1], bool(value[2] & FLAG_FINAL), bytes(value[HEADER_LEN:])


class Reassembler:
    """One per characteristic, on the receiving side (the phone; test_client.py).

    feed(value) -> the complete message bytes when a FINAL chunk completes a message, else None.
    Rules: a chunk with a new msg_id starts over (drops an incomplete previous message); chunks
    must arrive in order (0, 1, 2, ... as BLE notifications on one link do): a gap, duplicate or a
    chunk_index > 0 with no message in progress drops the partial message until the next chunk 0.
    """

    def __init__(self, max_bytes: int = 64 * 1024):
        self.max_bytes = max_bytes
        self.msg_id: Optional[int] = None
        self.next_index = 0
        self.parts: list[bytes] = []
        self.size = 0
        self.dropped = 0             # messages abandoned (new id before FINAL, gap, overflow, bad DEFLATE)
        self.completed = 0
        self.compressed = 0          # completed messages that came deflated
        self.chunks = 0              # notification values fed
        self.z = False               # the message in progress is deflated

    def _reset(self) -> None:
        self.msg_id, self.next_index, self.parts, self.size = None, 0, [], 0

    def feed(self, value: bytes) -> Optional[bytes]:
        try:
            mid, idx, final, payload = parse_header(value)
        except ValueError:
            return None
        self.chunks += 1
        if self.msg_id is not None and mid != self.msg_id:
            self.dropped += 1                      # new message before the old one finished
            self._reset()
        if idx == 0:
            if self.msg_id == mid and self.parts:   # a repeated chunk 0 of the same id: restart
                self.dropped += 1
            self._reset()
            self.msg_id = mid
            self.z = bool(value[2] & FLAG_Z)
        elif self.msg_id != mid or idx != self.next_index:
            if self.msg_id is not None:
                self.dropped += 1
            self._reset()                           # gap or stray chunk: wait for the next chunk 0
            return None
        self.parts.append(payload)
        self.size += len(payload)
        self.next_index = idx + 1
        if self.size > self.max_bytes or (not final and self.next_index >= MAX_CHUNKS):
            self.dropped += 1
            self._reset()
            return None
        if final:
            msg, z = b"".join(self.parts), self.z
            self._reset()
            if z:
                try:
                    msg = inflate(msg)
                except ValueError:
                    self.dropped += 1
                    return None
                self.compressed += 1
            self.completed += 1
            return msg
        return None


def dumps(obj: Any) -> bytes:
    return json.dumps(obj, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


# ---------------------------------------------------------------- question / answer

def parse_voice(value: bytes) -> Optional[dict]:
    """A voice-settings write ({"voice": {"e", "v", "s"}}, PROTOCOL.md 5a) -> the app's POST /voice body
    {"engine", "grok_voice", "speed"}, or None when the write is not one (a question)."""
    if len(value) > MAX_QUESTION_BYTES:
        return None
    try:
        obj = json.loads(bytes(value).decode("utf-8"))
    except (ValueError, UnicodeDecodeError):
        return None
    v = obj.get("voice") if isinstance(obj, dict) else None
    if not isinstance(v, dict):
        return None
    return {"engine": v.get("e"), "grok_voice": v.get("v"), "speed": v.get("s")}


def parse_hello(value: bytes) -> Optional[dict]:
    """The phone's hello on connect ({"hello": {"z": 1}}, PROTOCOL.md 5c) -> {"z": bool: it inflates FLAG_Z
    messages}, or None when the write is not one."""
    if len(value) > MAX_QUESTION_BYTES:
        return None
    try:
        obj = json.loads(bytes(value).decode("utf-8"))
    except (ValueError, UnicodeDecodeError):
        return None
    h = obj.get("hello") if isinstance(obj, dict) else None
    if not isinstance(h, dict):
        return None
    return {"z": h.get("z") in (1, True)}


def parse_orient(value: bytes) -> Optional[dict]:
    """An orientation write ({"orient": {"front": "right"}}, PROTOCOL.md 5b) -> the app's POST /orientation
    body {"front"} (None: back to the configured seat), {} for an orient write naming no valid side (ignored),
    or None when the write is not one (a question)."""
    if len(value) > MAX_QUESTION_BYTES:
        return None
    try:
        obj = json.loads(bytes(value).decode("utf-8"))
    except (ValueError, UnicodeDecodeError):
        return None
    o = obj.get("orient") if isinstance(obj, dict) else None
    if not isinstance(o, dict):
        return None
    if "front" in o and o["front"] is None:
        return {"front": None}                # "use the rig's default": the app goes back to its configured seat
    front = str(o.get("front")).strip().lower()
    return {"front": front} if front in FRONTS else {}


def parse_question(value: bytes) -> tuple[int, Optional[str], Optional[str]]:
    """question write -> (id, text, error_text). error_text is set when it must be rejected."""
    qid = 0
    try:
        obj = json.loads(bytes(value).decode("utf-8"))
        if isinstance(obj, dict):
            raw_id = obj.get("id", 0)
            if isinstance(raw_id, (int, float)) and not isinstance(raw_id, bool):
                qid = int(raw_id) & 0xFFFF
    except (ValueError, UnicodeDecodeError):
        obj = None
    if len(value) > MAX_QUESTION_BYTES:
        return qid, None, TOO_LONG_TEXT
    if not isinstance(obj, dict):
        return qid, None, "I couldn't read that question."
    q = obj.get("q")
    if not isinstance(q, str) or not q.strip():
        return qid, None, "Ask me where something is, like: where are my keys?"
    return qid, q.strip(), None


def answer_msg(qid: int, ok: bool, text: str, point_at: Optional[str] = None,
               action: Optional[str] = None, target: Optional[list] = None, ms: int = 0) -> dict:
    return {"id": int(qid), "ok": bool(ok), "text": str(text), "point_at": point_at, "action": action,
            "target": target, "ms": int(ms)}


def room_msg(src: str, text: str, point_at: Optional[str] = None, action: Optional[str] = None,
             target: Optional[list] = None, **extra) -> dict:
    """An answer the phone didn't ask for (id null): a question asked in the room (src 'voice',
    'dashboard', ..., with 'q') or a reminder notice (src 'notice', with 'nid' and 'kind')."""
    return {"id": None, "src": src, **extra, "ok": True, "text": str(text), "point_at": point_at,
            "action": action, "target": target, "ms": None}


def target_of(state: Optional[dict], name: Optional[str], view: Optional[dict] = None) -> Optional[list]:
    """Resolved table-cm position of an entity in a WorldState (falls back to its last-seen spot); in the
    user's frame when view (GET /state's "view") is given."""
    if not name or not isinstance(state, dict):
        return None
    v = view_of(view)
    for e in state.get("entities") or []:
        if isinstance(e, dict) and e.get("name") == name:
            return _vxy(v, e.get("resolved_cm")) or _vxy(v, e.get("pos_cm"))
    return None


def view_action(action: Optional[str], view: Optional[dict] = None) -> Optional[str]:
    """An answer's action for the phone: 'sweep:<camera edge>' becomes 'sweep:<viewer edge>' (the laser
    already swept the camera edge in the app); anything else is unchanged."""
    v = view_of(view)
    if v is None or not isinstance(action, str) or not action.startswith("sweep:"):
        return action
    e = VIEW_EDGE[v["front"]].get(action[len("sweep:"):])
    return f"sweep:{e}" if e else action


# ---------------------------------------------------------------- the user's frame

def view_of(view: Any) -> Optional[dict]:
    """GET /state's "view" ({"front", "table", "m", "outline"}, core/viewframe.py) checked -> {"front", "table",
    "m", "outline"} with floats, or None (an older app without one, or anything malformed: camera frame)."""
    if not isinstance(view, dict):
        return None
    front = str(view.get("front") or "").strip().lower()
    m, t = view.get("m"), view.get("table")
    if front not in FRONTS or not isinstance(m, (list, tuple)) or len(m) != 2:
        return None
    try:
        rows = [[float(x) for x in r] for r in m]
        tw, th = float(t[0]), float(t[1])
    except (TypeError, ValueError, IndexError, KeyError):
        return None
    if any(len(r) != 3 for r in rows) or any(math.isnan(x) or math.isinf(x) for r in rows for x in r):
        return None
    return {"front": front, "table": (tw, th), "m": rows, "outline": bool(view.get("outline"))}


def _vxy(v: Optional[dict], p: Any) -> Optional[list]:
    """A camera-frame [x, y] -> [x, y] cm, 1 decimal: the user's with v (as checked by view_of), else as it is;
    None when p isn't a finite point."""
    q = _xy(p)
    if v is None or q is None:
        return q
    x, y = float(p[0]), float(p[1])            # finite: _xy checked them
    (a, b, c), (d, e, f) = v["m"]
    return _xy((a * x + b * y + c, d * x + e * y + f))


# ---------------------------------------------------------------- state

def _r1(v: Any) -> Optional[float]:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if math.isnan(f) or math.isinf(f):
        return None
    return round(f, 1)


def _xy(p: Any) -> Optional[list]:
    if not isinstance(p, (list, tuple)) or len(p) < 2:
        return None
    x, y = _r1(p[0]), _r1(p[1])
    if x is None or y is None:
        return None
    return [x, y]


def _conf2(v: Any) -> Optional[float]:
    """A confidence rounded to 2 dp, or None when it isn't a finite number."""
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return None
    if math.isnan(v) or math.isinf(v):
        return None
    return round(float(v), 2)


REGISTRY_STATES = ("visible", "hidden", "carried", "last_seen", "unknown")   # core/permanence.py
NOT_OBJECT = "not an object"          # core/grok_check.py NOT_OBJECT: the belief's clutter label


def _best_guess(e: dict) -> tuple[Optional[str], Optional[float]]:
    """(g, gc): the automatic guess (core/auto_name.py 'guess') or Grok's fused belief (its top real label); with both,
    the one with the higher confidence (a missing confidence loses; a tie keeps the guess)."""
    opts = []
    g = e.get("guess")
    if isinstance(g, dict) and g.get("name"):
        opts.append((str(g["name"]), _conf2(g.get("confidence"))))
    b = e.get("belief")
    top = next((x for x in (b if isinstance(b, list) else []) if isinstance(x, (list, tuple)) and x and x[0]
                and x[0] != NOT_OBJECT), None)        # the top real label: clutter is never a guess
    if top is not None:
        opts.append((str(top[0]), _conf2(top[1]) if len(top) > 1 else None))
    if not opts:
        return None, None
    return max(opts, key=lambda o: -1.0 if o[1] is None else o[1])   # max keeps the first on a tie


def compact_entity(e: dict, view: Optional[dict] = None) -> dict:
    """One WorldState entity -> the compact phone form. Unknown extra fields are ignored. With view (as
    checked by view_of) xy, r and edge are the user's."""
    status = str(e.get("status") or "UNKNOWN")
    out: dict = {"n": str(e.get("name")),
                 "k": KIND_LETTER.get(str(e.get("kind")), "t"),
                 "s": STATUS_LETTER.get(status, "X")}
    if e.get("parent"):
        out["p"] = str(e["parent"])
    xy = _vxy(view, e.get("pos_cm"))
    if xy:
        out["xy"] = xy
    r = _vxy(view, e.get("resolved_cm"))
    if r:
        out["r"] = r
    try:
        c = round(float(e.get("confidence")), 2)
        if not math.isnan(c):
            out["c"] = c
    except (TypeError, ValueError):
        pass
    if e.get("edge"):
        edge = str(e["edge"])
        out["edge"] = VIEW_EDGE[view["front"]].get(edge, edge) if view else edge
    zone = e.get("zone")
    if zone and zone != "table":                          # room memory (spec 0009/0010): the room zone it is in
        out["z"] = str(zone)
    is_thing = str(e.get("name")).startswith("thing:")
    if is_thing:                                           # open-world things only
        aliases = e.get("aliases")
        out["a"] = [str(a) for a in aliases] if isinstance(aliases, list) else []
    m = e.get("maybe_same_as")
    if isinstance(m, list) and m:
        mm = []
        for pair in m:
            if isinstance(pair, (list, tuple)) and len(pair) >= 2:
                try:
                    mm.append([str(pair[0]), round(float(pair[1]), 2)])
                except (TypeError, ValueError):
                    continue
        if mm:
            out["m"] = mm
    g, gc = _best_guess(e)
    if g:
        out["g"] = g
        if gc is not None:
            out["gc"] = gc
    if is_thing and e.get("named_by") == "grok":            # aliases[0] bound by the Grok settle check
        out["as"] = "grok"
    ls = _r1(e.get("last_seen"))
    if ls is not None:
        out["ls"] = ls
    reg = e.get("registry")                                # object permanence (spec 0011): the registry's state
    if isinstance(reg, dict) and reg.get("state") in REGISTRY_STATES:
        out["rg"] = str(reg["state"])
        if reg.get("tentative"):
            out["rt"] = 1
    return out


STALE_THING_S = 600.0


def _stale_thing(e: dict, now: float) -> bool:
    """An unnamed thing:N that is GONE / UNKNOWN and was last seen over STALE_THING_S ago: clutter on
    the phone, so it is left out. Named things and configured objects are always sent."""
    if not str(e.get("name")).startswith("thing:") or e.get("aliases") or isinstance(e.get("registry"), dict):
        return False                                       # named, configured, or in the registry: always sent
    if str(e.get("status") or "UNKNOWN") not in ("GONE", "UNKNOWN"):
        return False
    ls = _r1(e.get("last_seen"))
    return ls is not None and now - ls > STALE_THING_S


def compact_state(state: Optional[dict], table_cm: tuple, now: float, view: Optional[dict] = None,
                  sides: Optional[dict] = None) -> dict:
    """WorldState (GET /state's "state") -> the state notification JSON. state=None: an empty room.

    view: GET /state's "view" (core/viewframe.py). With one, positions, edges and "table" are in the user's
    frame (x their left to right, y far to near, 'bottom' the edge nearest them) and "view" says so:
    {"f": front, "o": outline, "s": {camera side: label}} (GET /state's "sides", keyed like "f" by camera
    side, only when there are any). Without one (an older app), the camera frame and table_cm as before."""
    st = state if isinstance(state, dict) else {}
    v = view_of(view)
    laser = st.get("laser") if isinstance(st.get("laser"), dict) else {}
    las = {"on": bool(laser.get("on")), "target": str(laser["target"]) if laser.get("target") else None}
    ents = [compact_entity(e, v) for e in (st.get("entities") or [])
            if isinstance(e, dict) and e.get("name") and not _stale_thing(e, now)]
    size = v["table"] if v else table_cm
    out = {"v": 1, "t": round(now, 1), "table": [_r1(size[0]), _r1(size[1])],
           "online": bool(st.get("online")), "laser": las, "e": ents}
    if v:
        out["view"] = {"f": v["front"], "o": v["outline"]}
        edges = VIEW_EDGE[v["front"]]
        lab = {str(k): str(val) for k, val in (sides.items() if isinstance(sides, dict) else ())
               if str(k) in edges and val}
        if lab:
            out["view"]["s"] = lab
    return out


SIGHTING_TTL_S = 1800.0
SIGHTING_MAX = 8
SIGHTING_KINDS = ("look", "recall")       # answer evidence kinds (core/evidence.py) that come from Grok seeing
_NEGATIVE = re.compile(r"\b(don't|do not|can't|cannot|couldn't|could not|no longer|not|isn't|aren't|nowhere|"
                       r"no sign|haven't|didn't)\b", re.I)


def zone_phrases(zones) -> list:
    """[(zone id, [phrases])] from room layout zones ({"id", "say"}) or room_zones.json-style {id: {"say"}}:
    the say without a leading "the" ("floor by the doorway") and the id in words ("doorway")."""
    items = zones.items() if isinstance(zones, dict) else ((z.get("id"), z) for z in zones or [] if isinstance(z, dict))
    out = []
    for zid, z in items:
        if not zid:
            continue
        say = str((z or {}).get("say") or "").strip().lower()
        say = say[4:] if say.startswith("the ") else say
        words = {p for p in (say, str(zid).replace("_", " ").lower()) if p}
        out.append((str(zid), sorted(words, key=len, reverse=True)))
    return out


def sighting_of(row: dict, phrases: list) -> Optional[tuple]:
    """(name, zone id, t, source) when an answer says Grok saw an object in a room zone ("I see glasses on the
    couch": evidence kind look or recall, the answer's obj, a zone named in its text), else None. A negative
    answer ("I don't see your glasses") is never a sighting."""
    if not isinstance(row, dict) or not row.get("obj") or not isinstance(row.get("text"), str):
        return None
    kinds = [e.get("kind") for e in row.get("evidence") or [] if isinstance(e, dict)]
    src = next((k for k in kinds if k in SIGHTING_KINDS), None)
    text = row["text"].lower()
    if src is None or _NEGATIVE.search(text):
        return None
    name = str(row["obj"])
    at = text.find(name.replace("_", " ").lower())
    best = None
    for zid, words in phrases:
        for w in words:
            for m in re.finditer(rf"\b{re.escape(w)}\b", text):
                key = (m.start() < at, m.start())          # after the object's name first, then earliest
                if best is None or key < best[0]:
                    best = (key, zid)
    if best is None:
        return None
    t = row.get("t")
    return name, best[1], float(t) if isinstance(t, (int, float)) and not isinstance(t, bool) else None, src


def sightings_msg(sightings: dict, now: float, ttl: float = SIGHTING_TTL_S, cap: int = SIGHTING_MAX) -> list:
    """{name: (zone, t, src)} -> the state's "sg": [[name, zone, t, src], ...] newest first, fresh ones only."""
    rows = [[n, z, round(t, 1), src] for n, (z, t, src) in sightings.items() if now - t <= ttl]
    return sorted(rows, key=lambda r: -r[2])[:cap]


STATE_MAX_BYTES = 12_000


def _cut_rank(e: dict) -> tuple:
    """Which entity goes first when a state is over budget (lower first): unnamed things, then ones with only
    a guess, each lost or gone first, then hidden, then visible, oldest first; named things, configured
    objects and registry objects only after all of them."""
    thing = str(e.get("n", "")).startswith("thing:")
    named = not thing or bool(e.get("a")) or "rg" in e      # configured, taught or in the registry
    tier = 2 if named else 1 if e.get("g") else 0           # a Grok guess alone doesn't protect a duplicate
    status = {"X": 0, "G": 0, "U": 1, "I": 1, "H": 2, "V": 2}.get(e.get("s"), 0)
    return (tier, status, float(e.get("ls") or 0.0))


def cap_state(msg: dict, max_bytes: int = STATE_MAX_BYTES) -> dict:
    """A state notification no bigger than max_bytes of JSON: the least useful entities (_cut_rank) are
    left out and "more" says how many (the phone can say "and N more"). Keeps the link from carrying
    100 duplicate things at 1.5 Hz (rig, 27 Sep: 28 KB states, the phone dropped every minute)."""
    if len(dumps(msg)) <= max_bytes:
        return msg
    ents = list(msg.get("e") or [])
    order = sorted(range(len(ents)), key=lambda i: _cut_rank(ents[i]))
    sizes = [len(dumps(e)) + 1 for e in ents]
    over = len(dumps(msg)) - max_bytes + 16          # room for "more"
    drop = set()
    for i in order:
        if over <= 0:
            break
        drop.add(i)
        over -= sizes[i]
    out = dict(msg, e=[e for i, e in enumerate(ents) if i not in drop])
    out["more"] = len(drop)
    return out


POS_DEADBAND_CM = 0.5
CONF_DEADBAND = 0.05


def _moved(a: Optional[list], b: Optional[list]) -> bool:
    if (a is None) != (b is None):
        return True
    if a is None:
        return False
    return math.hypot(a[0] - b[0], a[1] - b[1]) > POS_DEADBAND_CM


def state_changed(prev: Optional[dict], cur: dict) -> bool:
    """True when cur differs from the last SENT state in a way the phone should see. Ignores the
    timestamps and detector jitter (positions within 0.5 cm, confidence and guess confidence within
    0.05)."""
    if prev is None:
        return True
    for k in ("table", "online", "laser", "view", "lh", "sg"):
        if prev.get(k) != cur.get(k):
            return True
    pe = {e["n"]: e for e in prev.get("e", [])}
    ce = {e["n"]: e for e in cur.get("e", [])}
    if pe.keys() != ce.keys():
        return True
    for n, c in ce.items():
        p = pe[n]
        for k in ("k", "s", "p", "edge", "a", "m", "g", "as", "rg", "rt"):
            if p.get(k) != c.get(k):
                return True
        if _moved(p.get("xy"), c.get("xy")) or _moved(p.get("r"), c.get("r")):
            return True
        if abs(float(p.get("c", 0)) - float(c.get("c", 0))) > CONF_DEADBAND:
            return True
        if ("gc" in p) != ("gc" in c) or abs(float(p.get("gc", 0)) - float(c.get("gc", 0))) > CONF_DEADBAND:
            return True
        if c.get("s") != "V" and p.get("ls") != c.get("ls"):
            return True       # a hidden object's last-seen time changing is news; a visible one's isn't
    return False


# ---------------------------------------------------------------- status

def status_msg(app_up: bool, state: Optional[dict], cal: bool, laser_cal: bool) -> dict:
    st = state if isinstance(state, dict) else {}
    fps = _r1(st.get("fps")) if app_up else 0.0
    online = bool(st.get("online")) if app_up else False
    gc = st.get("grok_check")                 # GrokCheck.status(); only present when the check is on
    gk = online and isinstance(gc, dict) and bool(gc.get("enabled", True))
    spk = app_up and isinstance(st.get("speaker"), dict) and bool(st["speaker"].get("ok"))
    return {"app": "up" if app_up else "down", "fps": fps or 0.0, "online": online,
            "cal": bool(cal), "laser_cal": bool(laser_cal), "gk": gk, "spk": spk}


def status_changed(prev: Optional[dict], cur: dict, fps_step: float = 1.0) -> bool:
    if prev is None:
        return True
    for k in ("app", "online", "cal", "laser_cal", "gk", "spk"):
        if prev.get(k) != cur.get(k):
            return True
    return abs(float(prev.get("fps") or 0) - float(cur.get("fps") or 0)) >= fps_step


# ---------------------------------------------------------------- outbound queue

class Outbox:
    """Notification chunks waiting to go out, drained by the bridge's GLib timer.

    Priority answer > status > state. For "latest wins" characteristics (state, status) a newly
    pushed message replaces queued messages of that characteristic that have not started sending;
    a message already partly sent is finished first, so the phone never sees a torn message.
    """

    ORDER = ("answer", "status", "state")
    LATEST_WINS = {"state", "status"}

    def __init__(self):
        self.queues: dict[str, list[list[bytes]]] = {c: [] for c in self.ORDER}
        self.started: dict[str, bool] = {c: False for c in self.ORDER}   # head message partly sent
        self.replaced = 0

    def push(self, char: str, chunks: list[bytes]) -> None:
        q = self.queues[char]
        if char in self.LATEST_WINS and q:
            keep = q[:1] if self.started[char] else []
            self.replaced += len(q) - len(keep)
            q[:] = keep
        q.append(list(chunks))

    def peek(self) -> Optional[bytes]:
        """The chunk pop(1) would return next, or None."""
        for c in self.ORDER:
            if self.queues[c]:
                return self.queues[c][0][0]
        return None

    def pending(self, char: Optional[str] = None) -> int:
        chars = [char] if char else list(self.ORDER)
        return sum(len(m) for c in chars for m in self.queues[c])

    def clear(self, char: str) -> None:
        self.queues[char] = []
        self.started[char] = False

    def pop(self, n: int) -> list[tuple[str, bytes]]:
        out: list[tuple[str, bytes]] = []
        for c in self.ORDER:
            q = self.queues[c]
            while q and len(out) < n:
                msg = q[0]
                out.append((c, msg.pop(0)))
                self.started[c] = True
                if not msg:
                    q.pop(0)
                    self.started[c] = False
            if len(out) >= n:
                break
        return out


class Pacer:
    """A token bucket over notification bytes: the pump sends a chunk only when take(len) allows it, so
    chunks leave at the link's pace instead of piling up unbounded inside bluetoothd, where no newer
    state can replace them (the Outbox's latest-wins only works on what is still ours)."""

    def __init__(self, rate_bps: float, burst: float, clock: Callable[[], float]):
        self.rate, self.burst, self.clock = float(rate_bps), float(burst), clock
        self.tokens, self.t = float(burst), clock()

    def take(self, n: int) -> bool:
        now = self.clock()
        self.tokens = min(self.burst, self.tokens + (now - self.t) * self.rate)
        self.t = now
        if self.tokens < n:
            return False
        self.tokens -= n
        return True


def pump(outbox: "Outbox", pacer: Pacer, send: Callable[[str, bytes], None], limit: int) -> int:
    """One pump tick: at most `limit` chunks, each only when the pacer allows its bytes, in the Outbox's
    priority order; send(char, value) notifies. Returns how many chunks left the Outbox."""
    n = 0
    while n < limit:
        nxt = outbox.peek()
        if nxt is None or not pacer.take(len(nxt)):
            break
        (char, value), = outbox.pop(1)
        send(char, value)
        n += 1
    return n


class MsgCounter:
    """Per-characteristic msg_id, 0..255 wrapping."""

    def __init__(self):
        self.ids: dict[str, int] = {}

    def next(self, char: str) -> int:
        i = self.ids.get(char, -1) + 1 & 0xFF
        self.ids[char] = i
        return i


Clock = Callable[[], float]
