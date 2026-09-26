"""Ask the Room BLE protocol: UUIDs, notify framing, and the JSON messages (mobile/PROTOCOL.md).

Pure Python 3.10, standard library only: imported by the Jetson bridge (ble_bridge.py), the Mac
test client (test_client.py) and tests/test_mobile_protocol.py.

Framing (every NOTIFY on answer / state / status):
    byte 0  msg_id       u8, per characteristic, increments per message and wraps 255 -> 0
    byte 1  chunk_index  u8, 0, 1, 2, ... within the message
    byte 2  flags        u8, bit0 = FINAL (last chunk of the message); bits 1-7 are 0
    3..     payload      UTF-8 JSON bytes, at most MTU - 3 (ATT) - 3 (this header) per chunk
A message is at most 256 chunks. A receiver keeps one partial message per characteristic; a chunk
with a different msg_id discards any incomplete previous message.
"""
from __future__ import annotations

import json
import math
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
MAX_CHUNKS = 256
DEFAULT_MTU = 185            # iOS negotiates ~185; used until a read/write tells us the real one
MIN_MTU = 23                 # the ATT minimum
MAX_QUESTION_BYTES = 180

STATUS_LETTER = {"VISIBLE": "V", "HELD": "H", "UNDER": "U", "INSIDE": "I", "GONE": "G", "UNKNOWN": "X"}
KIND_LETTER = {"target": "t", "container": "c", "cover": "v"}

ROOM_DOWN_TEXT = "The room isn't running right now."
TOO_LONG_TEXT = "Question too long."


# ---------------------------------------------------------------- framing

def chunk_payload_size(mtu: int) -> int:
    """JSON bytes per notification for a negotiated ATT MTU."""
    return max(1, int(mtu) - ATT_OVERHEAD - HEADER_LEN)


def frame(msg_id: int, payload: bytes, mtu: int = DEFAULT_MTU) -> list[bytes]:
    """Split one message into notification values, each <= mtu - 3 bytes."""
    size = chunk_payload_size(mtu)
    n = max(1, math.ceil(len(payload) / size))
    if n > MAX_CHUNKS:
        raise ValueError(f"message of {len(payload)} bytes needs {n} chunks (max {MAX_CHUNKS}) at MTU {mtu}")
    mid = msg_id & 0xFF
    out = []
    for i in range(n):
        flags = FLAG_FINAL if i == n - 1 else 0
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
        self.dropped = 0             # messages abandoned (new id before FINAL, gap, overflow)
        self.completed = 0

    def _reset(self) -> None:
        self.msg_id, self.next_index, self.parts, self.size = None, 0, [], 0

    def feed(self, value: bytes) -> Optional[bytes]:
        try:
            mid, idx, final, payload = parse_header(value)
        except ValueError:
            return None
        if self.msg_id is not None and mid != self.msg_id:
            self.dropped += 1                      # new message before the old one finished
            self._reset()
        if idx == 0:
            if self.msg_id == mid and self.parts:   # a repeated chunk 0 of the same id: restart
                self.dropped += 1
            self._reset()
            self.msg_id = mid
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
            msg = b"".join(self.parts)
            self._reset()
            self.completed += 1
            return msg
        return None


def dumps(obj: Any) -> bytes:
    return json.dumps(obj, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


# ---------------------------------------------------------------- question / answer

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


def target_of(state: Optional[dict], name: Optional[str]) -> Optional[list]:
    """Resolved table-cm position of an entity in a WorldState (falls back to its last-seen spot)."""
    if not name or not isinstance(state, dict):
        return None
    for e in state.get("entities") or []:
        if isinstance(e, dict) and e.get("name") == name:
            p = _xy(e.get("resolved_cm")) or _xy(e.get("pos_cm"))
            return p
    return None


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


def compact_entity(e: dict) -> dict:
    """One WorldState entity -> the compact phone form. Unknown extra fields are ignored."""
    status = str(e.get("status") or "UNKNOWN")
    out: dict = {"n": str(e.get("name")),
                 "k": KIND_LETTER.get(str(e.get("kind")), "t"),
                 "s": STATUS_LETTER.get(status, "X")}
    if e.get("parent"):
        out["p"] = str(e["parent"])
    xy = _xy(e.get("pos_cm"))
    if xy:
        out["xy"] = xy
    r = _xy(e.get("resolved_cm"))
    if r:
        out["r"] = r
    try:
        c = round(float(e.get("confidence")), 2)
        if not math.isnan(c):
            out["c"] = c
    except (TypeError, ValueError):
        pass
    if e.get("edge"):
        out["edge"] = str(e["edge"])
    if str(e.get("name")).startswith("thing:"):            # open-world things only
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
    g = e.get("guess")
    if isinstance(g, dict) and g.get("name"):              # automatic name guess (core/auto_name.py)
        out["g"] = str(g["name"])
    ls = _r1(e.get("last_seen"))
    if ls is not None:
        out["ls"] = ls
    return out


def compact_state(state: Optional[dict], table_cm: tuple, now: float) -> dict:
    """WorldState (GET /state's "state") -> the state notification JSON. state=None: an empty room."""
    st = state if isinstance(state, dict) else {}
    laser = st.get("laser") if isinstance(st.get("laser"), dict) else {}
    las = {"on": bool(laser.get("on")), "target": str(laser["target"]) if laser.get("target") else None}
    ents = [compact_entity(e) for e in (st.get("entities") or []) if isinstance(e, dict) and e.get("name")]
    return {"v": 1, "t": round(now, 1), "table": [_r1(table_cm[0]), _r1(table_cm[1])],
            "online": bool(st.get("online")), "laser": las, "e": ents}


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
    timestamps and detector jitter (positions within 0.5 cm, confidence within 0.05)."""
    if prev is None:
        return True
    for k in ("table", "online", "laser"):
        if prev.get(k) != cur.get(k):
            return True
    pe = {e["n"]: e for e in prev.get("e", [])}
    ce = {e["n"]: e for e in cur.get("e", [])}
    if pe.keys() != ce.keys():
        return True
    for n, c in ce.items():
        p = pe[n]
        for k in ("k", "s", "p", "edge", "a", "m", "g"):
            if p.get(k) != c.get(k):
                return True
        if _moved(p.get("xy"), c.get("xy")) or _moved(p.get("r"), c.get("r")):
            return True
        if abs(float(p.get("c", 0)) - float(c.get("c", 0))) > CONF_DEADBAND:
            return True
        if c.get("s") != "V" and p.get("ls") != c.get("ls"):
            return True       # a hidden object's last-seen time changing is news; a visible one's isn't
    return False


# ---------------------------------------------------------------- status

def status_msg(app_up: bool, state: Optional[dict], cal: bool, laser_cal: bool) -> dict:
    st = state if isinstance(state, dict) else {}
    fps = _r1(st.get("fps")) if app_up else 0.0
    return {"app": "up" if app_up else "down", "fps": fps or 0.0,
            "online": bool(st.get("online")) if app_up else False,
            "cal": bool(cal), "laser_cal": bool(laser_cal)}


def status_changed(prev: Optional[dict], cur: dict, fps_step: float = 1.0) -> bool:
    if prev is None:
        return True
    for k in ("app", "online", "cal", "laser_cal"):
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


class MsgCounter:
    """Per-characteristic msg_id, 0..255 wrapping."""

    def __init__(self):
        self.ids: dict[str, int] = {}

    def next(self, char: str) -> int:
        i = self.ids.get(char, -1) + 1 & 0xFF
        self.ids[char] = i
        return i


Clock = Callable[[], float]
