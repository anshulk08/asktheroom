"""Grok for open-ended (OTHER) questions (spec V10), on the voice path: voice/pipeline.py's default
answerer is ask_other(), which tries voice/local_llm.py's fixed templates first (instant, exact) and
then ask_grok(). Offline, it gives the fallback sentence. (Team decision, Fri night: all LLM/VLM work
goes through Grok; the local Qwen answerer, voice.local_llm.ask_local, is no longer on the path.)

Only question text and a compact world-state JSON leave the device. The model gets the state in
the system prompt, may call locate / history / changes_since, and must finish with respond().
Any failure (offline, no key, timeout, API error, bad output) returns the offline fallback;
ask_grok never raises.

xAI API notes (checked 2026-09-25):
- grok-4.3 lists reasoning efforts `none`, `low`, `medium`, `high`, `xhigh` (default `low`):
  https://docs.x.ai/developers/models/grok-4.3
- Chat Completions takes a top-level string `reasoning_effort` ("supported values and the
  default depend on the model"): https://docs.x.ai/developers/rest-api-reference/inference/chat-completions
- tool_choice accepts "auto" | "required" | "none" | {"type": "function", "function": {"name"}}:
  https://docs.x.ai/developers/tools/function-calling
"""
from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
from datetime import datetime
from typing import Any, Optional

from core.config import display_name, load_config
from core.labels import thing_labels
from core.narration_store import med_claim
from core.types import Answer, Status

log = logging.getLogger(__name__)

FALLBACK_TEXT = "I can tell you where things are, what happened to them, or what changed."
MAX_ROUNDS = 3
MAX_EVENTS = 20


def fallback() -> Answer:
    return Answer(FALLBACK_TEXT, None, None)


SYSTEM_TEMPLATE = """You are the voice of "Ask the Room", a device that watches a tabletop with an overhead camera and remembers where objects are, even when they are hidden inside the box or under the notebook. A person asked a question out loud; your reply is spoken by a text-to-speech voice and a laser can point at one object.

Rules:
- Answer in at most 2 short sentences of plain spoken English. No markdown, lists, emoji, coordinates, or numbers with units of centimetres.
- Use only facts from the world state and tool results below. Never guess or invent objects, people, or events.
- Never say or imply that medication or pills were taken, swallowed, or missed. You can only report where the pill bottle is and when it was picked up, moved, or covered.
- If an object's confidence is below {plain}, hedge with "probably". If its status is UNKNOWN, say "I lost track of" it and where it was last seen.
- Status meanings: VISIBLE on the table; HELD in a hand; INSIDE a container (parent); UNDER a cover (parent); GONE left the camera view (edge tells which side); UNKNOWN lost track.
- Say object names with spaces (pill bottle, not pill_bottle). Times: say "a minute ago", "about 5 minutes ago", etc.
- The objects are: {objects}. Things the person named appear by that name; "mug?" is a thing nobody has named that looks like a mug (say "what looks like a mug"), and "something new" is one with no name and no guess (describe it by where it is; never say a number in brackets). maybe_same_as lists objects that may be the same physical object.
- Always finish by calling respond(text, point_at). Set point_at to the object the answer is about when pointing at it helps (for hidden objects, point at the object itself; the laser follows it to its container), otherwise leave it empty.
- You may call locate, history, or changes_since first, but only if the state below is not enough. Be quick.

Current local time: {now_iso}
World state (JSON; area = which third of the table, seen_s_ago = seconds since last seen):
{state}"""


def _tool(name: str, desc: str, props: dict, required: list[str]) -> dict:
    return {"type": "function", "function": {
        "name": name, "description": desc,
        "parameters": {"type": "object", "properties": props, "required": required}}}


def _tools(names: list[str]) -> list[dict]:
    obj = {"type": "string", "enum": names, "description": "object name"}
    return [
        _tool("locate", "Where an object is now, following containers and covers.",
              {"object": obj}, ["object"]),
        _tool("history", "The most recent events for an object, newest first.",
              {"object": obj, "limit": {"type": "integer", "minimum": 1, "maximum": 10}},
              ["object"]),
        _tool("changes_since", "All events on the table since a time.",
              {"iso_time": {"type": "string",
                            "description": "ISO 8601 time, e.g. 2026-09-25T14:03:00-04:00"}},
              ["iso_time"]),
        _tool("respond", "Give the final spoken answer. Must be the last call.",
              {"text": {"type": "string", "description": "at most 2 spoken sentences"},
               "point_at": {"type": "string",
                            "description": "object name for the laser, or empty string"}},
              ["text"]),
    ]


# ---------------------------------------------------------------- world -> compact facts

def _area(cfg: dict, pos) -> Optional[str]:
    if not pos:
        return None
    width = float(((cfg.get("table") or {}).get("size_cm") or [90, 60])[0])
    x = float(pos[0])
    return "left" if x < width / 3 else ("right" if x > 2 * width / 3 else "middle")


def _ago(wall: Optional[float], now: float) -> Optional[int]:
    return None if wall is None else max(0, int(round(now - wall)))


def compact_state(world, cfg: dict) -> list[dict]:
    """world.state_json() minus noise (fps, laser, raw coordinates, redundant edges). Open-world
    things go by their taught name, else 'mug?' or 'something new', never the internal id (core/labels.py)."""
    now = time.time()
    out = []
    st = world.state_json()
    labels = thing_labels(st)
    room = st.get("room") or {}                # spec 0009: entities in a room zone (none when it's off)
    for e in st.get("entities", []):
        d: dict[str, Any] = {"name": labels.get(e["name"], e["name"]), "status": e["status"]}
        if e.get("parent") and e["status"] != Status.VISIBLE.value:
            d["parent"] = labels.get(e["parent"], e["parent"])
        r = room.get(e["name"]) if e["name"] != "conflicts" else None
        if isinstance(r, dict) and r.get("say"):
            d["room_zone"] = r["say"]          # 'the bookshelf': no table area (pos_cm may be a flicker)
            if r.get("absent"):
                d["not_seen_there_now"] = True
        else:
            area = _area(cfg, e.get("resolved_cm") or e.get("pos_cm"))
            if area:
                d["area"] = area
        d["confidence"] = round(float(e.get("confidence", 1.0)), 2)
        if e.get("candidates"):
            d["candidates"] = [labels.get(c, c) for c in e["candidates"]]
        if e.get("edge"):
            d["edge"] = e["edge"]
        if len(e.get("aliases") or []) > 1:
            d["also_called"] = e["aliases"][1:]
        if e.get("maybe_same_as"):
            d["maybe_same_as"] = [labels.get(m[0], m[0]) for m in e["maybe_same_as"]]
        ago = _ago(e.get("last_seen"), now)
        if ago is not None:
            d["seen_s_ago"] = ago
        out.append(d)
    return out


def _event_dict(ev, now: float, with_obj: bool, labels: Optional[dict] = None) -> dict:
    labels = labels or {}
    d: dict[str, Any] = {}
    if with_obj:
        d["object"] = labels.get(ev.obj, ev.obj)
    d["type"] = ev.type
    d["s_ago"] = _ago(ev.wall, now)
    if ev.parent:
        d["parent"] = labels.get(ev.parent, ev.parent)
    if ev.edge:
        d["edge"] = ev.edge
    if ev.confidence is not None and ev.confidence < 1.0:
        d["confidence"] = round(ev.confidence, 2)
    return d


def _parse_iso(s: str) -> float:
    s = s.strip()
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"            # fromisoformat on 3.10 rejects 'Z'
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.astimezone()             # naive -> local time
    return dt.timestamp()


class _Tools:
    """Dispatches model tool calls against the world and event log."""

    def __init__(self, world, events, cfg: dict):
        self.world, self.events, self.cfg = world, events, cfg
        self.names = _entity_names(world)
        self.labels = _labels(world)          # thing:N -> what Grok calls it
        self.ids = {v.lower(): k for k, v in self.labels.items() if k in self.names}

    def _say(self, name: Optional[str]) -> Optional[str]:
        return self.labels.get(name, name) if name else name

    def _name(self, raw: Any) -> str:
        n = str(raw or "").strip().lower()
        if n in self.ids:                     # a thing by its taught name / 'mug?' / 'something new (2)'
            return self.ids[n]
        find = getattr(self.world, "find", None)
        hit = find(n) if callable(find) and n else None     # 'my charger', plurals, taught aliases
        if hit in self.names:
            return hit
        n = (self.cfg.get("synonyms") or {}).get(n, n).replace(" ", "_")
        if n not in self.names:
            raise ValueError(f"unknown object {raw!r}")
        return n

    def locate(self, object: str) -> dict:
        name = self._name(object)
        e = self.world.get(name)
        pos, chain = self.world.resolve(name)
        d: dict[str, Any] = {"object": self._say(name), "status": e.status.value,
                             "chain": [self._say(c) for c in chain], "confidence": round(e.confidence, 2)}
        if e.parent and e.status != Status.VISIBLE:
            d["parent"] = self._say(e.parent)
        if _area(self.cfg, pos):
            d["area"] = _area(self.cfg, pos)
        if e.candidates:
            d["candidates"] = [self._say(c) for c in e.candidates]
        if e.edge:
            d["edge"] = e.edge
        if e.last_seen is not None:
            d["seen_s_ago"] = _ago(e.last_seen, time.time())
        return d

    def history(self, object: str, limit: int = 3) -> dict:
        name = self._name(object)
        n = max(1, min(10, int(limit or 3)))
        now = time.time()
        return {"object": self._say(name),
                "events": [_event_dict(ev, now, False, self.labels) for ev in self.world.history(name, n)]}

    def changes_since(self, iso_time: str) -> dict:
        t = _parse_iso(iso_time)
        log_ = self.events if self.events is not None else getattr(self.world, "events", None)
        evs = sorted(log_.since(t), key=lambda ev: ev.wall) if log_ is not None else []
        now = time.time()
        return {"since_s_ago": _ago(t, now), "count": len(evs),
                "events": [_event_dict(ev, now, True, self.labels) for ev in evs[-MAX_EVENTS:]]}

    def call(self, name: str, args: dict) -> dict:
        try:
            if name == "locate":
                return self.locate(args.get("object"))
            if name == "history":
                return self.history(args.get("object"), args.get("limit", 3))
            if name == "changes_since":
                return self.changes_since(args.get("iso_time", ""))
            return {"error": f"unknown tool {name}"}
        except Exception as ex:                 # report to the model, don't crash
            return {"error": str(ex)}


def _entity_names(world) -> list[str]:
    try:
        return [e["name"] for e in world.state_json().get("entities", [])]
    except Exception:
        return []


def _labels(world) -> dict[str, str]:
    try:
        return thing_labels(world.state_json())
    except Exception:
        return {}


# ---------------------------------------------------------------- answer post-processing

_MD = re.compile(r"[*_#`>\[\]]+")
_BOTTLE = re.compile(r"\bpill[ _]bottles?\b", re.I)   # the object, not medication


def _med_claim(text: str) -> bool:
    """Any sentence saying medication was taken, missed or skipped (narration_store.med_claim, the
    wider rule the visual answers use); 'pill bottle' is the object and doesn't count as medication."""
    return any(med_claim(_BOTTLE.sub("bottle", s)) for s in re.split(r"(?<=[.!?])\s+", text))


PILLS_SAFE = ("I can't tell whether medication was taken; I can only tell you where the pill "
              "bottle is and when it was moved.")


def clean_text(text: str) -> str:
    """Plain spoken text, at most 2 sentences."""
    t = _MD.sub("", text or "").replace("\n", " ")
    t = re.sub(r"\s+", " ", t).strip()
    parts = re.split(r"(?<=[.!?])\s+", t)
    return " ".join(parts[:2]).strip()


def to_answer(text: str, point_at: Optional[str], names: list[str], cfg: dict,
              labels: Optional[dict] = None) -> Answer:
    """respond(...) args -> Answer; drops unknown point_at and blocks 'pills were taken' claims.
    labels (thing:N -> name, see core/labels.py) turn a thing's name back into its entity id."""
    text = clean_text(text)
    if not text:
        return fallback()
    p = (point_at or "").strip().lower()
    ids = {v.lower(): k for k, v in (labels or {}).items() if k in names}
    p = ids.get(p) or (cfg.get("synonyms") or {}).get(p, p).replace(" ", "_")
    target = p if p in names else None
    if _med_claim(text):
        text, target = PILLS_SAFE, ("pill_bottle" if "pill_bottle" in names else None)
    return Answer(text, target, "point" if target else None)


# ---------------------------------------------------------------- the call

def _make_client(base_url: str, api_key: str, timeout: float):
    """Seam for tests. core.xai: plain requests on the shared, pre-warmed connection to xAI."""
    from core.xai import Client
    return Client(base_url, api_key, timeout=timeout)


def _args(tc) -> dict:
    try:
        a = json.loads(tc.function.arguments or "{}")
        return a if isinstance(a, dict) else {}
    except (ValueError, TypeError):
        return {}


def _run(question: str, world, events, cfg: dict, api_key: str, deadline: float) -> Answer:
    llm = cfg.get("llm") or {}
    names = _entity_names(world)
    tools = _Tools(world, events, cfg)
    said = [tools.labels.get(n, n) for n in names]        # things by name, never thing:N
    objects = ", ".join(f"{n} ({display_name(cfg, n)})" if "_" in n else n for n in said)
    system = SYSTEM_TEMPLATE.format(
        plain=cfg.get("answer_plain", 0.7), objects=objects,
        now_iso=datetime.now().astimezone().isoformat(timespec="seconds"),
        state=json.dumps(compact_state(world, cfg), separators=(",", ":")))
    messages: list[dict] = [{"role": "system", "content": system},
                            {"role": "user", "content": question}]
    client = _make_client(llm.get("base_url", "https://api.x.ai/v1"), api_key,
                          max(0.1, deadline - time.monotonic()))
    extra: dict[str, Any] = {}
    if llm.get("reasoning_effort"):
        extra["reasoning_effort"] = str(llm["reasoning_effort"])

    for rnd in range(MAX_ROUNDS):
        remaining = deadline - time.monotonic()
        if remaining <= 0.05:
            log.info("grok: out of time before round %d", rnd)
            return fallback()
        last = rnd == MAX_ROUNDS - 1
        choice = ({"type": "function", "function": {"name": "respond"}} if last else "required")
        resp = client.chat.completions.create(
            model=llm.get("model", "grok-4.3"), messages=messages, tools=_tools(said),
            tool_choice=choice, max_completion_tokens=300, timeout=remaining, **extra)
        msg = resp.choices[0].message
        calls = list(msg.tool_calls or [])
        for tc in calls:
            if tc.function.name == "respond":
                a = _args(tc)
                return to_answer(a.get("text", ""), a.get("point_at"), names, cfg, tools.labels)
        if not calls:
            # model ignored tool_choice; accept plain content if any
            return to_answer(msg.content or "", None, names, cfg, tools.labels)
        messages.append({"role": "assistant", "content": msg.content or "",
                         "tool_calls": [{"id": tc.id, "type": "function",
                                         "function": {"name": tc.function.name,
                                                      "arguments": tc.function.arguments or "{}"}}
                                        for tc in calls]})
        for tc in calls:
            result = tools.call(tc.function.name, _args(tc))
            messages.append({"role": "tool", "tool_call_id": tc.id,
                             "content": json.dumps(result, separators=(",", ":"))})
    return fallback()


def ask_other(question: str, world, events, cfg: dict | None = None, online: bool = True) -> Answer:
    """The pipeline's answerer for OTHER: fixed templates for the common open questions, then Grok
    when online, else the fallback sentence. understand.backend qwen (or auto, offline) uses the local
    Qwen (voice.local_llm) instead. Never raises."""
    try:
        cfg = cfg if cfg is not None else load_config()
        from voice.local_llm import ask_local, templated
        fixed = templated(question, world, cfg)
        if fixed is not None:
            return fixed
        backend = str((cfg.get("understand") or {}).get("backend", "grok"))
        has_key = bool(os.environ.get("XAI_API_KEY", "").strip())
        if backend == "qwen" or (backend == "auto" and not (online and has_key)):
            return ask_local(question, world, events, cfg)
    except Exception:
        log.exception("templated answer failed")
    return ask_grok(question, world, events, cfg, online=online)


def ask_grok(question: str, world, events, cfg: dict | None = None,
             online: bool = True) -> Answer:
    """Answer an open-ended question with Grok, or the offline fallback. Never raises and
    returns within cfg llm.timeout_s (plus a few ms)."""
    try:
        if not online:
            return fallback()
        api_key = os.environ.get("XAI_API_KEY", "").strip()
        if not api_key:
            return fallback()
        cfg = cfg if cfg is not None else load_config()
        timeout_s = float((cfg.get("llm") or {}).get("timeout_s", 4))
        deadline = time.monotonic() + timeout_s
    except Exception:
        log.exception("grok setup failed")
        return fallback()

    box: dict[str, Answer] = {}

    def work() -> None:
        try:
            box["a"] = _run(question, world, events, cfg, api_key, deadline)
        except Exception as ex:
            log.warning("grok failed: %s: %s", type(ex).__name__, ex)

    th = threading.Thread(target=work, name="ask_grok", daemon=True)
    th.start()
    th.join(max(0.0, deadline - time.monotonic()))
    if th.is_alive():
        log.warning("grok timed out after %.1f s", timeout_s)
        return fallback()
    return box.get("a") or fallback()
