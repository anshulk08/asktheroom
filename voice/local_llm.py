"""Fixed answers for the common open (OTHER) questions (templated(), used by voice.llm.ask_other on
the voice path), and the local Qwen answerer for the long tail (ask_local: an offline option, no longer
on the voice path since the team moved all LLM work to Grok): "what's in the box", "which things are
hidden", "is anything under the notebook".

One call, no tool loop: the prompt carries the compact world state and the last few events
(voice.llm.compact_state), and a JSON schema makes the reply {action, point_at, text} in that
order. The laser target is an enum of the objects, so the model never invents coordinates. The
same post-processing as Grok's answers runs on it (voice.llm.to_answer: two plain sentences, no
"pills were taken"). The laser only moves if the sentence names the object it points at.

The rules and templates still answer WHERE/HISTORY/HANDLED/CHANGES (voice.answers): exact,
tested, and instant. The open questions visitors ask most get templates too (templated(): what's
in or under something, what's hidden, what's on the table, "are you recording me", "what can you
do"), because a 1.7B model reading JSON state still slips ("3 things on the table: keys, ...").
Qwen writes only the long tail. Any failure (server down, timeout, bad JSON) returns voice.llm's
fallback sentence; ask_local never raises.

    python -m voice.local_llm "what's in the box"      # demo world, llama-server on understand.url
"""
from __future__ import annotations

import json
import logging
import re
import sys
import time
from typing import Optional

import requests

from core.config import display_name, load_config
from core.types import Answer
from voice.llm import PILLS_SAFE, _entity_names, _event_dict, compact_state, fallback, to_answer

log = logging.getLogger(__name__)

RECENT_S = 30 * 60                      # events from the last half hour...
RECENT_N = 12                           # ...at most this many, newest last

# Static, so llama-server's prompt cache keeps it between questions; the state goes in the user turn.
SYSTEM = """You are the voice of "Ask the Room", a device that watches a tabletop with an overhead camera and remembers where objects are, even hidden inside the box or under the notebook. Someone asked a question out loud. Your text is spoken aloud and a laser can point at one object.

Rules:
- text: at most 2 short spoken sentences. No lists, markdown, coordinates or centimetres.
- Use only the facts in the state and events. Never invent objects, people or events. If the state doesn't say, say you don't know.
- Never say or imply that medication or pills were taken, swallowed or missed. Only say where the pill bottle is and when it was moved.
- Status: VISIBLE on the table; HELD in a hand; INSIDE a container (parent); UNDER a cover (parent); GONE left the table (edge); UNKNOWN lost track. Confidence below 0.7: say "probably".
- Say names with spaces (pill bottle). Times: "a minute ago", "about 5 minutes ago".
- action: "point" at the one object the answer is about, "circle" it if it is hidden or unsure, "none" if no single object fits. point_at is that object, or "none".
Objects: {objects}.
Reply with JSON only."""


def _names(cfg: dict) -> list[str]:
    return list(cfg.get("objects") or {})


def schema(cfg: dict) -> dict:
    return {"type": "object", "additionalProperties": False, "required": ["action", "point_at", "text"],
            "properties": {"action": {"type": "string", "enum": ["point", "circle", "none"]},
                           "point_at": {"type": "string", "enum": _names(cfg) + ["none"]},
                           "text": {"type": "string", "maxLength": 220}}}


def system_prompt(cfg: dict) -> str:
    return SYSTEM.format(objects=", ".join(_names(cfg)))


def user_prompt(question: str, world, events, cfg: dict) -> str:
    now = time.time()
    evs = []
    if events is not None:
        try:
            evs = sorted(events.since(now - RECENT_S), key=lambda ev: ev.wall)[-RECENT_N:]
        except Exception:
            pass
    state = json.dumps(compact_state(world, cfg), separators=(",", ":"))
    recent = json.dumps([_event_dict(ev, now, True, cfg=cfg) for ev in evs], separators=(",", ":"))
    return f"State: {state}\nRecent events (s_ago = seconds ago): {recent}\nQuestion: {question}"


def _list(names: list[str], cfg: dict) -> str:
    said = [f"the {display_name(cfg, n)}" for n in names]
    return said[0] if len(said) == 1 else ", ".join(said[:-1]) + f" and {said[-1]}"


PRIVACY = ("Audio stays on this device and I never save a recording. When I'm online, questions I can't "
           "answer myself go to Grok, sometimes with a picture of the table, and I keep a day of snapshots.")
_MEDS_TAKEN = re.compile(r"\b(take|took|taken|had|swallow\w*)\b.*\b(pills?|meds|medicine|medication)\b")
_PRIVACY = re.compile(r"\b(record\w*|camera|listening|spy\w*|video|privacy|private|saving|save)\b")
_HELP = re.compile(r"\b(what can you do|what do you do|how do(es)? (you|this|it) work|what are you|help)\b")
_HIDDEN = re.compile(r"\b(hidden|hiding|can'?t see|out of sight|covered up)\b")
_ON_TABLE = re.compile(r"\b(on the table|do you see|can you see|what'?s here|how many)\b")


def templated(question: str, world, cfg: dict) -> Optional[Answer]:
    """Fixed answers for the open questions visitors ask most; None for everything else."""
    q = question.lower().replace("\u2019", "'")
    st = compact_state(world, cfg)
    kinds = cfg.get("objects") or {}
    for place in [n for n, k in kinds.items() if k in ("container", "cover")]:
        m = re.search(r"\b(in|inside|into|under|underneath|beneath|below)\s+(the\s+|my\s+|that\s+)?"
                      + re.escape(display_name(cfg, place).lower()), q)
        if m:
            inside = [e["name"] for e in st if e.get("parent") == place and e["status"] in ("INSIDE", "UNDER")]
            rel = "under" if kinds[place] == "cover" else "inside"
            where = f"{rel} the {display_name(cfg, place)}"
            if not inside:
                return Answer(f"I don't know of anything {where}.", place, "circle")
            be = "is" if len(inside) == 1 and inside[0] not in ("keys", "glasses") else "are"
            return Answer(f"{_list(inside, cfg)[0].upper()}{_list(inside, cfg)[1:]} {be} {where}.",
                          place, "circle")
    if _MEDS_TAKEN.search(q) and "pill_bottle" in kinds:
        return Answer(PILLS_SAFE, "pill_bottle", "point")
    if _PRIVACY.search(q):
        return Answer(PRIVACY)
    if _HIDDEN.search(q):
        hid = [e for e in st if e["status"] in ("INSIDE", "UNDER") and e.get("parent") in kinds]
        if not hid:
            return Answer("Nothing is hidden right now.")
        parts = [f"the {display_name(cfg, e['name'])} {'under' if e['status'] == 'UNDER' else 'inside'} "
                 f"the {display_name(cfg, e['parent'])}" for e in hid[:3]]
        return Answer(f"Hidden right now: {', '.join(parts[:-1]) + ' and ' if len(parts) > 1 else ''}{parts[-1]}.")
    if _HELP.search(q):
        targets = [n for n, k in kinds.items() if k == "target"]
        return Answer(f"I keep track of {_list(targets, cfg).replace('the ', 'your ')}. Ask me where one is "
                      "or what changed, and I'll point at it.")
    if _ON_TABLE.search(q):
        vis = [e["name"] for e in st if e["status"] == "VISIBLE"]
        if not vis:
            return Answer("I don't see anything on the table right now.")
        return Answer(f"I can see {_list(vis, cfg)} on the table.")
    return None


def to_local_answer(raw: str, names: list[str], cfg: dict) -> Answer:
    d = json.loads(raw)
    ans = to_answer(str(d.get("text", "")), d.get("point_at"), names, cfg)
    if ans.point_at and display_name(cfg, ans.point_at).lower() not in ans.text.lower():
        ans = Answer(ans.text, None, None)       # don't point at something the sentence doesn't mention
    elif ans.point_at and d.get("action") == "circle":
        ans = Answer(ans.text, ans.point_at, "circle")
    elif d.get("action") == "none" and ans.point_at != "pill_bottle":
        ans = Answer(ans.text, None, None)
    return ans


def ask_local(question: str, world, events, cfg: Optional[dict] = None, online: bool = True) -> Answer:
    """Same signature as voice.llm.ask_grok; `online` is ignored (it's local). Never raises."""
    try:
        cfg = cfg if cfg is not None else load_config()
        fixed = templated(question, world, cfg)
        if fixed is not None:
            return fixed
        u = cfg.get("understand") or {}
        if not u.get("enabled", True):
            return fallback()
        url = str(u.get("url", "http://127.0.0.1:8081/v1")).rstrip("/")
        names = _entity_names(world) or _names(cfg)
        r = requests.post(f"{url}/chat/completions", timeout=float(u.get("answer_timeout_s", 4)), json={
            "model": str(u.get("model", "qwen")), "temperature": 0, "max_tokens": 90, "cache_prompt": True,
            "chat_template_kwargs": {"enable_thinking": False},
            "messages": [{"role": "system", "content": system_prompt(cfg)},
                         {"role": "user", "content": user_prompt(question, world, events, cfg)}],
            "response_format": {"type": "json_schema", "json_schema": {"name": "answer", "schema": schema(cfg)}}})
        r.raise_for_status()
        return to_local_answer(r.json()["choices"][0]["message"]["content"], names, cfg)
    except Exception as ex:
        log.warning("local answer failed (%s: %s); fallback sentence", type(ex).__name__, ex)
        return fallback()


def main(argv=None) -> int:
    from core.fakeworld import demo_world
    logging.basicConfig(level=logging.INFO, format="%(name)s %(message)s")
    cfg = load_config()
    world = demo_world()
    for q in (sys.argv[1:] if argv is None else argv) or ["what's in the box", "which things are hidden"]:
        t0 = time.monotonic()
        a = ask_local(q, world, world.events, cfg)
        print(f"{1000 * (time.monotonic() - t0):5.0f} ms  {q!r} -> {a.text!r} [{a.action} {a.point_at}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
