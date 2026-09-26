"""Spoken wording shared by the care layer (reminders, reports, profile, follow-ups).

One place for the sentences that are spoken without being asked: times ('8 AM', '9 o'clock'), whose
thing it is ('your keys', 'the box'), where it is ('It's under the notebook.'), and the pill rule. The
pill rule is the same one voice/llm.py and core/narration_store.py enforce: nothing we say may state or
imply that medication was taken, swallowed or missed; we only know whether the pill bottle was picked up.
Every care sentence is built so it cannot break the rule, and pill_guard() is the belt on top.

Ideas for proactive, plain-language prompts for people with dementia are credited to Project Memoria
(MIT); the wording and code here are our own, built on our world model.
"""
from __future__ import annotations

import re
import time
from datetime import datetime
from typing import Optional

from core.config import display_name
from core.types import Event, Status
from voice.answers import area

__all__ = ["pill_claim", "pill_guard", "say_time", "say_clock", "say_oclock", "say_day", "spoken", "plural",
           "your", "where_sentence", "event_place", "second_person", "med_phrase", "NEUTRAL_PILLS"]

UNNAMED = "thing I haven't been told about"
PLURAL = {"keys", "glasses", "pills", "headphones", "scissors", "earbuds"}

# ---------------------------------------------------------------- the pill rule

# Medication words; 'pill bottle' is the object we track, not a claim about pills.
MEDS = (r"\b(?:pills?(?!\s*bottles?)|medications?|medicines?|meds|doses?|dosage|tablets?|capsules?|vitamins?"
        r"|prescriptions?)\b")
CLAIM = (r"\b(?:took|taken|take|takes|taking|had|miss(?:ed|es|ing)?|skip(?:ped|s|ping)?|forg[eo]t\w*"
         r"|popp(?:ed|ing)|consum\w*|ate|dosed)\b")
ALWAYS = r"\b(?:swallow\w*|ingest\w*)\b"
_MEDS, _CLAIM, _ALWAYS = (re.compile(p, re.I) for p in (MEDS, CLAIM, ALWAYS))
NEUTRAL_PILLS = "I can only tell you where the pill bottle is and when it was picked up."


def _sentences(text: str) -> list[str]:
    return [s for s in re.split(r"(?<=[.!?;])\s+", (text or "").strip()) if s]


def pill_claim(text: str) -> bool:
    """True if any sentence states or implies medication was taken, swallowed or missed. Stricter than
    the Grok filter on purpose: even 'take your pills' is out, so a reminder can never read as a record."""
    for s in _sentences(text):
        if _ALWAYS.search(s) or (_MEDS.search(s) and _CLAIM.search(s)):
            return True
    return False


def pill_guard(text: str) -> str:
    """text without the sentences that break the pill rule (never empty: the neutral line instead)."""
    keep = [s for s in re.split(r"(?<=[.!?])\s+", (text or "").strip()) if s and not pill_claim(s)]
    return " ".join(keep) or NEUTRAL_PILLS


def med_phrase(task: str) -> Optional[str]:
    """'take my pills' -> 'your pills': the medication a task is about, said without its verb."""
    m = _MEDS.search(task or "")
    return f"your {m.group(0).lower()}" if m else None


# ---------------------------------------------------------------- times

def say_time(h: int, m: int = 0) -> str:
    """'8 AM', '8:30 PM', 'noon', 'midnight'."""
    if (h, m) == (12, 0):
        return "noon"
    if (h, m) == (0, 0):
        return "midnight"
    h12 = h % 12 or 12
    return f"{h12}{f':{m:02d}' if m else ''} {'AM' if h < 12 else 'PM'}"


def say_clock(wall: float) -> str:
    lt = time.localtime(wall)
    return say_time(lt.tm_hour, lt.tm_min)


def say_oclock(h: int, m: int = 0) -> str:
    """How 'It's ...' continues: '9 o'clock', '3:30', 'noon', 'midnight'."""
    if (h, m) in ((12, 0), (0, 0)):
        return say_time(h, m)
    h12 = h % 12 or 12
    return f"{h12} o'clock" if m == 0 else f"{h12}:{m:02d}"


def say_day(t: float, now: float) -> str:
    """'today', 'tomorrow', 'on Saturday' (this week), else 'on September 30'."""
    d, n = datetime.fromtimestamp(t).date(), datetime.fromtimestamp(now).date()
    k = (d - n).days
    if k == 0:
        return "today"
    if k == 1:
        return "tomorrow"
    if 1 < k < 7:
        return f"on {d.strftime('%A')}"
    return f"on {d.strftime('%B')} {d.day}"


# ---------------------------------------------------------------- names

def _labels(world) -> dict:
    try:
        return world.thing_labels() if hasattr(world, "thing_labels") else {}
    except Exception:
        return {}


def spoken(world, cfg: dict, obj: str) -> str:
    """How an entity is said: its taught name for a thing, else the display name; a thing id never."""
    if obj.startswith("thing:"):
        return _labels(world).get(obj) or UNNAMED
    return display_name(cfg, obj)


def plural(word: str) -> bool:
    w = word.split()[-1] if word else ""
    return w in PLURAL or (w.endswith("s") and not w.endswith("ss"))


def _kind(world, cfg: dict, obj: str) -> str:
    try:
        return world.get(obj).kind
    except Exception:
        return (cfg.get("objects") or {}).get(obj, "target")


def your(world, cfg: dict, obj: str) -> str:
    """'your' for the user's things, 'the' for the box and the notebook."""
    return "your" if _kind(world, cfg, obj) == "target" else "the"


def second_person(text: str) -> str:
    """'call my daughter' -> 'call your daughter' (names keep their case)."""
    swap = {"my": "your", "me": "you", "i": "you", "myself": "yourself", "mine": "yours", "am": "are",
            "im": "you're", "i'm": "you're", "our": "your", "we": "you", "us": "you"}
    return " ".join(swap.get(w.lower(), w) for w in (text or "").split())


# ---------------------------------------------------------------- places

def _parent_phrase(world, cfg: dict, parent: Optional[str], status: Status) -> str:
    prep = "under" if status == Status.UNDER else "inside"
    if not parent or parent == "unknown":
        return f"{prep} something"
    if parent.startswith("hand"):
        return "in someone's hand"
    return f"{prep} the {spoken(world, cfg, parent)}"


def where_sentence(world, cfg: dict, obj: str, named: bool = False) -> tuple[str, Optional[str]]:
    """(sentence, laser action) for where obj is now. named=False says 'It's ...' (the object was just
    named); named=True says 'Your pill bottle is ...'."""
    try:
        e = world.get(obj)
    except Exception:
        return "", None
    n = spoken(world, cfg, obj)
    pl = plural(n)
    subj = f"{your(world, cfg, obj).capitalize()} {n}" if named else ("They" if pl else "It")
    is_, was = ("are", "were") if pl else ("is", "was")
    s = f"{subj} {is_}" if named else ("They're" if pl else "It's")
    obj_pron = f"{your(world, cfg, obj)} {n}" if named else ("them" if pl else "it")
    if e.status == Status.VISIBLE:
        return f"{s} on the table, {area(e.pos_cm, cfg)}.", "point"
    if e.status in (Status.INSIDE, Status.UNDER):
        return f"{s} {_parent_phrase(world, cfg, e.parent, e.status)}.", "point"
    if e.status == Status.HELD:
        return f"Someone is holding {obj_pron} right now.", "point"
    if e.status == Status.GONE:
        side = f"the {e.edge} side of the table" if e.edge else "the table"
        return f"{subj} {was} carried off {side}.", (f"sweep:{e.edge}" if e.edge else "circle")
    if e.pos_cm is None and e.last_seen is None:
        return f"I haven't seen {obj_pron} yet.", None
    return (f"I lost track of {obj_pron}; I last saw {'them' if pl else 'it'} {area(e.pos_cm, cfg)}.",
            "circle")


def event_place(ev: Event, world, cfg: dict) -> Optional[str]:
    """Where an event left its object: 'inside the box', 'on the table, near the top left', or None for
    events that leave it nowhere in particular (picked up, lost from view)."""
    p = ev.parent if ev.parent and ev.parent != "unknown" and not ev.parent.startswith("hand") else None
    t = str(ev.type)
    if t == "PUT_INSIDE":
        return f"inside the {spoken(world, cfg, p)}" if p else "inside something"
    if t == "COVERED":
        return f"under the {spoken(world, cfg, p)}" if p else "under something"
    if t == "EXITED_VIEW":
        return f"off the {ev.edge} side of the table" if ev.edge else "off the table"
    if t in ("PUT_BACK", "MOVED", "APPEARED", "FOUND", "TAKEN_OUT", "UNCOVERED", "CORRECTED"):
        return f"on the table, {area(ev.to_cm, cfg)}" if ev.to_cm else "on the table"
    return None
