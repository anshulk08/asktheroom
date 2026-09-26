"""Offline rule-based intent parser (spec V3): transcript -> Intent(kind, obj, raw).

Pipeline: lowercase, drop apostrophes/punctuation, drop filler words, map synonyms and object
names to canonical names (longest phrase first, whole words), then trigger regexes in priority
order. Priority (first match wins), chosen so overlaps resolve sensibly:

  RECAL  > RESET                    'reset the calibration' is a recalibration
  WHERE ('where...')                'where did anyone move my wallet' is a location question
  HANDLED (did/has <someone> <touch verb>, or passive 'been moved')
                                    'did I move my keys' -> HANDLED; 'who moved the box' -> HISTORY
  CHANGES (changed/different/while I was gone); with an object it becomes HISTORY
  HISTORY (what happened / who / when)
                                    'when did I last see my keys' -> HISTORY (the answer's event
                                    times say when; a 'where' in the question would win instead)
  WHERE (find/seen/lost/locate, or 'is/are my X', 'did I leave X', or just 'my X')
  OTHER                             open-ended; routed to the LLM when online

HISTORY/HANDLED with no object but a general word ('anything', 'stuff', 'what happened')
become CHANGES: 'what happened while I was away', 'did anyone touch anything'.
"""
from __future__ import annotations

import re
from typing import Optional

from core.config import display_name
from core.types import Intent

__all__ = ["Intent", "parse", "normalize"]

FILLER = {"um", "umm", "uh", "uhh", "uhm", "er", "erm", "ah", "eh", "hmm", "hm", "hey", "hi",
          "okay", "ok", "so", "like", "please", "well", "oh", "actually", "yo"}

_SUBJ = r"(?:i|anyone|anybody|someone|somebody|you|we|they|he|she)"
_VERB = (r"(?:take|takes|took|taken|taking|touch\w*|pick\w*|move[ds]?|moving|grab\w*|"
         r"handle[ds]?|handling|use[ds]?|using|open\w*)\b")

RECAL = re.compile(r"\b(?:re)?calibrat\w*")
RESET = re.compile(r"\breset\b|\bstart (?:over|fresh)\b")
WHERE = re.compile(r"\bwhere")  # where, whered, wheres, whereabouts (not 'anywhere')
HANDLED = re.compile(
    rf"\b(?:(?:did|have|has|had)\s+{_SUBJ}|ive|weve)\b(?:\s+\w+){{0,2}}?\s+{_VERB}"
    r"|\b(?:been|get|got|gotten|was|were)\s+(?:touched|moved|picked|taken|handled|grabbed|used|opened)\b")
CHANGES = re.compile(r"\bchang(?:e|ed|es|ing)\b|\bdifferent\b|\bwhat did i miss\b|\bmissed\b"
                     r"|\bwhile i was (?:gone|away|out)\b|\banything new\b|\bwhats new\b")
HISTORY = re.compile(r"\bwhat(?:s| has| had)? happen\w*|\bhappened to\b|\bwho\b|\bwhen\b"
                     r"|\bhistory\b|\blast time\b|\bwhat did (?:i|you|someone|somebody|anyone) do with\b")
WHERE2 = re.compile(r"\b(?:find|found|seen|locate\w*|lost|misplaced|look(?:ing)? for|spot(?:ted)?)\b")
GENERAL = re.compile(r"\b(?:anything|something|everything|stuff|things|what happened"
                     r"|whats happened|while i was)\b")
ARTICLES = {"my", "the", "a", "your", "our"}


def normalize(text: str) -> str:
    """Lowercase, drop apostrophes ('where'd' -> 'whered'), punctuation -> space, drop fillers."""
    t = re.sub(r"['’`]", "", text.lower())
    t = re.sub(r"[^a-z0-9\s]", " ", t)
    return " ".join(w for w in t.split() if w not in FILLER)


def _vocab(cfg: dict) -> tuple[dict[str, str], re.Pattern]:
    """Spoken phrase -> canonical object, plus one whole-word regex (longest phrases first)."""
    objs = cfg.get("objects") or {}
    phrases: dict[str, str] = {}
    for o in objs:
        for p in (o, o.replace("_", " "), display_name(cfg, o)):
            phrases[normalize(p) or p] = o
    for k, v in (cfg.get("synonyms") or {}).items():
        if v in objs:
            phrases[normalize(str(k))] = v
    alts = sorted(phrases, key=lambda p: (-len(p.split()), -len(p)))
    rx = re.compile(r"(?<!\w)(" + "|".join(re.escape(p) for p in alts) + r")(?:e?s)?(?!\w)")
    return phrases, rx


def _pick(found: list[str], cfg: dict) -> Optional[str]:
    """First target mentioned wins ('are my keys in the box' -> keys); else first object."""
    kinds = cfg.get("objects") or {}
    for o in found:
        if kinds.get(o) == "target":
            return o
    return found[0] if found else None


def parse(text: str, cfg: dict) -> Intent:
    """Classify a transcript into an Intent with a canonical object name (or None)."""
    phrases, rx = _vocab(cfg)
    found: list[str] = []

    def sub(m: re.Match) -> str:
        found.append(phrases[m.group(1)])
        return found[-1]

    t = rx.sub(sub, normalize(text))
    obj = _pick(found, cfg)

    if RECAL.search(t):
        kind = "RECAL"
    elif RESET.search(t):
        kind = "RESET"
    elif WHERE.search(t):
        kind = "WHERE"
    elif HANDLED.search(t):
        kind = "HANDLED"
    elif CHANGES.search(t):
        kind = "HISTORY" if obj else "CHANGES"
    elif HISTORY.search(t):
        kind = "HISTORY"
    elif WHERE2.search(t):
        kind = "WHERE"
    elif obj and (re.search(rf"\b(?:is|are)\s+(?:my|the|your|our)?\s*{re.escape(obj)}\b", t)
                  or re.search(r"\b(?:leave|left)\b", t)
                  or [w for w in t.split() if w not in ARTICLES] == [obj]):
        kind = "WHERE"
    else:
        kind = "OTHER"

    if kind in ("HISTORY", "HANDLED") and obj is None and GENERAL.search(t):
        kind = "CHANGES"
    if kind in ("RESET", "RECAL", "CHANGES"):
        obj = None
    return Intent(kind=kind, obj=obj, raw=text)
