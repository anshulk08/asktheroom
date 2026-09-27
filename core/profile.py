"""Profile facts the user states about themselves (care layer, feature 4).

Stable facts, stored in the EventLog's SQLite file (care_profile) as (subject, relation, value):

  user     preferred_name 'Grandpa Joe' ('call me Grandpa Joe'), speech_rate 'slower', speech_volume 'louder'
  person   daughter 'Sarah' ('my daughter is Sarah', 'Sarah is my daughter')
  routine  'take your pills' -> 'at 8 AM and 8 PM' ('I take my pills at 8 and 8'), 'walk the dog' -> 'every morning'

Deterministic patterns cover the common forms offline. When online, a statement the patterns don't
cover may go to Grok (voice/llm.py's client and config) for strict-JSON extraction; its output is kept
only if the relation is one we store and the value is words the user actually said, so nothing is
inferred. There is no 'health' relation: a routine is stored as said ('take your pills at 8'), never
as a claim about taking them. The same subject + relation replaces the old value; the old row stays
as history. 'forget that' drops the latest fact, 'forget everything about me' all of them.

A routine with times suggests a condition reminder ('Should I remind you at 8 AM every day if you
haven't picked up the pill bottle?'): voice/care.py asks and creates it on 'yes'. The preferred name is
used by the morning report and greetings.

Remembering personal facts to personalise help for people with dementia is an idea from Project Memoria
(MIT); the patterns, validation and storage here are our own.

    python -m core.profile --selftest ["I love gardening and my granddaughter Lily visits on Sundays"]
    # runs the Grok extractor for real (needs XAI_API_KEY) and prints what would be stored
"""
from __future__ import annotations

import json
import logging
import os
import re
import sys
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from core.carewords import med_phrase, pill_guard, say_time, second_person
from core.reminders import CareDB, normalize, raw_words
from core.types import Answer

log = logging.getLogger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS care_profile (id INTEGER PRIMARY KEY, t REAL, subject TEXT, relation TEXT, value TEXT,
  times TEXT, active INTEGER, said TEXT, source TEXT);
CREATE INDEX IF NOT EXISTS care_profile_key ON care_profile(subject, relation, active);
"""

PEOPLE = ("daughter", "son", "wife", "husband", "partner", "grandson", "granddaughter", "grandchild", "caregiver",
          "carer", "nurse", "doctor", "neighbor", "neighbour", "friend", "sister", "brother", "mother", "mom",
          "mum", "father", "dad", "niece", "nephew", "aide", "helper", "cousin", "therapist")
USER_RELATIONS = {"preferred_name", "speech_rate", "speech_volume", "likes", "dislikes", "pet", "hobby",
                  "hometown", "occupation"}
ROUTINE_VERBS = ("take", "walk", "eat", "have", "go", "feed", "call", "water", "do", "read", "watch", "visit", "get",
                 "see", "play", "drink", "shower", "bathe", "nap", "exercise", "swim", "cook", "meet", "pray", "wake",
                 "brush", "check", "phone", "lock", "make", "garden", "work", "sleep", "use", "listen", "attend")
# Words that are never a person's name ('my daughter is sick', 'where is my daughter').
NOT_NAMES = {"here", "there", "home", "coming", "visiting", "sick", "ill", "late", "away", "out", "fine", "okay", "ok",
             "busy", "tired", "gone", "dead", "married", "pregnant", "a", "an", "the", "not", "very", "so", "at",
             "in", "on", "where", "what", "who", "how", "which", "when", "why", "she", "he", "this", "that", "it",
             "they", "is", "and", "or", "my", "your", "also", "still", "back", "later", "tomorrow", "today", "you"}

_REL = "|".join(PEOPLE)
_NAME = r"[a-z][a-z'-]*(?: [a-z][a-z'-]*){0,2}"
NAME_RX = [re.compile(p) for p in (
    rf"^(?:(?:you can|you may|please|just|from now on|i want you to|id like you to|i would like you to)\s+)*"
    rf"call me (?P<v>{_NAME})(?: please| from now on)?$",
    rf"^my name is (?P<v>{_NAME})$",
    rf"^(?:im|i am) called (?P<v>{_NAME})$",
)]
PERSON_RX = [re.compile(p) for p in (
    rf"^(?:and )?my (?P<rel>{_REL})(?:s name is| is called| is named| is) (?P<v>{_NAME})$",
    rf"^(?P<v>{_NAME}) is my (?P<rel>{_REL})$",
)]
SPEECH_RX = re.compile(r"\b(?:speak|talk|say (?:it|things)|go)\s+(?:a (?:bit|little) )?(?P<v>more slowly|more quietly|"
                       r"slower|slowly|faster|quicker|louder|softer|quieter|more loudly)\b")
ROUTINE_RX = re.compile(
    rf"^i (?:usually |always |normally |often |generally |also |typically )*(?P<act>(?:{'|'.join(ROUTINE_VERBS)})\b.*?)"
    r"\s+(?P<when>(?:every|each)\s+\w+(?:\s+(?:at|around)\s+.+)?|(?:at|around)\s+(?:\d|noon|midnight).*"
    r"|in the (?:morning|afternoon|evening)s?|twice a day|once a day)$")
KNOW_RX = re.compile(r"\bwhat do you (?:know|remember) (?:about|of) me\b|\bwhat have i told you\b|^who am i$"
                     r"|\bwhat do you know about me\b")
FORGET_LAST = re.compile(r"^(?:please )?(?:no )?forget (?:that|it|what i (?:just )?said|the last thing(?: i said)?)"
                         r"(?: please)?$")
FORGET_ALL = re.compile(r"\bforget (?:everything|all)(?: (?:you know|of it|about me|that you know))*(?: about me)?\b")
FORGET_REL = re.compile(rf"^(?:please )?forget (?:about )?my (?P<rel>{_REL}|name)$")
STATEMENT = re.compile(r"^(?:i|im|i am|my|we|our|call me)\b")
_CLOCK = re.compile(r"\b(?:(?P<h>\d{1,2})(?:[:.](?P<m>\d\d))?(?:\s*(?P<ap>am|pm))?|(?P<w>noon|midnight))\b")
_DAYS = ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")


@dataclass
class Statement:
    subject: str
    relation: str
    value: str
    times: list = field(default_factory=list)       # [(h, m)] for routines


@dataclass
class Fact:
    id: int
    t: float
    subject: str
    relation: str
    value: str
    times: list
    active: bool
    said: str
    source: str


@dataclass
class Suggestion:
    obj_phrase: str                                  # 'take your pills': the reminder's object is resolved from it
    times: list


@dataclass
class Reply:
    answer: Answer
    suggest: Optional[Suggestion] = None


# ---------------------------------------------------------------- extraction

def _norm(text: str) -> str:
    t = normalize(text)
    words = "one two three four five six seven eight nine ten eleven twelve".split()
    return re.sub(rf"\b(and|or|then)\s+({'|'.join(words)})\b", lambda m: f"{m.group(1)} {words.index(m.group(2)) + 1}", t)


def routine_times(when: str) -> list[tuple[int, int]]:
    """Times in a routine, read the way people say them: 'at 8 and 8' is 8 AM and 8 PM, a lone '8' is
    morning (5-11 AM), '1'-'4' afternoon, and each later bare hour is the next one after the previous."""
    out: list[tuple[int, int]] = []
    prev = None
    for m in _CLOCK.finditer(when):
        if m.group("w"):
            hm = (12, 0) if m.group("w") == "noon" else (0, 0)
        else:
            h, mi, ap = int(m.group("h")), int(m.group("m") or 0), m.group("ap")
            if h > 23 or mi > 59:
                continue
            if ap and 1 <= h <= 12:
                H = h % 12 + (12 if ap == "pm" else 0)
            elif h > 12 or h == 0:
                H = h
            elif prev is None:
                H = h if 5 <= h <= 11 else (12 if h == 12 else h + 12)
            else:
                later = [c for c in (h % 12, h % 12 + 12) if (c, mi) > prev]
                H = min(later) if later else h % 12
            hm = (H, mi)
        out.append(hm)
        prev = hm
    return out


def _when_value(when: str, times: list) -> str:
    """'at 8 and 8' -> 'at 8 AM and 8 PM'; 'every sunday at 10' -> 'every Sunday at 10 AM'."""
    head = re.split(r"\s*\b(?:at|around)\b\s*", when, maxsplit=1)[0] if times else when
    head = " ".join(w.capitalize() if w in _DAYS else w for w in head.split())
    if not times:
        return head
    ts = [say_time(*t) for t in times]
    at = "at " + (ts[0] if len(ts) == 1 else ", ".join(ts[:-1]) + " and " + ts[-1])
    return f"{head} {at}".strip()


def _name_ok(raw: str, value: str) -> bool:
    """A person's name, not 'sick' or 'where': not a stop word, and capitalized if the transcript uses
    capitals at all (speech-to-text capitalizes names)."""
    words = value.split()
    if not words or any(w in NOT_NAMES for w in words):
        return False
    if raw == raw.lower():
        return True
    return bool(re.search(r"\b" + r"\W+".join(re.escape(w.capitalize()) for w in words) + r"\b", raw))


def _title(raw: str, value: str) -> str:
    got = raw_words(raw, value)
    return got if got != got.lower() else " ".join(w.capitalize() for w in value.split())


def extract(text: str) -> list[Statement]:
    """Deterministic facts in one utterance (empty if it states none)."""
    t = _norm(text)
    raw = (text or "").strip()
    for rx in NAME_RX:
        m = rx.match(t)
        if m and not re.match(r"(?:at|in|when|if|later|back|tomorrow|a|an|the|tonight|on)\b", m.group("v")):
            return [Statement("user", "preferred_name", _title(raw, m.group("v")))]
    m = SPEECH_RX.search(t)
    if m:
        v = m.group("v")
        if v in ("slower", "slowly", "more slowly"):
            return [Statement("user", "speech_rate", "slower")]
        if v in ("faster", "quicker"):
            return [Statement("user", "speech_rate", "faster")]
        return [Statement("user", "speech_volume", "louder" if v in ("louder", "more loudly") else "softer")]
    for rx in PERSON_RX:
        m = rx.match(t)
        if m and _name_ok(raw, m.group("v")):
            return [Statement("person", m.group("rel"), _title(raw, m.group("v")))]
    m = ROUTINE_RX.match(t)
    if m:
        times = routine_times(m.group("when"))
        return [Statement("routine", second_person(m.group("act")), _when_value(m.group("when"), times), times)]
    return []


def _said(value: str, text: str) -> bool:
    """value is made of words the user said (my/your-insensitive), in order."""
    v = [w for w in normalize(second_person(value)).split()]
    t = normalize(second_person(text))
    return bool(v) and re.search(r"\b" + r"\s+".join(map(re.escape, v)) + r"\b", t) is not None


def validate(raw: Any, text: str) -> list[Statement]:
    """Keep only well-formed facts with a relation we store and a value the user actually said."""
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except ValueError:
            return []
    facts = raw.get("facts") if isinstance(raw, dict) else None
    if not isinstance(facts, list):
        return []
    out = []
    for f in facts:
        if not isinstance(f, dict):
            continue
        s, r, v = (str(f.get(k) or "").strip() for k in ("subject", "relation", "value"))
        r = r.lower().replace(" ", "_") if s != "routine" else second_person(r.lower())
        if not (s and r and v) or len(v) > 60:
            continue
        if s == "user" and r not in USER_RELATIONS:
            continue
        if s == "person" and r not in PEOPLE:
            continue
        if s == "routine" and not _said(r, text):
            continue
        if s not in ("user", "person", "routine") or not _said(v, text):
            continue
        out.append(Statement(s, r, v, routine_times(v) if s == "routine" else []))
    return out


# ---------------------------------------------------------------- Grok extraction

EXTRACT_PROMPT = """Extract stable personal facts the speaker states about themselves from one spoken sentence.
Return JSON only: {"facts": [{"subject": "user" | "person" | "routine", "relation": "...", "value": "..."}]}.
- user relations: preferred_name, speech_rate, speech_volume, likes, dislikes, pet, hobby, hometown, occupation.
- person: relation is how they are related to the speaker (daughter, son, wife, nurse, neighbor, ...); value is their name.
- routine: relation is the activity in the speaker's words (e.g. "walk the dog"); value is when (e.g. "every morning").
- Use only words the speaker said. Do not infer, guess, summarise or add anything. No health conditions or diagnoses.
- Never say or imply that medication was taken; a routine is stored exactly as said.
- If the sentence states no such fact, return {"facts": []}."""


def make_grok_extractor(cfg: dict) -> Callable[[str], Optional[dict]]:
    """extract(text) -> parsed JSON or None, through Grok with voice/llm.py's client and cfg['llm']
    (model, base_url, reasoning_effort). None without XAI_API_KEY or on any failure."""
    llm = cfg.get("llm") or {}

    def ex(text: str) -> Optional[dict]:
        key = os.environ.get("XAI_API_KEY", "").strip()
        if not key:
            return None
        try:
            from voice.llm import _make_client
            timeout = float(llm.get("timeout_s", 4))
            client = _make_client(llm.get("base_url", "https://api.x.ai/v1"), key, timeout)
            kw: dict = dict(model=llm.get("model", "grok-4.3"),
                            messages=[{"role": "system", "content": EXTRACT_PROMPT},
                                      {"role": "user", "content": text}],
                            response_format={"type": "json_object"}, max_completion_tokens=200, timeout=timeout)
            if llm.get("reasoning_effort"):
                kw["reasoning_effort"] = str(llm["reasoning_effort"])
            resp = client.chat.completions.create(**kw)
            out = json.loads(resp.choices[0].message.content or "")
            return out if isinstance(out, dict) else None
        except Exception as e:
            log.warning("profile extraction failed: %s: %s", type(e).__name__, e)
            return None

    return ex


# ---------------------------------------------------------------- the store

def _sentence(s: str) -> str:
    return s[:1].upper() + s[1:] + "."


_COLS = "id, t, subject, relation, value, times, active, said, source"


def _fact(r) -> Fact:
    i, t, s, rel, v, times, active, said, source = r
    return Fact(i, t, s, rel, v, [tuple(x) for x in json.loads(times or "[]")], bool(active), said or "", source or "")


class Profile(CareDB):
    """care_profile: every fact ever stated, the current one per (subject, relation) active."""

    def __init__(self, events, extractor: Optional[Callable[[str], Any]] = None):
        super().__init__(events, SCHEMA)
        self.extractor = extractor

    def remember(self, subject: str, relation: str, value: str, said: str = "", now: Optional[float] = None,
                 times=(), source: str = "rule") -> Fact:
        self.exec("UPDATE care_profile SET active = 0 WHERE subject = ? AND relation = ? AND active = 1",
                  (subject, relation))
        i = self.insert(f"INSERT INTO care_profile ({_COLS}) VALUES (NULL,?,?,?,?,?,1,?,?)",
                        (now or 0.0, subject, relation, value, json.dumps([list(x) for x in times]), said, source))
        return self._get(i)

    def _get(self, i: int) -> Fact:
        return _fact(self.rows(f"SELECT {_COLS} FROM care_profile WHERE id = ?", (i,))[0])

    def facts(self) -> list[Fact]:
        return [_fact(r) for r in self.rows(f"SELECT {_COLS} FROM care_profile WHERE active = 1 ORDER BY id")]

    def history(self, subject: str, relation: str) -> list[Fact]:
        return [_fact(r) for r in self.rows(f"SELECT {_COLS} FROM care_profile WHERE subject = ? AND relation = ? "
                                            "ORDER BY id", (subject, relation))]

    def get(self, subject: str, relation: str) -> Optional[str]:
        r = self.rows("SELECT value FROM care_profile WHERE subject = ? AND relation = ? AND active = 1 "
                      "ORDER BY id DESC LIMIT 1", (subject, relation))
        return r[0][0] if r else None

    def preferred_name(self) -> Optional[str]:
        return self.get("user", "preferred_name")

    def speech(self) -> dict:
        """What the user asked of the voice ({'speech_rate': 'slower'}), for a TTS that can apply it."""
        out = {}
        for k in ("speech_rate", "speech_volume"):
            v = self.get("user", k)
            if v:
                out[k] = v
        return out

    def forget_last(self) -> Optional[Fact]:
        fs = self.facts()
        if not fs:
            return None
        self.exec("UPDATE care_profile SET active = 0 WHERE id = ?", (fs[-1].id,))
        return fs[-1]

    def forget_all(self) -> int:
        return self.exec("UPDATE care_profile SET active = 0 WHERE active = 1")

    # -- words

    def phrase(self, f: Fact) -> str:
        """A fact as it is said back: 'your daughter is Sarah', 'your pills at 8 AM and 8 PM'."""
        if f.subject == "person":
            return f"your {f.relation} is {f.value}"
        if f.subject == "routine":
            meds = med_phrase(f.relation)
            return f"{meds} {f.value}" if meds else f"you {f.relation} {f.value}"
        if f.relation == "preferred_name":
            return f"you'd like me to call you {f.value}"
        if f.relation in ("speech_rate", "speech_volume"):
            return f"you'd like me to speak {f.value}"
        return f"your {f.relation.replace('_', ' ')} is {f.value}" if f.relation not in ("likes", "dislikes") \
            else f"you {'like' if f.relation == 'likes' else 'dislike'} {f.value}"

    def describe(self) -> str:
        fs = self.facts()
        if not fs:
            return ("I don't know anything about you yet. You can tell me things like 'call me Joe' or "
                    "'my daughter is Sarah'.")
        out = []
        name = self.preferred_name()
        if name:
            out.append(f"You asked me to call you {name}.")
        people = [self.phrase(f) for f in fs if f.subject == "person"]
        if people:
            out.append(_sentence(", and ".join(people)))
        routines = [self.phrase(f) for f in fs if f.subject == "routine"]
        if routines:
            out.append(f"Your routine{'s' if len(routines) > 1 else ''}: {'; '.join(routines)}.")
        speech = [f.value for f in fs if f.relation in ("speech_rate", "speech_volume")]
        if speech:
            out.append(f"You asked me to speak {' and '.join(speech)}.")
        other = [self.phrase(f) for f in fs if f.subject == "user" and f.relation not in
                 ("preferred_name", "speech_rate", "speech_volume")]
        if other:
            out.append(_sentence("; ".join(other)))
        return pill_guard(" ".join(out))

    def confirm(self, f: Fact) -> str:
        return pill_guard(self._confirm(f))

    def _confirm(self, f: Fact) -> str:
        if f.relation == "preferred_name":
            return f"Okay, I'll call you {f.value}."
        if f.subject == "person":
            return f"Got it, your {f.relation} is {f.value}."
        if f.subject == "routine":
            return f"Got it: {self.phrase(f)}."
        if f.relation in ("speech_rate", "speech_volume"):
            return f"Okay, I've noted that you'd like me to speak {f.value}."
        return f"Got it: {self.phrase(f)}."

    # -- voice

    def handle(self, text: str, now: float) -> Optional[Reply]:
        """'what do you know about me', 'forget that', or a stated fact; None for anything else."""
        t = _norm(text)
        if KNOW_RX.search(t):
            return Reply(Answer(self.describe()))
        if FORGET_ALL.search(t):
            n = self.forget_all()
            return Reply(Answer("Okay, I've forgotten everything you told me about yourself." if n
                                else "There's nothing to forget."))
        m = FORGET_REL.match(t)
        if m:
            rel = m.group("rel")
            subj, rel = ("user", "preferred_name") if rel == "name" else ("person", rel)
            gone = self.exec("UPDATE care_profile SET active = 0 WHERE subject = ? AND relation = ? AND active = 1",
                             (subj, rel))
            return Reply(Answer("Okay, I've forgotten that." if gone else "I didn't know that anyway."))
        if FORGET_LAST.match(t):
            f = self.forget_last()
            return Reply(Answer(pill_guard(f"Okay, I've forgotten that {self.phrase(f)}.") if f
                                else "There's nothing to forget."))
        stmts = extract(text)
        if not stmts:
            return None
        facts = [self.remember(s.subject, s.relation, s.value, text, now, s.times) for s in stmts]
        f = facts[-1]
        sug = Suggestion(f.relation, list(f.times)) if f.subject == "routine" and f.times else None
        return Reply(Answer(" ".join(self.confirm(x) for x in facts)), sug)

    def learn(self, text: str, now: float, online: bool) -> list[Fact]:
        """Free-form statements through the model (online only); stores what validates. Never raises."""
        if not online or self.extractor is None or not STATEMENT.match(_norm(text)) or text.strip().endswith("?"):
            return []
        try:
            stmts = validate(self.extractor(text), text)
        except Exception as e:
            log.warning("profile extraction failed: %s", e)
            return []
        return [self.remember(s.subject, s.relation, s.value, text, now, s.times, source="llm") for s in stmts]


def _selftest(argv: list[str]) -> int:
    from core.config import load_config
    text = " ".join(argv) or "I love gardening and my granddaughter Lily visits on Sundays"
    if not os.environ.get("XAI_API_KEY", "").strip():
        print("XAI_API_KEY is not set; nothing to test against Grok.")
        return 2
    raw = make_grok_extractor(load_config())(text)
    print("said:     ", text)
    print("grok:     ", json.dumps(raw))
    print("kept:     ", [(s.subject, s.relation, s.value) for s in validate(raw, text)] if raw else [])
    print("patterns: ", [(s.subject, s.relation, s.value) for s in extract(text)])
    return 0 if raw is not None else 1


if __name__ == "__main__":
    if sys.argv[1:2] == ["--selftest"]:
        raise SystemExit(_selftest(sys.argv[2:]))
    print(__doc__)
