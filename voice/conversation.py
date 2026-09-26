"""Short-term conversation memory (care layer, feature 3): follow-up questions, resolved deterministically.

Each source (voice, dashboard, sms, ...) keeps its last few turns: the question, what the router was
asked, the intent, the entity it was about, the answer and any alternatives the answer offered. Turns
expire after care.conversation_ttl_s of silence. A follow-up is either rewritten into a full question
for the normal router (so the offline templates answer it) or answered here:

  'and my glasses?' / 'what about my keys?'   same intent as last time, new object     -> rewritten
  'when?' / 'how long ago?'                   history of the last entity                -> rewritten
  'who moved them?' / 'did anyone touch it?'  pronoun -> the last entity                -> rewritten
  'point at it again' / 'show me'             re-aim at what was just pointed at        -> answered
  'where was it before that?'                 the previous place, from the event log    -> answered
  'what about the other one?'                 the next alternative the answer offered   -> answered
                                              (another possible container, or a look-alike thing)

context(source) gives the live turns as chat messages for an LLM. voice/llm.py has no parameter for
them yet; see voice/care.py for where they would be passed.

Conversational follow-ups for people with memory loss are an idea from Project Memoria (MIT); the
resolution rules here are our own, over our intents, world model and event log.
"""
from __future__ import annotations

import re
import threading
from collections import deque
from dataclasses import dataclass, field, replace
from typing import Callable, Optional

from core.carewords import event_place, plural, spoken, where_sentence, your
from core.config import display_name
from core.types import Answer
from voice.answers import ago, area
from voice.intents import normalize, parse

OBJECT_KINDS = ("WHERE", "HISTORY", "HANDLED")
TEMPLATES = {"WHERE": "where is my {n}", "HISTORY": "what happened to my {n}", "HANDLED": "did anyone move my {n}"}

OTHER_ONE = re.compile(r"\b(?:the )?other one\b|\bthe other\b|\banother one\b|\bwhere else\b|\bany others?\b")
AGAIN = re.compile(r"^(?:(?:can|could) you )?(?:point (?:at|to) (?:it|them|that|those)(?: again)?|show me"
                   r"(?: (?:it|them|that|again|where|please))*|show (?:it|them)(?: to me)?(?: again)?|point again"
                   r"|again|where (?:is it|are they|was it|were they) again|one more time)$")
BEFORE = re.compile(r"\bbefore that\b|\bwhere (?:was|were) (?:it|they) before\b|\bwhere did (?:it|they) "
                    r"(?:come from|used? to be)\b|\bwhere (?:was|were) (?:it|they) previously\b")
WHEN = re.compile(r"^(?:and )?(?:when|what time|how long ago|how long|since when)(?: (?:was|did) "
                  r"(?:that|it|this|they)(?: happen)?)?$")
AND_OBJ = re.compile(r"^(?:and|what about|how about|and what about|what of|and how about)\s+(?P<n>.+)$")
PRONOUN = re.compile(r"\b(?:it|them|that|they|those)\b")


@dataclass
class Turn:
    t: float
    question: str                       # as heard
    resolved: str                       # what the router was asked (or the question, if answered here)
    kind: str
    obj: Optional[str]                  # the entity it was about
    name: Optional[str]                 # how to say it in a rewritten question
    answer: str
    point_at: Optional[str]
    action: Optional[str]
    alternatives: list = field(default_factory=list)   # [('place' | 'thing', entity)]
    shown: int = 0                      # alternatives already offered by 'the other one'


@dataclass
class Followup:
    text: Optional[str] = None          # a rewritten question for the router
    answer: Optional[Answer] = None     # or the answer itself
    base: Optional[Turn] = None         # the turn it follows up


def _aliases(world) -> list:
    try:
        return world.alias_phrases() if hasattr(world, "alias_phrases") else []
    except Exception:
        return []


def entity_of(text: str, world, cfg: dict) -> tuple[str, Optional[str], Optional[str]]:
    """(intent kind, entity, spoken name) for a question, resolved like the answers do."""
    it = parse(text, cfg, aliases=_aliases(world))
    obj = None
    for n in (it.obj, it.name):
        if not n:
            continue
        if n in (cfg.get("objects") or {}) and not hasattr(world, "find"):
            obj = n
        elif hasattr(world, "find"):
            try:
                obj = world.find(n)
            except Exception:
                obj = None
        if obj:
            break
    if obj is None:
        return it.kind, None, it.name
    name = display_name(cfg, obj) if not obj.startswith("thing:") else (
        spoken(world, cfg, obj) if spoken(world, cfg, obj) != "thing I haven't been told about" else it.name)
    return it.kind, obj, name


def _alternatives(world, obj: Optional[str]) -> list:
    if obj is None:
        return []
    out = []
    try:
        e = world.get(obj)
        out += [("place", c) for c in e.candidates if c != e.parent]
        out += [("thing", n) for n, _ in (world.similar_to(obj) if hasattr(world, "similar_to") else [])]
        out += [("thing", n) for n, _ in (getattr(e, "maybe_same_as", None) or [])]
    except Exception:
        pass
    seen, uniq = set(), []
    for a in out:
        if a[1] not in seen:
            seen.add(a[1])
            uniq.append(a)
    return uniq


class Conversation:
    """Per-source turns; thread-safe (voice, dashboard and SMS threads ask concurrently)."""

    def __init__(self, cfg: Optional[dict] = None, ttl_s: Optional[float] = None, max_turns: Optional[int] = None):
        c = (cfg or {}).get("care") or {}
        self.ttl_s = float(ttl_s if ttl_s is not None else c.get("conversation_ttl_s", 120))
        self.max_turns = int(max_turns if max_turns is not None else c.get("conversation_turns", 5))
        self._turns: dict[str, deque] = {}
        self._lock = threading.Lock()

    def turns(self, source: str, now: float) -> list[Turn]:
        """Live turns, oldest first; all of them expire together after ttl_s of silence."""
        with self._lock:
            dq = self._turns.get(source)
            if not dq:
                return []
            if now - dq[-1].t > self.ttl_s:
                dq.clear()
            return list(dq)

    def _last(self, source: str, now: float) -> tuple[Optional[Turn], Optional[Turn]]:
        """(latest turn, latest turn about an entity)."""
        ts = self.turns(source, now)
        return (ts[-1] if ts else None), next((t for t in reversed(ts) if t.obj), None)

    # -- resolution

    def resolve(self, text: str, source: str, world, events, cfg: dict, now: float) -> Optional[Followup]:
        """A follow-up for text given this source's live turns, or None (ask the router as is)."""
        last, ent = self._last(source, now)
        if last is None:
            return None
        t = normalize(text)
        if ent is not None and OTHER_ONE.search(t):
            return Followup(answer=self._other_one(ent, world, cfg), base=ent)
        if ent is not None and AGAIN.match(t):
            if ent.point_at:
                pl = plural(spoken(world, cfg, ent.point_at))
                return Followup(answer=Answer("Here they are." if pl else "Here it is.", point_at=ent.point_at,
                                              action=ent.action or "point"), base=ent)
            return Followup(text=TEMPLATES["WHERE"].format(n=ent.name), base=ent) if ent.name else None
        if ent is not None and BEFORE.search(t):
            return Followup(answer=self._before(ent, world, events, cfg, now), base=ent)
        if ent is not None and ent.name and WHEN.match(t):
            kind = "HANDLED" if ent.kind == "HANDLED" else "HISTORY"
            return Followup(text=TEMPLATES[kind].format(n=ent.name), base=ent)
        m = AND_OBJ.match(t)
        if m and not PRONOUN.fullmatch(m.group("n")):
            n = re.sub(r"^(?:(?:my|the|your|our)\s+)+", "", m.group("n")).strip()
            _, obj, name = entity_of(f"where is my {n}", world, cfg)
            if obj or re.match(r"(?:my|our)\s", m.group("n")):
                kind = last.kind if last.kind in OBJECT_KINDS else "WHERE"
                return Followup(text=TEMPLATES[kind].format(n=name or n), base=last)
            return None
        if ent is not None and ent.name and PRONOUN.search(t):
            kind, obj, name = entity_of(text, world, cfg)
            if kind in OBJECT_KINDS and obj is None and name is None:
                return Followup(text=PRONOUN.sub(f"my {ent.name}", t, count=1), base=ent)
        return None

    def _other_one(self, ent: Turn, world, cfg: dict) -> Answer:
        alts = ent.alternatives
        if not alts:
            return Answer(f"I only know of one {ent.name or spoken(world, cfg, ent.obj)}.")
        if ent.shown >= len(alts):
            return Answer("That's the only other place I know of." if alts[-1][0] == "place"
                          else "That's the only other one I know of.")
        kind, other = alts[ent.shown]
        if kind == "place":
            pron = "they" if plural(spoken(world, cfg, ent.obj)) else "it"
            try:
                cover = world.get(other).kind == "cover"
            except Exception:
                cover = (cfg.get("objects") or {}).get(other) == "cover"
            return Answer(f"Or {pron} could be {'under' if cover else 'inside'} the {spoken(world, cfg, other)}.",
                          point_at=other, action="point")
        s, act = where_sentence(world, cfg, other)
        text = re.sub(r"^(?:It's|They're)\b", "The other one is", s)
        if text == s:
            text = f"The other one: {s[0].lower()}{s[1:]}"
        return Answer(text, point_at=other if act else None, action=act)

    def _before(self, ent: Turn, world, events, cfg: dict, now: float) -> Answer:
        obj = ent.obj
        n = f"{your(world, cfg, obj)} {spoken(world, cfg, obj)}"
        was = "were" if plural(spoken(world, cfg, obj)) else "was"
        try:
            evs = list(reversed(world.history(obj, 50) if hasattr(world, "history") else events.last(obj, 50)))
        except Exception:
            evs = []
        segments, here = [], None          # (place, left at); where it is now
        for ev in evs:
            typ = str(ev.type)
            if typ == "PICKED_UP" and here is None and ev.from_cm:
                here = f"on the table, {area(ev.from_cm, cfg)}"
            new = event_place(ev, world, cfg)
            if typ in ("PICKED_UP", "LOST_TRACK") or (new is not None and new != here):
                if here is not None:
                    segments.append((here, ev.wall))
                here = new
        before = next(((p, t) for p, t in reversed(segments) if p != here), None)
        if before is None:
            if segments:
                return Answer(f"{n[0].upper()}{n[1:]} {was} {segments[-1][0]} before that too.")
            return Answer(f"I don't know where {n} {was} before that.")
        return Answer(f"Before that, {n} {was} {before[0]}, until {ago(before[1], now)}.")

    # -- recording and the whole step

    def record(self, source: str, question: str, resolved: str, ans: Answer, world, cfg: dict, now: float,
               follow: Optional[Followup] = None) -> Turn:
        if follow is not None and follow.answer is not None and follow.base is not None:
            b = follow.base
            shown = b.shown + (1 if OTHER_ONE.search(normalize(question)) and ans.point_at else 0)
            turn = replace(b, t=now, question=question, resolved=question, answer=ans.text,
                           point_at=ans.point_at or b.point_at, action=ans.action or b.action, shown=shown)
        else:
            kind, obj, name = entity_of(resolved, world, cfg)
            turn = Turn(now, question, resolved, kind, obj if kind in OBJECT_KINDS else None, name, ans.text,
                        ans.point_at, ans.action, _alternatives(world, obj) if kind in OBJECT_KINDS else [])
        with self._lock:
            dq = self._turns.get(source)
            if dq is None or (dq and now - dq[-1].t > self.ttl_s):
                dq = self._turns[source] = deque(maxlen=self.max_turns)
            dq.append(turn)
        return turn

    def ask(self, text: str, source: str, base_ask: Callable[[str, str], Answer], world, events, cfg: dict,
            now: float) -> Answer:
        """Resolve a follow-up, ask the router (or answer here), and remember the turn."""
        follow = self.resolve(text, source, world, events, cfg, now)
        if follow is not None and follow.answer is not None:
            ans, resolved = follow.answer, text
        else:
            resolved = follow.text if follow is not None else text
            ans = base_ask(resolved, source)
        self.record(source, text, resolved, ans, world, cfg, now, follow)
        return ans

    def context(self, source: str, now: float) -> list[dict]:
        """The live turns as chat messages (oldest first), e.g. for the Grok router."""
        out = []
        for t in self.turns(source, now):
            out += [{"role": "user", "content": t.question}, {"role": "assistant", "content": t.answer}]
        return out
