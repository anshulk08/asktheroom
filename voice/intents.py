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

TEACH ('this is my X', 'remember this as X', 'call this my X') is checked first, on the words as
spoken, and only as the whole sentence ('lets call it a day', 'what do you call that', 'it is a mess'
don't teach): obj and name are the new name ('phone charger'), even when it is a configured object's name,
so the answer can refuse it politely. Open world: when a WHERE / HISTORY / HANDLED question names no
configured object, Intent.name keeps the spoken noun phrase after my/the ('where is my charger' ->
name 'charger', obj None); answers resolve it through world.find. parse(..., aliases=[...]) matches
taught aliases like object names (longest phrase first), with obj = the alias.

HISTORY/HANDLED with no object but a general word ('anything', 'stuff', 'what happened')
become CHANGES: 'what happened while I was away', 'did anyone touch anything'.

WHAT_DOING (narration memory): 'what was I doing before lunch', 'what did I do this morning', and a
CHANGES question with a time window ('what happened at 3') that is not about being away. It sits after
HANDLED, so 'did I take my pills this morning' stays HANDLED.
"""
from __future__ import annotations

import re
from typing import Optional

from core.config import display_name
from core.narration_store import has_time_phrase
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
AWAY = re.compile(r"\bwhile i was (?:gone|away|out)\b|\bwhat did i miss\b|\bsince i left\b")
# Narration memory (core/narration.py): 'what was I doing before lunch', 'what did I do this morning'.
# 'what did I do with X' stays HISTORY; with an object ('what was I doing with my keys') it becomes HISTORY.
WHAT_DOING = re.compile(r"\bwhat (?:was|were|am|have|had) (?:i|we)(?: been)? (?:doing|up to|working on|busy with)\b"
                        r"|\bwhat did (?:i|we) (?:do|get up to|work on|get done)\b(?!\s+with\b)"
                        r"|\bwhat (?:went on|was going on|was happening)\b|\bsummar(?:y|ize|ise)\b")
HISTORY = re.compile(r"\bwhat(?:s| has| had)? happen\w*|\bhappened to\b|\bwho\b|\bwhen\b"
                     r"|\bhistory\b|\blast time\b|\bwhat did (?:i|you|someone|somebody|anyone) do with\b")
WHERE2 = re.compile(r"\b(?:find|found|seen|locate\w*|lost|misplaced|look(?:ing)? for|spot(?:ted)?)\b")
GENERAL = re.compile(r"\b(?:anything|something|everything|stuff|things|what happened"
                     r"|whats happened|while i was)\b")
ARTICLES = {"my", "the", "a", "your", "our"}

_ART = r"(?:my|the|our|a|an|his|her|their)"
_OWN = r"(?:my|our|his|her|their)"
# Whole sentence, said as a statement: 'what do you call that thing', 'lets call it a day', 'it is a
# mess' and 'thats the problem' are not teaching. 'a/an' only after 'called'.
_START = r"^(?:and\s+|now\s+)?"
TEACH = [re.compile(p) for p in (
    rf"{_START}(?:this|that)(?:\s+one|\s+thing)?\s+is\s+(?:called\s+(?:{_ART}\s+)?|(?:{_OWN}|the)\s+)(?P<n>.+)$",
    rf"{_START}thats\s+{_OWN}\s+(?P<n>.+)$",
    rf"{_START}(?:(?:can|could|will|would)\s+you\s+)?remember\s+(?:this|it|that)(?:\s+one|\s+thing)?\s+as\s+"
    rf"(?:{_ART}\s+)?(?P<n>.+)$",
    rf"{_START}call\s+(?:(?:this|that)(?:\s+one|\s+thing)?\s+(?:{_ART}\s+)?|it\s+{_OWN}\s+)(?P<n>.+)$",
)]
TEACH_TAIL = {"here", "now", "right", "please", "thanks", "thank", "you", "ok", "okay"}
NOT_A_NAME = {"day", "mess", "point", "bad", "fault", "problem", "idea", "thing", "deal", "plan", "turn", "job",
              "life", "way", "one", "question", "guess", "best", "worst", "last", "first", "end", "even", "quits"}
# Words that end a spoken name ('where is my charger in the box' -> 'charger'), or are not one.
NAME_STOP = {
    "is", "are", "was", "were", "be", "been", "go", "gone", "went", "to", "at", "in", "on", "under",
    "inside", "into", "near", "by", "from", "with", "and", "or", "but", "now", "today", "tonight",
    "yesterday", "morning", "right", "this", "that", "these", "those", "it", "them", "anywhere",
    "again", "did", "do", "does", "i", "you", "we", "he", "she", "they", "last", "put", "left", "leave",
    "before", "after", "here", "there", "get", "got", "table", "side", "room", "top", "bottom",
    "middle", "anything", "something", "everything", "stuff", "things", "thing", "one", "ones",
    "weather", "time", "please", "day", "calibration", "laser", "for", "of", "about", "lately",
    "show", "showed", "shown", "appear", "appeared", "arrive", "arrived", "turn", "turned", "come", "came",
}
_POSS = re.compile(r"\b(?:my|the|your|our)\s+([a-z0-9]+(?:\s+[a-z0-9]+){0,3})")


def normalize(text: str) -> str:
    """Lowercase, drop apostrophes ('where'd' -> 'whered'), punctuation -> space, drop fillers."""
    t = re.sub(r"['’`]", "", text.lower())
    t = re.sub(r"[^a-z0-9\s]", " ", t)
    return " ".join(w for w in t.split() if w not in FILLER)


def _vocab(cfg: dict, aliases=()) -> tuple[dict[str, str], re.Pattern]:
    """Spoken phrase -> canonical object (or taught alias), plus one whole-word regex (longest
    phrases first)."""
    objs = cfg.get("objects") or {}
    phrases: dict[str, str] = {}
    for o in objs:
        for p in (o, o.replace("_", " "), display_name(cfg, o)):
            phrases[normalize(p) or p] = o
    for k, v in (cfg.get("synonyms") or {}).items():
        if v in objs:
            phrases[normalize(str(k))] = v
    for a in aliases:
        if normalize(a):
            phrases[normalize(a)] = normalize(a)
    alts = sorted(phrases, key=lambda p: (-len(p.split()), -len(p)))
    rx = re.compile(r"(?<!\w)(" + "|".join(re.escape(p) for p in alts) + r")(?:e?s)?(?!\w)")
    return phrases, rx


def _pick(found: list[str], cfg: dict) -> Optional[str]:
    """First target mentioned wins ('are my keys in the box' -> keys); else first object. A taught
    alias counts as a target (it names one of the user's things)."""
    kinds = cfg.get("objects") or {}
    for o in found:
        if kinds.get(o, "target") == "target":
            return o
    return found[0] if found else None


def _teach_name(t: str) -> Optional[str]:
    """The new name in 'this is my X' / 'remember this as X' / 'call this X', as spoken."""
    for rx in TEACH:
        m = rx.search(t)
        if m:
            words = m.group("n").split()
            while words and words[-1] in TEACH_TAIL:
                words.pop()
            while words and words[0] in ARTICLES:
                words.pop(0)
            name = " ".join(words[:4])
            return name if name and name not in NOT_A_NAME else None   # 'thats my point'
    return None


def _spoken_name(t: str) -> Optional[str]:
    """First noun phrase after my/the/your/our, cut at the first word that cannot be part of a name."""
    for m in _POSS.finditer(t):
        words = []
        for w in m.group(1).split():
            if w in NAME_STOP or w in ARTICLES:
                break
            words.append(w)
        if words:
            return " ".join(words)
    return None


def parse(text: str, cfg: dict, aliases=()) -> Intent:
    """Classify a transcript into an Intent with a canonical object name (or None). aliases: names
    taught for things (world.alias_phrases()), matched like object names."""
    taught = _teach_name(normalize(text))
    if taught:
        return Intent(kind="TEACH", obj=taught, raw=text, name=taught)
    phrases, rx = _vocab(cfg, aliases)
    found: list[str] = []

    def sub(m: re.Match) -> str:
        found.append(phrases[m.group(1)])
        return found[-1]

    t = rx.sub(sub, normalize(text))
    obj = _pick(found, cfg)
    spoken = _spoken_name(t)
    if obj is not None and (cfg.get("objects") or {}).get(obj, "target") != "target" \
            and spoken and spoken != obj:
        obj = None                      # 'is my charger in the box': the box is where, not what

    if RECAL.search(t):
        kind = "RECAL"
    elif RESET.search(t):
        kind = "RESET"
    elif WHERE.search(t):
        kind = "WHERE"
    elif HANDLED.search(t):
        kind = "HANDLED"
    elif WHAT_DOING.search(t):
        kind = "HISTORY" if obj or re.search(r"\bwith (?:my|the|our)\b", t) else "WHAT_DOING"
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
    elif obj is None and (re.search(r"\b(?:is|are)\s+(?:my|your|our)\s", t)
                          or re.fullmatch(r"(?:my|our)\s+[a-z0-9]+(?:\s+[a-z0-9]+)?", t)):
        kind = "WHERE"                  # 'is my charger in the box', 'my charger?'
    else:
        kind = "OTHER"

    name = spoken if kind in ("WHERE", "HISTORY", "HANDLED") and obj is None else None
    if kind in ("HISTORY", "HANDLED") and obj is None and name is None and GENERAL.search(t):
        kind = "CHANGES"
    if kind == "CHANGES" and not AWAY.search(t) and has_time_phrase(text):
        kind = "WHAT_DOING"             # 'what happened this morning': a time window, not 'what changed'
    if kind in ("RESET", "RECAL", "CHANGES", "WHAT_DOING"):
        obj = None
    return Intent(kind=kind, obj=obj, raw=text, name=name)
