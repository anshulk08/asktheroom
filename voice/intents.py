"""Offline rule-based intent parser (spec V3): transcript -> Intent(kind, obj, raw).

Pipeline: lowercase, drop apostrophes/punctuation, drop filler words, map synonyms and object
names to canonical names (longest phrase first, whole words), then trigger regexes in priority
order. Priority (first match wins), chosen so overlaps resolve sensibly:

  RECAL  > RESET                    'reset the calibration' is a recalibration
  WHERE ('where...')                'where did anyone move my wallet' is a location question
  HISTORY ('who' / 'when' opening the question)
                                    'when did somebody last grab the pills' asks when, not yes or no
  HANDLED (did/has <someone> <touch verb>, or passive 'been moved' / 'was the remote touched')
                                    'did I move my keys' -> HANDLED; 'who moved the box' -> HISTORY
  CHANGES (changed/different/while I was gone); with an object it becomes HISTORY
  HISTORY (what happened / who / when)
                                    'when did I last see my keys' -> HISTORY (the answer's event
                                    times say when; a 'where' in the question would win instead)
  WHERE (find/seen/lost/locate, or 'is/are my X', 'did I leave X', or just 'my X'; with a thing
         named: show / point at / light up / highlight, which want the laser)
  OTHER                             open-ended; routed to the LLM when online

TEACH ('this is my X', 'remember this as X', 'call this my X') is checked first, on the words as
spoken, and only as the whole sentence ('lets call it a day', 'what do you call that', 'it is a mess'
don't teach): obj and name are the new name ('phone charger'), even when it is a configured object's name,
so the answer can refuse it politely. Open world: when a WHERE / HISTORY / HANDLED question names no
configured object, Intent.name keeps the spoken noun phrase after my/the ('where is my charger' ->
name 'charger', obj None); answers resolve it through world.find. parse(..., aliases=[...]) matches
taught aliases like object names (longest phrase first), with obj = the alias.

Mishearings: Whisper turns 'where's my wallet' into 'wears my wall it'. A leading 'wear(s)' before
my/the/is/are reads as 'where', and when no name matches exactly, a 1-2 word span after my/the/your/our
that sounds like an object or synonym (fuzzy_match: same spelling without spaces, close spelling, or the
same consonants) stands for it: 'wall it' -> wallet, 'kiss' -> keys, 'note book' -> notebook.
matched_exactly() tells the interpreter (voice.understand) which it was, so a model can overrule a guess.

HISTORY/HANDLED with no object but a general word ('anything', 'stuff', 'what happened')
become CHANGES: 'what happened while I was away', 'did anyone touch anything'.

WHAT_DOING (narration memory): 'what was I doing before lunch', 'what did I do this morning', and a
CHANGES question with a time window ('what happened at 3') that is not about being away. It sits after
HANDLED, so 'did I take my pills this morning' stays HANDLED.
"""
from __future__ import annotations

import difflib
import re
from typing import Optional

from core.config import display_name
from core.narration_store import has_time_phrase
from core.types import Intent

__all__ = ["Intent", "parse", "normalize", "fuzzy_match", "matched_exactly", "firm_question", "names_a_person"]

FILLER = {"um", "umm", "uh", "uhh", "uhm", "er", "erm", "ah", "eh", "hmm", "hm", "hey", "hi",
          "okay", "ok", "so", "like", "please", "well", "oh", "actually", "yo"}

_SUBJ = r"(?:i|anyone|anybody|someone|somebody|you|we|they|he|she)"
_VERB = (r"(?:take|takes|took|taken|taking|touch\w*|pick\w*|move[ds]?|moving|grab\w*|"
         r"handle[ds]?|handling|use[ds]?|using|open\w*|mess\w*|fiddl\w*|tamper\w*|disturb\w*|"
         r"been (?:near|at|into|in|through)|g[eo]t into)\b")
_DONE = r"(?:touched|moved|picked|taken|handled|grabbed|used|opened|messed|disturbed|tampered)"

RECAL = re.compile(r"\b(?:re)?calibrat\w*")
RESET = re.compile(r"\breset\b|\bstart (?:over|fresh)\b")
WHERE = re.compile(r"\bwhere")  # where, whered, wheres, whereabouts (not 'anywhere')
HANDLED_ASKED = re.compile(                      # 'did anyone move', 'has it been touched'
    rf"\b(?:(?:did|have|has|had)\s+(?:{_SUBJ}|(?:my|the|our|your)\s+[a-z]+)|ive|weve)\b(?:\s+\w+){{0,2}}?\s+{_VERB}"
    rf"|\b(?:been|get|got|gotten|was|were)\s+{_DONE}\b")
HANDLED_PASSIVE = re.compile(                    # 'was the remote touched' (also 'were the posters moved')
    rf"\b(?:was|were|is|are|has|have)\s+(?:my|the|our|your)\s+[a-z_]+(?:\s+[a-z_]+)?\s+(?:been\s+)?{_DONE}\b")
HANDLED = re.compile(f"{HANDLED_ASKED.pattern}|{HANDLED_PASSIVE.pattern}")
# 'who ...' / 'when ...' opening the question asks for the history, even with a touch verb in it.
ASKS_WHO_WHEN = re.compile(r"^(?:and\s+|now\s+|then\s+)?(?:who|whos|when|whens)\b")
CHANGES = re.compile(r"\bchang(?:e|ed|es|ing)\b|\bdifferent\b|\bwhat did i miss\b|\bmissed\b"
                     r"|\bwhile i was (?:gone|away|out)\b|\banything new\b|\bwhats new\b")
AWAY = re.compile(r"\bwhile i was (?:gone|away|out)\b|\bwhat did i miss\b|\bsince i left\b")
# Narration memory (core/narration.py): 'what was I doing before lunch', 'what did I do this morning'.
# 'what did I do with X' stays HISTORY; with an object ('what was I doing with my keys') it becomes HISTORY.
WHAT_DOING = re.compile(r"\bwhat (?:was|were|am|have|had) (?:i|we)(?: been)? (?:doing|up to|working on|busy with)\b"
                        r"|\bwhat did (?:i|we) (?:do|get up to|work on|get done)\b(?!\s+with\b)"
                        r"|\bwhat (?:went on|was going on|was happening)\b|\bsummar(?:y|ize|ise)\b")
HISTORY = re.compile(r"\bwhat(?:s| has| had)? happen\w*|\bhappened to\b|\bwho\b|\bwhos\b|\bwhen\b"
                     r"|\b(?:story|deal|going on) with\b"
                     r"|\bhistory\b|\blast time\b|\bwhat did (?:i|you|someone|somebody|anyone) do with\b")
WHERE2 = re.compile(r"\b(?:find|found|seen|locate\w*|lost|misplaced|look(?:ing)? for|spot(?:ted)?)\b")
# Wants the laser on a named thing: 'show me my keys', 'point at the pills', 'light up the remote'.
SHOW = re.compile(r"\b(?:show|point|pointing|light up|highlight|shine|flash|aim|where\s*abouts)\b")
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
    # participles end a name too: 'was my charger moved' -> 'charger'
    "moved", "touched", "taken", "took", "grabbed", "picked", "handled", "used", "opened", "messed",
    "disturbed", "tampered", "gone", "missing", "lost", "stolen",
}
# Places in a room (spec 0010: the room demo). A spoken phrase made only of these is where something is,
# never the name of a thing: 'where's the couch' is not an untaught object, 'the kitchen counter' is a spot.
PLACES = {
    "couch", "sofa", "loveseat", "armchair", "chair", "chairs", "stool", "bench", "ottoman", "side", "end",
    "coffee", "table", "tables", "desk", "counter", "countertop", "kitchen", "living", "dining", "bedroom",
    "room", "shelf", "shelves", "bookshelf", "bookcase", "cabinet", "cupboard", "drawer", "dresser",
    "nightstand", "bed", "floor", "rug", "carpet", "corner", "window", "windowsill", "sill", "door",
    "doorway", "hallway", "hall", "wall", "tv", "stand", "fridge", "sink", "stove", "microwave", "island",
    "mantel", "mantle", "fireplace", "entryway", "closet", "porch", "bathroom",
}
# 'is it on the counter?', 'are they under the couch': a follow-up asking whether the last thing is there
PRONOUN_WHERE = re.compile(r"^(?:and\s+)?(?:is|are|was|were)\s+(?:it|they|them|those|that|these|this)\s+(?:still\s+)?"
                           r"(?:on|in|under|at|near|by|inside|behind|beside|next to|underneath|beneath|over)\b")


def _names_a_place(t: str) -> bool:
    """t (normalized) says 'the/my <place>' and asks about it, not about an 'it' on it."""
    if PRONOUN_WHERE.search(t):
        return False
    for m in _POSS.finditer(t):
        words = []
        for w in m.group(1).split():
            if w in ARTICLES or (w in NAME_STOP and w not in PLACES):
                break
            words.append(w)
        if words and is_place(" ".join(words)):
            return True
    return False


def is_place(name: Optional[str]) -> bool:
    """A spoken phrase that names a place in the room ('couch', 'kitchen counter'), not a thing."""
    words = (name or "").split()
    return bool(words) and all(w in PLACES for w in words)


_POSS = re.compile(r"\b(?:my|the|your|our)\s+([a-z0-9]+(?:\s+[a-z0-9]+){0,3})")


# Whisper's 'where' mishearings, only as the question's first word: 'wears my wall it'.
_MISHEARD_WHERE = re.compile(r"^(?:wear|wears|ware|wares|wheres?e)\b(?=\s+(?:my|the|your|our|is|are|did|do|does|has|have)\b)")


def normalize(text: str) -> str:
    """Lowercase, drop apostrophes ('where'd' -> 'whered'), punctuation -> space, drop fillers, and
    read a leading 'wears my' as 'wheres my'."""
    t = re.sub(r"['’`]", "", text.lower())
    t = re.sub(r"[^a-z0-9\s]", " ", t)
    t = " ".join(w for w in t.split() if w not in FILLER)
    return _MISHEARD_WHERE.sub("wheres", t)


FUZZY_RATIO = 0.8                       # difflib ratio of the spans without spaces ('wall it' ~ wallet)
_SOUNDS = [(r"ph", "f"), (r"ck", "k"), (r"[cq]", "k"), (r"x", "ks"), (r"z", "s")]


def _consonants(s: str) -> str:
    """A rough sound key: first letter, then the consonants, with look-alike letters merged and
    repeats collapsed. 'kiss' and 'keys' -> 'ks'; 'wall it' and 'wallet' -> 'wlt'."""
    s = re.sub(r"[^a-z]", "", s)
    if not s:
        return ""
    for a, b in _SOUNDS:
        s = re.sub(a, b, s)
    key = s[0] + re.sub(r"[aeiouyhw]", "", s[1:])
    return re.sub(r"(.)\1+", r"\1", key)


def _stem(s: str) -> str:
    return re.sub(r"(?:es|s)$", "", s) if len(s) > 3 else s


def fuzzy_match(span: str, name: str) -> float:
    """How much a heard span sounds like a name, 0..1 (0: not a match). The same letters without
    spaces ('note book'), a close spelling (ratio >= FUZZY_RATIO), or the same consonant key with the
    same first letter and half the letters in common, for short words only ('kiss' ~ 'keys'; 'kids',
    'case' and 'papers' ~ 'purse' are not)."""
    a, b = span.replace(" ", ""), name.replace(" ", "").replace("_", "")
    if len(a) < 3 or len(b) < 3:
        return 0.0
    if a == b or _stem(a) == _stem(b):
        return 1.0
    if _stem(a).startswith(_stem(b)) and len(_stem(a)) - len(_stem(b)) >= 2:
        return 0.0                      # a longer word: 'pillow' is not the pills, 'keyboard' not the keys
    r = difflib.SequenceMatcher(None, _stem(a), _stem(b)).ratio()
    if r >= FUZZY_RATIO:
        return r
    ka, kb = _consonants(a), _consonants(b)
    raw = difflib.SequenceMatcher(None, a, b).ratio()
    if len(kb) >= 2 and ka == kb and a[0] == b[0] and raw >= 0.5 and max(len(a), len(b)) <= 4:
        return 0.5 + raw / 4
    return 0.0


def _join_split(t: str, phrases: dict[str, str]) -> str:
    """Rejoin a name Whisper split in two: 'the note book' -> 'the notebook'."""
    words = t.split()
    out, i = [], 0
    while i < len(words):
        if i + 1 < len(words) and words[i] + words[i + 1] in phrases:
            out.append(words[i] + words[i + 1])
            i += 2
        else:
            out.append(words[i])
            i += 1
    return " ".join(out)


_DET_SPAN = re.compile(r"\b(?:my|the|your|our)\s+([a-z]+)(?:\s+([a-z]+))?")


def _fuzzy(t: str, phrases: dict[str, str]) -> Optional[tuple[str, str]]:
    """(heard span, phrase) for the best 1-2 word span after my/the/your/our that sounds like an
    object name or synonym, or None. The span must cover the whole spoken noun phrase: in 'my wall
    charger', 'wall' is part of a longer name, not a misheard wallet ('wall it' is: 'it' ends a name)."""
    best, score = None, 0.0
    for m in _DET_SPAN.finditer(t):
        one = m.group(1)
        if one in NAME_STOP or one in ARTICLES:
            continue
        noun = 0                        # words in the spoken noun phrase after the determiner
        for w in t[m.start(1):].split()[:4]:
            if w in NAME_STOP or w in ARTICLES:
                break
            noun += 1
        spans = [one] + ([f"{one} {m.group(2)}"] if m.group(2) else [])
        for span in spans:
            if len(span.split()) < noun:
                continue
            for p in phrases:
                sc = fuzzy_match(span, p)
                if sc > score:
                    best, score = (span, p), sc
    return best


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


def firm_question(text: str) -> bool:
    """Phrased as a question about a thing, not just with a show verb or a passive ('show me the money',
    'were the posters moved' are just as likely said to a person). Overheard speech about a name nobody
    taught needs this (voice.understand.screen)."""
    t = normalize(text)
    return bool(WHERE.search(t) or WHERE2.search(t) or HANDLED_ASKED.search(t) or HISTORY.search(t)
                or ASKS_WHO_WHEN.search(t))


# 'this is my wife Karen' introduces a person; overheard, it isn't teaching a thing's name.
PEOPLE = {"wife", "husband", "partner", "friend", "friends", "girlfriend", "boyfriend", "fiance", "fiancee",
          "son", "daughter", "kid", "kids", "child", "children", "mom", "mum", "mother", "dad", "father",
          "brother", "sister", "sibling", "cousin", "aunt", "uncle", "grandma", "grandpa", "grandmother",
          "grandfather", "family", "boss", "manager", "colleague", "coworker", "teammate", "team", "mentor",
          "roommate", "buddy", "pal", "classmate", "professor", "teacher", "advisor", "neighbor", "neighbour",
          "guy", "guys", "man", "woman", "baby", "dog", "cat", "group", "project", "demo", "startup", "hack"}


def names_a_person(name: Optional[str], raw: str = "") -> bool:
    """A TEACH name that is a person, pet or the project ('wife karen', 'friend', 'team'), not a thing
    ('friend's mug', said with the possessive, is a mug)."""
    words = (name or "").split()
    if not words or words[0] not in PEOPLE:
        return False
    stem = words[0][:-1] if words[0].endswith("s") else words[0]    # normalize drops the apostrophe
    return not re.search(rf"\b{re.escape(stem)}['’]s\b", raw.lower())


def matched_exactly(text: str, obj: Optional[str], cfg: dict, aliases=()) -> bool:
    """Was obj (a parse() result) named in text as spoken, not guessed from a mishearing?"""
    if obj is None:
        return False
    phrases, rx = _vocab(cfg, aliases)
    return any(phrases[m.group(1)] == obj for m in rx.finditer(_join_split(normalize(text), phrases)))


def _without_wake_word(t: str, cfg: dict) -> str:
    """'room this is my mug' -> 'this is my mug': a teaching sentence may open with the wake word."""
    for w in (cfg.get("listen") or {}).get("wake_words") or ["room"]:
        w = normalize(str(w))
        if w and (t == w or t.startswith(w + " ")):
            return t[len(w):].strip()
    return t


def parse(text: str, cfg: dict, aliases=()) -> Intent:
    """Classify a transcript into an Intent with a canonical object name (or None). aliases: names
    taught for things (world.alias_phrases()), matched like object names."""
    taught = _teach_name(_without_wake_word(normalize(text), cfg))
    if taught:
        return Intent(kind="TEACH", obj=taught, raw=text, name=taught)
    phrases, rx = _vocab(cfg, aliases)
    found: list[str] = []

    def sub(m: re.Match) -> str:
        found.append(phrases[m.group(1)])
        return found[-1]

    t = rx.sub(sub, _join_split(normalize(text), phrases))
    if not found:
        heard = _fuzzy(t, phrases)
        if heard is not None:           # 'wears my wall it' -> 'wheres my wallet'
            found.append(phrases[heard[1]])
            t = re.sub(rf"\b{re.escape(heard[0])}\b", found[-1], t, count=1)
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
    elif ASKS_WHO_WHEN.search(t):
        kind = "HISTORY"
    elif HANDLED.search(t):
        kind = "HANDLED"
    elif WHAT_DOING.search(t):
        kind = "HISTORY" if obj or re.search(r"\bwith (?:my|the|our)\b", t) else "WHAT_DOING"
    elif CHANGES.search(t):
        kind = "HISTORY" if obj else "CHANGES"
    elif HISTORY.search(t):
        kind = "HISTORY"
    elif WHERE2.search(t) or (SHOW.search(t) and (obj or spoken)):
        kind = "WHERE"
    elif obj and (re.search(rf"\b(?:is|are)\s+(?:my|the|your|our)?\s*{re.escape(obj)}\b", t)
                  or re.search(r"\b(?:leave|left|put|set|drop|dropped|place|placed)\b", t)
                  or [w for w in t.split() if w not in ARTICLES] == [obj]):
        kind = "WHERE"
    elif obj is None and (re.search(r"\b(?:is|are)\s+(?:my|your|our)\s", t)
                          or re.fullmatch(r"(?:my|our)\s+[a-z0-9]+(?:\s+[a-z0-9]+)?", t)):
        kind = "WHERE"                  # 'is my charger in the box', 'my charger?'
    elif obj is None and PRONOUN_WHERE.search(t):
        kind = "WHERE"                  # 'is it on the counter?': voice.conversation fills in the thing
    else:
        kind = "OTHER"
    if kind == "WHERE" and obj is None and is_place(spoken) and PRONOUN_WHERE.search(t):
        spoken = None                   # 'is it on the counter?': 'it' is the thing, the counter only where
    if kind == "WHERE" and obj is None and (spoken is None or is_place(spoken)) and _names_a_place(t):
        kind, spoken = "OTHER", None    # 'where's the couch': a place, not a thing to find ('did I use the
                                        # stove' keeps its name: the narration memory answers that)

    name = spoken if kind in ("WHERE", "HISTORY", "HANDLED") and obj is None else None
    if kind in ("HISTORY", "HANDLED") and obj is None and name is None and GENERAL.search(t):
        kind = "CHANGES"
    if kind == "CHANGES" and not AWAY.search(t) and has_time_phrase(text):
        kind = "WHAT_DOING"             # 'what happened this morning': a time window, not 'what changed'
    if kind in ("RESET", "RECAL", "CHANGES", "WHAT_DOING"):
        obj = None
    return Intent(kind=kind, obj=obj, raw=text, name=name)
