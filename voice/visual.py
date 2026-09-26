"""Visual questions: what the camera sees now (look) and what it saw before (recall), and the routing
that decides which layer answers a question.

Routing (VisualQA.route, called by voice/pipeline.py before the offline templates):
  - object location / history questions about known things stay with the world model (fast, offline);
  - activity questions (WHAT_DOING, 'while I was away') stay with narrations + events (voice/answers.py);
  - OTHER questions about the scene now go to look(); in the past tense, to recall(). An OTHER question
    that isn't about the table or what the camera sees ('what time is it', 'tell me a joke') returns None
    so the pipeline's `other` answerer takes it, online or offline;
  - WHERE for a name the world model doesn't know ('where is my red mug?') goes to look(), which can
    point at it; HISTORY / HANDLED for such a name goes to recall() when no narration mentions it.
  - WHERE for a tracked object the world has never placed (or, offline, for a name it doesn't know) uses
    the Grok settle check's newest sighting of it (core/grok_check.py, when on and it has one).
Offline, a question only the VLM could answer gets a spoken 'I need the connection' (the world model
still answers everything else). Over the hourly cap, route() returns None and the old path answers.

look(): set-of-marks grounding. Every VISIBLE tracked entity with a box (known objects and thing:N) is
drawn on the current frame (<= look_px) as a numbered yellow box; that marked frame, upscaled close-ups of
entities the question names (from core.crops.active(): only views the store knows were of that entity, each
labelled with when it was taken; attach() binds views to things after every world.update), the compact
world state and the question go to the VLM, which replies with strict JSON {answer, confidence,
mark: null | number, point: null | {x, y} fractions of image 1}. A mark points at its entity (the laser then follows it through world.resolve); a
point is mapped to table cm (frame px -> table homography) and points only when it lands inside a marked
entity's box (the smallest, when that one lies inside the others). The laser never aims at a raw table
position: a point on no tracked entity, or on an ambiguous overlap, gets the spoken answer and a brief
"not sure exactly where". Low confidence abstains: "I can't tell from here."

One observation per question: the frame is read once and the world state once right after it (marks,
names, tracker text and close-ups all come from that reading); a box the world last saw more than
MARK_FRESH_S from the frame's time is not drawn. When the reply comes back the chosen entity is checked
against the world as it is now: gone, merged, off the table, or (still visible) moved more than JUMP_CM
means no pointing and a short "it moved" note; hidden meanwhile, the laser follows it through
world.resolve as usual.

Why marks: measured with grok-4.3 on a 5-object desk photo, asking for boxes (Project Memoria's Gemini
box_2d convention, [ymin, xmin, ymax, xmax] 0-1000) scored 0/5 (mean IoU 0.14; pixel and fraction boxes
2/5), a bare point landed on 4/5, and picking among numbered candidate boxes got 5/5 in ~0.75 s. The
tracker already has the boxes; the VLM only has to choose.

recall(): archived frames (core/visual_memory.py) picked by text similarity and/or the question's time
window, each with its time and the world model's digest, plus the narrations and world events of that
window, go to the VLM; the answer cites times ('a red mug was on the left from about 9:10 to 9:40').
When nothing in the archive matches the words well enough, it abstains without a VLM call.

Every VLM answer passes the medication rule (core/narration_store.redact_meds) and is cut to two spoken
sentences. The prompts say the camera sees only the tabletop, and nothing else is claimed.
"""
from __future__ import annotations

import logging
import re
import time
from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Optional

import numpy as np

from core.config import display_name
from core.narration import NarrationConfig, NarrationError, ProviderError, _parse_json, event_line, make_provider
from core.narration_store import parse_window, redact_meds
from core.types import Answer, Intent, Status
from core.visual_memory import VisualArchive, VisualConfig, digest_text, make_embedder

if TYPE_CHECKING:
    from core.crops import Crop

log = logging.getLogger(__name__)

OFFLINE = ("I'm offline, so I can't look at the table for that right now. I can still tell you where "
           "your things are and what happened to them.")
CANT_SEE = "I can't see the table right now."
ABSTAIN = "I can't tell from here."
PILLS_SAFE = ("I can't tell whether medication was taken; I can only tell you where the pill bottle is "
              "and when it was moved.")
UNSURE_WHERE = "I'm not sure exactly where, so I won't point."
MOVED = "It moved while I was looking, so I won't point at it."
MARK_FRESH_S = 0.5         # a box the world last saw further than this from the frame's time is not drawn
JUMP_CM = 5.0              # a visible entity that moved more than this during the VLM call is not pointed at
HIDDEN = (Status.HELD.value, Status.UNDER.value, Status.INSIDE.value)
POINT_MARGIN_CM = 2.0      # a VLM point this close outside a marked box still lands on it

PAST = re.compile(r"\b(?:was|were|did|had|earlier|before|ago|yesterday|used to|show(?:ed|n)? up|appeared"
                  r"|last time|when i left|before i left|while i was)\b")
# An OTHER question is about the table (so the camera can answer it) when it says so; 'what time is it'
# or 'tell me a joke' is not, and goes to the pipeline's `other` answerer instead.
SCENE = re.compile(r"\b(?:table|desk|here|this|that|these|those|see|seen|saw|look|looks|looking|colou?rs?"
                   r"|whats on|what is on|how many|read|reads|note|notes|written|writing|shape|size)\b")
THING_Q = re.compile(r"^(?:is|are) (?:the|my|your|our|a|an|there)\b")     # 'is the charger plugged in'
NOT_THING = re.compile(r"\b(?:weather|time|date|day|news|temperature)\b")  # 'is the weather nice'

LOOK_SYSTEM = """You answer spoken questions about a tabletop seen by an overhead camera that looks straight down. You see the table surface, the objects on it and sometimes hands; nothing beyond the table's edge, no faces, no room. Image 1 is the whole table right now. Any later images are enlarged close-ups of objects the question names; each says when it was taken, and an older close-up shows how the object looked then, not necessarily now. You also get what an object tracker believes about known objects; it remembers hidden ones (an object INSIDE the box or UNDER the notebook can't be seen but is there).

Rules:
- Answer in one or two short spoken sentences: plain words, no lists, no coordinates, no markdown.
- Say only what you can see or what the tracker states. If you can't tell, say so and set confidence below 0.5.
- Never state or imply that medication was taken, swallowed, skipped or missed.
- Never describe anything beyond the table; the camera cannot see it.
- Image 1 has numbered yellow boxes (marks) around objects the tracker follows; the text lists them. If the answer is about one visible thing and a mark is on it, set mark to that number and point to null.
- If the thing has no mark, set mark to null and point to its centre in image 1 as {"x": fraction of the width, "y": fraction of the height}, each 0 to 1.
- If the answer is not about one visible thing, mark and point are both null.
- Never mention the marks, their numbers or the yellow boxes in the answer: the listener can't see them. Describe places on the table, not the image, the way a person would ("on the left, next to the cup").
- confidence: 0 to 1.
Reply with the JSON object only."""

LOOK_SCHEMA = {
    "type": "object", "additionalProperties": False, "required": ["answer", "confidence", "mark", "point"],
    "properties": {
        "answer": {"type": "string"}, "confidence": {"type": "number"},
        "mark": {"anyOf": [{"type": "null"}, {"type": "integer"}]},
        "point": {"anyOf": [{"type": "null"}, {
            "type": "object", "additionalProperties": False, "required": ["x", "y"],
            "properties": {"x": {"type": "number"}, "y": {"type": "number"}}}]}},
}

RECALL_SYSTEM = """You answer spoken questions about what was on a tabletop earlier. You get frames an overhead camera saved at the listed times (it looks straight down: table, objects, sometimes hands; nothing beyond the table, no faces), what an object tracker believed at each frame, and the tracker's events and activity notes for that time.

Rules:
- Answer in one or two short spoken sentences with times, e.g. "A red mug was on the left side from about 9:10 to 9:40." Use the frame times; say "about".
- Say only what the frames or notes show. If none of the frames show what was asked, say you didn't see it in the saved pictures. If you can't tell, set confidence below 0.5.
- Never state or imply that medication was taken, swallowed, skipped or missed.
- Never describe anything beyond the table.
- frames: the numbers of the frames your answer relies on.
- confidence: 0 to 1.
Reply with the JSON object only."""

RECALL_SCHEMA = {
    "type": "object", "additionalProperties": False, "required": ["answer", "confidence", "frames"],
    "properties": {"answer": {"type": "string"}, "confidence": {"type": "number"},
                   "frames": {"type": "array", "items": {"type": "integer"}}},
}


# ---------------------------------------------------------------- geometry

def point_to_px(point, orig_wh: tuple) -> Optional[tuple]:
    """{x, y} from the VLM -> (x, y) pixels of the ORIGINAL frame. Fractions 0-1 (what the prompt asks;
    slightly out of range is clamped); values up to 1000 are read as the 0-1000 convention some models
    fall back to; anything else is None."""
    try:
        x, y = float(point["x"]), float(point["y"])
    except (TypeError, ValueError, KeyError):
        return None
    if not (x == x and y == y):
        return None
    m = max(abs(x), abs(y))
    if m > 1000:
        return None
    scale = 1.0 if m <= 1.5 else 1000.0
    ow, oh = orig_wh
    return (min(max(x / scale, 0.0), 1.0) * ow, min(max(y / scale, 0.0), 1.0) * oh)


def px_to_cm(table, pts) -> Optional[np.ndarray]:
    """Frame px -> table cm with whatever the table offers: px_to_cm (core.table.Table, main.FlatTable),
    or the inverse of cm_to_px from its four corners (server.sim.SimTable)."""
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    if table is None:
        return None
    if hasattr(table, "px_to_cm"):
        return np.asarray(table.px_to_cm(pts), dtype=np.float64).reshape(-1, 2)
    if hasattr(table, "cm_to_px"):
        import cv2
        w, h = getattr(table, "W", 90.0), getattr(table, "H", 60.0)
        cm = np.array([[0, 0], [w, 0], [w, h], [0, h]], np.float32)
        px = np.asarray(table.cm_to_px(cm), np.float32).reshape(-1, 2)
        H = cv2.getPerspectiveTransform(px, cm)
        return cv2.perspectiveTransform(pts.reshape(-1, 1, 2), H).reshape(-1, 2)
    return None


def cm_box_to_px(table, box_cm) -> Optional[tuple]:
    """A table-cm box -> the (x1, y1, x2, y2) frame-px box around its projected corners."""
    if table is None or not hasattr(table, "cm_to_px"):
        return None
    x1, y1, x2, y2 = (float(v) for v in box_cm)
    try:
        px = np.asarray(table.cm_to_px([[x1, y1], [x2, y1], [x2, y2], [x1, y2]]), dtype=np.float64).reshape(-1, 2)
    except Exception:
        return None
    (a, b), (c, d) = px.min(axis=0), px.max(axis=0)
    return (float(a), float(b), float(c), float(d)) if c > a and d > b else None


def draw_marks(img: np.ndarray, boxes: list) -> np.ndarray:
    """A copy of img with each px box drawn in yellow and numbered from 1 (black-outlined white digits in
    a corner tag), sized for the frame so they survive the downscale to look_px."""
    import cv2
    out = img.copy()
    h, w = out.shape[:2]
    th = max(2, round(max(h, w) / 400))
    fs = max(0.6, max(h, w) / 1100)
    for i, (x1, y1, x2, y2) in enumerate(boxes, 1):
        p1 = (int(max(0, x1)), int(max(0, y1)))
        p2 = (int(min(w - 1, x2)), int(min(h - 1, y2)))
        cv2.rectangle(out, p1, p2, (0, 255, 255), th)
        label = str(i)
        (tw, tht), base = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fs, th)
        tx = min(max(p1[0], 0), w - tw - 6)
        ty = p1[1] - 4 if p1[1] - tht - 8 >= 0 else min(p1[1] + tht + 6, h - 4)
        cv2.rectangle(out, (tx, ty - tht - 4), (tx + tw + 6, ty + base), (0, 255, 255), -1)
        cv2.putText(out, label, (tx + 3, ty), cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 0, 0), th + 1, cv2.LINE_AA)
    return out


def _jpeg(img: np.ndarray, long_side: Optional[int] = None, upscale: bool = False) -> tuple[bytes, tuple]:
    """JPEG bytes and the (w, h) actually encoded. long_side shrinks larger images (and enlarges smaller
    ones with upscale=True: tiny crops hurt VLM accuracy)."""
    import cv2
    h, w = img.shape[:2]
    if long_side:
        s = long_side / max(h, w)
        if s < 1 or upscale:
            img = cv2.resize(img, (max(1, round(w * s)), max(1, round(h * s))),
                             interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ok:
        raise RuntimeError("jpeg encode failed")
    return buf.tobytes(), (img.shape[1], img.shape[0])


def tracked_marks(world, table) -> list[tuple[str, tuple]]:
    """(entity, frame px box) for every VISIBLE tracked entity with a box, in world order: the numbered
    candidates a VLM picks from. Empty without a calibrated table."""
    if table is None or getattr(table, "ok", True) is False:
        return []
    try:
        ents = world.state_json().get("entities", [])
    except Exception:
        return []
    out = []
    for d in ents:
        if d.get("status") != "VISIBLE":
            continue
        try:
            box = world.get(d["name"]).box_cm
        except Exception:
            box = None
        px = cm_box_to_px(table, box) if box is not None else None
        if px is not None:
            out.append((d["name"], px))
    return out


def spoken_names(world, cfg: dict) -> dict:
    """entity -> spoken name, for everything the world tracks (unnamed things: None)."""
    try:
        ents = [e["name"] for e in world.state_json().get("entities", [])]
    except Exception:
        ents = list((cfg.get("objects") or {}))
    try:
        labels = world.thing_labels() if hasattr(world, "thing_labels") else {}
    except Exception:
        labels = {}
    return {n: (labels.get(n) if n.startswith("thing:") else display_name(cfg, n)) for n in ents}


def _clock(wall: float) -> str:
    lt = time.localtime(wall)
    return f"{lt.tm_hour % 12 or 12}:{lt.tm_min:02d} {'AM' if lt.tm_hour < 12 else 'PM'}"


def _crop_store():
    try:
        from core.crops import active
        return active()
    except Exception:
        return None


def _taken(dt: float) -> str:
    """When a close-up was taken, relative to image 1 (dt = image 1's time minus the crop's, s)."""
    if abs(dt) <= 1.0:
        return "taken at the same time as image 1"
    if dt < 0:
        return "taken just after image 1"
    n = f"{dt / 60:.0f} minutes" if dt >= 90 else f"{dt:.0f} seconds"
    return f"taken {n} before image 1 (it may have changed since)"


def _with_note(text: str, note: str, where: bool = True) -> str:
    """The answer plus a short note, still at most two spoken sentences: a one-sentence answer keeps
    all of it; of two, a 'where' answer keeps its first (the place) and any other keeps both and
    drops the note (what the note says matters more than where it is)."""
    said = [x for x in re.split(r"(?<=[.!?])\s+", (text or "").strip()) if x]
    if len(said) < 2:
        return f"{' '.join(said)} {note}".strip()
    return f"{said[0]} {note}" if where else " ".join(said[:2])


@dataclass
class Mark:
    """A numbered box drawn on image 1: the tracked entity it stands for, as the world had it then."""
    name: str
    box_px: tuple
    box_cm: tuple
    pos_cm: Optional[tuple] = None
    seen: Optional[float] = None      # wall time the world last observed it


@dataclass
class Observation:
    """Everything one look() question is answered from, captured once while perception keeps running:
    the frame (with its idx and times), the world state read once for it, the marks drawn on that frame
    (mark n -> marks[n - 1].name) and the close-ups with their capture times (crop.t, Frame.t clock)."""
    img: np.ndarray
    idx: int
    t: float                          # Frame.t (time.monotonic())
    wall: float                       # Frame.wall
    state: dict
    names: dict
    marks: list
    crops: list


class _Frozen:
    """A world whose state_json is one captured reading (for compact_state)."""

    def __init__(self, state: dict):
        self.state = state

    def state_json(self) -> dict:
        return self.state


def _spoken(text: str) -> str:
    """Plain spoken text, at most two sentences, with the medication rule applied."""
    t = re.sub(r"[*_#`>\[\]]+", "", text or "").replace("\n", " ")
    t = re.sub(r"\s+", " ", t).strip()
    t = " ".join(re.split(r"(?<=[.!?])\s+", t)[:2]).strip()
    t, n = redact_meds(t)
    if not t and n:
        return PILLS_SAFE
    return t if not t or t.endswith((".", "!", "?")) else t + "."


_MARK_REFS = [re.compile(p, re.I) for p in (
    r"\s*\(\s*(?:mark(?:ed)?|#)\s*#?\d+\s*\)",                                    # "(mark 3)"
    r"\bmark\s*#?\d+\s+is\s+",                                                  # "Mark 2 is your mug"
    r",?\s*(?:(?:under|at|in|by|near|beside|with)\s+)?(?:the\s+)?mark(?:ed)?\s*(?:number\s*)?#?\d+\b",
)]


def _unmark(text: str) -> str:
    """Drop references to the numbered marks: the listener never sees image 1."""
    for rx in _MARK_REFS:
        text = rx.sub("", text)
    text = re.sub(r"\s+([,.!?])", r"\1", text)
    text = re.sub(r",\s*([,.!?])", r"\1", text).strip(" ,")
    return text[:1].upper() + text[1:]


def _conf(d: dict) -> float:
    try:
        c = float(d.get("confidence", 0.0))
    except (TypeError, ValueError):
        return 0.0
    return min(1.0, max(0.0, c)) if c == c else 0.0


_QUERY = [re.compile(p) for p in (
    r"\b(?:was|were|is|are) there (?:ever )?((?:an?|any|some|the|my) [a-z0-9 ]+?)(?= here\b| on\b| in\b| at\b"
    r"| this\b| earlier\b| today\b| yesterday\b| before\b| after\b| when\b| while\b|$)",
    r"\bwhen did (?:the |my |a |an )?([a-z0-9 ]+?) (?:show up|appear|arrive|get here|come|turn up|go|leave|disappear)",
    r"\b(?:did|have) (?:i|you|anyone) (?:see|seen|leave|left|put) ((?:an?|any|the|my) [a-z0-9 ]+?)"
    r"(?= here\b| on\b| in\b| at\b| this\b| earlier\b| today\b| yesterday\b| before\b|$)",
    r"\bwhere was (?:the |my |a |an )?([a-z0-9 ]+?)(?= this\b| earlier\b| today\b| yesterday\b| before\b| at\b|$)",
)]


def query_phrase(question: str) -> Optional[str]:
    """The thing a past-tense question asks about ('was there a red mug here this morning' -> 'a red
    mug'), or None ('what was on the table before I left' asks about everything)."""
    from voice.intents import normalize
    t = normalize(question)
    for rx in _QUERY:
        m = rx.search(t)
        if m:
            p = m.group(1).strip()
            return p if p not in ("anything", "something", "any", "some") else None
    return None


# ---------------------------------------------------------------- the answerer

class VisualQA:
    def __init__(self, cfg: dict, world, events, frames=None, table=None, provider=None,
                 archive: Optional[VisualArchive] = None, online: Optional[Callable[[], bool]] = None,
                 clock: Callable[[], float] = time.time, c: Optional[VisualConfig] = None):
        self.cfg, self.world, self.events, self.frames, self.table = cfg, world, events, frames, table
        self.c = c or VisualConfig.from_dict(cfg.get("visual_memory"))
        self.provider = provider or make_provider(NarrationConfig.from_dict({
            "provider": self.c.provider, "model": self.c.model, "base_url": self.c.base_url,
            "api_key_env": self.c.api_key_env, "reasoning_effort": self.c.reasoning_effort,
            "timeout_s": self.c.timeout_s, "max_tokens": self.c.max_tokens}))
        self.archive = archive
        self.online = online or (lambda: True)
        self.clock = clock
        self._calls: deque = deque()
        self.last: Optional[dict] = None
        self.grok_check = None          # core.grok_check.GrokCheck when on (main.build sets it)

    # -- the VLM call

    def _capped(self) -> bool:
        now = self.clock()
        while self._calls and self._calls[0] <= now - 3600:
            self._calls.popleft()
        return len(self._calls) >= self.c.max_per_hour

    def _vlm(self, system: str, parts: list, schema: dict) -> dict:
        self._calls.append(self.clock())
        reply = self.provider.narrate(system, parts, schema)
        d = _parse_json(reply.text)
        self.last = {"latency_ms": reply.latency_ms, "usage": reply.usage, "reply": d}
        return d

    # -- names

    def _names(self, state: Optional[dict] = None) -> dict:
        """entity -> spoken name, for everything the world tracks (unnamed things: None); from state
        (one captured state_json) when given."""
        try:
            ents = [e["name"] for e in (self.world.state_json() if state is None else state)["entities"]]
        except Exception:
            ents = list((self.cfg.get("objects") or {}))
        try:
            labels = self.world.thing_labels() if hasattr(self.world, "thing_labels") else {}
        except Exception:
            labels = {}
        return {n: (labels.get(n) if n.startswith("thing:") else display_name(self.cfg, n)) for n in ents}

    def _focus(self, intent: Optional[Intent], text: str, names: Optional[dict] = None) -> list[str]:
        """Entities the question names, for close-ups: the intent's target, then any spoken name or alias
        that appears in the question."""
        out = []
        if intent is not None:
            from voice.answers import _target
            try:
                t = _target(intent, self.world, self.cfg) if (intent.obj or intent.name) else None
            except Exception:
                t = None
            if t:
                out.append(t)
        low = f" {text.lower()} "
        for n, spoken in (self._names() if names is None else names).items():
            if spoken and re.search(rf"\b{re.escape(spoken.lower())}s?\b", low) and n not in out:
                out.append(n)
        return out[: self.c.max_crops]

    def _state_text(self, state: Optional[dict] = None) -> str:
        from voice.llm import compact_state
        try:
            names = self._names(state)
            rows = []
            for d in compact_state(self.world if state is None else _Frozen(state), self.cfg):
                spoken = names.get(d["name"]) or "unnamed object"
                s = f"{spoken}: {d['status'].lower()}"
                if d.get("parent"):
                    s += f" ({names.get(d['parent'], d['parent'])})"
                if d.get("area"):
                    s += f", {d['area']} of the table"
                rows.append(s)
            return "; ".join(rows) or "nothing tracked"
        except Exception:
            return "unavailable"

    # -- A: the table now

    def look(self, question: str, intent: Optional[Intent] = None) -> Answer:
        obs = self._observe(question, intent)
        if obs is None:
            return Answer(CANT_SEE)
        oh, ow = obs.img.shape[:2]
        names, marks = obs.names, obs.marks
        full, _ = _jpeg(draw_marks(obs.img, [mk.box_px for mk in marks]) if marks else obs.img, self.c.look_px)
        parts: list = [("text", "Image 1: the whole table now, from above."), ("image", full)]
        for i, (n, crop) in enumerate(obs.crops, 2):
            parts += [("text", f"Image {i}: close-up of the {names.get(n) or 'unnamed object'}, "
                               f"{_taken(obs.t - crop.t)}."),
                      ("image", _jpeg(crop.img, self.c.crop_px, upscale=True)[0])]
        listed = ", ".join(f"{i} = {names.get(mk.name) or 'unnamed object'}" for i, mk in enumerate(marks, 1))
        parts.append(("text", f"Marks: {listed or 'none'}.\nTracker: {self._state_text(obs.state)}.\n"
                              f"Question: {question}"))
        try:
            d = self._vlm(LOOK_SYSTEM, parts, LOOK_SCHEMA)
        except (ProviderError, NarrationError) as ex:
            log.warning("look failed: %s", ex)
            return Answer("Sorry, I couldn't look at the table just now.")
        text = _spoken(_unmark(str(d.get("answer") or "")))
        if not text or _conf(d) < self.c.abstain_below:
            return Answer(ABSTAIN)
        where = (intent is not None and intent.kind == "WHERE") or "where" in question.lower()
        m = d.get("mark")
        if isinstance(m, int) and not isinstance(m, bool) and 1 <= m <= len(marks):
            return self._verified(text, marks[m - 1], where)
        return self._pointed(text, d.get("point"), (ow, oh), marks, where)

    def _observe(self, question: str, intent: Optional[Intent]) -> Optional[Observation]:
        """One consistent observation for a question: the frame read once, the world state read once
        right after it, the marks fresh for that frame, the close-ups with their times."""
        f = self.frames.latest() if self.frames is not None else None
        if f is None or getattr(f, "img", None) is None:
            return None
        try:
            state = self.world.state_json()
        except Exception:
            state = {}
        marks = self._marks(state, f.wall)
        names = self._names(state)
        crops = self._crops(self._focus(intent, question, names))
        return Observation(f.img, getattr(f, "idx", 0), float(f.t), float(f.wall), state, names, marks, crops)

    def _marks(self, state: dict, wall: float) -> list[Mark]:
        """A Mark for every entity VISIBLE in state with a box the world saw within MARK_FRESH_S of the
        frame's wall time, in world order: the candidates the VLM picks from. Box, position and time
        are read together; an entity whose live status already differs is left out."""
        if self.table is None or getattr(self.table, "ok", True) is False:
            return []
        out = []
        for d in state.get("entities", []):
            if d.get("status") != "VISIBLE":
                continue
            try:
                ent = self.world.get(d["name"])
                status, box, pos, seen = str(ent.status), ent.box_cm, ent.pos_cm, ent.last_seen
            except Exception:
                continue
            if status != "VISIBLE" or box is None or seen is None or abs(wall - seen) > MARK_FRESH_S:
                continue
            px = cm_box_to_px(self.table, box)
            if px is not None:
                out.append(Mark(d["name"], px, tuple(box), tuple(pos) if pos is not None else None, seen))
        return out

    def _verified(self, text: str, mk: Mark, where: bool = True) -> Answer:
        """Point at mk's entity only if it still holds now that the answer is back: it still exists and
        wasn't merged, and if still visible it hasn't moved more than JUMP_CM (hidden: the laser follows
        it through world.resolve as usual). Otherwise the answer, a short note and no pointing."""
        try:
            ent = self.world.get(mk.name)
            status, pos, merged = str(ent.status), ent.pos_cm, getattr(ent, "merged_into", None)
        except Exception:
            ent = None
        if ent is None or merged:
            return Answer(_with_note(text, MOVED, where))
        if status == "VISIBLE":
            if mk.pos_cm is None or pos is None or np.hypot(pos[0] - mk.pos_cm[0], pos[1] - mk.pos_cm[1]) > JUMP_CM:
                return Answer(_with_note(text, MOVED, where))
        elif status not in HIDDEN:
            return Answer(_with_note(text, MOVED, where))
        return Answer(text, point_at=mk.name, action="point")

    def _crops(self, focus: list[str]) -> list[tuple[str, Crop]]:
        """(entity, crop) close-ups of focus entities: only crops the store knows were of that entity
        (core.crops ownership), each with its capture time."""
        store = _crop_store()
        out = []
        for n in focus:
            try:
                tr = store.for_entity(self.world.get(n)) if store is not None else None
            except Exception:
                tr = None
            crop = (tr.best or tr.recent) if tr is not None else None
            if crop is not None and crop.img is not None:
                out.append((n, crop))
        return out

    @staticmethod
    def _bind_crops(world) -> None:
        """Tell the crop store which view each thing was in the frame the world just processed."""
        store = _crop_store()
        if store is None or not hasattr(world, "thing_labels"):
            return
        try:
            for n in world.thing_labels():
                store.bind(world.get(n))
        except Exception:
            log.debug("binding crops to things failed", exc_info=True)

    def _pointed(self, text: str, point, orig_wh: tuple, marks: list[Mark], where: bool = True) -> Answer:
        """The answer, pointing where the VLM pointed only when that is a marked (tracked) entity; a
        point on nothing tracked, or on an ambiguous overlap, is spoken as unsure and not aimed at."""
        if point is None:
            return Answer(text)
        px = point_to_px(point, orig_wh) if isinstance(point, dict) else None
        c = px_to_cm(self.table, [px]) if px is not None and getattr(self.table, "ok", True) is not False else None
        hit = self._mark_at(float(c[0][0]), float(c[0][1]), marks) if c is not None else None
        if hit is None:
            return Answer(_with_note(text, UNSURE_WHERE, where))
        return self._verified(text, hit, where)

    @staticmethod
    def _mark_at(x: float, y: float, marks: list[Mark]) -> Optional[Mark]:
        """The mark whose box holds (x, y) cm (else within POINT_MARGIN_CM); of several, the smallest
        when it lies inside all the others (keys on the notebook), else None (ambiguous)."""
        from core import geom
        hits: list[Mark] = []
        for margin in (0.0, POINT_MARGIN_CM):
            hits = [mk for mk in marks if mk.box_cm[0] - margin <= x <= mk.box_cm[2] + margin
                    and mk.box_cm[1] - margin <= y <= mk.box_cm[3] + margin]
            if hits:
                break
        if not hits:
            return None
        small = min(hits, key=lambda mk: geom.area(mk.box_cm))
        if all(mk is small or geom.overlap_frac(mk.box_cm, small.box_cm) >= 0.9 for mk in hits):
            return small
        return None

    # -- B: the table before

    def _window(self, question: str, now: float) -> tuple[float, float, str]:
        from voice.answers import _away_since
        from voice.intents import normalize
        t = normalize(question)
        if re.search(r"\b(?:before|when) i left\b|\bwhile i was (?:gone|away|out)\b", t):
            since = _away_since(self.events, now)
            if since is not None:
                if "while i was" in t:
                    return since, now, " while you were away"
                return since - 900, since, " before you left"
        w = parse_window(question, now)
        if w is not None:
            return w.t0, w.t1, f" {w.label}"
        return now - self.c.keep_h * 3600, now, ""

    def recall(self, question: str) -> Answer:
        if self.archive is None:
            return Answer("I don't keep pictures of the table, so I can't tell.")
        now = self.clock()
        t0, t1, label = self._window(question, now)
        phrase = query_phrase(question)
        hits = self.archive.search(phrase, t0, t1, self.c.recall_frames) if phrase else []
        if phrase and hits and hits[0][1] < self.c.min_sim:
            return Answer(f"I don't remember seeing {phrase}{label}.")
        rows = [r for r, _ in hits] or (self.archive.sample(t0, t1, self.c.recall_frames) if "before you left"
                                        not in label else self.archive.store.window(t0, t1)[-self.c.recall_frames:])
        rows = sorted((r for r in rows if r.path), key=lambda r: r.t)
        if not rows:
            return Answer(f"I don't have any saved pictures of the table{label}.")
        parts: list = [("text", self._context(t0, t1))]
        import cv2
        n = 0
        for r in rows:
            img = cv2.imread(r.path)
            if img is None:
                continue
            n += 1
            parts += [("text", f"Frame {n} at {_clock(r.t)}{', a hand in view' if r.hands else ''}; tracker: "
                               f"{digest_text(r.digest)}."), ("image", _jpeg(img, self.c.recall_px)[0])]
        if n == 0:
            return Answer(f"I don't have any saved pictures of the table{label}.")
        parts.append(("text", f"Question: {question}"))
        try:
            d = self._vlm(RECALL_SYSTEM, parts, RECALL_SCHEMA)
        except (ProviderError, NarrationError) as ex:
            log.warning("recall failed: %s", ex)
            return Answer("Sorry, I couldn't go through the saved pictures just now.")
        text = _spoken(str(d.get("answer") or ""))
        if not text or _conf(d) < self.c.abstain_below:
            return Answer("I can't tell from the pictures I saved.")
        return Answer(text)

    def _context(self, t0: float, t1: float) -> str:
        """Narrations and world events in the window, as text for the recall prompt."""
        from core.narration_store import store_for
        lines = [f"Time window: {_clock(t0)} to {_clock(t1)}."]
        st = store_for(self.events, create=False)
        narr = [r for r in (st.between(t0, t1) if st else []) if r.summary][-5:]
        if narr:
            lines.append("Activity notes: " + " ".join(f"{_clock(r.t_start)}: {redact_meds(r.summary)[0]}"
                                                        for r in narr))
        try:
            evs = [e for e in self.events.since(t0) if e.wall <= t1][-15:]
        except Exception:
            evs = []
        if evs:
            names = self._names()
            lines.append("Tracker events: " + "; ".join(
                event_line({"wall": e.wall, "obj": e.obj, "type": str(e.type), "parent": e.parent,
                            "edge": e.edge}, names, self.cfg) for e in evs))
        return "\n".join(lines)

    # -- C: routing

    def route(self, intent: Intent, text: str, online: Optional[bool] = None) -> Optional[Answer]:
        """The answer when this question belongs to the visual layers, else None (the world model,
        narrations and templates answer as before)."""
        from voice.answers import _narrated_about, _target
        from voice.intents import normalize
        k = intent.kind
        t = normalize(text)
        past = bool(PAST.search(t))
        if k == "OTHER":
            if not self._about_table(intent, t):
                return None
            how = "recall" if past else "look"
        elif k in ("WHERE", "HISTORY", "HANDLED") and (intent.name or intent.obj):
            try:
                target = _target(intent, self.world, self.cfg)
                known = target is not None
            except Exception:
                target, known = None, True
            if known:
                return self._sighting(ent=target) if k == "WHERE" and target else None
            said = [w for w in (intent.name or intent.obj, query_phrase(text)) if w]
            if k == "WHERE":
                how = "look"
            elif self.archive is not None and _narrated_about(said, self.events, self.clock()) is None:
                how = "recall"
            else:
                return None
        else:
            return None
        online = self.online() if online is None else online
        if not online:
            if k == "OTHER":
                return Answer(OFFLINE)
            return self._sighting(said=said[0]) if k == "WHERE" else None
        if self._capped():
            log.info("visual questions: hourly cap reached; answering without the camera")
            return None
        return self.look(text, intent) if how == "look" else self.recall(text)

    def _sighting(self, ent: Optional[str] = None, said: Optional[str] = None) -> Optional[Answer]:
        """WHERE for something the world has never had a position for (a tracked object still UNKNOWN,
        or offline, a name it doesn't know), from the Grok settle check's newest sighting of it. Stored
        rows, so it works offline. The laser circles the spot, since the sighting may be minutes old.
        None: the templates answer as before."""
        if self.grok_check is None:
            return None
        if ent is not None:
            try:
                e = self.world.get(ent)
            except Exception:
                return None
            if e.status != "UNKNOWN" or e.pos_cm is not None or e.last_seen is not None:
                return None
            said = self._names().get(ent) or said
        if not said:
            return None
        try:
            hit = self.grok_check.lookup(said)
        except Exception:
            log.exception("grok check lookup failed")
            return None
        if hit is None:
            return None
        text = _spoken(f"I haven't tracked your {said}, but at {_clock(hit['wall'])} I saw what looked like "
                       f"your {said} about here.")
        return Answer(text, action="circle", target_cm=(hit["x_cm"], hit["y_cm"]))

    def _about_table(self, intent: Intent, t: str) -> bool:
        """Whether an OTHER question (t normalized) is about the table or what the camera sees: it says
        so, names a tracked object or thing label, or asks about a thing ('was there a red mug')."""
        from voice.intents import normalize
        if intent.obj or intent.name or SCENE.search(t) or query_phrase(t):
            return True
        if THING_Q.search(t) and not NOT_THING.search(t):
            return True
        low = f" {t} "
        return any(spoken and re.search(rf"\b{re.escape(normalize(spoken))}s?\b", low)
                   for spoken in self._names().values())

    # -- status

    def status(self) -> dict:
        arch = self.archive.store.stats() if self.archive is not None else None
        return {"enabled": True, "provider": self.provider.name, "model": self.provider.model,
                "archive": arch,
                "disclosure": (f"Visual questions are on: the current camera frame of the table and, for "
                               f"questions about earlier, up to {self.c.recall_frames} saved frames are sent to "
                               f"{self.provider.name} ({self.provider.model}). Saved frames stay on this "
                               f"device for {self.c.keep_h:g} h.")}

    def attach(self, world) -> "VisualQA":
        """Adds the visual status to world.state_json and, after every world.update, binds the crop
        store's views to the things the world saw in them (so close-ups follow identity)."""
        state = world.state_json
        update = getattr(world, "update", None)
        if callable(update):
            def update_and_bind(dets, frame):
                out = update(dets, frame)
                self._bind_crops(world)
                return out

            world.update = update_and_bind

        def state_json(*a, **kw):
            st = state(*a, **kw)
            try:
                st["visual_memory"] = self.status()
            except Exception:
                log.exception("visual status failed")
            return st

        world.state_json = state_json
        return self

    def stop(self) -> None:
        if self.archive is not None:
            self.archive.stop()


def from_config(cfg: dict, world, events, frames=None, table=None, online=None, start: bool = True
                ) -> Optional[VisualQA]:
    """The app's VisualQA (with its archive attached to the world), or None when cfg visual_memory.enabled
    is false (the default)."""
    c = VisualConfig.from_dict(cfg.get("visual_memory"))
    if not c.enabled:
        return None
    archive = VisualArchive(cfg, events, world, embedder=make_embedder(c), start=start, c=c).attach(world)
    qa = VisualQA(cfg, world, events, frames, table, archive=archive, online=online, c=c).attach(world)
    log.info("visual memory on: %s", qa.status()["disclosure"])
    return qa


# ---------------------------------------------------------------- selftest (real provider calls)

def _selftest(argv=None) -> int:
    """python -m voice.visual --selftest [--image photo.jpg] [--question '...'] [--recall a.jpg b.jpg]

    look(): the photo (default: the first stock desk photo) as the current frame (frame == table,
    90 x 60 cm), a question, the configured provider (default Grok; needs $XAI_API_KEY); prints the
    answer, latency, usage and where it would point. --recall: the given photos archived 30 minutes
    apart (this morning), embedded with the configured embedder, then a past-tense question."""
    import argparse
    import os
    import tempfile

    import cv2

    from core.config import ROOT, load_config
    from core.events import EventLog
    from core.fakeworld import FakeWorld
    from core.types import Frame
    from core.visual_memory import VisualArchive, _thumb
    ap = argparse.ArgumentParser(description=_selftest.__doc__)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--image", default=str(ROOT / "experiments/openset/frames/proxy_getty_a.jpg"))
    ap.add_argument("--question", default="Where are the keys?")
    ap.add_argument("--recall", nargs="*")
    ap.add_argument("--recall-question", default="When did the calculator show up?")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    cfg = load_config()
    c = VisualConfig.from_dict(dict(cfg.get("visual_memory") or {}, enabled=True))
    key_env = c.api_key_env or ("XAI_API_KEY" if c.provider == "grok" else "ANTHROPIC_API_KEY")
    if c.provider != "fake" and not os.environ.get(key_env):
        print(f"${key_env} is not set: the real {c.provider} call is pending a key.")
        return 2
    img = cv2.imread(a.image)
    if img is None:
        raise SystemExit(f"can't read {a.image}")
    h, w = img.shape[:2]

    class Frames:
        def latest(self):
            return Frame(t=0.0, wall=time.time(), img=img, idx=1)

    class Table:                                  # frame == table
        ok = True

        def px_to_cm(self, pts):
            return np.asarray(pts, float).reshape(-1, 2) * [90.0 / w, 60.0 / h]

    events = EventLog(":memory:", tempfile.mkdtemp(prefix="visual_selftest_"))
    world = FakeWorld([], events)
    qa = VisualQA(cfg, world, events, Frames(), Table(), c=c)
    t0 = time.perf_counter()
    ans = qa.look(a.question)
    print(f"\nlook: {a.question!r} -> {ans}\n  {(time.perf_counter() - t0) * 1000:.0f} ms total; "
          f"provider {qa.last and qa.last['latency_ms']} ms; usage {qa.last and qa.last['usage']}\n  "
          f"reply {qa.last and qa.last['reply']}")
    if a.recall:
        emb = make_embedder(c)
        arch = VisualArchive(cfg, events, world, embedder=emb, start=False, c=c)
        base = time.time() - 3 * 3600
        for i, p in enumerate(a.recall):
            im = cv2.imread(p)
            f = Frame(t=float(i), wall=base + i * 1800, img=im, idx=i)
            arch._save(f, _thumb(im), 5.0, False, "change")
        arch.drain()
        qa.archive = arch
        t0 = time.perf_counter()
        ans = qa.recall(a.recall_question)
        print(f"\nrecall: {a.recall_question!r} -> {ans}\n  {(time.perf_counter() - t0) * 1000:.0f} ms total; "
              f"provider {qa.last and qa.last['latency_ms']} ms; usage {qa.last and qa.last['usage']}\n  "
              f"reply {qa.last and qa.last['reply']}")
    events.close()
    return 0


if __name__ == "__main__":
    import sys
    raise SystemExit(_selftest(sys.argv[1:]))
