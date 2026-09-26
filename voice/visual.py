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
Offline, a question only the VLM could answer gets a spoken 'I need the connection' (the world model
still answers everything else). Over the hourly cap, route() returns None and the old path answers.

look(): set-of-marks grounding. Every VISIBLE tracked entity with a box (known objects and thing:N) is
drawn on the current frame (<= look_px) as a numbered yellow box; that marked frame, upscaled close-ups of
entities the question names (from core.crops.active(): only views the store knows were of that entity, each
labelled with when it was taken; attach() binds views to things after every world.update), the compact
world state and the question go to
the VLM, which replies with strict JSON {answer, confidence, mark: null | number, point: null | {x, y}
fractions of image 1}. A mark points at its entity (the laser then follows it through world.resolve); a
point (for something no mark covers) is mapped to table cm (frame px -> table homography), attached to a
tracked entity it lands on, else carried as the raw table position in target_cm. Low confidence abstains:
"I can't tell from here."

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
from typing import TYPE_CHECKING, Callable, Optional

import numpy as np

from core.config import display_name
from core.narration import NarrationConfig, NarrationError, ProviderError, _parse_json, event_line, make_provider
from core.narration_store import parse_window, redact_meds
from core.types import Answer, Intent
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
POINT_NEAR_CM = 6.0

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

    def _names(self) -> dict:
        """entity -> spoken name, for everything the world tracks (unnamed things: None)."""
        try:
            ents = [e["name"] for e in self.world.state_json().get("entities", [])]
        except Exception:
            ents = list((self.cfg.get("objects") or {}))
        try:
            labels = self.world.thing_labels() if hasattr(self.world, "thing_labels") else {}
        except Exception:
            labels = {}
        return {n: (labels.get(n) if n.startswith("thing:") else display_name(self.cfg, n)) for n in ents}

    def _focus(self, intent: Optional[Intent], text: str) -> list[str]:
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
        for n, spoken in self._names().items():
            if spoken and re.search(rf"\b{re.escape(spoken.lower())}s?\b", low) and n not in out:
                out.append(n)
        return out[: self.c.max_crops]

    def _state_text(self) -> str:
        from voice.llm import compact_state
        try:
            names = self._names()
            rows = []
            for d in compact_state(self.world, self.cfg):
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
        f = self.frames.latest() if self.frames is not None else None
        if f is None or getattr(f, "img", None) is None:
            return Answer(CANT_SEE)
        oh, ow = f.img.shape[:2]
        names = self._names()
        marks = self._marks()
        full, _ = _jpeg(draw_marks(f.img, [b for _, b in marks]) if marks else f.img, self.c.look_px)
        parts: list = [("text", "Image 1: the whole table now, from above."), ("image", full)]
        focus = self._focus(intent, question)
        crops = self._crops(focus)
        for i, (n, crop) in enumerate(crops, 2):
            parts += [("text", f"Image {i}: close-up of the {names.get(n) or 'unnamed object'}, "
                               f"{_taken(f.t - crop.t)}."),
                      ("image", _jpeg(crop.img, self.c.crop_px, upscale=True)[0])]
        listed = ", ".join(f"{i} = {names.get(n) or 'unnamed object'}" for i, (n, _) in enumerate(marks, 1))
        parts.append(("text", f"Marks: {listed or 'none'}.\nTracker: {self._state_text()}.\n"
                              f"Question: {question}"))
        try:
            d = self._vlm(LOOK_SYSTEM, parts, LOOK_SCHEMA)
        except (ProviderError, NarrationError) as ex:
            log.warning("look failed: %s", ex)
            return Answer("Sorry, I couldn't look at the table just now.")
        text = _spoken(_unmark(str(d.get("answer") or "")))
        if not text or _conf(d) < self.c.abstain_below:
            return Answer(ABSTAIN)
        m = d.get("mark")
        if isinstance(m, int) and not isinstance(m, bool) and 1 <= m <= len(marks):
            return Answer(text, point_at=marks[m - 1][0], action="point")
        return self._pointed(text, d.get("point"), (ow, oh))

    def _marks(self) -> list[tuple[str, tuple]]:
        """(entity, frame px box) for every VISIBLE tracked entity with a box, in world order: the
        candidates the VLM picks from."""
        if self.table is None or getattr(self.table, "ok", True) is False:
            return []
        try:
            ents = self.world.state_json().get("entities", [])
        except Exception:
            return []
        out = []
        for d in ents:
            if d.get("status") != "VISIBLE":
                continue
            try:
                box = self.world.get(d["name"]).box_cm
            except Exception:
                box = None
            px = cm_box_to_px(self.table, box) if box is not None else None
            if px is not None:
                out.append((d["name"], px))
        return out

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

    def _pointed(self, text: str, point, orig_wh: tuple) -> Answer:
        """The answer, pointing where the VLM pointed: a tracked entity it lands on, else the spot."""
        if not isinstance(point, dict) or getattr(self.table, "ok", True) is False:
            return Answer(text)
        px = point_to_px(point, orig_wh)
        if px is None:
            return Answer(text)
        c = px_to_cm(self.table, [px])
        if c is None:
            return Answer(text)
        x, y = float(c[0][0]), float(c[0][1])
        w, h = ((self.cfg.get("table") or {}).get("size_cm") or [90, 60])
        if not (-5 <= x <= w + 5 and -5 <= y <= h + 5):
            return Answer(text)
        ent = self._entity_at(x, y)
        if ent is not None:
            return Answer(text, point_at=ent, action="point")
        return Answer(text, action="point", target_cm=(round(x, 1), round(y, 1)))

    def _entity_at(self, x: float, y: float) -> Optional[str]:
        best, bd = None, POINT_NEAR_CM
        try:
            ents = self.world.state_json().get("entities", [])
        except Exception:
            return None
        for d in ents:
            if d.get("status") != "VISIBLE" or not d.get("pos_cm"):
                continue
            try:
                box = self.world.get(d["name"]).box_cm
            except Exception:
                box = None
            if box is not None and box[0] - 2 <= x <= box[2] + 2 and box[1] - 2 <= y <= box[3] + 2:
                dist = 0.0
            else:
                dist = float(np.hypot(d["pos_cm"][0] - x, d["pos_cm"][1] - y))
            if dist <= bd:
                best, bd = d["name"], dist
        return best

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
                known = _target(intent, self.world, self.cfg) is not None
            except Exception:
                known = True
            if known:
                return None
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
            return Answer(OFFLINE) if k == "OTHER" else None
        if self._capped():
            log.info("visual questions: hourly cap reached; answering without the camera")
            return None
        return self.look(text, intent) if how == "look" else self.recall(text)

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
