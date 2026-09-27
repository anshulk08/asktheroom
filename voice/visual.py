"""Visual questions: what the camera sees now (look) and what it saw before (recall), and the routing
that decides which layer answers a question.

Routing (VisualQA.route, called by voice/pipeline.py before the offline templates):
  - object location / history questions about known things stay with the world model (fast, offline);
  - activity questions (WHAT_DOING, 'while I was away') stay with narrations + events (voice/answers.py);
  - OTHER questions about the scene now go to look(); in the past tense, to recall(). An OTHER question
    that isn't about the table or what the camera sees ('what time is it', 'tell me a joke') returns None
    so the pipeline's `other` answerer takes it, online or offline;
  - WHERE for a name the world model doesn't know ('where is my red mug?') goes to pick(): Grok only says
    which mark shows it and what it is, the world model says where (no marks to pick from: look()).
    HISTORY / HANDLED for such a name goes to recall() when no narration mentions it.
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

pick(): the narrow form of look() for 'where is my X' when X is a name the world doesn't know. Grok gets
only the marked frame and replies {mark, label, confidence}: which box shows X and what the object is
(1-4 words). Where it is comes from the world model (the WHERE template), so Grok writes no sentence. An
unnamed thing picked at >= BIND_CONF takes the asked-for name (world.bind_alias; a taught name is never
replaced), so the next question is answered offline with no Grok call. No marks: falls back to look().

recall(): archived frames (core/visual_memory.py) picked by text similarity and/or the question's time
window, each with its time and the world model's digest, plus the narrations and world events of that
window, go to the VLM; the answer cites times ('a red mug was on the left from about 9:10 to 9:40').
When nothing in the archive matches the words well enough, it abstains without a VLM call.

look_room(): with room memory on (spec 0009/0010), the camera in a room corner and the table view cut from
it, a question about the room (and, by default, any camera question that doesn't say "table") gets the
whole camera view, plus native-resolution close-ups of drawn zones (room_zones.json): the zones the question
names, else the far (small) ones, at most room_crops. A cup on the far counter is ~40 px of the 2560 px
frame and ~20 px of the 1280 px view Grok gets; its close-up shows it at ~15x that area. Spoken only (no
laser off the table). A thing Grok looks for and doesn't see is an honest "I don't see it" (confidence is
in the answer, so an absence can be sure); too far or dark to tell abstains. Recall of a room question
uses the archive's whole-room frames (view 'room') when it has any in the window.

Every VLM answer passes the medication rule (core/narration_store.redact_meds) and is cut to two spoken
sentences. The table prompts keep answers to the table; people are only ever "someone".
"""
from __future__ import annotations

import logging
import os
import re
import time
from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Optional

import numpy as np

from core.config import display_name
from core.narration import NarrationConfig, NarrationError, ProviderError, _parse_json, event_line, make_provider
from core.narration_store import med_claim, parse_window, redact_meds
from core.types import Answer, Intent, Status
from core.visual_memory import VisualArchive, VisualConfig, digest_text, make_embedder
from net import call_with_deadline

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
POINT_NEAR_CM = 6.0
NEW_THING = "something new"      # a thing with no name and no guess, in VLM prompts (core/labels.py)
BIND_CONF = 0.7                  # a picked unnamed thing takes the asked-for name at or above this confidence
VLM_MARGIN_S = 2.0               # wall-clock deadline per Grok call: visual_memory.timeout_s plus this

PAST = re.compile(r"\b(?:was|were|did|had|earlier|before|ago|yesterday|used to|show(?:ed|n)? up|appeared"
                  r"|last time|when i left|before i left|while i was)\b")
# An OTHER question is about the table (so the camera can answer it) when it says so; 'what time is it'
# or 'tell me a joke' is not, and goes to the pipeline's `other` answerer instead.
SCENE = re.compile(r"\b(?:table|desk|here|this|that|these|those|see|seen|saw|look|looks|looking|colou?rs?"
                   r"|whats on|what is on|how many|read|reads|note|notes|written|writing|shape|size)\b")
THING_Q = re.compile(r"^(?:is|are) (?:the|my|your|our|a|an|there)\b")     # 'is the charger plugged in'
NOT_THING = re.compile(r"\b(?:weather|time|date|day|news|temperature)\b")  # 'is the weather nice'

LOOK_SYSTEM = """You answer spoken questions about a tabletop. Image 1 is the camera's view of the whole table right now; the camera is above the table, looking straight down or, mounted high in a corner of the room, down at an angle, so the edges of the image may show the floor, furniture or people around the table. Any later images are enlarged close-ups of objects the question names; each says when it was taken, and an older close-up shows how the object looked then, not necessarily now. You also get what an object tracker believes about known objects; it remembers hidden ones (an object INSIDE the box or UNDER the notebook can't be seen but is there). A name ending in "?" is a guess ("mug?": say "what looks like a mug"); "something new" is a thing nobody has named (describe it by what you see). Never say "unnamed" or a number in brackets.

Rules:
- Answer in one or two short spoken sentences: plain words, no lists, no coordinates, no markdown.
- Say only what you can see or what the tracker states. If you can't tell, say so and set confidence below 0.5.
- Never state or imply that medication was taken, swallowed, skipped or missed.
- Answer about the table and what is on it. Don't describe people beyond "someone"; never guess who they are.
- Image 1 has numbered yellow boxes (marks) around objects the tracker follows; the text lists them. If the answer is about one visible thing and a mark is on it, set mark to that number and point to null.
- If the thing has no mark, set mark to null and point to its centre in image 1 as {"x": fraction of the width, "y": fraction of the height}, each 0 to 1.
- If the answer is not about one visible thing, mark and point are both null.
- Never mention the marks, their numbers or the yellow boxes in the answer: the listener can't see them. Describe places on the table, not the image, the way a person would ("on the left, next to the cup").
- confidence: 0 to 1.
Reply with the JSON object only."""

# A question about the room around the table (with room memory on, spec 0009): its words, or a drawn zone's.
ROOM_Q = re.compile(r"\b(?:room|couch|sofa|shelf|bookshelf|bed|floor|chair|dresser|counter|cabinet|windowsill"
                    r"|around|anywhere|elsewhere|in here)\b")
# With room memory on the whole room is the default view; a question that says "table" keeps the table look.
TABLE_Q = re.compile(r"\b(?:table|desk)\b")

ROOM_SYSTEM = """You answer spoken questions about a room seen by one camera mounted high in a corner, looking down across the room at an angle. Near the camera things look big; across the room they look small. Image 1 is the camera's whole view right now. The text names the areas of the room the rig knows (for example "the couch") and says where each one is in image 1. Any later images are sharp close-ups of some of those areas, cut from the same moment at full resolution: small things far from the camera are clearest there. You also get what an object tracker believes about known objects, including ones it saw moved off the table into those areas.

Rules:
- Answer in one or two short spoken sentences: plain words, no lists, no coordinates, no markdown.
- Say where things are using the named areas or plain room words ("on the couch", "on the floor by the doorway"), never positions in the image or image numbers.
- seen: first, briefly (at most about 50 words) list what you see in image 1 and in each close-up: the objects and where they are. Then answer from that list. The listener never hears seen.
- Say only what you can see or what the tracker states.
- If you are asked where something is or whether it is there: look for it in image 1 and every close-up. If it isn't in any of them, say plainly that you don't see it (for example, that you don't see a mug) with a confidence for how sure you are that it isn't in view; it may be hidden or out of view, so don't say it isn't in the room. If you see something close to what was asked (a cup for a mug), say what you see and where.
- If what was asked about is too small, dark or blocked to make out, say you can't tell and set confidence below 0.5. Don't guess.
- Don't describe people beyond "someone"; never guess who they are.
- Never state or imply that medication was taken, swallowed, skipped or missed.
- confidence: 0 to 1, how sure you are that your answer is right.
Reply with the JSON object only."""

# seen comes first: listing what is in view before answering stopped "I don't see X" for things in plain
# view (eval/room_look.py: 4 of 4 visible things found, against 0 of 4 answering directly).
ROOM_SCHEMA = {
    "type": "object", "additionalProperties": False, "required": ["seen", "answer", "confidence"],
    "properties": {"seen": {"type": "string"}, "answer": {"type": "string"}, "confidence": {"type": "number"}},
}

LOOK_SCHEMA = {
    "type": "object", "additionalProperties": False, "required": ["answer", "confidence", "mark", "point"],
    "properties": {
        "answer": {"type": "string"}, "confidence": {"type": "number"},
        "mark": {"anyOf": [{"type": "null"}, {"type": "integer"}]},
        "point": {"anyOf": [{"type": "null"}, {
            "type": "object", "additionalProperties": False, "required": ["x", "y"],
            "properties": {"x": {"type": "number"}, "y": {"type": "number"}}}]}},
}

PICK_SYSTEM = """You find one object on a tabletop seen by a camera above it (looking straight down, or down at an angle from high in a corner of the room). The image has numbered yellow boxes (marks) around the objects a tracker follows. Your only job is to say which mark shows the object the person asks about and what that object is. A name ending in "?" is a guess ("mug?": say "what looks like a mug"); "something new" is a thing nobody has named (describe it by what you see). Never say "unnamed" or a number in brackets.

Rules:
- mark: the number of the box around the object asked about; null if no box shows it.
- label: what the object in that box is, the way a person would say it: 1 to 4 plain words, colour first if it helps ("red mug", "phone charger"). Null if mark is null. For any medicine container say only "pill bottle".
- confidence: 0 to 1; below 0.5 if you aren't sure.
Reply with the JSON object only."""

# A room answer that says it didn't see the thing (normalize drops apostrophes: "isn't" -> "isnt").
NEGATION = re.compile(r"\b(?:not|no|nothing|none|isnt|arent|wasnt|werent|dont|doesnt|didnt|cant|cannot|couldnt"
                      r"|never|unable|without|nowhere)\b")
WHERE_Q = re.compile(r"\b(?:where|is there|are there|do you see|can you see|have you seen)\b")


def normalize_text(text: str) -> str:
    from voice.intents import normalize
    return normalize(text)


# The second look when the marked frame gave no pick: every marked thing as its own numbered close-up. On
# the rig (27 Sep 02:56) "where are the batteries?" got no mark among ~14 on the whole table view, then a
# confident "I don't see any batteries." from the room look; 40 s later the same pick found them.
PICK_SHEET_SYSTEM = """You find one object among the things on a tabletop. The image is a grid of numbered close-ups, one per object a tracker follows, cut from a camera above the table (it may look down at an angle, so a close-up can also show a bit of what is around the object). Your only job is to say which numbered close-up shows the object the person asks about and what that object is.

Rules:
- mark: the number of the close-up whose main object is the one asked about; null if none shows it.
- label: what the object in that close-up is, the way a person would say it: 1 to 4 plain words, colour first if it helps ("red mug", "phone charger"). Null if mark is null. For any medicine container say only "pill bottle".
- confidence: 0 to 1; below 0.5 if you aren't sure.
Reply with the JSON object only."""
SHEET_TILE = 192                 # px per close-up in the pick sheet
SHEET_MAX = 20                   # close-ups per sheet (5 x 4)

PICK_SCHEMA = {
    "type": "object", "additionalProperties": False, "required": ["mark", "label", "confidence"],
    "properties": {"mark": {"anyOf": [{"type": "null"}, {"type": "integer"}]},
                   "label": {"anyOf": [{"type": "null"}, {"type": "string"}]},
                   "confidence": {"type": "number"}},
}

RECALL_SYSTEM = """You answer spoken questions about what was on a tabletop earlier. You get frames a camera above the table saved at the listed times (the table, objects and sometimes hands; the camera may look at an angle, so the edges can show the floor or furniture around the table), what an object tracker believed at each frame, and the tracker's events and activity notes for that time. A name ending in "?" is a guess ("mug?": say "what looks like a mug"); "something new" is a thing nobody has named (describe it by what you see). Never say "unnamed" or a number in brackets.

Rules:
- seen: first, briefly (a few words per frame) list for each frame its time and what it shows that bears on the question. Then answer from that list. The listener never hears seen.
- Answer in one or two short spoken sentences with times, e.g. "A red mug was on the left side from about 9:10 to 9:40." Use the frame times; say "about".
- Say only what the frames or notes show. If none of the frames show what was asked, say you didn't see it in the saved pictures. If you can't tell, set confidence below 0.5.
- Never state or imply that medication was taken, swallowed, skipped or missed.
- Answer about the table and what is on it. Don't describe people beyond "someone"; never guess who they are.
- confidence: 0 to 1.
- pictures: the numbers of the frames your answer rests on (at most 3; empty if none shows it).
Reply with the JSON object only."""

ROOM_RECALL_SYSTEM = """You answer spoken questions about what the room looked like earlier. You get pictures of the whole room that one camera, mounted high in a corner and looking down across the room at an angle, saved at the listed times (near the camera things look big; across the room, small), what an object tracker believed at each picture, and the tracker's events and activity notes for that time. A name ending in "?" is a guess ("mug?": say "what looks like a mug"); "something new" is a thing nobody has named (describe it by what you see). Never say "unnamed" or a number in brackets. The text names the areas of the room the rig knows and where each one is in the pictures.

Rules:
- seen: first, briefly (a few words per picture) list for each picture its time and what it shows that bears on the question: people, objects and where they are. Then answer from that list. The listener never hears seen.
- Answer in one or two short spoken sentences with times, e.g. "There was a laptop on the couch at about 6:20 and again at about 11." Use the picture times; say "about".
- Say where things were using the named areas or plain room words ("on the couch", "on the kitchen counter"), never positions in the image or picture numbers.
- Say only what the pictures or notes show. If none of the pictures show what was asked, say you didn't see it in the saved pictures. If you can't tell, set confidence below 0.5.
- Don't describe people beyond "someone"; never guess who they are.
- Never state or imply that medication was taken, swallowed, skipped or missed.
- confidence: 0 to 1.
- pictures: the numbers of the pictures your answer rests on (at most 3; empty if none shows it).
Reply with the JSON object only."""

DESCRIBE_SYSTEM = """You say where one object is, for someone in the room looking for it. The pictures are from a camera mounted high in a corner of the room, looking down at an angle; a numbered yellow box is drawn around the object. Image 1 is the camera's view, image 2 a close-up around the box.

Rules:
- where: one short phrase (at most 10 words) placing the object by the things right next to it or the part of the furniture it is on, e.g. "next to the laptop charger", "by the armrest nearest the TV", "on the stack of papers".
- Never use left, right, in front of or behind: the listener is not where the camera is.
- Don't name the object itself, and don't mention the box, the image or the camera.
- Never state or imply that medication was taken, swallowed, skipped or missed.
- confidence: 0 to 1; below 0.5 if you can't tell.
Reply with the JSON object only."""

DESCRIBE_SCHEMA = {
    "type": "object", "additionalProperties": False, "required": ["where", "confidence"],
    "properties": {"where": {"type": "string"}, "confidence": {"type": "number"}},
}
DESCRIBE_DEADLINE_S = 1.5        # a WHERE answer waits this long for Grok's description, then goes without
SIDE_WORDS = re.compile(r"\b(?:left|behind|in front of|to the right|on the right|right of|right side|right-hand)\b")

RECALL_SCHEMA = {                  # seen first, as ROOM_SCHEMA; pictures: which ones the answer rests on (evidence)
    "type": "object", "additionalProperties": False, "required": ["seen", "answer", "confidence", "pictures"],
    "properties": {"seen": {"type": "string"}, "answer": {"type": "string"}, "confidence": {"type": "number"},
                   "pictures": {"type": "array", "items": {"type": "integer"}}},
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


def zone_box(poly, frame_wh: tuple, min_px: int = 256) -> Optional[tuple]:
    """The frame px box a zone's close-up is cut from: the polygon's box grown by 15% on each side, upwards
    by its height (at least 0.4 of its width), since a zone is drawn on a surface and what stands on it rises
    above it in the image, and downwards by 40% of its height (things at a surface's front edge), then
    widened to at least min_px, inside the frame. None if empty."""
    if len(poly) < 3:
        return None
    xs, ys = [p[0] for p in poly], [p[1] for p in poly]
    x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
    w, h = x2 - x1, y2 - y1
    x1, x2 = x1 - 0.15 * w, x2 + 0.15 * w
    y1, y2 = y1 - max(h, 0.4 * w) - 0.15 * h, y2 + 0.4 * h
    fw, fh = frame_wh
    for lo, hi, n in ((0, 2, fw), (1, 3, fh)):
        b = [x1, y1, x2, y2]
        if b[hi] - b[lo] < min_px:
            c = (b[lo] + b[hi]) / 2
            b[lo], b[hi] = c - min_px / 2, c + min_px / 2
        x1, y1, x2, y2 = b
    x1, y1, x2, y2 = max(0, round(x1)), max(0, round(y1)), min(fw, round(x2)), min(fh, round(y2))
    return (x1, y1, x2, y2) if x2 - x1 > 1 and y2 - y1 > 1 else None


def image_place(poly, frame_wh: tuple) -> str:
    """Where a zone is in the image, in words for the prompt ('right side, middle height')."""
    fw, fh = frame_wh
    cx = sum(p[0] for p in poly) / len(poly) / fw
    cy = sum(p[1] for p in poly) / len(poly) / fh
    x = "left side" if cx < 1 / 3 else "right side" if cx > 2 / 3 else "middle"
    y = "top" if cy < 1 / 3 else "bottom" if cy > 2 / 3 else "middle height"
    return f"{x}, {y}"


def shown_names(names: dict, state: Optional[dict]) -> dict:
    """entity -> what the VLM is told it is: its spoken name (names), else for a thing core.labels' name
    from state ('mug?' for a guess, else 'something new'). Never None."""
    try:
        from core.labels import thing_labels
        things = thing_labels(state)
    except Exception:
        things = {}
    return {n: v or things.get(n) or NEW_THING for n, v in names.items()}


def pick_sheet(img: np.ndarray, boxes: list, tile: int = SHEET_TILE, cols: int = 5) -> np.ndarray:
    """A grid of close-ups, one per px box (grown by 40%, at least 48 px), each fitted into a tile x tile
    cell and numbered from 1 in its corner, like draw_marks' tags."""
    import cv2
    h, w = img.shape[:2]
    rows = max(1, -(-len(boxes) // cols))
    sheet = np.full((rows * tile, min(cols, max(1, len(boxes))) * tile, 3), 40, np.uint8)
    for i, (x1, y1, x2, y2) in enumerate(boxes):
        bw, bh = max(48.0, (x2 - x1) * 1.4), max(48.0, (y2 - y1) * 1.4)
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        a, b = int(max(0, cx - bw / 2)), int(max(0, cy - bh / 2))
        c, d = int(min(w, cx + bw / 2)), int(min(h, cy + bh / 2))
        crop = img[b:d, a:c] if d > b and c > a else np.zeros((8, 8, 3), np.uint8)
        s = (tile - 8) / max(crop.shape[:2])
        crop = cv2.resize(crop, (max(1, round(crop.shape[1] * s)), max(1, round(crop.shape[0] * s))),
                          interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)
        r, k = divmod(i, cols)
        oy, ox = r * tile + (tile - crop.shape[0]) // 2, k * tile + (tile - crop.shape[1]) // 2
        sheet[oy:oy + crop.shape[0], ox:ox + crop.shape[1]] = crop
        label = str(i + 1)
        (tw, th), base = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        cv2.rectangle(sheet, (k * tile, r * tile), (k * tile + tw + 8, r * tile + th + base + 6), (0, 255, 255), -1)
        cv2.putText(sheet, label, (k * tile + 4, r * tile + th + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2,
                    cv2.LINE_AA)
    return sheet


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


def _label(v) -> Optional[str]:
    """Grok's name for what it picked, as something safe to say: 1-4 plain lower-case words, never a
    medication claim, or None."""
    t = re.sub(r"[^a-z' -]", "", str(v or "").lower()).strip()
    t = re.sub(r"^(?:a|an|the|my|your)\s+", "", t)
    if not t or len(t.split()) > 4 or len(t) > 30 or med_claim(t) or t in ("object", "thing", "item", "unknown"):
        return None
    return t


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
        self.room_zones: Optional[list] = None   # [(name, spoken)] when room memory is on (main.build): room look
        self._polys: Optional[dict] = None       # the zones' polygons (room_memory.zones_path), read on first use

    # -- the VLM call

    def _capped(self) -> bool:
        now = self.clock()
        while self._calls and self._calls[0] <= now - 3600:
            self._calls.popleft()
        return len(self._calls) >= self.c.max_per_hour

    def _vlm(self, system: str, parts: list, schema: dict, limit: Optional[float] = None) -> dict:
        """One Grok call, bounded in wall time: requests' timeout is per phase (DNS, connect, each read),
        so a stalling connection could hold a spoken answer far past visual_memory.timeout_s. Past
        timeout_s + VLM_MARGIN_S (or `limit`) it raises ProviderError, and the callers' usual failure
        answer is said."""
        self._calls.append(self.clock())
        limit = float(self.c.timeout_s) + VLM_MARGIN_S if limit is None else float(limit)
        try:
            reply = call_with_deadline(self.provider.narrate, limit, system, parts, schema, name="visual-grok")
        except TimeoutError:
            raise ProviderError(f"no reply after {limit:.1f} s") from None
        d = _parse_json(reply.text)
        self.last = {"latency_ms": reply.latency_ms, "usage": reply.usage, "reply": d}
        return d

    # -- where exactly (voice/answers.py WHERE, when no landmark says it)

    def describe_where(self, ent: str, place=None) -> Optional[str]:
        """A short phrase for where `ent` is ('next to the laptop charger', 'by the armrest nearest the TV'),
        from one quick Grok look at the live frame with it boxed: the full view for a room place (its
        box_px), else the table view (its box_cm). None when offline, over the hourly cap, with nothing to
        box, or after DESCRIBE_DEADLINE_S: the template answer goes out without it. Never left/right: Grok
        sees from the camera, not the user's seat."""
        if not self.online() or self._capped():
            return None
        try:
            if place is not None and getattr(place, "kind", None) == "room":
                f, box, on = self._room_frame(), getattr(place, "box_px", None), place.say
            else:
                f = self.frames.latest() if self.frames is not None else None
                box_cm = self.world.get(ent).box_cm
                box, on = (cm_box_to_px(self.table, box_cm) if box_cm is not None else None), "the table"
            if f is None or getattr(f, "img", None) is None or box is None:
                return None
            img = f.img
            h, w = img.shape[:2]
            x1, y1, x2, y2 = (float(v) for v in box)
            side = max(x2 - x1, y2 - y1) * 3 + 160                      # the box and what surrounds it
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            cx1, cy1 = int(max(0, cx - side / 2)), int(max(0, cy - side / 2))
            cx2, cy2 = int(min(w, cx + side / 2)), int(min(h, cy + side / 2))
            marked = draw_marks(img, [(x1, y1, x2, y2)])
            parts = [("text", f"Image 1: the camera's view; the boxed object is on {on}."),
                     ("image", _jpeg(marked, self.c.look_px)[0]),
                     ("text", "Image 2: close-up around it."),
                     ("image", _jpeg(marked[cy1:cy2, cx1:cx2], self.c.crop_px, upscale=True)[0]),
                     ("text", "Where exactly is it?")]
        except Exception:
            log.debug("describe_where: nothing to look at", exc_info=True)
            return None
        try:
            d = self._vlm(DESCRIBE_SYSTEM, parts, DESCRIBE_SCHEMA, limit=DESCRIBE_DEADLINE_S)
        except (ProviderError, NarrationError) as ex:
            log.info("describe_where: none in time (%s)", ex)
            return None
        where = _spoken(str(d.get("where") or "")).strip().rstrip(".")
        where = re.sub(r"^(?:it(?:'s| is)|they(?:'re| are))\s+", "", where, flags=re.I)
        where = re.sub(rf"^on {re.escape(on)},?\s*", "", where, flags=re.I)
        if (not where or _conf(d) < self.c.abstain_below or SIDE_WORDS.search(where.lower())
                or len(where.split()) > 12 or where == PILLS_SAFE.rstrip(".")):
            return None
        return where[:1].lower() + where[1:]

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

    def _state_text(self, state: Optional[dict] = None, placed_only: bool = False) -> str:
        """The tracker's beliefs as text. placed_only (the room look): only named objects it has a place
        for; a list of never-seen names or unnamed blobs made Grok 'see' them in the room."""
        from voice.llm import compact_state
        try:
            names = self._names(state)
            taught = {v for v in names.values() if v}
            rows = []
            for d in compact_state(self.world if state is None else _Frozen(state), self.cfg):
                # compact_state names a thing by its core.labels name (its taught name, guess or 'something new')
                spoken = names.get(d["name"]) or d["name"] or NEW_THING
                if placed_only and (spoken not in taught or (d["status"] == Status.UNKNOWN.value
                                                             and not d.get("room_zone"))):
                    continue
                s = f"{spoken}: {d['status'].lower()}"
                if d.get("parent"):
                    s += f" ({names.get(d['parent'], d['parent'])})"
                if d.get("room_zone"):
                    s += f", in {d['room_zone']}" + (" (not seen there now)" if d.get("not_seen_there_now") else "")
                elif d.get("area"):
                    s += f", {d['area']} of the table"
                rows.append(s)
            return "; ".join(rows) or "nothing tracked"
        except Exception:
            return "unavailable"

    # -- A: the table now

    def _room_frame(self):
        """The camera's whole view when room memory is on (TableView.latest_full), else None."""
        full = getattr(self.frames, "latest_full", None)
        if full is None or not getattr(self, "room_zones", None):
            return None
        try:
            return full()
        except Exception:
            log.debug("full frame unavailable", exc_info=True)
            return None

    def _room_on(self) -> bool:
        """Room memory is on: a full frame to look at and the drawn zones."""
        return getattr(self.frames, "latest_full", None) is not None and bool(getattr(self, "room_zones", None))

    def _about_room(self, t: str) -> bool:
        """Whether a question (t normalized) is about the room around the table: room words, or a drawn
        zone's name or spoken name. Only with room memory on (a full frame to look at). The wake word
        ("room, what's on the table?") is not a room word."""
        if not self._room_on():
            return False
        from voice.intents import _without_wake_word
        t = _without_wake_word(t, self.cfg)
        return bool(ROOM_Q.search(t) or self._named_zones(t))

    def _room_default(self, t: str) -> bool:
        """With room memory on, a question the camera answers looks at the whole room unless it says
        "table" (and names no zone: "the side table" is the room). t normalized."""
        return self._room_on() and (self._about_room(t) or not TABLE_Q.search(t))

    def _zone_polys(self, frame_wh: tuple) -> dict:
        """zone name -> polygon in this frame's px, from room_memory.zones_path (read once; scaled when the
        frame's size differs from the one the zones were drawn at). {} if the file can't be read."""
        if self._polys is None:
            self._polys = {}
            try:
                from core.room_zones import Zones
                zs = Zones.load((self.cfg.get("room_memory") or {}).get("zones_path", "room_zones.json"))
                self._polys = {"size": zs.size_px, "zones": {z.name: z.poly for z in zs.zones.values()}}
            except Exception:
                log.warning("room look: zone polygons unavailable; no close-ups", exc_info=True)
        if not self._polys:
            return {}
        (dw, dh), (fw, fh) = self._polys["size"], frame_wh
        sx, sy = (fw / dw if dw else 1.0), (fh / dh if dh else 1.0)
        return {n: [(x * sx, y * sy) for x, y in poly] for n, poly in self._polys["zones"].items()}

    def _named_zones(self, t: str) -> list[str]:
        """Zone names the question (t normalized, wake word removed) names, in zone order."""
        from voice.intents import normalize
        low, out = f" {t} ", []
        for name, say in self.room_zones:
            for w in (name.replace("_", " "), say):
                w = re.sub(r"^(?:the|a|an) ", "", normalize(w))
                if w and re.search(rf"\b{re.escape(w)}\b", low) and name not in out:
                    out.append(name)
        return out

    def _zone_crops(self, img: np.ndarray, question: str, every: bool = False) -> list[tuple[str, np.ndarray]]:
        """(spoken zone name, native-px close-up) for the zones the question names, else for the far
        zones (each under room_crop_max_frac of the frame), at most room_crops. Not the table: a close-up
        of its clutter made Grok 'see' glasses and a phone there (eval/room_look.py), and the tracker
        answers for what is on the table."""
        from voice.intents import _without_wake_word, normalize
        if self.c.room_crops <= 0:
            return []
        h, w = img.shape[:2]
        polys = self._zone_polys((w, h))
        if not polys:
            return []
        say = dict(self.room_zones)
        named = self._named_zones(_without_wake_word(normalize(question), self.cfg))
        out = []
        for name in named or [n for n, _ in self.room_zones]:
            box = zone_box(polys[name], (w, h)) if name in polys else None
            if box is None:
                continue
            x1, y1, x2, y2 = box
            if not named and not every and (x2 - x1) * (y2 - y1) > self.c.room_crop_max_frac * w * h:
                continue                     # near and big: image 1 shows it well enough
            out.append((say.get(name, name), img[y1:y2, x1:x2]))
        return out if every else out[: self.c.room_crops]

    def look_room(self, question: str, again: bool = True) -> Answer:
        """A question about the room: Grok gets the camera's whole view, close-ups of the zones it asks
        about (else the far ones), the zone names and where they are, and the tracker's beliefs. Spoken
        only: the laser never aims off the table from this."""
        f = self._room_frame()
        if f is None or getattr(f, "img", None) is None:
            return Answer("I can't see the room right now.")
        img, _ = _jpeg(f.img, self.c.look_px)
        parts: list = [("text", "Image 1: the camera's whole view of the room now."), ("image", img)]
        for i, (say, crop) in enumerate(self._zone_crops(f.img, question, every=not again), 2):
            parts += [("text", f"Image {i}: close-up of {say}, now."),
                      ("image", _jpeg(crop, self.c.room_crop_px, upscale=True)[0])]
        parts.append(("text", f"Areas the rig knows: {self._zone_text(f.img)}.\n"
                              f"Tracker: {self._state_text(placed_only=True)}.\nQuestion: {question}"))
        try:
            d = self._vlm(ROOM_SYSTEM, parts, ROOM_SCHEMA)
        except (ProviderError, NarrationError) as ex:
            log.warning("room look failed: %s", ex)
            return Answer("Sorry, I couldn't look at the room just now.")
        text = _spoken(_unmark(str(d.get("answer") or "")))
        if not text or _conf(d) < self.c.abstain_below:
            return Answer(ABSTAIN)
        if again and WHERE_Q.search(question.lower()) and NEGATION.search(normalize_text(text)) and not self._capped():
            # "I don't see the notebook." for a notebook in plain view on the couch (rig 27 Sep 05:49; asked again
            # 15 s later: "The notebook is on the couch."): one more look, with every zone's close-up
            second = self.look_room(question, again=False)
            if second.text != ABSTAIN and not NEGATION.search(normalize_text(second.text)):
                log.info("room look: found on the second look: %r (first: %r)", second.text, text)
                return second
        return Answer(text, evidence=self._look_evidence(img, f.wall))

    def _look_evidence(self, jpg: bytes, wall: float) -> list:
        """The whole-room frame a look sent, saved in the snapshot dir as <ms>_look.jpg (pruned by age with
        the event snapshots), as the answer's evidence."""
        from core import evidence
        snap = getattr(self.events, "snap_dir", None)
        if not snap:
            return []
        path = os.path.join(snap, f"{int(wall * 1000)}_look.jpg")
        try:
            with open(path, "wb") as fh:
                fh.write(jpg)
        except OSError:
            log.warning("room look: could not save its frame for evidence", exc_info=True)
            return []
        return evidence.trim([evidence.item("look", path, wall, f"What the camera saw at {_clock(wall)}", snap)])

    def _zone_text(self, img: Optional[np.ndarray]) -> str:
        """The zones' spoken names, each with where it is in an image of the whole view (img None: in the
        view the zones were drawn on, which saved room frames are, shrunk)."""
        if img is not None:
            h, w = img.shape[:2]
        else:
            self._zone_polys((1, 1))
            w, h = (self._polys or {}).get("size") or (1, 1)
        polys = self._zone_polys((w, h)) if w and h else {}
        out = [f"{say} ({image_place(polys[n], (w, h))} of the view)" if n in polys else say
               for n, say in self.room_zones]
        rect = getattr(self.frames, "rect", None)                 # the table view's cut, camera px
        if rect is not None and w and h:
            x1, y1, x2, y2 = rect
            out.append(f"the table the tracker watches ({image_place([(x1, y1), (x2, y1), (x2, y2), (x1, y2)], (w, h))}"
                       f" of the view; call it \"the table\", it is none of the other areas)")
        return ", ".join(out)

    def look(self, question: str, intent: Optional[Intent] = None) -> Answer:
        obs = self._observe(question, intent)
        if obs is None:
            return Answer(CANT_SEE)
        oh, ow = obs.img.shape[:2]
        names, marks = shown_names(obs.names, obs.state), obs.marks
        full, _ = _jpeg(draw_marks(obs.img, [mk.box_px for mk in marks]) if marks else obs.img, self.c.look_px)
        parts: list = [("text", "Image 1: the whole table now, from above."), ("image", full)]
        for i, (n, crop) in enumerate(obs.crops, 2):
            nm = names.get(n) or NEW_THING
            parts += [("text", f"Image {i}: close-up of {nm if nm == NEW_THING or nm.endswith('?') else 'the ' + nm}, "
                               f"{_taken(obs.t - crop.t)}."),
                      ("image", _jpeg(crop.img, self.c.crop_px, upscale=True)[0])]
        listed = ", ".join(f"{i} = {names.get(mk.name) or NEW_THING}" for i, mk in enumerate(marks, 1))
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

    def pick(self, question: str, said: str) -> Answer:
        """'Where is my red mug?' for a name the world doesn't know: Grok only says which mark shows it and
        what it is; the world model says where (the WHERE template). A picked unnamed thing keeps the
        name, so the next question needs no Grok call. No marks to pick from: the full look. Pointing is
        verified like look()'s: an entity that moved or merged during the call is not pointed at.
        With room memory on, a thing not picked on the table (or no marks there) is looked for in the
        whole room, unless the question says "table"."""
        from voice.intents import normalize
        room = self._room_default(normalize(question))
        ob = self._observe(question, None)
        if ob is None:
            return self.look_room(question) if room else Answer(CANT_SEE)
        if not ob.marks:
            return self.look_room(question) if room else self.look(question)
        names = dict(ob.names)
        full, _ = _jpeg(draw_marks(ob.img, [mk.box_px for mk in ob.marks]), self.c.look_px)
        shown = shown_names(names, ob.state)
        listed = ", ".join(f"{i} = {shown.get(mk.name) or NEW_THING}" for i, mk in enumerate(ob.marks, 1))
        parts: list = [("image", full), ("text", f"Marks: {listed}.\nFind: {said}\nQuestion: {question}")]
        try:
            d = self._vlm(PICK_SYSTEM, parts, PICK_SCHEMA)
        except (ProviderError, NarrationError) as ex:
            log.warning("pick failed: %s", ex)
            return Answer("Sorry, I couldn't look at the table just now.")
        m, conf = d.get("mark"), _conf(d)
        picked = isinstance(m, int) and not isinstance(m, bool) and 1 <= m <= len(ob.marks)
        if not picked or conf < self.c.abstain_below:
            # Before any "I don't see it": the marked things as close-ups, where a small one is ~4x bigger
            d2 = self._pick_sheet(ob, shown, said, question)
            m2, conf2 = d2.get("mark"), _conf(d2)
            if (isinstance(m2, int) and not isinstance(m2, bool) and 1 <= m2 <= min(len(ob.marks), SHEET_MAX)
                    and conf2 >= max(self.c.abstain_below, BIND_CONF)):     # a second chance, so a firmer bar
                log.info("visual pick: the close-ups found %s (%.2f) after the marked frame gave %s", said, conf2,
                         m if picked else "none")
                d, m, conf, picked = d2, m2, conf2, True
        if room and (not picked or conf < self.c.abstain_below):
            return self.look_room(question)
        if not picked:
            return Answer(f"I can't see your {said} on the table right now." if conf >= self.c.abstain_below
                          else ABSTAIN)
        if conf < self.c.abstain_below:
            return Answer(ABSTAIN)
        mk = ob.marks[m - 1]
        ent = mk.name
        label = _label(d.get("label"))
        if ent.startswith("thing:") and not names.get(ent) and self._guess_disagrees(ent, said):
            # Its own name guess says it's something else (rig 27 Sep 05:17: 'laptop' picked the notebook,
            # guessed 'notebook' 0.8, while the laptop was on the couch): no pick, no name taught.
            log.info("visual pick: %s picked for '%s' but guessed '%s'; not taken", ent, said,
                     (self.world.thing_guess(ent) or {}).get("name"))
            return self.look_room(question) if room else Answer(f"I can't see your {said} on the table right now.")
        holds = self._still_holds(mk)       # checked before any naming: a thing that moved keeps no name
        if holds and ent.startswith("thing:"):
            names[ent] = self._names().get(ent) or names.get(ent)      # a name taught during the call wins
        if (holds and ent.startswith("thing:") and not names.get(ent) and conf >= BIND_CONF
                and hasattr(self.world, "bind_alias")):
            try:
                if self.world.bind_alias(ent, said):
                    log.info("visual pick: %s is now '%s' (Grok saw %s, %.2f)", ent, said, label, conf)
                    names[ent] = said
            except Exception:
                log.exception("bind_alias failed")
        from voice.answers import answer
        try:
            where = answer(Intent("WHERE", None, "", name=ent), self.world, self.events, self.cfg)
        except Exception:
            log.exception("pick: WHERE template failed")
            where = Answer("")
        if ent.startswith("thing:") and not names.get(ent):
            text = f"I think your {said} is this {label}." if label and label != said.lower() else \
                f"I think this is your {said}."
        else:
            text = where.text
            known = names.get(ent) or ""
            if known.lower() != said.lower():           # Grok picked something the rig knows by another name
                text = f"I think your {said} is what I call your {known}. {text}".strip()
        out = self._verified(text or f"I think this is your {said}.", mk)
        out.evidence, out.obj = list(where.evidence or []), ent    # the picked thing's receipt
        return out

    def _guess_disagrees(self, ent: str, said: str) -> bool:
        """The auto-namer's guess for ent (core/auto_name) is confident (>= BIND_CONF) and fits `said` not at
        all, alternatives included (match_score 0)."""
        try:
            g = self.world.thing_guess(ent) if hasattr(self.world, "thing_guess") else None
        except Exception:
            return False
        if not isinstance(g, dict) or float(g.get("confidence") or 0) < BIND_CONF:
            return False
        from core.auto_name import match_score
        return match_score(said, g) <= 0

    def _pick_sheet(self, ob, shown: dict, said: str, question: str) -> dict:
        """The pick again on a sheet of the marked things' close-ups (the first SHEET_MAX marks); {} if the
        call fails or the cap is reached."""
        if self._capped():
            return {}
        marks = ob.marks[:SHEET_MAX]
        sheet, _ = _jpeg(pick_sheet(ob.img, [mk.box_px for mk in marks]), self.c.look_px)
        listed = ", ".join(f"{i} = {shown.get(mk.name) or NEW_THING}" for i, mk in enumerate(marks, 1))
        parts: list = [("image", sheet), ("text", f"Close-ups: {listed}.\nFind: {said}\nQuestion: {question}")]
        try:
            return self._vlm(PICK_SHEET_SYSTEM, parts, PICK_SCHEMA)
        except (ProviderError, NarrationError) as ex:
            log.warning("pick (close-ups) failed: %s", ex)
            return {}

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
        if not self._still_holds(mk):
            return Answer(_with_note(text, MOVED, where))
        return Answer(text, point_at=mk.name, action="point")

    def _still_holds(self, mk: Mark) -> bool:
        """mk's entity still exists unmerged, and is hidden (followed by the world) or still visible
        within JUMP_CM of where the mark had it."""
        try:
            ent = self.world.get(mk.name)
            status, pos, merged = str(ent.status), ent.pos_cm, getattr(ent, "merged_into", None)
        except Exception:
            return False
        if merged:
            return False
        if status == "VISIBLE":
            return not (mk.pos_cm is None or pos is None
                        or np.hypot(pos[0] - mk.pos_cm[0], pos[1] - mk.pos_cm[1]) > JUMP_CM)
        return status in HIDDEN

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
        if px is not None and (self.cfg.get("room") or {}).get("enabled") and not self._on_table(c):
            return Answer(text, action=f"room:{px[0]:.0f},{px[1]:.0f}")    # off the table: the image spot (spec 0006)
        hit = self._mark_at(float(c[0][0]), float(c[0][1]), marks) if c is not None else None
        if hit is None:
            return Answer(_with_note(text, UNSURE_WHERE, where))
        return self._verified(text, hit, where)

    def _on_table(self, c) -> bool:
        """A point in table cm (px_to_cm's output) lies on the table, within 5 cm of its edge."""
        if c is None:
            return False
        x, y = float(c[0][0]), float(c[0][1])
        w, h = ((self.cfg.get("table") or {}).get("size_cm") or [90, 60])
        return -5 <= x <= w + 5 and -5 <= y <= h + 5

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

    def _recall_view(self, question: str, t0: float, t1: float) -> str:
        """'room' when room memory is on, the question doesn't keep to the table and the archive has room
        frames in [t0, t1]; else 'table'."""
        from voice.intents import normalize
        if not self._room_default(normalize(question)):
            return "table"
        return "room" if any(r.path for r in self.archive.store.window(t0, t1, "room")) else "table"

    def recall(self, question: str) -> Answer:
        if self.archive is None:
            return Answer("I don't keep pictures of the table, so I can't tell.")
        now = self.clock()
        t0, t1, label = self._window(question, now)
        view = self._recall_view(question, t0, t1)
        what = "the room" if view == "room" else "the table"
        phrase = query_phrase(question)
        hits = self.archive.search(phrase, t0, t1, self.c.recall_frames, view=view) if phrase else []
        if phrase and hits and hits[0][1] < self.c.min_sim:
            # On room frames a small far thing scores like an absent one (a cup on the counter 0.195, an
            # umbrella nowhere 0.212; eval/room_look.py): Grok looks at frames spread over the window instead.
            if view == "table":
                return Answer(f"I don't remember seeing {phrase}{label}.")
            hits = []
        rows = [r for r, _ in hits] or (self.archive.sample(t0, t1, self.c.recall_frames, view=view)
                                        if "before you left" not in label
                                        else self.archive.store.window(t0, t1, view)[-self.c.recall_frames:])
        rows = sorted((r for r in rows if r.path), key=lambda r: r.t)
        if not rows:
            return Answer(f"I don't have any saved pictures of {what}{label}.")
        context = self._context(t0, t1)
        if view == "room":
            context += f"\nAreas the rig knows: {self._zone_text(None)}."
        parts: list = [("text", context)]
        import cv2
        from voice.intents import _without_wake_word, normalize
        named = self._named_zones(_without_wake_word(normalize(question), self.cfg)) if view == "room" else []
        say = dict(self.room_zones or [])
        n, sent = 0, []
        for r in rows:
            img = cv2.imread(r.path)
            if img is None:
                continue
            n += 1
            sent.append(r)
            parts += [("text", f"{'Picture' if view == 'room' else 'Frame'} {n} at {_clock(r.t)}"
                               f"{', a hand in view' if r.hands else ''}; tracker: {digest_text(r.digest)}."),
                      ("image", _jpeg(img, self.c.look_px if view == "room" else self.c.recall_px)[0])]
            polys = self._zone_polys(img.shape[1::-1]) if named else {}
            for z in named[:2]:              # the areas asked about, enlarged from the saved picture
                box = zone_box(polys[z], img.shape[1::-1], min_px=128) if z in polys else None
                if box is not None:
                    x1, y1, x2, y2 = box
                    parts += [("text", f"Picture {n}, close-up of {say.get(z, z)}."),
                              ("image", _jpeg(img[y1:y2, x1:x2], self.c.room_crop_px // 2, upscale=True)[0])]
        if n == 0:
            return Answer(f"I don't have any saved pictures of {what}{label}.")
        parts.append(("text", f"Question: {question}"))
        try:
            d = self._vlm(ROOM_RECALL_SYSTEM if view == "room" else RECALL_SYSTEM, parts, RECALL_SCHEMA)
        except (ProviderError, NarrationError) as ex:
            log.warning("recall failed: %s", ex)
            return Answer("Sorry, I couldn't go through the saved pictures just now.")
        text = _spoken(str(d.get("answer") or ""))
        if not text or _conf(d) < self.c.abstain_below:
            return Answer("I can't tell from the pictures I saved.")
        return Answer(text, evidence=self._recall_evidence(d.get("pictures"), sent))

    def _recall_evidence(self, cited, sent: list) -> list:
        """The saved frames Grok says its answer rests on (reply 'pictures': 1-based numbers), newest first."""
        from core import evidence
        snap = getattr(self.events, "snap_dir", None)
        nums = [int(i) for i in (cited if isinstance(cited, list) else [])
                if isinstance(i, (int, float)) and not isinstance(i, bool) and 1 <= int(i) <= len(sent)]
        rows = sorted({sent[i - 1].id: sent[i - 1] for i in nums}.values(), key=lambda r: -r.t)
        return evidence.trim(evidence.item("recall", r.path, r.t, f"Saved picture, {_clock(r.t)}", snap)
                             for r in rows)

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
        target = None
        if k == "OTHER":
            about_room = self._about_room(t)
            if not about_room and not self._about_table(intent, t):
                return None
            if past:
                how = "recall"          # the room's saved frames when room memory is on (recall picks)
            elif about_room:
                how = "room"
            elif intent.obj or intent.name or not self._room_default(t):
                how = "look"            # "table" said, or a tracked prop named: the table look and its close-ups
            else:
                how = "room"            # "what do you see?" with room memory on: the whole room
        elif k in ("WHERE", "HISTORY", "HANDLED") and (intent.name or intent.obj):
            try:
                target = _target(intent, self.world, self.cfg)
                known = target is not None
            except Exception:
                target, known = None, True
            said = [w for w in (intent.name or intent.obj, query_phrase(text)) if w]
            if known:
                if k != "WHERE" or not target:
                    return None
                seen = self._sighting(ent=target)
                if seen is not None or not (self._room_default(t) and (self._never_placed(target)
                                                                         or self._stale_room_place(target))):
                    return seen
                how = "room"            # never seen, or its room place is stale: Grok looks around the room
                said = []
            elif k == "WHERE":
                how = "pick"
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
            return self._sighting(said=said[0]) if k == "WHERE" and said else None
        if self._capped():
            log.info("visual questions: hourly cap reached; answering without the camera")
            return None
        if how == "pick":
            return self.pick(text, said[0])
        if how == "room":
            a = self.look_room(text)
            a.obj = target                  # a known object looked for in the room (None for other questions)
            return a
        return self.look(text, intent) if how == "look" else self.recall(text)

    def _sighting(self, ent: Optional[str] = None, said: Optional[str] = None) -> Optional[Answer]:
        """WHERE for something the world has never had a position for (a tracked object still UNKNOWN,
        or offline, a name it doesn't know), from the Grok settle check's newest sighting of it. Stored
        rows, so it works offline. Spoken only: the spot is a VLM estimate that may be minutes old, and the
        laser only ever aims at tracked entities.
        None: the templates answer as before."""
        if self.grok_check is None:
            return None
        if ent is not None:
            if not self._never_placed(ent):
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
        from voice.answers import area
        text = _spoken(f"I haven't tracked your {said}, but at {_clock(hit['wall'])} I saw what looked like "
                       f"your {said} {area((hit['x_cm'], hit['y_cm']), self.cfg)}.")
        return Answer(text)                 # an old, ungrounded VLM sighting: spoken only, never aimed at

    def _never_placed(self, ent: str) -> bool:
        """A tracked object the world has never had a position for (UNKNOWN, never seen)."""
        try:
            e = self.world.get(ent)
        except Exception:
            return False
        return e.status == "UNKNOWN" and e.pos_cm is None and e.last_seen is None

    def _stale_room_place(self, ent: str) -> bool:
        """The world places ent in a room zone but hasn't seen it there lately (or sees the spot empty): a
        look at the room now beats repeating an old place. The tracker text still says where it was."""
        try:
            place = self.world.place(ent, self.clock()) if hasattr(self.world, "place") else None
        except Exception:
            return False
        return place is not None and place.kind == "room" and not place.fresh

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
                "disclosure": (f"Visual questions are on: the current camera frame of the table"
                               f"{', the whole room view with close-ups of its areas (the default view for questions with room memory on)' if self._room_frame() is not None else ''}"
                               f" and, for questions about earlier, up to {self.c.recall_frames} saved frames"
                               f"{' (of the table or the whole room)' if self._room_frame() is not None else ''} are sent to "
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
    archive = VisualArchive(cfg, events, world, embedder=make_embedder(c), start=start, c=c,
                            frames=frames, table=table).attach(world)
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
