"""Object permanence (spec 0011): a registry of the objects that matter, re-found anywhere in the room.

Only with `permanence.mode: registry` (default off: make_permanence returns None and nothing changes). The
registry holds the configured props and taught things, each with a few reference crops and their DINOv2
appearance embeddings (core/embed.py, the reid: section with `enabled` forced on for the registry only). A slow
loop looks at the native full camera frame one view at a time (a tile grid plus zoomed views of far areas):
YOLOE proposes candidates, people are masked out, embeddings are compared with the references, and a candidate
at a new spot is confirmed by a closed Grok question. Each registered object has one state: visible (where),
hidden (a person covers its last spot), carried (a hand or person touched it and it left), last seen (place,
time), found again (anywhere). There is no table-to-zone handoff: the couch, the floor and the doorway all work
the same way.

Positions here are full-frame pixels (the camera's native 2560x1440 frame); places are said in human terms
('the couch', 'the floor by the doorway'). Table centimetres stay with the table world (the laser, the map).
The registry never creates identities from unnamed proposals.

attach(world) wraps world.place and world.state_json (as core/auto_name.attach wraps state_json), so answers,
/state, the phone and the bridge read registry states without core/world.py changing. State changes are
ordinary events in the EventLog (FOUND, PICKED_UP, LOST_TRACK, MOVED): history, 'what changed' and HANDLED
read them as before. Python 3.10 (JetPack 6).
"""
from __future__ import annotations

import argparse
import logging
import math
import queue
import threading
import time
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Callable, Optional, Sequence

import cv2
import numpy as np

from core import geom
from core.room_types import Place
from core.things import ExemplarBank
from core.types import BoxPx, Event, Status

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]

VISIBLE, HIDDEN, CARRIED, LAST_SEEN, UNKNOWN = "visible", "hidden", "carried", "last_seen", "unknown"
REFIND = "refind"       # how: found by a re-find backend (answers hedge until appearance alone re-finds it)
PEOPLE = ("person", "man", "woman", "child", "boy", "girl", "people")
# never a registered object: people's parts and clothing, and the room itself
IGNORE = PEOPLE + ("hand", "arm", "finger", "leg", "knee", "foot", "feet", "toe", "sock", "shoe", "sneaker",
                   "boot", "sandal", "slipper", "sleeve", "shirt", "t-shirt", "jeans", "pants", "shorts",
                   "trousers", "jacket", "sweater", "hoodie", "hair", "head", "face", "table", "dining table",
                   "desk", "coffee table", "tabletop", "countertop", "floor", "wall", "ceiling", "wood",
                   "wood floor", "hardwood", "carpet", "rug", "couch", "sofa", "chair", "door", "window",
                   "cabinet", "refrigerator", "oven", "stove", "microwave", "curtain")
DUP_SIM = 0.97          # a reference this close to a stored one adds nothing


@dataclass
class PermanenceConfig:
    """config.yaml `permanence:` (a new section at the end). Every key optional; these are the defaults."""
    mode: str = "off"                       # off | registry
    objects: Optional[list] = None          # names registered at start (None: every configured object)
    every_n: int = 1                        # one view every N perception frames
    tiles: tuple = (3, 2)                   # grid over the native full frame (cols, rows)
    tile_overlap: float = 0.15
    zoom: object = "zones"                  # 'zones' (each drawn zone's box, native), [] or [[x1,y1,x2,y2], ...]
    view_px: int = 1280                     # a view with a longer side is resized down to this for YOLOE
    imgsz: int = 640
    conf: float = 0.15                      # YOLOE box confidence
    min_side_px: int = 12                   # full-frame px; smaller boxes aren't candidates
    max_area_frac: float = 0.03             # of the full frame; bigger boxes are furniture, not objects
    reuse_iou: float = 0.8                  # a candidate this much on one from the view's last look keeps its ...
    reembed_s: float = 10.0                 # ... embedding for this long (a still room costs no embeddings)
    # Measured on rig crops (spec 0011 section 3.3): the same prop at the same spot scores ~0.85 (p25 0.69),
    # but the same prop elsewhere can score 0.1-0.3 and clutter at one spot up to 0.9. So appearance only
    # keeps an object at its spot; a new spot needs arrival evidence and Grok.
    sim_accept: float = 0.55                # embedding similarity: the object still at its last spot
    sim_accept_far: float = 0.88            # ... anywhere else without asking Grok (rare)
    sim_backstop: float = 0.35              # an object never seen: candidates this similar are asked about
    learn_sim: float = 0.75                 # a match this good adds a reference view ...
    learn_every_s: float = 30.0             # ... at most this often per object
    refs_max: int = 8
    refs_dir: str = "data/registry"         # <refs_dir>/<name>/*.jpg: enrolled reference crops
    near_frac: float = 1.0                  # 'at its last spot': centre within this many box diagonals
    person_inside: float = 0.6              # a candidate this much inside a person box is the person
    hidden_overlap: float = 0.3             # a person box covering this much of the last spot: hidden
    contact_pad: float = 0.2                # the object's box grown by this (each side) for contact
    contact_s: float = 3.0                  # a person or hand this recently before it went: carried
    miss_visits: int = 2                    # valid views without it ...
    miss_s: float = 3.0                     # ... and this long since it was seen: carried or last seen
    fresh_s: float = 10.0                   # seen this recently: said in the present tense
    verify: bool = True                     # re-find missing objects at new spots (below: the backend)
    refind: str = "grok_marks"              # grok_marks: Grok picks among numbered suspects (set-of-marks)
    confirm: bool = True                    # ... and a closed same-object question confirms the pick
    confirm_conf: float = 0.8
    verify_conf: float = 0.7
    verify_per_minute: int = 20
    verify_timeout_s: float = 8.0
    ask_every_s: float = 5.0                # a missing object is asked about at most this often
    arrival_s: float = 2.0                  # a candidate first seen this long before it went still counts
    marks: int = 6                          # candidates marked in one question
    backstop_s: float = 180.0               # objects with no new arrivals: asked about their best matches (0: off)
    reask_s: float = 60.0                   # a candidate re-find turned down isn't asked about again this soon
    places: list = field(default_factory=list)   # [{name, say, poly: [[x, y], ...]}] full-frame px, after zones
    table_say: str = "the table"
    near_px: float = 250.0                  # outside every place: 'near <zone>' within this many px
    table_fresh_s: float = 2.0              # the table world had it this recently: its answer wins
    hide_unnamed: bool = True               # /state leaves out unnamed things that aren't visible now

    @classmethod
    def from_dict(cls, raw: Optional[dict]) -> "PermanenceConfig":
        known = {f.name for f in fields(cls)}
        c = cls(**{k: v for k, v in (raw or {}).items() if k in known})
        c.tiles = tuple(int(v) for v in c.tiles)
        return c


# ================================================================================ places

@dataclass
class Region:
    name: str
    say: str
    poly: list

    def contains(self, pt) -> bool:
        if len(self.poly) < 3:
            return False
        contour = np.asarray(self.poly, np.float32).reshape(-1, 1, 2)
        return cv2.pointPolygonTest(contour, (float(pt[0]), float(pt[1])), False) >= 0

    def dist(self, pt) -> float:
        if len(self.poly) < 3:
            return math.inf
        contour = np.asarray(self.poly, np.float32).reshape(-1, 1, 2)
        return max(0.0, -cv2.pointPolygonTest(contour, (float(pt[0]), float(pt[1])), True))


class Places:
    """A full-frame point said in human terms: the table, a drawn zone, an extra place, 'near <zone>', else a
    side of the room."""

    def __init__(self, regions: Sequence[Region] = (), table_rect: Optional[BoxPx] = None,
                 table_say: str = "the table", frame_wh: tuple = (2560, 1440), near_px: float = 250.0):
        self.regions = list(regions)
        self.table_rect, self.table_say = table_rect, table_say
        self.frame_wh, self.near_px = frame_wh, near_px

    def at(self, box: BoxPx) -> tuple[str, str]:
        """(zone name, spoken place) for a box; the box's bottom centre is where it rests."""
        pt = ((box[0] + box[2]) / 2, box[3] - 0.1 * (box[3] - box[1]))
        if self.table_rect is not None:
            x1, y1, x2, y2 = self.table_rect
            if x1 <= pt[0] <= x2 and y1 <= pt[1] <= y2:
                return "table", self.table_say
        for r in self.regions:
            if r.contains(pt):
                return r.name, r.say
        near = min(self.regions, key=lambda r: r.dist(pt), default=None)
        if near is not None and near.dist(pt) <= self.near_px:
            return f"near:{near.name}", f"near {near.say}"
        w = self.frame_wh[0]
        side = "left side" if pt[0] < w / 3 else "right side" if pt[0] > 2 * w / 3 else "middle"
        return f"room:{side.split()[0]}", f"the {side} of the room"


def load_places(cfg: dict, c: PermanenceConfig) -> Places:
    """Zones from room_memory.zones_path (if the file loads), then permanence.places; the table rect from
    room_memory.table_view_rect."""
    rm = (cfg or {}).get("room_memory") or {}
    regions: list[Region] = []
    try:
        from core.room_zones import Zones
        zs = Zones.load(rm.get("zones_path", "room_zones.json"))
        regions += [Region(z.name, z.say, [list(p) for p in z.poly]) for z in zs.zones.values()]
    except Exception:
        pass
    regions += [Region(str(p.get("name") or p.get("say")), str(p.get("say") or p.get("name")),
                       [list(v) for v in p.get("poly") or []]) for p in c.places or []]
    rect = rm.get("table_view_rect")
    size = tuple(rm.get("capture_size") or (2560, 1440))
    return Places(regions, tuple(rect) if rect else None, c.table_say, size, c.near_px)


# ================================================================================ registry

@dataclass
class Candidate:
    box: BoxPx                      # full-frame px
    conf: float
    cls: str = ""
    emb: Optional[np.ndarray] = None
    in_person: bool = False         # mostly inside a person box (a knee, or a phone on a lap): asked about last
    first_wall: float = 0.0         # first seen at this spot (the view's previous candidates carry it)
    emb_wall: float = 0.0           # when emb was computed


@dataclass
class RegObject:
    name: str                       # the world entity it answers for ('remote', or a taught 'thing:7')
    bank: ExemplarBank
    refs: list = field(default_factory=list)    # small BGR reference crops (for Grok and the dashboard)
    state: str = UNKNOWN
    box: Optional[BoxPx] = None
    zone: Optional[str] = None
    say: Optional[str] = None
    seen_wall: Optional[float] = None
    since_wall: Optional[float] = None          # when the current state began
    arrived_wall: Optional[float] = None        # when it appeared at the current place
    arrival_observed: bool = False              # seen somewhere else first (so 'appeared there' is true)
    contact_wall: Optional[float] = None        # a person or hand last touched its box
    misses: int = 0
    learned_wall: float = 0.0
    asking: bool = False                        # a Grok question about it is in flight
    asked_wall: float = float("-inf")
    search_view: int = 0                        # a grounding re-find's next view
    rejected: list = field(default_factory=list)   # [(box, wall)] candidates a re-find turned down
    how: str = ""                               # how it was last found: 'look', 'teach', 'refind:<source>'


def _crop(img: np.ndarray, box, margin: float = 0.15, min_side: int = 0) -> Optional[np.ndarray]:
    h, w = img.shape[:2]
    x1, y1, x2, y2 = box
    mx, my = (x2 - x1) * margin, (y2 - y1) * margin
    if min_side:
        cx, cy, s = (x1 + x2) / 2, (y1 + y2) / 2, max(x2 - x1 + 2 * mx, y2 - y1 + 2 * my, min_side)
        x1, y1, x2, y2 = cx - s / 2, cy - s / 2, cx + s / 2, cy + s / 2
        mx = my = 0
    a, b = max(0, int(x1 - mx)), max(0, int(y1 - my))
    c, d = min(w, int(math.ceil(x2 + mx))), min(h, int(math.ceil(y2 + my)))
    return img[b:d, a:c].copy() if c - a >= 2 and d - b >= 2 else None


def _small(img: Optional[np.ndarray], side: int = 160) -> Optional[np.ndarray]:
    if img is None:
        return None
    s = side / max(img.shape[:2])
    return cv2.resize(img, (max(1, round(img.shape[1] * s)), max(1, round(img.shape[0] * s))),
                      interpolation=cv2.INTER_AREA) if s < 1 else img


def ref_patch(img: np.ndarray, box: BoxPx) -> np.ndarray:
    """What Grok is shown as a reference: the object in a red box inside some of its surroundings."""
    x1, y1, x2, y2 = box
    s = max(160, 3 * max(x2 - x1, y2 - y1))
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    h, w = img.shape[:2]
    a, b = int(max(0, cx - s / 2)), int(max(0, cy - s / 2))
    c, d = int(min(w, cx + s / 2)), int(min(h, cy + s / 2))
    patch = img[b:d, a:c].copy()
    k = min(3.0, max(1.0, 320 / max(1, min(patch.shape[:2]))))
    patch = cv2.resize(patch, (round(patch.shape[1] * k), round(patch.shape[0] * k)), interpolation=cv2.INTER_CUBIC)
    cv2.rectangle(patch, (int((x1 - a) * k), int((y1 - b) * k)), (int((x2 - a) * k), int((y2 - b) * k)), (0, 0, 255), 2)
    return patch


def _covers(a: BoxPx, b: BoxPx) -> float:
    return geom.overlap_frac(a, b)


def _grow(box: BoxPx, frac: float) -> BoxPx:
    dx, dy = (box[2] - box[0]) * frac, (box[3] - box[1]) * frac
    return (box[0] - dx, box[1] - dy, box[2] + dx, box[3] + dy)


def tile_boxes(w: int, h: int, grid: tuple = (3, 2), overlap: float = 0.15) -> list[BoxPx]:
    """Grid tiles over a w x h frame, each grown by overlap of its size (clipped): full-frame px."""
    cols, rows = max(1, int(grid[0])), max(1, int(grid[1]))
    tw, th = w / cols, h / rows
    out = []
    for r in range(rows):
        for c in range(cols):
            x1, y1 = c * tw - overlap * tw, r * th - overlap * th
            x2, y2 = (c + 1) * tw + overlap * tw, (r + 1) * th + overlap * th
            out.append((max(0, int(x1)), max(0, int(y1)), min(w, int(math.ceil(x2))), min(h, int(math.ceil(y2)))))
    return out


class Permanence:
    """The registry and its slow whole-frame loop. Injected callables keep it testable without models:
    detect(img) -> [(class name, conf, box px in img)], embed(img, boxes) -> [unit vector | None], and
    refind(name, refs, view img, view box, suspect boxes) -> [(box, confidence, source)] (full-frame px): the
    re-find backend for a missing object (marks_refinder: Grok picks among numbered suspects; a grounding
    backend may return a box anywhere in the view). It runs on a worker thread."""

    def __init__(self, c: PermanenceConfig, detect: Callable, embed: Callable, events=None,
                 places: Optional[Places] = None, refind: Optional[Callable] = None,
                 names: Optional[Callable[[str], str]] = None, clock: Callable[[], float] = time.time,
                 zoom_boxes: Sequence[BoxPx] = (), online: Callable[[], bool] = lambda: True):
        self.c = c
        self.detect, self.embed, self.events = detect, embed, events
        self.places = places or Places(table_say=c.table_say)
        self.refind = refind if c.verify else None
        self.names = names or (lambda n: n.replace("_", " "))
        self.clock, self.online = clock, online
        self.zoom_boxes = [tuple(int(v) for v in b) for b in zoom_boxes]
        self.objects: dict[str, RegObject] = {}
        self.lock = threading.RLock()
        self._n = 0
        self._next = 0
        self._views: list[BoxPx] = []
        self._views_wh: Optional[tuple] = None
        self._cands: dict[int, list[Candidate]] = {}      # view -> its latest candidates (the backstop's pool)
        self._people: dict[int, list[BoxPx]] = {}
        self._frame: Optional[np.ndarray] = None
        self._results: queue.Queue = queue.Queue()
        self._asks: queue.Queue = queue.Queue(maxsize=8)
        self._ask_times: list[float] = []
        self._worker: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._backstop_now: set[str] = set()
        self.last_ms = 0.0
        self.embedded = 0                                 # candidates embedded in the last look
        self.sweep_wall: Optional[float] = None           # when the last view of a full sweep was done

    # ----- the registry

    def register(self, name: str) -> RegObject:
        with self.lock:
            if name not in self.objects:
                self.objects[name] = RegObject(name, ExemplarBank(self.c.refs_max, DUP_SIM))
            return self.objects[name]

    def add_ref(self, name: str, crop: np.ndarray, emb: Optional[np.ndarray] = None,
                context: Optional[np.ndarray] = None) -> bool:
        """A reference view of name (a BGR crop around the object); embedded here unless emb is given. context
        (the object boxed in its surroundings, ref_patch) is what Grok is shown, if given: a tight crop of a
        small object is a coloured blob Grok can't vouch for (rig, Sun 27 Sep)."""
        if crop is None or crop.size == 0:
            return False
        if emb is None:
            h, w = crop.shape[:2]
            emb = (self.embed(crop, [(0, 0, w, h)]) or [None])[0]
        if emb is None:
            return False
        o = self.register(name)
        with self.lock:
            added = o.bank.add(np.asarray(emb, np.float32))
            if added:
                o.refs = (o.refs + [context if context is not None else _small(crop)])[-self.c.refs_max:]
            return added

    def enroll_dir(self, root: Optional[str] = None) -> int:
        """Load <refs_dir>/<name>/*.jpg for every name there. Returns the views added."""
        base = Path(root or self.c.refs_dir)
        base = base if base.is_absolute() else ROOT / base
        n = 0
        for d in sorted(p for p in base.glob("*") if p.is_dir()) if base.exists() else []:
            for f in sorted(d.glob("*.jpg")) + sorted(d.glob("*.png")):
                img = cv2.imread(str(f))
                n += bool(img is not None and self.add_ref(d.name, img))
        return n

    def teach(self, name: str, img: np.ndarray, box: BoxPx) -> bool:
        """'this is my X' bound entity name to the object at box (full-frame px of img): register it."""
        crop = _crop(img, box, 0.0)
        emb = (self.embed(img, [tuple(box)]) or [None])[0]
        ok = self.add_ref(name, crop, emb, ref_patch(img, box)) if crop is not None else False
        if ok:
            with self.lock:
                o = self.objects[name]
                if o.state != VISIBLE:
                    self._seen(o, Candidate(tuple(box), 1.0, emb=emb), self.clock(), [], how="teach")
        return ok

    def reset(self) -> None:
        """Forget where everything is (a new judge); keep what everything looks like."""
        with self.lock:
            for o in list(self.objects.values()):
                self.objects[o.name] = RegObject(o.name, o.bank, o.refs)
            self._cands.clear()
            self._people.clear()

    # ----- the slow loop

    def views(self, w: int, h: int) -> list[BoxPx]:
        if self._views_wh != (w, h):
            self._views = tile_boxes(w, h, self.c.tiles, self.c.tile_overlap)
            self._views += [(max(0, b[0]), max(0, b[1]), min(w, b[2]), min(h, b[3])) for b in self.zoom_boxes]
            self._views = [v for v in self._views if v[2] - v[0] >= 16 and v[3] - v[1] >= 16]
            self._views_wh = (w, h)
        return self._views

    def step(self, img: Optional[np.ndarray], wall: Optional[float] = None,
             blockers: Sequence[BoxPx] = ()) -> list[Event]:
        """One perception frame: every every_n frames, one view of the native full frame. blockers: hand
        boxes in full-frame px from the fast loop (contact evidence)."""
        wall = self.clock() if wall is None else wall
        evs = self._apply_results(wall)
        self._n += 1
        if img is None or self._n % max(1, self.c.every_n):
            return evs
        vs = self.views(img.shape[1], img.shape[0])
        if not vs:
            return evs
        i = self._next % len(vs)
        self._next += 1
        evs += self.look(img, i, wall, blockers)
        if i == len(vs) - 1:
            self.sweep_wall = wall
        return evs

    def sweep(self, img: np.ndarray, wall: Optional[float] = None, blockers: Sequence[BoxPx] = ()) -> list[Event]:
        """Every view once (offline replays and tests)."""
        wall = self.clock() if wall is None else wall
        evs = self._apply_results(wall)
        for i in range(len(self.views(img.shape[1], img.shape[0]))):
            evs += self.look(img, i, wall, blockers)
        self.sweep_wall = wall
        return evs

    def _detect_view(self, img: np.ndarray, vb: BoxPx) -> tuple[list[BoxPx], list[Candidate]]:
        x1, y1, x2, y2 = vb
        crop = img[y1:y2, x1:x2]
        s = min(1.0, self.c.view_px / max(crop.shape[:2]))
        small = cv2.resize(crop, (round(crop.shape[1] * s), round(crop.shape[0] * s)),
                           interpolation=cv2.INTER_AREA) if s < 1 else crop
        people, cands = [], []
        fw, fh = img.shape[1], img.shape[0]
        for cls, conf, b in self.detect(small):
            if conf < self.c.conf:
                continue
            box = (x1 + b[0] / s, y1 + b[1] / s, x1 + b[2] / s, y1 + b[3] / s)
            name = str(cls).lower()
            if name in PEOPLE:
                people.append(box)
                continue
            if name in IGNORE:
                continue
            bw, bh = box[2] - box[0], box[3] - box[1]
            if min(bw, bh) < self.c.min_side_px or bw * bh > self.c.max_area_frac * fw * fh:
                continue
            cands.append(Candidate(tuple(int(round(v)) for v in box), float(conf), name))
        for k in cands:     # a knee or a sleeve; or a phone on a lap, which only a strong match may claim
            k.in_person = any(_covers(p, k.box) >= self.c.person_inside for p in people)
        return people, _nms(cands)

    def look(self, img: np.ndarray, i: int, wall: float, blockers: Sequence[BoxPx] = ()) -> list[Event]:
        """Process view i of img: candidates, contact, matching, misses. Returns the events."""
        t0 = time.perf_counter()
        vb = self.views(img.shape[1], img.shape[0])[i]
        people, cands = self._detect_view(img, vb)
        with self.lock:
            last = list(self._cands.get(i, []))
        for k in cands:                         # arrival evidence: new at this spot since the view's last look
            prev = max(last, key=lambda q: geom.iou(q.box, k.box), default=None)
            iou = geom.iou(prev.box, k.box) if prev is not None else 0.0
            k.first_wall = prev.first_wall if iou >= 0.5 else wall
            if iou >= self.c.reuse_iou and prev.emb is not None and wall - prev.emb_wall < self.c.reembed_s:
                k.emb, k.emb_wall = prev.emb, prev.emb_wall            # a still thing: no new embedding
        todo = [k for k in cands if k.emb is None]
        if todo:
            for k, e in zip(todo, self.embed(img, [k.box for k in todo]) or []):
                k.emb, k.emb_wall = (None if e is None else np.asarray(e, np.float32)), wall
        cands = [k for k in cands if k.emb is not None]
        self.embedded = len(todo)
        with self.lock:
            self._frame = img
            self._cands[i], self._people[i] = cands, people
            # people from every view's latest look: a tile cuts one person into a head here and a torso there
            people = [b for ps in self._people.values() for b in ps]
            touch = list(people) + [tuple(b) for b in blockers]
            for o in self.objects.values():
                if o.box is not None and o.state in (VISIBLE, HIDDEN) and _inside(o.box, vb) \
                        and any(geom.intersection(_grow(o.box, self.c.contact_pad), t) for t in touch):
                    o.contact_wall = wall
            evs: list[Event] = []
            matched: set[str] = set()
            pairs = sorted(((o.bank.sim(k.emb), o.name, j) for o in self.objects.values() if o.bank.vecs
                            for j, k in enumerate(cands)), reverse=True)
            used: set[int] = set()
            for sim, name, j in pairs:
                if sim < self.c.sim_accept:
                    break
                if name in matched or j in used:
                    continue
                o, k = self.objects[name], cands[j]
                near = self._near(o, k.box)
                if near or sim >= self.c.sim_accept_far:
                    how = o.how if near and o.how.startswith(REFIND) else "look"   # found by re-find: hedged
                    evs += self._seen(o, k, wall, people, how=how, sim=sim)
                    matched.add(name)
                    used.add(j)
            for o in self.objects.values():
                if o.name not in matched and o.box is not None and _inside(o.box, vb):
                    evs += self._missed(o, wall, people)
            self._maybe_ask(img, wall)
        self.last_ms = 1000 * (time.perf_counter() - t0)
        return evs

    def _near(self, o: RegObject, box: BoxPx) -> bool:
        if o.box is None or o.state not in (VISIBLE, HIDDEN):
            return False
        diag = max(20.0, math.hypot(o.box[2] - o.box[0], o.box[3] - o.box[1]))
        return geom.dist(geom.center(o.box), geom.center(box)) <= self.c.near_frac * diag

    # ----- states

    def _event(self, name: str, typ: str, wall: float) -> Event:
        ev = Event(t=time.monotonic(), wall=wall, obj=name, type=typ)
        if self.events is not None:
            try:
                self.events.add(ev)
            except Exception:
                log.exception("permanence: event not stored")
        return ev

    def _seen(self, o: RegObject, k: Candidate, wall: float, people, how: str, sim: float = 1.0) -> list[Event]:
        evs = []
        zone, say = self.places.at(k.box)
        moved = o.zone is not None and zone != o.zone
        if o.state != VISIBLE:
            if o.state in (CARRIED, LAST_SEEN) or (o.state == HIDDEN and moved):
                evs.append(self._event(o.name, "FOUND", wall))
            if o.state == UNKNOWN or moved or o.state in (CARRIED, LAST_SEEN):
                o.arrived_wall = wall
                o.arrival_observed = o.state != UNKNOWN
            o.state, o.since_wall = VISIBLE, wall
        elif moved:                             # carried and put down between two looks at it
            evs.append(self._event(o.name, "MOVED", wall))
            o.arrived_wall, o.arrival_observed = wall, True
        o.box, o.zone, o.say, o.seen_wall, o.misses, o.how = k.box, zone, say, wall, 0, how
        if how == "look" and k.emb is not None and sim >= self.c.learn_sim \
                and wall - o.learned_wall >= self.c.learn_every_s and self._frame is not None:
            if o.bank.add(k.emb):
                o.refs = (o.refs + [ref_patch(self._frame, k.box)])[-self.c.refs_max:]
            o.learned_wall = wall
        return evs

    def _missed(self, o: RegObject, wall: float, people) -> list[Event]:
        """A valid view of o's last spot without o in it."""
        if o.state not in (VISIBLE, HIDDEN):
            return []
        if any(_covers(p, o.box) >= self.c.hidden_overlap for p in people):
            if o.state != HIDDEN:
                o.state, o.since_wall = HIDDEN, wall
            o.contact_wall = wall
            return []
        if o.state == HIDDEN:                   # the person moved away and it's gone: they took it
            o.state, o.since_wall, o.misses = CARRIED, wall, 0
            return [self._event(o.name, "PICKED_UP", wall)]
        o.misses += 1
        if o.misses < self.c.miss_visits or wall - (o.seen_wall or wall) < self.c.miss_s:
            return []
        o.misses = 0
        if o.contact_wall is not None and o.contact_wall >= (o.seen_wall or 0.0) - self.c.contact_s:
            o.state, o.since_wall = CARRIED, wall
            return [self._event(o.name, "PICKED_UP", wall)]
        o.state, o.since_wall = LAST_SEEN, wall
        return [self._event(o.name, "LOST_TRACK", wall)]

    # ----- Grok: which of these marked candidates is the object (set-of-marks, spec 0010 section 4)

    def _can_ask(self, wall: float) -> bool:
        self._ask_times = [t for t in self._ask_times if wall - t < 60.0]
        return len(self._ask_times) < self.c.verify_per_minute and self.online()

    def _claimed(self) -> list[BoxPx]:
        return [o.box for o in self.objects.values() if o.box is not None and o.state in (VISIBLE, HIDDEN)]

    def suspects(self, o: RegObject, wall: float) -> list[tuple[int, Candidate]]:
        """Where a missing object may be now, best first: candidates that appeared after it went (a carried
        object turns up somewhere new), else, for one never seen or once backstop_s is due, the candidates that
        look most like it. Candidates another registered object holds are never suspects."""
        claimed = self._claimed()
        o.rejected = [(b, t) for b, t in o.rejected if wall - t < self.c.reask_s]
        pool = [(i, k) for i, ks in self._cands.items() for k in ks
                if not any(geom.iou(k.box, b) >= 0.5 for b in claimed)
                and not any(geom.iou(k.box, b) >= 0.7 for b, _ in o.rejected)]
        if o.state in (CARRIED, LAST_SEEN) and o.seen_wall is not None:
            new = [(i, k) for i, k in pool if k.first_wall >= o.seen_wall - self.c.arrival_s]
            if new:
                return sorted(new, key=lambda q: (q[1].in_person, -o.bank.sim(q[1].emb)))
        backstop = self.c.backstop_s > 0 and wall - o.asked_wall >= self.c.backstop_s and \
            (o.state == UNKNOWN or wall - (o.since_wall or wall) >= self.c.backstop_s)
        if not backstop and o.name not in self._backstop_now:
            return []
        good = [(i, k) for i, k in pool if o.bank.sim(k.emb) >= self.c.sim_backstop]
        return sorted(good, key=lambda q: (q[1].in_person, -o.bank.sim(q[1].emb)))

    def _maybe_ask(self, img: np.ndarray, wall: float) -> None:
        if self.refind is None:
            return
        grounding = not getattr(self.refind, "needs_suspects", True)
        views = self.views(img.shape[1], img.shape[0])
        for o in self.objects.values():
            if o.state in (VISIBLE, HIDDEN) or o.asking or not o.refs or wall - o.asked_wall < self.c.ask_every_s:
                continue
            if not self._can_ask(wall):
                return
            sus = self.suspects(o, wall)
            if not sus and not grounding:
                continue
            if sus:
                view = sus[0][0]
            else:                               # a grounding backend searches the views in turn
                view, o.search_view = o.search_view % len(views), o.search_view + 1
            ks = [k for i, k in sus if i == view][:self.c.marks]
            x1, y1, x2, y2 = views[view]
            try:
                self._asks.put_nowait((o.name, o.refs[-2:], img[y1:y2, x1:x2].copy(), views[view], ks))
            except queue.Full:
                return
            o.asking, o.asked_wall = True, wall
            self._backstop_now.discard(o.name)
            self._ask_times.append(wall)
            self._ensure_worker()

    def request(self, name: str) -> None:
        """Someone asked where name is: if it isn't visible, ask Grok about its best matches soon."""
        with self.lock:
            if name in self.objects:
                self._backstop_now.add(name)
                self.objects[name].asked_wall = min(self.objects[name].asked_wall, 0.0)

    def _ensure_worker(self) -> None:
        if self._worker is None or not self._worker.is_alive():
            self._worker = threading.Thread(target=self._work, name="permanence-verify", daemon=True)
            self._worker.start()

    def _work(self) -> None:
        while not self._stop.is_set():
            try:
                name, refs, view_img, view, ks = self._asks.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                r = list(self.refind(self.names(name), refs, view_img, view, [k.box for k in ks]) or [])
            except Exception:
                log.exception("permanence: re-find failed")
                r = []
            self._results.put((name, ks, r))

    def _apply_results(self, wall: float) -> list[Event]:
        evs = []
        while True:
            try:
                name, ks, found = self._results.get_nowait()
            except queue.Empty:
                return evs
            with self.lock:
                o = self.objects.get(name)
                if o is None:
                    continue
                o.asking = False
                if o.state in (VISIBLE, HIDDEN):
                    continue
                if not any(float(c) >= self.c.verify_conf for _, c, _ in found):   # none of them: don't re-ask
                    o.rejected += [(k.box, wall) for k in ks]
                claimed = self._claimed()
                for box, conf, source in sorted(found, key=lambda f: -float(f[1])):
                    if float(conf) < self.c.verify_conf or any(geom.iou(box, b) >= 0.5 for b in claimed):
                        continue
                    k = self._candidate_at(box)
                    if k is None and source != "marks" and self._frame is not None:   # grounding: a box of its own
                        emb = (self.embed(self._frame, [tuple(int(v) for v in box)]) or [None])[0]
                        k = Candidate(tuple(int(v) for v in box), float(conf), source, emb=emb, first_wall=wall)
                    if k is None:
                        continue
                    evs += self._seen(o, k, wall, [], how=f"{REFIND}:{source}")
                    if k.emb is not None and o.bank.add(k.emb):    # this spot's look: refreshes need no re-find
                        if self._frame is not None:
                            o.refs = (o.refs + [ref_patch(self._frame, k.box)])[-self.c.refs_max:]
                    break

    def _candidate_at(self, box) -> Optional[Candidate]:
        """The latest views' candidate at box (IoU 0.5), or None: it has moved on since the question."""
        best = max((c for cs in self._cands.values() for c in cs), key=lambda c: geom.iou(box, c.box), default=None)
        return best if best is not None and geom.iou(box, best.box) >= 0.5 else None

    def _still_there(self, k: Candidate) -> bool:
        """The candidate Grok picked is still proposed where it was (in the latest views)."""
        return any(geom.iou(k.box, c.box) >= 0.5 for cs in self._cands.values() for c in cs)

    def stop(self) -> None:
        self._stop.set()

    # ----- what the world and answers read

    def fresh(self, o: RegObject, now: float) -> bool:
        return o.state == VISIBLE and o.seen_wall is not None and now - o.seen_wall <= self.c.fresh_s

    def place(self, name: str, now: Optional[float] = None) -> Optional["RegPlace"]:
        now = self.clock() if now is None else now
        with self.lock:
            o = self.objects.get(name)
            if o is None or o.say is None:
                return None
            status = Status.VISIBLE if o.state in (VISIBLE, HIDDEN) else \
                Status.HELD if o.state == CARRIED else Status.UNKNOWN
            return RegPlace(kind="room", zone=o.zone or "room", say=o.say, status=status, chain=[name], via=name,
                            box_px=o.box, fresh=self.fresh(o, now) or o.state == HIDDEN,
                            absent=o.state == LAST_SEEN, arrived_wall=o.arrived_wall, last_seen_wall=o.seen_wall,
                            arrival_observed=o.arrival_observed, tentative=o.how.startswith(REFIND), state=o.state,
                            since_wall=o.since_wall)

    def status(self) -> dict:
        with self.lock:
            return {"mode": self.c.mode, "objects": len(self.objects), "views": len(self._views),
                    "last_ms": round(self.last_ms, 1), "sweep_wall": self.sweep_wall,
                    "states": {n: o.state for n, o in self.objects.items()}}

    def snapshot(self, now: Optional[float] = None) -> dict:
        """name -> the registry side of /state."""
        now = self.clock() if now is None else now
        with self.lock:
            return {n: {"state": o.state, "zone": o.zone, "say": o.say, "box_px": list(o.box) if o.box else None,
                        "seen_wall": o.seen_wall, "since_wall": o.since_wall, "fresh": self.fresh(o, now),
                        "arrival_observed": o.arrival_observed,
                        "refs": len(o.bank.vecs)} for n, o in self.objects.items()}


@dataclass
class RegPlace(Place):
    state: str = UNKNOWN                    # the registry state (answers speak hidden and carried)
    since_wall: Optional[float] = None


def _inside(box: BoxPx, view: BoxPx, frac: float = 0.7) -> bool:
    """box lies (mostly) inside view: a valid look at it."""
    return geom.overlap_frac(view, box) >= frac


def _nms(cands: list[Candidate], thr: float = 0.6) -> list[Candidate]:
    out: list[Candidate] = []
    for k in sorted(cands, key=lambda k: -k.conf):
        if all(geom.iou(k.box, o.box) < thr and geom.overlap_frac(o.box, k.box) < 0.85 for o in out):
            out.append(k)
    return out


def marked_view(img: np.ndarray, view: BoxPx, boxes: Sequence[BoxPx], out_px: int = 1280) -> np.ndarray:
    """The view, native, with each candidate in a red box numbered from 1 (0010: a marked object in context,
    and a closed question, is what Grok gets right)."""
    x1, y1, x2, y2 = view
    out = img[y1:y2, x1:x2].copy()
    k = min(1.0, out_px / max(out.shape[:2]))
    if k < 1:
        out = cv2.resize(out, (round(out.shape[1] * k), round(out.shape[0] * k)), interpolation=cv2.INTER_AREA)
    return _mark(out, [((b[0] - x1) * k, (b[1] - y1) * k, (b[2] - x1) * k, (b[3] - y1) * k) for b in boxes])


def _mark(img: np.ndarray, boxes) -> np.ndarray:
    """Red boxes numbered from 1. Each number sits beside its own box, at the first spot (above, left, right,
    below) that overlaps no other box or number: a number between two boxes is misread (rig, Sun 27 Sep)."""
    h, w = img.shape[:2]
    boxes = [tuple(int(v) for v in b) for b in boxes]
    for b in boxes:
        cv2.rectangle(img, b[:2], b[2:], (0, 0, 255), 2)
    taken: list = []
    for n, b in enumerate(boxes, 1):
        (tw, th), _ = cv2.getTextSize(str(n), cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        lw, lh = tw + 6, th + 8
        spots = [(b[0], b[1] - lh - 2), (b[0] - lw - 2, b[1]), (b[2] + 2, b[1]), (b[0], b[3] + 2)]
        others = [q for q in boxes if q != b] + taken
        ok = [(x, y) for x, y in spots if 0 <= x and x + lw <= w and 0 <= y and y + lh <= h]
        free = [(x, y) for x, y in ok if not any(geom.intersection((x, y, x + lw, y + lh), q) for q in others)]
        x, y = (free or ok or [(max(0, b[0]), max(0, b[1]))])[0]
        taken.append((x, y, x + lw, y + lh))
        cv2.rectangle(img, (x, y), (x + lw, y + lh), (255, 255, 255), -1)
        cv2.putText(img, str(n), (x + 3, y + th + 3), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    return img


def question_image(view_img: np.ndarray, boxes: Sequence[BoxPx], min_px: int = 640, max_px: int = 1280,
                   margin_px: int = 160) -> tuple[np.ndarray, list]:
    """The part of a view around the suspects (view px), enlarged so small objects are big enough to judge
    (up to 3x), with the suspects numbered; returns (image, the boxes in its px)."""
    h, w = view_img.shape[:2]
    xs1, ys1 = min(b[0] for b in boxes), min(b[1] for b in boxes)
    xs2, ys2 = max(b[2] for b in boxes), max(b[3] for b in boxes)
    m = max(margin_px, 0.5 * max(xs2 - xs1, ys2 - ys1))
    a, b0 = int(max(0, xs1 - m)), int(max(0, ys1 - m))
    c, d = int(min(w, xs2 + m)), int(min(h, ys2 + m))
    crop = view_img[b0:d, a:c]
    k = min(3.0, max(1.0, min_px / max(1, min(crop.shape[:2]))), max_px / max(crop.shape[:2]))
    out = cv2.resize(crop, (round(crop.shape[1] * k), round(crop.shape[0] * k)),
                     interpolation=cv2.INTER_CUBIC if k > 1 else cv2.INTER_AREA)
    local = [((q[0] - a) * k, (q[1] - b0) * k, (q[2] - a) * k, (q[3] - b0) * k) for q in boxes]
    return _mark(out, local), local


def distinct(boxes: Sequence[BoxPx], thr: float = 0.3) -> list[int]:
    """Indices of boxes that don't overlap an earlier (better) one: two boxes on one spot confuse the marks."""
    keep: list[int] = []
    for i, b in enumerate(boxes):
        if all(geom.iou(b, boxes[j]) < thr and geom.overlap_frac(boxes[j], b) < 0.6
               and geom.overlap_frac(b, boxes[j]) < 0.6 for j in keep):
            keep.append(i)
    return keep


# ================================================================================ the world adapter

TABLE_HAS = (Status.VISIBLE, Status.HELD, Status.UNDER, Status.INSIDE)


def attach(p: Permanence, world) -> Permanence:
    """Wrap world.place and world.state_json with the registry (world.get stays: the World calls it itself).
    The table world's answer wins while the table has the object."""
    place = getattr(world, "place", None)
    state = world.state_json

    def table_has(name: str, now: float) -> bool:
        try:
            e = world.get(name)
        except Exception:
            return False
        return getattr(e, "zone", "table") == "table" and e.status in TABLE_HAS and e.last_seen is not None \
            and now - e.last_seen <= p.c.table_fresh_s

    def reg_place(name: str, now: Optional[float] = None):
        now = p.clock() if now is None else now
        rp = p.place(name, now)
        if rp is None or table_has(name, now):
            return place(name, now) if callable(place) else None
        if rp.state != VISIBLE:
            p.request(name)
        return rp

    def state_json(*a, **kw):
        st = state(*a, **kw)
        try:
            now = p.clock()
            snap = p.snapshot(now)
            room = st.get("room")
            if not isinstance(room, dict):
                room = st["room"] = {}
            ents = []
            for e in st.get("entities") or []:
                n = e.get("name") if isinstance(e, dict) else None
                r = snap.get(n)
                if r is not None and r["state"] != UNKNOWN and not table_has(n, now):
                    e["zone"] = r["zone"] if r["zone"] == "table" else (r["zone"] or "room")
                    e["status"] = {VISIBLE: "VISIBLE", HIDDEN: "VISIBLE", CARRIED: "HELD"}.get(r["state"], "UNKNOWN")
                    if r["zone"] != "table":
                        e["pos_cm"] = e["resolved_cm"] = None
                    room[n] = {"zone": r["zone"], "say": r["say"], "box_px": r["box_px"], "seen_wall": r["seen_wall"],
                               "absent": r["state"] == LAST_SEEN, "arrival_observed": r["arrival_observed"]}
                if r is not None:
                    e["registry"] = r
                if p.c.hide_unnamed and n and str(n).startswith("thing:") and r is None \
                        and not e.get("label") and e.get("status") != "VISIBLE":
                    continue
                ents.append(e)
            st["entities"] = ents
            st["permanence"] = p.status()
        except Exception:
            log.exception("permanence: state failed")
        return st

    world.place = reg_place
    world.state_json = state_json
    world.permanence = p
    return p


# ================================================================================ construction

def yoloe_detect(model, imgsz: int = 640, conf: float = 0.15) -> Callable:
    """detect(img) -> [(class name, conf, box)] from a YOLOE prompt-free model (ultralytics or
    core.yoloe_fast.ReducedYOLOE: the same predict interface as core/proposals.py uses)."""
    def detect(img: np.ndarray) -> list:
        r = model.predict(img, imgsz=imgsz, conf=conf, iou=0.5, agnostic_nms=True, max_det=150, verbose=False)[0]
        names = getattr(r, "names", None) or getattr(model, "names", {}) or {}
        xyxy = r.boxes.xyxy.cpu().numpy() if hasattr(r.boxes.xyxy, "cpu") else np.asarray(r.boxes.xyxy)
        cf = r.boxes.conf.cpu().numpy() if hasattr(r.boxes.conf, "cpu") else np.asarray(r.boxes.conf)
        cl = r.boxes.cls.cpu().numpy() if hasattr(r.boxes.cls, "cpu") else np.asarray(r.boxes.cls)
        return [(str(names.get(int(k), "")), float(s), tuple(float(v) for v in b)) for b, s, k in zip(xyxy, cf, cl)]
    return detect


def registry_embedder(cfg: dict):
    """The reid: embedder with enabled forced on, for the registry only (reid.enabled stays as it is, so
    the table world's things are unchanged). Returns embed(img, boxes) -> list, or None without the model."""
    from core.embed import Embedder, ReidConfig, _path
    rc = ReidConfig.from_dict({**((cfg or {}).get("reid") or {}), "enabled": True, "background": False})
    if not _path(rc.model).exists():
        log.warning("permanence: %s not found; the registry needs the re-ID model", _path(rc.model))
        return None
    e = Embedder(rc)
    e.load()
    return e.batch


def marks_refinder(ask: Callable, confirm: Optional[Callable] = None, confirm_conf: float = 0.8) -> Callable:
    """A re-find backend from set-of-marks questions: ask(name, refs, marked image, n) -> (mark, confidence),
    mark 0 = none; then, with confirm(name, refs, patch) -> (same, confidence), the pick must be confirmed by a
    closed same-object question (Grok leans to yes: 0010 section 3). Only suspects can be picked."""
    def refind(name: str, refs: list, view_img: np.ndarray, view: BoxPx, suspects: list):
        keep = distinct(suspects)
        if not keep:
            return []
        x1, y1 = view[0], view[1]
        local = [(suspects[i][0] - x1, suspects[i][1] - y1, suspects[i][2] - x1, suspects[i][3] - y1) for i in keep]
        img, _ = question_image(view_img, local)
        r = ask(name, refs, img, len(local))
        if not r or not 1 <= int(r[0]) <= len(local):
            return []
        pick = keep[int(r[0]) - 1]
        conf = float(r[1])
        if confirm is not None:
            lb = local[int(r[0]) - 1]
            patch, _ = question_image(view_img, [lb], min_px=320, max_px=640, margin_px=60)
            c = confirm(name, refs, patch)
            if not c or not c[0] or float(c[1]) < confirm_conf:
                return []
            conf = min(conf, float(c[1]))
        return [(suspects[pick], conf, "marks")]
    refind.needs_suspects = True
    return refind


VERIFY_SYSTEM = ("You find one person's object in a room photo. The first image(s) show the owner's object. "
                 "The last image is part of the room with numbered red boxes. Say which box holds that same "
                 "object: the same kind, colours, shape and markings. A different object of the same kind is not "
                 "it. Often none of the boxes is the object: then mark is 0. Never guess.")
VERIFY_SCHEMA = {"type": "object", "additionalProperties": False, "required": ["mark", "confidence"],
                 "properties": {"mark": {"type": "integer"}, "confidence": {"type": "number"}}}
SAME_SYSTEM = ("You compare photos from one room camera. The first image(s) show a person's object in a red box. "
               "The last image shows an object in a red box, maybe from another angle, distance or light. Is it "
               "the same object: the same kind, colour, shape and markings? A different object of the same kind "
               "(another bottle, another box) is same=false.")
SAME_SCHEMA = {"type": "object", "additionalProperties": False, "required": ["same", "confidence", "what"],
               "properties": {"same": {"type": "boolean"}, "confidence": {"type": "number"},
                              "what": {"type": "string"}}}


def _provider(cfg: dict):
    from core.narration import NarrationConfig, make_provider
    from core.visual_memory import VisualConfig
    v = VisualConfig.from_dict((cfg or {}).get("visual_memory"))
    return make_provider(NarrationConfig.from_dict({
        "provider": v.provider, "model": v.model, "base_url": v.base_url, "api_key_env": v.api_key_env,
        "reasoning_effort": v.reasoning_effort, "timeout_s": v.timeout_s, "max_tokens": 120}))


def _jpg(img: np.ndarray) -> bytes:
    return cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])[1].tobytes()


def _ref_parts(name: str, refs: list) -> list:
    parts: list = []
    for j, ref in enumerate(refs, 1):
        big = cv2.resize(ref, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC) if max(ref.shape[:2]) < 200 else ref
        parts += [("text", f"Image {j}: the owner's {name} (in the red box)."), ("image", _jpg(big))]
    return parts


def grok_marks(cfg: dict, timeout_s: float = 8.0) -> Optional[Callable]:
    """ask(name, refs, marked image, n) -> (mark, confidence), through the visual_memory provider (Grok);
    mark 0 is none of them."""
    try:
        from core.narration import _parse_json
        from net import call_with_deadline
        provider = _provider(cfg)
    except Exception:
        log.exception("permanence: no Grok provider; re-find off")
        return None

    def ask(name: str, refs: list, marked: np.ndarray, n: int):
        parts = _ref_parts(name, refs) + [
            ("text", f"Image {len(refs) + 1}: part of the room, boxes 1 to {n}. Which box is the owner's {name}? "
                     f"0 if none."), ("image", _jpg(marked))]
        reply = call_with_deadline(provider.narrate, timeout_s, VERIFY_SYSTEM, parts, VERIFY_SCHEMA,
                                   name="permanence-refind")
        d = _parse_json(reply.text) or {}
        return int(d.get("mark") or 0), float(d.get("confidence") or 0.0)
    return ask


def grok_same(cfg: dict, timeout_s: float = 8.0) -> Optional[Callable]:
    """confirm(name, refs, patch) -> (same, confidence): the closed question after a mark is picked."""
    try:
        from core.narration import _parse_json
        from net import call_with_deadline
        provider = _provider(cfg)
    except Exception:
        log.exception("permanence: no Grok provider; no confirmation")
        return None

    def confirm(name: str, refs: list, patch: np.ndarray):
        parts = _ref_parts(name, refs) + [
            ("text", f"Image {len(refs) + 1}: is the object in the red box the owner's {name}?"), ("image", _jpg(patch))]
        reply = call_with_deadline(provider.narrate, timeout_s, SAME_SYSTEM, parts, SAME_SCHEMA,
                                   name="permanence-confirm")
        d = _parse_json(reply.text) or {}
        return bool(d.get("same")), float(d.get("confidence") or 0.0)
    return confirm


def make_refinder(cfg: dict, c: PermanenceConfig) -> Optional[Callable]:
    """The re-find backend permanence.refind names: 'grok_marks' (default), or off with verify: false."""
    if not c.verify:
        return None
    if c.refind == "grok_marks":
        ask = grok_marks(cfg, c.verify_timeout_s)
        confirm = grok_same(cfg, c.verify_timeout_s) if c.confirm else None
        return marks_refinder(ask, confirm, c.confirm_conf) if ask is not None else None
    log.warning("permanence: unknown refind backend %r; re-find off", c.refind)
    return None


def make_permanence(cfg: dict, world=None, detector=None, events=None, online=None) -> Optional[Permanence]:
    """The registry the config asks for, attached to the world, or None (mode off, or no model)."""
    c = PermanenceConfig.from_dict((cfg or {}).get("permanence"))
    if c.mode != "registry":
        return None
    embed = registry_embedder(cfg)
    if embed is None:
        return None
    proposer = getattr(detector, "proposer", None)
    model = getattr(proposer, "model", None)
    if model is None:
        from core.proposals import YOLOEConfig
        from ultralytics import YOLO
        model = YOLO(YOLOEConfig.from_dict(((cfg or {}).get("proposals") or {}).get("yoloe")).model)
    places = load_places(cfg, c)
    zoom = [r_box(r) for r in places.regions] if c.zoom == "zones" else list(c.zoom or [])
    display = (cfg or {}).get("display_names") or {}
    p = Permanence(c, yoloe_detect(model, c.imgsz, c.conf), embed, events=events, places=places,
                   refind=make_refinder(cfg, c),
                   names=lambda n: display.get(n, n.replace("_", " ")), zoom_boxes=zoom,
                   online=online or (lambda: bool(getattr(world, "online", True))))
    for n in (c.objects if c.objects is not None else list((cfg or {}).get("objects") or {})):
        p.register(n)
    log.info("permanence: registry of %d objects, %d reference views", len(p.objects), p.enroll_dir())
    if world is not None:
        attach(p, world)
    return p


def r_box(r: Region) -> BoxPx:
    xs, ys = [q[0] for q in r.poly], [q[1] for q in r.poly]
    return (int(min(xs)), int(min(ys)), int(math.ceil(max(xs))), int(math.ceil(max(ys))))


def table_to_full(box, rect: Sequence[int], out: tuple = (1280, 720)) -> BoxPx:
    """A table-view px box (the 1280x720 cut) in full-frame px."""
    x1, y1, x2, y2 = rect
    sx, sy = (x2 - x1) / out[0], (y2 - y1) / out[1]
    return (x1 + box[0] * sx, y1 + box[1] * sy, x1 + box[2] * sx, y1 + box[3] * sy)


def hook_teach(p: Permanence, world, frames, table, rect) -> None:
    """After 'this is my X' binds a thing (world.teach, unchanged), register that thing with its box cut from
    the native full frame, so it is re-found anywhere from then on."""
    teach = getattr(world, "teach", None)
    if not callable(teach):
        return

    def teach_and_register(name, *a, **kw):
        thing = teach(name, *a, **kw)
        try:
            box = full_box(world.get(thing), table, rect) if thing else None
            full = frames.latest_full() if box is not None else None
            if full is not None and not p.teach(thing, full.img, box):
                log.info("permanence: %s taught but not registered (no appearance)", thing)
        except Exception:
            log.exception("permanence: registering %s failed", thing)
        return thing

    world.teach = teach_and_register


def full_box(e, table, rect) -> Optional[BoxPx]:
    """An entity's table-cm box in full-frame px (table cm -> table-view px -> full frame), or None."""
    b = getattr(e, "box_cm", None)
    if b is None or table is None or not getattr(table, "ok", False) or rect is None:
        return None
    pts = np.asarray(table.cm_to_px([(b[0], b[1]), (b[2], b[3]), (b[0], b[3]), (b[2], b[1])]), float).reshape(-1, 2)
    xs, ys = pts[:, 0].tolist(), pts[:, 1].tolist()
    return tuple(int(round(v)) for v in table_to_full((min(xs), min(ys), max(xs), max(ys)), rect))


# ================================================================================ command line

def main(argv: Optional[list] = None) -> int:
    ap = argparse.ArgumentParser(description="Object permanence registry (spec 0011)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    en = sub.add_parser("enroll", help="save a reference crop of an object from a raw full frame")
    en.add_argument("--name", required=True)
    en.add_argument("--image", required=True)
    en.add_argument("--box", type=float, nargs=4, required=True, metavar=("X1", "Y1", "X2", "Y2"))
    en.add_argument("--dir", default=None)
    a = ap.parse_args(argv)
    from core.config import load_config
    c = PermanenceConfig.from_dict(load_config().get("permanence"))
    img = cv2.imread(a.image)
    if img is None:
        ap.error(f"can't read {a.image}")
    crop = _crop(img, a.box, 0.0)
    d = Path(a.dir or c.refs_dir)
    d = (d if d.is_absolute() else ROOT / d) / a.name
    d.mkdir(parents=True, exist_ok=True)
    out = d / f"{Path(a.image).stem}_{int(a.box[0])}_{int(a.box[1])}.jpg"
    cv2.imwrite(str(out), crop)
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
