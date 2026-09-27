"""Open-world objects for the world model. Class-agnostic proposals (Detection.cls == 'thing') become
dynamic entities thing:1, thing:2, ... that then follow exactly the same rules as the configured
objects (debounce, hand pick-up, cover, container dwell, lifted cover, decay). This module decides
only WHICH entity a proposal is; world.py still decides what happened to it.

Lifecycle: candidate -> confirmed (APPEARED) -> visible / hidden (HELD, UNDER, INSIDE by the world's
rules) -> exited (GONE) or lost (UNKNOWN) -> archived. Entities are never deleted; a merged identity
keeps its entity with merged_into set.

Identity is causal first; appearance only confirms. Each proposal goes through ordered rules, not a
weighted score, so every decision can be explained in one sentence:
  per batch, while the thing is in view or in a hand (continuity):
    (b)    a visible thing: the nearest proposal within gate_cm, of a size like its latest box (or,
           while an arm hides part of it, over the box it rests in: the whole of it again)
    (b)    a HELD thing: a proposal at its holding hand's recent boxes
    merge  one proposal covering two visible things' last boxes: those not seen on their own are kept
           where they were, at ambiguity_penalty, until they come apart (then nearest, or appearance if
           decisive); the blob itself is never a candidate
  otherwise the proposal is a candidate; after the k-of-n debounce, clear of hands, and not lying on a
  thing in view (the same box, or the same spot at a like size: that thing seen twice):
    (a)    causal: it came out of a container / cover that holds hidden things (a hand touched that
           parent since they hid), or appeared where an UNDER thing lay: that child. Several
           children: appearance may pick one, else a new thing linked to all of them
    (b)    a thing lost in place (no hand involved) seen again at the same spot (the nearest, if several
           were lost there); with thing_identity:
           rebirth_s, also one lost recently after a pick-up, a false pick-up or an edge exit
    (c)    appearance resurrects an archived thing only above resurrect_sim AND by resurrect_margin
           over every other thing (a twin on the table blocks it)
    (d)    a new thing, with maybe_same_as links to archived things that look similar; only inside
           the tabletop outline clear of its edge band (table_area:, core/table_area.py; its centre or,
           for a tall thing at the far edge, where it stands), and never
           from a proposal flagged occluded (inside a person box). Either can still be an existing
           thing, so one slid off the table leaves as itself and a carried one stays itself
Exemplars (per-thing appearance banks) are learned only from isolated, confident, unambiguous
views, so an ambiguous association can never drift an identity.

The World class mixes this in; methods share its state (entities, _bits, _contacts, ...).
"""
from __future__ import annotations

import re
import time
from collections import deque
from dataclasses import dataclass, field, fields

import cv2
import numpy as np

from core import geom, relations
from core.table_area import TableArea
from core.types import Detection, Entity, EventType, Status

THING = 'thing'                    # Detection.cls of a class-agnostic proposal
PREFIX = 'thing:'
HIDDEN = (Status.INSIDE, Status.UNDER)
ARCHIVED = (Status.GONE, Status.UNKNOWN)
# Events after which a thing sits on the table where it was just put: what 'this is my X' refers to.
PLACED = {EventType.APPEARED, EventType.MOVED, EventType.PUT_BACK, EventType.TAKEN_OUT,
          EventType.UNCOVERED, EventType.CORRECTED}
OPEN_INSIDE = 0.8   # this much of a visible object's box within an open container's box: lying in it
STARTUP_S = 3.0     # configured objects first seen this soon after the first batch were not put down
ARTICLES = {'my', 'the', 'a', 'an', 'your', 'our', 'this', 'that', 'his', 'her', 'their'}
NEG = float('-inf')
SAME_SIZE = 2.0     # 'a like size' for the same-spot birth veto: areas within this factor
REST_IOU = 0.5      # a proposal this much over the box a partly hidden thing rests in: the whole of it
CROSS_UV = 1.0      # a hand in and out of a thing container on opposite sides of its middle, this far
                    # apart (in half-widths): it crossed the thing carrying something past, not into it


@dataclass
class ThingsConfig:
    """The openworld: section of config.yaml; every key is optional."""
    enabled: bool = True
    proposal_conf: float = 0.25        # proposals below this are ignored
    area_cm2: tuple[float, float] = (4.0, 900.0)    # plausible object size (the notebook is ~450)
    dup_iou: float = 0.5               # IoU with a known object / hand / kept proposal: a duplicate
    hand_contain: float = 0.8          # a proposal covering this much of a hand box is hand or arm
    new_hand_overlap_max: float = 0.1  # a candidate is decided only once hands cover less of it
    still_cm: float = 2.0              # ... and its centre stayed within this ...
    still_s: float = 0.5               # ... for this long: objects people put down stay put, an arm the
                                       # hand detector missed keeps moving (on the rig, every pass of a
                                       # hand left phantom things behind)
    gate_cm: float = 8.0               # visible thing: nearest proposal within this per batch
    size_ratio_max: float = 4.0        # ... and no more than this many times larger or smaller
    hand_gate_cm: float = 4.0          # HELD thing: proposal centre within this of its hand's box
    hand_trail_s: float = 2.0          # ... at any time in the last this many seconds
    emerge_cm: float = 12.0            # 'next to' a container / cover: within its box grown by this
    same_spot_cm: float = 5.0          # 'at the same spot' as an UNDER or lost-in-place thing
    tie_sim: float = 0.80              # appearance picking among causal / split candidates ...
    tie_margin: float = 0.05           # ... needs this lead over the runner-up
    resurrect_sim: float = 0.92        # appearance alone bringing back an archived thing ...
    resurrect_margin: float = 0.10     # ... needs this lead over every other thing
    maybe_sim: float = 0.70            # a new thing is linked (maybe_same_as) above this
    maybe_max: int = 3
    merge_cover: float = 0.6           # a proposal covering this much of 2+ things' boxes: merged
    member_iou: float = 0.6            # ... unless it is plainly one of them
    ambiguous_s: float = 2.0           # no exemplar learning this long after a merge / tie-break
    exemplar_max: int = 8
    exemplar_every_s: float = 1.0
    exemplar_min_px: int = 16
    exemplar_dup_sim: float = 0.97     # a view this close to a stored one adds nothing
    teach_recent_s: float = 30.0       # no teach zone: things put down within this long

    @classmethod
    def from_config(cls, cfg) -> 'ThingsConfig':
        raw = dict(getattr(cfg, 'openworld', None) or {})
        known = {f.name for f in fields(cls)}
        kw = {k: v for k, v in raw.items() if k in known}
        if 'area_cm2' in kw:
            kw['area_cm2'] = tuple(kw['area_cm2'])
        return cls(**kw)


@dataclass
class ContainerConfig:
    """The thing_containers: section of config.yaml (every key optional): which things may hold others."""
    enabled: bool = True
    min_area_cm2: float = 100.0        # footprint of a mug or larger: a tub, a bag, a hat; not keys, a coin

    @classmethod
    def from_config(cls, cfg) -> 'ContainerConfig':
        raw = dict(getattr(cfg, 'thing_containers', None) or {})
        return cls(**{k: v for k, v in raw.items() if k in {f.name for f in fields(cls)}})


@dataclass
class IdentityConfig:
    """The thing_identity: section of config.yaml (every key optional): one object, one thing:N. On the
    rig most new things were an old one back at its spot after a hand passed (31 of 39 births in 13 min
    were within 5 cm of an earlier thing), because only a thing lost in place with no hand came back."""
    rebirth_s: float = 0.0             # a thing lost this recently (UNKNOWN / GONE / HELD, hand or no hand)
                                       # seen again at its spot is that thing again; 0 = off (rule (b) only)
    rebirth_cm: float = 5.0            # 'at its spot': within this of where it was last seen or picked up

    @classmethod
    def from_config(cls, cfg) -> 'IdentityConfig':
        raw = dict(getattr(cfg, 'thing_identity', None) or {})
        return cls(**{k: v for k, v in raw.items() if k in {f.name for f in fields(cls)}})


@dataclass
class Candidate:
    """An unexplained proposal track, debounced exactly like an object's presence bits."""
    det: Detection
    bits: deque
    born: float = 0.0
    trail: deque = field(default_factory=lambda: deque(maxlen=64))   # (t, centre cm) while matched


class ExemplarBank:
    """Up to cap unit appearance vectors for one thing. A near-duplicate view is not added; when
    full, the most redundant stored view goes, so the bank stays diverse."""

    def __init__(self, cap: int, dup_sim: float):
        self.cap, self.dup_sim = cap, dup_sim
        self.vecs: list[np.ndarray] = []

    def sim(self, v: np.ndarray | None) -> float:
        if v is None or not self.vecs:
            return NEG
        return max(float(e @ v) for e in self.vecs)

    def add(self, v: np.ndarray) -> bool:
        if self.sim(v) >= self.dup_sim:
            return False
        self.vecs.append(v)
        if len(self.vecs) > self.cap:
            m = np.stack(self.vecs) @ np.stack(self.vecs).T
            np.fill_diagonal(m, NEG)
            del self.vecs[int(np.argmax(m.max(axis=1)))]
        return True

    def extend(self, other: 'ExemplarBank') -> None:
        for v in other.vecs:
            self.add(v)


# ----- appearance: an offline stand-in ------------------------------------------------------------

def hsv_embed(img, box_px) -> np.ndarray | None:
    """STAND-IN for a CLIP / DINO crop embedding: an HSV colour histogram plus a coarse 2x2 colour
    layout, as one unit vector (cosine = dot product). Cheap (~0.1 ms per crop) and offline, but it
    only knows colour: two different mugs of one colour look the same to it, which is why appearance
    never overrides causal or spatial evidence. None without an image or for a crop under 8 px."""
    if img is None or img.ndim != 3:
        return None
    h, w = img.shape[:2]
    x1, y1 = max(0, int(box_px[0])), max(0, int(box_px[1]))
    x2, y2 = min(w, int(box_px[2])), min(h, int(box_px[3]))
    if x2 - x1 < 8 or y2 - y1 < 8:
        return None
    crop = img[y1:y2, x1:x2]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hs = np.sqrt(cv2.calcHist([hsv], [0, 1], None, [12, 4], [0, 180, 0, 256]).ravel())
    v = np.sqrt(cv2.calcHist([hsv], [2], None, [4], [0, 256]).ravel())
    layout = cv2.resize(crop, (2, 2), interpolation=cv2.INTER_AREA).astype(np.float32).ravel()
    parts = [(hs, 0.6), (v, 0.2), (layout, 0.2)]
    vec = np.concatenate([p / (np.linalg.norm(p) or 1.0) * wt for p, wt in parts])
    return vec / np.linalg.norm(vec)


# ----- names -------------------------------------------------------------------------------------

def norm_name(text: str | None) -> str:
    """'My Charger!' -> 'charger': lowercase, no apostrophes or punctuation, no leading articles."""
    t = re.sub(r"['’`]", '', (text or '').lower())
    words = re.sub(r'[^a-z0-9\s]', ' ', t).split()
    while words and words[0] in ARTICLES:
        words.pop(0)
    return ' '.join(words)


def is_thing(name: str | None) -> bool:
    return bool(name) and name.startswith(PREFIX)


def _singular(key: str) -> str:
    return key[:-1] if len(key) > 3 and key.endswith('s') and not key.endswith('ss') else key


def _grow(box, d: float):
    return (box[0] - d, box[1] - d, box[2] + d, box[3] + d)


def _footprint(box):
    """Where a box stands on the table: the middle of its side nearest the camera (table y grows toward
    it). A tall thing at the far edge (a speaker, a paper bag, the pill bottle) has its centre past the
    edge while it stands on the table."""
    return ((box[0] + box[2]) / 2, box[3])


def _box_dist(p, box) -> float:
    """Distance from point p to box (0 inside)."""
    dx = max(box[0] - p[0], 0.0, p[0] - box[2])
    dy = max(box[1] - p[1], 0.0, p[1] - box[3])
    return float(np.hypot(dx, dy))


# ----- the mixin ----------------------------------------------------------------------------------

class ThingRules:
    """Open-world association, lifecycle, appearance banks and naming, mixed into World."""

    embed = None       # embed(frame_img, box_px) -> unit np.ndarray | None; set by World.__init__

    def _reset_things(self) -> None:
        self._tcfg = ThingsConfig.from_config(self.cfg)
        self._icfg = IdentityConfig.from_config(self.cfg)
        self._area = TableArea.from_dict(getattr(self.cfg, 'table_area', None))   # tabletop outline, or none
        # Never reuse an id: the event log outlives RESET and the process, and a new thing:N would
        # inherit an old thing:N's history. Duck-typed sinks without max_number start at 0.
        logged = getattr(self.events, 'max_number', None)
        self._thing_n = max(getattr(self, '_thing_n', 0), logged(PREFIX) if logged else 0)
        self._things: list[str] = []                   # every thing ever confirmed, oldest first
        self._born: dict[str, float] = {}              # thing -> wall time it was confirmed
        self._alias_by: dict[str, str] = {}             # alias -> who bound it ('grok'), absent = taught
        self._vetoes: list[tuple] = []                  # (box cm, until wall, margin cm): retired clutter
        self._cands: list[Candidate] = []
        self._banks: dict[str, ExemplarBank] = {}
        self._aliases: dict[str, str] = {}              # normalised alias -> thing
        self._hidden_at: dict[str, float] = {}          # thing -> when it became INSIDE / UNDER
        self._placed_t: dict[str, float] = {}           # thing or configured object -> when last put down in view
        self._known_seen: set[str] = set()              # configured objects ever seen visible
        self._t_start: float | None = None              # the first batch: the scene then was not put down
        self._rest_pos: dict[str, tuple] = {}           # entity -> where it was last seen visible
        self._merged: dict[str, frozenset] = {}         # thing -> things sharing its merged proposal
        self._unsure_until: dict[str, float] = {}       # thing -> no exemplar learning before this
        self._learn_t: dict[str, float] = {}
        self._clean: dict[str, Detection] = {}          # this batch's unambiguous associations
        self._trail: dict[str, deque] = {}              # hand -> recent (t, box_cm)
        self._batch_boxes: list[tuple] = []             # this batch's object / proposal / hand boxes
        self._ccfg = ContainerConfig.from_config(self.cfg)
        self._in_thing: dict[tuple, list] = {}          # (hand, thing container) -> [entry t, entry uv, last uv]
        self._thing_visits: dict[tuple, tuple] = {}     # (hand, thing, exit t) -> (entry t, crossed it)
        self._known_names: dict[str, str] = {}
        for n in self.cfg.names():
            self._known_names[norm_name(n)] = n
            self._known_names[norm_name(n.replace('_', ' '))] = n
        for spoken, n in (self.cfg.synonyms or {}).items():
            if n in self.entities:
                self._known_names.setdefault(norm_name(str(spoken)), n)

    # ----- public API -----------------------------------------------------------------------------

    def find(self, name: str) -> str | None:
        """Entity for a spoken name: a configured object, synonym or taught alias ('my charger',
        'The Charger', 'chargers'), or an entity name. A merged identity resolves to the survivor."""
        with self.lock:
            if name in self.entities:
                return self._survivor(name)
            key = norm_name(name)
            for k in dict.fromkeys((key, _singular(key))):
                hit = self._known_names.get(k) or self._aliases.get(k)
                if hit:
                    return self._survivor(hit)
            return None

    def open_container_of(self, name: str) -> str | None:
        """The container a VISIBLE object is seen lying in: a visible configured container, or a thing
        that may hold others (thing_containers), whose box holds at least OPEN_INSIDE of the object's
        box. From overhead an open box shows its contents, so they never go INSIDE; answers say 'in'."""
        with self.lock:
            ent = self.entities.get(name)
            if ent is None or ent.status != Status.VISIBLE or ent.box_cm is None:
                return None
            boxes = {n: self.entities[n].box_cm for n in self.cfg.names('container')
                     if n != name and self.entities[n].status == Status.VISIBLE and self.entities[n].box_cm}
            boxes.update({n: b for n, b in self._thing_containers().items() if n != name})
            inside = [(geom.area(b), n) for n, b in boxes.items()
                      if geom.area(b) > geom.area(ent.box_cm) and geom.overlap_frac(b, ent.box_cm) >= OPEN_INSIDE]
            return min(inside)[1] if inside else None

    def teach_target(self) -> str | None:
        """What 'this is my X' would name now (see teach), without binding anything."""
        with self.lock:
            return self._teach_target()

    def teach(self, name: str) -> str | None:
        """Bind a spoken name to exactly one entity by a fixed rule: the visible thing or configured
        object most recently APPEARED or put down inside teach_zone_cm (without a zone: put down within
        teach_recent_s), so 'this is my brown wallet' names the wallet the detector knows. None when
        nothing qualifies or the name already belongs to a configured object."""
        with self.lock:
            key = norm_name(name)
            if not key or self._is_known_name(key):
                return None
            target = self._teach_target()
            if target is not None:
                self._bind(target, key)
            return target

    def bind_alias(self, entity: str, name: str, by: str | None = None) -> bool:
        """Bind a name to a given thing (the 'yes, that's it' flow). A name keeps one owner: it
        moves here from any other thing. Configured objects' names are refused. by: who named it when
        not a person ('grok'); shown as named_by, so answers and the phone can hedge."""
        with self.lock:
            key, ent = norm_name(name), self._survivor(entity)
            if not key or self._is_known_name(key) or not is_thing(ent) or ent not in self.entities:
                return False
            self._bind(ent, key)
            if by:
                self._alias_by[key] = by
            return True

    def retire_thing(self, name: str, since: float, veto_s: float = 120.0, veto_cm: float = 5.0) -> bool:
        """Drop an unnamed thing judged not an object (Grok, over several settle checks: a cable, part of
        the desk). Only one lying VISIBLE on the table, without a taught name, holding nothing, with no hand
        contact since monotonic time `since` (the judged frame). It leaves the state (merged_into itself,
        history kept) and no new thing is born on its box, grown by veto_cm, for veto_s (a cable is long:
        a radius around its centre misses the rest of it). No event: nothing moved."""
        with self.lock:
            e = self.entities.get(name)
            if e is None or not is_thing(name) or e.merged_into is not None or e.status != Status.VISIBLE \
                    or e.pos_cm is None or any(self._alias_by.get(a) is None for a in e.aliases) \
                    or any(o.parent == name for o in self.entities.values()) \
                    or max(self._contacts.get(name, {}).values(), default=NEG) > since:
                return False
            now = self._wall if self._wall is not None else time.time()
            box = tuple(e.box_cm) if e.box_cm is not None else (*e.pos_cm, *e.pos_cm)
            self._vetoes = [v for v in self._vetoes if v[1] > now] + [(box, now + veto_s, veto_cm)]
            for a in list(e.aliases):
                self._aliases.pop(a, None)
                self._alias_by.pop(a, None)
            e.merged_into, e.status, e.parent, e.confidence, e.aliases = name, Status.UNKNOWN, None, 0.0, []
            self._bits[name] = self._new_bits()
            self._present[name] = False
            self._merged.pop(name, None)
            return True

    def confirm_same(self, a: str, b: str) -> str | None:
        """Fold two thing identities into one (a person confirmed they are the same object). The older
        identity survives with the newer one's current state if that is fresher, all aliases,
        exemplars and history; the other is kept, archived, with merged_into set. Refused (None) for
        two things visible at once, or anything but two distinct things."""
        with self.lock:
            a, b = self._survivor(a), self._survivor(b)
            if a == b or not (is_thing(a) and is_thing(b)) or a not in self.entities or b not in self.entities:
                return None
            ea, eb = self.entities[a], self.entities[b]
            if ea.status == Status.VISIBLE and eb.status == Status.VISIBLE:
                return None
            keep, drop = sorted((a, b), key=self._thing_number)
            if self._now is None:                   # before any update: events still need a time
                self._now, self._wall = time.monotonic(), time.time()
            self._absorb(keep, drop)
            return keep

    def merge_same_kind(self, older: str, newer: str) -> str | None:
        """One object per kind: the thing `newer` was just named what `older` answers to (Grok's label on
        the settle check). Fold newer into older only when they cannot be two objects: older lost (UNKNOWN
        / GONE) since before newer was confirmed, so never seen together; newer lying on the table; and
        nothing inside or under either. Two of a kind seen at once stay two. Returns the survivor or None."""
        with self.lock:
            a, b = self._survivor(older), self._survivor(newer)
            if a == b or not (is_thing(a) and is_thing(b)) or a not in self.entities or b not in self.entities:
                return None
            ea, eb, born = self.entities[a], self.entities[b], self._born.get(b)
            if ea.status not in ARCHIVED or eb.status != Status.VISIBLE or born is None \
                    or ea.last_seen is None or ea.last_seen >= born:
                return None
            if any(e.parent in (a, b) for e in self.entities.values()):
                return None
            return self.confirm_same(a, b)

    def similar_to(self, name: str) -> list[tuple[str, float]]:
        """Things that may be the same object as name (their maybe_same_as links to it), best first."""
        with self.lock:
            out = [(n, s) for n in self._things for m, s in self.entities[n].maybe_same_as
                   if m == name and self.entities[n].merged_into is None]
            return sorted(out, key=lambda x: -x[1])

    def exemplars(self, name: str) -> list[np.ndarray]:
        with self.lock:
            bank = self._banks.get(name)
            return list(bank.vecs) if bank else []

    def alias_phrases(self) -> list[str]:
        with self.lock:
            return list(self._aliases)

    def thing_labels(self) -> dict[str, str | None]:
        """Every live thing -> its spoken name (newest alias) or None: for answers and the dashboard."""
        with self.lock:
            return {n: (e.aliases[0] if e.aliases else None) for n, e in
                    ((n, self.entities[n]) for n in self._things) if e.merged_into is None}

    # ----- per-batch association (called by World.update before the debounce) --------------------

    def _associate_things(self, dets, seen: dict[str, Detection]) -> list:
        """Adds this batch's thing detections to seen (thing name -> proposal) and returns APPEARED
        events for things confirmed now. With no proposals and no open candidates this does nothing,
        so the configured objects behave exactly as without the open world."""
        tc = self._tcfg
        self._clean = {}
        for h in dets.hands:
            self._trail.setdefault(h.cls, deque(maxlen=64)).append((self._now, h.box_cm))
        if len(self._trail) > len(dets.hands) + 8:      # hand ids only grow: forget long-gone hands
            for hid in [k for k, tr in self._trail.items() if self._now - tr[-1][0] > 60.0]:
                del self._trail[hid]
        if not tc.enabled:
            return []
        props = self._proposals(dets.items, seen, dets.hands)
        self._batch_boxes = ([d.box_cm for d in seen.values()] + [d.box_cm for d in props]
                             + [h.box_cm for h in dets.hands])
        if not props and not self._cands:
            return []
        props, blobs = self._merged_blobs(props)
        props = self._match_visible(props, seen)
        props = self._match_held(props, seen)
        self._match_merged(blobs, seen)
        return self._track_candidates(props, seen, dets.hands)

    def _proposals(self, items, seen, hands) -> list[Detection]:
        """'thing' proposals that are plausible objects and not duplicates of a configured object
        (detected now or believed present) or of a hand, highest confidence first."""
        tc = self._tcfg
        known = [d.box_cm for d in seen.values()]
        known += [self.entities[n].box_cm for n in self.cfg.names()
                  if self.entities[n].box_cm is not None and (self._present[n] or any(self._bits[n]))]
        hand_boxes = [h.box_cm for h in hands]
        holders = list(self._thing_containers().values()) if hand_boxes else []
        kept: list[Detection] = []
        for d in sorted((d for d in items if d.cls == THING and d.conf >= tc.proposal_conf),
                        key=lambda d: -d.conf):
            if not tc.area_cm2[0] <= geom.area(d.box_cm) <= tc.area_cm2[1]:
                continue
            if any(geom.iou(d.box_cm, k) >= tc.dup_iou for k in known + [p.box_cm for p in kept]):
                continue
            # a hand resting inside a tub lies wholly within the tub's box: that box is still the tub
            is_holder = any(geom.iou(d.box_cm, b) >= tc.member_iou for b in holders)
            if not is_holder and any(geom.iou(d.box_cm, h) >= tc.dup_iou
                                     or geom.overlap_frac(d.box_cm, h) >= tc.hand_contain for h in hand_boxes):
                continue
            kept.append(d)
        return kept

    def _visible_things(self) -> list[str]:
        return [n for n in self._things if self.entities[n].status == Status.VISIBLE
                and self.entities[n].merged_into is None and self.entities[n].zone == 'table'
                and self.entities[n].box_cm is not None]

    def _merged_blobs(self, props) -> tuple[list[Detection], list[tuple[Detection, list[str]]]]:
        """Splits off the proposals lying over 2+ visible things' last boxes (and plainly none of them):
        (the rest, [(blob, the things under it)]). A blob is never a candidate: what it shows is known."""
        tc = self._tcfg
        vis = self._visible_things()
        rest, blobs = [], []
        for d in props:
            inside = [n for n in vis if geom.overlap_frac(d.box_cm, self.entities[n].box_cm) >= tc.merge_cover]
            if len(inside) < 2 or any(geom.iou(d.box_cm, self.entities[n].box_cm) >= tc.member_iou for n in inside):
                rest.append(d)
            else:
                blobs.append((d, inside))
        return rest, blobs

    def _match_merged(self, blobs, seen) -> None:
        """After the one-to-one rules: a thing under a blob that no proposal of its own explained stays
        where it was (its identity is not decidable inside the blob), at a confidence penalty, learning
        nothing. One still seen on its own is just that: holding it here too left its own proposal
        unmatched, and it was born again every few seconds (on the rig, 100 things over one cable pile)."""
        tc = self._tcfg
        for d, inside in blobs:
            group = frozenset(inside)
            for n in inside:
                if n in seen:
                    continue
                ent = self.entities[n]
                seen[n] = Detection(THING, d.conf, self._box_px.get(n, d.box_px), ent.pos_cm, ent.box_cm)
                self._merged[n] = group
                ent.confidence = min(ent.confidence, self.cfg.ambiguity_penalty)
                self._unsure_until[n] = self._now + tc.ambiguous_s

    def _match_visible(self, props, seen) -> list[Detection]:
        """Rule (b), in view: each visible thing takes its nearest proposal within gate_cm."""
        tc = self._tcfg
        names = [n for n in self._visible_things() if n not in seen]
        pairs = []
        for i, d in enumerate(props):
            for n in names:
                ent = self.entities[n]
                dd = geom.dist(d.center_cm, ent.pos_cm)
                if dd <= tc.gate_cm and self._size_fits(n, d.box_cm):
                    pairs.append((dd, i, n))
        taken = self._greedy(pairs, props, seen)
        self._resolve_splits(taken, props, seen)
        return [d for i, d in enumerate(props) if i not in taken]

    def _match_held(self, props, seen) -> list[Detection]:
        """Rule (b), in a hand: a HELD thing takes a proposal at its holding hand's recent boxes,
        newest first, so two hands that cross each keep their own thing."""
        tc = self._tcfg
        held = [n for n in self._things if self.entities[n].status == Status.HELD and n not in seen
                and self.entities[n].parent in self._trail]
        pairs = []
        for i, d in enumerate(props):
            for n in held:
                for age, (t, box) in enumerate(reversed(self._trail[self.entities[n].parent])):
                    if self._now - t > tc.hand_trail_s:
                        break
                    gap = _box_dist(d.center_cm, box)
                    if gap <= tc.hand_gate_cm:
                        pairs.append(((age, gap), i, n))
                        break
        taken = self._greedy(pairs, props, seen)
        return [d for i, d in enumerate(props) if i not in taken]

    def _greedy(self, pairs, props, seen) -> dict[int, str]:
        """Assign cheapest first, one proposal per thing. An assignment is clean (may teach the
        exemplar bank) when neither side had any other option within its gate."""
        per_prop: dict[int, int] = {}
        per_name: dict[str, int] = {}
        for _, i, n in pairs:
            per_prop[i] = per_prop.get(i, 0) + 1
            per_name[n] = per_name.get(n, 0) + 1
        taken: dict[int, str] = {}
        for _, i, n in sorted(pairs, key=lambda p: p[0]):
            if i in taken or n in seen:
                continue
            taken[i], seen[n] = n, props[i]
            if per_prop[i] == 1 and per_name[n] == 1:
                self._clean[n] = props[i]
        return taken

    def _resolve_splits(self, taken: dict[int, str], props, seen) -> None:
        """Things leaving a merged proposal: nearest-to-where-they-were stands unless appearance
        decisively says a pair came apart the other way round. Either way nothing is learned for
        ambiguous_s, and confidence returns to normal."""
        tc = self._tcfg
        split = {n: i for i, n in taken.items() if n in self._merged}
        groups: dict[frozenset, list[str]] = {}
        for n in split:
            groups.setdefault(self._merged.pop(n), []).append(n)
            self.entities[n].confidence = 1.0
            self._unsure_until[n] = self._now + tc.ambiguous_s
            self._clean.pop(n, None)
        for members in groups.values():
            if len(members) != 2:
                continue
            a, b = sorted(members)
            va, vb = self._embed(props[split[a]].box_px), self._embed(props[split[b]].box_px)
            ba, bb = self._banks.get(a), self._banks.get(b)
            if va is None or vb is None or not (ba and ba.vecs and bb and bb.vecs):
                continue
            kept, swapped = ba.sim(va) + bb.sim(vb), ba.sim(vb) + bb.sim(va)
            if swapped - kept >= 2 * tc.tie_margin and min(ba.sim(vb), bb.sim(va)) >= tc.tie_sim:
                seen[a], seen[b] = props[split[b]], props[split[a]]

    # ----- candidates -------------------------------------------------------------------------------

    def _track_candidates(self, props, seen, hands) -> list:
        tc, k, n = self._tcfg, self.cfg.present_k, self.cfg.present_n
        pairs = sorted((geom.dist(d.center_cm, c.det.center_cm), i, j)
                       for i, d in enumerate(props) for j, c in enumerate(self._cands))
        used_p, used_c = set(), set()
        for dd, i, j in pairs:
            if dd <= tc.gate_cm and i not in used_p and j not in used_c:
                used_p.add(i)
                used_c.add(j)
                self._cands[j].det = props[i]
                self._cands[j].bits.append(True)
                self._cands[j].trail.append((self._now, props[i].center_cm))
        for j, c in enumerate(self._cands):
            if j not in used_c:
                c.bits.append(False)
        self._cands = [c for c in self._cands if any(c.bits)]
        for i, d in enumerate(props):
            if i not in used_p:
                c = Candidate(d, deque([True], maxlen=n), self._now)
                c.trail.append((self._now, d.center_cm))
                self._cands.append(c)
        out = []
        for c in list(self._cands):
            if sum(c.bits) < k or not c.bits[-1]:
                continue
            if any(geom.overlap_frac(h.box_cm, c.det.box_cm) > tc.new_hand_overlap_max for h in hands):
                continue                       # still being put down: decide once the hand is off it
            if not self._still(c):
                continue                       # moving: an arm, or an object still being slid
            self._cands.remove(c)
            out += self._decide(c, seen)
        return out

    def _still(self, c: Candidate) -> bool:
        """The candidate has been matched for still_s and its centre stayed within still_cm of where it
        is now over that time."""
        tc = self._tcfg
        if self._now - c.born < tc.still_s - 1e-9:
            return False
        recent = [p for t, p in c.trail if t >= self._now - tc.still_s - 1e-9]
        return bool(recent) and all(geom.dist(p, c.det.center_cm) <= tc.still_cm for p in recent)

    def _decide(self, c: Candidate, seen) -> list:
        """A persistent, unexplained proposal: an existing identity (its debounce is seeded with the
        candidate's bits, so the world's observation rule reports TAKEN_OUT / UNCOVERED / CORRECTED /
        MOVED as for a configured object) or a new thing."""
        d = c.det
        if self._on_a_thing(d, seen):
            return []                          # a thing in view seen twice (two boxes on one object)
        vec = self._embed(d.box_px)
        verdict, links = self._identify(d, vec, seen, self._may_create(d))
        if verdict == 'skip':
            return []
        if verdict is not None:
            self._bits[verdict] = self._new_bits(list(c.bits)[:-1])
            seen[verdict] = d
            return []
        return self._new_thing(d, c.bits, vec, links, seen)

    def _on_a_thing(self, d: Detection, seen) -> bool:
        """The proposal lies on a thing in view (visible, or placed this batch): the same box (IoU >=
        dup_iou), or the same spot at a like size (either box holds the other's centre, areas within
        SAME_SIZE). Then it is that thing seen twice, never a new one nor a lost one brought back beside
        it. A phone laid on a notebook is neither (a third of its box, and far smaller)."""
        tc = self._tcfg
        boxes = [self.entities[n].box_cm for n in self._visible_things()]
        boxes += [s.box_cm for n, s in seen.items() if is_thing(n) and s.box_cm is not None]
        a = geom.area(d.box_cm)
        for b in boxes:
            if geom.iou(d.box_cm, b) >= tc.dup_iou:
                return True
            ab = geom.area(b)
            if a > 0 and ab > 0 and max(a, ab) / min(a, ab) <= SAME_SIZE and (
                    geom.contains_point(b, d.center_cm) or geom.contains_point(d.box_cm, geom.center(b))):
                return True
        return False

    def _may_create(self, d: Detection) -> bool:
        """A proposal may start a NEW identity only inside the tabletop outline, clear of its edge
        band (table_area:), and only if it is not flagged occluded (mostly inside a person box: a
        finger, a knee, a carried object). Otherwise it can still be an existing thing (rules a-c)."""
        if self._vetoes and any(until > (self._wall or 0) and b[0] - r <= d.center_cm[0] <= b[2] + r
                                and b[1] - r <= d.center_cm[1] <= b[3] + r for b, until, r in self._vetoes):
            return False                    # where retired clutter lay (retire_thing)
        return not d.occluded and (self._area.interior(d.center_cm) or self._area.interior(_footprint(d.box_cm)))

    def _identify(self, d: Detection, vec, seen, may_create: bool = True):
        """Ordered identity rules (a) causal, (b) continuity, (c) decisive appearance; returns
        (thing, []) / ('skip', []) / (None, maybe_same_as links) for a new thing. may_create False: no
        new thing, so 'skip' where one would be made (and no confidence is spent on an ambiguity)."""
        tc = self._tcfg
        # a configured object under a cover reappearing at its spot is the class detector's to confirm
        for n in self.cfg.names('target'):
            e = self.entities[n]
            if e.status == Status.UNDER and e.pos_cm is not None and \
                    geom.dist(d.center_cm, e.pos_cm) <= tc.same_spot_cm:
                return 'skip', []
        # (a) causal
        cause = [n for n in self._things if self.entities[n].status in HIDDEN and self._emerged(n, d)]
        if len(cause) == 1:
            return cause[0], []
        if cause:
            pick = self._by_look(vec, cause, tc.tie_sim, tc.tie_margin)
            if pick:
                self._unsure_until[pick] = self._now + tc.ambiguous_s
                return pick, []
            # Hidden only by 'something undetected' (world._background_changed: the pixels there are not
            # bare table): no cover, so no event tells them apart, and a new thing per return piles up
            # (on the rig, a knee at the table edge made 7 things in 30 s). The nearest one of the right
            # size, most recently hidden on a tie, is back. A real cover keeps the ambiguity below.
            blind = [n for n in cause if self.entities[n].parent == 'unknown' and n not in seen
                     and self.entities[n].pos_cm is not None and self._size_ok(d.box_cm, self.entities[n].box_cm)]
            if blind and all(self.entities[n].parent == 'unknown' for n in cause):
                return min(blind, key=lambda n: (geom.dist(d.center_cm, self.entities[n].pos_cm),
                                                 -self._hidden_at.get(n, NEG))), []
            if not may_create:
                return 'skip', []
            for n in cause:        # one of them came out, but which is unknown
                self.entities[n].confidence *= self.cfg.ambiguity_penalty
            return None, [(n, self._look_score(vec, n)) for n in cause]
        # (b) continuity: lost in place (not from a hand), seen again where it was
        spot = [n for n in self._things if self.entities[n].status == Status.UNKNOWN
                and self.entities[n].merged_into is None and self.entities[n].pre_pickup_pos is None
                and self.entities[n].pos_cm is not None
                and geom.dist(d.center_cm, self.entities[n].pos_cm) <= tc.same_spot_cm
                and self._size_fits(n, d.box_cm)]
        if spot:           # several lost there (duplicates of one object): the nearest, then the latest
            return min(spot, key=lambda n: (geom.dist(d.center_cm, self.entities[n].pos_cm),
                                            -(self.entities[n].last_seen or NEG))), []
        back = self._reborn(d, seen)
        if back:
            return back, []
        # (c) appearance: only a strict, clear match brings an archived thing back
        # A visible thing unseen for two batches may have jumped out of its gate (knocked, or a hand
        # the detector missed): it is still 'visible' only because absence takes ~9 batches to declare.
        archived = [n for n in self._things if self.entities[n].merged_into is None
                    and (self.entities[n].status in ARCHIVED
                         or (self.entities[n].status == Status.VISIBLE and n not in seen
                             and not any(list(self._bits[n])[-2:])))]
        rivals = [n for n in self._things if self.entities[n].merged_into is None]
        pick = self._by_look(vec, archived, tc.resurrect_sim, tc.resurrect_margin, rivals)
        if pick:
            return pick, []
        # (d) new, linked to archived things it resembles
        if not may_create:
            return 'skip', []
        links = sorted(((n, self._look_score(vec, n)) for n in archived), key=lambda x: -x[1])
        return None, [(n, s) for n, s in links if s >= tc.maybe_sim][:tc.maybe_max]

    def _reborn(self, d: Detection, seen) -> str | None:
        """(b') thing_identity.rebirth_s: a thing lost recently, with or without a hand (picked up and put
        back, a false pick-up while a hand passed over it, an arm at the edge), seen again of its size at
        the spot it was last seen or picked up from: the nearest one, most recently seen on a tie."""
        ic = self._icfg
        if ic.rebirth_s <= 0 or self._wall is None:
            return None
        back = []
        for n in self._things:
            e = self.entities[n]
            if e.merged_into is not None or n in seen or e.status not in ARCHIVED + (Status.HELD,) \
                    or e.last_seen is None or self._wall - e.last_seen > ic.rebirth_s \
                    or not self._size_fits(n, d.box_cm):
                continue
            dd = min((geom.dist(d.center_cm, p) for p in (e.pos_cm, e.pre_pickup_pos) if p is not None),
                     default=float('inf'))
            if dd <= ic.rebirth_cm:
                back.append((dd, -e.last_seen, n))
        return min(back)[2] if back else None

    def _emerged(self, name: str, d: Detection) -> bool:
        """The proposal can be this hidden thing: it lies where the thing sat under its cover, or
        next to a container / cover on its parent chain that a hand touched since the thing hid."""
        tc, ent = self._tcfg, self.entities[name]
        if ent.status == Status.UNDER and ent.pos_cm is not None and \
                geom.dist(d.center_cm, ent.pos_cm) <= tc.same_spot_cm:
            return True
        since = self._hidden_at.get(name, NEG)
        _, chain = relations.resolve_chain(name, self.entities, self.cfg.max_nesting)
        if not self._size_ok(d.box_cm, ent.box_cm):
            return False            # e.g. the tub it lies in, seen again after being carried: not it
        for p in chain[1:]:
            box = self.entities[p].box_cm
            if box is not None and geom.contains_point(_grow(box, tc.emerge_cm), d.center_cm) \
                    and self._touched_since(p, box, since):
                return True
        return False

    def _touched_since(self, parent: str, box, since: float) -> bool:
        if max(self._contacts[parent].values(), default=NEG) > since:
            return True
        if any(c == parent for hid in self._hands for c, _ in self._dwell.visits_since(hid, since)):
            return True
        return any(t > since and geom.intersection(b, box) for trail in self._trail.values() for t, b in trail)

    def _by_look(self, vec, names, thr: float, margin: float, rivals=None) -> str | None:
        """The best-looking of names if it scores >= thr and leads every rival by margin."""
        if vec is None or not names:
            return None
        best = max(names, key=lambda n: self._look_score(vec, n))
        s1 = self._look_score(vec, best)
        s2 = max((self._look_score(vec, n) for n in (rivals or names) if n != best), default=NEG)
        return best if s1 >= thr and s1 - s2 >= margin else None

    def _look_score(self, vec, name: str) -> float:
        bank = self._banks.get(name)
        s = bank.sim(vec) if bank else NEG
        return round(s, 3) if s != NEG else NEG

    def _new_thing(self, d: Detection, bits, vec, links, seen) -> list:
        self._thing_n += 1
        name = f'{PREFIX}{self._thing_n}'
        ent = Entity(name=name, kind='target', confidence=1.0, pos_cm=d.center_cm, box_cm=d.box_cm)
        # a causal link (one of several hidden things) stands without appearance: score 0 = no look
        ent.maybe_same_as = [(n, float(s) if s != NEG else 0.0) for n, s in links]
        self.entities[name] = ent
        self._things.append(name)
        self._born[name] = self._wall if self._wall is not None else time.time()
        self._path[name] = deque(maxlen=60)
        self._bits[name] = self._new_bits(list(bits)[:-1])
        self._present[name] = False
        self._contacts[name] = {}
        self._banks[name] = ExemplarBank(self._tcfg.exemplar_max, self._tcfg.exemplar_dup_sim)
        if vec is not None and self._isolated(d):
            self._banks[name].add(vec)
            self._learn_t[name] = self._now
        seen[name] = d
        return [self._emit(name, EventType.APPEARED, to_cm=d.center_cm)]

    # ----- after the rules ----------------------------------------------------------------------------

    def _after_things(self, events) -> None:
        """Bookkeeping once this batch's rules ran: when things were put down / hid, merge groups of
        things no longer in view, and exemplar learning."""
        for ev in events:
            if ev.type in PLACED and ev.obj in self.entities and not self._refound(ev):
                self._placed_t[ev.obj] = self._now
        if self._t_start is None:
            self._t_start = self._now
        for n, ent in self.entities.items():
            if ent.status == Status.VISIBLE and ent.pos_cm is not None:
                self._rest_pos[n] = tuple(ent.pos_cm)
        for n in self.cfg.names():                  # a configured object's first sighting emits no event
            if self.entities[n].status == Status.VISIBLE and n not in self._known_seen:
                self._known_seen.add(n)
                if self._now - self._t_start >= STARTUP_S:
                    self._placed_t[n] = self._now
        for n in self._things:
            ent = self.entities[n]
            if ent.status in HIDDEN:
                self._hidden_at.setdefault(n, self._now)
            else:
                self._hidden_at.pop(n, None)
            if ent.status != Status.VISIBLE and self._merged.pop(n, None) is not None:
                self._unsure_until[n] = self._now + self._tcfg.ambiguous_s
        if self.embed is not None:
            self._learn_looks()

    def _learn_looks(self) -> None:
        tc = self._tcfg
        for n, d in self._clean.items():
            ent = self.entities[n]
            if ent.status != Status.VISIBLE or ent.confidence < self.cfg.answer_plain or ent.candidates \
                    or n in self._merged or self._unsure_until.get(n, NEG) > self._now:
                continue
            if self._now - self._learn_t.get(n, NEG) < tc.exemplar_every_s or not self._isolated(d):
                continue
            self._learn_t[n] = self._now
            vec = self._embed(d.box_px)
            if vec is not None:
                self._banks[n].add(vec)

    def _isolated(self, d: Detection) -> bool:
        """A view worth remembering: no hand or other box touches it and the crop is big enough."""
        tc = self._tcfg
        others = [b for b in self._batch_boxes if b is not d.box_cm and geom.iou(b, d.box_cm) < 0.95]
        if any(geom.intersection(b, d.box_cm) for b in others):
            return False
        x1, y1, x2, y2 = d.box_px
        w, h = self.cfg.frame_size_px
        return min(x2 - x1, y2 - y1) >= tc.exemplar_min_px and x1 >= 0 and y1 >= 0 and x2 <= w and y2 <= h

    def _embed(self, box_px) -> np.ndarray | None:
        if self.embed is None:
            return None
        img = self._frame.img if self._frame is not None else None
        try:
            v = self.embed(img, box_px)
        except Exception:
            return None                        # appearance is optional evidence, never a failure
        if v is None:
            return None
        v = np.asarray(v, np.float32).ravel()
        nv = float(np.linalg.norm(v))
        return v / nv if nv > 0 else None

    # ----- naming and merging -------------------------------------------------------------------------

    def _is_known_name(self, key: str) -> bool:
        return key in self._known_names or _singular(key) in self._known_names

    def _refound(self, ev) -> bool:
        """A CORRECTED sighting where the object was last seen, or an UNCOVERED one from under something
        unknown (an arm or a blanket lay over it): a lost track recovered, not a put-down."""
        prev = self._rest_pos.get(ev.obj)
        again = ev.type == EventType.CORRECTED or (
            ev.type == EventType.UNCOVERED and ev.obj in getattr(self, '_unknown_uncovered', ()))
        return again and prev is not None and ev.to_cm is not None and geom.dist(prev, ev.to_cm) < self.cfg.moved_min_cm

    def _teach_target(self) -> str | None:
        zone = self.cfg.teach_zone_cm
        known = [n for n in self.cfg.names() if self.entities[n].status == Status.VISIBLE
                 and self.entities[n].zone == 'table' and self.entities[n].pos_cm is not None]
        pool = [n for n in self._visible_things() + known if n in self._placed_t]
        if zone:
            pool = [n for n in pool if geom.contains_point(tuple(zone), self.entities[n].pos_cm)]
        elif self._now is not None:
            pool = [n for n in pool if self._now - self._placed_t[n] <= self._tcfg.teach_recent_s]
        if not pool:
            return None
        return max(pool, key=lambda n: (self._placed_t[n], -self._thing_number(n) if is_thing(n) else 0))

    def _bind(self, name: str, key: str) -> None:
        self._alias_by.pop(key, None)
        old = self._aliases.get(key)
        if old is not None and key in self.entities[old].aliases:
            self.entities[old].aliases.remove(key)
        self._aliases[key] = name
        aliases = self.entities[name].aliases
        if key in aliases:
            aliases.remove(key)
        aliases.insert(0, key)

    def _survivor(self, name: str) -> str:
        seen = set()
        while name in self.entities and self.entities[name].merged_into and name not in seen:
            seen.add(name)
            name = self.entities[name].merged_into
        return name

    @staticmethod
    def _thing_number(name: str) -> int:
        try:
            return int(name.split(':', 1)[1])
        except (IndexError, ValueError):
            return 0

    def _absorb(self, keep: str, drop: str) -> None:
        ek, ed = self.entities[keep], self.entities[drop]
        if (ed.last_seen or NEG) >= (ek.last_seen or NEG):      # the dropped identity is the current one
            for f in ('status', 'parent', 'pos_cm', 'box_cm', 'last_seen', 'confidence', 'candidates',
                      'edge', 'pre_pickup_pos', 'zone', 'held_since'):
                setattr(ek, f, getattr(ed, f))
            for d in (self._seen_t, self._rest, self._rest_box, self._carry, self._box_px, self._hidden_at,
                      self._placed_t, self._lifted_at, self._learn_t, self._unsure_until, self._matched_t,
                      self._waiting):
                if drop in d:
                    d[keep] = d.pop(drop)
                else:
                    d.pop(keep, None)
            for d in (self._path, self._bits, self._present, self._contacts):
                d[keep] = d[drop]
            self._look_px.pop(keep, None)           # its remembered patch is stale; refreshed soon
            if drop in self._confirmed:
                self._confirmed.add(keep)
        self._banks[keep].extend(self._banks[drop])
        mine = list(ek.aliases)
        for key in reversed(ed.aliases):
            by = self._alias_by.get(key)
            self._bind(keep, key)
            if by:
                self._alias_by[key] = by
        ek.aliases = mine + [a for a in ek.aliases if a not in mine]     # its own names stay first
        links: dict[str, float] = {}
        for n, sc in ek.maybe_same_as + ed.maybe_same_as:
            if n not in (keep, drop):
                links[n] = max(sc, links.get(n, NEG))
        ek.maybe_same_as = sorted(links.items(), key=lambda x: -x[1])
        for n in self._things:                      # links to the dropped identity now mean the survivor
            if n != keep:
                e = self.entities[n]
                e.maybe_same_as = [(keep if m == drop else m, sc) for m, sc in e.maybe_same_as]
        ed.merged_into, ed.status, ed.parent, ed.confidence, ed.aliases = keep, Status.UNKNOWN, None, 0.0, []
        self._bits[drop] = self._new_bits()
        self._present[drop] = False
        self._path[drop] = deque(maxlen=60)
        self._contacts[drop] = {}
        self._merged.pop(drop, None)
        self._emit(keep, EventType.CORRECTED, to_cm=ek.pos_cm)

    def _size_ok(self, a, b) -> bool:
        if b is None:
            return True
        aa, ab = geom.area(a), geom.area(b)
        return aa > 0 and ab > 0 and max(aa, ab) / min(aa, ab) <= self._tcfg.size_ratio_max

    def _size_fits(self, name: str, box) -> bool:
        """Size gate for a proposal to be this thing: a size like its latest box or, while that has shrunk
        (part of the thing hidden), the box it rests in again. An arm over most of a big thing leaves a
        proposal for the uncovered part, and the thing follows it; when the arm goes, the whole thing is
        still itself, not a new thing (hands_1: a tub)."""
        last, rest = self.entities[name].box_cm, self._rest_box.get(name)
        if self._size_ok(box, last):
            return True
        return (rest is not None and last is not None and geom.area(last) < geom.area(rest)
                and geom.iou(box, rest) >= REST_IOU)

    def _thing_json(self, ent: Entity, out: dict) -> dict:
        if is_thing(ent.name):
            out.update(label=ent.aliases[0] if ent.aliases else None, aliases=list(ent.aliases),
                       maybe_same_as=[[n, s] for n, s in ent.maybe_same_as],
                       named_by=self._alias_by.get(ent.aliases[0]) if ent.aliases else None)
        return out

    # ----- things as containers ---------------------------------------------------------------------
    # A large thing (a tub, a bag, a hat, a cup) holds what a hand leaves in it, by the world's container
    # rule (rules 7-8): the hand holding the object dwells inside the thing's box, leaves, and the object
    # is not seen again. The configured box keeps priority, and a hand that crossed the thing was
    # carrying something past it (over a placemat, a sheet of paper), not into it.

    def _thing_containers(self) -> dict[str, tuple]:
        """Things that may hold others -> box: visible on the table (so inside or under nothing), with a
        footprint of at least min_area_cm2. While an arm hides part of one, the box it rests in."""
        if not self._ccfg.enabled:
            return {}
        out = {}
        for n in self._visible_things():
            ent = self.entities[n]
            box = self._rest_box[n] if self._partly_hidden(n, ent) else ent.box_cm
            if geom.area(box) >= self._ccfg.min_area_cm2:
                out[n] = box
        return out

    def _track_crossings(self, hands, containers: dict[str, tuple]) -> None:
        """Where each hand's centre went into a thing container and where it last was inside, in the
        thing's own coordinates. Called with the same hands and boxes as the dwell tracker, so a visit
        that ends here ends there in the same update (same exit time)."""
        inside = set()
        for h in hands:
            c = geom.center(h.box_cm)
            for n, box in containers.items():
                if geom.contains_point(box, c):
                    inside.add((h.cls, n))
                    self._in_thing.setdefault((h.cls, n), [self._now, _uv(c, box), None])[2] = _uv(c, box)
        for key in [k for k in self._in_thing if k not in inside]:
            entry_t, a, b = self._in_thing.pop(key)
            crossed = any(p * q < 0 and abs(p - q) >= CROSS_UV for p, q in zip(a, b))
            self._thing_visits[(key[0], key[1], self._now)] = (entry_t, crossed)
        for v in [v for v in self._thing_visits if self._now - v[2] > relations.DwellTracker.HISTORY_S]:
            del self._thing_visits[v]

    def _holding_visits(self, name: str, hid: str, since: float) -> list[tuple[str, float]]:
        """The hand's completed container visits since `since` that could have left `name` inside,
        oldest first; the last one is the container to use. A thing is skipped when the hand crossed it
        or when holding name would nest deeper than max_nesting. A configured container the hand left
        while it was inside the last thing wins over that thing: a box standing on a tray."""
        out = []
        for c, t in self._dwell.visits_since(hid, since):
            if c == name or c not in self.entities:
                continue
            if is_thing(c) and (self._thing_visits.get((hid, c, t), (t, False))[1] or not self._may_hold(c, name)):
                continue
            out.append((c, t))
        if out and is_thing(out[-1][0]):
            entered = self._thing_visits.get((hid, *out[-1]), (out[-1][1],))[0]
            known = [v for v in out if not is_thing(v[0]) and v[1] >= entered]
            if known:
                out.remove(known[-1])
                out.append(known[-1])
        return out

    def _may_hold(self, container: str, name: str) -> bool:
        _, chain = relations.resolve_chain(container, self.entities, self.cfg.max_nesting + 1)
        if name in chain:
            return False            # a thing inside the object cannot hold it
        return len(chain) + self._depth_below(name) <= self.cfg.max_nesting

    def _depth_below(self, name: str) -> int:
        """Levels of entities hidden in name (keys in the box in hand: the box has 1)."""
        depth, frontier, seen = 0, {name}, {name}
        while True:
            nxt = {n for n, e in self.entities.items() if e.parent in frontier and n not in seen}
            if not nxt:
                return depth
            depth, frontier = depth + 1, nxt
            seen |= nxt


def _uv(p, box) -> tuple[float, float]:
    """p in box's own coordinates: -1 at its left / top edge, 0 in the middle, 1 at its right / bottom."""
    hw, hh = (box[2] - box[0]) / 2 or 1.0, (box[3] - box[1]) / 2 or 1.0
    return ((p[0] - box[0]) / hw - 1.0, (p[1] - box[1]) / hh - 1.0)
