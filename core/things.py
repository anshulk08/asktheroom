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
    merge  one proposal covering two visible things' last boxes: both are kept where they were, at
           ambiguity_penalty, until they come apart (then nearest, or appearance if decisive)
    (b)    a visible thing: the nearest proposal within gate_cm
    (b)    a HELD thing: a proposal at its holding hand's recent boxes
  otherwise the proposal is a candidate; after the k-of-n debounce, clear of hands:
    (a)    causal: it came out of a container / cover that holds hidden things (a hand touched that
           parent since they hid), or appeared where an UNDER thing lay: that child. Several
           children: appearance may pick one, else a new thing linked to all of them
    (b)    a thing lost in place (no hand involved) seen again at the same spot
    (c)    appearance resurrects an archived thing only above resurrect_sim AND by resurrect_margin
           over every other thing (a twin on the table blocks it)
    (d)    a new thing, with maybe_same_as links to archived things that look similar; only inside
           the tabletop outline clear of its edge band (table_area:, core/table_area.py), and never
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
STARTUP_S = 3.0     # configured objects first seen this soon after the first batch were not put down
ARTICLES = {'my', 'the', 'a', 'an', 'your', 'our', 'this', 'that', 'his', 'her', 'their'}
NEG = float('-inf')


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
        self._area = TableArea.from_dict(getattr(self.cfg, 'table_area', None))   # tabletop outline, or none
        # Never reuse an id: the event log outlives RESET and the process, and a new thing:N would
        # inherit an old thing:N's history. Duck-typed sinks without max_number start at 0.
        logged = getattr(self.events, 'max_number', None)
        self._thing_n = max(getattr(self, '_thing_n', 0), logged(PREFIX) if logged else 0)
        self._things: list[str] = []                   # every thing ever confirmed, oldest first
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

    def bind_alias(self, entity: str, name: str) -> bool:
        """Bind a name to a given thing (the 'yes, that's it' flow). A name keeps one owner: it
        moves here from any other thing. Configured objects' names are refused."""
        with self.lock:
            key, ent = norm_name(name), self._survivor(entity)
            if not key or self._is_known_name(key) or not is_thing(ent) or ent not in self.entities:
                return False
            self._bind(ent, key)
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
        props = self._match_merged(props, seen)
        props = self._match_visible(props, seen)
        props = self._match_held(props, seen)
        return self._track_candidates(props, seen, dets.hands)

    def _proposals(self, items, seen, hands) -> list[Detection]:
        """'thing' proposals that are plausible objects and not duplicates of a configured object
        (detected now or believed present) or of a hand, highest confidence first."""
        tc = self._tcfg
        known = [d.box_cm for d in seen.values()]
        known += [self.entities[n].box_cm for n in self.cfg.names()
                  if self.entities[n].box_cm is not None and (self._present[n] or any(self._bits[n]))]
        hand_boxes = [h.box_cm for h in hands]
        kept: list[Detection] = []
        for d in sorted((d for d in items if d.cls == THING and d.conf >= tc.proposal_conf),
                        key=lambda d: -d.conf):
            if not tc.area_cm2[0] <= geom.area(d.box_cm) <= tc.area_cm2[1]:
                continue
            if any(geom.iou(d.box_cm, k) >= tc.dup_iou for k in known + [p.box_cm for p in kept]):
                continue
            if any(geom.iou(d.box_cm, h) >= tc.dup_iou or geom.overlap_frac(d.box_cm, h) >= tc.hand_contain
                   for h in hand_boxes):
                continue
            kept.append(d)
        return kept

    def _visible_things(self) -> list[str]:
        return [n for n in self._things if self.entities[n].status == Status.VISIBLE
                and self.entities[n].merged_into is None and self.entities[n].zone == 'table'
                and self.entities[n].box_cm is not None]

    def _match_merged(self, props, seen) -> list[Detection]:
        """One proposal over 2+ visible things' last boxes: keep every one of them where it was (their
        identities are not decidable inside the blob), at a confidence penalty, and learn nothing."""
        tc = self._tcfg
        vis = self._visible_things()
        rest = []
        for d in props:
            inside = [n for n in vis if geom.overlap_frac(d.box_cm, self.entities[n].box_cm) >= tc.merge_cover]
            if len(inside) < 2 or any(geom.iou(d.box_cm, self.entities[n].box_cm) >= tc.member_iou for n in inside):
                rest.append(d)
                continue
            group = frozenset(inside)
            for n in inside:
                ent = self.entities[n]
                seen[n] = Detection(THING, d.conf, self._box_px.get(n, d.box_px), ent.pos_cm, ent.box_cm)
                self._merged[n] = group
                ent.confidence = min(ent.confidence, self.cfg.ambiguity_penalty)
                self._unsure_until[n] = self._now + tc.ambiguous_s
        return rest

    def _match_visible(self, props, seen) -> list[Detection]:
        """Rule (b), in view: each visible thing takes its nearest proposal within gate_cm."""
        tc = self._tcfg
        names = [n for n in self._visible_things() if n not in seen]
        pairs = []
        for i, d in enumerate(props):
            for n in names:
                ent = self.entities[n]
                dd = geom.dist(d.center_cm, ent.pos_cm)
                if dd <= tc.gate_cm and self._size_ok(d.box_cm, ent.box_cm):
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
        vec = self._embed(d.box_px)
        verdict, links = self._identify(d, vec, seen, self._may_create(d))
        if verdict == 'skip':
            return []
        if verdict is not None:
            self._bits[verdict] = deque(list(c.bits)[:-1], maxlen=self.cfg.present_n)
            seen[verdict] = d
            return []
        return self._new_thing(d, c.bits, vec, links, seen)

    def _may_create(self, d: Detection) -> bool:
        """A proposal may start a NEW identity only inside the tabletop outline, clear of its edge
        band (table_area:), and only if it is not flagged occluded (mostly inside a person box: a
        finger, a knee, a carried object). Otherwise it can still be an existing thing (rules a-c)."""
        return not d.occluded and self._area.interior(d.center_cm)

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
                and self._size_ok(d.box_cm, self.entities[n].box_cm)]
        if len(spot) == 1:
            return spot[0], []
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

    def _emerged(self, name: str, d: Detection) -> bool:
        """The proposal can be this hidden thing: it lies where the thing sat under its cover, or
        next to a container / cover on its parent chain that a hand touched since the thing hid."""
        tc, ent = self._tcfg, self.entities[name]
        if ent.status == Status.UNDER and ent.pos_cm is not None and \
                geom.dist(d.center_cm, ent.pos_cm) <= tc.same_spot_cm:
            return True
        since = self._hidden_at.get(name, NEG)
        _, chain = relations.resolve_chain(name, self.entities, self.cfg.max_nesting)
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
        self._path[name] = deque(maxlen=60)
        self._bits[name] = deque(list(bits)[:-1], maxlen=self.cfg.present_n)
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
        """A CORRECTED sighting where the object was last seen: a lost track recovered, not a put-down."""
        prev = self._rest_pos.get(ev.obj)
        return (ev.type == EventType.CORRECTED and prev is not None and ev.to_cm is not None
                and geom.dist(prev, ev.to_cm) < self.cfg.moved_min_cm)

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
            for d in (self._seen_t, self._rest, self._carry, self._box_px, self._hidden_at,
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
            self._bind(keep, key)
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
        self._bits[drop] = deque(maxlen=self.cfg.present_n)
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

    def _thing_json(self, ent: Entity, out: dict) -> dict:
        if is_thing(ent.name):
            out.update(label=ent.aliases[0] if ent.aliases else None, aliases=list(ent.aliases),
                       maybe_same_as=[[n, s] for n, s in ent.maybe_same_as])
        return out
