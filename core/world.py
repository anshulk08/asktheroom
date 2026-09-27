"""Deterministic rule-based world model (spec 3.5). Owns every Entity; update() applies the rules
in order once per detection batch and returns the events it emitted. Spatial and image helpers
(covers, container dwell, parent chains, background / appearance) live in relations.py.

Open-world objects (class-agnostic 'thing' proposals -> entities thing:1, thing:2, ...) are matched
to identities by core/things.py (mixed in), then run through these same rules.

Room memory (spec 0009: props carried off the table into drawn room zones) is core/room_world.py
(mixed in): room_update() applies a zone visit, _observe() hands a room entity back to the table.

Readers (answers, llm, server, eval) use only get / resolve / history / state_json / place (WorldAPI), plus
find / teach / bind_alias / confirm_same / similar_to for named things (duck-typed: FakeWorld has
only the read API, for tests and --fake runs)."""
from __future__ import annotations

import threading
import time
from collections import deque
from typing import Protocol, Union

import cv2

from core import geom, relations
from core.config import Config
from core.presence import Presence
from core.room_world import RoomRules
from core.surround import SurroundMemory, UnknownCoverConfig
from core.things import ThingRules, is_thing
from core.types import Detection, Detections, Entity, Event, EventType, Frame, Point, Status

HISTORY_MAX = 1000
TRANSIT_S, TRANSIT_CM = 0.5, 4.0   # a hand whose centre moved this far this recently is passing over
LABEL_LEFT_IOU = 0.3         # a target's label this clear of its box while its pixels stay: a swap
LABEL_ON_THING_IOU = 0.5    # a configured label on a thing's box overlapping this much ...
ESTABLISHED_S = 2.0         # ... in place this long before the object was last seen elsewhere: the thing
PARTLY_HIDDEN_INSIDE = 0.8  # a smaller box this much inside the box an object rests in: part of it is hidden
THING_COVER_AREA = 1.5      # a thing at least this many times an object's footprint can lie over it
PERSON_OVER = 0.5           # a person box over this share of an object's box hides its spot (_person_over)
PERSON_HOLD_S = 20.0        # ... for at most this long (unknown_cover.person_hold_s)
OUTLINE_IOU = 0.5           # a class-agnostic proposal this much like a configured object's box outlines it

# A rule verdict is (Status, parent, confidence, EventType[, candidates]), or NO_CHANGE: the rule
# claims the object this update (later rules are skipped) but leaves the belief as it is and emits
# nothing, e.g. while waiting out reappear_wait_s after a container visit.
NO_CHANGE = ('NO_CHANGE',)

# Event emitted when an object seen before is observed again after being believed hidden or lost.
REAPPEAR_EVENT = {
    Status.UNDER: EventType.UNCOVERED,
    Status.INSIDE: EventType.TAKEN_OUT,
    Status.GONE: EventType.CORRECTED,
    Status.UNKNOWN: EventType.CORRECTED,
}


class WorldAPI(Protocol):
    """What readers (answers, llm, server, eval) may call. World and FakeWorld both fit."""

    def get(self, name: str) -> Entity: ...

    def resolve(self, name: str) -> tuple[Point | None, list[str]]: ...

    def history(self, name: str, n: int = 3) -> list[Event]: ...

    def state_json(self) -> dict: ...

    def place(self, name: str, now: float | None = None): ...   # -> core.room_types.Place


class World(ThingRules, RoomRules):
    def __init__(self, cfg: Union[Config, dict], events=None, embed=None):
        """embed(frame_img, box_px) -> unit vector | None: appearance evidence for things (e.g. a
        CLIP / DINO crop embedding; core.things.hsv_embed is an offline stand-in). None: no appearance."""
        self.cfg = cfg if isinstance(cfg, Config) else Config.from_dict(cfg)
        self._room_raw = None if isinstance(cfg, Config) else cfg.get('room_memory')   # Config: room defaults
        self.events = events
        self.embed = embed
        self.lock = threading.RLock()
        # Set by the app, not the rules; carried in state_json for the dashboard and Grok.
        self.online = False
        self.fps = 0.0
        self.laser = {'on': False, 'target': None, 'err_cm': None}
        # detector weights (core.detect.weights_info) and whether the camera feed is stale (main.py)
        self.perception: dict = {}
        self.reset()

    # ----- public API -------------------------------------------------------------------------

    def reset(self) -> None:
        with self.lock:
            self.entities = {n: Entity(name=n, kind=self.cfg.kind_of(n), confidence=0.0)
                             for n in self.cfg.names()}
            self._seen_t: dict[str, float] = {}         # obj -> monotonic t of last detection
            self._rest: dict[str, tuple] = {}           # obj -> where it last sat still (visible)
            self._rest_box: dict[str, tuple] = {}       # obj -> its box there
            self._carry: dict[str, tuple] = {}          # obj -> (hand, origin) while moved in view
            self._path: dict[str, deque] = {n: deque(maxlen=60) for n in self.entities}  # (t, pos)
            self._bits = {n: self._new_bits() for n in self.entities}
            self._present = {n: False for n in self.entities}
            self._confirmed: set[str] = set()      # names ever observed (debounced present / found)
            self._contacts: dict[str, dict[str, float]] = {n: {} for n in self.entities}  # obj -> hand -> t
            self._hands: dict[str, tuple[float, tuple, tuple]] = {}   # hand -> (last t, box_px, box_cm)
            self._hands_now: set[str] = set()
            self._now: float | None = None          # t of the latest update
            self._wall: float | None = None
            self._frame: Frame | None = None
            self._history: deque[Event] = deque(maxlen=HISTORY_MAX)   # used when no EventLog
            self._covers = relations.CoverMotion(self.cfg.cover_moved_min_cm)
            self._dwell = relations.DwellTracker(self.cfg.container_dwell_s)
            self._lifted_at: dict[str, float] = {}      # child -> when its cover moved off it
            self._box_px: dict[str, tuple] = {}         # obj -> last detected box_px
            self._bg = relations.BackgroundModel(self.cfg.bg_frames, self.cfg.bg_change_threshold)
            self._looks = relations.AppearanceMemory()
            self._look_px: dict[str, tuple] = {}        # obj -> box_px its remembered patch came from
            self._matched_t: dict[str, float] = {}      # obj -> last t its patch matched while undetected
            self._waiting: dict[str, float] = {}        # obj -> when an absence no rule explained began
            self._bg_t: float | None = None             # last background / appearance refresh
            self._gray_img = None                       # this update's grey frame, made on demand
            self._ucfg = UnknownCoverConfig.from_config(self.cfg)
            self._bands = SurroundMemory(self._ucfg)    # the band of table around each object (unknown covers)
            self._laid_wait: dict[str, float] = {}      # obj -> since when its absence waits on the band
            self._person_wait: dict[str, float] = {}    # obj -> since when a person has hidden its spot
            self._people: list = []                     # this update's person boxes (cm), from the proposer
            self._bare_at: dict[str, float] = {}        # obj UNDER 'unknown' -> since when its band looks bare
            self._unknown_uncovered: set[str] = set()   # last seen again from under 'unknown'
            self._reset_things()
            self._reset_room()

    def get(self, name: str) -> Entity:
        with self.lock:
            return self.entities[name]

    def update(self, dets: Detections, frame: Frame) -> list[Event]:
        with self.lock:
            self._decay(dets.t)
            self._now, self._wall, self._frame = dets.t, frame.wall if frame else dets.t, frame
            self._gray_img = None
            self._batch_things = [d for d in dets.items if d.cls == 'thing']
            self._people = [d.box_cm for d in getattr(dets, 'people', None) or ()]
            out: list[Event] = []
            seen = self._best_detections(dets.items, dets.hands)
            self._track_covers(seen)
            out += self._associate_things(dets, seen)     # adds thing:N -> proposal to seen
            fell: list[str] = []
            for name, ent in self.entities.items():
                was = self._present[name]
                self._debounce(name, ent, seen.get(name))
                if self._present[name] and (not was or ent.status != Status.VISIBLE):
                    out += self._observe(name, ent)
                elif was and not self._present[name]:
                    fell.append(name)
            self._track_hands(dets.hands)
            self._record_contacts()
            for name, ent in self.entities.items():
                if name in seen and self._present[name] and ent.status == Status.VISIBLE:
                    out += self._track_motion(name, ent)
            for name in fell:
                if self.entities[name].zone == 'table':    # other zones: overhead absence means nothing
                    out += self._disappear(name, self.entities[name])
            self._track_dwell(dets.hands)
            for name, ent in self.entities.items():
                if ent.status == Status.HELD and not self._reappearing(name):
                    out += self._update_held(name, ent)
            out += self._things_laid_over()
            out += self._lifted_covers()
            self._refresh_images(seen, dets.hands)
            self._remember_bands(seen, dets.hands)
            self._after_things(out)
            return out

    def resolve(self, name: str) -> tuple[tuple[float, float] | None, list[str]]:
        """Follow parent links through entity names (up to max_nesting). Returns the outermost
        entity's last position and the chain, e.g. ((60, 40), ['keys', 'box', 'notebook']).
        Children never move themselves: a carried or moved parent carries them by this lookup.
        An outermost entity off the table (a room zone) gives (None, chain): no table-cm path may use its
        stale table spot. Keyed on the zone, not pos_cm, which one table flicker can rewrite."""
        with self.lock:
            pos, chain = relations.resolve_chain(name, self.entities, self.cfg.max_nesting)
            if chain and self.entities[chain[-1]].zone != 'table':
                return None, chain
            return pos, chain

    def state_json(self) -> dict:
        with self.lock:
            return {'t': self._wall, 'online': self.online, 'fps': self.fps,
                    'entities': [self._entity_json(ent) for ent in self.entities.values()
                                 if ent.merged_into is None],
                    'edges': self._edges(), 'laser': dict(self.laser), 'perception': dict(self.perception),
                    'aliases': dict(self._aliases),
                    'merged': {n: e.merged_into for n, e in self.entities.items() if e.merged_into},
                    'room': self.room_json()}

    def box_px(self, name: str) -> tuple | None:
        """The last detected box of name in the table view's px (what its event snapshots show), or None."""
        with self.lock:
            return self._box_px.get(name)

    def history(self, name: str, n: int = 3) -> list[Event]:
        """Latest n events for name, newest first."""
        with self.lock:
            names = {name} | {m for m, e in self.entities.items() if e.merged_into == name}
            if self.events is not None:
                evs = [ev for m in names for ev in self.events.last(m, n)]
                return sorted(evs, key=lambda ev: ev.wall, reverse=True)[:n] if len(names) > 1 else evs
            return [ev for ev in reversed(self._history) if ev.obj in names][:n]

    def observe_external(self, name: str, pos_cm, zone: str) -> list[Event]:
        """Stretch: the search camera found the object off the table (or anywhere)."""
        with self.lock:
            if self._now is None:
                self._now, self._wall = time.monotonic(), time.time()
            self._frame = None
            ent = self.entities[name]
            ent.status, ent.parent, ent.candidates, ent.confidence, ent.edge = Status.VISIBLE, None, [], 1.0, None
            ent.pos_cm, ent.box_cm, ent.zone, ent.last_seen = tuple(pos_cm), None, zone, self._wall
            self._seen_t[name] = self._now
            ent.pre_pickup_pos, ent.held_since = None, None
            self._confirmed.add(name)
            return [self._emit(name, EventType.FOUND, to_cm=ent.pos_cm)]

    def _entity_json(self, ent: Entity) -> dict:
        resolved, _ = self.resolve(ent.name)
        return self._thing_json(ent, {
            'name': ent.name, 'kind': ent.kind, 'status': ent.status.value, 'parent': ent.parent,
            'pos_cm': _list(ent.pos_cm), 'resolved_cm': _list(resolved), 'confidence': ent.confidence,
            'candidates': list(ent.candidates), 'last_seen': ent.last_seen, 'zone': ent.zone,
            'edge': ent.edge})

    def _edges(self) -> list[list[str]]:
        edges = []
        for ent in self.entities.values():
            if ent.merged_into is not None:
                continue
            if ent.parent in self.entities or _is_hand(ent.parent):
                edges.append([ent.name, ent.status.value, ent.parent])
            elif ent.status == Status.VISIBLE:
                edges.append([ent.name, 'ON', ent.zone])
        return edges

    # ----- rule 1: debounce ---------------------------------------------------------------------

    def _best_detections(self, items: list[Detection], hands=()) -> dict[str, Detection]:
        best: dict[str, Detection] = {}
        for d in items:
            if d.cls in self.entities and d.conf >= self.cfg.threshold(d.cls):
                if self._label_on_a_thing(d, hands) or self._label_left_behind(d):
                    continue
                if d.cls not in best or d.conf > best[d.cls].conf:
                    best[d.cls] = d
        return best

    def _label_left_behind(self, d: Detection) -> bool:
        """The object's label read moved_min_cm or more from where it lies while its remembered pixels
        still match there: the detector is calling a neighbour by its name (a label swap), the object
        itself has not moved. A real move leaves the old spot looking different. Targets only, and only
        a label clear of the old box: a big, plain cover or container slid part-way can still look
        alike over its old spot."""
        ent = self.entities[d.cls]
        return (ent.kind == 'target' and ent.status == Status.VISIBLE and ent.pos_cm is not None
                and ent.box_cm is not None and geom.dist(ent.pos_cm, d.center_cm) >= self.cfg.moved_min_cm
                and geom.iou(ent.box_cm, d.box_cm) < LABEL_LEFT_IOU and self._still_there(d.cls))

    def _label_on_a_thing(self, d: Detection, hands) -> bool:
        """A configured label read on a thing that was already in place while that object was seen
        elsewhere, with no hand at the spot: two objects side by side, so the detector is misnaming the
        thing (a fine-tuned detector calls unknown things by its few classes), not the object moving
        onto it. A thing that appeared only after the object was last seen may be the object itself."""
        ent, seen = self.entities[d.cls], self._seen_t.get(d.cls)
        if (seen is None or ent.pos_cm is None or not hasattr(self, '_placed_t')
                or geom.dist(ent.pos_cm, d.center_cm) < self.cfg.moved_min_cm
                or any(geom.overlap_frac(h.box_cm, d.box_cm) > 0 for h in hands)):
            return False
        in_place = self._visible_things() + [    # or hidden where it lay (a blanket just lifted off it)
            n for n in self._things if self.entities[n].status == Status.UNDER
            and self.entities[n].merged_into is None and self.entities[n].box_cm is not None]
        return any(geom.iou(self.entities[n].box_cm, d.box_cm) >= LABEL_ON_THING_IOU
                   and self._placed_t.get(n, seen) + ESTABLISHED_S <= seen
                   for n in in_place)

    def _new_bits(self, bits=()) -> Presence:
        """A presence window: the last present_n updates, or present_n updates' time at presence_hz."""
        return Presence(bits, self.cfg.present_n, self.cfg.presence_hz)

    def _debounce(self, name: str, ent: Entity, det: Detection | None) -> None:
        """Push this batch's presence bit; refresh position while detected (even before a flip)."""
        bits = self._bits[name]
        bits.push(self._now, det is not None)
        if det is not None:
            ent.pos_cm, ent.box_cm, ent.last_seen = det.center_cm, det.box_cm, self._wall
            self._seen_t[name] = self._now
            self._box_px[name] = det.box_px
        count = bits.hits()
        if count >= self.cfg.present_k - 1e-9:
            self._present[name] = True
        elif count <= self.cfg.absent_max + 1e-9:
            self._present[name] = False

    # ----- rule 2: observation wins -------------------------------------------------------------

    def _observe(self, name: str, ent: Entity) -> list[Event]:
        prev, origin, known = ent.status, ent.pre_pickup_pos, name in self._confirmed
        from_room = ent.zone != 'table' and name in self._room
        # seen again where something unknown lay over it: found in place, not put down (teach, _refound)
        (self._unknown_uncovered.add if prev == Status.UNDER and ent.parent == 'unknown'
         else self._unknown_uncovered.discard)(name)
        ent.status, ent.parent, ent.candidates, ent.confidence, ent.edge = Status.VISIBLE, None, [], 1.0, None
        ent.zone, ent.pre_pickup_pos, ent.held_since = 'table', None, None
        self._confirmed.add(name)
        self._rest[name], self._rest_box[name] = ent.pos_cm, ent.box_cm
        self._carry.pop(name, None)
        self._waiting.pop(name, None)
        self._settled_band(name)
        if from_room:                       # back from a room zone: Return (core/room_world.py)
            return self._room_return(name, ent)
        if prev == Status.HELD:
            moved = origin is not None and geom.dist(ent.pos_cm, origin) >= self.cfg.moved_min_cm
            etype = EventType.MOVED if moved else EventType.PUT_BACK
            return [self._emit(name, etype, from_cm=origin, to_cm=ent.pos_cm)]
        if known and prev in REAPPEAR_EVENT:
            return [self._emit(name, REAPPEAR_EVENT[prev], to_cm=ent.pos_cm)]
        return []

    # ----- per-update bookkeeping: cover motion, container dwell, background / appearance --------

    def _track_covers(self, seen: dict[str, Detection]) -> None:
        for name in self.cfg.names('cover'):
            if name in seen:
                self._covers.update(name, seen[name].box_cm, self._now)

    def _track_dwell(self, hands: list[Detection]) -> None:
        containers = {c: self.entities[c].box_cm for c in self.cfg.names('container')
                      if self.entities[c].status == Status.VISIBLE and self.entities[c].box_cm is not None}
        things = self._thing_containers()          # large things hold too (core/things.py)
        self._track_crossings(hands, things)
        self._dwell.update(self._now, hands, {**containers, **things})

    def _gray(self):
        """This update's frame in grayscale, converted at most once per update; None without an image."""
        img = self._frame.img if self._frame is not None else None
        if self._gray_img is None and img is not None:
            self._gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
        return self._gray_img

    def _refresh_images(self, seen: dict[str, Detection], hands: list[Detection]) -> None:
        """Background and appearance refresh, at most every bg_update_every_s: a background update
        costs up to ~25 ms at 320 px against a 5 ms per-update budget. Object and hand boxes are
        never background; a patch is only remembered with no hand over the object."""
        if self._frame is None or self._frame.img is None:
            return
        if self._bg_t is not None and self._now - self._bg_t < self.cfg.bg_update_every_s:
            return
        self._bg_t, gray = self._now, self._gray()
        hand_px = [h.box_px for h in hands]
        self._bg.update(gray, hand_px + self._object_boxes_px())
        for name, det in seen.items():
            hand_over = any(geom.intersection(det.box_px, h) for h in hand_px)
            if self.entities[name].status == Status.VISIBLE and not hand_over:
                self._looks.remember(name, gray, det.box_px)
                self._look_px[name] = det.box_px

    def _object_boxes_px(self) -> list[tuple]:
        """Last boxes of objects believed present or detected within the debounce window."""
        return [box for n, box in self._box_px.items() if self._present[n] or any(self._bits[n])]

    # ----- rule 3: contacts ---------------------------------------------------------------------

    def _track_hands(self, hands: list[Detection]) -> None:
        self._hands_now = {h.cls for h in hands}
        for h in hands:
            self._hands[h.cls] = (self._now, h.box_px, h.box_cm)

    def _record_contacts(self) -> None:
        for name, ent in self.entities.items():
            if ent.box_cm is None:
                continue
            for hid in self._hands_now:
                if geom.overlap_frac(self._hands[hid][2], ent.box_cm) >= self.cfg.contact_overlap:
                    self._contacts[name][hid] = self._now

    # ----- rule 3b: moved in plain view -----------------------------------------------------------

    def _track_motion(self, name: str, ent: Entity) -> list[Event]:
        """A carried object often stays detected, so it never goes absent and rule 4 never fires.
        Leaving its resting spot by moved_min_cm with a hand on it is a pick-up; untouched for
        settle_s and still within settle_cm since the last touch is a set-down (MOVED / PUT_BACK).
        A box that only shrank inside the one the object rests in is the object partly hidden (an arm
        over it), not moved: its centre shifts, the object does not."""
        pos = ent.pos_cm
        self._path[name].append((self._now, pos))
        rest = self._rest.setdefault(name, pos)
        self._rest_box.setdefault(name, ent.box_cm)
        touches = self._contacts[name]
        carry = self._carry.get(name)
        if carry is None:
            if geom.dist(pos, rest) < self.cfg.moved_min_cm or self._partly_hidden(name, ent):
                return []
            recent = sorted(((t, h) for h, t in touches.items()
                             if t >= self._now - self.cfg.contact_window_s), reverse=True)
            if not recent:                  # shifted with no hand (e.g. re-detected): new rest spot
                self._rest[name], self._rest_box[name] = pos, ent.box_cm
                self._settled_band(name)
                return []
            self._carry[name] = (recent[0][1], rest)
            return [self._emit(name, EventType.PICKED_UP, from_cm=rest, parent=recent[0][1])]
        last_touch = max(touches.values(), default=-1e18)
        if self._now - last_touch < self.cfg.settle_s:
            return []
        since = [p for t, p in self._path[name] if t > last_touch]
        if not since or any(geom.dist(p, pos) > self.cfg.settle_cm for p in since):
            return []
        origin = self._carry.pop(name)[1]
        self._rest[name], self._rest_box[name] = pos, ent.box_cm
        self._settled_band(name)
        etype = EventType.MOVED if geom.dist(pos, origin) >= self.cfg.moved_min_cm else EventType.PUT_BACK
        return [self._emit(name, etype, from_cm=origin, to_cm=pos)]

    def _partly_hidden(self, name: str, ent: Entity) -> bool:
        rest = self._rest_box.get(name)
        return (rest is not None and ent.box_cm is not None and geom.area(ent.box_cm) < geom.area(rest)
                and geom.overlap_frac(rest, ent.box_cm) >= PARTLY_HIDDEN_INSIDE)

    # ----- rule 4: disappearance ----------------------------------------------------------------

    def _disappear(self, name: str, ent: Entity) -> list[Event]:
        if self._still_there(name):
            self._present[name] = True      # detector miss: re-checked each update while it lasts
            self._matched_t[name] = self._now
            return []
        if self._hand_over(ent):            # the hand hides the spot (a wave, or a grab in progress):
            self._present[name] = True      # decided once it moves on, by the rules below
            return []
        if self._person_over(name, ent):    # someone is in front of it: decided once they move away
            self._present[name] = True
            return []
        waiting = self._now - self._waiting.get(name, -1e18) <= self.cfg.lost_grace_s
        if not waiting:                     # already unexplained: a hand that comes after did not take it
            slid_over = self._slid_over_by_hand(name, ent)
            if slid_over is not None:
                return self._apply_verdict(name, ent, slid_over)
            if self._touched(name):         # a hand holding an unknown cover touches what it covers
                laid = self._laid_over(name, ent)
                if laid is not None:
                    return self._defer_or_apply(name, ent, laid)
            held = self._picked_up(name, ent)
            if held or ent.status == Status.HELD:
                return held
        hidden = self._hidden_by(name, ent, ent.box_cm, self._now, self._frame)
        if hidden is not None:
            return self._apply_verdict(name, ent, hidden)
        if self._now - self._last_evidence(name) < self.cfg.lost_grace_s:
            if not waiting:                 # unexplained: an arm no detector saw, or a detector miss
                self._waiting[name] = self._now
            self._present[name] = True      # re-checked each update; seen again meanwhile: nothing logged
            return []
        self._waiting.pop(name, None)
        self._depart_anchor = self._last_evidence(name)   # room memory: last seen, not the grace's end
        laid = self._laid_over(name, ent, settled=True)     # untouched, still unseen: an unknown cover?
        if laid is not None and laid is not NO_CHANGE:
            return self._apply_verdict(name, ent, laid)       # (COVERED by 'unknown' is a departure too)
        return self._lose(name, ent)

    def _still_there(self, name: str) -> bool:
        """Review fix: the pixels still match the remembered patch, so the detector just missed it.
        Compared where the patch was taken, not at the last box: detection boxes wobble a few px and
        a shifted crop decorrelates. Checked before the hand rule too: a touch within the contact
        window followed by a detector miss must not read as a pick-up."""
        gray, box = self._gray(), self._look_px.get(name)
        return self._outlined(name) or (gray is not None and box is not None
                                        and self._looks.still_there(name, gray, box, self.cfg.appearance_match))

    def _outlined(self, name: str) -> bool:
        """A visible configured object the detector no longer names while a class-agnostic proposal still
        outlines its box (OUTLINE_IOU), with no hand on it: it is still there. On blanket_1t the phone's
        screen lit up, the fine-tuned detector called the tape roll 'phone' instead, the patch no longer
        matched, and the phone was lost and reborn as a thing, though YOLOE still proposed its box."""
        ent = self.entities[name]
        if is_thing(name) or ent.status != Status.VISIBLE or ent.box_cm is None:
            return False
        if any(geom.overlap_frac(self._hands[h][2], ent.box_cm) > 0 for h in self._hands_now):
            return False
        return any(geom.iou(d.box_cm, ent.box_cm) >= OUTLINE_IOU for d in getattr(self, '_batch_things', ()))

    def _last_evidence(self, name: str) -> float:
        """When the object was last detected or its patch last matched. An absence no rule explains is
        not a loss until lost_grace_s after this: on the rig the detector drops objects while an arm is
        near them and a dim-light arm is often not detected as a hand, so the patch check fails whenever
        the arm overlaps the box as absence is declared, and untouched objects flickered LOST_TRACK ->
        CORRECTED (shell_1: unseen up to 1.9 s, then found in place). Meanwhile the object stays VISIBLE
        where it was, the cover and background rules still apply, and hands arriving later are ignored:
        the absence came first (shell_1: one frame of a hand over the already unseen phone)."""
        return max(self._seen_t.get(name, -1e18), self._matched_t.get(name, -1e18))

    def _lose(self, name: str, ent: Entity) -> list[Event]:
        """UNKNOWN at the last known position; confidence is left as it was."""
        ent.status, ent.parent, ent.candidates, ent.held_since = Status.UNKNOWN, None, [], None
        return [self._emit(name, EventType.LOST_TRACK)]

    def _person_over(self, name: str, ent: Entity) -> bool:
        """WS2 proposal (room demo, Sun 27 Sep): a person box (YOLOE 'person', 'arm', a worn shoe; from the
        proposer) over PERSON_OVER of the object's last box hides the spot, as a moving hand does. Live at
        03:00, 52 'COVERED by something' in 20 min on 13 real props were a torso, a head or an arm leaning
        over the table with no hand box (median 3 s, 13 of them 10 s or more): the spot is not seen, so
        nothing is decided until they move away, for at most person_hold_s (someone parked in front of
        it: then the other rules decide, as before)."""
        if ent.box_cm is None or not any(geom.overlap_frac(p, ent.box_cm) >= PERSON_OVER for p in self._people):
            self._person_wait.pop(name, None)
            return False
        since = self._person_wait.get(name)
        if since is None or since < self._last_evidence(name):
            since = self._person_wait[name] = self._now
        hold = (getattr(self.cfg, 'unknown_cover', None) or {}).get('person_hold_s', PERSON_HOLD_S)
        return self._now - since <= float(hold)

    def _hand_over(self, ent: Entity) -> bool:
        """A hand in view, still moving (a wave passing over, or a grab already on its way out), covers
        any of the object's last box: the spot cannot be seen (nor its pixels checked) yet. A hand at
        rest on it is a grasp: the pick-up rule decides at once."""
        return ent.box_cm is not None and any(
            geom.overlap_frac(self._hands[h][2], ent.box_cm) > 0 and self._in_transit(h) for h in self._hands_now)

    def _in_transit(self, hid: str) -> bool:
        trail = getattr(self, '_trail', {}).get(hid) or ()
        now_c = geom.center(self._hands[hid][2])
        return any(self._now - t <= TRANSIT_S and geom.dist(geom.center(b), now_c) >= TRANSIT_CM for t, b in trail)

    def _touched(self, name: str) -> bool:
        since = self._seen_t[name] - self.cfg.contact_window_s
        return any(t >= since for t in self._contacts[name].values())

    def _picked_up(self, name: str, ent: Entity) -> list[Event]:
        # Debounce declares absence ~0.6-0.9 s late, so the window is anchored at last_seen, not now.
        # The hand need not still be in view (grab-and-leave): if it has gone, the HELD pass later in
        # this same update applies the hand-lost rule (edge -> EXITED_VIEW, else LOST_TRACK).
        since = self._seen_t[name] - self.cfg.contact_window_s
        hands = sorted(((t, h) for h, t in self._contacts[name].items() if t >= since), reverse=True)
        carry = self._carry.pop(name, None)       # already picked up in view: no second event
        if not hands:
            return []
        ent.status, ent.parent, ent.candidates = Status.HELD, hands[0][1], [h for _, h in hands[1:]]
        ent.confidence = self.cfg.conf_held * (self.cfg.ambiguity_penalty if ent.candidates else 1.0)
        ent.pre_pickup_pos, ent.held_since = (carry[1] if carry else ent.pos_cm), self._now
        if carry:
            return []
        return [self._emit(name, EventType.PICKED_UP, from_cm=ent.pos_cm, parent=ent.parent)]

    def _slid_over_by_hand(self, name: str, ent: Entity):
        """The cover rule wins over the hand rule when every hand that touched the object is resting
        on the cover that now lies over it: the touch was the cover being slid across, not a pick-up."""
        since = self._seen_t[name] - self.cfg.contact_window_s
        hands = [h for h, t in self._contacts[name].items() if t >= since]
        if not hands:
            return None
        verdict = self._covered(name, ent.box_cm, self._now)
        if verdict is None:
            return self._thing_put_over(name, ent, hands, since)
        cover = self._covers.state(verdict[1])
        on_cover = all(geom.overlap_frac(cover.box_cm, self._hands[h][2]) >= self.cfg.contact_overlap
                       for h in hands)
        return verdict if on_cover else None

    def _thing_put_over(self, name: str, ent: Entity, hands: list[str], since: float):
        """WS2, room mode (prop labels off, so a notebook or a cup is an unlabelled thing:N; additive to
        the named-cover rule above): a thing in view now lying over the object's box (cover_overlap of it,
        THING_COVER_AREA times its footprint), held by a hand that touched the object or touched by one
        since just before the object was last seen, was put over it by that hand: UNDER it, not picked up. A
        real pick-up leaves nothing lying there. A cup is too small for the band rule (_laid_over: it
        leaves most of the band bare), so without this the shell game read as the keys carried off."""
        box = ent.box_cm
        if box is None:
            return None
        best = []
        for n in self._things:
            e = self.entities[n]
            if n == name or e.merged_into is not None or e.box_cm is None or self._seen_t.get(n, -1e18) < since \
                    or e.status not in (Status.VISIBLE, Status.HELD) \
                    or geom.area(e.box_cm) < THING_COVER_AREA * geom.area(box):
                continue
            by_hand = (e.parent in hands if e.status == Status.HELD
                       else any(self._contacts[n].get(h, -1e18) >= since for h in hands))
            ov = geom.overlap_frac(e.box_cm, box)
            if by_hand and ov >= self.cfg.cover_overlap:
                best.append((ov, n))
        if not best:
            return None
        return (Status.UNDER, max(best)[1], self.cfg.conf_under, EventType.COVERED)

    def _hidden_by(self, name, ent, box_cm, now, frame):
        """Cover rule, then background-change rule. Returns a verdict (see NO_CHANGE) when the
        object is judged hidden rather than lost, else None."""
        return self._covered(name, box_cm, now) or self._background_changed(name)

    def _covered(self, name, box_cm, now):
        """A present cover that moved in the last cover_moved_window_s now lies over the last box."""
        states = {c: self._covers.state(c) for c in self.cfg.names('cover') if c != name and self._present[c]}
        covers = {c: st for c, st in states.items() if st is not None}
        parents = relations.covering_candidates(box_cm, covers, now, self.cfg.cover_overlap,
                                                self.cfg.cover_moved_window_s)
        if not parents:
            return None
        return self._hidden_verdict(parents + self._visited_containers(name), self.cfg.conf_under)

    def _background_changed(self, name: str):
        """Something undetected now lies where the object was: the background model has seen the
        bare table in (most of) that box and the pixels there differ from it."""
        gray, box = self._gray(), self._box_px.get(name)
        if gray is None or box is None or not self._bg.ready:
            return None
        if self._bg.known(box) and self._bg.changed(gray, box):
            return (Status.UNDER, 'unknown', self.cfg.conf_under_unknown, EventType.COVERED)
        return None

    def _visited_containers(self, name: str) -> list[str]:
        """Containers entered by a hand after it last touched this object: a rival parent when a
        cover also qualifies. (A touch within contact_window_s of last_seen is a pick-up instead.)"""
        out: list[str] = []
        for hid, touched in self._contacts[name].items():
            for container, _ in self._holding_visits(name, hid, touched):
                if container not in out:
                    out.append(container)
        return out

    def _hidden_verdict(self, parents: list[str], confidence: float):
        """UNDER the first parent; any others are ambiguity candidates at a confidence penalty."""
        if len(parents) > 1:
            confidence *= self.cfg.ambiguity_penalty
        return (Status.UNDER, parents[0], confidence, EventType.COVERED, parents[1:])

    # ----- rule 4b: a cover no detector knows (a blanket, a jacket, a napkin) ----------------------
    # The hands laying it touch what it covers, so the hand rule alone reads each touched object as
    # picked up. Once no hand hides the spot, a real pick-up leaves the band of table around it as it
    # was; a cover leaves it changed all round (core/surround.py). That, settled for settle_s, is UNDER
    # the cover: a named cover that qualifies, a thing laid over it, else 'unknown'.

    def _laid_over(self, name: str, ent: Entity, settled: bool = False):
        """A verdict once the band around the object's spot looks covered and has been at rest for
        settle_s (a cover lying there; an arm keeps moving); NO_CHANGE while it settles or hands hide
        too much of it to tell (at most wait_max_s); None when the band shows table (or there is no
        memory of it): the other rules decide. settled: the lost grace has run out, decide now."""
        uc, img = self._ucfg, (self._frame.img if self._frame is not None else None)
        band = self._bands.memory(name) if uc.enabled and img is not None else None
        if band is None:
            return None
        now, hands = self._now, [self._hands[h][1] for h in self._hands_now]
        self._bands.sample(name, img, now)
        if not settled and now - self._laid_wait.setdefault(name, now) > uc.wait_max_s:
            return None
        look = self._bands.look(name, img, hands)
        if look is False:
            self._laid_wait.pop(name, None)
            return None
        if look is None or not self._bands.still(name, img, now, uc.settle_s, hands):
            return None if settled else NO_CHANGE
        self._laid_wait.pop(name, None)
        ent.box_cm, ent.pos_cm = band.box_cm, geom.center(band.box_cm)    # it lies where it rested
        self._box_px[name] = band.box_px
        return self._under_verdict(name, band.box_cm)

    def _defer_or_apply(self, name: str, ent: Entity, laid) -> list[Event]:
        if laid is NO_CHANGE:
            self._present[name] = True      # re-checked each update until the band settles
            return []
        self._waiting.pop(name, None)
        return self._apply_verdict(name, ent, laid)

    def _under_verdict(self, name: str, box_cm):
        named = self._covered(name, box_cm, self._now)
        if named is not None:
            return named
        thing = self._thing_over(name, box_cm)
        if thing is not None:
            return (Status.UNDER, thing, self.cfg.conf_under, EventType.COVERED)
        return (Status.UNDER, 'unknown', self.cfg.conf_under_unknown, EventType.COVERED)

    def _thing_over(self, name: str, box_cm) -> str | None:
        """A visible thing lying over box_cm (cover_overlap of it, at least THING_COVER_AREA times its
        footprint), put down there no earlier than just before the object was last seen (a mat the object
        already lay on was there before), and established: untouched for ESTABLISHED_S since. A napkin the
        proposer sees whole qualifies; a piece of a blanket still in a hand does not (on the rig, one
        taken as the cover made its child 'come out' at the next piece of blanket the proposer saw)."""
        if box_cm is None:
            return None
        since = self._seen_t.get(name, -1e18) - self.cfg.contact_window_s
        best = []
        for n in self._visible_things():
            b, placed = self.entities[n].box_cm, self._placed_t.get(n, -1e18)
            if n == name or geom.area(b) < THING_COVER_AREA * geom.area(box_cm) or placed < since \
                    or self._now - placed < ESTABLISHED_S or max(self._contacts[n].values(), default=-1e18) > placed:
                continue
            ov = geom.overlap_frac(b, box_cm)
            if ov >= self.cfg.cover_overlap:
                best.append((ov, n))
        return max(best)[1] if best else None

    def _things_laid_over(self) -> list[Event]:
        """An object UNDER 'unknown' with a thing now seen lying over it: that thing is its cover."""
        out = []
        for name, ent in self.entities.items():
            if ent.status == Status.UNDER and ent.parent == 'unknown':
                thing = self._thing_over(name, ent.box_cm)
                if thing is not None:
                    ent.parent, ent.confidence, ent.candidates = thing, self.cfg.conf_under, []
                    self._bare_at.pop(name, None)
                    out.append(self._emit(name, EventType.COVERED, parent=thing))
        return out

    def _unknown_cover_lifted(self, name: str, ent: Entity) -> list[Event]:
        """The band around an object UNDER 'unknown' looks as remembered again (the cover lifted off). If
        the object's own remembered pixels are back at its spot, it is there (UNCOVERED, as a detector
        miss). If it has not been seen within reappear_wait_s, it went with the cover, so it is lost,
        with lifted_cover_penalty, like one under a named cover that was lifted."""
        img = self._frame.img if self._frame is not None else None
        hands = [self._hands[h][1] for h in self._hands_now]
        if img is None or not self._bands.bare(name, img, hands):
            self._bare_at.pop(name, None)
            return []
        if self._still_there(name):         # uncovered and its own pixels are back: it is there
            self._present[name], self._matched_t[name] = True, self._now
            return self._observe(name, ent)
        since = self._bare_at.setdefault(name, self._now)
        if self._now - since < self.cfg.reappear_wait_s or self._reappearing(name):
            return []
        del self._bare_at[name]
        ent.confidence *= self.cfg.lifted_cover_penalty
        return self._lose(name, ent)

    def _settled_band(self, name: str) -> None:
        """Seen again, or set down: no cover is settling over it; a band remembered elsewhere is stale."""
        self._laid_wait.pop(name, None)
        self._bare_at.pop(name, None)
        self._bands.drop_samples(name)
        band, pos = self._bands.memory(name), self.entities[name].pos_cm
        if band is not None and pos is not None and geom.dist(geom.center(band.box_cm), pos) >= self.cfg.moved_min_cm:
            self._bands.forget(name)

    def _remember_bands(self, seen: dict[str, Detection], hands: list[Detection]) -> None:
        img = self._frame.img if self._frame is not None else None
        if not self._ucfg.enabled or img is None:
            return
        hands_px = [h.box_px for h in hands]
        for name, det in seen.items():
            ent = self.entities[name]
            if ent.status == Status.VISIBLE and self._present[name] and ent.zone == 'table':
                self._bands.observe(name, img, det.box_px, det.box_cm, self._now, hands_px)
        for name, ent in self.entities.items():     # not seen now: keep views, to tell a cover at rest
            if name not in seen and ent.status in (Status.VISIBLE, Status.HELD) and ent.zone == 'table':
                self._bands.sample(name, img, self._now)

    # ----- rule 5: HELD objects ------------------------------------------------------------------

    def _reappearing(self, name: str) -> bool:
        """Present, or seen often enough lately that the debounce may still flip to present. A put-down
        is judged by observation, so the hand rules wait rather than race it (a hand that drops an
        object and leaves at once would otherwise read as lost before the object is confirmed)."""
        return self._present[name] or self._bits[name].hits() > self.cfg.absent_max + 1e-9

    def _update_held(self, name: str, ent: Entity) -> list[Event]:
        laid = self._laid_over(name, ent)       # it was a cover laid over it, not a pick-up
        if laid is NO_CHANGE:
            return []
        if laid is not None:
            return self._apply_verdict(name, ent, laid)
        inside = self._check_inside(name, ent, self._now)
        if inside is not None:
            return self._apply_verdict(name, ent, inside)
        hand_t, hand_px, _ = self._hands[ent.parent]
        if self._now - hand_t > self.cfg.hand_lost_s:
            self._depart_anchor = hand_t    # room memory: it left when its hand was last seen, not now
            edge = self._edge_of(hand_px)
            if edge:
                ent.status, ent.parent, ent.candidates, ent.edge = Status.GONE, None, [], edge
                ent.held_since = None
                return [self._emit(name, EventType.EXITED_VIEW, edge=edge)]
            return self._lose(name, ent)
        if self._now - ent.held_since > self.cfg.held_timeout_s:
            return self._lose(name, ent)
        return []

    def _check_inside(self, name, ent, now):
        """Container dwell -> INSIDE: the holding hand's latest completed container visit since the
        object left the table ended reappear_wait_s ago and the object is still absent. While that
        wait runs, NO_CHANGE holds off the hand-lost / timeout rules; with no visit, None falls
        through to them."""
        # Anchored at last_seen, not held_since: debounce declares HELD 0.6-0.9 s late, and a quick
        # drop into the box can be over by then.
        visits = self._holding_visits(name, ent.parent, self._seen_t[name])
        if not visits:
            return None
        container, exit_t = visits[-1]
        if now - exit_t >= self.cfg.reappear_wait_s:
            return (Status.INSIDE, container, self.cfg.conf_inside, EventType.PUT_INSIDE)
        return NO_CHANGE        # likely dropped in: wait for a reappearance before the hand rules

    def _apply_verdict(self, name: str, ent: Entity, verdict) -> list[Event]:
        """verdict = (Status, parent, confidence, EventType[, candidates]) or NO_CHANGE."""
        if verdict is NO_CHANGE:
            return []
        status, parent, confidence, etype, *rest = verdict
        ent.status, ent.parent, ent.confidence = status, parent, confidence
        ent.candidates, ent.held_since = list(rest[0]) if rest else [], None
        return [self._emit(name, etype, parent=ent.parent)]

    def _edge_of(self, box_px) -> str | None:
        """Frame side the box lies within edge_margin of (the nearest one if several), else None."""
        w, h = self.cfg.frame_size_px
        mx, my = self.cfg.edge_margin * w, self.cfg.edge_margin * h
        gaps = {'left': box_px[0], 'right': w - box_px[2], 'top': box_px[1], 'bottom': h - box_px[3]}
        margin = {'left': mx, 'right': mx, 'top': my, 'bottom': my}
        near = [(g, side) for side, g in gaps.items() if g <= margin[side]]
        return min(near)[1] if near else None

    # ----- rule 5b: lifted cover -------------------------------------------------------------------

    def _lifted_covers(self) -> list[Event]:
        """A child UNDER a named cover whose cover has left its last box (or left view) and that has
        not reappeared within reappear_wait_s of that moment is lost, at a confidence penalty."""
        out: list[Event] = []
        for name, ent in self.entities.items():
            cover = ent.parent if ent.status == Status.UNDER else None
            if cover == 'unknown':
                out += self._unknown_cover_lifted(name, ent)
                continue
            if cover not in self.entities or not self._cover_lifted(cover, ent):
                self._lifted_at.pop(name, None)
                continue
            if self._cover_in_hand(cover):  # WS2: a cover still in a hand is decided once it is put down
                self._lifted_at.pop(name, None)
                continue
            since = self._lifted_at.setdefault(name, self._now)
            if self._now - since >= self.cfg.reappear_wait_s and not self._reappearing(name):
                del self._lifted_at[name]
                went = self._went_with_cover(name, ent, cover)
                if went:
                    out += went
                    continue
                ent.confidence *= self.cfg.lifted_cover_penalty
                out += self._lose(name, ent)
        return out

    def _cover_in_hand(self, cover: str) -> bool:
        return self.entities[cover].status == Status.HELD or cover in self._carry

    def _went_with_cover(self, name: str, child: Entity, cover: str) -> list[Event]:
        """WS2 (the shell game): a thing cover (a cup, an unlabelled notebook; configured covers keep their
        rule) moved off the child's spot and now rests elsewhere on the table,
        the child was not seen again, and nothing is at its spot (neither its remembered pixels nor any
        proposal): it went along under the cover (a cup slid across the table takes the keys with it). It
        stays UNDER, now where the cover is (MOVED). Anything else, or no cover on the table: lost, as
        before."""
        c = self.entities[cover]
        if not is_thing(cover) or c.status != Status.VISIBLE or not self._present[cover] or c.box_cm is None \
                or child.box_cm is None:
            return []                       # a configured cover keeps its rule: lifted, not seen, lost
        if self._still_there(name) or any(geom.overlap_frac(d.box_cm, child.box_cm) >= 0.5
                                          for d in getattr(self, '_batch_things', ())):
            return []                       # something is at the old spot: not a slide-along
        frm = child.pos_cm
        cx, cy = geom.center(c.box_cm)
        w, h = child.box_cm[2] - child.box_cm[0], child.box_cm[3] - child.box_cm[1]
        child.box_cm, child.pos_cm = (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2), (cx, cy)
        self._box_px.pop(name, None)        # its old pixels are not where it is now
        child.confidence *= self.cfg.ambiguity_penalty      # inferred, not seen: answers hedge
        return [self._emit(name, EventType.MOVED, from_cm=frm, to_cm=child.pos_cm, parent=cover)]

    def _cover_lifted(self, cover: str, child: Entity) -> bool:
        if is_thing(cover):                 # a thing laid over it: its box is the cover's
            c = self.entities[cover]
            return (not self._present[cover] or c.status != Status.VISIBLE or c.box_cm is None
                    or child.box_cm is None or geom.overlap_frac(c.box_cm, child.box_cm) < self.cfg.lifted_overlap_max)
        st = self._covers.state(cover)
        if st is None or not self._present[cover] or child.box_cm is None:
            return True
        return geom.overlap_frac(st.box_cm, child.box_cm) < self.cfg.lifted_overlap_max

    # ----- rule 6: decay -----------------------------------------------------------------------

    def _decay(self, t: float) -> None:
        # Applied before this batch's rules: it ages beliefs held over the elapsed interval, so a
        # belief formed in this batch starts at its full confidence.
        if self._now is None:
            return
        factor = self.cfg.decay_per_min ** ((t - self._now) / 60.0)
        for ent in self.entities.values():
            if ent.status != Status.VISIBLE:
                ent.confidence *= factor

    # ----- events -------------------------------------------------------------------------------

    def _emit(self, name: str, etype: EventType, t=None, wall=None, img=None, context=None, **fields) -> Event:
        """t / wall / img: a room event's capture time and zone crop (its snapshot, never the last table
        frame); by default this update's time and frame. context: the whole camera view, saved beside the
        snapshot for answer evidence (a room arrival)."""
        frame = self._frame if t is None else (Frame(t, wall, img, -1) if img is not None else None)
        ev = Event(t=self._now if t is None else t, wall=self._wall if wall is None else wall, obj=name,
                   type=etype, confidence=self.entities[name].confidence, **fields)
        self._room_departure(ev)
        if self.events is not None:
            if context is not None:
                self.events.add(ev, frame, context=context)
            else:
                self.events.add(ev, frame)
        else:
            self._history.append(ev)
        return ev


def _list(p):
    return None if p is None else list(p)


def _is_hand(parent: str | None) -> bool:
    return parent is not None and parent.startswith('hand')
