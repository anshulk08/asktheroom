"""Room memory on the World side (spec 0009 M0, section 3 Transitions). RoomTracker (core/room.py) hands
World.room_update one ZoneVisit per processed zone; these rules decide what its confirmed tracks mean for
the entities. Every change to a room entity is exactly one operation:

  Acquire    the entity has an unconsumed table departure within handoff_s, is GONE / UNKNOWN on the
             table, and the track was first seen after that departure: zone = the track's zone, VISIBLE,
             FOUND. The departure is consumed: it authorises at most one acquisition.
  Refresh    the entity's own associated track matched again: room timestamps only, no event.
  Absence    its own track missed absent_visits valid visits: UNKNOWN, zone kept, LOST_TRACK.
  Reacquire  UNKNOWN in zone Z; a track in Z at its last spot (IoU >= 0.3), first seen after the
             absence: VISIBLE again, FOUND.
  Return     confirmed table presence (World._observe after the presence flip, never one detection):
             room state and conflicts dropped, zone 'table', MOVED.
  Conflict   any other confirmed track of the entity's class: recorded as a sighting, entity unchanged.
             A conflict track never upgrades, by repetition or by a later departure.

A departure is EXITED_VIEW or LOST_TRACK emitted while the entity's zone is 'table' (hand-lost and
held-timeout losses are LOST_TRACK too). Room absence (LOST_TRACK in a room zone) is not one.

One instance per prop class is an M0 demo assumption, not identity evidence: a second key ring that first
appears in a zone after the keys leave the table would inherit them. The first-seen-after-departure and
consume-once rules only keep decoys that were already there, or later ones, from doing so.

Unnamed things (spec 0009 section 3, 'Identity: unnamed things'). A thing:N has no class the room detector
could recognise, so the room pass tracks unnamed objects as cls 'thing' tracks and Grok names each crop
(RoomTrack.guess); the table thing's own Grok name comes from world.thing_guess (core/auto_name.py). A
confirmed 'thing' track that is not a thing's own track is handed over (Acquire, tentative) only when all
hold: exactly one candidate (a thing, not merged, GONE / UNKNOWN on the table, with an unconsumed table
departure within handoff_s and before the track's first_seen); no other pending 'thing' track anywhere in
the rooms first seen after that departure (one-to-one: two new objects, one departure, could be either);
and both Grok names exist and match (names_match). A mismatch is final ('ignored'); anything still open
waits thing_name_wait_s from the track's first_seen, then is given up ('ignored'). 'ignored' and 'conflict'
thing tracks are never re-decided. Refresh, Absence, Reacquire (same zone, one missing or absent thing
there) and Return are the props' rules, keyed by track id (_thing_tracks) instead of class. Nothing about
things is a conflict: 'thing' is not identity evidence.

Room positions are full-frame px (RoomState.box_px); pos_cm stays None for a room entity, and freshness
reads RoomState timestamps, never ent.last_seen (one table flicker rewrites that through _debounce).

The World class mixes this in; methods share its state (entities, lock, _bits, _present, _seen_t, ...).
"""
from __future__ import annotations

import time
from typing import Optional

from core import relations
from core.auto_name import match_score
from core.room_types import TABLE, Conflict, Place, RoomConfig, RoomState, RoomTrack, ZoneVisit
from core.things import is_thing
from core.types import Entity, Event, EventType, Status

DEPART_BACKDATE_MAX_S = 5.0  # a departure is dated back to the last table evidence at most this far
DEPARTURES = (EventType.EXITED_VIEW, EventType.LOST_TRACK)
TABLE_HIDDEN = (Status.UNDER, Status.INSIDE, Status.HELD)
OFF_TABLE = (Status.GONE, Status.UNKNOWN)
UNKNOWN_COVER = 'unknown'   # UNDER an unknown cover: from a high corner an arm grabbing it reads this way
TABLE_SAY = 'the table'
THING = 'thing'                         # RoomTrack.cls of an unnamed object in a zone
FINAL = ('conflict', 'ignored')         # a thing track in one of these roles is never re-decided


def names_match(a: Optional[dict], b: Optional[dict], min_score: float) -> bool:
    """Two Grok guesses name the same kind of object: some phrase of one (its name or an alternative) fits
    the other at min_score or better (core.auto_name.match_score: 2 = the head noun shared), either way
    round. 'remote' fits {'remote control', also 'remote'}; 'shoe' fits nothing about a remote."""
    if not isinstance(a, dict) or not isinstance(b, dict):
        return False
    for g, other in ((a, b), (b, a)):
        for phrase in [g.get('name')] + list(g.get('also') or []):
            if isinstance(phrase, str) and match_score(phrase, other) >= min_score:
                return True
    return False


class RoomRules:
    """Room association, absence, return and place(), mixed into World."""

    def _reset_room(self) -> None:
        self._room: dict[str, RoomState] = {}
        self._departures: dict[str, tuple[float, float]] = {}   # name -> (t, wall) of its unconsumed departure
        self._conflicts: dict[str, dict[str, Conflict]] = {}     # name -> track id -> Conflict
        self._thing_tracks: dict[str, str] = {}                  # 'thing' track id -> the thing:N it is
        self._pending_things: dict[str, RoomTrack] = {}          # confirmed 'thing' tracks not decided yet
        self.room_cfg = RoomConfig.from_dict(getattr(self, '_room_raw', None))

    def thing_guess(self, name: str) -> Optional[dict]:
        """Grok's {name, also, confidence} for a thing on the table. core/auto_name.py's attach replaces
        this on the instance; without a namer no thing has a name, so none is ever handed over."""
        return None

    # ----- public API -----------------------------------------------------------------------------

    def room_update(self, visit: ZoneVisit) -> list[Event]:
        """Apply one zone visit. Tracks of a class that is not an entity (nor 'thing') are ignored."""
        with self.lock:
            out: list[Event] = []
            things = self.room_cfg.things
            # Each entity's own track first: its refresh this visit must be seen by the other tracks'
            # decisions (a same-class track while the own one still matches stays a conflict).
            own = [t for t in visit.confirmed if self._own_track(t) is not None]
            rest = [t for t in visit.confirmed if t not in own]
            # Every new thing track of this visit is known before any is decided: two that confirm together
            # make each other ambiguous, whichever is decided first.
            for trk in rest:
                if things and trk.cls == THING and trk.role not in FINAL:
                    if trk.role == 'assoc':           # its thing went back to the table (Return)
                        trk.role, trk.entity = 'pending', None
                    self._pending_things[trk.tid] = trk
            for trk in own + rest:
                if trk.cls == THING:
                    if things:
                        out += self._room_thing(trk, visit)
                elif trk.cls in self.entities:
                    out += self._room_confirmed(trk.cls, trk, visit)
            for trk in visit.missed:
                out += self._room_missed(trk, visit)
            for trk in visit.dropped:
                self._room_dropped(trk)
            return out

    def place(self, name: str, now: Optional[float] = None) -> Place:
        """Where to say `name` is (now: wall time). The chain's outermost entity decides: in a room zone,
        a room place with room freshness; otherwise today's table place (resolve_chain's position)."""
        now = time.time() if now is None else now
        with self.lock:
            if name not in self.entities:
                return Place(kind='none', zone='', say='', status=Status.UNKNOWN, chain=[], via=name)
            pos, chain = relations.resolve_chain(name, self.entities, self.cfg.max_nesting)
            via = chain[-1]
            vent = self.entities[via]
            fresh_s = self.room_cfg.fresh_s
            conflicts = [c for c in self._conflicts.get(name, {}).values() if now - c.seen_wall <= fresh_s]
            st = self._room.get(via)
            if vent.zone != TABLE and st is not None:
                rc = self.room_cfg
                stale = st.misses >= rc.fresh_visits and now - st.seen_wall >= rc.stale_min_s
                fresh = not st.absent and not stale and now - st.seen_wall <= fresh_s
                return Place(kind='room', zone=st.zone, say=st.say, status=vent.status, chain=chain, via=via,
                             box_px=st.box_px, observed_directly=via == name, fresh=fresh, absent=st.absent,
                             arrived_wall=st.arrived_wall, last_seen_wall=st.seen_wall,
                             arrival_observed=st.arrival_observed, conflicts=conflicts,
                             tentative=st.tentative)
            say = TABLE_SAY if vent.zone == TABLE else f'the {vent.zone}'
            return Place(kind='table', zone=vent.zone, say=say, status=vent.status, chain=chain, via=via,
                         pos_cm=pos, observed_directly=via == name, conflicts=conflicts)

    def room_json(self) -> dict:
        """For state_json: name -> its room state, plus 'conflicts' (every recorded conflict sighting)."""
        # 'tentative' only when true (a thing handed over by name): a prop's entry keeps its M0 shape.
        out: dict = {n: {'zone': st.zone, 'say': st.say, 'box_px': list(st.box_px), 'seen_wall': st.seen_wall,
                         'absent': st.absent, 'arrival_observed': st.arrival_observed,
                         **({'tentative': True} if st.tentative else {})}
                     for n, st in self._room.items()}
        out['conflicts'] = [{'entity': c.entity, 'zone': c.zone, 'say': c.say, 'track': c.track,
                             'box_px': list(c.box_px), 'seen_wall': c.seen_wall}
                            for per in self._conflicts.values() for c in per.values()]
        return out

    # ----- hooks called by world.py ------------------------------------------------------------------

    def _room_departure(self, ev: Event) -> None:
        """_emit hook: an EXITED_VIEW / LOST_TRACK while the entity is on the table is its latest
        departure (a newer one replaces an unconsumed older one)."""
        anchor = self.__dict__.pop('_depart_anchor', None)
        if _is_departure(ev) and self.entities[ev.obj].zone == TABLE:
            # The table decides a departure up to ~2 s after the object left (absence debounce, hand_lost_s,
            # lost_grace_s); a room zone near the table can see it before that. Date the departure at the
            # last table evidence the rule set just before emitting (the holding hand's last sighting, or the
            # object's own), so a zone sighting in between is still 'first seen after the departure'.
            t = ev.t
            if anchor is not None and 0.0 <= ev.t - anchor <= DEPART_BACKDATE_MAX_S:
                t = anchor
            self._departures[ev.obj] = (t, ev.wall - (ev.t - t))

    def _room_return(self, name: str, ent: Entity) -> list[Event]:
        """_observe hook, after it reset the zone to 'table': confirmed table presence of a room entity.
        _observe runs only after the table presence flip, so one flicker never gets here. The tracker keeps
        the old track; the next visit re-decides it against the table (ignored while the table sees it)."""
        self._room.pop(name, None)
        self._conflicts.pop(name, None)
        for tid in [tid for tid, n in self._thing_tracks.items() if n == name]:
            del self._thing_tracks[tid]       # a thing's old room track is just an unnamed track again
        return [self._emit(name, EventType.MOVED, to_cm=ent.pos_cm)]

    # ----- transitions ---------------------------------------------------------------------------------

    def _own_track(self, trk: RoomTrack) -> Optional[str]:
        """The entity whose associated track trk is (a prop by its class, a thing by the track id), or None."""
        name = self._thing_tracks.get(trk.tid) if trk.cls == THING else trk.cls
        st = self._room.get(name) if name is not None else None
        return name if st is not None and st.track == trk.tid else None

    def _room_confirmed(self, name: str, trk: RoomTrack, visit: ZoneVisit) -> list[Event]:
        st = self._room.get(name)
        if st is not None and st.track == trk.tid:
            self._room_refresh(name, st, trk)
            return []
        if trk.role == 'conflict':            # never upgrades
            self._room_conflict(name, trk, visit)
            return []
        op = self._room_decide(name, trk, visit)
        if op == 'acquire':
            return self._room_acquire(name, trk, visit)
        if op == 'reacquire':
            return self._room_reacquire(name, trk, visit)
        if op == 'ignored':
            trk.role, trk.entity = 'ignored', None
            return []
        self._room_conflict(name, trk, visit)
        return []

    def _room_decide(self, name: str, trk: RoomTrack, visit: ZoneVisit) -> str:
        """'acquire' | 'reacquire' | 'ignored' | 'conflict' for a confirmed track that is not the
        entity's own. The table wins while it has the object: seen there within table_fresh_s, the track
        is something else (re-decided next visit); believed hidden there (UNDER / INSIDE / HELD), the
        belief stands. Only a GONE / UNKNOWN entity can be acquired: one seen on the table again after its
        departure is not authorised by that old departure."""
        ent, rc = self.entities[name], self.room_cfg
        if ent.zone == TABLE:
            if ent.status == Status.VISIBLE and visit.t - self._seen_t.get(name, float('-inf')) <= rc.table_fresh_s:
                return 'ignored'
            dep = self._departures.get(name)
            if (_off_table(ent) and dep is not None and visit.t - dep[0] <= rc.handoff_s
                    and trk.first_seen > dep[0]):
                return 'acquire'
            return 'conflict'
        st = self._room.get(name)
        # Same zone, first seen after the entity's own track was last matched, while that track is missing
        # (valid empty visits) or already absent: the object moved within its zone (nudged along the shelf).
        # A same-class track while its own track still matches stays a conflict. Relies on the M0 demo
        # assumption of one instance per prop class, like Acquire (spec 0009, M0).
        if (st is not None and visit.zone == st.zone and trk.first_seen > st.seen_t
                and (st.absent or st.misses > 0)):
            return 'reacquire'
        return 'conflict'

    def _room_thing(self, trk: RoomTrack, visit: ZoneVisit) -> list[Event]:
        """A confirmed 'thing' track: its thing's own track refreshes; a final one stays as it is; any other
        is decided (see the module docstring) and, once handed over, is that thing's track."""
        name = self._own_track(trk)
        if name is not None:
            self._room_refresh(name, self._room[name], trk)
            return []
        if trk.role in FINAL:
            return []
        op, name = self._thing_decide(trk, visit)
        if op == 'pending':
            return []
        self._pending_things.pop(trk.tid, None)
        if op == 'ignored':
            trk.role, trk.entity = 'ignored', None
            return []
        old = self._room[name].track if op == 'reacquire' else None
        self._thing_tracks.pop(old, None)
        self._thing_tracks[trk.tid] = name
        if op == 'reacquire':
            return self._room_reacquire(name, trk, visit)     # keeps RoomState.tentative
        return self._room_acquire(name, trk, visit, tentative=True)

    def _thing_decide(self, trk: RoomTrack, visit: ZoneVisit) -> tuple[str, Optional[str]]:
        """('acquire' | 'reacquire', thing) or ('pending' | 'ignored', None) for a thing track that is not
        its thing's own. Reacquire first (a thing missing in this zone: the props' rule, one such thing
        only), then the handoff: one candidate, one new track, matching names. Undecided stays pending
        while the track is younger than thing_name_wait_s (a name or the other track's fate may still
        come); after that it is given up."""
        rc = self.room_cfg
        name = self._thing_reacquire(trk, visit)
        if name is not None:
            return 'reacquire', name
        cands = self._thing_candidates(trk, visit)
        if len(cands) == 1:
            name, dep_t = cands[0]
            others = [p for tid, p in self._pending_things.items()
                      if tid != trk.tid and p.role == 'pending' and p.first_seen > dep_t]
            mine, theirs = trk.guess, self.thing_guess(name)
            if not others and mine is not None and theirs is not None:
                if names_match(mine, theirs, rc.name_match_min):
                    return 'acquire', name
                return 'ignored', None    # a shoe left the table; this is a remote: never that thing
        if visit.t - trk.first_seen >= rc.thing_name_wait_s:
            return 'ignored', None
        return 'pending', None

    def _thing_candidates(self, trk: RoomTrack, visit: ZoneVisit) -> list[tuple[str, float]]:
        """(thing, departure t) for every thing this track could be: carried off the table (unconsumed
        departure within handoff_s, before the track was first seen), still off it, not merged away and
        not already some other track's."""
        taken = set(self._thing_tracks.values())
        out = []
        for name, (dep_t, _) in self._departures.items():
            ent = self.entities.get(name)
            if (ent is None or not is_thing(name) or ent.merged_into is not None or name in taken
                    or name in self._room):
                continue
            if (ent.zone == TABLE and _off_table(ent) and visit.t - dep_t <= self.room_cfg.handoff_s
                    and dep_t < trk.first_seen):
                out.append((name, dep_t))
        return out

    def _thing_reacquire(self, trk: RoomTrack, visit: ZoneVisit) -> Optional[str]:
        """The one thing in this zone whose own track is missing or absent and was last matched before trk
        was first seen (moved within the zone), or None. Two such things: none (a thing has no class to
        tell them apart). A track whose Grok name contradicts the thing's is not it."""
        hits = []
        for name, st in self._room.items():
            ent = self.entities.get(name)
            if (not is_thing(name) or ent is None or ent.merged_into is not None or st.zone != visit.zone
                    or trk.first_seen <= st.seen_t or not (st.absent or st.misses > 0)):
                continue
            theirs = self.thing_guess(name)
            if trk.guess is not None and theirs is not None and not names_match(trk.guess, theirs,
                                                                               self.room_cfg.name_match_min):
                continue
            hits.append(name)
        return hits[0] if len(hits) == 1 else None

    def _room_acquire(self, name: str, trk: RoomTrack, visit: ZoneVisit, tentative: bool = False) -> list[Event]:
        del self._departures[name]            # consumed: one departure, one acquisition
        ent = self.entities[name]
        self._room[name] = RoomState(zone=visit.zone, say=visit.say, box_px=trk.box_px, track=trk.tid,
                                     seen_t=trk.last_seen, seen_wall=trk.last_wall, arrived_wall=trk.first_wall,
                                     arrival_observed=True, table_pos_cm=ent.pos_cm, tentative=tentative)
        return self._room_settle(name, ent, trk, visit)

    def _room_reacquire(self, name: str, trk: RoomTrack, visit: ZoneVisit) -> list[Event]:
        st = self._room[name]
        st.box_px, st.track, st.seen_t, st.seen_wall = trk.box_px, trk.tid, trk.last_seen, trk.last_wall
        st.arrived_wall, st.arrival_observed = trk.first_wall, False
        st.misses, st.absent, st.absent_t = 0, False, None
        return self._room_settle(name, self.entities[name], trk, visit)

    def _room_settle(self, name: str, ent: Entity, trk: RoomTrack, visit: ZoneVisit) -> list[Event]:
        """The entity is VISIBLE in the visit's zone. No table cm: pos_cm / box_cm are cleared, and the table
        debounce restarts so that only a new presence flip (Return) brings it back to the table."""
        ent.status, ent.zone, ent.parent, ent.candidates, ent.confidence, ent.edge = \
            Status.VISIBLE, visit.zone, None, [], 1.0, None
        ent.pos_cm, ent.box_cm, ent.held_since, ent.pre_pickup_pos = None, None, None, None
        ent.last_seen = trk.last_wall
        self._box_px.pop(name, None)          # table-view pixels: meaningless for a room entity
        self._look_px.pop(name, None)
        self._bits[name].clear()
        self._present[name] = False
        self._confirmed.add(name)
        trk.role, trk.entity = 'assoc', name
        return [self._emit(name, EventType.FOUND, t=visit.t, wall=visit.wall, img=visit.crop)]

    def _room_refresh(self, name: str, st: RoomState, trk: RoomTrack) -> None:
        """Its own track matched again. An observation older than the one last applied is ignored."""
        if trk.last_seen < st.seen_t:
            return
        st.box_px, st.seen_t, st.seen_wall, st.misses = trk.box_px, trk.last_seen, trk.last_wall, 0
        ent = self.entities.get(name)
        if ent is not None:
            ent.last_seen = trk.last_wall

    def _room_conflict(self, name: str, trk: RoomTrack, visit: ZoneVisit) -> None:
        trk.role = 'conflict'
        self._conflicts.setdefault(name, {})[trk.tid] = Conflict(
            entity=name, zone=visit.zone, say=visit.say, track=trk.tid, box_px=trk.box_px,
            seen_t=trk.last_seen, seen_wall=trk.last_wall)

    def _room_missed(self, trk: RoomTrack, visit: ZoneVisit) -> list[Event]:
        """A valid visit without a match for the entity's own track. Blocked visits never reach here (the
        tracker leaves them out), so a person in front of the zone counts for nothing."""
        name = self._own_track(trk)
        if name is None:
            return []
        st = self._room[name]
        st.misses = trk.misses
        rc = self.room_cfg
        if st.misses < rc.absent_visits or visit.t - st.seen_t < rc.absent_min_s or st.absent:
            return []           # a zone is visited ~3x a second: a few detector misses are not absence
        st.absent, st.absent_t = True, visit.t
        ent = self.entities[name]
        ent.status, ent.parent, ent.candidates, ent.held_since = Status.UNKNOWN, None, [], None
        # zone stays the room zone, so _emit does not count this LOST_TRACK as a departure
        return [self._emit(name, EventType.LOST_TRACK, t=visit.t, wall=visit.wall, img=visit.crop)]

    def _room_dropped(self, trk: RoomTrack) -> None:
        name = self._own_track(trk)
        if name is not None:
            self._room[name].track = None     # released; the state stays (absent) for reacquire
        self._thing_tracks.pop(trk.tid, None)
        self._pending_things.pop(trk.tid, None)
        for name in list(self._conflicts):
            self._conflicts[name].pop(trk.tid, None)
            if not self._conflicts[name]:
                del self._conflicts[name]


def _is_departure(ev: Event) -> bool:
    """EXITED_VIEW / LOST_TRACK, or COVERED by an unknown cover. On the rig (spec 0009: the Brio high in a
    corner) the detector rarely sees hands, so a grab reads as the object vanishing under something
    unknown (an arm), not as HELD; counting it lets the handoff happen. A real unknown cover is safe: a
    handoff still needs a new room track first seen after it, one candidate and, for things, a Grok name
    match."""
    return ev.type in DEPARTURES or (ev.type == EventType.COVERED and ev.parent == UNKNOWN_COVER)


def _off_table(ent: Entity) -> bool:
    """Gone from the table as far as a handoff is concerned: GONE, UNKNOWN, or UNDER an unknown cover."""
    return ent.status in OFF_TABLE or (ent.status == Status.UNDER and ent.parent == UNKNOWN_COVER)
