"""Room memory on the World side (spec 0009 M0, section 3 Transitions): acquire, refresh, absence,
reacquire, return, conflict, place() and the legacy-reader guard in resolve().

The table side is a tests.synth.Scene; the room side is hand-built RoomTracker output (ZoneVisit /
RoomTrack) on the same clock (Scene wall == t), always after the scene's current t."""
import json
from pathlib import Path

import numpy as np
import pytest

from core.config import Config, load_config
from core.room_types import RoomConfig, RoomTrack, ZoneVisit
from core.types import EventType, Frame, Status
from core.world import World
from tests.synth import Scene
from tests.test_world_core import pick_up

ROOT = Path(__file__).resolve().parents[1]
SAY = {'bookshelf': 'the bookshelf', 'couch': 'the couch'}
SHELF_BOX = (1500, 200, 1560, 240)       # full-frame px
COUCH_BOX = (300, 800, 360, 840)


@pytest.fixture
def cfg():
    return Config.load(ROOT / 'config.yaml')


@pytest.fixture
def scene(cfg):
    return Scene(cfg, fps=10, t0=1000.0)


@pytest.fixture
def world(cfg):
    return World(cfg)


def types(events):
    return [e.type for e in events]


class FakeLog:
    def __init__(self):
        self.added = []

    def add(self, ev, frame=None):
        self.added.append((ev, frame))

    def last(self, obj, n):
        return [ev for ev, _ in reversed(self.added) if ev.obj == obj][:n]


class CoveredWorld(World):
    """Every vanishing object is under the notebook (stands in for the cover rule)."""

    def _hidden_by(self, name, ent, box_cm, now, frame):
        return (Status.UNDER, 'notebook', self.cfg.conf_under, EventType.COVERED)


class KeysInBoxWorld(World):
    """Held keys go straight into the box (stands in for the container rule); nothing else does."""

    def _check_inside(self, name, ent, now):
        if name == 'keys':
            return (Status.INSIDE, 'box', self.cfg.conf_inside, EventType.PUT_INSIDE)
        return None


# ----- helpers: the table side --------------------------------------------------------------------

def depart(scene, world, name='keys', hid=1, at=(40, 30)):
    """Pick the object up and carry it out by the left edge: EXITED_VIEW while its zone is 'table'."""
    pick_up(scene, world, name, hid, at)
    scene.hand(hid, 5, at[1])
    scene.run(world, 0.3)
    scene.hand_off(hid)
    events = scene.run(world, 1.0)
    assert EventType.EXITED_VIEW in types(events)
    return events


# ----- helpers: the room side (hand-built tracker output) -----------------------------------------

def appear(scene, world, tid, cls='keys', zone='bookshelf', box=SHELF_BOX):
    """A new room track's first observation, after the table ran another 0.5 s (unconfirmed: the World
    is not told)."""
    scene.run(world, 0.5)
    t = scene.t
    return RoomTrack(tid=tid, zone=zone, cls=cls, box_px=box, first_seen=t, first_wall=t,
                     last_seen=t, last_wall=t)


def seen(scene, world, *tracks, zone='bookshelf', crop=None, dt=0.5):
    """One zone visit, dt after the table's last batch, in which these tracks matched (confirmed)."""
    scene.run(world, dt)
    t = scene.t
    for trk in tracks:
        trk.last_seen = trk.last_wall = t
        trk.hits, trk.misses, trk.confirmed = trk.hits + 1, 0, True
    return world.room_update(ZoneVisit(zone, SAY[zone], t, t, scene.idx, confirmed=list(tracks), crop=crop))


def missed(scene, world, *tracks, zone='bookshelf', dt=1.2):
    """One valid zone visit in which these confirmed tracks were not matched; dropped, like the tracker
    does, at absent_visits misses and absent_min_s since the last match (3 misses 1.2 s apart)."""
    scene.run(world, dt)
    t = scene.t
    for trk in tracks:
        trk.misses += 1
    rc = world.room_cfg
    dropped = [trk for trk in tracks if trk.misses >= rc.absent_visits and t - trk.last_seen >= rc.absent_min_s]
    return world.room_update(ZoneVisit(zone, SAY[zone], t, t, scene.idx, missed=list(tracks), dropped=dropped))


def acquired(scene, world, tid='r:1'):
    """The demo moment: the keys leave the table, a new shelf track confirms on its second visit."""
    depart(scene, world)
    trk = appear(scene, world, tid)
    events = seen(scene, world, trk)
    assert types(events) == [EventType.FOUND]
    return trk


# ----- acquire ------------------------------------------------------------------------------------

def test_demo_moment_acquires_the_keys_on_the_bookshelf(scene, world):
    depart(scene, world)
    trk = appear(scene, world, 'r:1')
    events = seen(scene, world, trk)
    keys = world.get('keys')
    assert types(events) == [EventType.FOUND]
    assert (keys.status, keys.zone, keys.parent, keys.pos_cm, keys.box_cm) == \
        (Status.VISIBLE, 'bookshelf', None, None, None)
    assert keys.confidence == 1.0
    assert (trk.role, trk.entity) == ('assoc', 'keys')
    assert world.resolve('keys') == (None, ['keys'])
    p = world.place('keys', now=scene.t)
    assert (p.kind, p.zone, p.say, p.via, p.chain) == ('room', 'bookshelf', 'the bookshelf', 'keys', ['keys'])
    assert p.fresh and p.arrival_observed and p.observed_directly and not p.absent
    assert p.box_px == SHELF_BOX
    assert p.arrived_wall == trk.first_wall
    assert p.last_seen_wall == trk.last_wall
    assert p.pos_cm is None and p.conflicts == []
    assert 'keys' not in world._departures                 # consumed
    assert world._room['keys'].table_pos_cm == pytest.approx((40, 30))


def test_lost_track_on_the_table_is_a_departure_too(scene, world):
    scene.place('keys', 40, 30)
    scene.run(world, 1.0)
    scene.remove('keys')
    assert types(scene.run(world, world.cfg.lost_grace_s + 0.2)) == [EventType.LOST_TRACK]
    trk = appear(scene, world, 'r:1')
    assert types(seen(scene, world, trk)) == [EventType.FOUND]
    assert world.get('keys').zone == 'bookshelf'


def test_room_found_event_carries_the_visit_time_and_crop(scene, cfg):
    log = FakeLog()
    world = World(cfg, events=log)
    depart(scene, world)
    trk = appear(scene, world, 'r:1')
    scene.run(world, 0.5)
    t = scene.t - 0.05                    # the room frame was captured before the table's latest batch
    trk.last_seen = trk.last_wall = t
    trk.hits, trk.confirmed = 2, True
    crop = np.zeros((40, 60, 3), np.uint8)
    (ev,) = world.room_update(ZoneVisit('bookshelf', 'the bookshelf', t, t + 0.01, 7, confirmed=[trk], crop=crop))
    assert (ev.type, ev.t, ev.wall) == (EventType.FOUND, t, t + 0.01)
    assert ev.t != world._now
    logged, frame = log.added[-1]
    assert logged is ev
    assert isinstance(frame, Frame) and frame.img is crop and frame.t == t


def test_room_event_without_a_crop_has_no_table_snapshot(scene, cfg):
    log = FakeLog()
    world = World(cfg, events=log)
    depart(scene, world)
    trk = appear(scene, world, 'r:1')
    seen(scene, world, trk)
    assert log.added[-1][0].type == EventType.FOUND
    assert world._frame is not None and log.added[-1][1] is None


def test_a_track_first_seen_before_the_departure_is_a_conflict(scene, world):
    pick_up(scene, world)
    trk = appear(scene, world, 'r:1')     # already on the shelf while the keys are in a hand
    scene.hand(1, 5, 30)
    scene.run(world, 0.3)
    scene.hand_off(1)
    assert EventType.EXITED_VIEW in types(scene.run(world, 1.0))
    assert seen(scene, world, trk) == []
    assert trk.role == 'conflict'
    assert world.get('keys').status == Status.GONE
    assert world.get('keys').zone == 'table'
    (c,) = world.place('keys', now=scene.t).conflicts
    assert (c.entity, c.zone, c.say, c.track, c.box_px) == ('keys', 'bookshelf', 'the bookshelf', 'r:1', SHELF_BOX)


def test_a_departure_older_than_handoff_s_authorises_nothing(scene, world):
    world.room_cfg.handoff_s = 1.0
    depart(scene, world)
    scene.run(world, 1.0)
    trk = appear(scene, world, 'r:1')
    assert seen(scene, world, trk) == []
    assert trk.role == 'conflict'
    assert world.get('keys').zone == 'table'


def test_departure_is_consumed_by_one_acquisition(scene, world):
    acquired(scene, world)
    trk = appear(scene, world, 'r:2', zone='couch', box=COUCH_BOX)
    assert seen(scene, world, trk, zone='couch') == []
    assert trk.role == 'conflict'
    assert world.get('keys').zone == 'bookshelf'
    assert world._room['keys'].track == 'r:1'
    assert [c.zone for c in world.place('keys', now=scene.t).conflicts] == ['couch']


def test_a_conflict_track_never_upgrades_even_after_a_later_departure(scene, world):
    trk = appear(scene, world, 'r:1')     # startup: the keys were never on the table
    assert seen(scene, world, trk) == []
    assert trk.role == 'conflict'
    depart(scene, world)
    for _ in range(3):
        assert seen(scene, world, trk) == []
    assert trk.role == 'conflict'
    assert world.get('keys').zone == 'table'



def test_a_conflict_track_stays_one_when_the_rules_would_now_allow_it(scene, world):
    """The role itself is sticky: a track decided as a conflict is never re-decided."""
    depart(scene, world)
    world.room_cfg.handoff_s = 0.1        # too old: a conflict
    trk = appear(scene, world, 'r:1')
    assert seen(scene, world, trk) == []
    assert trk.role == 'conflict'
    world.room_cfg.handoff_s = 120.0      # the same departure would now authorise a new track
    assert seen(scene, world, trk) == []
    assert trk.role == 'conflict'
    assert world.get('keys').zone == 'table' and 'keys' in world._departures


def test_a_departure_is_void_once_the_table_sees_the_object_again(scene, world):
    """LOST_TRACK, then CORRECTED on the table: that old departure must not hand the keys to a shelf track
    while the table still holds them (VISIBLE, only briefly unseen)."""
    world.room_cfg.table_fresh_s = 0.5
    scene.place('keys', 40, 30)
    scene.run(world, 1.0)
    scene.remove('keys')
    assert types(scene.run(world, world.cfg.lost_grace_s + 0.2)) == [EventType.LOST_TRACK]
    scene.place('keys', 40, 30)
    assert types(scene.run(world, 1.0)) == [EventType.CORRECTED]
    scene.remove('keys')                  # a detector miss: VISIBLE for lost_grace_s, but not fresh
    trk = appear(scene, world, 'r:1')
    assert seen(scene, world, trk) == []
    assert world.get('keys').status == Status.VISIBLE
    assert trk.role == 'conflict'
    assert world.get('keys').zone == 'table'


def test_a_class_that_is_not_an_entity_is_ignored(scene, world):
    depart(scene, world)
    trk = appear(scene, world, 'r:1', cls='cat')
    assert seen(scene, world, trk) == []
    assert trk.role == 'pending'
    assert world._room == {} and world._conflicts == {}
    assert 'keys' in world._departures


def test_room_update_before_any_table_batch_is_a_conflict_not_a_crash(world):
    trk = RoomTrack('r:1', 'bookshelf', 'keys', SHELF_BOX, 5.0, 5.0, 6.0, 6.0, hits=2, confirmed=True)
    assert world.room_update(ZoneVisit('bookshelf', 'the bookshelf', 6.0, 6.0, 1, confirmed=[trk])) == []
    assert trk.role == 'conflict'
    assert world.get('keys').status == Status.UNKNOWN


# ----- the table wins while it has the object ----------------------------------------------------

def test_keys_visible_on_the_table_make_a_shelf_track_not_the_keys(scene, world):
    scene.place('keys', 40, 30)
    scene.run(world, 1.0)
    trk = appear(scene, world, 'r:1')
    assert seen(scene, world, trk) == []
    assert trk.role == 'ignored'
    keys = world.get('keys')
    assert (keys.status, keys.zone) == (Status.VISIBLE, 'table')
    assert world.place('keys', now=scene.t).conflicts == []
    # The keys then leave: the old track was first seen before that departure, so it is a conflict.
    scene.remove('keys')
    scene.run(world, world.cfg.lost_grace_s + 1.0)
    assert world.get('keys').status == Status.UNKNOWN
    assert seen(scene, world, trk) == []
    assert trk.role == 'conflict'


def test_keys_under_the_notebook_keep_their_belief_against_shelf_sightings(scene, cfg):
    world = CoveredWorld(cfg)
    scene.place('notebook', 60, 40)
    scene.place('keys', 60, 40)
    scene.run(world, 1.0)
    scene.remove('keys')
    assert types(scene.run(world, 1.0)) == [EventType.COVERED]
    trk = appear(scene, world, 'r:1')
    for _ in range(4):
        assert seen(scene, world, trk) == []
    keys = world.get('keys')
    assert (keys.status, keys.parent, keys.zone) == (Status.UNDER, 'notebook', 'table')
    assert trk.role == 'conflict'
    p = world.place('keys', now=scene.t)
    assert (p.kind, p.via, p.chain) == ('table', 'notebook', ['keys', 'notebook'])
    assert p.pos_cm == pytest.approx((60, 40))
    assert [c.say for c in p.conflicts] == ['the bookshelf']


# ----- refresh, absence, reacquire ---------------------------------------------------------------

def test_own_track_only_refreshes(scene, world):
    trk = acquired(scene, world)
    for _ in range(4):
        assert seen(scene, world, trk) == []
    st = world._room['keys']
    assert (st.track, st.seen_t, st.seen_wall, st.misses) == ('r:1', scene.t, scene.t, 0)
    assert world._conflicts.get('keys', {}) == {}
    assert world.place('keys', now=scene.t).fresh


def test_an_out_of_order_refresh_is_rejected(scene, world):
    trk = acquired(scene, world)
    st = world._room['keys']
    before = (st.seen_t, st.seen_wall)
    trk.last_seen = trk.last_wall = before[0] - 1.0
    old = ZoneVisit('bookshelf', 'the bookshelf', before[0] - 1.0, before[1] - 1.0, 1, confirmed=[trk])
    assert world.room_update(old) == []
    assert (st.seen_t, st.seen_wall) == before


def test_absence_after_absent_visits_valid_misses(scene, world):
    trk = acquired(scene, world)
    assert missed(scene, world, trk) == []
    assert world.place('keys', now=scene.t).fresh                 # one miss: still fresh
    assert missed(scene, world, trk) == []
    p = world.place('keys', now=scene.t)
    assert not p.fresh and not p.absent                           # fresh_visits misses: last-seen wording
    events = missed(scene, world, trk)
    keys = world.get('keys')
    assert types(events) == [EventType.LOST_TRACK]
    assert events[0].t == scene.t
    assert (keys.status, keys.zone) == (Status.UNKNOWN, 'bookshelf')
    assert 'keys' not in world._departures                        # room absence is not a departure
    p = world.place('keys', now=scene.t)
    assert (p.kind, p.zone, p.absent, p.fresh) == ('room', 'bookshelf', True, False)
    assert p.status == Status.UNKNOWN
    assert world.resolve('keys') == (None, ['keys'])
    assert world._room['keys'].track is None                      # the dropped track is released


def test_freshness_expires_with_time_on_room_timestamps(scene, world):
    acquired(scene, world)
    seen_wall = world._room['keys'].seen_wall
    assert world.place('keys', now=seen_wall + world.room_cfg.fresh_s).fresh
    assert not world.place('keys', now=seen_wall + world.room_cfg.fresh_s + 0.1).fresh


def test_decoy_after_room_absence_is_a_conflict(scene, world):
    trk = acquired(scene, world)
    for _ in range(3):
        missed(scene, world, trk)
    assert world.get('keys').status == Status.UNKNOWN
    decoy = appear(scene, world, 'r:2', zone='couch', box=COUCH_BOX)
    assert seen(scene, world, decoy, zone='couch') == []
    assert decoy.role == 'conflict'
    keys = world.get('keys')
    assert (keys.status, keys.zone) == (Status.UNKNOWN, 'bookshelf')
    assert [c.zone for c in world.place('keys', now=scene.t).conflicts] == ['couch']


def test_held_timeout_decoy_stays_a_conflict(scene, world, cfg):
    pick_up(scene, world)
    scene.hand(1, 60, 40)                 # the hand stays in view, mid-table
    decoy = appear(scene, world, 'r:1')
    assert seen(scene, world, decoy) == []
    assert decoy.role == 'conflict'       # the keys are HELD: keep the belief
    events = scene.run(world, cfg.held_timeout_s)
    assert EventType.LOST_TRACK in types(events)
    assert 'keys' in world._departures
    for _ in range(3):
        assert seen(scene, world, decoy) == []
    keys = world.get('keys')
    assert (keys.status, keys.zone) == (Status.UNKNOWN, 'table')
    assert decoy.role == 'conflict'


def test_held_timeout_decoy_first_seen_during_the_hold_is_a_conflict_when_it_confirms_later(scene, world, cfg):
    pick_up(scene, world)
    scene.hand(1, 60, 40)
    decoy = appear(scene, world, 'r:1')   # first seen while HELD, confirmed only after the timeout
    assert EventType.LOST_TRACK in types(scene.run(world, cfg.held_timeout_s))
    assert seen(scene, world, decoy) == []
    assert decoy.role == 'conflict'
    assert world.get('keys').zone == 'table'


def test_reacquire_at_the_last_spot(scene, world):
    trk = acquired(scene, world)
    arrived = world._room['keys'].arrived_wall
    for _ in range(3):
        missed(scene, world, trk)
    again = appear(scene, world, 'r:2', box=(1505, 205, 1565, 245))
    events = seen(scene, world, again)
    keys = world.get('keys')
    assert types(events) == [EventType.FOUND]
    assert (keys.status, keys.zone) == (Status.VISIBLE, 'bookshelf')
    assert (again.role, again.entity) == ('assoc', 'keys')
    st = world._room['keys']
    assert (st.track, st.absent, st.misses, st.arrival_observed) == ('r:2', False, 0, False)
    assert st.arrived_wall == again.first_wall != arrived
    p = world.place('keys', now=scene.t)
    assert p.fresh and not p.arrival_observed


def test_moved_along_its_zone_reacquires_but_another_zone_does_not(scene, world):
    """Nudged along the shelf: a new track in the same zone after its own track went missing is the keys
    (one instance per class, the M0 demo assumption); the same keys class on the couch is a conflict."""
    trk = acquired(scene, world)
    for _ in range(3):
        missed(scene, world, trk)
    couch = appear(scene, world, 'r:3', zone='couch', box=SHELF_BOX)
    assert seen(scene, world, couch, zone='couch') == []
    assert couch.role == 'conflict' and world.get('keys').status == Status.UNKNOWN
    far = appear(scene, world, 'r:2', box=(1000, 200, 1060, 240))
    assert types(seen(scene, world, far)) == [EventType.FOUND]
    keys = world.get('keys')
    assert (keys.status, keys.zone, far.role) == (Status.VISIBLE, 'bookshelf', 'assoc')
    assert world.place('keys', now=scene.t).box_px == (1000, 200, 1060, 240)


def test_moved_along_its_zone_before_absence_reacquires(scene, world):
    """Nudged while its old track has only started missing: no absence first, no permanent conflict."""
    trk = acquired(scene, world)
    missed(scene, world, trk)
    moved = appear(scene, world, 'r:2', box=(1000, 200, 1060, 240))
    assert types(seen(scene, world, moved)) == [EventType.FOUND]
    assert world._room['keys'].track == 'r:2' and world.get('keys').status == Status.VISIBLE


def test_a_track_confirmed_while_the_own_track_matches_stays_a_conflict(scene, world):
    """Two key rings on the shelf at once: the second is a conflict and never takes over, even after the
    keys' own track goes missing."""
    trk = acquired(scene, world)
    other = appear(scene, world, 'r:2', box=(1000, 200, 1060, 240))
    assert seen(scene, world, trk, other) == []          # both seen: the keys' own track still matches
    assert other.role == 'conflict'
    for _ in range(3):
        missed(scene, world, trk)
    assert seen(scene, world, other) == []
    assert other.role == 'conflict' and world.get('keys').status == Status.UNKNOWN


def test_dropped_tracks_take_their_conflicts_with_them(scene, world):
    acquired(scene, world)
    decoy = appear(scene, world, 'r:2', zone='couch', box=COUCH_BOX)
    seen(scene, world, decoy, zone='couch')
    assert list(world._conflicts['keys']) == ['r:2']
    for _ in range(3):
        missed(scene, world, decoy, zone='couch')
    assert world._conflicts.get('keys', {}) == {}


def test_stale_conflicts_are_left_out_of_place(scene, world):
    acquired(scene, world)
    decoy = appear(scene, world, 'r:2', zone='couch', box=COUCH_BOX)
    seen(scene, world, decoy, zone='couch')
    seen_wall = decoy.last_wall
    assert world.place('keys', now=seen_wall + world.room_cfg.fresh_s).conflicts
    assert world.place('keys', now=seen_wall + world.room_cfg.fresh_s + 0.1).conflicts == []


# ----- return -------------------------------------------------------------------------------------

def test_table_return_needs_the_presence_flip_and_clears_the_room(scene, world):
    trk = acquired(scene, world)
    decoy = appear(scene, world, 'r:2', zone='couch', box=COUCH_BOX)
    seen(scene, world, decoy, zone='couch')
    scene.place('keys', 60, 40)
    events = scene.run(world, 1.0)
    keys = world.get('keys')
    assert types(events) == [EventType.MOVED]
    assert events[0].to_cm == pytest.approx((60, 40))
    assert events[0].from_cm is None
    assert (keys.status, keys.zone) == (Status.VISIBLE, 'table')
    assert 'keys' not in world._room and 'keys' not in world._conflicts
    assert world.resolve('keys') == (pytest.approx((60, 40)), ['keys'])
    p = world.place('keys', now=scene.t)
    assert (p.kind, p.zone, p.say) == ('table', 'table', 'the table')
    assert p.pos_cm == pytest.approx((60, 40))
    for _ in range(3):                     # the shelf is empty now: nothing changes
        assert missed(scene, world, trk) == []
    assert (keys.status, keys.zone) == (Status.VISIBLE, 'table')


def test_a_single_table_flicker_does_not_return(scene, world):
    acquired(scene, world)
    before = world.place('keys', now=scene.t)
    scene.place('keys', 60, 40)
    events = world.update(*scene.step())  # one detection above threshold: no presence flip
    scene.remove('keys')
    events += scene.run(world, 2.0)
    keys = world.get('keys')
    assert events == []
    assert keys.last_seen is not None and keys.pos_cm is not None      # the flicker wrote these ...
    assert (keys.status, keys.zone) == (Status.VISIBLE, 'bookshelf')    # ... and changed nothing else
    assert world.resolve('keys') == (None, ['keys'])
    after = world.place('keys', now=before.last_seen_wall)
    assert (after.kind, after.zone, after.fresh, after.last_seen_wall, after.pos_cm) == \
        ('room', 'bookshelf', True, before.last_seen_wall, None)


def test_an_absent_room_entity_seen_on_the_table_returns(scene, world):
    trk = acquired(scene, world)
    for _ in range(3):
        missed(scene, world, trk)
    scene.place('keys', 60, 40)
    assert types(scene.run(world, 1.0)) == [EventType.MOVED]
    assert world.get('keys').zone == 'table' and world._room == {}


def test_after_return_the_old_shelf_track_is_not_the_keys(scene, world):
    trk = acquired(scene, world)
    scene.place('keys', 60, 40)
    scene.run(world, 1.0)
    assert seen(scene, world, trk) == []       # still confirming there: the keys are on the table
    assert trk.role == 'ignored'
    assert world.get('keys').zone == 'table'


def test_round_trip_twice(scene, world):
    acquired(scene, world, 'r:1')
    scene.place('keys', 40, 30)
    assert types(scene.run(world, 1.0)) == [EventType.MOVED]
    depart(scene, world)
    trk = appear(scene, world, 'r:2', zone='couch', box=COUCH_BOX)
    assert types(seen(scene, world, trk, zone='couch')) == [EventType.FOUND]
    assert world.get('keys').zone == 'couch'


# ----- containers, legacy readers, state ----------------------------------------------------------

def test_box_with_the_keys_inside_carried_to_the_shelf(scene, cfg):
    world = KeysInBoxWorld(cfg)
    scene.place('box', 30, 50)
    assert types(pick_up(scene, world)) == [EventType.PICKED_UP, EventType.PUT_INSIDE]
    scene.hand_off(1)
    scene.run(world, 0.5)
    depart(scene, world, name='box', hid=2, at=(30, 50))
    trk = appear(scene, world, 'r:1', cls='box')
    assert types(seen(scene, world, trk)) == [EventType.FOUND]
    assert world.get('keys').parent == 'box'
    assert world.resolve('keys') == (None, ['keys', 'box'])
    p = world.place('keys', now=scene.t)
    assert (p.kind, p.zone, p.via, p.chain) == ('room', 'bookshelf', 'box', ['keys', 'box'])
    assert not p.observed_directly and p.fresh
    assert p.status == Status.VISIBLE                     # of the box


def test_place_of_a_table_object_and_of_an_unknown_name(scene, world):
    scene.place('wallet', 80, 40)
    scene.run(world, 1.0)
    p = world.place('wallet', now=scene.t)
    assert (p.kind, p.zone, p.say, p.via, p.status) == ('table', 'table', 'the table', 'wallet', Status.VISIBLE)
    assert p.pos_cm == pytest.approx((80, 40)) and p.box_px is None
    assert world.place('unicorn').kind == 'none'


def test_state_json_carries_room_state(scene, world):
    acquired(scene, world)
    decoy = appear(scene, world, 'r:2', zone='couch', box=COUCH_BOX)
    seen(scene, world, decoy, zone='couch')
    state = world.state_json()
    json.dumps(state)
    room = state['room']
    assert room['keys'] == {'zone': 'bookshelf', 'say': 'the bookshelf', 'box_px': list(SHELF_BOX),
                            'seen_wall': world._room['keys'].seen_wall, 'absent': False,
                            'arrival_observed': True}
    (c,) = room['conflicts']
    assert (c['entity'], c['zone'], c['track']) == ('keys', 'couch', 'r:2')
    keys = {e['name']: e for e in state['entities']}['keys']
    assert (keys['zone'], keys['resolved_cm']) == ('bookshelf', None)
    assert ['keys', 'ON', 'bookshelf'] in state['edges']


def test_reset_clears_room_state(scene, world):
    acquired(scene, world)
    world.reset()
    assert (world._room, world._departures, world._conflicts) == ({}, {}, {})
    assert world.get('keys').zone == 'table'


def test_room_config_comes_from_the_raw_dict_or_defaults(cfg):
    raw = load_config(ROOT / 'config.yaml')
    raw.pop('room_memory', None)
    assert World(raw).room_cfg == RoomConfig()
    raw['room_memory'] = {'enabled': True, 'absent_visits': 5, 'handoff_s': 30}
    rc = World(raw).room_cfg
    assert (rc.enabled, rc.absent_visits, rc.handoff_s) == (True, 5, 30)
    assert World(cfg).room_cfg == RoomConfig()



def test_own_track_refreshes_first_whatever_the_visit_order(scene, world):
    trk = acquired(scene, world)
    missed(scene, world, trk)                               # the own track missed once (misses 1)
    other = appear(scene, world, 'r:2', box=(1000, 200, 1060, 240))
    assert seen(scene, world, other, trk) == []             # other listed first; both matched this visit
    assert other.role == 'conflict' and world._room['keys'].track == trk.tid
