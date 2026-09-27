"""Room memory for Grok-named things on the World side (spec 0009 section 3, unnamed things; room_world.py):
a thing:N that Grok named on the table and that was carried off it is handed over to a room 'thing' track
only one-to-one (one departed candidate, no other new pending thing track) and only when the two Grok
names match. The place is tentative (answers hedge).

The table side is a rendered tests.synth.Scene with a real AutoNamer attached (fake Grok), so the table
thing's guess reaches the World the way it does in the app (world.thing_guess). The room side is hand-built
RoomTracker output, as in tests/test_room_world.py."""
from pathlib import Path

import pytest

import core.crops
from core.auto_name import AutoNameConfig, AutoNamer
from core.config import Config, load_config
from core.room_types import RoomTrack
from core.types import EventType, Status
from core.world import World
from tests.synth import Scene
from tests.test_auto_name import FakeGrok, Clock
from tests.test_room_world import COUCH_BOX, SHELF_BOX, appear, depart, missed, seen, types

ROOT = Path(__file__).resolve().parents[1]
CFG = load_config()
REMOTE = {"name": "remote control", "also": ["remote"], "confidence": 0.7}
MUG = {"name": "coffee mug", "also": ["mug"], "confidence": 0.8}   # a non-junk name that is not a remote
COUCH_BOX_2 = (420, 800, 480, 840)


@pytest.fixture
def cfg():
    return Config.load(ROOT / 'config.yaml')


@pytest.fixture
def scene(cfg):
    return Scene(cfg, fps=10, t0=1000.0, render=True)    # rendered: the namer's close-up is a frame crop


@pytest.fixture
def world(cfg):
    return World(cfg)


@pytest.fixture(autouse=True)
def no_crop_store():
    old = core.crops.active()
    core.crops.set_active(None)
    yield
    core.crops.set_active(old)


def namer_for(world, *replies):
    c = AutoNameConfig(enabled=True, max_per_minute=100)
    return AutoNamer(CFG, world, provider=FakeGrok(*replies), online=lambda: True, clock=Clock(), c=c,
                     start=False).attach(world)


def named_thing_leaves(scene, world, namer, key='remote', at=(40, 30), thing='thing:1', guess=REMOTE):
    """An unnamed object put on the table becomes `thing`, Grok names it, then a hand picks it up and
    carries it out by the left edge: EXITED_VIEW while its zone is 'table' (a table departure)."""
    scene.thing(key, *at)
    scene.run(world, 1.5)
    assert world.get(thing).status == Status.VISIBLE
    assert namer.step() is True
    assert world.thing_guess(thing) == guess
    scene.hand(1, *at)
    scene.run(world, 0.3)
    scene.remove(key)
    scene.run(world, 1.0)
    assert world.get(thing).status == Status.HELD
    scene.hand(1, 5, at[1])
    scene.run(world, 0.3)
    scene.hand_off(1)
    events = scene.run(world, 1.0)
    assert [e.type for e in events if e.obj == thing] == [EventType.EXITED_VIEW]
    assert thing in world._departures


def room_thing(scene, world, tid, guess=REMOTE, zone='couch', box=COUCH_BOX):
    """A new room 'thing' track's first observation (0.5 s later), with Grok's name for its crop (or
    None: not named yet)."""
    trk = appear(scene, world, tid, cls='thing', zone=zone, box=box)
    trk.guess = guess
    trk.changed = True                 # it arrived: its spot changed when first seen
    return trk


def handed_over(scene, world, namer=None, tid='r:1'):
    named_thing_leaves(scene, world, namer or namer_for(world, REMOTE))
    trk = room_thing(scene, world, tid, guess={"name": "remote", "also": [], "confidence": 0.6})
    assert types(seen(scene, world, trk, zone='couch')) == [EventType.FOUND]
    return trk


# ----- the handoff --------------------------------------------------------------------------------

def test_a_named_thing_carried_to_the_couch_is_found_there_tentatively(scene, world):
    named_thing_leaves(scene, world, namer_for(world, REMOTE))
    trk = room_thing(scene, world, 'r:1', guess={"name": "remote", "also": [], "confidence": 0.6})
    (ev,) = seen(scene, world, trk, zone='couch')
    assert (ev.type, ev.obj, ev.t) == (EventType.FOUND, 'thing:1', scene.t)
    ent = world.get('thing:1')
    assert (ent.status, ent.zone, ent.pos_cm, ent.box_cm, ent.parent) == (Status.VISIBLE, 'couch', None, None, None)
    assert (trk.role, trk.entity) == ('assoc', 'thing:1')
    assert world.resolve('thing:1') == (None, ['thing:1'])
    p = world.place('thing:1', now=scene.t)
    assert (p.kind, p.zone, p.say, p.via) == ('room', 'couch', 'the couch', 'thing:1')
    assert p.tentative and p.fresh and p.arrival_observed and p.box_px == COUCH_BOX
    assert world.room_json()['thing:1']['tentative'] is True
    assert 'thing:1' not in world._departures                 # consumed


def test_a_known_prop_place_is_not_tentative(scene, world):
    depart(scene, world)
    trk = appear(scene, world, 'r:1')
    assert types(seen(scene, world, trk)) == [EventType.FOUND]
    assert world.place('keys', now=scene.t).tentative is False
    assert 'tentative' not in world.room_json()['keys']       # a prop's entry keeps its M0 shape


def test_names_that_do_not_match_hand_over_nothing(scene, world):
    """A mug left the table; the thing on the couch is a remote: not the mug."""
    named_thing_leaves(scene, world, namer_for(world, MUG), key='mug', guess=MUG)
    trk = room_thing(scene, world, 'r:1', guess=REMOTE)
    assert seen(scene, world, trk, zone='couch') == []
    assert (trk.role, trk.entity) == ('ignored', None)
    mug = world.get('thing:1')
    assert (mug.status, mug.zone) == (Status.GONE, 'table')
    assert 'thing:1' in world._departures and world._room == {}


def test_an_unnamed_room_track_waits_for_its_name_then_is_handed_over(scene, world):
    named_thing_leaves(scene, world, namer_for(world, REMOTE))
    trk = room_thing(scene, world, 'r:1', guess=None)
    assert seen(scene, world, trk, zone='couch') == []
    assert trk.role == 'pending'
    assert seen(scene, world, trk, zone='couch') == []
    trk.guess = {"name": "tv remote", "also": ["remote control"], "confidence": 0.8}   # Grok answered
    assert types(seen(scene, world, trk, zone='couch')) == [EventType.FOUND]
    assert world.get('thing:1').zone == 'couch' and trk.entity == 'thing:1'


def test_a_room_track_still_unnamed_after_the_wait_is_given_up_for_good(scene, world):
    world.room_cfg.thing_name_wait_s = 2.0
    named_thing_leaves(scene, world, namer_for(world, REMOTE))
    trk = room_thing(scene, world, 'r:1', guess=None)
    assert seen(scene, world, trk, zone='couch') == []
    assert trk.role == 'pending'
    assert seen(scene, world, trk, zone='couch', dt=2.0) == []
    assert trk.role == 'ignored'
    trk.guess = REMOTE                                        # too late: 'ignored' is final
    assert seen(scene, world, trk, zone='couch') == []
    assert trk.role == 'ignored' and world.get('thing:1').zone == 'table'
    assert 'thing:1' in world._departures


def test_a_room_track_first_seen_before_the_departure_is_not_the_thing(scene, world):
    world.room_cfg.thing_name_wait_s = 3.0
    namer = namer_for(world, REMOTE)
    trk = room_thing(scene, world, 'r:1')                     # already on the couch
    named_thing_leaves(scene, world, namer)
    assert seen(scene, world, trk, zone='couch') == []
    assert trk.role == 'ignored'                              # it could never be: first_seen is too old
    assert world.get('thing:1').zone == 'table' and 'thing:1' in world._departures


def test_two_same_named_departed_things_are_one_object_the_latest_is_handed_over(scene, world):
    """Two departed things both named remote are taken for duplicates of one remote (the corner camera
    re-births an object while a hand places it; one of each object is the M0 demo assumption): the latest
    is handed over and the other's departure dropped. Revised after the trial run (was: nothing)."""
    namer = namer_for(world, REMOTE, {"name": "remote", "also": ["tv remote"], "confidence": 0.8})
    named_thing_leaves(scene, world, namer, key='remote', at=(40, 30))
    named_thing_leaves(scene, world, namer, key='remote2', at=(40, 50), thing='thing:2',
                       guess={"name": "remote", "also": ["tv remote"], "confidence": 0.8})
    trk = room_thing(scene, world, 'r:1')
    assert [(e.type, e.obj) for e in seen(scene, world, trk, zone='couch')] == [(EventType.FOUND, 'thing:2')]
    assert world.get('thing:1').zone == 'table' and 'thing:1' not in world._departures


def test_no_handoff_by_name_while_a_same_named_thing_is_still_on_the_table(scene, world):
    """Live Sun 27 Sep 04:42: a 4 s duplicate of the pill bottle (born while someone leaned over the table)
    was lost, and a counter object Grok also called a pill bottle took its identity while the real one lay
    on the table. A same-named thing still on the table is what the name means: nothing is handed over."""
    namer = namer_for(world, REMOTE, REMOTE)
    scene.thing('remote', 80, 30)
    scene.run(world, 1.5)
    assert namer.step() is True and world.thing_guess('thing:1') == REMOTE
    named_thing_leaves(scene, world, namer, key='dup', at=(40, 50), thing='thing:2')
    assert world.get('thing:1').status == Status.VISIBLE
    trk = room_thing(scene, world, 'r:1')
    assert seen(scene, world, trk, zone='couch') == []
    assert world.get('thing:2').zone == 'table' and world.get('thing:1').zone == 'table'


def test_two_new_room_tracks_hand_over_nothing(scene, world):
    named_thing_leaves(scene, world, namer_for(world, REMOTE))
    a = room_thing(scene, world, 'r:1')
    b = room_thing(scene, world, 'r:2', box=COUCH_BOX_2)
    assert seen(scene, world, a, b, zone='couch') == []
    assert (a.role, b.role) == ('pending', 'pending')
    assert world.get('thing:1').zone == 'table'


def test_the_first_confirmed_same_named_arrival_takes_the_handoff(scene, world):
    """Zones are visited one at a time, so of two arrivals Grok both calls a remote the first confirmed
    takes it; the departure is used up, so the second gets nothing (an unnamed arrival never competes)."""
    named_thing_leaves(scene, world, namer_for(world, REMOTE))
    shelf = room_thing(scene, world, 'r:1', zone='bookshelf', box=SHELF_BOX, guess=dict(REMOTE))
    couch = room_thing(scene, world, 'r:2')
    assert types(seen(scene, world, shelf, zone='bookshelf')) == [EventType.FOUND]
    assert seen(scene, world, couch, zone='couch') == []
    assert world.get('thing:1').zone == 'bookshelf' and couch.role != 'assoc'

def test_the_departure_is_consumed_by_one_handoff(scene, world):
    handed_over(scene, world)
    trk = room_thing(scene, world, 'r:2', zone='bookshelf', box=SHELF_BOX)
    assert seen(scene, world, trk, zone='bookshelf') == []
    assert trk.entity is None and trk.role == 'pending'
    assert world.get('thing:1').zone == 'couch' and world._room['thing:1'].track == 'r:1'


# ----- refresh, absence, reacquire ------------------------------------------------------------------

def test_its_own_thing_track_only_refreshes(scene, world):
    trk = handed_over(scene, world)
    for _ in range(3):
        assert seen(scene, world, trk, zone='couch') == []
    st = world._room['thing:1']
    assert (st.track, st.seen_t, st.misses, st.tentative) == ('r:1', scene.t, 0, True)
    assert world.get('thing:1').last_seen == scene.t
    assert world.place('thing:1', now=scene.t).fresh


def test_misses_make_a_thing_absent_in_its_zone_and_a_new_track_there_reacquires_it(scene, world):
    trk = handed_over(scene, world)
    events = []
    for _ in range(3):
        events += missed(scene, world, trk, zone='couch')
    assert types(events) == [EventType.LOST_TRACK]
    ent = world.get('thing:1')
    assert (ent.status, ent.zone) == (Status.UNKNOWN, 'couch')
    assert 'thing:1' not in world._departures                 # room absence is not a departure
    p = world.place('thing:1', now=scene.t)
    assert p.absent and p.tentative and p.zone == 'couch'
    assert world._room['thing:1'].track is None               # the tracker dropped it
    again = room_thing(scene, world, 'r:2', box=COUCH_BOX_2)
    assert types(seen(scene, world, again, zone='couch')) == [EventType.FOUND]
    st = world._room['thing:1']
    assert (st.track, st.absent, st.tentative, st.arrival_observed) == ('r:2', False, True, False)
    assert (again.role, again.entity) == ('assoc', 'thing:1')
    assert world.get('thing:1').status == Status.VISIBLE
    assert seen(scene, world, again, zone='couch') == []      # now its own track: refresh


# ----- props and the hook ---------------------------------------------------------------------------

def test_props_are_unaffected_by_thing_tracks(scene, world):
    depart(scene, world)                                      # the keys leave the table
    trk = room_thing(scene, world, 'r:1', zone='bookshelf', box=SHELF_BOX,
                     guess={"name": "keys", "also": ["key ring"], "confidence": 0.9})
    assert seen(scene, world, trk) == []
    assert world.get('keys').zone == 'table' and 'keys' in world._departures
    keys = appear(scene, world, 'r:2')
    assert types(seen(scene, world, keys)) == [EventType.FOUND]
    assert world.get('keys').zone == 'bookshelf' and world.place('keys', now=scene.t).tentative is False


def test_without_a_namer_no_thing_has_a_guess(world):
    assert world.thing_guess('thing:1') is None


def test_a_grab_read_as_covered_by_something_unknown_is_a_departure_too(scene, world):
    """Rig run (Sat 26 Sep, corner camera): the pickup of the remote was logged COVERED by 'unknown' (no
    hand seen), so no handoff happened although the couch zone saw a 'remote control' right after."""
    namer = namer_for(world, REMOTE)
    scene.thing('remote', 40, 30)
    scene.run(world, 1.5)
    assert namer.step() is True
    ent = world.get('thing:1')
    ev = world._apply_verdict('thing:1', ent, (Status.UNDER, 'unknown', 0.6, EventType.COVERED))
    assert [e.type for e in ev] == [EventType.COVERED] and 'thing:1' in world._departures
    scene.remove('remote')                                  # carried away under the arm: gone from view, so the
    world._bits['thing:1'].clear()                          # presence debounce restarts, as it would have after
    world._present['thing:1'] = False                       # the object vanished (the verdict came after that)
    trk = room_thing(scene, world, 'r:1', guess={"name": "remote control", "also": [], "confidence": 0.7})
    assert types(seen(scene, world, trk, zone='couch')) == [EventType.FOUND]
    assert world.get('thing:1').zone == 'couch' and world.place('thing:1', now=scene.t).tentative


def test_covered_by_a_known_cover_is_not_a_departure(scene, world):
    namer = namer_for(world, REMOTE)
    scene.thing('remote', 40, 30)
    scene.run(world, 1.5)
    namer.step()
    world._apply_verdict('thing:1', world.get('thing:1'), (Status.UNDER, 'notebook', 0.85, EventType.COVERED))
    assert 'thing:1' not in world._departures
    trk = room_thing(scene, world, 'r:1', guess={"name": "remote control", "also": [], "confidence": 0.7})
    seen(scene, world, trk, zone='couch')
    assert world.get('thing:1').zone == 'table'


def test_handoff_hints_are_the_names_of_things_that_just_left(scene, world):
    assert world.room_handoff_hints(scene.t) == []
    named_thing_leaves(scene, world, namer_for(world, REMOTE))
    assert world.room_handoff_hints(scene.t) == [REMOTE]
    assert world.room_handoff_hints(scene.t + world.room_cfg.handoff_s + 1) == []    # too long ago
    trk = room_thing(scene, world, 'r:1', guess=dict(REMOTE))
    seen(scene, world, trk, zone='couch')
    assert world.room_handoff_hints(scene.t) == []                                   # consumed


def test_a_carried_thing_is_handed_off_before_a_stale_room_copy_reacquires(scene, world):
    """Trial run 1: an old couch copy of the remote (missing from the couch) took the new couch track, and
    the carried thing's departure stayed open. The handoff wins now."""
    trk = handed_over(scene, world)                           # thing:1 on the couch
    missed(scene, world, trk, zone='couch')                    # it's picked up from the couch (missing there)
    named_thing_leaves(scene, world, namer_for(world, REMOTE), key='remote2', thing='thing:2')
    new = room_thing(scene, world, 'r:9', guess=dict(REMOTE))
    evs = seen(scene, world, new, zone='couch')
    assert [(e.type, e.obj) for e in evs] == [(EventType.FOUND, 'thing:2')]
    assert 'thing:2' not in world._departures


def test_same_named_duplicates_count_as_one_candidate(scene, world):
    """Trial run: one placement made thing:1 and thing:2 (both 'remote control'); both left the table.
    A couch track named remote is the latest of them, and the other's departure is dropped."""
    namer = namer_for(world, REMOTE)
    named_thing_leaves(scene, world, namer)
    named_thing_leaves(scene, world, namer, key='remote2', thing='thing:2')
    trk = room_thing(scene, world, 'r:1', guess=dict(REMOTE))
    evs = seen(scene, world, trk, zone='couch')
    assert [(e.type, e.obj) for e in evs] == [(EventType.FOUND, 'thing:2')]
    assert world._departures == {}


def test_static_clutter_is_never_handed_off_even_with_a_matching_name(scene, world):
    """Trial run: the stove panel, re-proposed every few seconds, got a Grok 'remote control'. A track
    whose spot never changed did not arrive: no handoff, whatever its name."""
    named_thing_leaves(scene, world, namer_for(world, REMOTE))
    trk = room_thing(scene, world, 'r:1', zone='bookshelf', box=SHELF_BOX, guess=dict(REMOTE))
    trk.changed = False
    assert seen(scene, world, trk, zone='bookshelf') == []
    assert world.get('thing:1').zone == 'table'


def test_an_unnamed_new_track_does_not_block_a_handoff(scene, world):
    """Trial run: stove / counter clutter Grok can't name stayed pending and blocked the couch handoff."""
    named_thing_leaves(scene, world, namer_for(world, REMOTE))
    clutter = room_thing(scene, world, 'r:7', zone='bookshelf', box=SHELF_BOX, guess=None)
    seen(scene, world, clutter, zone='bookshelf')
    trk = room_thing(scene, world, 'r:8', guess=dict(REMOTE))
    assert types(seen(scene, world, trk, zone='couch')) == [EventType.FOUND]


def test_a_thing_that_barely_sat_on_the_table_does_not_make_the_room_hot(scene, world):
    """Rig: a foot at the table edge is born, named 'sneaker' and gone 3 s later. It may still be handed
    off (hints stay), but it must not switch on the fast cadence, which halves the table's fps; a thing
    that sat for handoff_min_dwell_s does, and only for hot_max_s."""
    namer = namer_for(world, {"name": "pen", "also": ["ballpoint"], "confidence": 0.8}, REMOTE)
    scene.thing('pen', 40, 30)
    scene.run(world, 1.5)
    assert namer.step() is True
    scene.remove('pen')
    scene.run(world, 3.0)
    assert world.get('thing:1').status != Status.VISIBLE and 'thing:1' in world._departures
    assert len(world.room_handoff_hints(scene.t)) == 1 and world.room_hot(scene.t) is False
    named_thing_leaves(scene, world, namer, key='remote', at=(60, 30), thing='thing:2')
    world._placed_t['thing:2'] = world._departures['thing:2'][0] - 5.0      # sat 5 s before leaving
    assert world.room_hot(scene.t) is True
    world.room_cfg.hot_max_s = 0.5
    assert world.room_hot(scene.t + 1.0) is False and len(world.room_handoff_hints(scene.t + 1.0)) == 2


def test_a_departed_thing_named_like_clothing_is_never_handed_off(scene, world):
    """Rig: the couch-sitter's feet were born on the table edge as 'sock', 'gone', and handed off to
    their own arrival on the couch. Body and clothing names never open a handoff. (A 'sock' is no longer
    kept as a name at all, core.auto_name.NOT_OBJECTS; a shoe can be a real object, so it is named.)"""
    sock = {"name": "white sneaker", "also": ["sneaker"], "confidence": 0.8}
    named_thing_leaves(scene, world, namer_for(world, sock), guess=sock)
    assert world.room_handoff_hints(scene.t) == [] and world.room_hot(scene.t) is False
    trk = room_thing(scene, world, 'r:1', guess=dict(sock))
    assert seen(scene, world, trk, zone='couch') == []
    assert world.get('thing:1').zone == 'table'
