"""Core world-model rules (spec 3.5): debounce, observation, contacts, disappearance, HELD, decay."""
import json
from pathlib import Path

import pytest

from core.config import Config
from core.types import EventType, Status
from core.world import World
from tests.synth import Scene

ROOT = Path(__file__).resolve().parents[1]


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


def test_first_sighting_becomes_visible_without_event(scene, world):
    scene.place('keys', 40, 30)
    events = scene.run(world, 1.0)
    keys = world.get('keys')
    assert keys.status == Status.VISIBLE
    assert keys.pos_cm == pytest.approx((40, 30))
    assert keys.confidence == 1.0
    assert events == []


@pytest.mark.parametrize('missed', [2, 3])
def test_visible_object_survives_a_few_missed_batches(scene, world, missed):
    scene.place('keys', 40, 30)
    scene.run(world, 1.0)
    scene.remove('keys')
    events = scene.run(world, missed * scene.dt)
    scene.place('keys', 40, 30)
    events += scene.run(world, 0.5)
    assert world.get('keys').status == Status.VISIBLE
    assert events == []


def test_vanishing_without_a_hand_is_lost_track(scene, world):
    scene.place('keys', 40, 30)
    scene.run(world, 1.0)
    scene.remove('keys')
    events = scene.run(world, world.cfg.lost_grace_s + 0.2)       # unseen that long: lost
    keys = world.get('keys')
    assert types(events) == [EventType.LOST_TRACK]
    assert keys.status == Status.UNKNOWN
    assert keys.parent is None
    assert keys.pos_cm == pytest.approx((40, 30))
    assert events[0].obj == 'keys'


def pick_up(scene, world, name='keys', hid=1, at=(40, 30)):
    """Place, settle, put a hand over the object, then lift it (hand stays). Returns the pickup events."""
    scene.place(name, *at)
    scene.run(world, 1.0)
    scene.hand(hid, *at)
    scene.run(world, 0.3)
    scene.remove(name)
    return scene.run(world, 1.0)


def test_pick_up_makes_object_held_by_touching_hand(scene, world, cfg):
    events = pick_up(scene, world)
    keys = world.get('keys')
    assert types(events) == [EventType.PICKED_UP]
    assert keys.status == Status.HELD
    assert keys.parent == 'hand:1'
    assert keys.pre_pickup_pos == pytest.approx((40, 30))
    assert events[0].parent == 'hand:1'
    assert events[0].confidence == pytest.approx(cfg.conf_held)


def test_contact_window_is_measured_from_last_seen_not_from_now(scene, world):
    scene.place('keys', 40, 30)
    scene.run(world, 1.0)
    scene.hand(1, 40, 30)
    scene.run(world, 0.1)             # last contact
    scene.hand(1, 70, 50)             # hand moves away but stays in view
    scene.run(world, 0.8)             # object still seen 0.8 s after the touch
    scene.remove('keys')
    events = scene.run(world, 0.9)    # debounce declares absence ~0.9 s later: 1.7 s after contact
    assert types(events) == [EventType.PICKED_UP]
    assert world.get('keys').status == Status.HELD


def put_down(scene, world, at, name='keys', hid=1):
    scene.hand(hid, *at)
    scene.place(name, *at)
    return scene.run(world, 1.0)


def test_put_down_near_pickup_spot_is_put_back(scene, world):
    pick_up(scene, world)
    events = put_down(scene, world, (43, 30))
    assert types(events) == [EventType.PUT_BACK]
    assert world.get('keys').status == Status.VISIBLE
    assert world.get('keys').parent is None


def test_put_down_far_from_pickup_spot_is_moved_with_from_and_to(scene, world):
    pick_up(scene, world)
    events = put_down(scene, world, (60, 50))
    assert types(events) == [EventType.MOVED]
    assert events[0].from_cm == pytest.approx((40, 30))
    assert events[0].to_cm == pytest.approx((60, 50))
    assert events[0].confidence == 1.0
    assert world.get('keys').pos_cm == pytest.approx((60, 50))


def test_holding_hand_leaving_by_left_edge_is_exited_view(scene, world):
    pick_up(scene, world)
    scene.hand(1, 5, 30)              # hand box x 1-11 cm: inside the 6.4 cm left margin
    scene.run(world, 0.3)
    scene.hand_off(1)
    events = scene.run(world, 1.0)
    keys = world.get('keys')
    assert types(events) == [EventType.EXITED_VIEW]
    assert events[0].edge == 'left'
    assert keys.status == Status.GONE
    assert keys.edge == 'left'
    assert keys.parent is None


def test_holding_hand_vanishing_mid_table_is_lost_track(scene, world):
    pick_up(scene, world)
    scene.hand_off(1)
    events = scene.run(world, 1.0)
    keys = world.get('keys')
    assert types(events) == [EventType.LOST_TRACK]
    assert keys.status == Status.UNKNOWN
    assert keys.parent is None
    assert keys.edge is None


def test_held_longer_than_timeout_is_lost_track_even_with_hand_present(scene, world, cfg):
    pick_up(scene, world)
    scene.hand(1, 60, 40)
    assert scene.run(world, cfg.held_timeout_s - 1) == []
    events = scene.run(world, 2.0)
    assert types(events) == [EventType.LOST_TRACK]
    assert world.get('keys').status == Status.UNKNOWN


def test_two_touching_hands_give_ambiguous_held_with_candidates(scene, world, cfg):
    scene.place('keys', 40, 30)
    scene.run(world, 1.0)
    scene.hand(1, 36, 30)
    scene.hand(2, 44, 30)
    scene.run(world, 0.3)
    scene.hand(1, 20, 20)             # hand 1 withdraws first; hand 2 was the last to touch
    scene.run(world, 0.1)
    scene.remove('keys')
    events = scene.run(world, 1.0)
    keys = world.get('keys')
    assert types(events) == [EventType.PICKED_UP]
    assert keys.status == Status.HELD
    assert keys.parent == 'hand:2'
    assert keys.candidates == ['hand:1']
    assert events[0].confidence == pytest.approx(cfg.conf_held * cfg.ambiguity_penalty)


def test_reappearing_after_gone_is_corrected(scene, world):
    pick_up(scene, world)
    scene.hand(1, 5, 30)
    scene.run(world, 0.3)
    scene.hand_off(1)
    scene.run(world, 1.0)
    assert world.get('keys').status == Status.GONE
    scene.place('keys', 70, 40)
    events = scene.run(world, 1.0)
    keys = world.get('keys')
    assert types(events) == [EventType.CORRECTED]
    assert keys.status == Status.VISIBLE
    assert keys.edge is None
    assert events[0].to_cm == pytest.approx((70, 40))


def test_reappearing_after_lost_track_is_corrected(scene, world):
    scene.place('keys', 40, 30)
    scene.run(world, 1.0)
    scene.remove('keys')
    scene.run(world, world.cfg.lost_grace_s + 0.2)
    scene.place('keys', 40, 30)
    assert types(scene.run(world, 1.0)) == [EventType.CORRECTED]


class CoveredWorld(World):
    """Stands in for the integrator's cover rule: every vanishing object is under the notebook."""

    def _hidden_by(self, name, ent, box_cm, now, frame):
        return (Status.UNDER, 'notebook', self.cfg.conf_under, EventType.COVERED)


class BoxedWorld(World):
    """Stands in for the integrator's container rule: held objects go straight into the box."""

    def _check_inside(self, name, ent, now):
        return (Status.INSIDE, 'box', self.cfg.conf_inside, EventType.PUT_INSIDE)


def test_hidden_by_hook_verdict_is_applied_and_uncovering_is_reported(scene, cfg):
    world = CoveredWorld(cfg)
    scene.place('keys', 40, 30)
    scene.run(world, 1.0)
    scene.remove('keys')
    events = scene.run(world, 1.0)
    assert types(events) == [EventType.COVERED]
    assert events[0].parent == 'notebook'
    assert world.get('keys').status == Status.UNDER
    assert events[0].confidence == pytest.approx(cfg.conf_under)
    scene.place('keys', 40, 30)
    assert types(scene.run(world, 1.0)) == [EventType.UNCOVERED]


def test_check_inside_hook_runs_before_hand_rules_and_taking_out_is_reported(scene, cfg):
    world = BoxedWorld(cfg)
    events = pick_up(scene, world)
    assert types(events) == [EventType.PICKED_UP, EventType.PUT_INSIDE]
    assert world.get('keys').parent == 'box'
    scene.place('keys', 40, 30)
    assert types(scene.run(world, 1.0)) == [EventType.TAKEN_OUT]


def test_confidence_decays_for_hidden_beliefs_but_not_for_visible_ones(scene, world, cfg):
    scene.place('wallet', 80, 40)
    scene.place('keys', 40, 30)
    scene.run(world, 1.0)
    scene.remove('keys')
    scene.run(world, cfg.lost_grace_s + 0.2)
    before = world.get('keys').confidence
    scene.dt = 60.0                   # one batch per simulated minute
    scene.run(world, 10 * 60.0)
    assert world.get('keys').status == Status.UNKNOWN
    assert world.get('keys').confidence == pytest.approx(before * cfg.decay_per_min ** 10)
    assert world.get('wallet').confidence == 1.0


def test_held_confidence_decays_while_held(scene, world, cfg):
    (pickup,) = pick_up(scene, world)
    scene.run(world, 12.0)
    keys = world.get('keys')
    minutes = (scene.t - pickup.t) / 60
    assert keys.status == Status.HELD
    assert keys.confidence == pytest.approx(cfg.conf_held * cfg.decay_per_min ** minutes)
    assert keys.confidence < cfg.conf_held


def test_observe_external_marks_found_and_visible_in_zone(scene, world):
    pick_up(scene, world)
    scene.hand_off(1)
    scene.run(world, 1.0)
    events = world.observe_external('keys', (150.0, 20.0), 'floor')
    keys = world.get('keys')
    assert types(events) == [EventType.FOUND]
    assert events[0].to_cm == pytest.approx((150, 20))
    assert events[0].t == scene.t
    assert keys.status == Status.VISIBLE
    assert keys.zone == 'floor'
    assert keys.pos_cm == pytest.approx((150, 20))
    assert keys.parent is None
    assert keys.confidence == 1.0
    assert keys.last_seen == scene.t


def test_entity_in_another_zone_ignores_overhead_absence(scene, world):
    scene.place('keys', 125, 30)
    scene.run(world, 1.0)
    world.observe_external('keys', (150.0, 20.0), 'floor')
    scene.remove('keys')              # fell off the table: overhead camera no longer sees it
    events = scene.run(world, 2.0)
    assert events == []
    assert world.get('keys').status == Status.VISIBLE
    assert world.get('keys').zone == 'floor'


class FakeLog:
    """Minimal EventLog: records add() calls and serves last() newest first."""

    def __init__(self):
        self.added = []

    def add(self, ev, frame):
        self.added.append((ev, frame))

    def last(self, obj, n):
        return [ev for ev, _ in reversed(self.added) if ev.obj == obj][:n]


def test_history_is_newest_first_in_memory(scene, world):
    pick_up(scene, world)
    put_down(scene, world, (60, 50))
    scene.place('wallet', 80, 40)
    scene.run(world, 1.0)
    assert types(world.history('keys')) == [EventType.MOVED, EventType.PICKED_UP]
    assert types(world.history('keys', n=1)) == [EventType.MOVED]
    assert world.history('wallet') == []


def test_events_go_to_the_sink_with_their_frame_and_history_reads_it(scene, cfg):
    log = FakeLog()
    world = World(cfg, events=log)
    events = pick_up(scene, world)
    (ev, frame), = log.added
    assert ev is events[0]
    assert frame.t == ev.t
    world.observe_external('keys', (150.0, 20.0), 'floor')
    assert log.added[-1][1] is None
    assert types(world.history('keys')) == [EventType.FOUND, EventType.PICKED_UP]


def test_resolve_follows_entity_parents_to_the_outermost_positioned_one(scene, cfg):
    world = CoveredWorld(cfg)
    scene.place('notebook', 60, 40)
    scene.place('keys', 60, 40)
    scene.run(world, 1.0)
    scene.remove('keys')
    scene.run(world, 1.0)
    assert world.resolve('keys') == (pytest.approx((60, 40)), ['keys', 'notebook'])


def test_resolve_stops_at_a_hand_and_uses_last_known_position(scene, world):
    pick_up(scene, world)
    assert world.resolve('keys') == (pytest.approx((40, 30)), ['keys'])


def test_resolve_never_seen_object_has_no_position(world):
    assert world.resolve('remote') == (None, ['remote'])


def test_state_json_is_serialisable_and_lists_edges(scene, world):
    scene.place('wallet', 80, 40)
    pick_up(scene, world)
    state = world.state_json()
    json.dumps(state)
    assert state['t'] == scene.t
    by_name = {e['name']: e for e in state['entities']}
    assert set(by_name) == set(world.cfg.names())
    keys = by_name['keys']
    assert keys['status'] == 'HELD'
    assert keys['parent'] == 'hand:1'
    assert keys['kind'] == 'target'
    assert keys['resolved_cm'] == pytest.approx([40, 30])
    assert set(keys) == {'name', 'kind', 'status', 'parent', 'pos_cm', 'resolved_cm', 'confidence',
                         'candidates', 'last_seen', 'zone', 'edge'}
    assert ['keys', 'HELD', 'hand:1'] in state['edges']
    assert ['wallet', 'ON', 'table'] in state['edges']
    assert not any(edge[0] == 'remote' for edge in state['edges'])


def test_reset_returns_everything_to_never_seen(scene, world):
    pick_up(scene, world)
    world.reset()
    keys = world.get('keys')
    assert (keys.status, keys.parent, keys.pos_cm, keys.last_seen, keys.confidence) == \
        (Status.UNKNOWN, None, None, None, 0.0)
    assert world.history('keys') == []
    scene.hand_off(1)
    scene.place('keys', 40, 30)
    assert scene.run(world, 1.0) == []     # a first sighting again, not a CORRECTED
    assert world.state_json()['edges'] == [['keys', 'ON', 'table']]


LAYOUT = {'keys': (15, 15), 'pill_bottle': (35, 15), 'wallet': (55, 15), 'glasses': (75, 15),
          'phone': (95, 15), 'remote': (115, 15), 'box': (30, 50), 'notebook': (90, 50)}


def test_static_scene_for_60_s_emits_nothing_after_first_sightings(scene, world):
    for name, at in LAYOUT.items():
        scene.place(name, *at)
    assert scene.run(world, 1.0) == []
    events = []
    for i in range(600):
        for k, (name, at) in enumerate(LAYOUT.items()):   # detector misses ~1 batch in 4 per object
            if (i + k) % 4 == 0:
                scene.remove(name)
            else:
                scene.place(name, *at)
        scene.hand(1, 60 + 10 * ((i // 20) % 3), 34)       # a hand wandering in empty space
        events += world.update(*scene.step())
    assert events == []
    assert all(world.get(n).status == Status.VISIBLE for n in LAYOUT)


def test_put_down_then_quick_hand_withdrawal_is_put_back_not_lost(scene, world):
    pick_up(scene, world)
    scene.place('keys', 40, 30)
    scene.hand_off(1)                 # hand drops the keys and leaves at once
    events = scene.run(world, 0.2)
    scene.remove('keys')              # one missed detection delays the debounce past hand_lost_s
    events += scene.run(world, 0.1)
    scene.place('keys', 40, 30)
    events += scene.run(world, 1.0)
    assert types(events) == [EventType.PUT_BACK]


def test_history_of_a_merged_thing_is_ordered_by_wall_time(cfg, tmp_path):
    """Merged identities' events are interleaved by wall time: monotonic t restarts with every boot, so an
    event from an earlier run can carry a larger t than today's."""
    from core.events import EventLog
    from core.types import Entity, Event
    log = EventLog(':memory:', str(tmp_path))
    world = World(cfg, events=log)
    for n in ('thing:1', 'thing:2'):
        world.entities[n] = Entity(name=n, kind='target')
    world.entities['thing:2'].merged_into = 'thing:1'
    log.add(Event(t=9000.0, wall=1000.0, obj='thing:2', type=EventType.APPEARED))    # earlier run
    log.add(Event(t=50.0, wall=2000.0, obj='thing:1', type=EventType.MOVED))         # this run
    assert types(world.history('thing:1')) == [EventType.MOVED, EventType.APPEARED]
    log.close()
