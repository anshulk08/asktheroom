"""WHERE from the room tracker's named tracks (room.aim_tracks): a fresh Grok-named track on the couch that
fits the asked name answers and is aimed at through the room aim path; worn things, stale, missed,
unconfirmed or tentative tracks never; a table thing wins."""
import copy
from pathlib import Path

import pytest

from core.auto_name import AutoNameConfig, AutoNamer
from core.config import Config, load_config
from core.room_types import RoomTrack
from core.world import World
from tests.synth import Scene
from voice.pipeline import make_ask
from voice.room_tracks import answer_from_tracks, worn_or_person

ROOT = Path(__file__).resolve().parents[1]
RAW = load_config(ROOT / 'config.yaml')
NOW = 5000.0
SAYS = {'couch': 'the couch', 'side_table': 'the side table'}


def track(tid='r:387', zone='couch', name='laptop', box=(1200, 700, 1500, 860), last=NOW - 1.0, **kw):
    t = RoomTrack(tid=tid, zone=zone, cls='thing', box_px=box, first_seen=1.0, first_wall=NOW - 30,
                  last_seen=2.0, last_wall=last, confirmed=True,
                  guess={'name': name, 'also': [], 'confidence': 0.9} if name else None)
    for k, v in kw.items():
        setattr(t, k, v)
    return t


def test_a_named_couch_track_answers_and_aims_at_its_box():
    a = answer_from_tracks('laptop', [track()], SAYS, NOW)
    assert a.text == 'Your laptop is on the couch.'
    assert a.action == 'room:1350,780,1200,700,1500,860'


@pytest.mark.parametrize('name', ['left leg', 'hand', 'person'])
def test_body_parts_and_people_are_never_answered_or_aimed_at(name):
    assert worn_or_person({'name': name, 'also': []})
    assert answer_from_tracks(name, [track(name=name)], SAYS, NOW) is None


@pytest.mark.parametrize('said,name', [('shoe', 'white sneaker'), ('sneaker', 'shoe'), ('shoe', 'shoe')])
def test_a_shoe_is_a_valid_target(said, name):
    """The user's call: shoes by the table are the biggest laser targets. Whether someone is wearing it is
    the aim gate's (test_worn_target_gate below)."""
    a = answer_from_tracks(said, [track(name=name, zone='side_table')], SAYS, NOW)
    assert a is not None and a.text.endswith('on the side table.') and a.action.startswith('room:')


@pytest.mark.parametrize('name', ['eyeglasses', 'sunglasses', 'reading glasses', 'spectacles', 'pair of glasses'])
def test_glasses_answer_to_their_synonyms(name):
    a = answer_from_tracks('glasses', [track(name=name)], SAYS, NOW)
    assert a is not None and a.text == 'Your glasses are on the couch.'


@pytest.mark.parametrize('kw', [dict(last=NOW - 30), dict(misses=1), dict(confirmed=False), dict(name=None)])
def test_stale_missed_unconfirmed_or_unnamed_tracks_are_not_used(kw):
    assert answer_from_tracks('laptop', [track(**kw)], SAYS, NOW) is None


def test_a_tentative_handed_over_things_track_is_left_to_the_world():
    t = track(entity='thing:7', role='assoc')
    assert answer_from_tracks('laptop', [t], SAYS, NOW, tentative=lambda n: n == 'thing:7') is None


def test_the_best_fitting_name_wins():
    a = answer_from_tracks('laptop', [track('r:1', 'side_table', name='laptop charger'),
                                      track('r:2', 'couch', name='laptop')], SAYS, NOW)
    assert a.text.endswith('on the couch.')


# ----- the ask pipeline ---------------------------------------------------------------------------

class NoGrok:
    def narrate(self, system, parts, schema):
        raise RuntimeError('offline')


def ask_for(world, tracks, on=True):
    raw = copy.deepcopy(RAW)
    raw['room'] = dict(raw.get('room') or {}, aim_tracks=on)
    return make_ask(raw, world, world.events, clock=lambda: NOW,
                    room_tracks=lambda: (tracks, SAYS, lambda n: False))


@pytest.fixture
def world():
    from core.events import EventLog
    return World(Config.load(ROOT / 'config.yaml'), EventLog(':memory:', '/tmp/ws2_room_tracks'))


def test_where_asks_the_room_tracks_when_the_world_has_no_place(world):
    a = ask_for(world, [track()])('where is my laptop')
    assert a.text == 'Your laptop is on the couch.' and a.action.startswith('room:')


def test_off_by_default_nothing_changes(world):
    a = ask_for(world, [track()], on=False)('where is my laptop')
    assert not (a.action or '').startswith('room:')


def test_a_table_thing_wins_over_a_room_track(world):
    s = Scene(Config.load(ROOT / 'config.yaml'), fps=10, t0=1000.0)
    s.thing('laptop', 50, 30, 30, 20)
    s.run(world, 2.0)
    namer = AutoNamer(RAW, world, provider=NoGrok(), online=lambda: False, c=AutoNameConfig(enabled=True),
                      start=False).attach(world)
    namer._guesses['thing:1'] = {'name': 'laptop', 'also': [], 'confidence': 0.9}
    a = ask_for(world, [track()])('where is my laptop')
    assert not (a.action or '').startswith('room:') and 'couch' not in a.text


def test_where_are_my_glasses_aims_at_an_eyeglasses_track_on_the_couch(world):
    a = ask_for(world, [track(name='eyeglasses', box=(900, 600, 980, 640))])('where are my glasses?')
    assert a.text == 'Your glasses are on the couch.' and a.action == 'room:940,620,900,600,980,640'


# ----- the aim gate: lone footwear is not a person ------------------------------------------------

from types import SimpleNamespace  # noqa: E402

from act.room_map import beam_blocked  # noqa: E402
from main import Room  # noqa: E402

SNEAKER = (1000, 1200, 1120, 1260)
REMOTE = (1140, 1210, 1260, 1240)               # lying right next to the sneaker on the table


def gate(people, footwear, target):
    me = SimpleNamespace(_people_footwear=footwear, worn_near_px=40.0)
    blockers = Room._blockers(me, people)
    return beam_blocked(((target[0] + target[2]) / 2, (target[1] + target[3]) / 2), target, blockers, None, 120)


def test_a_remote_next_to_a_lone_sneaker_on_the_table_is_aimed_at():
    assert gate([SNEAKER], [SNEAKER], REMOTE) is False


def test_the_lone_sneaker_itself_is_aimed_at():
    assert gate([SNEAKER], [SNEAKER], SNEAKER) is False


def test_a_sneaker_touching_a_leg_blocks():
    leg = (980, 800, 1100, 1210)
    assert gate([SNEAKER, leg], [SNEAKER], REMOTE) is True


def test_a_lone_shirt_next_to_the_target_always_blocks():
    shirt = (1150, 1100, 1300, 1200)
    assert gate([shirt], [], REMOTE) is True


def test_without_footwear_boxes_nothing_changes():
    assert gate([SNEAKER], [], REMOTE) is True
