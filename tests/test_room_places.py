"""'Point to the couch': the laser shows a drawn room zone (voice/room_places.py)."""
from __future__ import annotations

from pathlib import Path

import pytest

from core.config import Config, load_config
from core.world import World
from voice.pipeline import make_ask
from voice.room_places import answer_for_zone, pick_zone

ROOT = Path(__file__).resolve().parents[1]
# the rig's zones (room_zones.json, 2560x1440 full-frame px, 27 Sep)
ZONES = [('couch', 'the couch', [[973, 1053], [1320, 1053], [1320, 1440], [906, 1440]]),
         ('side_table', 'the side table', [[0, 740], [286, 740], [286, 866], [0, 866]]),
         ('counter', 'the kitchen counter', [[2240, 720], [2560, 746], [2560, 886], [2240, 813]]),
         ('doorway', 'the floor by the doorway', [[1485, 850], [1760, 850], [1930, 1050], [1510, 1050]])]


def room_px(action):
    u, v, x1, y1, x2, y2 = (float(s) for s in action.split(':', 1)[1].split(','))
    return (u, v), (x1, y1, x2, y2)


@pytest.mark.parametrize('q, zone', [
    ('Point to the couch.', 'couch'), ('point at the sofa', 'couch'), ('Show me the couch', 'couch'),
    ('Hey Room, point to the side table.', 'side_table'), ('point to the kitchen', 'counter'),
    ('Point to the kitchen counter please.', 'counter'), ('point to the counter', 'counter'),
    ('Quite to the couch.', 'couch'),                 # whisper's 'point to'
])
def test_a_point_question_naming_a_zone_picks_it(q, zone):
    assert pick_zone(q, ZONES)[0] == zone


@pytest.mark.parametrize('q', [
    'point to the remote on the couch', 'show me whats on the side table', 'point to my keys',
    'point to the table', 'point to it', 'where is the couch',
])
def test_a_thing_the_table_or_no_zone_is_not_a_zone(q):
    assert pick_zone(q, ZONES) is None
    assert answer_for_zone(q, ZONES) is None


@pytest.mark.parametrize('q', ["don't point at the couch", 'stop pointing to the couch', 'where is the couch'])
def test_no_point_cue_or_a_negation_never_aims(q):
    assert answer_for_zone(q, ZONES) is None


@pytest.mark.parametrize('name, say, poly', ZONES)
def test_the_aim_is_inside_the_zone_and_the_box_is_its_outline(name, say, poly):
    import cv2
    import numpy as np
    a = answer_for_zone(f'point to {say}', ZONES)
    assert a.text == f"That's {say}."
    uv, box = room_px(a.action)
    assert cv2.pointPolygonTest(np.float32(poly).reshape(-1, 1, 2), uv, False) >= 0
    xs, ys = [p[0] for p in poly], [p[1] for p in poly]
    assert box == (min(xs), min(ys), max(xs), max(ys))


def test_a_concave_outline_still_aims_inside_it():
    ell = [('shelf', 'the shelf', [[0, 0], [100, 0], [100, 20], [20, 20], [20, 100], [0, 100]])]
    a = answer_for_zone('point to the shelf', ell)
    import cv2
    import numpy as np
    assert cv2.pointPolygonTest(np.float32(ell[0][2]).reshape(-1, 1, 2), room_px(a.action)[0], False) >= 0


def test_two_zones_sharing_a_word_need_the_full_name():
    zs = [('side_table', 'the side table', ZONES[1][2]), ('coffee_table', 'the coffee table', ZONES[0][2])]
    assert pick_zone('point to the coffee table', zs)[0] == 'coffee_table'
    assert pick_zone('point to the side table', zs)[0] == 'side_table'


@pytest.fixture
def world():
    from core.events import EventLog
    return World(Config.load(ROOT / 'config.yaml'), EventLog(':memory:', '/tmp/point_zones'))


def test_the_router_answers_a_zone_before_the_interpreter_and_logs_it(world):
    def interpret(text):
        raise AssertionError('a zone point never reaches the interpreter')
    ask = make_ask(load_config(), world, world.events, interpret=interpret, room_zones=lambda: ZONES)
    a = ask('point to the couch')
    assert a.text == "That's the couch." and a.action.startswith('room:')


def test_the_router_without_zones_or_for_a_thing_goes_on_as_before(world):
    ask = make_ask(load_config(), world, world.events, room_zones=lambda: ZONES)
    assert not (ask('point to my keys').action or '').startswith('room:')
    ask = make_ask(load_config(), world, world.events)
    assert ask('point to the couch').text != "That's the couch."
