"""Covers and cups the detector has no label for (room mode: the prop labels are off, so the notebook, a
cup and the keys are all unnamed thing:N, named by Grok). A thing laid over another is its cover: the
keys go UNDER it, "where are my keys?" says under the notebook, lifting it brings the keys back as
themselves, and a cup slid across the table with the keys under it takes them along (the shell game).
Rendered scenes, as on the rig: no appearance embedder (reid is off there)."""
from pathlib import Path

import pytest

from core.auto_name import AutoNameConfig, AutoNamer
from core.config import Config, load_config
from core.events import EventLog
from core.types import EventType, Status
from core.world import World
from tests.synth import Scene
from voice.answers import answer
from voice.intents import parse

ROOT = Path(__file__).resolve().parents[1]
RAW = load_config(ROOT / 'config.yaml')
KEYS_AT = (40, 30)


@pytest.fixture
def cfg():
    return Config.load(ROOT / 'config.yaml')


@pytest.fixture
def rscene(cfg):
    return Scene(cfg, fps=10, t0=1000.0, render=True)


@pytest.fixture
def world(cfg, tmp_path):
    return World(cfg, EventLog(':memory:', str(tmp_path / 'snaps')))


def of(events, obj):
    return [e.type for e in events if e.obj == obj]


def where(world, text='where are my keys'):
    return answer(parse(text, RAW), world, world.events, RAW, now=world._wall).text


class NoGrok:
    def narrate(self, system, parts, schema):
        raise RuntimeError('offline')


def guesses(world, **by_thing):
    """Grok's guesses, as core/auto_name.py attaches them on the rig: thing_1='keys' -> thing:1."""
    namer = AutoNamer(RAW, world, provider=NoGrok(), online=lambda: False, c=AutoNameConfig(enabled=True),
                      start=False).attach(world)
    for k, name in by_thing.items():
        namer._guesses[k.replace('_', ':')] = {'name': name, 'also': [], 'confidence': 0.9}
    return namer


def settle(s, w, seconds=2.5):
    return s.run(w, seconds)


def carry(s, w, key, frm, to, size, over=None, steps=5):
    """A hand picks key up at frm and puts it down at to (w x h = size); over: what it hides when laid
    there, removed as it lands. The hand then leaves."""
    s.hand(1, *frm)
    events = s.run(w, 0.4)
    s.remove(key)
    for i in range(1, steps + 1):
        x, y = frm[0] + (to[0] - frm[0]) * i / steps, frm[1] + (to[1] - frm[1]) * i / steps
        s.hand(1, x, y)
        events += w.update(*s.step())
    if over:
        s.remove(over)
    s.thing(key, *to, *size)
    events += s.run(w, 0.4)
    s.hand(1, 120, 5)
    events += s.run(w, 0.3)
    s.hand_off(1)
    return events + s.run(w, 3.0)


def slide(s, w, key, frm, to, size, steps=8):
    """A hand rests on key and slides it along the table (it stays in view and proposed)."""
    events = []
    for i in range(steps + 1):
        x, y = frm[0] + (to[0] - frm[0]) * i / steps, frm[1] + (to[1] - frm[1]) * i / steps
        s.thing(key, x, y, *size)
        s.hand(1, x, y)
        events += w.update(*s.step())
    s.hand(1, 120, 5)
    events += s.run(w, 0.3)
    s.hand_off(1)
    return events + s.run(w, 3.0)


def keys_and_notebook(s, w):
    s.thing('keys', *KEYS_AT, 6, 4)
    s.thing('notebook', 80, 40, 15, 21)
    settle(s, w)
    guesses(w, thing_1='keys', thing_2='notebook')


def test_keys_under_a_grok_named_notebook_are_under_it_and_where_says_so(rscene, world):
    keys_and_notebook(rscene, world)
    events = carry(rscene, world, 'notebook', (80, 40), KEYS_AT, (15, 21), over='keys')
    assert of(events, 'thing:1')[-1] == EventType.COVERED
    keys = world.get('thing:1')
    assert (keys.status, keys.parent) == (Status.UNDER, 'thing:2')
    assert 'under the notebook' in where(world)


def test_the_notebook_lifted_off_brings_the_keys_back_as_themselves(rscene, world):
    keys_and_notebook(rscene, world)
    carry(rscene, world, 'notebook', (80, 40), KEYS_AT, (15, 21), over='keys')
    rscene.hand(1, *KEYS_AT)
    events = rscene.run(world, 0.4)
    rscene.remove('notebook')
    rscene.thing('keys', *KEYS_AT, 6, 4)
    for x in (50, 60, 70):
        rscene.hand(1, x, 40)
        events += world.update(*rscene.step())
    rscene.thing('notebook', 80, 40, 15, 21)
    events += rscene.run(world, 0.3)
    rscene.hand_off(1)
    events += rscene.run(world, 3.0)
    assert EventType.UNCOVERED in of(events, 'thing:1')
    assert EventType.APPEARED not in [e.type for e in events]
    assert world.get('thing:1').status == Status.VISIBLE


def test_shell_game_the_keys_go_with_the_cup_slid_over_them(rscene, world):
    """Three cups; one is put over the keys, then slid to a new spot and the others shuffled slowly.
    'Where are my keys?' names the cup they are under, where it is now."""
    rscene.thing('keys', *KEYS_AT, 6, 4)
    for k, at in (('cup_a', (60, 20)), ('cup_b', (80, 30)), ('cup_c', (60, 50))):
        rscene.thing(k, *at, 9, 9)
    settle(rscene, world)
    guesses(world, thing_1='keys', thing_2='cup', thing_3='cup', thing_4='cup')
    carry(rscene, world, 'cup_b', (80, 30), KEYS_AT, (9, 9), over='keys')
    assert (world.get('thing:1').status, world.get('thing:1').parent) == (Status.UNDER, 'thing:3')
    slide(rscene, world, 'cup_b', KEYS_AT, (40, 55), (9, 9))           # the keys go along under it
    slide(rscene, world, 'cup_a', (60, 20), (40, 30), (9, 9))          # another cup to the keys' old spot
    keys = world.get('thing:1')
    assert (keys.status, keys.parent) == (Status.UNDER, 'thing:3')
    assert world.resolve('thing:1')[0] == pytest.approx((40, 55), abs=1.5)
    assert 'under the cup' in where(world)
    rscene.hand(1, 40, 55)                                            # lift that cup: the keys are there
    rscene.run(world, 0.4)
    rscene.remove('cup_b')
    rscene.thing('keys', 40, 55, 6, 4)
    rscene.hand(1, 90, 60)
    events = rscene.run(world, 0.3)
    rscene.hand_off(1)
    events += rscene.run(world, 3.0)
    assert EventType.UNCOVERED in of(events, 'thing:1')
    assert world.get('thing:1').pos_cm == pytest.approx((40, 55), abs=1.5)
    assert EventType.APPEARED not in [e.type for e in events]


def test_keys_picked_up_right_after_a_cup_is_put_down_beside_them_are_picked_up(rscene, world):
    """The same hand touched a cup and the keys, but the cup does not lie over them: a pick-up."""
    rscene.thing('keys', *KEYS_AT, 6, 4)
    settle(rscene, world)
    carry(rscene, world, 'cup', (90, 50), (52, 30), (9, 9))           # a new cup put down beside them
    rscene.hand(1, *KEYS_AT)
    events = rscene.run(world, 0.4)
    rscene.remove('keys')
    for x in (30, 20, 10):
        rscene.hand(1, x, 30)
        events += world.update(*rscene.step())
    events += rscene.run(world, 0.5)
    assert of(events, 'thing:1') == [EventType.PICKED_UP]


def test_a_cup_carried_out_of_view_with_the_keys_gone_loses_them(rscene, world, cfg):
    rscene.thing('keys', *KEYS_AT, 6, 4)
    rscene.thing('cup', 80, 30, 9, 9)
    settle(rscene, world)
    carry(rscene, world, 'cup', (80, 30), KEYS_AT, (9, 9), over='keys')
    assert world.get('thing:1').parent == 'thing:2'
    rscene.remove('cup')                            # gone, keys and all: no cover on the table to follow
    events = rscene.run(world, cfg.lost_grace_s + cfg.reappear_wait_s + 1.5)
    assert of(events, 'thing:1') == [EventType.LOST_TRACK]
