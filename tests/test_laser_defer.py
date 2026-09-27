"""The laser dot beside a prop is a local pixel change: on the film build, props read 'covered by
something' as a dot went off next to them. While the dot is lit (world.laser['on'], set by main.py)
and LASER_SETTLE_S after, such a verdict waits; then the rules decide as before. Rendered scenes."""
from pathlib import Path

import pytest

from core.config import Config
from core.relations import BackgroundModel
from core.types import Status
from core.world import LASER_SETTLE_S, World
from tests.synth import Scene

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def cfg():
    return Config.load(ROOT / 'config.yaml')


def setup(cfg):
    s = Scene(cfg, fps=10, t0=1000.0, render=True)
    w = World(cfg)
    s.run(w, (BackgroundModel.MIN_READY + 0.5) * cfg.bg_update_every_s)
    s.thing('pills', 40, 30, 4, 5)
    s.run(w, 2.5)
    assert w.get('thing:1').status == Status.VISIBLE
    return s, w


def test_no_under_something_while_the_laser_is_lit(cfg):
    s, w = setup(cfg)
    w.laser = {'on': True, 'target': 'thing:2', 'err_cm': 3.0}
    s.remove('pills')
    s.overlay('dot', 40, 30, 30, 20)               # the pixels there change
    assert s.run(w, 4.0) == []
    assert w.get('thing:1').status == Status.VISIBLE


def test_once_the_dot_is_off_the_rules_decide_as_before(cfg):
    s, w = setup(cfg)
    w.laser = {'on': True, 'target': 'thing:2', 'err_cm': 3.0}
    s.remove('pills')
    s.overlay('dot', 40, 30, 30, 20)
    s.run(w, 2.0)
    w.laser = {'on': False, 'target': None, 'err_cm': None}
    s.run(w, LASER_SETTLE_S + 3.0)
    assert (w.get('thing:1').status, w.get('thing:1').parent) == (Status.UNDER, 'unknown')


def test_without_the_laser_nothing_changes(cfg):
    s, w = setup(cfg)
    s.remove('pills')
    s.overlay('dot', 40, 30, 30, 20)
    s.run(w, 4.0)
    assert (w.get('thing:1').status, w.get('thing:1').parent) == (Status.UNDER, 'unknown')
