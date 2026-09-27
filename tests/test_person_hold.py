"""Someone in front of a prop (room demo): a torso, a head or an arm the hand detector does not box hides
the spot. With YOLOE's person boxes (Detections.people, from the proposer) the absence waits until they
move away, as for a hand passing over, instead of 'covered by something' (live Sun 27 Sep 03:00: 52 of
them in 20 min on 13 real props). Rendered scenes: the unknown-cover rule reads pixels."""
from pathlib import Path

import pytest

from core.config import Config
from core.relations import BackgroundModel
from core.types import EventType, Status
from core.world import World
from tests.synth import Scene, _box

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def cfg():
    return Config.load(ROOT / 'config.yaml')


def run(s, w, seconds, people=()):
    """Scene steps with YOLOE person boxes (x, y, w, h in table cm) attached, as core/detect.py does."""
    out = []
    for _ in range(int(round(seconds / s.dt))):
        dets, frame = s.step()
        dets.people = [type('P', (), {'box_cm': _box(*p)})() for p in people]
        out += w.update(dets, frame)
    return out


def setup(cfg):
    s = Scene(cfg, fps=10, t0=1000.0, render=True)
    w = World(cfg)
    run(s, w, (BackgroundModel.MIN_READY + 0.5) * cfg.bg_update_every_s)
    s.thing('pills', 40, 30, 4, 5)
    run(s, w, 2.5)
    assert w.get('thing:1').status == Status.VISIBLE
    return s, w


def lean_over(s):
    s.remove('pills')
    s.overlay('torso', 40, 30, 30, 20)             # the pixels there are not bare table


def test_a_person_leaning_over_a_prop_is_not_a_cover(cfg):
    s, w = setup(cfg)
    lean_over(s)
    events = run(s, w, 8.0, people=[(40, 30, 30, 20)])
    assert events == [] and w.get('thing:1').status == Status.VISIBLE
    s.overlays.pop('torso')
    s.thing('pills', 40, 30, 4, 5)                  # they move away: it is there
    assert run(s, w, 2.0) == [] and w.get('thing:1').status == Status.VISIBLE


def test_without_a_person_box_it_is_under_something_as_before(cfg):
    s, w = setup(cfg)
    lean_over(s)
    run(s, w, 4.0)
    assert (w.get('thing:1').status, w.get('thing:1').parent) == (Status.UNDER, 'unknown')


def test_someone_parked_in_front_for_long_lets_the_other_rules_decide(cfg):
    s, w = setup(cfg)
    lean_over(s)
    hold = cfg.unknown_cover.get('person_hold_s', 20)
    events = run(s, w, hold + 4.0, people=[(40, 30, 30, 20)])
    assert EventType.COVERED in [e.type for e in events]


def test_a_person_box_elsewhere_changes_nothing(cfg):
    s, w = setup(cfg)
    lean_over(s)
    run(s, w, 4.0, people=[(90, 50, 10, 10)])
    assert w.get('thing:1').status == Status.UNDER


def test_a_pick_up_behind_a_person_box_is_still_a_pick_up_once_they_move(cfg):
    """Deferred, not lost: the touch is remembered from when the prop was last seen."""
    s, w = setup(cfg)
    s.hand(1, 40, 30)
    events = run(s, w, 0.4, people=[(40, 30, 30, 20)])
    s.remove('pills')
    events += run(s, w, 3.0, people=[(40, 30, 30, 20)])
    s.hand_off(1)
    events += run(s, w, 3.0)
    assert EventType.PICKED_UP in [e.type for e in events if e.obj == 'thing:1']
    assert w.get('thing:1').status != Status.UNDER
