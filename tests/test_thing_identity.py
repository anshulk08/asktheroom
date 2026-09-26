"""thing_identity: (core/things.py IdentityConfig): a thing lost recently and seen again at its spot is
that thing again, not a new thing:N. No appearance embedder here, as on the rig (reid off)."""
from pathlib import Path

import pytest

from core.config import Config
from core.types import EventType, Status
from core.world import World
from tests.synth import Scene

ROOT = Path(__file__).resolve().parents[1]


def make(rebirth_s=60.0, **kw):
    cfg = Config.load(ROOT / 'config.yaml')
    cfg.thing_identity = {'rebirth_s': rebirth_s, **kw}
    scene = Scene(cfg, fps=10, t0=1000.0)
    return scene, World(cfg)


def things(world):
    return sorted(n for n, e in world.entities.items() if n.startswith('thing:') and e.merged_into is None)


def picked_up_and_lost(scene, world, at=(40, 30)):
    """thing:1 at `at`, lifted by a hand that then leaves the view: lost (UNKNOWN) after a pick-up."""
    scene.thing('mug', *at)
    scene.run(world, 1.0)
    scene.hand(1, *at)
    scene.run(world, 0.3)
    scene.remove('mug')
    scene.run(world, 1.0)
    assert world.get('thing:1').status == Status.HELD
    scene.hand(1, at[0] + 20, at[1])
    scene.run(world, 0.5)
    scene.hand_off(1)
    scene.run(world, 2.0)
    assert world.get('thing:1').status == Status.UNKNOWN


def test_off_by_default_a_thing_put_back_after_a_pick_up_is_a_new_thing():
    scene, world = make(rebirth_s=0.0)
    picked_up_and_lost(scene, world)
    scene.thing('mug', 40.5, 30)
    events = scene.run(world, 2.0)
    assert [(e.obj, e.type) for e in events] == [('thing:2', EventType.APPEARED)]


def test_a_thing_put_back_at_its_spot_after_a_pick_up_is_itself():
    scene, world = make()
    picked_up_and_lost(scene, world)
    scene.thing('mug', 40.5, 30)
    events = scene.run(world, 2.0)
    assert [e.obj for e in events] == ['thing:1'] and EventType.APPEARED not in [e.type for e in events]
    assert things(world) == ['thing:1'] and world.get('thing:1').status == Status.VISIBLE


def test_a_false_pick_up_while_a_hand_hovers_elsewhere_is_itself():
    """A hand passes over the thing (hidden: HELD), moves on and stays in view; the thing was never
    lifted and shows again at its spot once the hand trail has aged out."""
    scene, world = make()
    scene.thing('mug', 40, 30)
    scene.run(world, 1.0)
    scene.hand(1, 40, 30)
    scene.run(world, 0.3)
    scene.miss('mug')
    scene.run(world, 1.0)
    assert world.get('thing:1').status == Status.HELD
    scene.hand(1, 80, 30)
    scene.run(world, 3.0)
    scene.miss('mug', False)
    scene.run(world, 2.0)
    assert things(world) == ['thing:1'] and world.get('thing:1').status == Status.VISIBLE


def test_too_long_ago_is_a_new_thing():
    scene, world = make(rebirth_s=2.0)
    picked_up_and_lost(scene, world)
    scene.run(world, 3.0)
    scene.thing('mug', 40.5, 30)
    scene.run(world, 2.0)
    assert things(world) == ['thing:1', 'thing:2']


def test_elsewhere_or_another_size_is_a_new_thing():
    scene, world = make()
    picked_up_and_lost(scene, world)
    scene.thing('mug', 60, 30)                         # 20 cm away
    scene.thing('tray', 40, 30, w=30, h=20)           # at the spot, 25 times the area
    scene.run(world, 2.0)
    assert things(world) == ['thing:1', 'thing:2', 'thing:3']
    assert world.get('thing:1').status == Status.UNKNOWN


def test_the_nearest_of_two_lost_things_comes_back():
    scene, world = make()
    picked_up_and_lost(scene, world, at=(40, 30))
    scene.thing('cup', 47, 30)                         # 7 cm from the lost mug's spot: a new thing
    scene.run(world, 1.0)
    scene.hand(1, 47, 30)
    scene.run(world, 0.3)
    scene.remove('cup')
    scene.run(world, 1.0)
    scene.hand(1, 70, 30)
    scene.run(world, 0.5)
    scene.hand_off(1)
    scene.run(world, 2.0)
    assert world.get('thing:2').status == Status.UNKNOWN
    scene.thing('cup', 44, 30)                         # 3 cm from the cup's spot, 4 from the mug's
    scene.run(world, 2.0)
    assert world.get('thing:2').status == Status.VISIBLE
    assert world.get('thing:1').status == Status.UNKNOWN and things(world) == ['thing:1', 'thing:2']
    assert world.get('thing:2').pos_cm == pytest.approx((44, 30), abs=0.5)
