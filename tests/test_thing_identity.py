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


# ----- one per kind: Grok names a new thing what a lost thing answers to --------------------------

def grok_names(world, ent, label, others=(), conf=0.95, merge_same=True):
    """One settle check whose reply names the mark on `ent` `label` (plus `others`: (entity, label))."""
    import json

    import numpy as np

    import tempfile

    from core.config import load_config
    from core.events import EventLog
    from core.grok_check import GrokCheck
    from core.narration import FakeProvider
    from server.sim import SimTable
    raw = load_config()
    marks = [(ent, (100, 100, 160, 160))] + [(o, (300 + 80 * i, 100, 360 + 80 * i, 160))
                                             for i, (o, _) in enumerate(others)]
    rep = json.dumps({"marks": [dict(mark=i, real=True, label=lab, confidence=conf)
                                for i, lab in enumerate([label] + [lab for _, lab in others], 1)],
                      "unmarked": []})
    g = GrokCheck({**raw, "grok_check": dict(enabled=True, merge_same=merge_same)},
                  EventLog(":memory:", tempfile.mkdtemp()),
                  table=SimTable(raw), provider=FakeProvider(rep), online=lambda: True, start=False)
    g.world = world
    return g.check(np.full((720, 1280, 3), 170, np.uint8), 2000.0, marks=marks,
                   names={n: None for n, _ in marks})


def lost_mug_then_new(scene, world, second_at=(70, 30)):
    """thing:1 'mug' (taught) is carried off unseen; later a mug-sized thing:2 is put down elsewhere."""
    picked_up_and_lost(scene, world)
    world.bind_alias('thing:1', 'mug')
    scene.run(world, 3.0)
    scene.thing('mug', *second_at)
    scene.run(world, 2.0)
    assert things(world) == ['thing:1', 'thing:2']


def test_a_new_thing_grok_names_like_a_lost_one_is_that_one():
    scene, world = make(rebirth_s=0.0)
    lost_mug_then_new(scene, world)
    s = grok_names(world, 'thing:2', 'mug')
    assert things(world) == ['thing:1'] and world.get('thing:2').merged_into == 'thing:1'
    assert world.get('thing:1').status == Status.VISIBLE
    assert world.get('thing:1').pos_cm == pytest.approx((70, 30), abs=0.5)
    assert world.find('mug') == 'thing:1' and s['bound'] == ['thing:2 = mug (thing:1)']


def test_two_of_a_kind_in_one_frame_stay_two():
    scene, world = make(rebirth_s=0.0)
    lost_mug_then_new(scene, world)
    scene.thing('mug2', 20, 20)
    scene.run(world, 2.0)
    grok_names(world, 'thing:2', 'mug', others=[('thing:3', 'blue mug')])
    assert things(world) == ['thing:1', 'thing:2', 'thing:3']


def test_seen_together_once_they_stay_two():
    """The mug was on the table while the second one was put down, then carried off: two mugs."""
    scene, world = make(rebirth_s=0.0)
    scene.thing('mug', 40, 30)
    scene.thing('cup', 70, 30)
    scene.run(world, 2.0)
    world.bind_alias('thing:1', 'mug')
    scene.remove('mug')
    scene.run(world, 4.0)
    assert world.get('thing:1').status == Status.UNKNOWN
    grok_names(world, 'thing:2', 'mug')
    assert things(world) == ['thing:1', 'thing:2']


def test_off_or_unsure_nothing_merges():
    scene, world = make(rebirth_s=0.0)
    lost_mug_then_new(scene, world)
    grok_names(world, 'thing:2', 'mug', merge_same=False)
    grok_names(world, 'thing:2', 'mug', conf=0.65)
    assert things(world) == ['thing:1', 'thing:2']


def test_a_lost_thing_still_on_the_marks_is_not_merged():
    scene, world = make(rebirth_s=0.0)
    lost_mug_then_new(scene, world)
    grok_names(world, 'thing:2', 'mug', others=[('thing:1', 'tape')])
    assert things(world) == ['thing:1', 'thing:2']
