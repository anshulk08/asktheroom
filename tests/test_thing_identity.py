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


# ----- belief: an unnamed thing's guesses over several checks name it or retire it as clutter ------

def checker(world, replies, **gc):
    """A GrokCheck on `world` whose settle checks reply, in turn, with `replies` (one mark each)."""
    import json
    import tempfile

    from core.config import load_config
    from core.events import EventLog
    from core.grok_check import GrokCheck
    from core.narration import FakeProvider
    from server.sim import SimTable
    raw = load_config()
    rep = [json.dumps({"marks": [dict(mark=1, real=r.get("real", True), label=r.get("label"),
                                      confidence=r.get("conf", 0.5), guesses=r.get("guesses", []),
                                      not_object=r.get("no", 0.0), held=False)], "unmarked": []})
           for r in replies]
    g = GrokCheck({**raw, "grok_check": {"enabled": True, "belief_enabled": True, **gc}},
                  EventLog(":memory:", tempfile.mkdtemp()),
                  table=SimTable(raw), provider=FakeProvider(rep), online=lambda: True, start=False)
    g.world = world
    return g


def settle_checks(g, ent, n, wall=2000.0, t=None, gap=10.0):
    import numpy as np
    img = np.full((720, 1280, 3), 170, np.uint8)
    for i in range(n):
        s = g.check(img, wall + i * gap, t=t, marks=[(ent, (100, 100, 160, 160))], names={ent: None})
    return s


def one_thing(scene, world, at=(40, 30)):
    scene.thing('blob', *at)
    scene.run(world, 2.0)
    assert things(world) == ['thing:1']


def test_three_checks_that_agree_on_a_guess_name_the_thing():
    scene, world = make(rebirth_s=0.0)
    one_thing(scene, world)
    g = checker(world, [dict(guesses=[dict(label='phone charger', p=0.6), dict(label='cable', p=0.2)])])
    s = settle_checks(g, 'thing:1', 2)
    assert world.get('thing:1').aliases == [] and g.belief('thing:1')[0][0] == 'phone charger'
    s = settle_checks(g, 'thing:1', 1)
    assert world.find('phone charger') == 'thing:1' and s['bound'] == ['thing:1 = phone charger']
    assert world.state_json()['entities'][-1]['named_by'] == 'grok'


def test_split_guesses_never_name_it():
    scene, world = make(rebirth_s=0.0)
    one_thing(scene, world)
    g = checker(world, [dict(guesses=[dict(label='phone', p=0.45), dict(label='remote', p=0.4)])])
    settle_checks(g, 'thing:1', 5)
    assert world.get('thing:1').aliases == []


def test_clutter_is_retired_and_nothing_is_born_there_for_a_while():
    scene, world = make(rebirth_s=0.0)
    one_thing(scene, world)
    g = checker(world, [dict(real=False, conf=0.9, no=0.9)])
    s = settle_checks(g, 'thing:1', 3, wall=scene.t, t=scene.t)
    assert s['retired'] == ['thing:1'] and things(world) == []
    assert 'thing:1' not in [e['name'] for e in world.state_json()['entities']]
    scene.run(world, 3.0)                              # still on the table: no new thing at that spot
    assert things(world) == []


def test_a_touched_thing_is_not_retired():
    scene, world = make(rebirth_s=0.0)
    one_thing(scene, world)
    t0 = scene.t
    scene.hand(1, 40, 30)
    scene.run(world, 0.3)
    scene.hand_off(1)
    scene.run(world, 2.0)
    g = checker(world, [dict(real=False, conf=0.9, no=0.9)])
    s = settle_checks(g, 'thing:1', 3, t=t0)             # judged a frame from before the touch
    assert s['retired'] == [] and things(world) == ['thing:1']


def test_named_things_have_no_belief_and_belief_is_off_by_default():
    scene, world = make(rebirth_s=0.0)
    one_thing(scene, world)
    g = checker(world, [dict(real=False, conf=0.9, no=0.9)], belief_enabled=False)
    settle_checks(g, 'thing:1', 3, t=scene.t)
    assert things(world) == ['thing:1'] and g.belief('thing:1') == []
    world.bind_alias('thing:1', 'tape')
    g = checker(world, [dict(real=False, conf=0.9, no=0.9)])
    settle_checks(g, 'thing:1', 3, t=scene.t)
    assert things(world) == ['thing:1']


def test_an_old_grok_checks_table_gains_the_new_columns():
    import tempfile

    from core.events import EventLog
    from core.grok_check import COLS, CheckStore
    ev = EventLog(":memory:", tempfile.mkdtemp())
    with ev._locked():
        ev._conn().execute("CREATE TABLE grok_checks (id INTEGER PRIMARY KEY, t REAL, wall REAL, episode TEXT, "
                           "mark INTEGER, entity TEXT, world_label TEXT, grok_label TEXT, verdict TEXT, x_cm REAL, "
                           "y_cm REAL, confidence REAL, latency_ms INTEGER, model TEXT)")
    store = CheckStore(ev)
    store.add([{"wall": 1.0, "verdict": "named", "guesses": '[["mug", 0.7]]', "not_object": 0.1, "held": 0}])
    assert set(store.rows()[0]) == set(COLS) and store.rows()[0]["guesses"] == '[["mug", 0.7]]'
