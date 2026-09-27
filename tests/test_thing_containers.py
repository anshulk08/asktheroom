"""Open-world containers: a large unnamed thing (a tub, a bag, a hat) holds what a hand leaves in it, by
the same container rule as the configured box (core/world.py rules 7-8, core/things.py). The evidence
is specific: a hand holding the object dwells inside the thing's box, leaves, and the object is not
seen again. A hand that crosses a large flat thing (in one side, out the other) was carrying past it."""
from pathlib import Path

import pytest

from core.config import Config, load_config
from core.events import EventLog
from core.types import EventType, Status
from core.world import World
from tests.synth import Scene
from voice.answers import answer
from voice.intents import parse

ROOT = Path(__file__).resolve().parents[1]
CFG = load_config()
TUB_AT, TUB = (80, 40), (22.0, 13.0)      # the Lego tub of shell_1: x 69-91, y 33.5-46.5 (286 cm2)
POD_AT, POD = (40, 30), (5.0, 6.0)        # an AirPods case (30 cm2)
OUT = (100, 60)


@pytest.fixture
def cfg():
    return Config.load(ROOT / 'config.yaml')


@pytest.fixture
def scene(cfg):
    return Scene(cfg, fps=10, t0=1000.0)


@pytest.fixture
def world(cfg, scene, tmp_path):
    return World(cfg, EventLog(':memory:', str(tmp_path / 'snaps')), embed=scene.embed)


def of(events, obj):
    return [e.type for e in events if e.obj == obj]


def tub_and_pods(scene, world):
    """The tub appears first (thing:1), then the AirPods case (thing:2)."""
    scene.thing('tub', *TUB_AT, *TUB)
    scene.run(world, 1.5)
    scene.thing('pods', *POD_AT, *POD)
    scene.run(world, 1.5)
    assert [world.get(n).status for n in ('thing:1', 'thing:2')] == [Status.VISIBLE] * 2


def carry_into(scene, world, key, at, into, hid=1, dwell=0.6, via=None):
    """A hand picks key up at `at`, carries it to `into`, dwells and withdraws to OUT (in view), resting
    at `via` for half a second on the way if given."""
    scene.hand(hid, *at)
    events = scene.run(world, 0.3)
    scene.remove(key)
    events += scene.run(world, 0.2)
    mid = ((at[0] + into[0]) / 2, (at[1] + into[1]) / 2)
    scene.hand(hid, *mid)
    events += scene.run(world, 0.3)
    scene.hand(hid, *into)
    events += scene.run(world, dwell)
    if via is not None:
        scene.hand(hid, *via)
        events += scene.run(world, 0.5)
    scene.hand(hid, *OUT)
    return events + scene.run(world, 0.1)


def pods_in_tub(scene, world, cfg, dwell=0.6):
    tub_and_pods(scene, world)
    events = carry_into(scene, world, 'pods', POD_AT, TUB_AT, dwell=dwell)
    return events + scene.run(world, cfg.reappear_wait_s + 0.2)


# ----- put inside, carried, taken out ------------------------------------------------------------

def test_an_object_left_in_a_large_thing_is_put_inside_it(scene, world, cfg):
    events = pods_in_tub(scene, world, cfg)
    assert of(events, 'thing:2') == [EventType.PICKED_UP, EventType.PUT_INSIDE]
    put = [e for e in events if e.type == EventType.PUT_INSIDE][0]
    assert put.parent == 'thing:1' and put.confidence == pytest.approx(cfg.conf_inside)
    pods = world.get('thing:2')
    assert (pods.status, pods.parent) == (Status.INSIDE, 'thing:1')
    assert of(events, 'thing:1') == []             # the tub itself never moved
    assert world.get('thing:1').status == Status.VISIBLE
    assert world.resolve('thing:2') == (pytest.approx(TUB_AT), ['thing:2', 'thing:1'])


def test_a_configured_object_can_be_put_inside_a_thing_too(scene, world, cfg):
    scene.thing('tub', *TUB_AT, *TUB)
    scene.place('keys', 40, 30)
    scene.run(world, 1.5)
    events = carry_into(scene, world, 'keys', (40, 30), TUB_AT) + scene.run(world, cfg.reappear_wait_s + 0.2)
    assert of(events, 'keys') == [EventType.PICKED_UP, EventType.PUT_INSIDE]
    assert (world.get('keys').status, world.get('keys').parent) == (Status.INSIDE, 'thing:1')


def test_a_long_dwell_in_the_tub_is_not_a_pick_up_of_the_tub(scene, world, cfg):
    """A hand resting inside the tub's box for over a second: the tub's proposal (which contains the
    whole hand box) is still the tub, not an arm, so the tub stays in view and holds the object."""
    events = pods_in_tub(scene, world, cfg, dwell=1.5)
    assert of(events, 'thing:1') == []
    assert (world.get('thing:2').status, world.get('thing:2').parent) == (Status.INSIDE, 'thing:1')


def test_a_thing_container_lifted_and_carried_30_cm_takes_its_contents_along(scene, world, cfg):
    pods_in_tub(scene, world, cfg)
    scene.hand(1, *TUB_AT)                         # grab the tub
    events = scene.run(world, 0.4)
    scene.remove('tub')
    for x in range(83, 111, 3):                    # carry it 30 cm to the right
        scene.hand(1, x, 40)
        events += world.update(*scene.step())
    scene.run(world, 0.2)
    scene.thing('tub', 110, 40, *TUB)              # set it down, let go, leave
    events += scene.run(world, 0.5)
    scene.hand_off(1)
    events += scene.run(world, 2.0)
    assert of(events, 'thing:1') == [EventType.PICKED_UP, EventType.MOVED]
    assert of(events, 'thing:2') == []
    assert (world.get('thing:2').status, world.get('thing:2').parent) == (Status.INSIDE, 'thing:1')
    assert world.resolve('thing:2') == (pytest.approx((110, 40)), ['thing:2', 'thing:1'])


def test_a_thing_container_slid_across_the_table_takes_its_contents_along(scene, world, cfg):
    pods_in_tub(scene, world, cfg)
    events = []
    for i in range(0, 11):                         # a hand on its left rim slides it 30 cm right
        x = TUB_AT[0] + 3 * i
        scene.thing('tub', x, 40, *TUB)
        scene.hand(1, x - 8, 40)
        events += world.update(*scene.step())
    scene.hand_off(1)
    events += scene.run(world, 2.0)
    assert of(events, 'thing:1') == [EventType.PICKED_UP, EventType.MOVED]
    assert world.get('thing:2').status == Status.INSIDE
    assert world.resolve('thing:2') == (pytest.approx((110, 40)), ['thing:2', 'thing:1'])


def test_an_object_taken_back_out_of_a_thing_container_is_taken_out(scene, world, cfg):
    pods_in_tub(scene, world, cfg)
    scene.hand(1, *TUB_AT)                         # reach in ...
    events = scene.run(world, 0.6)
    scene.hand(1, 100, 40)                         # ... and set it down beside the tub
    scene.thing('pods', 100, 40, *POD)
    events += scene.run(world, 0.5)
    scene.hand_off(1)
    events += scene.run(world, 2.0)
    assert of(events, 'thing:2') == [EventType.TAKEN_OUT]
    assert [n for n in world.entities if n.startswith('thing:')] == ['thing:1', 'thing:2']
    pods = world.get('thing:2')
    assert (pods.status, pods.parent) == (Status.VISIBLE, None)
    assert pods.pos_cm == pytest.approx((100, 40))


def test_a_big_thing_put_down_beside_a_thing_container_is_not_its_contents_coming_out(scene, world, cfg):
    pods_in_tub(scene, world, cfg)
    scene.hand(1, *TUB_AT)                         # a hand in the tub ...
    events = scene.run(world, 0.6)
    scene.hand(1, 90, 55)                          # ... then a book set down right beside it
    scene.thing('book', 90, 55, 20, 15)
    events += scene.run(world, 0.5)
    scene.hand_off(1)
    events += scene.run(world, 2.0)
    assert of(events, 'thing:2') == []
    assert (world.get('thing:2').status, world.get('thing:2').parent) == (Status.INSIDE, 'thing:1')
    assert of(events, 'thing:3') == [EventType.APPEARED]


# ----- the gate: which things can hold --------------------------------------------------------------

def test_a_small_thing_is_no_container(scene, world, cfg):
    scene.thing('coin', *TUB_AT, 6, 4)             # 24 cm2: the hand rests right on it
    scene.place('keys', 40, 30)
    scene.run(world, 1.5)
    events = carry_into(scene, world, 'keys', (40, 30), TUB_AT) + scene.run(world, cfg.reappear_wait_s + 0.5)
    assert EventType.PUT_INSIDE not in of(events, 'keys')
    assert (world.get('keys').status, world.get('keys').parent) == (Status.HELD, 'hand:1')


def test_the_configured_box_wins_over_a_thing_it_stands_on(scene, world, cfg):
    """A tray (a large thing) under the box: the hand inside the box is inside both boxes, and leaves
    the tray last (it rests on the tray's rim, beside the box, on the way out)."""
    scene.thing('tray', 80, 40, 35, 24)            # x 62.5-97.5, y 28-52; the box x 70-90, y 32.5-47.5
    scene.place('box', 80, 40)
    scene.place('keys', 40, 30)
    scene.run(world, 1.5)
    assert world.get('thing:1').status == Status.VISIBLE
    events = carry_into(scene, world, 'keys', (40, 30), (80, 40), via=(94, 50))
    events += scene.run(world, cfg.reappear_wait_s + 0.2)
    assert of(events, 'keys') == [EventType.PICKED_UP, EventType.PUT_INSIDE]
    assert (world.get('keys').status, world.get('keys').parent) == (Status.INSIDE, 'box')


def test_nesting_in_a_thing_container_stops_at_max_nesting(cfg, tmp_path):
    """The box, with the keys inside, is carried into a crate (a large thing): keys -> box -> crate."""
    def run(max_nesting):
        c = Config.load(ROOT / 'config.yaml')
        c.max_nesting = max_nesting
        s = Scene(c, fps=10, t0=1000.0)
        w = World(c, embed=s.embed)
        s.thing('crate', 60, 50, 36, 24)
        s.place('keys', 20, 10)
        s.place('box', 100, 20)
        s.run(w, 1.5)
        carry_into(s, w, 'keys', (20, 10), (100, 20))
        s.hand_off(1)
        s.run(w, c.reappear_wait_s + 0.2)
        assert w.get('keys').parent == 'box'
        events = carry_into(s, w, 'box', (100, 20), (60, 50), hid=2) + s.run(w, c.reappear_wait_s + 0.2)
        return w, events

    w, events = run(3)
    assert of(events, 'box') == [EventType.PICKED_UP, EventType.PUT_INSIDE]
    assert w.resolve('keys') == (pytest.approx((60, 50)), ['keys', 'box', 'thing:1'])
    w, events = run(1)                             # keys -> box -> crate would be two levels
    assert of(events, 'box') == [EventType.PICKED_UP]
    assert w.get('box').status == Status.HELD


# ----- a large flat thing (a placemat, a sheet of paper) does not swallow things --------------------

MAT_AT, MAT = (60, 40), (36.0, 24.0)       # x 42-78, y 28-52 (864 cm2, under openworld's 900 cap)


def test_a_hand_passing_over_a_large_flat_thing_leaves_nothing_inside(scene, world, cfg):
    scene.thing('mat', *MAT_AT, *MAT)
    scene.place('keys', 100, 15)
    scene.run(world, 1.5)
    events = []
    for x in range(20, 101, 4):                    # an empty hand sweeps across the mat
        scene.hand(1, x, 40)
        events += world.update(*scene.step())
    scene.hand(1, 60, 40)                          # and rests on it for a while
    events += scene.run(world, 1.0)
    scene.hand_off(1)
    events += scene.run(world, 3.0)
    assert events == []
    assert all(e.status == Status.VISIBLE for n, e in world.entities.items() if e.pos_cm is not None)


def test_keys_carried_across_a_mat_and_off_the_table_exit_view_not_inside(scene, world, cfg):
    scene.thing('mat', *MAT_AT, *MAT)
    scene.place('keys', 20, 40)
    scene.run(world, 1.5)
    scene.hand(1, 20, 40)
    events = scene.run(world, 0.3)
    scene.remove('keys')
    for x in range(20, 131, 3):                    # carried across the mat and off the right edge
        scene.hand(1, x, 40)
        events += world.update(*scene.step())
    scene.hand_off(1)
    events += scene.run(world, cfg.reappear_wait_s + 1.0)
    assert of(events, 'keys') == [EventType.PICKED_UP, EventType.EXITED_VIEW]
    assert (world.get('keys').status, world.get('keys').edge) == (Status.GONE, 'right')


def test_keys_put_down_next_to_a_mat_stay_on_the_table(scene, world, cfg):
    scene.thing('mat', *MAT_AT, *MAT)
    scene.place('keys', 20, 10)
    scene.run(world, 1.5)
    events = carry_into(scene, world, 'keys', (20, 10), (78, 40), dwell=0.2)
    scene.hand(1, 78, 40)                          # the hand, on the mat's edge, sets them down beside it
    scene.place('keys', 86, 40)
    events += scene.run(world, 0.6)
    scene.hand_off(1)
    events += scene.run(world, cfg.reappear_wait_s + 1.0)
    assert EventType.PUT_INSIDE not in of(events, 'keys')
    assert of(events, 'keys')[-1] == EventType.MOVED
    assert world.get('keys').status == Status.VISIBLE


# ----- answers and pointing ----------------------------------------------------------------------------

def ask(world, text):
    return answer(parse(text, CFG), world, world.events, CFG, now=world._wall)


def test_where_answer_names_the_thing_container_and_points_at_it(scene, world, cfg):
    pods_in_tub(scene, world, cfg)
    assert world.bind_alias('thing:2', 'airpods')
    a = ask(world, "where are my airpods?")
    assert a.text == "Your airpods are inside a container. They were put there just now."
    assert (a.point_at, a.action) == ('thing:2', 'point')
    assert world.resolve(a.point_at)[0] == pytest.approx(TUB_AT)   # the laser aims at the tub
    h = ask(world, "what happened to my airpods")
    assert "put inside a container" in h.text and "thing" not in h.text, h.text


def test_a_thing_container_is_spoken_by_its_name_or_guess(scene, world, cfg):
    pods_in_tub(scene, world, cfg)
    world.bind_alias('thing:2', 'airpods')
    state = world.state_json

    def with_guess():                              # as core/auto_name.AutoNamer.attach does
        st = state()
        for e in st['entities']:
            if e['name'] == 'thing:1':
                e['guess'] = {'name': 'plastic tub', 'also': [], 'confidence': 0.9}
        return st

    world.state_json = with_guess
    assert ask(world, "where are my airpods?").text.startswith("Your airpods are inside the plastic tub.")
    world.bind_alias('thing:1', 'toy bin')         # a taught name wins over the guess
    assert ask(world, "where are my airpods?").text.startswith("Your airpods are inside the toy bin.")


# ----- config ------------------------------------------------------------------------------------------

def test_config_section_matches_the_code_defaults(cfg):
    from core.things import ContainerConfig
    assert ContainerConfig.from_config(cfg) == ContainerConfig()
    assert ContainerConfig.from_config(Config.from_dict({})) == ContainerConfig()   # an older config.yaml


def test_with_thing_containers_disabled_only_configured_containers_hold(cfg, tmp_path):
    cfg.thing_containers = {'enabled': False}
    scene = Scene(cfg, fps=10, t0=1000.0)
    world = World(cfg, embed=scene.embed)
    events = pods_in_tub(scene, world, cfg)
    assert of(events, 'thing:2') == [EventType.PICKED_UP]
    assert world.get('thing:2').status == Status.HELD


# ----- seen lying in an open container ------------------------------------------------------------------

def test_keys_seen_lying_in_the_open_box_are_in_the_box(scene, world):
    """From overhead an open box shows what is in it: the keys stay VISIBLE, and the answer says where."""
    scene.place('box', 80, 40)                      # 20 x 15 cm
    scene.place('keys', 82, 41)
    scene.run(world, 1.5)
    assert world.open_container_of('keys') == 'box'
    a = ask(world, "where are my keys?")
    assert a.text == "Your keys are in the box." and a.point_at == 'keys', a.text


def test_an_object_lying_in_an_open_thing_container_is_in_it(scene, world):
    scene.thing('tub', *TUB_AT, *TUB)
    scene.run(world, 1.5)
    scene.place('keys', 78, 40)
    scene.run(world, 1.5)
    assert world.open_container_of('keys') == 'thing:1'
    assert ask(world, "where are my keys?").text == "Your keys are in a container."


def test_keys_beside_the_box_are_not_in_it(scene, world):
    scene.place('box', 80, 40)
    scene.place('keys', 95, 40)                     # half over the box's edge
    scene.run(world, 1.5)
    assert world.open_container_of('keys') is None
    assert "in the box" not in ask(world, "where are my keys?").text
