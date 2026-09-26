"""Teaching names to open-world things (world.teach / bind_alias / find) and answering about them."""
import re
from pathlib import Path

import pytest

from core.config import Config, load_config
from core.events import EventLog
from core.types import EventType, Intent, Status
from core.world import World
from tests.synth import Scene, look, look_like
from voice.answers import answer
from voice.intents import parse
from voice.pipeline import make_ask

ROOT = Path(__file__).resolve().parents[1]
CFG = load_config()
ZONE = (0.0, 40.0, 20.0, 60.0)      # a teach square in the bottom-left corner
BOX_AT = (80, 40)


@pytest.fixture
def cfg():
    c = Config.load(ROOT / 'config.yaml')
    c.teach_zone_cm = ZONE
    return c


@pytest.fixture
def scene(cfg):
    return Scene(cfg, fps=10, t0=1000.0)


@pytest.fixture
def world(cfg, scene, tmp_path):
    return World(cfg, EventLog(':memory:', str(tmp_path / 'snaps')), embed=scene.embed)


def put(scene, world, key, at, hid=2, **kw):
    """A hand sets a thing down at `at` and leaves."""
    scene.hand(hid, *at)
    scene.thing(key, *at, **kw)
    events = scene.run(world, 0.5)
    scene.hand_off(hid)
    return events + scene.run(world, 1.0)


def ask(world, text, **kw):
    return answer(parse(text, CFG), world, world.events, CFG, now=world._wall, **kw)


def spoken_ok(a):
    assert a.text and a.text[0].isupper() and a.text.endswith((".", "?")), a.text
    assert not re.search(r"[*#`_\[\]]|thing:", a.text), a.text


# ----- world: teach / find / bind_alias -----------------------------------------------------------

def test_teach_binds_the_thing_most_recently_put_down_in_the_teach_zone(scene, world):
    put(scene, world, 'lamp', (60, 30))            # thing:1, outside the square
    put(scene, world, 'mug', (10, 50))             # thing:2, inside
    put(scene, world, 'charger', (15, 55))         # thing:3, inside, most recent
    assert world.teach('my charger') == 'thing:3'
    for said in ('charger', 'The Charger', 'my charger', 'chargers', 'thing:3'):
        assert world.find(said) == 'thing:3', said
    assert world.get('thing:3').aliases == ['charger']


def test_moving_an_older_thing_within_the_zone_makes_it_the_teach_target(scene, world):
    put(scene, world, 'mug', (10, 50))
    put(scene, world, 'charger', (15, 55))
    scene.hand(1, 10, 50)                          # pick the mug up and set it down again
    scene.run(world, 0.3)
    scene.remove('mug')
    scene.run(world, 1.0)
    events = put(scene, world, 'mug', (6, 45), hid=1)
    assert [e.type for e in events if e.obj == 'thing:1'] == [EventType.MOVED]
    assert world.teach('mug') == 'thing:1'


def test_teach_refuses_when_nothing_is_in_the_zone(scene, world):
    put(scene, world, 'lamp', (60, 30))
    assert world.teach('lamp') is None
    assert world.find('lamp') is None


def test_without_a_zone_the_most_recent_put_down_counts_for_a_while(scene, world):
    world.cfg.teach_zone_cm = None
    put(scene, world, 'lamp', (60, 30))
    put(scene, world, 'mug', (90, 30))
    assert world.teach('mug') == 'thing:2'
    scene.run(world, 40.0)
    assert world.teach('lamp') is None


def test_configured_names_cannot_be_taught(scene, world):
    put(scene, world, 'charger', (10, 50))
    for said in ('keys', 'my cell phone', 'Pills', 'the pill bottle'):
        assert world.teach(said) is None, said
    assert not world.bind_alias('thing:1', 'phone')
    assert world.get('thing:1').aliases == []
    assert world.find('my car keys') == 'keys' and world.find('medication') == 'pill_bottle'
    assert world.find('toaster') is None


def test_a_name_has_one_owner_and_moves_when_bound_again(scene, world):
    put(scene, world, 'a', (10, 50))
    put(scene, world, 'b', (60, 30))
    assert world.teach('charger') == 'thing:1'
    assert world.bind_alias('thing:2', 'The charger')
    assert world.find('charger') == 'thing:2'
    assert world.get('thing:1').aliases == []
    assert world.alias_phrases() == ['charger']


def settle(scene, world):
    """The empty table for a while: what is in view at start-up was not put down."""
    scene.run(world, 3.5)


def put_known(scene, world, name, at, hid=2):
    """A hand sets a configured object (the detector knows its class) down at `at` and leaves."""
    scene.hand(hid, *at)
    scene.place(name, *at)
    scene.run(world, 0.5)
    scene.hand_off(hid)
    scene.run(world, 1.0)


def test_a_new_name_goes_to_a_configured_object_put_down_last(scene, world):
    """'this is my brown wallet' names the wallet the detector sees, not an older unnamed thing."""
    settle(scene, world)
    put(scene, world, 'lamp', (10, 50))            # thing:1, inside the square, earlier
    put_known(scene, world, 'wallet', (12, 48))
    assert world.teach('my brown wallet') == 'wallet'
    assert world.find('brown wallet') == 'wallet'
    assert world.get('thing:1').aliases == []


def test_a_configured_object_in_view_from_the_start_was_not_put_down(scene, world):
    """The scene at start-up is not a put-down: the box confirmed a frame after the thing does not
    take the name."""
    world.cfg.teach_zone_cm = None
    scene.place('box', 80, 40)
    scene.thing('lamp', 10, 50)
    scene.run(world, 2.0)
    assert world.teach('charger') == 'thing:1'


def test_an_object_found_again_where_it_was_lost_was_not_put_down(scene, world):
    """The phone flickers out (LOST_TRACK) and back at the same spot (CORRECTED) just after the wallet
    is put down: the wallet takes the name."""
    world.cfg.teach_zone_cm = None
    scene.place('phone', 60, 20)
    settle(scene, world)
    put_known(scene, world, 'wallet', (12, 48))
    scene.miss('phone')
    events = scene.run(world, 4.0)
    assert any(e.obj == 'phone' and e.type == EventType.LOST_TRACK for e in events)
    scene.miss('phone', False)
    events = scene.run(world, 1.0)
    assert any(e.obj == 'phone' and e.type == EventType.CORRECTED for e in events)
    assert world.teach('brown wallet') == 'wallet'


def test_a_thing_put_down_after_the_configured_object_still_wins(scene, world):
    settle(scene, world)
    put_known(scene, world, 'wallet', (12, 48))
    put(scene, world, 'lamp', (10, 50))
    assert world.teach('lamp') == 'thing:1'


# ----- answers ---------------------------------------------------------------------------------------

def test_teach_answer_confirms_and_points_at_the_thing(scene, world):
    put(scene, world, 'charger', (10, 50))
    a = ask(world, "This is my charger.")
    spoken_ok(a)
    assert a.text == "Got it, I'll remember this as your charger."
    assert (a.point_at, a.action) == ('thing:1', 'point')


def test_teach_answer_asks_for_the_teach_square_when_nothing_qualifies(world):
    a = ask(world, "this is my charger")
    spoken_ok(a)
    assert a.text == "Put it in the teach square first, then say 'this is my charger'."
    assert a.point_at is None


def test_teach_answer_refuses_a_configured_name_politely(scene, world):
    put(scene, world, 'charger', (10, 50))
    a = ask(world, "this is my phone")
    spoken_ok(a)
    assert a.text == "I already know your phone. Give this one a different name."
    assert world.get('thing:1').aliases == []


def test_naming_the_configured_object_just_put_down_confirms_it(scene, world):
    settle(scene, world)
    put_known(scene, world, 'wallet', (12, 48))
    a = ask(world, "this is my wallet")
    spoken_ok(a)
    assert a.text == "Yes, that's your wallet."
    assert (a.point_at, a.action) == ('wallet', 'point')


def test_where_is_a_name_taught_to_a_configured_object(scene, world):
    settle(scene, world)
    put_known(scene, world, 'wallet', (12, 48))
    ask(world, "this is my brown wallet")
    a = ask(world, "where is my brown wallet?")
    spoken_ok(a)
    assert a.point_at == 'wallet' and "brown wallet" in a.text, a.text


def test_question_about_an_unknown_name(world):
    for q in ("Where is my charger?", "What happened to my charger?", "Did anyone touch my charger?"):
        a = ask(world, q)
        spoken_ok(a)
        assert a.text == ("I don't know what your charger is yet. "
                          "Put it in the teach square and say 'this is my charger'."), q
        assert a.point_at is None


def test_where_is_a_taught_thing_uses_its_name(scene, world):
    put(scene, world, 'charger', (10, 50))
    world.teach('charger')
    a = ask(world, "where's my charger?")
    spoken_ok(a)
    assert a.text.startswith("Your charger is on the table")
    assert (a.point_at, a.action) == ('thing:1', 'point')


def test_taught_multiword_name_wins_over_a_configured_word_inside_it(scene, world):
    put(scene, world, 'charger', (10, 50))
    world.teach('phone charger')
    scene.place('phone', 60, 20)
    scene.run(world, 1.0)
    a = answer(Intent("WHERE", "phone", "where is my phone charger"), world, world.events, CFG)
    assert a.point_at == 'thing:1' and "phone charger" in a.text


def test_history_of_a_taught_thing(scene, world):
    put(scene, world, 'charger', (10, 50))
    world.teach('charger')
    a = ask(world, "what happened to my charger")
    spoken_ok(a)
    assert a.text.startswith("Your charger was first seen")


def test_changes_mentions_unnamed_things_without_internal_names(scene, world):
    put(scene, world, 'lamp', (60, 30))
    a = ask(world, "what changed while I was gone?", since=scene.t - 60)
    spoken_ok(a)
    assert "haven't been told about" in a.text


def exit_left(scene, world, at=(10, 50)):
    scene.hand(1, *at)
    scene.run(world, 0.3)
    scene.remove('charger')
    scene.run(world, 1.0)
    scene.hand(1, 5, 30)
    scene.run(world, 0.3)
    scene.hand_off(1)
    scene.run(world, 1.0)


def test_something_similar_coming_back_is_hedged_and_pointed_at(scene, world):
    put(scene, world, 'charger', (10, 50))
    scene.run(world, 1.5)                          # time to learn its look
    world.teach('charger')
    exit_left(scene, world)
    assert world.get('thing:1').status == Status.GONE
    put(scene, world, 'lookalike', (60, 30), look=look_like('lookalike', 'charger', 0.85))
    assert [n for n, _ in world.get('thing:2').maybe_same_as] == ['thing:1']
    a = ask(world, "where is my charger")
    spoken_ok(a)
    assert a.text.startswith("Your charger was carried off the left side of the table")
    assert "Something similar came back" in a.text and "it might be yours" in a.text
    assert (a.point_at, a.action) == ('thing:2', 'point')


def test_fake_world_without_open_world_support_still_answers():
    from core.fakeworld import demo_world
    w = demo_world()
    a = answer(parse("where is my charger", CFG), w, w.events, CFG)
    spoken_ok(a)
    assert "don't know what your charger is" in a.text
    assert answer(parse("this is my charger", CFG), w, w.events, CFG).text == \
        "I can't learn new names right now."


# ----- end to end: teach -> hide in the box -> move the box -> ask ------------------------------------

def test_teach_hide_in_box_move_box_then_where_points_at_the_box(scene, world):
    ask_fn = make_ask(CFG, world, world.events, other=lambda *a, **k: None)
    scene.place('box', *BOX_AT)
    put(scene, world, 'charger', (10, 50))
    ans = ask_fn("This is my charger", "voice")
    assert ans.text == "Got it, I'll remember this as your charger." and ans.point_at == 'thing:1'
    scene.hand(1, 10, 50)                          # pick it up, carry it into the box
    scene.run(world, 0.3)
    scene.remove('charger')
    scene.run(world, 0.3)
    for x, y in [(30, 46), (50, 43), (70, 41)]:
        scene.hand(1, x, y)
        world.update(*scene.step())
    scene.hand(1, *BOX_AT)
    scene.run(world, 0.6)
    scene.hand(1, 100, 60)
    scene.run(world, world.cfg.reappear_wait_s + 0.5)
    assert (world.get('thing:1').status, world.get('thing:1').parent) == (Status.INSIDE, 'box')
    scene.hand(1, *BOX_AT)                         # carry the box 30 cm to the right
    scene.run(world, 0.3)
    scene.remove('box')
    scene.run(world, 0.5)
    scene.hand(1, 110, 40)
    scene.place('box', 110, 40)
    scene.run(world, 0.5)
    scene.hand_off(1)
    scene.run(world, 1.0)
    ans = ask_fn("Where is my charger?", "voice")
    spoken_ok(ans)
    assert ans.text.startswith("Your charger is inside the box")
    assert (ans.point_at, ans.action) == ('thing:1', 'point')
    assert world.resolve(ans.point_at) == (pytest.approx((110, 40)), ['thing:1', 'box'])
