"""Open-world things (core/things.py + world.py): class-agnostic proposals become thing:N entities
that follow the same rules as the configured objects. Identity is causal first; appearance only
confirms, and an undecided identity stays a separate thing linked by maybe_same_as."""
import json
from pathlib import Path

import numpy as np
import pytest

from core.config import Config
from core.things import hsv_embed
from core.types import EventType, Status
from core.world import World
from tests.synth import Scene, look, look_like

ROOT = Path(__file__).resolve().parents[1]
BOX_AT = (80, 40)          # default 20 x 15 cm box: x 70-90, y 32.5-47.5


@pytest.fixture
def cfg():
    return Config.load(ROOT / 'config.yaml')


@pytest.fixture
def scene(cfg):
    return Scene(cfg, fps=10, t0=1000.0)


@pytest.fixture
def world(cfg, scene):
    return World(cfg, embed=scene.embed)


def types(events):
    return [e.type for e in events]


def of(events, obj):
    return [e.type for e in events if e.obj == obj]


def things(world):
    return sorted(n for n, e in world.entities.items() if n.startswith('thing:') and e.merged_into is None)


def appear(scene, world, key, at, seconds=1.0, **kw):
    scene.thing(key, *at, **kw)
    return scene.run(world, seconds)


# ----- lifecycle: candidate -> confirmed -------------------------------------------------------

def test_thing_becomes_an_entity_only_after_the_proposal_persists(scene, world):
    scene.thing('mug', 40, 30)
    assert scene.run(world, 0.4) == []            # 4 batches: below present_k
    assert things(world) == []
    events = scene.run(world, 0.6)
    assert types(events) == [EventType.APPEARED]
    assert events[0].obj == 'thing:1'
    assert events[0].to_cm == pytest.approx((40, 30))
    ent = world.get('thing:1')
    assert (ent.status, ent.kind, ent.confidence) == (Status.VISIBLE, 'target', 1.0)
    assert ent.pos_cm == pytest.approx((40, 30))
    assert scene.run(world, 3.0) == []             # the same proposal keeps its identity
    assert things(world) == ['thing:1']


def test_flickering_proposal_never_becomes_a_thing(scene, world):
    for i in range(40):                            # seen 3 batches in 10
        if i % 10 < 3:
            scene.thing('glint', 40, 30)
        else:
            scene.remove('glint')
        world.update(*scene.step())
    assert things(world) == []


def test_a_region_sweeping_across_the_table_never_becomes_a_thing(scene, world):
    """An arm the hand detector missed reaches the proposer as a region that keeps moving. Objects
    people put down stay put; measured on the rig, every pass of a hand left phantom things behind."""
    for i in range(40):                            # 4 s at 8 cm/s, no hand box anywhere
        scene.thing('arm', 10 + 0.8 * i, 30, w=10, h=6)
        world.update(*scene.step())
    assert things(world) == []


def test_an_object_slid_into_place_becomes_one_thing_once_it_stops(scene, world):
    for i in range(15):                            # slid 12 cm over 1.5 s ...
        scene.thing('mug', 20 + 0.8 * i, 30)
        world.update(*scene.step())
    assert things(world) == []
    events = scene.run(world, 1.5)                 # ... then left there
    assert types(events) == [EventType.APPEARED] and things(world) == ['thing:1']
    assert world.get('thing:1').pos_cm == pytest.approx((31.2, 30), abs=0.5)


def test_flicker_under_an_undetected_cover_does_not_pile_up_things(cfg):
    """On the rig, a knee at the table edge showed up as two overlapping regions that kept vanishing
    and coming back. Each vanishing read as 'covered by something undetected' (the pixels there are
    never bare table), and each return was ambiguous between the hidden ones, so a new thing was made
    every cycle: 7 things at one spot in 30 s. With nothing to tell them apart, the most recently
    hidden one of the right size comes back."""
    from core.relations import BackgroundModel
    rs = Scene(cfg, fps=10, t0=1000.0, render=True)
    w = World(cfg, embed=rs.embed)
    rs.run(w, (BackgroundModel.MIN_READY + 0.5) * cfg.bg_update_every_s)
    for _ in range(4):
        rs.overlays.pop('knee', None)
        rs.thing('k1', 12, 34, w=3, h=3)             # two regions 4 cm apart: one spot (same_spot_cm 5)
        rs.thing('k2', 16, 34, w=3, h=3)
        rs.run(w, 1.5)
        rs.remove('k1')
        rs.remove('k2')
        rs.overlay('knee', 14, 34, 12, 8)            # the pixels there are not bare table
        rs.run(w, 1.5)
    assert len(things(w)) == 2, [(n, w.get(n).status, w.get(n).parent) for n in things(w)]


def test_the_hand_itself_reported_as_a_proposal_is_ignored(scene, world):
    scene.hand(1, 40, 30)
    scene.thing('hand-blob', 40, 30, 12, 12)
    scene.thing('arm-blob', 60, 30, 30, 16)        # hand plus forearm: contains the hand box
    scene.hand(2, 55, 30)
    assert scene.run(world, 3.0) == []
    assert things(world) == []


def test_a_new_thing_waits_until_no_hand_overlaps_it(scene, world):
    scene.hand(1, 40, 30)
    scene.thing('mug', 40, 30)                     # being put down: the hand is still on it
    assert scene.run(world, 2.0) == []
    scene.hand_off(1)
    events = scene.run(world, 0.3)
    assert types(events) == [EventType.APPEARED]


def test_proposal_duplicating_a_known_object_is_ignored(scene, world):
    scene.place('keys', 40, 30)
    scene.thing('keys-again', 40, 30)
    scene.run(world, 1.0)
    scene.miss('keys')                             # class detector misses a couple of batches
    scene.run(world, 0.2)
    scene.miss('keys', False)
    assert scene.run(world, 2.0) == []
    assert things(world) == []
    assert world.get('keys').status == Status.VISIBLE


def test_implausibly_large_proposal_is_ignored(scene, world):
    scene.thing('table-blob', 64, 36, 120, 70)
    assert scene.run(world, 2.0) == []
    assert things(world) == []


def test_no_proposals_means_no_things_and_no_extra_state(scene, world):
    scene.place('keys', 40, 30)
    scene.run(world, 2.0)
    state = world.state_json()
    assert [e['name'] for e in state['entities']] == world.cfg.names()
    assert state['aliases'] == {} and state['merged'] == {}


# ----- the same rules as the known objects -------------------------------------------------------

def carry_into_box(scene, world, key='mug', at=(40, 30), hid=1):
    appear(scene, world, key, at)
    scene.place('box', *BOX_AT)
    scene.run(world, 1.0)
    scene.hand(hid, *at)
    events = scene.run(world, 0.3)
    scene.remove(key)
    events += scene.run(world, 0.2)
    scene.hand(hid, 60, 35)
    events += scene.run(world, 0.3)
    scene.hand(hid, *BOX_AT)
    events += scene.run(world, 0.6)
    scene.hand(hid, 100, 60)
    events += scene.run(world, 0.1)
    scene.hand_off(hid)
    return events + scene.run(world, world.cfg.reappear_wait_s + 0.5)


def test_thing_put_in_the_box_and_box_moved_resolves_to_the_box(scene, world):
    events = carry_into_box(scene, world)
    assert of(events, 'thing:1') == [EventType.PICKED_UP, EventType.PUT_INSIDE]
    assert (world.get('thing:1').status, world.get('thing:1').parent) == (Status.INSIDE, 'box')
    scene.hand(1, *BOX_AT)
    scene.run(world, 0.3)
    scene.remove('box')
    scene.run(world, 1.0)
    scene.hand(1, 110, 40)
    scene.place('box', 110, 40)
    scene.run(world, 0.5)
    scene.hand_off(1)
    scene.run(world, 1.0)
    assert world.resolve('thing:1') == (pytest.approx((110, 40)), ['thing:1', 'box'])
    assert things(world) == ['thing:1']


def test_thing_follows_the_cover_rule_and_is_uncovered_in_place(scene, world):
    appear(scene, world, 'mug', (40, 30))
    scene.place('notebook', 90, 30)
    scene.run(world, 1.0)
    for i in range(1, 6):
        scene.place('notebook', 90 - 10 * i, 30)
        world.update(*scene.step())
    scene.remove('mug')
    events = scene.run(world, 1.0)
    assert of(events, 'thing:1') == [EventType.COVERED]
    assert world.get('thing:1').parent == 'notebook'
    for i in range(1, 6):                          # lift it off again
        scene.place('notebook', 40 + 10 * i, 30)
        world.update(*scene.step())
    scene.thing('mug', 40, 30)
    events = scene.run(world, 1.0)
    assert of(events, 'thing:1') == [EventType.UNCOVERED]
    assert things(world) == ['thing:1']


def test_thing_lost_in_place_and_seen_again_at_the_same_spot_keeps_its_identity(scene, world):
    appear(scene, world, 'mug', (40, 30))
    scene.miss('mug')
    assert types(scene.run(world, 1.5)) == [EventType.LOST_TRACK]
    scene.miss('mug', False)
    events = scene.run(world, 1.0)
    assert types(events) == [EventType.CORRECTED]
    assert events[0].obj == 'thing:1'
    assert things(world) == ['thing:1']


# ----- identity rule (a): causal ----------------------------------------------------------------

def test_thing_coming_out_of_the_box_is_the_one_inside_even_if_it_looks_different(scene, world):
    carry_into_box(scene, world)
    scene.hand(1, *BOX_AT)                         # reach into the box ...
    scene.run(world, 0.5)
    scene.hand(1, 97, 40)                          # ... and set something down beside it
    scene.thing('mug-out', 97, 40, look=look('something-else'))
    events = scene.run(world, 0.5)
    scene.hand(1, 115, 62)
    events += scene.run(world, 1.5)
    assert of(events, 'thing:1') == [EventType.TAKEN_OUT]
    assert EventType.APPEARED not in types(events)
    assert world.get('thing:1').status == Status.VISIBLE
    assert world.get('thing:1').pos_cm == pytest.approx((97, 40))
    assert things(world) == ['thing:1']


def test_something_set_beside_the_box_without_a_hand_in_the_box_is_new(scene, world):
    carry_into_box(scene, world)
    scene.hand(2, 100, 58)                         # never touches the box
    scene.thing('other', 100, 55)
    scene.run(world, 0.5)
    scene.hand_off(2)
    events = scene.run(world, 1.0)
    assert [(e.obj, e.type) for e in events] == [('thing:2', EventType.APPEARED)]
    assert world.get('thing:1').status == Status.INSIDE


# ----- identity rule (b): continuity --------------------------------------------------------------

def test_two_lookalike_things_swapped_by_two_hands_keep_their_identities(scene, world):
    same = look('mug')
    appear(scene, world, 'a', (30, 30), look=same)
    appear(scene, world, 'b', (60, 30), look=same)
    assert world.get('thing:1').pos_cm == pytest.approx((30, 30))
    scene.hand(1, 30, 30)
    scene.hand(2, 60, 30)
    scene.run(world, 0.3)
    scene.remove('a')
    scene.remove('b')
    events = scene.run(world, 1.0)
    assert sorted((e.obj, e.parent) for e in events) == [('thing:1', 'hand:1'), ('thing:2', 'hand:2')]
    for f in (0.25, 0.5, 0.75, 1.0):               # the hands cross over
        scene.hand(1, 30 + 30 * f, 30 + 15 * f * (1 - f))
        scene.hand(2, 60 - 30 * f, 30 - 15 * f * (1 - f))
        world.update(*scene.step())
    scene.thing('a', 60, 30, look=same)            # hand 1 puts its thing down where b was
    scene.thing('b', 30, 30, look=same)
    events = scene.run(world, 0.5)
    scene.hand_off(1)
    scene.hand_off(2)
    events += scene.run(world, 1.0)
    assert sorted(types(events)) == [EventType.MOVED, EventType.MOVED]
    assert world.get('thing:1').pos_cm == pytest.approx((60, 30))
    assert world.get('thing:2').pos_cm == pytest.approx((30, 30))
    assert things(world) == ['thing:1', 'thing:2']


# ----- identity rule (c)/(d): appearance only when decisive; otherwise a new, linked thing ---------

def exit_left(scene, world, key='mug', at=(40, 30), twin_at=None, **kw):
    appear(scene, world, key, at, seconds=2.0, **kw)
    if twin_at is not None:                        # a look-alike that stays on the table
        appear(scene, world, 'twin', twin_at, seconds=2.0, **kw)
    scene.hand(1, *at)
    scene.run(world, 0.3)
    scene.remove(key)
    scene.run(world, 1.0)
    scene.hand(1, 5, 30)
    scene.run(world, 0.3)
    scene.hand_off(1)
    events = scene.run(world, 1.0)
    assert world.get('thing:1').status == Status.GONE
    return events


def put_down_new(scene, world, key, at, **kw):
    scene.hand(2, *at)
    scene.thing(key, *at, **kw)
    events = scene.run(world, 0.5)
    scene.hand_off(2)
    return events + scene.run(world, 1.0)


def test_similar_object_after_one_left_is_a_new_thing_that_may_be_the_same(scene, world):
    exit_left(scene, world)
    events = put_down_new(scene, world, 'cup', (50, 30), look=look_like('cup', 'mug', 0.85))
    assert [(e.obj, e.type) for e in events] == [('thing:2', EventType.APPEARED)]
    new = world.get('thing:2')
    assert [n for n, _ in new.maybe_same_as] == ['thing:1']
    assert new.maybe_same_as[0][1] == pytest.approx(0.85, abs=0.01)
    assert world.get('thing:1').status == Status.GONE
    assert world.similar_to('thing:1') == [('thing:2', pytest.approx(0.85, abs=0.01))]


def test_decisive_appearance_resurrects_the_thing_that_left(scene, world):
    exit_left(scene, world)
    events = put_down_new(scene, world, 'mug', (50, 30))
    assert [(e.obj, e.type) for e in events] == [('thing:1', EventType.CORRECTED)]
    assert world.get('thing:1').status == Status.VISIBLE
    assert world.get('thing:1').pos_cm == pytest.approx((50, 30))
    assert things(world) == ['thing:1']


def test_identical_look_is_not_decisive_while_a_twin_is_on_the_table(scene, world):
    same = look('mug')
    exit_left(scene, world, twin_at=(90, 50), look=same)                # the twin is thing:2
    events = put_down_new(scene, world, 'third', (50, 30), look=same)
    assert [(e.obj, e.type) for e in events] == [('thing:3', EventType.APPEARED)]
    assert [n for n, _ in world.get('thing:3').maybe_same_as] == ['thing:1']
    assert world.get('thing:1').status == Status.GONE


def test_without_an_embedder_a_returning_object_is_new_and_unlinked(cfg, scene):
    world = World(cfg)
    exit_left(scene, world)
    events = put_down_new(scene, world, 'mug', (50, 30))
    assert [(e.obj, e.type) for e in events] == [('thing:2', EventType.APPEARED)]
    assert world.get('thing:2').maybe_same_as == []


def test_confirm_same_folds_the_new_thing_into_the_old_identity(scene, world):
    exit_left(scene, world)
    put_down_new(scene, world, 'cup', (50, 30), look=look_like('cup', 'mug', 0.85))
    assert world.confirm_same('thing:2', 'thing:1') == 'thing:1'
    old = world.get('thing:1')
    assert (old.status, old.pos_cm) == (Status.VISIBLE, pytest.approx((50, 30)))
    assert world.get('thing:2').merged_into == 'thing:1'
    assert world.find('thing:2') == 'thing:1'
    assert scene.run(world, 2.0) == []            # the proposal now feeds thing:1
    assert world.get('thing:1').status == Status.VISIBLE
    assert things(world) == ['thing:1']
    state = world.state_json()
    assert 'thing:2' not in [e['name'] for e in state['entities']]
    assert state['merged'] == {'thing:2': 'thing:1'}
    assert types(world.history('thing:1', 10))[:2] == [EventType.CORRECTED, EventType.APPEARED]


def test_confirm_same_refuses_two_things_visible_at_once(scene, world):
    appear(scene, world, 'a', (30, 30))
    appear(scene, world, 'b', (60, 30))
    assert world.confirm_same('thing:1', 'thing:2') is None
    assert things(world) == ['thing:1', 'thing:2']


# ----- exemplar bank ----------------------------------------------------------------------------

def test_exemplars_are_learned_only_from_isolated_confident_views(scene, world):
    appear(scene, world, 'mug', (40, 30), seconds=2.0)
    bank = world.exemplars('thing:1')
    assert len(bank) == 1 and bank[0] @ look('mug') == pytest.approx(1.0)
    scene.looks['mug'] = look('mug-turned')        # a new view, but a hand overlaps it
    scene.hand(1, 44, 30)
    scene.run(world, 2.0)
    assert len(world.exemplars('thing:1')) == 1
    scene.hand_off(1)                              # isolated again: the new view is learned
    scene.run(world, 2.0)
    assert len(world.exemplars('thing:1')) == 2


def test_merge_then_split_keeps_both_identities_and_freezes_their_exemplars(scene, world, cfg):
    appear(scene, world, 'a', (40, 30), seconds=2.0)
    appear(scene, world, 'b', (50, 30), seconds=2.0)
    before = [len(world.exemplars(n)) for n in ('thing:1', 'thing:2')]
    scene.remove('a')
    scene.remove('b')
    scene.thing('blob', 45, 30, 22, 5)             # one proposal now covers both
    events = scene.run(world, 3.0)
    assert events == []
    for n in ('thing:1', 'thing:2'):
        ent = world.get(n)
        assert ent.status == Status.VISIBLE
        assert ent.confidence == pytest.approx(cfg.ambiguity_penalty)
    assert things(world) == ['thing:1', 'thing:2']
    scene.remove('blob')
    scene.thing('a', 40, 30, look=look_like('a-new', 'a', 0.5))    # new, learnable views ...
    scene.thing('b', 50, 30, look=look_like('b-new', 'b', 0.5))
    assert scene.run(world, 1.0) == []
    assert world.get('thing:1').pos_cm == pytest.approx((40, 30))
    assert world.get('thing:2').pos_cm == pytest.approx((50, 30))
    assert [len(world.exemplars(n)) for n in ('thing:1', 'thing:2')] == before    # ... not yet
    assert world.get('thing:1').confidence == 1.0
    assert things(world) == ['thing:1', 'thing:2']
    scene.run(world, cfg.openworld.get('ambiguous_s', 2.0) + 1.0)
    assert [len(world.exemplars(n)) for n in ('thing:1', 'thing:2')] == [b + 1 for b in before]


def test_split_with_swapped_positions_is_resolved_by_appearance(scene, world):
    appear(scene, world, 'a', (40, 30), seconds=2.0)
    appear(scene, world, 'b', (50, 30), seconds=2.0)
    scene.remove('a')
    scene.remove('b')
    scene.thing('blob', 45, 30, 22, 5)
    scene.run(world, 1.0)
    scene.remove('blob')
    scene.thing('a', 50, 30)                       # they came apart the other way round
    scene.thing('b', 40, 30)
    scene.run(world, 1.0)
    assert world.get('thing:1').pos_cm == pytest.approx((50, 30))
    assert world.get('thing:2').pos_cm == pytest.approx((40, 30))


# ----- state JSON and the default embedder ------------------------------------------------------

def test_state_json_lists_things_with_aliases_and_links(scene, world):
    exit_left(scene, world)
    put_down_new(scene, world, 'cup', (50, 30), look=look_like('cup', 'mug', 0.85))
    assert world.bind_alias('thing:1', 'my charger')
    state = world.state_json()
    json.dumps(state)
    by = {e['name']: e for e in state['entities']}
    assert by['thing:1']['aliases'] == ['charger'] and by['thing:1']['label'] == 'charger'
    assert by['thing:2']['label'] is None
    assert by['thing:2']['maybe_same_as'] == [['thing:1', pytest.approx(0.85, abs=0.01)]]
    assert ['thing:2', 'ON', 'table'] in state['edges']
    assert state['aliases'] == {'charger': 'thing:1'}
    assert set(by['keys']) == {'name', 'kind', 'status', 'parent', 'pos_cm', 'resolved_cm',
                               'confidence', 'candidates', 'last_seen', 'zone', 'edge'}


def test_hsv_embedder_is_a_unit_vector_that_separates_colours():
    img = np.zeros((100, 200, 3), np.uint8)
    img[:, :100] = (0, 0, 220)                     # red half (BGR)
    img[:, 100:] = (220, 60, 0)                    # blue half
    red, red2 = hsv_embed(img, (10, 10, 60, 60)), hsv_embed(img, (20, 30, 90, 90))
    blue = hsv_embed(img, (110, 10, 190, 90))
    assert np.linalg.norm(red) == pytest.approx(1.0)
    assert red @ red2 == pytest.approx(1.0, abs=1e-3)
    assert red @ blue < 0.5
    assert hsv_embed(None, (0, 0, 50, 50)) is None
    assert hsv_embed(img, (10, 10, 13, 13)) is None                     # too small to judge


def test_rendered_scene_with_the_default_embedder_runs(cfg):
    scene = Scene(cfg, fps=10, t0=1000.0, render=True)
    world = World(cfg, embed=hsv_embed)
    appear(scene, world, 'mug', (40, 30), seconds=2.0)
    assert things(world) == ['thing:1']
    assert len(world.exemplars('thing:1')) == 1


def test_confirm_same_keeps_the_survivors_other_links(scene, world):
    exit_left(scene, world)                                        # thing:1 leaves
    put_down_new(scene, world, 'cup', (50, 30), look=look_like('cup', 'mug', 0.85))   # thing:2 ~ 1
    scene.hand(1, 50, 30)                                          # thing:2 leaves too
    scene.run(world, 0.3)
    scene.remove('cup')
    scene.run(world, 1.0)
    scene.hand(1, 5, 30)
    scene.run(world, 0.3)
    scene.hand_off(1)
    scene.run(world, 1.0)
    cup, other = scene.looks['cup'], look('jar')
    other = other - (other @ cup) * cup
    jar = 0.8 * cup + 0.6 * other / np.linalg.norm(other)          # cosine 0.8 with the cup
    put_down_new(scene, world, 'jar', (70, 30), look=jar)          # thing:3 ~ thing:2
    assert [n for n, _ in world.get('thing:3').maybe_same_as] == ['thing:2']
    world.get('thing:3').maybe_same_as.append(('thing:1', 0.75))
    assert world.confirm_same('thing:3', 'thing:2') == 'thing:2'
    assert world.get('thing:2').maybe_same_as == [('thing:1', pytest.approx(0.85, abs=0.01))]


def test_a_thing_that_jumps_out_of_its_gate_is_kept_by_a_decisive_look(scene, world):
    appear(scene, world, 'mug', (40, 30), seconds=2.0)
    scene.thing('mug', 60, 30)                     # knocked 20 cm in one batch, no hand seen
    events = scene.run(world, 2.0)
    assert EventType.APPEARED not in types(events)
    assert world.get('thing:1').pos_cm == pytest.approx((60, 30))
    assert world.get('thing:1').status == Status.VISIBLE
    assert things(world) == ['thing:1']


def test_one_of_two_hidden_things_coming_out_is_new_and_linked_to_both(cfg, scene):
    world = World(cfg)                                             # no appearance to break the tie
    carry_into_box(scene, world, 'mug', (40, 30))
    carry_into_box(scene, world, 'pen', (40, 50))
    assert [world.get(n).status for n in ('thing:1', 'thing:2')] == [Status.INSIDE] * 2
    scene.hand(1, *BOX_AT)
    scene.run(world, 0.5)
    scene.hand(1, 97, 40)
    scene.thing('out', 97, 40)
    scene.run(world, 0.5)
    scene.hand(1, 115, 62)
    events = scene.run(world, 1.5)
    assert [(e.obj, e.type) for e in events] == [('thing:3', EventType.APPEARED)]
    assert sorted(n for n, _ in world.get('thing:3').maybe_same_as) == ['thing:1', 'thing:2']
    for n in ('thing:1', 'thing:2'):
        assert world.get(n).status == Status.INSIDE
        assert world.get(n).confidence < cfg.conf_inside


# ----- ids are never reused (the event log outlives the process and RESET) ---------------------

def test_thing_ids_continue_after_the_event_log_and_across_reset(cfg, scene, tmp_path):
    """A new thing:N must not inherit an unrelated old thing:N's history: numbering continues after the
    highest thing:N in the event log (a previous run), and RESET never lowers it."""
    from core.events import EventLog
    from core.types import Event
    events = EventLog(':memory:', str(tmp_path))
    for n in (2, 10):
        events.add(Event(t=5.0, wall=1.0, obj=f'thing:{n}', type=EventType.APPEARED, to_cm=(1.0, 1.0)))
    world = World(cfg, events, embed=scene.embed)
    assert [e.obj for e in appear(scene, world, 'mug', (40, 30))] == ['thing:11']
    world.reset()
    scene.remove('mug')
    assert [e.obj for e in appear(scene, world, 'book', (20, 20))] == ['thing:12']
    assert [e.type for e in world.history('thing:12', 5)] == [EventType.APPEARED]
    events.close()
