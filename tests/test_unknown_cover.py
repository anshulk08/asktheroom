"""A cover the detectors have no class for (a blanket, a jacket, a napkin) laid over objects by hand
(core/world.py rule 4, core/surround.py). The hands holding it touch what it covers, so the hand rule used
to read every touched object as picked up, then lost when the hands left. A real pick-up leaves the band
of table around the object's spot bare once the hand has moved off; a cover leaves it changed. So a spot
surrounded by change that is not a hand, settled, is UNDER the cover (a thing laid over it, else
'unknown'), and the objects come back UNCOVERED as themselves when it is lifted. Rendered scenes: the
rule reads pixels."""
from pathlib import Path

import numpy as np
import pytest

from core.config import Config, load_config
from core.events import EventLog
from core.surround import SurroundMemory, UnknownCoverConfig
from core.types import EventType, Status
from core.world import World
from tests.synth import Scene
from voice.answers import answer
from voice.intents import parse

ROOT = Path(__file__).resolve().parents[1]
KEYS_AT, WALLET_AT, PODS_AT = (40, 30), (62, 34), (50, 46)
BLANKET = (50, 38, 50, 34)                 # centre x, y, w, h: over all three (x 25-75, y 21-55)
RAW = load_config(ROOT / 'config.yaml')


@pytest.fixture
def cfg():
    return Config.load(ROOT / 'config.yaml')


@pytest.fixture
def rscene(cfg):
    return Scene(cfg, fps=10, t0=1000.0, render=True)


@pytest.fixture
def world(cfg, rscene, tmp_path):
    return World(cfg, EventLog(':memory:', str(tmp_path / 'snaps')), embed=rscene.embed)


def of(events, obj):
    return [e.type for e in events if e.obj == obj]


def table_of_three(s, w):
    """Keys and wallet (configured) and an AirPods case (a thing) lie still long enough to be known."""
    s.place('keys', *KEYS_AT)
    s.place('wallet', *WALLET_AT)
    s.thing('pods', *PODS_AT, 5, 6)
    s.run(w, 2.5)
    assert [w.get(n).status for n in ('keys', 'wallet', 'thing:1')] == [Status.VISIBLE] * 3


def lay_blanket(s, w, hold_s=0.4):
    """Two hands bring the blanket down: they rest on the keys and the wallet while it settles over all
    three, then slide off it and leave the view mid-table."""
    s.hand(1, 42, 32)                      # on the keys
    s.hand(2, 58, 40)                      # on the wallet (not the AirPods)
    events = s.run(w, 0.3)
    for k in ('keys', 'wallet', 'pods'):
        s.remove(k)
    s.overlay('blanket', *BLANKET)
    events += s.run(w, hold_s)
    s.hand(1, 90, 20)
    s.hand(2, 95, 62)
    events += s.run(w, 0.3)
    s.hand_off(1)
    s.hand_off(2)
    return events + s.run(w, 3.0)


def lift_blanket(s, w, back=('keys', 'wallet', 'pods')):
    """A hand lifts the blanket off; what lay under it is seen again."""
    s.hand(1, 50, 20)
    events = s.run(w, 0.3)
    s.overlays.pop('blanket')
    if 'keys' in back:
        s.place('keys', *KEYS_AT)
    if 'wallet' in back:
        s.place('wallet', *WALLET_AT)
    if 'pods' in back:
        s.thing('pods', *PODS_AT, 5, 6)
    s.hand(1, 50, 5)
    events += s.run(w, 0.3)
    s.hand_off(1)
    return events + s.run(w, 3.0)


# ----- laid over by hand: under it, not picked up ------------------------------------------------

def test_a_blanket_laid_by_hand_over_three_objects_covers_them_not_picks_them_up(rscene, world, cfg):
    table_of_three(rscene, world)
    events = lay_blanket(rscene, world)
    for n in ('keys', 'wallet', 'thing:1'):
        assert of(events, n) == [EventType.COVERED], n
        e = world.get(n)
        assert (e.status, e.parent) == (Status.UNDER, 'unknown'), n
        assert e.confidence == pytest.approx(cfg.conf_under_unknown, rel=0.01)
    assert world.get('keys').pos_cm == pytest.approx(KEYS_AT)        # still where it lay


def test_where_is_it_while_the_blanket_lies_there_answers_under_something(rscene, world, cfg):
    table_of_three(rscene, world)
    lay_blanket(rscene, world)
    text = answer(parse('where are my keys', RAW), world, world.events, RAW, now=world._wall).text
    assert 'under something' in text


def test_the_blanket_lifted_uncovers_the_same_identities(rscene, world):
    table_of_three(rscene, world)
    lay_blanket(rscene, world)
    events = lift_blanket(rscene, world)
    for n in ('keys', 'wallet', 'thing:1'):
        assert of(events, n) == [EventType.UNCOVERED], n
        assert world.get(n).status == Status.VISIBLE
    assert EventType.APPEARED not in [e.type for e in events]            # no new thing:N for the AirPods


def test_an_object_gone_when_the_blanket_is_lifted_is_lost_not_left_under_it(rscene, world, cfg):
    table_of_three(rscene, world)
    lay_blanket(rscene, world)
    events = lift_blanket(rscene, world, back=('keys', 'pods'))
    assert of(events, 'wallet') == [EventType.LOST_TRACK]
    wallet = world.get('wallet')
    assert wallet.status == Status.UNKNOWN
    assert wallet.confidence == pytest.approx(cfg.conf_under_unknown * cfg.lifted_cover_penalty, rel=0.02)


def test_a_blanket_laid_without_hands_seen_covers_objects_there_since_startup(cfg, world):
    """Objects on the table from the first frame: the background model never saw the table under them,
    so only the band around them tells a cover from a detector miss. Untouched, the absence first waits
    out the lost grace (an arm no detector saw lying across them looks alike for a while)."""
    s = Scene(cfg, fps=10, t0=1000.0, render=True)
    s.place('keys', *KEYS_AT)
    s.run(world, 2.0)
    s.remove('keys')
    s.overlay('blanket', *BLANKET)
    assert s.run(world, cfg.lost_grace_s - 0.5) == []
    events = s.run(world, 1.0)
    assert of(events, 'keys') == [EventType.COVERED]
    assert (world.get('keys').status, world.get('keys').parent) == (Status.UNDER, 'unknown')


# ----- a napkin the proposer sees: that thing is the parent -------------------------------------------

def test_a_napkin_laid_over_the_keys_is_their_parent_and_lifting_it_uncovers_them(rscene, world, cfg):
    rscene.place('keys', *KEYS_AT)
    rscene.run(world, 2.5)
    rscene.hand(1, 42, 32)
    events = rscene.run(world, 0.3)
    rscene.remove('keys')
    rscene.thing('napkin', 41, 31, 20, 16)          # seen by the proposer as one thing
    events += rscene.run(world, 0.4)
    rscene.hand(1, 90, 20)
    events += rscene.run(world, 0.3)
    rscene.hand_off(1)
    events += rscene.run(world, 3.0)
    assert of(events, 'keys')[-1] == EventType.COVERED
    assert EventType.PICKED_UP not in of(events, 'keys')
    keys = world.get('keys')
    assert (keys.status, keys.parent) == (Status.UNDER, 'thing:1')
    assert world.resolve('keys')[1] == ['keys', 'thing:1']
    rscene.hand(1, 41, 31)                          # lift the napkin off and away
    events = rscene.run(world, 0.3)
    rscene.remove('napkin')
    rscene.place('keys', *KEYS_AT)
    rscene.hand(1, 100, 10)
    events += rscene.run(world, 0.3)
    rscene.hand_off(1)
    events += rscene.run(world, 3.0)
    assert of(events, 'keys') == [EventType.UNCOVERED]


def test_a_napkin_lifted_with_the_keys_gone_loses_them(rscene, world, cfg):
    rscene.place('keys', *KEYS_AT)
    rscene.run(world, 2.5)
    rscene.remove('keys')
    rscene.thing('napkin', 41, 31, 20, 16)
    rscene.run(world, 3.0)
    assert world.get('keys').parent == 'thing:1'
    rscene.remove('napkin')                         # lifted, and the keys went with it
    events = rscene.run(world, cfg.lost_grace_s + cfg.reappear_wait_s + 1.5)    # the napkin is lost first
    assert of(events, 'keys') == [EventType.LOST_TRACK]


# ----- what must not change ----------------------------------------------------------------------

def test_keys_grabbed_and_carried_off_the_edge_are_still_picked_up_then_exited(rscene, world):
    rscene.place('keys', 100, 30)
    rscene.run(world, 2.5)
    rscene.hand(1, 100, 30)
    events = rscene.run(world, 0.4)
    rscene.remove('keys')
    for x in (106, 112, 118, 124):
        rscene.hand(1, x, 30)
        events += world.update(*rscene.step())
    rscene.hand_off(1)
    events += rscene.run(world, 2.0)
    assert of(events, 'keys') == [EventType.PICKED_UP, EventType.EXITED_VIEW]


def test_keys_grasped_a_while_then_carried_away_are_picked_up(rscene, world):
    """The hand rests on the keys (hiding the band around them) before it carries them off mid-table."""
    rscene.place('keys', 60, 30)
    rscene.run(world, 2.5)
    rscene.hand(1, 60, 30)
    events = rscene.run(world, 0.3)
    rscene.remove('keys')
    events += rscene.run(world, 1.2)
    for x in (66, 72, 78, 84):
        rscene.hand(1, x, 40)
        events += world.update(*rscene.step())
    events += rscene.run(world, 0.5)
    assert of(events, 'keys') == [EventType.PICKED_UP]
    assert world.get('keys').status == Status.HELD


def test_notebook_slid_over_keys_by_hand_is_still_under_the_notebook(rscene, world, cfg):
    rscene.place('keys', *KEYS_AT)
    rscene.place('notebook', 90, 30)
    rscene.run(world, 2.5)
    events = []
    for i in range(1, 6):
        rscene.place('notebook', 90 - 10 * i, 30)
        rscene.hand(1, 90 - 10 * i, 30)
        events += world.update(*rscene.step())
    rscene.remove('keys')
    events += rscene.run(world, 0.3)
    rscene.hand_off(1)
    events += rscene.run(world, 2.0)
    assert of(events, 'keys') == [EventType.COVERED]
    assert (world.get('keys').status, world.get('keys').parent) == (Status.UNDER, 'notebook')


def test_disabled_the_hand_rule_decides_as_before(cfg, tmp_path):
    cfg.unknown_cover = {'enabled': False}
    s = Scene(cfg, fps=10, t0=1000.0, render=True)
    w = World(cfg, EventLog(':memory:', str(tmp_path / 'snaps')), embed=s.embed)
    table_of_three(s, w)
    events = lay_blanket(s, w)
    assert of(events, 'keys')[0] == EventType.PICKED_UP


# ----- the band memory itself ------------------------------------------------------------------------

BOX_PX, BOX_CM = (400, 280, 460, 320), (40.0, 28.0, 46.0, 32.0)


def table_img():
    img = np.empty((720, 1280, 3), np.uint8)
    img[:] = (200, 210, 220)
    return img


def remembered(mem, img, t0=0.0, secs=1.2):
    for i in range(int(secs * 10) + 1):
        mem.observe('keys', img, BOX_PX, BOX_CM, t0 + i / 10, [])


def test_the_band_is_remembered_only_after_it_stayed_the_same():
    mem = SurroundMemory(UnknownCoverConfig())
    img = table_img()
    remembered(mem, img, secs=0.5)
    assert mem.memory('keys') is None
    remembered(mem, img, t0=0.6, secs=0.6)
    assert mem.memory('keys') is not None


def test_a_band_that_keeps_changing_is_not_remembered():
    mem = SurroundMemory(UnknownCoverConfig())
    for i in range(20):
        img = table_img()
        img[250:350, 360:500] = (i * 12) % 255          # something moving about the spot
        mem.observe('keys', img, BOX_PX, BOX_CM, i / 10, [])
    assert mem.memory('keys') is None


def test_a_band_covered_all_round_looks_covered_and_bare_looks_bare():
    mem = SurroundMemory(UnknownCoverConfig())
    img = table_img()
    remembered(mem, img)
    assert mem.look('keys', img, []) is False
    assert mem.bare('keys', img, [])
    covered = table_img()
    covered[200:400, 300:560] = (30, 30, 200)             # a red blanket over the spot and all round it
    assert mem.look('keys', covered, []) is True
    assert not mem.bare('keys', covered, [])


def test_an_arm_across_one_side_is_not_a_cover():
    mem = SurroundMemory(UnknownCoverConfig())
    img = table_img()
    remembered(mem, img)
    arm = table_img()
    arm[300:720, 390:470] = (140, 170, 215)               # an arm reaching in from below over the spot
    assert mem.look('keys', arm, []) is False


def test_a_band_hidden_by_hands_is_unclear():
    mem = SurroundMemory(UnknownCoverConfig())
    img = table_img()
    remembered(mem, img)
    covered = table_img()
    covered[200:400, 300:560] = (30, 30, 200)
    hands = [(350, 230, 520, 400)]                         # a hand box over the spot and most of the band
    assert mem.look('keys', covered, hands) is None


def test_an_undetected_arm_moving_over_untouched_keys_is_not_a_cover(cfg, world):
    """place_1: a sleeved arm (no hand detected) lay across the phone for the whole lost grace. It covers
    the band all round like a blanket, but it keeps moving; a cover lies at rest. So the phone is lost as
    before, not UNDER something (and found again when the arm goes)."""
    s = Scene(cfg, fps=10, t0=1000.0, render=True)
    s.place('keys', *KEYS_AT)
    s.run(world, 2.0)
    s.miss('keys')
    events = []
    for i in range(int((cfg.lost_grace_s + 1.0) * 10)):
        s.overlay('sleeve', 40 + (i % 5) - 2, 30 + (i % 3), 24, 16)    # shifts a little every batch
        events += world.update(*s.step())
    assert of(events, 'keys') == [EventType.LOST_TRACK]


def test_objects_seen_again_when_the_blanket_is_lifted_were_not_put_down(rscene, world):
    """'this is my X' names what was just put down; the blanket lifting off the table puts nothing down."""
    table_of_three(rscene, world)
    lay_blanket(rscene, world)
    placed = dict(world._placed_t)                 # when each was last put down in view
    lift_blanket(rscene, world)
    assert world._placed_t == placed


def test_a_label_read_on_a_thing_uncovered_in_place_is_not_the_object_moving(cfg, world):
    """blanket_1t, as the blanket came off: the detector read 'phone' on the tape roll (a thing) in the
    frame before the tape was seen again. The tape was hidden where it had lain since before the phone was
    last seen, so that label is the tape misnamed; the phone did not move onto it."""
    from tests.synth import DEFAULT_SIZE_CM
    s = Scene(cfg, fps=10, t0=1000.0, render=True)
    s.place('keys', *KEYS_AT)
    s.thing('tape', 56, 30, 6, 6)
    s.run(world, 3.5)
    for k in ('keys', 'tape'):
        s.remove(k)
    s.overlay('blanket', *BLANKET)
    s.run(world, cfg.lost_grace_s + 1.5)
    assert [world.get(n).status for n in ('keys', 'thing:1')] == [Status.UNDER] * 2
    del s.overlays['blanket']
    s.overlay('real keys', *KEYS_AT, *DEFAULT_SIZE_CM['target'])     # drawn, not named by the detector
    s.place('keys', 56, 30, 6, 6)                  # 'keys' read on the tape's box ...
    s.thing('tape', 56, 30, 6, 6)                  # ... which the proposer sees too
    s.run(world, 2.0)
    assert world.get('keys').pos_cm == pytest.approx(KEYS_AT, abs=1.0)
