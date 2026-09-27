"""core/presence.py: debounce windows counted by time, so live and replay rates agree."""
import random

import pytest

from core.config import Config, load_config
from core.presence import Presence
from core.types import Detections, Frame, Status
from core.world import World
from tests.synth import Scene
from tests.test_world_rules import ROOT


def test_without_hz_it_is_the_last_n_updates():
    p = Presence(n=3)
    for t, b in enumerate([True, True, False, True]):
        p.push(t, b)
    assert list(p) == [True, False, True] and p.hits() == 2


def test_at_the_reference_rate_with_jitter_it_is_exactly_the_last_n_updates():
    rnd = random.Random(1)
    p, q = Presence(n=10, hz=15), Presence(n=10)
    for i in range(300):
        t, b = 1000 + i / 15 + rnd.uniform(-0.004, 0.004), rnd.random() < 0.5
        p.push(t, b)
        q.push(t, b)
        assert list(p) == list(q) and p.hits() == sum(q)


@pytest.mark.parametrize('fps', [2.5, 7, 15, 30])
def test_the_window_spans_n_reference_periods_whatever_the_rate(fps):
    p = Presence(n=10, hz=15)
    for i in range(int(4 * fps)):
        p.push(i / fps, True)
    assert 10 <= p.hits() < 10 + 15 / fps + 1e-9 and p.hits() >= 6      # below 3.3 fps too


def test_after_a_slow_step_one_miss_does_not_drop_a_present_object():
    p = Presence(n=10, hz=15)
    for i in range(20):
        p.push(i / 15, True)
    p.push(19 / 15 + 0.6, False)                 # a 0.6 s step (hot room cadence), then one miss
    assert p.hits() > 1                          # absent_max 1: still not absent


def test_after_a_slow_step_one_false_hit_does_not_make_an_absent_object_present():
    p = Presence(n=10, hz=15)
    for i in range(20):
        p.push(i / 15, False)
    p.push(19 / 15 + 0.45, True)                 # a 0.45 s step, then one false detection
    assert p.hits() < 6                          # present_k 6: not present


def test_the_weight_is_continuous_with_no_jump_between_counting_modes():
    """At the room build's 10-13 fps a few ms of jitter must never switch a weight by 0.25 (a hard snap did)."""
    from core.presence import SNAP, weight
    for edge in (1 + SNAP, 1 + 2 * SNAP, 1 - SNAP, 1 - 2 * SNAP):
        assert abs(weight(edge + 1e-4) - weight(edge - 1e-4)) < 1e-3
    assert weight(1.2) == 1.0 and weight(1.5) == 1.5 and weight(2.14) == pytest.approx(2.14)   # 12, 10, 7 fps
    xs = [0.5 + i / 1000 for i in range(2000)]
    assert all(weight(b) >= weight(a) for a, b in zip(xs, xs[1:]))                          # monotone


def test_a_stall_is_no_evidence_one_miss_after_it_keeps_the_object():
    p = Presence(n=10, hz=15)
    for i in range(20):
        p.push(i / 15, True)
    p.push(20 / 15 + 2.0, False)                 # the loop stalled 2 s, then one miss
    assert p.hits() >= 6


def test_bits_seeded_from_a_candidate_count_one_each():
    p = Presence([True] * 5, n=10, hz=15)
    assert p.hits() == 5
    p.push(10.0, True)
    assert p.hits() == 6


def test_clear_forgets_the_weights_too():
    """A debounce restart (core/room_world.py clears an entity's bits) must not leave stale weights."""
    p = Presence(n=10, hz=15)
    for i in range(20):
        p.push(i / 15, False)
    p.clear()
    for i in range(6):
        p.push(5 + i / 15, True)
    assert list(p) == [True] * 6 and p.hits() == 6


def test_a_clock_that_goes_back_starts_the_window_over():
    p = Presence(n=10, hz=15)
    for i in range(20):
        p.push(100 + i / 15, True)
    p.push(0.0, False)
    assert list(p) == [False] and p.hits() == 0


# ----- the world model on it

def cfg_hz(hz):
    cfg = Config.load(ROOT / 'config.yaml')
    cfg.presence_hz = hz
    return cfg


@pytest.mark.parametrize('fps', [7, 15])
def test_a_put_down_object_is_confirmed_after_the_same_time_at_7_and_15_fps(fps):
    cfg = cfg_hz(15)
    scene, world = Scene(cfg, fps=fps, t0=1000.0), World(cfg)
    scene.place('wallet', 40, 30)
    t0 = scene.t
    while world.get('wallet').status != Status.VISIBLE:
        world.update(*scene.step())
    assert scene.t - t0 == pytest.approx((cfg.present_k - 1) / 15 + 1 / fps, abs=1 / fps + 1e-6)


@pytest.mark.parametrize('fps', [7, 15])
def test_absence_is_declared_after_the_same_time_at_7_and_15_fps(fps):
    cfg = cfg_hz(15)
    cfg.lost_grace_s = 0.0
    scene, world = Scene(cfg, fps=fps, t0=1000.0), World(cfg)
    scene.place('keys', 40, 30)
    scene.run(world, 2.0)
    scene.remove('keys')
    t0 = scene.t
    while world.get('keys').status == Status.VISIBLE:
        world.update(*scene.step())
        assert scene.t - t0 < 2.0
    assert scene.t - t0 <= 10 / 15 + 1 / fps + 1e-6


def play_story(cfg, times):
    """server/sim.py's demo story through the world at the given frame times."""
    from server.sim import story
    world = World(cfg)
    scene, _ = story(load_config(), seed=0)
    events = []
    for i, d in enumerate(scene.render()):
        t = times(i)
        events += world.update(Detections(t=t, frame_idx=i, items=d.items, hands=d.hands),
                               Frame(t=t, wall=t, idx=i, img=None))
    return world, [(e.obj, str(e.type), e.parent) for e in events]


def test_at_15_fps_with_jitter_the_world_is_exactly_as_with_frame_counting():
    rnd = random.Random(7)
    jit = [rnd.uniform(-0.004, 0.004) for _ in range(5000)]
    times = lambda i: 1000.0 + i / 15 + jit[i]                       # noqa: E731
    w1, e1 = play_story(cfg_hz(15), times)
    w0, e0 = play_story(cfg_hz(None), times)
    assert e1 == e0
    assert {n: (e.status, e.parent) for n, e in w1.entities.items()} == \
        {n: (e.status, e.parent) for n, e in w0.entities.items()}


def test_the_demo_story_at_2_5_fps_ends_with_the_right_beliefs():
    world, _ = play_story(cfg_hz(15), lambda i: 1000.0 + i / 2.5)
    assert (world.get("keys").status, world.get("keys").parent) == (Status.INSIDE, "box")
    assert (world.get("pill_bottle").status, world.get("pill_bottle").parent) == (Status.UNDER, "notebook")
    for name in ("wallet", "remote", "glasses", "box", "notebook"):
        assert world.get(name).status == Status.VISIBLE, name
