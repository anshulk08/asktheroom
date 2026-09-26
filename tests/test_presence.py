"""core/presence.py: debounce windows counted by time, so live and replay rates agree."""
import pytest

from core.config import Config
from core.presence import Presence
from core.types import Status
from core.world import World
from tests.synth import Scene
from tests.test_world_rules import ROOT


def test_without_hz_it_is_the_last_n_updates():
    p = Presence(n=3)
    for t, b in enumerate([True, True, False, True]):
        p.push(t, b)
    assert list(p) == [True, False, True] and p.hits() == 2


@pytest.mark.parametrize('fps', [7, 10, 15, 30])
def test_hits_measure_time_in_updates_of_the_reference_rate(fps):
    p = Presence(n=10, hz=15)
    for i in range(3 * fps):                     # detected for 3 s
        p.push(i / fps, True)
    assert p.hits() == pytest.approx(10, abs=1.6)          # the window is 10 / 15 s whatever the rate
    assert p._t[-1] - p._t[0] < p.window_s


def test_one_stalled_update_never_counts_as_many():
    p = Presence(n=10, hz=15)
    p.push(0.0, True)
    p.push(0.6, True)                            # the loop stalled 0.6 s
    assert p.hits() <= 1 + 2.5


def test_bits_seeded_from_a_candidate_are_one_update_apart():
    p = Presence([True] * 5, n=10, hz=15)
    assert p.hits() == 5
    p.push(10.0, True)
    assert p.hits() == pytest.approx(6)


@pytest.mark.parametrize('fps', [7, 15])
def test_a_put_down_object_is_confirmed_after_the_same_time_at_7_and_15_fps(fps):
    cfg = Config.load(ROOT / 'config.yaml')
    assert cfg.presence_hz == 15
    scene, world = Scene(cfg, fps=fps, t0=1000.0), World(cfg)
    scene.place('wallet', 40, 30)
    t0 = scene.t
    while world.get('wallet').status != Status.VISIBLE:
        world.update(*scene.step())
    assert scene.t - t0 == pytest.approx((cfg.present_k - 1) / 15 + 1 / fps, abs=1 / fps + 1e-6)


@pytest.mark.parametrize('fps', [7, 15])
def test_absence_is_declared_after_the_same_time_at_7_and_15_fps(fps):
    cfg = Config.load(ROOT / 'config.yaml')
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
