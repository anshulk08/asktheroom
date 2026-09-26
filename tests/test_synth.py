from pathlib import Path

import pytest

from core.config import Config
from core.types import Event, EventType
from tests.synth import Scene

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def scene():
    return Scene(Config.load(ROOT / 'config.yaml'), fps=10, t0=1000.0)


def test_step_reports_placed_objects_in_cm_and_px(scene):
    scene.place('keys', 40, 30)
    dets, frame = scene.step()
    (d,) = dets.items
    assert d.cls == 'keys'
    assert d.center_cm == pytest.approx((40, 30))
    assert d.box_cm == pytest.approx((37, 28, 43, 32))       # default target size 6 x 4 cm
    assert d.box_px == (370, 280, 430, 320)                    # 10 px per cm
    assert frame.idx == dets.frame_idx


def test_removed_object_is_not_reported(scene):
    scene.place('keys', 40, 30)
    scene.remove('keys')
    dets, _ = scene.step()
    assert dets.items == []


def test_hands_carry_track_ids(scene):
    scene.hand(1, 50, 30)
    dets, _ = scene.step()
    (h,) = dets.hands
    assert h.cls == 'hand:1'
    assert h.center_cm == pytest.approx((50, 30))


def test_time_advances_one_period_per_step(scene):
    a, _ = scene.step()
    b, _ = scene.step()
    assert b.t - a.t == pytest.approx(0.1)
    assert b.frame_idx == a.frame_idx + 1


def test_container_and_cover_get_larger_default_boxes(scene):
    scene.place('box', 60, 40)
    scene.place('notebook', 20, 20)
    dets, _ = scene.step()
    sizes = {d.cls: (d.box_cm[2] - d.box_cm[0], d.box_cm[3] - d.box_cm[1]) for d in dets.items}
    assert sizes['box'] == pytest.approx((20, 15))
    assert sizes['notebook'] == pytest.approx((25, 18))


def test_run_feeds_world_for_duration_and_collects_events(scene):
    class FakeWorld:
        calls = 0

        def update(self, dets, frame):
            FakeWorld.calls += 1
            return [Event(t=dets.t, wall=0, obj='keys', type=EventType.MOVED)]

    events = scene.run(FakeWorld(), seconds=1.0)
    assert FakeWorld.calls == 10
    assert len(events) == 10
