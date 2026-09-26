"""core/capture.py without hardware: a fake camera and a generated video file."""
import json
import threading
import time

import cv2
import numpy as np
import pytest

from core.capture import FrameBuffer, VideoFileSource


class FakeCam:
    """read() at about `fps`, each image filled with its frame number; can fail on chosen reads."""

    def __init__(self, fps=200.0, fail_every=0):
        self.dt, self.n, self.fail_every = 1.0 / fps, 0, fail_every
        self.released = threading.Event()

    def read(self):
        time.sleep(self.dt)
        self.n += 1
        if self.fail_every and self.n % self.fail_every == 0:
            return False, None
        return True, np.full((4, 4, 3), self.n % 256, np.uint8)

    def release(self):
        self.released.set()


def wait_for(cond, timeout=2.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if cond():
            return True
        time.sleep(0.005)
    return False


def test_latest_is_the_newest_frame_and_keeps_advancing():
    fb = FrameBuffer(FakeCam())
    assert wait_for(lambda: fb.latest() is not None and fb.latest().idx >= 5)
    a = fb.latest()
    assert wait_for(lambda: fb.latest().idx > a.idx)
    assert fb.latest().t > a.t and time.monotonic() - fb.latest().t < 0.05
    fb.stop()


def test_at_returns_the_ring_frame_nearest_a_time():
    fb = FrameBuffer(FakeCam())
    assert wait_for(lambda: fb.latest() is not None and fb.latest().idx >= 20)
    mid = fb.at(fb.latest().t - 0.05)
    ring = list(fb._ring)
    assert mid is min(ring, key=lambda f: abs(f.t - (fb.latest().t - 0.05))) or abs(mid.t - (fb.latest().t - 0.05)) < 0.01
    assert fb.at(0.0) is ring[0] or fb.at(0.0).idx <= ring[1].idx
    fb.stop()


def test_ring_keeps_only_ring_s_seconds():
    fb = FrameBuffer(FakeCam(), ring_s=0.1)
    assert wait_for(lambda: fb.latest() is not None and fb.latest().idx >= 60)
    ring = list(fb._ring)
    assert ring[-1].t - ring[0].t <= 0.1 + 0.02
    fb.stop()


def test_fps_is_measured_and_failures_are_counted_not_fatal():
    fb = FrameBuffer(FakeCam(fps=100, fail_every=5))
    assert wait_for(lambda: fb.failures >= 3 and fb.latest() is not None and fb.latest().idx >= 20)
    assert 40 < fb.fps < 110           # 4 of every 5 reads succeed
    fb.stop()


def test_wait_new_blocks_until_a_newer_frame():
    fb = FrameBuffer(FakeCam(fps=50))
    assert wait_for(lambda: fb.latest() is not None)
    f = fb.latest()
    g = fb.wait_new(f.idx, timeout=1.0)
    assert g is not None and g.idx > f.idx
    fb.stop()
    assert fb.cap.released.is_set()


@pytest.fixture
def clip(tmp_path):
    path = tmp_path / "video.avi"
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 20.0, (64, 48))
    for i in range(20):
        w.write(np.full((48, 64, 3), i * 10, np.uint8))
    w.release()
    (tmp_path / "timestamps.json").write_text(json.dumps([round(i * 0.05, 3) for i in range(20)]))
    return path


def test_video_frames_come_in_order_with_recorded_timestamps(clip):
    src = VideoFileSource(clip, start=False)
    frames = list(src.frames())
    assert [f.idx for f in frames] == list(range(20))
    assert [f.t for f in frames] == pytest.approx([i * 0.05 for i in range(20)])
    assert abs(float(frames[7].img.mean()) - 70) < 4        # MJPG is lossy


def test_video_source_as_fast_as_possible_delivers_every_frame(clip):
    src = VideoFileSource(clip, realtime=False)
    assert src.done.wait(2.0)
    assert src.latest().idx == 20
    src.stop()


def test_video_source_in_realtime_takes_the_clip_length(clip):
    t0 = time.monotonic()
    src = VideoFileSource(clip, realtime=True)
    assert src.done.wait(3.0)
    assert time.monotonic() - t0 >= 0.9                       # 20 frames at 0.05 s
    src.stop()
