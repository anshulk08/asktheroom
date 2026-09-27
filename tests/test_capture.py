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


# ----- a camera that drops off USB is reopened and gets its settings back ---------------------------------

class DyingCam(FakeCam):
    """Works for `good` reads, then fails forever (unplugged)."""
    def __init__(self, good=20, **kw):
        super().__init__(**kw)
        self.good = good

    def read(self):
        time.sleep(self.dt)
        self.n += 1
        return (True, np.zeros((4, 4, 3), np.uint8)) if self.n <= self.good else (False, None)


class Controls:
    def __init__(self):
        self.snaps, self.restored = [], []

    def snapshot(self, path):
        self.snaps.append(path)
        return ["exposure=333", "focus=40"]

    def restore(self, path, snap):
        self.restored.append((path, list(snap)))
        return len(snap)


def test_a_camera_that_drops_off_usb_is_reopened_with_its_settings_restored():
    first, second = DyingCam(good=20), FakeCam()
    opened = []

    def opener(src):
        opened.append(src)
        if len(opened) == 1:
            return first
        if len(opened) == 2:
            raise RuntimeError("can't open camera")          # still re-enumerating
        return second

    ctl = Controls()
    fb = FrameBuffer("/dev/v4l/by-id/cam", opener=opener, controls=ctl, reopen_after_s=0.05, retry_s=0.02)
    try:
        assert wait_for(lambda: fb.reconnects == 1 and fb.latest() is not None and fb.latest().idx > 40)
        assert opened == ["/dev/v4l/by-id/cam"] * 3 and first.released.is_set()
        assert ctl.snaps == ["/dev/v4l/by-id/cam"]               # snapshot once, when first opened
        assert ctl.restored == [("/dev/v4l/by-id/cam", ["exposure=333", "focus=40"])]
    finally:
        fb.stop()


def test_a_test_source_without_a_device_is_never_reopened():
    fb = FrameBuffer(DyingCam(good=5), reopen_after_s=0.02, retry_s=0.01)
    try:
        time.sleep(0.2)
        assert fb.reconnects == 0 and fb.failures > 0
    finally:
        fb.stop()


class RaisingCam(FakeCam):
    """cv2.error-like exceptions on some reads (a corrupt MJPG frame, a driver hiccup)."""
    def read(self):
        ok, img = super().read()
        if self.n % 4 == 0:
            raise cv2.error("corrupt frame")
        return ok, img


def test_a_read_that_raises_never_ends_the_capture_thread():
    fb = FrameBuffer(RaisingCam(fps=200))
    try:
        assert wait_for(lambda: fb.failures >= 5 and fb.latest() is not None and fb.latest().idx >= 30)
        assert fb._thread.is_alive() and fb.age() < 0.5
    finally:
        fb.stop()


def test_an_error_outside_the_read_is_survived_too(monkeypatch):
    fb = FrameBuffer(FakeCam(fps=200))
    try:
        assert wait_for(lambda: fb.latest() is not None)
        real, calls = fb._fresh.notify_all, []

        def boom():
            calls.append(1)
            if len(calls) == 1:
                raise RuntimeError("boom")
            real()
        monkeypatch.setattr(fb._fresh, "notify_all", boom)
        n = fb.latest().idx
        assert wait_for(lambda: fb.latest().idx > n + 10) and fb._thread.is_alive()
    finally:
        fb.stop()


# ----- decode only what the app uses: grab every frame, retrieve (JPEG decode) at most decode_fps -----

class GrabCam(FakeCam):
    """grab() takes a frame off the camera; retrieve() decodes the last grabbed one (counted)."""
    def __init__(self, fps=200.0):
        super().__init__(fps=fps)
        self.decodes = 0

    def grab(self):
        time.sleep(self.dt)
        self.n += 1
        return True

    def retrieve(self):
        self.decodes += 1
        return True, np.full((4, 4, 3), self.n % 256, np.uint8)

    def read(self):                                 # OpenCV's read() is grab() + retrieve()
        return (self.retrieve() if self.grab() else (False, None))


def test_frames_are_grabbed_at_camera_rate_but_decoded_at_most_decode_fps():
    cam = GrabCam(fps=200)
    fb = FrameBuffer(cam, decode_fps=20)
    try:
        time.sleep(1.0)
        grabbed, decoded = cam.n, cam.decodes
    finally:
        fb.stop()
    assert grabbed > 100                            # the driver is drained at camera rate
    assert 12 <= decoded <= 24                      # but only ~20 frames a second are decoded
    assert fb.latest() is not None


def test_the_newest_decoded_frame_is_fresh():
    cam = GrabCam(fps=200)
    fb = FrameBuffer(cam, decode_fps=20)
    try:
        time.sleep(0.5)
        f = fb.latest()
        assert f is not None and time.monotonic() - f.t < 0.1     # at most one decode period old
        assert (cam.n - int(f.img[0, 0, 0])) % 256 < 12        # decoded from a grab ~one period ago at most
    finally:
        fb.stop()


def test_without_decode_fps_every_frame_is_decoded_as_before():
    cam = GrabCam(fps=200)
    fb = FrameBuffer(cam)
    try:
        time.sleep(0.3)
    finally:
        fb.stop()
    assert cam.decodes >= cam.n - 1


def test_a_source_without_grab_still_works_with_decode_fps():
    fb = FrameBuffer(FakeCam(fps=100), decode_fps=10)
    try:
        assert wait_for(lambda: fb.latest() is not None)
    finally:
        fb.stop()


def test_skipped_decodes_are_not_failures_and_a_raising_decode_is_survived():
    """Merged room/detector's crash guard around capture-decode's _next(): a grabbed-not-decoded frame is
    neither a failure nor a reason to reopen; a retrieve() that raises counts as one failed read and the
    thread carries on."""
    cam = GrabCam(fps=200)
    fb = FrameBuffer(cam, decode_fps=20)
    try:
        time.sleep(0.6)
        assert cam.n > 60 and fb.failures == 0 and fb.reconnects == 0 and fb.age() < 0.5
        real, calls = cam.retrieve, []

        def flaky():
            calls.append(1)
            if len(calls) % 3 == 0:
                raise cv2.error("corrupt frame")
            return real()
        cam.retrieve = flaky
        assert wait_for(lambda: len(calls) >= 9)
        assert fb._thread.is_alive() and 1 <= fb.failures <= len(calls) and fb.age() < 0.5
    finally:
        fb.stop()
