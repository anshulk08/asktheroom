"""Camera and recording sources (spec 3.1). Owner: P.

FrameBuffer keeps the newest frame plus a ring of the last ring_s seconds, filled by a background
thread; VideoFileSource has the same read API over a recording, so replays and tests use the same
code as the live camera.

Camera settings measured on the icSpring overhead camera (Jetson, V4L2): MJPG is the only format that
does 30 fps at 1280x720; CAP_PROP_BUFFERSIZE=1 halves it to 15 fps (the driver misses every other frame
while the one buffer is out), 2 keeps 30. Staleness is avoided by draining continuously in the thread,
not by starving the driver of buffers. Run scripts/camera_setup.sh first: auto exposure drops this
camera to 7.5-15 fps indoors.
"""
from __future__ import annotations

import bisect
import json
import logging
import os
import sys
import threading
import time
from collections import deque
from pathlib import Path
from typing import Iterator, Optional, Union

import cv2
import numpy as np

from core.types import Frame

log = logging.getLogger(__name__)

W, H, FPS = 1280, 720, 30


def open_camera(index: Union[int, str] = 0, width: int = W, height: int = H, fps: int = FPS,
                buffers: int = 2) -> cv2.VideoCapture:
    """Open a camera with the settings above. V4L2 on Linux; the platform default elsewhere (Mac dev)."""
    backend = cv2.CAP_V4L2 if sys.platform.startswith("linux") else cv2.CAP_ANY
    cap = cv2.VideoCapture(index, backend)
    if not cap.isOpened():
        raise RuntimeError(f"can't open camera {index!r}")
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, buffers)
    got = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    if got != (width, height):
        log.warning("camera %r gives %dx%d, not %dx%d", index, *got, width, height)
    return cap


class FrameBuffer:
    """Live source: a thread reads continuously; latest() is the newest frame, at(t) the nearest in the ring.

    `source` is a camera index or device path (opened with open_camera), or anything with read() -> (ok, img)
    and release(), which is how tests feed it.

    A camera (index or path) that stops giving frames for reopen_after_s, e.g. dropped off USB, is released
    and reopened every retry_s; once it is back, the controls it had when first opened (after
    scripts/camera_setup.sh) are written back, since a replugged UVC camera comes back with auto exposure
    and autofocus on. Open it by its /dev/v4l/by-id/ path: an index can change when it re-enumerates. In
    the app's container only the /dev/video* nodes present at start exist: if the camera returns as a new
    node, the log says so and the app must be restarted.
    """

    def __init__(self, source: Union[int, str, object] = 0, ring_s: float = 10.0, name: str = "capture",
                 opener=None, controls=None, reopen_after_s: float = 1.0, retry_s: float = 1.0,
                 decode_fps: Optional[float] = None):
        self._device = source if isinstance(source, (int, str)) else None
        self._opener = opener or open_camera
        if controls is None:
            from core import v4l2ctl as controls
        self._controls = controls
        self.reopen_after_s, self.retry_s = reopen_after_s, retry_s
        self.decode_period = 1.0 / decode_fps if decode_fps else 0.0
        self._decoded_t = float("-inf")
        self.grabbed = 0                         # frames taken off the camera; failures + frames decoded
        self.reconnects = 0
        self.cap = self._opener(source) if self._device is not None else source
        self._snapshot = self._controls.snapshot(self._ctl_path()) if self._device is not None else []
        self.ring_s = ring_s
        self._ring: deque[Frame] = deque()
        self._lock = threading.Lock()
        self._fresh = threading.Condition(self._lock)
        self._stop = threading.Event()
        self._idx = 0
        self._times: deque[float] = deque(maxlen=60)     # read times, for fps
        self.failures = 0
        self._thread = threading.Thread(target=self._run, name=name, daemon=True)
        self._thread.start()

    def _ctl_path(self) -> Optional[str]:
        from core.v4l2ctl import device_path
        return device_path(self._device)

    def _reopen(self) -> None:
        """Release the dead camera and open it again every retry_s until it is back (or stop()), then
        restore its controls."""
        log.warning("camera %r gave no frames for %.1f s (unplugged?): reopening", self._device, self.reopen_after_s)
        try:
            self.cap.release()
        except Exception:
            log.debug("release of the dead camera failed", exc_info=True)
        tries = 0
        while not self._stop.is_set():
            tries += 1
            try:
                self.cap = self._opener(self._device)
                break
            except Exception as ex:
                if tries % 10 == 1:
                    path = self._ctl_path()
                    target = os.path.realpath(path) if path else None
                    hint = (f"; it now points at {target}, which this container may not have: restart the app"
                            if target and path and os.path.exists(path) and not os.path.exists(target) else "")
                    log.error("camera %r not back yet (%s)%s", self._device, ex, hint)
                self._stop.wait(self.retry_s)
        else:
            return
        n = self._controls.restore(self._ctl_path(), self._snapshot)
        self.reconnects += 1
        log.warning("camera %r is back (reconnect %d); %d of %d controls restored",
                    self._device, self.reconnects, n, len(self._snapshot))

    def _run(self) -> None:
        """The capture thread. Nothing may end it: perception would wait on it forever while the
        dashboard looks alive. A read that raises (a cv2.error mid-stream) counts as a failed read."""
        while not self._stop.is_set():
            try:
                self._read_loop()
            except Exception:
                log.exception("capture thread error; carrying on")
                self._stop.wait(0.1)

    def _read_loop(self) -> None:
        dead_since = None
        while not self._stop.is_set():
            try:
                ok, img = self._next()          # (True, None): grabbed, not decoded (decode_fps)
            except Exception as ex:
                ok, img = False, None
                if self.failures % 30 == 0:
                    log.warning("camera read raised %s: %s", type(ex).__name__, ex)
            now, wall = time.monotonic(), time.time()
            if ok and img is None:              # grabbed, not decoded: a frame nobody would use
                dead_since = None
                continue
            if not ok or img is None:
                self.failures += 1
                if self.failures % 30 == 1:
                    log.warning("camera read failed (%d so far)", self.failures)
                dead_since = dead_since or now
                if self._device is not None and now - dead_since >= self.reopen_after_s:
                    self._reopen()
                    dead_since = None
                    continue
                self._stop.wait(0.01)
                continue
            dead_since = None
            with self._lock:
                self._idx += 1
                self._ring.append(Frame(t=now, wall=wall, img=img, idx=self._idx))
                while self._ring and now - self._ring[0].t > self.ring_s:
                    self._ring.popleft()
                self._times.append(now)
                self._fresh.notify_all()

    def _next(self) -> tuple[bool, Optional[np.ndarray]]:
        """The next camera frame. With decode_period, every frame is grabbed (the driver never backs up)
        but only one per period is decoded (retrieve: the MJPEG decode, the costly part at 1440p/1080p);
        the others come back as (True, None). Sources without grab/retrieve (tests, recordings) read()."""
        grab = getattr(self.cap, "grab", None)
        if not self.decode_period or grab is None or not hasattr(self.cap, "retrieve"):
            return self.cap.read()
        if not grab():
            return False, None
        self.grabbed += 1
        now = time.monotonic()
        if now - self._decoded_t < self.decode_period:
            return True, None
        self._decoded_t = now
        return self.cap.retrieve()

    def latest(self) -> Optional[Frame]:
        with self._lock:
            return self._ring[-1] if self._ring else None

    def age(self) -> Optional[float]:
        """Seconds since the newest frame arrived, or None before the first."""
        with self._lock:
            return time.monotonic() - self._ring[-1].t if self._ring else None

    def wait_new(self, after_idx: int, timeout: float = 1.0) -> Optional[Frame]:
        """Block until a frame newer than after_idx arrives (None on timeout)."""
        deadline = time.monotonic() + timeout
        with self._lock:
            while not (self._ring and self._ring[-1].idx > after_idx):
                left = deadline - time.monotonic()
                if left <= 0 or self._stop.is_set():
                    return None
                self._fresh.wait(left)
            return self._ring[-1]

    def at(self, t: float) -> Optional[Frame]:
        """The ring frame whose timestamp is nearest t (monotonic), or None if the ring is empty."""
        with self._lock:
            if not self._ring:
                return None
            ts = [f.t for f in self._ring]
            i = bisect.bisect_left(ts, t)
            cands = [self._ring[j] for j in (i - 1, i) if 0 <= j < len(ts)]
            return min(cands, key=lambda f: abs(f.t - t))

    @property
    def fps(self) -> float:
        with self._lock:
            if len(self._times) < 2:
                return 0.0
            return (len(self._times) - 1) / (self._times[-1] - self._times[0])

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2)
        self.cap.release()


def load_timestamps(video: Union[str, Path]) -> Optional[list[float]]:
    """timestamps.json beside a recorded video (eval/record.py writes one), else None."""
    p = Path(video).with_name("timestamps.json")
    return json.loads(p.read_text()) if p.exists() else None


class VideoFileSource(FrameBuffer):
    """A recording with FrameBuffer's API. realtime=True paces frames by their timestamps; False plays
    as fast as it can. Iterate it (for frame in src.frames()) to process every frame in order."""

    def __init__(self, path: Union[str, Path], realtime: bool = True, loop: bool = False,
                 ring_s: float = 10.0, start: bool = True):
        self.path = str(path)
        self.realtime, self.loop = realtime, loop
        self.timestamps = load_timestamps(path)
        cap = cv2.VideoCapture(self.path)
        if not cap.isOpened():
            raise RuntimeError(f"can't open video {self.path}")
        self.video_fps = cap.get(cv2.CAP_PROP_FPS) or FPS
        cap.release()
        self.done = threading.Event()
        if start:
            super().__init__(_Unused(), ring_s=ring_s, name="video-source")
        else:                                   # iterate-only: no thread
            self.cap, self._stop = _Unused(), threading.Event()

    def _t(self, i: int) -> float:
        return self.timestamps[i] if self.timestamps and i < len(self.timestamps) else i / self.video_fps

    def frames(self) -> Iterator[Frame]:
        """Every frame in order, no pacing. t starts at 0 (video time); wall = now + t."""
        cap = cv2.VideoCapture(self.path)
        wall0, i = time.time(), 0
        try:
            while True:
                ok, img = cap.read()
                if not ok:
                    return
                t = self._t(i)
                yield Frame(t=t, wall=wall0 + t, img=img, idx=i)
                i += 1
        finally:
            cap.release()

    def _run(self) -> None:
        while not self._stop.is_set():
            t0 = time.monotonic()
            for f in self.frames():
                if self._stop.is_set():
                    return
                if self.realtime:
                    self._stop.wait(max(0.0, t0 + f.t - time.monotonic()))
                now = time.monotonic()
                with self._lock:
                    self._idx += 1
                    self._ring.append(Frame(t=now, wall=time.time(), img=f.img, idx=self._idx))
                    while self._ring and now - self._ring[0].t > self.ring_s:
                        self._ring.popleft()
                    self._times.append(now)
                    self._fresh.notify_all()
            if not self.loop:
                self.done.set()
                return


class _Unused:
    def read(self):
        return False, None

    def release(self) -> None:
        pass


def main(argv=None) -> None:
    """python -m core.capture [--device 0] [--seconds 10]: report fps and frame age (task P4 check)."""
    import argparse
    ap = argparse.ArgumentParser(description=main.__doc__)
    ap.add_argument("--device", default="0")
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--save", help="write the last frame here")
    ap.add_argument("--size", default=f"{W}x{H}", help="capture size, e.g. 2560x1440")
    ap.add_argument("--decode-fps", type=float, help="grab every frame, decode only this many a second")
    a = ap.parse_args(argv)
    dev = int(a.device) if a.device.isdigit() else a.device
    w, h = (int(v) for v in a.size.lower().split("x"))
    t_cpu = time.process_time()
    fb = FrameBuffer(dev, opener=lambda src: open_camera(src, w, h), decode_fps=a.decode_fps)
    ages, last_idx, t_end = [], 0, time.monotonic() + a.seconds
    while time.monotonic() < t_end:
        time.sleep(0.013)                        # sample at an unrelated rate
        f = fb.latest()
        if f is not None and f.idx > 30:         # skip warm-up
            ages.append(time.monotonic() - f.t)
            last_idx = f.idx
    cpu = time.process_time() - t_cpu
    print(f"fps {fb.fps:.1f} decoded ({fb.grabbed / a.seconds:.1f} grabbed/s); frames {last_idx}; read failures "
          f"{fb.failures}; latest() age median {1000 * float(np.median(ages)):.0f} ms, max {1000 * max(ages):.0f} ms; "
          f"CPU {100 * cpu / a.seconds:.0f}% of one core")
    if a.save and fb.latest() is not None:
        cv2.imwrite(a.save, fb.latest().img)
    fb.stop()


if __name__ == "__main__":
    main()
