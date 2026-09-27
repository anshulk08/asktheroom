import threading
import time

import requests

import net
from net import NetMonitor

CFG = {"net": {"check_host": "https://example.invalid", "interval_s": 0.02, "timeout_s": 0.3}}


class FakeHead:
    """Stands in for requests.head; `up` controls success."""

    def __init__(self, up: bool = True):
        self.up = up
        self.calls = 0

    def __call__(self, url, timeout=None, allow_redirects=None):
        self.calls += 1
        assert timeout == 0.3
        if not self.up:
            raise requests.ConnectionError("down")
        return object()


def test_flips_state_and_callbacks_once_per_change(monkeypatch):
    head = FakeHead(up=True)
    monkeypatch.setattr(net.requests, "head", head)
    m = NetMonitor(CFG)
    seen = []
    m.on_change(seen.append)
    assert m.online is False

    assert m.check_once() is True and m.online is True
    m.check_once()
    m.check_once()
    head.up = False
    assert m.check_once() is True and m.online is True      # one failed probe is not enough
    assert m.check_once() is False and m.online is False
    m.check_once()
    head.up = True
    m.check_once()
    assert seen == [True, False, True]


def test_one_slow_probe_does_not_flap_offline(monkeypatch):
    head = FakeHead(up=True)
    monkeypatch.setattr(net.requests, "head", head)
    m = NetMonitor(CFG, initial=True)
    seen = []
    m.on_change(seen.append)
    for up in (False, True, False, True, False, True):      # isolated failures between good probes
        head.up = up
        m.check_once()
        assert m.online is True
    assert seen == []
    head.up = False
    m.check_once()
    m.check_once()
    assert m.online is False and seen == [False]
    head.up = True
    assert m.check_once() is True and seen == [False, True]  # one good probe: online at once


def test_starting_offline_stays_offline_on_failures(monkeypatch):
    monkeypatch.setattr(net.requests, "head", FakeHead(up=False))
    m = NetMonitor(CFG)
    seen = []
    m.on_change(seen.append)
    assert m.check_once() is False and m.check_once() is False and seen == []


def test_probe_timeout_default_and_config():
    from core.config import load_config
    assert NetMonitor({}).timeout_s == 1.5
    assert NetMonitor(load_config()).timeout_s == 1.5


def test_call_with_deadline_returns_raises_and_times_out():
    assert net.call_with_deadline(lambda a, b=0: a + b, 1.0, 2, b=3) == 5

    def boom():
        raise ValueError("bad")

    try:
        net.call_with_deadline(boom, 1.0)
        raise AssertionError("expected ValueError")
    except ValueError as ex:
        assert str(ex) == "bad"
    release = threading.Event()
    t0 = time.perf_counter()
    try:
        net.call_with_deadline(release.wait, 0.1, 5, name="slow")
        raise AssertionError("expected TimeoutError")
    except TimeoutError as ex:
        assert "slow" in str(ex)
    assert time.perf_counter() - t0 < 0.5
    release.set()


def test_background_thread_tracks_state(monkeypatch):
    head = FakeHead(up=True)
    monkeypatch.setattr(net.requests, "head", head)
    m = NetMonitor(CFG)
    flipped = threading.Event()
    m.on_change(lambda on: flipped.set() if not on else None)
    m.start()
    try:
        deadline = time.time() + 2
        while not m.online and time.time() < deadline:
            time.sleep(0.005)
        assert m.online
        head.up = False
        assert flipped.wait(2)
        assert m.online is False
    finally:
        m.stop()


def test_callback_exception_does_not_kill_monitor(monkeypatch):
    monkeypatch.setattr(net.requests, "head", FakeHead(up=True))
    m = NetMonitor(CFG)

    def bad(_):
        raise RuntimeError("boom")

    m.on_change(bad)
    assert m.check_once() is True


def test_online_never_blocks_and_stop_joins_quickly(monkeypatch):
    def slow_head(url, timeout=None, allow_redirects=None):
        time.sleep(0.2)
        return object()

    monkeypatch.setattr(net.requests, "head", slow_head)
    m = NetMonitor({"net": {"check_host": "https://x", "interval_s": 5, "timeout_s": 0.3}}).start()
    t0 = time.perf_counter()
    _ = m.online
    assert time.perf_counter() - t0 < 0.01
    time.sleep(0.25)            # now waiting out the 5 s interval
    t0 = time.perf_counter()
    m.stop()
    assert time.perf_counter() - t0 < 0.1
    assert m._thread is None


def test_clock_behind_the_newest_saved_file_is_flagged(tmp_path):
    import os
    f = tmp_path / "events.db"
    f.write_text("")
    os.utime(f, (net.CLOCK_FLOOR + 7200, net.CLOCK_FLOOR + 7200))
    assert net.clock_behind([str(f)], now=net.CLOCK_FLOOR + 3600) == 3600
    assert net.clock_behind([str(f)], now=net.CLOCK_FLOOR + 7200 + 5) is None
    assert net.clock_behind([str(f)], now=net.CLOCK_FLOOR + 7200 - 30) is None     # within slack


def test_clock_before_the_hackathon_is_flagged_even_with_no_files():
    assert net.clock_behind(["", "/no/such/file"], now=0.0) == net.CLOCK_FLOOR
    assert net.clock_behind([], now=net.CLOCK_FLOOR + 1) is None
