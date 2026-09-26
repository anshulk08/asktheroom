"""Connectivity monitor (spec V9 / 3.11). A daemon thread HEADs cfg net.check_host every
cfg net.interval_s; readers just look at `.online`, which never blocks."""
from __future__ import annotations

import logging
import os
import threading
import time
from typing import Callable, Iterable, Optional

import requests

log = logging.getLogger(__name__)

Callback = Callable[[bool], None]


class NetMonitor:
    def __init__(self, cfg: dict, initial: bool = False):
        net = (cfg or {}).get("net") or {}
        self.host: str = net.get("check_host", "https://api.x.ai")
        self.interval_s: float = float(net.get("interval_s", 5))
        self.timeout_s: float = float(net.get("timeout_s", 0.3))
        self._online = bool(initial)
        self._callbacks: list[Callback] = []
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    @property
    def online(self) -> bool:
        """Last observed state. Starts False (assume offline until the first check succeeds)."""
        return self._online

    def on_change(self, callback: Callback) -> None:
        """Call callback(online) from the monitor thread whenever the state flips."""
        with self._lock:
            self._callbacks.append(callback)

    def probe(self) -> bool:
        """One HTTPS HEAD. Any HTTP response counts as online; any exception as offline."""
        try:
            requests.head(self.host, timeout=self.timeout_s, allow_redirects=False)
            return True
        except Exception:
            return False

    def check_once(self) -> bool:
        """Probe, update state, fire callbacks on a change. Returns the new state."""
        now = self.probe()
        with self._lock:
            changed = now != self._online
            self._online = now
            cbs = list(self._callbacks) if changed else []
        if changed:
            log.info("network %s", "online" if now else "offline")
        for cb in cbs:
            try:
                cb(now)
            except Exception:
                log.exception("net on_change callback failed")
        return now

    def _run(self) -> None:
        while not self._stop.is_set():
            self.check_once()
            self._stop.wait(self.interval_s)

    def start(self) -> "NetMonitor":
        if self._thread is not None and self._thread.is_alive():
            return self
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="NetMonitor", daemon=True)
        self._thread.start()
        return self

    def stop(self, timeout: float = 1.0) -> None:
        """Signal the thread and join; returns within about one probe timeout."""
        self._stop.set()
        t = self._thread
        if t is not None:
            t.join(timeout)
        self._thread = None


CLOCK_FLOOR = 1790294400.0      # 2026-09-25 00:00 UTC: the rig never ran before HackGT 13


def clock_behind(files: Iterable[str], now: Optional[float] = None, slack_s: float = 60.0) -> Optional[float]:
    """Seconds the wall clock is behind the newest of `files`' mtimes (or CLOCK_FLOOR), or None when it
    looks right. A Jetson with no RTC battery and no internet boots with a stale clock, and NTP only
    fixes it once online: until then every time the rig speaks ('at 3:14') and logs to n8n is off."""
    now = time.time() if now is None else now
    floor = CLOCK_FLOOR
    for f in files:
        try:
            floor = max(floor, os.path.getmtime(f))
        except (OSError, TypeError):
            pass
    return floor - now if now < floor - slack_s else None
