"""Connectivity monitor (spec V9 / 3.11). A daemon thread HEADs cfg net.check_host every
cfg net.interval_s; readers just look at `.online`, which never blocks. One good probe means online
at once; offline takes FAILS_TO_OFFLINE failed probes in a row, so one slow HEAD on busy venue Wi-Fi
doesn't flip every answer to the offline templates for a whole interval."""
from __future__ import annotations

import logging
import os
import threading
import time
from typing import Any, Callable, Iterable, Optional

import requests

log = logging.getLogger(__name__)

Callback = Callable[[bool], None]

TIMEOUT_S = 1.5             # per HEAD; 0.3 s timed out on ordinary venue Wi-Fi and flapped
FAILS_TO_OFFLINE = 2        # failed probes in a row before the state goes offline


def call_with_deadline(fn: Callable[..., Any], timeout_s: float, *args, name: str = "deadline", **kwargs) -> Any:
    """fn(*args, **kwargs) on a daemon thread; its result, or TimeoutError once timeout_s of wall time
    has passed. requests' timeout is per phase (connect, then each read) and doesn't cover DNS, so on
    flaky Wi-Fi a "1.5 s" call can take much longer; this bounds the whole call. fn's own exception is
    re-raised here. A call past its deadline keeps running in the background and its result is dropped."""
    box: dict = {}

    def work() -> None:
        try:
            box["v"] = fn(*args, **kwargs)
        except BaseException as ex:          # noqa: BLE001 - handed to the caller
            box["e"] = ex

    th = threading.Thread(target=work, name=name, daemon=True)
    th.start()
    th.join(max(0.0, float(timeout_s)))
    if th.is_alive():
        raise TimeoutError(f"{name}: no result after {timeout_s:.1f} s")
    if "e" in box:
        raise box["e"]
    return box.get("v")


class NetMonitor:
    def __init__(self, cfg: dict, initial: bool = False):
        net = (cfg or {}).get("net") or {}
        self.host: str = net.get("check_host", "https://api.x.ai")
        self.interval_s: float = float(net.get("interval_s", 5))
        self.timeout_s: float = float(net.get("timeout_s", TIMEOUT_S))
        self.fails_to_offline: int = FAILS_TO_OFFLINE
        self._online = bool(initial)
        self._fails = 0                 # failed probes in a row
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
        """Probe, update state, fire callbacks on a change. Returns the new state. Online after one
        good probe; offline only after fails_to_offline failed ones in a row."""
        ok = self.probe()
        with self._lock:
            self._fails = 0 if ok else self._fails + 1
            now = ok or (self._online and self._fails < self.fails_to_offline)
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
