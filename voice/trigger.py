"""Push-to-talk trigger (spec 3.9 / V1): a presentation clicker, read with evdev.

Presentation clickers are USB HID keyboards that send PageDown / PageUp. On Linux we read them
with python-evdev (https://python-evdev.readthedocs.io): list_devices(), InputDevice.capabilities()
to find one that has KEY_PAGEDOWN or KEY_PAGEUP, then read_loop() / select() for key-down events
(value 1). grab() keeps the presses from also scrolling whatever window has focus.

evdev is Linux-only. On a laptop, or when no clicker is plugged in, Clicker falls back to the
keyboard: Enter on stdin counts as a press, so the whole loop is testable anywhere.
"""
from __future__ import annotations

import logging
import queue
import sys
import threading
from typing import Optional

log = logging.getLogger(__name__)

KEYS = ("KEY_PAGEDOWN", "KEY_PAGEUP")


def find_clicker(name_hint: str = ""):
    """The first evdev device that can send PageDown or PageUp (name_hint narrows it), else None."""
    try:
        import evdev
        from evdev import ecodes
    except ImportError:
        return None
    want = {ecodes.ecodes[k] for k in KEYS}
    for path in evdev.list_devices():
        try:
            dev = evdev.InputDevice(path)
        except OSError:
            continue
        keys = set(dev.capabilities().get(ecodes.EV_KEY, []))
        if keys & want and name_hint.lower() in dev.name.lower():
            return dev
        dev.close()
    return None


class Clicker:
    """wait_press() blocks until the clicker (or Enter) is pressed. Presses are queued by a reader
    thread, so pressed() can also poll for a second press while recording."""

    def __init__(self, cfg: Optional[dict] = None, keyboard: Optional[bool] = None):
        """keyboard: True forces the Enter fallback; False requires a clicker; None tries the clicker
        first. cfg main.clicker_name narrows which evdev device counts."""
        m = (cfg or {}).get("main") or {}
        self._q: "queue.Queue[str]" = queue.Queue()
        self._stop = threading.Event()
        self.dev = None
        if not keyboard:
            self.dev = find_clicker(str(m.get("clicker_name", "")))
            if self.dev is None and keyboard is False:
                raise RuntimeError("no presentation clicker found (need a device with PageDown/PageUp)")
        if self.dev is not None:
            try:
                self.dev.grab()
            except OSError:
                log.warning("could not grab %s; presses will also reach other apps", self.dev.name)
            log.info("clicker: %s (%s)", self.dev.name, self.dev.path)
            target = self._read_evdev
        else:
            log.info("clicker: none found, press Enter to ask")
            target = self._read_stdin
        self.kind = "evdev" if self.dev is not None else "keyboard"
        threading.Thread(target=target, name="clicker", daemon=True).start()

    def _read_evdev(self) -> None:
        from evdev import ecodes
        want = {ecodes.ecodes[k] for k in KEYS}
        try:
            for ev in self.dev.read_loop():
                if self._stop.is_set():
                    return
                if ev.type == ecodes.EV_KEY and ev.code in want and ev.value == 1:
                    self._q.put(ecodes.KEY[ev.code])
        except OSError:
            log.exception("clicker disconnected")

    def _read_stdin(self) -> None:
        while not self._stop.is_set():
            try:
                line = sys.stdin.readline()
            except (OSError, ValueError):   # no usable stdin (pytest, a service)
                return
            if not line:            # stdin closed: stop quietly
                return
            self._q.put("ENTER")

    def wait_press(self, timeout: Optional[float] = None) -> bool:
        """Block until a press. Returns False on timeout or after close()."""
        while not self._stop.is_set():
            try:
                self._q.get(timeout=0.2 if timeout is None else min(0.2, timeout))
                return True
            except queue.Empty:
                if timeout is not None:
                    timeout -= 0.2
                    if timeout <= 0:
                        return False
        return False

    def pressed(self) -> bool:
        """Non-blocking: True (and consumes it) if a press arrived since the last check."""
        try:
            self._q.get_nowait()
            return True
        except queue.Empty:
            return False

    def clear(self) -> None:
        """Drop queued presses (e.g. extra clicks while an answer was playing)."""
        while self.pressed():
            pass

    def press(self) -> None:
        """Inject a press (tests, or a dashboard button)."""
        self._q.put("INJECTED")

    def close(self) -> None:
        self._stop.set()
        if self.dev is not None:
            try:
                self.dev.ungrab()
            except OSError:
                pass
            self.dev.close()
