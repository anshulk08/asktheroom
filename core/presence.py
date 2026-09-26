"""Presence debounce windows for the world model (rule 1: present k of n, absent at most absent_max of n)."""
from __future__ import annotations

from collections import deque
from typing import Iterable, Optional

MAX_WEIGHT = 2.5        # one update never counts for more than this many units (a stalled loop)


class Presence(deque):
    """Presence bits, newest last (True: detected in that update); any(), sum() and slicing work as on a
    deque. Frame-counted (the last n updates) when hz is None. Otherwise it keeps the last n / hz seconds
    of updates, and hits() weighs each detection by the time since the update before it, in updates of
    an hz loop: at hz updates a second that is the plain count, and k of n takes the same time at 7 fps
    live as at 15 fps in a replay."""

    def __init__(self, bits: Iterable[bool] = (), n: int = 10, hz: Optional[float] = None):
        super().__init__(bits, maxlen=None if hz else n)
        self.n = n
        self.unit = 1.0 / hz if hz else 1.0
        self.window_s = n / hz if hz else None
        self._w: deque = deque([1.0] * len(self))
        self._t: Optional[deque] = None                  # update times, once the first push says the clock
        self._last: Optional[float] = None

    def push(self, t: float, bit: bool) -> None:
        if not self.window_s:
            self.append(bit)
            return
        if self._t is None:                              # bits seeded from a candidate: one unit apart
            self._t = deque(t - self.unit * (len(self) - i) for i in range(len(self)))
            self._last = self._t[-1] if self._t else None
        w = 1.0 if self._last is None else min(max(t - self._last, 0.0) / self.unit, MAX_WEIGHT)
        self._last = t
        self.append(bit)
        self._w.append(w)
        self._t.append(t)
        while self._t and t - self._t[0] >= self.window_s - 1e-9:
            self.popleft()
            self._w.popleft()
            self._t.popleft()

    def hits(self) -> float:
        """Detections in the window, in updates at the nominal n / window_s rate."""
        if not self.window_s:
            return float(sum(self))
        return sum(w for b, w in zip(self, self._w) if b)
