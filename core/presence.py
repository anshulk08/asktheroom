"""Presence debounce windows for the world model (rule 1: present k of n, absent at most absent_max of n)."""
from __future__ import annotations

from collections import deque
from typing import Iterable, Optional

SNAP = 0.25             # an update this close to one reference period counts exactly one (camera jitter);
                        # from there to 2 x SNAP the weight ramps to the true time: no jump anywhere
MAX_W = 2.0             # no update counts for more than this: after a slow step one sample never decides


def weight(periods: float) -> float:
    """An update's weight from the time since the one before, in reference periods: 1 within SNAP of one
    period, the true time beyond 2 x SNAP, a straight ramp between (continuous: no counting-mode jump)."""
    d = periods - 1.0
    a = abs(d)
    if a <= SNAP:
        return 1.0
    return 1.0 + (d if a >= 2 * SNAP else 2.0 * (a - SNAP) * (1 if d > 0 else -1))


class Presence(deque):
    """Presence bits, newest last (True: detected in that update); any(), sum() and slicing work as on a
    deque. Frame-counted (the last n updates) when hz is None. Otherwise each update weighs the time since
    the one before, in periods of an hz loop (within SNAP of one period: exactly 1; at most MAX_W), the
    window is the newest updates weighing n, and hits() is the weight of the detections in it. At hz that
    is exactly the last n updates; at 7 fps each update weighs ~2 and k of n takes about the same time as
    at 15 fps; below 7.5 fps the cap makes it count updates again (never fewer than n / MAX_W of them). A
    gap longer than the whole window (a stalled loop) is no evidence either way: it weighs 1, drops nothing."""

    def __init__(self, bits: Iterable[bool] = (), n: int = 10, hz: Optional[float] = None):
        super().__init__(bits, maxlen=None if hz else n)
        self.n = n
        self.unit = 1.0 / hz if hz else None
        self._w: deque = deque([1.0] * len(self))
        self._last: Optional[float] = None

    def push(self, t: float, bit: bool) -> None:
        if self.unit is None:
            self.append(bit)
            return
        gap = None if self._last is None else t - self._last
        if gap is not None and gap < -self.n * self.unit:      # the clock went back (a new clip): start over
            self.clear()
            gap = None
        self._last = t
        if gap is None or gap < 0 or gap > self.n * self.unit:
            w = 1.0
        else:
            w = min(MAX_W, weight(gap / self.unit))
        self.append(bit)
        self._w.append(w)
        total = sum(self._w)
        while len(self) > 1 and total - self._w[0] >= self.n - 1e-9:
            total -= self._w.popleft()
            self.popleft()

    def clear(self) -> None:
        """Forget every update (a debounce restart, e.g. core/room_world.py), weights included."""
        super().clear()
        self._w.clear()
        self._last = None

    def hits(self) -> float:
        """Detections in the window, in updates of the reference rate (the plain count without hz)."""
        if self.unit is None:
            return float(sum(self))
        return sum(w for b, w in zip(self, self._w) if b)
