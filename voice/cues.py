"""Listening cues: a chime that says "speak now" and a listening light (spec: talk to the rig hands-free).

When the rig starts listening for a question (the wake word on its own, "Hey Room!", or a clicker press)
it plays the listen chime and turns the indicator on until the question has been heard. A question said
in one breath with the wake word ("Room, where are my keys?") gets the shorter ack tone instead, the
moment it is accepted.

Chimes are synthesized here (no sound files) and played on the speech output (TTS.play_cue), so they
come out of the same speaker as the answers. A Bluetooth speaker that has been idle clips the first
~0.2 s of sound it gets, so each chime starts with LEAD_S of silence; A2DP also plays ~0.2 s late, so
the mic waits TAIL_S after a chime before it records, or it would hear the chime.

A Bluetooth speaker switches itself off after ~20 min without sound (the Bose SoundLink Micro did twice on
the rig, Sat 26 Sep; the rig then spoke into the analog sink). keepalive_pcm is a near-inaudible puff of
low-passed noise (tts.keepalive_dbfs, about -50 dBFS RMS, 0.3 s) that TTS.keepalive plays when nothing has
played for tts.keepalive_s; digital silence may not count as sound to the speaker.

The indicator is software with a hardware slot. listen.indicator.backend:
  none  (default) log only: the rig has no LEDs yet
  gpio  an LED on a GPIO line (libgpiod's python bindings, `gpiod`): gpio_chip (e.g. gpiochip0) and
        gpio_line (the line offset of the 40-pin header pin), active_low for an LED wired to 3.3 V.
A GPIO that can't be opened is logged once and the indicator carries on as 'none': a missing light must
never cost an answer.
"""
from __future__ import annotations

import logging
import threading
from typing import Optional

import numpy as np

log = logging.getLogger(__name__)

RATE = 24000
LEAD_S = 0.25           # silence first: an idle Bluetooth speaker clips the start of what it plays
TAIL_S = 0.25           # after a chime, before the mic records (A2DP plays ~0.2 s late)
FADE_S = 0.008          # raised-cosine edges, so the tones don't click

# (frequency Hz, seconds) per note: listen rises (a question is wanted), ack is one short soft note
CHIMES = {
    "listen": ((880.0, 0.09), (1318.5, 0.14)),
    "ack": ((1046.5, 0.07),),
}
LEVEL = {"listen": 0.35, "ack": 0.2}


def chime_pcm(kind: str = "listen", rate: int = RATE, lead_s: float = LEAD_S) -> bytes:
    """int16 mono PCM of a chime: lead_s of silence, then its notes with faded edges."""
    parts = [np.zeros(int(lead_s * rate), np.float32)]
    for freq, dur in CHIMES[kind]:
        n = int(dur * rate)
        t = np.arange(n) / rate
        tone = np.sin(2 * np.pi * freq * t) * (0.7 + 0.3 * np.exp(-t * 18))     # a soft bell-like decay
        f = min(int(FADE_S * rate), n // 2)
        if f:
            ramp = 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, f))
            tone[:f] *= ramp
            tone[-f:] *= ramp[::-1]
        parts.append(tone.astype(np.float32))
    pcm = np.concatenate(parts) * LEVEL[kind]
    return (np.clip(pcm, -1, 1) * 32767).astype(np.int16).tobytes()


def keepalive_pcm(dbfs: float = -50.0, dur_s: float = 0.3, rate: int = RATE, seed: int = 0) -> bytes:
    """int16 mono PCM: dur_s of low-passed noise at dbfs RMS with faded edges (no click). Noise, not a tone:
    a small speaker plays a sub-audible tone as nothing, and may treat it as silence."""
    n = max(1, int(dur_s * rate))
    x = np.random.default_rng(seed).standard_normal(n)
    a = np.exp(-2 * np.pi * 400 / rate)                  # one-pole low-pass at ~400 Hz: a soft rumble, not hiss
    y = np.empty(n)
    acc = 0.0
    for i in range(n):
        acc = a * acc + (1 - a) * x[i]
        y[i] = acc
    f = min(int(0.05 * rate), n // 2)
    if f:
        ramp = 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, f))
        y[:f] *= ramp
        y[-f:] *= ramp[::-1]
    y *= 10 ** (dbfs / 20) / max(float(np.sqrt(np.mean(y ** 2))), 1e-12)
    return (np.clip(y, -1, 1) * 32767).astype(np.int16).tobytes()


class ListenIndicator:
    """on() while the rig listens for a question, off() after. Thread-safe; never raises."""

    def __init__(self, cfg: Optional[dict] = None):
        c = ((cfg or {}).get("listen") or {}).get("indicator") or {}
        self.backend = str(c.get("backend") or "none").lower()
        self.chip, self.line_no = c.get("gpio_chip", "gpiochip0"), c.get("gpio_line")
        self.active_low = bool(c.get("active_low", False))
        self.lit = False
        self._line = None
        self._lock = threading.Lock()
        if self.backend == "gpio":
            self._line = self._open_gpio()
            if self._line is None:
                self.backend = "none"

    def _open_gpio(self):
        if self.line_no is None:
            log.warning("listen.indicator: gpio needs gpio_line; the listening light is off")
            return None
        try:
            import gpiod
            chip = gpiod.Chip(str(self.chip))
            line = chip.get_line(int(self.line_no))
            line.request(consumer="askroom-listening", type=gpiod.LINE_REQ_DIR_OUT,
                         default_vals=[1 if self.active_low else 0])
            log.info("listening light on %s line %s", self.chip, self.line_no)
            return line
        except Exception as ex:
            log.warning("listen.indicator: %s line %s not usable (%s: %s); the listening light is off",
                        self.chip, self.line_no, type(ex).__name__, ex)
            return None

    def set(self, on: bool) -> None:
        with self._lock:
            if on == self.lit:
                return
            self.lit = on
            if self._line is not None:
                try:
                    self._line.set_value(int(on) ^ int(self.active_low))
                except Exception as ex:
                    log.warning("listening light failed (%s); turning it off for good", ex)
                    self._line = None
            log.debug("listening light %s", "on" if on else "off")

    def on(self) -> None:
        self.set(True)

    def off(self) -> None:
        self.set(False)

    def close(self) -> None:
        self.off()
        if self._line is not None:
            try:
                self._line.release()
            except Exception:
                pass
            self._line = None
