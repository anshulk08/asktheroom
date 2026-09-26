"""Acceptance test for the always-on mic (spec 0002): play a recording of hall noise and chatter
through the same path main.py uses (Silero VAD -> Whisper -> voice.understand overheard filter)
and count what the rig would have answered. Every answer to a recording with no questions for the
rig is a false trigger. Target: 0-1 in 10 minutes.

    python scripts/overheard_test.py hall.wav              # 16 kHz mono (other rates are resampled)
    python scripts/overheard_test.py hall.wav --mode wake  # only "room, ..." counts
    python scripts/overheard_test.py hall.wav --show       # also print what was said (local only)

Needs llama-server for Qwen (scripts/qwen_server.sh); without it the rules decide. The recording is
yours: keep it out of git (tests/stt_audio/ is ignored).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import voice.stt as stt_mod  # noqa: E402
from core.config import load_config  # noqa: E402
from voice.understand import IGNORE, Understander  # noqa: E402


class FileInput:
    """open_input() stand-in: serves the recording block by block, as fast as the VAD reads it."""

    def __init__(self, audio: np.ndarray, block: int):
        self.audio, self.block, self.pos = audio, block, 0

    def read(self, timeout: float = 1.0):
        if self.pos + self.block > len(self.audio):
            return None
        b = self.audio[self.pos:self.pos + self.block]
        self.pos += self.block
        return b

    def close(self) -> None:
        pass


def load(path: str) -> np.ndarray:
    import wave
    with wave.open(path, "rb") as w:
        rate, ch = w.getframerate(), w.getnchannels()
        a = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768
    if ch > 1:
        a = a.reshape(-1, ch).mean(axis=1)
    if rate != stt_mod.RATE:
        n = int(len(a) * stt_mod.RATE / rate)
        a = np.interp(np.linspace(0, len(a) - 1, n), np.arange(len(a)), a).astype(np.float32)
    return a


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("wav")
    ap.add_argument("--mode", choices=["always", "wake"], default=None, help="default: listen.mode")
    ap.add_argument("--show", action="store_true", help="print transcripts (never saved)")
    a = ap.parse_args(argv)
    cfg = load_config()
    if a.mode:
        cfg["listen"] = dict(cfg.get("listen") or {}, mode=a.mode)
    audio = load(a.wav)
    src = FileInput(audio, stt_mod.BLOCK)
    stt_mod.open_input = lambda rate, block, device=None: src
    stt = stt_mod.STT(cfg)
    stt.log_text = a.show
    u = Understander(cfg)
    u.warm()
    spoken = answered = 0
    t0 = time.monotonic()
    while src.pos + stt_mod.BLOCK <= len(audio):
        text = stt.hear(idle_s=1e9)
        if not text:
            continue
        spoken += 1
        i = u(text, overheard=True)
        if i.kind != IGNORE:
            answered += 1
            at = src.pos / stt_mod.RATE
            print(f"  {at:7.1f} s  ANSWERED {i.kind} {i.obj or ''} ({u.last_by})"
                  + (f": {text!r}" if a.show else ""))
        elif a.show:
            print(f"  ignored ({u.last_by}): {text!r}")
    mins = len(audio) / stt_mod.RATE / 60
    print(f"{mins:.1f} min of audio, {spoken} utterances transcribed, {answered} answered "
          f"({answered / max(mins, 1e-9) * 10:.1f} per 10 min), took {time.monotonic() - t0:.0f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
