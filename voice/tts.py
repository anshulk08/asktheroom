"""Speech out (spec V9): ElevenLabs streaming when online, Piper offline.

ElevenLabs (checked 2026-09-25): POST https://api.elevenlabs.io/v1/text-to-speech/{voice_id}/stream
with header xi-api-key, body {"text", "model_id"}, query output_format=pcm_22050 (raw signed 16-bit
little-endian mono, so no mp3 decoder). eleven_flash_v2_5 is the ~75 ms model.
  https://elevenlabs.io/docs/api-reference/text-to-speech/stream
  https://elevenlabs.io/docs/overview/models
Piper: piper-tts 1.8 `PiperVoice.load(model.onnx)`; `voice.synthesize(text)` yields one
AudioChunk per sentence (sample_rate, audio_int16_bytes). Voices live in models/piper/
(scripts/get_piper_voice.sh).

Audio goes through two seams tests can patch: `open_output(rate)` (a streaming int16 mono sink)
and `play_pcm(pcm, rate)` (whole buffer, blocking).
"""
from __future__ import annotations

import logging
import os
import queue
import threading
import time
from pathlib import Path
from typing import Iterator, Optional, Union

import numpy as np
import requests

from core.config import ROOT

log = logging.getLogger(__name__)

ELEVEN_URL = "https://api.elevenlabs.io/v1/text-to-speech/{voice_id}/stream"
ELEVEN_FORMAT = "pcm_22050"
ELEVEN_RATE = 22050
FIRST_BYTE_S = 1.5          # slower than this -> Piper for this utterance
CHUNK_BYTES = 2048          # ~46 ms at 22.05 kHz int16
PIPER_DIR = ROOT / "models" / "piper"


# ---------------------------------------------------------------- audio seam

class AudioOut:
    """Streaming int16 mono output on the default device (sounddevice/PortAudio)."""

    def __init__(self, rate: int):
        import sounddevice as sd
        self.stream = sd.RawOutputStream(samplerate=rate, channels=1, dtype="int16")
        self.stream.start()
        self._odd = b""

    def write(self, data: bytes) -> None:
        data = self._odd + data
        cut = len(data) - (len(data) % 2)
        self._odd = data[cut:]
        if cut:
            self.stream.write(data[:cut])      # blocks when the device buffer is full

    def close(self) -> None:
        """Drain what is queued, then close."""
        try:
            self.stream.stop()
        finally:
            self.stream.close()

    def abort(self) -> None:
        """Drop queued audio and close now."""
        try:
            self.stream.abort()
        finally:
            self.stream.close()


def open_output(rate: int) -> AudioOut:
    return AudioOut(rate)


def play_pcm(pcm: Union[np.ndarray, bytes], rate: int) -> None:
    """Play a whole int16 mono buffer and block until it ends."""
    data = pcm.astype(np.int16).tobytes() if isinstance(pcm, np.ndarray) else bytes(pcm)
    out = open_output(rate)
    out.write(data)
    out.close()


# ---------------------------------------------------------------- piper

_voice_cache: dict[str, object] = {}
_voice_lock = threading.Lock()


def load_piper(name: str, model_dir: Path = PIPER_DIR):
    """Load and cache a PiperVoice by name (e.g. en_US-lessac-medium)."""
    path = Path(model_dir) / f"{name}.onnx"
    with _voice_lock:
        v = _voice_cache.get(str(path))
        if v is None:
            if not path.exists():
                raise FileNotFoundError(f"{path} missing; run scripts/get_piper_voice.sh {name}")
            from piper import PiperVoice
            v = PiperVoice.load(str(path))
            _voice_cache[str(path)] = v
        return v


def piper_chunks(voice, text: str) -> Iterator[tuple[bytes, int]]:
    """(int16 bytes, sample rate) per sentence, as soon as each is synthesized."""
    for chunk in voice.synthesize(text):
        yield chunk.audio_int16_bytes, chunk.sample_rate


# ---------------------------------------------------------------- TTS

class _ElevenFailed(Exception):
    pass


class TTS:
    def __init__(self, cfg: dict, net=None):
        """net: anything with an `.online` bool (net.NetMonitor). None means assume online and
        let the first-byte timeout decide."""
        t = (cfg or {}).get("tts") or {}
        self.net = net
        self.piper_voice: str = t.get("piper_voice", "en_US-lessac-medium")
        self.eleven_model: str = t.get("elevenlabs_model", "eleven_flash_v2_5")
        self.eleven_voice_cfg: str = t.get("elevenlabs_voice_id") or ""
        self._stop = threading.Event()
        self._lock = threading.Lock()          # one utterance at a time
        self._out: Optional[AudioOut] = None
        self._resp = None
        self.last_engine: Optional[str] = None  # 'elevenlabs' | 'piper' | None
        self.last_first_audio_s: Optional[float] = None

    # -- routing

    def _eleven_creds(self) -> Optional[tuple[str, str]]:
        key = os.environ.get("ELEVENLABS_API_KEY", "").strip()
        voice = os.environ.get("ELEVENLABS_VOICE_ID", "").strip() or self.eleven_voice_cfg
        if not key or not voice:
            return None
        return key, voice

    def _online(self) -> bool:
        if self.net is None:
            return True
        try:
            return bool(self.net.online)
        except Exception:
            return False

    def speak(self, text: str) -> None:
        """Speak text; blocks until playback ends or stop() is called. Never raises."""
        text = (text or "").strip()
        if not text:
            return
        with self._lock:
            self._stop.clear()
            self.last_engine = None
            self.last_first_audio_s = None
            creds = self._eleven_creds()
            if creds and self._online():
                try:
                    self._speak_eleven(text, *creds)
                    self.last_engine = "elevenlabs"
                    return
                except _ElevenFailed as ex:
                    log.warning("ElevenLabs failed (%s); using Piper", ex)
                except Exception:
                    log.exception("ElevenLabs failed; using Piper")
            if self._stop.is_set():
                return
            try:
                self._speak_piper(text)
                self.last_engine = "piper"
            except Exception:
                log.exception("Piper failed; nothing spoken")

    @property
    def speaking(self) -> bool:
        """True while an answer plays (the always-on mic waits, or it would answer itself)."""
        return self._lock.locked()

    def stop(self) -> None:
        """Cut off current speech (safe from any thread)."""
        self._stop.set()
        out, resp = self._out, self._resp
        if resp is not None:
            try:
                resp.close()
            except Exception:
                pass
        if out is not None:
            try:
                out.abort()
            except Exception:
                pass

    # -- engines

    def _speak_eleven(self, text: str, key: str, voice_id: str) -> None:
        """Stream PCM to the speaker as it arrives. Raises _ElevenFailed before any audio is
        played (so the caller can fall back); errors after first audio just end the utterance."""
        t0 = time.monotonic()
        q: "queue.Queue[object]" = queue.Queue()
        DONE = object()
        abandon = threading.Event()

        def fetch() -> None:
            r = None
            try:
                r = requests.post(
                    ELEVEN_URL.format(voice_id=voice_id),
                    params={"output_format": ELEVEN_FORMAT},
                    headers={"xi-api-key": key, "Content-Type": "application/json"},
                    json={"text": text, "model_id": self.eleven_model},
                    stream=True, timeout=(FIRST_BYTE_S, 5.0))
                if abandon.is_set():
                    return
                self._resp = r
                if r.status_code != 200:
                    q.put(_ElevenFailed(f"HTTP {r.status_code}: {r.text[:200]}"))
                    return
                for chunk in r.iter_content(chunk_size=CHUNK_BYTES):
                    if self._stop.is_set() or abandon.is_set():
                        break
                    if chunk:
                        q.put(chunk)
                q.put(DONE)
            except Exception as ex:
                q.put(_ElevenFailed(f"{type(ex).__name__}: {ex}"))
            finally:
                if r is not None:
                    if self._resp is r:
                        self._resp = None
                    try:
                        r.close()
                    except Exception:
                        pass

        threading.Thread(target=fetch, name="eleven", daemon=True).start()
        try:
            first = q.get(timeout=FIRST_BYTE_S)
        except queue.Empty:
            abandon.set()
            self._close_resp()
            raise _ElevenFailed(f"no audio after {FIRST_BYTE_S} s")
        if isinstance(first, Exception):
            raise first
        if first is DONE:
            raise _ElevenFailed("empty audio stream")
        self.last_first_audio_s = time.monotonic() - t0
        out = self._out = open_output(ELEVEN_RATE)
        try:
            out.write(first)  # type: ignore[arg-type]
            while not self._stop.is_set():
                try:
                    item = q.get(timeout=5.0)
                except queue.Empty:
                    log.warning("ElevenLabs stream stalled")
                    break
                if item is DONE:
                    break
                if isinstance(item, Exception):
                    log.warning("ElevenLabs stream broke mid-utterance: %s", item)
                    break
                out.write(item)  # type: ignore[arg-type]
        finally:
            abandon.set()
            self._finish(out)

    def _speak_piper(self, text: str) -> None:
        t0 = time.monotonic()
        voice = load_piper(self.piper_voice)
        out: Optional[AudioOut] = None
        try:
            for pcm, rate in piper_chunks(voice, text):
                if self._stop.is_set():
                    break
                if out is None:
                    self.last_first_audio_s = time.monotonic() - t0
                    out = self._out = open_output(rate)
                out.write(pcm)
        finally:
            if out is not None:
                self._finish(out)

    def _finish(self, out: AudioOut) -> None:
        self._out = None
        try:
            if self._stop.is_set():
                out.abort()
            else:
                out.close()
        except Exception:
            pass

    def _close_resp(self) -> None:
        r = self._resp
        if r is not None:
            try:
                r.close()
            except Exception:
                pass

    # -- helpers

    def synthesize_piper(self, text: str) -> tuple[np.ndarray, int, float]:
        """Offline synthesis without playback: (int16 samples, rate, seconds to first chunk)."""
        t0 = time.monotonic()
        voice = load_piper(self.piper_voice)
        parts, rate, first = [], 22050, None
        for pcm, rate in piper_chunks(voice, text):
            if first is None:
                first = time.monotonic() - t0
            parts.append(np.frombuffer(pcm, dtype=np.int16))
        audio = np.concatenate(parts) if parts else np.zeros(0, dtype=np.int16)
        return audio, rate, (first if first is not None else time.monotonic() - t0)

    def warm(self) -> None:
        """Load the Piper voice now so the first offline answer is not slow."""
        try:
            load_piper(self.piper_voice)
        except Exception:
            log.exception("Piper warm-up failed")
