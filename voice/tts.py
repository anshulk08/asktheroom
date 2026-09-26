"""Speech out (spec V9): ElevenLabs streaming when online, Piper offline.

ElevenLabs (checked 2026-09-25): POST https://api.elevenlabs.io/v1/text-to-speech/{voice_id}/stream
with header xi-api-key, body {"text", "model_id"}, query output_format=pcm_22050 (raw signed 16-bit
little-endian mono, so no mp3 decoder). eleven_flash_v2_5 is the ~75 ms model.
  https://elevenlabs.io/docs/api-reference/text-to-speech/stream
  https://elevenlabs.io/docs/overview/models
Piper: piper-tts 1.8 `PiperVoice.load(model.onnx)`; `voice.synthesize(text)` yields one
AudioChunk per sentence (sample_rate, audio_int16_bytes). Voices live in models/piper/
(scripts/get_piper_voice.sh).

Audio goes through two seams tests can patch: `open_output(rate, device)` (a streaming int16 mono
sink) and `play_pcm(pcm, rate, device)` (whole buffer, blocking).

Output device: tts.output_device in config.yaml, a sounddevice index or part of a device name; null
is the default device, which in the Jetson container is HDMI. `python -m voice.tts --devices` lists
the outputs. The speakerphone and the camera can both be called "USB Audio"; only devices with output
channels are matched (the camera has none), and if a name still matches several the first is used
with a warning: then give the index instead (indices can change when USB devices are replugged, so
prefer a distinctive part of the name, e.g. "Jabra"). A device that refuses the voice's sample rate
(USB speakerphones under ALSA hw: often take only 48 or 16 kHz) is opened at its own default rate
and the audio resampled.
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

class _Resampler:
    """Streaming linear-interpolation resampler for int16 mono: plenty for speech, and only one
    sample of state carried between chunks, so chunk boundaries don't click."""

    def __init__(self, src: int, dst: int):
        self.step = src / float(dst)           # input samples per output sample
        self.t = 0.0                           # next output position, in input samples from self.prev
        self.prev: Optional[float] = None      # last input sample of the previous chunk

    def __call__(self, x: np.ndarray) -> np.ndarray:
        x = x.astype(np.float32)
        if self.prev is not None:
            x = np.concatenate(([self.prev], x))
        n = len(x)
        if n < 2:
            self.prev = float(x[-1]) if n else self.prev
            return np.zeros(0, np.int16)
        k = int(np.floor((n - 1 - self.t) / self.step)) + 1 if self.t <= n - 1 else 0
        y = np.interp(self.t + self.step * np.arange(k), np.arange(n), x)
        self.t += self.step * k - (n - 1)
        self.prev = float(x[-1])
        return np.clip(np.round(y), -32768, 32767).astype(np.int16)


class AudioOut:
    """Streaming int16 mono output (sounddevice/PortAudio) on device (an index; None = default)."""

    def __init__(self, rate: int, device: Optional[int] = None):
        import sounddevice as sd
        self._rs: Optional[_Resampler] = None
        try:
            self.stream = sd.RawOutputStream(samplerate=rate, channels=1, dtype="int16", device=device)
        except sd.PortAudioError:
            dev_rate = int(sd.query_devices(device, "output")["default_samplerate"])
            if dev_rate == rate:
                raise
            log.info("output device refuses %d Hz; playing at its %d Hz, resampled", rate, dev_rate)
            self.stream = sd.RawOutputStream(samplerate=dev_rate, channels=1, dtype="int16", device=device)
            self._rs = _Resampler(rate, dev_rate)
        self.stream.start()
        self._odd = b""

    def write(self, data: bytes) -> None:
        data = self._odd + data
        cut = len(data) - (len(data) % 2)
        self._odd = data[cut:]
        if cut:
            chunk = data[:cut]
            if self._rs is not None:
                chunk = self._rs(np.frombuffer(chunk, np.int16)).tobytes()
            self.stream.write(chunk)           # blocks when the device buffer is full

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


def open_output(rate: int, device: Optional[int] = None) -> AudioOut:
    return AudioOut(rate, device)


def play_pcm(pcm: Union[np.ndarray, bytes], rate: int, device: Optional[int] = None) -> None:
    """Play a whole int16 mono buffer and block until it ends."""
    data = pcm.astype(np.int16).tobytes() if isinstance(pcm, np.ndarray) else bytes(pcm)
    out = open_output(rate, device)
    out.write(data)
    out.close()


def output_devices() -> list[dict]:
    """sounddevice devices that can play (max_output_channels > 0), each with its 'index'."""
    import sounddevice as sd
    return [dict(d, index=i) for i, d in enumerate(sd.query_devices()) if d.get("max_output_channels", 0) > 0]


def resolve_output_device(spec) -> Optional[int]:
    """tts.output_device -> a sounddevice index, or None for the default device. An int (or digits)
    is taken as the index; any other string picks the output device whose name contains it (case
    insensitive), the first if several do (with a warning naming them). No match: the default device,
    with a warning, so an unplugged speaker costs the voice its device, not the answer."""
    if spec is None or (isinstance(spec, str) and not spec.strip()):
        return None
    if isinstance(spec, int) or (isinstance(spec, str) and spec.strip().isdigit()):
        return int(spec)
    want = str(spec).strip().lower()
    hits = [d for d in output_devices() if want in str(d.get("name", "")).lower()]
    if not hits:
        log.warning("tts.output_device %r: no output device matches; using the default "
                    "(python -m voice.tts --devices lists them)", spec)
        return None
    if len(hits) > 1:
        log.warning("tts.output_device %r: several output devices match (%s); using %d. Set a longer "
                    "part of the name, or the index.", spec,
                    "; ".join(f"{d['index']}: {d['name']}" for d in hits), hits[0]["index"])
    return int(hits[0]["index"])


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
        self.output_device = t.get("output_device")      # index | name substring | None (default)
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
        out = self._out = open_output(ELEVEN_RATE, self._device())
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
                    out = self._out = open_output(rate, self._device())
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

    def _device(self) -> Optional[int]:
        """Resolved per utterance: USB devices can be replugged and renumbered while the app runs."""
        if self.output_device is None:
            return None
        try:
            return resolve_output_device(self.output_device)
        except Exception:
            log.exception("output device lookup failed; using the default")
            return None

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


def main(argv=None) -> int:
    """python -m voice.tts --devices | --say "text" [--device N|name]"""
    import argparse

    from core.config import load_config
    ap = argparse.ArgumentParser(description="speech output: list devices, or speak a test sentence")
    ap.add_argument("--devices", action="store_true", help="list output devices")
    ap.add_argument("--say", help="speak this through the configured (or --device) output")
    ap.add_argument("--device", help="override tts.output_device (index or part of the name)")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    cfg = load_config()
    spec = a.device if a.device is not None else (cfg.get("tts") or {}).get("output_device")
    if a.devices or not a.say:
        import sounddevice as sd
        try:
            default_out = sd.query_devices(kind="output")["index"]
        except Exception:
            default_out = None
        chosen = resolve_output_device(spec)
        for d in output_devices():
            mark = ("*" if d["index"] == chosen else " ") + ("d" if d["index"] == default_out else " ")
            print(f"{mark} {d['index']:3d}  {d['name']}  ({d['max_output_channels']} ch, "
                  f"{int(d['default_samplerate'])} Hz)")
        print(f"tts.output_device = {spec!r} -> {chosen if chosen is not None else 'default'} "
              f"(* = used for speech, d = system default)")
        return 0
    if a.device is not None:
        cfg.setdefault("tts", {})["output_device"] = a.device
    t = TTS(cfg)
    t.speak(a.say)
    print(f"spoke with {t.last_engine} on {t._device() if t.output_device is not None else 'default'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
