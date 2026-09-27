"""Speech out (spec V9): the voice the phone app picked when online (Grok or ElevenLabs), Piper offline.

The voice follows the iPhone app's helper settings (sent over Bluetooth, mobile/PROTOCOL.md): engine
"grok" (the app's default, voice "eve"), "rig" (ElevenLabs, "Same as the rig" in the app) or "builtin"
(the iPhone's own voice, which the rig can't make: Piper). TTS.set_voice stores it in data/voice.json.

Grok (xAI, checked 2026-09-26): POST https://api.x.ai/v1/tts with Bearer XAI_API_KEY, body {"text",
"voice_id", "language", "speed" (0.7-1.5), "output_format": {"codec": "pcm", "sample_rate": 24000}}:
raw signed 16-bit little-endian mono, first bytes in ~0.12 s from the rig.

ElevenLabs (checked 2026-09-25): POST https://api.elevenlabs.io/v1/text-to-speech/{voice_id}/stream
with header xi-api-key, body {"text", "model_id"}, query output_format=pcm_22050 (raw signed 16-bit
little-endian mono, so no mp3 decoder). eleven_flash_v2_5 is the ~75 ms model.
  https://elevenlabs.io/docs/api-reference/text-to-speech/stream
  https://elevenlabs.io/docs/overview/models
Piper: piper-tts 1.8 `PiperVoice.load(model.onnx)`; `voice.synthesize(text)` yields one
AudioChunk per sentence (sample_rate, audio_int16_bytes). Voices live in models/piper/
(scripts/get_piper_voice.sh). A Piper voice that won't load is retried on every answer; meanwhile
espeak-ng (apt install espeak-ng), if installed, says it, robotic but not silent.

Every utterance plays on a worker thread under a deadline (playback_budget_s: 0.1 s per character + 3 s
from the moment the speaker opens). A speaker that stops taking audio blocks write/close forever; past the
deadline the stream is aborted (itself bounded, abort can hang too), a warning is logged and `speaking`
goes False, so the always-on mic, which waits while TTS.speaking, reopens.

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

import json
import logging
import os
import queue
import re
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator, Optional, Union

import numpy as np
import requests

from core.config import ROOT
from net import call_with_deadline

log = logging.getLogger(__name__)

ELEVEN_URL = "https://api.elevenlabs.io/v1/text-to-speech/{voice_id}/stream"
ELEVEN_FORMAT = "pcm_22050"
ELEVEN_RATE = 22050
FIRST_BYTE_S = 1.5          # slower than this -> Piper for this utterance
GROK_TTS_URL = "https://api.x.ai/v1/tts"
GROK_RATE = 24000
GROK_DEFAULT_VOICE = "eve"  # the iPhone app's default (Speaker.swift Grok.defaultVoice)
GROK_SPEEDS = (0.7, 1.5)
ENGINES = ("grok", "rig", "builtin")    # the app's Speaker.Engine raw values
VOICE_PATH = "data/voice.json"          # the phone's voice settings (tts.voice_path overrides)
SPEAKER_CHECK_S = 3.0                   # how long a speaker check is reused (GET /state polls often)
EXTERNAL_SINK = re.compile(r"^bluez_sink\.|usb", re.I)   # a PulseAudio sink that is a speaker, not the Jetson's own outputs
CHUNK_BYTES = 2048          # ~46 ms at 22.05 kHz int16
PIPER_DIR = ROOT / "models" / "piper"
MAX_SPOKEN_CHARS = 600      # longer text is cut at a sentence end: ~40 s of speech is already too long
PIPER_MAX_CHARS = 250       # Piper runs a sentence in one pass and its memory grows with the square of its
                            # length (6,600 chars: 8 GB); longer sentences go in pieces, cut at commas/spaces


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
        """Drop queued audio and close now. Only on the thread that writes: PortAudio doesn't allow
        closing a stream another thread is blocked writing to (ALSA may crash)."""
        try:
            self.stream.abort()
        finally:
            self.stream.close()

    def interrupt(self) -> None:
        """Drop queued audio and stop now, from any thread; a write blocked on the stream returns (or
        raises) and the writing thread closes it (abort())."""
        self.stream.abort()


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


def pulse_default_sink(timeout: float = 2.0) -> Optional[str]:
    """PulseAudio's default sink name (pactl, from pulseaudio-utils), or None without one."""
    import subprocess
    try:
        r = subprocess.run(["pactl", "get-default-sink"], capture_output=True, text=True, timeout=timeout)
        name = r.stdout.strip()
        if r.returncode == 0 and name:
            return name
        r = subprocess.run(["pactl", "info"], capture_output=True, text=True, timeout=timeout)
        for line in r.stdout.splitlines():
            if line.startswith("Default Sink:"):
                return line.split(":", 1)[1].strip() or None
    except (OSError, subprocess.SubprocessError):
        pass
    return None


def resolve_output_device(spec) -> Optional[int]:
    """tts.output_device -> a sounddevice index, or None for the default device. An int (or digits)
    is taken as the index; any other string picks the output device whose name contains it (case
    insensitive), the first if several do (with a warning naming them); a device whose whole name it is
    wins ("default" is not "sysdefault"). No match: the default device,
    with a warning, so an unplugged speaker costs the voice its device, not the answer."""
    if spec is None or (isinstance(spec, str) and not spec.strip()):
        return None
    if isinstance(spec, int) or (isinstance(spec, str) and spec.strip().isdigit()):
        return int(spec)
    want = str(spec).strip().lower()
    hits = [d for d in output_devices() if want in str(d.get("name", "")).lower()]
    exact = [d for d in hits if str(d.get("name", "")).strip().lower() == want]
    if exact:                   # "pulse" / "default" (the host PulseAudio: a Bluetooth speaker) are whole names;
        hits = exact[:1]        # "default" is also inside "sysdefault", which PortAudio lists first
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
    """Load and cache a PiperVoice by name (e.g. en_US-lessac-medium). A failed load is not cached,
    so the next answer tries again."""
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
    """(int16 bytes, sample rate) per sentence, as soon as each is synthesized; a sentence longer than
    PIPER_MAX_CHARS goes to Piper in pieces (one 21,000-character sentence took the rig out of memory)."""
    for piece in text_pieces(text):
        for chunk in voice.synthesize(piece):
            yield chunk.audio_int16_bytes, chunk.sample_rate


def text_pieces(text: str, limit: int = PIPER_MAX_CHARS) -> list[str]:
    """Sentences of at most limit characters: a longer one is cut after its last comma or semicolon that
    fits, else at its last space, else at limit."""
    out: list[str] = []
    for sent in re.split(r"(?<=[.!?])\s+", text.strip()):
        while len(sent) > limit:
            cut = max(sent.rfind(", ", 0, limit), sent.rfind("; ", 0, limit)) + 1
            if cut <= 0:
                cut = sent.rfind(" ", 0, limit)
            if cut <= 0:
                cut = limit
            out.append(sent[:cut].strip())
            sent = sent[cut:].strip()
        if sent:
            out.append(sent)
    return out


def speakable(text: str, limit: int = MAX_SPOKEN_CHARS) -> str:
    """text, or its whole sentences that fit in limit characters (else cut at a word, with a full stop)."""
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    head = text[:limit + 1]
    end = max(head.rfind(". "), head.rfind("! "), head.rfind("? "))
    if end > 0:
        return head[:end + 1]
    cut = head.rfind(" ", 0, limit)
    return (head[:cut] if cut > 0 else head[:limit]).rstrip(",;:") + "."


# ---------------------------------------------------------------- espeak (last resort)

ESPEAK_TIMEOUT_S = 5.0


def espeak_pcm(text: str) -> Optional[tuple[bytes, int]]:
    """(int16 mono bytes, rate) from espeak-ng / espeak, or None when neither is installed or it fails.
    The robotic last resort when Piper can't load: better than a silent rig."""
    import io
    import shutil
    import subprocess
    import wave
    exe = shutil.which("espeak-ng") or shutil.which("espeak")
    if exe is None:
        return None
    try:
        wav = subprocess.run([exe, "--stdout", "--", text], capture_output=True, check=True,
                             timeout=ESPEAK_TIMEOUT_S).stdout
        with wave.open(io.BytesIO(wav), "rb") as w:
            if w.getsampwidth() != 2 or w.getnchannels() != 1:
                return None
            return w.readframes(w.getnframes()), w.getframerate()
    except Exception as ex:
        log.warning("espeak failed: %s: %s", type(ex).__name__, ex)
        return None


# ---------------------------------------------------------------- TTS

# Playback deadline per utterance, from the moment the speaker is opened: a USB speaker that stops taking
# audio blocks AudioOut.write/close forever, and while TTS.speaking the always-on mic stays shut. Speech
# runs about 15 characters a second, so 0.1 s per character is ~1.5x the real length, plus slack for the
# drain at the end. Before any audio (ElevenLabs first byte, Piper load and first sentence) PREPARE_MAX_S.
PLAY_S_PER_CHAR = 0.1
PLAY_SLACK_S = 3.0
PREPARE_MAX_S = 15.0
ABORT_S = 1.0               # stream.abort() can hang on a wedged device too
WATCH_S = 0.05              # how often speak() checks the deadline


def playback_budget_s(text: str) -> float:
    return len(text) * PLAY_S_PER_CHAR + PLAY_SLACK_S


def _abort_quietly(out, close: bool = True) -> None:
    """out.abort() (close=True: on the writing thread) or out.interrupt() (close=False: from another
    thread, which must not close a stream being written to), given ABORT_S at most and never raising."""
    if out is None:
        return
    fn = out.abort if close else getattr(out, "interrupt", out.abort)
    try:
        call_with_deadline(fn, ABORT_S, name="tts-abort")
    except TimeoutError:
        log.warning("speaker abort hung for %.1f s; leaving it", ABORT_S)
    except Exception:
        pass


class _CloudFailed(Exception):
    """A cloud voice failed before any audio played: the next engine speaks instead."""


_ElevenFailed = _CloudFailed


@dataclass(frozen=True)
class VoiceChoice:
    """The phone app's voice settings, as the rig uses them."""
    engine: str = "grok"
    grok_voice: str = GROK_DEFAULT_VOICE
    speed: float = 1.0

    @classmethod
    def make(cls, engine=None, grok_voice=None, speed=None) -> "VoiceChoice":
        """Validated: an unknown engine is grok, an empty voice the default, speed clamped to GROK_SPEEDS."""
        e = str(engine or "grok").strip().lower()
        v = re.sub(r"[^a-z0-9_-]", "", str(grok_voice or "").strip().lower())[:32]
        try:
            sp = float(speed) if speed is not None else 1.0
        except (TypeError, ValueError):
            sp = 1.0
        if sp != sp:                            # NaN
            sp = 1.0
        return cls(e if e in ENGINES else "grok", v or GROK_DEFAULT_VOICE,
                   round(min(max(sp, GROK_SPEEDS[0]), GROK_SPEEDS[1]), 2))

    @classmethod
    def load(cls, path: Optional[Path]) -> "VoiceChoice":
        try:
            d = json.loads(Path(path).read_text()) if path else {}
            return cls.make(d.get("engine"), d.get("grok_voice"), d.get("speed"))
        except (OSError, ValueError, AttributeError):
            return cls()

    def save(self, path: Optional[Path]) -> None:
        if not path:
            return
        try:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            tmp = Path(path).with_suffix(".tmp")
            tmp.write_text(json.dumps(asdict(self)))
            tmp.replace(path)
        except OSError as ex:
            log.warning("voice settings not saved (%s); they last until a restart", ex)


class _Utterance:
    """One speak() call's state. A hung utterance's worker thread may outlive speak(); keeping its stop
    flag and stream here (not on the TTS) means it can't touch the next utterance's."""

    def __init__(self):
        self.stop = threading.Event()
        self.t0 = time.monotonic()
        self.t_audio: Optional[float] = None    # when the speaker was opened
        self.out: Optional[AudioOut] = None
        self.resp = None


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
        vp = t.get("voice_path", VOICE_PATH)
        self.voice_path: Optional[Path] = (ROOT / vp if not Path(vp).is_absolute() else Path(vp)) if vp else None
        self.voice = VoiceChoice.load(self.voice_path)   # the phone app's pick; its default until one arrives
        self._lock = threading.Lock()          # one utterance at a time
        self._utt: Optional[_Utterance] = None
        self.last_engine: Optional[str] = None  # 'grok' | 'elevenlabs' | 'piper' | 'espeak' | None
        self.last_first_audio_s: Optional[float] = None
        self.last_timed_out = False             # the last utterance hit its playback deadline

    # -- routing

    def set_voice(self, engine=None, grok_voice=None, speed=None) -> VoiceChoice:
        """The phone app's voice settings (engine grok | rig | builtin, Grok voice, speed): used from the
        next answer on and kept in voice_path across restarts."""
        v = VoiceChoice.make(engine, grok_voice, speed)
        if v != self.voice:
            log.info("voice: %s (Grok voice %s, speed %.2f), from the phone", v.engine, v.grok_voice, v.speed)
            self.voice = v
            v.save(self.voice_path)
        return v

    def speaker_status(self) -> dict:
        """{"ok", "name"}: whether speech goes to an external speaker now. Through PulseAudio
        (output_device 'pulse'/'default' with PULSE_SERVER set, the askroom:audio image): its default
        sink is a Bluetooth or USB one. A named ALSA device: it is plugged in. The phone stays quiet
        while ok (mobile/PROTOCOL.md, status 'spk'). Cached SPEAKER_CHECK_S."""
        now = time.monotonic()
        cached = getattr(self, "_spk", None)
        if cached is not None and now - cached[0] < SPEAKER_CHECK_S:
            return cached[1]
        spec = self.output_device
        st = {"ok": False, "name": ""}
        try:
            via_pulse = str(spec or "").strip().lower() in ("pulse", "default", "") and bool(os.environ.get("PULSE_SERVER"))
            if via_pulse:
                sink = pulse_default_sink()
                st = {"ok": bool(sink and EXTERNAL_SINK.search(sink)), "name": sink or ""}
            elif spec is not None and str(spec).strip():
                want = str(spec).strip().lower()
                hits = [str(d.get("name", "")) for d in output_devices() if want in str(d.get("name", "")).lower()]
                st = {"ok": bool(hits), "name": hits[0] if hits else ""}
        except Exception as ex:
            log.debug("speaker check failed: %s", ex)
        self._spk = (now, st)
        return st

    def attach(self, world) -> "TTS":
        """GET /state gains 'speaker' (speaker_status) and 'voice' (the phone's voice settings), as
        core.grok_check.GrokCheck.attach adds its own entry."""
        state = world.state_json

        def state_json(*a, **kw):
            st = state(*a, **kw)
            st["speaker"] = self.speaker_status()
            st["voice"] = asdict(self.voice)
            return st

        world.state_json = state_json
        return self

    @staticmethod
    def _grok_key() -> Optional[str]:
        return os.environ.get("XAI_API_KEY", "").strip() or None

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
        """Speak text; blocks until playback ends, stop() is called, or the playback deadline passes
        (a hung speaker: the stream is aborted and speaking goes False, so the mic reopens). Never raises."""
        text = (text or "").strip()
        if not text:
            return
        if len(text) > MAX_SPOKEN_CHARS:
            log.warning("answer is %d characters; speaking only the first %d", len(text), MAX_SPOKEN_CHARS)
            text = speakable(text)
        with self._lock:
            u = self._utt = _Utterance()
            self.last_engine = None
            self.last_first_audio_s = None
            self.last_timed_out = False
            worker = threading.Thread(target=self._say, args=(text, u), name="tts", daemon=True)
            worker.start()
            budget = playback_budget_s(text)
            while True:
                worker.join(WATCH_S)
                if not worker.is_alive():
                    return
                now = time.monotonic()
                if u.t_audio is not None and now - u.t_audio > budget:
                    what = f"playback still running {budget:.1f} s after the speaker opened"
                    break
                if u.t_audio is None and now - u.t0 > PREPARE_MAX_S:
                    what = f"no audio after {PREPARE_MAX_S:.0f} s"
                    break
            log.warning("speech output hung (%s); aborting it so the microphone reopens", what)
            self.last_timed_out = True
            self._cut(u)

    def _cloud_voices(self) -> list:
        """(name, speak) for the cloud voices to try, the phone's pick first: grok -> Grok then
        ElevenLabs; rig -> ElevenLabs then Grok; builtin (the iPhone's own voice) -> none (Piper)."""
        v = self.voice
        grok = ("grok", lambda text, u: self._speak_grok(text, key, v.grok_voice, v.speed, u)) \
            if (key := self._grok_key()) else None
        eleven = ("elevenlabs", lambda text, u: self._speak_eleven(text, *creds, u=u)) \
            if (creds := self._eleven_creds()) else None
        order = {"grok": [grok, eleven], "rig": [eleven, grok], "builtin": []}[v.engine]
        return [c for c in order if c is not None]

    def _say(self, text: str, u: _Utterance) -> None:
        """The utterance itself, on the worker thread: the phone's cloud voice, else Piper, else espeak."""
        try:
            if self._online():
                for name, speak in self._cloud_voices():
                    if u.stop.is_set():
                        return
                    try:
                        speak(text, u)
                        self.last_engine = name
                        return
                    except _CloudFailed as ex:
                        log.warning("%s voice failed (%s); trying the next", name, ex)
                    except Exception:
                        log.exception("%s voice failed; trying the next", name)
            if u.stop.is_set():
                return
            try:
                self._speak_piper(text, u)
                self.last_engine = "piper"
                return
            except Exception as ex:
                if u.t_audio is not None:           # broke mid-answer: part of it was said, don't repeat it
                    log.warning("Piper broke mid-utterance: %s: %s", type(ex).__name__, ex)
                    return
                log.error("Piper failed (%s: %s); trying espeak", type(ex).__name__, ex)
            if u.stop.is_set():
                return
            if self._speak_espeak(text, u):
                self.last_engine = "espeak"
            else:
                log.error("no voice could speak (Piper failed, espeak-ng not installed or failed): %r "
                          "was not said", text[:60])
        except Exception:
            log.exception("speech failed")

    @property
    def speaking(self) -> bool:
        """True while an answer plays (the always-on mic waits, or it would answer itself)."""
        return self._lock.locked()

    def stop(self) -> None:
        """Cut off current speech (safe from any thread; returns within about ABORT_S)."""
        u = self._utt
        if u is not None:
            self._cut(u)

    def _cut(self, u: _Utterance) -> None:
        u.stop.set()
        resp, out = u.resp, u.out
        if resp is not None:
            try:
                resp.close()
            except Exception:
                pass
        _abort_quietly(out, close=False)      # the worker closes it (_finish), unless it is stuck for good

    # -- engines

    def _open(self, rate: int, u: _Utterance) -> AudioOut:
        """Open the speaker for u; the playback deadline runs from here (opening can hang too)."""
        u.t_audio = time.monotonic()
        self.last_first_audio_s = u.t_audio - u.t0
        u.out = open_output(rate, self._device())
        if u.stop.is_set():                    # cut while the device was opening
            _abort_quietly(u.out)
        return u.out

    def _speak_eleven(self, text: str, key: str, voice_id: str, u: Optional[_Utterance] = None) -> None:
        self._stream_cloud("ElevenLabs", ELEVEN_RATE, u, lambda: requests.post(
            ELEVEN_URL.format(voice_id=voice_id),
            params={"output_format": ELEVEN_FORMAT},
            headers={"xi-api-key": key, "Content-Type": "application/json"},
            json={"text": text, "model_id": self.eleven_model},
            stream=True, timeout=(FIRST_BYTE_S, 5.0)))

    def _speak_grok(self, text: str, key: str, voice: str, speed: float,
                    u: Optional[_Utterance] = None) -> None:
        self._stream_cloud("Grok", GROK_RATE, u, lambda: requests.post(
            GROK_TTS_URL,
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            json={"text": text, "voice_id": voice, "language": "en", "speed": speed,
                  "output_format": {"codec": "pcm", "sample_rate": GROK_RATE}},
            stream=True, timeout=(FIRST_BYTE_S, 5.0)))

    def _stream_cloud(self, name: str, rate: int, u: Optional[_Utterance], post) -> None:
        """Stream a cloud voice's raw int16 mono PCM to the speaker as it arrives. post() makes the
        streaming request. Raises _CloudFailed before any audio is played (so the caller can fall
        back); errors after first audio just end the utterance."""
        u = u or _Utterance()
        q: "queue.Queue[object]" = queue.Queue()
        DONE = object()
        abandon = threading.Event()

        def fetch() -> None:
            r = None
            try:
                r = post()
                if abandon.is_set():
                    return
                u.resp = r
                if r.status_code != 200:
                    q.put(_CloudFailed(f"HTTP {r.status_code}: {r.text[:200]}"))
                    return
                for chunk in r.iter_content(chunk_size=CHUNK_BYTES):
                    if u.stop.is_set() or abandon.is_set():
                        break
                    if chunk:
                        q.put(chunk)
                q.put(DONE)
            except Exception as ex:
                q.put(_CloudFailed(f"{type(ex).__name__}: {ex}"))
            finally:
                if r is not None:
                    if u.resp is r:
                        u.resp = None
                    try:
                        r.close()
                    except Exception:
                        pass

        threading.Thread(target=fetch, name=name.lower(), daemon=True).start()
        try:
            first = q.get(timeout=FIRST_BYTE_S)
        except queue.Empty:
            abandon.set()
            self._close_resp(u)
            raise _CloudFailed(f"no audio after {FIRST_BYTE_S} s")
        if isinstance(first, Exception):
            raise first
        if first is DONE:
            raise _CloudFailed("empty audio stream")
        out = self._open(rate, u)
        try:
            out.write(first)  # type: ignore[arg-type]
            while not u.stop.is_set():
                try:
                    item = q.get(timeout=5.0)
                except queue.Empty:
                    log.warning("%s stream stalled", name)
                    break
                if item is DONE:
                    break
                if isinstance(item, Exception):
                    log.warning("%s stream broke mid-utterance: %s", name, item)
                    break
                out.write(item)  # type: ignore[arg-type]
        finally:
            abandon.set()
            self._finish(out, u)

    def _speak_piper(self, text: str, u: Optional[_Utterance] = None) -> None:
        u = u or _Utterance()
        voice = load_piper(self.piper_voice)
        out: Optional[AudioOut] = None
        try:
            for pcm, rate in piper_chunks(voice, text):
                if u.stop.is_set():
                    break
                if out is None:
                    out = self._open(rate, u)
                out.write(pcm)
        finally:
            if out is not None:
                self._finish(out, u)

    def _speak_espeak(self, text: str, u: _Utterance) -> bool:
        got = espeak_pcm(text)
        if got is None or u.stop.is_set():
            return False
        pcm, rate = got
        out = self._open(rate, u)
        try:
            out.write(pcm)
        finally:
            self._finish(out, u)
        return True

    def _finish(self, out: AudioOut, u: _Utterance) -> None:
        if u.out is out:
            u.out = None
        if u.stop.is_set():
            _abort_quietly(out)
            return
        try:
            out.close()
        except Exception:
            pass

    @staticmethod
    def _close_resp(u: _Utterance) -> None:
        r = u.resp
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

    def warm(self) -> bool:
        """Load the Piper voice now so the first offline answer is not slow. False (with an error log)
        if it failed: every answer then tries Piper again, and espeak-ng if Piper still won't load."""
        try:
            load_piper(self.piper_voice)
            return True
        except Exception as ex:
            log.error("Piper voice %s failed to load (%s: %s). Offline answers will retry it each time, "
                      "then fall back to espeak-ng (%s).", self.piper_voice, type(ex).__name__, ex,
                      "installed" if _espeak_available() else "NOT installed: offline answers will be silent")
            return False


def _espeak_available() -> bool:
    import shutil
    return bool(shutil.which("espeak-ng") or shutil.which("espeak"))


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
