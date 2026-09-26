"""tts.output_device: speech to a chosen sounddevice output (the speakerphone, not the container's
default HDMI), by index or name substring; sounddevice is faked."""
import sys
import types
from types import SimpleNamespace as NS

import numpy as np
import pytest

from core.config import load_config
from voice import tts
from voice.tts import TTS, resolve_output_device

DEVICES = [
    {"name": "NVIDIA Jetson Orin Nano HDA: HDMI 0 (hw:0,3)", "max_input_channels": 0, "max_output_channels": 8,
     "default_samplerate": 44100.0, "hostapi": 0},
    {"name": "HD Pro Webcam C920: USB Audio (hw:1,0)", "max_input_channels": 2, "max_output_channels": 0,
     "default_samplerate": 32000.0, "hostapi": 0},
    {"name": "Jabra SPEAK 410 USB: USB Audio (hw:2,0)", "max_input_channels": 1, "max_output_channels": 2,
     "default_samplerate": 48000.0, "hostapi": 0},
    {"name": "Other USB Audio speaker (hw:3,0)", "max_input_channels": 0, "max_output_channels": 2,
     "default_samplerate": 48000.0, "hostapi": 0},
    {"name": "default", "max_input_channels": 32, "max_output_channels": 32,
     "default_samplerate": 44100.0, "hostapi": 0},
]


class FakeSD(types.ModuleType):
    """Just enough of sounddevice: query_devices, default.device, RawOutputStream, PortAudioError."""

    class PortAudioError(Exception):
        pass

    def __init__(self, rates=None):
        super().__init__("sounddevice")
        self.default = NS(device=[4, 4])
        self.rates = rates          # device index -> the only sample rate it opens at (None: any)
        self.streams = []
        sd = self

        class RawOutputStream:
            def __init__(self, samplerate, channels, dtype, device=None, **kw):
                ok = (sd.rates or {}).get(device)
                if ok is not None and samplerate != ok:
                    raise sd.PortAudioError(f"Invalid sample rate [PaErrorCode -9997] {samplerate}")
                self.samplerate, self.device, self.data = samplerate, device, bytearray()
                sd.streams.append(self)

            def start(self):
                pass

            def write(self, b):
                self.data += bytes(b)

            def stop(self):
                pass

            def close(self):
                pass

            def abort(self):
                pass

        self.RawOutputStream = RawOutputStream

    def query_devices(self, device=None, kind=None):
        if device is not None:
            return dict(DEVICES[device], index=device)
        return [dict(d, index=i) for i, d in enumerate(DEVICES)]


@pytest.fixture
def sd(monkeypatch):
    fake = FakeSD()
    monkeypatch.setitem(sys.modules, "sounddevice", fake)
    return fake


# ----- picking the device -------------------------------------------------------------------------

def test_null_is_the_default_device(sd):
    assert resolve_output_device(None) is None
    assert resolve_output_device("") is None


def test_index_as_int_or_digits(sd):
    assert resolve_output_device(2) == 2
    assert resolve_output_device("3") == 3


def test_name_substring_is_case_insensitive_and_output_only(sd):
    assert resolve_output_device("jabra") == 2
    assert resolve_output_device("HDMI") == 0
    assert resolve_output_device("C920") is None          # the camera has a mic, no speaker


def test_ambiguous_name_takes_the_first_output_and_says_so(sd, caplog):
    assert resolve_output_device("USB Audio") == 2         # the webcam matches too, but has no output
    assert "several output devices match" in caplog.text and "Other USB Audio speaker" in caplog.text


def test_unknown_name_falls_back_to_the_default_with_a_warning(sd, caplog):
    assert resolve_output_device("bluetooth") is None
    assert "no output device matches" in caplog.text


# ----- playing on it ------------------------------------------------------------------------------

def test_audio_out_opens_the_device(sd):
    out = tts.open_output(22050, 2)
    out.write(np.arange(10, dtype=np.int16).tobytes())
    out.close()
    s = sd.streams[0]
    assert s.device == 2 and s.samplerate == 22050 and len(s.data) == 20


def test_audio_out_resamples_when_the_device_refuses_the_rate(sd):
    """USB speakerphones often only run at 48 kHz (or 16 kHz) under ALSA hw: devices; Piper and
    ElevenLabs send 22.05 kHz. Open at the device's rate and resample instead of going silent."""
    sd.rates = {2: 48000}
    out = tts.open_output(22050, 2)
    t = np.arange(22050) / 22050
    pcm = (8000 * np.sin(2 * np.pi * 440 * t)).astype(np.int16)
    for k in range(0, len(pcm), 1000):                     # streamed in odd-sized chunks
        out.write(pcm[k:k + 1000].tobytes())
    out.close()
    s = sd.streams[-1]
    assert s.samplerate == 48000
    got = np.frombuffer(bytes(s.data), np.int16)
    assert abs(len(got) - 48000) <= 3                      # one second in, one second out
    spec = np.abs(np.fft.rfft(got.astype(np.float64)))
    assert abs(np.argmax(spec) * 48000 / len(got) - 440) < 3     # same pitch


def test_tts_speaks_on_the_configured_device_with_piper_and_elevenlabs(monkeypatch, sd):
    cfg = load_config()
    cfg["tts"] = dict(cfg.get("tts") or {}, output_device="jabra")
    opened = []

    class Out:
        def write(self, b):
            pass

        def close(self):
            pass

        def abort(self):
            pass

    monkeypatch.setattr(tts, "open_output", lambda rate, device=None: opened.append((rate, device)) or Out())

    class Voice:
        def synthesize(self, text):
            yield NS(sample_rate=22050, audio_int16_bytes=b"\x00\x00" * 10)

    monkeypatch.setattr(tts, "load_piper", lambda name, model_dir=None: Voice())
    TTS(cfg, NS(online=False)).speak("offline")

    class Resp:
        status_code = 200

        def iter_content(self, chunk_size=None):
            yield b"\x01\x00" * 10

        def close(self):
            pass

    monkeypatch.setenv("ELEVENLABS_API_KEY", "k")
    monkeypatch.setenv("ELEVENLABS_VOICE_ID", "v")
    monkeypatch.setattr(tts.requests, "post", lambda *a, **k: Resp())
    t = TTS(cfg, NS(online=True))
    t.speak("online")
    assert t.last_engine == "elevenlabs"
    assert opened == [(22050, 2), (22050, 2)]


def test_devices_listing(sd, capsys):
    assert tts.main(["--devices"]) == 0
    out = capsys.readouterr().out
    assert "Jabra SPEAK 410" in out and "HDMI" in out
    assert "C920" not in out                                # input-only devices are not listed
    assert "tts.output_device" in out
