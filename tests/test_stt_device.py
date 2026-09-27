"""stt.input_device: the mic by index or name substring, and a mic that won't record at 16 kHz (an ALSA hw
device on the Jetson) recorded at its own rate and resampled to the VAD's 512-sample blocks. sounddevice
is faked."""
import sys
import types

import numpy as np
import pytest

from voice import stt
from voice.stt import BLOCK, RATE, _to_block, resolve_input_device

DEVICES = [
    {"name": "BRIO 4K Stream Edition: USB Audio (hw:2,0)", "max_input_channels": 2, "max_output_channels": 0,
     "default_samplerate": 48000.0},
    {"name": "USB PnP Sound Device: Audio (hw:3,0)", "max_input_channels": 1, "max_output_channels": 2,
     "default_samplerate": 44100.0},
    {"name": "NVIDIA Jetson HDA: HDMI 0 (hw:0,3)", "max_input_channels": 0, "max_output_channels": 8,
     "default_samplerate": 44100.0},
    {"name": "default", "max_input_channels": 32, "max_output_channels": 32, "default_samplerate": 44100.0},
]


class FakeSD(types.ModuleType):
    class PortAudioError(Exception):
        pass

    def __init__(self, rates=None):
        super().__init__("sounddevice")
        self.default = types.SimpleNamespace(device=[3, 3])
        self.rates, self.streams = rates or {}, []
        sd = self

        class InputStream:
            def __init__(self, samplerate, blocksize, channels, dtype, device=None, callback=None):
                ok = sd.rates.get(device)
                if ok is not None and samplerate != ok:
                    raise sd.PortAudioError(f"Invalid sample rate [PaErrorCode -9997] {samplerate}")
                self.samplerate, self.blocksize, self.device, self.callback = samplerate, blocksize, device, callback
                sd.streams.append(self)

            def start(self):
                pass

            def stop(self):
                pass

            def close(self):
                pass

        self.InputStream = InputStream

    def query_devices(self, device=None, kind=None):
        if device is not None:
            return dict(DEVICES[device], index=device)
        return [dict(d, index=i) for i, d in enumerate(DEVICES)]


@pytest.fixture
def sd(monkeypatch):
    fake = FakeSD()
    monkeypatch.setitem(sys.modules, "sounddevice", fake)
    return fake


def test_input_device_by_name_index_or_default(sd, caplog):
    assert resolve_input_device(None) is None and resolve_input_device("") is None
    assert resolve_input_device(1) == 1 and resolve_input_device("1") == 1
    assert resolve_input_device("pnp") == 1
    assert resolve_input_device("brio") == 0
    assert resolve_input_device("hdmi") is None and "no input device matches" in caplog.text   # output only
    assert resolve_input_device("usb") == 0 and "several input devices match" in caplog.text     # sounddevice would refuse


def _feed(stream, n):
    status = types.SimpleNamespace(input_overflow=False)
    t = np.arange(n) / stream.samplerate
    x = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)[:, None]
    stream.callback(x, n, None, status)


def test_a_mic_that_refuses_16k_is_recorded_at_its_rate_and_resampled(sd):
    sd.rates = {0: 48000}                       # the Brio: 48 kHz only
    src = stt.AudioIn(RATE, BLOCK, 0)
    s = sd.streams[-1]
    assert (src.device_rate, s.samplerate, s.blocksize) == (48000, 48000, 1536)
    _feed(s, 1536)
    got = src.read(0.1)
    assert got.shape == (BLOCK,) and got.dtype == np.float32 and 0.3 < np.abs(got).max() <= 0.5


def test_44k1_mic_blocks_come_out_as_vad_blocks(sd):
    sd.rates = {1: 44100}
    src = stt.AudioIn(RATE, BLOCK, 1)
    s = sd.streams[-1]
    assert s.blocksize == 1411
    _feed(s, 1411)
    assert src.read(0.1).shape == (BLOCK,)


def test_a_mic_at_16k_is_untouched(sd):
    src = stt.AudioIn(RATE, BLOCK, 3)
    s = sd.streams[-1]
    assert (src.device_rate, s.samplerate, s.blocksize) == (RATE, RATE, BLOCK)
    x = np.linspace(-1, 1, BLOCK, dtype=np.float32)[:, None]
    s.callback(x, BLOCK, None, types.SimpleNamespace(input_overflow=False))
    assert np.array_equal(src.read(0.1), x[:, 0])


def test_to_block_averages_whole_ratios():
    x = np.repeat(np.arange(BLOCK, dtype=np.float32), 3)
    assert np.array_equal(_to_block(x, BLOCK), np.arange(BLOCK, dtype=np.float32))


def test_level_meter_prints_level_and_speech(monkeypatch):
    class Src:
        device_rate = 48000

        def __init__(self):
            self.n = 0

        def read(self, timeout=1.0):
            self.n += 1
            return np.full(BLOCK, 0.1, np.float32)

        def close(self):
            pass

    class Vad:
        def reset(self):
            pass

        def __call__(self, x):
            return 0.9

    monkeypatch.setattr(stt, "open_input", lambda rate, block, device=None: Src())
    s = stt.STT({"stt": {"vad_threshold": 0.5}}, vad=Vad(), backend=object())
    lines = []
    assert stt.level_meter(s, 1.0, out=lines.append) == 0
    assert lines[0].startswith("input at 48000 Hz") and len(lines) == 4    # 0.5 s lines up to 1 s
    assert "-20.0 dBFS" in lines[1] and "SPEECH" in lines[1]
