import threading
import time
from types import SimpleNamespace as NS

import numpy as np
import pytest
import requests

from core.config import load_config
from voice import tts
from voice.tts import TTS

CFG = load_config()


class Recorder:
    """Stands in for open_output(); collects everything written."""

    def __init__(self):
        self.opened = []       # sample rates
        self.data = bytearray()
        self.closed = self.aborted = 0

    def __call__(self, rate):
        self.opened.append(rate)
        rec = self

        class Out:
            def write(self, b):
                rec.data += b

            def close(self):
                rec.closed += 1

            def abort(self):
                rec.aborted += 1

        return Out()


class FakeResp:
    def __init__(self, status=200, chunks=(b"\x01\x00" * 100, b"\x02\x00" * 100), delay=0.0):
        self.status_code = status
        self.chunks = list(chunks)
        self.delay = delay
        self.text = "error body"
        self.closed = False

    def iter_content(self, chunk_size=None):
        for c in self.chunks:
            yield c

    def close(self):
        self.closed = True


class FakePost:
    def __init__(self, resp=None, exc=None, delay=0.0):
        self.resp, self.exc, self.delay = resp or FakeResp(), exc, delay
        self.calls = []

    def __call__(self, url, **kw):
        self.calls.append((url, kw))
        if self.delay:
            time.sleep(self.delay)
        if self.exc:
            raise self.exc
        return self.resp


class FakeVoice:
    def __init__(self):
        self.texts = []

    def synthesize(self, text):
        self.texts.append(text)
        for i in range(2):
            yield NS(sample_rate=22050, audio_int16_bytes=np.full(50, 7 + i, np.int16).tobytes())


@pytest.fixture
def audio(monkeypatch):
    rec = Recorder()
    monkeypatch.setattr(tts, "open_output", rec)
    return rec


@pytest.fixture
def piper(monkeypatch):
    v = FakeVoice()
    monkeypatch.setattr(tts, "load_piper", lambda name, model_dir=None: v)
    return v


@pytest.fixture
def eleven_env(monkeypatch):
    monkeypatch.setenv("ELEVENLABS_API_KEY", "k")
    monkeypatch.setenv("ELEVENLABS_VOICE_ID", "voice123")


def online(v=True):
    return NS(online=v)


def test_online_with_key_streams_elevenlabs(monkeypatch, audio, piper, eleven_env):
    post = FakePost()
    monkeypatch.setattr(tts.requests, "post", post)
    t = TTS(CFG, online())
    t.speak("Your keys are in the box.")
    assert t.last_engine == "elevenlabs" and piper.texts == []
    url, kw = post.calls[0]
    assert url == "https://api.elevenlabs.io/v1/text-to-speech/voice123/stream"
    assert kw["params"] == {"output_format": "pcm_22050"}
    assert kw["headers"]["xi-api-key"] == "k"
    assert kw["json"] == {"text": "Your keys are in the box.", "model_id": "eleven_flash_v2_5"}
    assert kw["stream"] is True
    assert audio.opened == [22050] and bytes(audio.data) == b"\x01\x00" * 100 + b"\x02\x00" * 100
    assert audio.closed == 1
    assert t.last_first_audio_s is not None and t.last_first_audio_s < 0.4


def test_offline_uses_piper(monkeypatch, audio, piper, eleven_env):
    monkeypatch.setattr(tts.requests, "post", lambda *a, **k: pytest.fail("no network offline"))
    t = TTS(CFG, online(False))
    t.speak("Offline answer.")
    assert t.last_engine == "piper" and piper.texts == ["Offline answer."]
    assert audio.opened == [22050] and len(audio.data) == 200 and audio.closed == 1


def test_no_key_uses_piper(monkeypatch, audio, piper):
    monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)
    monkeypatch.setattr(tts.requests, "post", lambda *a, **k: pytest.fail("no key, no call"))
    t = TTS(CFG, online())
    t.speak("hello")
    assert t.last_engine == "piper"


@pytest.mark.parametrize("post", [
    FakePost(exc=requests.ConnectionError("dns")),
    FakePost(resp=FakeResp(status=401)),
    FakePost(resp=FakeResp(chunks=())),
])
def test_elevenlabs_error_falls_back_to_piper(monkeypatch, audio, piper, eleven_env, post):
    monkeypatch.setattr(tts.requests, "post", post)
    t = TTS(CFG, online())
    t.speak("fallback please")
    assert len(post.calls) == 1
    assert t.last_engine == "piper" and piper.texts == ["fallback please"]
    assert audio.opened == [22050]          # only Piper opened the device


def test_slow_first_byte_falls_back(monkeypatch, audio, piper, eleven_env):
    monkeypatch.setattr(tts, "FIRST_BYTE_S", 0.1)
    monkeypatch.setattr(tts.requests, "post", FakePost(delay=0.5))
    t = TTS(CFG, online())
    t0 = time.perf_counter()
    t.speak("slow")
    assert time.perf_counter() - t0 < 0.4
    assert t.last_engine == "piper"


def test_net_none_counts_as_online(monkeypatch, audio, piper, eleven_env):
    monkeypatch.setattr(tts.requests, "post", FakePost())
    t = TTS(CFG)
    t.speak("hi")
    assert t.last_engine == "elevenlabs"


def test_stop_cuts_piper(monkeypatch, audio, eleven_env):
    started = threading.Event()

    class SlowVoice:
        def synthesize(self, text):
            for i in range(50):
                if i == 2:
                    started.set()          # device is open by now
                time.sleep(0.02)
                yield NS(sample_rate=22050, audio_int16_bytes=b"\x00\x00" * 10)

    monkeypatch.setattr(tts, "load_piper", lambda name, model_dir=None: SlowVoice())
    t = TTS(CFG, online(False))
    th = threading.Thread(target=t.speak, args=("long",))
    th.start()
    assert started.wait(1)
    t.stop()
    th.join(1)
    assert not th.is_alive() and audio.aborted >= 1


def test_empty_text_is_noop(audio, piper):
    TTS(CFG, online(False)).speak("   ")
    assert audio.opened == [] and piper.texts == []


def test_play_pcm_seam(monkeypatch, audio):
    tts.play_pcm(np.array([1, 2, 3], np.int16), 16000)
    assert audio.opened == [16000] and bytes(audio.data) == np.array([1, 2, 3], np.int16).tobytes()


# ------------------------------------------------------------------ real local Piper check

@pytest.mark.skipif(not (tts.PIPER_DIR / "en_US-lessac-medium.onnx").exists(),
                    reason="run scripts/get_piper_voice.sh first")
def test_real_piper_synthesis():
    t = TTS(CFG, online(False))
    t.warm()                                   # model load is a one-time startup cost
    audio, rate, first_s = t.synthesize_piper("Your keys are probably in the box.")
    print(f"\nPiper time-to-first-audio (warm): {first_s * 1000:.0f} ms, "
          f"{len(audio) / rate:.2f} s of audio at {rate} Hz")
    assert rate == 22050 and audio.dtype == np.int16 and len(audio) > 0
    assert np.abs(audio).max() > 1000          # not silence
