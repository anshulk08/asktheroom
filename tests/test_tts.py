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
        self.devices = []      # output device per open (None = default)
        self.data = bytearray()
        self.closed = self.aborted = 0

    def __call__(self, rate, device=None):
        self.opened.append(rate)
        self.devices.append(device)
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


# ------------------------------------------------------------------ a hung speaker

class HungOut:
    """A speaker that stops taking audio: write() (and abort(), unless abort_ok) block until released."""

    def __init__(self, release, abort_ok=True):
        self.release, self.abort_ok = release, abort_ok
        self.aborted = 0

    def write(self, b):
        self.release.wait()

    def close(self):
        self.release.wait()

    def abort(self):
        self.aborted += 1
        if not self.abort_ok:
            self.release.wait()


@pytest.mark.parametrize("abort_ok", [True, False])
def test_hung_speaker_times_out_and_the_next_answer_plays(monkeypatch, piper, abort_ok, caplog):
    monkeypatch.setattr(tts, "PLAY_S_PER_CHAR", 0.01)
    monkeypatch.setattr(tts, "PLAY_SLACK_S", 0.2)
    monkeypatch.setattr(tts, "ABORT_S", 0.1)
    release = threading.Event()
    hung = HungOut(release, abort_ok)
    monkeypatch.setattr(tts, "open_output", lambda rate, device=None: hung)
    t = TTS(CFG, online(False))
    budget = tts.playback_budget_s("stuck answer")
    assert budget == pytest.approx(0.32)
    th = threading.Thread(target=t.speak, args=("stuck answer",), daemon=True)
    t0 = time.perf_counter()
    th.start()
    time.sleep(0.05)
    assert t.speaking                              # the mic waits while it plays ...
    th.join(budget + 1.0)
    assert not th.is_alive() and not t.speaking   # ... and reopens once the deadline passes
    assert time.perf_counter() - t0 < budget + 0.5
    assert hung.aborted >= 1 and t.last_timed_out
    assert "speech output hung" in caplog.text

    rec = Recorder()                               # the speaker is back: the next answer plays in full
    monkeypatch.setattr(tts, "open_output", rec)
    t.speak("next answer")
    assert t.last_engine == "piper" and not t.last_timed_out and len(rec.data) == 200 and rec.closed == 1
    release.set()                                  # the old worker wakes up: it must not touch the new one
    time.sleep(0.05)
    assert not t.speaking


def test_hung_device_open_times_out(monkeypatch, piper):
    """Opening the stream can hang as well: the deadline runs from the open."""
    monkeypatch.setattr(tts, "PLAY_S_PER_CHAR", 0.0)
    monkeypatch.setattr(tts, "PLAY_SLACK_S", 0.2)
    release = threading.Event()
    monkeypatch.setattr(tts, "open_output", lambda rate, device=None: release.wait() and HungOut(release))
    t = TTS(CFG, online(False))
    t0 = time.perf_counter()
    t.speak("hello")
    assert time.perf_counter() - t0 < 0.6 and not t.speaking and t.last_timed_out
    release.set()


def test_stop_returns_even_if_abort_hangs(monkeypatch, piper):
    monkeypatch.setattr(tts, "ABORT_S", 0.1)
    release = threading.Event()
    hung = HungOut(release, abort_ok=False)
    monkeypatch.setattr(tts, "open_output", lambda rate, device=None: hung)
    t = TTS(CFG, online(False))
    th = threading.Thread(target=t.speak, args=("a long answer " * 20,), daemon=True)
    th.start()
    time.sleep(0.05)
    t0 = time.perf_counter()
    t.stop()
    assert time.perf_counter() - t0 < 0.4
    release.set()
    th.join(1)
    assert not th.is_alive() and not t.speaking


# ------------------------------------------------------------------ Piper won't load

def test_piper_load_failure_is_retried_and_espeak_speaks_meanwhile(monkeypatch, audio, caplog):
    tries = []

    def broken(name, model_dir=None):
        tries.append(name)
        raise FileNotFoundError("models/piper/x.onnx missing")

    monkeypatch.setattr(tts, "load_piper", broken)
    monkeypatch.setattr(tts, "espeak_pcm", lambda text: (b"\x05\x00" * 30, 16000))
    t = TTS(CFG, online(False))
    assert t.warm() is False
    assert "failed to load" in caplog.text
    t.speak("still talking")
    assert t.last_engine == "espeak" and audio.opened == [16000] and len(audio.data) == 60
    good = FakeVoice()
    monkeypatch.setattr(tts, "load_piper", lambda name, model_dir=None: tries.append(name) or good)
    t.speak("piper again")
    assert t.last_engine == "piper" and good.texts == ["piper again"]
    assert len(tries) == 3                          # warm, first answer, second answer: each tried Piper


def test_no_voice_at_all_logs_clearly(monkeypatch, audio, caplog):
    monkeypatch.setattr(tts, "load_piper", lambda name, model_dir=None: (_ for _ in ()).throw(OSError("bad onnx")))
    monkeypatch.setattr(tts, "espeak_pcm", lambda text: None)
    t = TTS(CFG, online(False))
    t.speak("nobody hears this")
    assert t.last_engine is None and audio.opened == [] and not t.speaking
    assert "was not said" in caplog.text


class WedgedOut:
    """Like PortAudio: interrupt() (stream.abort) makes a blocked write return. Records, per call,
    whether a write was still in progress."""

    def __init__(self):
        self.unblock, self.calls, self.writing = threading.Event(), [], False

    def write(self, b):
        self.writing = True
        self.unblock.wait()
        time.sleep(0.02)                       # the write takes a moment to unwind
        self.writing = False

    def interrupt(self):
        self.calls.append(("interrupt", self.writing))
        self.unblock.set()

    def abort(self):
        self.calls.append(("abort", self.writing))
        self.unblock.set()

    def close(self):
        self.calls.append(("close", self.writing))


@pytest.mark.parametrize("how", ["deadline", "stop"])
def test_the_stream_is_never_closed_mid_write(monkeypatch, piper, how):
    """Closing a PortAudio stream another thread is blocked writing to is not allowed (ALSA may crash):
    the deadline and stop() only interrupt it; the worker, once its write returns, closes it."""
    monkeypatch.setattr(tts, "PLAY_S_PER_CHAR", 0.0)
    monkeypatch.setattr(tts, "PLAY_SLACK_S", 0.15)
    out = WedgedOut()
    monkeypatch.setattr(tts, "open_output", lambda rate, device=None: out)
    t = TTS(CFG, online(False))
    if how == "deadline":
        t.speak("wedged")
    else:
        th = threading.Thread(target=t.speak, args=("a long answer " * 20,), daemon=True)
        th.start()
        time.sleep(0.05)
        t.stop()
        th.join(1)
    assert wait_until(lambda: any(c == "abort" or c == "close" for c, _ in out.calls))
    assert out.calls[0] == ("interrupt", True)                  # from outside, mid-write: interrupt only
    assert all(not writing for c, writing in out.calls if c in ("abort", "close"))
    assert not t.speaking


def wait_until(cond, timeout=2.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if cond():
            return True
        time.sleep(0.01)
    return False


def test_audio_out_interrupt_does_not_close(monkeypatch):
    calls = []

    class Stream:
        def abort(self):
            calls.append("abort")

        def close(self):
            calls.append("close")

    out = tts.AudioOut.__new__(tts.AudioOut)
    out.stream = Stream()
    out.interrupt()
    assert calls == ["abort"]
    out.abort()
    assert calls == ["abort", "abort", "close"]


@pytest.mark.parametrize("chars", [10, 28, 82, 114, 200, 400])
def test_the_playback_deadline_leaves_room_for_a_bluetooth_speaker(chars):
    """Piper (en_US-lessac-medium) measured 16-19 characters a second on answers of 28-114 characters
    (laptop). A Bluetooth speaker through PulseAudio adds ~0.15-0.3 s, and the pulse plugin's buffer drains
    after the last write: with a second for both, the deadline still never cuts off a real answer."""
    speech_s = chars / 15.0                                     # slower than any measured answer
    assert tts.playback_budget_s("x" * chars) >= speech_s + 0.3 + 1.0


# -- long answers: Piper's memory grows with the square of a sentence's length (a 21,000-character
#    'what changed' list took the rig out of memory)

LONG_LIST = "The " + ", ".join(["thing I haven't been told about"] * 600) + " also changed."


def test_piper_gets_a_long_sentence_in_bounded_pieces(piper):
    pcm = list(tts.piper_chunks(piper, LONG_LIST))
    assert len(piper.texts) > 1 and max(map(len, piper.texts)) <= tts.PIPER_MAX_CHARS
    assert len(pcm) == 2 * len(piper.texts)
    assert " ".join(piper.texts).replace(" ,", ",") == LONG_LIST


def test_short_sentences_reach_piper_as_they_are(piper):
    list(tts.piper_chunks(piper, "Where are my keys? They are on the couch."))
    assert piper.texts == ["Where are my keys?", "They are on the couch."]


def test_a_piece_without_commas_is_cut_at_a_space_or_at_the_limit():
    words = tts.text_pieces("word " * 200, limit=50)
    assert all(len(p) <= 50 for p in words) and all(not p.startswith(" ") for p in words)
    assert all(len(p) <= 50 for p in tts.text_pieces("x" * 180, limit=50))


def test_speakable_keeps_whole_sentences_that_fit():
    assert tts.speakable("Short one. " * 80) == ("Short one. " * 54).strip()
    assert tts.speakable("Fine as is.") == "Fine as is."
    cut = tts.speakable(LONG_LIST)
    assert len(cut) <= tts.MAX_SPOKEN_CHARS and cut.endswith(".") and not cut.endswith(",.")


def test_speak_caps_a_long_answer(audio, piper, caplog, monkeypatch):
    monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)
    t = TTS(CFG, online(False))
    t.speak(LONG_LIST)
    assert sum(map(len, piper.texts)) <= tts.MAX_SPOKEN_CHARS
    assert "speaking only the first" in caplog.text


# -- the phone app's voice (Grok TTS by default)

def grok_key(monkeypatch, key="xk"):
    monkeypatch.setattr(tts.TTS, "_grok_key", staticmethod(lambda: key))


def test_online_the_rig_speaks_with_the_apps_default_grok_voice(monkeypatch, audio, piper):
    grok_key(monkeypatch)
    post = FakePost()
    monkeypatch.setattr(tts.requests, "post", post)
    t = TTS(CFG, online())
    t.speak("Your keys are on the couch.")
    assert t.last_engine == "grok" and piper.texts == []
    url, kw = post.calls[0]
    assert url == "https://api.x.ai/v1/tts" and kw["headers"]["Authorization"] == "Bearer xk"
    assert kw["json"] == {"text": "Your keys are on the couch.", "voice_id": "eve", "language": "en", "speed": 1.0,
                          "output_format": {"codec": "pcm", "sample_rate": 24000}}
    assert kw["stream"] is True and audio.opened == [24000] and audio.closed == 1


def test_the_phones_voice_is_used_and_kept_across_restarts(monkeypatch, audio, piper):
    grok_key(monkeypatch)
    post = FakePost()
    monkeypatch.setattr(tts.requests, "post", post)
    t = TTS(CFG, online())
    assert t.set_voice("grok", "Ara", 1.3) == tts.VoiceChoice("grok", "ara", 1.3)
    t2 = TTS(CFG, online())                                   # a restart reads data/voice.json
    t2.speak("Hello.")
    assert post.calls[0][1]["json"]["voice_id"] == "ara" and post.calls[0][1]["json"]["speed"] == 1.3


def test_voice_settings_are_validated():
    V = tts.VoiceChoice.make
    assert V("nonsense", "", "fast") == tts.VoiceChoice("grok", "eve", 1.0)
    assert V("rig", "Rex!!", 9) == tts.VoiceChoice("rig", "rex", 1.5)
    assert V("builtin", None, 0.1).speed == 0.7 and V(speed=float("nan")).speed == 1.0


def test_grok_failing_falls_back_to_piper(monkeypatch, audio, piper):
    grok_key(monkeypatch)
    monkeypatch.setattr(tts.requests, "post", FakePost(resp=FakeResp(status=401)))
    t = TTS(CFG, online())
    t.speak("Fallback please.")
    assert t.last_engine == "piper" and piper.texts == ["Fallback please."]


def test_offline_grok_is_not_tried(monkeypatch, audio, piper):
    grok_key(monkeypatch)
    monkeypatch.setattr(tts.requests, "post", lambda *a, **k: pytest.fail("no network offline"))
    t = TTS(CFG, online(False))
    t.speak("Offline.")
    assert t.last_engine == "piper"


def test_same_as_the_rig_picks_elevenlabs_and_the_iphone_voice_picks_piper(monkeypatch, audio, piper, eleven_env):
    grok_key(monkeypatch)
    post = FakePost()
    monkeypatch.setattr(tts.requests, "post", post)
    t = TTS(CFG, online())
    t.set_voice("rig")
    t.speak("One.")
    assert t.last_engine == "elevenlabs" and "elevenlabs" in post.calls[0][0]
    t.set_voice("builtin")
    t.speak("Two.")
    assert t.last_engine == "piper" and len(post.calls) == 1


# -- is the rig's speaker connected (status 'spk': the phone stays quiet)

def test_speaker_through_pulseaudio_is_a_bluetooth_or_usb_sink(monkeypatch):
    monkeypatch.setenv("PULSE_SERVER", "unix:/run/user/1000/pulse/native")
    cfg = dict(CFG, tts=dict(CFG["tts"], output_device="pulse"))
    for sink, ok in (("bluez_sink.2C_41_A1_76_9C_F0.a2dp_sink", True),
                     ("alsa_output.usb-Generic_USB2.0_Audio-00.analog-stereo", True),
                     ("alsa_output.platform-sound.analog-stereo", False), (None, False)):
        monkeypatch.setattr(tts, "pulse_default_sink", lambda s=sink: s)
        assert TTS(cfg).speaker_status() == {"ok": ok, "name": sink or ""}, sink


def test_speaker_on_a_named_alsa_device_is_there_when_plugged_in(monkeypatch):
    monkeypatch.delenv("PULSE_SERVER", raising=False)
    monkeypatch.setattr(tts, "output_devices", lambda: [{"name": "Jabra SPEAK 410: USB Audio (hw:3,0)", "index": 3}])
    assert TTS(dict(CFG, tts=dict(CFG["tts"], output_device="jabra"))).speaker_status()["ok"] is True
    assert TTS(dict(CFG, tts=dict(CFG["tts"], output_device="bose"))).speaker_status()["ok"] is False
    assert TTS(dict(CFG, tts=dict(CFG["tts"], output_device=None))).speaker_status()["ok"] is False  # HDMI


def test_state_carries_the_speaker_and_the_phones_voice(monkeypatch):
    monkeypatch.setenv("PULSE_SERVER", "unix:/x")
    monkeypatch.setattr(tts, "pulse_default_sink", lambda: "bluez_sink.b.a2dp_sink")
    world = NS(state_json=lambda: {"entities": []})
    t = TTS(dict(CFG, tts=dict(CFG["tts"], output_device="pulse"))).attach(world)
    t.set_voice("grok", "rex", 1.0)
    st = world.state_json()
    assert st["speaker"]["ok"] is True and st["voice"] == {"engine": "grok", "grok_voice": "rex", "speed": 1.0}
