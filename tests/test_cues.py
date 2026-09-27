"""Listening cues (voice/cues.py): the "speak now" chime and the listening light."""
import sys
import time
import types

import numpy as np
import pytest

from core.config import load_config
from tests.test_main import FakeClicker, FakeSTT, always_on, cal_path, make_room, stop_voice, wait_for  # noqa: F401 (fixture)
from voice import cues, tts
from voice.tts import TTS

CFG = load_config()


def test_chimes_start_silent_and_stay_in_range():
    for kind in ("listen", "ack"):
        pcm = np.frombuffer(cues.chime_pcm(kind), np.int16)
        lead = int(cues.LEAD_S * cues.RATE)
        assert not pcm[:lead].any()                               # an idle Bluetooth speaker clips this part
        assert np.abs(pcm[lead:]).max() > 3000 and np.abs(pcm).max() < 32767
        assert abs(int(pcm[-1])) < 200                            # faded out: no click
    assert len(cues.chime_pcm("ack")) < len(cues.chime_pcm("listen"))
    assert len(cues.chime_pcm("listen")) / 2 / cues.RATE < 0.6


def test_indicator_without_leds_just_keeps_state():
    ind = cues.ListenIndicator(CFG)
    assert ind.backend == "none"
    ind.on()
    assert ind.lit
    ind.off()
    ind.close()
    assert not ind.lit


class FakeLine:
    def __init__(self):
        self.values, self.released = [], False

    def request(self, consumer, type, default_vals):
        self.values.append(default_vals[0])

    def set_value(self, v):
        self.values.append(v)

    def release(self):
        self.released = True


def gpio_cfg(**kw):
    return dict(CFG, listen=dict(CFG.get("listen") or {}, indicator=dict(backend="gpio", gpio_chip="gpiochip0", **kw)))


def test_indicator_drives_a_gpio_line(monkeypatch):
    line = FakeLine()
    monkeypatch.setitem(sys.modules, "gpiod", types.SimpleNamespace(
        Chip=lambda name: types.SimpleNamespace(get_line=lambda n: line), LINE_REQ_DIR_OUT=1))
    ind = cues.ListenIndicator(gpio_cfg(gpio_line=12, active_low=True))
    assert ind.backend == "gpio"
    ind.on()
    ind.on()                                                      # no repeat writes
    ind.off()
    ind.close()
    assert line.values == [1, 0, 1] and line.released             # active low: 1 is dark


def test_an_unusable_gpio_leaves_the_light_off_but_never_raises(monkeypatch):
    monkeypatch.setitem(sys.modules, "gpiod", types.SimpleNamespace(
        Chip=lambda name: (_ for _ in ()).throw(PermissionError("no access")), LINE_REQ_DIR_OUT=1))
    ind = cues.ListenIndicator(gpio_cfg(gpio_line=12))
    assert ind.backend == "none"
    ind.on()
    ind.off()
    assert cues.ListenIndicator(gpio_cfg()).backend == "none"     # no line given


# -- TTS.play_cue

def test_play_cue_plays_on_the_speech_output(monkeypatch):
    got = []

    class Out:
        def write(self, b):
            got.append(len(b))

        def close(self):
            got.append("closed")

    monkeypatch.setattr(tts, "open_output", lambda rate, device=None: got.append(rate) or Out())
    assert TTS(CFG).play_cue(cues.chime_pcm("ack"), cues.RATE) is True
    assert got[0] == cues.RATE and got[-1] == "closed"


def test_play_cue_never_overlaps_an_answer_or_hangs(monkeypatch):
    t = TTS(CFG)
    with t._lock:                                                 # an answer is playing
        assert t.play_cue(b"\0\0" * 10, cues.RATE) is False
    monkeypatch.setattr(tts, "open_output", lambda rate, device=None: time.sleep(5))
    t0 = time.monotonic()
    assert t.play_cue(b"\0\0" * 10, cues.RATE, deadline_s=0.2) is False
    assert time.monotonic() - t0 < 1.0 and not t.speaking


# -- the voice loop

class CueLog:
    """Stands in for the room's TTS cue player and indicator; records the order of events."""

    def __init__(self, room):
        self.events = []
        room.tts.play_cue = lambda pcm, rate: self.events.append(("cue", len(pcm))) or True
        room.indicator.set = lambda on: self.events.append(("light", on))
        listen = room.stt.listen

        def listening():
            self.events.append(("listen", None))
            return listen()
        room.stt.listen = listening


def test_the_wake_word_alone_chimes_and_lights_up_while_listening(tmp_path, cal_path):
    stt = FakeSTT("where is my wallet", overheard=["Hey Room!"])
    room, _ = make_room(tmp_path, cal_path, stt=stt, clicker=FakeClicker())
    log = CueLog(room)
    t = always_on(room)
    assert wait_for(lambda: room.tts.said)
    stop_voice(room, t)
    kinds = [e[0] for e in log.events]
    assert kinds[:4] == ["cue", "light", "listen", "light"]
    assert log.events[0][1] == len(cues.chime_pcm("listen")) and log.events[1] == ("light", True)
    assert log.events[3] == ("light", False) and "wallet" in room.tts.said[0].lower()


def test_room_and_the_question_in_one_breath_gets_the_short_tone(tmp_path, cal_path):
    stt = FakeSTT("", overheard=["room, where is my wallet?"])
    room, _ = make_room(tmp_path, cal_path, stt=stt, clicker=FakeClicker())
    log = CueLog(room)
    t = always_on(room)
    assert wait_for(lambda: room.tts.said)
    stop_voice(room, t)
    assert log.events[0] == ("cue", len(cues.chime_pcm("ack")))
    assert ("listen", None) not in log.events and not any(e[0] == "light" for e in log.events)


@pytest.mark.parametrize("text", ["where did I leave my wallet in the room?", "did anyone move my wallet in this room"])
def test_room_later_in_the_sentence_gets_no_short_tone(tmp_path, cal_path, text):
    """The ack chime uses the answering rule (voice.understand.has_wake_word): "room" anywhere used to chime."""
    stt = FakeSTT("", overheard=[text])
    room, _ = make_room(tmp_path, cal_path, stt=stt, clicker=FakeClicker())
    log = CueLog(room)
    t = always_on(room)
    assert wait_for(lambda: room.tts.said)
    stop_voice(room, t)
    assert not any(e[0] == "cue" for e in log.events) and "wallet" in room.tts.said[0].lower()


def test_the_light_goes_off_when_the_mic_fails(tmp_path, cal_path):
    room, _ = make_room(tmp_path, cal_path, stt=FakeSTT(""), clicker=FakeClicker())
    log = CueLog(room)
    room.stt.listen = lambda: (_ for _ in ()).throw(OSError("mic gone"))
    room._asked(time.monotonic())
    assert log.events[-1] == ("light", False)


def test_no_chime_when_turned_off(tmp_path, cal_path):
    room, _ = make_room(tmp_path, cal_path, stt=FakeSTT("where is my wallet"), clicker=FakeClicker())
    log = CueLog(room)
    room.chime = False
    room._asked(time.monotonic())
    assert not any(e[0] == "cue" for e in log.events) and ("light", True) in log.events


# -- the speaker keep-alive (a Bluetooth speaker switches itself off after ~20 min of silence)

def test_keepalive_sound_is_quiet_shaped_noise_with_soft_edges():
    pcm = np.frombuffer(cues.keepalive_pcm(-50, 0.3), np.int16).astype(np.float64) / 32767
    rms = 20 * np.log10(np.sqrt(np.mean(pcm ** 2)))
    assert len(pcm) == int(0.3 * cues.RATE) and abs(rms + 50) < 0.5
    assert np.abs(pcm[:5]).max() < 1e-3 and np.abs(pcm[-5:]).max() < 1e-3 and np.count_nonzero(pcm) > len(pcm) // 2
    assert 20 * np.log10(np.sqrt(np.mean((np.frombuffer(cues.keepalive_pcm(-45), np.int16) / 32767.0) ** 2))) > -46


def keepalive_tts(monkeypatch, **kw):
    got = []

    class Out:
        def write(self, b):
            got.append(len(b))

        def close(self):
            pass

    monkeypatch.setattr(tts, "open_output", lambda rate, device=None: Out())
    return TTS(dict(CFG, tts=dict(CFG.get("tts") or {}, **kw))), got


def test_keepalive_plays_only_after_keepalive_s_of_silence(monkeypatch):
    t, got = keepalive_tts(monkeypatch, keepalive_s=480)
    assert t.keepalive() is False and got == []                  # just started
    t.last_audio_t -= 481
    assert t.keepalive() is True and got == [2 * int(0.3 * cues.RATE)]
    assert t.keepalive() is False and len(got) == 1              # the clock restarted
    t.last_audio_t -= 481
    t.play_cue(cues.chime_pcm("ack"), cues.RATE)                 # a chime counts as sound
    assert t.keepalive() is False


def test_keepalive_never_waits_for_or_overlaps_an_answer(monkeypatch):
    t, got = keepalive_tts(monkeypatch, keepalive_s=480)
    t.last_audio_t -= 481
    with t._lock:                                                # an answer is playing
        t0 = time.monotonic()
        assert t.keepalive() is False and time.monotonic() - t0 < 0.05 and got == []
    t2, got2 = keepalive_tts(monkeypatch, keepalive_s=0)        # off
    t2.last_audio_t -= 10_000
    assert t2.keepalive() is False and got2 == []


def test_keepalive_on_a_stalled_speaker_gives_up_within_its_deadline(monkeypatch):
    t, _ = keepalive_tts(monkeypatch, keepalive_s=480)
    monkeypatch.setattr(tts, "open_output", lambda rate, device=None: time.sleep(5))
    t.last_audio_t -= 481
    t0 = time.monotonic()
    assert t.keepalive() is False and time.monotonic() - t0 < 2.0 and not t.speaking


def test_the_voice_loop_plays_the_keepalive_between_turns_never_while_listening(tmp_path, cal_path):
    """Run it only when the voice loop is idle: never while the mic records or waits for the question after
    "Room!", and the mic waits the chime's tail after it, so it can't hear the keep-alive."""
    events = []
    stt = FakeSTT("where is my wallet", overheard=["Room!", "hello there"])
    room, _ = make_room(tmp_path, cal_path, stt=stt, clicker=FakeClicker())
    room.tts.keepalive = lambda: events.append("keepalive") or True
    hear, listen = stt.hear, stt.listen
    stt.hear = lambda idle_s: events.append("hear") or hear(idle_s)
    stt.listen = lambda: events.append("listen") or listen()
    t = always_on(room)
    assert wait_for(lambda: room.tts.said and events.count("hear") >= 3)
    stop_voice(room, t)
    assert events[:5] == ["keepalive", "hear", "listen", "keepalive", "hear"]
    assert all(events[i] == "keepalive" for i in range(len(events)) if events[i - 1:i] == ["listen"])
