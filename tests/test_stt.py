import json
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from core.config import load_config
from voice import stt
from voice.intents import parse
from voice.stt import BLOCK, RATE, STT
from voice.trigger import Clicker

CFG = load_config()
HERE = Path(__file__).resolve().parent
QUESTIONS = json.loads((HERE / "stt_questions.json").read_text())


class FakeMic:
    """Stands in for open_input(): plays blocks from a list, then None (no audio)."""

    def __init__(self, blocks):
        self.blocks = list(blocks)
        self.opened = self.closed = 0

    def __call__(self, rate, block, device=None):
        assert (rate, block) == (RATE, BLOCK)
        self.opened += 1
        mic = self

        class In:
            def read(self, timeout=1.0):
                return mic.blocks.pop(0) if mic.blocks else None

            def close(self):
                mic.closed += 1

        return In()


def loud(n):
    return [np.full(BLOCK, 0.5, np.float32) for _ in range(n)]


def quiet(n):
    return [np.zeros(BLOCK, np.float32) for _ in range(n)]


class AmpVAD:
    """Speech probability = block amplitude > 0.1 (the real one is Silero)."""

    def reset(self):
        self.calls = 0

    def __call__(self, b):
        self.calls += 1
        return 0.9 if np.abs(b).max() > 0.1 else 0.05


class FakeBackend:
    def __init__(self, text="where are my keys"):
        self.text, self.calls = text, []

    def transcribe(self, audio, prompt, audio_ctx):
        self.calls.append((len(audio), prompt, audio_ctx))
        return self.text


def make(monkeypatch, blocks, clicker=None, **stt_cfg):
    mic = FakeMic(blocks)
    monkeypatch.setattr(stt, "open_input", mic)
    cfg = dict(CFG, stt={**(CFG.get("stt") or {}), **stt_cfg})
    return STT(cfg, clicker=clicker, backend=FakeBackend(), vad=AmpVAD()), mic


# ---------------------------------------------------------------- helpers

def test_audio_ctx_scales_with_clip_and_caps_at_30s():
    assert stt.audio_ctx_for(3 * RATE) == 150 + stt.CTX_PAD
    assert stt.audio_ctx_for(RATE) == 50 + stt.CTX_PAD
    assert stt.audio_ctx_for(60 * RATE) == 1500


def test_prompt_lists_every_object_by_spoken_name():
    p = stt.initial_prompt(CFG)
    for name in ["keys", "pill bottle", "wallet", "glasses", "phone", "remote", "box", "notebook"]:
        assert name in p
    assert "pill_bottle" not in p


def test_clean_drops_non_speech_tags():
    assert stt.clean(" [BLANK_AUDIO] ") == ""
    assert stt.clean("(music) Where are  my keys? [inaudible]") == "Where are my keys?"


def test_wav_roundtrip(tmp_path):
    a = (np.sin(np.arange(RATE) / 10) * 0.3).astype(np.float32)
    stt.write_wav(tmp_path / "a.wav", a)
    b = stt.read_wav(tmp_path / "a.wav")
    assert len(b) == len(a) and np.abs(a - b).max() < 1e-3


# ---------------------------------------------------------------- recording

def test_stops_after_silence_and_trims(monkeypatch):
    s, mic = make(monkeypatch, quiet(30) + loud(40) + quiet(100), silence_ms=700)
    audio = s.record_until_silence()
    assert s.last_speech and mic.opened == mic.closed == 1
    pre, tail = int(stt.PREROLL_S * RATE / BLOCK), int(stt.TAIL_S * RATE / BLOCK)
    assert len(audio) == (pre + 40 + tail) * BLOCK
    # stopped ~700 ms after speech, not at max_s
    assert len(mic.blocks) == 100 - int(np.ceil(0.7 * RATE / BLOCK))


def test_short_pauses_do_not_end_the_question(monkeypatch):
    s, _ = make(monkeypatch, loud(20) + quiet(10) + loud(20) + quiet(60), silence_ms=700)
    audio = s.record_until_silence()
    assert len(audio) >= 50 * BLOCK


def test_no_speech_returns_empty_and_skips_whisper(monkeypatch):
    s, mic = make(monkeypatch, quiet(500), no_speech_s=2)
    audio = s.record_until_silence()
    assert len(audio) == 0 and not s.last_speech
    assert len(mic.blocks) == 500 - (int(np.ceil(2 * RATE / BLOCK)) + 1)
    assert s.transcribe(audio) == "" and s.backend.calls == []


def test_max_s_caps_recording(monkeypatch):
    s, _ = make(monkeypatch, loud(1000))
    audio = s.record_until_silence(max_s=2)
    assert len(audio) <= int(np.ceil(2 * RATE / BLOCK)) * BLOCK


def test_second_click_stops_recording(monkeypatch):
    clicker = Clicker(keyboard=True)
    clicker.close()                  # no stdin reader; presses only by press()
    s, mic = make(monkeypatch, loud(300), clicker=clicker)
    clicker.press()
    audio = s.record_until_silence()
    assert len(audio) > 0 and len(mic.blocks) == 299


def test_mic_with_no_audio_returns(monkeypatch):
    s, _ = make(monkeypatch, [])
    assert len(s.record_until_silence()) == 0


# ---------------------------------------------------------------- transcription

def test_transcribe_passes_prompt_and_clip_sized_ctx():
    b = FakeBackend(" [BLANK_AUDIO] Where are my keys? ")
    s = STT(CFG, backend=b, vad=AmpVAD())
    assert s.transcribe(np.zeros(3 * RATE, np.float32)) == "Where are my keys?"
    n, prompt, ctx = b.calls[0]
    assert n == 3 * RATE and ctx == stt.audio_ctx_for(3 * RATE) and "pill bottle" in prompt


def test_short_clip_is_padded_to_whisper_minimum():
    b = FakeBackend()
    STT(CFG, backend=b, vad=AmpVAD()).transcribe(np.ones(RATE // 4, np.float32) * 0.1)
    assert b.calls[0][0] == int(stt.MIN_WHISPER_S * RATE)


def test_backend_choice():
    with pytest.raises(ValueError):
        stt.make_backend({"stt": {"backend": "nope"}})
    with pytest.raises(FileNotFoundError):
        stt.make_backend({"stt": {"backend": "cli", "model": "/nonexistent/ggml-base.en.bin"}})


# ---------------------------------------------------------------- the 20 questions

def test_question_set_has_twenty():
    assert len(QUESTIONS) == 20


@pytest.mark.parametrize("q", QUESTIONS, ids=[q["text"] for q in QUESTIONS])
def test_question_text_parses(q):
    it = parse(q["text"], CFG)
    assert (it.kind, it.obj) == (q["intent"], q["obj"])


@pytest.mark.parametrize("q", QUESTIONS, ids=[q["text"] for q in QUESTIONS])
def test_fake_transcript_through_stt_parses(q):
    s = STT(CFG, backend=FakeBackend(q["text"]), vad=AmpVAD())
    it = parse(s.transcribe(np.ones(2 * RATE, np.float32) * 0.1), CFG)
    assert (it.kind, it.obj) == (q["intent"], q["obj"])


# ---------------------------------------------------------------- real models (opt-in)

AUDIO = os.environ.get("ASKROOM_STT_AUDIO") == "1"
SAY = shutil.which("say")


def clip(i: int, text: str, tmp: Path) -> Path:
    p = HERE / "stt_audio" / f"{i:02d}.wav"
    if p.exists():
        return p
    if not SAY:
        pytest.skip(f"no {p} and no macOS say to synthesize it")
    p = tmp / f"{i:02d}.wav"
    subprocess.run([SAY, "-o", str(p), "--data-format=LEI16@16000", text], check=True)
    return p


@pytest.fixture(scope="module")
def real_stt():
    if not AUDIO:
        pytest.skip("set ASKROOM_STT_AUDIO=1 to run Whisper on the 20 spoken questions")
    s = STT(CFG)
    s.warm()
    return s


@pytest.mark.parametrize("i", range(1, 21))
def test_spoken_question_parses(real_stt, i, tmp_path):
    q = QUESTIONS[i - 1]
    text = real_stt.transcribe(stt.read_wav(clip(i, q["text"], tmp_path)))
    it = parse(text, CFG)
    assert (it.kind, it.obj) == (q["intent"], q["obj"]), text


@pytest.mark.skipif(not stt.VAD_PATH.exists(), reason="models/silero_vad.onnx not downloaded")
def test_silero_hears_speech_not_silence_or_hum(tmp_path):
    vad = stt.SileroVAD()
    t = np.arange(2 * RATE) / RATE
    for a in (np.zeros(2 * RATE, np.float32), (0.2 * np.sin(2 * np.pi * 120 * t)).astype(np.float32)):
        vad.reset()
        assert max(vad(a[i:i + BLOCK]) for i in range(0, len(a) - BLOCK, BLOCK)) < 0.5
    if SAY:
        a = stt.read_wav(clip(1, QUESTIONS[0]["text"], tmp_path))
        vad.reset()
        assert max(vad(a[i:i + BLOCK]) for i in range(0, len(a) - BLOCK, BLOCK)) > 0.5
