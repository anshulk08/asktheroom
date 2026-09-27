import json
import os
import shutil
import subprocess
import time
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
    # pinned to sized: config.yaml now sets audio_ctx 0 (sized garbled transcripts on the Jetson)
    s = STT(dict(CFG, stt={**CFG["stt"], "audio_ctx": "sized"}), backend=b, vad=AmpVAD())
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


# ---------------------------------------------------------------- whisper-server backend, auto pick

import io  # noqa: E402
import sys  # noqa: E402
import threading  # noqa: E402
import wave  # noqa: E402
from email import policy  # noqa: E402
from email.parser import BytesParser  # noqa: E402
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer  # noqa: E402


class FakeWhisperServer:
    """whisper-server's two endpoints: GET /health, POST /inference (multipart). Records requests."""

    def __init__(self, text=" Where are my keys?\n", status="ok"):
        self.requests = []
        fake = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _send(self, code, body):
                b = json.dumps(body).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(b)))
                self.end_headers()
                self.wfile.write(b)

            def do_GET(self):
                if self.path == "/health":
                    self._send(200 if status == "ok" else 503, {"status": status})
                else:
                    self._send(404, {})

            def do_POST(self):
                body = self.rfile.read(int(self.headers["Content-Length"]))
                msg = BytesParser(policy=policy.default).parsebytes(
                    b"Content-Type: " + self.headers["Content-Type"].encode() + b"\r\n\r\n" + body)
                fields = {}
                for part in msg.iter_parts():
                    name = part.get_param("name", header="content-disposition")
                    fields[name] = part.get_payload(decode=True)
                fake.requests.append((self.path, fields))
                self._send(200, {"text": fake.text})

        self.text = text
        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.port = self.httpd.server_address[1]
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()


@pytest.fixture
def fake_server():
    s = FakeWhisperServer()
    yield s
    s.close()


def free_port() -> int:
    import socket
    with socket.socket() as so:
        so.bind(("127.0.0.1", 0))
        return so.getsockname()[1]


def test_model_name_with_dots_maps_to_ggml_file():
    # 'base.en' has Path.suffix '.en'; it must still mean models/whisper/ggml-base.en.bin
    assert stt.model_path("base.en") == stt.WHISPER_DIR / "ggml-base.en.bin"
    assert stt.model_path("tiny.en") == stt.WHISPER_DIR / "ggml-tiny.en.bin"
    assert stt.model_path("/x/ggml-small.bin") == Path("/x/ggml-small.bin")


def test_find_binary_looks_in_build_dirs(tmp_path, monkeypatch):
    exe = tmp_path / "build-mac" / "bin" / "whisper-server"
    exe.parent.mkdir(parents=True)
    exe.write_text("#!/bin/sh\n")
    exe.chmod(0o755)
    monkeypatch.setattr(stt, "WHISPER_BUILDS", [tmp_path / "build" / "bin", exe.parent])
    monkeypatch.setenv("PATH", str(tmp_path / "empty"))
    assert stt.find_binary("whisper-server") == str(exe)
    assert stt.find_binary(str(exe)) == str(exe)
    assert stt.find_binary("whisper-nothing") is None


def test_write_wav_to_file_object():
    buf = io.BytesIO()
    stt.write_wav(buf, np.zeros(RATE, np.float32))
    with wave.open(io.BytesIO(buf.getvalue())) as w:
        assert (w.getframerate(), w.getnchannels(), w.getnframes()) == (RATE, 1, RATE)


def test_server_backend_reuses_running_server_and_posts_fields(fake_server):
    b = stt.WhisperServerBackend("base.en", port=fake_server.port)
    assert b.proc is None                      # healthy server already there: not spawned
    s = STT(dict(CFG, stt={**CFG["stt"], "audio_ctx": "sized"}), backend=b, vad=AmpVAD())
    assert s.transcribe(np.ones(3 * RATE, np.float32) * 0.1) == "Where are my keys?"
    path, f = fake_server.requests[0]
    assert path == "/inference"
    assert f["prompt"].decode() == s.prompt
    assert int(f["audio_ctx"]) == stt.audio_ctx_for(3 * RATE)
    assert f["response_format"] == b"json" and f["no_timestamps"] == b"true"
    with wave.open(io.BytesIO(f["file"])) as w:
        assert (w.getframerate(), w.getnframes()) == (RATE, 3 * RATE)


def test_server_down_falls_back_to_cli(tmp_path):
    fb = FakeBackend("fallback text")
    b = stt.WhisperServerBackend("base.en", port=free_port(), fallback=fb, start=False)
    assert not b.healthy()
    assert b.transcribe(np.zeros(RATE, np.float32), "p", 64) == "fallback text"
    assert b.fallbacks == 1 and fb.calls == [(RATE, "p", 64)]


def test_server_down_without_fallback_raises():
    import requests
    b = stt.WhisperServerBackend("base.en", port=free_port(), start=False)
    with pytest.raises(requests.RequestException):
        b.transcribe(np.zeros(RATE, np.float32), "p", 64)


FAKE_SERVER_EXE = r'''#!{py}
import json, sys
from http.server import BaseHTTPRequestHandler, HTTPServer
port = int(sys.argv[sys.argv.index("--port") + 1])
if "--crash" in open(sys.argv[sys.argv.index("-m") + 1]).read():
    sys.stderr.write("model load failed\n"); sys.exit(3)
class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def _send(self, body):
        b = json.dumps(body).encode()
        self.send_response(200); self.send_header("Content-Length", str(len(b))); self.end_headers()
        self.wfile.write(b)
    def do_GET(self): self._send({{"status": "ok"}})
    def do_POST(self):
        self.rfile.read(int(self.headers["Content-Length"])); self._send({{"text": " spawned"}})
HTTPServer(("127.0.0.1", port), H).serve_forever()
'''


def fake_exe(tmp_path, model_text=""):
    exe = tmp_path / "whisper-server"
    exe.write_text(FAKE_SERVER_EXE.format(py=sys.executable))
    exe.chmod(0o755)
    model = tmp_path / "ggml-fake.bin"
    model.write_text(model_text)
    return exe, model


def test_server_backend_spawns_waits_for_health_and_closes(tmp_path):
    exe, model = fake_exe(tmp_path)
    b = stt.WhisperServerBackend(str(model), binary=str(exe), port=free_port(), start_timeout=20)
    try:
        assert b.proc is not None and b.proc.poll() is None and b.healthy()
        assert b.transcribe(np.zeros(RATE, np.float32), "p", 64) == " spawned"
    finally:
        b.close()
    assert b.proc is None and not b.healthy()


def test_server_that_dies_on_start_raises_with_its_log(tmp_path):
    exe, model = fake_exe(tmp_path, "--crash")
    with pytest.raises(RuntimeError, match="model load failed"):
        stt.WhisperServerBackend(str(model), binary=str(exe), port=free_port(), start_timeout=20)


def test_auto_prefers_server_then_cli_then_pywhispercpp(monkeypatch):
    made = []

    class Srv:
        def __init__(self, *a, **k):
            made.append(("server", k.get("fallback")))

    class Cli:
        def __init__(self, *a, **k):
            made.append(("cli", None))

    class Py:
        def __init__(self, *a, **k):
            made.append(("py", None))

    monkeypatch.setattr(stt, "WhisperServerBackend", Srv)
    monkeypatch.setattr(stt, "WhisperCliBackend", Cli)
    monkeypatch.setattr(stt, "PyWhisperCppBackend", Py)
    monkeypatch.setattr(stt, "model_path", lambda m, d=None: Path(__file__))   # "model exists"
    have = {"whisper-server", "whisper-cli"}
    monkeypatch.setattr(stt, "find_binary", lambda n: f"/bin/{n}" if n in have else None)
    auto = {"stt": {"backend": "auto"}}

    assert isinstance(stt.make_backend(auto), Srv)
    assert isinstance(made[-1][1], Cli)          # cli is the server's fallback
    have.discard("whisper-server")
    assert isinstance(stt.make_backend(auto), Cli)

    def no_cli(*a, **k):
        raise FileNotFoundError("no whisper-cli")
    monkeypatch.setattr(stt, "WhisperCliBackend", no_cli)
    assert isinstance(stt.make_backend(auto), Py)


def test_auto_falls_back_to_cli_when_server_will_not_start(monkeypatch):
    class Boom:
        def __init__(self, *a, **k):
            raise RuntimeError("did not come up")

    class Cli:
        def __init__(self, *a, **k):
            pass

    monkeypatch.setattr(stt, "WhisperServerBackend", Boom)
    monkeypatch.setattr(stt, "WhisperCliBackend", Cli)
    monkeypatch.setattr(stt, "model_path", lambda m, d=None: Path(__file__))
    monkeypatch.setattr(stt, "find_binary", lambda n: f"/bin/{n}")
    for kind in ("auto", "server"):
        assert isinstance(stt.make_backend({"stt": {"backend": kind}}), Cli)


def test_config_backend_is_known():
    assert str(CFG["stt"]["backend"]) in set(stt.BACKENDS) | {"auto"}


# ---------------------------------------------------------------- crowd noise (end_drop_db)

class AlwaysSpeechVAD(AmpVAD):
    """Silero in a crowd: background talkers score as speech whatever their level."""

    def __call__(self, b):
        return 0.95


def crowd(n, amp):
    return [np.full(BLOCK, amp, np.float32) for _ in range(n)]


def test_crowd_keeps_silero_alone_recording_to_max_s(monkeypatch):
    s, mic = make(monkeypatch, crowd(10, 0.03) + loud(40) + crowd(400, 0.03), end_drop_db=None)
    s._vad = AlwaysSpeechVAD()
    audio = s.record_until_silence(max_s=4)
    assert len(audio) >= int(np.ceil(4 * RATE / BLOCK)) * BLOCK - BLOCK


def test_level_gate_ends_question_when_asker_stops_in_a_crowd(monkeypatch):
    # asker at 0.5, crowd at 0.03 (-24 dB): 10 dB below the asker's loud blocks counts as quiet
    s, mic = make(monkeypatch, crowd(10, 0.03) + loud(40) + crowd(400, 0.03), end_drop_db=10)
    s._vad = AlwaysSpeechVAD()
    audio = s.record_until_silence(max_s=6)
    need = int(np.ceil(0.7 * RATE / BLOCK))
    assert len(mic.blocks) == 400 - need                 # stopped silence_ms after the asker
    assert len(audio) <= (10 + 40 + int(stt.TAIL_S * RATE / BLOCK)) * BLOCK


def test_level_gate_keeps_a_quieter_last_word(monkeypatch):
    # last word 6 dB softer than the rest: still speech at end_drop_db=10
    s, _ = make(monkeypatch, quiet(5) + loud(30) + crowd(15, 0.25) + quiet(60), end_drop_db=10)
    audio = s.record_until_silence()
    assert len(audio) >= 45 * BLOCK


def test_level_gate_off_by_default_in_code():
    assert STT({"stt": {}}, backend=FakeBackend(), vad=AmpVAD()).end_drop_db is None


def test_audio_ctx_config_sized_or_fixed():
    for setting, want in (("sized", stt.audio_ctx_for(3 * RATE)), (None, stt.audio_ctx_for(3 * RATE)),
                          (0, 0), (768, 768)):
        b = FakeBackend()
        STT({"stt": {"audio_ctx": setting}}, backend=b, vad=AmpVAD()).transcribe(np.ones(3 * RATE, np.float32))
        assert b.calls[0][2] == want, setting


def test_prompt_can_include_synonyms():
    p = stt.initial_prompt(CFG, synonyms=True)
    for w in ["keys", "pill bottle", "meds", "specs", "cell phone", "box"]:
        assert w in p
    assert "the box" not in p and p.count(" box,") == 1
    assert "meds" not in stt.initial_prompt(CFG)
    assert STT(CFG, backend=FakeBackend(), vad=AmpVAD()).prompt == stt.initial_prompt(
        CFG, synonyms=bool(CFG["stt"].get("prompt_synonyms")))


# ---------------------------------------------------------------- the rig's own voice

class Speaking:
    """Stands in for voice.tts.TTS: .speaking follows a script, one entry per mic block read."""

    def __init__(self, script):
        self.script, self.i = list(script), 0

    @property
    def speaking(self):
        v = self.script[min(self.i, len(self.script) - 1)]
        self.i += 1
        return v


def test_tts_starting_mid_recording_discards_the_clip(monkeypatch):
    s, mic = make(monkeypatch, loud(20) + quiet(100))
    s.tts = Speaking([False] * 11 + [True])          # a care notice starts 10 blocks into the question
    audio = s.record_until_silence()
    assert len(audio) == 0 and s.last_stop == "tts" and not s.last_speech
    assert mic.closed == 1 and len(mic.blocks) == 109   # stopped at block 11, not after max_s


def test_hear_returns_nothing_when_the_rig_talks_over_it(monkeypatch):
    s, _ = make(monkeypatch, quiet(5) + loud(40) + quiet(100))
    s.tts = Speaking([False] * 20 + [True])
    assert s.hear(idle_s=8) == "" and s.backend.calls == []


def test_speech_already_playing_at_the_start_only_counts_once_it_stopped(monkeypatch):
    """A click cuts the answer and listens at once: the cut answer's tail doesn't end the question,
    but a new answer starting later does."""
    s, _ = make(monkeypatch, loud(40) + quiet(100))
    s.tts = Speaking([True] * 3 + [False])
    assert len(s.record_until_silence()) > 0 and s.last_stop == "silence"
    s2, _ = make(monkeypatch, loud(40) + quiet(100))
    s2.tts = Speaking([True] * 3 + [False] * 5 + [True])
    assert len(s2.record_until_silence()) == 0 and s2.last_stop == "tts"


def test_tts_passed_to_the_constructor(monkeypatch):
    mic = FakeMic(loud(10) + quiet(100))
    monkeypatch.setattr(stt, "open_input", mic)
    tts = Speaking([False, True])
    s = STT(CFG, backend=FakeBackend(), vad=AmpVAD(), tts=tts)
    assert s.tts is tts and s.listen() == "" and s.last_stop == "tts"


def test_prompt_leads_with_the_greetings():
    """'Hey room' came back as 'Hey, bro!' and 'Hey Drew!' (rig, Sat 26 Sep): the prompt primes the greetings,
    with the wake word only opening sentences."""
    p = stt.initial_prompt(CFG)
    assert p.startswith("Hey Room! Okay Room. Room, where are my ") and "keys" in p
    jarvis = stt.initial_prompt(dict(CFG, listen={"wake_words": ["jarvis"]}))
    assert jarvis.startswith("Hey Jarvis! Okay Jarvis. Jarvis, where")


@pytest.mark.parametrize("text", ["you", "You.", "Thank you.", "Thanks!", "Thank you so much.", "you you"])
def test_whisper_fillers_for_noise_are_dropped(text, monkeypatch):
    """The rig's Whisper wrote 'you' 12 times and 'Thank you.' 6 times for noise (Sat 26 Sep)."""
    assert stt.filler_only(text)
    s, _ = make(monkeypatch, quiet(10) + loud(20) + quiet(200))
    s._backend = FakeBackend(text)
    assert s.hear() == ""


def test_a_clicked_thank_you_is_kept(monkeypatch):
    """'Thank you' after a click acknowledges a care notice (core/reminders.py): only the always-on mic drops it."""
    s, _ = make(monkeypatch, quiet(10) + loud(20) + quiet(200))
    s._backend = FakeBackend("Thank you.")
    assert s.listen() == "Thank you."


@pytest.mark.parametrize("text", ["Thank you, where are my keys?", "So.", "You know it!", "Room, what do you see?"])
def test_real_speech_with_you_is_kept(text):
    assert not stt.filler_only(text)


def test_speech_under_min_speech_ms_is_dropped(monkeypatch):
    """A cough or a clack (a few VAD blocks) is not sent to Whisper, which writes 'you' for it."""
    s, _ = make(monkeypatch, quiet(10) + loud(5) + quiet(200))        # 5 blocks: 160 ms
    assert s.hear() == "" and not s.last_speech and s.backend.calls == []
    s, _ = make(monkeypatch, quiet(10) + loud(10) + quiet(200))       # 320 ms: a short "Room!"
    assert s.hear() and s.last_speech
    s, _ = make(monkeypatch, quiet(10) + loud(5) + quiet(200))        # asked (a click, "Room!"): kept
    assert len(s.record_until_silence()) > 0


@pytest.mark.parametrize("text, echo", [
    ("Hey Room! Okay Room.", True), ("hey room okay room room where are my keys", True),
    (stt.initial_prompt(CFG), True), ("Okay room.", False), ("Hey Room!", False), ("Room, where are my keys?", False),
    ("", False)])
def test_whisper_writing_its_prompt_back_is_an_echo(text, echo):
    assert stt.echoes_prompt(text, stt.initial_prompt(CFG)) is echo, text


def test_last_speech_s_is_the_speech_span_and_last_clip_s_the_recording(monkeypatch):
    s, _ = make(monkeypatch, quiet(10) + loud(15) + quiet(200))
    s.record_until_silence()
    assert abs(s.last_speech_s - 15 * BLOCK / RATE) < 1e-6 and s.last_clip_s > s.last_speech_s
    s, _ = make(monkeypatch, quiet(500), no_speech_s=1)
    s.record_until_silence()
    assert s.last_speech_s == 0.0


# -- cloud speech to text, for questions only

class FakeCloud:
    provider, timeout_s = "xai", 0.5

    def __init__(self, text="What color is my pill bottle?", fail=None, delay=0.0, key="k"):
        self.text, self.fail, self.delay, self._key, self.calls = text, fail, delay, key, 0

    def key(self):
        return self._key

    def transcribe(self, audio, prompt, audio_ctx=0):
        self.calls += 1
        time.sleep(self.delay)
        if self.fail:
            raise self.fail
        return self.text


def cloud_stt(monkeypatch, cloud, local="What car is my belt?"):
    s, _ = make(monkeypatch, quiet(10) + loud(20) + quiet(200))
    s._backend, s.cloud = FakeBackend(local), cloud
    return s


def test_a_question_is_transcribed_in_the_cloud_and_overheard_speech_never_is(monkeypatch):
    """Rig: base.en heard "What color is my pill bottle?" as "What car is my belt?"."""
    c = FakeCloud()
    s = cloud_stt(monkeypatch, c)
    assert s.listen() == "What color is my pill bottle?" and s.last_by == "xai" and s.backend.calls == []
    s2 = cloud_stt(monkeypatch, c)
    assert s2.hear() == "What car is my belt?" and c.calls == 1 and s2.last_by == "local"


@pytest.mark.parametrize("cloud", [FakeCloud(fail=RuntimeError("HTTP 429")), FakeCloud(delay=2.0),
                                   FakeCloud(key="")])
def test_the_local_model_answers_when_the_cloud_fails_is_slow_or_has_no_key(monkeypatch, cloud):
    s = cloud_stt(monkeypatch, cloud)
    t0 = time.monotonic()
    assert s.listen() == "What car is my belt?" and s.last_by == "local" and time.monotonic() - t0 < 1.5


def test_after_a_cloud_failure_the_cloud_rests_for_retry_after_s(monkeypatch):
    c = FakeCloud(fail=RuntimeError("down"))
    s = cloud_stt(monkeypatch, c)
    s.listen()
    c.fail = None
    s2 = cloud_stt(monkeypatch, c)
    s2._cloud_failed_t, s2.cloud_down_s = time.monotonic(), 60
    assert s2.listen() == "What car is my belt?" and c.calls == 1
    s2._cloud_failed_t = time.monotonic() - 61
    s3 = cloud_stt(monkeypatch, c)
    assert s3.listen() == "What color is my pill bottle?"


def test_cloud_setup_from_config():
    assert stt.make_cloud(CFG) is None                                       # off by default
    c = stt.make_cloud(dict(CFG, stt=dict(CFG["stt"], cloud={"provider": "xai"})))
    assert c.provider == "xai" and c.model == "grok-voice-transcribe-2.0" and c.key_env == "XAI_API_KEY"
    assert "Room" in c.keyterms and "pill bottle" in c.keyterms and "meds" in c.keyterms
    g = stt.make_cloud(dict(CFG, stt=dict(CFG["stt"], cloud={"provider": "groq", "timeout_s": 3})))
    assert g.model == "whisper-large-v3-turbo" and g.timeout_s == 3
    with pytest.raises(ValueError):
        stt.make_cloud(dict(CFG, stt=dict(CFG["stt"], cloud={"provider": "nope"})))


def test_cloud_requests_carry_keyterms_or_the_prompt(monkeypatch):
    sent = []

    class Resp:
        def raise_for_status(self):
            pass

        def json(self):
            return {"text": " What color is my pill bottle? "}

    class Session:
        def post(self, url, headers, data, files, timeout):
            sent.append((url, headers, data, files, timeout))
            return Resp()

    monkeypatch.setenv("XAI_API_KEY", "x-key")
    monkeypatch.setenv("GROQ_API_KEY", "g-key")
    audio = np.zeros(RATE, np.float32)
    x = stt.CloudBackend("xai", keyterms=["Room", "pill bottle"], session=Session())
    assert x.transcribe(audio, "Hey Room!").strip() == "What color is my pill bottle?"
    url, headers, data, files, _ = sent[-1]
    assert url == "https://api.x.ai/v1/stt" and headers["Authorization"] == "Bearer x-key"
    assert ("keyterm", "pill bottle") in data and ("model", "grok-voice-transcribe-2.0") in data
    assert files["file"][1][:4] == b"RIFF"
    g = stt.CloudBackend("groq", session=Session())
    g.transcribe(audio, "Hey Room!")
    url, headers, data, _, _ = sent[-1]
    assert url.endswith("/openai/v1/audio/transcriptions") and data["prompt"] == "Hey Room!"
    assert data["model"] == "whisper-large-v3-turbo" and headers["Authorization"] == "Bearer g-key"


def test_a_question_gets_the_question_prompt_and_overheard_speech_the_greetings(monkeypatch):
    """Question words in the prompt fix "What color is my pill bottle?" (heard "What car is my belt?"), but
    Whisper writes a prompt's questions back on babble, so overheard speech keeps the short greeting prompt."""
    s, _ = make(monkeypatch, quiet(10) + loud(20) + quiet(200))
    s.listen()
    s2, _ = make(monkeypatch, quiet(10) + loud(20) + quiet(200))
    s2.hear()
    asked, overheard = s.backend.calls[-1][1], s2.backend.calls[-1][1]
    assert "What color is my pill bottle?" in asked and asked == stt.question_prompt(CFG)
    assert overheard == stt.initial_prompt(CFG, synonyms=True) and "What color" not in overheard


@pytest.mark.parametrize("text", ["Room. Room. Room. Room.", "Room, room, room, room, room, room.", "no no no"])
def test_one_word_over_and_over_is_whisper_on_noise(text, monkeypatch):
    assert stt.filler_only(text)
    s, _ = make(monkeypatch, quiet(10) + loud(20) + quiet(200))
    s._backend = FakeBackend(text)
    assert s.hear() == ""


def test_room_twice_is_still_a_wake_word():
    assert not stt.filler_only("Room, room.") and not stt.filler_only("Okay Room. Room.")
