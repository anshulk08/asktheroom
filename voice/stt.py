"""Speech in (spec 3.9 / V2): record until the speaker stops, then whisper.cpp base.en.

End of speech is Silero VAD, not an energy threshold (an expo hall is too loud for one). Silero v5
ONNX (checked 2026-09-25, https://github.com/snakers4/silero-vad, src/silero_vad/data/silero_vad.onnx):
inputs `input` float32 [1, 64 + 512] (the previous chunk's last 64 samples, then 512 new ones at
16 kHz), `state` float32 [2, 1, 128], `sr` int64; outputs the speech probability and the next
state. Run with onnxruntime, so there is no torch dependency (same on the Mac and the Jetson).
Speech starts at prob >= threshold and ends after silence_ms below threshold - 0.15 (Silero's own
hysteresis). A second clicker press also ends the recording.

Whisper backends, picked by cfg stt.backend:
  pywhispercpp  whisper.cpp via its Python binding (laptop; Metal on a Mac). Model.transcribe(audio,
                initial_prompt=, audio_ctx=, ...); ggml models download to models/whisper/.
                https://github.com/absadiki/pywhispercpp
  cli           the whisper-cli binary from a whisper.cpp build (Jetson: -DGGML_CUDA=ON
                -DCMAKE_CUDA_ARCHITECTURES=87, make -j2). Flags: -m -f -l en -nt -np --prompt -ac.
                https://github.com/ggml-org/whisper.cpp/tree/master/examples/cli
Both get an initial prompt naming the objects (so 'pill bottle' is spelled right) and audio_ctx
sized to the clip: the default encodes a full 30 s window (1500 positions, 50 per second) even for
a 3 s question.

Audio goes through one seam tests can patch: `open_input(rate, block)` (a float32 mono source with
read(timeout) -> one block or None, and close()).
"""
from __future__ import annotations

import logging
import math
import os
import queue
import re
import shutil
import subprocess
import tempfile
import time
import wave
from pathlib import Path
from typing import Optional, Protocol

import numpy as np

from core.config import ROOT, display_name

log = logging.getLogger(__name__)

RATE = 16000
BLOCK = 512                 # 32 ms: the only chunk size Silero v5 takes at 16 kHz
CONTEXT = 64                # samples of the previous chunk Silero v5 expects in front
PREROLL_S = 0.3             # kept before the first speech block (soft consonants start early)
TAIL_S = 0.2                # kept after the last speech block
MIN_WHISPER_S = 1.1         # whisper.cpp drops input under 1 s; pad short clips with silence
CTX_PER_S = 50              # encoder positions per second of audio (1500 = 30 s)
CTX_PAD = 32                # slack: at +16 a clip ending in 1.3 s of digital silence came out as "Where"
VAD_PATH = ROOT / "models" / "silero_vad.onnx"
VAD_URL = "https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.onnx"
WHISPER_DIR = ROOT / "models" / "whisper"
TAG = re.compile(r"\[[^\]]*\]|\([^)]*\)")    # [BLANK_AUDIO], (music), ...


# ---------------------------------------------------------------- audio seam

class AudioIn:
    """float32 mono blocks from the default input device (sounddevice/PortAudio)."""

    def __init__(self, rate: int, block: int, device=None):
        import sounddevice as sd
        self._q: "queue.Queue[np.ndarray]" = queue.Queue()
        self.overflows = 0

        def cb(indata, frames, t, status):
            if status.input_overflow:
                self.overflows += 1
            self._q.put(indata[:, 0].copy())

        self.stream = sd.InputStream(samplerate=rate, blocksize=block, channels=1, dtype="float32",
                                     device=device, callback=cb)
        self.stream.start()

    def read(self, timeout: float = 1.0) -> Optional[np.ndarray]:
        try:
            return self._q.get(timeout=timeout)
        except queue.Empty:
            return None

    def close(self) -> None:
        try:
            self.stream.stop()
        finally:
            self.stream.close()


def open_input(rate: int, block: int, device=None) -> AudioIn:
    return AudioIn(rate, block, device)


# ---------------------------------------------------------------- VAD

class SileroVAD:
    """Streaming Silero v5: feed 512-sample blocks in order, get a speech probability for each."""

    def __init__(self, path: Path = VAD_PATH, threads: int = 1):
        import onnxruntime as ort
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"{path} missing; download it: curl -L -o {path} {VAD_URL}")
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = opts.inter_op_num_threads = threads
        self.sess = ort.InferenceSession(str(path), sess_options=opts,
                                         providers=["CPUExecutionProvider"])
        self._sr = np.array(RATE, dtype=np.int64)
        self.reset()

    def reset(self) -> None:
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._ctx = np.zeros(CONTEXT, dtype=np.float32)

    def __call__(self, block: np.ndarray) -> float:
        block = np.asarray(block, dtype=np.float32).reshape(-1)
        if len(block) != BLOCK:
            raise ValueError(f"Silero needs {BLOCK}-sample blocks at {RATE} Hz, got {len(block)}")
        x = np.concatenate([self._ctx, block])[None, :]
        out, self._state = self.sess.run(None, {"input": x, "state": self._state, "sr": self._sr})
        self._ctx = block[-CONTEXT:]
        return float(out.reshape(-1)[0])


# ---------------------------------------------------------------- whisper

def initial_prompt(cfg: dict) -> str:
    """Primes Whisper's spelling with the object names: 'Where are my keys, pill bottle, ...?'"""
    names = [display_name(cfg, o) for o in (cfg.get("objects") or {})]
    return f"Where are my {', '.join(names)}?" if names else ""


def audio_ctx_for(n_samples: int, rate: int = RATE) -> int:
    """Encoder context for a clip of n_samples: 50 positions per second, plus slack, at most 1500."""
    return min(1500, int(math.ceil(n_samples / rate * CTX_PER_S)) + CTX_PAD)


def clean(text: str) -> str:
    """Drop Whisper's non-speech tags and extra spaces."""
    return " ".join(TAG.sub(" ", text or "").split())


class Backend(Protocol):
    def transcribe(self, audio: np.ndarray, prompt: str, audio_ctx: int) -> str: ...


class PyWhisperCppBackend:
    def __init__(self, model: str = "base.en", models_dir: Path = WHISPER_DIR, threads: int = 4):
        from pywhispercpp.model import Model
        Path(models_dir).mkdir(parents=True, exist_ok=True)
        self.model = Model(model, models_dir=str(models_dir), n_threads=threads,
                           print_progress=False, print_realtime=False, print_timestamps=False)

    def transcribe(self, audio: np.ndarray, prompt: str, audio_ctx: int) -> str:
        segs = self.model.transcribe(audio, initial_prompt=prompt, audio_ctx=audio_ctx, language="en",
                                     no_context=True, single_segment=True, suppress_blank=True)
        return " ".join(s.text for s in segs)


class WhisperCliBackend:
    """whisper-cli from a whisper.cpp build. Writes the clip to a temp WAV and reads stdout."""

    def __init__(self, model: str = "base.en", binary: str = "whisper-cli",
                 models_dir: Path = WHISPER_DIR, threads: int = 4):
        path = Path(model)
        if not path.suffix:
            path = Path(models_dir) / f"ggml-{model}.bin"
        if not path.exists():
            raise FileNotFoundError(f"{path} missing; whisper.cpp: models/download-ggml-model.sh {model}")
        exe = shutil.which(binary) or (binary if os.path.exists(binary) else None)
        if exe is None:
            raise FileNotFoundError(f"{binary} not found; build whisper.cpp or set stt.cli_binary")
        self.model, self.exe, self.threads = str(path), exe, threads

    def transcribe(self, audio: np.ndarray, prompt: str, audio_ctx: int) -> str:
        with tempfile.NamedTemporaryFile(suffix=".wav") as f:
            write_wav(f.name, audio)
            r = subprocess.run([self.exe, "-m", self.model, "-f", f.name, "-l", "en", "-nt", "-np",
                                "-t", str(self.threads), "-ac", str(audio_ctx), "--prompt", prompt],
                               capture_output=True, text=True, timeout=30)
        if r.returncode != 0:
            raise RuntimeError(f"whisper-cli failed ({r.returncode}): {r.stderr[-300:]}")
        return r.stdout


BACKENDS = {"pywhispercpp": PyWhisperCppBackend, "cli": WhisperCliBackend}


def make_backend(cfg: dict) -> Backend:
    s = (cfg or {}).get("stt") or {}
    kind = str(s.get("backend", "pywhispercpp")).lower()
    model, threads = s.get("model", "base.en"), int(s.get("threads", 4))
    if kind == "cli":
        return WhisperCliBackend(model, s.get("cli_binary", "whisper-cli"), threads=threads)
    if kind not in BACKENDS:
        raise ValueError(f"unknown stt.backend {kind!r}; expected one of {sorted(BACKENDS)}")
    return PyWhisperCppBackend(model, threads=threads)


def write_wav(path, audio: np.ndarray, rate: int = RATE) -> None:
    pcm = (np.clip(np.asarray(audio, dtype=np.float32), -1, 1) * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm.tobytes())


def read_wav(path) -> np.ndarray:
    """16 kHz mono int16 WAV -> float32 in [-1, 1] (resampled linearly if the rate differs)."""
    with wave.open(str(path), "rb") as w:
        rate, ch, n = w.getframerate(), w.getnchannels(), w.getnframes()
        x = np.frombuffer(w.readframes(n), dtype=np.int16).astype(np.float32) / 32768
    if ch > 1:
        x = x.reshape(-1, ch).mean(axis=1)
    if rate != RATE:
        t = np.arange(int(len(x) * RATE / rate)) * rate / RATE
        x = np.interp(t, np.arange(len(x)), x).astype(np.float32)
    return x


# ---------------------------------------------------------------- STT

class STT:
    def __init__(self, cfg: dict, clicker=None, backend: Optional[Backend] = None,
                 vad: Optional[SileroVAD] = None):
        """clicker: anything with pressed() -> bool; a press while recording ends it. backend and
        vad default to the cfg stt settings and are loaded on first use (or by warm())."""
        s = (cfg or {}).get("stt") or {}
        self.cfg = cfg
        self.clicker = clicker
        self.max_s = float(s.get("max_s", 6))
        self.silence_ms = float(s.get("silence_ms", 700))
        self.threshold = float(s.get("vad_threshold", 0.5))
        self.no_speech_s = float(s.get("no_speech_s", 4))
        self.device = s.get("input_device")
        self.prompt = initial_prompt(cfg or {})
        self._backend, self._vad = backend, vad
        self.last_speech = False            # did the last recording contain speech?
        self.last_ms: dict[str, float] = {}

    @property
    def vad(self) -> SileroVAD:
        if self._vad is None:
            self._vad = SileroVAD()
        return self._vad

    @property
    def backend(self) -> Backend:
        if self._backend is None:
            self._backend = make_backend(self.cfg)
        return self._backend

    def warm(self) -> None:
        """Load the VAD and the Whisper model now, and run one short clip through Whisper, so the
        first question is not slow."""
        _ = self.vad
        self.transcribe(np.zeros(RATE, dtype=np.float32), force=True)

    def record_until_silence(self, max_s: Optional[float] = None,
                             silence_ms: Optional[float] = None) -> np.ndarray:
        """Record 16 kHz mono float32 until silence_ms of non-speech follows speech, max_s passes,
        a clicker press, or no_speech_s with no speech at all. Returns the speech plus a little
        padding, or an empty array if nobody spoke."""
        max_s = self.max_s if max_s is None else max_s
        silence_ms = self.silence_ms if silence_ms is None else silence_ms
        vad = self.vad
        vad.reset()
        off_thr = max(0.0, self.threshold - 0.15)
        block_s = BLOCK / RATE
        need_quiet = int(math.ceil(silence_ms / 1000 / block_s))
        blocks: list[np.ndarray] = []
        first = last = None                 # first / last speech block index
        quiet = 0
        stop = "max_s"
        t0 = time.monotonic()
        src = open_input(RATE, BLOCK, self.device)
        try:
            while len(blocks) * block_s < max_s:
                b = src.read(timeout=1.0)
                if b is None:
                    stop = "no audio"
                    log.warning("microphone gave no audio for 1 s")
                    break
                p = vad(b)
                blocks.append(b)
                i = len(blocks) - 1
                if first is None:
                    if p >= self.threshold:
                        first = last = i
                    elif i * block_s >= self.no_speech_s:
                        stop = "no speech"
                        break
                elif p >= off_thr:
                    last, quiet = i, 0
                else:
                    quiet += 1
                    if quiet >= need_quiet:
                        stop = "silence"
                        break
                if self.clicker is not None and self.clicker.pressed():
                    stop = "click"
                    break
        finally:
            src.close()
        self.last_ms["record"] = 1000 * (time.monotonic() - t0)
        self.last_speech = first is not None
        log.info("recorded %.2f s, stopped by %s, speech=%s", len(blocks) * block_s, stop, self.last_speech)
        if first is None:
            return np.zeros(0, dtype=np.float32)
        a = max(0, first - int(PREROLL_S / block_s))
        b = min(len(blocks), last + 1 + int(TAIL_S / block_s))
        return np.concatenate(blocks[a:b]).astype(np.float32)

    def transcribe(self, audio: np.ndarray, force: bool = False) -> str:
        """Whisper base.en with the object-name prompt and a clip-sized audio_ctx. Empty audio
        (VAD heard nobody) returns '' without running Whisper, which would hallucinate on silence."""
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        if len(audio) == 0 and not force:
            return ""
        n_min = int(MIN_WHISPER_S * RATE)
        if len(audio) < n_min:
            audio = np.concatenate([audio, np.zeros(n_min - len(audio), dtype=np.float32)])
        t0 = time.monotonic()
        text = clean(self.backend.transcribe(audio, self.prompt, audio_ctx_for(len(audio))))
        self.last_ms["transcribe"] = 1000 * (time.monotonic() - t0)
        log.info("transcribed %.1f s in %.0f ms: %r", len(audio) / RATE, self.last_ms["transcribe"], text)
        return text

    def listen(self) -> str:
        """record_until_silence() then transcribe()."""
        return self.transcribe(self.record_until_silence())


def main(argv=None) -> int:
    """python -m voice.stt [--wav clip.wav ...]: transcribe files, or press Enter / the clicker and
    speak. Prints the transcript, the parsed intent, and timings."""
    import argparse

    from core.config import load_config
    from voice.intents import parse
    ap = argparse.ArgumentParser(description="speech-to-text check")
    ap.add_argument("--wav", nargs="*", help="transcribe these files instead of the microphone")
    ap.add_argument("--record-questions", metavar="DIR",
                    help="read out tests/stt_questions.json and save 01.wav ... into DIR")
    ap.add_argument("--config")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    cfg = load_config(a.config)
    if a.record_questions:
        return record_questions(cfg, Path(a.record_questions))
    if a.wav:
        stt = STT(cfg)
        for p in a.wav:
            text = stt.transcribe(read_wav(p))
            it = parse(text, cfg)
            print(f"{p}: {text!r} -> {it.kind} {it.obj}  ({stt.last_ms['transcribe']:.0f} ms)")
        return 0
    from voice.trigger import Clicker
    clicker = Clicker(cfg)
    stt = STT(cfg, clicker=clicker)
    stt.warm()
    while True:
        print("press the clicker (or Enter) and ask a question; Ctrl-C to quit")
        clicker.wait_press()
        text = stt.listen()
        it = parse(text, cfg)
        print(f"{text!r} -> {it.kind} {it.obj}  (record {stt.last_ms.get('record', 0):.0f} ms, "
              f"transcribe {stt.last_ms.get('transcribe', 0):.0f} ms)")


def record_questions(cfg: dict, out: Path) -> int:
    """Prompt for each test question, record it with the VAD, save it as NN.wav."""
    import json

    from voice.trigger import Clicker
    qs = json.loads((ROOT / "tests" / "stt_questions.json").read_text())
    out.mkdir(parents=True, exist_ok=True)
    clicker = Clicker(cfg)
    stt = STT(cfg, clicker=clicker)
    for i, q in enumerate(qs, 1):
        while True:
            print(f"[{i}/{len(qs)}] press the clicker (or Enter), then say: {q['text']!r}")
            clicker.wait_press()
            audio = stt.record_until_silence()
            if len(audio):
                break
            print("  heard no speech, again")
        write_wav(out / f"{i:02d}.wav", audio)
        print(f"  saved {len(audio) / RATE:.1f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
