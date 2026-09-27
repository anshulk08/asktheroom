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
  server        a long-lived whisper-server from the same build (scripts/build_whisper.sh), so the
                model stays loaded: whisper-cli pays ~0.5 s per question on the Jetson for process
                start + CUDA init + model load (measured 2026-09-25: base.en 139 ms median per
                question via the server vs 621 ms via whisper-cli). Falls back to cli if it dies.
  auto          server > cli > pywhispercpp, whichever this machine has (one config.yaml for both).
All get an initial prompt naming the objects (so 'pill bottle' is spelled right) and an audio_ctx:
`sized` to the clip, or a fixed value (cfg stt.audio_ctx; 0 = the full 30 s window, 1500
positions, 50 per second). Measured on 22 Piper questions (0.8-1.9 s): sized (~90-130 positions)
made base.en repeat or garble and run long (whisper-cli median 2.2 s, 13/22 exact); 0 gave 22/22
exact at 139 ms on the Jetson's GPU, so config.yaml uses 0.

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
FILLER_WORDS = {"you", "thank", "thanks", "so", "very", "much"}   # filler_only: nobody asks the rig this
CTX_PER_S = 50              # encoder positions per second of audio (1500 = 30 s)
CTX_PAD = 32                # slack: at +16 a clip ending in 1.3 s of digital silence came out as "Where"
VAD_PATH = ROOT / "models" / "silero_vad.onnx"
VAD_URL = "https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.onnx"
WHISPER_DIR = ROOT / "models" / "whisper"
TAG = re.compile(r"\[[^\]]*\]|\([^)]*\)")    # [BLANK_AUDIO], (music), ...


# ---------------------------------------------------------------- audio seam

def _to_block(x: np.ndarray, block: int) -> np.ndarray:
    """One device block resampled to exactly `block` samples: an average over each group when the
    ratio is whole (48 kHz -> 16 kHz: 3 samples, which also filters what would alias), else linear."""
    n = len(x)
    if n == block:
        return x
    if n % block == 0:
        return x.reshape(block, n // block).mean(axis=1).astype(np.float32)
    return np.interp(np.linspace(0, n - 1, block), np.arange(n), x).astype(np.float32)


_DEVICE_RATE: dict = {}        # (device, wanted rate) -> the rate it records at instead (one failed open per run)


class AudioIn:
    """float32 mono blocks of `block` samples at `rate` from an input device (sounddevice/PortAudio).
    A device that refuses `rate` (an ALSA hw device on the Jetson often takes only 44.1 / 48 kHz) is
    opened at its own default rate, and each block is resampled to `block` samples at `rate` (the VAD
    needs exactly 512 at 16 kHz)."""

    def __init__(self, rate: int, block: int, device=None):
        import sounddevice as sd
        self._q: "queue.Queue[np.ndarray]" = queue.Queue()
        self.overflows = 0
        self.device_rate = rate
        known = _DEVICE_RATE.get((device, rate))              # this device refused `rate` before

        def cb(indata, frames, t, status):
            if status.input_overflow:
                self.overflows += 1
            x = indata[:, 0].copy()
            self._q.put(x if len(x) == block else _to_block(x, block))

        try:
            if known:
                raise sd.PortAudioError("known to refuse this rate")
            self.stream = sd.InputStream(samplerate=rate, blocksize=block, channels=1, dtype="float32",
                                         device=device, callback=cb)
        except sd.PortAudioError:
            dev_rate = known or int(sd.query_devices(device, "input")["default_samplerate"])
            if dev_rate == rate:
                raise
            if not known:
                log.info("input device refuses %d Hz; recording at its %d Hz, resampled", rate, dev_rate)
                _DEVICE_RATE[(device, rate)] = dev_rate
            self.device_rate = dev_rate
            self.stream = sd.InputStream(samplerate=dev_rate, blocksize=round(block * dev_rate / rate),
                                         channels=1, dtype="float32", device=device, callback=cb)
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


def input_devices() -> list[dict]:
    """sounddevice devices that can record (max_input_channels > 0), each with its 'index'."""
    import sounddevice as sd
    return [dict(d, index=i) for i, d in enumerate(sd.query_devices()) if d.get("max_input_channels", 0) > 0]


def find_input_device(spec) -> tuple[Optional[int], str, list]:
    """stt.input_device -> (sounddevice index or None for the default mic, what it is, the matches). Like
    tts.output_device: an int (or digits) is the index; any other string, the first input device whose name
    contains it, case insensitive (sounddevice itself refuses a name several devices share: the Brio, a USB
    mic and a USB speaker are all "USB Audio"). A device whose whole name is the string wins over those that
    only contain it ("default" is not "sysdefault"). The status is default | index | match | ambiguous | missing;
    missing means a name was set and no input device has it."""
    if spec is None or (isinstance(spec, str) and not spec.strip()):
        return None, "default", []
    if isinstance(spec, int) or (isinstance(spec, str) and spec.strip().isdigit()):
        return int(spec), "index", []
    want = str(spec).strip().lower()
    hits = [d for d in input_devices() if want in str(d.get("name", "")).lower()]
    if not hits:
        return None, "missing", []
    exact = [d for d in hits if str(d.get("name", "")).strip().lower() == want]
    if exact:                   # "default" / "pulse" (ALSA plugins through the host PulseAudio) are whole names;
        return int(exact[0]["index"]), "match", exact    # "default" is also inside "sysdefault", listed first
    return int(hits[0]["index"]), ("ambiguous" if len(hits) > 1 else "match"), hits


def resolve_input_device(spec) -> Optional[int]:
    """find_input_device's index, logging a missing mic (ERROR: the default mic may be the Brio across the
    room) or an ambiguous name (every match, and the one used)."""
    idx, status, hits = find_input_device(spec)
    if status == "missing":
        log.error("stt.input_device %r: no input device has that name; using the default mic "
                  "(python -m voice.stt --devices lists them)", spec)
    elif status == "ambiguous":
        log.info("stt.input_device %r matches %s; using %d: %s", spec,
                 "; ".join(f"{d['index']}: {d['name']}" for d in hits), idx, hits[0]["name"])
    elif status == "match":
        log.info("microphone: %d: %s", idx, hits[0]["name"])
    return idx


def record_seconds(seconds: float, spec=None, rate: int = RATE) -> np.ndarray:
    """`seconds` of float32 mono from the stt.input_device mic, the way the voice loop opens it (by name,
    resampled if it won't record at `rate`). RuntimeError if a named mic isn't there: demo_check's mic
    check must fail then, not pass on the default mic."""
    idx, status, _ = find_input_device(spec)
    if status == "missing":
        raise RuntimeError(f"stt.input_device {spec!r}: no input device has that name")
    src = open_input(rate, BLOCK, idx)
    out, need = [], int(seconds * rate)
    try:
        while sum(len(b) for b in out) < need:
            b = src.read(timeout=2.0)
            if b is None:
                raise RuntimeError("no audio from the microphone")
            out.append(b)
    finally:
        src.close()
    return np.concatenate(out)[:need] if out else np.zeros(0, np.float32)


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


class LevelGate:
    """Optional end-of-speech helper for rooms full of other talkers (cfg stt.end_drop_db).

    Silero hears speech at almost any level: measured on Piper clips, 6 overlapping background
    talkers 20 dB below the asker still score p ~0.97, so in an expo hall Silero alone never sees the
    question end and the recording runs to max_s. This is not an absolute energy threshold: after
    speech starts, a block also counts as quiet when its RMS is drop_db below the 90th percentile of
    the block RMS since speech started (the asker is much closer to the mic than the crowd, so their
    loudest blocks set the bar; a percentile, not the peak, so one cough or clicker clack does not).
    drop_db None: off (Silero only)."""

    PCT = 90

    def __init__(self, drop_db: Optional[float]):
        self.ratio = None if drop_db is None else 10 ** (-float(drop_db) / 20)
        self._rms: list[float] = []

    @staticmethod
    def rms(b: np.ndarray) -> float:
        return float(np.sqrt(np.mean(np.square(b, dtype=np.float64))))

    def speech(self, b: np.ndarray) -> None:
        """Called for the first speech block; loud() records every later block itself."""
        if self.ratio is not None and not self._rms:
            self._rms.append(self.rms(b))

    def loud(self, b: np.ndarray) -> bool:
        if self.ratio is None:
            return True
        r = self.rms(b)
        self._rms.append(r)
        return r >= float(np.percentile(self._rms, self.PCT)) * self.ratio


# ---------------------------------------------------------------- whisper

def initial_prompt(cfg: dict, synonyms: bool = False) -> str:
    """Primes Whisper's spelling with the object names: 'Where are my keys, pill bottle, ...?'
    synonyms (cfg stt.prompt_synonyms) also lists the spoken synonyms ('meds', 'specs', ...): with
    names only, base.en heard Piper's 'Where are my meds?' as 'mids' on both the Mac and the Jetson."""
    names = [display_name(cfg, o) for o in (cfg.get("objects") or {})]
    if synonyms:
        objs = cfg.get("objects") or {}
        names += [re.sub(r"^the ", "", str(k)) for k, v in (cfg.get("synonyms") or {}).items() if v in objs]
        names = list(dict.fromkeys(names))
    if not names:
        return ""
    # The greetings lead, as they are spoken: on its own, a one-word "Room!" came back as "Hit the Arduino
    # table." from the ceiling mic, and "Hey room" as "Hey, bro!" or "Hey Drew!" (rig, Sat 26 Sep). The wake
    # word only opens sentences here, so Whisper isn't primed to write "room" mid-sentence.
    wake = str(((cfg.get("listen") or {}).get("wake_words") or ["room"])[0]).strip().capitalize()
    return f"Hey {wake}! Okay {wake}. {wake}, where are my {', '.join(names)}?"


def audio_ctx_for(n_samples: int, rate: int = RATE) -> int:
    """Encoder context for a clip of n_samples: 50 positions per second, plus slack, at most 1500."""
    return min(1500, int(math.ceil(n_samples / rate * CTX_PER_S)) + CTX_PAD)


def clean(text: str) -> str:
    """Drop Whisper's non-speech tags and extra spaces."""
    return " ".join(TAG.sub(" ", text or "").split())


def filler_only(text: str) -> bool:
    """What Whisper writes for noise: 'you' (12 times on the rig, Sat 26 Sep), 'Thank you.' (6), 'Thanks!'."""
    words = re.findall(r"[a-z]+", text.lower())
    return bool(words) and set(words) <= FILLER_WORDS and bool({"you", "thanks"} & set(words))


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
    """whisper-cli from a whisper.cpp build. Writes the clip to a temp WAV and reads stdout. The WAV
    goes in /dev/shm (RAM) where there is one (the Jetson), so audio never touches the disk."""

    def __init__(self, model: str = "base.en", binary: str = "whisper-cli",
                 models_dir: Path = WHISPER_DIR, threads: int = 4):
        path = model_path(model, models_dir)
        if not path.exists():
            raise FileNotFoundError(f"{path} missing; whisper.cpp: models/download-ggml-model.sh {model}")
        exe = shutil.which(binary) or (binary if os.path.exists(binary) else None)
        if exe is None:
            raise FileNotFoundError(f"{binary} not found; build whisper.cpp or set stt.cli_binary")
        self.model, self.exe, self.threads = str(path), exe, threads

    def transcribe(self, audio: np.ndarray, prompt: str, audio_ctx: int) -> str:
        ram = "/dev/shm" if os.path.isdir("/dev/shm") else None
        with tempfile.NamedTemporaryFile(suffix=".wav", dir=ram) as f:
            write_wav(f.name, audio)
            r = subprocess.run([self.exe, "-m", self.model, "-f", f.name, "-l", "en", "-nt", "-np",
                                "-t", str(self.threads), "-ac", str(audio_ctx), "--prompt", prompt],
                               capture_output=True, text=True, timeout=30)
        if r.returncode != 0:
            raise RuntimeError(f"whisper-cli failed ({r.returncode}): {r.stderr[-300:]}")
        return r.stdout


WHISPER_BUILDS = [ROOT / "third_party" / "whisper.cpp" / d / "bin" for d in ("build", "build-mac")]


def find_binary(name: str) -> Optional[str]:
    """A whisper.cpp binary: `name` as a path (absolute, or relative to the cwd or the repo), then
    PATH, then scripts/build_whisper.sh's output (third_party/whisper.cpp/build{,-mac}/bin)."""
    p = Path(name)
    for c in ([p] if p.is_absolute() else [p, ROOT / p]):
        if c.is_file() and os.access(c, os.X_OK):
            return str(c)
    exe = shutil.which(name)
    if exe:
        return exe
    for d in WHISPER_BUILDS:
        c = d / p.name
        if c.is_file() and os.access(c, os.X_OK):
            return str(c)
    return None


def model_path(model: str, models_dir: Path = WHISPER_DIR) -> Path:
    """'base.en' -> models/whisper/ggml-base.en.bin; a .bin file or a path is used as given.
    (Not Path.suffix: 'base.en' has the suffix '.en'.)"""
    m = str(model)
    if m.endswith(".bin") or os.sep in m or "/" in m:
        return Path(m)
    return Path(models_dir) / f"ggml-{m}.bin"


def _die_with_parent():
    """Popen preexec_fn on Linux: SIGTERM the server if the app dies, even by SIGKILL, so no orphan
    keeps the port and the GPU memory. prctl is looked up here, in the parent, so the forked child
    only makes one C call (no imports, which could deadlock after fork in a threaded app)."""
    if not Path("/proc/self").exists():
        return None
    try:
        import ctypes
        prctl = ctypes.CDLL(None, use_errno=True).prctl
    except Exception:
        return None
    return lambda: prctl(1, 15)            # PR_SET_PDEATHSIG, SIGTERM


class WhisperServerBackend:
    """A long-lived whisper-server (whisper.cpp examples/server), started once, so the model and the
    CUDA context stay loaded between questions: launching whisper-cli per question pays process
    start + CUDA init + model load every time. The server takes per-request multipart fields and
    resets them after each request (server.cpp get_req_parameters, checked 2026-09-25 at d09f61a):
    `file` (WAV bytes), `prompt`, `audio_ctx`, `no_timestamps`, `response_format=json` -> {"text"}.
    GET /health is 200 {"status":"ok"} once the model is loaded.

    If a healthy server already answers on host:port it is reused (e.g. left over from a run that
    was SIGKILLed on the Mac). If the server cannot be reached for a request, it is restarted once
    (when we own it) and the request goes to `fallback` (whisper-cli) meanwhile."""

    def __init__(self, model: str = "base.en", binary: str = "whisper-server",
                 models_dir: Path = WHISPER_DIR, threads: int = 4, host: str = "127.0.0.1",
                 port: int = 8178, start_timeout: float = 60.0, request_timeout: float = 15.0,
                 fallback: Optional[Backend] = None, start: bool = True):
        import requests
        self._http = requests.Session()
        self.url = f"http://{host}:{port}"
        self.host, self.port, self.threads = host, int(port), threads
        self.start_timeout, self.request_timeout = start_timeout, request_timeout
        self.fallback = fallback
        self.model = str(model_path(model, models_dir))
        self.exe = find_binary(binary)
        self.proc: Optional[subprocess.Popen] = None
        self._log = None
        self.fallbacks = 0              # requests answered by the fallback
        if start and not self.healthy():
            if not Path(self.model).exists():
                raise FileNotFoundError(f"{self.model} missing; whisper.cpp: models/download-ggml-model.sh {model}")
            if self.exe is None:
                raise FileNotFoundError(f"{binary} not found; run scripts/build_whisper.sh or set stt.server_binary")
            self.start()

    def healthy(self, timeout: float = 0.5) -> bool:
        try:
            r = self._http.get(self.url + "/health", timeout=timeout)
            return r.status_code == 200 and r.json().get("status") == "ok"
        except Exception:
            return False

    def start(self) -> None:
        """Spawn the server and wait (up to start_timeout) for /health. Raises if it never comes up."""
        import atexit
        self._log = tempfile.NamedTemporaryFile(prefix="whisper-server-", suffix=".log", delete=False)
        cmd = [self.exe, "-m", self.model, "-t", str(self.threads), "-l", "en", "-nt",
               "--host", self.host, "--port", str(self.port)]
        log.info("starting %s", " ".join(cmd))
        self.proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=self._log,
                                     stdin=subprocess.DEVNULL,
                                     preexec_fn=_die_with_parent())
        atexit.register(self.close)
        t0 = time.monotonic()
        while time.monotonic() - t0 < self.start_timeout:
            if self.proc.poll() is not None:
                break
            if self.healthy():
                log.info("whisper-server up in %.1f s (%s)", time.monotonic() - t0, self.url)
                return
            time.sleep(0.1)
        tail = Path(self._log.name).read_text(errors="replace")[-500:]
        self.close()
        raise RuntimeError(f"whisper-server did not come up on {self.url}: {tail}")

    def _post(self, audio: np.ndarray, prompt: str, audio_ctx: int) -> str:
        import io
        buf = io.BytesIO()
        write_wav(buf, audio)
        data = {"prompt": prompt, "audio_ctx": str(int(audio_ctx)), "no_timestamps": "true",
                "response_format": "json", "temperature": "0.0"}
        r = self._http.post(self.url + "/inference", files={"file": ("clip.wav", buf.getvalue(), "audio/wav")},
                            data=data, timeout=self.request_timeout)
        r.raise_for_status()
        return str(r.json().get("text", ""))

    def transcribe(self, audio: np.ndarray, prompt: str, audio_ctx: int) -> str:
        import requests
        try:
            return self._post(audio, prompt, audio_ctx)
        except requests.RequestException as e:
            log.warning("whisper-server request failed (%s)", e)
            if self.proc is not None and self.proc.poll() is not None:
                try:
                    self.start()
                    return self._post(audio, prompt, audio_ctx)
                except Exception as e2:
                    log.warning("whisper-server restart failed: %s", e2)
            if self.fallback is None:
                raise
            self.fallbacks += 1
            return self.fallback.transcribe(audio, prompt, audio_ctx)

    def close(self) -> None:
        p, self.proc = self.proc, None
        if p is not None and p.poll() is None:
            p.terminate()
            try:
                p.wait(timeout=3)
            except subprocess.TimeoutExpired:
                p.kill()


BACKENDS = {"pywhispercpp": PyWhisperCppBackend, "cli": WhisperCliBackend,
            "server": WhisperServerBackend}


def _server(s: dict, model: str, threads: int, fallback: Optional[Backend]) -> WhisperServerBackend:
    return WhisperServerBackend(model, s.get("server_binary", "whisper-server"), threads=threads,
                                port=int(s.get("server_port", 8178)), fallback=fallback)


def _cli_or_none(s: dict, model: str, threads: int) -> Optional[Backend]:
    try:
        return WhisperCliBackend(model, find_binary(s.get("cli_binary", "whisper-cli")) or
                                 s.get("cli_binary", "whisper-cli"), threads=threads)
    except FileNotFoundError:
        return None


def make_backend(cfg: dict) -> Backend:
    """stt.backend: server | cli | pywhispercpp | auto. auto (one config for the Mac and the Jetson)
    takes the first that is available here: server (whisper-server binary + model; whisper-cli as
    its fallback), then cli, then pywhispercpp."""
    s = (cfg or {}).get("stt") or {}
    kind = str(s.get("backend", "pywhispercpp")).lower()
    model, threads = s.get("model", "base.en"), int(s.get("threads", 4))
    if kind == "cli":
        binary = s.get("cli_binary", "whisper-cli")
        return WhisperCliBackend(model, find_binary(binary) or binary, threads=threads)
    if kind in ("server", "auto"):
        cli = _cli_or_none(s, model, threads)
        if kind == "server" or (find_binary(s.get("server_binary", "whisper-server"))
                                and model_path(model).exists()):
            try:
                return _server(s, model, threads, cli)
            except Exception as e:
                if cli is None and kind == "server":
                    raise
                log.warning("stt %s: whisper-server failed (%s); trying whisper-cli", kind, e)
        if cli is not None:
            return cli
        if kind == "server":
            raise RuntimeError("stt server: whisper-server and whisper-cli both unavailable")
        try:
            return PyWhisperCppBackend(model, threads=threads)
        except ImportError:
            raise RuntimeError("stt auto: no whisper backend; run scripts/build_whisper.sh "
                               "or pip install pywhispercpp") from None
    if kind not in BACKENDS:
        raise ValueError(f"unknown stt.backend {kind!r}; expected one of {sorted(BACKENDS) + ['auto']}")
    return PyWhisperCppBackend(model, threads=threads)


def write_wav(path, audio: np.ndarray, rate: int = RATE) -> None:
    pcm = (np.clip(np.asarray(audio, dtype=np.float32), -1, 1) * 32767).astype(np.int16)
    with wave.open(path if hasattr(path, "write") else str(path), "wb") as w:   # a path or a file object
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
                 vad: Optional[SileroVAD] = None, tts=None):
        """clicker: anything with pressed() -> bool; a press while recording ends it. backend and
        vad default to the cfg stt settings and are loaded on first use (or by warm()). tts: anything
        with a `.speaking` bool (voice.tts.TTS); when the rig starts talking mid-recording, the clip
        (its own voice from now on) is dropped."""
        s = (cfg or {}).get("stt") or {}
        self.cfg = cfg
        self.clicker = clicker
        self.tts = tts
        self.max_s = float(s.get("max_s", 6))
        self.silence_ms = float(s.get("silence_ms", 700))
        self.threshold = float(s.get("vad_threshold", 0.5))
        self.no_speech_s = float(s.get("no_speech_s", 4))
        self.min_speech_s = float(s.get("min_speech_ms", 250)) / 1000   # always-on mic: a cough, a clack
        ctx = s.get("audio_ctx", "sized")
        self.audio_ctx = "sized" if ctx in (None, "", "sized") else int(ctx)   # int: fixed (0 = full 30 s)
        drop = s.get("end_drop_db")
        self.end_drop_db = None if drop in (None, "", 0) else float(drop)
        self.device = s.get("input_device")
        self._dev, self._dev_ok = None, False
        self.input_status = ""              # default | index | match | ambiguous | missing | error, once resolved
        self.prompt = initial_prompt(cfg or {}, synonyms=bool(s.get("prompt_synonyms", False)))
        self._backend, self._vad = backend, vad
        self.last_speech = False            # did the last recording contain speech?
        self.last_stop = ""                 # why it ended: silence | max_s | click | no speech | no audio | tts
        self.log_text = True                # always-on mic: main.py turns this off, so chatter isn't logged
        self.last_ms: dict[str, float] = {}

    def _input(self):
        """stt.input_device as a sounddevice index, resolved once (logged then: the device used, or an
        ERROR for a named mic that isn't there) and again only after an open failed (a replugged mic may
        have a new index). None: the default mic."""
        if not self._dev_ok:
            try:
                self._dev = resolve_input_device(self.device)
                self.input_status = find_input_device(self.device)[1] if self.device not in (None, "") else "default"
            except Exception:
                log.exception("input device lookup failed; using the default mic")
                self._dev, self.input_status = None, "error"
            self._dev_ok = True
        return self._dev

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
                             silence_ms: Optional[float] = None,
                             no_speech_s: Optional[float] = None, min_speech_s: float = 0.0) -> np.ndarray:
        """Record 16 kHz mono float32 until silence_ms of non-speech follows speech, max_s passes,
        a clicker press, or no_speech_s with no speech at all. Returns the speech plus a little
        padding, or an empty array if nobody spoke (or for under min_speech_s, unless a click ended it).
        If the TTS starts speaking while this records (a care notice, a dashboard or phone answer), the
        recording ends and the clip is discarded (empty array,
        last_stop 'tts'): the mic would hear the rig and answer itself. Speech already playing when the
        recording starts (a clicked answer still cutting off) doesn't count until it has stopped once."""
        max_s = self.max_s if max_s is None else max_s
        silence_ms = self.silence_ms if silence_ms is None else silence_ms
        no_speech_s = self.no_speech_s if no_speech_s is None else no_speech_s
        vad = self.vad
        vad.reset()
        off_thr = max(0.0, self.threshold - 0.15)
        level = LevelGate(self.end_drop_db)
        block_s = BLOCK / RATE
        need_quiet = int(math.ceil(silence_ms / 1000 / block_s))
        blocks: list[np.ndarray] = []
        first = last = None                 # first / last speech block index
        quiet = 0
        stop = "max_s"
        was_quiet = not self._tts_speaking()    # the rig's voice counts once it has been off
        t0 = time.monotonic()
        try:
            src = open_input(RATE, BLOCK, self._input())
        except Exception:
            self._dev_ok = False            # look the name up again next time (unplugged, replugged)
            raise
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
                        level.speech(b)
                    elif i * block_s >= no_speech_s:
                        stop = "no speech"
                        break
                elif level.loud(b) and p >= off_thr:
                    last, quiet = i, 0
                else:
                    quiet += 1
                    if quiet >= need_quiet:
                        stop = "silence"
                        break
                if self.clicker is not None and self.clicker.pressed():
                    stop = "click"
                    break
                if self._tts_speaking():
                    if was_quiet:
                        stop = "tts"
                        break
                else:
                    was_quiet = True
        finally:
            src.close()
        note = ""
        if stop == "tts":
            first, note = None, " (the rig started speaking: discarded)"   # nothing here is a question
        elif first is not None and stop != "click" and (last - first + 1) * block_s < min_speech_s:
            first, note = None, f" (under {min_speech_s * 1000:.0f} ms of speech: discarded)"
        self.last_ms["record"] = 1000 * (time.monotonic() - t0)
        self.last_speech = first is not None
        self.last_stop = stop
        log.info("recorded %.2f s, stopped by %s, speech=%s%s", len(blocks) * block_s, stop, self.last_speech, note)
        if first is None:
            return np.zeros(0, dtype=np.float32)
        a = max(0, first - int(PREROLL_S / block_s))
        b = min(len(blocks), last + 1 + int(TAIL_S / block_s))
        return np.concatenate(blocks[a:b]).astype(np.float32)

    def _tts_speaking(self) -> bool:
        try:
            return bool(getattr(self.tts, "speaking", False))
        except Exception:
            return False

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
        ctx = audio_ctx_for(len(audio)) if self.audio_ctx == "sized" else self.audio_ctx
        text = clean(self.backend.transcribe(audio, self.prompt, ctx))
        self.last_ms["transcribe"] = 1000 * (time.monotonic() - t0)
        log.info("transcribed %.1f s in %.0f ms%s", len(audio) / RATE, self.last_ms["transcribe"],
                 f": {text!r}" if self.log_text else "")
        return text

    def listen(self) -> str:
        """record_until_silence() then transcribe()."""
        return self.transcribe(self.record_until_silence())

    def hear(self, idle_s: float = 8.0) -> str:
        """Always-on mic: wait up to idle_s for someone to speak, then record until they stop and
        transcribe. The audio lives only in memory and is dropped here. '' if nobody spoke, for under
        stt.min_speech_ms of speech, or for Whisper's fillers ('you', 'Thank you.': filler_only; a clicked
        "thank you" still acknowledges a care notice)."""
        text = self.transcribe(self.record_until_silence(no_speech_s=idle_s, min_speech_s=self.min_speech_s))
        if filler_only(text):
            log.info("dropped Whisper's filler for noise")
            return ""
        return text


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
    ap.add_argument("--devices", action="store_true", help="list input devices (* = stt.input_device, d = default)")
    ap.add_argument("--level", type=float, metavar="SECONDS",
                    help="mic level and speech probability every half second (the 2 m check), then quit")
    ap.add_argument("--config")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    cfg = load_config(a.config)
    if a.devices:
        import sounddevice as sd
        spec = (cfg.get("stt") or {}).get("input_device")
        chosen, default_in = resolve_input_device(spec), sd.default.device[0]
        print(f"stt.input_device: {spec!r}")
        for d in input_devices():
            mark = ("*" if d["index"] == chosen else " ") + ("d" if d["index"] == default_in else " ")
            print(f"{mark} {d['index']:3d}  {d['name']}  ({d['max_input_channels']} ch, "
                  f"{int(d['default_samplerate'])} Hz)")
        return 0
    if a.level:
        return level_meter(STT(cfg), a.level)
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


def level_meter(stt: "STT", seconds: float, out=print) -> int:
    """Print the mic level (dBFS) and the VAD's highest speech probability every half second for `seconds`.
    Stand where the judge will and talk: speech should reach stt.vad_threshold with the room's noise below
    it. Nothing is recorded or kept."""
    src = open_input(RATE, BLOCK, stt._input())
    vad = stt.vad
    vad.reset()
    per = max(1, RATE // BLOCK // 2)
    out(f"input at {getattr(src, 'device_rate', RATE)} Hz; speech threshold {stt.threshold:.2f}")
    try:
        t, n, sq, pmax = 0.0, 0, 0.0, 0.0
        while t < seconds:
            x = src.read(timeout=1.0)
            if x is None:
                out("no audio from the input device")
                return 1
            sq += float(np.mean(x.astype(np.float64) ** 2))
            pmax = max(pmax, vad(x))
            n += 1
            if n == per:
                t += n * BLOCK / RATE
                db = 10 * np.log10(sq / n + 1e-12)
                out(f"{t:5.1f} s  level {db:6.1f} dBFS  speech {pmax:.2f} {'#' * int(pmax * 20):20s}"
                    f"{'  SPEECH' if pmax >= stt.threshold else ''}")
                n, sq, pmax = 0, 0.0, 0.0
    finally:
        src.close()
    return 0


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
