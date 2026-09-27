"""Scores speech-to-text for the demo: word error rate, the question read right, latency, and Whisper on noise.

A clip set is a directory of WAVs with truth.json ([{"wav", "text", "note"}]; text "" for a clip with no
question in it). Real recordings come from the rig's mic (record_demo.py, WS7's session); --synthesize makes a
set from the same questions with Piper (and Grok's voices when $XAI_API_KEY is set), clean and with babble or
reverb added, plus noise-only clips.

Each run is one STT setup: a whisper.cpp model (pywhispercpp here; the rig runs whisper-server with the same
ggml file) or a cloud API, and a prompt ("rig": voice.stt.initial_prompt, the overheard prompt; "demo": voice.stt.question_prompt;
"none"). Per setup it prints:
  WER       word error rate over the question clips (words normalized; "okay"/"ok" and "color"/"colour" alike)
  intent    voice.intents.parse of the transcript gives the truth's kind and object
  latency   median / 90th percentile ms per clip on this machine (the Jetson is slower: measure there)
  noise     noise-only clips Whisper wrote something for, and of those how many wake the rig
            (voice.understand.bare_wake / has_wake_word) or are the prompt written back (voice.stt.echoes_prompt)

    python scripts/eval_stt.py --synthesize data/stt_eval/synth
    python scripts/eval_stt.py data/stt_eval/rec --models base.en,small.en --prompts rig,demo
    python scripts/eval_stt.py data/stt_eval/rec --cloud xai,groq  # needs $XAI_API_KEY / $GROQ_API_KEY
    python scripts/eval_stt.py data/stt_eval/rec --server 127.0.0.1:8179 --label small.en-q5_1   # the Jetson
"""
from __future__ import annotations

import argparse
import io
import json
import os
import re
import statistics
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.config import load_config  # noqa: E402
from voice.intents import parse  # noqa: E402
from voice.stt import (RATE, clean, echoes_prompt, filler_only, initial_prompt, question_prompt, read_wav,  # noqa: E402
                       write_wav)
from voice.understand import bare_wake, has_wake_word  # noqa: E402

# The questions the demo is asked (the same list WS7 recorded on the rig with record_demo.py).
QUESTIONS = [
    "What color is my pill bottle?", "Where are my keys?", "What's on the kitchen counter?",
    "Room, this is my lucky mug.", "When did I last pick up my pills?", "What changed?", "Hey Room!", "Okay Room.",
    "Room, what do you see?", "Where did I leave my wallet?", "Is my phone on the couch?",
    "Did anyone touch my glasses?", "Where's the remote?", "What's in the box?", "Room, where is my notebook?",
    "What color are the laptops?", "Hey Room, what's on the table right now?", "Who moved my pill bottle?",
    "What did I miss?", "Show me my keys.",
]
SPELL = {"ok": "okay", "colour": "color", "whats": "what is", "wheres": "where is", "im": "i am"}


def words(text: str) -> list[str]:
    t = re.sub(r"['’`]", "", text.lower())
    out = []
    for w in re.findall(r"[a-z0-9]+", t):
        out += SPELL.get(w, w).split()
    return out


def wer(ref: str, hyp: str) -> tuple[int, int]:
    """(word edits, reference words)."""
    r, h = words(ref), words(hyp)
    d = list(range(len(h) + 1))
    for i in range(1, len(r) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(h) + 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (r[i - 1] != h[j - 1]))
    return d[len(h)], len(r)


def load_set(d: Path) -> list[dict]:
    items = json.loads((d / "truth.json").read_text())
    for it in items:
        it["audio"] = read_wav(d / it["wav"])
    return items


# ---------------------------------------------------------------- STT setups

class Whisper:
    def __init__(self, model: str, models_dir: str):
        from voice.stt import PyWhisperCppBackend
        self.name = model
        self.b = PyWhisperCppBackend(str(Path(models_dir) / f"ggml-{model}.bin") if models_dir else model)

    def __call__(self, audio: np.ndarray, prompt: str) -> str:
        n = int(1.1 * RATE)
        if len(audio) < n:
            audio = np.concatenate([audio, np.zeros(n - len(audio), np.float32)])
        return clean(self.b.transcribe(audio, prompt, 0))     # the rig runs audio_ctx 0 (config.yaml)


class Server:
    """A running whisper-server (how the rig runs whisper.cpp: the model stays loaded on the GPU)."""

    def __init__(self, url: str, label: str):
        from voice.stt import WhisperServerBackend
        host, port = url.rsplit(":", 1)
        self.name = label or f"server:{port}"
        self.b = WhisperServerBackend(start=False, host=host, port=int(port))

    def __call__(self, audio: np.ndarray, prompt: str) -> str:
        n = int(1.1 * RATE)
        if len(audio) < n:
            audio = np.concatenate([audio, np.zeros(n - len(audio), np.float32)])
        return clean(self.b.transcribe(audio, prompt, 0))


class Cloud:
    """An OpenAI-compatible /audio/transcriptions endpoint (Groq, OpenAI, ...)."""

    APIS = {"groq": ("https://api.groq.com/openai/v1", "whisper-large-v3-turbo", "GROQ_API_KEY"),
            "openai": ("https://api.openai.com/v1", "gpt-4o-mini-transcribe", "OPENAI_API_KEY")}

    def __init__(self, which: str):
        import requests
        self.url, self.model, env = self.APIS[which]
        self.key = os.environ.get(env, "").strip()
        if not self.key:
            raise SystemExit(f"{which}: set ${env}")
        self.name, self.s = f"{which}:{self.model}", requests.Session()

    def __call__(self, audio: np.ndarray, prompt: str) -> str:
        buf = io.BytesIO()
        write_wav(buf, audio)
        r = self.s.post(f"{self.url}/audio/transcriptions", headers={"Authorization": f"Bearer {self.key}"},
                        files={"file": ("clip.wav", buf.getvalue(), "audio/wav")},
                        data={"model": self.model, "prompt": prompt, "language": "en", "temperature": "0",
                              "response_format": "json"}, timeout=15)
        r.raise_for_status()
        return clean(r.json().get("text", ""))


class XaiStt:
    """xAI's speech to text (POST https://api.x.ai/v1/stt, $XAI_API_KEY, the rig's key): keyterms, not a prompt."""

    KEYTERMS = ["Room", "pill bottle", "keys", "wallet", "glasses", "remote", "notebook", "laptops", "kitchen counter",
                "couch", "meds", "specs"]

    def __init__(self, model: str = "grok-voice-transcribe-2.0"):
        import requests
        self.key = os.environ.get("XAI_API_KEY", "").strip()
        if not self.key:
            raise SystemExit("xai: set $XAI_API_KEY")
        self.model, self.name, self.s = model, f"xai:{model}", requests.Session()

    def __call__(self, audio: np.ndarray, prompt: str) -> str:
        buf = io.BytesIO()
        write_wav(buf, audio)
        data = [("model", self.model), ("language", "en")] + ([("keyterm", k) for k in self.KEYTERMS] if prompt else [])
        r = self.s.post("https://api.x.ai/v1/stt", headers={"Authorization": f"Bearer {self.key}"},
                        files={"file": ("clip.wav", buf.getvalue(), "audio/wav")}, data=data, timeout=15)
        r.raise_for_status()
        return clean(r.json().get("text", ""))


def run(stt, items: list[dict], prompt: str, cfg: dict) -> dict:
    edits = total = right = asked = 0
    ms, noise, misses = [], [], []
    for it in items:
        t0 = time.monotonic()
        got = stt(it["audio"], prompt)
        ms.append(1000 * (time.monotonic() - t0))
        it.setdefault("got", {})[f"{stt.name}|{prompt[:12]}"] = got
        if not it["text"]:
            if got and not filler_only(got):
                noise.append((it["wav"], got, bare_wake(got, cfg) or has_wake_word(got, cfg), echoes_prompt(got, prompt)))
            continue
        e, n = wer(it["text"], got)
        edits, total = edits + e, total + n
        a, b = parse(it["text"], cfg), parse(got, cfg)
        asked += 1
        ok = (a.kind, a.obj) == (b.kind, b.obj) and (bare_wake(it["text"], cfg) == bare_wake(got, cfg))
        right += ok
        if e:
            misses.append((it["wav"], it["text"], got, ok))
    return {"wer": edits / max(total, 1), "intent": (right, asked), "ms": ms, "noise": noise, "misses": misses,
            "noise_clips": sum(1 for it in items if not it["text"])}


def report(name: str, prompt_name: str, r: dict, show: bool) -> None:
    ms = sorted(r["ms"])
    p90 = ms[int(0.9 * (len(ms) - 1))] if ms else 0
    woke = sum(1 for n in r["noise"] if n[2])
    echo = sum(1 for n in r["noise"] if n[3])
    print(f"{name:28} {prompt_name:5} WER {100 * r['wer']:5.1f}%  intent {r['intent'][0]:2}/{r['intent'][1]:<2}  "
          f"{statistics.median(ms) if ms else 0:6.0f} / {p90:6.0f} ms  noise {len(r['noise'])}/{r['noise_clips']} "
          f"(wakes {woke}, prompt echo {echo})")
    if show:
        for wav, ref, got, ok in r["misses"]:
            print(f"    {'  ' if ok else 'X '}{wav}: {ref!r} -> {got!r}")
        for wav, got, w, e in r["noise"]:
            print(f"    noise {wav}: {got!r}{'  WAKES' if w else ''}{'  ECHO' if e else ''}")


# ---------------------------------------------------------------- synthetic set

def _grok_tts(text: str, voice: str) -> np.ndarray | None:
    import requests
    key = os.environ.get("XAI_API_KEY", "").strip()
    if not key:
        return None
    r = requests.post("https://api.x.ai/v1/tts", headers={"Authorization": f"Bearer {key}"}, timeout=30,
                      json={"text": text, "voice_id": voice, "language": "en",
                            "output_format": {"codec": "pcm", "sample_rate": RATE}})
    r.raise_for_status()
    return np.frombuffer(r.content, np.int16).astype(np.float32) / 32768


def _to_rate(x: np.ndarray, rate: int) -> np.ndarray:
    if rate == RATE:
        return x
    t = np.arange(int(len(x) * RATE / rate)) * rate / RATE
    return np.interp(t, np.arange(len(x)), x).astype(np.float32)


def _mix(x: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    noise = np.resize(noise, len(x))
    px, pn = np.mean(x ** 2) + 1e-12, np.mean(noise ** 2) + 1e-12
    return (x + noise * np.sqrt(px / pn / 10 ** (snr_db / 10))).astype(np.float32)


def _reverb(x: np.ndarray, rt60: float, rng) -> np.ndarray:
    n = int(rt60 * RATE)
    ir = rng.standard_normal(n) * np.exp(-6.9 * np.arange(n) / n)
    ir[0] = 1.0
    y = np.convolve(x, ir / np.sqrt(np.sum(ir ** 2)))[:len(x) + n // 2]
    return (y / (np.abs(y).max() + 1e-9) * np.abs(x).max()).astype(np.float32)


def synthesize(out: Path, cfg: dict) -> int:
    """Piper (and Grok voices if $XAI_API_KEY) saying QUESTIONS: clean, over babble at 10 dB, with a 0.5 s room
    reverb; plus noise-only clips (room tone, babble of reversed speech: no words in it)."""
    from voice.tts import TTS
    rng = np.random.default_rng(7)
    out.mkdir(parents=True, exist_ok=True)
    tts, voices, truth, clips = TTS(cfg), [], [], {}
    for q in QUESTIONS:
        a, rate, _ = tts.synthesize_piper(q)
        clips.setdefault("piper", []).append((q, _to_rate(a.astype(np.float32) / 32768, rate)))
    for v in ("eve", "ara", "rex"):
        try:
            got = [(q, _grok_tts(q, v)) for q in QUESTIONS]
        except Exception as ex:
            print(f"grok voice {v}: {ex}")
            continue
        if all(a is not None for _, a in got):
            clips[f"grok-{v}"] = got
    babble = np.concatenate([a[::-1] for _, a in clips["piper"]])     # reversed speech: speech-like, no words
    babble = sum(np.roll(babble, k * 7919) for k in range(4)) / 4
    k = 0
    for voice, qs in clips.items():
        for q, a in qs:
            a = np.concatenate([np.zeros(int(0.3 * RATE), np.float32), a, np.zeros(int(0.3 * RATE), np.float32)])
            for cond, x in (("clean", a), ("babble10", _mix(a, babble, 10)), ("reverb", _reverb(a, 0.5, rng))):
                k += 1
                name = f"{k:03d}-{voice}-{cond}.wav"
                write_wav(out / name, x)
                truth.append({"wav": name, "text": q, "note": f"{voice} {cond}"})
    for i in range(6):
        k += 1
        tone = rng.standard_normal(3 * RATE).astype(np.float32) * 10 ** ((-55 + 5 * (i % 3)) / 20)
        x = tone if i < 3 else _mix(tone, babble[i * RATE:], 0) * 0.05
        name = f"{k:03d}-noise-{'room' if i < 3 else 'babble'}.wav"
        write_wav(out / name, x)
        truth.append({"wav": name, "text": "", "note": "noise only"})
    (out / "truth.json").write_text(json.dumps(truth, indent=1))
    print(f"{len(truth)} clips in {out} (voices: {', '.join(clips)})")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("dir", nargs="?", help="a clip set: WAVs + truth.json")
    ap.add_argument("--synthesize", metavar="DIR", help="make a synthetic clip set in DIR and stop")
    ap.add_argument("--models", default="base.en", help="whisper.cpp models, comma-separated")
    ap.add_argument("--models-dir", default="", help="where the ggml-*.bin files are (default: models/whisper)")
    ap.add_argument("--prompts", default="rig,demo", help="rig | demo | none, comma-separated")
    ap.add_argument("--server", default="", help="host:port of a running whisper-server, e.g. 127.0.0.1:8179")
    ap.add_argument("--label", default="", help="the --server's model, for the report")
    ap.add_argument("--cloud", default="", help="xai | groq | openai, comma-separated (API key in the environment)")
    ap.add_argument("-v", "--verbose", action="store_true", help="print each miss and noise transcript")
    ap.add_argument("--json", help="write every transcript to this file")
    a = ap.parse_args(argv)
    cfg = load_config()
    if a.synthesize:
        return synthesize(Path(a.synthesize), cfg)
    if not a.dir:
        ap.error("a clip set directory, or --synthesize DIR")
    items = load_set(Path(a.dir))
    rig = initial_prompt(cfg, synonyms=bool((cfg.get("stt") or {}).get("prompt_synonyms", False)))
    prompts = {"rig": rig, "demo": question_prompt(cfg), "none": ""}
    setups = [Whisper(m, a.models_dir) for m in a.models.split(",") if m and not a.server]
    setups += [Server(a.server, a.label)] if a.server else []
    setups += [XaiStt() if c == "xai" else Cloud(c) for c in a.cloud.split(",") if c]
    print(f"{len(items)} clips ({sum(1 for i in items if i['text'])} questions) in {a.dir}")
    for stt in setups:
        for p in a.prompts.split(","):
            stt(items[0]["audio"], prompts[p])                   # warm up
            report(stt.name, p, run(stt, items, prompts[p], cfg), a.verbose)
    if a.json:
        Path(a.json).write_text(json.dumps([{k: v for k, v in i.items() if k != "audio"} for i in items], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
