"""Closed-loop STT benchmark: Piper speaks the test questions -> 16 kHz WAV -> voice.stt backends.

    python3 third_party/stt_bench.py --models base.en tiny.en --ctx sized 256 384 0
    python3 third_party/stt_bench.py --vad            # Silero end-of-speech on padded / noisy clips

Reference intent = voice.intents.parse(reference text). Latency is wall time of one transcribe call.
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.config import load_config  # noqa: E402
from voice import stt  # noqa: E402
from voice.intents import parse  # noqa: E402

EXTRA = ["Where are my keys?", "Did I take my pills?", "What changed?"]
WAVS = ROOT / "third_party" / "stt_bench" / "wavs"


def questions() -> list[str]:
    qs = [q["text"] for q in json.loads((ROOT / "tests" / "stt_questions.json").read_text())]
    return qs + [q for q in EXTRA if q not in qs]


def resample(a: np.ndarray, rate: int) -> np.ndarray:
    try:
        from math import gcd

        from scipy.signal import resample_poly
        g = gcd(stt.RATE, rate)
        return resample_poly(a, stt.RATE // g, rate // g).astype(np.float32)
    except ImportError:
        t = np.arange(int(len(a) * stt.RATE / rate)) * rate / stt.RATE
        return np.interp(t, np.arange(len(a)), a).astype(np.float32)


def clips(cfg) -> list[tuple[str, np.ndarray]]:
    WAVS.mkdir(parents=True, exist_ok=True)
    out, tts = [], None
    for i, q in enumerate(questions(), 1):
        p = WAVS / f"{i:02d}.wav"
        if not p.exists():
            if tts is None:
                from voice.tts import TTS
                tts = TTS(cfg)
            a, rate, _ = tts.synthesize_piper(q)
            stt.write_wav(p, resample(a.astype(np.float32) / 32768, rate))
        out.append((q, stt.read_wav(p)))
    return out


def norm(t: str) -> str:
    return re.sub(r"[^a-z ]", "", t.lower().replace("'", "")).strip()


def pad(a: np.ndarray) -> np.ndarray:
    n = int(stt.MIN_WHISPER_S * stt.RATE)
    return np.concatenate([a, np.zeros(max(0, n - len(a)), np.float32)])


def run(label, fn, data, cfg, rows):
    ok = exact = 0
    ms = []
    for q, a in data:
        t0 = time.monotonic()
        text = stt.clean(fn(a))
        dt = 1000 * (time.monotonic() - t0)
        ref, got = parse(q, cfg), parse(text, cfg)
        good = (ref.kind, ref.obj) == (got.kind, got.obj)
        ok += good
        exact += norm(text) == norm(q)
        ms.append(dt)
        rows.append({"cfg": label, "q": q, "text": text, "ms": round(dt), "intent_ok": good})
        print(f"  {label:28s} {dt:6.0f} ms {'OK ' if good else 'BAD'} {q!r} -> {text!r}", flush=True)
    s = (f"{label:28s} intent {ok}/{len(data)}  exact {exact}/{len(data)}  median {statistics.median(ms):.0f} ms"
         f"  p90 {sorted(ms)[int(0.9 * (len(ms) - 1))]:.0f} ms  max {max(ms):.0f} ms")
    print("SUMMARY " + s, flush=True)
    return s


def cli_timings(exe, model, wav, prompt, ctx) -> dict:
    """One whisper-cli run WITHOUT -np so it prints whisper_print_timings; split the wall time."""
    t0 = time.monotonic()
    r = subprocess.run([exe, "-m", model, "-f", str(wav), "-l", "en", "-nt", "-t", "4", "-ac", str(ctx),
                        "--prompt", prompt], capture_output=True, text=True, timeout=60)
    wall = 1000 * (time.monotonic() - t0)
    t = {k: float(v) for k, v in re.findall(r"whisper_print_timings:\s+(\w+) time =\s+([\d.]+) ms", r.stderr)}
    t["wall"] = wall
    dev = [ln.strip() for ln in r.stderr.splitlines() if "CUDA" in ln or "Metal" in ln or "MTL" in ln][:3]
    t["device_lines"] = dev
    return t


def bench(a):
    cfg = load_config()
    data = clips(cfg)
    prompt = stt.initial_prompt(cfg, synonyms=a.synonyms)
    print(f"{len(data)} clips, {min(len(x) for _, x in data) / stt.RATE:.2f}-"
          f"{max(len(x) for _, x in data) / stt.RATE:.2f} s; prompt {prompt!r}")
    rows, summ = [], []
    for model in a.models:
        mcfg = dict(cfg, stt=dict(cfg.get("stt") or {}, model=model, backend="cli"))
        cli = stt.make_backend(mcfg)
        if "cli" in a.backends:
            # the teammate's path, unmodified: STT.transcribe with the cli backend
            for ctx in a.cli_ctx:
                s = stt.STT(dict(mcfg, stt=dict(mcfg["stt"], audio_ctx=ctx, prompt_synonyms=a.synonyms)))
                s.warm()
                summ.append(run(f"cli {model} ctx={ctx}", s.transcribe, data, cfg, rows))
            # overhead split on three clips
            for q, x in data[:3]:
                wav = WAVS / "tmp.wav"
                stt.write_wav(wav, pad(x))
                for ctx in (stt.audio_ctx_for(len(pad(x))), 0):
                    t = cli_timings(cli.exe, cli.model, wav, prompt, ctx)
                    inside = t.get("total", 0)
                    print(f"  CLI SPLIT {model} ctx={ctx:4d} wall {t['wall']:.0f} ms = total(in whisper) "
                          f"{inside:.0f} (load {t.get('load', 0):.0f}, encode {t.get('encode', 0):.0f}, "
                          f"decode {t.get('decode', 0):.0f}, batchd {t.get('batchd', 0):.0f}) + outside "
                          f"{t['wall'] - inside:.0f}  {q!r}", flush=True)
                    if t["device_lines"]:
                        print("   ", t["device_lines"])
        if "server" in a.backends:
            srv = stt.WhisperServerBackend(model, port=a.port, threads=4)
            try:
                if not srv.healthy():
                    raise RuntimeError("server not healthy")
                srv.transcribe(pad(np.zeros(stt.RATE, np.float32)), prompt, 0)   # warm-up
                for ctx in a.ctx:
                    def fn(x, ctx=ctx):
                        x = pad(x)
                        c = stt.audio_ctx_for(len(x)) if ctx == "sized" else int(ctx)
                        return srv.transcribe(x, prompt, c)
                    summ.append(run(f"server {model} ctx={ctx}", fn, data, cfg, rows))
            finally:
                srv.close()
    print("\n".join(["", "=== summary"] + summ))
    if a.json:
        Path(a.json).write_text(json.dumps(rows, indent=1))


def vad(a):
    """Feed padded / noisy clips through STT.record_until_silence (unmodified) via a fake mic."""
    cfg = load_config()
    data = clips(cfg)
    rng = np.random.default_rng(0)
    B, R = stt.BLOCK, stt.RATE
    # babble: 6 overlapping talkers, each a shuffled run of the other questions (a crowd nearby)
    streams = []
    for k in range(6):
        order = rng.permutation(len(data))
        st = np.concatenate([np.concatenate([data[i][1], np.zeros(int(R * 0.15), np.float32)]) for i in order])
        streams.append(np.roll(st, rng.integers(len(st))))
    babble = sum(st[:min(len(t) for t in streams)] for st in streams)

    def rms(x):
        return float(np.sqrt(np.mean(x ** 2)) + 1e-12)

    def pink(n):
        w = rng.standard_normal(n).astype(np.float32)
        f = np.fft.rfft(w)
        f /= np.sqrt(np.arange(1, len(f) + 1))
        return np.fft.irfft(f, n).astype(np.float32)

    conds = [("silence", None, None), ("white 20dB", "white", 20), ("white 10dB", "white", 10),
             ("pink 10dB", "pink", 10), ("pink 0dB", "pink", 0),
             ("babble 20dB", "babble", 20), ("babble 10dB", "babble", 10), ("babble 5dB", "babble", 5)]
    s = stt.STT(cfg)
    s.end_drop_db = a.drop
    backend = None
    if a.transcribe:
        backend = stt.make_backend(dict(cfg, stt=dict(cfg.get("stt") or {}, backend=a.vad_backend)))
    print(f"end_drop_db {a.drop}; lead 1.0 s, tail 6.0 s; silence_ms {s.silence_ms:.0f}, max_s {s.max_s}, threshold {s.threshold}")
    for name, kind, snr in conds:
        ends, lat, heard_ok, starts, hit_max = [], [], 0, [], 0
        for q, x in data:
            lead, tail = np.zeros(R, np.float32), np.zeros(6 * R, np.float32)
            sig = np.concatenate([lead, x, tail])
            if kind:
                n = {"white": lambda k: rng.standard_normal(k).astype(np.float32),
                     "pink": pink,
                     "babble": lambda k: np.resize(np.roll(babble, rng.integers(len(babble))), k)}[kind](len(sig))
                n *= rms(x) / rms(n) / (10 ** (snr / 20))
                sig = sig + n
            blocks = [sig[i:i + B] for i in range(0, len(sig) - B + 1, B)]
            fed = []

            def mic(rate, block, device=None, blocks=blocks, fed=fed):
                class In:
                    def read(self, timeout=1.0):
                        if not blocks:
                            return None
                        fed.append(1)
                        return blocks.pop(0)

                    def close(self):
                        pass
                return In()
            stt.open_input = mic
            t0 = time.monotonic()
            clip = s.record_until_silence()
            proc = time.monotonic() - t0
            stop_t = len(fed) * B / R                 # audio time at which recording stopped
            speech_end = (R + len(x)) / R
            hit_max += stop_t >= s.max_s - 0.04
            if len(clip):
                ends.append(stop_t - speech_end)
            lat.append(1000 * proc / max(1, len(fed)))
            if backend is not None and len(clip):
                c = stt.audio_ctx_for(len(pad(clip))) if a.vad_ctx == "sized" else int(a.vad_ctx)
                text = stt.clean(backend.transcribe(pad(clip), s.prompt, c))
                ref, got = parse(q, cfg), parse(text, cfg)
                heard_ok += (ref.kind, ref.obj) == (got.kind, got.obj)
                if (ref.kind, ref.obj) != (got.kind, got.obj):
                    print(f"    {name}: {q!r} -> {text!r}")
            starts.append(len(clip) / R)
        e = np.array(ends) if ends else np.array([np.nan])
        print(f"{name:12s} speech found {len(ends)}/{len(data)}; stop after speech end: median "
              f"{np.median(e):.2f} s, min {e.min():.2f}, max {e.max():.2f} "
              f"(hit max_s {hit_max}); clip {np.median(starts):.2f} s; VAD {np.mean(lat):.2f} ms/block"
              + (f"; intent ok {heard_ok}/{len(data)}" if backend is not None else ""), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=["base.en", "tiny.en"])
    ap.add_argument("--backends", nargs="*", default=["cli", "server"])
    ap.add_argument("--ctx", nargs="*", default=["sized", "0"])
    ap.add_argument("--port", type=int, default=8179)
    ap.add_argument("--cli-ctx", nargs="*", default=["sized", "0"])
    ap.add_argument("--json")
    ap.add_argument("--synonyms", action="store_true", help="prompt lists the synonyms too")
    ap.add_argument("--vad", action="store_true")
    ap.add_argument("--transcribe", action="store_true", help="with --vad: also transcribe the VAD clips")
    ap.add_argument("--vad-backend", default="server")
    ap.add_argument("--drop", type=float, default=None, help="with --vad: stt.end_drop_db")
    ap.add_argument("--vad-ctx", default="sized", help="with --vad --transcribe: audio_ctx (sized or a number)")
    a = ap.parse_args()
    vad(a) if a.vad else bench(a)
