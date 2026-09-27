"""Spoken end-to-end trials for the room voice (spec 0010 P0-1): a laptop 2 m from the rig's mic says the
question out loud, and the rig's own answer log (GET /state "answers") says what it heard, what it answered
and when. Done test: "where's my wallet" from 2 m, answered within 4 s, 5/5.

Runs on the laptop, not the rig: it only speaks through the laptop's speaker (macOS `say`, else espeak-ng)
and reads the app's HTTP API. The latency is from the end of the spoken question to the rig having its
answer (logged just before it speaks; add ~0.3 s to first sound). The laptop and rig clocks are aligned from
/state's server_t, so no clock sync is needed.

    python scripts/voice_trials.py --url http://RIG:8000 --expect counter                # 5 x "where's my wallet"
    python scripts/voice_trials.py --url http://RIG:8000 -q "where are my glasses" --expect couch --n 3
    python scripts/voice_trials.py --url http://RIG:8000 --expect counter --out data/voice_trials.json

Put the object in the expected zone first (--expect is a word the answer must contain). scripts/room_trials.py
(the room-memory trials) is separate and unchanged.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
import urllib.request
from typing import Callable, Optional

POLL_S = 0.1
WAIT_EXTRA_S = 6.0          # keep waiting this long past --limit before calling a trial unanswered


def get_state(url: str, timeout: float = 2.0) -> dict:
    with urllib.request.urlopen(f"{url.rstrip('/')}/state", timeout=timeout) as r:
        return json.loads(r.read().decode())


def say_cmd(text: str, voice: Optional[str] = None) -> list[str]:
    """The command that speaks text out of this machine's speaker and returns when it has finished."""
    if shutil.which("say"):
        return ["say"] + (["-v", voice] if voice else []) + [text]
    if shutil.which("espeak-ng"):
        return ["espeak-ng"] + (["-v", voice] if voice else []) + [text]
    raise SystemExit("no speech command: needs macOS `say` or espeak-ng")


def clock_offset(get: Callable[[], dict], clock: Callable[[], float] = time.time) -> float:
    """Rig wall clock minus ours, from one /state round trip (its server_t is read mid-request)."""
    t0 = clock()
    st = get()
    t1 = clock()
    return float(st.get("server_t", t1)) - (t0 + t1) / 2


def last_seq(state: dict) -> int:
    return max([int(a.get("seq", 0)) for a in state.get("answers") or []] or [0])


def trial(question: str, expect: str, limit_s: float, speak: Callable[[str], None], get: Callable[[], dict],
          offset: float, clock: Callable[[], float] = time.time, sleep: Callable[[float], None] = time.sleep) -> dict:
    """Say the question, wait for the rig's next voice answer. Returns one result row."""
    seq0 = last_seq(get())
    speak(question)
    t_end = clock()                                      # our clock: the question has been said
    deadline = t_end + limit_s + WAIT_EXTRA_S
    while clock() < deadline:
        for a in get().get("answers") or []:
            if int(a.get("seq", 0)) > seq0 and a.get("src") == "voice":
                latency = float(a.get("t", 0)) - offset - t_end
                text = str(a.get("text", ""))
                return {"q": question, "heard": a.get("q"), "text": text, "latency_s": round(latency, 2),
                        "ok": latency <= limit_s and expect.lower() in text.lower()}
        sleep(POLL_S)
    return {"q": question, "heard": None, "text": None, "latency_s": None, "ok": False}


def run(question: str, expect: str, n: int, limit_s: float, gap_s: float, speak, get,
        clock=time.time, sleep=time.sleep, out=print) -> list[dict]:
    offset = clock_offset(get, clock)
    out(f"rig clock {offset:+.2f} s from ours; {n} x {question!r}, expecting {expect!r} within {limit_s:.1f} s")
    rows = []
    for i in range(1, n + 1):
        r = trial(question, expect, limit_s, speak, get, offset, clock, sleep)
        rows.append(r)
        lat = "no answer" if r["latency_s"] is None else f"{r['latency_s']:.2f} s"
        out(f"[{i}/{n}] {'PASS' if r['ok'] else 'FAIL'}  {lat:>9}  heard {r['heard']!r} -> {r['text']!r}")
        if i < n:
            sleep(gap_s)                                   # let the rig finish speaking (and its echo tail)
    passed = sum(r["ok"] for r in rows)
    out(f"{passed}/{n} passed")
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--url", required=True, help="the rig app, e.g. http://10.90.84.178:8000")
    ap.add_argument("-q", "--question", default="where's my wallet")
    ap.add_argument("--expect", required=True, help="a word the answer must contain (the zone: counter, couch)")
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--limit", type=float, default=4.0, help="seconds from end of question to answer")
    ap.add_argument("--gap", type=float, default=8.0, help="seconds between trials")
    ap.add_argument("--voice", help="say / espeak-ng voice")
    ap.add_argument("--out", help="write the results as JSON here")
    a = ap.parse_args(argv)
    cmd = lambda text: subprocess.run(say_cmd(text, a.voice), check=True)   # noqa: E731
    rows = run(a.question, a.expect, a.n, a.limit, a.gap, cmd, lambda: get_state(a.url))
    if a.out:
        with open(a.out, "w") as f:
            json.dump({"question": a.question, "expect": a.expect, "limit_s": a.limit, "trials": rows}, f, indent=1)
    return 0 if all(r["ok"] for r in rows) else 1


if __name__ == "__main__":
    sys.exit(main())
