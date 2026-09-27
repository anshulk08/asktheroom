"""Replays the rig's logged transcripts (tests/wake_eval.json) through the wake-word path of the voice loop.

Every transcript the always-on mic heard on Sat 26 Sep, in order, goes through what main.Room._voice_turn does
with it, with the rules only (understand.enabled false): Whisper's fillers dropped (voice.stt.filler_only), a bare
wake word opens the mic (voice.understand.bare_wake) and the next line is the question (Understander.after_wake),
anything else must pass Understander.screen and the overheard reading. stt.min_speech_ms needs the audio, so it
isn't replayed. Prints how many of the labelled lines went the right way and every unlabelled line (chatter)
that was answered or woke the rig. Target: no false line answered, every genuine one answered, every wake
line opens the mic.

    python scripts/eval_wake.py                 # listen.mode wake, as on the rig
    python scripts/eval_wake.py --mode always   # the always-on default of config.yaml
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.config import load_config  # noqa: E402
from voice.stt import filler_only  # noqa: E402
from voice.understand import IGNORE, Understander, bare_wake, has_wake_word  # noqa: E402

EVAL = Path(__file__).resolve().parent.parent / "tests" / "wake_eval.json"
WANT = {"false": "-", "genuine": "answered", "wake": "woke"}


def rig_config(cfg: dict, mode: str = "wake") -> dict:
    cfg = copy.deepcopy(cfg)
    cfg["listen"] = dict(cfg.get("listen") or {}, mode=mode)
    cfg["understand"] = dict(cfg.get("understand") or {}, enabled=False)
    return cfg


def replay(cfg: dict, sessions: list[dict]) -> list[dict]:
    """Each line with got: answered | woke | - (dropped), and ack (the chime for "Room, ..." answered)."""
    out = []
    for s in sessions:
        u = Understander(cfg)                   # a fresh rig per log file, as it was restarted
        woke = False
        for e in s["lines"]:
            text = "" if filler_only(e["text"]) else e["text"]
            ack = False
            if woke and text and bare_wake(text, cfg):   # said again: main.Room._asked listens again (twice at most)
                got = "woke"
            elif woke:                          # main.Room._asked(after_wake=True)
                woke, got = False, "answered" if text and u.after_wake(text) else "-"
            elif not text:
                got = "-"
            elif bare_wake(text, cfg):
                woke, got = True, "woke"
            elif u.screen(text) and u(text, overheard=True).kind != IGNORE:
                got, ack = "answered", has_wake_word(text, cfg)
            else:
                got = "-"
            out.append(dict(e, log=s["log"], got=got, ack=ack))
    return out


def score(lines: list[dict]) -> dict[str, tuple[int, int]]:
    """label -> (lines that went the right way, lines with that label)."""
    return {k: (sum(1 for e in lines if e.get("label") == k and e["got"] == want),
                sum(1 for e in lines if e.get("label") == k)) for k, want in WANT.items()}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--file", default=str(EVAL))
    ap.add_argument("--mode", default="wake", choices=["wake", "always"], help="listen.mode (the rig: wake)")
    a = ap.parse_args(argv)
    doc = json.loads(Path(a.file).read_text())
    lines = replay(rig_config(load_config(), a.mode), doc["sessions"])
    print(f"{len(lines)} transcripts, listen.mode {a.mode}")
    for k, (ok, n) in score(lines).items():
        print(f"  {k:8} {ok}/{n} {'not answered' if k == 'false' else WANT[k]}")
    before = sum(1 for e in lines if e.get("answered"))
    now = sum(1 for e in lines if e["got"] == "answered")
    print(f"  answered: {now} (the rig then: {before}); woke: {sum(1 for e in lines if e['got'] == 'woke')}")
    bad = [e for e in lines if (e.get("label") in WANT and e["got"] != WANT[e["label"]])
           or (e.get("label") is None and e["got"] != "-")]   # always mode answers real questions without "room"
    for e in bad:
        print(f"  MISS {e.get('label') or 'chatter':8} got {e['got']:8} {e['log']} {e['t']}  {e['text']!r}")
    for e in lines:
        if e.get("label") == "unclear":
            print(f"  unclear          got {e['got']:8} {e['log']} {e['t']}  {e['text']!r}")
    return 1 if any(e.get("label") or a.mode == "wake" for e in bad) else 0


if __name__ == "__main__":
    raise SystemExit(main())
