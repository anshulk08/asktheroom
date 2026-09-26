"""Live Grok check: 10 open-ended questions against demo_world, printing answer + latency.

    XAI_API_KEY=... .venv/bin/python scripts/grok_smoke.py

Does nothing (exit 0) when XAI_API_KEY is not set.
"""
from __future__ import annotations

import logging
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.config import load_config  # noqa: E402
from core.fakeworld import demo_world  # noqa: E402
from voice.llm import FALLBACK_TEXT, ask_grok  # noqa: E402

QUESTIONS = [
    "What's hidden on the table right now?",
    "Could my keys be somewhere other than the box?",
    "Did I take my pills this morning?",
    "What happened in the last few minutes?",
    "Is anything missing from the table?",
    "Which things have I touched recently?",
    "Where did my phone go?",
    "Are my glasses safe?",
    "What's under the notebook?",
    "Give me a quick summary of everything.",
]


def main() -> int:
    if not os.environ.get("XAI_API_KEY"):
        print("XAI_API_KEY not set; skipping live Grok smoke test.")
        return 0
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    cfg = load_config()
    world = demo_world()
    lat, fallbacks = [], 0
    for q in QUESTIONS:
        t0 = time.perf_counter()
        a = ask_grok(q, world, world.events, cfg)
        ms = (time.perf_counter() - t0) * 1000
        lat.append(ms)
        fallbacks += a.text == FALLBACK_TEXT
        print(f"[{ms:6.0f} ms] Q: {q}\n           A: {a.text}  (point_at={a.point_at})")
    lat.sort()
    print(f"\nmedian {lat[len(lat) // 2]:.0f} ms, max {lat[-1]:.0f} ms, fallbacks {fallbacks}/10")
    return 0


if __name__ == "__main__":
    sys.exit(main())
