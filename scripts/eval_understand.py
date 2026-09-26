"""Scores the spoken-question interpreter (voice.understand) on tests/understand_eval.json.

Each item is {text, kind, obj, set}. Sets stt20 and loose are clicker questions (asked); overheard
is always-on mic speech, where kind IGNORE means the rig must stay quiet. Prints accuracy per set
for rules only and rules + Qwen, the misses, and Qwen's median latency. Start llama-server first
(scripts/qwen_server.sh); run it on the Jetson with YOLO and whisper loaded for real latency.

    python scripts/eval_understand.py                                  # understand.url from config.yaml
    python scripts/eval_understand.py --url http://127.0.0.1:8082/v1   # another model's server
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.config import load_config  # noqa: E402
from voice.understand import IGNORE, Understander  # noqa: E402

EVAL = Path(__file__).resolve().parent.parent / "tests" / "understand_eval.json"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--url", help="llama-server base url (default: understand.url)")
    ap.add_argument("--file", default=str(EVAL))
    a = ap.parse_args(argv)
    cfg = load_config()
    if a.url:
        cfg["understand"] = dict(cfg["understand"], url=a.url)
    items = json.loads(Path(a.file).read_text())
    qwen = Understander(cfg)
    if not qwen.warm():
        print(f"llama-server not answering at {cfg['understand']['url']}", file=sys.stderr)
        return 1
    rules = Understander(dict(cfg, understand=dict(cfg["understand"], enabled=False)))
    ms, total = [], {"rules": 0, "qwen": 0}
    for name in dict.fromkeys(q["set"] for q in items):
        got = {"rules": 0, "qwen": 0}
        qs = [q for q in items if q["set"] == name]
        for q in qs:
            overheard = name == "overheard"
            want = (q["kind"], q["obj"])
            for who, u in (("rules", rules), ("qwen", qwen)):
                i = u(q["text"], overheard)
                ok = (i.kind, None if i.kind == IGNORE else i.obj) == want
                got[who] += ok
                if who == "qwen":
                    if u.last_by == "qwen":
                        ms.append(u.last_ms)
                    if not ok:
                        print(f"  miss {q['text']!r}: want {want[0]} {want[1]}, got {i.kind} {i.obj} ({u.last_by})")
        print(f"{name:10s} rules {got['rules']:2d}/{len(qs)}   rules+qwen {got['qwen']:2d}/{len(qs)}")
        for k in total:
            total[k] += got[k]
    print(f"{'all':10s} rules {total['rules']:2d}/{len(items)}   rules+qwen {total['qwen']:2d}/{len(items)}")
    if ms:
        print(f"qwen calls {len(ms)}, median {statistics.median(ms):.0f} ms, max {max(ms):.0f} ms")
    return 0


if __name__ == "__main__":
    sys.exit(main())
