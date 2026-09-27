"""Scores the spoken-question interpreter (voice.understand) on tests/understand_eval.json.

Each item is {text, kind, obj, set}. Sets stt20 and loose are clicker questions (asked); overheard
is always-on mic speech, where kind IGNORE means the rig must stay quiet. Prints accuracy per set
for rules only and rules + the model (understand.backend: Grok by default, needs $XAI_API_KEY; or
Qwen via llama-server, scripts/qwen_server.sh), the misses, and the model's median latency.

    python scripts/eval_understand.py                                  # understand.backend from config.yaml
    python scripts/eval_understand.py --backend qwen --url http://127.0.0.1:8082/v1   # a local model
    python scripts/eval_understand.py --rules                          # offline: the rule parser only
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
    ap.add_argument("--backend", choices=["grok", "qwen"], help="default: understand.backend")
    ap.add_argument("--url", help="qwen: llama-server base url (default: understand.url)")
    ap.add_argument("--file", default=str(EVAL))
    ap.add_argument("--rules", action="store_true", help="rules only, as the rig answers offline (no model)")
    a = ap.parse_args(argv)
    cfg = load_config()
    if a.url:
        cfg["understand"] = dict(cfg["understand"], url=a.url)
    if a.backend:
        cfg["understand"] = dict(cfg["understand"], backend=a.backend)
    items = json.loads(Path(a.file).read_text())
    if a.rules:
        return rules_only(cfg, items)
    qwen = Understander(cfg)
    model = qwen._name()
    if not qwen.warm():
        print(f"{model} not answering at {qwen.model.url}", file=sys.stderr)
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
                    if u.last_by == model:
                        ms.append(u.last_ms)
                    if not ok:
                        print(f"  miss {q['text']!r}: want {want[0]} {want[1]}, got {i.kind} {i.obj} ({u.last_by})")
        print(f"{name:10s} rules {got['rules']:2d}/{len(qs)}   rules+{model} {got['qwen']:2d}/{len(qs)}")
        for k in total:
            total[k] += got[k]
    print(f"{'all':10s} rules {total['rules']:2d}/{len(items)}   rules+{model} {total['qwen']:2d}/{len(items)}")
    if ms:
        print(f"{model} calls {len(ms)}, median {statistics.median(ms):.0f} ms, max {max(ms):.0f} ms")
    return 0


def rules_only(cfg: dict, items: list[dict]) -> int:
    """Score the rule parser alone (the offline rig), printing each miss."""
    rules = Understander(dict(cfg, understand=dict(cfg["understand"], enabled=False)))
    total = 0
    for name in dict.fromkeys(q["set"] for q in items):
        qs = [q for q in items if q["set"] == name]
        got = 0
        for q in qs:
            i = rules(q["text"], name == "overheard")
            ok = (i.kind, None if i.kind == IGNORE else i.obj) == (q["kind"], q["obj"])
            got += ok
            if not ok:
                print(f"  miss {q['text']!r}: want {q['kind']} {q['obj']}, got {i.kind} {i.obj}")
        print(f"{name:10s} rules {got:2d}/{len(qs)}")
        total += got
    print(f"{'all':10s} rules {total:2d}/{len(items)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
