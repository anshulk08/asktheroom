"""Evaluation report (spec 3.13). Owner: eval.

    python -m eval.report --trials trials/ --out report.md [--db [data/events.db]]

Reads trials/*/results/<system>.json and prints a markdown table: rows = categories (plus a
'hidden' subtotal of covered+inside+inside_box_moved and an overall row), columns = systems,
cells = correct/total (percent). Below it: median laser error (cm) and median prediction latency
(ms) per system. With --db, also the median end-to-end question latency from the questions table
of an events.db (live runs). Writes report.md and prints it.
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path

if __package__ in (None, ""):  # allow `python eval/<name>.py` as well as `python -m eval.<name>`
    _sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

import argparse
import sqlite3
import sys
from pathlib import Path
from statistics import median
from typing import Optional

from eval.replay import SYSTEMS
from eval.trial import CATEGORIES, HIDDEN, STRETCH, load_trials


def collect(trials_dir: str) -> dict[str, list[dict]]:
    """{system: [result dicts]} across all trials."""
    out: dict[str, list[dict]] = {}
    for t in load_trials(trials_dir):
        for sys_name, r in t.results().items():
            out.setdefault(sys_name, []).append(r)
    return out


def _cell(rs: list[dict]) -> str:
    if not rs:
        return "-"
    done = [r for r in rs if not r.get("skipped")]
    if not done:
        return "skipped"
    ok = sum(bool(r.get("correct")) for r in done)
    return f"{ok}/{len(done)} ({100.0 * ok / len(done):.0f}%)"


def _med(vals: list[float]) -> Optional[float]:
    vals = [v for v in vals if v is not None]
    return median(vals) if vals else None


def _fmt(v: Optional[float], nd: int = 1, unit: str = "") -> str:
    return "-" if v is None else f"{v:.{nd}f}{unit}"


def db_latency(db_path: str) -> tuple[Optional[float], int]:
    """(median latency_ms, n) from an events.db questions table, or (None, 0)."""
    if not Path(db_path).exists():
        return None, 0
    try:
        con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        rows = con.execute("SELECT latency_ms FROM questions WHERE latency_ms IS NOT NULL").fetchall()
        con.close()
    except sqlite3.Error:
        return None, 0
    vals = [r[0] for r in rows]
    return (median(vals) if vals else None), len(vals)


def render(results: dict[str, list[dict]], db: Optional[str] = None) -> str:
    systems = [s for s in SYSTEMS if s in results] + sorted(s for s in results if s not in SYSTEMS)
    if not systems:
        return "# Ask the Room - evaluation\n\nNo results found. Run eval/replay.py first.\n"
    present = {r["category"] for rs in results.values() for r in rs}
    cats = [c for c in CATEGORIES if c in present] + sorted(present - set(CATEGORIES))

    def rows_for(sys_name: str, keep) -> list[dict]:
        return [r for r in results.get(sys_name, []) if keep(r["category"])]

    n_trials = max(len(rs) for rs in results.values())
    lines = ["# Ask the Room - evaluation", "",
             f"{n_trials} trial(s). Cells are correct/total (percent). "
             "INSIDE/UNDER scored on parent, GONE on edge, VISIBLE within 5 cm, HELD on status.", "",
             "| category | n | " + " | ".join(systems) + " |",
             "|---|---:|" + "---:|" * len(systems)]
    for c in cats:
        n = len(rows_for(systems[0], lambda x, c=c: x == c))
        label = f"{c} (stretch)" if c in STRETCH else c
        lines.append(f"| {label} | {n} | "
                     + " | ".join(_cell(rows_for(s, lambda x, c=c: x == c)) for s in systems) + " |")
    if any(c in present for c in HIDDEN):
        n = len(rows_for(systems[0], lambda x: x in HIDDEN))
        lines.append(f"| **hidden** ({'+'.join(HIDDEN)}) | {n} | "
                     + " | ".join(f"**{_cell(rows_for(s, lambda x: x in HIDDEN))}**" for s in systems)
                     + " |")
    lines.append(f"| **overall** | {len(results[systems[0]])} | "
                 + " | ".join(f"**{_cell(results[s])}**" for s in systems) + " |")

    lines += ["", "| metric | " + " | ".join(systems) + " |", "|---|" + "---:|" * len(systems)]
    live = {s: [r for r in results[s] if not r.get("skipped")] for s in systems}
    lines.append("| median laser error (cm) | " + " | ".join(
        _fmt(_med([r.get("laser_err_cm") for r in live[s]]), 1) if live[s] else "skipped"
        for s in systems) + " |")
    lines.append("| median predict latency (ms) | " + " | ".join(
        _fmt(_med([r.get("predict_ms") for r in live[s]]), 3) if live[s] else "skipped"
        for s in systems) + " |")
    lines.append("| median update per frame (ms) | " + " | ".join(
        _fmt(_med([r.get("update_ms_median") for r in live[s]]), 3) if live[s] else "skipped"
        for s in systems) + " |")

    notes = []
    for s in systems:
        sk = [r for r in results[s] if r.get("skipped")]
        if sk:
            notes.append(f"- `{s}` skipped on {len(sk)} trial(s): {sk[0].get('reason')}")
        er = [r for r in results[s] if r.get("error")]
        if er:
            notes.append(f"- `{s}` raised on {len(er)} trial(s) (counted wrong), e.g. trial "
                         f"{er[0]['trial_id']}: {er[0]['error']}")
    if db:
        med, n = db_latency(db)
        notes.append(f"- end-to-end question latency from `{db}` questions table: "
                     + (f"median {med:.0f} ms over {n} question(s)" if n else "no questions logged"))
    if notes:
        lines += ["", *notes]
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--trials", default="trials")
    ap.add_argument("--out", default="report.md")
    ap.add_argument("--db", nargs="?", const="data/events.db", default=None,
                    help="events.db to read question latency from (default data/events.db)")
    a = ap.parse_args(argv)
    md = render(collect(a.trials), db=a.db)
    Path(a.out).write_text(md)
    print(md)
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
