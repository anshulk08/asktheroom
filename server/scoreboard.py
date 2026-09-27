"""Room handoff scoreboard (spec 0010 P2-2): real trial counts from scripts/room_trials.py, never synthetic.

The trial driver runs on a laptop and writes a JSON list of runs to its --out file, one record per run:
    {"run", "zone", "on_table_s", "handoff_s", "answer", "result": "pass" | "fail" | "no table sighting",
     "return_answer", "return_result": "pass" | "fail"}
The records name neither the object nor the time, so the dashboard keeps its own copies: the laptop
uploads each results file with the object named,

    curl -X POST -H 'content-type: application/json' --data @data/room/trials.json \\
         'http://<rig>:8080/scoreboard/trials?object=remote'

and it is stored in scoreboard.trials_dir as {"object", "uploaded", "records"}. Plain driver files copied
into that directory by hand also count: the object is taken from a record's "object" field, else from the
file name (remote_2130.json -> remote), the time from the file's modification time.

The summary for one day (local time) counts:
  handoffs   runs that got as far as the carry step (result pass / fail) and how many passed; a run whose
             object was never seen on the table ("no table sighting") is counted apart as skipped
  returns    table returns attempted (return_result present) and passed
  median_s   the median handoff_s of passed handoffs (seconds from "carry it" to the rig seeing it there)
per object and per zone as well. No files, no numbers: the dashboard then says nothing was recorded.
"""
from __future__ import annotations

import datetime as dt
import json
import logging
import re
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

log = logging.getLogger("askroom.scoreboard")

OBJECT_RE = re.compile(r"^[a-z0-9][a-z0-9 _-]{0,31}$")
MAX_RECORDS = 200            # per upload: a trial session is 5-10 runs
RESULTS = ("pass", "fail", "no table sighting")


@dataclass
class Run:
    obj: str
    zone: str
    result: str                          # pass / fail / no table sighting
    handoff_s: Optional[float]
    return_result: Optional[str]         # pass / fail, or None when not attempted
    t: float                             # when the results were uploaded or the file was written


def trials_dir(cfg: dict) -> Path:
    return Path((cfg.get("scoreboard") or {}).get("trials_dir", "data/room"))


def valid_object(name: str) -> bool:
    return bool(OBJECT_RE.match(name or ""))


def valid_records(records) -> bool:
    """A room_trials.py results list: dicts with a known result and a zone."""
    return (isinstance(records, list) and 0 < len(records) <= MAX_RECORDS
            and all(isinstance(r, dict) and r.get("result") in RESULTS and isinstance(r.get("zone"), str)
                    for r in records))


def save_upload(folder: Path, obj: str, records: list, now: Optional[float] = None) -> Path:
    """Store one uploaded results file; returns its path. ValueError on a bad object name or records."""
    obj = (obj or "").strip().lower()
    if not valid_object(obj):
        raise ValueError("object must be 1-32 lowercase letters, digits, spaces, _ or -")
    if not valid_records(records):
        raise ValueError(f"expected room_trials.py results: a list of 1-{MAX_RECORDS} runs with zone and result")
    now = time.time() if now is None else now
    folder.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.fromtimestamp(now).strftime("%Y%m%d-%H%M%S")
    path = folder / f"{obj.replace(' ', '_')}_{stamp}.json"
    n = 1
    while path.exists():                 # two uploads in one second
        n += 1
        path = folder / f"{obj.replace(' ', '_')}_{stamp}-{n}.json"
    path.write_text(json.dumps({"object": obj, "uploaded": now, "records": records}, indent=1))
    return path


def _num(v) -> Optional[float]:
    return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else None


def load_runs(folder: Path) -> list[Run]:
    """Every run in every results file in folder; unreadable or foreign files are skipped (logged)."""
    runs: list[Run] = []
    if not folder.is_dir():
        return runs
    for p in sorted(folder.glob("*.json")):
        try:
            data = json.loads(p.read_text())
        except (OSError, ValueError):
            log.warning("scoreboard: %s is unreadable; skipped", p)
            continue
        if isinstance(data, dict):
            records, obj, t = data.get("records"), data.get("object"), _num(data.get("uploaded"))
        else:
            records, obj, t = data, None, None
        if not valid_records(records):
            log.warning("scoreboard: %s is not a room_trials results file; skipped", p)
            continue
        if t is None:
            t = p.stat().st_mtime
        file_obj = obj if isinstance(obj, str) and obj else p.stem.split("_")[0].split("-")[0]
        for r in records:
            o = r.get("object") if isinstance(r.get("object"), str) and r.get("object") else file_obj
            rr = r.get("return_result")
            runs.append(Run(obj=str(o).lower(), zone=r["zone"], result=r["result"], handoff_s=_num(r.get("handoff_s")),
                            return_result=rr if rr in ("pass", "fail") else None, t=float(t)))
    return runs


def _day(date: Optional[str], now: float) -> Optional[dt.date]:
    """'today' / None -> today, 'yesterday', 'all' -> None (every day), or YYYY-MM-DD; ValueError otherwise."""
    today = dt.datetime.fromtimestamp(now).date()
    if date in (None, "", "today"):
        return today
    if date == "yesterday":
        return today - dt.timedelta(days=1)
    if date == "all":
        return None
    return dt.date.fromisoformat(date)


def _tally(runs: Iterable[Run]) -> dict:
    runs = list(runs)
    tried = [r for r in runs if r.result in ("pass", "fail")]
    passed = [r for r in tried if r.result == "pass"]
    rets = [r for r in runs if r.return_result is not None]
    times = [r.handoff_s for r in passed if r.handoff_s is not None]
    return {"handoffs": {"passed": len(passed), "total": len(tried)},
            "returns": {"passed": sum(r.return_result == "pass" for r in rets), "total": len(rets)},
            "skipped": sum(r.result == "no table sighting" for r in runs),
            "median_s": round(statistics.median(times), 1) if times else None}


def summarize(runs: list[Run], date: Optional[str] = None, now: Optional[float] = None) -> dict:
    """The scoreboard for one day (see the module docstring). ValueError on a bad date."""
    now = time.time() if now is None else now
    day = _day(date, now)
    sel = [r for r in runs if day is None or dt.datetime.fromtimestamp(r.t).date() == day]
    out = {"date": "all" if day is None else day.isoformat(), "runs": len(sel), **_tally(sel)}
    out["by_object"] = {o: _tally(r for r in sel if r.obj == o) for o in sorted({r.obj for r in sel})}
    out["by_zone"] = {z: _tally(r for r in sel if r.zone == z) for z in sorted({r.zone for r in sel})}
    out["last_t"] = max((r.t for r in sel), default=None)
    return out
