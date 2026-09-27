"""Hands-free room memory trials (spec 0009, the M0 "5/5 demo-moment runs" check), driven from a laptop.

The laptop speaks each instruction (macOS `say`; printed elsewhere), watches the rig's dashboard API until
the step happened, asks the rig "where is my <object>?" through POST /ask, speaks the answer and records
pass/fail. Each run: put the object on the table (the rig tracks it and Grok names it), carry it to the
run's zone (the rig hands it off), ask; then bring it back to the table and ask again (table return).

    python scripts/room_trials.py --rig http://10.90.84.178:8080 --zones couch side_table couch counter couch
    python scripts/room_trials.py --rig ... --object remote --out data/room/trials.json

Needs nothing but the standard library (and core.auto_name.match_score for the name check).
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.auto_name import match_score  # noqa: E402

SAY = {"couch": "the couch", "side_table": "the side table", "counter": "the kitchen counter",
       "stove": "the stove"}


def speak(text: str) -> None:
    print(f"\n>>> {text}", flush=True)
    if shutil.which("say"):
        subprocess.run(["say", "-r", "185", text], check=False)


def get_state(rig: str) -> dict:
    with urllib.request.urlopen(f"{rig}/state", timeout=5) as r:
        return json.load(r)["state"]


def ask(rig: str, text: str) -> str:
    req = urllib.request.Request(f"{rig}/ask", data=json.dumps({"text": text}).encode(),
                                 headers={"content-type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=20) as r:
        return json.load(r)["text"]


def matches(obj: str, e: dict) -> bool:
    """The configured prop itself, or an unnamed thing whose Grok guess fits the object's name."""
    if e.get("name") == obj:
        return True
    g = e.get("guess")
    return bool(g) and match_score(obj, g) >= 1.5


def where(st: dict, obj: str) -> list[tuple[str, str, str]]:
    """(entity, zone, status) of every entity that is the object (by name or Grok guess) and not UNKNOWN."""
    room = st.get("room") or {}
    out = []
    for e in st.get("entities") or []:
        if matches(obj, e) and e.get("status") not in ("UNKNOWN", None):
            r = room.get(e["name"])
            zone = r.get("zone") if isinstance(r, dict) else e.get("zone", "table")
            out.append((e["name"], zone or "table", e["status"]))
    return out


def wait_for(rig: str, obj: str, zone: str, timeout_s: float, poll_s: float = 1.0):
    """Until the object is seen in `zone` ('table' or a room zone). Returns (entity, seconds) or None."""
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout_s:
        try:
            hits = [h for h in where(get_state(rig), obj) if h[1] == zone and h[2] == "VISIBLE"]
        except Exception as ex:                      # the rig restarting, a dropped request: keep polling
            print(f"   (state unavailable: {ex})")
            hits = []
        if hits:
            return hits[0][0], time.monotonic() - t0
        time.sleep(poll_s)
    return None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rig", default="http://10.90.84.178:8080")
    ap.add_argument("--object", default="remote")
    ap.add_argument("--zones", nargs="+", default=["couch", "side_table", "couch", "counter", "couch"])
    ap.add_argument("--table-timeout", type=float, default=60.0)
    ap.add_argument("--carry-timeout", type=float, default=90.0)
    ap.add_argument("--out", default="data/room/trials.json")
    a = ap.parse_args(argv)
    obj, results = a.object, []
    speak(f"Room memory check. {len(a.zones)} runs with the {obj}. Follow my instructions.")
    for i, zone in enumerate(a.zones, 1):
        say = SAY.get(zone, zone.replace("_", " "))
        res = {"run": i, "zone": zone}
        speak(f"Run {i}. Put the {obj} on the coffee table, and leave it there.")
        got = wait_for(a.rig, obj, "table", a.table_timeout)
        res["on_table_s"] = None if got is None else round(got[1], 1)
        if got is None:
            speak(f"I don't see a {obj} on the table. Skipping run {i}.")
            res["result"] = "no table sighting"
            results.append(res)
            continue
        speak(f"Got it. Now carry the {obj} to {say}, put it down, and step away.")
        got = wait_for(a.rig, obj, zone, a.carry_timeout)
        res["handoff_s"] = None if got is None else round(got[1], 1)
        time.sleep(2.0)
        ans = ask(a.rig, f"where is my {obj}")
        res["answer"] = ans
        ok = say in ans.lower()
        res["result"] = "pass" if ok else "fail"
        speak(f"The rig says: {ans}")
        speak("Pass." if ok else f"That's a fail: I expected {say}.")
        speak(f"Now bring the {obj} back to the coffee table.")
        back = wait_for(a.rig, obj, "table", a.table_timeout)
        time.sleep(2.0)
        ans2 = ask(a.rig, f"where is my {obj}")
        res["return_answer"] = ans2
        res["return_result"] = "pass" if (back is not None and "table" in ans2.lower()) else "fail"
        speak(f"Back on the table. The rig says: {ans2}")
        results.append(res)
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(results, indent=1))
    n = sum(r["result"] == "pass" for r in results)
    m = sum(r.get("return_result") == "pass" for r in results)
    speak(f"Done. Handoffs: {n} of {len(a.zones)} passed. Table returns: {m} of {len(a.zones)} passed.")
    print(json.dumps(results, indent=1))
    return 0 if n == len(a.zones) else 1


if __name__ == "__main__":
    sys.exit(main())
