"""The /live engineering view's question timeline: an in-memory ring of the app's own log records, grouped
into questions, so every stage of an answer shows with its time (heard -> route -> answer -> point cue ->
laser). Read only: nothing here can move or light the laser.

LiveLog is a logging.Handler on the loggers that tell the story (askroom.main, voice, core.room,
core.auto_name). A question opens at main's "heard: ..." (the mic) or at the router's "asked (...)" (the
phone, the dashboard); records within QUESTION_S of it join it, the rest go to the ambient feed (room-track
naming, locks, re-homes). The router and the laser log a `live` extra dict (voice/pipeline.py, main.py) with
the fields the page shows; messages are only parsed for what older lines carry.
"""
from __future__ import annotations

import collections
import logging
import re
import threading
import time
from typing import Optional

LOGGERS = ("askroom.main", "voice", "core.room", "core.auto_name")
QUESTION_S = 30.0          # a question's aim, lock and timing lines come within this of its start
MERGE_S = 5.0              # the router's "asked" this soon after "heard" is the same question
MAX_QUESTIONS = 50
MAX_RECORDS = 2000
MAX_AMBIENT = 200

HIT = {"in_box", "within_tol"}
ROOM_AIM = re.compile(r"laser -> (?:(?P<name>\S+) at )?room px \((?P<u>-?[\d.]+), (?P<v>-?[\d.]+)\): (?P<reason>\w+) "
                      r"after (?P<tries>\d+) tries, err (?P<err>[\d.]+|inf) px")
TABLE_AIM = re.compile(r"laser -> (?P<name>\S+) at \((?P<x>-?[\d.]+), (?P<y>-?[\d.]+)\) via (?P<chain>\S+?)"
                       r"(?:, err (?P<err>[\d.]+) cm)?$")
REFUSED = re.compile(r"^(?:laser refused|room aim refused|table aim refused|edge sweep refused|no position for"
                     r"|laser not aimed|laser failed)")
HEARD = re.compile(r"^heard: (?P<q>.+)$")
NAMING = re.compile(r"looks like|no usable name|naming room track")


def _stage(rec: logging.LogRecord, msg: str) -> str:
    """What a record is in a question's story."""
    live = getattr(rec, "live", None) or {}
    if live.get("stage"):
        return str(live["stage"])
    if HEARD.match(msg):
        return "heard"
    if msg.startswith("heard the wake word alone"):
        return "wake"
    if msg.startswith("no point cue"):
        return "cue"
    if msg.startswith("laser -> "):
        return "laser"
    if REFUSED.match(msg):
        return "refused"
    if msg.startswith("LASER LOCKED"):
        return "lock"
    if msg.startswith("timing "):
        return "timing"
    if rec.name == "voice.stt":
        return "stt"
    if rec.name.startswith("voice.tts"):
        return "speak"
    if rec.name.startswith("voice."):
        return "route"
    return "log"


def _laser_fields(msg: str) -> dict:
    m = ROOM_AIM.search(msg)
    if m:
        err = float(m["err"])
        return {"kind": "room", "target": m["name"], "px": [float(m["u"]), float(m["v"])], "reason": m["reason"],
                "tries": int(m["tries"]), "err_px": None if err == float("inf") else err}
    m = TABLE_AIM.search(msg)
    if m:
        return {"kind": "table", "target": m["name"], "cm": [float(m["x"]), float(m["y"])], "via": m["chain"],
                "err_cm": float(m["err"]) if m["err"] else None, "reason": "aimed"}
    return {}


class LiveLog(logging.Handler):
    """Bounded ring of records and the questions they make up."""

    def __init__(self, question_s: float = QUESTION_S):
        super().__init__(logging.INFO)
        self.question_s = question_s
        self.records: collections.deque = collections.deque(maxlen=MAX_RECORDS)
        self.questions: collections.deque = collections.deque(maxlen=MAX_QUESTIONS)
        self.ambient: collections.deque = collections.deque(maxlen=MAX_AMBIENT)
        self.wakes: collections.deque = collections.deque(maxlen=3)   # the wake word alone, for the next question
        self._seq = 0
        self._mu = threading.Lock()

    def emit(self, rec: logging.LogRecord) -> None:
        try:
            msg = rec.getMessage()
        except Exception:
            return
        try:
            self.add(rec, msg)
        except Exception:                   # the log must never break the app
            pass

    def add(self, rec: logging.LogRecord, msg: str) -> None:
        stage = _stage(rec, msg)
        live = dict(getattr(rec, "live", None) or {})
        row = {"t": rec.created, "logger": rec.name, "level": rec.levelname, "stage": stage, "msg": msg}
        if live:
            row["live"] = live
        if stage in ("laser", "refused"):
            row["laser"] = {**_laser_fields(msg), **(live.get("laser") or {})}
            if stage == "refused":
                row["laser"].setdefault("reason", "refused")
                row["laser"]["why"] = msg
        with self._mu:
            self.records.append(row)
            if stage == "stt":                                  # the mic's every clip: noise in a timeline
                return
            q = self.questions[-1] if self.questions else None
            if stage == "heard" or stage == "asked":
                text = (live.get("text") if stage == "asked" else None) or (HEARD.match(msg) or {"q": msg})["q"]
                if (stage == "asked" and q is not None and q["open"] and not q["asked"]
                        and rec.created - q["t"] <= MERGE_S):
                    q["asked"] = True                           # the mic's question reaching the router
                    q["source"] = live.get("source", q["source"])
                    q["stages"].append(row)
                    return
                self._seq += 1
                q = {"id": self._seq, "t": rec.created, "text": str(text).strip("'\""), "open": True,
                     "asked": stage == "asked", "source": live.get("source", "voice"), "stages": []}
                wake = [r for r in self.wakes if rec.created - r["t"] < 15]
                q["stages"] += wake[-1:]
                q["stages"].append(row)
                self.questions.append(q)
                return
            if q is not None and q["open"] and rec.created - q["t"] <= self.question_s and stage != "wake" \
                    and not (rec.name == "core.room" and NAMING.search(msg)) and rec.name != "core.auto_name":
                q["stages"].append(row)
                if stage in ("lock",):
                    self.ambient.append(row)
                return
            if q is not None and q["open"] and rec.created - q["t"] > self.question_s:
                q["open"] = False
            if stage == "wake":
                self.wakes.append(row)
            elif stage != "speak":
                self.ambient.append(row)

    def snapshot(self, now: Optional[float] = None) -> dict:
        """Questions newest first, each with its outcome; the ambient feed newest first."""
        now = time.time() if now is None else now
        with self._mu:
            qs = [dict(q, stages=list(q["stages"])) for q in self.questions]
            amb = list(self.ambient)
        return {"questions": [summarize(q, now) for q in reversed(qs)], "ambient": list(reversed(amb[-80:]))}


def summarize(q: dict, now: float) -> dict:
    """A question's stages with times relative to its start, the route, the answer and the laser outcome:
    hit (dot on it), miss (budget, jumped, not seen, ...), refused (a gate said no), no_cue (no 'point to'),
    spoken (nothing to aim at) or pending (still within its window with no answer yet)."""
    t0 = q["t"]
    stages, route, answer, laser, cue, timing = [], None, None, None, None, None
    for r in q["stages"]:
        s = dict(r, dt=round(r["t"] - t0, 3))
        stages.append(s)
        live = r.get("live") or {}
        if r["stage"] == "answer":
            route, answer = live.get("path"), live
        elif r["stage"] in ("laser", "refused"):
            laser = r.get("laser")
        elif r["stage"] == "cue":
            cue = "no point cue"
        elif r["stage"] == "timing":
            timing = r["msg"][len("timing "):]
    if laser and laser.get("reason") in HIT | {"aimed"}:
        outcome = "hit"
    elif laser and laser.get("reason") == "refused":
        outcome = "refused"
    elif laser:
        outcome = "miss"
    elif cue:
        outcome = "no_cue"
    elif answer is None and now - t0 < 20:
        outcome = "pending"
    else:
        outcome = "spoken"
    for i in range(1, len(stages)):
        stages[i - 1]["took"] = round(stages[i]["t"] - stages[i - 1]["t"], 3)
    return {"id": q["id"], "t": t0, "text": q["text"], "source": q["source"], "route": route,
            "answer": (answer or {}).get("answer"), "action": (answer or {}).get("action"), "cue": cue,
            "laser": laser, "timing": timing, "outcome": outcome, "stages": stages}


_LIVE: Optional[LiveLog] = None
_install_mu = threading.Lock()


def install() -> LiveLog:
    """The process's one LiveLog, attached to LOGGERS once (create_app runs more than once in tests)."""
    global _LIVE
    with _install_mu:
        if _LIVE is None:
            _LIVE = LiveLog()
            for name in LOGGERS:                    # levels are the app's (main.py logs INFO)
                logging.getLogger(name).addHandler(_LIVE)
        return _LIVE
