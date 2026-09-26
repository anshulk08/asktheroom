"""The care layer: reminders, the morning report and caregiver summary, conversation memory and profile
facts, in front of the normal question router (voice/pipeline.py).

    care = Care(cfg, world, events, base_ask)      # base_ask: voice.pipeline.make_ask(...)
    care.ask(text, source) -> Answer               # use it wherever ask(text, source) was used
    care.on_notice = room.respond                  # proactive notices are spoken and aimed like answers
    care.start() / care.stop()                     # the scheduler thread: care.tick() every care.tick_s

ask() handles, in order: a yes / no to a suggested reminder, 'okay' after a notice, reminder requests,
'give me today's summary', profile statements and questions, greetings; everything else goes through
conversation memory (follow-ups rewritten or answered) to the router. The morning report is added in
front of the first answer of the morning on spoken sources. A failure anywhere in the care layer is
logged and the question goes to the router as if the layer weren't there.

Conversation context for Grok: voice/llm.ask_grok has no parameter for earlier turns, so OTHER questions
reach it without them. Care.conversation.context(source, now) returns them as chat messages; passing
them would be one additive `history` argument to ask_grok, inserted before the user message in _run.

main.py wires it with attach_care(room, cfg) (see there), and create_app(..., care=care) adds `notices`
to /state, POST /notices/{id}/ack and GET /report.
"""
from __future__ import annotations

import logging
import re
import threading
import time
from dataclasses import replace
from typing import Callable, Optional

from core.carewords import say_time, spoken
from core.profile import Profile, STATEMENT, Suggestion, make_grok_extractor
from core.reminders import Notice, Reminders, care_cfg, daily_when, normalize, resolve_obj
from core.reports import MorningReport, caregiver_summary, morning_report, spoken_summary, summary_data
from core.types import Answer
from voice.conversation import Conversation

log = logging.getLogger(__name__)

TEXT_ONLY = ("sms", "n8n")          # answered in text: no morning report, no profile statements
SUGGEST_TTL_S = 120
YES = re.compile(r"^(?:yes|yeah|yep|yup|sure|ok|okay|please|please do|do it|go ahead|sounds good|alright|all right"
                 r"|yes please|that would be good|that would be great|good idea)(?: please| thanks| thank you)*$")
NO = re.compile(r"^(?:no|nope|nah|dont|do not|not now|never mind|nevermind|no need)(?: thanks| thank you| please)*$")
SUMMARY = re.compile(r"\b(?:(?P<day>today|yesterday)s?|daily|the days?|my days?)\s+(?:summary|report|recap)\b"
                     r"|\bsumm(?:ary|arize|arise)\s+(?:of\s+)?(?P<day2>today|yesterday|my day)\b"
                     r"|\bhow was (?P<day3>today|yesterday|my day)\b")
MORNING = re.compile(r"\bmorning (?:report|briefing|summary)\b")
GREETING = re.compile(r"^(?P<g>hi|hello|hey|good morning|good afternoon|good evening|morning|evening)"
                      r"(?: there| room| again)?$")


class Care:
    def __init__(self, cfg: dict, world, events, base_ask: Callable[[str, str], Answer],
                 clock: Callable[[], float] = time.time, extractor=None, online: Optional[Callable[[], bool]] = None):
        self.cfg, self.world, self.events, self.base_ask, self.clock = cfg, world, events, base_ask, clock
        self.c = care_cfg(cfg)
        self.reminders = Reminders(cfg, world, events)
        if extractor is None and self.c.get("profile_llm", True):
            extractor = make_grok_extractor(cfg)
        self.profile = Profile(events, extractor=extractor)
        self.morning = MorningReport(cfg, world, events, self.reminders, self.profile)
        self.conversation = Conversation(cfg)
        self.online = online or (lambda: bool(getattr(world, "online", False)))
        self.on_notice: Optional[Callable[[Answer], object]] = None
        self.tick_s = float(self.c.get("tick_s", 10))
        # demo.hold_notices: during judging nothing speaks unasked. Reminders are still recorded and shown
        # (/state notices, the phone); the morning report is only said when asked for.
        self.hold = bool((cfg.get("demo") or {}).get("hold_notices", False))
        self._pending: dict[str, tuple[float, str, list, str]] = {}   # source -> (t, obj, times, said)
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    # ---------------------------------------------------------------- questions

    def ask(self, text: str, source: str = "voice") -> Answer:
        now = self.clock()
        t0 = time.perf_counter()
        try:
            got = self._care(text, source, now)
        except Exception:
            log.exception("care layer failed on %r; answering without it", text)
            got = None
        if got is not None:
            ans, kind = got
            self._log(text, kind, ans, t0)
            return ans
        try:
            ans = self.conversation.ask(text, source, self.base_ask, self.world, self.events, self.cfg, now)
        except Exception:
            log.exception("conversation memory failed on %r", text)
            ans = self.base_ask(text, source)
        self._learn(text, source, now)
        return self._with_morning(ans, source, now)

    def _care(self, text: str, source: str, now: float) -> Optional[tuple[Answer, str]]:
        spoken_src = source not in TEXT_ONLY
        t = normalize(text)
        ans = self._answer_suggestion(t, source, now)
        if ans is not None:
            return ans, "CARE_CONFIRM"
        ans = self.reminders.acknowledge(text, now) if spoken_src else None    # only the room acknowledges
        if ans is not None:
            return ans, "CARE_ACK"
        ans = self.reminders.handle(text, now)
        if ans is not None:
            return ans, "CARE_REMINDER"
        m = SUMMARY.search(t)
        if m:
            day = m.group("day") or m.group("day2") or m.group("day3") or "today"
            day = "today" if day.startswith("my") or day.startswith("the") else day
            return Answer(spoken_summary(day, self.world, self.events, self.cfg, self.reminders, now)), "CARE_SUMMARY"
        if MORNING.search(t):
            return Answer(morning_report(self.world, self.events, self.cfg, now, self.reminders, self.profile)), \
                "CARE_SUMMARY"
        if spoken_src:
            reply = self.profile.handle(text, now)
            if reply is not None:
                return self._suggest(reply.answer, reply.suggest, source, text, now), "CARE_PROFILE"
            m = GREETING.match(t)
            if m:
                report = None if self.hold else self.morning.deliver(now)   # None if not due or already said
                if report:
                    return Answer(report), "CARE_GREETING"
                g = m.group("g")
                g = "Hello" if g in ("hi", "hello", "hey") else ("Good " + g if g in ("morning", "evening")
                                                                else g.capitalize())
                name = self.profile.preferred_name()
                return Answer(f"{g}, {name}." if name else f"{g}."), "CARE_GREETING"
        return None

    def _with_morning(self, ans: Answer, source: str, now: float) -> Answer:
        if source in TEXT_ONLY or self.hold:
            return ans
        try:
            text = self.morning.deliver(now)
        except Exception:
            log.exception("morning report failed")
            text = None
        return replace(ans, text=f"{text} {ans.text}") if text else ans

    def _learn(self, text: str, source: str, now: float) -> None:
        """Free-form personal statements go to the model in the background (online only), never delaying
        the answer; what validates is stored."""
        if source in TEXT_ONLY or self.profile.extractor is None or not STATEMENT.match(normalize(text)):
            return
        try:
            online = bool(self.online())
        except Exception:
            online = False
        if online:
            threading.Thread(target=self.profile.learn, args=(text, now, True), name="care-learn",
                             daemon=True).start()

    def _log(self, text: str, kind: str, ans: Answer, t0: float) -> None:
        try:
            self.events.log_question(text, kind, ans.point_at, ans.text, bool(self.online()),
                                     int((time.perf_counter() - t0) * 1000))
        except Exception:
            pass                    # logging must never cost an answer

    # ---------------------------------------------------------------- suggested reminders

    def _suggest(self, ans: Answer, sug: Optional[Suggestion], source: str, said: str, now: float) -> Answer:
        """'I take my pills at 8 and 8' -> offer a daily condition reminder for the object it names."""
        if sug is None or not sug.times:
            return ans
        obj = resolve_obj(sug.obj_phrase, self.cfg, self.world)
        if obj is None:
            return ans
        have = {(r.at_h, r.at_m) for r in self.reminders.store.active()
                if r.kind == "condition" and r.obj == obj and r.daily}
        times = sorted(set(sug.times) - have)
        if not times:
            return ans
        self._pending[source] = (now, obj, times, said)
        ts = " and ".join(say_time(*x) for x in times)
        return replace(ans, text=f"{ans.text} Should I remind you at {ts} every day if you haven't picked up "
                                f"the {spoken(self.world, self.cfg, obj)}?")

    def _answer_suggestion(self, t: str, source: str, now: float) -> Optional[Answer]:
        p = self._pending.get(source)
        if p is None:
            return None
        if now - p[0] > SUGGEST_TTL_S:
            self._pending.pop(source, None)
            return None
        if NO.match(t) or t in ("no thanks", "no thank you"):
            self._pending.pop(source, None)
            return Answer("Okay, I won't.")
        if not YES.match(t):
            return None
        self._pending.pop(source, None)
        _, obj, times, said = p
        prev = None
        for h, m in times:
            since = None
            if prev is not None:                   # a later check only counts pickups after the midpoint
                mid = ((prev[0] * 60 + prev[1]) + (h * 60 + m)) // 2
                since = (mid // 60, mid % 60)
            self.reminders.store.add("condition", daily_when(h, m, now), obj=obj, since=since,
                                     said=f"suggested: {said}", now=now)
            prev = (h, m)
        ts = " and ".join(say_time(*x) for x in times)
        return Answer(f"Okay. I'll check your {spoken(self.world, self.cfg, obj)} every day at {ts}.")

    # ---------------------------------------------------------------- the scheduler

    def tick(self, now: Optional[float] = None) -> list[Notice]:
        """Fire due reminders and, on the first activity of the morning, the morning report. Notices that
        may be spoken go to on_notice (never inside quiet hours or past the hourly cap)."""
        now = self.clock() if now is None else now
        out = []
        for n, ans in self.reminders.tick(now):
            out.append(n)
            if ans is not None:
                self._say(ans)
        if not self.hold and self.c.get("morning_on_activity", True) and self.morning.due(now) and self.reminders.may_speak(now) \
                and self.morning.activity_since_morning(now):
            text = self.morning.deliver(now)          # None if a question delivered it since due()
            n = self.reminders.store.add_notice(now, "morning", text, None, None, True) if text else None
            if n is not None:
                out.append(n)
                self._say(Answer(text))
        return out

    def _say(self, ans: Answer) -> None:
        if self.hold:
            log.info("notice held (demo.hold_notices): %s", ans.text)
            return
        if self.on_notice is None:
            log.info("notice (no speaker): %s", ans.text)
            return
        try:
            self.on_notice(ans)
        except Exception:
            log.exception("speaking a notice failed")

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                self.tick()
            except Exception:
                log.exception("care tick failed")
            self._stop.wait(self.tick_s)

    def start(self) -> "Care":
        if self._thread is None or not self._thread.is_alive():
            self._stop.clear()
            self._thread = threading.Thread(target=self._loop, name="care", daemon=True)
            self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)

    # ---------------------------------------------------------------- for the dashboard and other channels

    def notices_json(self, now: Optional[float] = None) -> list[dict]:
        return self.reminders.notices_json(self.clock() if now is None else now)

    def ack(self, notice_id: int) -> bool:
        return self.reminders.store.ack(notice_id, self.clock())

    def report(self, day=None, fmt: str = "markdown"):
        """The caregiver summary for day ('YYYY-MM-DD', 'today', 'yesterday'): Markdown, text or a dict (json)."""
        now = self.clock()
        if fmt == "json":
            return summary_data(day, self.world, self.events, self.cfg, self.reminders, now)
        return caregiver_summary(day, self.world, self.events, self.cfg, self.reminders, fmt, now)


def attach_care(room, cfg: dict) -> Care:
    """Put the care layer in front of room's router and speak its notices through room.respond. The room
    starts it (care.start()) in run() and passes it to the dashboard."""
    care = Care(cfg, room.world, room.events, room.base_ask)
    room.base_ask = care.ask
    room.care = care
    care.on_notice = room.respond
    return care
