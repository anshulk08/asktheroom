"""scripts/voice_trials.py with a fake rig: a fake clock, a fake speaker and a scripted /state."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import voice_trials as V  # noqa: E402


class FakeRig:
    """The rig's clock runs `skew` s ahead of ours; it answers `delay` s after each question ends."""

    def __init__(self, clock, answers, skew=100.0, delay=1.5):
        self.clock, self.answers, self.skew, self.delay = clock, answers, skew, delay
        self.log, self.pending = [], []

    def state(self):
        now = self.clock["t"]
        for due, a in list(self.pending):
            if now >= due:
                self.log.append(dict(a, seq=len(self.log) + 1, t=due + self.skew))
                self.pending.remove((due, a))
        return {"server_t": now + self.skew, "answers": list(self.log[-10:])}

    def speak(self, text):
        self.clock["t"] += 1.0                             # saying it takes a second
        if self.answers:
            self.pending.append((self.clock["t"] + self.delay, self.answers.pop(0)))


def rig_run(answers, n, delay=1.5, limit=4.0):
    clock = {"t": 1000.0}
    rig = FakeRig(clock, answers, delay=delay)

    def sleep(s):
        clock["t"] += s

    lines = []
    rows = V.run("where's my wallet", "counter", n, limit, 8.0, rig.speak, rig.state,
                 clock=lambda: clock["t"], sleep=sleep, out=lines.append)
    return rows, lines


def voice(text, q="where's my wallet"):
    return {"src": "voice", "q": q, "text": text}


def test_five_quick_right_answers_pass():
    rows, lines = rig_run([voice("Your wallet, I think, is on the kitchen counter.")] * 5, 5)
    assert all(r["ok"] for r in rows) and lines[-1] == "5/5 passed"
    assert all(abs(r["latency_s"] - 1.5) < 0.15 for r in rows)          # the clock skew is taken out
    assert lines[0].startswith("rig clock +100.00 s")


def test_slow_wrong_or_missing_answers_fail():
    rows, _ = rig_run([voice("Your wallet, I think, is on the kitchen counter.")], 1, delay=5.0)
    assert not rows[0]["ok"] and rows[0]["latency_s"] > 4.0
    rows, _ = rig_run([voice("Your wallet is on the couch.")], 1)
    assert not rows[0]["ok"]
    rows, _ = rig_run([], 1)
    assert rows[0] == {"q": "where's my wallet", "heard": None, "text": None, "latency_s": None, "ok": False}


def test_other_sources_are_not_the_spoken_answer():
    rows, _ = rig_run([{"src": "phone", "q": "x", "text": "on the counter"}], 1)
    assert not rows[0]["ok"] and rows[0]["heard"] is None


def test_say_command_prefers_say(monkeypatch):
    monkeypatch.setattr(V.shutil, "which", lambda c: "/usr/bin/say" if c == "say" else None)
    assert V.say_cmd("hi", "Samantha") == ["say", "-v", "Samantha", "hi"]
    monkeypatch.setattr(V.shutil, "which", lambda c: "/usr/bin/espeak-ng" if c == "espeak-ng" else None)
    assert V.say_cmd("hi") == ["espeak-ng", "hi"]
