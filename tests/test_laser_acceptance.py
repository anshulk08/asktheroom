"""scripts/laser_acceptance.py: the scoring rules, the /ask + /state flow against a fake app, and --sim."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import laser_acceptance as L  # noqa: E402

BOX = [100.0, 200.0, 180.0, 240.0]          # 80 px wide


def row(**kw):
    r = {"object": "keys", "on_target": True, "err_px": None, "err_cm": None, "box": BOX, "target": None,
         "width_cm": 8.0, "tape_cm": None}
    r.update(kw)
    return r


# ---------------------------------------------------------------- scoring

def test_est_cm_scales_px_by_known_width_over_box_width():
    assert L.est_cm(20, BOX, 8.0) == pytest.approx(2.0)          # 10 px per cm
    assert L.est_cm(20, None, 8.0) is None and L.est_cm(None, BOX, 8.0) is None
    assert L.est_cm(20, BOX, None) is None and L.est_cm(float("inf"), BOX, 8.0) is None
    assert L.est_cm(5, [10, 0, 10, 5], 8.0) is None              # degenerate box


def test_tape_wins_over_the_estimate():
    assert L.score(row(err_px=5, tape_cm=4.9))["passed"] is True
    r = L.score(row(err_px=5, tape_cm=6.0))                      # est 0.5 cm, but the tape says 6
    assert r["passed"] is False and r["why"] == "tape 6.0 cm"


def test_estimate_used_without_tape_and_tol_is_inclusive():
    assert L.score(row(err_px=40), tol_cm=4.0)["passed"] is True        # 4.0 cm
    r = L.score(row(err_px=41), tol_cm=4.0)
    assert r["passed"] is False and r["est_cm"] == pytest.approx(4.1)


def test_reported_table_cm_is_the_estimate_when_there_is_no_box():
    r = L.score(row(box=None, err_cm=0.8))
    assert r["passed"] is True and r["est_cm"] == 0.8


def test_missing_box_and_no_error_fails_as_no_box():
    r = L.score(row(box=None))
    assert r["passed"] is False and r["why"] == "no box"


def test_box_fallback_needs_the_error_circle_inside_the_box():
    # no known width: fall back to the box; target 10 px from the left edge
    assert L.score(row(width_cm=None, err_px=8, target=[110, 220]))["passed"] is True
    assert L.score(row(width_cm=None, err_px=12, target=[110, 220]))["passed"] is False
    assert L.score(row(width_cm=None, err_px=15))["passed"] is True       # box centre: 20 px margin
    assert L.score(row(width_cm=None, dot=[179, 239]))["passed"] is True  # dot px known: plain containment
    assert L.score(row(width_cm=None))["passed"] is False                  # nothing says where the dot is


def test_not_on_target_always_fails():
    r = L.score(row(on_target=False, err_px=1, tape_cm=0.5))
    assert r["passed"] is False and r["why"] == "laser not on target"


def test_need_threshold():
    rows = [{"passed": p} for p in (True, True, True, True, False)]
    assert L.verdict(rows, 4) == {"passed": 4, "of": 5, "need": 4, "ok": True}
    assert L.verdict(rows, 5)["ok"] is False


def test_parse_widths_overrides_defaults():
    w = L.parse_widths(["keys=9", "mug=10"])
    assert w["keys"] == 9 and w["mug"] == 10 and w["pill_bottle"] == 4.5


# ---------------------------------------------------------------- HTTP flow (fake app)

class FakeApp:
    """Answers /ask and /state like server/app.py; the laser comes on on the second /state poll."""

    def __init__(self, places):
        self.places, self.asked, self.polls, self.laser = places, [], 0, {"on": False}

    def __call__(self, method, url, body=None, timeout=30.0):
        if url.endswith("/ask"):
            assert method == "POST"
            self.asked.append(body["text"])
            self.polls, self.cur = 0, body["text"].removeprefix("where is my ").rstrip("?").replace(" ", "_")
            self.laser = {"on": False}
            p = self.places.get(self.cur)
            if p is None:
                return {"text": "I haven't seen it.", "point_at": None, "action": None, "latency_ms": 5}
            return {"text": f"The {self.cur} is here.", "point_at": self.cur, "action": p.get("action", "point"),
                    "latency_ms": 5}
        assert url.endswith("/state") and method == "GET"
        self.polls += 1
        p = self.places.get(getattr(self, "cur", None)) or {}
        if self.polls >= 2 and "laser" in p:
            self.laser = p["laser"]
        room = {self.cur: {"zone": "couch", "box_px": p["box"]}} if p.get("box") else {}
        return {"state": {"laser": dict(self.laser), "room": room, "entities": []}}


def args(**kw):
    d = dict(url="http://rig:8080/", objects=["keys", "wallet", "remote", "glasses", "pill_bottle"],
             known_width_cm=[], tol_cm=5.0, need=4, no_prompt=False, wait_s=5.0, clear_s=5.0, poll_s=0.1)
    d.update(kw)
    return type("A", (), d)()


def fake_clock():
    t = [0.0]
    return (lambda: t[0]), (lambda s: t.__setitem__(0, t[0] + s))


def test_http_flow_scores_each_object():
    app = FakeApp({
        "keys": {"box": BOX, "laser": {"on": True, "err_px": 10.0, "err_cm": None}},            # est 1 cm
        "wallet": {"box": [0, 0, 110, 50], "laser": {"on": True, "err_px": 60.0}},               # est 6 cm
        "remote": {"action": "room:140,220,100,200,180,240", "laser": {"on": True, "err_px": 5.0}},
        "glasses": {"box": BOX},                                                                  # aim refused
        # pill_bottle: not known, no aim
    })
    clock, sleep = fake_clock()
    tapes = iter(["3", "", "x"])                     # keys: tape 3 cm; wallet: skipped; remote: a typo
    rows = L.run_live(args(), http=app, ask_tape=lambda _: next(tapes), sleep=sleep, clock=clock)
    assert app.asked == ["where is my keys?", "where is my wallet?", "where is my remote?",
                         "where is my glasses?", "where is my pill bottle?"]
    by = {r["object"]: r for r in rows}
    assert by["keys"]["tape_cm"] == 3.0 and by["keys"]["passed"] is True
    assert by["wallet"]["est_cm"] == pytest.approx(6.0) and by["wallet"]["passed"] is False
    assert by["remote"]["box"] == BOX and by["remote"]["target"] == [140, 220]    # from the room action
    assert by["remote"]["tape_cm"] is None and by["remote"]["est_cm"] == pytest.approx(18 * 5 / 80) and by["remote"]["passed"] is True
    assert by["glasses"]["laser_on"] is False and by["glasses"]["why"] == "laser not on target"
    assert by["pill_bottle"]["answer"] == "I haven't seen it." and by["pill_bottle"]["passed"] is False
    assert L.verdict(rows, 4)["ok"] is False and L.verdict(rows, 2)["ok"] is True


def test_main_writes_json_and_exit_code(monkeypatch, tmp_path, capsys):
    app = FakeApp({o: {"box": BOX, "laser": {"on": True, "err_px": 4.0}} for o in ("keys", "wallet", "remote")})
    monkeypatch.setattr(L, "http_json", app)
    out = tmp_path / "out.json"
    code = L.main(["--url", "http://rig:8080", "--objects", "keys", "wallet", "remote", "--no-prompt",
                   "--need", "3", "--clear-s", "0", "--poll-s", "0", "--json", str(out)])
    assert code == 0
    d = json.loads(out.read_text())
    assert d["verdict"] == {"passed": 3, "of": 3, "need": 3, "ok": True}
    assert [r["tape_cm"] for r in d["rows"]] == [None, None, None]
    assert "PASS: 3/3" in capsys.readouterr().out


def test_waits_for_the_previous_dot_to_go_off(monkeypatch):
    app = FakeApp({"keys": {"box": BOX, "laser": {"on": True, "err_px": 4.0}}})
    app.laser = {"on": True}                         # the last answer's dot is still on
    clock, sleep = fake_clock()
    slept = []
    L.run_live(args(objects=["keys"], no_prompt=True), http=app, sleep=lambda s: (slept.append(s), sleep(s)),
               clock=clock)
    assert sum(slept) >= 5.0                         # clear_s spent before asking


def test_another_objects_dot_is_not_counted():
    app = FakeApp({"keys": {"box": BOX, "laser": {"on": True, "target": "wallet", "err_px": 1.0}},
                   "wallet": {"box": BOX, "laser": {"on": True, "target": "wallet", "err_px": 1.0}}})
    clock, sleep = fake_clock()
    rows = L.run_live(args(objects=["keys", "wallet"], no_prompt=True, clear_s=0), http=app, sleep=sleep,
                      clock=clock)
    assert [r["laser_on"] for r in rows] == [False, True] and rows[0]["why"] == "laser not on target"


# ---------------------------------------------------------------- --sim

def test_sim_run_scores_five_scene_objects(tmp_path, capsys):
    out = tmp_path / "sim.json"
    L.main(["--sim", "--grid", "10", "8", "--no-prompt", "--json", str(out)])
    d = json.loads(out.read_text())
    assert len(d["rows"]) == 5 and d["sim"] is True
    for r in d["rows"]:
        assert r["box"] is not None and r["width_cm"] > 0
        assert r["est_cm"] is not None and r["tape_cm"] is not None
    assert sum(r["on_target"] for r in d["rows"]) >= 4
    assert "objects (need 4" in capsys.readouterr().out
