"""Room handoff scoreboard (server/scoreboard.py, GET /scoreboard, POST /scoreboard/trials): counts come
only from scripts/room_trials.py results files, per object and zone, for one local day."""
import datetime as dt
import json
import os
import time

import pytest
from fastapi.testclient import TestClient

from core.config import load_config
from core.events import EventLog
from core.fakeworld import demo_world
from server import scoreboard
from server.app import create_app

NOW = dt.datetime(2026, 9, 26, 21, 30).timestamp()


def run(i, zone, result="pass", handoff_s=8.0, ret="pass"):
    """One record exactly as scripts/room_trials.py writes it."""
    r = {"run": i, "zone": zone, "on_table_s": 3.1}
    if result == "no table sighting":
        return dict(r, on_table_s=None, result=result)
    return dict(r, handoff_s=handoff_s, answer="Your remote, I think, is on the couch.", result=result,
                return_answer="Your remote is on the table.", return_result=ret)


REMOTE = [run(1, "couch", handoff_s=9.0), run(2, "side_table", handoff_s=24.0), run(3, "couch", handoff_s=8.0),
          run(4, "counter", handoff_s=20.0), run(5, "couch", handoff_s=10.0)]


# ----- summary -----------------------------------------------------------------------------------

def test_an_uploaded_session_counts_handoffs_returns_and_the_median_time(tmp_path):
    scoreboard.save_upload(tmp_path, "remote", REMOTE, now=NOW)
    sb = scoreboard.summarize(scoreboard.load_runs(tmp_path), now=NOW)
    assert sb["date"] == "2026-09-26" and sb["runs"] == 5
    assert sb["handoffs"] == {"passed": 5, "total": 5} and sb["returns"] == {"passed": 5, "total": 5}
    assert sb["median_s"] == 10.0 and sb["skipped"] == 0
    assert sb["by_zone"]["couch"]["handoffs"] == {"passed": 3, "total": 3}
    assert sb["by_zone"]["couch"]["median_s"] == 9.0
    assert sb["by_zone"]["side_table"]["median_s"] == 24.0
    assert list(sb["by_object"]) == ["remote"]


def test_fails_and_skipped_runs_are_counted_apart_and_only_passes_are_timed(tmp_path):
    recs = [run(1, "couch", handoff_s=7.0), run(2, "counter", result="fail", handoff_s=None, ret="fail"),
            run(3, "couch", result="no table sighting")]
    scoreboard.save_upload(tmp_path, "wallet", recs, now=NOW)
    sb = scoreboard.summarize(scoreboard.load_runs(tmp_path), now=NOW)
    assert sb["handoffs"] == {"passed": 1, "total": 2}           # the skipped run never reached the handoff
    assert sb["returns"] == {"passed": 1, "total": 2}
    assert sb["skipped"] == 1 and sb["median_s"] == 7.0
    assert sb["by_zone"]["counter"]["median_s"] is None


def test_objects_are_kept_apart_and_only_the_asked_day_counts(tmp_path):
    scoreboard.save_upload(tmp_path, "remote", REMOTE, now=NOW)
    scoreboard.save_upload(tmp_path, "pill bottle", [run(1, "side_table", result="fail", ret="fail")], now=NOW)
    scoreboard.save_upload(tmp_path, "wallet", [run(1, "couch")], now=NOW - 86400)     # yesterday
    runs = scoreboard.load_runs(tmp_path)
    today = scoreboard.summarize(runs, now=NOW)
    assert sorted(today["by_object"]) == ["pill bottle", "remote"]
    assert today["handoffs"] == {"passed": 5, "total": 6}
    assert today["by_object"]["pill bottle"]["handoffs"] == {"passed": 0, "total": 1}
    yesterday = scoreboard.summarize(runs, "yesterday", now=NOW)
    assert list(yesterday["by_object"]) == ["wallet"] and yesterday["handoffs"] == {"passed": 1, "total": 1}
    assert scoreboard.summarize(runs, "all", now=NOW)["runs"] == 7
    assert scoreboard.summarize(runs, "2026-09-25", now=NOW)["runs"] == 1


def test_no_results_means_no_numbers(tmp_path):
    sb = scoreboard.summarize(scoreboard.load_runs(tmp_path / "missing"), now=NOW)
    assert sb["runs"] == 0 and sb["handoffs"] == {"passed": 0, "total": 0} and sb["median_s"] is None
    assert sb["by_object"] == {} and sb["last_t"] is None


def test_a_driver_file_copied_in_by_hand_takes_its_object_from_the_name_and_its_day_from_mtime(tmp_path):
    p = tmp_path / "glasses_2130.json"
    p.write_text(json.dumps([run(1, "couch"), run(2, "couch", result="fail", ret="fail")]))
    os.utime(p, (NOW, NOW))
    sb = scoreboard.summarize(scoreboard.load_runs(tmp_path), now=NOW)
    assert list(sb["by_object"]) == ["glasses"] and sb["handoffs"] == {"passed": 1, "total": 2}


def test_foreign_or_broken_files_in_the_folder_are_skipped(tmp_path):
    (tmp_path / "notes.json").write_text('{"hello": 1}')
    (tmp_path / "broken.json").write_text("{not json")
    (tmp_path / "eval.json").write_text(json.dumps([{"frame": 1}]))
    scoreboard.save_upload(tmp_path, "remote", REMOTE[:1], now=NOW)
    assert scoreboard.summarize(scoreboard.load_runs(tmp_path), now=NOW)["runs"] == 1


@pytest.mark.parametrize("obj, records", [
    ("", REMOTE), ("../etc", REMOTE), ("Remote!", REMOTE), ("x" * 40, REMOTE),
    ("remote", []), ("remote", {"run": 1}), ("remote", [{"zone": "couch", "result": "maybe"}]),
    ("remote", [{"result": "pass"}]), ("remote", [run(1, "couch")] * 201),
])
def test_bad_uploads_are_refused(tmp_path, obj, records):
    with pytest.raises(ValueError):
        scoreboard.save_upload(tmp_path, obj, records, now=NOW)
    assert not list(tmp_path.glob("*.json"))


def test_bad_dates_are_refused(tmp_path):
    with pytest.raises(ValueError):
        scoreboard.summarize([], "last tuesday", now=NOW)


def test_two_uploads_in_one_second_are_both_kept(tmp_path):
    a = scoreboard.save_upload(tmp_path, "remote", REMOTE[:1], now=NOW)
    b = scoreboard.save_upload(tmp_path, "remote", REMOTE[:2], now=NOW)
    assert a != b and scoreboard.summarize(scoreboard.load_runs(tmp_path), now=NOW)["runs"] == 3


# ----- endpoints ---------------------------------------------------------------------------------

@pytest.fixture
def client(tmp_path):
    cfg = load_config()
    cfg["scoreboard"] = {"trials_dir": str(tmp_path / "room")}
    events = EventLog(str(tmp_path / "e.db"), str(tmp_path / "snaps"))
    app = create_app(cfg, demo_world(events), events)
    with TestClient(app) as c:
        yield c, tmp_path / "room"


def test_the_scoreboard_is_empty_until_trials_are_uploaded(client):
    c, _ = client
    r = c.get("/scoreboard")
    assert r.status_code == 200 and r.json()["runs"] == 0


def test_uploading_a_room_trials_file_puts_it_on_the_scoreboard(client):
    c, folder = client
    r = c.post("/scoreboard/trials?object=remote", content=json.dumps(REMOTE),
               headers={"content-type": "application/json"})
    assert r.status_code == 200 and r.json()["runs"] == 5
    assert r.json()["today"]["handoffs"] == {"passed": 5, "total": 5}
    saved = json.loads((folder / r.json()["saved"]).read_text())
    assert saved["object"] == "remote" and saved["records"] == REMOTE and saved["uploaded"] <= time.time()
    sb = c.get("/scoreboard").json()
    assert sb["handoffs"] == {"passed": 5, "total": 5} and sb["median_s"] == 10.0
    assert c.get("/scoreboard?date=all").json()["runs"] == 5


def test_bad_uploads_and_dates_get_400_and_store_nothing(client):
    c, folder = client
    assert c.post("/scoreboard/trials?object=remote", content=b"{not json").status_code == 400
    assert c.post("/scoreboard/trials?object=../x", content=json.dumps(REMOTE)).status_code == 400
    assert c.post("/scoreboard/trials?object=remote", content=json.dumps([{"a": 1}])).status_code == 400
    assert c.post("/scoreboard/trials?object=remote", content=b" " * (300 * 1024)).status_code == 413
    assert not folder.exists() or not list(folder.glob("*.json"))
    assert c.get("/scoreboard?date=soon").status_code == 400


def test_the_dashboard_has_the_scoreboard_widget(client):
    c, _ = client
    html = c.get("/").text
    assert 'id="st-score"' in html and 'id="score-rows"' in html
    assert "/scoreboard" in c.get("/static/app.js").text
