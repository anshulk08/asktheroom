"""A guided clip carries the tabletop outline set with its calibration, and a replay restores it, so the
edge rules (no births at the outline's edge) run in replays as on the rig."""
import json
from pathlib import Path
from types import SimpleNamespace

from core import table_area
from eval import raw_record, score_clip

H = [[0.1, 0.0, -5.0], [0.0, 0.1, -3.0], [0.0, 0.0, 1.0]]
POLY = [[5.0, 5.0], [95.0, 5.0], [95.0, 55.0], [5.0, 55.0]]


def _saved(dirpath: Path) -> dict:
    (dirpath / "table_cal.json").write_text(json.dumps({"H": H, "size_cm": [100, 60]}))
    area = {"polygon_cm": POLY, "cal_hash": table_area.cal_hash(H)}
    (dirpath / "table_area.json").write_text(json.dumps(area))
    return {"paths": {"table_cal": str(dirpath / "table_cal.json")}}


def test_the_recorded_meta_holds_the_outline_next_to_the_calibration(tmp_path):
    meta = raw_record.meta_for(_saved(tmp_path), "/dev/video0", {}, "abc")
    assert meta["table_cal"]["H"] == H
    assert meta["table_area"]["polygon_cm"] == POLY


def test_no_outline_saved_records_none(tmp_path):
    cfg = _saved(tmp_path)
    (tmp_path / "table_area.json").unlink()
    assert raw_record.meta_for(cfg, "/dev/video0", {}, "abc")["table_area"] is None


def test_the_replay_restores_the_outline_and_it_loads_against_the_recorded_calibration(tmp_path):
    rec = tmp_path / "rec"
    rec.mkdir()
    meta = raw_record.meta_for(_saved(rec), "/dev/video0", {}, "abc")
    work = tmp_path / "work"
    work.mkdir()
    cfg = score_clip.prepare_config(SimpleNamespace(meta=meta), {"objects": {}}, str(work))
    assert table_area.load_saved(table_area.area_path(cfg), table_area._saved_h(cfg)) == \
        [tuple(p) for p in POLY]
    assert table_area.apply_saved_area(cfg)["table_area"]["polygon_cm"] == POLY


def test_an_old_clip_without_an_outline_replays_without_one(tmp_path):
    meta = {"table_cal": {"H": H}}
    cfg = score_clip.prepare_config(SimpleNamespace(meta=meta), {"objects": {}}, str(tmp_path))
    assert not table_area.area_path(cfg).exists()
