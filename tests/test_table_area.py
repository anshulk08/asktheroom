"""core/table_area.py: the operator-set tabletop outline (table_area: in config.yaml, or table_area.json
written by python -m core.table --outline), its edge band, and its tie to the calibration it was set with."""
import json
import logging

import numpy as np
import pytest

import core.config
from core.config import Config, load_config
from core.table import Table
from core.table import main as table_main
from core.table_area import TableArea, apply_saved_area, area_path, set_outline

CFG = load_config()
# px -> cm: 10 px per cm, image (50, 20) px is table (0, 0)
H = np.array([[0.1, 0.0, -5.0], [0.0, 0.1, -2.0], [0.0, 0.0, 1.0]])
RECT = [[0, 0], [100, 0], [100, 60], [0, 60]]


def conf(tmp_path, **area):
    """A config whose calibration lives in tmp_path."""
    return dict(CFG, paths=dict(CFG['paths'], table_cal=str(tmp_path / 'table_cal.json')),
                table_tag=dict(CFG.get('table_tag') or {}, enabled=False),
                table_area=dict(CFG.get('table_area') or {}, **area))


def rig(tmp_path, h=H, **area):
    """conf() with that calibration saved."""
    save_cal(tmp_path, h)
    return conf(tmp_path, **area)


def save_cal(tmp_path, h):
    (tmp_path / 'table_cal.json').write_text(json.dumps(
        {'H': np.asarray(h).tolist(), 'markers_px': {}, 'size_cm': [128, 72], 't': 0}))


# ----- the outline itself ---------------------------------------------------------------------------

def test_the_default_config_has_no_outline_so_the_whole_view_is_the_table():
    area = TableArea.from_dict(CFG.get('table_area'))
    assert not area.defined
    assert area.contains((-500, 900)) and area.interior((-500, 900))
    assert Config.from_dict(CFG).table_area == CFG['table_area']


def test_the_interior_is_the_outline_minus_its_edge_band():
    area = TableArea.from_dict({'polygon_cm': RECT, 'edge_cm': 3})
    assert area.defined
    assert area.interior((50, 30)) and area.contains((50, 30))
    assert area.contains((1.5, 30)) and not area.interior((1.5, 30))        # in the edge band
    assert not area.contains((-1, 30)) and not area.interior((-1, 30))      # off the table
    assert not area.interior((98, 58))                                      # near a corner


def test_a_concave_outline_is_followed_exactly():
    ell = [[0, 0], [100, 0], [100, 20], [30, 20], [30, 60], [0, 60]]        # an L-shaped desk
    area = TableArea.from_dict({'polygon_cm': ell, 'edge_cm': 2})
    assert area.interior((15, 40)) and area.interior((70, 10))
    assert not area.contains((70, 40))                                      # the notch


def test_fewer_than_three_points_is_no_outline(caplog):
    with caplog.at_level(logging.WARNING):
        area = TableArea.from_dict({'polygon_cm': [[0, 0], [10, 0]]})
    assert not area.defined and 'table_area' in caplog.text


# ----- saved with the calibration -------------------------------------------------------------------

def test_an_outline_clicked_in_image_px_is_saved_in_table_cm_next_to_the_calibration(tmp_path):
    cfg = rig(tmp_path, edge_cm=4)
    table = Table(cfg)
    got = set_outline(table, cfg, [(50, 20), (1050, 20), (1050, 620), (50, 620)], px=True)
    assert np.allclose(got, RECT)
    path = area_path(cfg)
    assert path == tmp_path / 'table_area.json' and path.exists()
    fresh = conf(tmp_path, edge_cm=4)
    apply_saved_area(fresh)
    assert np.allclose(fresh['table_area']['polygon_cm'], RECT)
    assert fresh['table_area']['edge_cm'] == 4                               # other keys kept
    assert TableArea.from_dict(fresh['table_area']).interior((50, 30))


def test_an_outline_typed_in_table_cm_is_saved_as_is(tmp_path):
    cfg = rig(tmp_path)
    set_outline(Table(cfg), cfg, [(5, 5), (80, 5), (80, 55), (5, 55)])
    apply_saved_area(cfg)
    assert np.allclose(cfg['table_area']['polygon_cm'], [[5, 5], [80, 5], [80, 55], [5, 55]])


def test_a_new_calibration_invalidates_the_saved_outline(tmp_path, caplog):
    """Table cm move with the calibration (one-tag mode puts the origin at the view's corner), so an
    outline saved with another calibration would cut the wrong part of the table."""
    cfg = rig(tmp_path)
    set_outline(Table(cfg), cfg, [(5, 5), (80, 5), (80, 55), (5, 55)])
    save_cal(tmp_path, H @ np.array([[1, 0, 30], [0, 1, 0], [0, 0, 1]]))     # recalibrated: shifted
    fresh = conf(tmp_path)
    with caplog.at_level(logging.WARNING):
        apply_saved_area(fresh)
    assert fresh['table_area']['polygon_cm'] == []
    assert 'table_area.json' in caplog.text and '--outline' in caplog.text


def test_no_saved_outline_keeps_the_configured_one(tmp_path):
    cfg = rig(tmp_path, polygon_cm=RECT)
    apply_saved_area(cfg)
    assert cfg['table_area']['polygon_cm'] == RECT


def test_an_uncalibrated_table_cannot_take_an_outline(tmp_path):
    cfg = rig(tmp_path)
    (tmp_path / 'table_cal.json').unlink()
    with pytest.raises(RuntimeError):
        set_outline(Table(cfg), cfg, [(5, 5), (80, 5), (80, 55)])


# ----- python -m core.table --outline ------------------------------------------------------------

@pytest.fixture
def cli(tmp_path, monkeypatch):
    cfg = rig(tmp_path)
    monkeypatch.setattr(core.config, 'load_config', lambda *a, **k: cfg)
    return cfg


def test_cli_without_points_prints_how_to_set_the_outline(cli, capsys, tmp_path):
    assert table_main(['--outline']) == 0
    out = capsys.readouterr().out
    assert '--outline-px' in out and 'frame.jpg' in out
    assert not (tmp_path / 'table_area.json').exists()


def test_cli_saves_corners_in_table_cm(cli, capsys, tmp_path):
    assert table_main(['--outline', '0,0', '100,0', '100,60', '0,60']) == 0
    saved = json.loads((tmp_path / 'table_area.json').read_text())
    assert np.allclose(saved['polygon_cm'], RECT)
    assert 'table_area.json' in capsys.readouterr().out


def test_cli_converts_corners_clicked_in_image_px(cli, tmp_path):
    assert table_main(['--outline-px', '50,20', '1050,20', '1050,620', '50,620']) == 0
    saved = json.loads((tmp_path / 'table_area.json').read_text())
    assert np.allclose(saved['polygon_cm'], RECT)
    assert np.allclose(saved['polygon_px'], [[50, 20], [1050, 20], [1050, 620], [50, 620]])


def test_cli_refuses_malformed_or_too_few_points(cli, capsys, tmp_path):
    assert table_main(['--outline', '0,0', '100']) == 2
    assert table_main(['--outline', '0,0', '100,0']) == 2
    assert not (tmp_path / 'table_area.json').exists()
