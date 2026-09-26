from pathlib import Path

import pytest

from core.config import Config, load_config

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def cfg():
    return Config.load(ROOT / 'config.yaml')


def test_loads_eight_objects_with_kinds(cfg):
    assert len(cfg.objects) == 8
    assert cfg.kind_of('keys') == 'target'
    assert cfg.kind_of('box') == 'container'
    assert cfg.kind_of('notebook') == 'cover'


def test_spec_start_values(cfg):
    assert (cfg.present_k, cfg.present_n) == (6, 10)
    assert cfg.absent_max == 1
    assert cfg.contact_overlap == pytest.approx(0.30)
    assert cfg.contact_window_s == pytest.approx(1.0)
    assert cfg.cover_overlap == pytest.approx(0.60)
    assert cfg.moved_min_cm == pytest.approx(5)
    assert cfg.held_timeout_s == pytest.approx(30)


def test_synonyms_map_to_canonical_names(cfg):
    assert cfg.synonyms['pills'] == 'pill_bottle'


def test_names_by_kind(cfg):
    assert cfg.names('container') == ['box']
    assert cfg.names('cover') == ['notebook']
    assert len(cfg.names('target')) == 6


def test_typed_view_reads_the_shared_yaml_layout(cfg):
    """config.yaml keeps the dict layout voice/eval/server use; Config maps it for the world."""
    assert cfg.conf_threshold == pytest.approx(0.35)          # from conf_threshold.default
    assert cfg.table_size_cm == (90, 60)                     # from table.size_cm
    assert cfg.frame_size_px == (1280, 720)
    assert 'keys' in next(o for o in cfg.objects if o.name == 'keys').prompts


def test_per_object_conf_thresholds_reach_the_typed_view():
    c = Config.from_dict({'objects': {'keys': 'target', 'wallet': 'target'},
                          'conf_threshold': {'default': 0.35, 'wallet': 0.2}})
    assert c.threshold('wallet') == pytest.approx(0.2)
    assert c.threshold('keys') == pytest.approx(0.35)
    assert Config.from_dict({'objects': {'keys': 'target'}, 'conf_threshold': 0.4}).threshold('keys') == 0.4


def test_from_dict_ignores_other_modules_keys():
    c = Config.from_dict({'objects': {'keys': 'target'}, 'llm': {'model': 'x'}, 'present_k_of_n': [4, 8]})
    assert c.names() == ['keys'] and (c.present_k, c.present_n) == (4, 8)


def _write(d, name, text):
    (d / name).write_text(text)
    return d / name


def test_a_local_file_overrides_config_yaml_key_by_key(tmp_path, monkeypatch):
    monkeypatch.delenv("ASKROOM_NO_LOCAL_CONFIG", raising=False)
    base = _write(tmp_path, "config.yaml", "actuator: fake\nn8n:\n  webhook_url: ''\n  token: t\nlist: [1, 2]\n")
    _write(tmp_path, "config.local.yaml", "actuator: pca9685\nn8n:\n  webhook_url: http://x\nlist: [3]\n")
    cfg = load_config(base)
    assert cfg["actuator"] == "pca9685"
    assert cfg["n8n"] == {"webhook_url": "http://x", "token": "t"}     # the untouched key survives
    assert cfg["list"] == [3]                                           # lists replace, not append


def test_no_local_file_is_just_config_yaml(tmp_path, monkeypatch):
    monkeypatch.delenv("ASKROOM_NO_LOCAL_CONFIG", raising=False)
    base = _write(tmp_path, "config.yaml", "actuator: fake\n")
    assert load_config(base) == {"actuator": "fake"}


def test_the_opt_out_ignores_the_local_file(tmp_path, monkeypatch):
    monkeypatch.setenv("ASKROOM_NO_LOCAL_CONFIG", "1")
    base = _write(tmp_path, "config.yaml", "actuator: fake\n")
    _write(tmp_path, "config.local.yaml", "actuator: pca9685\n")
    assert load_config(base)["actuator"] == "fake"


def test_the_typed_view_sees_the_local_file(tmp_path, monkeypatch):
    monkeypatch.delenv("ASKROOM_NO_LOCAL_CONFIG", raising=False)
    base = _write(tmp_path, "config.yaml", "present_k_of_n: [6, 10]\n")
    _write(tmp_path, "config.local.yaml", "present_k_of_n: [4, 8]\n")
    assert (Config.load(base).present_k, Config.load(base).present_n) == (4, 8)


def test_the_example_local_file_parses_and_names_real_keys():
    import yaml
    ex = yaml.safe_load((ROOT / "config.local.yaml.example").read_text())
    base = load_config(ROOT / "config.yaml")
    assert set(ex) <= set(base)
