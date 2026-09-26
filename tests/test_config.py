from pathlib import Path

import pytest

from core.config import Config

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


def test_from_dict_ignores_other_modules_keys():
    c = Config.from_dict({'objects': {'keys': 'target'}, 'llm': {'model': 'x'}, 'present_k_of_n': [4, 8]})
    assert c.names() == ['keys'] and (c.present_k, c.present_n) == (4, 8)
