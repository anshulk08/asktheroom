"""Load config.yaml into a plain dict. Every module takes this dict as `cfg`; the world model
reads the same file through the typed Config view at the bottom.

A gitignored config.local.yaml next to it (per device: the rig's `actuator: pca9685`, the laptop's
n8n webhook) is merged over it: nested sections merge key by key, anything else replaces.
ASKROOM_NO_LOCAL_CONFIG=1 skips it (the tests set it, so they read the committed file only)."""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field, fields
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent


LOCAL_NAME = "config.local.yaml"


def load_config(path: str | os.PathLike | None = None) -> dict:
    p = Path(path) if path else ROOT / "config.yaml"
    with open(p) as f:
        cfg = yaml.safe_load(f)
    local = p.with_name(LOCAL_NAME)
    if local != p and local.is_file() and not os.environ.get("ASKROOM_NO_LOCAL_CONFIG"):
        with open(local) as f:
            over = yaml.safe_load(f) or {}
        merge_into(cfg, over)
        logging.getLogger("askroom.config").info("%s overrides: %s", local.name, ", ".join(sorted(over)))
    return cfg


def merge_into(base: dict, over: dict) -> dict:
    """Deep-merge `over` into `base` in place: dicts merge recursively, other values replace."""
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            merge_into(base[k], v)
        else:
            base[k] = v
    return base


def display_name(cfg: dict, obj: str) -> str:
    """How an object name is spoken: 'pill_bottle' -> 'pill bottle'."""
    return (cfg.get("display_names") or {}).get(obj, obj.replace("_", " "))


# ---------------------------------------------------------------------------------------------
# Typed view for the world model. Built from the same config.yaml dict everyone else reads, so
# there is one config file; fields not in the yaml keep the defaults below.

@dataclass
class ObjectSpec:
    name: str
    kind: str                   # 'target' | 'container' | 'cover'
    prompts: list[str] = field(default_factory=list)


@dataclass
class Config:
    objects: list[ObjectSpec] = field(default_factory=list)
    table_size_cm: tuple[float, float] = (90.0, 60.0)
    frame_size_px: tuple[int, int] = (1280, 720)

    conf_threshold: float = 0.35
    present_k: int = 6
    present_n: int = 10
    absent_max: int = 1

    contact_overlap: float = 0.30
    contact_window_s: float = 1.0
    cover_overlap: float = 0.60
    cover_moved_window_s: float = 3.0
    cover_moved_min_cm: float = 2.0
    container_dwell_s: float = 0.3
    reappear_wait_s: float = 2.0
    edge_margin: float = 0.05
    held_timeout_s: float = 30.0
    hand_lost_s: float = 0.5
    moved_min_cm: float = 5.0
    settle_s: float = 0.5
    settle_cm: float = 1.5
    max_nesting: int = 3
    lifted_overlap_max: float = 0.2

    conf_held: float = 0.9
    conf_under: float = 0.85
    conf_under_unknown: float = 0.6
    conf_inside: float = 0.85
    lifted_cover_penalty: float = 0.5
    ambiguity_penalty: float = 0.7
    decay_per_min: float = 0.99
    answer_plain: float = 0.7
    answer_hedge: float = 0.5

    bg_frames: int = 30
    bg_change_threshold: float = 25.0
    appearance_match: float = 0.7
    bg_update_every_s: float = 1.0
    # An object on the table unseen (no detection, no patch match) for less than this is kept where it
    # was before LOST_TRACK: an arm the detectors miss, or a detector that drops objects near an arm.
    lost_grace_s: float = 2.0

    synonyms: dict[str, str] = field(default_factory=dict)
    edge_drop_cm: float = 10.0
    floor_zones: list[dict] = field(default_factory=list)

    # Open world (core/things.py): the openworld: section, read there; a square on the table where a
    # new thing is shown to be named ('this is my charger'), or None for 'most recently put down'.
    openworld: dict = field(default_factory=dict)
    teach_zone_cm: tuple[float, float, float, float] | None = None
    # The tabletop outline (core/table_area.py, table_area: section): new things are born only inside it.
    table_area: dict = field(default_factory=dict)
    # Large unnamed things that hold others (core/things.py ContainerConfig, thing_containers: section).
    thing_containers: dict = field(default_factory=dict)
    # Fewer duplicate things (core/things.py IdentityConfig, thing_identity: section).
    thing_identity: dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, raw: dict) -> "Config":
        """Map the shared config.yaml layout onto the world's fields; other modules' keys are ignored."""
        known = {f.name for f in fields(cls)}
        kw = {k: v for k, v in raw.items() if k in known and k not in ("objects", "conf_threshold", "floor_zones")}
        prompts = raw.get("prompts") or {}
        kw["objects"] = [ObjectSpec(n, k, list(prompts.get(n, []))) for n, k in (raw.get("objects") or {}).items()]
        ct = raw.get("conf_threshold")
        if ct is not None:
            kw["conf_threshold"] = ct.get("default", 0.35) if isinstance(ct, dict) else ct
        if "present_k_of_n" in raw:
            kw["present_k"], kw["present_n"] = raw["present_k_of_n"]
        if "absent_k_of_n" in raw:
            kw["absent_max"] = raw["absent_k_of_n"][0]
        size = (raw.get("table") or {}).get("size_cm")
        if size:
            kw["table_size_cm"] = tuple(size)
        if "frame_size_px" in raw:
            kw["frame_size_px"] = tuple(raw["frame_size_px"])
        zones = raw.get("floor_zones") or []
        kw["floor_zones"] = [{"name": n, **z} for n, z in zones.items()] if isinstance(zones, dict) else list(zones)
        return cls(**kw)

    @classmethod
    def load(cls, path: str | os.PathLike | None = None) -> "Config":
        return cls.from_dict(load_config(path))

    def kind_of(self, name: str) -> str:
        for o in self.objects:
            if o.name == name:
                return o.kind
        raise KeyError(name)

    def names(self, kind: str | None = None) -> list[str]:
        return [o.name for o in self.objects if kind is None or o.kind == kind]
