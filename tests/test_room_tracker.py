"""RoomTracker (core/room.py, spec 0009 M0): per-zone tracks, confirmation, valid/invalid visits, absence."""
from __future__ import annotations

from core.room import RoomTracker
from core.room_types import RoomConfig, RoomObservation

KEYS = (100, 100, 140, 130)         # a 40x30 box in full-frame px
WALLET = (300, 100, 350, 140)


def obs(cls: str, box, t: float, zone: str = "shelf", conf: float = 0.9) -> RoomObservation:
    return RoomObservation(zone=zone, cls=cls, conf=conf, box_px=box, t=t, wall=1000.0 + t, frame_idx=int(t))


class Run:
    """Drives one tracker with a visit clock."""

    def __init__(self, **cfg):
        self.tr = RoomTracker(RoomConfig(**cfg))
        self.t = 0.0

    def visit(self, *observations, zone: str = "shelf", blockers=(), changes=(), lum=None, crop=None):
        self.t += 1.0
        o = [obs(c, b, self.t, zone) for c, b in observations]
        return self.tr.visit(zone, "the shelf", o, list(blockers), list(changes), self.t, 1000.0 + self.t,
                             int(self.t), lum=lum, crop=crop)


def test_confirm_after_two_visits():
    r = Run()
    v1 = r.visit(("keys", KEYS))
    assert v1.confirmed == [] and v1.missed == [] and v1.dropped == []
    [tr] = r.tr.tracks("shelf")
    assert tr.hits == 1 and not tr.confirmed and tr.first_seen == 1.0 and tr.first_wall == 1001.0
    v2 = r.visit(("keys", (102, 101, 142, 131)))
    assert [x.tid for x in v2.confirmed] == [tr.tid]
    assert tr.confirmed and tr.hits == 2 and tr.misses == 0
    assert tr.box_px == (102, 101, 142, 131) and tr.last_seen == 2.0 and tr.last_wall == 1002.0
    assert tr.first_seen == 1.0
    assert v2.zone == "shelf" and v2.say == "the shelf" and v2.t == 2.0 and v2.wall == 1002.0 and v2.frame_idx == 2


def test_match_by_centre_distance_when_iou_is_low():
    r = Run()
    r.visit(("keys", KEYS))
    # shrunk box: IoU < 0.3 but the centre is within half a diagonal
    v = r.visit(("keys", (115, 110, 125, 120)))
    assert len(r.tr.tracks()) == 1 and len(v.confirmed) == 1


def test_far_detection_is_a_new_track():
    r = Run()
    r.visit(("keys", KEYS))
    r.visit(("keys", (400, 300, 440, 330)))
    ids = [x.tid for x in r.tr.tracks()]
    assert len(ids) == 1                 # the old unconfirmed track had a valid miss: dropped
    assert ids[0] != "r:1"


def test_unconfirmed_dropped_on_valid_miss():
    r = Run()
    r.visit(("keys", KEYS))
    v = r.visit()
    assert r.tr.tracks() == []
    assert [x.tid for x in v.dropped] == ["r:1"] and v.missed == []


def test_blocked_miss_changes_nothing():
    r = Run()
    r.visit(("keys", KEYS))
    r.visit(("keys", KEYS))
    [tr] = r.tr.tracks()
    v = r.visit(blockers=[(90, 90, 150, 140)])           # a hand over the spot
    assert v.missed == [] and v.dropped == [] and tr.misses == 0 and tr.hits == 2
    r2 = Run()
    r2.visit(("keys", KEYS))
    r2.visit(blockers=[(90, 90, 150, 140)])
    assert len(r2.tr.tracks()) == 1                       # unconfirmed survives a blocked visit too


def test_small_blocker_overlap_is_still_valid():
    r = Run()
    r.visit(("keys", KEYS))
    r.visit(("keys", KEYS))
    [tr] = r.tr.tracks()
    v = r.visit(blockers=[(135, 125, 200, 200)])          # covers well under 30% of the box
    assert [x.tid for x in v.missed] == [tr.tid] and tr.misses == 1


def test_big_change_blob_blocks_small_one_does_not():
    r = Run()
    r.visit(("keys", KEYS))
    r.visit(("keys", KEYS))
    [tr] = r.tr.tracks()
    v = r.visit(changes=[(50, 50, 200, 200)])             # much bigger than the keys, over them
    assert v.missed == [] and tr.misses == 0
    v = r.visit(changes=[(105, 105, 125, 125)])           # smaller than the keys: they were taken
    assert [x.tid for x in v.missed] == [tr.tid] and tr.misses == 1


def test_big_change_blob_elsewhere_does_not_block():
    r = Run()
    r.visit(("keys", KEYS))
    r.visit(("keys", KEYS))
    [tr] = r.tr.tracks()
    v = r.visit(changes=[(500, 500, 700, 700)])
    assert len(v.missed) == 1 and tr.misses == 1


def test_dark_or_bright_box_blocks():
    r = Run()
    r.visit(("keys", KEYS))
    r.visit(("keys", KEYS))
    [tr] = r.tr.tracks()
    assert r.visit(lum=lambda box: 10.0).missed == []
    assert r.visit(lum=lambda box: 250.0).missed == []
    assert tr.misses == 0
    seen = []
    v = r.visit(lum=lambda box: seen.append(box) or 120.0)
    assert len(v.missed) == 1 and seen == [KEYS]


def test_confirmed_track_misses_then_dropped():
    r = Run()
    r.visit(("keys", KEYS))
    r.visit(("keys", KEYS))
    [tr] = r.tr.tracks()
    for n in (1, 2):
        v = r.visit()
        assert [x.tid for x in v.missed] == [tr.tid] and tr.misses == n and v.dropped == []
        assert r.tr.tracks() == [tr]
    v = r.visit()
    assert [x.tid for x in v.missed] == [tr.tid] and tr.misses == 3
    assert [x.tid for x in v.dropped] == [tr.tid]
    assert r.tr.tracks() == []


def test_match_resets_misses():
    r = Run()
    r.visit(("keys", KEYS))
    r.visit(("keys", KEYS))
    [tr] = r.tr.tracks()
    r.visit()
    r.visit()
    v = r.visit(("keys", KEYS))
    assert tr.misses == 0 and tr.hits == 3 and v.confirmed == [tr]


def test_two_classes_do_not_cross_match():
    r = Run()
    r.visit(("keys", KEYS))
    r.visit(("wallet", KEYS))                              # same spot, other class
    tr = r.tr.tracks()
    assert sorted(x.cls for x in tr) == ["keys", "wallet"]  # the wallet on its spot: no valid miss for keys
    r = Run()
    r.visit(("keys", KEYS), ("wallet", WALLET))
    v = r.visit(("keys", KEYS), ("wallet", WALLET))
    assert sorted(x.cls for x in v.confirmed) == ["keys", "wallet"]


def test_greedy_matching_prefers_best_iou():
    r = Run()
    a, b = (100, 100, 140, 140), (130, 100, 170, 140)
    r.visit(("keys", a), ("keys", b))
    t1, t2 = r.tr.tracks()
    r.visit(("keys", (131, 100, 171, 140)), ("keys", (101, 100, 141, 140)))
    assert t1.box_px == (101, 100, 141, 140) and t2.box_px == (131, 100, 171, 140)


def test_ids_increase_across_zones():
    r = Run()
    r.visit(("keys", KEYS))
    r.visit(("keys", KEYS), zone="couch")
    r.visit(("wallet", WALLET))
    ids = [x.tid for x in r.tr.tracks("couch")] + [x.tid for x in r.tr.tracks("shelf")]
    nums = [int(i.split(":")[1]) for i in ids]
    assert ids[0] == "r:2" and nums == sorted(nums) and len(set(nums)) == len(nums)
    assert {x.zone for x in r.tr.tracks("couch")} == {"couch"}


def test_zones_are_independent():
    r = Run()
    r.visit(("keys", KEYS))
    r.visit(("keys", KEYS))
    r.visit(zone="couch")                                  # an empty couch visit is no miss for the shelf
    [tr] = r.tr.tracks("shelf")
    assert tr.misses == 0 and tr.confirmed


def test_role_and_entity_survive_visits():
    r = Run()
    r.visit(("keys", KEYS))
    v = r.visit(("keys", KEYS))
    v.confirmed[0].role, v.confirmed[0].entity = "assoc", "keys"
    v = r.visit(("keys", KEYS))
    assert v.confirmed[0].role == "assoc" and v.confirmed[0].entity == "keys"


def test_crop_passed_through():
    r = Run()
    marker = object()
    assert r.visit(crop=marker).crop is marker


def test_confirm_visits_one_confirms_at_once():
    r = Run(confirm_visits=1)
    v = r.visit(("keys", KEYS))
    assert len(v.confirmed) == 1 and v.confirmed[0].confirmed


def test_another_object_on_the_spot_is_not_a_miss():
    r = Run()
    r.visit(("keys", KEYS))
    r.visit(("keys", KEYS))
    [keys] = r.tr.tracks()
    for _ in range(4):                                     # a wallet put down on top of the keys
        v = r.visit(("wallet", (95, 95, 145, 135)))
        assert v.missed == [] and keys.misses == 0
    assert keys in r.tr.tracks()
