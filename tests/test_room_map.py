import math

import numpy as np
import pytest

from act.room_map import RoomMap, SweepAborted, beam_blocked, people_boxes_hog, sweep
from act.sim import RoomRig


def dist(a, b):
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))


def center(box):
    return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)


@pytest.fixture(scope="module", params=[3.0, 10.0, 25.0], ids=["b3", "b10", "b25"])
def mapped(request):
    """A room rig with pivot offset b, swept once (16x12 grid + refine)."""
    rig = RoomRig(b_cm=request.param, seed=1)
    laser = rig.make_laser()
    rm = sweep(laser, grid=(16, 12), n_pairs=1)
    return rig, laser, rm


# ---------------------------------------------------------------- map

def test_sweep_maps_the_room_and_refines_edges(mapped):
    rig, laser, rm = mapped
    assert len(rm.pulses) > 16 * 12                      # the refine pass added points
    assert rm.n_seen > 100 and rm.size_px == (640, 360)
    assert laser.state["on"] is False and rig.act.laser_on is False
    seen = rm.px[rm.seen]
    surfaces = {rig.surface_at(p) for p in seen}
    assert {"floor", "table", "back_wall"} <= surfaces


def test_map_inverts_to_within_the_servo_slop(mapped):
    rig, _, rm = mapped
    rng = np.random.default_rng(0)
    slop = np.array(rig.geom.k) * rig.backlash_deg / 2     # one-sided approach: servo stops this short
    errs = []
    for i in rng.choice(rm.n_seen, 30, replace=False):
        uv = rm._x[i] + rng.normal(0, 6, 2)
        g = rm.pulses_for_px(uv)
        got = rig.geom.dot_px(*(np.array(g.pulses) - slop))
        if got is not None:
            errs.append(dist(got, uv))
    assert len(errs) >= 25 and np.median(errs) < 2.0


def test_save_load_round_trip_keeps_misses_and_zones(mapped, tmp_path):
    _, _, rm = mapped
    rm.zones = {"shelf": [[380, 60], [490, 60], [490, 80], [380, 80]]}
    p = str(tmp_path / "room_map.json")
    rm.save(p)
    rm2 = RoomMap.load(p)
    assert rm2.n_seen == rm.n_seen and np.allclose(rm2.pulses, rm.pulses)
    assert np.isnan(rm2.px[~rm.seen]).all()
    assert rm2.zone_at((430, 70)) == "shelf" and rm2.zone_at((100, 300)) is None


def synthetic(fold_px=0.0, left_only=False):
    """10x10 pulse grid, px = linear map; columns >= 5 shifted by fold_px (a depth edge)."""
    P, X = [], []
    for t in np.linspace(1300, 1700, 10):
        for p in np.linspace(1300, 1700, 10):
            u, v = 100 + (p - 1300) * 1.0, 50 + (t - 1300) * 0.6
            if p > 1500:
                u += fold_px
            if left_only and p > 1500:
                u = v = math.nan
            P.append((p, t))
            X.append((u, v))
    return RoomMap(P, X, (400 / 9, 400 / 9), (640, 360))


def test_unmapped_gap_is_refused():
    rm = synthetic(left_only=True)
    assert rm.pulses_for_px((150, 150)) is not None
    assert rm.pulses_for_px((480, 150), max_gap_px=40) is None


def test_fold_is_detected_and_the_guess_uses_the_consistent_side():
    rm = synthetic(fold_px=80.0)
    g = rm.pulses_for_px((260, 170))                      # just left of the edge (columns <= 1500)
    assert g.fold
    assert abs(g.pulses[0] - 1460) < 12 and abs(g.J[0, 0] - 1.0) < 0.2
    assert not synthetic().pulses_for_px((260, 170)).fold


# ---------------------------------------------------------------- aiming

def targets(rig):
    """(name, target px, box px or None) on the floor, table, coffee table, back wall and a bottle."""
    out = [("floor", (320.0, 300.0), None), ("back_wall", (200.0, 60.0), None)]
    for name in ("table", "coffee_table", "bottle"):
        box = rig.box_px(name)
        out.append((name, center(box), box if name == "bottle" else None))
    return out


def test_aim_px_converges_on_every_surface(mapped):
    rig, laser, rm = mapped
    for name, t, box in targets(rig):
        r = laser.aim_px(t, box, room_map=rm, tol_px=3.0)
        true = rig.true_dot_px()
        assert r.on_target and r.seen and r.tries <= 8, (name, r)
        assert true is not None and rig.surface_at(true) == name, (name, true)
        assert dist(true, t) < (max(box[2] - box[0], box[3] - box[1]) / 2 if box else 4.5), (name, r, true)
    laser.off()


@pytest.mark.parametrize("scale", [2.0, 0.5])
def test_aim_px_recovers_from_a_wrong_jacobian(mapped, scale, monkeypatch):
    rig, laser, rm = mapped
    real = rm.pulses_for_px

    def skewed(uv, **kw):                                  # Jacobian off by `scale`, guess off by ~3 deg
        g = real(uv, **kw)
        g.J, g.pulses = g.J * scale, (g.pulses[0] + 30, g.pulses[1] - 25)
        return g
    monkeypatch.setattr(rm, "pulses_for_px", skewed)
    monkeypatch.setattr(laser, "max_on_s", 60.0)           # convergence, not the on-time budget (tested apart)
    rig.act.move(1500, 1500)                               # start far away: the feedforward is exact-ish
    t = center(rig.box_px("table"))
    t = (t[0] + 20, t[1] - 5)                              # off the sample grid
    r = laser.aim_px(t, room_map=rm, tol_px=2.0)
    assert r.first_err_px > 10 and r.on_target and 2 <= r.tries <= 8, r
    assert dist(rig.true_dot_px(), t) < 3.5
    laser.off()


def test_blocked_dot_is_not_a_success(mapped):
    rig, laser, rm = mapped
    rig.blocked = True
    try:
        r = laser.aim_px(center(rig.box_px("table")), room_map=rm)
    finally:
        rig.blocked = False
    assert not r.on_target and not r.seen and r.reason == "not_seen" and r.tries == 3
    laser.off()


def test_unmapped_target_is_refused_without_moving():
    rig = RoomRig(seed=2)
    laser = rig.make_laser()
    rm = synthetic(left_only=True)
    n = len(rig.act.calls)
    r = laser.aim_px((600, 300), room_map=rm)
    assert r.reason == "unmapped" and not r.on_target and len(rig.act.calls) == n
    with pytest.raises(RuntimeError, match="room map"):
        laser.aim_px((600, 300))


# ---------------------------------------------------------------- dot detection

def test_averaged_pairs_find_a_dim_dot_that_one_pair_misses():
    rig = RoomRig(dot_gain=0.2, seed=3)
    laser = rig.make_laser()
    hits = {1: 0, 8: 0}
    for pt in [(1500, 1600), (1400, 1650), (1600, 1550), (1450, 1700)]:
        laser.move_to(*pt)
        rig.clock.sleep(0.2)
        truth = rig.true_dot_px()
        for n in hits:
            d = laser.find_dot_px(n)
            if d is not None:
                assert dist(d, truth) < 3                  # never a false dot
                hits[n] += 1
    assert hits[1] <= 1 and hits[8] >= 3, hits


def test_shape_gate_ignores_a_streak():
    from act.laser import _dot_blob
    mask = np.zeros((100, 100), np.uint8)
    w = np.zeros((100, 100), np.float32)
    mask[10, 5:60], w[10, 5:60] = 1, 200                   # a 1-px streak (a moving edge), brighter
    mask[50:54, 50:54], w[50:54, 50:54] = 1, 80            # a dot
    x, y = _dot_blob(mask, w, 2, 3000)
    assert abs(x - 51.5) < 0.1 and abs(y - 51.5) < 0.1


# ---------------------------------------------------------------- sweep safety

def test_sweep_aborts_when_a_person_appears_and_leaves_the_laser_off():
    rig = RoomRig(seed=4)
    laser = rig.make_laser()
    calls = []

    def stop():
        calls.append(1)
        return len(calls) > 5
    with pytest.raises(SweepAborted):
        sweep(laser, grid=(6, 5), n_pairs=1, stop=stop)
    assert len(calls) == 6 and rig.act.laser_on is False


def test_beam_blocked():
    person = (100, 50, 180, 300)
    assert beam_blocked((150, 200), None, [person])                          # target on the person
    assert beam_blocked((400, 200), (60, 250, 420, 330), [person])          # box overlaps
    assert not beam_blocked((400, 200), (380, 180, 420, 220), [person])
    assert beam_blocked((400, 200), None, [person], head_px=(0, 200))       # person on the beam path
    assert not beam_blocked((400, 200), None, [person], head_px=(640, 200))
    assert not beam_blocked((400, 200), None, [])


def test_hog_finds_nobody_in_the_empty_room():
    import cv2
    rig = RoomRig(seed=5)
    got = people_boxes_hog(rig.frames.latest().img)
    if not hasattr(cv2, "HOGDescriptor"):
        assert got is None                                 # OpenCV 5: unknown, never "nobody"
    else:
        assert got == []


# ---------------------------------------------------------------- main.py routing (action "room:u,v")

@pytest.fixture
def room_main(tmp_path, monkeypatch):
    """main.Room on the simulated room, room pointing on, one zone over the table."""
    import main
    from core.config import load_config
    from core.events import EventLog
    from core.fakeworld import demo_world
    from core.types import Answer
    cfg = load_config()
    cfg = dict(cfg, room=dict(cfg["room"], enabled=True, room_dwell_s=0.3))
    rig = RoomRig(b_cm=3.0, seed=6)
    laser = rig.make_laser()
    laser.room_map = sweep(laser, grid=(12, 9), n_pairs=1)
    tb = rig.box_px("table")
    laser.room_map.zones = {"table": [[tb[0], tb[1]], [tb[2], tb[1]], [tb[2], tb[3]], [tb[0], tb[3]]]}
    events = EventLog(":memory:", str(tmp_path))
    room = main.Room(cfg, demo_world(events), events, None, rig.frames, laser,
                     lambda text, source: Answer("x"))
    room.laser_timeout_s = 60
    room._people_now = lambda img: []           # a person detector that sees nobody (this OpenCV may have no HOG)
    return room, rig, Answer


def wait_for(cond, timeout=3.0):
    import time
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if cond():
            return True
        time.sleep(0.01)
    return False


def test_main_routes_a_room_point_to_aim_px_and_caps_the_dwell(room_main):
    room, rig, Answer = room_main
    t = center(rig.box_px("table"))
    room.aim(Answer("It's on the side table.", action=f"room:{t[0]:.0f},{t[1]:.0f}"))
    assert room.world.laser["on"] is True and room.laser.last_aim["reason"] in ("within_tol", "in_box")
    assert dist(rig.true_dot_px(), t) < 13
    assert wait_for(lambda: room.world.laser["on"] is False)            # room_dwell_s, not laser_timeout_s
    assert rig.act.laser_on is False


def test_main_refuses_room_aims_outside_zones_near_hands_or_when_off(room_main):
    import time
    room, rig, Answer = room_main
    t = center(rig.box_px("table"))
    act = f"room:{t[0]:.0f},{t[1]:.0f}"
    n = len(rig.act.calls)
    room.aim(Answer("x", action="room:320,300"))                        # the floor: no zone there
    room._hand_boxes.append((time.monotonic(), (int(t[0]) - 10, int(t[1]) - 10, int(t[0]) + 10, int(t[1]) + 10)))
    room.aim(Answer("x", action=act))                                   # a hand on the target
    room._hand_boxes.clear()
    room.room_enabled = False
    room.aim(Answer("x", action=act))
    assert len(rig.act.calls) == n and room.world.laser["on"] is False


def test_main_turns_the_laser_off_when_the_dot_is_never_seen(room_main):
    room, rig, Answer = room_main
    t = center(rig.box_px("table"))
    rig.blocked = True
    room.aim(Answer("x", action=f"room:{t[0]:.0f},{t[1]:.0f}"))
    assert room.laser.last_aim["reason"] == "not_seen"
    assert rig.act.laser_on is False and room.world.laser["on"] is False


def test_visual_point_off_the_table_becomes_a_room_action():
    import json
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))
    import test_visual as tv
    from core.events import EventLog
    log = EventLog(":memory:", tempfile_dir())
    reply = json.dumps({"answer": "On the floor.", "confidence": 0.9, "mark": None,
                        "point": {"x": 0.5, "y": 0.5}})                  # (640, 360) px = (45, 30) cm
    small = {"size_cm": [20, 20]}                                        # ... which is off this table
    q, _ = tv.qa(log, reply)
    q.cfg = dict(q.cfg, table=small)
    assert q.look("where's my shoe?").action is None                     # room off: speak only
    q, _ = tv.qa(log, reply)
    q.cfg = dict(q.cfg, table=small, room={"enabled": True})
    a = q.look("where's my shoe?")
    u, v = (float(x) for x in a.action.split(":")[1].split(","))
    assert a.action.startswith("room:") and abs(u - 640) <= 1 and abs(v - 360) <= 1


def tempfile_dir():
    import tempfile
    return tempfile.mkdtemp(prefix="askroom_room_")


# ----- room pointing on the room build (WS10): safety gates, routing from place(), the on-time cap

def test_a_person_near_the_target_or_the_beam_refuses_the_aim(room_main):
    room, rig, Answer = room_main
    t = center(rig.box_px("table"))
    n = len(rig.act.calls)
    room._people_now = lambda img: [(t[0] + 30, t[1] - 200, t[0] + 120, t[1] + 20)]    # someone beside it
    room.aim(Answer("x", action=f"room:{t[0]:.0f},{t[1]:.0f}"))
    assert len(rig.act.calls) == n and room.world.laser["on"] is False


def test_no_person_detector_able_to_look_means_no_aim(room_main):
    room, rig, Answer = room_main
    t = center(rig.box_px("table"))
    n = len(rig.act.calls)
    room._people_now = lambda img: None
    room.aim(Answer("x", action=f"room:{t[0]:.0f},{t[1]:.0f}"))
    assert len(rig.act.calls) == n and room.world.laser["on"] is False


def test_a_point_at_an_object_with_a_full_frame_box_is_a_room_aim_at_its_centre(room_main):
    from types import SimpleNamespace
    room, rig, Answer = room_main
    b = rig.box_px("table")
    room.world.place = lambda name: SimpleNamespace(kind="room", box_px=b, pos_cm=None, via=name)
    room.aim(Answer("Your keys are on the side table.", point_at="keys"))
    assert room.world.laser["on"] is True and room.world.laser["target"] == "keys"
    assert dist(rig.true_dot_px(), center(b)) < 13


def test_hand_boxes_in_the_table_view_are_mapped_into_the_full_frame(room_main):
    room, _, _ = room_main
    room.view_rect = (0, 980, 817, 1440)                               # the rig's table view in 2560x1440
    assert room._view_to_full((0, 0)) == (0, 980)
    assert room._view_to_full((1280, 720)) == pytest.approx((817, 1440))


def test_no_aim_stays_on_longer_than_max_on_s(room_main):
    room, rig, Answer = room_main
    room.max_on_s, room.room_dwell_s = 0.2, 30
    t = center(rig.box_px("table"))
    room.aim(Answer("x", action=f"room:{t[0]:.0f},{t[1]:.0f}"))
    assert room.world.laser["on"] is True
    assert wait_for(lambda: room.world.laser["on"] is False, 1.5) and rig.act.laser_on is False


def test_people_are_looked_for_on_the_perception_thread(room_main):
    """YOLOE isn't thread-safe: the aim asks, the perception thread runs the model on the newest full frame."""
    import threading
    room, _, _ = room_main
    seen = []

    class Prop:
        def people(self, img):
            seen.append(threading.current_thread().name)
            return [(1, 2, 3, 4)]

    room.detector = type("D", (), {"proposer": Prop()})()
    del room._people_now                                 # the real one
    t = threading.Thread(target=lambda: [room._people_want.wait(2), room._serve_people()], name="perception")
    t.start()
    assert room._people_now(None) == [(1, 2, 3, 4)] and seen == ["perception"]
    t.join(2)


# ----- eye safety inside aim_px (review of ws/laser): dark moves, per-try check, jumps, the on-time budget

def lit_moves(act, since: float) -> list:
    """Servo writes made while the laser was on, after `since`."""
    log = [(t, on) for t, on in act.laser_log]
    bad = []
    for t, pan, tilt in act.writes:
        if t < since:
            continue
        state = [on for tl, on in log if tl <= t]
        if state and state[-1]:
            bad.append((t, pan, tilt))
    return bad


def test_aim_px_never_moves_the_head_with_the_laser_on(mapped):
    rig, laser, rm = mapped
    t0 = rig.clock.now()
    for box in ("table", "shelf"):
        laser.aim_px(center(rig.box_px(box)), room_map=rm)      # the second aim slews from the first spot
    assert lit_moves(rig.act, t0) == []
    laser.off()


def test_aim_px_stops_dark_when_the_safety_check_objects(mapped):
    rig, laser, rm = mapped
    calls = []

    def check():
        calls.append(1)
        return "a person near the beam" if len(calls) >= 2 else None
    r = laser.aim_px(center(rig.box_px("table")), room_map=rm, tol_px=0.01, check=check)
    assert r.reason == "unsafe" and not r.on_target and rig.act.laser_on is False
    assert laser.last_aim["unsafe"] == "a person near the beam"


def test_aim_px_stops_dark_at_a_jump(mapped, monkeypatch):
    rig, laser, rm = mapped
    dots = iter([(100.0, 100.0), (600.0, 400.0), (100.0, 100.0), (100.0, 100.0)])   # 2nd look jumps far
    monkeypatch.setattr(laser, "find_dot_px", lambda *a, **k: next(dots))
    r = laser.aim_px(center(rig.box_px("table")), room_map=rm, tol_px=0.01)
    assert r.reason == "jumped" and r.tries == 2 and not r.on_target and rig.act.laser_on is False


def test_aim_px_stops_dark_at_the_on_time_budget(mapped, monkeypatch):
    rig, laser, rm = mapped
    monkeypatch.setattr(laser, "max_on_s", 0.3)
    r = laser.aim_px(center(rig.box_px("table")), room_map=rm, tol_px=0.01)
    assert r.reason == "budget" and not r.on_target and rig.act.laser_on is False
    assert laser.last_aim["lit_s"] >= 0.3
