"""eval/scorecard.py on a small synthetic trace: one prop that became two identities, a phantom sock while
people move, a wallet carried to the couch, names from Grok, a removed prop that stays on the table."""
import json

import pytest

from eval import scorecard as sc
from eval.clip import Clip
from eval.score_clip import Sample, Trace

DT = 0.1
A_AT, B_AT, C_AT = (20.0, 20.0), (50.0, 20.0), (90.0, 40.0)


def _truth(steps, props, **kw):
    for s in steps:
        s.setdefault("obj", None)
        s.setdefault("parent", None)
    return dict({"props": props, "steps": steps, "commands": [], "questions": [], "checkpoints": []}, **kw)


def _trace(duration, world, events):
    """world(t) -> (ents, zones, names) per sample every DT from 0 to duration."""
    samples, kinds = [], {}
    for i in range(int(round(duration / DT)) + 1):
        t = round(i * DT, 3)
        ents, zones, names = world(t)
        for n in ents:
            kinds[n] = "target"
        samples.append(Sample(t=t, ents=ents, hands=[], seen={}, zones=zones, names=names))
    return Trace(samples=samples, events=events, kinds=kinds, objects={}, detector="synthetic")


def _clip(truth, duration):
    return Clip(dir=None, meta={"clip": "synthetic"}, truth=truth, t=[0.0, duration], wall=[1.7e9, 1.7e9 + duration])


def _appeared(obj, t, pos):
    return {"t": t, "obj": obj, "type": "APPEARED", "parent": None, "to_cm": list(pos), "from_cm": None}


def room_clip():
    steps = [{"t": 0.0, "event": "hands_out", "seg": "still"},
             {"t": 1.0, "event": "place", "obj": "A", "seg": "people"},
             {"t": 3.0, "event": "place", "obj": "B", "seg": "people"},
             {"t": 5.0, "event": "hands_out", "seg": "still"},
             {"t": 20.0, "event": "carry_to", "obj": "A", "zone": "couch", "seg": "people"},
             {"t": 32.0, "event": "hands_out", "seg": "still"}]
    truth = _truth(steps, {"A": "wallet", "B": "keys"})

    def world(t):
        ents, zones, names = {}, {}, {}
        if t >= 2.0:
            if t < 20.0:
                ents["thing:1"] = ("VISIBLE", None, A_AT if t < 12.0 else (23.0, 20.0))
            elif t < 25.0:
                ents["thing:1"] = ("HELD", "hand:1", None)
            else:
                ents["thing:1"] = ("VISIBLE", None, None)
                zones["thing:1"] = "couch"
            names["thing:1"] = {"guess": {"name": "brown wallet", "also": [], "confidence": 0.8}}
        if t >= 4.0:
            ents["thing:2"] = ("VISIBLE", None, B_AT)
            names["thing:2"] = {"guess": {"name": "car keys"}}
        if 10.0 <= t < 20.0:
            ents["thing:3"] = ("VISIBLE", None, (22.0, 21.0))          # a second identity on the wallet
        if t >= 30.0:
            ents["thing:4"] = ("VISIBLE", None, (80.0, 50.0))          # a foot at the table edge
            names["thing:4"] = {"guess": {"name": "gray sock"}}
        return ents, zones, names

    events = [_appeared("thing:1", 2.0, A_AT), _appeared("thing:2", 4.0, B_AT),
              _appeared("thing:3", 10.0, (22.0, 21.0)), _appeared("thing:4", 30.0, (80.0, 50.0))]
    return _trace(45.0, world, events), _clip(truth, 45.0)


def crit(r, name):
    return next(c for c in r["criteria"] if c["name"] == name)


def test_segments_split_still_from_people_and_still_starts_after_settling():
    steps = [{"t": 0.0, "event": "hands_out"}, {"t": 1.0, "event": "place"}, {"t": 5.0, "event": "hands_out"},
             {"t": 20.0, "event": "carry_to"}, {"t": 30.0, "event": "sit", "seg": "people"}]
    segs = sc.segments(steps, 0.0, 40.0, settle_s=3.0)
    assert segs == [(0.0, 8.0, "people"), (8.0, 20.0, "still"), (20.0, 40.0, "people")]


def test_scorecard_counts_duplicates_phantoms_body_parts_and_handoffs():
    trace, clip = room_clip()
    r = sc.scorecard(trace, clip)
    assert r["mapping"]["A"][0]["entity"] == "thing:1" and r["mapping"]["B"][0]["entity"] == "thing:2"
    epo = r["entities_per_object"]
    assert epo["per_prop"]["A"]["max"] == 2 and epo["per_prop"]["A"]["duplicates"] == ["thing:3"]
    assert epo["per_prop"]["B"]["max"] == 1 and epo["worst"] == 2
    assert crit(r, "entities per real object")["result"] == "FAIL"
    births = r["phantom_births"]
    assert [b["entity"] for b in births["still"]["births"]] == ["thing:3"]      # the placed ones are explained
    assert [b["entity"] for b in births["people"]["births"]] == ["thing:4"]
    assert births["still"]["minutes"] == pytest.approx((12.0 + 10.0) / 60, abs=1e-3)
    assert births["people"]["per_min"] == pytest.approx(1 / ((1.0 + 7.0 + 15.0) / 60), abs=1e-2)
    assert crit(r, "phantom births/min still")["result"] == "FAIL"
    assert r["body_things"]["n"] == 1 and r["body_things"]["things"][0]["entity"] == "thing:4"
    assert r["position_still"]["max_cm"] == pytest.approx(3.0)
    assert crit(r, "position error while still")["result"] == "PASS"
    h = r["handoffs"]["rows"][0]
    assert h["ok"] and h["entity"] == "thing:1" and h["got"] == "couch" and h["delay_s"] == pytest.approx(5.0)
    assert r["naming"]["ok"] == 2 and r["naming"]["named"] == 2
    assert r["things"]["identities_created"] == 4 and r["things"]["expected_end"] == 1
    assert r["pass"] is False


def test_without_names_the_name_metrics_are_not_applicable():
    trace, clip = room_clip()
    for s in trace.samples:
        s.names = {}
    r = sc.scorecard(trace, clip)
    assert crit(r, "body-part / clothing things")["result"] == "n/a"
    assert crit(r, "naming (hook)")["result"] == "n/a"


def test_a_clean_still_clip_passes():
    steps = [{"t": 0.0, "event": "hands_out"}, {"t": 1.0, "event": "place", "obj": "A"},
             {"t": 5.0, "event": "hands_out"}]
    truth = _truth(steps, {"A": "wallet"})

    def world(t):
        return ({"thing:1": ("VISIBLE", None, A_AT)} if t >= 2.0 else {}), {}, {}
    trace = _trace(40.0, world, [_appeared("thing:1", 2.0, A_AT)])
    r = sc.scorecard(trace, _clip(truth, 40.0))
    assert r["entities_per_object"]["worst"] == 1
    assert r["phantom_births"]["still"]["n"] == 0 and r["phantom_births"]["people"]["n"] == 0
    assert crit(r, "room handoffs")["result"] == "n/a" and r["pass"] is True


def test_a_removed_prop_still_seen_at_its_spot_is_a_ghost_and_a_wrong_zone_is_no_handoff():
    steps = [{"t": 0.0, "event": "hands_out"}, {"t": 1.0, "event": "place", "obj": "A"},
             {"t": 3.0, "event": "place", "obj": "C"},
             {"t": 10.0, "event": "remove", "obj": "A"}, {"t": 12.0, "event": "carry_to", "obj": "C",
                                                           "zone": "counter"}]
    truth = _truth(steps, {"A": "wallet", "C": "phone"})

    def world(t):
        ents, zones = {}, {}
        if t >= 2.0:
            ents["thing:1"] = ("VISIBLE", None, A_AT)            # the world never lets the wallet go
        if 4.0 <= t < 12.0:
            ents["thing:2"] = ("VISIBLE", None, C_AT)
        elif t >= 16.0:
            ents["thing:2"] = ("VISIBLE", None, None)
            zones["thing:2"] = "couch"
        return ents, zones, {}
    trace = _trace(30.0, world, [_appeared("thing:1", 2.0, A_AT), _appeared("thing:2", 4.0, C_AT)])
    r = sc.scorecard(trace, _clip(truth, 30.0))
    assert r["removals"]["ghosts"] == 1 and r["removals"]["rows"][0]["seen_there"] == 1.0
    h = r["handoffs"]["rows"][0]
    assert not h["ok"] and h["got"] == "couch"
    assert crit(r, "removed, not ghosted")["result"] == "FAIL" and crit(r, "room handoffs")["result"] == "FAIL"


def test_body_part_names_and_name_scores():
    assert sc.body_part({"guess": {"name": "blue jeans"}}) == "blue jeans"
    assert sc.body_part({"guess": {"name": "remote", "also": ["bare foot"]}}) == "bare foot"
    assert sc.body_part({"label": "charger", "aliases": ["charger"]}) is None
    assert sc.name_score("pill bottle", {"guess": {"name": "pill bottle"}}) == 3.0
    assert sc.name_score("keys", {"aliases": ["my keys"]}) >= 2.0
    assert sc.name_score("wallet", {"guess": {"name": "phone"}}) == 0.0


def test_traces_round_trip_through_json_and_the_cli_scores_them(tmp_path, capsys):
    trace, clip = room_clip()
    back = sc.trace_from_json(json.loads(json.dumps(sc.trace_to_json(trace))))
    assert back.samples[30].ents == trace.samples[30].ents and back.samples[-1].zones == {"thing:1": "couch"}
    assert sc.scorecard(back, clip)["entities_per_object"] == sc.scorecard(trace, clip)["entities_per_object"]
    d = tmp_path / "room_x"
    d.mkdir()
    (d / "truth.json").write_text(json.dumps(clip.truth))
    (d / "frames.json").write_text(json.dumps({"t": clip.t, "wall": clip.wall}))
    (d / "trace.json").write_text(json.dumps(sc.trace_to_json(trace)))
    out = tmp_path / "card.json"
    assert sc.main([str(d), "--from-trace", "--json", str(out)]) == 1
    printed = capsys.readouterr().out
    assert "entities per real object" in printed and "FAIL" in printed
    cards = json.loads(out.read_text())
    assert cards[0]["clip"] == "room_x" and cards[0]["entities_per_object"]["worst"] == 2


def test_room_placement_false_handoff_and_occlusion():
    steps = [{"t": 0.0, "event": "hands_out"}, {"t": 1.0, "event": "place", "obj": "A"},
             {"t": 3.0, "event": "place", "obj": "PB"}, {"t": 5.0, "event": "hands_out"},
             {"t": 10.0, "event": "place_room", "obj": "C", "zone": "couch"},
             {"t": 12.0, "event": "place_room", "obj": "B", "zone": "floor"},
             {"t": 20.0, "event": "block", "obj": "PB"}, {"t": 30.0, "event": "unblock", "obj": "PB"},
             {"t": 34.0, "event": "hands_out"}]
    truth = _truth(steps, {"A": "wallet", "PB": "pill bottle", "C": "phone", "B": "keys"})
    PB_AT = (60.0, 30.0)

    def world(t):
        ents, zones, names = {}, {}, {}
        if t >= 2.0:
            ents["thing:1"] = ("VISIBLE", None, A_AT) if t < 14.0 else ("VISIBLE", None, None)
            if t >= 14.0:
                zones["thing:1"] = "couch"           # the wallet on the table handed to the couch: false
        if t >= 4.0:
            ents["thing:2"] = ("UNKNOWN", None, PB_AT) if 21.0 <= t < 31.0 else ("VISIBLE", None, PB_AT)
        if t >= 13.0:
            ents["thing:3"] = ("VISIBLE", None, None)
            zones["thing:3"] = "couch"               # the phone, first new entity on the couch
        if t >= 25.0:
            ents["thing:4"] = ("VISIBLE", None, None)
            zones["thing:4"] = "side_table"          # a foot handed off
            names["thing:4"] = {"guess": {"name": "sneaker"}}
        return ents, zones, names
    trace = _trace(45.0, world, [_appeared("thing:1", 2.0, A_AT), _appeared("thing:2", 4.0, PB_AT)])
    trace.room = True
    clip = _clip(truth, 45.0)
    clip.meta["room_zones"] = {"zones": {"couch": {}, "side_table": {}, "counter": {}}}
    r = sc.scorecard(trace, clip)
    rows = r["room_placements"]["rows"]
    assert rows[0]["ok"] and rows[0]["entity"] == "thing:3" and rows[0]["delay_s"] == pytest.approx(3.0)
    assert rows[1]["drawn"] is False and rows[1]["ok"] is None and r["room_placements"]["n"] == 1
    assert crit(r, "room placements")["result"] == "PASS"
    fh = {(x["entity"], x["zone"]): x for x in r["false_handoffs"]["rows"]}
    assert set(fh) == {("thing:1", "couch"), ("thing:4", "side_table")}
    assert fh[("thing:1", "couch")]["prop"] == "A" and fh[("thing:4", "side_table")]["name"] == "sneaker"
    assert crit(r, "false handoffs")["result"] == "FAIL"
    o = r["occlusions"]["rows"][0]
    assert o["ok"] and o["before"] == "thing:2" == o["after"] and o["while_hidden"] == "UNKNOWN"
    assert crit(r, "identity through occlusion")["result"] == "PASS"
