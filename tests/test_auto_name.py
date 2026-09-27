"""Automatic names for new things (core/auto_name.py): one Grok look at a close-up of each new thing,
stored as a soft guess that questions fall back to after taught names, with hedged answers."""
import json
import time
from pathlib import Path

import numpy as np
import pytest

import core.crops
from core.auto_name import AutoNameConfig, AutoNamer, clean_name, from_config, match_score, not_object
from core.config import Config, load_config
from core.events import EventLog
from core.labels import thing_labels
from core.narration import ProviderError, Reply
from core.types import Frame, Status
from core.world import World
from mobile.bridge import bleproto as P
from tests.synth import Scene
from voice.answers import answer
from voice.intents import parse

ROOT = Path(__file__).resolve().parents[1]
CFG = load_config()
DEO = {"name": "deodorant stick", "also": ["deodorant"], "confidence": 0.9}


class FakeGrok:
    """provider.narrate(system, parts, schema) -> Reply; replies are dicts or exceptions, used in order
    (the last one repeats). Records every call."""
    name, model = "grok", "fake"

    def __init__(self, *replies):
        self.replies = list(replies) or [DEO]
        self.calls = []

    def narrate(self, system, parts, schema):
        self.calls.append((system, parts, schema))
        r = self.replies[min(len(self.calls) - 1, len(self.replies) - 1)]
        if isinstance(r, Exception):
            raise r
        return Reply(json.dumps(r), {}, 5)


class Clock:
    def __init__(self, t=0.0):
        self.t = t

    def __call__(self):
        return self.t


@pytest.fixture
def cfg():
    return Config.load(ROOT / "config.yaml")


@pytest.fixture
def scene(cfg):
    return Scene(cfg, fps=10, t0=1000.0, render=True)


@pytest.fixture
def world(cfg, scene, tmp_path):
    return World(cfg, EventLog(":memory:", str(tmp_path / "snaps")), embed=scene.embed)


@pytest.fixture(autouse=True)
def no_crop_store():
    old = core.crops.active()
    core.crops.set_active(None)
    yield
    core.crops.set_active(old)


def make(world, *replies, online=True, clock=None, **kw):
    net = {"on": online}
    grok = FakeGrok(*replies)
    c = AutoNameConfig(**{"enabled": True, **kw})
    namer = AutoNamer(CFG, world, provider=grok, online=lambda: net["on"], clock=clock or Clock(), c=c,
                      start=False).attach(world)
    return namer, grok, net


def put(scene, world, key, at, **kw):
    scene.thing(key, *at, **kw)
    return scene.run(world, 1.5)


def hide_under_notebook(scene, world, key, at):
    """Slide the notebook over the thing (as in test_openworld's cover test)."""
    x, y = at
    scene.place("notebook", x + 50, y)
    scene.run(world, 1.0)
    for i in range(1, 6):
        scene.place("notebook", x + 50 - 10 * i, y)
        world.update(*scene.step())
    scene.remove(key)
    scene.run(world, 1.5)


def ask(world, text):
    return answer(parse(text, CFG), world, world.events, CFG, now=world._wall)


# ----- the namer -----------------------------------------------------------------------------------

def test_a_new_thing_is_named_in_the_background_and_the_guess_is_in_state(scene, world):
    namer, grok, _ = make(world)
    put(scene, world, "deo", (40, 30))
    assert world.get("thing:1").status == Status.VISIBLE
    assert grok.calls == []                         # perception never waits for Grok
    assert namer.step() is True
    assert len(grok.calls) == 1
    system, parts, schema = grok.calls[0]
    assert [k for k, _ in parts].count("image") == 2          # the close-up and the marked spot, nothing else
    assert set(schema["required"]) == {"object", "name", "also", "confidence"}
    assert namer.guess("thing:1") == {"name": "deodorant stick", "also": ["deodorant"], "confidence": 0.9}
    st = world.state_json()
    ent = next(e for e in st["entities"] if e["name"] == "thing:1")
    assert ent["guess"]["name"] == "deodorant stick"
    assert ent["label"] is None                     # a guess is not a taught name
    assert "disclosure" in st["auto_name"] and "close-up" in st["auto_name"]["disclosure"]
    assert namer.step() is False                    # one successful name per thing
    scene.run(world, 3.0)
    assert namer.step() is False and len(grok.calls) == 1


def test_the_close_up_comes_from_the_crop_store_when_it_has_one(scene, world):
    seen = []

    class Store:
        def for_entity(self, ent):
            seen.append(ent.name)
            crop = core.crops.Crop(img=np.full((40, 60, 3), 90, np.uint8), box_px=(0, 0, 60, 40),
                                   box_cm=(0, 0, 6, 4), t=0.0, score=1.0)
            return core.crops.CropTrack("p1", "thing", (0, 0, 60, 40), (0, 0, 6, 4), 0.0, best=crop)

    core.crops.set_active(Store())
    namer, grok, _ = make(world)
    put(scene, world, "deo", (40, 30))
    namer.step()
    assert seen == ["thing:1"] and len(grok.calls) == 1


def test_hidden_under_the_notebook_where_is_my_deodorant_is_answered_hedged(scene, world):
    namer, _, _ = make(world)
    put(scene, world, "deo", (40, 30))
    namer.step()
    hide_under_notebook(scene, world, "deo", (40, 30))
    assert (world.get("thing:1").status, world.get("thing:1").parent) == (Status.UNDER, "notebook")
    a = ask(world, "where is my deodorant")
    assert a.text == "Your deodorant, I think, is under the notebook."
    assert (a.point_at, a.action) == ("thing:1", "point")
    b = ask(world, "where's my deodorant stick?")
    assert b.text == "Your deodorant stick, I think, is under the notebook."
    c = ask(world, "what happened to my deodorant")
    assert c.text.startswith("Your deodorant, I think, was first seen"), c.text
    assert "notebook" in c.text


def test_a_visible_guessed_thing_is_answered_hedged(scene, world):
    namer, _, _ = make(world)
    put(scene, world, "deo", (40, 30))
    namer.step()
    a = ask(world, "where are my deodorants")
    assert a.text.startswith("Your deodorants, I think, are on the table") or \
        a.text.startswith("Your deodorants, I think,"), a.text
    assert a.point_at == "thing:1"


def test_without_a_guess_the_name_is_still_unknown(scene, world):
    namer, _, _ = make(world, online=False)
    put(scene, world, "deo", (40, 30))
    namer.step()
    assert ask(world, "where is my deodorant").text.startswith("I don't know what your deodorant is yet")


def test_a_taught_name_wins_over_a_guess(scene, world):
    namer, _, _ = make(world)
    put(scene, world, "deo", (40, 30))
    namer.step()                                     # thing:1 guessed 'deodorant stick'
    put(scene, world, "other", (80, 30))
    assert world.bind_alias("thing:2", "deodorant")
    a = ask(world, "where is my deodorant")
    assert "I think" not in a.text and a.point_at == "thing:2", a.text
    # a guessed thing that was then taught another name is no longer found by its guess
    assert world.bind_alias("thing:1", "lotion")
    assert namer.find_guess("deodorant stick") == []
    assert ask(world, "where is my lotion").point_at == "thing:1"


def test_a_thing_with_a_taught_name_is_never_sent(scene, world):
    namer, grok, _ = make(world)
    put(scene, world, "deo", (40, 30))
    assert world.bind_alias("thing:1", "charger")
    namer.step()
    assert grok.calls == [] and namer.guess("thing:1") is None


def test_a_failed_call_is_retried_once_later_then_given_up(scene, world):
    clock = Clock(100.0)
    namer, grok, _ = make(world, ProviderError("timeout"), DEO, clock=clock, retry_after_s=30)
    put(scene, world, "deo", (40, 30))
    assert namer.step() is True and len(grok.calls) == 1
    assert namer.guess("thing:1") is None
    assert namer.step() is False and len(grok.calls) == 1      # not due yet
    clock.t += 31
    assert namer.step() is True and len(grok.calls) == 2
    assert namer.guess("thing:1")["name"] == "deodorant stick"

    put(scene, world, "cup", (80, 30))                           # thing:2: fails twice
    grok.replies = [ProviderError("down")]
    namer.step()
    clock.t += 31
    namer.step()
    clock.t += 300
    assert namer.step() is False and len(grok.calls) == 4
    assert namer.guess("thing:2") is None and namer.status()["failed"] == 1


def test_offline_no_call_until_the_connection_is_back(scene, world):
    namer, grok, net = make(world, online=False)
    put(scene, world, "deo", (40, 30))
    assert namer.step() is False and grok.calls == []
    net["on"] = True
    assert namer.step() is True and len(grok.calls) == 1


def test_calls_are_rate_limited_per_minute(scene, world):
    clock = Clock(0.0)
    namer, grok, _ = make(world, clock=clock, max_per_minute=2)
    for i, x in enumerate((20, 50, 80)):
        put(scene, world, f"t{i}", (x, 30))
    for _ in range(5):
        namer.step()
    assert len(grok.calls) == 2
    clock.t += 61
    namer.step()
    assert len(grok.calls) == 3
    assert {namer.guess(f"thing:{i}")["name"] for i in (1, 2, 3)} == {"deodorant stick"}


def test_a_low_confidence_name_is_not_kept_or_retried(scene, world):
    namer, grok, _ = make(world, {"name": "thing", "also": [], "confidence": 0.9},
                          {"name": "stapler", "also": [], "confidence": 0.2})
    put(scene, world, "a", (40, 30))
    put(scene, world, "b", (80, 30))
    namer.step()
    namer.step()
    namer.step()
    assert len(grok.calls) == 2
    assert namer.guess("thing:1") is None and namer.guess("thing:2") is None


def test_a_body_part_worn_thing_or_colour_is_no_name(scene, world):
    """Rig logs, Sat 26 Sep: hands, arms, watches on wrists and 'white object' were kept as names."""
    namer, grok, _ = make(world, {"object": False, "name": "watch", "also": [], "confidence": 0.9},
                          {"object": True, "name": "Person's Hand", "also": [], "confidence": 0.9},
                          {"object": True, "name": "white object", "also": [], "confidence": 0.9},
                          {"object": True, "name": "tv remote", "also": ["remote", "hand", "arm"], "confidence": 0.9})
    for i, x in enumerate((20, 50, 80, 110)):
        put(scene, world, f"k{i}", (x, 30))
    for _ in range(4):
        namer.step()
    assert len(grok.calls) == 4
    assert [namer.guess(f"thing:{i}") for i in (1, 2, 3)] == [None, None, None]
    assert namer.guess("thing:4") == {"name": "tv remote", "also": ["remote"], "confidence": 0.9}


def test_the_prompt_is_for_a_corner_camera_not_an_overhead_one(scene, world):
    namer, grok, _ = make(world)
    put(scene, world, "deo", (40, 30))
    namer.step()
    system, parts, _ = grok.calls[0]
    assert "corner" in system and "straight down" not in system and "overhead" not in system
    assert "red box" in system and "object: false" in system
    assert [p for k, p in parts if k == "text"][1].endswith("red box:")


class FullFrames:
    """A TableView stand-in: the full frame is the table-view frame at 2x, the view cut from all of it."""
    rect, out_size = (0, 0, 2560, 1440), (1280, 720)

    def __init__(self, same_idx=True):
        self.last, self.same_idx = None, same_idx

    def full_at(self, t):
        import cv2
        f = self.last
        big = cv2.resize(f.img, (2560, 1440), interpolation=cv2.INTER_NEAREST)
        return Frame(t=f.t, wall=f.wall, img=big, idx=f.idx if self.same_idx else f.idx + 1)


def with_full(world, frames):
    """make() with a full-frame source; world.update records each frame for it."""
    grok = FakeGrok()
    namer = AutoNamer(CFG, world, provider=grok, online=lambda: True, clock=Clock(),
                      c=AutoNameConfig(enabled=True), start=False, frames=frames).attach(world)
    inner = world.update

    def update(dets, frame):
        frames.last = frame
        return inner(dets, frame)

    world.update = update
    return namer, grok


def test_the_close_up_is_cut_from_the_full_frame_at_native_resolution(scene, world, cfg, tmp_path):
    namer, _, _ = make(world)
    put(scene, world, "deo", (40, 30))
    small = namer._jobs[0].img
    scene2 = Scene(cfg, fps=10, t0=1000.0, render=True)
    world2 = World(cfg, EventLog(":memory:", str(tmp_path / "s2")), embed=scene2.embed)
    namer2, grok = with_full(world2, FullFrames())
    put(scene2, world2, "deo", (40, 30))
    job = namer2._jobs[0]
    assert abs(job.img.shape[0] - 2 * small.shape[0]) <= 3 and abs(job.img.shape[1] - 2 * small.shape[1]) <= 3
    red = (job.ctx[..., 2] > 200) & (job.ctx[..., 1] < 50) & (job.ctx[..., 0] < 50)
    assert job.ctx.shape[0] >= 240 and red.any()          # the spot, with the thing in a red box
    namer2.step()
    assert [k for k, _ in grok.calls[0][1]].count("image") == 2


def test_queued_views_are_kept_at_their_send_size(scene, world):
    """A big thing's native close-up and 3x context patch are shrunk when queued (Jetson memory)."""
    class Huge(FullFrames):
        rect = (0, 0, 12800, 7200)                         # a 10x full frame: every view is over 384 px

        def full_at(self, t):
            import cv2
            f = self.last
            return Frame(t=f.t, wall=f.wall, img=cv2.resize(f.img, (12800, 7200), interpolation=cv2.INTER_NEAREST),
                         idx=f.idx)

    namer, _ = with_full(world, Huge())
    put(scene, world, "deo", (40, 30))
    job = namer._jobs[0]
    assert max(job.img.shape[:2]) == 384 and max(job.ctx.shape[:2]) == 384


def test_a_full_frame_from_another_moment_is_not_used(scene, world):
    namer, _ = with_full(world, FullFrames(same_idx=False))
    put(scene, world, "deo", (40, 30))
    job = namer._jobs[0]
    assert job.img.shape[0] < 100                          # the table-view cut, not the 2x full frame


def test_no_context_view_when_off(scene, world):
    namer, grok, _ = make(world, context=False)
    put(scene, world, "deo", (40, 30))
    namer.step()
    assert [k for k, _ in grok.calls[0][1]].count("image") == 1


def test_a_pill_bottle_is_named_plainly_and_the_prompt_forbids_claims(scene, world):
    namer, grok, _ = make(world, {"name": "Pill Bottle", "also": ["medicine", "pills taken today"],
                                  "confidence": 0.8})
    put(scene, world, "meds", (40, 30))
    namer.step()
    system = grok.calls[0][0]
    assert "medication" in system.lower() and "never" in system.lower()
    g = namer.guess("thing:1")
    assert g["name"] == "pill bottle"
    assert all("taken" not in a for a in g["also"])


def test_equal_guesses_prefer_the_visible_one_else_both_places_are_said(scene, world):
    namer, _, _ = make(world, {"name": "mug", "also": [], "confidence": 0.9})
    put(scene, world, "m1", (40, 30))
    put(scene, world, "m2", (100, 30))
    namer.step()
    namer.step()
    a = ask(world, "where is my mug")                   # both in view: genuinely ambiguous
    assert a.text == "Two things might be your mug: one is on your right, the other is in the middle."
    assert a.point_at == "thing:2"                      # the most recently seen of the two
    hide_under_notebook(scene, world, "m1", (40, 30))
    a = ask(world, "where is my mug")
    assert (a.point_at, a.text) == ("thing:2", "Your mug, I think, is on the table, on your right. It showed up there just now."), a.text
    scene.remove("m2")                                  # the other one is lost from view
    scene.run(world, 3.0)
    assert world.get("thing:2").status in (Status.UNKNOWN, Status.UNDER)
    a = ask(world, "where is my mug")
    assert a.text == ("Two things might be your mug: one was last seen on your right, "
                      "the other is under the notebook."), a.text


def test_the_worker_thread_names_things_and_stops(scene, world):
    grok = FakeGrok()
    namer = AutoNamer(CFG, world, provider=grok, online=lambda: True,
                      c=AutoNameConfig(enabled=True), start=True).attach(world)
    try:
        put(scene, world, "deo", (40, 30))
        deadline = time.monotonic() + 3.0
        while namer.guess("thing:1") is None and time.monotonic() < deadline:
            time.sleep(0.02)
        assert namer.guess("thing:1")["name"] == "deodorant stick"
    finally:
        namer.stop()
    assert not namer._thread.is_alive()


# ----- pieces ------------------------------------------------------------------------------------

def test_clean_name():
    assert clean_name("A Deodorant Stick.") == "deodorant stick"
    assert clean_name("  the   blue coffee mug with handle ") == "blue coffee mug"
    assert clean_name("object") is None and clean_name("") is None and clean_name(None) is None
    assert clean_name("unknown") is None
    assert clean_name("white") is None and clean_name("white object") is None
    assert clean_name("small black thing") is None
    assert clean_name("white mug") == "white mug" and clean_name("orange") == "orange"
    assert clean_name("white bar of soap") == "bar of soap" and clean_name("small box of pens for") == "box of pens"


def test_not_object():
    for n in ("hand", "hands", "persons leg", "table leg", "grey shirt", "socks", "person", "wooden table",
              "jeans button", "clothing tag"):
        assert not_object(n), n
    for n in ("tv remote", "watch", "white sneaker", "glasses", "hand sanitizer", "laptop", "arm band",
              "robot arm", "shower head", "power button", "microphone arm", "remote button"):
        assert not not_object(n), n
    for n in ("left hand", "persons hand", "bare foot", "white hand", "hand fingers", "denim jeans", "button"):
        assert not_object(n), n


def test_match_score():
    g = {"name": "deodorant stick", "also": ["deodorant"], "confidence": 0.9}
    assert match_score("deodorant stick", g) > match_score("deodorant", {"name": "deodorant stick", "also": []}) > 0
    assert match_score("deodorant", {"name": "deodorant stick", "also": []}) > 0     # token overlap
    assert match_score("deodorants", g) > 0                                          # plural
    assert match_score("my blue mug", {"name": "coffee mug", "also": []}) > 0         # head noun
    assert match_score("phone case", {"name": "phone charger", "also": []}) == 0
    assert match_score("stick", {"name": "deodorant stick", "also": []}) > 0
    assert match_score("glue stick", {"name": "deodorant stick", "also": []}) == 0


def test_config_off_by_default_and_from_config():
    assert AutoNameConfig.from_dict(None).enabled is False
    assert from_config({}, object()) is None
    c = AutoNameConfig.from_dict(CFG.get("auto_name"))
    assert c.enabled and c.max_per_minute >= 1 and 0 < c.min_confidence < 1
    keys = list(CFG)                                # a new section goes at the end (AGENTS.md): after
    assert keys.index("auto_name") > keys.index("table_area")      # the sections that came before it


def test_labels_and_phone_carry_the_guess():
    st = {"entities": [{"name": "thing:3", "label": None, "guess": {"name": "deodorant stick"}},
                       {"name": "thing:4", "label": "charger", "guess": {"name": "cable"}}]}
    assert thing_labels(st) == {"thing:3": "deodorant stick?", "thing:4": "charger"}
    e = P.compact_entity({"name": "thing:3", "kind": "target", "status": "VISIBLE",
                          "guess": {"name": "deodorant stick", "also": [], "confidence": 0.9}})
    assert e["g"] == "deodorant stick"
    assert "g" not in P.compact_entity({"name": "thing:4", "kind": "target", "status": "VISIBLE"})
    prev = {"e": [P.compact_entity({"name": "thing:3", "kind": "target", "status": "VISIBLE"})]}
    assert P.state_changed(prev, {"e": [e]})           # a new guess is news for the phone


def test_no_usable_name_gets_one_fresh_close_up_later(scene, world):
    """Rig run (corner camera): the first close-up of a small dark remote got no usable name; a settled
    view 20 s later is asked once more."""
    clock = Clock()
    namer, grok, _ = make(world, {"name": "object", "also": [], "confidence": 0.3}, REMOTE_REPLY, clock=clock)
    put(scene, world, "a", (40, 30))
    assert namer.step() is True and namer.guess("thing:1") is None
    scene.run(world, 0.5)
    assert namer.step() is False                         # not yet: rename_after_s
    clock.t += 21
    scene.run(world, 0.5)                                 # observe queues the fresh close-up
    assert namer.step() is True
    assert namer.guess("thing:1")["name"] == "remote control" and len(grok.calls) == 2
    clock.t += 60
    scene.run(world, 0.5)
    assert namer.step() is False and len(grok.calls) == 2   # named: never again


def test_a_second_unusable_reply_is_final(scene, world):
    clock = Clock()
    namer, grok, _ = make(world, {"name": "object", "also": [], "confidence": 0.3}, clock=clock)
    put(scene, world, "a", (40, 30))
    namer.step()
    for _ in range(3):
        clock.t += 21
        scene.run(world, 0.5)
        namer.step()
    assert len(grok.calls) == 2 and namer.guess("thing:1") is None


REMOTE_REPLY = {"name": "remote control", "also": ["remote"], "confidence": 0.7}


@pytest.mark.parametrize("reply", [{"object": False, "name": "", "also": [], "confidence": 0.0},
                                   {"object": True, "name": "left hand", "also": [], "confidence": 0.9}])
def test_no_object_is_never_asked_again(scene, world, reply):
    """Only an unsure reply gets the rename's second look: a hand is not asked about twice."""
    clock = Clock()
    namer, grok, _ = make(world, reply, REMOTE_REPLY, clock=clock)
    put(scene, world, "a", (40, 30))
    namer.step()
    for _ in range(3):
        clock.t += 21
        scene.run(world, 0.5)
        namer.step()
    assert len(grok.calls) == 1 and namer.guess("thing:1") is None


def test_a_guessed_answer_receipt_uses_the_asked_for_name(scene, world):
    """Rig 27 Sep 04:34: 'Your remote, I think, is on the table…' came with the caption 'Your thing that looks
    like a remote control, first seen…'; the receipt now says 'Your remote' like the answer."""
    namer, _, _ = make(world)
    put(scene, world, "deo", (40, 30))
    namer.step()
    world.events.flush()
    a = ask(world, "where is my deodorant")
    assert a.text.startswith("Your deodorant, I think,"), a.text
    assert a.evidence and a.evidence[0]["caption"].startswith("Your deodorant, "), a.evidence


def test_match_score_with_colours_rules_out_other_colours_and_ranks_the_one_said():
    from core.auto_name import match_score
    blue, white, plain = ({'name': n, 'also': []} for n in ('blue cup', 'white cup', 'cup'))
    assert match_score('blue cup', white) == 2.0                      # two guesses compared: colour ignored
    assert match_score('blue cup', white, colours=True) == 0.0
    assert match_score('blue cup', blue, colours=True) > match_score('blue cup', plain, colours=True) > 0
    assert match_score('gray mug', {'name': 'grey mug', 'also': []}, colours=True) > 2.0
    assert match_score('orange', {'name': 'orange', 'also': []}, colours=True) == 3.0   # the fruit
