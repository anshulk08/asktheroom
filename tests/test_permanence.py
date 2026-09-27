"""Object permanence registry (spec 0011, core/permanence.py) on painted scenes: each object is a rectangle of
its own colour, the fake detector finds colour blobs (people by their colour), and the fake embedder gives each
colour a fixed unit vector, so the real tiling, matching, people and state rules run without models."""
import queue
import time

import numpy as np
import pytest

from core.config import load_config
from core.fakeworld import FakeWorld
from core.permanence import (CARRIED, HIDDEN, LAST_SEEN, UNKNOWN, VISIBLE, Permanence, PermanenceConfig,
                             Places, Region, attach, make_permanence, tile_boxes)
from core.types import Entity, Status
from voice.answers import answer
from voice.intents import Intent

CFG = load_config()
W, H = 1280, 720
PERSON = (200, 200, 200)
rng = np.random.default_rng(0)


def unit(v):
    v = np.asarray(v, np.float32)
    return v / np.linalg.norm(v)


BASE = {c: unit(rng.normal(size=16)) for c in [(0, 0, 200), (0, 200, 0), (200, 0, 0), (0, 200, 200)]}


def like(v, cos):
    """A unit vector at exactly this cosine to v."""
    n = rng.normal(size=16)
    n = unit(n - (n @ v) * v)
    return unit(cos * v + (1 - cos ** 2) ** 0.5 * n)


LOOKS = dict(BASE)
LOOKS[(10, 10, 190)] = like(BASE[(0, 0, 200)], 0.58)          # a look-alike of the remote
REMOTE, KEYS, MUG, DECOY = (0, 0, 200), (0, 200, 0), (200, 0, 0), (10, 10, 190)


class Scene:
    def __init__(self):
        self.items = {}                  # name -> (colour, box)
        self.people = {}

    def put(self, name, colour, box):
        self.items[name] = (colour, box)

    def take(self, name):
        self.items.pop(name, None)

    def person(self, name, box):
        self.people[name] = box

    def leave(self, name):
        self.people.pop(name, None)

    def frame(self):
        img = np.full((H, W, 3), 90, np.uint8)
        for colour, (x1, y1, x2, y2) in self.items.values():
            img[y1:y2, x1:x2] = colour
        for x1, y1, x2, y2 in self.people.values():          # people are in front
            img[y1:y2, x1:x2] = PERSON
        return img


def detect(img):
    out = []
    for colour in list(LOOKS) + [PERSON]:
        m = np.all(img == colour, axis=2)
        if m.sum() < 30:
            continue
        ys, xs = np.nonzero(m)
        out.append(("person" if colour == PERSON else "object", 0.9,
                    (float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1))))
    return out


def embed(img, boxes):
    out = []
    for x1, y1, x2, y2 in boxes:
        c = tuple(int(v) for v in img[int((y1 + y2) / 2), int((x1 + x2) / 2)])
        out.append(LOOKS.get(c))
    return out


class Clock:
    def __init__(self, t=1000.0):
        self.t = t

    def __call__(self):
        return self.t


COUCH = Region("couch", "the couch", [[900, 400], [1280, 400], [1280, 720], [900, 720]])
PLACES = Places([COUCH], table_rect=(0, 450, 600, 720), table_say="the table", frame_wh=(W, H), near_px=120)


def reg(clock=None, verify=None, **kw):
    c = PermanenceConfig.from_dict({"mode": "registry", "tiles": [3, 2], "zoom": [], "miss_s": 2.0,
                                    "verify": verify is not None, **kw})
    p = Permanence(c, detect, embed, places=PLACES, verify=verify, clock=clock or Clock())
    for name, colour in (("remote", REMOTE), ("keys", KEYS)):
        crop = np.full((40, 40, 3), colour, np.uint8)
        assert p.add_ref(name, crop)
    return p


def run(p, scene, clock, sweeps=1, dt=1.0, blockers=()):
    evs = []
    for _ in range(sweeps):
        clock.t += dt
        evs += p.sweep(scene.frame(), clock.t, blockers)
    return evs


# ----- construction ----------------------------------------------------------------------------------

def test_mode_off_builds_nothing_and_leaves_the_world_alone():
    fw = FakeWorld([Entity("remote", "target", Status.VISIBLE, pos_cm=(10.0, 10.0))])
    place, state = fw.place, fw.state_json
    assert make_permanence({**CFG, "permanence": {"mode": "off"}}, fw) is None
    assert make_permanence(CFG, fw) is None              # the committed default is off
    assert fw.place == place and fw.state_json == state


def test_tiles_cover_the_frame_with_overlap():
    tiles = tile_boxes(2560, 1440, (3, 2), 0.15)
    assert len(tiles) == 6 and tiles[0][0] == 0 and tiles[-1][2] == 2560 and tiles[-1][3] == 1440
    assert tiles[1][0] < tiles[0][2]                      # neighbours overlap


# ----- registry and matching -------------------------------------------------------------------------

def test_a_registered_object_is_found_and_said_where_it_is():
    clock, s = Clock(), Scene()
    p = reg(clock)
    s.put("remote", REMOTE, (100, 500, 160, 540))
    run(p, s, clock)
    o = p.objects["remote"]
    assert o.state == VISIBLE and o.say == "the table" and o.zone == "table"
    assert p.objects["keys"].state == UNKNOWN                     # never seen: nothing said about it


def test_keys_put_straight_on_the_couch_are_found_there_without_a_handoff():
    clock, s = Clock(), Scene()
    p = reg(clock)
    s.put("keys", KEYS, (1000, 600, 1040, 630))                  # never on the table first
    run(p, s, clock)
    assert p.objects["keys"].state == VISIBLE and p.objects["keys"].say == "the couch"


def test_an_unregistered_object_never_becomes_an_identity():
    clock, s = Clock(), Scene()
    p = reg(clock)
    s.put("mug", MUG, (300, 100, 360, 160))
    run(p, s, clock, sweeps=3)
    assert set(p.objects) == {"remote", "keys"} and all(o.state == UNKNOWN for o in p.objects.values())


def test_a_look_alike_is_not_taken_for_the_object_and_one_candidate_serves_one_object():
    clock, s = Clock(), Scene()
    p = reg(clock)
    s.put("decoy", DECOY, (300, 100, 360, 160))                   # similar, below sim_accept_far
    run(p, s, clock, sweeps=2)
    assert p.objects["remote"].state == UNKNOWN
    s.put("remote", REMOTE, (100, 500, 160, 540))
    run(p, s, clock)
    assert p.objects["remote"].box == (100, 500, 160, 540)


# ----- states ----------------------------------------------------------------------------------------

def test_carried_away_then_found_again_on_the_couch():
    clock, s = Clock(), Scene()
    p = reg(clock)
    s.put("remote", REMOTE, (100, 500, 160, 540))
    run(p, s, clock)
    s.take("remote")
    evs = run(p, s, clock, sweeps=3, blockers=[(120, 480, 200, 560)])     # a hand at it as it went
    o = p.objects["remote"]
    assert o.state == CARRIED and [e.type for e in evs] == ["PICKED_UP"]
    s.put("remote", REMOTE, (1000, 600, 1060, 640))
    evs = run(p, s, clock)
    assert o.state == VISIBLE and o.say == "the couch" and o.arrival_observed
    assert [e.type for e in evs] == ["FOUND"]


def test_gone_without_contact_is_last_seen():
    clock, s = Clock(), Scene()
    p = reg(clock)
    s.put("remote", REMOTE, (100, 500, 160, 540))
    run(p, s, clock)
    s.take("remote")
    evs = run(p, s, clock, sweeps=3)
    assert p.objects["remote"].state == LAST_SEEN and [e.type for e in evs] == ["LOST_TRACK"]


def test_a_person_in_front_makes_it_hidden_and_leaving_without_it_makes_it_carried():
    clock, s = Clock(), Scene()
    p = reg(clock)
    s.put("remote", REMOTE, (1000, 600, 1060, 640))
    run(p, s, clock)
    s.person("judge", (950, 300, 1150, 720))                       # stands in front of it
    evs = run(p, s, clock, sweeps=5)
    assert p.objects["remote"].state == HIDDEN and evs == []
    s.leave("judge")                                               # still there: visible again, no event
    evs = run(p, s, clock)
    assert p.objects["remote"].state == VISIBLE and evs == []
    s.person("judge", (950, 300, 1150, 720))
    run(p, s, clock)
    s.take("remote")
    s.leave("judge")                                               # left, and it's gone: they took it
    evs = run(p, s, clock)
    assert p.objects["remote"].state == CARRIED and [e.type for e in evs] == ["PICKED_UP"]


def test_views_that_do_not_contain_it_count_for_nothing():
    clock, s = Clock(), Scene()
    p = reg(clock)
    s.put("remote", REMOTE, (100, 100, 160, 140))                  # top-left tile only
    run(p, s, clock)
    s.take("remote")
    img = s.frame()
    far = [i for i, v in enumerate(p.views(W, H)) if v[0] > 600 and v[1] > 300]   # bottom-right tile
    for _ in range(10):
        clock.t += 1
        p.look(img, far[0], clock.t)
    assert p.objects["remote"].state == VISIBLE


def carried_off(p, s, clock):
    s.put("remote", REMOTE, (100, 500, 160, 540))
    run(p, s, clock)
    s.take("remote")
    run(p, s, clock, sweeps=3, blockers=[(120, 480, 200, 560)])
    assert p.objects["remote"].state == CARRIED


def settle(p, s, clock):
    deadline = time.time() + 2
    while any(o.asking for o in p.objects.values()) and time.time() < deadline:
        run(p, s, clock, dt=0.1)
    run(p, s, clock)


@pytest.mark.parametrize("mark, online, want", [(1, True, VISIBLE), (0, True, CARRIED), (1, False, CARRIED)])
def test_a_carried_object_is_found_at_a_new_arrival_only_when_grok_picks_it(mark, online, want):
    # measured: the same remote on another surface can look nothing like its references (cosine 0.1-0.3)
    clock, s = Clock(), Scene()
    asked = []

    def verify(name, refs, marked, n):
        asked.append((name, n))
        return mark, 0.9

    p = reg(clock, verify=verify)
    p.online = lambda: online
    carried_off(p, s, clock)
    LOOKS[(20, 20, 180)] = like(BASE[REMOTE], 0.2)                 # the remote on the couch, in other light
    try:
        s.put("remote", (20, 20, 180), (1000, 600, 1060, 640))
        run(p, s, clock)
        settle(p, s, clock)
        o = p.objects["remote"]
        assert o.state == want and bool(asked) == online
        if want == VISIBLE:
            assert o.say == "the couch" and p.place("remote").tentative        # found by Grok: hedged
            assert o.bank.sim(LOOKS[(20, 20, 180)]) > 0.99                    # and learned: no Grok next time
            asked.clear()
            run(p, s, clock, sweeps=3)
            assert asked == [] and o.state == VISIBLE
    finally:
        del LOOKS[(20, 20, 180)]


def test_suspects_are_new_arrivals_not_what_another_object_holds_and_people_last():
    clock, s = Clock(), Scene()
    p = reg(clock, verify=lambda *a: (0, 0.0))
    s.put("keys", KEYS, (700, 100, 740, 130))                      # keys: registered and visible
    carried_off(p, s, clock)
    LOOKS[(20, 20, 180)] = like(BASE[REMOTE], 0.3)
    LOOKS[(30, 30, 170)] = like(BASE[REMOTE], 0.8)
    try:
        s.person("sitter", (850, 250, 1250, 720))
        s.put("on lap", (30, 30, 170), (1000, 500, 1040, 530))          # inside the person box
        s.put("new", (20, 20, 180), (300, 100, 340, 130))
        img = s.frame()
        img[500:530, 1000:1040] = (30, 30, 170)
        clock.t += 1
        p.sweep(img, clock.t)
        sus = [k for _, k in p.suspects(p.objects["remote"], clock.t)]
        assert [k.box for k in sus] == [(300, 100, 340, 130), (1000, 500, 1040, 530)]
        assert all(k.box != (700, 100, 740, 130) for k in sus)           # the keys' own candidate
    finally:
        del LOOKS[(20, 20, 180)], LOOKS[(30, 30, 170)]


def test_reset_forgets_places_but_keeps_looks():
    clock, s = Clock(), Scene()
    p = reg(clock)
    s.put("remote", REMOTE, (100, 500, 160, 540))
    run(p, s, clock)
    p.reset()
    assert p.objects["remote"].state == UNKNOWN and p.objects["remote"].bank.vecs


# ----- answers and /state through the world adapter --------------------------------------------------

def world_with(p):
    fw = FakeWorld([Entity("remote", "target", Status.GONE, pos_cm=None, last_seen=900.0),
                    Entity("thing:3", "target", Status.UNKNOWN, pos_cm=None)])
    attach(p, fw)
    return fw


def ask(fw, text, now):
    from voice.intents import parse
    return answer(parse(text, CFG), fw, fw.events, CFG, now=now).text


def test_where_reads_the_registry_for_each_state():
    clock, s = Clock(), Scene()
    p = reg(clock)
    fw = FakeWorld([Entity("remote", "target", Status.GONE, pos_cm=None, last_seen=900.0)])
    attach(p, fw)
    s.put("remote", REMOTE, (1000, 600, 1060, 640))
    run(p, s, clock)
    assert ask(fw, "where is my remote", clock.t).startswith("Your remote is on the couch.")
    s.person("judge", (950, 300, 1150, 720))
    run(p, s, clock)
    assert ask(fw, "where is my remote", clock.t).startswith("Someone is in front of your remote right now.")
    s.take("remote")
    s.leave("judge")
    run(p, s, clock)
    a = ask(fw, "where is my remote", clock.t)
    assert a.startswith("Someone picked up your remote from the couch") and "haven't seen where it went" in a
    p.objects["remote"].state = LAST_SEEN
    assert ask(fw, "where is my remote", clock.t).startswith("I last saw your remote on the couch")


def test_the_table_world_wins_while_the_table_has_it():
    clock, s = Clock(), Scene()
    p = reg(clock)
    fw = FakeWorld([Entity("remote", "target", Status.VISIBLE, pos_cm=(40.0, 30.0), last_seen=clock.t)])
    attach(p, fw)
    s.put("remote", REMOTE, (1000, 600, 1060, 640))
    clock.t -= 1
    run(p, s, clock)                                              # clock back at the table sighting
    assert fw.place("remote", clock.t).kind == "table"
    assert fw.place("remote", clock.t + 10).kind == "room"        # the table lost it: the registry answers


def test_history_and_what_changed_read_the_registry_events():
    clock, s = Clock(time.time()), Scene()
    p = reg(clock)
    fw = FakeWorld([Entity("remote", "target", Status.GONE, pos_cm=None)])
    p.events = fw.events
    attach(p, fw)
    s.put("remote", REMOTE, (100, 500, 160, 540))
    run(p, s, clock)
    s.take("remote")
    run(p, s, clock, sweeps=3, blockers=[(120, 480, 200, 560)])
    s.put("remote", REMOTE, (1000, 600, 1060, 640))
    run(p, s, clock)
    assert [e.type for e in fw.history("remote", 3)] == ["FOUND", "PICKED_UP"]
    text = answer(Intent("CHANGES", None, "what changed"), fw, fw.events, CFG, now=clock.t).text
    assert text.startswith("Your remote was picked up") or text.startswith("The remote was picked up"), text


def test_state_json_carries_the_registry_and_hides_unnamed_things():
    clock, s = Clock(), Scene()
    p = reg(clock)
    fw = world_with(p)
    s.put("remote", REMOTE, (1000, 600, 1060, 640))
    run(p, s, clock)
    st = fw.state_json()
    ents = {e["name"]: e for e in st["entities"]}
    assert "thing:3" not in ents                                  # unnamed and not visible: temporary
    r = ents["remote"]
    assert r["zone"] == "couch" and r["status"] == "VISIBLE" and r["registry"]["say"] == "the couch"
    assert st["room"]["remote"]["say"] == "the couch" and st["permanence"]["objects"] == 2


def test_teach_registers_the_named_object_from_the_full_frame():
    clock, s = Clock(), Scene()
    p = reg(clock)
    s.put("mug", MUG, (300, 100, 360, 160))
    assert p.teach("thing:7", s.frame(), (300, 100, 360, 160))
    assert p.objects["thing:7"].state == VISIBLE
    s.take("mug")
    s.put("mug", MUG, (1000, 600, 1060, 660))
    run(p, s, clock)
    assert p.objects["thing:7"].say == "the couch"


def test_a_still_room_costs_no_new_embeddings():
    clock, s = Clock(), Scene()
    calls = []

    def counting(img, boxes):
        calls.append(len(boxes))
        return embed(img, boxes)

    c = PermanenceConfig.from_dict({"mode": "registry", "tiles": [3, 2], "zoom": [], "verify": False})
    p = Permanence(c, detect, counting, places=PLACES, clock=clock)
    p.add_ref("remote", np.full((40, 40, 3), REMOTE, np.uint8))
    s.put("remote", REMOTE, (100, 500, 160, 540))
    s.put("mug", MUG, (700, 100, 760, 160))
    run(p, s, clock)
    first = sum(calls)
    run(p, s, clock, sweeps=3)
    assert sum(calls) == first                                     # nothing moved: every embedding reused
    s.put("mug", MUG, (300, 300, 360, 360))
    run(p, s, clock)
    assert sum(calls) > first                                       # moved: embedded again
    clock.t += c.reembed_s
    before = sum(calls)
    run(p, s, clock)
    assert sum(calls) > before                                      # and refreshed now and then


def test_places_are_said_in_human_terms():
    assert PLACES.at((100, 500, 160, 540)) == ("table", "the table")
    assert PLACES.at((1000, 600, 1060, 640)) == ("couch", "the couch")
    assert PLACES.at((800, 600, 840, 640)) == ("near:couch", "near the couch")
    assert PLACES.at((100, 100, 140, 140)) == ("room:left", "the left side of the room")
