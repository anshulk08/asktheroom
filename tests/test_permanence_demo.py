"""Demo acceptance for "The Room Remembers" (spec 0011): one test per shot of the video, from painted full
frames through the object registry, the world adapter and the spoken answer templates. The same shots are
replayed on the guided rig clips with eval/permanence_replay.py once they are recorded."""
import re
import time
from types import SimpleNamespace as NS

import numpy as np

from core.fakeworld import FakeWorld
from core.permanence import CARRIED, HIDDEN, VISIBLE, attach, hook_teach
from core.types import Entity, Status
from tests.test_permanence import BASE, CFG, KEYS, LOOKS, MUG, REMOTE, Clock, Scene, like, reg, run, settle
from voice.answers import answer
from voice.intents import parse

WALLET = (0, 200, 200)
PILLS = REMOTE                                  # the painted pill bottle wears the remote's colour and look
AT = r"\d{1,2}:\d\d (AM|PM)"


def say(fw, text, now):
    a = answer(parse(text, CFG), fw, fw.events, CFG, now=now).text
    assert not re.search(r"\b(taken|took|swallow|thing:|unnamed)", a, re.I), a
    return a


def demo(clock, **kw):
    """The registry with keys, the pill bottle and the wallet enrolled; Grok (faked) picks the right mark."""
    def marks(name, refs, marked, n):
        return 1, 0.9
    p = reg(clock, verify=marks, **kw)
    p.objects["pill_bottle"] = p.objects.pop("remote")
    p.objects["pill_bottle"].name = "pill_bottle"
    p.add_ref("wallet", np.full((40, 40, 3), WALLET, np.uint8))
    return p


def world(p, *names):
    fw = FakeWorld([Entity(n, "target", Status.UNKNOWN, confidence=0.0) for n in names])
    p.events = fw.events
    attach(p, fw)
    return fw


def test_shot1_keys_put_straight_on_the_couch():
    clock, s = Clock(time.time()), Scene()
    p = demo(clock)
    fw = world(p, "keys", "pill_bottle", "wallet")
    s.put("keys", KEYS, (1000, 600, 1040, 630))                  # never on the table
    run(p, s, clock)
    a = say(fw, "Room, where are my keys?", clock.t)
    assert a.startswith("Your keys are on the couch.") and re.search(AT, a), a


def test_shot2_the_wallet_carried_out_then_found_again_on_a_new_spot():
    clock, s = Clock(time.time()), Scene()
    p = demo(clock)
    fw = world(p, "keys", "pill_bottle", "wallet")
    s.put("wallet", WALLET, (200, 520, 260, 550))
    run(p, s, clock)
    assert say(fw, "where is my wallet", clock.t).startswith("Your wallet is on the table.")
    s.take("wallet")                                              # picked up and carried out of the room
    run(p, s, clock, sweeps=3, blockers=[(190, 500, 280, 570)])
    assert p.objects["wallet"].state == CARRIED
    a = say(fw, "where is my wallet", clock.t)
    assert a.startswith("Someone picked up your wallet from the table") and "haven't seen where it went" in a, a
    LOOKS[(15, 190, 190)] = like(BASE[WALLET], 0.2)               # back, on the couch, in other light
    try:
        s.put("wallet", (15, 190, 190), (1000, 600, 1060, 640))
        run(p, s, clock)
        settle(p, s, clock)
    finally:
        del LOOKS[(15, 190, 190)]
    assert p.objects["wallet"].state == VISIBLE
    a = say(fw, "where is my wallet", clock.t)
    assert a.startswith("Your wallet, I think, is on the couch.") and "appeared there" in a, a
    assert [e.type for e in fw.history("wallet", 2)] == ["FOUND", "PICKED_UP"]


def test_shot3_someone_sits_in_front_of_the_pill_bottle():
    clock, s = Clock(time.time()), Scene()
    p = demo(clock)
    fw = world(p, "keys", "pill_bottle", "wallet")
    s.put("pills", PILLS, (1000, 600, 1040, 650))
    run(p, s, clock)
    s.person("sitter", (900, 250, 1200, 720))
    run(p, s, clock, sweeps=10)                                    # sitting there a while: never 'gone'
    assert p.objects["pill_bottle"].state == HIDDEN
    a = say(fw, "where are my pills", clock.t)
    assert a.startswith("Your pill bottle is probably still on the couch, behind someone."), a
    s.leave("sitter")
    run(p, s, clock)
    assert say(fw, "where are my pills", clock.t).startswith("Your pill bottle is on the couch.")
    assert fw.history("pill_bottle", 3) == []                      # nothing happened to it: no events


def test_shot4_the_lucky_mug_is_taught_carried_across_the_room_and_found_by_name():
    clock, s = Clock(time.time()), Scene()
    p = demo(clock)

    class TeachWorld(FakeWorld):
        """A world where 'this is my lucky mug' binds thing:9, the mug on the table (1 cm = 10 px here)."""
        def teach(self, name):
            self.entities["thing:9"].aliases = [name]
            return "thing:9"

        def find(self, name):
            return "thing:9" if name in self.entities["thing:9"].aliases else None

        def alias_phrases(self):
            return list(self.entities["thing:9"].aliases)

        def thing_labels(self):
            return {"thing:9": (self.entities["thing:9"].aliases or [None])[0]}

    fw = TeachWorld([Entity("thing:9", "target", Status.VISIBLE, pos_cm=(33.0, 53.0), box_cm=(30.0, 50.0, 36.0, 56.0),
                            last_seen=clock.t - 60)])
    p.events = fw.events
    attach(p, fw)
    s.put("mug", MUG, (300, 500, 360, 560))
    table = NS(ok=True, cm_to_px=lambda pts: np.asarray(pts, float) * 10)
    hook_teach(p, fw, NS(latest_full=lambda: NS(img=s.frame())), table, (0, 0, 1280, 720))
    from voice.teach import teach_answer
    teach_answer("lucky mug", fw, CFG)                              # "Room, this is my lucky mug"
    assert p.objects["thing:9"].state == VISIBLE
    s.take("mug")
    run(p, s, clock, sweeps=3, blockers=[(290, 480, 380, 580)])
    LOOKS[(190, 10, 10)] = like(BASE[MUG], 0.25)                   # across the room, from another angle
    try:
        s.put("mug", (190, 10, 10), (1000, 600, 1060, 660))
        run(p, s, clock)
        settle(p, s, clock)
    finally:
        del LOOKS[(190, 10, 10)]
    a = say(fw, "where's my lucky mug", clock.t)
    assert "lucky mug" in a and "on the couch" in a, a


def test_shot5_when_did_i_last_pick_up_my_pills_never_says_they_were_taken():
    clock, s = Clock(time.time() - 300), Scene()
    p = demo(clock)
    fw = world(p, "keys", "pill_bottle", "wallet")
    s.put("pills", PILLS, (200, 520, 240, 570))
    run(p, s, clock)
    s.take("pills")
    run(p, s, clock, sweeps=3, blockers=[(190, 500, 260, 590)])     # picked up ...
    s.put("pills", PILLS, (200, 520, 240, 570))
    run(p, s, clock)                                                # ... and put back
    a = say(fw, "When did I last pick up my pills?", clock.t + 120)
    assert re.match(rf"The pill bottle was picked up at {AT}, 2 minutes ago\.", a), a
