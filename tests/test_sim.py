"""The dashboard demo story (server/sim.py) run through the real world model, headless."""
import pytest

from core.config import load_config
from core.types import Detections, Frame, Status
from core.world import World
from server.sim import FPS, Painter, story

CFG = load_config()


def play(pixels: bool = True):
    world = World(CFG)
    scene, _ = story(CFG, seed=0)
    painter = Painter(CFG) if pixels else None
    events = []
    for i, (snap, d) in enumerate(zip(scene.snaps, scene.render())):
        t = 1000.0 + i / FPS
        frame = Frame(t=t, wall=t, idx=i, img=painter.draw(snap, "") if pixels else None)
        events += world.update(Detections(t=t, frame_idx=i, items=d.items, hands=d.hands), frame)
    return world, events


@pytest.mark.parametrize("pixels", [True, False])
def test_demo_story_ends_with_the_right_beliefs(pixels):
    world, _ = play(pixels)
    assert (world.get("keys").status, world.get("keys").parent) == (Status.INSIDE, "box")
    assert world.resolve("keys")[0] == world.get("box").pos_cm          # followed the moved box
    assert (world.get("pill_bottle").status, world.get("pill_bottle").parent) == (Status.UNDER, "notebook")
    assert (world.get("phone").status, world.get("phone").edge) == (Status.GONE, "right")
    for name in ("wallet", "remote", "glasses", "box", "notebook"):
        assert world.get(name).status == Status.VISIBLE, name


def test_objects_moved_in_plain_view_are_logged():
    """Carried objects stay detected in the hand; moving them must still be logged."""
    _, events = play()
    moved = {(e.obj, str(e.type)) for e in events}
    assert ("remote", "MOVED") in moved and ("wallet", "PUT_BACK") in moved
