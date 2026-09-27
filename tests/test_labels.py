"""Taught names instead of internal ids (core/labels.py) in the overlay and in Grok's world state."""
import json
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pytest

from core.config import Config, load_config
from core.events import EventLog
from core.labels import spoken, thing_label, thing_labels
from core.world import World
from server import overlay
from tests.synth import Scene
from voice import llm

ROOT = Path(__file__).resolve().parents[1]
CFG = load_config()


def test_thing_label():
    assert thing_label('thing:7') == 'something new'
    assert thing_label('thing:7', guess='mug') == 'mug?'
    assert thing_label('thing:7', 'charger') == 'charger'
    assert thing_label('keys') == 'keys'


def test_thing_labels_follow_merges_and_skip_configured_objects():
    st = {'entities': [{'name': 'keys'}, {'name': 'thing:1', 'label': 'charger'}, {'name': 'thing:4', 'label': None}],
          'merged': {'thing:2': 'thing:1', 'thing:3': 'thing:2'}}
    assert thing_labels(st) == {'thing:1': 'charger', 'thing:4': 'something new',
                                'thing:2': 'charger', 'thing:3': 'charger'}
    assert thing_labels(None) == {}


def test_thing_labels_never_number_a_thing_but_keep_each_name_unique():
    # Grok picks a thing by its name (tool enum, point_at), so two look-alikes can't share one
    st = {'entities': [{'name': f'thing:{i}', 'label': None} for i in (3, 8, 12)]
                      + [{'name': 'thing:5', 'guess': {'name': 'mug'}}, {'name': 'thing:6', 'guess': {'name': 'mug'}}]}
    assert thing_labels(st) == {'thing:3': 'something new', 'thing:8': 'something new (2)',
                                'thing:12': 'something new (3)', 'thing:5': 'mug?', 'thing:6': 'mug? (2)'}


def test_spoken():
    labels = {'thing:1': 'charger'}
    assert spoken('thing:1', labels) == 'charger'
    assert spoken('thing:9', labels) == 'something new'
    assert spoken('pill_bottle', labels) == 'pill bottle'
    assert spoken('hand:2', labels) == 'hand 2'
    assert spoken(None, labels) == ''


# ----- a real world with one taught and one untaught thing ----------------------------------------

@pytest.fixture
def world(tmp_path):
    cfg = Config.load(ROOT / 'config.yaml')
    cfg.teach_zone_cm = (0.0, 40.0, 20.0, 60.0)
    scene = Scene(cfg, fps=10, t0=1000.0)
    w = World(cfg, EventLog(':memory:', str(tmp_path / 'snaps')), embed=scene.embed)
    for key, at in (('mug', (50, 20)), ('charger', (10, 50))):
        scene.hand(2, *at)
        scene.thing(key, *at)
        scene.run(w, 0.5)
        scene.hand_off(2)
        scene.run(w, 1.0)
    assert w.teach('my charger') == 'thing:2'
    return w


def test_world_emits_labels(world):
    ents = {e['name']: e for e in world.state_json()['entities']}
    assert ents['thing:2']['label'] == 'charger' and ents['thing:1']['label'] is None


def test_overlay_labels_use_taught_names(world):
    st = world.state_json()
    labels = thing_labels(st)
    ents = {e['name']: e for e in st['entities']}
    assert overlay._label(ents['thing:2'], labels) == 'charger: on table'
    assert overlay._label(ents['thing:1'], labels) == 'something new: on table'
    assert overlay._label(ents['pill_bottle'], labels).startswith('pill bottle: ')
    inside = dict(ents['keys'], status='INSIDE', parent='thing:2')
    assert overlay._label(inside, labels) == 'keys: inside charger'
    assert overlay._label(ents['keys']) .startswith('keys: ')             # labels are optional
    img = np.zeros((720, 1280, 3), np.uint8)
    st['laser'] = {'on': True, 'target': 'thing:2', 'err_cm': 1.0}
    assert overlay.draw(img, st).shape == (540, 960, 3)


def test_grok_world_state_says_names_not_ids(world):
    s = llm.compact_state(world, CFG)
    text = json.dumps(s)
    assert 'thing:' not in text
    names = [e['name'] for e in s]
    assert 'charger' in names and 'something new' in names and 'keys' in names


def test_grok_maybe_same_as_and_parents_use_names():
    class W:
        def state_json(self):
            return {'entities': [
                {'name': 'thing:3', 'status': 'VISIBLE', 'confidence': 1.0, 'label': None,
                 'maybe_same_as': [['thing:1', 0.8]], 'candidates': []},
                {'name': 'thing:1', 'status': 'INSIDE', 'parent': 'box', 'confidence': 0.8,
                 'label': 'charger', 'candidates': ['thing:3']},
                {'name': 'keys', 'status': 'UNDER', 'parent': 'thing:1', 'confidence': 0.85}]}
    s = {e['name']: e for e in llm.compact_state(W(), CFG)}
    assert s['something new']['maybe_same_as'] == ['charger']
    assert s['charger']['candidates'] == ['something new']
    assert s['keys']['parent'] == 'charger'


def test_grok_tools_and_pointing_accept_the_names(world):
    tools = llm._Tools(world, world.events, CFG)
    loc = tools.call('locate', {'object': 'charger'})
    assert loc['object'] == 'charger' and loc['status'] == 'VISIBLE' and 'thing:' not in json.dumps(loc)
    assert tools.call('locate', {'object': 'something new'})['object'] == 'something new'
    assert tools.call('history', {'object': 'charger'})['object'] == 'charger'
    assert 'thing:' not in json.dumps(tools.call('changes_since', {'iso_time': '2000-01-01T00:00:00Z'}))

    names = llm._entity_names(world)
    labels = llm._labels(world)
    a = llm.to_answer('Your charger is on the left.', 'charger', names, CFG, labels)
    assert a.point_at == 'thing:2' and a.action == 'point'          # the laser gets the entity id
    assert llm.to_answer('It is there.', 'something new', names, CFG, labels).point_at == 'thing:1'
    assert llm.to_answer('Keys.', 'keys', names, CFG, labels).point_at == 'keys'


def test_grok_request_lists_names_in_the_prompt_and_enum(monkeypatch, world):
    monkeypatch.setenv('XAI_API_KEY', 'k')
    seen = {}

    def create(**kw):
        seen.update(kw)
        call = NS(id='c1', type='function', function=NS(
            name='respond', arguments=json.dumps({'text': 'Your charger is on the left.', 'point_at': 'charger'})))
        return NS(choices=[NS(message=NS(content=None, tool_calls=[call]))])

    monkeypatch.setattr(llm, '_make_client', lambda *a: NS(chat=NS(completions=NS(create=create))))
    ans = llm.ask_grok('where did I leave my charger', world, world.events, CFG)
    assert ans.point_at == 'thing:2'
    system = seen['messages'][0]['content']
    assert 'thing:' not in system and 'charger' in system
    enum = seen['tools'][0]['function']['parameters']['properties']['object']['enum']
    assert 'charger' in enum and 'something new' in enum and not any(n.startswith('thing:') for n in enum)
