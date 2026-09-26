import pytest

from core.config import load_config
from voice.intents import Intent, normalize, parse

CFG = load_config()

CASES = [
    # WHERE
    ("Where are my keys?", "WHERE", "keys"),
    ("Have you seen my glasses?", "WHERE", "glasses"),
    ("Find my wallet.", "WHERE", "wallet"),
    ("Where'd I put the remote?", "WHERE", "remote"),
    ("Um, where's my phone?", "WHERE", "phone"),
    ("Hey, uh, where are my car keys?", "WHERE", "keys"),
    ("Okay so where did I leave my reading glasses", "WHERE", "glasses"),
    ("WHERE IS MY CELL PHONE", "WHERE", "phone"),
    ("where is the TV remote", "WHERE", "remote"),
    ("Can you find my purse?", "WHERE", "wallet"),
    ("Where is my medicine?", "WHERE", "pill_bottle"),
    ("where's the pill bottle", "WHERE", "pill_bottle"),
    ("Where did I put my pills?", "WHERE", "pill_bottle"),
    ("Have you seen my keychain anywhere?", "WHERE", "keys"),
    ("Where is the book?", "WHERE", "notebook"),
    ("Where's the notepad", "WHERE", "notebook"),
    ("Where did my iPhone go?", "WHERE", "phone"),
    ("My keys?", "WHERE", "keys"),
    ("Are my glasses in the box?", "WHERE", "glasses"),
    ("Is my wallet under the notebook?", "WHERE", "wallet"),
    ("Where are my keys, the ones I put in the box?", "WHERE", "keys"),
    ("Where did anyone move my wallet to?", "WHERE", "wallet"),
    ("I can't find my spectacles.", "WHERE", "glasses"),
    ("where is it", "WHERE", None),
    # HISTORY
    ("What happened to my keys?", "HISTORY", "keys"),
    ("Who moved the box?", "HISTORY", "box"),
    ("When did I last have my phone?", "HISTORY", "phone"),
    ("When did I last see my keys?", "HISTORY", "keys"),
    ("Uh, what happened to the clicker?", "HISTORY", "remote"),
    ("What changed with my wallet?", "HISTORY", "wallet"),
    ("who moved it", "HISTORY", None),
    # HANDLED
    ("Did I pick up my pills?", "HANDLED", "pill_bottle"),
    ("Did I touch my medicine today?", "HANDLED", "pill_bottle"),
    ("Did anyone move my wallet?", "HANDLED", "wallet"),
    ("Did I move my keys?", "HANDLED", "keys"),
    ("Have I taken my meds?", "HANDLED", "pill_bottle"),
    ("Did I take my medication this morning?", "HANDLED", "pill_bottle"),
    ("Has anyone touched my spectacles?", "HANDLED", "glasses"),
    ("I've taken my pills already, right?", "HANDLED", "pill_bottle"),
    ("Did I remember to take my pills?", "HANDLED", "pill_bottle"),
    ("Did you see anyone grab my wallet?", "HANDLED", "wallet"),
    ("Has my wallet been moved?", "HANDLED", "wallet"),
    # CHANGES
    ("What changed while I was gone?", "CHANGES", None),
    ("Is anything different?", "CHANGES", None),
    ("What happened while I was away?", "CHANGES", None),
    ("Did anyone touch anything?", "CHANGES", None),
    ("Hey, um, what's different about the table?", "CHANGES", None),
    # RESET / RECAL
    ("Reset the table.", "RESET", None),
    ("Okay, reset the table please.", "RESET", None),
    ("Recalibrate.", "RECAL", None),
    ("Can you recalibrate the laser?", "RECAL", None),
    ("Reset the calibration.", "RECAL", None),
    # OTHER
    ("Is there anything on the left side?", "OTHER", None),
    ("Which things are hidden?", "OTHER", None),
    ("What's the weather like?", "OTHER", None),
    ("What's in the box?", "OTHER", "box"),
]


@pytest.mark.parametrize("text,kind,obj", CASES)
def test_parse(text, kind, obj):
    it = parse(text, CFG)
    assert isinstance(it, Intent)
    assert (it.kind, it.obj) == (kind, obj), text
    assert it.raw == text


def test_enough_cases():
    assert len(CASES) >= 30


def test_normalize_strips_filler_and_punctuation():
    assert normalize("Um, okay so... Where'd I put the REMOTE?!") == "whered i put the remote"


def test_synonyms_whole_word_only():
    # 'key' must not match inside 'keyboard'; 'pill' must not fire inside 'pillow'
    assert parse("where is the keyboard", CFG).obj is None
    assert parse("where is my pillow", CFG).obj is None


def test_multiword_synonym_beats_single_word():
    assert parse("where is the remote control", CFG).obj == "remote"
    assert parse("where are the car keys", CFG).obj == "keys"


def test_plural_spoken_forms():
    assert parse("where are my phones", CFG).obj == "phone"
    assert parse("where are the pill bottles", CFG).obj == "pill_bottle"


def test_empty_text_is_other():
    assert parse("", CFG).kind == "OTHER"
    assert parse("um uh", CFG).kind == "OTHER"
