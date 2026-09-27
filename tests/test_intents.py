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


# ---------- open world: teaching names, and names not in the config ----------

TEACH_CASES = [
    ("This is my charger.", "charger"),
    ("this is my phone charger", "phone charger"),
    ("Okay, this is the blue mug", "blue mug"),
    ("Remember this as my headphones", "headphones"),
    ("remember this as Grandma's ring", "grandmas ring"),
    ("Call this my lucky coin", "lucky coin"),
    ("Call it my lucky coin", "lucky coin"),
    ("This one is my charger, please", "charger"),
    ("That's my water bottle", "water bottle"),
    ("This is called my stapler", "stapler"),
    ("this is my phone", "phone"),                  # a known name: the answer refuses it
]


@pytest.mark.parametrize("text,name", TEACH_CASES)
def test_teach_intent(text, name):
    it = parse(text, CFG)
    assert (it.kind, it.obj, it.name) == ("TEACH", name, name), text


@pytest.mark.parametrize("text", [
    "Let's call it a day.", "What do you call that thing on the table?", "It is a mess.", "This is a mess.",
    "That's my point.", "That's the problem.", "That's the thing.", "This is the best!",
    "Do you remember it as bigger?", "Call it even.", "That's a lot of stuff.", "It is my keys.",
    "I can't remember what to call this.", "Is this my charger?",
])
def test_everyday_phrases_do_not_teach(text):
    assert parse(text, CFG).kind != "TEACH", text


@pytest.mark.parametrize("text,kind,name", [
    ("Where is my charger?", "WHERE", "charger"),
    ("where did I put the blue mug", "WHERE", "blue mug"),
    ("What happened to my charger?", "HISTORY", "charger"),
    ("Did anyone touch my charger today?", "HANDLED", "charger"),
    ("Is my charger in the box?", "WHERE", "charger"),
    ("My charger?", "WHERE", "charger"),
    ("Have you seen my headphones anywhere?", "WHERE", "headphones"),
])
def test_names_not_in_the_config_are_kept_as_spoken(text, kind, name):
    it = parse(text, CFG)
    assert (it.kind, it.obj, it.name) == (kind, None, name), text


@pytest.mark.parametrize("text", ["where is it", "who moved it", "Is there anything on the left side?",
                                  "What's the weather like?", "Did anyone touch anything?",
                                  "Where are my keys?", "What's different about the table?"])
def test_no_spoken_name_where_there_is_none(text):
    assert parse(text, CFG).name is None


def test_taught_aliases_are_matched_like_object_names():
    al = ["phone charger", "blue mug"]
    assert parse("where is my phone charger", CFG, aliases=al).obj == "phone charger"
    assert parse("where is my phone", CFG, aliases=al).obj == "phone"
    it = parse("what happened to the blue mugs", CFG, aliases=al)
    assert (it.kind, it.obj) == ("HISTORY", "blue mug")
    assert parse("Did anyone move my blue mug?", CFG, aliases=al).kind == "HANDLED"


@pytest.mark.parametrize("text,kind,obj", [
    # asking for the laser on a named thing is a location question (offline this was OTHER: no laser)
    ("Show me my wallet", "WHERE", "wallet"),
    ("Could you point to my glasses?", "WHERE", "glasses"),
    ("point at the remote please", "WHERE", "remote"),
    ("light up my phone", "WHERE", "phone"),
    ("highlight the pills", "WHERE", "pill_bottle"),
    ("can you locate my specs", "WHERE", "glasses"),
    ("show me the charger", "WHERE", None),                 # open world: name 'charger'
    ("show me something cool", "OTHER", None),              # nothing named: not a location question
    # yes/no touch questions, more ways to say them
    ("was my wallet moved", "HANDLED", "wallet"),
    ("were the keys touched while I was out", "HANDLED", "keys"),
    ("did my son pick up my glasses", "HANDLED", "glasses"),
    ("did the dog get into my wallet", "HANDLED", "wallet"),
    ("has anyone messed with the remote", "HANDLED", "remote"),
    ("has somebody been at my pills", "HANDLED", "pill_bottle"),
    ("has the phone been picked up", "HANDLED", "phone"),
    # who / when opening the question asks for the history
    ("when did someone last touch my keys", "HISTORY", "keys"),
    ("who's been near my wallet", "HISTORY", "wallet"),
    ("who has touched my phone", "HISTORY", "phone"),
    ("what's going on with my remote", "HISTORY", "remote"),
    ("what's the deal with the pills", "HISTORY", "pill_bottle"),
])
def test_wider_phrasings(text, kind, obj):
    it = parse(text, CFG)
    assert (it.kind, it.obj) == (kind, obj), (text, it)


@pytest.mark.parametrize("text,obj", [
    ("wears my wall it", "wallet"),
    ("wear is my wallet", "wallet"),
    ("where are my kiss", "keys"),
    ("where did I leave my wall et", "wallet"),
    ("wheres the note book", "notebook"),
    ("where is my phon", "phone"),
    ("where are my glases", "glasses"),
])
def test_misheard_names_sound_like_objects(text, obj):
    it = parse(text, CFG)
    assert (it.kind, it.obj) == ("WHERE", obj), (text, it)


@pytest.mark.parametrize("text,name", [
    ("where are my kids", "kids"),             # a real word that only looks like keys stays a name
    ("where is my case", "case"),
    ("where are the papers", "papers"),
    ("where is my mug", "mug"),
])
def test_other_words_are_not_forced_onto_objects(text, name):
    it = parse(text, CFG)
    assert (it.kind, it.obj, it.name) == ("WHERE", None, name), (text, it)


def test_matched_exactly_tells_a_guess_from_a_name():
    from voice.intents import matched_exactly
    assert matched_exactly("where are my keys", "keys", CFG)
    assert matched_exactly("where is the clicker", "remote", CFG)
    assert matched_exactly("wheres the note book", "notebook", CFG)   # the same letters, split
    assert not matched_exactly("where are my kiss", "keys", CFG)
    assert not matched_exactly("where are my keys", None, CFG)


def test_wear_is_only_where_as_the_first_word():
    assert normalize("wears my wallet") == "wheres my wallet"
    assert normalize("what she wears my wallet") == "what she wears my wallet"
    assert normalize("wear sunscreen") == "wear sunscreen"


def test_passive_participles_end_a_spoken_name():
    it = parse("was my charger moved", CFG)
    assert (it.kind, it.name) == ("HANDLED", "charger")
    assert parse("has my stapler been taken", CFG).name == "stapler"


def test_teaching_a_person_is_recognised():
    from voice.intents import names_a_person
    assert names_a_person(parse("This is my wife Karen", CFG).name)
    assert names_a_person("friend") and not names_a_person("travel charger")
    assert not names_a_person("friends mug", "this is my friend's mug")


@pytest.mark.parametrize("text", ["room, this is my mug", "Hey room, this is my mug", "room this is my mug please"])
def test_a_teaching_sentence_may_open_with_the_wake_word(text):
    it = parse(text, CFG)
    assert (it.kind, it.name) == ("TEACH", "mug")


def test_the_wake_word_is_only_stripped_at_the_start():
    assert parse("the room this is my mug", CFG).kind != "TEACH"
    assert parse("roommate this is my mug", CFG).kind != "TEACH"


# -- the room demo (spec 0010): places are where things are, never things

@pytest.mark.parametrize("text, obj", [
    ("where's my wallet", "wallet"), ("where is my glasses case", "glasses"), ("where are my pills", "pill_bottle"),
    ("where did I leave the remote", "remote"), ("is my wallet on the couch", "wallet"),
    ("is the pill bottle on the kitchen counter", "pill_bottle"), ("did I put my glasses on the side table", "glasses"),
    ("where did I put the remote in the living room", "remote"),
])
def test_room_questions_about_the_demo_set(text, obj):
    it = parse(text, CFG)
    assert (it.kind, it.obj) == ("WHERE", obj), (text, it)


@pytest.mark.parametrize("text", ["where's the couch", "where is the kitchen counter", "where's the side table",
                                  "where is the kitchen", "show me the couch", "where's the living room"])
def test_a_place_is_not_an_untaught_thing(text):
    it = parse(text, CFG)
    assert it.kind == "OTHER" and it.name is None, (text, it)


@pytest.mark.parametrize("text", ["is it on the counter?", "are they under the couch", "is it still on the side table",
                                  "and is it in the kitchen"])
def test_a_pronoun_asking_about_a_place_is_a_where_follow_up(text):
    it = parse(text, CFG)
    assert (it.kind, it.obj, it.name) == ("WHERE", None, None), (text, it)


def test_a_thing_named_with_a_place_word_is_still_a_thing():
    assert parse("is my charger on the couch", CFG).name == "charger"
    assert parse("where is my couch cushion", CFG).name == "couch cushion"
