import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Tests read the committed config.yaml only, never a device's config.local.yaml.
import os
os.environ["ASKROOM_NO_LOCAL_CONFIG"] = "1"

import pytest


@pytest.fixture(autouse=True)
def _no_real_rig_voice(monkeypatch, tmp_path):
    """Speech tests never call Grok with a real key from the environment, nor read or write the
    device's data/voice.json (the phone's voice settings)."""
    from voice import tts
    monkeypatch.setattr(tts.TTS, "_grok_key", staticmethod(lambda: None))
    monkeypatch.setattr(tts, "VOICE_PATH", str(tmp_path / "voice.json"))
