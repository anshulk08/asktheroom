import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Tests read the committed config.yaml only, never a device's config.local.yaml.
import os
os.environ["ASKROOM_NO_LOCAL_CONFIG"] = "1"
