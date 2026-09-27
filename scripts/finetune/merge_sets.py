"""Merge capture sets (one per view: capture.py --room table / --room <zone>, plus public hands) into one
training dir, the view's tag put into every stem so nothing collides and train.py's groups survive:

    python scripts/finetune/merge_sets.py --out data/ft-corner table=data/ft-corner-table \\
        couch=data/ft-corner-couch pubhand=data/hands_public

'cap0_keys-03' from the 'couch' set becomes 'cap0_couch-keys-03' (common.trial_of still gives 'cap0', so
--val-trials cap3 holds out the same poses of every view); 'synth_12' becomes 'synth_couch-12'.
Images and labels only (run synthesize.py in each view's dir first: its backgrounds are that view's)."""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import class_names, write_data_yaml  # noqa: E402


def tagged(stem: str, tag: str) -> str:
    group, _, rest = stem.rpartition("_")
    return f"{group}_{tag}-{rest}" if group else f"{tag}-{rest}"


def merge(out: Path, sets: dict[str, Path]) -> dict[str, int]:
    out = Path(out)
    for sub in ("images", "labels"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    counts = {}
    for tag, src in sets.items():
        if "_" in tag:
            raise ValueError(f"tag {tag!r}: no '_' (train.py splits groups on it)")
        n = 0
        for img in sorted((Path(src) / "images").glob("*.jpg")):
            lab = Path(src) / "labels" / f"{img.stem}.txt"
            if not lab.exists():
                continue
            st = tagged(img.stem, tag)
            shutil.copyfile(img, out / "images" / f"{st}.jpg")
            shutil.copyfile(lab, out / "labels" / f"{st}.txt")
            n += 1
        counts[tag] = n
    return counts


def main(argv=None) -> int:
    from core.config import load_config
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("sets", nargs="+", metavar="TAG=DIR")
    a = ap.parse_args(argv)
    sets = dict(s.split("=", 1) for s in a.sets)
    counts = merge(Path(a.out), {t: Path(d) for t, d in sets.items()})
    write_data_yaml(Path(a.out), class_names(load_config()))
    print(f"merged into {a.out}: {counts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
