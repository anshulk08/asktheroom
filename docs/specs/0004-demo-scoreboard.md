# 0004: Live scoreboard from real trials

Status: planned. It builds on `eval/`, which is implemented and tested (`tests/test_eval.py`). The dashboard part is not built.

## Problem

Judges hear many "it works" claims. A number from real trials on this table, compared against simple baselines, is a stronger claim. The only numbers we have so far are synthetic (`eval.synth`, 255/300). **Never show or quote the synthetic score as accuracy.** It is a regression check.

## Decision

1. **Record real trials** with the existing tools. The trial format is in `eval/trial.py`.
   ```bash
   python -m eval.record --trial-id 17 --category covered --object keys --truth notebook --detect
   python -m eval.replay --trials trials
   python -m eval.report --trials trials --out report.md --db data/events.db
   ```
   The report's hidden subtotal (covered + inside + inside_box_moved) is the headline, shown next to the three baselines `current_frame`, `last_seen` and `nearest_object`.
2. **Only count real trials.** The scoreboard counts only trials whose `truth.json` has `"source": "camera"` or `"file"`. `"synth"` never counts. Today `eval.report` doesn't filter by source, so add a filter (`--real-only`, or skip `source == "synth"` in `eval.report.collect`). The eval code is not a teammate-owned core file, so this is ours to change.
3. **Scoreboard on the dashboard.**
   - A small panel shows the hidden subtotal and overall, for `full` vs the best baseline, as `correct/total`. It also shows the median laser error.
   - It reads a JSON file written by `eval.report` (for example `trials/scoreboard.json`, next to `report.md`) via a new read-only route, `GET /scoreboard`, in `server/app.py`. The server doesn't run replays.
   - If no file exists, the panel is hidden.
4. **Judge trials at the expo.**
   - After a judge's shell game, the operator records the outcome with one command. It uses the clip if the recorder was running; otherwise it is a truth-only row with the answer the rig gave.
   - Re-running report refreshes the panel.
   - Truth is what the operator saw, typed with `--truth`. Nobody edits results by hand.

## Rules

- Count every real trial, including the failures. The panel shows `n`.
- Tune thresholds only from replays (PLANS E1). Re-run replay on all trials after any change, and show the new numbers, not the best ones.
- Real-trial rows in the README appendix stay "TBD" until D4 passes.

## Acceptance tests

| # | Test | Pass | Status |
|---|---|---|---|
| D1 | `.venv/bin/python -m pytest -q tests/test_eval.py` | passes | passing |
| D2 | New test: a synth trial plus a real trial in a temp `trials/`; report with the real-only filter | counts only the real trial | not built |
| D3 | New test: `GET /scoreboard` with and without the JSON file | 200 with the numbers / 404 (panel hidden) | not built |
| D4 | At least 3 real trials per core category recorded and replayed (PLANS F5) | `report.md` has real numbers for `full` and all three baselines | not run |
| D5 | On the rig: one judge-style trial recorded and report re-run | the dashboard panel updates within 1 minute; `n` goes up by 1 | not built |
| D6 | `grep -rn "255" README.md Demo/README.md docs/` | the synthetic score is never presented as accuracy | passing |
