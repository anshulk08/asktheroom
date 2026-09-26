# 0003: Grok on the detection side (measure first)

Status: planned, spec only. No code. **Needs team OK before any xAI spend.** Nothing here runs on the voice path (spec 0001).

## Problem

The detector is the weakest link. YOLO-World misses small or shiny objects (keys, pill bottle) from overhead, and it can only see the 8 prompted classes. A vision LLM might help in three ways, listed from most to least likely to survive:

- **(a) Auto-labelling.** Pre-label frames for the YOLO11 fine-tune (`scripts/finetune/`), replacing or backing up the YOLO-World pre-labels from `scripts/finetune/autolabel.py`. This is offline, has no demo-time dependency and gives the best payoff.
- **(b) Second opinion.** When the world model is about to decide an object was lost (UNKNOWN), send one still frame and ask where the object is. This is online and slow (seconds), and it only helps a rule that is already uncertain.
- **(c) `find_new`.** Answer "where's my water bottle?" for objects outside the 8 classes. This is the planned case in the README privacy statement: "a single still frame to Grok only when you ask about something the local detector can't see".

None of these is worth building if Grok's boxes are not accurate enough on our overhead view, so measure first.

## Step 1: measurement (the only step approved to plan)

1. On the rig, capture 20 still frames of varied layouts: all 8 objects, some touching or partly overlapping, hands in 5 of them. Save them to `data/grok_eval/` and don't commit them (`.gitignore` does not cover that folder yet). No people's faces: the camera looks straight down at the table.
2. Ground truth for each object:
   - its centre in table cm, measured with a ruler from ArUco marker 0, or
   - hand-fixed YOLO boxes (step 3 of `scripts/finetune/README.md`) mapped to cm with `core.table`'s homography.
3. Ask Grok (model from config `llm.model`) for a box per object in pixel coordinates, one request per frame, via a JSON schema. Map box centres to cm with the same homography.
4. Report per object:
   - hit rate (found when present)
   - false positives (found when absent)
   - median and 90th-percentile centre error in cm
   - latency per frame
   - cost for the 20 frames
5. Run the current YOLO-World detector on the same 20 frames, and report the same numbers next to Grok's.

Write it as a throwaway script (`scripts/grok_boxes_eval.py`, not yet written). It sends images only; no audio or transcripts are involved.

## Decision rule

| Result | Next |
|---|---|
| Grok hit rate at least 95% and median error at most 2 cm on keys and pill bottle | Build (a): Grok pre-labels in `autolabel.py` behind a flag. Hand-fixing is still required |
| Better than YOLO-World on keys and pill bottle, but not the bar above | (a) only as a second pre-labeller, merged for review |
| Not better than YOLO-World | Drop 0003. Fix labels by hand |
| (b) or (c) | Only after the expo, and only if latency per frame is under 2 s and the team accepts sending a still frame off the device. Update the README privacy line before shipping |

## Constraints

- Never on the voice path. Never needed for a demo answer.
- Offline must still work: every use degrades to "not available" with no answer change.
- Spend needs team OK. Keep the eval to about 20 requests.
- (b) and (c) send a still frame of the table off the device. The privacy statement must say so before either ships.

## Acceptance tests

| # | Test | Pass | Status |
|---|---|---|---|
| C1 | Team OK recorded (in PLANS.md or chat) before any xAI call | yes | not done |
| C2 | Measurement on 20 frames, Grok vs YOLO-World, with the table from step 1.4 added to this spec | table present, both systems | not run |
| C3 | If (a) is built: `.venv/bin/python -m pytest -q tests/test_finetune.py` with the Grok path mocked | passes; existing labels are never overwritten without `--overwrite` | not built |
| C4 | If (a) is built: a fine-tuned model trained with Grok pre-labels, on a held-out video | no worse than hand-only labels on the finetune "done when" check | not built |
| C5 | With (b) or (c) enabled and the network unplugged | answers unchanged; no hang longer than 100 ms on the perception thread | not built |
