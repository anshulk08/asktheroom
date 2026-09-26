# 0007: Grok settle check (Grok audits the table, YOLO stays Stage 1)

Status: implemented, off by default (`grok_check.enabled: false`). Supersedes 0003 (b) second opinion and (c) `find_new` in the form below. Not measured on rig frames yet (C2).

## Question

YOLO is finicky on the rig: hands come out as objects, there are phantom objects, wrong classes, and misses on small or shiny props. Should Stage 1 (per-frame detection) be replaced by a VLM (Grok) that writes object positions and timestamps to SQLite?

## Verdict: no. Keep YOLO as Stage 1, fix it, and add Grok as a checker when the table settles

1. **Hands and timing.** Stage 1 is the only source of hand boxes. HELD, INSIDE, GONE, carried-in-view, cover slides, UNDER and new-thing all need hand and object boxes at 10 fps or more (contact within 1 s, container dwell 0.3 s, 6-of-10 debounce, edge exit). A VLM at 0.3–1 Hz misses the covering and carrying moments the shell game is built on.
2. **VLM coordinates are weak.** Our own laptop test gave Grok boxes 0–2/5, a bare point 4/5 and set-of-marks 5/5 (`docs/FEATURE_STATUS.md`). The Set-of-Mark paper (Yang et al. 2023) gives GPT-4V 25.7 with raw coordinates vs 86.4 with marks on RefCOCOg. xAI documents no grounding and has no localization benchmark. AMBER (Wang et al. 2023) finds about 10% phantom "yes" answers for absent objects, and VLMs give no usable confidence. Object permanence from single frames is weak (CLEVR-POC, TCOW, Perception Test at 46% vs 91% for humans).
3. **Prior art is hybrid.** ReMEmbR (NVIDIA 2024), ConceptGraphs (2023), Frigate GenAI and Home Assistant LLM Vision all run a cheap continuous detector or motion trigger, then make event-triggered VLM calls for labels and captions, stored as rows of label, position, time and caption. None of them track hidden or contained objects; our rule-based world model does.
4. **Offline.** PLANS E3 requires the demo with the network unplugged. VLM-only perception sees nothing offline, and expo Wi-Fi and xAI tail latency are unknown. Cost is not the problem: about $7–12 for a 2 h expo at one call per 2 s.
5. **Time and ownership.** A swap means new perception, retuning about 200 world-rule tests, new trials and edits to owned files (`core/detect.py`, `core/world.py`, `core/hands.py`) on freeze day.
6. **SQLite with positions and times already exists.** `events` in `data/events.db` holds cm from/to, monotonic and wall times, and snapshots.

**Track A (the fix for finicky YOLO):** the fine-tune fast path in `scripts/finetune/README.md` (capture with hands, synthesize hand-over-object negatives, train YOLO26s, TensorRT export on the Jetson after asking), then tune `detect_filter` and thresholds from F5 replays. No new code.

**Track B (this spec):** the VLM + SQLite idea in its workable form. YOLO and the world model propose; Grok verifies by picking marks on a settled frame, and its verdicts are stored as rows.

## Design (`core/grok_check.py`)

- **Trigger.** The narration `Segmenter` (`core/narration.py`): hands or events open an episode, `quiet_s` of quiet closes it. The close is "the table settled". There is also one check at start-up, after `quiet_s`. `feed()` runs on the perception thread, is O(1), never raises, and puts the settled frame in a one-slot box (newest wins).
- **Worker** (thread `grok-check`). It calls only when online, under `max_per_hour` and at least `min_gap_s` after the last call. It drops the frame (counted in `dropped`) if an episode has reopened since the frame was taken.
- **Call.** The frame with numbered marks for the world's VISIBLE entities (`voice/visual.tracked_marks`, the same set-of-marks as look/pick), one Grok call (`grok-4.3`, `reasoning_effort: none`, JSON schema): `{marks: [{mark, real, label, confidence}], unmarked: [{label, point {x, y}, confidence}]}`.
- **Verdicts,** one row per mark or unmarked find:

| Verdict | Meaning |
|---|---|
| `agree` | real, and Grok's label matches the world's name (head-noun match) |
| `relabel` | real, but Grok names it something else |
| `named` | real, and the mark is an unnamed `thing:N` that Grok named |
| `phantom` | Grok says nothing is there (confidence ≥ `min_conf`) |
| `unsure` | below `min_conf` |
| `unmarked` | something on the table no mark covers (up to 8, points inside a mark or off the table skipped) |

- **Store.** Table `grok_checks` in `data/events.db` (own table, same connection pattern as `core/narration_store.py`; `core/events.py` unchanged): `id, t, wall, episode, mark, entity, world_label, grok_label, verdict, x_cm, y_cm, confidence, latency_ms, model`. Rows older than `keep_h` are pruned at start. No frame is kept.
- **Effects.**
  1. An unnamed `thing:N` named at `bind_conf` or more takes the name through `world.bind_alias`, only if it has no alias and no entity already has that name. Taught names are never replaced.
  2. Phantoms are recorded, never acted on. The world is unchanged. (Letting a phantom veto an entity would need a `core/world.py` change: a proposal for the owner, after judging.)
  3. `state_json()["grok_check"]` carries the status. The dashboard memory pill shows "check (N to review)" (phantom + relabel + unmarked) and the disclosure in its tooltip.
  4. "Where is my X?" when the world has no position (UNKNOWN, never seen) or the name is unknown: `VisualQA` answers from the newest sighting (`agree`, `relabel`, `named`, `unmarked`) within `sighting_max_age_s`: "I haven't tracked your X, but at 10:42 I saw what looked like your X about here." It circles the laser at that cm. This reads stored rows, so it also works offline. The answer passes the pill filter.
- **Eval.** `python -m core.grok_check --eval DIR [--n 20]` sends up to 20 saved frames (optional sidecar `<frame>.json`: `[{"name", "box_px"}]`, frame equals the table) and prints latency p50/p90, prompt tokens and verdict counts. It needs `XAI_API_KEY` and a team OK for spend (about $0.05).

## Config (`grok_check:`, last section of `config.yaml`)

`enabled` (false), `provider` (grok; `fake` for tests and `--fake`), `model` (grok-4.3), `reasoning_effort` (none), `quiet_s` (1.5), `min_gap_s` (5), `max_per_hour` (120), `look_px` (1280), `timeout_s` (8), `min_conf` (0.6), `bind_names` (true), `bind_conf` (0.7), `sighting_max_age_s` (900), `keep_h` (24).

## Privacy

With `grok_check.enabled: true` and online, one still frame of the table (objects, sometimes hands) goes to Grok each time the table settles. No frame is kept; only the verdict rows are, for `keep_h`. The README privacy statement says so, and `status()["disclosure"]` shows it on the dashboard.

## Acceptance tests

| # | Test | Pass | Status |
|---|---|---|---|
| G1 | `.venv/bin/python -m pytest -q tests/test_grok_check.py` | 22 pass: one call per settle and none during hand activity; none offline, over the cap or inside `min_gap_s`; stale frames dropped; `feed` never blocks on a stalled call; rows with cm and times; phantom leaves the world unchanged; binding rules; sighting fallback only when the world has no position; pill wording neutral | pass (laptop) |
| G2 | `python main.py --fake` with `grok_check: {enabled: true, provider: fake}` and the network up | one check per sim settle, none while the sim hand moves | pass (laptop, 6 checks in 40 s) |
| G3 | Network unplugged, enabled | no calls; answers unchanged; no perception stall over 100 ms (0003 C5) | unit-tested (`build` in fake mode, offline); not run on the rig |
| G4 | Team OK for spend, then `--eval` on 20 rig frames after the Track A model is in | latency p50 under 2 s on the hotspot; numbers added to this spec and `docs/FEATURE_STATUS.md` | not run |
| G5 | Decide whether to enable for judging | only if G4 passes and the team accepts a still frame per settle leaving the device | not decided |
