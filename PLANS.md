# Plan

The current plan for HackGT 13, with checkpoints up to the feature freeze and the expo. Feature status is in `docs/FEATURE_STATUS.md`. Designs for adopted ideas are in `docs/specs/`.

## Deadlines (EDT)

| When | What |
|---|---|
| Fri Sep 25, 8 PM | Hacking started (first commit 20:05) |
| **Sat Sep 26, 6 PM** | **Feature freeze.** After this: bug fixes, tuning from replays, docs |
| Sun Sep 27, 8 AM | Hacking ends. Submit to Devpost before this |
| Sun Sep 27, 9:00–11:15 AM | Expo judging, Klaus. Judges rotate, so the demo must reset in under a minute |

## Goal

A judge runs the shell game on the real table (keys under the notebook, notebook into the box, box slid across) and asks out loud where the keys are. The rig answers correctly and the laser lands on the box, every time, with no network needed. A live scoreboard of real trials backs the claim.

## Decisions (settled)

- The mic is always listening (`listen.mode: always`). The clicker is a "listen now" override and interrupts an answer (barge-in).
- A local Qwen on the Jetson turns speech into commands. Grok is used only on the detection side.
- YOLO stays Stage 1 (Sat research, spec 0007): a VLM can't give hand and object boxes at 10 fps, place boxes well, or work offline. Fix YOLO with the fine-tune fast path; Grok only checks the table when it settles.
- Audio and transcripts that aren't used are deleted: audio lives only in RAM, and ignored speech is never logged.
- The world model stays rule-based and deterministic.

## Handoff ideas: verdicts

| Idea | Verdict | Why (tied to the code) | Where |
|---|---|---|---|
| Local command interpreter | **Adopted, done** | `voice/understand.py` (rules first, Qwen3-1.7B with a JSON schema, thinking off, `sounds_like` guard) and `voice/local_llm.py` replace Grok on the voice path. 58/64 on the eval (laptop) | spec 0001 |
| Action set point/circle/trace/tour/sweep/off/ignore/find_new | **Adapted** | We kept point, circle, sweep (already used for GONE) and none. `IGNORE` exists as an intent for overheard speech. trace, tour and find_new need new laser and world code, so they are deferred to the roadmap | spec 0001 |
| Templates vs LLM sentences | **Keep templates** for WHERE/HISTORY/HANDLED/CHANGES and common open questions | Deterministic, tested (`tests/test_answers.py`), instant. The 1.7B model still slips on counts and lists | spec 0001 |
| Throttle the detector to ~5 fps while Qwen generates | **Deferred** | Measure first with `tegrastats` on the Jetson, with YOLO, whisper and Qwen all loaded | spec 0001 |
| Stream the answer action-first | **Deferred** | Qwen answers in about 0.3–0.5 s on the laptop and templates are instant, so there isn't much to gain yet. The schema is already action-first | spec 0001 |
| Always-listening pipeline | **Adopted, done** | VAD → whisper → keyword gate → addressed check → rules/Qwen. The mic is shut while speaking plus `echo_tail_s`. Wake and click modes exist. It still needs the 10-minute hall-noise acceptance run on the Jetson | spec 0002 |
| Grok on the detection side (auto-label, second opinion, find_new) | **Spec only; measure first** | Measure box error on 20 frames against ArUco ground truth before building anything. xAI spend needs team OK. (a) auto-labelling is the most likely to survive | spec 0003 |
| Replace Stage 1 YOLO with Grok + SQLite positions and times | **Rejected; adapted as the Grok settle check** | Hands need boxes at 10 fps or more; VLM boxes are weak (ours 0–2/5 vs marks 5/5); offline would see nothing. `core/grok_check.py` checks the tracked marks when the table settles and stores verdict rows in `grok_checks`. Off by default; `--eval` needs team spend OK | spec 0007 |
| Judge-run shell game | **Adopted** | This is what sets us apart. Covered by the world rules and `tests/test_world_rules.py` | README demo flow |
| Laser circles hidden/unsure targets | **Exists** | `act/laser.py` `circle()`. Used for UNKNOWN and low confidence, and by `local_llm` for hidden objects. Traced history is deferred | spec 0001 |
| Live scoreboard from real trials | **Adopted** | Built on `eval.record` / `eval.replay` / `eval.report`. Never quote the synthetic 255/300 as real | spec 0004 |
| Lamp-head enclosure | **Optional** | Only if the rig is stable by Sat afternoon. The Hive laser cutters are open Sat 3–9 PM, and ArUco recalibration makes re-mounting cheap | roadmap |
| Snapshot retention | **Exists** | `EventLog` prunes snapshots and state snapshots older than 24 h on start | README privacy |
| Honest privacy statement | **Done** | README and CONTEXT.md. The `local_llm` privacy template answers out loud | README |
| n8n changes | **Handled separately** | Another workstream owns `n8n/` | `n8n/README.md` |
| Devpost disclosure | **None needed** beyond crediting open-source models and libraries | All code was written after the Friday 8 PM start | Devpost |

## Checkpoints to the freeze (Sat 6 PM)

Owners are TBD until the team assigns them.

| # | Checkpoint | Done when | Owner |
|---|---|---|---|
| F1 | llama.cpp built on the Jetson, Qwen3-1.7B served | `scripts/qwen_server.sh` answers `/health` on the Jetson. Ask before building | TBD |
| F2 | Jetson timing with everything loaded | `tegrastats` peak RAM, `scripts/eval_understand.py` median Qwen ms, and perception fps recorded in `docs/FEATURE_STATUS.md` | TBD |
| F3 | Hall-noise acceptance (spec 0002) | `scripts/overheard_test.py` on 10 min of recorded hall noise: 0–1 false triggers, or switch the default to `wake` | TBD |
| F4 | whisper.cpp `whisper-cli` on the Jetson | `stt.backend: cli` transcribes the 20 test clips | TBD |
| F5 | Real recorded trials | At least 3 per core category recorded with `eval.record`, replayed against the baselines | TBD |
| F6 | Laser calibrated on the rig | `demo_check.py` check 4 green (fit < 1.5 cm, centre < 3 cm) | TBD |
| F7 | Scoreboard on the dashboard (spec 0004) | Shows real-trial counts only, updates after a judge trial | TBD |
| F8 | Docs current | README, CONTEXT.md, FEATURE_STATUS.md match the code | TBD |

## Checkpoints from freeze to expo

| # | Checkpoint | Done when | Owner |
|---|---|---|---|
| E1 | Tune from replays only | Thresholds changed only with a replay that shows the gain | TBD |
| E2 | Demo rehearsal x5 | Five shell games in a row, all correct, each reset in under 1 minute | TBD |
| E3 | Offline rehearsal | Same demo with the network unplugged: Piper voice, every answer still correct | TBD |
| E4 | Devpost | Submitted before Sun 8 AM, with the demo GIF, real-trial numbers and open-source credits | TBD |
| E5 | Morning of expo | `python demo_check.py` all green, kill switch checked, event DB fresh | TBD |

## Roadmap (after the event, or if time allows)

- YOLO11 fine-tune on our own overhead frames (`scripts/finetune/`), with hard negatives (Jetson case, clicker, cables).
- Laser `trace` (replay an object's history), `tour` (several objects in order), `find_new` (objects outside the 8 classes, via spec 0003).
- Detector throttling while Qwen generates, if F2 shows GPU contention.
- Streaming action-first answers.
- Qwen3-4B-Instruct-2507, only if `tegrastats` shows headroom.
- Lamp-head enclosure. Floor search camera and room map (config `floor_zones`, `room_map`); room calibration with laser-guided tag placement is specced in `docs/specs/0005-room-calibration.md`.
