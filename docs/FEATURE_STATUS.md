# Feature Status

Status meanings:

- **validated**: backed by passing tests or a real measurement (the evidence column says which). Laptop measurements are labelled as laptop measurements.
- **implemented**: the code exists and runs, but hasn't been tested or measured on the real rig.
- **planned**: a spec or roadmap item with no code yet.

Test suite at time of writing: `.venv/bin/python -m pytest -q` gives 608 passed, 20 skipped (the skips need optional models or hardware). Nothing has been measured on the Jetson yet. See PLANS.md checkpoints F1–F6.

## Perception and world model

| Spec | Feature | Status | Evidence |
|---|---|---|---|
| — | Camera capture, newest frame + ring (`core/capture.py`) | validated | `tests/test_capture.py` |
| — | ArUco table frame, px→cm homography (`core/table.py`) | validated | `tests/test_table.py` (synthetic frames) |
| — | YOLO-World detector, TensorRT on the Jetson (`core/detect.py`) | implemented | `tests/test_detect.py` covers the post-processing only. There is no accuracy measurement on our objects yet |
| — | YOLO11 fine-tune on overhead frames (`scripts/finetune/`) | implemented | `tests/test_finetune.py` covers the tooling. No trained model yet |
| — | Hand tracking via detector `hand` class (`core/hands.py`) | validated | `tests/test_hands.py` |
| — | World rules: HELD, UNDER, INSIDE, GONE, UNKNOWN, parent chains, decay (`core/world.py`, `core/relations.py`) | validated | `tests/test_world_rules.py`, `tests/test_world_core.py`, `tests/test_relations.py` (synthetic detections) |
| — | Shell game (keys → notebook → box → box moved) | validated | Synthetic: world-rule tests and `eval.synth`. Real-table trials not yet recorded (F5) |
| — | Event log + snapshots, 24 h snapshot pruning (`core/events.py`) | validated | `tests/test_events.py` |

## Voice

| Spec | Feature | Status | Evidence |
|---|---|---|---|
| — | Silero VAD + whisper.cpp base.en (`voice/stt.py`) | validated | `tests/test_stt.py`; 20/20 of the recorded test questions parsed correctly (`stt20` set, laptop) |
| — | Rule parser (`voice/intents.py`) | validated | `tests/test_intents.py`; rules alone 48/64 on `tests/understand_eval.json` |
| 0001 | Local interpreter: rules then Qwen3-1.7B, schema, `sounds_like` guard (`voice/understand.py`) | validated (laptop) | `tests/test_understand.py`; `scripts/eval_understand.py` 58/64, median 114 ms on the laptop. Not measured on the Jetson (F1, F2) |
| 0001 | Local answerer for open questions: templates, then one-shot Qwen, action-first (`voice/local_llm.py`) | validated (unit) | `tests/test_local_llm.py`. There is no accuracy eval of free-form answers |
| 0001 | Template answers for WHERE/HISTORY/HANDLED/CHANGES, confidence wording (`voice/answers.py`) | validated | `tests/test_answers.py` |
| 0001 | Pill-wording filter (`voice/llm.to_answer`) | validated | `tests/test_local_llm.py`, `tests/test_answers.py` |
| 0002 | Always listening, overheard filter, wake and click modes (`main.py`, `voice/understand.py`) | validated (unit) | `tests/test_main.py` (answers questions and drops chatter, waits while speaking, clicker is listen-now); overheard set 16/16 on the laptop. The 10-minute hall-noise run hasn't been done yet (F3) |
| 0002 | Echo guard: mic shut while speaking + `echo_tail_s` | validated (unit) | `tests/test_main.py::test_always_listening_waits_while_the_rig_speaks`. Not tested with a real speaker and mic |
| — | ElevenLabs streaming TTS with Piper fallback (`voice/tts.py`) | validated (unit) | `tests/test_tts.py` (fallback on no network and slow first byte) |
| — | whisper-cli backend on the Jetson (`stt.backend: cli`) | implemented | Not run on the Jetson yet (F4) |

## Laser

| Spec | Feature | Status | Evidence |
|---|---|---|---|
| — | Polynomial calibration + closed-loop aim (`act/laser.py`, `act/calibrate.py`) | validated (sim) | `tests/test_laser_fit.py`, `tests/test_laser_sim.py` (non-linear head with backlash and latency). Not calibrated on the rig yet (F6) |
| — | Circle and edge sweep actions | validated (sim) | `tests/test_laser_sim.py`, `tests/test_main.py::test_sweep_and_circle_actions` |
| — | Actuators: serial, PCA9685, sim; clamp, easing, laser timeout (`act/actuator.py`) | validated (unit) | `tests/test_actuator.py` |
| — | Hardware kill switch | implemented | `demo_check.py` check 8 prompts for a manual check |
| — | Laser `trace`, `tour`, `find_new` | planned | Roadmap in PLANS.md |

## Interfaces and ops

| Spec | Feature | Status | Evidence |
|---|---|---|---|
| — | Dashboard, `/ask`, `/state`, `/events`, `/ws`, video (`server/app.py`) | validated | `tests/test_server.py` |
| — | SMS via Twilio with signature check + whitelist (`/sms`) | validated (unit) | `tests/test_server.py`. Not tested against live Twilio |
| — | n8n question log report per answered question | validated (unit) | `tests/test_main.py::test_voice_loop_reports_each_question_to_n8n`. The workflow itself is in `n8n/` (separate workstream) |
| — | Network monitor, never blocks (`net.py`) | validated | `tests/test_net.py` |
| — | Pre-judge checklist (`demo_check.py`) | validated (fake rig) | `tests/test_demo_check.py`. Not run on the real rig yet |
| — | Whole program without hardware (`main.py --fake`, `server.sim`) | validated | `tests/test_main.py::test_build_fake_runs_without_hardware`, `tests/test_sim.py` |

## Evaluation

| Spec | Feature | Status | Evidence |
|---|---|---|---|
| — | Trial format, record, replay vs 3 baselines, report (`eval/`) | validated | `tests/test_eval.py` |
| — | Synthetic trials (`eval.synth`) | validated | `tests/test_synth.py`. The synthetic score is a regression check, never a claim of real accuracy |
| 0004 | Real-trial accuracy numbers | planned | No real trials recorded yet (F5) |
| 0004 | Live scoreboard on the dashboard | planned | Spec 0004 |

## Detection-side Grok

| Spec | Feature | Status | Evidence |
|---|---|---|---|
| 0003 | Measure Grok box error vs ArUco on 20 frames | planned | Spec 0003. Needs team OK for xAI spend |
| 0003 | (a) Grok auto-labelling for the YOLO11 fine-tune | planned | Depends on the measurement |
| 0003 | (b) Second opinion on low-confidence frames, (c) `find_new` | planned | Depends on the measurement; lower priority |
