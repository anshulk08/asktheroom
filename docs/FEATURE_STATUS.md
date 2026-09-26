# Feature Status

Status meanings:

- **validated**: backed by passing tests or a real measurement (the evidence column says which). Laptop measurements are labelled as laptop measurements.
- **implemented**: the code exists and runs, but hasn't been tested or measured on the real rig.
- **planned**: a spec or roadmap item with no code yet.

Test suite at time of writing (Sat 26 Sep, after the overnight merge): `.venv/bin/python -m pytest -q` gives 1370 passed, 21 skipped on the laptop (the skips need optional models or hardware). Jetson measurements are labelled "Jetson". See PLANS.md checkpoints F1–F6.

## Perception and world model

| Spec | Feature | Status | Evidence |
|---|---|---|---|
| — | Camera capture, newest frame + ring (`core/capture.py`) | validated | `tests/test_capture.py`; 29.9 fps live on the Jetson (MJPG 1280x720, manual exposure) |
| — | ArUco table frame, px→cm homography (`core/table.py`) | validated | `tests/test_table.py` (synthetic frames) |
| — | One-tag table calibration: one printed AprilTag 36h11 anywhere on the table, no measuring; tracked area = what the camera sees, axes follow the image (`core/table.py`, `table_tag:` in config, on by default; `scripts/make_markers.py --tag`) | validated (synthetic) | `tests/test_table.py`: distances across the view within 1.5 cm on noisy synthetic frames, same area whether the tag lies straight or at 40°; `tests/test_demo_check.py` drift check. Real-table accuracy not measured yet |
| — | YOLO-World detector, TensorRT on the Jetson (`core/detect.py`) | implemented | `tests/test_detect.py` covers the post-processing only. There is no accuracy measurement on our objects yet |
| — | YOLO11 fine-tune on overhead frames (`scripts/finetune/`) | implemented | `tests/test_finetune.py` covers the tooling. No trained model yet |
| — | Hand tracking via detector `hand` class (`core/hands.py`) | validated | `tests/test_hands.py` |
| — | World rules: HELD, UNDER, INSIDE, GONE, UNKNOWN, parent chains, decay (`core/world.py`, `core/relations.py`) | validated | `tests/test_world_rules.py`, `tests/test_world_core.py`, `tests/test_relations.py` (synthetic detections) |
| — | Shell game (keys → notebook → box → box moved) | validated | Synthetic: world-rule tests and `eval.synth`. Real-table trials not yet recorded (F5) |
| — | Event log + snapshots, 24 h snapshot pruning (`core/events.py`) | validated | `tests/test_events.py` |
| — | Open-world identity: unnamed `thing:N` entities, causal-first re-identification, UNKNOWN survives (`maybe_same_as`, no silent merges), exemplar bank (`core/things.py`) | validated (synthetic) | `tests/test_openworld.py`. The real-table check (D17: teach, hide, move the box, ask; at least 4/5) is not done yet |
| — | Teaching by voice: "this is my charger" names the thing just put down (`voice/teach.py`) | validated (unit) | `tests/test_teach.py` |
| — | Object proposals: change-detection proposer and YOLOE-26s prompt-free adapter, reduced-head fast path (`core/proposals.py`, `core/yoloe_fast.py`) | implemented | `tests/test_proposals.py`, `tests/test_yoloe_fast.py`. Measured on desk photos only, not on the rig |
| — | Re-identification embedder, DINOv2-S/14 in TensorRT (`core/embed.py`) | implemented, off by default | `tests/test_embed.py`; 3.7–4.7 ms per crop on the Jetson. Same-object cosine mean 0.72 vs different 0.21 on proxy crops; the `openworld` thresholds must be retuned for it before `reid.enabled` goes on |
| — | Close-up crop store for visual questions (`core/crops.py`) | validated (unit) | `tests/test_crops.py` |
| — | Model-free fine-tune labelling: background difference + copy-paste synthesis (`scripts/finetune/bglabel.py`, `synthesize.py`) | validated (unit) | `tests/test_bglabel.py`, `tests/test_synthesize.py`. No real capture session yet |

## Voice

| Spec | Feature | Status | Evidence |
|---|---|---|---|
| — | Silero VAD + whisper.cpp base.en (`voice/stt.py`) | validated | `tests/test_stt.py`; 20/20 of the recorded test questions parsed correctly (`stt20` set, laptop) |
| — | Rule parser (`voice/intents.py`) | validated | `tests/test_intents.py`; rules alone 48/64 on `tests/understand_eval.json` |
| 0001 | Interpreter: rules, then Grok (`understand.backend: grok`, the default; optional local Qwen3-1.7B via `qwen` / `auto`, not installed on the Jetson), schema, `sounds_like` guard (`voice/understand.py`) | validated (laptop) | Grok: `scripts/eval_understand.py` 58/64, median 853 ms (Sat, laptop on campus Wi-Fi). Qwen: `tests/test_understand.py`; `scripts/eval_understand.py` 58/64, median 114 ms on the laptop. Not measured on the Jetson (F1, F2) |
| 0001 | Open questions: templates, then Grok with the world state and lookup tools (`voice/llm.ask_other`); local Qwen answerer kept as an option (`voice/local_llm.py`) | validated (unit) | `tests/test_pipeline.py`; real Grok on the demo world 1.3–3.2 s, grounded answers. `tests/test_local_llm.py`. There is no accuracy eval of free-form answers |
| 0001 | Template answers for WHERE/HISTORY/HANDLED/CHANGES, confidence wording (`voice/answers.py`) | validated | `tests/test_answers.py` |
| 0001 | Pill-wording filter (`voice/llm.to_answer`): any sentence saying medication was taken, missed, skipped or had (the wider `narration_store.med_claim` rule); "pill bottle" is the object | validated | `tests/test_local_llm.py`, `tests/test_answers.py` |
| 0002 | Always listening, overheard filter, wake and click modes (`main.py`, `voice/understand.py`) | validated (unit) | `tests/test_main.py` (answers questions and drops chatter, waits while speaking, clicker is listen-now); overheard set 16/16 on the laptop. The 10-minute hall-noise run hasn't been done yet (F3) |
| 0002 | Echo guard: mic shut while speaking + `echo_tail_s` | validated (unit) | `tests/test_main.py::test_always_listening_waits_while_the_rig_speaks`. Not tested with a real speaker and mic |
| — | ElevenLabs streaming TTS with Piper fallback (`voice/tts.py`) | validated (unit) | `tests/test_tts.py` (fallback on no network and slow first byte) |
| — | whisper.cpp on the Jetson GPU (`stt.backend: server`, `whisper-server` base.en, `audio_ctx 0`) | validated (Jetson) | 22/22 test questions right, median 139 ms on the Jetson (`scripts/build_whisper.sh`, `third_party/stt_bench.py`) |
| — | Conversation memory: follow-ups like "and my wallet?" (`voice/conversation.py`) | validated (unit) | `tests/test_conversation.py` |
| — | Reminders from events ("remind me if I haven't picked up my pill bottle by 9"), neutral pill wording (`core/reminders.py`, `voice/care.py`) | validated (unit) | `tests/test_reminders.py`, `tests/test_care.py` |
| — | Morning report and caregiver summary (`core/reports.py`) | validated (unit) | `tests/test_reports.py` |
| — | Thinking cue: a short "Let me look." when a spoken answer takes over `demo.thinking_cue_s` (`main.py`) | validated (unit) | `tests/test_main.py` (`test_slow_answer_gets_a_thinking_cue_first`); not yet timed with real Grok |
| — | Demo hold: `demo.hold_notices` keeps reminders and the morning report silent unless asked (`voice/care.py`) | validated (unit) | `tests/test_care.py` (`test_demo_hold_notices_keeps_the_rig_quiet_unless_asked`) |
| — | Profile facts ("my daughter is Sarah") (`core/profile.py`) | validated (unit) | `tests/test_profile.py` |
| — | TTS output device selection (`tts.output_device`) | validated (unit) | `tests/test_tts_device.py` |

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
| — | "Recalibrate" in one-tag mode: feeds fresh frames until the tag fit completes (about 2 s), warns if the table frame moved over 2 cm under a fitted laser; startup warns when `laser_cal.json` is older than `table_cal.json` (`main.py`) | validated (unit) | `tests/test_main.py` (`test_recalibrate_*`, `test_laser_fitted_before_the_table_calibration_is_flagged`) |
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
| 0003 | Measure Grok box error vs ArUco on 20 frames | partly measured (laptop, desk photos) | grok-4.3 asked for boxes: Gemini `box_2d` 0/5 (mean IoU 0.14), pixel or fraction boxes 2/5, a bare point 4/5. Choosing among numbered candidate boxes (set-of-marks): 5/5. Grok boxes are too loose to auto-label a fine-tune; picking a mark works |
| 0003 | (a) Grok auto-labelling for the YOLO11 fine-tune | planned | Depends on the measurement |
| 0003 | (b) Second opinion on low-confidence frames, (c) `find_new` | planned | Depends on the measurement; lower priority |

## Grok (visual questions and memory)

All LLM/VLM work goes through Grok (grok-4.3 via the xAI API, `XAI_API_KEY`). Every answer passes the pill filter. Offline, rules and templates answer and visual questions say they need the connection.

| Spec | Feature | Status | Evidence |
|---|---|---|---|
| — | Look now, set-of-marks: tracked objects drawn as numbered boxes, Grok picks a mark (the laser follows that entity) or gives a point (`voice/visual.py`) | validated (laptop, desk photos) | `tests/test_visual.py`; real Grok with 23 unnamed marks on two desk photos: 20/23 right (misses: two sugar packets it declined to name, a notepad under a calculator), about 1.0 s median. Not yet on rig frames |
| — | Pick: "where is my X" for a name the rig doesn't know. Grok only says which mark shows X and what it is (`{mark, label, confidence}`); the world model says where. An unnamed thing picked at confidence 0.7 or more takes the name (`world.bind_alias`), so the next ask needs no Grok call; a taught name is never replaced (`VisualQA.pick`) | validated (unit) | `tests/test_visual.py` (`test_pick_*`). Not yet run against real Grok |
| — | Recall: saved keyframes found by MobileCLIP2 text search, then Grok answers with times (`voice/visual.py`, `core/visual_memory.py`, `core/clip_tokenizer.py`) | validated (unit) | `tests/test_visual.py`; one real Grok recall on desk photos, 1.0 s, abstained correctly. On in `config.yaml` since Sat (`visual_memory.enabled`) |
| — | Episode narration: Grok describes what happened in a short clip; "what was I doing this morning?" (`core/narration.py`, `core/narration_store.py`) | validated (unit) | `tests/test_narration.py`, `tests/test_narration_answers.py`; real Grok self-test 15.6 s with reasoning "low" for 4 frames (laptop). Off by default |
| — | Qwen interpreter and answerer replaced by Grok | validated (laptop) | `tests/test_understand.py`, `tests/test_pipeline.py`; eval numbers above. Rules and templates stay first; offline, rules and the fallback sentence |

## Phone app

| Spec | Feature | Status | Evidence |
|---|---|---|---|
| — | Bluetooth LE bridge on the Jetson, chunked JSON protocol (`mobile/bridge/`, `mobile/PROTOCOL.md`) | implemented | `tests/test_mobile_protocol.py`. Needs `sudo hcitool` advertising fix on the Jetson; not yet tested with an iPhone |
| — | Native iPhone app (SwiftUI + CoreBluetooth) (`mobile/ios/`) | implemented | Xcode project from the mobile teammate; framing and wire models reviewed against `mobile/PROTOCOL.md`. Not yet tested against the Jetson |
