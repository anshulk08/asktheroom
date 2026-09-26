# Feature Status

Status meanings:

- **validated**: backed by passing tests or a real measurement (the evidence column says which). Laptop measurements are labelled as laptop measurements.
- **implemented**: the code exists and runs, but hasn't been tested or measured on the real rig.
- **planned**: a spec or roadmap item with no code yet.

Test suite at time of writing (Sat 26 Sep, after the overnight merge): `.venv/bin/python -m pytest -q` gives 1432 passed, 25 skipped on the laptop (branch `grok-settle-check`) (the skips need optional models or hardware). Branch `thing-identity` (spec 0008): 1574 passed, 27 skipped. Jetson measurements are labelled "Jetson". See PLANS.md checkpoints F1–F6.

## Perception and world model

| Spec | Feature | Status | Evidence |
|---|---|---|---|
| — | Camera capture, newest frame + ring (`core/capture.py`) | validated | `tests/test_capture.py`; 29.9 fps live on the Jetson (MJPG 1280x720, manual exposure) |
| — | ArUco table frame, px→cm homography (`core/table.py`) | validated | `tests/test_table.py` (synthetic frames) |
| — | One-tag table calibration: one printed AprilTag 36h11 anywhere on the table, no measuring; tracked area = what the camera sees, axes follow the image (`core/table.py`, `table_tag:` in config, on by default; `scripts/make_markers.py --tag`) | validated (synthetic) | `tests/test_table.py`: distances across the view within 1.5 cm on noisy synthetic frames, same area whether the tag lies straight or at 40°; `tests/test_demo_check.py` drift check. Real-table accuracy not measured yet |
| — | YOLO-World detector, TensorRT on the Jetson (`core/detect.py`) | implemented | `tests/test_detect.py` covers the post-processing only. There is no accuracy measurement on our objects yet. Known-object boxes that sit on a hand box (IoU >= 0.6: a hand YOLO-World also called 'phone'), are too big for any prop, or are off the table are dropped (`detect_filter:`); unit-tested, not measured on the rig |
| — | YOLO11 fine-tune on overhead frames (`scripts/finetune/`) | implemented | `tests/test_finetune.py` covers the tooling. No trained model yet |
| — | Hand tracking via detector `hand` class (`core/hands.py`) | validated | `tests/test_hands.py` |
| — | World rules: HELD, UNDER, INSIDE, GONE, UNKNOWN, parent chains, decay (`core/world.py`, `core/relations.py`) | validated | `tests/test_world_rules.py`, `tests/test_world_core.py`, `tests/test_relations.py` (synthetic detections) |
| — | Briefly unseen is not lost: an absence no rule explains waits `lost_grace_s` (2 s) before `LOST_TRACK`; seen again in place meanwhile logs nothing (`core/world.py` rule 4) | validated (synthetic + replay) | `tests/test_world_rules.py`. shell_1 replay (model 1): wallet and phone `LOST_TRACK`/`CORRECTED` events 13 → 2, identity changes 1 → 0. A removed object is lost ~1.1 s later than before |
| — | Shell game (keys → notebook → box → box moved) | validated | Synthetic: world-rule tests and `eval.synth`. Real-table trials not yet recorded (F5) |
| — | Things as containers: a large unnamed `thing:N` (tub, bag, hat, mug; footprint ≥ `thing_containers.min_area_cm2`, 100 cm²) holds what a hand leaves in it, by the box's dwell rule; contents follow it when it is carried or slid, `TAKEN_OUT` when seen beside it; the configured box wins when the hand was in both; a hand that crossed the thing was carrying past it; answers say "inside a container" / its name or guess (`core/things.py`, `core/world.py` rules 7-8, `voice/answers.py`) | validated (synthetic) | `tests/test_thing_containers.py`. Jetson scoring of the five recorded clips (model 1): every metric unchanged. shell_1 cannot show it: the AirPods case lies in plain view in the open tub, so it is never absent after the drop (and hands are detected in one frame of it) |
| — | Event log + snapshots, 24 h snapshot pruning (`core/events.py`) | validated | `tests/test_events.py` |
| — | Open-world identity: unnamed `thing:N` entities, causal-first re-identification, UNKNOWN survives (`maybe_same_as`, no silent merges), exemplar bank (`core/things.py`) | validated (synthetic) | `tests/test_openworld.py`. The real-table check (D17: teach, hide, move the box, ask; at least 4/5) is not done yet |
| 0008 | One object, one `thing:N`: a thing lost within `rebirth_s` (60 s) and seen again, of its size, within `rebirth_cm` (5 cm) of where it was last seen or picked up from is itself (`thing_identity:`, `core/things.py`); one object per kind when Grok names a new thing like a lost one (`grok_check.merge_same`, `bind_conf` 0.9); Grok label belief, a summed tally of top-3 guesses and `not_object` that names or retires clutter (`grok_check.belief_enabled`); phone gets `g`, `gc`, `as` | implemented | Rebirth on by default in `config.yaml`; merge needs the settle check (off by default); belief off by default. `tests/test_thing_identity.py`. Live on the Jetson for ~6 min (spec 0008): 7 clutter things retired and 2 named in 3.5 min; births at the bottom edge (an arm) not yet fewer. Before: 13 min, 39 births, 31 within 5 cm of an earlier thing |
| — | Teaching by voice: "this is my charger" names the thing just put down (`voice/teach.py`) | validated (unit) | `tests/test_teach.py` |
| — | Object proposals: change-detection proposer and YOLOE-26s prompt-free adapter, reduced-head fast path (`core/proposals.py`, `core/yoloe_fast.py`) | implemented | `tests/test_proposals.py`, `tests/test_yoloe_fast.py`. Measured on desk photos only, not on the rig. Hands and arms the detector misses: moving regions are rejected (`proposals.change` `still_frames` / `moving_frac`), and a new thing is confirmed only after it holds still (`openworld` `still_cm` 2, `still_s` 0.5); synthetic tests only |
| — | Tabletop outline and person boxes: an operator-set outline in table cm (`table_area:`, `python -m core.table --outline` / `--outline-px`, saved to `table_area.json` and ignored once the table is recalibrated); proposals only inside it, new things only clear of its `edge_cm` band while existing ones are still followed there; YOLOE boxes inside a person box are flagged `occluded` (never a new thing, still matched to an existing one) instead of dropped (`core/table_area.py`, `core/proposals.py`, `core/things.py`) | validated (synthetic) | `tests/test_table_area.py`, `tests/test_proposals.py`, `tests/test_openworld.py`. Not yet set or checked on the rig |
| — | Re-identification embedder, DINOv2-S/14 in TensorRT (`core/embed.py`) | implemented, off by default | `tests/test_embed.py`; 3.7–4.7 ms per crop on the Jetson. Same-object cosine mean 0.72 vs different 0.21 on proxy crops; the `openworld` thresholds must be retuned for it before `reid.enabled` goes on |
| — | Close-up crop store for visual questions (`core/crops.py`) | validated (unit) | `tests/test_crops.py` |
| — | Model-free fine-tune labelling: background difference + copy-paste synthesis (`scripts/finetune/bglabel.py`, `synthesize.py`), with distractor negatives (`capture.py --distractors`: unknown things labelled empty, pasted unboxed) | validated (unit) | `tests/test_bglabel.py`, `tests/test_synthesize.py`. First capture session: the model called unseen things `phone` at 0.53-0.72, hence distractors; not yet captured or retrained with them |

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
| — | "Recalibrate" in one-tag mode: feeds fresh frames until the tag fit completes (about 2 s), warns if the table frame moved over 2 cm under a fitted laser; startup warns when `laser_cal.json` is older than `table_cal.json` (`main.py`). A spoken recalibrate that fails says so, and one that measures a new tracked area asks for a restart (the world, laser and detector read the size at startup); the BLE bridge follows the new size on its own | validated (unit) | `tests/test_main.py` (`test_recalibrate_*`, `test_a_*recalibrate*`, `test_laser_fitted_before_the_table_calibration_is_flagged`), `tests/test_mobile_protocol.py` (`test_the_bridge_follows_a_recalibrated_table_size`) |
| — | Pre-judge checklist (`demo_check.py`) | validated (fake rig) | `tests/test_demo_check.py`. Not run on the real rig yet |
| — | Clock check: `main.py` warns at startup and `demo_check.py` check 9 fails when the wall clock is behind the last saved file (event DB, calibrations, config) or before Sep 25; a Jetson offline with no RTC battery boots stale and every spoken time and n8n timestamp is off (`net.clock_behind`) | validated (unit) | `tests/test_net.py`, `tests/test_demo_check.py` |
| — | Whole program without hardware (`main.py --fake`, `server.sim`) | validated | `tests/test_main.py::test_build_fake_runs_without_hardware`, `tests/test_sim.py` |

## Evaluation

| Spec | Feature | Status | Evidence |
|---|---|---|---|
| — | Trial format, record, replay vs 3 baselines, report (`eval/`) | validated | `tests/test_eval.py` |
| — | Synthetic trials (`eval.synth`) | validated | `tests/test_synth.py`. The synthetic score is a regression check, never a claim of real accuracy |
| — | Guided-clip replay and scorer (`eval.score_clip`): replays a recorded clip through `Room.perceive` and the ask pipeline (answers clocked on clip time), maps props to entities by position (an unrelated object the detector merely found again in place is no placement; undeclared configured objects are never guessed for a prop), PASS/FAIL per metric | implemented | `tests/test_score_clip.py` (synthetic clips, stub detector); runs end to end on all five `data/clips` on the Jetson with the TensorRT detector |
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
| — | Automatic names for new things: one close-up of each new `thing:N` goes to Grok once, in the background (online only, `max_per_minute`, one retry), for a soft guess ("deodorant stick"); "where's my deodorant?" falls back to it after configured names and taught aliases, hedged ("Your deodorant, I think, is under the notebook."); a taught name always wins and a taught thing is never sent (`core/auto_name.py`, `voice/answers.py`) | validated (unit + laptop) | `tests/test_auto_name.py` (fake Grok: appear, hide, ask; taught wins; retry; offline; rate limit). `scripts/auto_name_smoke.py`: real grok-4.3 named 5/5 crops of a desk photo sensibly, median 1.15 s (laptop). Not yet on rig frames. On in `config.yaml` (`auto_name.enabled`) |
| — | Qwen interpreter and answerer replaced by Grok | validated (laptop) | `tests/test_understand.py`, `tests/test_pipeline.py`; eval numbers above. Rules and templates stay first; offline, rules and the fallback sentence |

## Phone app

| Spec | Feature | Status | Evidence |
|---|---|---|---|
| — | Bluetooth LE bridge on the Jetson, chunked JSON protocol (`mobile/bridge/`, `mobile/PROTOCOL.md`) | implemented | `tests/test_mobile_protocol.py`. Needs `sudo hcitool` advertising fix on the Jetson; not yet tested with an iPhone |
| — | Native iPhone app (SwiftUI + CoreBluetooth) (`mobile/ios/`) | implemented | Xcode project from the mobile teammate; framing and wire models reviewed against `mobile/PROTOCOL.md`. Ask timeouts nest (server 10 s < bridge 12 s < app 15 s); a rig answer later than 10 s is not spoken or aimed (`tests/test_main.py`). Not yet tested against the Jetson |
| — | Phone: reminders and the morning report shown on Home as the rig fires them; answers to questions asked out loud, on the dashboard or by SMS shown when a helper turns it on (`id: null` answers, `mobile/PROTOCOL.md` 6a) | implemented (branch `mobile-app`) | `RoomStoreTests`, `ModelsTests` (the 6a examples); simulator screenshot. Not yet tested against the Jetson |
| — | Phone: Bluetooth receive path: reads `status` before subscribing (MTU first), reconnects at once after a drop, reconnects a link silent for 15 s | implemented (branch `mobile-app`) | Simulator has no Bluetooth; needs an iPhone next to the rig |
| — | Phone: optional read-aloud in a Grok voice (28 voices, speed, matched to the output: iPhone speaker, headphones, headset, AirPlay), the rig's ElevenLabs voice, or the iPhone voice (fallback) | implemented (branch `mobile-app`) | `SpeakerTests`; one real xAI TTS call (0.6 s, 24 kHz MP3). Playback not yet heard on an iPhone |
| — | Phone: Find My-style map pins with each thing's picture (emoji, SF Symbol, or a helper's own emoji or Genmoji) | implemented (branch `mobile-app`) | `ThingIconTests`, `MapLayoutTests` (no overlaps on an iPhone-width map); simulator screenshots. Genmoji needs an Apple Intelligence iPhone |
