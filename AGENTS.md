# Repository Guidelines

Rules for teammates and coding agents working on Ask the Room. Read `CONTEXT.md` first. It is the compact project map and the source of truth for the n8n "ask-the-repo" bot, which reads it live by URL. This file adds the working rules. Both files stay at the repo root.

## Project purpose

An overhead camera on a Jetson tracks eight tabletop objects, including hidden ones (under the notebook, inside the box). People ask out loud where something is, and the rig answers in speech and points a laser at the spot. It is a HackGT 13 entry. Judging is a live demo at the expo, so the demo must be repeatable and reset in under a minute.

## Where things are

| Need | File |
|---|---|
| Project map, data flow, repo map | `CONTEXT.md` |
| Plan, checkpoints, deadlines | `PLANS.md` |
| How the world model, perception, voice and laser work | `TECHNICAL_DESIGN.md` |
| Status of each feature (validated / implemented / planned) | `docs/FEATURE_STATUS.md` |
| One spec per adopted idea, with acceptance tests | `docs/specs/NNNN-name.md` |
| Setup, running, API | `README.md` |
| Interfaces and pass tests (section numbers like 3.5, V6, H4 in code comments) | the team spec linked in `CONTEXT.md` |

## Working rules

- **Python 3.10.** The Jetson runs JetPack 6. Don't use `match` statements or 3.11+ stdlib (`tomllib`, `ExceptionGroup`, `typing.Self` and so on).
- **Config.** Add new config keys in new sections at the end of `config.yaml`. Never rename existing keys. Tune thresholds from recorded replays (`eval.replay`, `scripts/eval_understand.py`, `scripts/overheard_test.py`), never by guessing during a live run.
- **Units.** Positions are table centimetres (origin at ArUco marker 0, x right, y down). A field is in pixels only if its name says so (`box_px`).
- **World readers** (answers, LLM prompts, server, eval) use only `get`, `resolve`, `history` and `state_json` (`WorldAPI` in `core/world.py`).
- **Spoken answers** are 1–2 short sentences with no markdown. Pill-bottle wording stays neutral: never say or imply that medication was "taken". The pill filter in `voice/llm.py` (`to_answer`, `PILLS_SAFE`) runs on every LLM answer.
- **Shared types** in `core/types.py` change only as a team.
- **Owned files.** Don't edit `core/capture.py`, `core/detect.py`, `core/hands.py`, `core/table.py`, `core/world.py`, `core/relations.py` or `core/events.py`. A teammate owns them, so propose changes to them instead.
- **Grok** is used only on the detection side (spec 0003, not built). Nothing on the voice path calls it. `voice/llm.py` keeps `compact_state`, `to_answer` and the pill filter, which the local answerer reuses. `ask_grok` is not used on the voice path.
- **Privacy.** Audio stays in memory only. Overheard speech the rig ignores is never logged or stored. Only accepted questions go to the `questions` table. Keep the privacy statement in `README.md` accurate whenever data handling changes.
- **Jetson.** The Orin Nano has 8 GB of RAM shared with the GPU and runs out of memory easily. Ask before running anything heavy there (engine builds, llama.cpp builds, model downloads, evals with everything loaded). The limit is RAM, not disk. The Jetson is `guru@192.168.55.1` over USB-C. Build TensorRT engines inside the container that runs them (`scripts/dock.sh`).
- **Tests.** Keep them passing: `.venv/bin/python -m pytest -q`. None need hardware. Add tests with new behaviour.
- **Never commit** `tests/stt_audio/*.wav`, anything under `models/`, `.env`, calibration files (`table_cal.json`, `laser_cal.json`) or `data/events.db`.
- **Commits.** Small, one topic each, with conventional prefixes (`feat:`, `fix:`, `docs:`, `chore:`). No AI attribution lines. Dated commits are part of the hackathon record: all code was written after the Friday 8 PM start (the first commit is Fri Sep 25 20:05 EDT).
- **Freeze.** After Sat Sep 26 6 PM EDT: only bug fixes, tuning from replays, and docs. Anything new becomes a spec or roadmap item, not half-built code.
- **Docs in the same change.** When a feature's state changes, update `docs/FEATURE_STATUS.md`, and `CONTEXT.md` if the map or status moved. The n8n bot is only as current as `CONTEXT.md`.

## Commands

```bash
.venv/bin/python -m pytest -q                       # all tests
python main.py --fake                               # whole program without hardware
python -m server.sim                                # dashboard on a synthetic camera
scripts/qwen_server.sh                              # local Qwen on :8081
python scripts/eval_understand.py                   # interpreter accuracy (needs llama-server)
python scripts/overheard_test.py hall.wav           # always-on false triggers
python demo_check.py                                # before every judge
```

## Quality bar

- Specs and docs must let a teammate implement or verify without guessing. Every spec has acceptance tests.
- Mark something "validated" only when tests or real measurements back it. Laptop numbers are labelled as laptop numbers.
- Never quote the synthetic eval (255/300) as real accuracy.
- Prefer small, deterministic, testable changes. Templates beat LLM text wherever a template can answer.
