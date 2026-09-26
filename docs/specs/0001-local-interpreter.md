# 0001: Local command interpreter and answerer

Status: implemented; validated on the laptop. Jetson numbers pending (PLANS F1, F2).

## Problem

Spoken questions used to fall through to Grok when the rule parser missed. That needed the network, sent what visitors said off the device, and added a second or more of latency. The expo hall Wi-Fi is not reliable.

## Decision

Understanding and answering run on the Jetson. Grok is not on the voice path.

1. **Rules first.** `voice/intents.py` parses the transcript. If it found both the question type and the object that type needs (`rules_sure`), that result is used and no model runs.
2. **Qwen when the rules aren't sure.** `voice/understand.Understander` calls llama-server (`scripts/qwen_server.sh`, port 8081) running Qwen3-1.7B Q4_K_M:
   - It uses the `response_format` `json_schema` `{kind, object}`, with enums built from the intent kinds and the config object names.
   - `chat_template_kwargs.enable_thinking: false` (llama.cpp skips grammar enforcement while thinking), `temperature: 0`, `max_tokens: 40`, and a static system prompt so `cache_prompt` reuses it.
   - Guard: Qwen's object must `sounds_like` a word actually said (difflib ratio at least 0.6, or a synonym). An object the rules recognised always wins.
   - RESET and RECAL come from the rules only.
   - If Qwen times out (`understand.timeout_s`, 1.5 s), is unreachable or returns bad output, the rules' result stands.
3. **Templates answer.** WHERE, HISTORY, HANDLED and CHANGES go to `voice/answers.py`.
4. **Open questions (OTHER)** go to `voice/local_llm.ask_local`:
   - First, templates for: what is in or under X (circle on X), what is hidden, what is on the table, meds/pills (always `PILLS_SAFE`), privacy, help.
   - Otherwise, one Qwen call. The prompt holds `compact_state(world)` and the last 12 events from the past 30 minutes. The schema is action-first: `{action: point|circle|none, point_at: <object>|none, text (max 220 chars)}`. Limits are `max_tokens` 90 and `answer_timeout_s` 4.
   - The output goes through `voice/llm.to_answer`: two sentences at most, no markdown, pill filter.
   - The pointer is dropped if the sentence doesn't name that object.
   - Any failure gives `FALLBACK_TEXT`.
5. **Action set.**
   - **Kept:** `point`, `circle`, `sweep:<edge>` (from templates for GONE) and none.
   - **Deferred:** `trace`, `tour` and `find_new`. They need new laser and world code.
   - `IGNORE` is an intent, not a laser action (spec 0002).

## Model choice

Measured with `python scripts/eval_understand.py` on `tests/understand_eval.json` (64 items), on the laptop:

| | stt20 | loose | overheard | all | median Qwen latency |
|---|---|---|---|---|---|
| rules only | 20/20 | 14/28 | 14/16 | 48/64 | — |
| rules + Qwen2.5-1.5B | not recorded | not recorded | not recorded | 58/64 | 142 ms |
| rules + Qwen3-1.7B | 20/20 | 22/28 | 16/16 | 58/64 | 114 ms |

Both models tie on accuracy. Qwen3-1.7B was picked because it is faster. Qwen3-4B is on the roadmap only if `tegrastats` shows RAM headroom.

## Deferred, with reasons

- **Throttle the detector to about 5 fps while Qwen generates.** Measure GPU contention with `tegrastats` first, with YOLO, whisper and Qwen all loaded (F2).
- **Stream the answer action-first.** Qwen already answers open questions in roughly 0.3–0.5 s on the laptop, and templates are instant. The schema is action-first already, so streaming can be added later without a format change.

## Acceptance tests

| # | Test | Pass | Status |
|---|---|---|---|
| A1 | `.venv/bin/python -m pytest -q tests/test_understand.py tests/test_local_llm.py tests/test_answers.py` | all pass | passing |
| A2 | `python scripts/eval_understand.py` on the laptop | at least 58/64 overall, stt20 20/20, overheard 16/16 | passing (laptop) |
| A3 | Same script on the Jetson, with YOLO and whisper loaded | at least 58/64, median Qwen latency under 300 ms | not run |
| A4 | Qwen server stopped (`understand.url` unreachable) | every stt20 question still answered by rules; OTHER gives the fallback sentence; nothing hangs longer than `timeout_s` | covered by unit tests; to check on the rig |
| A5 | Ask "did I take my meds" | answer is `PILLS_SAFE` and points at the pill bottle; never says "taken" | passing (`tests/test_local_llm.py`) |
| A6 | Network unplugged, full demo | all answers correct; Piper voice | not run (PLANS E3) |
| A7 | `grep -rn "ask_grok(" main.py voice/pipeline.py voice/understand.py voice/local_llm.py` | no matches | passing |
