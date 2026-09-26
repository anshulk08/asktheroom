# 0002: Always listening

Status: implemented; unit-tested. The hall-noise acceptance run on the Jetson is pending (PLANS F3).

## Problem

Needing a clicker press before every question slows the demo, and judges forget to press. But a mic that is always on in a loud expo hall will hear people talking to each other. Answering them is worse than staying silent, and storing what they say would be a privacy problem.

## Decision

`listen.mode` in `config.yaml` sets how the mic listens. It can be overridden with `python main.py --listen always|wake|click`.

| Mode | Behaviour |
|---|---|
| `always` (default) | Every utterance goes through the overheard filter below |
| `wake` | Only speech containing a wake word (`listen.wake_words`, default `[room]`) counts. Use this if the hall test fails |
| `click` | Listen only after a clicker press (the old behaviour) |

In every mode, a clicker press means "listen now": the next utterance is treated as asked, not overheard. A press also stops the current answer (barge-in).

### Pipeline (`main.Room.voice_loop`)

1. While TTS is speaking, the loop doesn't read the mic. After speech ends, it waits `listen.echo_tail_s` (0.4 s) more, so the rig never hears itself.
2. `stt.hear(idle_s)` listens with Silero VAD for up to `listen.idle_s` (8 s), then transcribes with whisper.cpp. The audio is a numpy array in memory only, and `stt.log_text` is off, so transcripts are not logged.
3. `Understander.interpret(text, overheard=True)`:
   - **Keyword gate**: needs an object name or synonym, a command word, or the wake word.
   - **Addressed**: needs the wake word, or a question/request opening (`where`, `did`, `can`, `show`, ...) after fillers.
   - Then the same rules → Qwen path as spec 0001.
   - Extra guards: RESET and RECAL need the wake word. OTHER needs the wake word or an object.
   - Anything that fails becomes `IGNORE`.
   - Qwen is not asked to judge IGNORE. It scored 8/16 on the overheard set, while the rule checks score 14/16 alone and 16/16 with Qwen parsing the survivors.
4. `IGNORE` is dropped. It isn't stored, logged or sent. Only a counter (`ignored_since_last`) goes to n8n with the next real question.
5. Accepted questions are answered like clicker questions. The n8n report says `mode: overheard` and gives `speech_end_to_laser_s`.

### Privacy

- Audio is never written to disk.
- Ignored speech leaves no trace.
- Accepted questions are stored in the `questions` table and reported to n8n, just like clicker questions.
- `scripts/overheard_test.py --show` prints transcripts to the terminal only.

## Risks

- **Echo.** The rig's own voice is blocked by the mute plus the tail. Room reverb longer than 0.4 s could still get through. Raise `echo_tail_s` if the hall test shows self-triggers.
- **Loud hall.** Whisper misreads chatter as object names. The fallback is `wake` mode, a config change only.
- **Latency.** VAD end-of-speech (`stt.silence_ms`, 700 ms) adds to every answer. The clicker path has the same cost.

## Acceptance tests

| # | Test | Pass | Status |
|---|---|---|---|
| B1 | `.venv/bin/python -m pytest -q tests/test_main.py tests/test_understand.py` | all pass, including `test_always_listening_answers_questions_and_drops_chatter`, `test_always_listening_waits_while_the_rig_speaks` and `test_clicker_is_listen_now_in_always_mode` | passing |
| B2 | `python scripts/eval_understand.py`, overheard set | 16/16 (laptop) | passing (laptop) |
| B3 | Record 10 minutes of real hall noise and chatter with no questions for the rig. Run `python scripts/overheard_test.py hall.wav` on the Jetson with Qwen up | 0–1 false triggers. If more, run with `--mode wake`; if that passes, set `listen.mode: wake` | not run |
| B4 | On the rig, ask 10 stt20 questions without the clicker, with people chatting nearby | at least 9 answered correctly, no answer to the chatter | not run |
| B5 | On the rig, let it answer a long question; count self-triggers over 10 answers | 0 | not run |
| B6 | Press the clicker mid-answer | speech stops within 0.3 s and the rig listens | covered by unit test; to check on the rig |
| B7 | After B3–B5, `sqlite3 data/events.db "select count(*) from questions"` | only the accepted questions; no ignored text anywhere on disk | not run |
