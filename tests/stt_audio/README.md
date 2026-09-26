Recorded clips for tests/test_stt.py, one per line of tests/stt_questions.json: `01.wav` ... `20.wav`
(16 kHz mono). Record them with `python -m voice.stt --record-questions tests/stt_audio`.

`ASKROOM_STT_AUDIO=1 python -m pytest -q tests/test_stt.py` runs them through Whisper and the intent
parser. Missing clips are synthesized with macOS `say` when it is available.
