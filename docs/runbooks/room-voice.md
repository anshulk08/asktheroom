# Room voice bring-up (spec 0010 P0-1)

The rig answers spoken questions in the room: "where's my wallet?" said from about 2 m away gets a spoken
answer within 4 s. This page takes a teammate from a USB mic and speaker in a bag to that test passing, then
sets up the phone as the safety net. It takes about 45 minutes. Every step says what "good" looks like and
what to change if it isn't.

Commands marked **(container)** run inside the room app's container (`docker exec -it <container> bash`,
in the app's checkout); **(host)** runs on the Jetson host; **(laptop)** runs on a laptop with this repo.
Settings go in `config.local.yaml` of the checkout the room app runs from (gitignored; see
`config.local.yaml.example`, "The room demo").

## 0. Before you start

- The room app holds the camera. It must be running **with voice on** for steps 5-8 (no `--no-voice`), and
  **stopped** for steps 2-3 if the mic is an ALSA `hw:` device (only one program can record from it). Ask
  the rig owner before stopping or restarting it.
- You need: a USB mic (or USB speakerphone) and a USB speaker, a laptop 2 m away with this repo, the
  demo object (the wallet) on a known zone (say the kitchen counter).

## 1. Place the hardware

- **Mic:** chest height, near where the judge will stand, pointing at them, 1-2 m away. Not in the ceiling
  corner: the Brio's mic up there does not hear a judge over expo noise.
- **Speaker:** near the judge too, but **at least 1 m from the mic and facing away from it**. The rig drops
  any recording its own voice starts in, but a speaker pointed at the mic still makes the echo check (step 6)
  harder.
- Plug both into the Jetson directly (not a hub shared with the Brio if you can avoid it).

## 2. Find the devices (container)

```bash
python -m voice.stt --devices        # inputs: * = what stt.input_device picks, d = the default
python -m voice.tts --devices        # outputs: the same for tts.output_device
```

Pick a **part of each name that only that device has**. The Brio, a USB mic and a USB speaker often all say
"USB Audio"; use the product part instead ("PnP Sound", "Jabra", "UACDemo"). If several match, the log
warns and names them all, and the first is used. Then in `config.local.yaml`:

```yaml
stt:
  input_device: "PnP Sound"          # part of the mic's name (or its index, but indexes change on replug)
tts:
  output_device: "UACDemo"           # part of the speaker's name
listen:
  wake_words: [ask the room, askroom, ask room]   # not "room": the room demo says it constantly
demo:
  hold_notices: true                 # reminders never talk over a judge
```

A mic that only records at 44.1 or 48 kHz is fine: the rig records at its rate and resamples (the log says
"input device refuses 16000 Hz; recording at its 48000 Hz, resampled").

## 3. Speaker check (container)

```bash
python -m voice.tts --say "Your wallet, I think, is on the kitchen counter."
```

Good: you hear it from the USB speaker, not the HDMI screen, at a level a judge hears at 2 m over the hall.
Set the volume with `alsamixer` (host; F6 picks the card) or `amixer -c <card> set PCM 90%`. If the log says
"speech output hung", the device took no audio: check the name and replug; the rig gives up on a hung
speaker after the answer's length plus 3 s, so it never goes deaf, but it says nothing either.

## 4. The 2 m mic check (container)

```bash
python -m voice.stt --level 30       # level and speech probability every half second; nothing is kept
```

1. Say nothing for 10 s with the room as noisy as the expo will be. Good: `speech` stays **below 0.5**
   (`stt.vad_threshold`) and no line says `SPEECH`.
2. Stand where the judge will, 2 m away, and ask "where's my wallet" at a normal voice, 5 times. Good: each
   question shows `SPEECH` lines, with the level at least **15 dB above** the quiet lines.

If speech doesn't reach `SPEECH`: move the mic closer or aim it better, raise its gain (`alsamixer`, capture
view, F4), and only then lower `stt.vad_threshold` (0.4, then 0.35). If the quiet room already shows `SPEECH`,
the mic hears the hall too well: turn the gain down or move the mic away from the noise.

Question length: the rig ends a question after `stt.silence_ms` (700) of quiet and never records past
`stt.max_s` (6). In a loud hall `stt.end_drop_db` (10) also ends it when the level falls back to the noise.
If questions get cut off mid-sentence, raise `silence_ms` to 900; if the rig waits long after the question,
lower `end_drop_db` to 8.

## 5. First questions (the app, voice on)

Restart the room app with voice on and the new `config.local.yaml` (the rig owner's call). Ask from 2 m:

- "where's my wallet?" → "Your wallet, I think, is on the kitchen counter..." and nothing else.
- "is it on the counter?" (right after) → the same wallet answer: follow-ups work for 2 minutes after a
  question.
- "where's the couch?" said to someone else → nothing (a place, not a thing).
- "let's reset the room" → nothing. "ask the room, reset" → the reset answer. Overheard reset only counts
  when the wake phrase opens the sentence; the phone and the dashboard can always reset.

The log line `heard: '...'` shows what Whisper wrote; the rig logs only what it answers, never chatter.

## 6. Echo check

Ask 5 questions with the speaker at its demo volume. Good: after each answer the rig is silent until you
speak again. It must never answer itself (a second `heard:` line with its own words).

If it does: first turn the speaker away from the mic or move them apart. Then raise `listen.echo_tail_s`
from 0.4 to 0.8 (a room-sized speaker rings longer than a desk one), and 1.2 at most: the mic stays shut that
long after each answer, so a judge's quick follow-up is lost if it is too long.

## 7. Hall noise, 10 minutes (host, then container)

Record 10 minutes of the room with people talking (the expo, or a recording of one played at expo volume),
**with no questions for the rig**, from the demo mic:

```bash
arecord -l                                        # (host) the mic's card number
arecord -D plughw:<card>,0 -f S16_LE -r 16000 -c 1 -d 600 hall.wav
python scripts/overheard_test.py hall.wav          # (container, in the checkout; keep hall.wav out of git)
```

Good: **0-1 false triggers** in 10 minutes. More than 1: run the demo in wake mode (`listen.mode: wake`, and
judges say "ask the room, where's my wallet"), and rerun with `--mode wake` to confirm.

## 8. The done test (laptop)

Put the wallet on the kitchen counter and let the rig see it arrive. From the laptop, 2 m from the mic with
its speaker at conversation volume:

```bash
python scripts/voice_trials.py --url http://<rig-ip>:8000 --expect counter
```

It says "where's my wallet" 5 times, 8 s apart, and reads the rig's answers from `/state`. Good: **5/5 PASS**
with each latency **under 4 s** (end of the question to the rig having its answer; the first sound follows
~0.3 s later). Then move the wallet to the couch and run it with `--expect couch`. Save the results with
`--out data/voice_trials_<zone>.json` for the scoreboard.

A FAIL with "no answer" is the mic (step 4) or the overheard filter (the log's `heard:` line is missing);
a FAIL with the wrong place is room memory, not voice (the rig owner's `scripts/room_trials.py`); a slow
PASS/FAIL over 4 s: check `online` in `/state` (Grok being slow on Wi-Fi: the rules answer after at most 10 s,
and a "Let me look." cue covers the wait).

## 9. Phone safety net (host)

The iPhone app talks to the BLE bridge, a host process that calls the room app's HTTP API on port 8000,
so it works with whichever build serves that port.

```bash
systemctl status askroom-ble        # installed as a service? its --repo points at ~/askroom
mobile/bridge/run_bridge.sh status  # or run by hand, from the room app's checkout:
cd ~/askroom_room && mobile/bridge/run_bridge.sh start
```

Run **one** bridge, not both. Started from the room checkout, it reads that checkout's config (table size,
calibration). Check from the laptop with `.venv/bin/python mobile/bridge/test_client.py -q "where's my
wallet?"`, or on the phone: the answer card shows "...on the kitchen counter" and the phone reads it aloud.
Objects in a room zone arrive in the state with `z` (the zone) and no table position (`mobile/PROTOCOL.md`).

If the rig's voice fails during judging, hand the judge the phone. If both fail: the dashboard's `/ask` box.

## 10. Offline rehearsal

With Wi-Fi off, repeat step 8 once. Good: the same answers (room places and where-is questions need no
network), in Piper's voice. Misheard names still work offline ("wears my wall it" is the wallet). Open
questions answer "I'm offline". If there is no voice at all offline, Piper's voice file is missing
(`scripts/get_piper_voice.sh`); the log says so, and `espeak-ng`, if installed, speaks meanwhile.
