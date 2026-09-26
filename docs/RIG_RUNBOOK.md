# Rig runbook: the table shell-game demo

For the teammate at the rig. Goal (PLANS.md): a judge plays the shell game (keys under the notebook, notebook
into the box, box slid across), asks where the keys are, and the rig answers correctly with the laser on the
box, every time, even offline. Work top to bottom; each step says what "done" looks like. Write every number
you measure into the log table at the end and send it back.

Conventions:

- The Jetson is `guru@10.90.84.178` (Wi-Fi) or `guru@192.168.55.1` (USB). Every command below runs **on the
  Jetson, in the deploy directory**, unless it says "Mac".
- The deploy directory is `~/askroom_rig`: a clean copy of the integration branch, prepared for you (it has
  `config.local.yaml` from `~/askroom`, and `.env` and `models/` linked to `~/askroom`'s). Calibration files
  (`table_cal.json`, `table_area.json`, `laser_cal.json`) are written there.
- Python runs in the app container: `D=scripts/dock.sh` below, so `$D python3 -m ...`.
- The camera is the Brio, always by its stable path:
  `CAM=/dev/v4l/by-id/usb-046d_Logitech_BRIO_3675F8D2-video-index0` (`video-index2` is its IR node; never
  use a bare index, it moves on replug).
- Only one thing may hold the camera at a time: the app, `demo_check`, `eval.record` or `core.table`.

```bash
cd ~/askroom_rig
D=scripts/dock.sh
CAM=/dev/v4l/by-id/usb-046d_Logitech_BRIO_3675F8D2-video-index0
```

## 1. Hygiene (only after the orchestrator's OK, given at that moment)

These stop other people's processes. Do not run them on your own.

```bash
docker ps --format '{{.Names}}  {{.Image}}  {{.Status}}'      # note what runs
docker top <room container> -o pid,etime,args                   # the room app: main.py --no-voice, from ~/askroom_room
docker stop <room container>                                    # frees the Brio and ~2.8 GB
( cd <baby-tau compose dir> && docker compose stop )            # or: docker stop baby-tau-ollama-1 baby-tau-whisper-1 baby-tau-piper-tts-1 baby-tau-python-1
free -m                                                         # expect > 4.5 GB available
ls -d ~/askroom_*                                               # stale scratch copies; remove only the ones the orchestrator lists
```

- Reconnect the USB cable (Jetson to laptop) so SSH works without Wi-Fi; step 6 needs it:
  `ping -c1 192.168.55.1` from the Mac.
- Clock: the Jetson has no RTC battery. `date` on the Jetson and the Mac must agree within a few seconds.
  If not: `sudo date -s "$(ssh <mac> date -u +%FT%TZ)"` or, online, `sudo systemctl restart systemd-timesyncd`.
- Done when: only what the demo needs is running, `free -m` shows > 4.5 GB available, `date` is right.

## 2. Light, exposure, table frame, outline, then freeze

1. **Even light.** Light the table evenly: no sun patch or hard shadow across it, no glare on the box. Keep
   the same light for every later step; if it changes, redo steps 2 and 3.
2. **Lock the camera** (auto exposure and autofocus off; values measured over this table on Sat 26 Sep):

   ```bash
   scripts/camera_setup.sh 166 $CAM 80 10 160 3200   # exposure 16.6 ms, gain 80, focus 10, zoom 160, 3200 K
   ```

   Too dark or too bright in venue light: change only the gain (3rd value) in steps of 20 and rerun. Run it
   again after any replug (the app also restores the controls itself when the camera drops off USB).
3. **Table frame (one AprilTag).** Lay the printed tag (36h11, id 0) flat near the middle of the table.
   `table_tag.size_cm` in `config.local.yaml` must be the printed black square's side: 16.0 for the printed
   tag, 14.5 for the iPad stand-in. Then:

   ```bash
   $D python3 -m core.table --device $CAM
   ```

   Done when it prints `calibrated: True`, a tracked area, and `saved table_cal.json`. Take the tag away: the
   app loads `table_cal.json` at startup.
4. **Tabletop outline** (objects are only born inside it; an arm or the floor at the frame edge is not):

   ```bash
   $D python3 -m core.table --outline                  # prints these steps
   # grab a frame: with the app running, Mac: curl -o frame.jpg http://10.90.84.178:8000/frame.jpg
   # read the tabletop's corners in px (a little inside the real edge), in order around the table, then:
   $D python3 -m core.table --outline-px X1,Y1 X2,Y2 X3,Y3 X4,Y4 --image frame.jpg
   ```

   Check the drawn `frame_outline.jpg`: the outline hugs the tabletop. Restart the app afterwards.
5. **Freeze.** From here on nothing may move the table frame: not the camera, not the table, and nobody says
   "recalibrate" (it refits and invalidates the outline and the laser fit). Keep a copy to restore:

   ```bash
   mkdir -p calib_frozen && cp -p table_cal.json table_area.json calib_frozen/
   # (after step 3 also: cp -p laser_cal.json calib_frozen/)
   # restore: cp -p calib_frozen/*.json . && restart the app
   ```

## 3. Laser calibration and 10 ruler spots

*Laser section: supplied by the laser/demo_check workstream (asktheroom-6c), pasted here when its branch is
merged.* It covers `act.calibrate --rig` (latency, servo limits, fit, save), aiming at 10 ruler spots with the
error in cm, the kill switch check, and setting `actuator: pca9685` in `config.local.yaml`.

Done when: fit error < 1.5 cm, a point at the table centre lands < 3 cm away, the 10 ruler spots are logged,
the kill switch cuts the dot, and `cp -p laser_cal.json calib_frozen/` is done.

## 4. Real trials: 10 shell games and 10 place/hide

Each trial is one clip recorded with `eval.record`. Record with the app **stopped** (it holds the camera).
The question is asked at the end of the clip; `--truth` is what is true at that moment. Trial ids are
unique: shell games 101-110, place/hide 201-210.

```bash
# shell game: keys under the notebook, notebook into the box, box slid across. Truth: the keys are in the box
$D python3 -m eval.record --camera $CAM --trial-id 101 --category inside_box_moved --object keys --truth box \
    --question "where are my keys" --no-preview           # Enter to stop at the end of the game
# ... 102 to 110 the same, varying the hand, speed and where the box ends up

# place/hide: alternate these, 2-3 each
$D python3 -m eval.record --camera $CAM --trial-id 201 --category covered --object keys --truth notebook --no-preview
$D python3 -m eval.record --camera $CAM --trial-id 202 --category inside --object wallet --truth box --no-preview
$D python3 -m eval.record --camera $CAM --trial-id 203 --category moved --object phone --truth visible --no-preview
$D python3 -m eval.record --camera $CAM --trial-id 204 --category carried_away --object glasses --truth left --no-preview
```

Then detect, replay and score:

```bash
for t in trials/*/; do $D python3 -m eval.record --trial-id $(basename $t) --detect-only; done
$D python3 -m eval.replay --trials trials/ --system all
$D python3 -m eval.report --trials trials/ --out report.md
```

Done when: `report.md` has 20 trials under `full`, and the shell-game row (`inside_box_moved`) is 10/10 with
the laser error median < 3 cm. These are the scoreboard's real numbers; never quote the synthetic ones.
Copy `report.md` into the log. A wrong trial: keep it (do not re-record over it) and note what happened.

## 5. Hall noise and the rig's own speaker

1. **Audio devices.** The container's default output is HDMI. Find the speaker and mic, and pin them:

   ```bash
   $D python3 -m voice.tts --devices          # then in config.local.yaml:  tts: {output_device: "<name part>"}
   arecord -l                                  # the mic card; listen.input_device if it isn't the default
   ```

2. **10 minutes of hall noise** at the venue, with people talking but nobody talking to the rig:

   ```bash
   arecord -D plughw:<card>,0 -f S16_LE -r 16000 -c 1 -d 600 ~/hall.wav
   cp ~/hall.wav tests/stt_audio/hall.wav                          # git-ignored; never commit it
   $D python3 scripts/overheard_test.py tests/stt_audio/hall.wav
   ```

   Done when: 0 or 1 false triggers. More than 1: run it again with `--mode wake`; if that is 0-1, set
   `listen: {mode: wake}` in `config.local.yaml` (the judge then says "room, where are my keys").
3. **Own-speaker echo test.** Start the app with voice (step 7's command), stand at the judge's spot and ask
   5 questions ("where are my keys", "where is my wallet", "what did I move", "what's on the table", "where is
   the notebook"), waiting for each answer.

   ```bash
   python3 -c "import sqlite3,time;[print(time.strftime('%H:%M:%S',time.localtime(t)),q,'->',a) for t,q,a in sqlite3.connect('data/events.db').execute('select t,text,answer from questions order by t desc limit 8')]"
   ```

   Done when: exactly 5 new rows, one per question you asked, none whose text is the rig's own answer.

## 6. Offline rehearsal (E3)

Over the USB link (step 1), not Wi-Fi: turning Wi-Fi off drops a Wi-Fi SSH session.

```bash
ssh guru@192.168.55.1
cd ~/askroom_rig && nmcli radio wifi off        # and unplug Ethernet if any
# start the app (step 7); the dashboard shows "offline" within a few seconds and the voice is Piper
```

Play 3 shell games and ask each question out loud, including "show me my keys". Done when: every answer is
correct, spoken in the Piper voice, with the laser on the box, and nothing waits on the network (no answer
takes > 2 s). Then `nmcli radio wifi on`.

## 7. Demo check (E5), then 5 clean games in a row (E2)

Start the app for the demo:

```bash
$D python3 main.py --camera $CAM
# dashboard: http://10.90.84.178:8000
```

Before every judge session, with the app **stopped**:

```bash
$D python3 demo_check.py --camera $CAM          # every line green; it asks you to confirm the tone and the kill switch
```

Then five shell games in a row, each reset in under 1 minute (props back home, say "room, reset" if the
board is off). Done when: 5/5 correct, laser on the box each time, resets < 1 min. A failure restarts the
count; write down what went wrong.

## Log (send this back)

| Step | Measure | Value |
|---|---|---|
| 1 | `free -m` available / clock right | |
| 2 | camera_setup gain used; tracked area (cm) | |
| 3 | laser fit error cm / centre error cm / 10 ruler spots (cm each) / kill switch ok | |
| 4 | shell games correct /10; place-hide correct /10; laser error median cm | |
| 5 | hall noise false triggers (always / wake); echo test rows = 5? | |
| 6 | offline games correct /3 | |
| 7 | demo_check all green?; clean games in a row | |
