# Expo demo script (Musts only)

Judges rotate past the table, so each run is about 2 minutes and the reset takes under 1 minute. This script uses only features that work offline and are covered by tests: the world model, template answers and the laser. Grok, the dashboard and the phone are extras (see the end). Wording and timings are drafts: rehearse five times (PLANS E2) and adjust.

## Before judging (once)

1. `python demo_check.py`: all green. Check the kill switch by hand (check 8).
2. After any camera replug: `scripts/camera_setup.sh <exposure> /dev/v4l/by-id/<camera> <gain>`.
3. Table tag in view, flat, near the centre. Say "room, recalibrate". The log should say `table recalibration ok`. If it warns that the table frame moved, recalibrate the laser (`python -m act.calibrate`).
4. `config.yaml`: `demo.hold_notices: true` if reminders might talk over a judge.
5. Wi-Fi check. If the venue Wi-Fi is bad, switch the rig to the phone hotspot. Offline, everything below still works (Piper voice).

## Staging

- On the table: keys, notebook, box, wallet and pill bottle. Leave the other props off to keep the view clear.
- Position the keys so the camera sees them but the judge's line of sight from the front of the table does not (behind the box, from the judge's side).
- A light, matte table surface, so the laser dot is visible.

## The run (about 2 minutes)

| # | Presenter says / does | Judge does | Rig |
|---|---|---|---|
| 1 | "It watches the table from above and remembers where things are, even when they're hidden." | — | — |
| 2 | Hand the keys to the judge: "Hide these however you like." | Keys under the notebook, notebook into the box, slide the box. | Tracks the chain. |
| 3 | "Now ask it." | "Where are my keys?" | "Your keys are under the notebook, which is inside the box." Laser on the box. |
| 4 | "It knows how they got there." | "What happened to my keys?" | The history, in one or two sentences. |
| 5 | "It's careful about medication." | "Did I take my pills?" | Neutral wording: where the bottle is and when it moved, never whether anything was taken. |
| 6 | Carry the wallet off the left edge. | "Where's my wallet?" | "Carried off the left side." The laser sweeps that edge. |

If an answer is wrong, say so plainly and ask again. Don't cover it up.

## Reset (under 1 minute)

1. Put the keys, notebook, box and wallet back at the start positions. Leave the pill bottle where it is.
2. Say "room, reset" (or type `reset` into the dashboard).
3. Wait until the laser is off and the dashboard shows every object VISIBLE, then start the next judge.

## Extras, only if the run above went cleanly and the judge has time

- Grok (online only): "What colour is the notebook?" or "Where is my red mug?" for an object it hasn't been taught. Grok picks it on the camera frame, and the rig remembers the name.
- Teach a name: put an object down and say "this is my charger", then "where is my charger?".
- Dashboard / phone: show the live object list and the answer log.

Don't demo the face-safety claim: there is no face check in the pipeline.
