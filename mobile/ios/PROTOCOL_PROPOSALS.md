# Protocol proposals from the iOS app

For the main session, which owns the bridge (`mobile/bridge/`) and the spec. Section 3 of the spec is the contract, so none of these are implemented against the rig until you agree. The app side of each is built behind a flag, ready either way.

## P1. Double trigger when a judge dictates next to the rig

A judge who dictates "Where are my keys?" into the phone is also heard by the rig's always-on mic. That gives two answers and fires the laser twice.

Proposal: after the bridge forwards a BLE question, the room app ignores voice triggers for 3 s. Where that's awkward, it could drop a voice question whose transcript nearly matches the BLE question from the last 3 s instead.

Until then, the app shows only the answer whose `id` matches its latest question, as the spec says.

## P2. Show questions spoken to the room on the phone

With the mic always on, most questions will be spoken to the room, not typed on the phone. Showing those answers on the map is a strong demo moment.

Proposal: the bridge also sends answers to voice questions on the answer characteristic, with `"id": null` and `"src": "voice"`. Optionally it adds `"q"` with the transcript, so the phone can show what was heard. Example:

```json
{"id": null, "src": "voice", "q": "where are my keys", "ok": true,
 "text": "Your keys are inside the box.", "point_at": "keys", "action": "point", "target": [70.4, 38.1]}
```

The app shows these as a "Heard in the room" card with the same map pulse. The card is off by default (`showRoomVoiceAnswers`), and turning it on needs no other app change. A v1 app that doesn't know `src` ignores the message, because its `id` is null.

## P3. Growing action set

Spec v1 defines `point`, `circle`, `sweep:<edge>` or no action. The planning session also proposed `trace`, `tour`, `off`, `ignore` and `find_new`.

Proposal: spec section 3 says `action` is an open string, and clients must handle values they don't know. The app already does this: for an unknown action it pulses `point_at` at `target` if present, otherwise it shows the text only. It never fails to decode.

Payloads the app would need before it can draw the new actions (for you to define):

- **`trace`** (replay an object's path): an ordered list of table-cm points the laser visits, with enough timing for the map to move in step with the laser.
- **`tour`** (several objects in order): an ordered list of entity names or targets, with the time spent on each.

## Spec wording that is now out of date

- Section 4, `online`: now it means the cloud voice (ElevenLabs) and the Grok detection extras are available. Answers still work offline. The app words its banner neutrally: "Offline: using the on-device voice."
- Section 3, timing: "up to about 3 s when it asks the cloud model" is now "up to about 2–3 s when the local model handles an unusual question". The 3 s budget and the 6 s timeout stand.
- Section 5, suggestion chips: "Did I take my pills?" invites an answer the rig must refuse (pill wording stays neutral). The app uses "When did I last pick up my pills?" instead.
