# Ask the Room BLE protocol (v1)

The iPhone app talks to the Jetson directly over Bluetooth LE. There is no cloud server and no Wi-Fi,
and the link is one phone to one rig. The Jetson is the GATT **peripheral** (`mobile/bridge/ble_bridge.py`,
BlueZ 5.64). The phone is the **central** (CoreBluetooth). The reference implementation of everything
below is `mobile/bridge/bleproto.py`; `tests/test_mobile_protocol.py` pins it down.

Positions are table centimetres. Whenever the app publishes a view (`GET /state` `"view"`, and then the state
message carries `"view"`, section 7), every position and size the phone gets is in the **user's frame**, the
table as seen from the side they sit at: x runs from the user's left to their right, y from the far side to
the side nearest them, (0, 0) is the far-left corner, and the `bottom` edge is the one nearest the user.
That covers `table`, `e[].xy`, `e[].r`, `e[].edge`, the answer's `target` and the edge in a `sweep:` action;
the phone draws them as they come, with `bottom` at the bottom of the screen. The user picks their side with
the orientation write (section 5b). An older app without a view sends the camera's frame: origin at the
top-left of the camera's table view, x to the right, y down, and `table` = `[w, h]` from `config.yaml`
`table.size_cm` (default `[90, 60]`).

## 1. Advertising

| Field | Value |
|---|---|
| Advertising data (21 of 31 bytes) | Flags `0x06` (LE General Discoverable, BR/EDR not supported), complete list of 128-bit service UUIDs = `8A1E0001-6B7F-4C2B-9E3A-2F5D7C1A0001` |
| Scan response (12 bytes) | Complete local name `AskTheRoom`. BlueZ moves it here because it doesn't fit next to the 128-bit UUID. |
| PDU | Legacy `ADV_IND` (connectable, scannable), sent through the controller's extended-advertising commands |
| Interval | 100–150 ms (`--adv-interval`; the BlueZ default is 1.28 s, which makes discovery slow) |

The phone scans with `scanForPeripherals(withServices: [8A1E0001-…])`. Don't filter on the name: iOS
reports `CBAdvertisementDataLocalNameKey` only after it receives a scan response, and `peripheral.name`
can show the Jetson's GAP name (`guru-desktop`) or a cached name.

## 2. GATT table

Service `8A1E0001-6B7F-4C2B-9E3A-2F5D7C1A0001` (primary). Every UUID follows the pattern
`8A1E000N-6B7F-4C2B-9E3A-2F5D7C1A000N`.

| N | Name | UUID | Properties | Direction | Payload |
|---|---|---|---|---|---|
| 2 | question | `8A1E0002-6B7F-4C2B-9E3A-2F5D7C1A0002` | Write (with response) | phone → rig | unframed UTF-8 JSON, ≤ 180 bytes |
| 3 | answer | `8A1E0003-6B7F-4C2B-9E3A-2F5D7C1A0003` | Notify | rig → phone | framed JSON |
| 4 | state | `8A1E0004-6B7F-4C2B-9E3A-2F5D7C1A0004` | Notify | rig → phone | framed JSON |
| 5 | status | `8A1E0005-6B7F-4C2B-9E3A-2F5D7C1A0005` | Read + Notify | rig → phone | Read: unframed JSON. Notify: framed JSON |

No characteristic needs encryption or pairing, so iOS never shows a pairing prompt.

## 3. MTU and chunk size

A notification value can be at most `ATT_MTU − 3` bytes. Each chunk's payload is at most
`ATT_MTU − 3 − 3` bytes: 3 bytes of ATT overhead, then our 3-byte header.

| ATT MTU | Notification value max | JSON bytes per chunk |
|---|---|---|
| 185 (typical iPhone) | 182 | **179** |
| 23 (ATT minimum) | 20 | 17 |
| 517 | 514 | 511 |

The bridge learns the MTU from BlueZ's `mtu` option on every read and write. Until it has heard from the
phone it assumes 185 (`--mtu`). With several centrals it uses the smallest MTU. If it learns an MTU smaller
than the one it assumed, it resends the state snapshot, because the earlier one may have been cut short.
**Recommended phone sequence:** discover → read `status` (this tells the bridge the MTU) → subscribe to
`answer`, `status`, `state`.

## 4. Framing (every notification: answer, state, status)

```
byte 0     msg_id        u8   per characteristic; +1 per message, wraps 255 → 0
byte 1     chunk_index   u8   0, 1, 2, … within this message
byte 2     flags         u8   bit0 = FINAL (last chunk of the message); bit1 = COMPRESSED; bits 2–7 are 0
byte 3..   payload            UTF-8 JSON bytes (or, COMPRESSED, raw DEFLATE bytes), ≤ MTU − 6 per chunk
```

- **COMPRESSED** (bit1) is set on every chunk of a message whose concatenated payload is raw DEFLATE
  (RFC 1951, no zlib header: Apple's `.zlib`, Python `zlib.compressobj(9, zlib.DEFLATED, -15)`) of the JSON.
  Inflate, then decode and parse. The bridge compresses only once every connected central has said
  `{"hello": {"z": 1}}` (5c), and only when it makes the message smaller. A message that doesn't inflate is
  dropped like a gap.

- A message has 1 to 256 chunks. An empty or short message is a single chunk with FINAL set.
- The JSON is split on byte boundaries, which can fall inside a multi-byte UTF-8 character. Concatenate
  the payloads first and only then decode UTF-8 and parse JSON.
- **Reassembly (phone):** keep one partial message per characteristic. If a chunk has a different `msg_id`,
  discard the incomplete previous message. On chunk 0, start a new message. Append chunks in order. On FINAL,
  concatenate, decode and parse. If the index skips (a gap) or a chunk arrives with nothing in progress, drop
  the partial message and wait for the next chunk 0. Notifications on one link arrive in order, so gaps
  should not happen, but they must never produce a corrupt message.
- `msg_id` numbers are independent per characteristic. The bridge may skip ids: it drops a state or status
  message that was superseded before its first chunk went out.

Example: an answer as msg 7, in one chunk at MTU 185 (147 bytes on the air):

```
07 00 01  7B 22 69 64 22 3A 33 32 31 2C …   {"id":321,"ok":true,"text":"Your keys are inside the box. …"}
^msg ^idx ^FINAL
```

A 414-byte state snapshot at MTU 185 is 3 chunks: `03 00 00 …179 B`, `03 01 00 …179 B`, `03 02 01 …56 B`.

## 5. question (write with response)

```json
{"id": 321, "q": "where are my keys?"}
```

- `id`: integer 0–65535, chosen by the phone and echoed in the answer. `q`: the question text, as typed or dictated.
- The whole write is **≤ 180 bytes** of UTF-8 (`JSONSerialization` output). That fits one ATT write at MTU
  185, so no long or prepared write is needed. Truncate the question on the phone to stay under the limit.
- The ATT write always succeeds. Every rejection comes back as an answer with `ok: false`:
  - longer than 180 bytes: `{"id": id, "ok": false, "text": "Question too long."}` (the id is echoed if the JSON parses)
  - unparseable: `"I couldn't read that question."`; empty `q`: `"Ask me where something is, like: where are my keys?"`
  - a question while another is still in flight: `"I'm still answering your last question."`
    (one question is answered at a time, and one more can wait in the queue)
- The bridge sends the question to `POST http://127.0.0.1:8000/ask` with `{"text": q, "source": "phone"}`.
  `phone` and `dashboard` are the sources a client may name, and both speak and move the laser
  (`main.Room.ask_and_act` treats only `sms` as text-only; n8n only receives a report of each answer). Asking from the phone has the same
  effect as asking from the dashboard: the answer is spoken on the rig and the laser points.
- **Dictating next to the rig (P1).** The always-on mic also hears a question the judge dictates into the
  phone. The rig drops a voice question that nearly matches (same words after normalizing) a phone question
  from the last 3 s, so it is answered once. The phone shows only the answer whose `id` matches its question.

## 5a. voice settings (write to question)

```json
{"voice": {"e": "grok", "s": 1.1, "v": "ara"}}
```

The phone's helper voice settings, written to the **question** characteristic (a write without `q` is not a
question). The rig's speaker then uses the same voice as the phone would.

| Key | Meaning |
|---|---|
| `e` | engine, the app's `Speaker.Engine` raw value: `grok` (default), `rigVoice` ("Same as the rig", ElevenLabs), `builtIn` (the iPhone's own voice, which the rig can't make: it uses Piper) |
| `v` | the Grok voice id (`eve` when empty) |
| `s` | speed, 0.7 to 1.5 (clamped) |

- Send it **on every connect** (after subscribing) and whenever the helper changes a voice setting. The rig keeps
  the last one across restarts (`data/voice.json`); until it gets one it uses the app's default (Grok `eve`, 1.0).
- No answer comes back. The bridge sends it to `POST /voice` (`{"engine", "grok_voice", "speed"}`).
- Offline, or when the cloud voice fails before any audio, the rig speaks with Piper.

## 5b. orientation (write to question)

```json
{"orient": {"front": "right"}}
```

The phone's "I sit here": which side of the table the user sits at, written to the **question**
characteristic. `front` is that side **in the camera's frame** (the frame of an app without a view):
`bottom` (the camera's side, the default), `right`, `top` or `left`. The phone can work it out from the map it
is showing: the side the user tapped, turned back through the current `view.f` with the table below.

- No answer comes back. The bridge sends it to `POST /orientation` (`{"front"}`); the rig keeps it across
  restarts (`data/viewer.json`, over `config.yaml` `viewer.front`). Spoken answers use it from the next one
  ("carried off the table on your left", "at the far right").
- Right after the app accepts it the bridge sends a fresh state message, so the map turns at once.
- `{"orient": {"front": null}}` goes back to the rig's configured seat (config `viewer.front`) and forgets the saved one: the phone's "use the rig's default".
- An `orient` object naming anything else is ignored (no answer); a write over 180 bytes is not an orientation write.

How a camera side becomes the user's, per `front` (camera edge → user edge; the camera table is `[w, h]`,
the user's `table` is `[w, h]` for bottom and top and `[h, w]` for right and left):

| `front` | camera `left` | camera `right` | camera `top` | camera `bottom` | camera point (x, y) → user (x, y) |
|---|---|---|---|---|---|
| `bottom` | left | right | top | bottom | (x, y) |
| `top` | right | left | bottom | top | (w − x, h − y) |
| `right` | top | bottom | right | left | (h − y, x) |
| `left` | bottom | top | left | right | (y, w − x) |

Example: a 100 × 60 cm camera table, `front: right`: the user's `table` is `[60, 100]`, and the camera
view's top-right point (90, 5) is at (55, 90), near the user on their right. With a tabletop outline
(`table_area.json`) the user's table is the outline's enclosing rectangle, squared to it; the affine `m` in
`GET /state` `"view"` does both, and the bridge applies it.

## 5c. hello (write to question)

```json
{"hello": {"z": 1}}
```

The phone writes this first on every connect (after reading status). `z: 1`: it inflates COMPRESSED
messages (section 4). No answer comes back; the bridge re-sends the state snapshot compressed. A central
that never says hello (an older app) gets plain JSON, and while it is connected nobody gets compressed
messages (notifications go to every subscriber alike).

## 6. answer (notify)

```json
{"id": 321, "ok": true, "text": "Your keys are inside the box. I'm pointing at it.",
 "point_at": "keys", "action": "point", "target": [70.4, 38.1], "ms": 812}
```

| Key | Type | Meaning |
|---|---|---|
| `id` | int | the question's id |
| `ok` | bool | false = the bridge's own reply (rejected, room down, timeout, error) |
| `text` | str | what the rig speaks; show it on the answer card |
| `point_at` | str \| null | entity the laser aims at (`keys`, `box`, `thing:3`, …) |
| `action` | str \| null | `point`, `circle` (lost track: circling the last-seen spot), `sweep:left\|right\|top\|bottom` (carried off that edge, the user's edge with a view: `bottom` is nearest them), or null. **An open string (P3):** later versions may add values (`trace`, `tour`, …); a client that doesn't know one pulses `point_at` at `target` if present, else shows the text only |
| `target` | [x, y] \| null | table-cm position of `point_at` (the user's frame with a view): its resolved position (a hidden object inherits its parent's), falling back to its last-seen spot; 1 decimal |
| `ms` | int | bridge time from receiving the write to having the answer (includes `/ask` and one `/state`) |

Timeouts nest so exactly one answer comes back and the rig never contradicts it: the server gives up after 10 s (`server/app.py` `ASK_TIMEOUT_S`) and answers "Sorry, that took too long. Please ask again."; an answer that finishes later is neither spoken nor aimed (`main.ANSWER_LATE_S`). The bridge waits 12 s (`ask_timeout_s`) before its own `ok: false` reply, and the app waits 15 s (`RoomStore.answerTimeout`).

All keys are always present. `ok: false` texts: `"The room isn't running right now."` (the app's HTTP API is
unreachable), `"Sorry, that took too long. Please ask again."` (over 12 s),
`"Sorry, something went wrong answering that."` (HTTP error).

### 6a. Answers the phone didn't ask for (P2, notices)

The same characteristic also carries, with `"id": null`:

- **Room answers**: every question answered by the rig from another source (spoken to the room, the dashboard,
  SMS), with `src` = that source and `q` = the question as heard. The phone's own questions are not repeated.
- **Notices**: a reminder or the morning report as it fires, with `src: "notice"`, `nid` (the notice id; the
  phone acknowledges it with the dashboard's `POST /notices/{nid}/ack` when it has Wi-Fi, otherwise not at all
  in v1) and `kind` (`reminder`, `morning`, …).

```json
{"id": null, "src": "voice", "q": "where are my keys", "ok": true, "text": "Your keys are inside the box.",
 "point_at": "keys", "action": "point", "target": [70.4, 38.1], "ms": null}
{"id": null, "src": "notice", "nid": 12, "kind": "reminder", "ok": true,
 "text": "It's 9 and the pill bottle hasn't been picked up yet.", "point_at": "pill_bottle", "action": "point",
 "target": [30.2, 12.0], "ms": null}
```

Only what happens while a phone is subscribed is sent; nothing is replayed on connect. The bridge learns of
them from `GET /state` (`answers`, the last 10 with a `seq`; `notices`), so they arrive within ~0.25 s of the
answer. A v1 client that ignores `src` drops them, because their `id` matches none of its questions.

The laser stays on for `laser_timeout_s` (10 s). Its live state comes in `state.laser`, so the phone can
pulse the target while `laser.on && laser.target == point_at`.

## 7. state (notify)

A full snapshot every time; there are no diffs. It is sent:

1. right after the phone subscribes to `state`
2. when something changed, at most **2 Hz** (the bridge polls `GET /state` at 4 Hz)
3. at least every **5 s**, as a heartbeat

"Changed" ignores detector jitter: a position has to move more than 0.5 cm, or confidence (`c`) or guess
confidence (`gc`) more than 0.05 (`gc` appearing or disappearing counts), or a status, parent, edge, alias,
`maybe_same_as`, guess (`g`) or `as` value has to change, or an entity has to be added or removed, or `online`,
`laser`, `table` or `view` has to change. A visible object's `ls` ticking does not count.

Stale things are left out: an unnamed `thing:N` (no aliases) whose status is GONE or UNKNOWN and whose
last-seen time is more than **600 s** before the snapshot's `t` is not in `e` (one with no last-seen time
is kept). Named things and the configured objects are always sent. When a stale thing drops out, that is an
"entity removed" change.

```json
{"v": 1, "t": 1790389843.0, "table": [90.0, 60.0], "online": false,
 "view": {"f": "bottom", "o": false},
 "laser": {"on": true, "target": "box"},
 "e": [{"n": "keys", "k": "t", "s": "I", "p": "box", "xy": [41.2, 29.0], "r": [70.4, 38.1], "c": 0.85, "ls": 1790389800.4},
       {"n": "box", "k": "c", "s": "V", "xy": [70.4, 38.1], "r": [70.4, 38.1], "c": 1.0, "ls": 1790389843.0},
       {"n": "thing:3", "k": "t", "s": "V", "xy": [20.0, 12.5], "r": [20.0, 12.5], "c": 0.9, "a": ["charger"], "m": [["thing:1", 0.74]]},
       {"n": "thing:5", "k": "t", "s": "V", "xy": [60.1, 45.0], "r": [60.1, 45.0], "c": 0.8, "a": ["stapler"], "as": "grok"},
       {"n": "thing:6", "k": "t", "s": "V", "xy": [8.0, 50.2], "r": [8.0, 50.2], "c": 0.9, "a": [], "g": "deodorant stick", "gc": 0.82}]}
```

| Key | Type | Meaning |
|---|---|---|
| `v` | int | protocol version, 1 |
| `t` | float | wall time of the snapshot (Unix s, 1 decimal) |
| `table` | [w, h] | table size in cm, as the user sees it with a view (width across from their seat, depth away from them) |
| `view` | {f, o, s?} | present when the app publishes the user's frame; then `table`, `xy`, `r`, `edge` (and answers' `target` and `sweep:` edge) are the user's (see the top of this file and 5b). `f`: the side they sit at, in the camera's frame (`bottom`/`right`/`top`/`left`). `o`: true when the map is the tabletop outline's rectangle, false for the whole calibrated area. `s`: optional labels for the table's sides from `config.yaml` `viewer.sides`, keyed like `f` by camera side, e.g. `{"right": "couch"}` (the phone turns them to its edges with `f`). Absent: an older app, camera frame |
| `online` | bool | the rig has internet: the cloud voice (ElevenLabs) and Grok (questions about what the camera sees, open questions, narration). Where-is and history answers always work offline |
| `laser` | {on, target} | laser on, and which entity it points at (`target` is present, null when none) |
| `e` | list | every entity |
| `e[].n` | str | name (`keys`, `pill_bottle`, …, or open-world `thing:N`). Always present |
| `e[].k` | `t`/`c`/`v` | kind: target / container / cover. Always present |
| `e[].s` | `V`/`H`/`U`/`I`/`G`/`X` | VISIBLE / HELD / UNDER / INSIDE / GONE / UNKNOWN. Always present |
| `e[].p` | str | parent: an entity name (`box`, `notebook`), `hand:N`, or `unknown` |
| `e[].xy` | [x, y] | last observed centre (cm, 1 decimal; the user's frame with `view`) |
| `e[].r` | [x, y] | resolved position: where it is now, through the parent chain (keys → box → table). Draw hidden objects here |
| `e[].c` | float | confidence 0–1, 2 decimals (a heuristic, not a probability) |
| `e[].edge` | str | `left`/`right`/`top`/`bottom`: the edge a GONE object left by (the user's edge with `view`: `bottom` is nearest them) |
| `e[].z` | str | room memory (specs 0009, 0010): the room zone the object is in (`couch`, `side_table`, `counter`), when it is off the table. Such an object has no `xy`/`r` (no table position): list it by zone, don't draw it on the table. The answer's `text` already says the place ("Your wallet, I think, is on the kitchen counter."). Optional |
| `e[].a` | [str] | taught names (aliases), newest first. **Things only**, and present for every thing (may be `[]`) |
| `e[].m` | [[name, score]] | "maybe the same as" an earlier thing (score 2 decimals). Optional |
| `e[].g` | str | an automatic guess of what an unnamed thing is (`deodorant stick`): the server's `guess.name` (`core/auto_name.py`) or, failing that, `belief[0]` (Grok's fused guess across settle checks); with both, whichever has the higher confidence. Not a taught name: show it hedged ("deodorant stick?"). Optional |
| `e[].gc` | float | the confidence of `g`, 0–1, 2 decimals. Only with `g`, and only when the server gave a number |
| `e[].as` | `grok` | `a[0]` was bound automatically by the Grok settle check, not taught by a person. Show it, but as the rig's name ("stapler (named by Grok)"). Absent: taught. Things only, optional |
| `e[].ls` | float | last-seen wall time (Unix s, 1 decimal) |
| `e[].rg` | str | object permanence (spec 0011, `permanence.mode: registry`): the registry's state, `visible`/`hidden`/`carried`/`last_seen`/`unknown`. A registry object is never cut from a capped state. Optional |
| `e[].rt` | 1 | the registry found it by a re-find (Grok), not by appearance alone: tentative. Optional |
| `tx` | int | state chunks the bridge sent before this message. Between two states, `tx` grows by the chunks sent; the phone compares that with the chunks it received for its "lost" count |
| `more` | int | entities left out to keep the message under the bridge's cap (`--state-max`, 12 KB of JSON): unnamed things lost or gone first, then hidden, then visible, oldest first; named things last. Optional |
| `lh` | str | the room layout's hash (`GET /room_layout`, the room map from the user's seat). Absent: no room map |
| `lay` | object | the room layout itself, `{"v", "size": [W, H], "front", "table": {"rect", "origin"}, "zones": [{"id", "say", "rect", "kind"}], "you": [x, y]}` in the user's frame. Sent on subscribe and when `lh` changes; keep the last one |

Keys whose value is null are **omitted**, except `n`, `k` and `s`. Unknown extra keys may appear in later
versions and must be ignored. Measured sizes: the live rig with 8 untracked objects is 414 B (3 chunks at
MTU 185). With all 8 objects tracked it is about 870 B (5 chunks). With 8 objects and 12 things with
aliases it is about 2.8 KB (16 chunks); the raw `/state` JSON for that is 5.7 KB. With the 12 things unnamed
and each carrying `g` and `gc` it is about 2.9 KB (17 chunks). Dropping stale things keeps a long session from
growing past that.

## 8. status (read + notify)

```json
{"app": "up", "fps": 13.1, "online": true, "cal": true, "laser_cal": false, "gk": true, "spk": true}
```

| Key | Type | Meaning |
|---|---|---|
| `app` | `up`/`down` | the room app's HTTP API answered the last `GET /state` |
| `fps` | float | perception frames per second (0.0 when down) |
| `online` | bool | the rig has internet |
| `cal` | bool | table calibrated: `table_cal.json` exists, or the world is being updated |
| `laser_cal` | bool | `laser_cal.json` exists (false: answers are spoken, but the laser can't aim) |
| `spk` | bool | the rig's speaker is connected (a Bluetooth or USB speaker is where its speech goes: `GET /state` `speaker.ok`). While true the phone **does not read answers aloud**, even with "Read answers aloud" on: the room hears the rig. False from older servers |
| `gk` | bool | the Grok settle check can run: it is enabled (`GET /state` carries `grok_check`) and the rig is online. False from older servers. When true, a still frame of the table goes to Grok each time the table settles |

- **Read** returns this JSON **unframed**. It is about 76 bytes, and always under 180.
- **Notify** is framed (section 4). It is sent right after subscribing and then on change, at most 1 Hz,
  except that an `app` change goes out at once. An fps change under 1.0 doesn't count; a `gk` or `spk` change does.

## 9. Timing budget

| What | Target |
|---|---|
| answer | notified as soon as `/ask` returns. Offline template answers take about 10–60 ms on the rig; Grok answers about what the camera sees take about 1–2 s; the bridge gives up at 12 s |
| state | ≤ 2 Hz, heartbeat 5 s, 1–16 chunks (compressed); a new state waits until the last has gone out |
| status | on change, ≤ 1 Hz |
| queueing | the bridge sends notifications in priority order answer > status > state, paced to 8 KB/s (`--rate`, a 2 KB burst): bluetoothd queues notifications without limit, and the old unpaced pump (4 chunks per 5 ms, 28 KB states at 1.5 Hz on 27 Sep) filled it until the phone went 15 s without a chunk and reconnected. A newer state or status replaces a queued one that hasn't started sending; a message already partly sent is always finished |
| link check | `mobile/bridge/test_client.py --soak 180` from a Mac: chunks lost (from `tx`), dropped partials, longest silence; the bridge log's `alive:` line every 10 s while subscribed: chunks notified per characteristic, B/s, queued, replaced, deferred, notify errors, wire/raw ratio |

## 10. Rig-specific: Realtek controller workaround

The Jetson's Bluetooth controller is a Realtek (LE features `BD 5F 66 00`: extended advertising, no LL
Privacy). With extended advertising it reports a new connection only as **LE Enhanced Connection
Complete**. Kernel 5.15 unmasks that event only for controllers with LL Privacy, so the event never reaches
the kernel. btmon showed the symptom: the central connects, its `Exchange MTU Request` arrives for an
unknown handle and is never answered, and the central gives up after 30 s (bleak on the Mac:
`BleakError: disconnected` during service discovery). The fix is to set the LE event mask to the kernel's
bits plus bit 9, once after every adapter power-on:

```
sudo hcitool -i hci0 cmd 0x08 0x0001 DF 1F 0A 00 00 00 00 00
```

The bridge sends this itself at startup and whenever the adapter powers on. That works only when the bridge
runs as root, or when `hcitool` has `cap_net_raw`. Otherwise the bridge logs the exact command to run.
See `mobile/README.md`.
