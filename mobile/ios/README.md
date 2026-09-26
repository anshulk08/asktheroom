# Ask the Room: iPhone app

SwiftUI app (iOS 17+, no third-party packages) that talks to the rig over Bluetooth LE. The spec is `mobile/SPEC.md` (for now, `Ask the Room — iPhone App Spec.md` in the parent folder); section 3 is the protocol contract. Proposed protocol changes are in `PROTOCOL_PROPOSALS.md`.

## Open and run

Open `AskTheRoom/AskTheRoom.xcodeproj` in Xcode. Source folders sync automatically: new `.swift` files in `AskTheRoom/` or `AskTheRoomTests/` join their target with no project edits.

- **Simulator:** the Simulator has no Bluetooth, so tap "Use demo mode" (or pass `-mock YES` under Scheme → Run → Arguments). Switch demo mode on or off anywhere from the gear (Helper settings) next to the status pill.
- **iPhone:** target → Signing & Capabilities → your personal team. If the bundle id `com.asktheroom.app` is taken, add a suffix. On the phone, turn on Settings → Privacy & Security → Developer Mode.

## Screens

After the rig connects (or in demo mode) the app opens on **Home**, an assistant-style dashboard inspired by [Project Memoria](https://github.com/gamefreakoneone/Project-Memoria_Dementia-Assistant). There are three tabs:

- **Home**
  - the greeting, today's date and the time in large type, for orientation;
  - **The room noticed**: things moved off the table, things the room can't see right now (with when they were last seen), and new unnamed objects. Each has a big "Help me find it" button (asks the rig, so the laser points) and "Got it" to put it away;
  - **Your things**: a tile per tracked object with where it is in plain words ("Inside the box", "Someone is holding it"); tap to ask the room.
- **Table**: the spec's Room screen, the live map with a short key under it. Tap a thing on the map and one card underneath says where it is in words, with "Ask the room" and "More about it"; otherwise the card is the latest answer. Past questions live on Recent.
- **Recent**: what changed since the app connected and the questions asked, newest first, with clock times, under "In the last hour" and "Earlier". Changes are worked out on the phone by comparing snapshots (no protocol change). Pill wording stays neutral: "picked up", never "taken".

Asking from Home, or tapping a row on Recent, opens an **answer sheet** over that screen rather than jumping to another tab. It shows the answer (or where the thing is), a map with it ringed until the person taps Done, and large buttons: "Ask the room where it is", "More about it" (the detail sheet) and "Show the whole table" (the only thing that switches tabs).

The design follows dementia-friendly guidance: one thing at a time, no surprise screen changes, plain wording that never blames, large text and targets (one column of tiles at the largest text sizes), calm colours (red only where the laser points), and no hidden gestures.

The map was decluttered the same way: no grid, one faint edge on plain things and a strong dashed edge only on hidden or held ones, block names inside the block with what's inside sitting under the name, a short "unnamed" label (with a link mark for "might be an older thing") instead of "unnamed object 9 ? link", and no status words on the map. The words are one tap away (the card, and VoiceOver reads them on every chip), and a key under the map explains only the marks in use. Suggestions wrap instead of scrolling off the edge. At accessibility text sizes the key and suggestions scroll with the card so the map keeps its size. Sources: W3C COGA [Avoid too much content](https://www.w3.org/WAI/WCAG2/supplemental/patterns/o5p03-manageable-quantity/), Apple's [map decluttering](https://developer.apple.com/documentation/MapKit/decluttering-a-map-with-mapkit-annotation-clustering), and progressive disclosure (labels on demand) from map labelling practice.

**Helper settings** (the gear next to the status pill) are for a family member or carer: demo mode, "Read answers aloud" (off by default, since the rig already speaks), "Show answers to questions asked in the room" (PROTOCOL_PROPOSALS.md P2), the rig's status, and "Show them again" for notices put away with "Got it".

**The rig's voice.** On `main` the rig speaks with ElevenLabs (`voice/tts.py`; Grok is only the language model, not the voice). With a helper's ElevenLabs key and the rig's voice ID (`ELEVENLABS_VOICE_ID`) entered in helper settings, "Read answers aloud" uses the same voice and model (`eleven_flash_v2_5`), so the phone sounds like the room. Without them, or offline, or if ElevenLabs takes over 3 s, the iPhone's built-in voice reads it, like the rig's Piper fallback. "Try the voice" plays a sample.

**Privacy.** The ElevenLabs key is kept in this phone's Keychain (this device only) and never in the repo. When read aloud is on with a key, the text of each answer is sent to ElevenLabs to be spoken; nothing else is sent, and no audio is saved. With no key, speech stays on the phone.

## Tests

```sh
cd mobile/ios/AskTheRoom
xcodebuild test -project AskTheRoom.xcodeproj -scheme AskTheRoom \
  -destination 'platform=iOS Simulator,name=iPhone 17 Pro'
```

If `xcode-select` points at the Command Line Tools, prefix the command with `DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer`. If the name doesn't match, use a simulator id from `xcrun simctl list devices` (`-destination 'id=<id>'`).

| File | What it does |
| --- | --- |
| `Framing.swift` | Chunk reassembly per characteristic (spec section 3) |
| `Models.swift` | Wire types; optional keys, open-string `action` |
| `MockData.swift` | The spec's sample snapshot |
| `MockRoom.swift` | Mock mode: the sample snapshot, a looping story, faked answers |
| `RoomStore.swift` | App state: snapshot, status, questions and answers, highlight, timeouts |
| `MapLayout.swift` | What to draw per entity (section 5 table), where without overlaps, and the key |
| `TableMapView.swift`, `FlowLayout.swift` | The table map, pulse, laser reticle, sweep and circle, key; wrapping rows |
| `AnswerCard.swift`, `AskBar.swift` | Answer card; suggestion chips, text field, mic |
| `Dashboard.swift` | Home's words: where each thing is, what the room noticed, what changed between snapshots |
| `HomeView.swift` | Home (start screen) and the Home / Table / Recent tabs |
| `RecentView.swift` | The Recent tab |
| `AnswerSheet.swift` | The answer sheet over Home and Recent |
| `HelperSettings.swift`, `Speaker.swift` | Helper settings; reading answers aloud in the rig's ElevenLabs voice, or the iPhone's |
| `RoomView.swift` | Connect and Table screens, status pill, banners, detail sheet |
| `RoomLink.swift` | CoreBluetooth: scan, connect to the strongest rig, subscribe, reconnect with backoff |
| `Dictation.swift` | Hold-to-talk, on-device speech recognition only |
| `tools/make_icon.swift` | Draws `AppIcon.png` (`swift tools/make_icon.swift <out.png>`) |

## See it without opening Xcode

`./run.sh <name> [launch args]` builds, installs and launches on the simulator, then saves a screenshot to `build/shots/<name>.png`:

```sh
./run.sh keys -mock YES -mockPaused YES -mockTab table -mockAsk "Where are my keys?"
```

Set `SIM=<id>` for a different simulator and `WAIT=<seconds>` to wait longer before the screenshot. Mock launch arguments:

| Argument | Effect |
| --- | --- |
| `-mock YES` | Start in demo mode |
| `-mockPaused YES` | Hold the sample snapshot; don't play the story |
| `-mockOffline YES` | Rig reports no internet |
| `-mockAppDown YES` | Rig's room app is down (banner, greyed map) |
| `-mockAsk "a\|b"` | Ask these questions, one every 2.5 s (on Home each opens the answer sheet) |
| `-mockSelect keys` | Open that entity's detail sheet (with `-mockTab table`, pick it on the map) |
| `-mockFocus phone` | Open the answer sheet on that entity |
| `-mockTab table` | Open on the Table (or `recent`) tab instead of Home |
| `-mockScroll YES` | Scroll Home to the bottom |
