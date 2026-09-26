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
  - **The room noticed**: reminders and the morning report as the rig fires them (in the rig's own words, first in the list), things moved off the table, things the room can't see right now (with when they were last seen), and new unnamed objects. Each has a big "Help me find it" button (asks the rig, so the laser points) and "Got it" to put it away;
  - **Your things**: a tile per tracked object with where it is in plain words ("Inside the box", "Someone is holding it"); tap to ask the room.
- **Table**: the spec's Room screen, the live map with a short key under it. Tap a thing on the map and one card underneath says where it is in words, with "Ask the room" and "More about it"; otherwise the card is the latest answer. Past questions live on Recent.
- **Recent**: what changed since the app connected and the questions asked, newest first, with clock times, under "In the last hour" and "Earlier". Changes are worked out on the phone by comparing snapshots (no protocol change). Pill wording stays neutral: "picked up", never "taken".

Asking from Home, or tapping a row on Recent, opens an **answer sheet** over that screen rather than jumping to another tab. It shows the answer (or where the thing is), a map with it ringed until the person taps Done, and large buttons: "Ask the room where it is", "More about it" (the detail sheet) and "Show the whole table" (the only thing that switches tabs).

The design follows dementia-friendly guidance: one thing at a time, no surprise screen changes, plain wording that never blames, large text and targets (one column of tiles at the largest text sizes), calm colours (red only where the laser points), and no hidden gestures.

The map reads like Find My: each thing is a round pin with its own picture and its name underneath. Status shows as a badge on the pin (a hand when held, a question mark when the room can't see it, a link for "might be an older thing"), a dashed ring when it's hidden, and a fade when the room isn't sure. Containers and covers stay drawn to scale with their picture and name inside, so the keys' pin sits inside the box. Tapping a pin enlarges it, rings it in the accent colour and gives a light haptic.

**Pictures.** Things get an emoji where a good one exists (🔑 👛 👓 📱 💊 📦 📓, and 🔌 for "my charger") or an SF Symbol where none does (the remote). A helper can change any of them under "More about it" → "Change picture": one tap for common emoji and symbols, or the emoji keyboard for anything else, including a Genmoji on iPhones with Apple Intelligence (iOS 18+). The same picture is used on the map, Home, Recent and the cards. Picks are saved on this phone (`thing-icons.json` in Application Support). `RESEARCH.md` compares emoji, SF Symbols, Genmoji, Image Playground and custom art.

The map was also decluttered: no grid, one faint edge on plain things and a strong dashed edge only on hidden or held ones, a short "unnamed" label instead of "unnamed object 9 ? link", and no status words on the map. The words are one tap away (the card, and VoiceOver reads them on every chip), and a key under the map explains only the marks in use. Suggestions wrap instead of scrolling off the edge. At accessibility text sizes the key and suggestions scroll with the card so the map keeps its size. Sources: W3C COGA [Avoid too much content](https://www.w3.org/WAI/WCAG2/supplemental/patterns/o5p03-manageable-quantity/), Apple's [map decluttering](https://developer.apple.com/documentation/MapKit/decluttering-a-map-with-mapkit-annotation-clustering), and progressive disclosure (labels on demand) from map labelling practice.

**Helper settings** (the gear next to the status pill) are for a family member or carer: demo mode, "Read answers aloud" (off by default, since the rig already speaks), "Show answers to questions asked in the room" (off by default: questions asked out loud, on the dashboard or by text, PROTOCOL.md 6a), the rig's status, and "Show them again" for notices put away with "Got it".

**Voice.** "Read answers aloud" uses Grok's voice from xAI by default (`POST https://api.x.ai/v1/tts`). Under Helper settings → Voice a helper can:
- pick the voice: Grok, "Same as the rig" (the rig's ElevenLabs voice ID and model `eleven_flash_v2_5`, from `voice/tts.py`) or the iPhone's own voice;
- pick a Grok voice (Eve by default; the full list is fetched once a key is saved);
- set the speed;
- see where the sound is playing.

This is phone-only: the rig's voice path is unchanged and doesn't call Grok.

Each answer is matched to the audio output:

| Output | Speed | Audio quality |
| --- | --- | --- |
| iPhone speaker | slightly slower | 24 kHz |
| Headphones and AirPods | chosen speed | full quality, 44.1 kHz |
| Bluetooth headset call profile | chosen speed | 16 kHz, all that link carries |
| AirPlay, TV, car | slower | 48 kHz |

If headphones or a Bluetooth speaker disconnect mid-answer, speech stops rather than carrying on out loud, per Apple's route-change guidance. Without a key, offline, or if the voice takes over 3 s, the iPhone's built-in voice reads the answer (it adapts its speed too), like the rig's Piper fallback. "Try the voice" plays a sample. Details and trade-offs are in `RESEARCH.md`.

**Privacy.** The xAI and ElevenLabs keys are typed into helper settings and kept only in this phone's Keychain (this device only), never in the repo or the app. When read aloud is on with a cloud voice, the text of each answer is sent to that service (xAI for Grok, ElevenLabs for "Same as the rig") to be spoken; nothing else is sent, and no audio is saved. With the iPhone voice, or no key, speech stays on the phone.

## Receiving from the Jetson

Everything reaches the phone over Bluetooth from `mobile/bridge/ble_bridge.py` on the Jetson (on `main`; spec in `mobile/PROTOCOL.md`). The phone does its part as follows:

- **Fewer chunks.** After connecting it reads `status` before subscribing, so the bridge learns the link's MTU before it sends the first snapshot (PROTOCOL.md section 3). It subscribes to `answer`, then `status`, then `state` last, because subscribing to `state` sends a snapshot at once.
- **Nothing dropped.** Answers the phone didn't ask for (`id: null`) are no longer thrown away. Reminders and morning reports (`src: "notice"`) always show on Home. Questions asked out loud, on the dashboard or by SMS show as a card on the Table tab when the helper setting is on.
- **Back quickly.** When a working link drops, the phone asks to connect to the remembered rig straight away. The system finishes that as soon as the rig advertises again. Only failed attempts back off, up to 5 s.
- **Silent links.** The bridge sends state at least every 5 s. If nothing arrives for 15 s on a link that still looks connected (the bridge hung, or lost the subscription), the phone disconnects and reconnects.
- **Torn messages** are dropped and logged (`RoomLink`, in Console). The next snapshot replaces them within 0.5 s.

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
| `TableMapView.swift`, `FlowLayout.swift` | The table map with Find My-style pins, pulse, laser reticle, sweep and circle, key; wrapping rows |
| `ThingIcon.swift` | Each thing's picture (emoji, SF Symbol or Genmoji), helpers' picks, the picture picker |
| `AnswerCard.swift`, `AskBar.swift` | Answer card; suggestion chips, text field, mic |
| `Dashboard.swift` | Home's words: where each thing is, what the room noticed, what changed between snapshots |
| `HomeView.swift` | Home (start screen) and the Home / Table / Recent tabs |
| `RecentView.swift` | The Recent tab |
| `AnswerSheet.swift` | The answer sheet over Home and Recent |
| `HelperSettings.swift`, `Speaker.swift` | Helper settings; reading answers aloud with Grok, the rig's ElevenLabs voice or the iPhone's, matched to the audio output |
| `RoomView.swift` | Connect and Table screens, status pill, banners, detail sheet |
| `RoomLink.swift` | CoreBluetooth: scan, connect to the strongest rig, read status then subscribe, reconnect at once after a drop, reconnect if nothing arrives for 15 s |
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
| `-mockSettings YES` | Open helper settings |
| `-mockIconPicker YES` | With `-mockSelect`, open the picture picker over the detail sheet |
| `-mockIcons "remote=📺"` | Show these pictures instead of the usual ones (not saved) |
| `-mockNotice "text"` | A second after launch, the rig fires this reminder about the pill bottle |
