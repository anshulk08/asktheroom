# Ask the Room: iPhone app

SwiftUI app (iOS 17+, no third-party packages) that talks to the rig over Bluetooth LE. The spec is `mobile/SPEC.md` (for now, `Ask the Room — iPhone App Spec.md` in the parent folder); section 3 is the protocol contract. Proposed protocol changes are in `PROTOCOL_PROPOSALS.md`.

## Open and run

Open `AskTheRoom/AskTheRoom.xcodeproj` in Xcode. Source folders sync automatically: new `.swift` files in `AskTheRoom/` or `AskTheRoomTests/` join their target with no project edits.

- **Simulator:** the Simulator has no Bluetooth, so tap "Use demo mode" (or pass `-mock YES` under Scheme → Run → Arguments). Long-press the status pill to switch demo mode on or off anywhere.
- **iPhone:** target → Signing & Capabilities → your personal team. If the bundle id `com.asktheroom.app` is taken, add a suffix. On the phone, turn on Settings → Privacy & Security → Developer Mode.

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
| `MapLayout.swift` | What to draw per entity (section 5 table) and where, without overlaps |
| `TableMapView.swift` | The table map, pulse, laser reticle, sweep and circle |
| `AnswerCard.swift`, `AskBar.swift` | Answer and history; suggestion chips, text field, mic |
| `RoomView.swift` | Connect and Room screens, status pill, banners, detail sheet |
| `RoomLink.swift` | CoreBluetooth: scan, connect to the strongest rig, subscribe, reconnect with backoff |
| `Dictation.swift` | Hold-to-talk, on-device speech recognition only |
| `tools/make_icon.swift` | Draws `AppIcon.png` (`swift tools/make_icon.swift <out.png>`) |

## See it without opening Xcode

`./run.sh <name> [launch args]` builds, installs and launches on the simulator, then saves a screenshot to `build/shots/<name>.png`:

```sh
./run.sh keys -mock YES -mockPaused YES -mockAsk "Where are my keys?"
```

Set `SIM=<id>` for a different simulator and `WAIT=<seconds>` to wait longer before the screenshot. Mock launch arguments:

| Argument | Effect |
| --- | --- |
| `-mock YES` | Start in demo mode |
| `-mockPaused YES` | Hold the sample snapshot; don't play the story |
| `-mockOffline YES` | Rig reports no internet |
| `-mockAppDown YES` | Rig's room app is down (banner, greyed map) |
| `-mockAsk "a\|b"` | Ask these questions, one every 2.5 s |
| `-mockSelect keys` | Open that entity's detail sheet |
