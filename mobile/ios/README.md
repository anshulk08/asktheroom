# Ask the Room: iPhone app

SwiftUI app (iOS 17+, no third-party packages) that talks to the rig over Bluetooth LE. The spec is `mobile/SPEC.md` (for now, `Ask the Room — iPhone App Spec.md` in the parent folder); section 3 is the protocol contract. Proposed protocol changes are in `PROTOCOL_PROPOSALS.md`.

## Open and run

Open `AskTheRoom/AskTheRoom.xcodeproj` in Xcode. Source folders sync automatically: new `.swift` files in `AskTheRoom/` or `AskTheRoomTests/` join their target with no project edits.

- **Simulator:** the UI runs in mock mode. The Simulator has no Bluetooth.
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
