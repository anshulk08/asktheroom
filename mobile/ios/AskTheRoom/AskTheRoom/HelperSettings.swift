import SwiftUI

/// For a family member or carer: demo mode, speech and the rig's details, kept off the main
/// screens but behind a labelled button rather than a hidden gesture.
struct HelperSettings: View {
    let store: RoomStore
    @AppStorage(Speaker.enabledKey) private var readAloud = false
    @AppStorage(Speaker.voiceIDKey) private var voiceID = ""
    @State private var keyDraft = ""
    @State private var hasKey = Speaker.apiKey != nil
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        NavigationStack {
            Form {
                Section {
                    Toggle("Demo mode", isOn: Binding(get: { store.isMock }, set: { store.setMock($0) }))
                } footer: {
                    Text("Shows a pretend table, for trying the app without the rig.")
                }

                Section {
                    Toggle("Read answers aloud", isOn: $readAloud)
                    Toggle("Show answers to questions asked in the room", isOn: Binding(
                        get: { store.showRoomVoiceAnswers },
                        set: { store.showRoomVoiceAnswers = $0 }))
                } header: {
                    Text("Answers")
                } footer: {
                    Text("The rig already says its answers out loud. Read aloud is for using the phone away from the table.")
                }

                Section {
                    TextField("Voice ID", text: $voiceID)
                        .textInputAutocapitalization(.never)
                        .autocorrectionDisabled()
                    SecureField(hasKey ? "API key (saved)" : "API key", text: $keyDraft)
                        .textInputAutocapitalization(.never)
                        .autocorrectionDisabled()
                        .onSubmit(saveKey)
                    if hasKey {
                        Button("Forget the key", role: .destructive) {
                            Speaker.apiKey = nil
                            hasKey = false
                        }
                    }
                    Button("Try the voice") {
                        saveKey()
                        Speaker.shared.speak("Your keys are inside the box.")
                    }
                } header: {
                    Text("The rig's voice")
                } footer: {
                    Text("Use the same ElevenLabs voice ID as the rig (ELEVENLABS_VOICE_ID) so the phone sounds like the room. The key stays in this phone's Keychain. When read aloud is on, answer text goes to ElevenLabs; without a key or internet the iPhone's own voice reads it.")
                }

                Section("The room") {
                    LabeledContent("Connection", value: connection)
                    if let status = store.status {
                        LabeledContent("Room app", value: status.appIsUp ? "Running" : "Not running")
                        if let fps = status.fps {
                            LabeledContent("Camera", value: "\(Int(fps.rounded())) frames a second")
                        }
                        if let online = status.online {
                            LabeledContent("Cloud voice", value: online ? "Available" : "Offline")
                        }
                        if let cal = status.cal {
                            LabeledContent("Table set up", value: cal ? "Yes" : "No")
                        }
                        if let laser = status.laser_cal {
                            LabeledContent("Laser set up", value: laser ? "Yes" : "No")
                        }
                    }
                }

                Section {
                    Button("Show them again") { store.restoreNotices() }
                        .disabled(store.dismissedNotices.isEmpty)
                } header: {
                    Text("Notices")
                } footer: {
                    Text(store.dismissedNotices.isEmpty
                         ? "No notices have been put away."
                         : "\(store.dismissedNotices.count) put away with “Got it”.")
                }
            }
            .onDisappear(perform: saveKey)
            .navigationTitle("Helper settings")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .confirmationAction) {
                    Button("Done") { dismiss() }
                }
            }
        }
    }

    private func saveKey() {
        let key = keyDraft.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !key.isEmpty else { return }
        Speaker.apiKey = key
        keyDraft = ""
        hasKey = true
    }

    private var connection: String {
        if store.isMock { return "Demo mode" }
        switch store.link {
        case .connected: return "Connected"
        case .connecting: return "Connecting"
        case .reconnecting: return "Reconnecting"
        case .searching: return "Looking for the rig"
        case .bluetoothOff: return "Bluetooth is off"
        case .unauthorized: return "Bluetooth not allowed"
        case .unsupported: return "No Bluetooth on this device"
        }
    }
}

/// The gear next to the status pill that opens `HelperSettings`.
struct HelperSettingsButton: View {
    let store: RoomStore
    @State private var open = false

    var body: some View {
        Button { open = true } label: {
            Image(systemName: "gearshape")
                .font(.title3)
                .frame(width: 44, height: 44)
                .contentShape(Rectangle())
        }
        .foregroundStyle(.secondary)
        .accessibilityLabel("Helper settings")
        .sheet(isPresented: $open) { HelperSettings(store: store) }
    }
}

#Preview {
    HelperSettings(store: RoomStore(mock: true))
}
