import AVFoundation
import SwiftUI

/// For a family member or carer: demo mode, speech and the rig's details, kept off the main
/// screens but behind a labelled button rather than a hidden gesture.
struct HelperSettings: View {
    let store: RoomStore
    @AppStorage(Speaker.enabledKey) private var readAloud = false
    @AppStorage(Speaker.engineKey) private var engine = Speaker.Engine.grok
    @AppStorage(Speaker.grokVoiceKey) private var grokVoice = Grok.defaultVoice
    @AppStorage(Speaker.speedKey) private var speed = 1.0
    @AppStorage(Speaker.voiceIDKey) private var voiceID = ""
    @State private var grokKeyDraft = ""
    @State private var hasGrokKey = Speaker.grokKey != nil
    @State private var keyDraft = ""
    @State private var hasKey = Speaker.apiKey != nil
    @State private var voices = Grok.knownVoices
    @State private var route = VoiceRoute.current
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
                    Text("The rig already says its answers out loud. Read aloud is for using the phone away from the table, and stays quiet while the rig's speaker is on.")
                }

                voiceSection

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
            .onDisappear(perform: saveKeys)
            .task(id: hasGrokKey) { voices = await Grok.fetchVoices(key: Speaker.grokKey) }
            // The rig's speaker uses the same voice (PROTOCOL.md section 5a).
            .onChange(of: engine) { store.sendVoiceSettings() }
            .onChange(of: grokVoice) { store.sendVoiceSettings() }
            .onChange(of: speed) { store.sendVoiceSettings() }
            .onReceive(NotificationCenter.default.publisher(for: AVAudioSession.routeChangeNotification)
                .receive(on: RunLoop.main)) { _ in route = VoiceRoute.current }
            .navigationTitle("Helper settings")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .confirmationAction) {
                    Button("Done") { dismiss() }
                }
            }
        }
    }

    private func saveKeys() {
        if let key = trimmed(grokKeyDraft) {
            Speaker.grokKey = key
            grokKeyDraft = ""
            hasGrokKey = true
        }
        if let key = trimmed(keyDraft) {
            Speaker.apiKey = key
            keyDraft = ""
            hasKey = true
        }
    }

    private func trimmed(_ draft: String) -> String? {
        let key = draft.trimmingCharacters(in: .whitespacesAndNewlines)
        return key.isEmpty ? nil : key
    }

    @ViewBuilder
    private var voiceSection: some View {
        Section {
            Picker("Voice", selection: $engine) {
                ForEach(Speaker.Engine.allCases) { Text($0.title).tag($0) }
            }
            switch engine {
            case .grok:
                Picker("Grok voice", selection: $grokVoice) {
                    ForEach(voiceChoices) { Text($0.name).tag($0.id) }
                }
                keyField(hasGrokKey ? "xAI API key (saved)" : "xAI API key", draft: $grokKeyDraft)
                if hasGrokKey {
                    Button("Forget the xAI key", role: .destructive) {
                        Speaker.grokKey = nil
                        hasGrokKey = false
                    }
                }
            case .rigVoice:
                TextField("ElevenLabs voice ID", text: $voiceID)
                    .textInputAutocapitalization(.never)
                    .autocorrectionDisabled()
                keyField(hasKey ? "ElevenLabs API key (saved)" : "ElevenLabs API key", draft: $keyDraft)
                if hasKey {
                    Button("Forget the ElevenLabs key", role: .destructive) {
                        Speaker.apiKey = nil
                        hasKey = false
                    }
                }
            case .builtIn:
                EmptyView()
            }
            VStack(alignment: .leading) {
                LabeledContent("Speed", value: speed.formatted(.number.precision(.fractionLength(1))) + "×")
                Slider(value: $speed, in: Grok.speeds, step: 0.1) {
                    Text("Speed")
                } minimumValueLabel: {
                    Image(systemName: "tortoise").accessibilityLabel("Slower")
                } maximumValueLabel: {
                    Image(systemName: "hare").accessibilityLabel("Faster")
                }
            }
            LabeledContent("Playing through", value: route.summary)
            Button("Try the voice") {
                saveKeys()
                Speaker.shared.speak("Your keys are inside the box.")
            }
        } header: {
            Text("Voice")
        } footer: {
            Text(voiceFooter)
        }
    }

    private var voiceChoices: [Grok.Voice] {
        voices.contains { $0.id.caseInsensitiveCompare(grokVoice) == .orderedSame }
            ? voices : voices + [Grok.Voice(id: grokVoice, name: grokVoice.capitalized)]
    }

    private func keyField(_ title: String, draft: Binding<String>) -> some View {
        SecureField(title, text: draft)
            .textInputAutocapitalization(.never)
            .autocorrectionDisabled()
            .onSubmit(saveKeys)
    }

    private var voiceFooter: String {
        let adapts = "The voice is matched to where it plays: a little slower on the phone's speaker or across a room, full quality in headphones, and it stops if headphones are unplugged."
        switch engine {
        case .grok:
            return "Grok's voice from xAI. The key stays in this phone's Keychain. When read aloud is on, answer text goes to xAI; without a key or internet the iPhone's own voice reads it. " + adapts
        case .rigVoice:
            return "Use the same ElevenLabs voice ID as the rig (ELEVENLABS_VOICE_ID) so the phone sounds like the room. The key stays in this phone's Keychain. When read aloud is on, answer text goes to ElevenLabs; without a key or internet the iPhone's own voice reads it. " + adapts
        case .builtIn:
            return "The iPhone's own voice. Nothing leaves the phone. " + adapts
        }
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
        button.onAppear {
            if store.isMock, UserDefaults.standard.bool(forKey: "mockSettings") { open = true }
        }
    }

    private var button: some View {
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
