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
    /// The camera side the person sits at, or empty for the rig's default.
    @AppStorage(Seat.savedKey) private var seat = ""
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

                seatSection

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

                if let stats = store.linkStats {
                    ConnectionSection(stats: stats)
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

    @ViewBuilder
    private var seatSection: some View {
        let view = store.snapshot?.view
        Section {
            SeatDiagram(view: view) { edge in
                guard let view else { return }
                Seat.choose(view.cameraSide(at: edge))
                store.sendSeat()
            }
            Button {
                Seat.choose(nil)
                store.sendSeat()
            } label: {
                HStack {
                    Text("Use the rig's default")
                    Spacer()
                    if seat.isEmpty {
                        Image(systemName: "checkmark").foregroundStyle(Theme.accent)
                    }
                }
            }
            .foregroundStyle(.primary)
            .accessibilityAddTraits(seat.isEmpty ? .isSelected : [])
        } header: {
            Text("I sit here")
        } footer: {
            Text(view == nil
                 ? "Left, right and the map follow where you sit. Connect to the rig to choose."
                 : "Left, right and the map follow where you sit. Tap the side of the table you sit at.")
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

/// How the Bluetooth link is doing, for someone chasing dropouts. Collapsed by default, at the bottom.
private struct ConnectionSection: View {
    let stats: LinkStats
    /// `-mockConnection YES` opens it, for screenshots.
    @State private var open = UserDefaults.standard.bool(forKey: "mockConnection")

    var body: some View {
        Section {
            DisclosureGroup(isExpanded: $open) {
                // Ticks once a second only while open, for the "ago" and "connected for" times.
                TimelineView(.periodic(from: .now, by: 1)) { context in
                    VStack(alignment: .leading, spacing: 6) {
                        line(stats.connectionLine(now: context.date))
                        line(stats.channelLine(.state))
                        line(stats.lossLine)
                        line(stats.lastUpdateLine(now: context.date))
                        line(stats.reconnectsLine)
                        line(stats.channelLine(.answer))
                        line(stats.channelLine(.status))
                    }
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .accessibilityElement(children: .contain)
                }
            } label: {
                Text("Connection")
            }
        } footer: {
            Text("Counts are since the phone last connected to the rig. Lost chunks are over the last minute.")
        }
    }

    private func line(_ text: String) -> some View {
        Text(text)
            .font(.footnote.monospacedDigit())
            .foregroundStyle(.secondary)
            .fixedSize(horizontal: false, vertical: true)
    }
}

/// The table as the map shows it, with the person always at the bottom. Each side is a button
/// that means "I sit here"; the rig then turns the map so that side comes to the bottom.
private struct SeatDiagram: View {
    let view: ViewInfo?
    let choose: (Edge) -> Void
    @Environment(\.colorScheme) private var scheme

    var body: some View {
        VStack(spacing: 6) {
            side(.top)
            HStack(spacing: 6) {
                side(.left)
                RoundedRectangle(cornerRadius: 10)
                    .fill(Theme.surface(scheme))
                    .overlay(RoundedRectangle(cornerRadius: 10).strokeBorder(.primary.opacity(0.2)))
                    .frame(width: 120, height: 80)
                    .accessibilityHidden(true)
                side(.right)
            }
            side(.bottom)
        }
        .frame(maxWidth: .infinity)
        .padding(.vertical, 6)
        .disabled(view == nil)
    }

    private func side(_ edge: Edge) -> some View {
        let name = view?.name(at: edge).map(\.localizedCapitalized)
        let isYou = edge == .bottom
        return Button { choose(edge) } label: {
            HStack(spacing: 4) {
                if isYou { Image(systemName: "person.fill") }
                Text(isYou ? ["You", name].compactMap { $0 }.joined(separator: " · ") : name ?? Self.word(for: edge))
                    .lineLimit(1)
                    .minimumScaleFactor(0.7)
            }
            .font(.footnote.weight(.semibold))
            .frame(minWidth: edge == .left || edge == .right ? 56 : 88, minHeight: 28)
        }
        .buttonStyle(.bordered)
        .tint(isYou ? Theme.accent : .secondary)
        .accessibilityLabel(Self.accessibilityLabel(for: edge, name: name))
        .accessibilityAddTraits(isYou ? .isSelected : [])
    }

    static func word(for edge: Edge) -> String {
        switch edge {
        case .top: return "Far"
        case .left: return "Left"
        case .right: return "Right"
        case .bottom: return "Near"
        }
    }

    /// "Sit at the far side, couch"; the bottom is where the person already sits.
    static func accessibilityLabel(for edge: Edge, name: String?) -> String {
        let place = "the \(word(for: edge).lowercased()) side" + (name.map { ", \($0.lowercased())" } ?? "")
        return edge == .bottom ? "You sit at \(place)" : "Sit at \(place)"
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
