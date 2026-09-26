import SwiftUI

/// For a family member or carer: demo mode, speech and the rig's details, kept off the main
/// screens but behind a labelled button rather than a hidden gesture.
struct HelperSettings: View {
    let store: RoomStore
    @AppStorage(Speaker.enabledKey) private var readAloud = false
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
            .navigationTitle("Helper settings")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .confirmationAction) {
                    Button("Done") { dismiss() }
                }
            }
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
