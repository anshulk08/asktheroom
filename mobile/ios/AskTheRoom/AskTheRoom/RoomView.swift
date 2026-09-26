import SwiftUI

/// Connect until the rig first connects, then Room for everything else (spec section 5).
struct RootView: View {
    let store: RoomStore

    var body: some View {
        if store.isMock || store.hasConnected {
            MainView(store: store)
        } else {
            ConnectView(store: store)
        }
    }
}

struct RoomView: View {
    let store: RoomStore
    @State private var selected: String?

    var body: some View {
        VStack(spacing: 0) {
            HStack {
                Text("Ask the Room").font(.title2.bold()).minimumScaleFactor(0.6)
                Spacer()
                StatusPill(store: store)
            }
            .lineLimit(1)
            .dynamicTypeSize(...DynamicTypeSize.accessibility1)
            .padding(.horizontal, 16)
            .padding(.top, 8)

            Banners(store: store)
                .padding(.horizontal, 16)
                .padding(.top, 8)

            Group {
                if let snapshot = store.snapshot {
                    TableMapView(snapshot: snapshot, highlight: store.highlight,
                                 greyed: store.isRoomAppDown) { selected = $0 }
                } else {
                    ContentUnavailableView("Waiting for the room", systemImage: "rectangle.dashed",
                                           description: Text("The map appears once the rig sends what it sees."))
                        .frame(maxHeight: 260)
                }
            }
            .padding(.horizontal, 4)

            ScrollView {
                VStack(spacing: 14) {
                    if store.showRoomVoiceAnswers, let heard = store.heardInRoom {
                        HeardInRoomCard(answer: heard)
                    }
                    if let current = store.current {
                        AnswerCard(exchange: current, onRetry: store.retry)
                    }
                    HistoryList(exchanges: store.history)
                }
                .padding(.horizontal, 16)
                .padding(.vertical, 8)
                .animation(.easeOut(duration: 0.2), value: store.exchanges)
            }
            .scrollDismissesKeyboard(.interactively)

            AskPanel(onAsk: store.ask)
        }
        .entityDetail($selected, store: store)
    }
}

private struct SelectedEntity: Identifiable {
    let id: String
}

extension View {
    /// The detail sheet for a tapped thing, on the map or on Home.
    func entityDetail(_ selected: Binding<String?>, store: RoomStore) -> some View {
        sheet(item: Binding(get: { selected.wrappedValue.map(SelectedEntity.init) },
                            set: { selected.wrappedValue = $0?.id })) { pick in
            EntityDetailView(name: pick.id, store: store)
                .presentationDetents([.medium, .large])
        }
    }
}

// MARK: Status

/// Connection state; long-press toggles mock mode (spec section 7).
struct StatusPill: View {
    let store: RoomStore

    private var label: (String, Color) {
        if store.isMock { return ("Demo mode", .orange) }
        switch store.link {
        case .connected: return ("Connected", .green)
        case .connecting: return ("Connecting…", .yellow)
        case .reconnecting: return ("Reconnecting…", .yellow)
        case .searching: return ("Looking…", .yellow)
        case .bluetoothOff, .unauthorized, .unsupported: return ("No Bluetooth", .red)
        }
    }

    var body: some View {
        HStack(spacing: 6) {
            Circle().fill(label.1).frame(width: 9, height: 9)
            Text(label.0).font(.footnote.weight(.medium))
        }
        .padding(.horizontal, 10)
        .padding(.vertical, 6)
        .background(Capsule().fill(Color(.secondarySystemBackground)))
        .onLongPressGesture(minimumDuration: 0.8) { store.setMock(!store.isMock) }
        .accessibilityElement(children: .combine)
        .accessibilityHint("Long-press to switch demo mode \(store.isMock ? "off" : "on")")
        .accessibilityAction(named: store.isMock ? "Turn off demo mode" : "Turn on demo mode") {
            store.setMock(!store.isMock)
        }
    }
}

struct Banners: View {
    let store: RoomStore

    var body: some View {
        VStack(spacing: 8) {
            switch store.link {
            case .bluetoothOff where !store.isMock, .unauthorized where !store.isMock, .unsupported where !store.isMock:
                BluetoothExplainer(state: store.link)
            default:
                EmptyView()
            }
            if store.isRoomAppDown {
                Banner(icon: "exclamationmark.triangle.fill", tint: .orange,
                       text: "The room app isn't running. The map may be out of date.")
            } else if store.isOffline {
                Banner(icon: "icloud.slash", tint: .secondary, text: "Offline: using the on-device voice.")
            }
        }
    }
}

struct Banner: View {
    let icon: String
    let tint: Color
    let text: String

    var body: some View {
        Label {
            Text(text).font(.subheadline.weight(.medium)).fixedSize(horizontal: false, vertical: true)
        } icon: {
            Image(systemName: icon).foregroundStyle(tint)
        }
        .padding(10)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(Color(.secondarySystemBackground), in: RoundedRectangle(cornerRadius: 12))
    }
}

/// Full-width explainer when Bluetooth is off or not allowed.
struct BluetoothExplainer: View {
    let state: LinkState
    @Environment(\.openURL) private var openURL

    private var message: String {
        switch state {
        case .unauthorized: return "Ask the Room needs Bluetooth permission to reach the rig on the table."
        case .unsupported: return "This device can't use Bluetooth Low Energy."
        default: return "Turn on Bluetooth to connect to the rig on the table."
        }
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            Label("Bluetooth is needed", systemImage: "antenna.radiowaves.left.and.right.slash")
                .font(.headline)
            Text(message).font(.subheadline).fixedSize(horizontal: false, vertical: true)
            if state != .unsupported {
                Button("Open Settings") {
                    if let url = URL(string: UIApplication.openSettingsURLString) { openURL(url) }
                }
                .buttonStyle(.borderedProminent)
            }
        }
        .padding(14)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(Color(.secondarySystemBackground), in: RoundedRectangle(cornerRadius: 16))
    }
}

// MARK: Connect

struct ConnectView: View {
    let store: RoomStore
    @State private var showTips = false
    @State private var pulsing = false
    @Environment(\.accessibilityReduceMotion) private var reduceMotion

    var body: some View {
        VStack(spacing: 24) {
            HStack {
                Spacer()
                StatusPill(store: store)
            }
            Spacer()
            if store.link == .bluetoothOff || store.link == .unauthorized || store.link == .unsupported {
                BluetoothExplainer(state: store.link)
                if store.link == .unsupported {
                    Button("Use demo mode") { store.setMock(true) }
                        .buttonStyle(.bordered)
                }
            } else {
                ZStack {
                    Circle().stroke(Theme.laser.opacity(0.35), lineWidth: 2)
                        .frame(width: 120, height: 120)
                        .scaleEffect(pulsing ? 1.25 : 0.9)
                        .opacity(pulsing ? 0.2 : 1)
                    Circle().fill(Theme.laser).frame(width: 18, height: 18)
                }
                .onAppear {
                    guard !reduceMotion else { return }
                    withAnimation(.easeInOut(duration: 1.4).repeatForever(autoreverses: true)) { pulsing = true }
                }
                .accessibilityHidden(true)
                Text(store.link == .connecting ? "Connecting…" : "Looking for the room…")
                    .font(.title2.weight(.semibold))
                if showTips {
                    VStack(alignment: .leading, spacing: 8) {
                        Label("Check Bluetooth is on.", systemImage: "antenna.radiowaves.left.and.right")
                        Label("Move closer to the rig on the table.", systemImage: "figure.walk")
                        Label("Make sure the rig is switched on.", systemImage: "power")
                    }
                    .font(.body)
                    .foregroundStyle(.secondary)
                    .transition(.opacity)
                }
            }
            Spacer()
        }
        .padding(24)
        .task {
            try? await Task.sleep(for: .seconds(10))
            withAnimation { showTips = true }
        }
    }
}

// MARK: Detail

struct EntityDetailView: View {
    let name: String
    let store: RoomStore

    var body: some View {
        NavigationStack {
            Group {
                if let snapshot = store.snapshot, let entity = snapshot.entity(named: name) {
                    List {
                        Section {
                            LabeledContent("Status", value: EntityDetailView.statusWords(entity, in: snapshot))
                            if let seen = entity.lastSeen {
                                LabeledContent("Last seen", value: EntityDetailView.ago(seen, now: snapshot.time ?? Date()))
                            }
                            if entity.confidence < 1 {
                                LabeledContent("Confidence", value: "\(Int((entity.confidence * 100).rounded()))%")
                            }
                        }
                        let chain = snapshot.chain(from: entity.name).dropFirst()
                        if !chain.isEmpty {
                            Section("Where it is") {
                                ForEach(Array(zip(snapshot.chain(from: entity.name), chain)), id: \.1.id) { child, parent in
                                    LabeledContent(child.displayName,
                                                   value: "\(child.status == .under ? "under" : "inside") \(parent.displayName)")
                                }
                            }
                        }
                        if !entity.aliases.isEmpty {
                            Section("Also called") {
                                ForEach(entity.aliases, id: \.self) { Text($0) }
                            }
                        }
                        if !entity.maybeSameAs.isEmpty {
                            Section("Might be") {
                                ForEach(entity.maybeSameAs, id: \.name) { m in
                                    LabeledContent(snapshot.entity(named: m.name)?.displayName ?? Entity.displayName(for: m.name),
                                                   value: "\(Int((m.score * 100).rounded()))% alike")
                                }
                            }
                        }
                    }
                } else {
                    ContentUnavailableView("No longer on the map", systemImage: "questionmark.circle")
                }
            }
            .navigationTitle(store.snapshot?.entity(named: name)?.displayName ?? Entity.displayName(for: name))
            .navigationBarTitleDisplayMode(.inline)
        }
    }

    static func statusWords(_ e: Entity, in snapshot: Snapshot) -> String {
        let parent = e.parent.map { snapshot.entity(named: $0)?.displayName ?? Entity.displayName(for: $0) }
        let words: String
        switch e.status {
        case .visible: words = "On the table"
        case .held: words = "In someone's hand"
        case .inside: words = "Inside \(parent ?? "something")"
        case .under: words = "Under \(parent ?? "something")"
        case .gone: words = e.edge.map { "Left the table (\($0.rawValue) edge)" } ?? "Left the table"
        case .lost: words = "Lost track of it"
        case .unrecognized: words = "Unknown"
        }
        return e.isUncertain && e.status != .lost ? "Probably: " + words.prefix(1).lowercased() + words.dropFirst() : words
    }

    static func ago(_ date: Date, now: Date) -> String {
        let seconds = max(0, now.timeIntervalSince(date))
        if seconds < 60 { return "Just now" }
        let f = RelativeDateTimeFormatter()
        f.unitsStyle = .short
        return f.localizedString(for: date, relativeTo: now)
    }
}

#Preview("Room, mock") {
    RootView(store: RoomStore(mock: true))
}
