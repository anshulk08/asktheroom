import SwiftUI

/// Connect until the rig first connects, then Room for everything else (spec section 5).
/// With a map saved from last time, the app opens straight on it while it finds the rig.
struct RootView: View {
    let store: RoomStore

    var body: some View {
        if store.isMock || store.hasConnected || store.snapshot != nil {
            MainView(store: store)
        } else {
            ConnectView(store: store)
        }
    }
}

/// The Table tab: the map, a short key, and one thing at a time underneath it (what was
/// tapped, or the latest answer), so the screen never shows more than one story.
struct RoomView: View {
    let store: RoomStore
    /// The thing tapped on the map; MainView sets it for "Show the whole table".
    @Binding var selected: String?
    @State private var details: String?
    @Environment(\.dynamicTypeSize) private var typeSize

    /// At the largest text sizes the key and suggestions scroll with the card, so the map
    /// and the card aren't squeezed out.
    private var compact: Bool { typeSize.isAccessibilitySize }

    var body: some View {
        VStack(spacing: 0) {
            HStack {
                Text("Ask the Room").font(.title2.bold()).minimumScaleFactor(0.6)
                Spacer()
                StatusPill(store: store)
                HelperSettingsButton(store: store)
            }
            .lineLimit(1)
            .dynamicTypeSize(...DynamicTypeSize.accessibility1)
            .padding(.horizontal, 16)
            .padding(.top, 8)

            Banners(store: store)
                .padding(.horizontal, 16)
                .padding(.top, 8)

            if let snapshot = store.snapshot {
                TableMapView(snapshot: snapshot, highlight: store.highlight,
                             greyed: store.isRoomAppDown || store.isMapStale, selected: selected) { name in
                    selected = selected == name ? nil : name
                }
                .padding(.horizontal, 4)
                // The map keeps its full width; the card area below scrolls instead.
                .layoutPriority(1)
                if !compact {
                    legend(for: snapshot)
                        .padding(.horizontal, 16)
                }
            } else {
                ContentUnavailableView("Waiting for the room", systemImage: "rectangle.dashed",
                                       description: Text("The map appears once the rig sends what it sees."))
                    .frame(maxHeight: 260)
            }

            ScrollView {
                VStack(spacing: 14) {
                    if let name = selected, store.snapshot?.entity(named: name) != nil {
                        SelectedThingCard(name: name, store: store,
                                          onAsk: { question in selected = nil; store.ask(question) },
                                          onMore: { details = name },
                                          onClose: { selected = nil })
                    } else if let current = store.current {
                        AnswerCard(exchange: current, onRetry: store.retry)
                    } else if store.snapshot != nil {
                        Label("Tap something on the map to see where it is.", systemImage: "hand.tap")
                            .font(.body)
                            .foregroundStyle(.secondary)
                            .frame(maxWidth: .infinity, alignment: .leading)
                            .padding(.top, 4)
                    }
                    if store.showRoomVoiceAnswers, let heard = store.heardInRoom {
                        HeardInRoomCard(answer: heard)
                    }
                    if compact, let snapshot = store.snapshot {
                        legend(for: snapshot)
                    }
                    if compact {
                        SuggestionChips { question in
                            selected = nil
                            store.ask(question)
                        }
                        .padding(.horizontal, -16)
                    }
                }
                .padding(.horizontal, 16)
                .padding(.vertical, 12)
                .animation(.easeOut(duration: 0.2), value: selected)
                .animation(.easeOut(duration: 0.2), value: store.exchanges)
            }
            .scrollDismissesKeyboard(.interactively)

            AskPanel(showSuggestions: !compact) { question in
                selected = nil
                store.ask(question)
            }
        }
        .entityDetail($details, store: store)
    }

    private func legend(for snapshot: Snapshot) -> some View {
        MapLegend(entries: MapLayout.legend(for: MapLayout.items(for: snapshot)))
            .frame(maxWidth: .infinity, alignment: .leading)
    }
}

/// What was tapped on the map, in words, with what can be done about it.
private struct SelectedThingCard: View {
    let name: String
    let store: RoomStore
    var onAsk: (String) -> Void
    var onMore: () -> Void
    var onClose: () -> Void

    private var askable: Entity? {
        guard let e = store.snapshot?.entity(named: name), e.kind == .target,
              !e.isThing || !e.aliases.isEmpty else { return nil }
        return e
    }

    var body: some View {
        VStack(spacing: 10) {
            ThingSummary(name: name, store: store)
                .overlay(alignment: .topTrailing) {
                    Button(action: onClose) {
                        Image(systemName: "xmark")
                            .font(.body.weight(.semibold))
                            .frame(width: 44, height: 44)
                            .contentShape(Rectangle())
                    }
                    .foregroundStyle(.secondary)
                    .accessibilityLabel("Close")
                }
            ViewThatFits(in: .horizontal) {
                HStack(spacing: 10) { buttons }
                VStack(spacing: 10) { buttons }
            }
            .buttonStyle(.bordered)
            .tint(.primary)
            .controlSize(.large)
        }
    }

    @ViewBuilder private var buttons: some View {
        if let e = askable {
            Button { onAsk(Dashboard.question(for: e)) } label: {
                Label("Ask the room", systemImage: "questionmark.bubble").frame(maxWidth: .infinity)
            }
        }
        Button(action: onMore) {
            Label("More about it", systemImage: "info.circle").frame(maxWidth: .infinity)
        }
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

/// Connection state. Demo mode is switched in helper settings, not by a hidden gesture;
/// VoiceOver keeps a direct action for it.
struct StatusPill: View {
    let store: RoomStore

    private var label: (String, Color) {
        // Words carry the meaning; the dot is only green when all is well.
        if store.isMock { return ("Demo mode", .gray) }
        switch store.link {
        case .connected: return ("Connected", .green)
        case .connecting: return ("Connecting…", .gray)
        case .reconnecting: return ("Reconnecting…", .gray)
        case .searching: return ("Looking…", .gray)
        case .bluetoothOff, .unauthorized, .unsupported: return ("No Bluetooth", .gray)
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
        .accessibilityElement(children: .combine)
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
            if store.isMapStale, let time = store.snapshot?.time {
                Banner(icon: "clock.arrow.circlepath", tint: .secondary,
                       text: "This is the map from \(Banners.when(time)). It updates when the room is back in reach.")
            }
            if store.isRoomAppDown {
                Banner(icon: "exclamationmark.triangle", tint: .secondary,
                       text: "The room app isn't running. The map may be out of date.")
            } else if store.isOffline {
                Banner(icon: "icloud.slash", tint: .secondary, text: "Offline: using the on-device voice.")
            }
        }
    }

    /// "1:05 PM" today, "Friday 1:05 PM" before that.
    static func when(_ date: Date, now: Date = Date(), calendar: Calendar = .current) -> String {
        calendar.isDate(date, inSameDayAs: now)
            ? date.formatted(.dateTime.hour().minute())
            : date.formatted(.dateTime.weekday(.wide).hour().minute())
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
                HelperSettingsButton(store: store)
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
    @State private var picking = UserDefaults.standard.bool(forKey: "mockIconPicker")

    var body: some View {
        NavigationStack {
            Group {
                if let snapshot = store.snapshot, let entity = snapshot.entity(named: name) {
                    List {
                        Section {
                            Button { picking = true } label: {
                                HStack(spacing: 14) {
                                    ThingIconView(icon: IconStore.shared.icon(for: name, title: entity.displayName), size: 30)
                                        .frame(width: 48, height: 48)
                                        .background(Circle().fill(Theme.iconWell))
                                    Text("Change picture")
                                }
                            }
                            .tint(.primary)
                        }
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
                        if let guess = entity.hedgedName, let number = entity.thingNumber {
                            Section("Not named yet") {
                                LabeledContent("Looks like", value: guess)
                                LabeledContent("Object", value: number)
                            }
                        }
                        if !entity.aliases.isEmpty {
                            Section("Also called") {
                                ForEach(entity.aliases, id: \.self) { Text($0) }
                            }
                        }
                        let matches = snapshot.knownMatches(of: entity)
                        if !matches.isEmpty {
                            Section("Might be") {
                                ForEach(matches, id: \.name) { m in
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
            .navigationTitle(title)
            .navigationBarTitleDisplayMode(.inline)
            .sheet(isPresented: $picking) { IconPicker(name: name, title: title) }
        }
    }

    private var title: String {
        store.snapshot?.entity(named: name)?.displayName ?? Entity.displayName(for: name)
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
