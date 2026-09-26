import SwiftUI

/// What's open over Home or Recent: an answer, or one thing.
enum Focus: Equatable {
    /// A question by id; nil follows the latest one.
    case answer(Int?)
    case thing(String)
}

/// One thing at a time, over the screen the person was on, so asking never jumps to another tab:
/// the answer (or where a thing is), the map with it lit up until they close it, and big buttons.
struct FocusSheet: View {
    let store: RoomStore
    @Binding var focus: Focus?
    /// "Show the whole table": switch to the Table tab, lighting up this thing if there is one.
    var showTable: (String?) -> Void

    /// Keeps the content while the sheet slides away after `focus` goes nil.
    @State private var last: Focus = .answer(nil)
    @State private var highlight: Highlight?
    @State private var details: String?

    private var current: Focus { focus ?? last }

    private var exchange: Exchange? {
        guard case .answer(let id) = current else { return nil }
        guard let id else { return store.current }
        return store.exchanges.first { $0.id == id }
    }

    /// The thing this is about: the one tapped, or the one the answer points at.
    private var entityName: String? {
        switch current {
        case .thing(let name): return name
        case .answer: return exchange?.answer?.pointAt
        }
    }

    private var askable: Entity? {
        guard case .thing(let name) = current, let e = store.snapshot?.entity(named: name),
              e.kind == .target, !e.isThing || !e.aliases.isEmpty else { return nil }
        return e
    }

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(alignment: .leading, spacing: 16) {
                    switch current {
                    case .thing(let name):
                        ThingSummary(name: name, store: store)
                    case .answer:
                        if let exchange {
                            AnswerCard(exchange: exchange, onRetry: store.retry)
                        }
                    }
                    if let snapshot = store.snapshot {
                        TableMapView(snapshot: snapshot, highlight: highlight,
                                     greyed: store.isRoomAppDown, steady: true, showsLaser: false) { name in
                            focus = .thing(name)
                        }
                    }
                    actions
                }
                .padding(16)
                .animation(.easeOut(duration: 0.2), value: exchange)
            }
            .navigationTitle(title)
            .navigationBarTitleDisplayMode(.inline)
            .safeAreaInset(edge: .bottom) {
                Button { focus = nil } label: {
                    Text("Done").frame(maxWidth: .infinity)
                }
                .buttonStyle(.borderedProminent)
                .controlSize(.extraLarge)
                .padding(.horizontal, 16)
                .padding(.vertical, 8)
                .background(.bar)
            }
        }
        .onChange(of: focus, initial: true) {
            if let focus { last = focus }
            updateHighlight()
        }
        .onChange(of: exchange?.answer) { updateHighlight() }
        .entityDetail($details, store: store)
    }

    private var title: String {
        switch current {
        case .answer: return "Answer"
        case .thing(let name):
            return Dashboard.capitalized(store.snapshot?.entity(named: name)?.displayName ?? Entity.displayName(for: name))
        }
    }

    private var actions: some View {
        VStack(spacing: 10) {
            if let e = askable {
                Button {
                    store.ask(Dashboard.question(for: e))
                    focus = .answer(nil)
                } label: {
                    Label("Ask the room where it is", systemImage: "questionmark.bubble")
                        .frame(maxWidth: .infinity)
                }
            }
            if let name = entityName, store.snapshot?.entity(named: name) != nil {
                Button { details = name } label: {
                    Label("More about it", systemImage: "info.circle").frame(maxWidth: .infinity)
                }
            }
            Button { showTable(entityName) } label: {
                Label("Show the whole table", systemImage: "square.grid.3x2").frame(maxWidth: .infinity)
            }
        }
        .buttonStyle(.bordered)
        // Full-contrast labels; only Done carries the accent.
        .tint(.primary)
        .controlSize(.extraLarge)
    }

    private func updateHighlight() {
        switch current {
        case .thing(let name):
            let target = store.snapshot?.entity(named: name).flatMap(MapLayout.position(of:))
            highlight = target.map { Highlight(entity: name, target: $0, action: .point) }
        case .answer:
            highlight = exchange?.answer.flatMap(store.highlight(for:))
        }
    }
}

/// Where one thing is, in words: the top of the sheet, and the card under the Table map.
struct ThingSummary: View {
    let name: String
    let store: RoomStore

    var body: some View {
        if let snapshot = store.snapshot, let e = snapshot.entity(named: name) {
            HStack(alignment: .top, spacing: 14) {
                ThingIconView(icon: IconStore.shared.icon(for: e.name, title: e.displayName), size: 36)
                    .frame(width: 56, height: 56)
                    .background(Circle().fill(Theme.iconWell))
                VStack(alignment: .leading, spacing: 4) {
                    Text(Dashboard.capitalized(e.displayName))
                        .font(.title2.bold())
                    Text(Dashboard.whereabouts(e, in: snapshot))
                        .font(.title3)
                    if let seen = Dashboard.lastSeen(e, now: snapshot.time ?? Date()) {
                        Text(seen).font(.body).foregroundStyle(.secondary)
                    }
                }
                .fixedSize(horizontal: false, vertical: true)
                Spacer(minLength: 0)
            }
            .padding(14)
            .background(Color(.secondarySystemBackground), in: RoundedRectangle(cornerRadius: 18))
            .accessibilityElement(children: .combine)
        } else {
            Text("The room can't see that right now.")
                .font(.title3)
                .foregroundStyle(.secondary)
        }
    }
}

#Preview("Thing") {
    FocusSheet(store: RoomStore(mock: true), focus: .constant(.thing("keys")), showTable: { _ in })
}
