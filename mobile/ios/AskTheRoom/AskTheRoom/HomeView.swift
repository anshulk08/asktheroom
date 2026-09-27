import SwiftUI

/// Home, the Table map and Recent as tabs. Asking from Home, or tapping something on Recent,
/// opens the answer over that screen; only "Show the whole table" switches tabs.
struct MainView: View {
    enum Tab: String { case home, table, recent }

    let store: RoomStore
    /// `-mockTab table` or `-mockTab recent` opens on that tab, for screenshots.
    @State private var tab = Tab(rawValue: UserDefaults.standard.string(forKey: "mockTab") ?? "") ?? .home
    /// What's open over Home or Recent; nil when nothing is.
    @State private var focus: Focus?
    /// What's picked on the Table map. `-mockTab table -mockSelect name` picks it at launch.
    @State private var tablePick: String? = UserDefaults.standard.string(forKey: "mockTab") == "table"
        ? UserDefaults.standard.string(forKey: "mockSelect") : nil

    var body: some View {
        TabView(selection: $tab) {
            HomeView(store: store, ask: store.ask, show: { focus = .thing($0) })
                .tabItem { Label("Home", systemImage: "house.fill") }
                .tag(Tab.home)
            RoomView(store: store, selected: $tablePick)
                .tabItem { Label("Table", systemImage: "square.grid.3x2.fill") }
                .tag(Tab.table)
            RecentView(store: store) { entry in
                switch entry {
                case .change(let event): focus = .thing(event.entity)
                case .question(let exchange): focus = .answer(exchange.id)
                }
            }
                .tabItem { Label("Recent", systemImage: "clock") }
                .tag(Tab.recent)
        }
        // The Table tab shows answers in place; elsewhere a new question opens its answer.
        .onChange(of: store.current?.id) { _, id in
            if id != nil, tab != .table { focus = .answer(nil) }
        }
        // Off unless a helper turns it on, and never while the rig's speaker is on: the room hears the rig.
        .onChange(of: store.current?.answer) { _, answer in
            if let answer, store.status?.rigSpeaks != true { Speaker.shared.say(answer.text) }
        }
        .task {
            if store.isMock, let name = UserDefaults.standard.string(forKey: "mockFocus") { focus = .thing(name) }
        }
        .sheet(isPresented: Binding(get: { focus != nil }, set: { if !$0 { focus = nil } })) {
            FocusSheet(store: store, focus: $focus) { name in
                focus = nil
                if let name { store.showOnMap(name) }
                tablePick = name
                tab = .table
            }
        }
    }
}

/// The start screen: the day and where each thing is.
/// Everything is a big tap target that asks the room.
struct HomeView: View {
    let store: RoomStore
    var ask: (String) -> Void
    var show: (String) -> Void
    @State private var selected: String?
    @Environment(\.dynamicTypeSize) private var typeSize

    private var columns: [GridItem] {
        // One column at the largest text sizes, so names never squeeze.
        typeSize.isAccessibilitySize ? [GridItem(.flexible())] : [GridItem(.adaptive(minimum: 150), spacing: 12)]
    }

    var body: some View {
        ScrollViewReader { proxy in
        ScrollView {
            VStack(alignment: .leading, spacing: 22) {
                DayHeader(store: store)
                Banners(store: store)

                HomeSection(title: "Your things") {
                    if let snapshot = store.snapshot {
                        LazyVGrid(columns: columns, spacing: 12) {
                            ForEach(Dashboard.things(in: snapshot)) { thing in
                                ThingTile(thing: thing, snapshot: snapshot,
                                          onTap: { ask(Dashboard.question(for: thing)) },
                                          onDetails: { selected = thing.name })
                            }
                        }
                    } else {
                        Text("Waiting for the room to see the table…")
                            .font(.body)
                            .foregroundStyle(.secondary)
                    }
                }

                Color.clear.frame(height: 1).id("end")
            }
            .padding(.horizontal, 16)
            .padding(.top, 8)
            .padding(.bottom, 16)
        }
        .scrollDismissesKeyboard(.interactively)
        .safeAreaInset(edge: .top, spacing: 0) {
            // Keeps scrolled cards from running under the clock.
            Color.clear.frame(height: 0).background(.bar, ignoresSafeAreaEdges: .top)
        }
        .safeAreaInset(edge: .bottom) {
            // The tiles are the suggestions here.
            AskPanel(showSuggestions: false, onAsk: ask)
                .background(.bar)
        }
        .task {
            guard store.isMock else { return }
            if let name = UserDefaults.standard.string(forKey: "mockSelect"),
               UserDefaults.standard.string(forKey: "mockTab") != "table" { selected = name }
            if UserDefaults.standard.bool(forKey: "mockScroll") {
                try? await Task.sleep(for: .seconds(1))
                proxy.scrollTo("end", anchor: .bottom)
            }
        }
        }
        .entityDetail($selected, store: store)
    }
}

private struct HomeSection<Content: View>: View {
    let title: String
    @ViewBuilder var content: Content

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text(title)
                .font(.title3.bold())
                .accessibilityAddTraits(.isHeader)
            content
        }
    }
}

/// Greeting, today's date and the time, for orientation, with the connection pill.
private struct DayHeader: View {
    let store: RoomStore

    var body: some View {
        TimelineView(.everyMinute) { context in
            VStack(alignment: .leading, spacing: 2) {
                // Full width for the words, so the date is never cut short.
                Group {
                    Text(Dashboard.greeting(at: context.date))
                        .font(.largeTitle.bold())
                    Text(context.date.formatted(.dateTime.weekday(.wide).month(.wide).day()))
                        .font(.title3)
                        .foregroundStyle(.secondary)
                }
                .fixedSize(horizontal: false, vertical: true)
                HStack(alignment: .center) {
                    Text(context.date.formatted(date: .omitted, time: .shortened))
                        .font(.title.weight(.semibold))
                        .monospacedDigit()
                        .lineLimit(1)
                    Spacer()
                    HStack(spacing: 2) {
                        StatusPill(store: store)
                        HelperSettingsButton(store: store)
                    }
                    .dynamicTypeSize(...DynamicTypeSize.accessibility1)
                }
                .padding(.top, 2)
            }
        }
    }
}

/// One of the person's things: tap to ask where it is. Details are in the answer sheet
/// ("More about it"); the context menu is only a shortcut.
private struct ThingTile: View {
    let thing: Entity
    let snapshot: Snapshot
    let onTap: () -> Void
    let onDetails: () -> Void

    /// Grows with the text so big icons stay inside their circle.
    @ScaledMetric(relativeTo: .title2) private var well: CGFloat = 44

    private var isAway: Bool { [.gone, .lost].contains(thing.status) }
    private var isHidden: Bool { [.inside, .under, .held].contains(thing.status) }

    var body: some View {
        Button(action: onTap) {
            VStack(alignment: .leading, spacing: 8) {
                HStack {
                    ThingIconView(icon: IconStore.shared.icon(for: thing.name, title: thing.displayName), size: well * 0.62)
                        .frame(width: well, height: well)
                        .background(Circle().fill(Theme.iconWell))
                    Spacer()
                    if isHidden || isAway {
                        Image(systemName: isAway ? "questionmark.circle" : "eye.slash")
                            .foregroundStyle(.secondary)
                    }
                }
                Text(Dashboard.capitalized(thing.displayName))
                    .font(.title3.bold())
                    .foregroundStyle(.primary)
                    .lineLimit(2)
                Text(Dashboard.whereabouts(thing, in: snapshot))
                    .font(.body)
                    .foregroundStyle(.primary)
                    .fixedSize(horizontal: false, vertical: true)
                if let seen = Dashboard.lastSeen(thing, now: snapshot.time ?? Date()) {
                    Text(seen)
                        .font(.subheadline)
                        .foregroundStyle(.secondary)
                }
            }
            .multilineTextAlignment(.leading)
            .padding(14)
            .frame(maxWidth: .infinity, minHeight: 150, alignment: .topLeading)
            .background(Color(.secondarySystemBackground), in: RoundedRectangle(cornerRadius: 18))
            .opacity(thing.isUncertain ? 0.8 : 1)
        }
        .buttonStyle(.plain)
        .contextMenu {
            Button("Where is it?", systemImage: "scope", action: onTap)
            Button("Details", systemImage: "info.circle", action: onDetails)
        }
        .accessibilityElement(children: .combine)
        .accessibilityHint("Asks the room where it is")
        .accessibilityAction(named: "Details", onDetails)
    }
}

#Preview("Home, mock") {
    MainView(store: RoomStore(mock: true))
}
