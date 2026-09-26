import SwiftUI

/// Home, the Table map and Recent as tabs. Asking from Home, or tapping "Show me", jumps to the map.
struct MainView: View {
    enum Tab: String { case home, table, recent }

    let store: RoomStore
    /// `-mockTab table` opens on the map, for screenshots.
    @State private var tab = Tab(rawValue: UserDefaults.standard.string(forKey: "mockTab") ?? "") ?? .home

    var body: some View {
        TabView(selection: $tab) {
            HomeView(store: store,
                     ask: { q in
                         store.ask(q)
                         tab = .table
                     },
                     show: { name in
                         store.showOnMap(name)
                         tab = .table
                     })
                .tabItem { Label("Home", systemImage: "house.fill") }
                .tag(Tab.home)
            RoomView(store: store)
                .tabItem { Label("Table", systemImage: "square.grid.3x2.fill") }
                .tag(Tab.table)
            RecentView(store: store)
                .tabItem { Label("Recent", systemImage: "clock") }
                .tag(Tab.recent)
        }
    }
}

/// The start screen: the day, what the room noticed, and where each thing is.
/// Everything is a big tap target that asks the room.
struct HomeView: View {
    let store: RoomStore
    var ask: (String) -> Void
    var show: (String) -> Void
    @State private var selected: String?

    var body: some View {
        ScrollViewReader { proxy in
        ScrollView {
            VStack(alignment: .leading, spacing: 22) {
                DayHeader(store: store)
                Banners(store: store)

                if !store.notices.isEmpty {
                    HomeSection(title: "The room noticed") {
                        ForEach(store.notices) { notice in
                            NoticeCard(notice: notice,
                                       onShow: { notice.question.map(ask) ?? show(notice.entity) },
                                       onDismiss: { withAnimation { store.dismiss(notice) } })
                        }
                    }
                }

                HomeSection(title: "Your things") {
                    if let snapshot = store.snapshot {
                        LazyVGrid(columns: [GridItem(.adaptive(minimum: 150), spacing: 12)], spacing: 12) {
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
            .animation(.easeOut(duration: 0.25), value: store.notices)
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
            if let name = UserDefaults.standard.string(forKey: "mockSelect") { selected = name }
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

/// Greeting and today's date, for orientation, with the connection pill.
private struct DayHeader: View {
    let store: RoomStore

    var body: some View {
        TimelineView(.everyMinute) { context in
            HStack(alignment: .top) {
                VStack(alignment: .leading, spacing: 2) {
                    Text(Dashboard.greeting(at: context.date))
                        .font(.largeTitle.bold())
                        .minimumScaleFactor(0.6)
                    Text(context.date.formatted(.dateTime.weekday(.wide).month(.wide).day()))
                        .font(.title3)
                        .foregroundStyle(.secondary)
                }
                .lineLimit(1)
                .accessibilityElement(children: .combine)
                Spacer()
                StatusPill(store: store)
                    .dynamicTypeSize(...DynamicTypeSize.accessibility1)
            }
        }
    }
}

private struct NoticeCard: View {
    let notice: Notice
    let onShow: () -> Void
    let onDismiss: () -> Void

    private var icon: (String, HierarchicalShapeStyle) {
        switch notice.kind {
        case .leftTable: return ("arrow.left.square", .secondary)
        case .lostTrack: return ("questionmark.circle", .secondary)
        case .unnamed: return ("sparkles", .secondary)
        }
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack(alignment: .top, spacing: 12) {
                Image(systemName: icon.0)
                    .font(.title2)
                    .foregroundStyle(icon.1)
                    .accessibilityHidden(true)
                Text(notice.text)
                    .font(.title3.weight(.semibold))
                    .fixedSize(horizontal: false, vertical: true)
            }
            HStack(spacing: 10) {
                Button(action: onShow) {
                    Label(notice.kind == .unnamed ? "Show me" : "Help me find it", systemImage: "scope")
                        .frame(maxWidth: .infinity)
                }
                .buttonStyle(.borderedProminent)
                Button("Got it", action: onDismiss)
                    .buttonStyle(.bordered)
            }
            .controlSize(.large)
        }
        .padding(14)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(Color(.secondarySystemBackground), in: RoundedRectangle(cornerRadius: 18))
    }
}

/// One of the person's things: tap to ask where it is (the map lights it up), long-press for details.
private struct ThingTile: View {
    let thing: Entity
    let snapshot: Snapshot
    let onTap: () -> Void
    let onDetails: () -> Void

    private var isAway: Bool { [.gone, .lost].contains(thing.status) }
    private var isHidden: Bool { [.inside, .under, .held].contains(thing.status) }

    var body: some View {
        Button(action: onTap) {
            VStack(alignment: .leading, spacing: 8) {
                HStack {
                    Image(systemName: Dashboard.symbol(for: thing.name))
                        .font(.title2)
                        .frame(width: 44, height: 44)
                        .background(Circle().fill(Theme.iconWell))
                        .foregroundStyle(.primary)
                    Spacer()
                    if isHidden || isAway {
                        Image(systemName: isAway ? "questionmark.circle" : "eye.slash")
                            .foregroundStyle(.secondary)
                    }
                }
                Text(Dashboard.capitalized(thing.displayName))
                    .font(.headline)
                    .foregroundStyle(.primary)
                    .lineLimit(2)
                Text(Dashboard.whereabouts(thing, in: snapshot))
                    .font(.subheadline)
                    .foregroundStyle(.primary)
                    .fixedSize(horizontal: false, vertical: true)
                if isAway, let seen = thing.lastSeen {
                    Text("Seen \(Dashboard.ago(seen, now: snapshot.time ?? Date()).lowercased())")
                        .font(.footnote)
                        .foregroundStyle(.secondary)
                }
            }
            .multilineTextAlignment(.leading)
            .padding(14)
            .frame(maxWidth: .infinity, minHeight: 140, alignment: .topLeading)
            .background(Color(.secondarySystemBackground), in: RoundedRectangle(cornerRadius: 18))
            .opacity(thing.isUncertain ? 0.8 : 1)
        }
        .buttonStyle(.plain)
        .contextMenu {
            Button("Where is it?", systemImage: "scope", action: onTap)
            Button("Details", systemImage: "info.circle", action: onDetails)
        }
        .accessibilityElement(children: .combine)
        .accessibilityHint("Asks the room and shows it on the table")
        .accessibilityAction(named: "Details", onDetails)
    }
}

#Preview("Home, mock") {
    MainView(store: RoomStore(mock: true))
}
