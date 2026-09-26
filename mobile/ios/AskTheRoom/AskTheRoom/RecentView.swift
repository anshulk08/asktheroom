import SwiftUI

/// The Recent tab: what changed on the table and what was asked, newest first, with clock times.
struct RecentView: View {
    let store: RoomStore
    var onSelect: (RecentEntry) -> Void = { _ in }

    var body: some View {
        NavigationStack {
            TimelineView(.periodic(from: .now, by: 30)) { context in
                let groups = Dashboard.recent(activity: store.activity, exchanges: store.exchanges, now: context.date)
                List {
                    if groups.lastHour.isEmpty && groups.earlier.isEmpty {
                        Text("Nothing has happened yet. Changes on the table and your questions will show up here.")
                            .font(.body)
                            .foregroundStyle(.secondary)
                            .listRowBackground(Color.clear)
                    }
                    if !groups.lastHour.isEmpty {
                        Section("In the last hour") { rows(groups.lastHour, now: context.date) }
                    }
                    if !groups.earlier.isEmpty {
                        Section("Earlier") { rows(groups.earlier, now: context.date) }
                    }
                }
                .listStyle(.insetGrouped)
                .animation(.easeOut(duration: 0.25), value: groups.lastHour)
            }
            .navigationTitle("Recent")
        }
    }

    private func rows(_ entries: [RecentEntry], now: Date) -> some View {
        ForEach(entries) { entry in
            Button { onSelect(entry) } label: {
                RecentRow(entry: entry, now: now)
            }
            .buttonStyle(.plain)
        }
    }
}

private struct RecentRow: View {
    let entry: RecentEntry
    let now: Date

    var body: some View {
        HStack(alignment: .top, spacing: 12) {
            icon
                .frame(width: 30)
                .accessibilityHidden(true)
            VStack(alignment: .leading, spacing: 4) {
                switch entry {
                case .change(let event):
                    Text(event.text)
                        .font(.body.weight(.medium))
                case .question(let exchange):
                    Text("You asked: \(exchange.question)")
                        .font(.body.weight(.medium))
                    if let answer = exchange.answer {
                        Text(answer.text).font(.body).foregroundStyle(.secondary)
                    } else if exchange.timedOut {
                        Text("The room didn't answer.").font(.body).foregroundStyle(.secondary)
                    }
                }
                Text("\(entry.time.formatted(date: .omitted, time: .shortened)) · \(Dashboard.ago(entry.time, now: now))")
                    .font(.subheadline)
                    .foregroundStyle(.secondary)
            }
            .fixedSize(horizontal: false, vertical: true)
            Spacer(minLength: 0)
        }
        .padding(.vertical, 6)
        .contentShape(Rectangle())
        .accessibilityElement(children: .combine)
    }

    @ViewBuilder private var icon: some View {
        switch entry {
        case .change(let event):
            ThingIconView(icon: IconStore.shared.icon(for: event.entity), size: 26)
        case .question:
            Image(systemName: "bubble.left").font(.title3).foregroundStyle(.secondary)
        }
    }
}
