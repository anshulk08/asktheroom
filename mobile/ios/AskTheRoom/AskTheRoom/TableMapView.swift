import SwiftUI

enum Theme {
    /// Same red as the laser dot (spec section 6). Only for "the laser points here":
    /// map highlights, the reticle and the connect pulse. Buttons use `accent`.
    static let laser = Color(red: 1, green: 0.231, blue: 0.188)
    /// Calm slate blue (AccentColor in the asset catalog) for buttons and selection.
    static let accent = Color.accentColor
    /// Behind icons on cards.
    static let iconWell = Color(.tertiarySystemFill)

    static func surface(_ scheme: ColorScheme) -> Color {
        scheme == .dark ? Color(red: 0.17, green: 0.20, blue: 0.24) : Color(red: 0.95, green: 0.91, blue: 0.84)
    }

    static func block(_ scheme: ColorScheme) -> Color {
        scheme == .dark ? Color(red: 0.31, green: 0.36, blue: 0.43) : Color(red: 0.82, green: 0.74, blue: 0.62)
    }

    /// A pin's disc and the badge behind a name.
    static func chip(_ scheme: ColorScheme) -> Color {
        scheme == .dark ? Color(red: 0.10, green: 0.11, blue: 0.13) : .white
    }
}

/// The live table map (spec section 5): table to scale, then every entity by its status.
struct TableMapView: View {
    var snapshot: Snapshot
    var highlight: Highlight?
    var greyed = false
    /// Keep a ring on the highlight after the pulses, until it's cleared (the answer sheet).
    var steady = false
    /// Where the real laser dot is. Off in the answer sheet, so only one thing is lit up there.
    var showsLaser = true
    /// The thing tapped on the Table tab, ringed in the accent colour.
    var selected: String? = nil
    var onSelect: (String) -> Void = { _ in }

    @Environment(\.accessibilityReduceMotion) private var reduceMotion

    var body: some View {
        let items = MapLayout.items(for: snapshot)
        GeometryReader { proxy in
            let geo = MapGeometry(table: snapshot.tableSize, size: proxy.size, showsYou: showsYou)
            let places = MapLayout.placements(for: items, in: geo)

            ZStack {
                TableSurface(geo: geo)

                if showsYou {
                    YouMarker()
                        .position(geo.youPoint)
                }

                ForEach(items.filter { $0.exitEdge != nil }) { item in
                    let from = places[item.id] ?? .zero
                    ExitArrow(from: from, to: geo.exitPoint(from: from, through: item.exitEdge!))
                        .stroke(.primary.opacity(0.6), style: StrokeStyle(lineWidth: 2, lineCap: .round, lineJoin: .round))
                        .opacity(item.opacity)
                }

                ForEach(items) { item in
                    MapItemView(item: item, scale: geo.scale, isSelected: item.id == selected)
                        .position(MapLayout.viewCenter(of: item, at: places[item.id] ?? .zero))
                        .zIndex(Double(item.layer))
                        .onTapGesture { onSelect(item.id) }
                        .accessibilityElement(children: .ignore)
                        .accessibilityLabel(item.accessibilityLabel)
                        .accessibilityAddTraits(.isButton)
                }

                if let highlight {
                    HighlightView(highlight: highlight, geo: geo, at: highlightPoint(highlight, places: places, geo: geo),
                                  steady: steady)
                        .id(highlight.id)
                        .zIndex(90)
                        .allowsHitTesting(false)
                }

                if showsLaser, let laser = snapshot.laser, laser.on, let target = laser.target, let point = places[target] {
                    Reticle()
                        .position(point)
                        .zIndex(100)
                        .allowsHitTesting(false)
                        .transition(.opacity)
                        .accessibilityHidden(true)
                }
            }
            .animation(reduceMotion ? nil : .easeInOut(duration: 0.3), value: items)
            .animation(reduceMotion ? nil : .spring(duration: 0.4), value: snapshot.laser)
            .animation(reduceMotion ? nil : .spring(duration: 0.3, bounce: 0.3), value: selected)
        }
        .sensoryFeedback(.selection, trigger: selected)
        .aspectRatio(MapGeometry.aspectRatio(for: snapshot.tableSize, showsYou: showsYou), contentMode: .fit)
        .dynamicTypeSize(...DynamicTypeSize.xLarge)
        .saturation(greyed ? 0 : 1)
        .opacity(greyed ? 0.45 : 1)
        .accessibilityElement(children: .contain)
        .accessibilityLabel("Table map")
    }

    /// Only a rig that turns the map to the person's frame (it sends a `view`) puts them at the bottom.
    private var showsYou: Bool { snapshot.view != nil }

    private func highlightPoint(_ h: Highlight, places: [String: CGPoint], geo: MapGeometry) -> CGPoint? {
        if let name = h.entity, let p = places[name] { return p }
        return h.target.map(geo.pointOnTable)
    }
}

// MARK: Pieces

private struct TableSurface: View {
    let geo: MapGeometry
    @Environment(\.colorScheme) private var scheme

    var body: some View {
        Canvas { context, _ in
            let rect = geo.tableRect
            let table = Path(roundedRect: rect, cornerRadius: 14)
            context.fill(table, with: .color(Theme.surface(scheme)))

            // No grid: the table's edge is the only line that isn't a thing.
            context.stroke(table, with: .color(.primary.opacity(0.2)), lineWidth: 1)
        }
        .accessibilityHidden(true)
    }
}

private struct MapItemView: View {
    let item: MapItem
    let scale: CGFloat
    var isSelected = false
    @Environment(\.colorScheme) private var scheme

    private var icon: ThingIcon { IconStore.shared.icon(for: item.id, title: item.title) }

    var body: some View {
        switch item.shape {
        case .block(let w, let h):
            block(width: w * scale, height: h * scale)
        case .pin:
            pin
        }
    }

    /// Plain things get a faint edge; only hidden and held things get a strong dashed one,
    /// so the lines that are there mean something.
    private var edge: (Color, StrokeStyle) {
        item.dashed
            ? (.primary.opacity(0.6), StrokeStyle(lineWidth: 1.5, dash: [4, 3]))
            : (.primary.opacity(0.12), StrokeStyle(lineWidth: 1))
    }

    private func block(width: CGFloat, height: CGFloat) -> some View {
        RoundedRectangle(cornerRadius: 8)
            .fill(Theme.block(scheme))
            .overlay(RoundedRectangle(cornerRadius: 8).strokeBorder(edge.0, style: edge.1))
            .overlay {
                if isSelected {
                    RoundedRectangle(cornerRadius: 10).strokeBorder(Theme.accent, lineWidth: 3).padding(-3)
                }
            }
            .frame(width: width, height: height)
            .overlay(alignment: .top) {
                // The picture and name sit inside the block, so nothing floats above it.
                HStack(spacing: 3) {
                    ThingIconView(icon: icon, size: 14)
                    Text(item.label).font(.caption.weight(.semibold))
                }
                .fixedSize()
                .frame(height: MapLayout.blockTitleHeight)
            }
            .opacity(item.opacity)
            .contentShape(Rectangle())
    }

    /// Find My's pin: the thing's picture on a disc, its name underneath, and status as a
    /// badge (held, lost), a dashed ring (hidden) or a fade (not sure).
    private var pin: some View {
        let d = MapLayout.pinDiameter(for: item)
        return VStack(spacing: 1) {
            ThingIconView(icon: icon, size: d * 0.62)
                .frame(width: d, height: d)
                .background(Circle().fill(Theme.chip(scheme)).shadow(color: .black.opacity(0.2), radius: 3, y: 1.5))
                .overlay(Circle().strokeBorder(edge.0, style: edge.1))
                .overlay(alignment: .topTrailing) {
                    if let glyph = item.glyph { PinBadge(symbol: glyph).offset(x: 5, y: -4) }
                }
                .overlay(alignment: .topLeading) {
                    if item.linkBadge { PinBadge(symbol: "link", quiet: true).offset(x: -5, y: -4) }
                }
                .overlay {
                    if isSelected { Circle().strokeBorder(Theme.accent, lineWidth: 3).padding(-4) }
                }
                .scaleEffect(isSelected ? 1.15 : 1)
            Text(item.label)
                .font(.caption.weight(.semibold))
                .lineLimit(1)
                .fixedSize()
                .padding(.horizontal, 5)
                .frame(height: MapLayout.pinLabelHeight - 1)
                .background(Capsule().fill(Theme.chip(scheme).opacity(0.85)))
        }
        .frame(minWidth: 44, minHeight: d + MapLayout.pinLabelHeight, alignment: .top)
        .opacity(item.opacity)
        .contentShape(Rectangle())
    }
}

/// Where the person sits: just past the near edge, since the map is turned to face them.
private struct YouMarker: View {
    var body: some View {
        HStack(spacing: 3) {
            Image(systemName: "person.fill")
                .font(.system(size: 11, weight: .semibold))
            Text("You")
                .font(.caption.weight(.semibold))
        }
        .foregroundStyle(.secondary)
        .fixedSize()
        .frame(height: MapGeometry.youHeight)
        .accessibilityElement(children: .ignore)
        .accessibilityLabel("You, at the near edge of the table")
    }
}

/// A status mark on a pin's corner, the way Find My and Messages badge a picture.
private struct PinBadge: View {
    let symbol: String
    var quiet = false

    var body: some View {
        Image(systemName: symbol)
            .font(.system(size: 9, weight: .bold))
            .foregroundStyle(quiet ? Color.primary : Color(.systemBackground))
            .frame(width: 17, height: 17)
            .background(Circle().fill(quiet ? Color(.systemBackground) : Color.primary))
            .overlay(Circle().strokeBorder(Color(.systemBackground), lineWidth: 1.5))
            .accessibilityHidden(true)
    }
}

/// The key under the map: only the marks in use, each with its words.
struct MapLegend: View {
    let entries: [MapLayout.LegendEntry]
    @Environment(\.colorScheme) private var scheme

    var body: some View {
        if !entries.isEmpty {
            FlowLayout(spacing: 14, lineSpacing: 6) {
                ForEach(entries, id: \.self) { entry in
                    HStack(spacing: 6) {
                        mark(entry)
                        Text(entry.words)
                    }
                }
            }
            .font(.footnote)
            .foregroundStyle(.secondary)
            .accessibilityElement(children: .ignore)
            .accessibilityLabel("Key: " + entries.map(\.words).joined(separator: ", "))
        }
    }

    @ViewBuilder private func mark(_ entry: MapLayout.LegendEntry) -> some View {
        switch entry {
        case .held:
            PinBadge(symbol: "hand.raised.fill")
        case .hidden:
            Circle().strokeBorder(.primary.opacity(0.6), style: StrokeStyle(lineWidth: 1.5, dash: [3, 2]))
                .frame(width: 17, height: 17)
        case .left:
            Image(systemName: "arrow.left").foregroundStyle(.primary)
        case .lost:
            PinBadge(symbol: "questionmark")
        case .unsure:
            Circle().fill(Theme.chip(scheme)).overlay(Circle().strokeBorder(.primary.opacity(0.18)))
                .frame(width: 17, height: 17)
                .opacity(MapLayout.uncertainOpacity)
        }
    }
}

/// A line off the table with an arrowhead, for objects that left through an edge.
private struct ExitArrow: Shape {
    var from: CGPoint
    var to: CGPoint

    func path(in rect: CGRect) -> Path {
        var p = Path()
        p.move(to: from)
        p.addLine(to: to)
        let angle = atan2(to.y - from.y, to.x - from.x)
        for side in [-1.0, 1.0] {
            let a = angle + .pi + side * .pi / 6
            p.move(to: to)
            p.addLine(to: CGPoint(x: to.x + 9 * cos(a), y: to.y + 9 * sin(a)))
        }
        return p
    }
}

/// Where the physical laser dot is: a red double ring that lands on its target.
private struct Reticle: View {
    @Environment(\.accessibilityReduceMotion) private var reduceMotion
    @State private var landed = false

    var body: some View {
        // Hugs the pin, so the names around it stay readable.
        ZStack {
            Circle().stroke(Theme.laser, lineWidth: 3)
            Circle().stroke(Theme.laser.opacity(0.45), lineWidth: 1.5).padding(-4)
        }
        .frame(width: 40, height: 40)
        .scaleEffect(landed || reduceMotion ? 1 : 2.5)
        .onAppear {
            withAnimation(.spring(duration: 0.4)) { landed = true }
        }
    }
}

/// The answer's highlight: a 1 s pulse on the target, plus a sweep along an edge or a circle.
/// `steady` leaves a still ring (and the sweep) behind once the pulses finish.
private struct HighlightView: View {
    let highlight: Highlight
    let geo: MapGeometry
    let at: CGPoint?
    var steady = false

    @Environment(\.accessibilityReduceMotion) private var reduceMotion
    @State private var progress: CGFloat = 0

    var body: some View {
        ZStack {
            if let at {
                if steady {
                    Circle()
                        .stroke(Theme.laser, lineWidth: 3)
                        .frame(width: 52, height: 52)
                        .position(at)
                }
                Circle()
                    .stroke(Theme.laser, lineWidth: 3)
                    .frame(width: 36, height: 36)
                    .scaleEffect(reduceMotion ? 1.3 : 0.7 + 1.6 * progress)
                    .opacity(reduceMotion ? 0.9 : 1 - Double(progress))
                    .position(at)

                if highlight.action == .circle {
                    Circle()
                        .trim(from: 0, to: reduceMotion ? 1 : min(1, progress * 1.5))
                        .stroke(Theme.laser, style: StrokeStyle(lineWidth: 3, lineCap: .round))
                        .rotationEffect(.degrees(-90))
                        .frame(width: 70, height: 70)
                        .position(at)
                }
            }
            if case .sweep(let edge) = highlight.action {
                let (a, b) = geo.edgeLine(edge)
                Path { p in p.move(to: a); p.addLine(to: b) }
                    .trim(from: 0, to: reduceMotion ? 1 : min(1, progress * 1.5))
                    .stroke(Theme.laser, style: StrokeStyle(lineWidth: 5, lineCap: .round))
            }
        }
        .onAppear {
            guard !reduceMotion else { return }
            withAnimation(.easeOut(duration: 1).repeatCount(3, autoreverses: false)) { progress = 1 }
        }
    }
}

#Preview("Sample snapshot") {
    TableMapView(snapshot: MockData.sampleSnapshot,
                 highlight: Highlight(entity: "keys", target: TablePoint(x: 70.4, y: 38.1), action: .point))
        .padding()
}
