import SwiftUI

/// Colours for the room map. Dark matches the dashboard's mat and the /demo room map; the
/// status colours are the dashboard's (visible, held, hidden), with a light-mode twin each.
enum RoomTheme {
    static func mat(_ s: ColorScheme) -> Color { s == .dark ? hex(0x16302A) : hex(0xEEF2EC) }
    static func ink(_ s: ColorScheme) -> Color { s == .dark ? hex(0xE4EDE6) : hex(0x1D2A25) }
    static func ink2(_ s: ColorScheme) -> Color { s == .dark ? hex(0xA7BDB2) : hex(0x55675E) }
    static func zoneStroke(_ s: ColorScheme) -> Color { s == .dark ? hex(0x7F968B).opacity(0.75) : hex(0x8A9A90) }
    static func zoneFill(_ s: ColorScheme) -> Color { s == .dark ? hex(0xE4EDE6).opacity(0.05) : Color.black.opacity(0.03) }
    static func seatFill(_ s: ColorScheme) -> Color { s == .dark ? hex(0xBFE6C9).opacity(0.12) : hex(0x2E8B57).opacity(0.10) }
    static func wood(_ s: ColorScheme) -> Color { s == .dark ? hex(0x6E5038) : hex(0xC9A27A) }
    static func woodEdge(_ s: ColorScheme) -> Color { s == .dark ? hex(0x9C7552) : hex(0xA27C56) }

    static func pin(_ style: RoomMapLayout.PinStyle, _ s: ColorScheme) -> Color {
        switch style {
        case .visible: return s == .dark ? hex(0xBFE6C9) : hex(0x2E8B57)
        case .hidden: return s == .dark ? hex(0x8EC5FF) : hex(0x2F7FD0)
        case .carried: return s == .dark ? hex(0xF2B84B) : hex(0xC7860B)
        case .ghost: return ink(s)
        case .sighted: return pin(.visible, s)
        }
    }

    private static func hex(_ v: UInt32) -> Color {
        Color(red: Double((v >> 16) & 0xFF) / 255, green: Double((v >> 8) & 0xFF) / 255, blue: Double(v & 0xFF) / 255)
    }
}

/// The whole room, turned to the person's seat: zones as rounded boxes, the table in wood,
/// a pin for every thing and "You" where they sit. Pins glide when things move.
struct RoomMapView: View {
    var plan: RoomPlan
    var snapshot: Snapshot
    var greyed = false
    var selected: String? = nil
    var onSelect: (String) -> Void = { _ in }

    @Environment(\.colorScheme) private var scheme
    @Environment(\.accessibilityReduceMotion) private var reduceMotion

    var body: some View {
        GeometryReader { proxy in
            let layout = RoomMapLayout(plan: plan, snapshot: snapshot, size: proxy.size)
            ZStack(alignment: .topLeading) {
                RoundedRectangle(cornerRadius: 16).fill(RoomTheme.mat(scheme))

                ForEach(layout.zones) { zone in zoneView(zone) }
                if let strip = layout.elsewhere { zoneView(strip, dashed: true) }

                if let table = layout.table {
                    RoundedRectangle(cornerRadius: 6)
                        .fill(RoomTheme.wood(scheme))
                        .overlay(RoundedRectangle(cornerRadius: 6).strokeBorder(RoomTheme.woodEdge(scheme), lineWidth: 1.5))
                        .frame(width: table.width, height: table.height)
                        .position(x: table.midX, y: table.midY)
                        .accessibilityHidden(true)
                }

                if let you = layout.you {
                    youMarker.position(x: you.midX, y: you.midY)
                }

                ForEach(layout.overflows) { more in
                    Text("+\(more.count)")
                        .font(.system(size: 12, weight: .bold))
                        .foregroundStyle(RoomTheme.ink2(scheme))
                        .frame(width: more.rect.width, height: more.rect.height, alignment: .leading)
                        .position(x: more.rect.midX, y: more.rect.midY)
                        .accessibilityLabel("\(more.count) more")
                }

                ForEach(layout.pins) { pin in pinName(pin) }
                ForEach(layout.pins) { pin in pinDot(pin) }
            }
            .animation(reduceMotion ? nil : .spring(duration: 0.6, bounce: 0.15), value: layout.pins)
            .animation(reduceMotion ? nil : .spring(duration: 0.3, bounce: 0.3), value: selected)
        }
        .aspectRatio(RoomMapLayout.aspectRatio(for: plan, snapshot: snapshot), contentMode: .fit)
        .dynamicTypeSize(...DynamicTypeSize.large)
        .saturation(greyed ? 0 : 1)
        .opacity(greyed ? 0.45 : 1)
        .sensoryFeedback(.selection, trigger: selected)
        .accessibilityElement(children: .contain)
        .accessibilityLabel("Room map")
    }

    // MARK: Pieces

    private func zoneView(_ zone: RoomMapLayout.ZoneBox, dashed: Bool = false) -> some View {
        let isDoor = zone.kind == .door || dashed
        let fill = zone.kind == .seat ? RoomTheme.seatFill(scheme) : RoomTheme.zoneFill(scheme)
        let stroke = StrokeStyle(lineWidth: 2, dash: isDoor ? [6, 4] : [])
        return ZStack(alignment: .topLeading) {
            RoundedRectangle(cornerRadius: 8)
                .fill(fill)
                .overlay(RoundedRectangle(cornerRadius: 8).strokeBorder(RoomTheme.zoneStroke(scheme), style: stroke))
                .frame(width: zone.rect.width, height: zone.rect.height)
                .position(x: zone.rect.midX, y: zone.rect.midY)
            zoneLabel(zone)
        }
        .accessibilityElement(children: .ignore)
        .accessibilityLabel(zone.label.prefix(1).uppercased() + zone.label.dropFirst())
    }

    @ViewBuilder private func zoneLabel(_ zone: RoomMapLayout.ZoneBox) -> some View {
        let text = Text(zone.label.uppercased())
            .font(.system(size: 11, weight: .semibold))
            .tracking(1.2)
            .foregroundStyle(RoomTheme.ink2(scheme))
            .lineLimit(1)
            .minimumScaleFactor(0.7)
        let r = zone.labelRect
        if zone.labelVertical {
            text.fixedSize()
                .rotationEffect(.degrees(-90))
                .frame(width: r.width, height: r.height)
                .position(x: r.midX, y: r.midY)
        } else {
            text.frame(width: r.width + 20, height: r.height, alignment: .leading)
                .position(x: r.midX + 10, y: r.midY)
        }
    }

    private var youMarker: some View {
        HStack(spacing: 3) {
            Image(systemName: "person.fill").font(.system(size: 10, weight: .bold))
            Text("YOU").font(.system(size: 11, weight: .bold)).tracking(1)
        }
        .foregroundStyle(RoomTheme.ink(scheme))
        .padding(.horizontal, 6)
        .frame(height: RoomMapLayout.youSize.height)
        .background(Capsule().fill(RoomTheme.mat(scheme).opacity(0.9)))
        .overlay(Capsule().strokeBorder(RoomTheme.ink2(scheme).opacity(0.6), lineWidth: 1))
        .fixedSize()
        .accessibilityElement(children: .ignore)
        .accessibilityLabel("You")
    }

    /// A tap on a pin selects its thing; a sighting of a thing the room doesn't track selects nothing.
    private func tap(_ pin: RoomMapLayout.Pin) {
        guard pin.style != .sighted || snapshot.entity(named: pin.selects) != nil else { return }
        onSelect(pin.selects)
    }

    /// Ghosts and sightings are hollow rings; the rest are solid dots in their state's colour.
    private func pinDot(_ pin: RoomMapLayout.Pin) -> some View {
        let color = RoomTheme.pin(pin.style, scheme)
        let isSelected = pin.selects == selected
        return ZStack {
            if pin.style == .ghost {
                Circle().strokeBorder(color, lineWidth: 2)
            } else if pin.style == .sighted {
                Circle().strokeBorder(color, lineWidth: 2.5)
            } else {
                Circle().fill(color)
                    .overlay(Circle().strokeBorder(RoomTheme.mat(scheme).opacity(0.8), lineWidth: 1.5))
            }
        }
        .frame(width: pin.diameter, height: pin.diameter)
        .opacity(pin.opacity)
        .overlay {
            if isSelected { Circle().strokeBorder(Theme.accent, lineWidth: 2.5).padding(-4) }
        }
        // A bigger tap target round the dot; the name takes taps too.
        .frame(width: 30, height: 30)
        .contentShape(Rectangle())
        .position(pin.point)
        .onTapGesture { tap(pin) }
        .accessibilityElement(children: .ignore)
        .accessibilityLabel(pin.accessibilityLabel)
        .accessibilityAddTraits(.isButton)
        .accessibilityAddTraits(isSelected ? .isSelected : [])
    }

    @ViewBuilder private func pinName(_ pin: RoomMapLayout.Pin) -> some View {
        if let r = pin.labelRect {
            let alignment: Alignment = r.minX >= pin.point.x ? .leading : r.maxX <= pin.point.x ? .trailing : .center
            Text(pin.name)
                .font(.system(size: pin.isOnTable ? 10 : 12, weight: pin.isOnTable ? .medium : .semibold))
                .foregroundStyle(RoomTheme.ink(scheme))
                .lineLimit(1)
                .truncationMode(.tail)
                .frame(width: r.width, height: r.height, alignment: alignment)
                .opacity(pin.opacity)
                .position(x: r.midX, y: r.midY)
                .contentShape(Rectangle())
                .onTapGesture { tap(pin) }
                .accessibilityHidden(true)
        }
    }
}

/// The key under the room map: only the looks in use.
struct RoomMapLegend: View {
    let styles: [RoomMapLayout.PinStyle]
    @Environment(\.colorScheme) private var scheme

    var body: some View {
        if !styles.isEmpty {
            FlowLayout(spacing: 14, lineSpacing: 6) {
                ForEach(styles, id: \.self) { style in
                    HStack(spacing: 6) {
                        Group {
                            if style == .ghost {
                                Circle().strokeBorder(Color.primary, lineWidth: 2).opacity(RoomMapLayout.ghostOpacity)
                            } else if style == .sighted {
                                Circle().strokeBorder(RoomTheme.pin(style, scheme), lineWidth: 2.5)
                            } else {
                                Circle().fill(RoomTheme.pin(style, scheme))
                            }
                        }
                        .frame(width: 12, height: 12)
                        Text(style.words)
                    }
                }
            }
            .font(.footnote)
            .foregroundStyle(.secondary)
            .accessibilityElement(children: .ignore)
            .accessibilityLabel("Key: " + styles.map(\.words).joined(separator: ", "))
        }
    }
}

#Preview("Room map") {
    RoomMapView(plan: RoomPlan(MockData.sampleLayout)!, snapshot: MockData.roomSnapshot)
        .padding()
        .preferredColorScheme(.dark)
}
