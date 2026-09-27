import CoreGraphics
import Foundation

/// A room layout (state `lay`) checked and read into centimetre rects, already in the person's
/// frame: x their left to right, y far to near. Nil when there's nothing sensible to draw, and
/// then the phone shows only the table map.
struct RoomPlan: Equatable {
    enum ZoneKind: Equatable {
        case seat, surface, door
    }

    struct Zone: Equatable {
        var id: String
        /// What the map prints: the rig's `say` without a leading "the" ("couch").
        var label: String
        var kind: ZoneKind
        var rect: CGRect
    }

    var size: CGSize
    var table: CGRect?
    /// Where the table map's (0, 0) sits in the room.
    var tableOrigin: CGPoint
    var zones: [Zone]
    var you: CGPoint?

    init(size: CGSize, table: CGRect?, tableOrigin: CGPoint? = nil, zones: [Zone], you: CGPoint?) {
        self.size = size
        self.table = table
        self.tableOrigin = tableOrigin ?? table?.origin ?? .zero
        self.zones = zones
        self.you = you
    }

    /// Nil for a layout without a usable size, or with neither a table nor a zone to draw.
    /// Zones with a bad rect or no id are left out; the rest still draw.
    init?(_ layout: RoomLayout?) {
        guard let layout, let size = Self.size(layout.size) else { return nil }
        let table = Self.rect(layout.table?.rect)
        var zones: [Zone] = []
        for z in layout.zones ?? [] {
            guard let id = z.id?.trimmingCharacters(in: .whitespaces), !id.isEmpty, id != "room", id != "table",
                  let rect = Self.rect(z.rect), !zones.contains(where: { $0.id == id }) else { continue }
            let kind: ZoneKind = z.kind == "seat" ? .seat : z.kind == "door" ? .door : .surface
            zones.append(Zone(id: id, label: Self.label(say: z.say, id: id), kind: kind, rect: rect))
        }
        guard table != nil || !zones.isEmpty else { return nil }
        self.init(size: size, table: table, tableOrigin: Self.point(layout.table?.origin),
                  zones: zones, you: Self.point(layout.you))
    }

    func zone(_ id: String) -> Zone? { zones.first { $0.id == id } }

    /// "the couch" -> "couch"; no `say` falls back to the id, "side_table" -> "side table".
    static func label(say: String?, id: String) -> String {
        var words = (say ?? "").trimmingCharacters(in: .whitespaces)
        if words.lowercased().hasPrefix("the ") { words = String(words.dropFirst(4)).trimmingCharacters(in: .whitespaces) }
        return words.isEmpty ? id.replacingOccurrences(of: "_", with: " ") : words
    }

    private static func finite(_ values: [Double]) -> Bool { values.allSatisfy(\.isFinite) }

    private static func size(_ a: [Double]?) -> CGSize? {
        guard let a, a.count == 2, finite(a), a[0] > 0, a[1] > 0 else { return nil }
        return CGSize(width: a[0], height: a[1])
    }

    private static func rect(_ a: [Double]?) -> CGRect? {
        guard let a, a.count == 4, finite(a), a[2] > 0, a[3] > 0 else { return nil }
        return CGRect(x: a[0], y: a[1], width: a[2], height: a[3])
    }

    private static func point(_ a: [Double]?) -> CGPoint? {
        guard let a, a.count == 2, finite(a) else { return nil }
        return CGPoint(x: a[0], y: a[1])
    }
}

/// The room map in points: the plan scaled to fit with a margin, each zone's box and label,
/// a pin for every thing (fanned out in a grid inside its zone, or on the table at
/// origin + its table position), "+N" where a zone is full, things in no drawn zone in an
/// "Elsewhere" strip along the bottom, and the "You" marker.
struct RoomMapLayout: Equatable {
    /// How a pin looks: seen, hidden, carried, a faded hollow "ghost" where it was last seen, or
    /// a full-strength hollow ring where the camera just saw it (a live sighting, state `sg`).
    enum PinStyle: CaseIterable, Equatable {
        case visible, hidden, carried, ghost, sighted

        init(_ presence: Entity.Presence) {
            switch presence {
            case .seen: self = .visible
            case .hidden: self = .hidden
            case .carried: self = .carried
            case .lastSeen: self = .ghost
            }
        }

        var words: String {
            switch self {
            case .visible: return "Seen"
            case .hidden: return "Hidden"
            case .carried: return "Carried"
            case .ghost: return "Last seen"
            case .sighted: return "Seen by the camera"
            }
        }
    }

    struct ZoneBox: Identifiable, Equatable {
        var id: String
        var label: String
        var kind: RoomPlan.ZoneKind
        var rect: CGRect
        var labelRect: CGRect
        /// Too narrow for the label across: it runs up the left edge instead.
        var labelVertical: Bool
    }

    struct Pin: Identifiable, Equatable {
        var id: String
        var name: String
        /// The dot's centre.
        var point: CGPoint
        var diameter: CGFloat
        var style: PinStyle
        /// The zone it's drawn in, `elsewhereID` for the strip, or nil on the table.
        var zone: String?
        /// Where the name goes; nil when there's no room for it (table pins only).
        var labelRect: CGRect?
        var opacity: Double
        var accessibilityLabel: String
        /// The thing a tap selects; the pin's id unless the id is a sighting's.
        var entityName: String? = nil

        var selects: String { entityName ?? id }

        var isOnTable: Bool { zone == nil }
        var dotRect: CGRect {
            CGRect(x: point.x - diameter / 2, y: point.y - diameter / 2, width: diameter, height: diameter)
        }
        /// The dot and its name together: what a tap hits.
        var frame: CGRect { labelRect.map { dotRect.union($0) } ?? dotRect }
    }

    struct Overflow: Identifiable, Equatable {
        /// The zone id.
        var id: String
        var count: Int
        var rect: CGRect
    }

    static let elsewhereID = "elsewhere"
    static let elsewhereLabel = "elsewhere"
    static let margin: CGFloat = 12
    static let pinSize: CGFloat = 14
    static let tablePinSize: CGFloat = 10
    /// The shortest row a pin and its name need.
    static let cellHeight: CGFloat = 18
    /// Room for a dot and a name like "headphones".
    static let minCellWidth: CGFloat = 84
    /// A cell cut short by a label or "You" still takes a pin if this much is left.
    static let minTrimmedWidth: CGFloat = 56
    static let inset: CGFloat = 6
    static let labelHeight: CGFloat = 14
    static let labelGap: CGFloat = 5
    static let tableLabelHeight: CGFloat = 13
    /// The strip under the room for things in no drawn zone.
    static let elsewhereHeight: CGFloat = 26
    static let youSize = CGSize(width: 42, height: 16)
    static let ghostOpacity = 0.45
    static let unsureOpacity = 0.7

    let size: CGSize
    let scale: CGFloat
    /// Where the plan's (0, 0) lands.
    let offset: CGPoint
    /// The room's outline in points.
    let bounds: CGRect
    private(set) var zones: [ZoneBox] = []
    private(set) var table: CGRect?
    private(set) var you: CGRect?
    /// The "Elsewhere" strip, when something needs it.
    private(set) var elsewhere: ZoneBox?
    private(set) var pins: [Pin] = []
    private(set) var overflows: [Overflow] = []

    /// Things on the room map that are in no drawn zone: `room`, or an id the layout lacks.
    static func needsElsewhere(_ plan: RoomPlan, snapshot: Snapshot) -> Bool {
        snapshot.entities.contains { e in shows(e) && e.zone.map { plan.zone($0) == nil } == true }
            || snapshot.liveSightings.contains { plan.zone($0.zone) == nil }
    }

    /// A sighting pin's id, kept apart from the entity pins' ids.
    static func sightingID(_ name: String) -> String { "sg:" + name }

    /// Width over height for a view showing the whole plan, its margins and any strip.
    static func aspectRatio(for plan: RoomPlan, snapshot: Snapshot, width: CGFloat = 390) -> CGFloat {
        let s = (width - 2 * margin) / plan.size.width
        let strip = needsElsewhere(plan, snapshot: snapshot) ? elsewhereHeight + 6 : 0
        return width / (plan.size.height * s + 2 * margin + strip)
    }

    init(plan: RoomPlan, snapshot: Snapshot, size: CGSize) {
        self.size = size
        let strip = Self.needsElsewhere(plan, snapshot: snapshot) ? Self.elsewhereHeight + 6 : 0
        let sx = (size.width - 2 * Self.margin) / plan.size.width
        let sy = (size.height - 2 * Self.margin - strip) / plan.size.height
        scale = max(0, min(sx, sy))
        offset = CGPoint(x: (size.width - plan.size.width * scale) / 2,
                         y: (size.height - strip - plan.size.height * scale) / 2)
        bounds = CGRect(origin: offset, size: CGSize(width: plan.size.width * scale, height: plan.size.height * scale))

        table = plan.table.map(rect)
        you = plan.you.map { p in
            let c = point(p)
            return CGRect(x: c.x - Self.youSize.width / 2, y: c.y - Self.youSize.height / 2,
                          width: Self.youSize.width, height: Self.youSize.height)
        }
        zones = plan.zones.map(box)
        if strip > 0 {
            let r = CGRect(x: bounds.minX, y: bounds.maxY + 6, width: bounds.width, height: Self.elsewhereHeight)
            let w = Self.labelWidth(Self.elsewhereLabel)
            elsewhere = ZoneBox(id: Self.elsewhereID, label: Self.elsewhereLabel, kind: .surface, rect: r,
                                labelRect: CGRect(x: r.minX + Self.inset + 2, y: r.midY - Self.labelHeight / 2,
                                                  width: w, height: Self.labelHeight),
                                labelVertical: false)
        }

        // Things in a zone, grouped; things in no drawn zone to the strip; the rest on the table.
        var inZone: [String: [Entity]] = [:]
        var away: [Entity] = []
        var onTable: [Entity] = []
        for e in snapshot.entities where Self.shows(e) {
            if let z = e.zone {
                if plan.zone(z) != nil { inZone[z, default: []].append(e) } else { away.append(e) }
            } else if MapLayout.position(of: e) != nil, plan.table != nil {
                onTable.append(e)
            }
        }

        // Things the camera just saw ("I see glasses on the couch") that the map can't otherwise place.
        var seenInZone: [String: [Sighting]] = [:]
        var seenAway: [Sighting] = []
        for s in snapshot.liveSightings {
            if plan.zone(s.zone) != nil { seenInZone[s.zone, default: []].append(s) } else { seenAway.append(s) }
        }

        for zone in zones {
            let things = inZone[zone.id] ?? [], seen = seenInZone[zone.id] ?? []
            guard !things.isEmpty || !seen.isEmpty else { continue }
            place(things, sightings: seen, in: zone, cells: cells(for: zone), snapshot: snapshot)
        }
        if let elsewhere, !away.isEmpty || !seenAway.isEmpty {
            place(away, sightings: seenAway, in: elsewhere, cells: stripCells(elsewhere), snapshot: snapshot)
        }
        placeOnTable(onTable, plan: plan, snapshot: snapshot)
    }

    // MARK: Plan to points

    func point(_ p: CGPoint) -> CGPoint {
        CGPoint(x: offset.x + p.x * scale, y: offset.y + p.y * scale)
    }

    func rect(_ r: CGRect) -> CGRect {
        CGRect(x: offset.x + r.minX * scale, y: offset.y + r.minY * scale, width: r.width * scale, height: r.height * scale)
    }

    /// Rough width of a zone label in 11 pt semibold capitals, letter-spaced.
    static func labelWidth(_ text: String) -> CGFloat { CGFloat(text.count) * 8.4 + 2 }

    /// Rough width of a pin's name.
    static func nameWidth(_ text: String, onTable: Bool) -> CGFloat {
        CGFloat(text.count) * (onTable ? 5.8 : 6.8) + 4
    }

    /// The label at the zone's top left, or its bottom left if "You" sits there, or up the
    /// left edge when the zone is too narrow for it.
    private func box(_ zone: RoomPlan.Zone) -> ZoneBox {
        let r = rect(zone.rect)
        let width = Self.labelWidth(zone.label)
        let vertical = width > r.width - 2 * Self.inset && r.height > r.width
        if vertical {
            let label = CGRect(x: r.minX + 3, y: r.minY + Self.inset, width: Self.labelHeight,
                               height: min(width, r.height - 2 * Self.inset))
            return ZoneBox(id: zone.id, label: zone.label, kind: zone.kind, rect: r, labelRect: label, labelVertical: true)
        }
        let w = min(width, r.width - 2 * Self.inset - 2)
        // Top left, unless "You" is there: then bottom left, top right, bottom right.
        let left = r.minX + Self.inset + 2, right = r.maxX - Self.inset - 2 - w
        let top = r.minY + 4, bottom = r.maxY - 4 - Self.labelHeight
        let spots = [CGPoint(x: left, y: top), CGPoint(x: left, y: bottom), CGPoint(x: right, y: top), CGPoint(x: right, y: bottom)]
            .map { CGRect(origin: $0, size: CGSize(width: w, height: Self.labelHeight)) }
        let label = spots.first { spot in you.map { !$0.intersects(spot.insetBy(dx: -2, dy: -2)) } ?? true } ?? spots[0]
        return ZoneBox(id: zone.id, label: zone.label, kind: zone.kind, rect: r, labelRect: label, labelVertical: false)
    }

    // MARK: Pins in zones

    /// A neat grid inside the zone in reading order, leaving out cells its label or the "You"
    /// marker covers. At least one cell, even in a zone too small for one.
    func cells(for zone: ZoneBox) -> [CGRect] {
        let area = zone.rect.insetBy(dx: Self.inset, dy: 3)
        let cols = max(1, Int(area.width / Self.minCellWidth))
        // As many rows as fit at the minimum height, sharing the height out evenly.
        let rows = max(1, Int((area.height + 1) / Self.cellHeight))
        let rowHeight = max(Self.cellHeight, area.height / CGFloat(rows))
        let top = area.minY
        let w = area.width / CGFloat(cols)
        var out: [CGRect] = []
        for row in 0..<rows {
            for col in 0..<cols {
                let cell = CGRect(x: area.minX + CGFloat(col) * w, y: top + CGFloat(row) * rowHeight,
                                  width: w, height: rowHeight)
                if let free = Self.trim(cell, clearOf: [zone.labelRect] + (you.map { [$0] } ?? [])) {
                    out.append(free)
                }
            }
        }
        if out.isEmpty {
            out.append(CGRect(x: area.minX, y: area.maxY - Self.cellHeight, width: max(area.width, Self.pinSize + 4),
                              height: Self.cellHeight))
        }
        return out
    }

    /// A cell a label or "You" partly covers keeps the wider free side, if a short name still fits.
    static func trim(_ cell: CGRect, clearOf obstacles: [CGRect]) -> CGRect? {
        var free = cell
        for o in obstacles {
            let hit = o.insetBy(dx: -2, dy: 0)
            guard hit.intersects(free) else { continue }
            let left = CGRect(x: free.minX, y: free.minY, width: max(0, hit.minX - free.minX), height: free.height)
            let right = CGRect(x: hit.maxX, y: free.minY, width: max(0, free.maxX - hit.maxX), height: free.height)
            free = left.width >= right.width ? left : right
        }
        return free.width >= minTrimmedWidth ? free : nil
    }

    /// One row after the strip's label.
    private func stripCells(_ strip: ZoneBox) -> [CGRect] {
        let left = strip.labelRect.maxX + 8
        let width = strip.rect.maxX - Self.inset - left
        let cols = max(1, Int(width / Self.minCellWidth))
        let w = width / CGFloat(cols)
        return (0..<cols).map { CGRect(x: left + CGFloat($0) * w, y: strip.rect.midY - Self.cellHeight / 2,
                                       width: w, height: Self.cellHeight) }
    }

    private enum Placed {
        case thing(Entity)
        case sighting(Sighting)
    }

    private mutating func place(_ things: [Entity], sightings: [Sighting] = [], in zone: ZoneBox, cells: [CGRect],
                                snapshot: Snapshot) {
        // Seen things first, then fresh sightings, then ghosts; by name within each, so pins keep
        // their places while others come and go.
        func key(_ p: Placed) -> (Int, String, String) {
            switch p {
            case .thing(let e): return (Self.style(for: e) == .ghost ? 2 : 0, e.displayName.lowercased(), e.name)
            case .sighting(let s): return (1, Self.sightingName(s, in: snapshot).lowercased(), s.name)
            }
        }
        let sorted = (things.map(Placed.thing) + sightings.map(Placed.sighting)).sorted { key($0) < key($1) }
        let fits = sorted.count <= cells.count ? sorted.count : cells.count - 1
        for (item, cell) in zip(sorted.prefix(fits), cells) {
            let d = Self.pinSize
            let dot = CGPoint(x: cell.minX + d / 2 + 1, y: cell.midY)
            let labelX = dot.x + d / 2 + Self.labelGap
            let label = CGRect(x: labelX, y: cell.midY - 8, width: max(0, cell.maxX - labelX - 2), height: 16)
            switch item {
            case .thing(let e):
                pins.append(pin(e, at: dot, diameter: d, zone: zone.id, label: label, snapshot: snapshot))
            case .sighting(let s):
                pins.append(Self.sightingPin(s, at: dot, diameter: d, zone: zone.id, label: label, snapshot: snapshot))
            }
        }
        if sorted.count > fits {
            overflows.append(Overflow(id: zone.id, count: sorted.count - fits, rect: cells[fits]))
        }
    }

    // MARK: Pins on the table

    private mutating func placeOnTable(_ things: [Entity], plan: RoomPlan, snapshot: Snapshot) {
        guard let table else { return }
        let d = Self.tablePinSize
        let inner = table.insetBy(dx: min(d / 2 + 2, table.width / 2), dy: min(d / 2 + 2, table.height / 2))
        var dots: [CGRect] = []
        // Names may spill off the table but never onto zone labels, the "You" marker or each other.
        var taken: [CGRect] = zones.map(\.labelRect) + (you.map { [$0] } ?? [])
        // Hidden things first: they sit at their parent's place, and the parent moves over.
        let ordered = things.enumerated().sorted { a, b in
            let ra = [.inside, .under].contains(a.element.status) ? 0 : 1
            let rb = [.inside, .under].contains(b.element.status) ? 0 : 1
            return (ra, a.offset) < (rb, b.offset)
        }.map(\.element)

        var placed: [(dot: CGPoint, entity: Entity)] = []
        for e in ordered {
            guard let p = MapLayout.position(of: e) else { continue }
            let cm = CGPoint(x: plan.tableOrigin.x + p.x, y: plan.tableOrigin.y + p.y)
            let start = Self.clamp(point(cm), in: inner)
            let dot = Self.freeSpot(near: start, diameter: d, avoiding: dots, in: inner)
            dots.append(CGRect(x: dot.x - d / 2, y: dot.y - d / 2, width: d, height: d))
            placed.append((dot, e))
        }
        taken += dots
        // The inset is small: only named things (configured or taught) get a name; guesses stay plain dots
        // (VoiceOver and a tap still say them).
        for (dot, e) in placed where !Self.namedOnTable(e) {
            pins.append(pin(e, at: dot, diameter: d, zone: nil, label: nil, snapshot: snapshot))
        }
        for (dot, e) in placed where Self.namedOnTable(e) {
            let w = Self.nameWidth(e.displayName, onTable: true)
            let h = Self.tableLabelHeight
            let candidates = [
                CGRect(x: dot.x + d / 2 + 3, y: dot.y - h / 2, width: w, height: h),
                CGRect(x: dot.x - d / 2 - 3 - w, y: dot.y - h / 2, width: w, height: h),
                CGRect(x: dot.x - w / 2, y: dot.y + d / 2 + 1, width: w, height: h),
                CGRect(x: dot.x - w / 2, y: dot.y - d / 2 - 1 - h, width: w, height: h),
            ]
            let label = candidates.first { c in bounds.contains(c) && !taken.contains { $0.intersects(c) } }
            if let label { taken.append(label) }
            pins.append(pin(e, at: dot, diameter: d, zone: nil, label: label, snapshot: snapshot))
        }
    }

    /// A thing on the table inset gets a printed name only if it has a real one: a configured object or a
    /// taught name, not a guess.
    static func namedOnTable(_ e: Entity) -> Bool { !e.isThing || !e.aliases.isEmpty }

    /// `p`, or the nearest spot around it clear of the dots already placed.
    private static func freeSpot(near p: CGPoint, diameter d: CGFloat, avoiding dots: [CGRect], in area: CGRect) -> CGPoint {
        func clear(_ q: CGPoint) -> Bool {
            let r = CGRect(x: q.x - d / 2, y: q.y - d / 2, width: d, height: d).insetBy(dx: -1, dy: -1)
            return !dots.contains { $0.intersects(r) }
        }
        if clear(p) { return p }
        for ring in 1...4 {
            let radius = CGFloat(ring) * (d + 2)
            for k in 0..<8 {
                let a = Double(k) * .pi / 4
                let q = clamp(CGPoint(x: p.x + radius * cos(a), y: p.y + radius * sin(a)), in: area)
                if clear(q) { return q }
            }
        }
        return p
    }

    private static func clamp(_ p: CGPoint, in r: CGRect) -> CGPoint {
        CGPoint(x: min(max(p.x, r.minX), r.maxX), y: min(max(p.y, r.minY), r.maxY))
    }

    // MARK: Style

    private func pin(_ e: Entity, at p: CGPoint, diameter: CGFloat, zone: String?, label: CGRect?, snapshot: Snapshot) -> Pin {
        Pin(id: e.name, name: e.displayName, point: p, diameter: diameter, style: Self.style(for: e), zone: zone,
            labelRect: label, opacity: Self.opacity(for: e),
            accessibilityLabel: Self.accessibilityLabel(for: e, in: snapshot))
    }

    /// The thing's name as the map shows it: its display name if the room knows it, else the name.
    static func sightingName(_ s: Sighting, in snapshot: Snapshot) -> String {
        snapshot.entity(named: s.name)?.displayName ?? Entity.displayName(for: s.name)
    }

    /// "glasses · seen 4:37 PM".
    static func sightingLabel(_ s: Sighting, in snapshot: Snapshot, now: Date = Date()) -> String {
        "\(sightingName(s, in: snapshot)) · seen \(Banners.when(s.time, now: now))"
    }

    /// "Glasses, seen on the couch at 4:37 PM".
    static func sightingAccessibilityLabel(_ s: Sighting, in snapshot: Snapshot, now: Date = Date()) -> String {
        let words = Dashboard.sightingWords(s, in: snapshot, now: now)
        return "\(Dashboard.capitalized(sightingName(s, in: snapshot))), \(words.prefix(1).lowercased() + words.dropFirst())"
    }

    private static func sightingPin(_ s: Sighting, at p: CGPoint, diameter: CGFloat, zone: String, label: CGRect,
                                    snapshot: Snapshot) -> Pin {
        Pin(id: sightingID(s.name), name: sightingLabel(s, in: snapshot), point: p, diameter: diameter, style: .sighted,
            zone: zone, labelRect: label, opacity: 1,
            accessibilityLabel: sightingAccessibilityLabel(s, in: snapshot), entityName: s.name)
    }

    /// Things drawn on the room map: every thing with a name, and anything the registry keeps.
    static func shows(_ e: Entity) -> Bool {
        !e.n.hasPrefix("hand:") && (!e.isNameless || e.registry != nil)
    }

    /// The registry's word wins when there is one; otherwise the status decides.
    static func style(for e: Entity) -> PinStyle { PinStyle(e.presence) }

    /// Ghosts fade most; a tentative or unsure thing a little.
    static func opacity(for e: Entity) -> Double {
        if style(for: e) == .ghost { return ghostOpacity }
        return e.isTentative || e.isUncertain ? unsureOpacity : 1
    }

    /// "Keys, on the couch", "Umbrella, last seen by the doorway", "Wallet, on the table".
    static func accessibilityLabel(for e: Entity, in snapshot: Snapshot) -> String {
        let words = Dashboard.whereabouts(e, in: snapshot)
        return "\(Dashboard.capitalized(e.displayName)), \(words.prefix(1).lowercased() + words.dropFirst())"
    }

    /// Only the looks this map uses, for the key under it.
    static func legend(for pins: [Pin]) -> [PinStyle] {
        PinStyle.allCases.filter { style in pins.contains { $0.style == style } }
    }
}
