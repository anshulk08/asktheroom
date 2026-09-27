import CoreGraphics
import Foundation

/// Converts table centimetres to view points with one uniform scale (spec section 5):
/// the table fits the space, keeps its aspect ratio and sits centred, with a margin
/// outside it so "left the table" arrows have somewhere to go.
struct MapGeometry {
    static let margin: CGFloat = 16
    /// The margin below the near edge when the map shows where the person sits: room for the
    /// "You" marker under the band the "left the table" arrows use.
    static let youMargin: CGFloat = 36
    /// The "You" marker's height, glyph and word on one line.
    static let youHeight: CGFloat = 18

    var table: TablePoint
    var size: CGSize
    /// The map is in the person's frame (the snapshot has a `view`): mark them at the bottom.
    var showsYou = false

    init(table: TablePoint, size: CGSize, showsYou: Bool = false) {
        self.table = table
        self.size = size
        self.showsYou = showsYou
    }

    var bottomMargin: CGFloat { Self.bottomMargin(showsYou: showsYou) }

    static func bottomMargin(showsYou: Bool) -> CGFloat { showsYou ? youMargin : margin }

    /// Width over height for a view that shows the whole table plus its margins.
    static func aspectRatio(for table: TablePoint, width: CGFloat = 390, showsYou: Bool = false) -> CGFloat {
        guard table.x > 0, table.y > 0 else { return 1.5 }
        let height = (width - 2 * margin) * table.y / table.x + margin + bottomMargin(showsYou: showsYou)
        return width / height
    }

    var scale: CGFloat {
        guard table.x > 0, table.y > 0 else { return 0 }
        let sx = (size.width - 2 * Self.margin) / table.x
        let sy = (size.height - Self.margin - bottomMargin) / table.y
        return max(0, min(sx, sy))
    }

    /// Centred across; down, centred in the space between the top margin and the bottom one.
    var tableRect: CGRect {
        let w = table.x * scale
        let h = table.y * scale
        let top = Self.margin + (size.height - Self.margin - bottomMargin - h) / 2
        return CGRect(x: (size.width - w) / 2, y: top, width: w, height: h)
    }

    /// The centre of the "You" marker: just outside the near edge, below the arrows' band.
    var youPoint: CGPoint {
        let rect = tableRect
        return CGPoint(x: rect.midX, y: rect.maxY + Self.youMargin - Self.youHeight / 2 - 2)
    }

    func point(_ p: TablePoint) -> CGPoint {
        let rect = tableRect
        return CGPoint(x: rect.minX + p.x * scale, y: rect.minY + p.y * scale)
    }

    /// `point`, pulled onto the table: the rig can report things in the band just past the
    /// tabletop's edge, and those are drawn at the edge rather than off the map.
    func pointOnTable(_ p: TablePoint) -> CGPoint {
        let rect = tableRect
        let q = point(p)
        return CGPoint(x: min(max(q.x, rect.minX), rect.maxX), y: min(max(q.y, rect.minY), rect.maxY))
    }

    func length(_ cm: Double) -> CGFloat { cm * scale }

    /// Where an arrow leaving through `edge` ends: just past the table, level with `from`.
    func exitPoint(from p: CGPoint, through edge: Edge) -> CGPoint {
        let rect = tableRect
        let past = Self.margin - 4
        switch edge {
        case .left: return CGPoint(x: rect.minX - past, y: p.y)
        case .right: return CGPoint(x: rect.maxX + past, y: p.y)
        case .top: return CGPoint(x: p.x, y: rect.minY - past)
        case .bottom: return CGPoint(x: p.x, y: rect.maxY + past)
        }
    }

    /// Both ends of a table edge, for the sweep animation.
    func edgeLine(_ edge: Edge) -> (CGPoint, CGPoint) {
        let r = tableRect
        switch edge {
        case .left: return (CGPoint(x: r.minX, y: r.minY), CGPoint(x: r.minX, y: r.maxY))
        case .right: return (CGPoint(x: r.maxX, y: r.minY), CGPoint(x: r.maxX, y: r.maxY))
        case .top: return (CGPoint(x: r.minX, y: r.minY), CGPoint(x: r.maxX, y: r.minY))
        case .bottom: return (CGPoint(x: r.minX, y: r.maxY), CGPoint(x: r.maxX, y: r.maxY))
        }
    }
}

/// One thing to draw on the map, decided from an entity and its snapshot.
struct MapItem: Identifiable, Equatable {
    enum Shape: Equatable {
        /// Containers and covers: a large rectangle sized in table cm.
        case block(width: Double, height: Double)
        /// Targets: a round pin with the thing's picture, its name underneath, like Find My.
        case pin
    }

    var id: String
    var title: String
    /// What the map prints: the title, or just "unnamed" for a thing without a name, so a
    /// long "unnamed object 9" doesn't crowd the table. VoiceOver and the card say it in full.
    /// A guess keeps its question mark: "phone charger?".
    var label: String
    /// The entity this one is inside or under, if any.
    var parent: String?
    /// Status in words ("inside box"); nil for a plainly visible object. Read by VoiceOver;
    /// on screen the ring, badge and legend carry it, and a tap shows it in full.
    var caption: String?
    var shape: Shape
    var status: EntityStatus
    var center: TablePoint
    /// Hidden objects are dashed, so status reads in greyscale.
    var dashed: Bool
    var opacity: Double
    /// SF Symbol badged on the pin's corner (held, lost).
    var glyph: String?
    /// Set for gone objects: draw an arrow off the table through this edge.
    var exitEdge: Edge?
    /// A link badge for things that might be an older thing.
    var linkBadge: Bool
    /// Where this pin sits among children sharing a parent, so they don't stack exactly.
    var siblingIndex = 0
    var siblingCount = 1
    /// For U: the pin peeks out from under its parent, whose size is needed to place it.
    var peekFrom: Shape?
    /// Drawing order: lower first.
    var layer: Int

    var accessibilityLabel: String {
        [title, caption].compactMap { $0 }.joined(separator: ", ")
    }
}

enum MapLayout {
    static let uncertainOpacity = 0.6
    static let lostOpacity = 0.45
    static let coveringOpacity = 0.85

    /// Rough real sizes in cm: a shoebox-sized container, a notebook-sized cover.
    static func blockSize(for kind: EntityKind) -> (Double, Double)? {
        switch kind {
        case .container: return (18, 13)
        case .cover: return (20, 14)
        default: return nil
        }
    }

    static func items(for snapshot: Snapshot) -> [MapItem] {
        let covered = Set(snapshot.entities.filter { $0.status == .under }.compactMap(\.parent))

        // Children drawn inside the same parent are fanned out instead of piled up.
        var siblings: [String: [String]] = [:]
        for e in snapshot.entities where e.status == .inside {
            if let p = e.parent { siblings[p, default: []].append(e.name) }
        }

        return snapshot.entities.compactMap { entity -> MapItem? in
            guard let center = position(of: entity) else { return nil }

            let blockShape = blockSize(for: entity.kind).map { MapItem.Shape.block(width: $0.0, height: $0.1) }
            let depth = snapshot.chain(from: entity.name).count - 1

            var item = MapItem(
                id: entity.name,
                title: entity.displayName,
                label: entity.isNameless ? "unnamed" : entity.displayName,
                parent: entity.status == .inside || entity.status == .under ? entity.parent : nil,
                caption: caption(for: entity, in: snapshot),
                shape: blockShape ?? .pin,
                status: entity.status,
                center: center,
                dashed: [.held, .inside, .under].contains(entity.status),
                opacity: 1,
                glyph: glyph(for: entity.status),
                exitEdge: entity.status == .gone ? entity.edge : nil,
                linkBadge: entity.isThing && !entity.maybeSameAs.isEmpty,
                layer: 0
            )

            // Containers and covers first so children stay visible; something under a
            // cover goes below it, and the cover is drawn over it translucently.
            if entity.status == .under {
                item.layer = depth
                if let parent = entity.parent.flatMap(snapshot.entity(named:)),
                   let size = blockSize(for: parent.kind) {
                    item.peekFrom = .block(width: size.0, height: size.1)
                }
            } else if blockShape != nil {
                item.layer = 10 + depth
            } else {
                item.layer = 20 + depth
            }

            if entity.status == .lost {
                item.opacity = lostOpacity
            } else if entity.isUncertain {
                item.opacity = uncertainOpacity
            } else if blockShape != nil, covered.contains(entity.name) {
                item.opacity = coveringOpacity
            }

            if entity.status == .inside, let p = entity.parent, let group = siblings[p], group.count > 1 {
                item.siblingIndex = group.firstIndex(of: entity.name) ?? 0
                item.siblingCount = group.count
            }
            return item
        }
        .sorted { $0.layer < $1.layer }
    }

    /// Held, gone and lost objects are drawn where they were last seen; the rest at
    /// their resolved position, which for hidden objects is their parent's.
    static func position(of entity: Entity) -> TablePoint? {
        switch entity.status {
        case .held, .gone, .lost: return entity.xy ?? entity.r
        default: return entity.r ?? entity.xy
        }
    }

    static func glyph(for status: EntityStatus) -> String? {
        switch status {
        case .held: return "hand.raised.fill"
        case .lost: return "questionmark"
        default: return nil
        }
    }

    static func arrow(for edge: Edge) -> String {
        switch edge {
        case .left: return "←"
        case .right: return "→"
        case .top: return "↑"
        case .bottom: return "↓"
        }
    }

    /// One line of the key under the map.
    enum LegendEntry: CaseIterable, Equatable {
        case held, hidden, left, lost, unsure

        var words: String {
            switch self {
            case .held: return "In someone's hand"
            case .hidden: return "Hidden under or inside"
            case .left: return "Left the table"
            case .lost: return "Can't see it now"
            case .unsure: return "Faded: not sure"
            }
        }
    }

    /// Only the marks this map actually uses, so the key stays short.
    static func legend(for items: [MapItem]) -> [LegendEntry] {
        LegendEntry.allCases.filter { entry in
            items.contains { item in
                switch entry {
                case .held: return item.status == .held
                case .hidden: return item.status == .inside || item.status == .under
                case .left: return item.exitEdge != nil
                case .lost: return item.status == .lost
                case .unsure: return item.status != .lost && item.opacity == uncertainOpacity
                }
            }
        }
    }

    /// Status words for VoiceOver and the detail, per the table in spec section 5.
    static func caption(for entity: Entity, in snapshot: Snapshot) -> String? {
        let parentName = entity.parent.map { name in
            snapshot.entity(named: name)?.displayName ?? Entity.displayName(for: name)
        }
        var words: String?
        switch entity.status {
        case .held: words = "in a hand"
        case .inside: words = parentName.map { "inside \($0)" } ?? "inside something"
        case .under: words = parentName.map { "under \($0)" } ?? "under something"
        case .gone: words = entity.edge.map { "left table \(arrow(for: $0))" } ?? "left the table"
        case .lost: return "lost track, last seen here"
        case .visible, .unrecognized: words = nil
        }
        if words == nil, entity.isThing, let other = entity.maybeSameAs.first {
            let otherName = snapshot.entity(named: other.name)?.displayName ?? Entity.displayName(for: other.name)
            words = "might be \(otherName)"
        }
        guard entity.isUncertain else { return words }
        return "probably " + (words ?? "here")
    }
}

// MARK: Placement in points

extension MapLayout {
    /// A pin's disc; smaller for things inside or under something, so they fit in their block.
    static let pinSize: CGFloat = 32
    static let innerPinSize: CGFloat = 26
    /// The name under the disc.
    static let pinLabelHeight: CGFloat = 16
    /// A block's picture and name sit inside its top edge; what's inside sits below them.
    static let blockTitleHeight: CGFloat = 18

    static func pinDiameter(for item: MapItem) -> CGFloat {
        item.status == .inside || item.status == .under ? innerPinSize : pinSize
    }

    /// Rough on-screen size of a pin and its name, for keeping names apart. Measured against
    /// SF Pro caption-semibold at the default text size; never under a 44 pt tap target wide.
    static func footprint(of item: MapItem) -> CGSize {
        let d = pinDiameter(for: item)
        return CGSize(width: max(44, CGFloat(item.label.count) * 7 + 14), height: d + pinLabelHeight)
    }

    /// The room a pin and its name take up, for a pin whose disc is centred on `p`.
    static func rect(of item: MapItem, at p: CGPoint) -> CGRect {
        let size = footprint(of: item)
        return CGRect(x: p.x - size.width / 2, y: p.y - pinDiameter(for: item) / 2, width: size.width, height: size.height)
    }

    /// Where to put a view so its disc (the top of it) lands on `p`; blocks are centred.
    static func viewCenter(of item: MapItem, at p: CGPoint) -> CGPoint {
        guard item.shape == .pin else { return p }
        let r = rect(of: item, at: p)
        return CGPoint(x: r.midX, y: r.midY)
    }

    /// Where each item's centre goes on screen (a pin's disc). Blocks sit exactly at their
    /// position. Pins start there (below the name inside a parent and fanned out, or peeking
    /// from under a cover), stay on the table, and then take the nearby spot that covers the
    /// least of the names and blocks already placed, so every name stays readable.
    static func placements(for items: [MapItem], in geo: MapGeometry) -> [String: CGPoint] {
        var out: [String: CGPoint] = [:]
        var taken: [(owner: String, isBody: Bool, rect: CGRect)] = []
        let table = geo.tableRect

        for item in items {
            guard case .block(let w, let h) = item.shape else { continue }
            let p = geo.pointOnTable(item.center)
            out[item.id] = p
            let body = CGRect(x: p.x - geo.length(w) / 2, y: p.y - geo.length(h) / 2, width: geo.length(w), height: geo.length(h))
            let titleWidth = CGFloat(item.label.count) * 7 + 8 + blockTitleHeight
            taken.append((item.id, true, body))
            taken.append((item.id, false, CGRect(x: p.x - titleWidth / 2, y: body.minY + 1, width: titleWidth, height: blockTitleHeight - 2)))
        }

        // Hidden children first: they belong inside their parent and move least.
        let pins = items.enumerated()
            .filter { $0.element.shape == .pin }
            .sorted { (rank($0.element), $0.offset) < (rank($1.element), $1.offset) }
            .map(\.element)

        for item in pins {
            let size = footprint(of: item)
            var p = geo.point(item.center)
            if item.status == .inside {
                p.y += blockTitleHeight / 2
            }
            if item.siblingCount > 1 {
                p.x += (CGFloat(item.siblingIndex) - CGFloat(item.siblingCount - 1) / 2) * (size.width + 2)
            }
            if case .block(let w, let h) = item.peekFrom {
                // Peek out below the cover's lower edge, towards its right.
                p.x += geo.length(w) * 0.2
                p.y += geo.length(h) / 2 + pinDiameter(for: item) * 0.2
            }

            // A pin may sit on its own parent's body, never on anyone's name.
            let obstacles = taken.filter { !($0.isBody && $0.owner == item.parent) }.map(\.rect)
            let best = bestSpot(for: item, near: p, avoiding: obstacles, in: table)
            out[item.id] = best
            taken.append((item.id, false, rect(of: item, at: best)))
        }
        return out
    }

    /// The nearby point where a pin covers the least; distance from `p` breaks ties.
    private static func bestSpot(for item: MapItem, near p: CGPoint, avoiding obstacles: [CGRect], in table: CGRect) -> CGPoint {
        let size = footprint(of: item)
        let step = size.height / 2 + 4
        var best = clamp(p, item: item, in: table)
        var bestScore = CGFloat.infinity
        for dy in [0, 1, -1, 2, -2, 3, -3] as [CGFloat] {
            for dx in [0, 0.5, -0.5, 1, -1] as [CGFloat] {
                let candidate = clamp(CGPoint(x: p.x + dx * size.width, y: p.y + dy * step), item: item, in: table)
                let r = rect(of: item, at: candidate)
                let overlap = obstacles.reduce(CGFloat(0)) { sum, o in
                    let i = o.intersection(r)
                    return i.isNull ? sum : sum + i.width * i.height
                }
                let score = overlap * 10 + hypot(candidate.x - p.x, candidate.y - p.y)
                if score < bestScore {
                    bestScore = score
                    best = candidate
                }
            }
        }
        return best
    }

    private static func rank(_ item: MapItem) -> Int {
        switch item.status {
        case .inside, .under: return 0
        case .visible, .held: return 1
        default: return 2
        }
    }

    /// Keeps a pin and its name on the table.
    private static func clamp(_ p: CGPoint, item: MapItem, in table: CGRect) -> CGPoint {
        let size = footprint(of: item)
        let above = pinDiameter(for: item) / 2
        let below = size.height - above
        let halfW = min(size.width / 2, table.width / 2)
        return CGPoint(x: min(max(p.x, table.minX + halfW + 4), table.maxX - halfW - 4),
                       y: min(max(p.y, table.minY + above + 4), table.maxY - below - 4))
    }
}
