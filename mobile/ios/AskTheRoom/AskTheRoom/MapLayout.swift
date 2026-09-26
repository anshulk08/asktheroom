import CoreGraphics
import Foundation

/// Converts table centimetres to view points with one uniform scale (spec section 5):
/// the table fits the space, keeps its aspect ratio and sits centred, with a margin
/// outside it so "left the table" arrows have somewhere to go.
struct MapGeometry {
    static let margin: CGFloat = 16

    var table: TablePoint
    var size: CGSize

    /// Width over height for a view that shows the whole table plus its margin.
    static func aspectRatio(for table: TablePoint, width: CGFloat = 390) -> CGFloat {
        guard table.x > 0, table.y > 0 else { return 1.5 }
        let height = (width - 2 * margin) * table.y / table.x + 2 * margin
        return width / height
    }

    var scale: CGFloat {
        guard table.x > 0, table.y > 0 else { return 0 }
        let sx = (size.width - 2 * Self.margin) / table.x
        let sy = (size.height - 2 * Self.margin) / table.y
        return max(0, min(sx, sy))
    }

    var tableRect: CGRect {
        let w = table.x * scale
        let h = table.y * scale
        return CGRect(x: (size.width - w) / 2, y: (size.height - h) / 2, width: w, height: h)
    }

    func point(_ p: TablePoint) -> CGPoint {
        let rect = tableRect
        return CGPoint(x: rect.minX + p.x * scale, y: rect.minY + p.y * scale)
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
        /// Targets: a rounded chip sized by its label.
        case chip
    }

    var id: String
    var title: String
    /// The entity this one is inside or under, if any.
    var parent: String?
    /// Status in words under the title ("inside box"); nil for a plainly visible object.
    var caption: String?
    var shape: Shape
    var status: EntityStatus
    var center: TablePoint
    /// Hidden objects are dashed, so status reads in greyscale.
    var dashed: Bool
    var opacity: Double
    /// SF Symbol shown before the title.
    var glyph: String?
    /// Set for gone objects: draw an arrow off the table through this edge.
    var exitEdge: Edge?
    /// A "? link" badge for things that might be an older thing.
    var linkBadge: Bool
    /// Where this chip sits among children sharing a parent, so they don't stack exactly.
    var siblingIndex = 0
    var siblingCount = 1
    /// For U: the chip peeks out from under its parent, whose size is needed to place it.
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
                parent: entity.status == .inside || entity.status == .under ? entity.parent : nil,
                caption: caption(for: entity, in: snapshot),
                shape: blockShape ?? .chip,
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

    /// The label under a chip, per the table in spec section 5.
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
    static let chipHeight: CGFloat = 28
    static let captionLineHeight: CGFloat = 18
    static let captionMaxWidth: CGFloat = 150

    /// Rough on-screen size of a chip plus its caption, for keeping labels apart.
    /// Measured against SF Pro subheadline-semibold / footnote at the default text size.
    static func footprint(of item: MapItem, withCaption: Bool = true) -> CGSize {
        let chipWidth = CGFloat(item.title.count) * 8 + 16 + (item.glyph == nil ? 0 : 18) + (item.linkBadge ? 48 : 0)
        guard withCaption, let caption = item.caption else { return CGSize(width: chipWidth, height: chipHeight) }
        let captionWidth = CGFloat(caption.count) * 7.2 + 8
        let lines = (captionWidth / captionMaxWidth).rounded(.up)
        return CGSize(width: max(chipWidth, min(captionWidth, captionMaxWidth)),
                      height: chipHeight + lines * captionLineHeight)
    }

    /// Where each item's centre goes on screen. Blocks sit exactly at their position.
    /// Chips start there (fanned out inside a shared parent, or peeking from under a
    /// cover), stay on the table, and then take the nearby spot that covers the least of
    /// the labels and blocks already placed, so every name stays readable. A caption
    /// that can't fit without covering something is dropped; the chip's outline still
    /// shows the status and the detail sheet has the words.
    static func placements(for items: [MapItem], in geo: MapGeometry) -> Placement {
        var out: [String: CGPoint] = [:]
        var hiddenCaptions: Set<String> = []
        var taken: [(owner: String, isBody: Bool, rect: CGRect)] = []
        let table = geo.tableRect

        for item in items {
            guard case .block(let w, let h) = item.shape else { continue }
            let p = geo.point(item.center)
            out[item.id] = p
            let body = CGRect(x: p.x - geo.length(w) / 2, y: p.y - geo.length(h) / 2, width: geo.length(w), height: geo.length(h))
            let labelWidth = CGFloat(item.title.count) * 7.5 + 8
            taken.append((item.id, true, body))
            taken.append((item.id, false, CGRect(x: p.x - labelWidth / 2, y: body.minY - 20, width: labelWidth, height: 18)))
        }

        // Hidden children first: they belong inside their parent and move least.
        let chips = items.enumerated()
            .filter { $0.element.shape == .chip }
            .sorted { (rank($0.element), $0.offset) < (rank($1.element), $1.offset) }
            .map(\.element)

        for item in chips {
            var p = geo.point(item.center)
            if item.siblingCount > 1 {
                p.y += (CGFloat(item.siblingIndex) - CGFloat(item.siblingCount - 1) / 2) * (chipHeight + 2)
            }
            if case .block(let w, let h) = item.peekFrom {
                // Peek out below the cover's lower edge, towards its right.
                p.x += geo.length(w) * 0.2
                p.y += geo.length(h) / 2 + chipHeight * 0.2
            }

            // A chip may sit on its own parent's body, never on anyone's label.
            let obstacles = taken.filter { !($0.isBody && $0.owner == item.parent) }.map(\.rect)

            var (best, size, overlap) = bestSpot(near: p, size: footprint(of: item), avoiding: obstacles, in: table)
            if overlap > 0, item.caption != nil {
                let bare = footprint(of: item, withCaption: false)
                let alt = bestSpot(near: p, size: bare, avoiding: obstacles, in: table)
                if alt.overlap < overlap {
                    (best, size, overlap) = alt
                    hiddenCaptions.insert(item.id)
                }
            }
            out[item.id] = best
            taken.append((item.id, false, rect(at: best, size: size)))
        }
        return Placement(points: out, hiddenCaptions: hiddenCaptions)
    }

    struct Placement {
        var points: [String: CGPoint]
        var hiddenCaptions: Set<String>
    }

    /// The nearby point where a footprint covers the least; distance from `p` breaks ties.
    private static func bestSpot(near p: CGPoint, size: CGSize, avoiding obstacles: [CGRect], in table: CGRect)
        -> (point: CGPoint, size: CGSize, overlap: CGFloat) {
        let step = chipHeight + 4
        var best = (point: clamp(p, size: size, in: table), size: size, overlap: CGFloat.infinity)
        var bestScore = CGFloat.infinity
        for dy in [0, 1, -1, 2, -2, 3, -3] as [CGFloat] {
            for dx in [0, 0.5, -0.5] as [CGFloat] {
                let candidate = clamp(CGPoint(x: p.x + dx * size.width, y: p.y + dy * step), size: size, in: table)
                let r = rect(at: candidate, size: size)
                let overlap = obstacles.reduce(CGFloat(0)) { sum, o in
                    let i = o.intersection(r)
                    return i.isNull ? sum : sum + i.width * i.height
                }
                let score = overlap * 10 + hypot(candidate.x - p.x, candidate.y - p.y)
                if score < bestScore {
                    bestScore = score
                    best = (candidate, size, overlap)
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

    /// The chip is centred on the point; its caption hangs below.
    static func rect(at p: CGPoint, size: CGSize) -> CGRect {
        CGRect(x: p.x - size.width / 2, y: p.y - chipHeight / 2, width: size.width, height: size.height)
    }

    private static func clamp(_ p: CGPoint, size: CGSize, in table: CGRect) -> CGPoint {
        let halfW = min(size.width / 2, table.width / 2)
        let minY = table.minY + chipHeight / 2 + 10
        let maxY = max(minY, table.maxY - (size.height - chipHeight / 2) - 4)
        return CGPoint(x: min(max(p.x, table.minX + halfW + 4), table.maxX - halfW - 4),
                       y: min(max(p.y, minY), maxY))
    }
}
