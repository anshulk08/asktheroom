import Foundation

// Wire types for protocol v1 (spec sections 3 and 4). Every key the spec lets the rig
// omit is optional, and unrecognised enum values decode to a fallback instead of
// failing, so a newer rig never breaks an older app.

/// A position in table centimetres: origin at the top-left marker, x right, y down.
struct TablePoint: Codable, Equatable, Hashable {
    var x: Double
    var y: Double

    init(x: Double, y: Double) {
        self.x = x
        self.y = y
    }

    init(from decoder: Decoder) throws {
        var c = try decoder.unkeyedContainer()
        x = try c.decode(Double.self)
        y = try c.decode(Double.self)
    }

    func encode(to encoder: Encoder) throws {
        var c = encoder.unkeyedContainer()
        try c.encode(x)
        try c.encode(y)
    }
}

enum EntityKind: String, Codable {
    case target = "t", container = "c", cover = "v", unrecognized

    init(from decoder: Decoder) throws {
        self = EntityKind(rawValue: try decoder.singleValueContainer().decode(String.self)) ?? .unrecognized
    }
}

enum EntityStatus: String, Codable {
    case visible = "V", held = "H", under = "U", inside = "I", gone = "G", lost = "X", unrecognized

    init(from decoder: Decoder) throws {
        self = EntityStatus(rawValue: try decoder.singleValueContainer().decode(String.self)) ?? .unrecognized
    }
}

enum Edge: String, Codable, CaseIterable {
    case left, right, top, bottom
}

/// "Possibly the same as": an unconfirmed link to an older thing, sent as `[name, score]`.
struct MaybeSame: Codable, Equatable, Hashable {
    var name: String
    var score: Double

    init(name: String, score: Double) {
        self.name = name
        self.score = score
    }

    init(from decoder: Decoder) throws {
        var c = try decoder.unkeyedContainer()
        name = try c.decode(String.self)
        score = try c.decode(Double.self)
    }

    func encode(to encoder: Encoder) throws {
        var c = encoder.unkeyedContainer()
        try c.encode(name)
        try c.encode(score)
    }
}

struct Entity: Codable, Equatable, Identifiable {
    var n: String
    var k: EntityKind
    var s: EntityStatus
    var p: String?
    var xy: TablePoint?
    var r: TablePoint?
    var c: Double?
    var edge: Edge?
    var a: [String]?
    var m: [MaybeSame]?
    var ls: Double?

    var id: String { n }
    var name: String { n }
    var kind: EntityKind { k }
    var status: EntityStatus { s }
    var parent: String? { p }
    var confidence: Double { c ?? 1 }
    var aliases: [String] { a ?? [] }
    var maybeSameAs: [MaybeSame] { m ?? [] }
    var lastSeen: Date? { ls.map(Date.init(timeIntervalSince1970:)) }

    /// Where to draw and highlight: the resolved position, else where it was last seen.
    var drawPoint: TablePoint? { r ?? xy }

    /// Below 0.7 the rig says "probably"; the map fades it and says so too.
    var isUncertain: Bool { confidence < 0.7 }

    var isThing: Bool { n.hasPrefix("thing:") }

    /// Held by a hand rather than inside or under another entity.
    var isInHand: Bool { p?.hasPrefix("hand:") ?? false }

    var displayName: String { Entity.displayName(for: n, aliases: aliases) }

    static func displayName(for name: String, aliases: [String] = []) -> String {
        if name.hasPrefix("thing:") {
            if let first = aliases.first { return first }
            return "unnamed object \(name.dropFirst("thing:".count))"
        }
        return name.replacingOccurrences(of: "_", with: " ")
    }
}

struct LaserState: Codable, Equatable {
    var on: Bool
    var target: String?
}

struct Snapshot: Codable, Equatable {
    static let defaultTable = TablePoint(x: 90, y: 60)

    var v: Int?
    var t: Double?
    var table: TablePoint?
    var online: Bool?
    var laser: LaserState?
    var e: [Entity]

    var entities: [Entity] { e }
    var tableSize: TablePoint { table ?? Self.defaultTable }
    var time: Date? { t.map(Date.init(timeIntervalSince1970:)) }

    func entity(named name: String) -> Entity? { e.first { $0.n == name } }

    /// The chain from an entity out to the outermost thing holding it, e.g. keys → notebook → box.
    /// Stops at hands, missing parents and cycles; the rig nests at most 3 levels.
    func chain(from name: String) -> [Entity] {
        var out: [Entity] = []
        var seen: Set<String> = []
        var next: String? = name
        while let current = next, !seen.contains(current), let entity = entity(named: current) {
            out.append(entity)
            seen.insert(current)
            next = entity.isInHand ? nil : entity.p
        }
        return out
    }
}

/// What the laser does with an answer. `action` is an open string on the wire (handoff item 4).
enum LaserAction: Equatable {
    case point
    case circle
    case sweep(Edge)
    /// A newer action this app doesn't know (e.g. `trace`, `tour`): pulse `point_at` if present.
    case other(String)

    init(_ raw: String) {
        switch raw {
        case "point": self = .point
        case "circle": self = .circle
        default:
            if raw.hasPrefix("sweep:"), let edge = Edge(rawValue: String(raw.dropFirst("sweep:".count))) {
                self = .sweep(edge)
            } else {
                self = .other(raw)
            }
        }
    }
}

struct Answer: Codable, Equatable {
    /// Echo of the question id. Nil for answers the phone didn't ask for.
    var id: Int?
    var ok: Bool?
    var text: String
    var point_at: String?
    var action: String?
    var target: TablePoint?
    var ms: Int?

    var succeeded: Bool { ok ?? true }
    var pointAt: String? { point_at }
    var laserAction: LaserAction? { action.map(LaserAction.init) }
}

struct RigStatus: Codable, Equatable {
    var app: String?
    var fps: Double?
    var online: Bool?
    var cal: Bool?
    var laser_cal: Bool?

    var appIsUp: Bool { app == "up" }
}

struct Question: Codable, Equatable {
    /// Protocol limit for one unframed write.
    static let maxBytes = 180

    var id: Int
    var q: String

    /// The JSON to write, trimming the question until it fits in one write.
    func encoded() -> Data? {
        let encoder = JSONEncoder()
        encoder.outputFormatting = .sortedKeys
        var text = q
        while true {
            guard let data = try? encoder.encode(Question(id: id, q: text)) else { return nil }
            if data.count <= Self.maxBytes { return data }
            if text.isEmpty { return nil }
            text.removeLast()
        }
    }
}

enum Wire {
    /// Decodes one reassembled message. Malformed JSON returns nil: the caller logs and drops it.
    static func decode<T: Decodable>(_ type: T.Type, from data: Data) -> T? {
        try? JSONDecoder().decode(type, from: data)
    }
}
