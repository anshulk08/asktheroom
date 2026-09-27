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
    /// Unnamed things only: Grok's soft guess of what it is, and how sure (missing means 0.6).
    var g: String?
    var gc: Double?
    /// `as` on the wire: "grok" when `a[0]` was set by Grok, not taught by a person.
    var aliasSource: String?

    private enum CodingKeys: String, CodingKey {
        case n, k, s, p, xy, r, c, edge, a, m, ls, g, gc
        case aliasSource = "as"
    }

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

    // Naming. The views use only these, so a thing is called the same everywhere:
    //   taught alias        "my charger"       ("your" in sentences)
    //   alias from Grok     "tape roll?"       ("what looks like a tape roll", never "your")
    //   Grok guess >= 0.5   "phone charger?"   (same hedging)
    //   nothing, weak guess "unnamed object 9"

    /// A guess counts from here up; below it the thing stays "unnamed".
    static let guessThreshold = 0.5
    /// Older bridges send `g` without `gc`.
    static let defaultGuessConfidence = 0.6

    /// A thing a person named: it has an alias, and Grok didn't set the newest one.
    var hasTaughtName: Bool { isThing && !aliases.isEmpty && aliasSource != "grok" }

    /// What the room thinks it is, without a person's say-so: Grok's alias, else a confident guess.
    var hedgedName: String? {
        guard isThing, !hasTaughtName else { return nil }
        if let alias = aliases.first { return alias }
        guard let g, !g.isEmpty, (gc ?? Self.defaultGuessConfidence) >= Self.guessThreshold else { return nil }
        return g
    }

    var isHedged: Bool { hedgedName != nil }

    /// A thing with nothing to call it but its number.
    var isNameless: Bool { isThing && aliases.isEmpty && !isHedged }

    /// "7" for `thing:7`, kept for the detail sheet once a guess replaces it in the title.
    var thingNumber: String? { isThing ? String(n.dropFirst("thing:".count)) : nil }

    var displayName: String {
        hedgedName.map { "\($0)?" } ?? Entity.displayName(for: n, aliases: aliases)
    }

    /// How a sentence names it: "what looks like a tape roll" when hedged, else `displayName`.
    var phrase: String {
        hedgedName.map { Entity.looksLike($0) } ?? displayName
    }

    static let hedgePrefix = "what looks like "

    /// "what looks like a tape roll".
    static func looksLike(_ name: String) -> String { hedgePrefix + withArticle(name) }

    /// "a tape roll", "an apple".
    static func withArticle(_ name: String) -> String {
        let vowel = name.lowercased().first.map { "aeiou".contains($0) } ?? false
        return "\(vowel ? "an" : "a") \(name)"
    }

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

    /// Older things `entity` might be, leaving out ones with no name to tell them by:
    /// "might be something new" says nothing.
    func knownMatches(of entity: Entity) -> [MaybeSame] {
        entity.maybeSameAs.filter { self.entity(named: $0.name).map { !$0.isNameless } ?? false }
    }

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
    /// Only when `id` is nil (PROTOCOL.md 6a): where the question came from (`voice`, `dashboard`,
    /// `sms`), or `notice` for a reminder or the morning report the rig just fired.
    var src: String?
    /// Room answers: the question as the rig heard it.
    var q: String?
    /// Notices: the rig's notice id, and what fired it (`reminder`, `morning`, …).
    var nid: Int?
    var kind: String?

    var succeeded: Bool { ok ?? true }
    /// A question someone else asked the rig: out loud, on the dashboard or by SMS.
    var isRoomAnswer: Bool { id == nil && src != nil && src != "notice" }
    var isNotice: Bool { id == nil && src == "notice" }
    var pointAt: String? { point_at }
    var laserAction: LaserAction? { action.map(LaserAction.init) }
}

struct RigStatus: Codable, Equatable {
    var app: String?
    var fps: Double?
    var online: Bool?
    var cal: Bool?
    var laser_cal: Bool?
    /// The rig's speaker is connected (PROTOCOL.md section 8). Missing from older rigs.
    var spk: Bool?

    var appIsUp: Bool { app == "up" }
    /// The room hears answers from the rig, so the phone doesn't read them aloud too.
    var rigSpeaks: Bool { appIsUp && spk == true }
}

/// The helper's voice, sent to the rig so its speaker talks the same way (PROTOCOL.md section 5a).
struct VoiceSettings: Codable, Equatable {
    struct Choice: Codable, Equatable {
        /// `Speaker.Engine` raw value.
        var e: String
        /// Grok voice id.
        var v: String
        /// Speed, 0.7 to 1.5.
        var s: Double
    }

    var voice: Choice

    /// The JSON to write to the question characteristic, or nil if it can't fit one write.
    func encoded() -> Data? {
        let encoder = JSONEncoder()
        encoder.outputFormatting = .sortedKeys
        guard let data = try? encoder.encode(self), data.count <= Question.maxBytes else { return nil }
        return data
    }
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
