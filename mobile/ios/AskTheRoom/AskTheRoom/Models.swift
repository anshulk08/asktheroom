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

/// A side of the table in the camera's frame, as the rig's calibration names it. Same four
/// words as `Edge`, which in a snapshot with a `view` is already in the person's frame.
typealias Side = Edge

/// Which way the map faces (PROTOCOL.md, "view"). When a snapshot has one, its table size,
/// positions, edges and answer targets are all in the person's frame: x runs their left to
/// right, y far to near, and `bottom` is the side they sit at.
struct ViewInfo: Codable, Equatable {
    /// The camera-frame side of the table the person sits at.
    var front: Side
    /// The map is cropped to a real tabletop outline.
    var outline: Bool
    /// Names for camera sides, e.g. `right: "couch"`. Often empty.
    var sides: [Side: String]

    init(front: Side, outline: Bool = false, sides: [Side: String] = [:]) {
        self.front = front
        self.outline = outline
        self.sides = sides
    }

    private enum CodingKeys: String, CodingKey { case f, o, s }

    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        // An unknown side reads as the camera's own, which is what an unturned map shows.
        front = (try? c.decodeIfPresent(String.self, forKey: .f)).flatMap { Side(rawValue: $0) } ?? .bottom
        outline = (try? c.decodeIfPresent(Bool.self, forKey: .o)) ?? false
        let raw = (try? c.decodeIfPresent([String: String].self, forKey: .s)) ?? [:]
        sides = Dictionary(uniqueKeysWithValues: raw.compactMap { key, name in
            Side(rawValue: key).map { ($0, name) }
        })
    }

    func encode(to encoder: Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encode(front.rawValue, forKey: .f)
        try c.encode(outline, forKey: .o)
        try c.encode(Dictionary(uniqueKeysWithValues: sides.map { ($0.key.rawValue, $0.value) }), forKey: .s)
    }

    /// Where a camera side shows on this map.
    func viewerEdge(of side: Side) -> Edge { Seat.viewerEdge(of: side, front: front) }

    /// The camera side drawn at a map edge.
    func cameraSide(at edge: Edge) -> Side { Seat.cameraSide(at: edge, front: front) }

    /// The rig's name for the side at a map edge ("couch"), if it has one.
    func name(at edge: Edge) -> String? {
        sides[cameraSide(at: edge)].flatMap { $0.isEmpty ? nil : $0 }
    }
}

/// Turning between the camera's sides and the person's map edges. The rig turns camera
/// coordinates so the side the person sits at (`front`) is the map's bottom; this is the
/// same table, so the phone can say which camera side a tapped edge is.
enum Seat {
    /// Clockwise, as seen from above.
    private static let clockwise: [Edge] = [.top, .right, .bottom, .left]

    /// Quarter turns clockwise that bring `front` to the bottom.
    private static func turns(for front: Side) -> Int {
        (2 - clockwise.firstIndex(of: front)! + 4) % 4
    }

    /// The map edge a camera side lands on when the person sits at `front`.
    static func viewerEdge(of side: Side, front: Side) -> Edge {
        clockwise[(clockwise.firstIndex(of: side)! + turns(for: front)) % 4]
    }

    /// The camera side at a map edge when the person sits at `front`.
    static func cameraSide(at edge: Edge, front: Side) -> Side {
        clockwise[(clockwise.firstIndex(of: edge)! - turns(for: front) + 4) % 4]
    }

    /// Where helper settings keep the chosen seat: a camera side, or nothing for the rig's default.
    static let savedKey = "seatFront"
    /// Set when the person goes back to the rig's default, until the reset has gone out on a connect.
    static let resetKey = "seatReset"

    static var saved: Side? {
        UserDefaults.standard.string(forKey: savedKey).flatMap(Side.init(rawValue:))
    }

    /// Chooses a seat, or nil for the rig's default. Going back to the default owes the rig one
    /// reset, sent now and again on the next connect in case this one didn't reach it.
    static func choose(_ side: Side?) {
        let defaults = UserDefaults.standard
        defaults.set(side?.rawValue ?? "", forKey: savedKey)
        defaults.set(side == nil, forKey: resetKey)
    }

    /// What to write after a change: the seat, a reset if one is owed, or nothing.
    static var savedOrient: OrientSettings? {
        if let saved { return OrientSettings(front: saved) }
        return UserDefaults.standard.bool(forKey: resetKey) ? .reset : nil
    }

    /// What to write on connect. A reset goes out on one connect only; after that the rig
    /// already uses its default and nothing is sent.
    static func orientForConnect() -> OrientSettings? {
        let orient = savedOrient
        if orient == .reset { UserDefaults.standard.set(false, forKey: resetKey) }
        return orient
    }
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
    //   nothing, weak guess "something new"    (never its number; hidden from Recent/Dashboard)

    /// A guess counts from here up; below it the thing stays nameless.
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

    /// A thing with nothing to call it: shown as "something new", left out of Recent and Home.
    var isNameless: Bool { isThing && aliases.isEmpty && !isHedged }

    var displayName: String {
        hedgedName.map { "\($0)?" } ?? Entity.displayName(for: n, aliases: aliases)
    }

    /// How a sentence names it: "what looks like a tape roll" when hedged, else `displayName`.
    var phrase: String {
        hedgedName.map { Entity.looksLike($0) } ?? displayName
    }

    static let hedgePrefix = "what looks like "
    /// What a nameless thing is called. Its number means nothing to the person.
    static let namelessName = "something new"

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
            return namelessName
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
    /// Present when the rig has turned the map to the person's frame. Missing from older rigs,
    /// whose maps are in the camera's frame.
    var view: ViewInfo?
    /// State-characteristic chunks the bridge had sent before this message, for measuring link
    /// loss (`LinkStats`). Diagnostics only; missing from older bridges.
    var tx: Int?
    /// Hash of the room layout. `lay` comes only when it changes or on subscribe, so the store
    /// keeps the last one and checks it against this.
    var lh: String?
    var lay: RoomLayout?

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

/// The room around the table (state `lay`): its size, the table's place in it, named zones and
/// where the person is. Every field is optional, and a field of the wrong type is ignored rather
/// than failing the whole state message.
struct RoomLayout: Codable, Equatable {
    struct Table: Codable, Equatable {
        var rect: [Double]?
        var origin: [Double]?

        init(rect: [Double]? = nil, origin: [Double]? = nil) {
            self.rect = rect
            self.origin = origin
        }

        init(from decoder: Decoder) throws {
            let c = try? decoder.container(keyedBy: CodingKeys.self)
            rect = try? c?.decodeIfPresent([Double].self, forKey: .rect)
            origin = try? c?.decodeIfPresent([Double].self, forKey: .origin)
        }
    }

    struct Zone: Codable, Equatable {
        var id: String?
        var say: String?
        var rect: [Double]?
        var kind: String?

        init(id: String? = nil, say: String? = nil, rect: [Double]? = nil, kind: String? = nil) {
            self.id = id
            self.say = say
            self.rect = rect
            self.kind = kind
        }

        init(from decoder: Decoder) throws {
            let c = try? decoder.container(keyedBy: CodingKeys.self)
            id = try? c?.decodeIfPresent(String.self, forKey: .id)
            say = try? c?.decodeIfPresent(String.self, forKey: .say)
            rect = try? c?.decodeIfPresent([Double].self, forKey: .rect)
            kind = try? c?.decodeIfPresent(String.self, forKey: .kind)
        }
    }

    var v: Int?
    var size: [Double]?
    var front: String?
    var table: Table?
    var zones: [Zone]?
    var you: [Double]?

    init(v: Int? = nil, size: [Double]? = nil, front: String? = nil, table: Table? = nil,
         zones: [Zone]? = nil, you: [Double]? = nil) {
        self.v = v
        self.size = size
        self.front = front
        self.table = table
        self.zones = zones
        self.you = you
    }

    init(from decoder: Decoder) throws {
        let c = try? decoder.container(keyedBy: CodingKeys.self)
        v = try? c?.decodeIfPresent(Int.self, forKey: .v)
        size = try? c?.decodeIfPresent([Double].self, forKey: .size)
        front = try? c?.decodeIfPresent(String.self, forKey: .front)
        table = try? c?.decodeIfPresent(Table.self, forKey: .table)
        zones = try? c?.decodeIfPresent([Zone].self, forKey: .zones)
        you = try? c?.decodeIfPresent([Double].self, forKey: .you)
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

/// Where the person sits, sent to the rig so it turns the map to face them. `front` is a
/// camera side, or null to go back to the rig's configured seat. No answer comes back; the
/// next state has the new `view`.
struct OrientSettings: Codable, Equatable {
    struct Orient: Codable, Equatable {
        var front: String?

        private enum CodingKeys: String, CodingKey { case front }

        // Written out so a reset sends `"front":null`; the synthesized one would leave it out.
        func encode(to encoder: Encoder) throws {
            var c = encoder.container(keyedBy: CodingKeys.self)
            if let front {
                try c.encode(front, forKey: .front)
            } else {
                try c.encodeNil(forKey: .front)
            }
        }
    }

    var orient: Orient

    init(front: Side?) {
        orient = Orient(front: front?.rawValue)
    }

    /// Back to the rig's configured seat; the rig forgets the saved one.
    static let reset = OrientSettings(front: nil)

    /// Nil for a reset (or a side this app doesn't know).
    var front: Side? { orient.front.flatMap(Side.init(rawValue:)) }

    /// The JSON to write to the question characteristic, like `VoiceSettings`.
    func encoded() -> Data? {
        let encoder = JSONEncoder()
        encoder.outputFormatting = .sortedKeys
        guard let data = try? encoder.encode(self), data.count <= Question.maxBytes else { return nil }
        return data
    }
}

/// Written first on every connect (PROTOCOL.md): tells the bridge this app can inflate
/// compressed messages (`z`: 1), so it may set the COMPRESSED flag. Exactly `{"hello":{"z":1}}`.
struct Hello: Codable, Equatable {
    struct Caps: Codable, Equatable {
        var z: Int
    }

    var hello: Caps

    static let current = Hello(hello: Caps(z: 1))

    /// The JSON to write to the question characteristic, like `VoiceSettings`.
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
