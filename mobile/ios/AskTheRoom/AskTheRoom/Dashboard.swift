import Foundation

// What the Home dashboard shows, worked out from snapshots alone (no protocol changes):
// where each thing is in plain words, what the room noticed, and a log of recent changes.

/// One change the app saw between two snapshots, e.g. "keys went into box".
struct ActivityEvent: Identifiable, Equatable {
    let id = UUID()
    var entity: String
    var text: String
    var time: Date

    static func == (a: ActivityEvent, b: ActivityEvent) -> Bool {
        a.entity == b.entity && a.text == b.text && a.time == b.time
    }
}

/// Something worth telling the person without being asked ("The room noticed").
struct Notice: Identifiable, Equatable {
    /// `rig`: a reminder or morning report the rig fired, with its `kind`.
    enum Kind: Equatable { case leftTable, lostTrack, unnamed, rig(String?) }

    var kind: Kind
    var entity: String
    var text: String
    /// What "Help me find it" asks the room; nil means just show it on the map.
    var question: String?
    /// For "can't see": when it was last seen, shown under the text. Kept out of `text`
    /// so the notice doesn't change (and come back after "Got it") every minute.
    var lastSeen: Date? = nil
    var seenWords: String? = nil
    /// The rig's notice id, for notices the rig sent.
    var rigID: Int? = nil

    /// "They were last seen on the table, 5 minutes ago."
    func detail(now: Date) -> String? {
        guard let seenWords else { return nil }
        return seenWords + (lastSeen.map { ", \(Dashboard.ago($0, now: now).lowercased())" } ?? "") + "."
    }

    /// Changes when the situation changes, so a dismissed notice comes back if it happens again.
    var id: String { rigID.map { "rig|\($0)" } ?? "\(entity)|\(text)" }

    /// The rig's own words, unchanged: it already keeps pill wording neutral.
    init(rig answer: Answer) {
        self.init(kind: .rig(answer.kind), entity: answer.pointAt ?? "", text: answer.text, rigID: answer.nid)
    }

    init(kind: Kind, entity: String, text: String, question: String? = nil,
         lastSeen: Date? = nil, seenWords: String? = nil, rigID: Int? = nil) {
        self.kind = kind
        self.entity = entity
        self.text = text
        self.question = question
        self.lastSeen = lastSeen
        self.seenWords = seenWords
        self.rigID = rigID
    }
}

/// A line on the Recent tab: something that changed, or something the person asked.
enum RecentEntry: Identifiable, Equatable {
    case change(ActivityEvent)
    case question(Exchange)

    var id: String {
        switch self {
        case .change(let e): return "c\(e.id)"
        case .question(let x): return "q\(x.id)"
        }
    }

    var time: Date {
        switch self {
        case .change(let e): return e.time
        case .question(let x): return x.askedAt
        }
    }
}

enum Dashboard {
    /// Moves smaller than this are sensor jitter, not worth a line on the Recent tab.
    static let moveThreshold = 15.0
    static let activityLimit = 50
    static let recentHour: TimeInterval = 60 * 60

    /// Home's "Your things": targets, unknown objects someone has named, and things in sight the room
    /// has a guess for ("remote control?"), in sight first. Containers and covers are furniture here;
    /// they show up in "where" words instead. A configured prop the room has never seen is left out:
    /// eight "Can't see it right now" tiles read as broken (rig, 27 Sep 06:40).
    static func things(in snapshot: Snapshot) -> [Entity] {
        let shown = snapshot.entities.filter { (isTheirs($0) || isGuessedInSight($0)) && !neverSeen($0) }
        return shown.enumerated()
            .sorted { (inSight($0.element) ? 0 : 1, $0.offset) < (inSight($1.element) ? 0 : 1, $1.offset) }
            .map(\.element)
    }

    /// A thing the room only has a guess for, somewhere it can be shown now.
    static func isGuessedInSight(_ e: Entity) -> Bool {
        e.kind == .target && e.isHedged && inSight(e) && (e.xy != nil || e.r != nil || e.zone != nil)
    }

    static func inSight(_ e: Entity) -> Bool {
        [.visible, .held, .inside, .under].contains(e.status)
    }

    /// A configured prop with no sighting at all yet.
    static func neverSeen(_ e: Entity) -> Bool {
        !e.isThing && e.status == .lost && e.lastSeen == nil && e.zone == nil
    }

    /// Something the person owns and named: "your keys", never "your tape roll" from Grok.
    static func isTheirs(_ e: Entity) -> Bool {
        e.kind == .target && (!e.isThing || e.hasTaughtName)
    }

    /// Where a thing off the table is, in the room: "on the couch", "by the doorway",
    /// "somewhere in the room". Nil for a thing on the table.
    static func place(of e: Entity, in snapshot: Snapshot) -> String? {
        e.zone.map { place(zone: $0, layout: snapshot.lay) }
    }

    /// The layout's own words for a zone ("the kitchen counter"), else its id tidied up
    /// ("side_table" -> "the side table"). `room` is the registry's "somewhere, no zone drawn".
    static func place(zone id: String, layout: RoomLayout?) -> String {
        if id == "room" { return "somewhere in the room" }
        if let zone = layout?.zones?.first(where: { $0.id == id }) {
            let say = zone.say?.trimmingCharacters(in: .whitespaces) ?? ""
            let name = say.isEmpty ? "the " + id.replacingOccurrences(of: "_", with: " ") : say
            return (zone.kind == "door" ? "by " : "on ") + name
        }
        return "on the " + id.replacingOccurrences(of: "_", with: " ")
    }

    /// "On the couch", "Inside the box, on the couch", "Last seen by the doorway"… for a thing
    /// in a room zone; nil for one on the table.
    static func roomWhereabouts(_ e: Entity, in snapshot: Snapshot) -> String? {
        guard let place = place(of: e, in: snapshot) else { return nil }
        let words: String
        switch e.presence {
        case .seen:
            words = capitalized(place)
        case .hidden:
            let parent = e.parent.map { snapshot.entity(named: $0)?.displayName ?? Entity.displayName(for: $0) }
            words = parent.map { "\(e.status == .under ? "Under" : "Inside") \(the($0)), \(place)" } ?? "Hidden \(place)"
        case .carried:
            words = "Carried, last seen \(place)"
        case .lastSeen:
            return "Last seen \(place)"
        }
        return e.isUncertain || e.isTentative ? "Probably " + words.prefix(1).lowercased() + words.dropFirst() : words
    }

    /// "Seen on the couch at 4:37 PM": where the camera last saw a thing (state `sg`).
    static func sightingWords(_ s: Sighting, in snapshot: Snapshot, now: Date = Date()) -> String {
        "Seen \(place(zone: s.zone, layout: snapshot.lay)) at \(Banners.when(s.time, now: now))"
    }

    /// The sighting words for a thing the map can't otherwise place, else nil.
    static func sightingWhereabouts(_ e: Entity, in snapshot: Snapshot, now: Date = Date()) -> String? {
        snapshot.sighting(for: e.name).map { sightingWords($0, in: snapshot, now: now) }
    }

    /// "On the table", "Inside the box", "Off the table, on the left side"… Sentence case.
    static func whereabouts(_ e: Entity, in snapshot: Snapshot) -> String {
        if let words = sightingWhereabouts(e, in: snapshot) { return words }
        if let words = roomWhereabouts(e, in: snapshot) { return words }
        let parent = e.parent.map { snapshot.entity(named: $0)?.displayName ?? Entity.displayName(for: $0) }
        let it = isPlural(e.displayName) ? "them" : "it"
        let words: String
        switch e.status {
        case .visible: words = "On the table"
        case .held: words = "Someone is holding \(it)"
        case .inside: words = parent.map { "Inside \(the($0))" } ?? "Inside something"
        case .under: words = parent.map { "Under \(the($0))" } ?? "Under something"
        case .gone: words = e.edge.map { "Off the table, on the \($0.rawValue) side" } ?? "Off the table"
        case .lost: words = "Can't see \(it) right now"
        case .unrecognized: words = "Unknown"
        }
        guard e.isUncertain, e.status != .lost else { return words }
        return "Probably " + words.prefix(1).lowercased() + words.dropFirst()
    }

    /// "Last seen 5 minutes ago", for things off the table or out of sight.
    static func lastSeen(_ e: Entity, now: Date) -> String? {
        guard [.gone, .lost].contains(e.status) || e.presence == .lastSeen, let seen = e.lastSeen else { return nil }
        return "Last seen \(ago(seen, now: now).lowercased())"
    }

    /// The question a tap on a thing asks. A name Grok gave isn't the person's: "the tape roll".
    static func question(for e: Entity) -> String {
        if let name = e.hedgedName { return "Where is the \(name)?" }
        return "Where \(isPlural(e.displayName) ? "are" : "is") \(e.displayName.hasPrefix("my ") ? "" : "my ")\(e.displayName)?"
    }

    /// Things to point out, most urgent first: gone, then lost, then newcomers the room has a
    /// guess for. Nameless things get no card: a crowded table would bury the rest in them.
    static func notices(in snapshot: Snapshot) -> [Notice] {
        var out: [Notice] = []
        for e in snapshot.entities {
            let name = e.displayName
            // Off the table but somewhere in the room: the room knows where, so no "left the table".
            let place = self.place(of: e, in: snapshot)
            switch e.status {
            case .gone where (!e.isThing || e.hasTaughtName) && place == nil:
                let side = e.edge.map { ", on the \($0.rawValue) side" } ?? ""
                out.append(Notice(kind: .leftTable, entity: e.name,
                                  text: "\(capitalized(your(e))) \(was(name)) moved off the table\(side).",
                                  question: question(for: e)))
            case .lost where !e.isThing:
                out.append(Notice(kind: .lostTrack, entity: e.name,
                                  text: "The room can't see \(your(e)) right now.",
                                  question: question(for: e),
                                  lastSeen: e.lastSeen,
                                  seenWords: "\(isPlural(name) ? "They" : "It") \(was(name)) last seen \(place ?? "on the table")"))
            default:
                break
            }
            if e.isThing, !e.hasTaughtName, !e.isNameless, e.status == .visible {
                // The room's own guess, hedged, then an older thing only if the person named it.
                let looks = e.hedgedName.map { " It looks like \(Entity.withArticle($0))." } ?? ""
                let known = e.maybeSameAs.lazy.compactMap { snapshot.entity(named: $0.name) }
                    .first { !$0.isThing || $0.hasTaughtName }
                let guess = known.map { " It might be \(your($0))." } ?? ""
                out.append(Notice(kind: .unnamed, entity: e.name,
                                  text: "Something new is \(place ?? "on the table").\(looks)\(guess)",
                                  question: nil))
            }
        }
        let order: [Notice.Kind] = [.leftTable, .lostTrack, .unnamed]
        return out.enumerated()
            .sorted { (order.firstIndex(of: $0.element.kind)!, $0.offset) < (order.firstIndex(of: $1.element.kind)!, $1.offset) }
            .map(\.element)
    }

    /// What changed from `old` to `new`, one line per entity. Neutral wording throughout:
    /// the pill bottle is "picked up", never "taken". Nameless things are left out: a line
    /// about "something new" in Recent says nothing the map doesn't.
    static func changes(from old: Snapshot, to new: Snapshot, now: Date = Date()) -> [ActivityEvent] {
        let time = new.time ?? now
        var out: [ActivityEvent] = []
        for e in new.entities where !e.isNameless {
            let name = e.phrase
            let place = self.place(of: e, in: new)
            guard let before = old.entity(named: e.name) else {
                if e.kind == .target {
                    out.append(ActivityEvent(entity: e.name, text: "\(capitalized(name)) appeared \(place ?? "on the table")", time: time))
                }
                continue
            }
            // Moving between the table and the room's zones says where it went.
            if e.zone != before.zone {
                var text: String?
                if let place, e.presence != .lastSeen {
                    text = "\(capitalized(name)) \(isPlural(name) ? "are" : "is") \(place) now"
                } else if e.zone == nil, e.status == .visible {
                    text = "\(capitalized(name)) came back to the table"
                }
                if let text {
                    out.append(ActivityEvent(entity: e.name, text: text, time: time))
                    continue
                }
            }
            let parent = e.parent.map { new.entity(named: $0)?.displayName ?? Entity.displayName(for: $0) }
            let oldParent = before.parent.map { old.entity(named: $0)?.displayName ?? Entity.displayName(for: $0) }
            var text: String?
            if e.status != before.status || (e.parent != before.parent && !(e.isInHand && before.isInHand)) {
                switch e.status {
                case .inside: text = "\(capitalized(name)) went into \(the(parent ?? "something"))"
                case .under: text = "\(capitalized(name)) went under \(the(parent ?? "something"))"
                case .held: text = "\(capitalized(name)) \(was(name)) picked up"
                case .gone: text = "\(capitalized(name)) left the table" + (e.edge.map { " on the \($0.rawValue)" } ?? "")
                case .lost: text = "Lost track of \(the(name))"
                case .visible:
                    switch before.status {
                    case .inside: text = "\(capitalized(name)) came out of \(the(oldParent ?? "something"))"
                    case .under: text = "\(capitalized(name)) \(was(name)) uncovered"
                    case .held: text = "\(capitalized(name)) \(was(name)) put down"
                    case .gone: text = "\(capitalized(name)) came back to the table"
                    case .lost: text = "Found \(the(name)) again"
                    default: text = nil
                    }
                case .unrecognized: text = nil
                }
            } else if e.status == .visible, let a = before.drawPoint, let b = e.drawPoint,
                      hypot(a.x - b.x, a.y - b.y) >= moveThreshold {
                text = "\(capitalized(name)) moved"
            }
            if let text { out.append(ActivityEvent(entity: e.name, text: text, time: time)) }
        }
        return out
    }

    /// Changes and questions together, newest first, split into the last hour and earlier.
    static func recent(activity: [ActivityEvent], exchanges: [Exchange], now: Date = Date())
        -> (lastHour: [RecentEntry], earlier: [RecentEntry]) {
        let all = (activity.map(RecentEntry.change) + exchanges.map(RecentEntry.question))
            .enumerated()
            // Newest first; ties keep their order (activity is already newest first).
            .sorted { ($0.element.time, -$0.offset) > ($1.element.time, -$1.offset) }
            .map(\.element)
        let cutoff = now.addingTimeInterval(-recentHour)
        return (all.filter { $0.time >= cutoff }, all.filter { $0.time < cutoff })
    }

    /// "the box", but "my charger" stays as the person named it, a guess stays hedged,
    /// and "something new" takes no article.
    static func the(_ name: String) -> String {
        name.hasPrefix("my ") || name.hasPrefix(Entity.hedgePrefix) || name == "something" || name == Entity.namelessName
            ? name : "the \(name)"
    }

    /// "your keys" for the person's things, "the box" for furniture, "something new" for
    /// nameless things, "my charger" as the person named it, "what looks like a tape roll"
    /// for the room's guess.
    static func your(_ e: Entity) -> String {
        if e.isHedged { return e.phrase }
        let name = e.displayName
        if name.hasPrefix("my ") { return name }
        return isTheirs(e) ? "your \(name)" : the(name)
    }

    /// Names that take "are": "Where are my keys?", "Keys were picked up".
    static let pluralNames: Set<String> = ["keys", "glasses", "headphones", "earbuds", "scissors"]

    static func isPlural(_ name: String) -> Bool {
        pluralNames.contains(name.lowercased())
    }

    static func was(_ name: String) -> String {
        isPlural(name) ? "were" : "was"
    }

    static func capitalized(_ s: String) -> String {
        s.prefix(1).uppercased() + s.dropFirst()
    }

    static func ago(_ date: Date, now: Date) -> String {
        let seconds = max(0, now.timeIntervalSince(date))
        if seconds < 60 { return "Just now" }
        let f = RelativeDateTimeFormatter()
        f.unitsStyle = .full
        return f.localizedString(for: date, relativeTo: now)
    }

    /// "Good morning" etc., for the top of Home.
    static func greeting(at date: Date, calendar: Calendar = .current) -> String {
        switch calendar.component(.hour, from: date) {
        case 5..<12: return "Good morning"
        case 12..<17: return "Good afternoon"
        default: return "Good evening"
        }
    }
}
