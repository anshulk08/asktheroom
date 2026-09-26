import Foundation

/// A pretend rig for building the UI without Bluetooth (spec section 7, "Mock mode").
/// Starts from the section 4 sample, then loops a scripted story: keys into the box,
/// the box moves, the phone is carried off the left edge. Answers are faked locally.
///
/// Launch arguments, for demos and screenshots:
///   -mock YES          start in mock mode
///   -mockPaused YES    hold the sample snapshot still
///   -mockOffline YES   report the cloud voice as unavailable
///   -mockAppDown YES   report the room app as down
///   -mockAsk "a|b"     ask these, a second after launch and then every 2.5 s; on Home each opens the answer sheet
///   -mockSelect name   open this entity's detail sheet (with -mockTab table: pick it on the map)
///   -mockFocus name    open the answer sheet on this entity
///   -mockTab table     open on the Table (or `recent`) tab rather than Home
///   -mockScroll YES    scroll Home to the bottom
///   -mockSettings YES  open helper settings
@MainActor
final class MockRoom: RoomTransport {
    static let stepInterval: Duration = .seconds(5)
    static let laserOnFor: Duration = .seconds(5)
    static let answerDelay: Duration = .milliseconds(700)

    private weak var store: RoomStore?
    private(set) var snapshot: Snapshot
    private(set) var lastChange = "Your keys went into the box, and the box moved."
    private var step = 0
    private var tasks: [Task<Void, Never>] = []
    private var laserTask: Task<Void, Never>?

    init(store: RoomStore, defaults: UserDefaults = .standard, autoplay: Bool = true) {
        self.store = store
        snapshot = Self.startingSnapshot()
        snapshot.online = !defaults.bool(forKey: "mockOffline")

        store.receive(status: RigStatus(
            app: defaults.bool(forKey: "mockAppDown") ? "down" : "up",
            fps: 15, online: snapshot.online, cal: true, laser_cal: true))
        publish()
        laserOff(after: Self.laserOnFor)

        if autoplay, !defaults.bool(forKey: "mockPaused") {
            tasks.append(Task { [weak self] in
                while !Task.isCancelled {
                    try? await Task.sleep(for: Self.stepInterval)
                    self?.advance()
                }
            })
        }
        if let script = defaults.string(forKey: "mockAsk") {
            tasks.append(Task { [weak store] in
                try? await Task.sleep(for: .seconds(1))
                for q in script.split(separator: "|") {
                    store?.ask(String(q))
                    try? await Task.sleep(for: .seconds(2.5))
                }
            })
        }
    }

    /// The spec sample, moved so its times are "now" and the detail sheet reads sensibly.
    static func startingSnapshot(now: Date = Date()) -> Snapshot {
        var s = MockData.sampleSnapshot
        let offset = now.timeIntervalSince1970 - (s.t ?? now.timeIntervalSince1970)
        s.t = now.timeIntervalSince1970
        for i in s.e.indices {
            s.e[i].ls = s.e[i].ls.map { $0 + offset }
        }
        s.update("pill_bottle") { $0.ls = now.timeIntervalSince1970 - 25 * 60 }
        return s
    }

    func stop() {
        tasks.forEach { $0.cancel() }
        laserTask?.cancel()
    }

    func send(_ question: Question) {
        let answer = Self.answer(to: question.q, id: question.id, in: snapshot, lastChange: lastChange)
        tasks.append(Task { [weak self] in
            try? await Task.sleep(for: Self.answerDelay)
            guard let self, !Task.isCancelled else { return }
            if let name = answer.pointAt {
                self.snapshot.laser = LaserState(on: true, target: name)
                self.publish()
                self.laserOff(after: Self.laserOnFor)
            }
            self.store?.receive(answer: answer)
        })
    }

    // MARK: Story

    private struct Step {
        let change: String
        let apply: (inout Snapshot, Double) -> Void
    }

    private static let story: [Step] = [
        Step(change: "Your phone came back to the table.") { s, now in
            s.update("phone") { $0.s = .visible; $0.xy = TablePoint(x: 12, y: 30); $0.r = $0.xy; $0.edge = nil; $0.ls = now }
        },
        Step(change: "You took your keys out of the box.") { s, now in
            s.update("keys") { $0.s = .visible; $0.p = nil; $0.xy = TablePoint(x: 41.2, y: 29); $0.r = $0.xy; $0.ls = now }
        },
        Step(change: "You picked up your keys.") { s, now in
            s.update("keys") { $0.s = .held; $0.p = "hand:1"; $0.xy = TablePoint(x: 55, y: 32); $0.r = $0.xy; $0.ls = now }
        },
        Step(change: "Your keys went into the box.") { s, now in
            let box = s.entity(named: "box")?.r
            s.update("keys") { $0.s = .inside; $0.p = "box"; $0.r = box; $0.ls = now }
        },
        Step(change: "The box moved, with your keys inside.") { s, now in
            s.move("box", to: TablePoint(x: 50, y: 18), now: now)
        },
        Step(change: "You picked up your phone.") { s, now in
            s.update("phone") { $0.s = .held; $0.p = "hand:1"; $0.xy = TablePoint(x: 8, y: 30); $0.r = $0.xy; $0.ls = now }
        },
        Step(change: "Your phone left the table on the left.") { s, now in
            s.update("phone") { $0.s = .gone; $0.p = nil; $0.xy = TablePoint(x: 3, y: 30); $0.r = nil; $0.edge = .left; $0.ls = now }
        },
        Step(change: "The box moved, with your keys inside.") { s, now in
            s.move("box", to: TablePoint(x: 70.4, y: 38.1), now: now)
        },
    ]

    private func advance() {
        let now = Date().timeIntervalSince1970
        let next = Self.story[step % Self.story.count]
        step += 1
        next.apply(&snapshot, now)
        snapshot.t = now
        lastChange = next.change
        publish()
    }

    private func publish() {
        store?.receive(state: snapshot)
    }

    private func laserOff(after delay: Duration) {
        laserTask?.cancel()
        laserTask = Task { [weak self] in
            try? await Task.sleep(for: delay)
            guard let self, !Task.isCancelled else { return }
            self.snapshot.laser = LaserState(on: false, target: nil)
            self.publish()
        }
    }

    // MARK: Fake answers

    /// Roughly what the rig says (Software Spec answer templates), enough to exercise the UI.
    static func answer(to question: String, id: Int, in snapshot: Snapshot, lastChange: String, now: Date = Date()) -> Answer {
        let words = Set(stems(of: question))
        if words.contains("chang") || words.contains("happen") {
            return Answer(id: id, ok: true, text: "Most recently: \(lastChange.prefix(1).lowercased() + lastChange.dropFirst())", ms: 700)
        }
        guard let entity = bestMatch(for: words, in: snapshot) else {
            return Answer(id: id, ok: false, text: "I can't see anything like that on the table.", ms: 700)
        }

        let subject = spokenName(entity)
        let verb = isPlural(entity) ? "are" : "is"
        let probably = entity.isUncertain ? "probably " : ""
        let position = MapLayout.position(of: entity)

        if words.contains("last"), words.contains("pick") || words.contains("mov") || words.contains("touch") {
            let when = entity.ls.map { ago(now.timeIntervalSince1970 - $0) }
            let text = when.map { "You last picked up \(subject) \($0)." } ?? "I haven't seen \(subject) picked up yet."
            return Answer(id: id, ok: true, text: capitalized(text), point_at: entity.name, action: "point", target: position, ms: 700)
        }

        var action = "point"
        let text: String
        switch entity.status {
        case .visible, .unrecognized:
            text = "\(subject) \(verb) \(probably)on the table."
        case .inside, .under:
            text = "\(subject) \(verb) \(probably)\(whereHidden(entity, in: snapshot))."
        case .held:
            text = "Someone is \(probably)holding \(subject)."
        case .gone:
            let side = entity.edge.map { $0 == .left || $0 == .right ? " on the \($0.rawValue)" : " at the \($0.rawValue)" } ?? ""
            text = "\(subject) left the table\(side)."
            action = entity.edge.map { "sweep:\($0.rawValue)" } ?? "point"
        case .lost:
            text = "I lost track of \(subject). I last saw \(isPlural(entity) ? "them" : "it") here."
            action = "circle"
        }
        return Answer(id: id, ok: true, text: capitalized(text), point_at: entity.name, action: action, target: position, ms: 700)
    }

    private static let stopWords: Set<String> = ["my", "the", "where", "are", "is", "did", "unnamed", "object", "thing", "what", "when", "you", "see"]

    static func stems(of text: String) -> [String] {
        text.lowercased()
            .split { !$0.isLetter }
            .map { word in
                var w = String(word)
                if w.hasSuffix("ed"), w.count > 4 { w.removeLast(2) }
                if w.hasSuffix("e"), w.count > 3 { w.removeLast() }
                if w.hasSuffix("s"), w.count > 3 { w.removeLast() }
                return w
            }
            .filter { $0.count > 2 && !stopWords.contains($0) }
    }

    /// The entity whose name or aliases share the most words with the question; targets win ties.
    static func bestMatch(for words: Set<String>, in snapshot: Snapshot) -> Entity? {
        let scored = snapshot.entities.map { e -> (Entity, Int) in
            let names = [e.displayName, e.name.replacingOccurrences(of: "_", with: " ")] + e.aliases
            let terms = Set(names.flatMap { stems(of: $0) })
            return (e, terms.intersection(words).count)
        }
        return scored
            .filter { $0.1 > 0 }
            .max { a, b in a.1 != b.1 ? a.1 < b.1 : (a.0.kind != .target && b.0.kind == .target) }?
            .0
    }

    private static func spokenName(_ e: Entity) -> String {
        if e.kind != .target { return "the \(e.displayName)" }
        if e.isThing {
            guard let alias = e.aliases.first else { return "that object" }
            return alias.hasPrefix("my ") ? "your " + alias.dropFirst(3) : "the \(alias)"
        }
        return "your \(e.displayName)"
    }

    private static func isPlural(_ e: Entity) -> Bool {
        !e.isThing && e.displayName.hasSuffix("s")
    }

    /// "inside the box", or for chains "inside the wallet, under the notebook".
    private static func whereHidden(_ e: Entity, in snapshot: Snapshot) -> String {
        let chain = snapshot.chain(from: e.name)
        return zip(chain, chain.dropFirst())
            .map { child, parent in "\(child.status == .under ? "under" : "inside") the \(parent.displayName)" }
            .joined(separator: ", ")
    }

    private static func capitalized(_ s: String) -> String {
        s.prefix(1).uppercased() + s.dropFirst()
    }

    static func ago(_ seconds: Double) -> String {
        let minutes = Int(seconds / 60)
        switch minutes {
        case ..<1: return "just now"
        case 1: return "a minute ago"
        case ..<60: return "\(minutes) minutes ago"
        case ..<120: return "an hour ago"
        default: return "\(minutes / 60) hours ago"
        }
    }
}

extension Snapshot {
    mutating func update(_ name: String, _ change: (inout Entity) -> Void) {
        guard let i = e.firstIndex(where: { $0.n == name }) else { return }
        change(&e[i])
    }

    /// Moves an entity and everything hidden in or under it, as the rig would report.
    mutating func move(_ name: String, to point: TablePoint, now: Double) {
        update(name) { $0.xy = point; $0.r = point; $0.ls = now }
        for child in e where child.p == name && (child.s == .inside || child.s == .under) {
            move(child.n, to: point, now: now)
        }
    }
}
