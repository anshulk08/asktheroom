import XCTest
@testable import AskTheRoom

final class DashboardTests: XCTestCase {
    private let sample = MockData.sampleSnapshot

    func testThingsAreTargetsAndNamedUnknowns() {
        let names = Dashboard.things(in: sample).map(\.name)
        // Every configured object as a fixed tile (box and notebook too), a named thing, then the room's guesses.
        XCTAssertEqual(names, ["keys", "box", "pill_bottle", "notebook", "wallet", "phone", "glasses", "remote",
                               "thing:7", "thing:11", "thing:12"])
    }

    func testWhereaboutsInPlainWords() {
        func words(_ n: String) -> String { Dashboard.whereabouts(sample.entity(named: n)!, in: sample) }
        XCTAssertEqual(words("keys"), "Inside the box")
        XCTAssertEqual(words("pill_bottle"), "Under the notebook")
        XCTAssertEqual(words("wallet"), "On the table")
        XCTAssertEqual(words("phone"), "Off the table, on the left side")
        XCTAssertEqual(words("glasses"), "Can't see them right now")
        XCTAssertEqual(words("remote"), "Someone is holding it")
        XCTAssertEqual(words("thing:9"), "Probably on the table")
    }

    func testQuestionsKeepThePersonsOwnNames() {
        XCTAssertEqual(Dashboard.question(for: sample.entity(named: "keys")!), "Where are my keys?")
        XCTAssertEqual(Dashboard.question(for: sample.entity(named: "pill_bottle")!), "Where is my pill bottle?")
        XCTAssertEqual(Dashboard.question(for: sample.entity(named: "thing:7")!), "Where is my charger?")
    }

    func testNoticesOrderedGoneLostNew() {
        let notices = Dashboard.notices(in: sample)
        XCTAssertEqual(notices.map(\.kind), [.leftTable, .lostTrack, .unnamed, .unnamed])
        XCTAssertEqual(notices[0].text, "Your phone was moved off the table, on the left side.")
        XCTAssertEqual(notices[1].text, "The room can't see your glasses right now.")
        XCTAssertEqual(notices[2].text, "Something new is on the table. It looks like a phone charger.")
        XCTAssertEqual(notices[3].text, "Something new is on the table. It looks like a tape roll.")
        // thing:9 has only a weak guess: no card, or a crowded table buries the rest.
        XCTAssertFalse(notices.contains { $0.entity == "thing:9" })
    }

    func testGrokNamedThingsArentThePersons() {
        let tape = sample.entity(named: "thing:12")!
        XCTAssertTrue(Dashboard.things(in: sample).contains { $0.name == "thing:12" })   // on Home as the room's guess
        XCTAssertEqual(Dashboard.your(tape), "what looks like a tape roll")
        XCTAssertEqual(Dashboard.question(for: tape), "Where is the tape roll?")
        XCTAssertEqual(Dashboard.your(sample.entity(named: "thing:11")!), "what looks like a phone charger")

        var gone = sample
        gone.update("thing:12") { $0.s = .gone }
        XCTAssertFalse(Dashboard.notices(in: gone).contains { $0.entity == "thing:12" }, "not \"your tape roll\"")
    }

    func testChangesHedgeGuesses() {
        var new = sample
        new.update("thing:11") { $0.s = .held; $0.p = "hand:1" }
        new.update("thing:12") { $0.s = .lost }
        XCTAssertEqual(Dashboard.changes(from: sample, to: new).map(\.text),
                       ["What looks like a phone charger was picked up", "Lost track of what looks like a tape roll"])

        var appeared = sample
        appeared.e.append(Entity(n: "thing:13", k: .target, s: .visible, xy: TablePoint(x: 5, y: 5), g: "mug", gc: 0.9))
        XCTAssertEqual(Dashboard.changes(from: sample, to: appeared).map(\.text), ["What looks like a mug appeared on the table"])
    }

    func testCantSeeNoticeSaysWhenLastSeen() {
        let now = Date(timeIntervalSince1970: 2_000_000_000)
        var s = sample
        s.update("glasses") { $0.ls = nil }
        var notice = Dashboard.notices(in: s).first { $0.kind == .lostTrack }!
        XCTAssertEqual(notice.detail(now: now), "They were last seen on the table.")

        s.update("glasses") { $0.ls = now.addingTimeInterval(-5 * 60).timeIntervalSince1970 }
        s.update("wallet") { $0.s = .lost; $0.ls = now.timeIntervalSince1970 }
        let notices = Dashboard.notices(in: s).filter { $0.kind == .lostTrack }
        notice = notices.first { $0.entity == "glasses" }!
        XCTAssertEqual(notice.detail(now: now), "They were last seen on the table, 5 minutes ago.")
        XCTAssertEqual(notices.first { $0.entity == "wallet" }?.detail(now: now), "It was last seen on the table, just now.")
        // The time lives outside the text, so "Got it" sticks as the minutes tick by.
        XCTAssertEqual(notice.id, "glasses|The room can't see your glasses right now.")
    }

    func testYourForThePersonsThingsOnly() {
        func your(_ n: String) -> String { Dashboard.your(sample.entity(named: n)!) }
        XCTAssertEqual(your("keys"), "your keys")
        XCTAssertEqual(your("pill_bottle"), "your pill bottle")
        XCTAssertEqual(your("thing:7"), "my charger")
        XCTAssertEqual(your("box"), "the box")
        XCTAssertEqual(your("thing:9"), "something new")
    }

    func testLastSeenOnlyForThingsOutOfSight() {
        let now = Date(timeIntervalSince1970: 2_000_000_000)
        var s = sample
        s.update("phone") { $0.ls = now.addingTimeInterval(-120).timeIntervalSince1970 }
        s.update("wallet") { $0.ls = now.timeIntervalSince1970 }
        XCTAssertEqual(Dashboard.lastSeen(s.entity(named: "phone")!, now: now), "Last seen 2 minutes ago")
        XCTAssertNil(Dashboard.lastSeen(s.entity(named: "wallet")!, now: now))
    }

    func testNoticeGuessesOnlyNamesThePersonKnows() {
        var s = sample
        s.update("thing:11") { $0.m = [MaybeSame(name: "thing:7", score: 0.7)] }
        XCTAssertEqual(Dashboard.notices(in: s).first { $0.entity == "thing:11" }?.text,
                       "Something new is on the table. It looks like a phone charger. It might be my charger.")
    }

    func testChangesDescribeEachMove() {
        var new = sample
        let t = Date(timeIntervalSince1970: 2_000_000_000)
        new.t = t.timeIntervalSince1970
        new.update("wallet") { $0.s = .inside; $0.p = "box" }
        new.update("phone") { $0.s = .visible; $0.edge = nil; $0.r = TablePoint(x: 10, y: 30) }
        new.update("pill_bottle") { $0.s = .held; $0.p = "hand:1" }
        new.update("glasses") { $0.s = .visible }
        new.update("remote") { $0.p = "hand:3" }           // hand to hand: not news
        new.move("box", to: TablePoint(x: 40, y: 20), now: t.timeIntervalSince1970)

        let lines = Dashboard.changes(from: sample, to: new).map(\.text)
        XCTAssertEqual(lines, [
            "Box moved",
            "Pill bottle was picked up",
            "Wallet went into the box",
            "Phone came back to the table",
            "Found the glasses again",
        ])
        XCTAssertTrue(Dashboard.changes(from: sample, to: new).allSatisfy { $0.time == t })
    }

    func testPillWordingStaysNeutral() {
        var new = sample
        for status in [EntityStatus.held, .visible, .gone, .inside, .under, .lost] {
            new.update("pill_bottle") { $0.s = status; $0.edge = status == .gone ? .left : nil }
            let pill = new.entity(named: "pill_bottle")!
            let notices = Dashboard.notices(in: new).filter { $0.entity == "pill_bottle" }
            let lines = Dashboard.changes(from: sample, to: new).map(\.text)
                + notices.map(\.text) + notices.compactMap { $0.detail(now: Date()) }
                + [Dashboard.whereabouts(pill, in: new)]
            for line in lines {
                XCTAssertFalse(line.lowercased().contains("taken"), line)
                XCTAssertFalse(line.lowercased().contains("took"), line)
            }
        }
    }

    func testPluralNames() {
        var new = sample
        new.update("keys") { $0.s = .held; $0.p = "hand:1" }
        XCTAssertEqual(Dashboard.changes(from: sample, to: new).map(\.text), ["Keys were picked up"])
    }

    func testSmallMovesAreIgnored() {
        var new = sample
        new.update("wallet") { $0.r = TablePoint(x: 65, y: 18) }
        XCTAssertTrue(Dashboard.changes(from: sample, to: new).isEmpty)
    }

    func testNewNamelessThingIsLeftOutOfRecent() {
        var new = sample
        new.e.append(Entity(n: "thing:14", k: .target, s: .visible, p: nil, xy: TablePoint(x: 5, y: 5), r: nil,
                            c: 1, edge: nil, a: nil, m: nil, ls: nil))
        XCTAssertTrue(Dashboard.changes(from: sample, to: new).isEmpty)
    }

    /// Recent and Home skip things with no name; named things and the room's guesses stay.
    func testNamelessThingsAreHiddenFromRecentAndHome() {
        var new = sample
        new.update("keys") { $0.s = .held; $0.p = "hand:1" }
        new.update("thing:9") { $0.s = .held; $0.p = "hand:2" }
        new.update("thing:11") { $0.s = .gone; $0.edge = .right }
        let changes = Dashboard.changes(from: sample, to: new)
        XCTAssertEqual(changes.map(\.entity), ["keys", "thing:11"])
        XCTAssertEqual(changes.map(\.text), ["Keys were picked up", "What looks like a phone charger left the table on the right"])
        XCTAssertFalse(Dashboard.things(in: sample).contains { $0.isNameless })
    }

    func testNamelessParentReadsNaturally() {
        var s = sample
        s.e.append(Entity(n: "thing:13", k: .container, s: .visible, xy: TablePoint(x: 5, y: 5)))
        s.update("wallet") { $0.s = .inside; $0.p = "thing:13" }
        XCTAssertEqual(Dashboard.whereabouts(s.entity(named: "wallet")!, in: s), "Inside something new")
    }

    func testRecentMergesChangesAndQuestionsNewestFirst() {
        let now = Date(timeIntervalSince1970: 2_000_000_000)
        let activity = [
            ActivityEvent(entity: "keys", text: "Keys went into the box", time: now.addingTimeInterval(-60)),
            ActivityEvent(entity: "phone", text: "Phone moved", time: now.addingTimeInterval(-2 * 3600)),
        ]
        var asked = Exchange(id: 1, question: "Where are my keys?")
        asked.askedAt = now.addingTimeInterval(-30)
        var old = Exchange(id: 2, question: "What changed?")
        old.askedAt = now.addingTimeInterval(-3 * 3600)

        let groups = Dashboard.recent(activity: activity, exchanges: [asked, old], now: now)
        XCTAssertEqual(groups.lastHour.map(\.id), ["q1", "c\(activity[0].id)"])
        XCTAssertEqual(groups.earlier.map(\.id), ["c\(activity[1].id)", "q2"])
    }

    func testGreeting() {
        var cal = Calendar(identifier: .gregorian)
        cal.timeZone = TimeZone(identifier: "America/New_York")!
        func at(_ h: Int) -> Date { cal.date(from: DateComponents(year: 2026, month: 9, day: 26, hour: h))! }
        XCTAssertEqual(Dashboard.greeting(at: at(8), calendar: cal), "Good morning")
        XCTAssertEqual(Dashboard.greeting(at: at(14), calendar: cal), "Good afternoon")
        XCTAssertEqual(Dashboard.greeting(at: at(21), calendar: cal), "Good evening")
        XCTAssertEqual(Dashboard.greeting(at: at(2), calendar: cal), "Good evening")
    }
}

@MainActor
final class RoomStoreActivityTests: XCTestCase {
    func testStoreLogsChangesNewestFirst() {
        let store = RoomStore()
        let first = MockData.sampleSnapshot
        store.receive(state: first)
        XCTAssertTrue(store.activity.isEmpty, "the first snapshot is the baseline, not news")

        var second = first
        second.update("wallet") { $0.s = .held; $0.p = "hand:1" }
        store.receive(state: second)
        var third = second
        third.update("wallet") { $0.s = .visible; $0.p = nil }
        store.receive(state: third)
        XCTAssertEqual(store.activity.map(\.text), ["Wallet was put down", "Wallet was picked up"])

        store.showOnMap("thing:11")
        XCTAssertEqual(store.highlight?.entity, "thing:11")
        XCTAssertEqual(store.highlight?.target, TablePoint(x: 45, y: 24))
    }

    func testRecentTabLeavesOutNamelessThings() {
        let store = RoomStore()
        let first = MockData.sampleSnapshot
        store.receive(state: first)
        var second = first
        second.update("keys") { $0.s = .held; $0.p = "hand:1" }
        second.update("thing:9") { $0.s = .held; $0.p = "hand:2" }
        second.update("thing:11") { $0.s = .gone; $0.edge = .right }
        store.receive(state: second)

        let recent = Dashboard.recent(activity: store.activity, exchanges: [], now: second.time!)
        let shown = recent.lastHour.compactMap { entry -> String? in
            if case .change(let e) = entry { return e.entity } else { return nil }
        }
        XCTAssertEqual(Set(shown), ["keys", "thing:11"])
    }
}
