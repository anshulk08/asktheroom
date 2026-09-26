import XCTest
@testable import AskTheRoom

final class DashboardTests: XCTestCase {
    private let sample = MockData.sampleSnapshot

    func testThingsAreTargetsAndNamedUnknowns() {
        let names = Dashboard.things(in: sample).map(\.name)
        XCTAssertEqual(names, ["keys", "pill_bottle", "wallet", "phone", "glasses", "remote", "thing:7"])
    }

    func testWhereaboutsInPlainWords() {
        func words(_ n: String) -> String { Dashboard.whereabouts(sample.entity(named: n)!, in: sample) }
        XCTAssertEqual(words("keys"), "Inside the box")
        XCTAssertEqual(words("pill_bottle"), "Under the notebook")
        XCTAssertEqual(words("wallet"), "On the table")
        XCTAssertEqual(words("phone"), "Left the table (left side)")
        XCTAssertEqual(words("glasses"), "Not sure where")
        XCTAssertEqual(words("remote"), "In someone's hand")
        XCTAssertEqual(words("thing:9"), "Probably on the table")
    }

    func testQuestionsKeepThePersonsOwnNames() {
        XCTAssertEqual(Dashboard.question(for: sample.entity(named: "keys")!), "Where are my keys?")
        XCTAssertEqual(Dashboard.question(for: sample.entity(named: "pill_bottle")!), "Where is my pill bottle?")
        XCTAssertEqual(Dashboard.question(for: sample.entity(named: "thing:7")!), "Where is my charger?")
    }

    func testNoticesOrderedGoneLostNew() {
        let notices = Dashboard.notices(in: sample)
        XCTAssertEqual(notices.map(\.kind), [.leftTable, .lostTrack, .unnamed])
        XCTAssertEqual(notices[0].text, "Phone left the table on the left.")
        XCTAssertEqual(notices[1].text, "Lost track of the glasses. It was last seen on the table.")
        // thing:9 might be thing:4, which has no name: don't guess.
        XCTAssertEqual(notices[2].text, "Something new is on the table.")
    }

    func testNoticeGuessesOnlyNamesThePersonKnows() {
        var s = sample
        s.update("thing:9") { $0.m = [MaybeSame(name: "thing:7", score: 0.7)] }
        XCTAssertEqual(Dashboard.notices(in: s).last?.text, "Something new is on the table. It might be my charger.")
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
        for status in [EntityStatus.held, .visible, .gone, .inside] {
            new.update("pill_bottle") { $0.s = status }
            for line in Dashboard.changes(from: sample, to: new).map(\.text) {
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

    func testNewUnnamedObjectAppears() {
        var new = sample
        new.e.append(Entity(n: "thing:12", k: .target, s: .visible, p: nil, xy: TablePoint(x: 5, y: 5), r: nil,
                            c: 1, edge: nil, a: nil, m: nil, ls: nil))
        XCTAssertEqual(Dashboard.changes(from: sample, to: new).map(\.text), ["Something new appeared on the table"])
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
    func testStoreLogsChangesNewestFirstAndDismissesNotices() {
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

        let unnamed = store.notices.first { $0.kind == .unnamed }!
        XCTAssertNil(unnamed.question, "the room can't be asked about a nameless thing; show it instead")
        store.showOnMap(unnamed.entity)
        XCTAssertEqual(store.highlight?.entity, "thing:9")
        XCTAssertEqual(store.highlight?.target, TablePoint(x: 82, y: 10))

        let phone = store.notices.first { $0.entity == "phone" }!
        store.dismiss(phone)
        XCTAssertFalse(store.notices.contains { $0.entity == "phone" })
    }
}
