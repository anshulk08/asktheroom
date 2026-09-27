import XCTest
@testable import AskTheRoom

@MainActor
final class MockRoomTests: XCTestCase {
    private let now = Date(timeIntervalSince1970: 1_790_380_000)
    private lazy var snapshot = MockRoom.startingSnapshot(now: now)

    private func ask(_ q: String) -> Answer {
        MockRoom.answer(to: q, id: 1, in: snapshot, lastChange: "The box moved, with your keys inside.", now: now)
    }

    func testTheDemoQuestion() {
        let a = ask("Where are my keys?")
        XCTAssertEqual(a.text, "Your keys are inside the box.")
        XCTAssertEqual(a.pointAt, "keys")
        XCTAssertEqual(a.laserAction, .point)
        XCTAssertEqual(a.target, TablePoint(x: 70.4, y: 38.1))
    }

    func testEachStatusGetsItsAction() {
        XCTAssertEqual(ask("where is my phone").text, "Your phone left the table on the left.")
        XCTAssertEqual(ask("where is my phone").laserAction, .sweep(.left))
        XCTAssertEqual(ask("where are my glasses").laserAction, .circle)
        XCTAssertEqual(ask("where are my glasses").text, "I lost track of your glasses. I last saw them here.")
        XCTAssertEqual(ask("where's the remote").text, "Someone is holding your remote.")
        XCTAssertEqual(ask("where is my charger").text, "Your charger is on the table.")
        XCTAssertEqual(ask("where is the pill bottle").text, "Your pill bottle is under the notebook.")
    }

    func testGrokNamesAreHedgedAndGuessesDontMatch() {
        XCTAssertEqual(ask("where is the tape roll").text, "What looks like a tape roll is on the table.")
        XCTAssertEqual(ask("where is my phone").pointAt, "phone", "a guessed phone charger isn't the phone")
    }

    func testNamelessContainerTakesNoArticle() {
        snapshot.e.append(Entity(n: "thing:20", k: .container, s: .visible, xy: TablePoint(x: 70, y: 38)))
        snapshot.update("keys") { $0.p = "thing:20" }
        XCTAssertEqual(ask("Where are my keys?").text, "Your keys are inside something new.")
    }

    func testPillWordingStaysNeutral() {
        let a = ask("When did I last pick up my pills?")
        XCTAssertEqual(a.text, "You last picked up your pill bottle 25 minutes ago.")
        XCTAssertEqual(a.pointAt, "pill_bottle")
        XCTAssertFalse(a.text.lowercased().contains("took"))
        XCTAssertFalse(a.text.lowercased().contains("taken"))
    }

    func testWhatChangedHasNoTarget() {
        let a = ask("What changed?")
        XCTAssertEqual(a.text, "Most recently: the box moved, with your keys inside.")
        XCTAssertNil(a.pointAt)
    }

    func testUnknownThing() {
        let a = ask("Where is the elephant?")
        XCTAssertFalse(a.succeeded)
        XCTAssertNil(a.pointAt)
    }

    func testMovingABoxCarriesWhatsInside() {
        var s = snapshot
        s.move("box", to: TablePoint(x: 50, y: 18), now: 5)
        XCTAssertEqual(s.entity(named: "box")?.r, TablePoint(x: 50, y: 18))
        XCTAssertEqual(s.entity(named: "keys")?.r, TablePoint(x: 50, y: 18))
        XCTAssertEqual(s.entity(named: "wallet")?.r, TablePoint(x: 60, y: 15), "things outside stay put")
    }

    func testStartingSnapshotIsTheSampleAtNow() {
        XCTAssertEqual(snapshot.t, now.timeIntervalSince1970)
        XCTAssertEqual(snapshot.entities.count, 16, "the sample and the room's four")
        XCTAssertEqual(snapshot.lay, MockData.sampleLayout)
        XCTAssertNotNil(snapshot.lh)
        XCTAssertEqual(snapshot.entity(named: "keys")?.ls, now.timeIntervalSince1970 - 120)
    }
}
