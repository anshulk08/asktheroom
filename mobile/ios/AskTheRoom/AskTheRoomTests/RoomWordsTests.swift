import XCTest
@testable import AskTheRoom

/// Things off the table, in a room zone (entity `z`), say where they are in words everywhere.
final class RoomWordsTests: XCTestCase {
    private func thing(_ name: String = "wallet", _ status: EntityStatus = .visible, zone: String? = "couch",
                       rg: RegistryState? = nil, parent: String? = nil, c: Double? = nil) -> Entity {
        Entity(n: name, k: .target, s: status, p: parent, c: c, z: zone, rg: rg)
    }

    private func snapshot(_ entities: [Entity], layout: RoomLayout? = MockData.sampleLayout) -> Snapshot {
        Snapshot(e: entities, lay: layout)
    }

    private func words(_ e: Entity, layout: RoomLayout? = MockData.sampleLayout) -> String {
        Dashboard.whereabouts(e, in: snapshot([e], layout: layout))
    }

    func testTheLayoutsOwnWords() {
        var layout = MockData.sampleLayout
        layout.zones?.append(RoomLayout.Zone(id: "kitchen", say: "the kitchen counter", rect: [0, 0, 10, 10], kind: "surface"))
        XCTAssertEqual(words(thing(zone: "kitchen"), layout: layout), "On the kitchen counter")
        XCTAssertEqual(words(thing(zone: "couch")), "On the couch")
        XCTAssertEqual(words(thing(zone: "side_table")), "On the TV stand")
        XCTAssertEqual(words(thing(zone: "doorway")), "By the doorway")
    }

    func testWithoutALayoutTheIdIsTidied() {
        XCTAssertEqual(words(thing(zone: "side_table"), layout: nil), "On the side table")
        XCTAssertEqual(words(thing(zone: "counter"), layout: nil), "On the counter")
        XCTAssertEqual(words(thing(zone: "hall_shelf")), "On the hall shelf", "an id the layout doesn't have")
        XCTAssertEqual(words(thing(zone: "room")), "Somewhere in the room")
        XCTAssertEqual(words(thing(zone: "room"), layout: nil), "Somewhere in the room")
    }

    func testTheTableIsNotAZone() {
        XCTAssertEqual(words(thing(zone: "table")), "On the table")
        XCTAssertEqual(words(thing(zone: nil)), "On the table")
    }

    func testRegistryStates() {
        XCTAssertEqual(words(thing(zone: "couch", rg: .lastSeen)), "Last seen on the couch")
        XCTAssertEqual(words(thing("wallet", .lost, zone: "couch")), "Last seen on the couch")
        XCTAssertEqual(words(thing(zone: "room", rg: .unknown)), "Last seen somewhere in the room")
        XCTAssertEqual(words(thing(zone: "counter", rg: .hidden)), "Hidden on the counter")
        XCTAssertEqual(words(thing("wallet", .held, zone: "counter", rg: .carried)), "Carried, last seen on the counter")
        XCTAssertEqual(words(thing(zone: "couch", c: 0.5)), "Probably on the couch")

        let box = Entity(n: "box", k: .container, s: .visible, z: "couch")
        let keys = thing("keys", .inside, zone: "couch", parent: "box")
        XCTAssertEqual(Dashboard.whereabouts(keys, in: snapshot([box, keys])), "Inside the box, on the couch")
    }

    func testTheDetailSheetSaysTheSame() {
        let e = thing(zone: "couch", rg: .lastSeen)
        XCTAssertEqual(EntityDetailView.statusWords(e, in: snapshot([e])), "Last seen on the couch")
    }

    /// Room things have no table position, and still show on Home.
    func testRoomThingsStayInTheLists() {
        let snap = MockData.roomSnapshot
        let names = Dashboard.things(in: snap).map(\.name)
        for name in ["headphones", "mug", "book", "umbrella"] {
            XCTAssertTrue(names.contains(name), name)
        }
        XCTAssertEqual(Dashboard.whereabouts(snap.entity(named: "headphones")!, in: snap), "On the couch")
        XCTAssertEqual(Dashboard.whereabouts(snap.entity(named: "umbrella")!, in: snap), "Last seen by the doorway")
        XCTAssertNotNil(Dashboard.lastSeen(Entity(n: "u", k: .target, s: .visible, ls: 0, z: "couch", rg: .lastSeen),
                                           now: Date()))
    }

    func testNoticesSayTheRoomNotTheTable() {
        let lost = Entity(n: "glasses", k: .target, s: .lost, z: "couch")
        let gone = Entity(n: "phone", k: .target, s: .gone, z: "counter")
        let notices = Dashboard.notices(in: snapshot([lost, gone]))
        XCTAssertEqual(notices.map(\.kind), [.lostTrack], "the room knows where the phone went")
        XCTAssertEqual(notices.first?.seenWords, "They were last seen on the couch")
    }

    func testRecentSaysWhereThingsWent() {
        let now = Date(timeIntervalSince1970: 1_790_380_000)
        let onTable = snapshot([Entity(n: "wallet", k: .target, s: .visible, xy: TablePoint(x: 1, y: 1))])
        let onCouch = snapshot([thing()])
        XCTAssertEqual(Dashboard.changes(from: onTable, to: onCouch, now: now).map(\.text), ["Wallet is on the couch now"])
        XCTAssertEqual(Dashboard.changes(from: onCouch, to: onTable, now: now).map(\.text), ["Wallet came back to the table"])
        let keys = snapshot([thing("keys", zone: "room")])
        XCTAssertEqual(Dashboard.changes(from: onCouch, to: keys, now: now).map(\.text), ["Keys appeared somewhere in the room"])
    }
}
