import XCTest
@testable import AskTheRoom

/// Live sightings (state `sg`): "I see glasses on the couch" puts the glasses on the room map.
final class SightingsTests: XCTestCase {
    private let plan = RoomPlan(MockData.sampleLayout)!
    private let phone = CGSize(width: 370, height: 480)
    private let t = 1790500620.4

    private func decode(_ json: String) -> Snapshot? {
        Wire.decode(Snapshot.self, from: Data(json.utf8))
    }

    private func snapshot(_ entities: [Entity], _ rows: [Sighting]) -> Snapshot {
        var s = Snapshot(e: entities, lay: MockData.sampleLayout)
        s.sg = SightingList(rows)
        return s
    }

    private func glassesSeen(on zone: String = "couch") -> Sighting {
        Sighting(name: "glasses", zone: zone, t: t, source: "look")
    }

    private let lostGlasses = Entity(n: "glasses", k: .target, s: .lost, rg: .unknown)

    // MARK: Decoding

    func testDecodesGoodRows() throws {
        let s = try XCTUnwrap(decode("""
        {"e":[],"sg":[["glasses","couch",1790500620.4,"look"],["thing:12","counter",1790500500,"recall"]]}
        """))
        XCTAssertEqual(s.sightings, [
            Sighting(name: "glasses", zone: "couch", t: 1790500620.4, source: "look"),
            Sighting(name: "thing:12", zone: "counter", t: 1790500500, source: "recall"),
        ])
        XCTAssertEqual(s.sightings[0].time, Date(timeIntervalSince1970: 1790500620.4))
    }

    func testSkipsMalformedRows() throws {
        let s = try XCTUnwrap(decode("""
        {"e":[],"sg":[["glasses"],null,42,"x",{"n":"a"},[1,"couch",3,"look"],["","couch",3,"look"],
                      ["keys","counter","soon","look"],["mug","doorway",1790500000,"look"]]}
        """))
        XCTAssertEqual(s.sightings.map(\.name), ["mug"])
    }

    func testABadSgNeverFailsTheSnapshot() throws {
        for sg in ["\"oops\"", "5", "{\"a\":1}", "null", "[]"] {
            let s = try XCTUnwrap(decode("{\"e\":[{\"n\":\"keys\",\"k\":\"t\",\"s\":\"V\"}],\"sg\":\(sg)}"), sg)
            XCTAssertEqual(s.entities.count, 1, sg)
            XCTAssertEqual(s.sightings, [], sg)
        }
    }

    func testAbsentSgIsNoSightings() throws {
        let s = try XCTUnwrap(decode(MockData.sampleSnapshotJSON))
        XCTAssertNil(s.sg)
        XCTAssertEqual(s.sightings, [])
    }

    func testRoundTripsForTheSavedMap() throws {
        let s = snapshot([], [glassesSeen()])
        let back = try XCTUnwrap(Wire.decode(Snapshot.self, from: try JSONEncoder().encode(s)))
        XCTAssertEqual(back.sightings, [glassesSeen()])
    }

    // MARK: Map

    func testSightingPinGoesInItsZone() throws {
        let l = RoomMapLayout(plan: plan, snapshot: snapshot([lostGlasses], [glassesSeen()]), size: phone)
        let pin = try XCTUnwrap(l.pins.first { $0.id == RoomMapLayout.sightingID("glasses") })
        XCTAssertEqual(pin.zone, "couch")
        XCTAssertEqual(pin.style, .sighted)
        XCTAssertEqual(pin.opacity, 1)
        XCTAssertEqual(pin.selects, "glasses")
        let couch = try XCTUnwrap(l.zones.first { $0.id == "couch" })
        XCTAssertTrue(couch.rect.contains(pin.point))
        let when = Banners.when(Date(timeIntervalSince1970: t))
        XCTAssertEqual(pin.name, "glasses · seen \(when)")
        XCTAssertEqual(pin.accessibilityLabel, "Glasses, seen on the couch at \(when)")
        XCTAssertEqual(RoomMapLayout.legend(for: l.pins).last, .sighted)
        XCTAssertEqual(RoomMapLayout.PinStyle.sighted.words, "Seen by the camera")
    }

    func testSightingOfAnUntrackedNameUsesTheName() throws {
        let s = snapshot([], [Sighting(name: "tv_remote", zone: "counter", t: t, source: "recall")])
        let l = RoomMapLayout(plan: plan, snapshot: s, size: phone)
        let pin = try XCTUnwrap(l.pins.first { $0.id == RoomMapLayout.sightingID("tv_remote") })
        XCTAssertEqual(pin.zone, "counter")
        XCTAssertTrue(pin.name.hasPrefix("tv remote · seen "))
    }

    func testNoSightingPinOnceAPlacedPinExists() {
        let onCouch = Entity(n: "glasses", k: .target, s: .visible, z: "couch", rg: .visible)
        let onTable = Entity(n: "glasses", k: .target, s: .visible, xy: TablePoint(x: 10, y: 10))
        for e in [onCouch, onTable] {
            let s = snapshot([e], [glassesSeen(on: "counter")])
            let l = RoomMapLayout(plan: plan, snapshot: s, size: phone)
            XCTAssertFalse(l.pins.contains { $0.style == .sighted })
            XCTAssertTrue(l.pins.contains { $0.id == "glasses" })
            XCTAssertNil(s.sighting(for: "glasses"))
        }
    }

    func testOnlyTheNewestSightingPerName() {
        let older = Sighting(name: "glasses", zone: "counter", t: t - 60, source: "recall")
        let l = RoomMapLayout(plan: plan, snapshot: snapshot([lostGlasses], [glassesSeen(), older]), size: phone)
        XCTAssertEqual(l.pins.filter { $0.style == .sighted }.map(\.zone), ["couch"])
    }

    func testUnknownZoneGoesElsewhere() throws {
        let s = snapshot([lostGlasses], [glassesSeen(on: "garage")])
        XCTAssertTrue(RoomMapLayout.needsElsewhere(plan, snapshot: s))
        let l = RoomMapLayout(plan: plan, snapshot: s, size: phone)
        XCTAssertNotNil(l.elsewhere)
        let pin = try XCTUnwrap(l.pins.first { $0.style == .sighted })
        XCTAssertEqual(pin.zone, RoomMapLayout.elsewhereID)
    }

    // MARK: Words

    func testWordsForAnUnplacedThing() {
        let s = snapshot([lostGlasses], [glassesSeen()])
        let when = Banners.when(Date(timeIntervalSince1970: t))
        XCTAssertEqual(Dashboard.whereabouts(lostGlasses, in: s), "Seen on the couch at \(when)")
        XCTAssertEqual(EntityDetailView.statusWords(lostGlasses, in: s), "Seen on the couch at \(when)")
        let door = snapshot([lostGlasses], [glassesSeen(on: "doorway")])
        XCTAssertEqual(Dashboard.whereabouts(lostGlasses, in: door), "Seen by the doorway at \(when)")
    }

    func testPlacedThingKeepsItsOwnWords() {
        let onCouch = Entity(n: "glasses", k: .target, s: .visible, z: "couch", rg: .visible)
        XCTAssertEqual(Dashboard.whereabouts(onCouch, in: snapshot([onCouch], [glassesSeen(on: "counter")])), "On the couch")
        XCTAssertEqual(Dashboard.whereabouts(lostGlasses, in: snapshot([lostGlasses], [])),
                       Dashboard.whereabouts(lostGlasses, in: Snapshot(e: [lostGlasses], lay: MockData.sampleLayout)))
    }

    // MARK: Demo

    @MainActor func testMockRoomShowsGlassesSeenOnTheCouch() {
        let s = MockRoom.startingSnapshot()
        XCTAssertEqual(s.sighting(for: "glasses")?.zone, "couch")
        let l = RoomMapLayout(plan: plan, snapshot: s, size: phone)
        XCTAssertTrue(l.pins.contains { $0.style == .sighted && $0.zone == "couch" && $0.selects == "glasses" })
    }
}
