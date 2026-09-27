import XCTest
@testable import AskTheRoom

/// Replays three state notifications exactly as the phone got them from the live rig at 06:22 on
/// 27 Sep (`rig_states_0627.json`, made with the bridge's own compact_state/cap_state): a tabletop
/// outline, front right, table 49.5 x 48. The real objects are unnamed `thing:N` with a guess; one is
/// on the side table with no position; the eight configured props are unknown with no position.
/// Each payload goes through the app's own path: chunked like the bridge, reassembled by the link
/// meter, decoded with `Wire`, then handed to the store.
@MainActor
final class RigReplayTests: XCTestCase {
    private var store: RoomStore!
    private var meter = LinkMeter()
    private var msgID: UInt8 = 0

    /// The visible things with a table position, and what the room guessed each one is.
    private let onTable = ["thing:50": "remote control", "thing:51": "pill bottle", "thing:52": "glasses",
                           "thing:56": "shipping label", "thing:57": "cardboard box"]
    private let props = ["keys", "pill_bottle", "wallet", "glasses", "phone", "remote", "box", "notebook"]

    override func setUp() async throws {
        store = RoomStore(mock: false, liveTransport: { _ in ReplayTransport() })
        store.linkChanged(.connected)
    }

    // MARK: Payloads

    /// The three payloads as JSON objects, in the order they arrived.
    private func payloads() throws -> [[String: Any]] {
        let url = try XCTUnwrap(Bundle(for: Self.self).url(forResource: "rig_states_0627", withExtension: "json"),
                                "rig_states_0627.json should be in the test bundle")
        let array = try JSONSerialization.jsonObject(with: Data(contentsOf: url)) as? [[String: Any]]
        return try XCTUnwrap(array)
    }

    /// Chunks one payload the way the bridge does (compressed, as after the app's hello, or not),
    /// feeds the chunks to the link meter and decodes what it reassembles, like `RoomLink.received`.
    @discardableResult
    private func deliver(_ payload: [String: Any], compressed: Bool = true) throws -> Snapshot {
        let json = try JSONSerialization.data(withJSONObject: payload)
        msgID &+= 1
        var message: Reassembler.Message?
        for chunk in Framing.chunks(json, msgID: msgID, compressed: compressed) {
            message = meter.receive(chunk, on: .state)
        }
        let data = try XCTUnwrap(message?.data, "the link meter should reassemble the whole state")
        let state = try XCTUnwrap(Wire.decode(Snapshot.self, from: data), "the real state should decode")
        meter.stateDecoded(tx: state.tx, chunks: message?.chunks ?? 0)
        store.receive(state: state)
        return state
    }

    private func entity(_ name: String) throws -> Entity {
        try XCTUnwrap(store.snapshot?.entity(named: name), name)
    }

    private func roomLayout(size: CGSize = CGSize(width: 390, height: 480)) throws -> RoomMapLayout {
        XCTAssertTrue(store.layoutIsCurrent, "the kept layout is the one `lh` names")
        let plan = try XCTUnwrap(RoomPlan(store.layout), "the rig's layout should make a room plan")
        return RoomMapLayout(plan: plan, snapshot: try XCTUnwrap(store.snapshot), size: size)
    }

    // MARK: Decoding

    func testEveryPayloadDecodesAndUpdatesTheStore() throws {
        let all = try payloads()
        XCTAssertEqual(all.count, 3)
        for (i, payload) in all.enumerated() {
            for compressed in [true, false] {
                let state = try deliver(payload, compressed: compressed)
                XCTAssertEqual(store.snapshot?.t, state.t, "payload \(i + 1) is what the store holds")
                XCTAssertEqual(state.e.count, 15)
                XCTAssertEqual(state.lh, "ce3f6baff2")
                XCTAssertEqual(state.lay != nil, i == 0, "only the first carries the layout")
                // The store puts the kept layout on the later ones, since they name it.
                XCTAssertNotNil(store.snapshot?.lay)
            }
        }
        XCTAssertFalse(store.isMapStale, "a live, connected state isn't greyed out")
        XCTAssertEqual(store.layout?.zones?.map(\.id), ["couch", "side_table", "doorway", "counter"])
    }

    func testPropsAreUnknownAndOffTheMap() throws {
        for payload in try payloads() { try deliver(payload) }
        for name in props {
            let e = try entity(name)
            XCTAssertEqual(e.status, .lost, name)
            XCTAssertNil(MapLayout.position(of: e), name)
        }
        let items = MapLayout.items(for: try XCTUnwrap(store.snapshot))
        XCTAssertFalse(items.contains { props.contains($0.id) }, "no position, no pin")
    }

    // MARK: (a) every visible thing is on both maps, with its guess

    func testVisibleThingsAreOnTheTableMapWithTheirGuess() throws {
        for payload in try payloads() {
            let snapshot = try deliver(payload)
            let items = MapLayout.items(for: snapshot)
            for (name, guess) in onTable {
                let item = try XCTUnwrap(items.first { $0.id == name }, "\(name) should be on the table map")
                XCTAssertEqual(item.label, "\(guess)?")
                XCTAssertEqual(item.title, "\(guess)?")
                XCTAssertEqual(item.status, .visible)
                XCTAssertEqual(item.center, try entity(name).r)
            }
            let e50 = try entity("thing:50")
            XCTAssertEqual(e50.displayName, "remote control?")
            XCTAssertFalse(e50.isNameless)
        }
    }

    func testVisibleThingsAreOnTheRoomMapWithTheirGuess() throws {
        for payload in try payloads() {
            try deliver(payload)
            let layout = try roomLayout()
            for (name, guess) in onTable {
                let pin = try XCTUnwrap(layout.pins.first { $0.id == name }, "\(name) should be on the room map")
                XCTAssertTrue(pin.isOnTable, "\(name) is drawn on the room map's table")
                XCTAssertEqual(pin.name, "\(guess)?")
                XCTAssertEqual(pin.style, .visible)
                XCTAssertTrue(pin.accessibilityLabel.lowercased().contains(guess), pin.accessibilityLabel)
                XCTAssertTrue(layout.table.map { $0.insetBy(dx: -1, dy: -1).contains(pin.point) } ?? false,
                              "\(name) sits on the table")
            }
        }
    }

    // MARK: (b) the thing on the side table

    func testThingOnTheSideTableIsInItsZoneAndSaysSo() throws {
        for payload in try payloads() {
            try deliver(payload)
            let e54 = try entity("thing:54")
            XCTAssertNil(e54.drawPoint)
            XCTAssertEqual(e54.zone, "side_table")

            let layout = try roomLayout()
            let pin = try XCTUnwrap(layout.pins.first { $0.id == "thing:54" }, "thing:54 should be on the room map")
            XCTAssertEqual(pin.zone, "side_table")
            XCTAssertEqual(pin.name, "pill bottle?")
            let zone = try XCTUnwrap(layout.zones.first { $0.id == "side_table" })
            XCTAssertTrue(zone.rect.insetBy(dx: -1, dy: -1).contains(pin.point), "the pin is inside the side table")

            let snapshot = try XCTUnwrap(store.snapshot)
            XCTAssertEqual(Dashboard.whereabouts(e54, in: snapshot), "On the side table")
            XCTAssertFalse(MapLayout.items(for: snapshot).contains { $0.id == "thing:54" },
                           "no table position, so not on the table map")
        }
    }

    // MARK: (c) things move when their position changes

    func testMovedThingMovesOnBothMaps() throws {
        var all = try payloads()
        // thing:51 moves in the real data (30.4, 2.2 -> 27.2, 4.2 -> 31.8, 1.3). Also slide the
        // cardboard box 20 cm in the third payload, a move big enough for Recent.
        all[2] = moving("thing:57", to: [8.9, 28.8], in: all[2])

        var tableCenters: [String: [TablePoint]] = [:]
        var tablePlaces: [String: [CGPoint]] = [:]
        var roomPoints: [String: [CGPoint]] = [:]
        for payload in all {
            let snapshot = try deliver(payload)
            let items = MapLayout.items(for: snapshot)
            let geo = MapGeometry(table: snapshot.tableSize, size: CGSize(width: 390, height: 380),
                                  showsYou: snapshot.view != nil)
            let places = MapLayout.placements(for: items, in: geo)
            let pins = try roomLayout().pins
            for name in ["thing:51", "thing:57", "thing:50"] {
                tableCenters[name, default: []].append(try XCTUnwrap(items.first { $0.id == name }?.center))
                tablePlaces[name, default: []].append(try XCTUnwrap(places[name]))
                roomPoints[name, default: []].append(try XCTUnwrap(pins.first { $0.id == name }?.point))
            }
        }

        XCTAssertEqual(tableCenters["thing:51"], [TablePoint(x: 30.4, y: 2.2), TablePoint(x: 27.2, y: 4.2),
                                                  TablePoint(x: 31.8, y: 1.3)])
        let p51 = try XCTUnwrap(tablePlaces["thing:51"])
        XCTAssertNotEqual(p51[0], p51[1], "the table map pin moves")
        XCTAssertNotEqual(p51[1], p51[2], "the table map pin moves back")
        let r51 = try XCTUnwrap(roomPoints["thing:51"])
        XCTAssertNotEqual(r51[0], r51[1], "the room map pin moves")
        XCTAssertNotEqual(r51[1], r51[2], "the room map pin moves back")

        let p57 = try XCTUnwrap(tablePlaces["thing:57"]), r57 = try XCTUnwrap(roomPoints["thing:57"])
        XCTAssertLessThan(p57[2].x, p57[1].x - 30, "20 cm to the left is a clear move on the table map")
        XCTAssertLessThan(r57[2].x, r57[1].x, "and on the room map")

        let p50 = try XCTUnwrap(tablePlaces["thing:50"])
        XCTAssertTrue(p50.allSatisfy { $0 == p50[0] }, "what didn't move stays put")
    }

    /// The store publishes every state, even one that only moves a thing: the maps are drawn from
    /// `store.snapshot` in `body`, so a new snapshot is a redraw.
    func testStoreKeepsEveryPositionChange() throws {
        let all = try payloads()
        try deliver(all[0])
        let first = try XCTUnwrap(store.snapshot)
        try deliver(all[1])
        let second = try XCTUnwrap(store.snapshot)
        XCTAssertNotEqual(first, second)
        XCTAssertEqual(second.entity(named: "thing:51")?.xy, TablePoint(x: 27.2, y: 4.2))
        XCTAssertNotEqual(MapLayout.items(for: first), MapLayout.items(for: second),
                          "the map's animation value changes, so SwiftUI animates the move")
    }

    // MARK: (d) the lists

    func testGuessedThingsAreNotNamelessAndReachRecent() throws {
        var all = try payloads()
        all[2] = moving("thing:57", to: [8.9, 28.8], in: all[2])
        for payload in all { try deliver(payload) }

        for name in Array(onTable.keys) + ["thing:54"] {
            let e = try entity(name)
            XCTAssertFalse(e.isNameless, "\(name) has a guess, so it isn't hidden as nameless")
            XCTAssertTrue(e.isHedged, name)
            XCTAssertTrue(RoomMapLayout.shows(e), name)
        }
        // Small real moves (thing:51 about 4 cm) are jitter; the 20 cm slide is news.
        let recent = Dashboard.recent(activity: store.activity, exchanges: [],
                                      now: try XCTUnwrap(store.snapshot?.time))
        let lines = recent.lastHour.compactMap { entry -> String? in
            if case .change(let e) = entry { return e.text } else { return nil }
        }
        XCTAssertEqual(lines, ["What looks like a cardboard box moved"])
    }

    func testANewGuessedThingAppearsOnRecent() throws {
        var all = try payloads()
        var e = try XCTUnwrap(all[1]["e"] as? [[String: Any]])
        e.append(["n": "thing:60", "k": "t", "s": "V", "xy": [20.0, 20.0], "r": [20.0, 20.0], "c": 1.0,
                  "a": [], "g": "coffee mug", "gc": 0.9])
        all[1]["e"] = e
        try deliver(all[0])
        try deliver(all[1])
        XCTAssertEqual(store.activity.map(\.text), ["What looks like a coffee mug appeared on the table"])
    }

    /// Home's "Your things" with the rig's real data: the 8 fixed demo tiles, then the things in sight with
    /// the room's guesses; no status line on a tile the rig can't see, a short place on one it can.
    func testHomeShowsTheFixedTilesThenGuessesAndNoStatusForUnseen() throws {
        for payload in try payloads() { try deliver(payload) }
        let snapshot = try XCTUnwrap(store.snapshot)
        let home = Dashboard.things(in: snapshot)
        XCTAssertEqual(Array(home.map(\.name).prefix(8)),
                       ["keys", "pill_bottle", "wallet", "glasses", "phone", "remote", "box", "notebook"])
        XCTAssertEqual(Set(home.map(\.name).dropFirst(8)),
                       ["thing:50", "thing:51", "thing:52", "thing:53", "thing:54", "thing:56", "thing:57"])
        for prop in home.prefix(8) { XCTAssertNil(Dashboard.tileLine(prop, in: snapshot), prop.name) }
        let remote = try XCTUnwrap(home.first { $0.name == "thing:50" })
        XCTAssertEqual(remote.displayName, "remote control?")
        XCTAssertEqual(Dashboard.tileLine(remote, in: snapshot), "On the table")
        XCTAssertEqual(Dashboard.tileLine(try XCTUnwrap(home.first { $0.name == "thing:54" }), in: snapshot),
                       "On the side table")
        XCTAssertEqual(Dashboard.question(for: remote), "Where is the remote control?")
        XCTAssertEqual(Dashboard.question(for: home[0]), "Where are my keys?")
    }

    // MARK: Helpers

    private func moving(_ name: String, to xy: [Double], in payload: [String: Any]) -> [String: Any] {
        var payload = payload
        let e = (payload["e"] as? [[String: Any]] ?? []).map { entity -> [String: Any] in
            guard entity["n"] as? String == name else { return entity }
            var moved = entity
            moved["xy"] = xy
            moved["r"] = xy
            return moved
        }
        payload["e"] = e
        return payload
    }
}

@MainActor
private final class ReplayTransport: RoomTransport {
    func send(_ question: Question) {}
    func stop() {}
}
