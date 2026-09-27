import XCTest
@testable import AskTheRoom

final class RoomMapLayoutTests: XCTestCase {
    private let plan = RoomPlan(MockData.sampleLayout)!
    private let phone = CGSize(width: 370, height: 480)

    private func layout(_ snapshot: Snapshot = MockData.roomSnapshot, size: CGSize? = nil, plan: RoomPlan? = nil) -> RoomMapLayout {
        RoomMapLayout(plan: plan ?? self.plan, snapshot: snapshot, size: size ?? phone)
    }

    private func pin(_ name: String, in l: RoomMapLayout) -> RoomMapLayout.Pin {
        l.pins.first { $0.id == name }!
    }

    private func zone(_ id: String, in l: RoomMapLayout) -> RoomMapLayout.ZoneBox {
        l.zones.first { $0.id == id }!
    }

    private func roomThing(_ name: String, zone: String, rg: RegistryState? = nil, _ s: EntityStatus = .visible) -> Entity {
        Entity(n: name, k: .target, s: s, z: zone, rg: rg)
    }

    // MARK: Plan

    func testPlanReadsTheLayout() {
        XCTAssertEqual(plan.size, CGSize(width: 420, height: 520))
        XCTAssertEqual(plan.table, CGRect(x: 66.9, y: 110, width: 73.1, height: 100.8))
        XCTAssertEqual(plan.zones.map(\.id), ["side_table", "couch", "counter", "doorway"])
        XCTAssertEqual(plan.zone("side_table")?.label, "TV stand", "the say, without its 'the'")
        XCTAssertEqual(plan.zone("couch")?.kind, .seat)
        XCTAssertEqual(plan.zone("doorway")?.kind, .door)
        XCTAssertEqual(plan.zone("counter")?.kind, .surface)
        XCTAssertEqual(plan.you, CGPoint(x: 103.4, y: 255.8))
        XCTAssertEqual(RoomPlan.label(say: nil, id: "side_table"), "side table")
        XCTAssertEqual(RoomPlan.label(say: "The kitchen counter", id: "k"), "kitchen counter")
    }

    func testMissingOrOddLayoutGivesNoRoomMap() {
        XCTAssertNil(RoomPlan(nil))
        XCTAssertNil(RoomPlan(RoomLayout()))
        XCTAssertNil(RoomPlan(RoomLayout(v: 1, size: [400, 300])), "nothing to draw")
        XCTAssertNil(RoomPlan(RoomLayout(size: [0, 300], table: .init(rect: [1, 1, 10, 10]))))
        XCTAssertNil(RoomPlan(RoomLayout(size: [400], table: .init(rect: [1, 1, 10, 10]))))
        XCTAssertNil(RoomPlan(RoomLayout(size: [400, 300], table: .init(rect: [1, 1, -10, 10]),
                                         zones: [.init(id: "couch", rect: [1, 2, 3])])), "only bad rects")
        let noID = RoomPlan(RoomLayout(size: [400, 300], zones: [.init(say: "the couch", rect: [0, 0, 50, 50]),
                                                                  .init(id: "counter", rect: [0, 0, 50, 50])]))
        XCTAssertEqual(noID?.zones.map(\.id), ["counter"], "a zone without an id is left out, the rest draw")
        XCTAssertNil(noID?.table)

        // A tolerant decode of junk gives no plan either.
        let junk = try? JSONDecoder().decode(RoomLayout.self, from: Data(#"{"size":"big","zones":7}"#.utf8))
        XCTAssertNil(RoomPlan(junk))
    }

    /// The store offers a plan only for the layout the latest state names.
    @MainActor
    func testOnlyACurrentLayoutMakesARoomMap() {
        let store = RoomStore()
        store.receive(state: MockData.sampleSnapshot)
        XCTAssertFalse(store.layoutIsCurrent)
        store.receive(state: MockData.roomSnapshot)
        XCTAssertTrue(store.layoutIsCurrent)
        XCTAssertNotNil(RoomPlan(store.layout))
    }

    // MARK: Scale

    func testScalesToFitWithAMarginAndKeepsTheAspect() {
        let l = layout()
        let aspect = l.bounds.width / l.bounds.height
        XCTAssertEqual(aspect, 420.0 / 520.0, accuracy: 0.001)
        XCTAssertGreaterThanOrEqual(l.bounds.minX, RoomMapLayout.margin - 0.001)
        XCTAssertGreaterThanOrEqual(l.bounds.minY, RoomMapLayout.margin - 0.001)
        XCTAssertLessThanOrEqual(l.bounds.maxX, phone.width - RoomMapLayout.margin + 0.001)
        XCTAssertLessThanOrEqual(l.bounds.maxY, phone.height - RoomMapLayout.margin + 0.001)
        XCTAssertEqual(l.bounds.midX, phone.width / 2, accuracy: 0.001, "centred")

        // A wide space: height limits, and the room is centred across.
        let wide = layout(size: CGSize(width: 800, height: 300))
        XCTAssertEqual(wide.bounds.height, 300 - 2 * RoomMapLayout.margin, accuracy: 0.001)
        XCTAssertEqual(wide.bounds.midX, 400, accuracy: 0.001)

        // The view's aspect ratio fits the room exactly across the width.
        let ratio = RoomMapLayout.aspectRatio(for: plan, snapshot: MockData.roomSnapshot, width: 370)
        let fitted = layout(size: CGSize(width: 370, height: 370 / ratio))
        XCTAssertEqual(fitted.bounds.width, 370 - 2 * RoomMapLayout.margin, accuracy: 0.01)
        XCTAssertEqual(fitted.table, fitted.rect(plan.table!))
    }

    // MARK: Pins in zones

    func testAPinSitsInsideItsZone() {
        let l = layout()
        for (name, id) in [("headphones", "couch"), ("mug", "counter"), ("book", "side_table"), ("umbrella", "doorway")] {
            let p = pin(name, in: l)
            XCTAssertEqual(p.zone, id)
            XCTAssertTrue(zone(id, in: l).rect.contains(p.dotRect), name)
            XCTAssertFalse(p.frame.intersects(zone(id, in: l).labelRect), "\(name) clear of the zone's label")
            XCTAssertEqual(p.diameter, RoomMapLayout.pinSize)
        }
    }

    private func assertFannedOut(_ pins: [RoomMapLayout.Pin], in zone: RoomMapLayout.ZoneBox, you: CGRect?,
                                 file: StaticString = #filePath, line: UInt = #line) {
        for p in pins {
            XCTAssertTrue(zone.rect.contains(p.frame), "\(p.id) inside \(zone.id)", file: file, line: line)
            XCTAssertFalse(p.frame.intersects(zone.labelRect), "\(p.id) clear of the label", file: file, line: line)
            if let you { XCTAssertFalse(p.frame.intersects(you), "\(p.id) clear of You", file: file, line: line) }
        }
        for (i, a) in pins.enumerated() {
            for b in pins[(i + 1)...] {
                XCTAssertFalse(a.frame.intersects(b.frame), "\(a.id) / \(b.id)", file: file, line: line)
            }
        }
    }

    func testFanOutHasNoOverlapsAndStaysInside() {
        // A big seat with the person in the middle of it.
        let big = RoomPlan(RoomLayout(size: [400, 300], zones: [.init(id: "rug", say: "the rug", rect: [20, 20, 360, 200], kind: "seat")],
                                      you: [200, 110]))!
        let names = ["keys", "wallet", "glasses", "charger", "book", "pen", "tissues", "cushion", "remote", "mug"]
        let l = RoomMapLayout(plan: big, snapshot: Snapshot(e: names.map { roomThing($0, zone: "rug") }), size: phone)
        let pins = l.pins.filter { $0.zone == "rug" }
        XCTAssertEqual(pins.count, names.count, "all fit")
        XCTAssertTrue(l.overflows.isEmpty)
        assertFannedOut(pins, in: zone("rug", in: l), you: l.you)
        XCTAssertGreaterThan(Set(pins.map(\.point.x)).count, 1, "a grid, not one column")

        // The real couch, with "You" on it, takes two things side by side with its label.
        var s = MockData.roomSnapshot
        s.update("wallet") { $0.xy = nil; $0.r = nil; $0.z = "couch" }
        let room = layout(s)
        let couch = room.pins.filter { $0.zone == "couch" }
        XCTAssertEqual(Set(couch.map(\.id)), ["headphones", "wallet"])
        assertFannedOut(couch, in: zone("couch", in: room), you: room.you)
    }

    func testOverflowShowsPlusN() {
        var s = MockData.roomSnapshot
        let extra = (1...30).map { roomThing("thing_\($0)", zone: "counter") }
        s.e += extra
        let l = layout(s)
        let counter = zone("counter", in: l)
        let cells = l.cells(for: counter)
        let shown = l.pins.filter { $0.zone == "counter" }
        let more = l.overflows.first { $0.id == "counter" }
        XCTAssertEqual(shown.count, cells.count - 1, "the last cell says +N")
        XCTAssertEqual(more?.count, 31 - shown.count, "30 extras and the mug")
        XCTAssertTrue(counter.rect.contains(more!.rect))
        XCTAssertFalse(shown.contains { $0.frame.intersects(more!.rect) })

        XCTAssertTrue(layout().overflows.isEmpty, "no +N when everything fits")
    }

    func testThingsInNoDrawnZoneGoElsewhere() {
        XCTAssertNil(layout().elsewhere, "no strip when nothing needs it")
        var s = MockData.roomSnapshot
        s.e += [roomThing("scarf", zone: "room"), roomThing("hat", zone: "hall_shelf")]
        let l = layout(s)
        let strip = try! XCTUnwrap(l.elsewhere)
        XCTAssertGreaterThan(strip.rect.minY, l.bounds.maxY, "along the bottom, under the room")
        XCTAssertLessThanOrEqual(strip.rect.maxY, phone.height)
        for name in ["scarf", "hat"] {
            XCTAssertEqual(pin(name, in: l).zone, RoomMapLayout.elsewhereID)
            XCTAssertTrue(strip.rect.contains(pin(name, in: l).dotRect))
        }
        XCTAssertEqual(pin("scarf", in: l).accessibilityLabel, "Scarf, somewhere in the room")
    }

    // MARK: Table

    func testTableObjectsSitAtOriginPlusTheirPosition() {
        let l = layout()
        let wallet = pin("wallet", in: l)
        XCTAssertNil(wallet.zone)
        XCTAssertEqual(wallet.diameter, RoomMapLayout.tablePinSize, "smaller on the table")
        let expected = l.point(CGPoint(x: 66.9 + 60, y: 110 + 15))
        XCTAssertEqual(wallet.point.x, expected.x, accuracy: 0.01)
        XCTAssertEqual(wallet.point.y, expected.y, accuracy: 0.01)

        // Beyond the table's edge (glasses at x 80 on a 73 cm table): clamped inside it.
        let glasses = pin("glasses", in: l)
        XCTAssertTrue(l.table!.contains(glasses.dotRect))

        // Things at one place (the keys inside the box) don't sit on top of each other.
        XCTAssertFalse(pin("keys", in: l).dotRect.intersects(pin("box", in: l).dotRect))
        let table = l.pins.filter(\.isOnTable)
        for p in table { XCTAssertTrue(l.table!.contains(p.dotRect), p.id) }
        // Names that are shown never overlap one another.
        let labels = table.compactMap(\.labelRect)
        for (i, a) in labels.enumerated() { for b in labels[(i + 1)...] { XCTAssertFalse(a.intersects(b)) } }
    }

    func testNamelessThingsAreLeftOff() {
        let l = layout()
        XCTAssertFalse(l.pins.contains { $0.id == "thing:9" }, "something new, with only a weak guess")
        XCTAssertTrue(l.pins.contains { $0.id == "thing:7" }, "my charger")
    }

    // MARK: Style

    func testStyleFromStatusAndRegistry() {
        func style(_ s: EntityStatus, _ rg: RegistryState? = nil) -> RoomMapLayout.PinStyle {
            RoomMapLayout.style(for: Entity(n: "x", k: .target, s: s, rg: rg))
        }
        XCTAssertEqual(style(.visible), .visible)
        XCTAssertEqual(style(.inside), .hidden)
        XCTAssertEqual(style(.under), .hidden)
        XCTAssertEqual(style(.held), .carried)
        XCTAssertEqual(style(.lost), .ghost)
        XCTAssertEqual(style(.gone), .ghost)
        XCTAssertEqual(style(.unrecognized), .ghost)
        XCTAssertEqual(style(.visible, .hidden), .hidden)
        XCTAssertEqual(style(.visible, .carried), .carried)
        XCTAssertEqual(style(.visible, .lastSeen), .ghost)
        XCTAssertEqual(style(.visible, .unknown), .ghost)
        XCTAssertEqual(style(.lost, .visible), .visible)
        XCTAssertEqual(style(.held, .unrecognized), .carried, "an unknown registry word falls back to the status")

        let l = layout()
        XCTAssertEqual(pin("headphones", in: l).style, .visible)
        XCTAssertEqual(pin("headphones", in: l).opacity, 1)
        XCTAssertEqual(pin("mug", in: l).style, .carried)
        XCTAssertEqual(pin("book", in: l).style, .hidden)
        XCTAssertEqual(pin("umbrella", in: l).style, .ghost)
        XCTAssertEqual(pin("umbrella", in: l).opacity, RoomMapLayout.ghostOpacity)
        XCTAssertEqual(pin("glasses", in: l).style, .ghost, "lost on the table")
        XCTAssertEqual(RoomMapLayout.legend(for: l.pins), [.visible, .hidden, .carried, .ghost])
    }

    func testAccessibilityLabels() {
        let l = layout()
        XCTAssertEqual(pin("headphones", in: l).accessibilityLabel, "Headphones, on the couch")
        XCTAssertEqual(pin("umbrella", in: l).accessibilityLabel, "Umbrella, last seen by the doorway")
        XCTAssertEqual(pin("mug", in: l).accessibilityLabel, "Mug, carried, last seen on the counter")
        XCTAssertEqual(pin("wallet", in: l).accessibilityLabel, "Wallet, on the table")
        XCTAssertEqual(pin("keys", in: l).accessibilityLabel, "Keys, inside the box")
    }

    func testYouMarkerAndLabelsKeepClear() {
        let l = layout()
        let you = try! XCTUnwrap(l.you)
        let expected = l.point(CGPoint(x: 103.4, y: 255.8))
        XCTAssertEqual(you.midX, expected.x, accuracy: 0.01)
        XCTAssertEqual(you.midY, expected.y, accuracy: 0.01)
        let map = CGRect(origin: .zero, size: l.size)
        for z in l.zones {
            XCTAssertFalse(z.labelRect.intersects(you), "\(z.id)'s label clear of You")
            // Just outside the zone, above it (or below), inside the map: the zone is all pins.
            XCTAssertTrue(abs(z.labelRect.maxY - (z.rect.minY - 1)) < 0.5 || abs(z.labelRect.minY - (z.rect.maxY + 1)) < 0.5, z.id)
            XCTAssertFalse(z.labelRect.intersects(z.rect.insetBy(dx: 0, dy: 0.5)), z.id)
            XCTAssertTrue(map.contains(z.labelRect), z.id)
            for p in l.pins {
                XCTAssertFalse(p.dotRect.intersects(z.labelRect), "\(p.id) under \(z.id)'s label")
            }
        }
    }

    func testOnlyNamedThingsAreLabelledOnTheTableInset() {
        let named = Entity(n: "keys", k: .target, s: .visible)
        var taught = Entity(n: "thing:3", k: .target, s: .visible)
        taught.a = ["charger"]
        var guessed = Entity(n: "thing:4", k: .target, s: .visible)
        guessed.a = []
        guessed.g = "phone charger"
        XCTAssertTrue(RoomMapLayout.namedOnTable(named))
        XCTAssertTrue(RoomMapLayout.namedOnTable(taught))
        XCTAssertFalse(RoomMapLayout.namedOnTable(guessed))
    }

    func testASmallZoneWithYouStillShowsAPinBesideItsCount() {
        let l = layout(size: CGSize(width: 300, height: 340))     // the compact tile on an iPhone
        let couch = zone("couch", in: l)
        let pinsInCouch = l.pins.filter { couch.rect.contains(CGPoint(x: $0.dotRect.midX, y: $0.dotRect.midY)) }
        if let over = l.overflows.first(where: { $0.id == "couch" }) {
            XCTAssertFalse(pinsInCouch.isEmpty, "the couch shows a pin, not only +\(over.count)")
            for p in pinsInCouch { XCTAssertFalse(p.frame.intersects(over.rect), p.id) }
        }
    }
}
