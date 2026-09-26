import XCTest
@testable import AskTheRoom

final class MapLayoutTests: XCTestCase {
    private let sample = MockData.sampleSnapshot

    private func item(_ name: String, in items: [MapItem]) -> MapItem {
        items.first { $0.id == name }!
    }

    func testEveryStatusInTheSampleGetsItsLabel() {
        let items = MapLayout.items(for: sample)
        XCTAssertEqual(items.count, 12)
        XCTAssertEqual(item("keys", in: items).caption, "inside box")
        XCTAssertEqual(item("pill_bottle", in: items).caption, "under notebook")
        XCTAssertEqual(item("phone", in: items).caption, "left table ←")
        XCTAssertEqual(item("phone", in: items).exitEdge, .left)
        XCTAssertEqual(item("glasses", in: items).caption, "lost track, last seen here")
        XCTAssertEqual(item("remote", in: items).caption, "in a hand")
        XCTAssertEqual(item("remote", in: items).glyph, "hand.raised.fill")
        XCTAssertNil(item("wallet", in: items).caption)
        XCTAssertEqual(item("thing:7", in: items).title, "my charger")
        XCTAssertEqual(item("thing:9", in: items).caption, "probably might be unnamed object 4")
        XCTAssertTrue(item("thing:9", in: items).linkBadge)
    }

    func testLineStyleAndOpacityCarryStatus() {
        let items = MapLayout.items(for: sample)
        for hidden in ["keys", "pill_bottle", "remote"] {
            XCTAssertTrue(item(hidden, in: items).dashed, hidden)
        }
        XCTAssertFalse(item("wallet", in: items).dashed)
        XCTAssertEqual(item("thing:9", in: items).opacity, MapLayout.uncertainOpacity)
        XCTAssertEqual(item("glasses", in: items).opacity, MapLayout.lostOpacity)
        XCTAssertEqual(item("notebook", in: items).opacity, MapLayout.coveringOpacity, "cover drawn over what's under it")
        XCTAssertEqual(item("box", in: items).opacity, 1)
    }

    func testDrawingOrder() {
        let items = MapLayout.items(for: sample)
        let order = items.map(\.id)
        func index(_ n: String) -> Int { order.firstIndex(of: n)! }
        XCTAssertLessThan(index("pill_bottle"), index("notebook"), "under goes below its cover")
        XCTAssertLessThan(index("box"), index("keys"), "containers before what's inside")
        XCTAssertLessThan(index("notebook"), index("wallet"))
        XCTAssertEqual(items.map(\.layer), items.map(\.layer).sorted())
    }

    func testPositionsFollowStatus() {
        let items = MapLayout.items(for: sample)
        XCTAssertEqual(item("keys", in: items).center, TablePoint(x: 70.4, y: 38.1), "inside: at r, the parent")
        XCTAssertEqual(item("phone", in: items).center, TablePoint(x: 3, y: 30), "gone: at xy")
        XCTAssertEqual(item("glasses", in: items).center, TablePoint(x: 80, y: 50))
    }

    func testEntityWithoutPositionIsSkipped() {
        var s = sample
        s.e.append(Entity(n: "mug", k: .target, s: .lost))
        XCTAssertNil(MapLayout.items(for: s).first { $0.id == "mug" })
    }

    func testSiblingsInsideOneParentFanOut() {
        var s = sample
        s.update("wallet") { $0.s = .inside; $0.p = "box"; $0.r = TablePoint(x: 70.4, y: 38.1) }
        let items = MapLayout.items(for: s)
        XCTAssertEqual(item("keys", in: items).siblingCount, 2)
        let geo = MapGeometry(table: s.tableSize, size: CGSize(width: 390, height: 280))
        let points = MapLayout.placements(for: items, in: geo)
        XCTAssertGreaterThanOrEqual(abs(points["keys"]!.x - points["wallet"]!.x), 40, "side by side, not stacked")
    }

    func testGeometryIsUniformAndCentred() {
        let geo = MapGeometry(table: TablePoint(x: 90, y: 60), size: CGSize(width: 390, height: 400))
        let rect = geo.tableRect
        XCTAssertEqual(rect.width / rect.height, 1.5, accuracy: 0.001)
        XCTAssertEqual(rect.midX, 195, accuracy: 0.001)
        XCTAssertEqual(rect.midY, 200, accuracy: 0.001)
        XCTAssertEqual(geo.point(TablePoint(x: 0, y: 0)), rect.origin)
        XCTAssertEqual(geo.point(TablePoint(x: 90, y: 60)).x, rect.maxX, accuracy: 0.001)
        XCTAssertEqual(geo.exitPoint(from: CGPoint(x: 50, y: 99), through: .left).y, 99)
        XCTAssertLessThan(geo.exitPoint(from: CGPoint(x: 50, y: 99), through: .left).x, rect.minX)
    }

    private func iPhoneGeometry(width: CGFloat = 394) -> MapGeometry {
        let size = CGSize(width: width, height: width / MapGeometry.aspectRatio(for: sample.tableSize, width: width))
        return MapGeometry(table: sample.tableSize, size: size)
    }

    /// On an iPhone-width map the sample's pins and names don't cover one another.
    func testSamplePinsDontOverlapOnAnIPhone() {
        let items = MapLayout.items(for: sample)
        let points = MapLayout.placements(for: items, in: iPhoneGeometry())
        let chips = items.filter { $0.shape == .pin }
        let rects = chips.map { MapLayout.rect(of: $0, at: points[$0.id]!) }
        for i in rects.indices {
            for j in rects.indices where j > i {
                // Footprints are estimates, so edges touching by a few points is fine.
                let overlap = rects[i].intersection(rects[j])
                XCTAssertTrue(overlap.isNull || min(overlap.width, overlap.height) <= 4,
                              "\(chips[i].id) overlaps \(chips[j].id) by \(overlap.size)")
            }
        }
    }

    /// The box's name sits inside its top edge; the keys' pin goes below it, still inside the box.
    func testInsidePinSitsUnderItsParentsName() {
        let items = MapLayout.items(for: sample)
        let geo = iPhoneGeometry()
        let keys = MapLayout.placements(for: items, in: geo)["keys"]!
        let box = item("box", in: items)
        guard case .block(let w, let h) = box.shape else { return XCTFail("box is a block") }
        let c = geo.point(box.center)
        let body = CGRect(x: c.x - geo.length(w) / 2, y: c.y - geo.length(h) / 2, width: geo.length(w), height: geo.length(h))
        let chip = MapLayout.rect(of: item("keys", in: items), at: keys)
        XCTAssertGreaterThanOrEqual(chip.minY, body.minY + MapLayout.blockTitleHeight - 1, "clear of the name")
        XCTAssertTrue(body.contains(CGPoint(x: keys.x, y: keys.y)), "still inside the box")
    }

    /// Unnamed things print a short label; the full name stays for VoiceOver and the card.
    func testUnnamedThingsGetAShortLabel() {
        let items = MapLayout.items(for: sample)
        XCTAssertEqual(item("thing:9", in: items).label, "unnamed")
        XCTAssertEqual(item("thing:9", in: items).title, "unnamed object 9")
        XCTAssertEqual(item("thing:7", in: items).label, "my charger", "named things keep their name")
        XCTAssertTrue(item("thing:9", in: items).accessibilityLabel.hasPrefix("unnamed object 9"))
    }

    /// The room's guesses print with a question mark, on the map and for VoiceOver.
    func testGuessesAreHedgedOnTheMap() {
        let items = MapLayout.items(for: sample)
        XCTAssertEqual(item("thing:11", in: items).label, "phone charger?")
        XCTAssertEqual(item("thing:11", in: items).title, "phone charger?")
        XCTAssertEqual(item("thing:12", in: items).label, "tape roll?")
        XCTAssertTrue(item("thing:12", in: items).accessibilityLabel.hasPrefix("tape roll?"))
    }

    /// The key only lists the marks the map is using.
    func testLegendListsOnlyMarksInUse() {
        XCTAssertEqual(MapLayout.legend(for: MapLayout.items(for: sample)), [.held, .hidden, .left, .lost, .unsure])
        var calm = sample
        calm.e = calm.e.filter { ["wallet", "box", "notebook"].contains($0.n) }
        XCTAssertEqual(MapLayout.legend(for: MapLayout.items(for: calm)), [])
    }
}
