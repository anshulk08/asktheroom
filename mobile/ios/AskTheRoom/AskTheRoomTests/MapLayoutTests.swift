import XCTest
@testable import AskTheRoom

final class MapLayoutTests: XCTestCase {
    private let sample = MockData.sampleSnapshot

    private func item(_ name: String, in items: [MapItem]) -> MapItem {
        items.first { $0.id == name }!
    }

    func testEveryStatusInTheSampleGetsItsLabel() {
        let items = MapLayout.items(for: sample)
        XCTAssertEqual(items.count, 10)
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
        let points = MapLayout.placements(for: items, in: geo).points
        XCTAssertNotEqual(points["keys"]!.y, points["wallet"]!.y)
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

    /// On an iPhone-width map the sample's chips and labels don't cover one another.
    func testSampleChipsDontOverlapOnAnIPhone() {
        let items = MapLayout.items(for: sample)
        let width: CGFloat = 394
        let size = CGSize(width: width, height: width / MapGeometry.aspectRatio(for: sample.tableSize, width: width))
        let placement = MapLayout.placements(for: items, in: MapGeometry(table: sample.tableSize, size: size))
        let chips = items.filter { $0.shape == .chip }
        let rects = chips.map { chip in
            MapLayout.rect(at: placement.points[chip.id]!,
                           size: MapLayout.footprint(of: chip, withCaption: !placement.hiddenCaptions.contains(chip.id)))
        }
        for i in rects.indices {
            for j in rects.indices where j > i {
                // Footprints are estimates, so edges touching by a few points is fine.
                let overlap = rects[i].intersection(rects[j])
                XCTAssertTrue(overlap.isNull || min(overlap.width, overlap.height) <= 4,
                              "\(chips[i].id) overlaps \(chips[j].id) by \(overlap.size)")
            }
        }
        XCTAssertFalse(placement.hiddenCaptions.contains("keys"), "the headline caption stays")
    }
}
