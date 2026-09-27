import XCTest
@testable import AskTheRoom

final class LinkStatsTests: XCTestCase {
    private let t0 = Date(timeIntervalSince1970: 1_000_000)

    private func sample(_ tx: Int, _ rx: Int, at seconds: TimeInterval) -> LossTracker.Sample {
        LossTracker.Sample(tx: tx, rx: rx, at: t0.addingTimeInterval(seconds))
    }

    func testLossIsSentMinusReceived() {
        XCTAssertEqual(LossTracker.loss(from: sample(100, 40, at: 0), to: sample(300, 220, at: 10)),
                       LinkStats.Loss(lost: 20, sent: 200))
        XCTAssertEqual(LossTracker.loss(from: sample(0, 0, at: 0), to: sample(50, 50, at: 1)),
                       LinkStats.Loss(lost: 0, sent: 50))
        // More received than sent (a count the bridge doesn't see) is never negative loss.
        XCTAssertEqual(LossTracker.loss(from: sample(0, 0, at: 0), to: sample(10, 12, at: 1))?.lost, 0)
        // The bridge restarted: its count went backwards.
        XCTAssertNil(LossTracker.loss(from: sample(500, 40, at: 0), to: sample(20, 60, at: 1)))
        XCTAssertEqual(LinkStats.Loss(lost: 20, sent: 200).percent, 10)
        XCTAssertEqual(LinkStats.Loss(lost: 0, sent: 0).percent, 0)
    }

    func testTrackerCoversTheLastMinute() {
        var tracker = LossTracker()
        XCTAssertNil(tracker.current)
        tracker.add(sample(0, 0, at: 0))
        XCTAssertNil(tracker.current, "one sample says nothing")
        tracker.add(sample(100, 90, at: 30))
        XCTAssertEqual(tracker.current, LinkStats.Loss(lost: 10, sent: 100))
        tracker.add(sample(200, 190, at: 61))
        tracker.add(sample(300, 290, at: 100))
        // The window starts at 40 s; the span runs from the last sample at or before it (30 s).
        XCTAssertEqual(tracker.samples.first?.tx, 100)
        XCTAssertEqual(tracker.current, LinkStats.Loss(lost: 0, sent: 200))
    }

    func testTrackerStartsOverWhenTheBridgeRestarts() {
        var tracker = LossTracker()
        tracker.add(sample(1000, 900, at: 0))
        tracker.add(sample(1100, 1000, at: 1))
        tracker.add(sample(5, 1010, at: 2))
        XCTAssertNil(tracker.current)
        tracker.add(sample(25, 1030, at: 3))
        XCTAssertEqual(tracker.current, LinkStats.Loss(lost: 0, sent: 20))
    }

    /// Two compressed state messages through the meter, with one chunk of a third lost in between.
    func testMeterCountsChunksMessagesAndLoss() throws {
        var meter = LinkMeter()
        meter.connected(mtu: 185, at: t0)
        var tx = 0
        func send(_ id: UInt8, skip: Int? = nil) throws -> Reassembler.Message? {
            var state = MockData.sampleSnapshot
            state.tx = tx
            let chunks = Framing.chunks(try JSONEncoder().encode(state), msgID: id, compressed: true)
            tx += chunks.count
            var out: Reassembler.Message?
            for (i, c) in chunks.enumerated() where i != skip {
                if let m = meter.receive(c, on: .state) { out = m }
            }
            if let out { meter.stateDecoded(tx: Wire.decode(Snapshot.self, from: out.data)?.tx, chunks: out.chunks, at: t0) }
            return out
        }
        let first = try XCTUnwrap(try send(1))
        XCTAssertTrue(first.compressed)
        XCTAssertNil(try send(2, skip: 1))
        XCTAssertNotNil(try send(3))

        let stats = meter.stats
        XCTAssertEqual(stats.state.messages, 2)
        XCTAssertEqual(stats.state.compressed, 2)
        XCTAssertEqual(stats.state.dropped, 1)
        XCTAssertEqual(stats.state.chunks, tx - 1)
        XCTAssertEqual(stats.loss, LinkStats.Loss(lost: 1, sent: tx - first.chunks))
        XCTAssertEqual(stats.mtu, 185)
        XCTAssertEqual(stats.lastStateAt, t0)
    }

    func testReconnectsAndLastDisconnect() {
        var meter = LinkMeter()
        meter.disconnected(reason: "failed attempt", at: t0)
        XCTAssertNil(meter.stats.lastDisconnect, "a failed attempt isn't a disconnect")
        meter.connected(mtu: 517, at: t0)
        XCTAssertEqual(meter.stats.reconnects, 0)
        _ = meter.receive(Data([0, 0, 1]) + Data("{}".utf8), on: .status)
        meter.disconnected(reason: "watchdog: nothing for 15 s", at: t0.addingTimeInterval(100))
        XCTAssertEqual(meter.stats.lastConnectionLasted, 100)
        meter.connected(mtu: 517, at: t0.addingTimeInterval(101))
        XCTAssertEqual(meter.stats.reconnects, 1)
        XCTAssertEqual(meter.stats.status, LinkStats.Channel(), "counts start again")
        XCTAssertEqual(meter.stats.reconnectsLine, "Reconnects: 1 (last: watchdog: nothing for 15 s)")
    }

    func testPlainWords() {
        var stats = LinkStats()
        stats.connectedAt = t0
        stats.mtu = 517
        stats.state = LinkStats.Channel(chunks: 812, bytes: 400_000, messages: 31, dropped: 0, compressed: 0)
        stats.loss = LinkStats.Loss(lost: 0, sent: 812)
        stats.lastStateAt = t0.addingTimeInterval(125.6)
        stats.reconnects = 3
        let now = t0.addingTimeInterval(126)
        XCTAssertEqual(stats.connectionLine(now: now), "Connected 2 min · MTU 517")
        XCTAssertEqual(stats.channelLine(.state), "State: 812 chunks, 31 messages, 0 dropped")
        XCTAssertEqual(stats.lossLine, "Lost 0 of 812 chunks (0%)")
        XCTAssertEqual(stats.lastUpdateLine(now: now), "Last update 0.4 s ago")
        XCTAssertEqual(stats.reconnectsLine, "Reconnects: 3")
        stats.state.compressed = 31
        XCTAssertEqual(stats.channelLine(.state), "State: 812 chunks, 31 messages (31 compressed), 0 dropped")
        stats.loss = LinkStats.Loss(lost: 3, sent: 80)
        XCTAssertEqual(stats.lossLine, "Lost 3 of 80 chunks (3.8%)")
        stats.connectedAt = nil
        stats.lastConnectionLasted = 95
        XCTAssertEqual(stats.connectionLine(now: now), "Not connected, lasted 1 min")
    }
}
