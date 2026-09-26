import XCTest
@testable import AskTheRoom

final class FramingTests: XCTestCase {
    private func json(bytes count: Int) -> Data {
        // Valid JSON of an exact size: {"x":"aaa…"}
        let filler = String(repeating: "a", count: count - 8)
        return Data("{\"x\":\"\(filler)\"}".utf8)
    }

    private func feed(_ chunks: [Data], into r: inout Reassembler) -> [Data] {
        chunks.compactMap { r.add($0) }
    }

    // M2: in-order chunks
    func testInOrderChunksReassemble() {
        let message = json(bytes: 500)
        let chunks = Framing.chunks(message, msgID: 7)
        XCTAssertEqual(chunks.count, 3)
        var r = Reassembler()
        XCTAssertEqual(feed(chunks, into: &r), [message])
    }

    // M2: a new msg_id mid-message
    func testNewMsgIDMidMessageDiscardsUnfinished() {
        let first = Framing.chunks(json(bytes: 500), msgID: 1)
        let second = json(bytes: 300)
        var r = Reassembler()
        XCTAssertNil(r.add(first[0]))
        XCTAssertNil(r.add(first[1]))
        XCTAssertEqual(feed(Framing.chunks(second, msgID: 2), into: &r), [second])
        // The rest of the abandoned message is ignored.
        XCTAssertNil(r.add(first[2]))
        XCTAssertEqual(r.dropped, 1)
    }

    // M2: a missing chunk
    func testGapDropsMessageAndRecovers() {
        let chunks = Framing.chunks(json(bytes: 500), msgID: 3)
        var r = Reassembler()
        XCTAssertEqual(feed([chunks[0], chunks[2]], into: &r), [])
        XCTAssertEqual(r.dropped, 1)
        let next = json(bytes: 40)
        XCTAssertEqual(feed(Framing.chunks(next, msgID: 4), into: &r), [next])
    }

    // M2: a single-chunk message
    func testSingleChunkMessage() {
        let status = Data(#"{"app":"up","fps":12.4,"online":true,"cal":true,"laser_cal":true}"#.utf8)
        let chunks = Framing.chunks(status, msgID: 9)
        XCTAssertEqual(chunks.count, 1)
        XCTAssertEqual(chunks[0][2], Framing.finalFlag)
        var r = Reassembler()
        XCTAssertEqual(r.add(chunks[0]), status)
    }

    // M2: a 3 KB message
    func testThreeKilobyteMessage() throws {
        let message = json(bytes: 3 * 1024)
        let chunks = Framing.chunks(message, msgID: 200)
        XCTAssertEqual(chunks.count, 18)
        XCTAssertTrue(chunks.allSatisfy { $0.count <= Framing.defaultMTU - 3 })
        var r = Reassembler()
        let out = feed(chunks, into: &r)
        XCTAssertEqual(out, [message])
        XCTAssertNoThrow(try JSONSerialization.jsonObject(with: out[0]))
    }

    func testMsgIDWrapsFrom255To0() {
        let a = json(bytes: 300), b = json(bytes: 300)
        var r = Reassembler()
        XCTAssertEqual(feed(Framing.chunks(a, msgID: 255) + Framing.chunks(b, msgID: 0), into: &r), [a, b])
        XCTAssertEqual(r.dropped, 0)
    }

    func testSameIDRestartingAtChunkZeroStartsOver() {
        let first = Framing.chunks(json(bytes: 500), msgID: 5)
        let again = json(bytes: 200)
        var r = Reassembler()
        XCTAssertNil(r.add(first[0]))
        XCTAssertEqual(feed(Framing.chunks(again, msgID: 5), into: &r), [again])
    }

    func testRuntChunkIsDroppedWithoutCrashing() {
        let chunks = Framing.chunks(json(bytes: 500), msgID: 1)
        var r = Reassembler()
        XCTAssertNil(r.add(chunks[0]))
        XCTAssertNil(r.add(Data([1, 1])))
        XCTAssertNil(r.add(Data()))
        XCTAssertEqual(feed(Array(chunks[1...]), into: &r), [])
    }

    func testEmptyMessageStillFrames() {
        let chunks = Framing.chunks(Data(), msgID: 0)
        XCTAssertEqual(chunks, [Data([0, 0, Framing.finalFlag])])
        var r = Reassembler()
        XCTAssertEqual(r.add(chunks[0]), Data())
    }
}
