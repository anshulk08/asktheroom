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

    // MARK: Compression (bit 1)

    /// Python `zlib.compressobj(9, zlib.DEFLATED, -15)` over `fixtureJSON`, as the bridge sends it.
    static let fixtureDeflate = "q1YqU7Iy1FEqAZJ6BkA6MSknVckq2tJAx8wgVkcpPy8nMw8oUFJUmqqjlJNYnFqkZFUNFFaySkvMKU4F6ShKTwVqzyvNyanVUQJpjq0FAA=="
    static let fixtureJSON = #"{"v":1,"t":1.0,"table":[90,60],"online":true,"laser":{"on":false,"target":null},"e":[]}"#

    /// A varied JSON message that doesn't compress to almost nothing.
    private func variedJSON(entities count: Int) -> Data {
        let e = (0..<count).map { #"{"n":"thing:\#($0)","k":"t","s":"V","xy":[\#(Double($0) * 1.37),\#(Double($0 * 7 % 60) + 0.25)],"ls":\#(1_758_000_000 + $0 * 13)}"# }
        return Data(#"{"v":1,"t":1.0,"e":[\#(e.joined(separator: ","))]}"#.utf8)
    }

    func testPythonFixtureInflatesAcrossThreeChunks() throws {
        let deflated = [UInt8](try XCTUnwrap(Data(base64Encoded: Self.fixtureDeflate)))
        let third = deflated.count / 3
        let parts = [deflated[..<third], deflated[third..<2 * third], deflated[(2 * third)...]]
        let flags: [UInt8] = [0x02, 0x02, 0x03]
        var r = Reassembler()
        var out: [Reassembler.Message] = []
        for (i, part) in parts.enumerated() {
            if let m = r.receive(Data([42, UInt8(i), flags[i]] + part)) { out.append(m) }
        }
        XCTAssertEqual(out.count, 1)
        XCTAssertEqual(String(decoding: out[0].data, as: UTF8.self), Self.fixtureJSON)
        XCTAssertTrue(out[0].compressed)
        XCTAssertEqual(out[0].chunks, 3)
        let snap = try XCTUnwrap(Wire.decode(Snapshot.self, from: out[0].data))
        XCTAssertEqual(snap.tableSize, TablePoint(x: 90, y: 60))
        XCTAssertEqual(r.dropped, 0)
    }

    /// Apple's `.zlib` makes the same raw DEFLATE as the bridge's Python.
    func testDeflateMatchesPython() {
        XCTAssertEqual(Framing.deflate(Data(Self.fixtureJSON.utf8)).base64EncodedString(), Self.fixtureDeflate)
    }

    func testCompressedRoundTripFlagsEveryChunk() {
        let message = variedJSON(entities: 200)
        let chunks = Framing.chunks(message, msgID: 12, mtu: 185, compressed: true)
        XCTAssertGreaterThan(chunks.count, 2)
        XCTAssertLessThan(chunks.count, Framing.chunks(message, msgID: 12, mtu: 185).count)
        XCTAssertTrue(chunks.allSatisfy { $0[2] & Framing.compressedFlag != 0 })
        XCTAssertTrue(chunks.dropLast().allSatisfy { $0[2] & Framing.finalFlag == 0 })
        XCTAssertEqual(chunks.last?[2], Framing.compressedFlag | Framing.finalFlag)
        var r = Reassembler()
        XCTAssertEqual(feed(chunks, into: &r), [message])
    }

    func testUncompressedChunksLeaveBitOneClear() {
        let chunks = Framing.chunks(json(bytes: 500), msgID: 1)
        XCTAssertTrue(chunks.allSatisfy { $0[2] & Framing.compressedFlag == 0 })
    }

    func testInflateFailureCountsAsDroppedAndRecovers() {
        var r = Reassembler()
        XCTAssertNil(r.add(Data([3, 0, 0x02]) + Data("not deflate at all".utf8)))
        XCTAssertNil(r.add(Data([3, 1, 0x03]) + Data([0xFF, 0xFF, 0xFF])))
        XCTAssertEqual(r.dropped, 1)
        XCTAssertEqual(r.lastDrop, .inflate)
        let next = json(bytes: 300)
        XCTAssertEqual(feed(Framing.chunks(next, msgID: 4, compressed: true), into: &r), [next])
        XCTAssertEqual(r.dropped, 1)
    }

    func testTruncatedDeflateCountsAsDropped() throws {
        let deflated = try XCTUnwrap(Data(base64Encoded: Self.fixtureDeflate))
        var r = Reassembler()
        XCTAssertNil(r.add(Data([5, 0, 0x03]) + deflated.prefix(deflated.count / 2)))
        XCTAssertEqual(r.dropped, 1)
        XCTAssertEqual(r.lastDrop, .inflate)
    }

    func testInflatedSizeIsCapped() {
        // 300 KB of JSON deflates to well under the wire limit, but mustn't be inflated past the cap.
        let huge = json(bytes: 300 * 1024)
        let chunks = Framing.chunks(huge, msgID: 6, mtu: 517, compressed: true)
        XCTAssertLessThan(chunks.count, 256)
        XCTAssertNil(Framing.inflate(Framing.deflate(huge), limit: Reassembler.maxMessageBytes))
        var r = Reassembler()
        XCTAssertEqual(feed(chunks, into: &r), [])
        XCTAssertEqual(r.dropped, 1)
        let fits = json(bytes: 200 * 1024)
        XCTAssertEqual(feed(Framing.chunks(fits, msgID: 7, mtu: 517, compressed: true), into: &r), [fits])
    }

    func testGapInCompressedMessageDrops() {
        let chunks = Framing.chunks(variedJSON(entities: 200), msgID: 8, compressed: true)
        var r = Reassembler()
        XCTAssertEqual(feed([chunks[0]] + chunks[2...], into: &r), [])
        XCTAssertEqual(r.dropped, 1)
        XCTAssertEqual(r.lastDrop, .gap)
    }

    /// The start of a message was missed: its later chunks count as one lost message, not a crash.
    func testStrayChunksWithNoMessageCountOnceAsAGap() {
        let chunks = Framing.chunks(json(bytes: 800), msgID: 9)
        var r = Reassembler()
        XCTAssertEqual(feed(Array(chunks[1...]), into: &r), [])
        XCTAssertEqual(r.dropped, 1)
        XCTAssertEqual(r.lastDrop, .gap)
        let next = json(bytes: 100)
        XCTAssertEqual(feed(Framing.chunks(next, msgID: 10), into: &r), [next])
        XCTAssertEqual(r.dropped, 1)
    }

    /// A new message's chunk 1 arriving mid-message loses both: the unfinished one and the new one.
    func testNewIDAtChunkOneLosesBothMessages() {
        let a = Framing.chunks(json(bytes: 500), msgID: 1)
        let b = Framing.chunks(json(bytes: 500), msgID: 2)
        var r = Reassembler()
        XCTAssertEqual(feed([a[0], b[1], b[2], a[1]], into: &r), [])
        XCTAssertEqual(r.dropped, 2)
    }
}
