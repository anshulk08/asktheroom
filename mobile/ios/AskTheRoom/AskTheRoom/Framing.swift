import Compression
import Foundation

/// Chunk framing for BLE notifications (spec section 3).
///
/// Every notification on answer, state and status is a 3-byte header
/// (`msg_id`, `chunk`, `flags`) followed by the next slice of a UTF-8 JSON message.
/// Bit 0 of `flags` marks the final chunk. Bit 1 marks a compressed message: the bridge sets it
/// on every chunk, and the reassembled payload is raw DEFLATE (RFC 1951, no zlib header) of the
/// JSON. The bridge only compresses after this app's hello (`Hello`) says it can inflate.
enum Framing {
    static let headerSize = 3
    static let finalFlag: UInt8 = 0x01
    static let compressedFlag: UInt8 = 0x02
    /// iOS negotiates an ATT MTU of about 185.
    static let defaultMTU = 185

    /// Splits a message into chunks the way the bridge does. Used by mock mode and tests.
    /// `compressed` deflates the message first and sets bit 1 on every chunk, like the bridge.
    static func chunks(_ message: Data, msgID: UInt8, mtu: Int = defaultMTU, compressed: Bool = false) -> [Data] {
        let size = max(1, mtu - 6)
        let bytes = [UInt8](compressed ? deflate(message) : message)
        var slices: [ArraySlice<UInt8>] = stride(from: 0, to: bytes.count, by: size).map {
            bytes[$0..<min($0 + size, bytes.count)]
        }
        if slices.isEmpty { slices = [[]] }
        let base: UInt8 = compressed ? compressedFlag : 0
        return slices.enumerated().map { index, slice in
            let flags = base | (index == slices.count - 1 ? finalFlag : 0)
            return Data([msgID, UInt8(truncatingIfNeeded: index), flags] + slice)
        }
    }

    /// Raw DEFLATE, byte for byte what Python's `zlib.compressobj(9, zlib.DEFLATED, -15)` makes
    /// for small messages. Apple's `.zlib` algorithm is raw DEFLATE despite the name.
    static func deflate(_ data: Data) -> Data {
        guard !data.isEmpty, let out = try? (data as NSData).compressed(using: .zlib) else { return Data() }
        return out as Data
    }

    /// Inflates raw DEFLATE, or nil if it's malformed, truncated or would come to more than
    /// `limit` bytes. Streams, so a hostile payload can't balloon in memory first.
    static func inflate(_ data: Data, limit: Int) -> Data? {
        guard !data.isEmpty else { return nil }
        let stream = UnsafeMutablePointer<compression_stream>.allocate(capacity: 1)
        defer { stream.deallocate() }
        guard compression_stream_init(stream, COMPRESSION_STREAM_DECODE, COMPRESSION_ZLIB) == COMPRESSION_STATUS_OK else {
            return nil
        }
        defer { compression_stream_destroy(stream) }
        let window = 16 * 1024
        let buffer = UnsafeMutablePointer<UInt8>.allocate(capacity: window)
        defer { buffer.deallocate() }

        return data.withUnsafeBytes { (src: UnsafeRawBufferPointer) -> Data? in
            guard let base = src.bindMemory(to: UInt8.self).baseAddress else { return nil }
            stream.pointee.src_ptr = base
            stream.pointee.src_size = src.count
            var out = Data()
            while true {
                let before = stream.pointee.src_size
                stream.pointee.dst_ptr = buffer
                stream.pointee.dst_size = window
                let status = compression_stream_process(stream, Int32(COMPRESSION_STREAM_FINALIZE.rawValue))
                let produced = window - stream.pointee.dst_size
                out.append(buffer, count: produced)
                if out.count > limit { return nil }
                switch status {
                case COMPRESSION_STATUS_END:
                    return out
                case COMPRESSION_STATUS_OK:
                    // No progress on either side: the input ended before the stream did.
                    if produced == 0, stream.pointee.src_size == before { return nil }
                default:
                    return nil
                }
            }
        }
    }
}

/// Reassembles one characteristic's chunks into whole messages.
///
/// Keep one per characteristic. A chunk with a new `msg_id` discards any unfinished
/// message; a gap in chunk numbers discards the message. The next state snapshot
/// replaces a dropped one within a few seconds, so dropping is always safe.
struct Reassembler {
    /// Guards against a runaway peer, on the wire and after inflating. State is 1–30 KB.
    static let maxMessageBytes = 256 * 1024

    enum DropReason: Equatable {
        /// A missing chunk, or a chunk with no message in progress (its start was missed).
        case gap
        /// A new message began before this one finished.
        case newID
        /// Too big, on the wire or inflated, or more than 256 chunks.
        case overflow
        /// Shorter than the header.
        case runt
        /// Marked compressed but not valid DEFLATE.
        case inflate
    }

    /// One whole message.
    struct Message: Equatable {
        /// The UTF-8 JSON, inflated if it came compressed.
        var data: Data
        /// How many notifications it took.
        var chunks: Int
        var compressed: Bool
    }

    private var msgID: UInt8?
    private var nextChunk = 0
    private var buffer = Data()
    private var compressed = false
    /// Messages recently given up on, so their remaining chunks aren't counted again.
    private var lost: [UInt8] = []

    /// Messages dropped since creation, for diagnostics.
    private(set) var dropped = 0
    private(set) var lastDrop: DropReason?

    /// Feeds one notification value. Returns the whole message on its final chunk.
    mutating func add(_ value: Data) -> Data? {
        receive(value)?.data
    }

    /// Feeds one notification value. Returns the whole message, with how it came, on its final chunk.
    mutating func receive(_ value: Data) -> Message? {
        let bytes = [UInt8](value)
        guard bytes.count >= Framing.headerSize else {
            discard(.runt)
            return nil
        }
        let id = bytes[0]
        let chunk = Int(bytes[1])
        let flags = bytes[2]
        let isFinal = flags & Framing.finalFlag != 0
        let payload = bytes[Framing.headerSize...]

        if chunk == 0 {
            // Always a fresh start, even if the same id was mid-message.
            if msgID != nil { discard(.newID) }
            msgID = id
            nextChunk = 0
            buffer = Data()
            compressed = false
            lost.removeAll { $0 == id }
        } else if id != msgID || chunk != nextChunk {
            // A new id mid-message, a gap, or a stray chunk whose start never came:
            // none of these can be completed.
            if msgID != nil { discard(id == msgID ? .gap : .newID) }
            if !lost.contains(id) {
                // A message we never saw the start of: lost too, counted once.
                note(.gap)
                remember(id)
            }
            return nil
        }

        buffer.append(contentsOf: payload)
        compressed = compressed || flags & Framing.compressedFlag != 0
        nextChunk = chunk + 1
        if buffer.count > Self.maxMessageBytes || (!isFinal && nextChunk > 255) {
            discard(.overflow)
            return nil
        }
        guard isFinal else { return nil }

        var message = Message(data: buffer, chunks: nextChunk, compressed: compressed)
        reset()
        if message.compressed {
            guard let inflated = Framing.inflate(message.data, limit: Self.maxMessageBytes) else {
                remember(id)
                note(.inflate)
                return nil
            }
            message.data = inflated
        }
        return message
    }

    private mutating func discard(_ reason: DropReason) {
        if let msgID {
            remember(msgID)
            note(reason)
        }
        reset()
    }

    private mutating func remember(_ id: UInt8) {
        lost.removeAll { $0 == id }
        lost.append(id)
        if lost.count > 4 { lost.removeFirst() }
    }

    private mutating func note(_ reason: DropReason) {
        dropped += 1
        lastDrop = reason
    }

    private mutating func reset() {
        msgID = nil
        nextChunk = 0
        buffer = Data()
        compressed = false
    }
}
