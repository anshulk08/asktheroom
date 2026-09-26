import Foundation

/// Chunk framing for BLE notifications (spec section 3).
///
/// Every notification on answer, state and status is a 3-byte header
/// (`msg_id`, `chunk`, `flags`) followed by the next slice of a UTF-8 JSON message.
/// Bit 0 of `flags` marks the final chunk.
enum Framing {
    static let headerSize = 3
    static let finalFlag: UInt8 = 0x01
    /// iOS negotiates an ATT MTU of about 185.
    static let defaultMTU = 185

    /// Splits a message into chunks the way the bridge does. Used by mock mode and tests.
    static func chunks(_ message: Data, msgID: UInt8, mtu: Int = defaultMTU) -> [Data] {
        let size = max(1, mtu - 6)
        let bytes = [UInt8](message)
        var slices: [ArraySlice<UInt8>] = stride(from: 0, to: bytes.count, by: size).map {
            bytes[$0..<min($0 + size, bytes.count)]
        }
        if slices.isEmpty { slices = [[]] }
        return slices.enumerated().map { index, slice in
            let flags: UInt8 = index == slices.count - 1 ? finalFlag : 0
            return Data([msgID, UInt8(truncatingIfNeeded: index), flags] + slice)
        }
    }
}

/// Reassembles one characteristic's chunks into whole messages.
///
/// Keep one per characteristic. A chunk with a new `msg_id` discards any unfinished
/// message; a gap in chunk numbers discards the message. The next state snapshot
/// replaces a dropped one within 0.5 s, so dropping is always safe.
struct Reassembler {
    /// Guards against a runaway peer. Real messages are 1–3 KB.
    static let maxMessageBytes = 64 * 1024

    private var msgID: UInt8?
    private var nextChunk = 0
    private var buffer = Data()

    /// Number of messages dropped since creation, for logging.
    private(set) var dropped = 0

    /// Feeds one notification value. Returns the whole message on its final chunk.
    mutating func add(_ value: Data) -> Data? {
        let bytes = [UInt8](value)
        guard bytes.count >= Framing.headerSize else {
            discard()
            return nil
        }
        let id = bytes[0]
        let chunk = Int(bytes[1])
        let isFinal = bytes[2] & Framing.finalFlag != 0
        let payload = bytes[Framing.headerSize...]

        if chunk == 0 {
            // Always a fresh start, even if the same id was mid-message.
            if msgID != nil { discard() }
            msgID = id
            nextChunk = 0
            buffer = Data()
        } else if id != msgID || chunk != nextChunk {
            // A new id mid-message, or a gap: this message can't be completed.
            discard()
            return nil
        }

        buffer.append(contentsOf: payload)
        nextChunk = chunk + 1
        if buffer.count > Self.maxMessageBytes || (!isFinal && nextChunk > 255) {
            discard()
            return nil
        }
        guard isFinal else { return nil }

        let message = buffer
        reset()
        return message
    }

    private mutating func discard() {
        if msgID != nil { dropped += 1 }
        reset()
    }

    private mutating func reset() {
        msgID = nil
        nextChunk = 0
        buffer = Data()
    }
}
