import Foundation

/// The three characteristics that notify the phone.
enum LinkChannel: CaseIterable {
    case answer
    case state
    case status
}

/// How the Bluetooth link is doing, for the Connection section of helper settings. Counts are
/// since the current connection began; reconnects and the last disconnect are since launch.
struct LinkStats: Equatable {
    struct Channel: Equatable {
        var chunks = 0
        var bytes = 0
        var messages = 0
        var dropped = 0
        var compressed = 0
    }

    /// State chunks the bridge sent over a span, and how many of them never arrived.
    struct Loss: Equatable {
        var lost: Int
        var sent: Int

        var percent: Double { sent > 0 ? Double(lost) * 100 / Double(sent) : 0 }
    }

    var connectedAt: Date?
    var mtu: Int?
    var reconnects = 0
    var lastDisconnect: String?
    /// How long the connection before this one lasted.
    var lastConnectionLasted: TimeInterval?
    var answer = Channel()
    var state = Channel()
    var status = Channel()
    var lastStateAt: Date?
    /// Over the last minute, or since connecting; nil until two state messages carried `tx`.
    var loss: Loss?

    subscript(channel: LinkChannel) -> Channel {
        get {
            switch channel {
            case .answer: return answer
            case .state: return state
            case .status: return status
            }
        }
        set {
            switch channel {
            case .answer: answer = newValue
            case .state: state = newValue
            case .status: status = newValue
            }
        }
    }
}

/// Measures link loss from state messages: the bridge's `tx` (state chunks it had sent before
/// the message) against the chunks the phone had received before the same message.
struct LossTracker {
    struct Sample: Equatable {
        var tx: Int
        var rx: Int
        var at: Date
    }

    static let window: TimeInterval = 60

    private(set) var samples: [Sample] = []

    /// Loss between two samples: chunks sent minus chunks received over the same span. Nil if
    /// the bridge's count went backwards (it restarted), so the two can't be compared.
    static func loss(from a: Sample, to b: Sample) -> LinkStats.Loss? {
        let sent = b.tx - a.tx
        let received = b.rx - a.rx
        guard sent >= 0, received >= 0 else { return nil }
        return LinkStats.Loss(lost: max(0, sent - received), sent: sent)
    }

    mutating func add(_ sample: Sample) {
        if let last = samples.last, sample.tx < last.tx || sample.rx < last.rx {
            samples = []
        }
        samples.append(sample)
        // Keep one sample at or before the window's start, so the span covers a whole minute.
        let start = sample.at.addingTimeInterval(-Self.window)
        while samples.count > 2, samples[1].at <= start { samples.removeFirst() }
    }

    mutating func reset() { samples = [] }

    /// Over the last minute, or since the first sample if that's more recent.
    var current: LinkStats.Loss? {
        guard samples.count >= 2, let first = samples.first, let last = samples.last else { return nil }
        return Self.loss(from: first, to: last)
    }
}

/// Reassembles and counts everything that arrives on the link. Owned by the transport, which
/// hands `stats` to the store from time to time.
struct LinkMeter {
    private(set) var stats = LinkStats()
    private var reassemblers: [LinkChannel: Reassembler] = [:]
    private var loss = LossTracker()

    /// A fresh connection: counts start again, reconnects and the last disconnect stay.
    mutating func connected(mtu: Int?, at now: Date = Date()) {
        if stats.connectedAt != nil || stats.lastDisconnect != nil { stats.reconnects += 1 }
        var fresh = LinkStats()
        fresh.connectedAt = now
        fresh.mtu = mtu
        fresh.reconnects = stats.reconnects
        fresh.lastDisconnect = stats.lastDisconnect
        fresh.lastConnectionLasted = stats.lastConnectionLasted
        stats = fresh
        reassemblers = [:]
        loss.reset()
    }

    /// Only a connection that was up counts; a failed attempt isn't a disconnect.
    mutating func disconnected(reason: String, at now: Date = Date()) {
        guard let since = stats.connectedAt else { return }
        stats.lastConnectionLasted = now.timeIntervalSince(since)
        stats.lastDisconnect = reason
        stats.connectedAt = nil
        reassemblers = [:]
        loss.reset()
    }

    /// Unfinished messages can't be completed, e.g. after the rig's service changed.
    mutating func resetReassembly() {
        reassemblers = [:]
    }

    /// Feeds one notification. Returns the whole message on its final chunk.
    mutating func receive(_ value: Data, on channel: LinkChannel) -> Reassembler.Message? {
        stats[channel].chunks += 1
        stats[channel].bytes += value.count
        var r = reassemblers[channel] ?? Reassembler()
        let droppedBefore = r.dropped
        let message = r.receive(value)
        stats[channel].dropped += r.dropped - droppedBefore
        reassemblers[channel] = r
        if let message {
            stats[channel].messages += 1
            if message.compressed { stats[channel].compressed += 1 }
        }
        return message
    }

    /// A message that reassembled but wasn't valid JSON for its characteristic.
    mutating func malformed(on channel: LinkChannel) {
        stats[channel].messages -= 1
        stats[channel].dropped += 1
    }

    /// A state message decoded. `tx` is the bridge's count of state chunks sent before it.
    mutating func stateDecoded(tx: Int?, chunks: Int, at now: Date = Date()) {
        stats.lastStateAt = now
        guard let tx else { return }
        loss.add(LossTracker.Sample(tx: tx, rx: stats.state.chunks - chunks, at: now))
        stats.loss = loss.current
    }
}

// MARK: In plain words (helper settings, "Connection")

extension LinkStats {
    /// "Connected 2 min · MTU 517", or how the last connection ended.
    func connectionLine(now: Date = Date()) -> String {
        guard let connectedAt else {
            let lasted = lastConnectionLasted.map { ", lasted \(Self.span($0))" } ?? ""
            return "Not connected" + lasted
        }
        let mtu = mtu.map { " · MTU \($0)" } ?? ""
        return "Connected \(Self.span(now.timeIntervalSince(connectedAt)))" + mtu
    }

    /// "State: 812 chunks, 31 messages, 0 dropped", with how many came compressed if any did.
    func channelLine(_ channel: LinkChannel) -> String {
        let c = self[channel]
        let name: String
        switch channel {
        case .answer: name = "Answers"
        case .state: name = "State"
        case .status: name = "Status"
        }
        let compressed = c.compressed > 0 ? " (\(c.compressed) compressed)" : ""
        return "\(name): \(Self.count(c.chunks, "chunk")), \(Self.count(c.messages, "message"))\(compressed), \(c.dropped) dropped"
    }

    /// "Lost 0 of 812 chunks (0%)", over the last minute or since connecting.
    var lossLine: String {
        guard let loss else { return "Lost: not measured yet" }
        let percent = loss.percent.formatted(.number.precision(.fractionLength(0...1)))
        return "Lost \(loss.lost) of \(Self.count(loss.sent, "chunk")) (\(percent)%)"
    }

    /// "Last update 0.4 s ago".
    func lastUpdateLine(now: Date = Date()) -> String {
        guard let lastStateAt else { return "No update yet" }
        let ago = max(0, now.timeIntervalSince(lastStateAt))
        if ago < 10 {
            return "Last update \(ago.formatted(.number.precision(.fractionLength(1)))) s ago"
        }
        return "Last update \(Self.span(ago)) ago"
    }

    /// "Reconnects: 3 (last: watchdog: nothing for 15 s)".
    var reconnectsLine: String {
        let last = lastDisconnect.map { " (last: \($0))" } ?? ""
        return "Reconnects: \(reconnects)\(last)"
    }

    static func span(_ seconds: TimeInterval) -> String {
        let s = Int(max(0, seconds))
        switch s {
        case ..<60: return "\(s) s"
        case ..<3600: return "\(s / 60) min"
        default: return "\(s / 3600) h \(s % 3600 / 60) min"
        }
    }

    private static func count(_ n: Int, _ noun: String) -> String {
        "\(n) \(noun)\(n == 1 ? "" : "s")"
    }
}
