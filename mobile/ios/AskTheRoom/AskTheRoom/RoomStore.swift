import Foundation
import Observation

/// Something that can carry a question to the rig: the Bluetooth link or the mock room.
@MainActor
protocol RoomTransport: AnyObject {
    func send(_ question: Question)
    func stop()
}

/// One question and, once it arrives, its answer.
struct Exchange: Identifiable, Equatable {
    let id: Int
    var question: String
    var answer: Answer?
    var timedOut = false
    /// Past `RoomStore.slowAfter`: still waiting, and the card says so.
    var slow = false
    var askedAt = Date()

    var isPending: Bool { answer == nil && !timedOut }
}

/// What the map should light up for an answer: a pulse at `target`, plus a sweep or circle.
struct Highlight: Identifiable, Equatable {
    let id = UUID()
    var entity: String?
    var target: TablePoint?
    var action: LaserAction?
}

enum LinkState: Equatable {
    case searching
    case connecting
    case connected
    case reconnecting
    case bluetoothOff
    case unauthorized
    case unsupported
}

@MainActor
@Observable
final class RoomStore {
    static let historyLimit = 10
    static let voiceAnswersKey = "showRoomVoiceAnswers"
    static let rigNoticeLimit = 3

    var link: LinkState = .searching
    private(set) var isMock = false
    /// True once the rig has connected this launch, so drops show "Reconnecting…" rather than the Connect screen.
    private(set) var hasConnected = false

    private(set) var snapshot: Snapshot?
    private(set) var status: RigStatus?

    /// Newest first, at most `historyLimit`.
    private(set) var exchanges: [Exchange] = []
    private(set) var highlight: Highlight?
    /// Latest answer to a question someone else asked the rig (PROTOCOL.md 6a).
    private(set) var heardInRoom: Answer?
    /// Reminders and morning reports the rig fired while connected, newest first.
    private(set) var rigNotices: [Notice] = []
    /// What changed since the app connected, newest first (Home, "Recently").
    private(set) var activity: [ActivityEvent] = []
    /// Notices the person has put away; each comes back if its situation changes.
    private(set) var dismissedNotices: Set<String> = []

    /// Off by default: the rig already speaks these, and the phone may be in another room.
    var showRoomVoiceAnswers = UserDefaults.standard.bool(forKey: RoomStore.voiceAnswersKey) {
        didSet { UserDefaults.standard.set(showRoomVoiceAnswers, forKey: Self.voiceAnswersKey) }
    }
    /// Outermost of the ask timeouts (server 10 s < bridge 12 s < this, PROTOCOL.md section 6), so the
    /// rig's own "took too long" reaches the phone first. Retrying sooner would only queue the same
    /// question behind the one still being answered.
    var answerTimeout: Duration = .seconds(15)
    /// Open questions answered by Grok can take several seconds.
    var slowAfter: Duration = .seconds(5)
    var highlightDuration: Duration = .seconds(5)

    private var transport: RoomTransport?
    private var makeLiveTransport: ((RoomStore) -> RoomTransport)?
    private var nextQuestionID = 1
    private var timeoutTask: Task<Void, Never>?
    private var highlightTask: Task<Void, Never>?

    var current: Exchange? { exchanges.first }
    var notices: [Notice] {
        (rigNotices + (snapshot.map(Dashboard.notices(in:)) ?? [])).filter { !dismissedNotices.contains($0.id) }
    }
    var history: ArraySlice<Exchange> { exchanges.dropFirst() }

    var isRoomAppDown: Bool { status.map { !$0.appIsUp } ?? false }
    /// Cloud voice and extras unavailable; answers still work (handoff decision 2).
    var isOffline: Bool { status?.online == false || snapshot?.online == false }

    init(mock: Bool = false, liveTransport: ((RoomStore) -> RoomTransport)? = nil) {
        makeLiveTransport = liveTransport
        setMock(mock)
    }

    // MARK: Source

    /// Switches between the mock room and the real rig (helper settings).
    func setMock(_ on: Bool) {
        transport?.stop()
        transport = nil
        isMock = on
        snapshot = nil
        status = nil
        highlight = nil
        heardInRoom = nil
        rigNotices = []
        exchanges = []
        activity = []
        dismissedNotices = []
        timeoutTask?.cancel()
        if on {
            link = .connected
            transport = MockRoom(store: self)
        } else {
            link = .searching
            hasConnected = false
            transport = makeLiveTransport?(self)
        }
    }

    // MARK: Asking

    func ask(_ text: String) {
        let q = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !q.isEmpty else { return }

        let id = nextQuestionID
        nextQuestionID = nextQuestionID == Int(Int32.max) ? 1 : nextQuestionID + 1

        exchanges.insert(Exchange(id: id, question: q), at: 0)
        if exchanges.count > Self.historyLimit { exchanges.removeLast(exchanges.count - Self.historyLimit) }
        setHighlight(nil)
        transport?.send(Question(id: id, q: q))

        timeoutTask?.cancel()
        timeoutTask = Task { [weak self, answerTimeout, slowAfter] in
            try? await Task.sleep(for: slowAfter)
            guard !Task.isCancelled else { return }
            self?.markSlow(id)
            try? await Task.sleep(for: answerTimeout - slowAfter)
            guard !Task.isCancelled else { return }
            self?.timeOut(id)
        }
    }

    func retry() {
        guard let last = current, last.timedOut else { return }
        ask(last.question)
    }

    private func markSlow(_ id: Int) {
        guard exchanges.first?.id == id, exchanges[0].answer == nil else { return }
        exchanges[0].slow = true
    }

    private func timeOut(_ id: Int) {
        guard exchanges.first?.id == id, exchanges[0].answer == nil else { return }
        exchanges[0].timedOut = true
    }

    // MARK: Incoming (called by the transport)

    func receive(answer: Answer) {
        guard let id = answer.id else {
            if answer.isNotice {
                receive(notice: answer)
            } else if answer.isRoomAnswer, showRoomVoiceAnswers {
                heardInRoom = answer
                setHighlight(highlight(for: answer))
            }
            return
        }
        // Only the latest question's answer counts; anything else is stale.
        guard exchanges.first?.id == id else { return }
        timeoutTask?.cancel()
        exchanges[0].answer = answer
        exchanges[0].timedOut = false
        setHighlight(highlight(for: answer))
    }

    /// A reminder always shows on Home, even with room answers off: it's meant for the person.
    private func receive(notice answer: Answer) {
        let notice = Notice(rig: answer)
        rigNotices.removeAll { $0.id == notice.id }
        rigNotices.insert(notice, at: 0)
        if rigNotices.count > Self.rigNoticeLimit { rigNotices.removeLast(rigNotices.count - Self.rigNoticeLimit) }
        setHighlight(highlight(for: answer))
    }

    func receive(state: Snapshot) {
        if let old = snapshot {
            activity.insert(contentsOf: Dashboard.changes(from: old, to: state).reversed(), at: 0)
            if activity.count > Dashboard.activityLimit { activity.removeLast(activity.count - Dashboard.activityLimit) }
        }
        snapshot = state
    }

    func dismiss(_ notice: Notice) {
        dismissedNotices.insert(notice.id)
    }

    /// Brings back every notice put away with "Got it" (helper settings).
    func restoreNotices() {
        dismissedNotices = []
    }

    func receive(status: RigStatus) {
        self.status = status
    }

    func linkChanged(_ state: LinkState) {
        if state == .connected { hasConnected = true }
        link = (state == .searching && hasConnected) ? .reconnecting : state
    }

    // MARK: Highlight

    /// Lights up a thing on the map without asking the rig (Home, "Show me").
    func showOnMap(_ name: String) {
        guard let entity = snapshot?.entity(named: name), let target = MapLayout.position(of: entity) else { return }
        setHighlight(Highlight(entity: name, target: target, action: .point))
    }

    /// Where an answer points on the map, if anywhere. The answer sheet keeps its own copy.
    func highlight(for answer: Answer) -> Highlight? {
        let entity = answer.pointAt.flatMap { snapshot?.entity(named: $0) }
        let target = answer.target ?? entity.flatMap(MapLayout.position(of:))
        let action = answer.laserAction
        let hasEdge: Bool = { if case .sweep = action { return true } else { return false } }()
        guard target != nil || hasEdge else { return nil }
        return Highlight(entity: answer.pointAt, target: target, action: action)
    }

    private func setHighlight(_ new: Highlight?) {
        highlight = new
        highlightTask?.cancel()
        guard let id = new?.id else { return }
        highlightTask = Task { [weak self, highlightDuration] in
            try? await Task.sleep(for: highlightDuration)
            guard !Task.isCancelled, self?.highlight?.id == id else { return }
            self?.highlight = nil
        }
    }
}
