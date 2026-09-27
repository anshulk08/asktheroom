import XCTest
@testable import AskTheRoom

@MainActor
private final class FakeTransport: RoomTransport {
    var sent: [Question] = []
    var orients: [OrientSettings] = []
    func send(_ question: Question) { sent.append(question) }
    func send(orient: OrientSettings) { orients.append(orient) }
    func stop() {}
}

@MainActor
final class RoomStoreTests: XCTestCase {
    private var transport: FakeTransport!
    private var store: RoomStore!

    override func setUp() async throws {
        let t = FakeTransport()
        transport = t
        store = RoomStore(mock: false, liveTransport: { _ in t })
        store.receive(state: MockData.sampleSnapshot)
    }

    private func answer(_ id: Int?, _ text: String = "Your keys are inside the box.") -> Answer {
        Answer(id: id, ok: true, text: text, point_at: "keys", action: "point", target: TablePoint(x: 70.4, y: 38.1))
    }

    private func sentOrients() -> [String] {
        transport.orients.compactMap { $0.encoded() }.map { String(decoding: $0, as: UTF8.self) }
    }

    /// Helper settings' seat goes to the rig; going back to the default sends a reset; never
    /// choosing sends nothing.
    func testSeatAndResetAreSent() {
        let defaults = UserDefaults.standard
        let clear = { [Seat.savedKey, Seat.resetKey].forEach(defaults.removeObject(forKey:)) }
        clear()
        defer { clear() }
        store.sendSeat()
        XCTAssertEqual(sentOrients(), [])
        Seat.choose(.right)
        store.sendSeat()
        Seat.choose(nil)
        store.sendSeat()
        XCTAssertEqual(sentOrients(), [#"{"orient":{"front":"right"}}"#, #"{"orient":{"front":null}}"#])
    }

    func testAskSendsAFreshID() {
        store.ask("Where are my keys?")
        store.ask("  ")
        store.ask("What changed?")
        XCTAssertEqual(transport.sent.map(\.q), ["Where are my keys?", "What changed?"])
        XCTAssertNotEqual(transport.sent[0].id, transport.sent[1].id)
        XCTAssertTrue(store.current!.isPending)
    }

    func testMatchingAnswerIsShownAndHighlighted() {
        store.ask("Where are my keys?")
        store.receive(answer: answer(transport.sent[0].id))
        XCTAssertEqual(store.current?.answer?.text, "Your keys are inside the box.")
        XCTAssertEqual(store.highlight?.entity, "keys")
        XCTAssertEqual(store.highlight?.target, TablePoint(x: 70.4, y: 38.1))
    }

    func testAnswerToAnOlderQuestionIsIgnored() {
        store.ask("Where are my keys?")
        store.ask("Where is my wallet?")
        store.receive(answer: answer(transport.sent[0].id))
        XCTAssertNil(store.current?.answer)
        XCTAssertNil(store.highlight)
    }

    /// The phone must not give up before the bridge does (12 s), or a retry queues behind the first question.
    func testPhoneWaitsLongerThanTheBridge() {
        XCTAssertGreaterThan(store.answerTimeout, .seconds(12))
        XCTAssertLessThan(store.slowAfter, store.answerTimeout)
    }

    func testSlowThenAnswered() async throws {
        store.slowAfter = .milliseconds(20)
        store.answerTimeout = .seconds(5)
        store.ask("What's on the table?")
        try await Task.sleep(for: .milliseconds(200))
        XCTAssertTrue(store.current!.slow)
        XCTAssertTrue(store.current!.isPending, "slow is still waiting, not timed out")
        store.receive(answer: answer(transport.sent[0].id))
        XCTAssertNotNil(store.current?.answer)
    }

    func testTimeoutThenLateAnswer() async throws {
        store.slowAfter = .milliseconds(10)
        store.answerTimeout = .milliseconds(50)
        store.ask("Where are my keys?")
        try await Task.sleep(for: .milliseconds(300))
        XCTAssertTrue(store.current!.timedOut)
        store.receive(answer: answer(transport.sent[0].id))
        XCTAssertFalse(store.current!.timedOut)
        XCTAssertNotNil(store.current?.answer)
    }

    func testRetryAsksAgain() async throws {
        store.slowAfter = .milliseconds(10)
        store.answerTimeout = .milliseconds(20)
        store.ask("Where are my keys?")
        try await Task.sleep(for: .milliseconds(200))
        store.retry()
        XCTAssertEqual(transport.sent.count, 2)
        XCTAssertEqual(transport.sent[1].q, "Where are my keys?")
    }

    func testHistoryKeepsTheLastTen() {
        for i in 1...13 { store.ask("q\(i)") }
        XCTAssertEqual(store.exchanges.count, RoomStore.historyLimit)
        XCTAssertEqual(store.current?.question, "q13")
        XCTAssertEqual(store.exchanges.last?.question, "q4")
        XCTAssertEqual(store.history.count, 9)
    }

    func testHighlightFallsBackToTheEntityPosition() {
        store.ask("Where is my wallet?")
        store.receive(answer: Answer(id: transport.sent[0].id, ok: true, text: "On the table.", point_at: "wallet", action: "point"))
        XCTAssertEqual(store.highlight?.target, TablePoint(x: 60, y: 15))
    }

    func testSweepHighlightsWithoutATarget() {
        store.ask("Where is my phone?")
        store.receive(answer: Answer(id: transport.sent[0].id, ok: true, text: "It left.", action: "sweep:left"))
        XCTAssertEqual(store.highlight?.action, .sweep(.left))
    }

    func testNoHighlightForTextOnlyAnswers() {
        store.ask("What changed?")
        store.receive(answer: Answer(id: transport.sent[0].id, ok: true, text: "The box moved."))
        XCTAssertNil(store.highlight)
        XCTAssertEqual(store.current?.answer?.text, "The box moved.")
    }

    func testRoomAnswersOnlyWithTheFlag() {
        var heard = answer(nil)
        heard.src = "voice"
        heard.q = "where are my keys"
        store.showRoomVoiceAnswers = false
        store.receive(answer: heard)
        XCTAssertNil(store.heardInRoom)

        store.showRoomVoiceAnswers = true
        store.receive(answer: heard)
        XCTAssertEqual(store.heardInRoom?.q, "where are my keys")
        XCTAssertEqual(store.highlight?.entity, "keys")
    }

    /// The bridge forwards dashboard and SMS questions the same way (PROTOCOL.md 6a).
    func testDashboardAndSMSAnswersShowToo() {
        store.showRoomVoiceAnswers = true
        for src in ["dashboard", "sms"] {
            var heard = answer(nil, "Your wallet is on the table.")
            heard.src = src
            store.receive(answer: heard)
            XCTAssertEqual(store.heardInRoom?.src, src)
        }
        var bare = answer(nil, "No source")
        bare.src = nil
        store.receive(answer: bare)
        XCTAssertEqual(store.heardInRoom?.text, "Your wallet is on the table.", "an id-less answer with no src is ignored")
    }

    private func notice(_ nid: Int, _ text: String, pointAt: String? = "pill_bottle", kind: String = "reminder") -> Answer {
        var a = Answer(id: nil, ok: true, text: text, point_at: pointAt, action: pointAt == nil ? nil : "point",
                       target: pointAt == nil ? nil : TablePoint(x: 30.2, y: 12))
        a.src = "notice"
        a.nid = nid
        a.kind = kind
        return a
    }

    /// Reminders show on Home first, even with room answers off, and light up what they point at.
    func testRigNoticesComeFirstWhateverTheSetting() {
        store.showRoomVoiceAnswers = false
        store.receive(answer: notice(12, "It's 9 and the pill bottle hasn't been picked up yet."))
        XCTAssertNil(store.heardInRoom)
        XCTAssertEqual(store.notices.first?.id, "rig|12")
        XCTAssertEqual(store.notices.first?.kind, .rig("reminder"))
        XCTAssertEqual(store.notices.first?.entity, "pill_bottle")
        XCTAssertEqual(store.highlight?.entity, "pill_bottle")
    }

    func testRigNoticesAreKeptShortAndCanBePutAway() {
        for nid in 1...5 { store.receive(answer: notice(nid, "Reminder \(nid)", pointAt: nil, kind: "morning")) }
        store.receive(answer: notice(5, "Reminder 5", pointAt: nil, kind: "morning"))
        XCTAssertEqual(store.rigNotices.map(\.rigID), [5, 4, 3], "newest first, no repeats, at most three")
        XCTAssertEqual(store.rigNotices.first?.entity, "")

        store.dismiss(store.rigNotices[0])
        XCTAssertFalse(store.notices.contains { $0.rigID == 5 })
        store.restoreNotices()
        XCTAssertTrue(store.notices.contains { $0.rigID == 5 })
    }

    func testDropAfterConnectingShowsReconnecting() {
        XCTAssertEqual(store.link, .searching)
        store.linkChanged(.connected)
        XCTAssertTrue(store.hasConnected)
        store.linkChanged(.searching)
        XCTAssertEqual(store.link, .reconnecting)
    }

    func testStatusBanners() {
        store.receive(status: RigStatus(app: "down", fps: 0, online: true, cal: true, laser_cal: true))
        XCTAssertTrue(store.isRoomAppDown)
        store.receive(status: RigStatus(app: "up", fps: 15, online: false, cal: true, laser_cal: true))
        XCTAssertFalse(store.isRoomAppDown)
        XCTAssertTrue(store.isOffline)
    }

    func testMockModeAnswersLocally() async throws {
        store.setMock(true)
        XCTAssertTrue(store.isMock)
        XCTAssertNotNil(store.snapshot)
        store.ask("Where are my keys?")
        try await Task.sleep(for: .seconds(1.2))
        XCTAssertEqual(store.current?.answer?.text, "Your keys are inside the box.")
        XCTAssertEqual(store.snapshot?.laser?.target, "keys")
        store.setMock(false)
        XCTAssertNil(store.snapshot)
    }

    // MARK: Link additions

    /// The hello goes first on every connect, before the voice and the seat.
    func testHelloIsWrittenFirstOnConnect() throws {
        let voice = VoiceSettings(voice: .init(e: "grok", v: "eve", s: 1))
        let writes = RoomLink.connectWrites(voice: voice, orient: OrientSettings(front: .right))
            .map { String(decoding: $0, as: UTF8.self) }
        XCTAssertEqual(writes.count, 3)
        XCTAssertEqual(writes[0], #"{"hello":{"z":1}}"#)
        XCTAssertEqual(writes[1], String(decoding: try XCTUnwrap(voice.encoded()), as: UTF8.self))
        XCTAssertEqual(writes[2], #"{"orient":{"front":"right"}}"#)
        XCTAssertEqual(RoomLink.connectWrites(voice: voice, orient: nil).first, Data(#"{"hello":{"z":1}}"#.utf8))
    }

    /// The layout comes only now and then; the store keeps it, and knows when the rig has moved on.
    func testLayoutIsKeptAcrossStates() {
        var withLayout = MockData.sampleSnapshot
        withLayout.lh = "h1"
        withLayout.lay = RoomLayout(v: 1, size: [400, 300])
        store.receive(state: withLayout)
        XCTAssertEqual(store.layout?.size, [400, 300])
        XCTAssertTrue(store.layoutIsCurrent)

        var plain = MockData.sampleSnapshot
        plain.lh = "h1"
        store.receive(state: plain)
        XCTAssertEqual(store.layout?.size, [400, 300], "kept")
        XCTAssertEqual(store.layoutHash, "h1")
        XCTAssertTrue(store.layoutIsCurrent)

        plain.lh = "h2"
        store.receive(state: plain)
        XCTAssertNotNil(store.layout, "the old one is still there to draw")
        XCTAssertFalse(store.layoutIsCurrent, "but the rig has a newer one")

        withLayout.lh = "h2"
        withLayout.lay = RoomLayout(v: 1, size: [500, 300])
        store.receive(state: withLayout)
        XCTAssertEqual(store.layout?.size, [500, 300])
        XCTAssertTrue(store.layoutIsCurrent)
    }

    func testLinkStatsArePublished() {
        XCTAssertNil(store.linkStats)
        var stats = LinkStats()
        stats.state.chunks = 3
        store.receive(linkStats: stats)
        XCTAssertEqual(store.linkStats?.state.chunks, 3)
        store.setMock(false)
        XCTAssertNil(store.linkStats)
    }
}

/// The last live map is kept on the phone, so the app opens on it while it finds the rig.
@MainActor
final class SavedMapTests: XCTestCase {
    private var url: URL!

    override func setUp() async throws {
        url = FileManager.default.temporaryDirectory.appending(path: "saved-map-\(UUID().uuidString).json")
    }

    override func tearDown() async throws {
        try? FileManager.default.removeItem(at: url)
    }

    private func liveStore() -> RoomStore {
        RoomStore(mock: false, liveTransport: { _ in FakeTransport() }, savedMap: url)
    }

    func testNoSavedMapOpensEmpty() {
        let store = liveStore()
        XCTAssertNil(store.snapshot)
        XCTAssertFalse(store.isSavedMap)
        XCTAssertFalse(store.isMapStale)
    }

    func testLiveMapIsSavedAndOpensNextTime() {
        liveStore().receive(state: MockData.sampleSnapshot)
        let next = liveStore()
        XCTAssertEqual(next.snapshot, MockData.sampleSnapshot)
        XCTAssertTrue(next.isSavedMap)
        XCTAssertTrue(next.isMapStale)
        XCTAssertTrue(next.activity.isEmpty)
    }

    func testFirstLiveMapReplacesTheSavedOneWithoutRecentLines() {
        liveStore().receive(state: MockData.sampleSnapshot)
        let store = liveStore()
        var moved = MockData.sampleSnapshot
        moved.e = moved.e.map { e in var e = e; if e.n == "keys" { e.s = .visible; e.p = nil }; return e }
        store.linkChanged(.connected)
        store.receive(state: moved)
        XCTAssertEqual(store.snapshot, moved)
        XCTAssertFalse(store.isSavedMap)
        XCTAssertFalse(store.isMapStale)
        // What changed while the app was closed happened at unknown times.
        XCTAssertTrue(store.activity.isEmpty)
    }

    func testDroppedLinkMarksTheMapStale() {
        let store = liveStore()
        store.linkChanged(.connected)
        store.receive(state: MockData.sampleSnapshot)
        XCTAssertFalse(store.isMapStale)
        store.linkChanged(.searching)
        XCTAssertTrue(store.isMapStale)
    }

    func testDemoModeNeitherSavesNorShowsTheSavedMap() {
        let mock = RoomStore(mock: true, savedMap: url)
        mock.receive(state: MockData.sampleSnapshot)
        XCTAssertFalse(FileManager.default.fileExists(atPath: url.path()))
        liveStore().receive(state: MockData.sampleSnapshot)
        let store = RoomStore(mock: true, savedMap: url)
        XCTAssertFalse(store.isSavedMap)
        XCTAssertFalse(store.isMapStale)
    }

    func testBannerTimeSaysTheDayWhenNotToday() {
        let now = Date(timeIntervalSince1970: 1_790_420_000)
        XCTAssertFalse(Banners.when(now.addingTimeInterval(-60), now: now).isEmpty)
        let yesterday = Banners.when(now.addingTimeInterval(-86_400), now: now)
        XCTAssertNotEqual(yesterday, Banners.when(now.addingTimeInterval(-60), now: now))
    }

}
