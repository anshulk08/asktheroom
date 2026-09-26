import XCTest
@testable import AskTheRoom

@MainActor
private final class FakeTransport: RoomTransport {
    var sent: [Question] = []
    func send(_ question: Question) { sent.append(question) }
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

    func testTimeoutThenLateAnswer() async throws {
        store.answerTimeout = .milliseconds(50)
        store.ask("Where are my keys?")
        try await Task.sleep(for: .milliseconds(300))
        XCTAssertTrue(store.current!.timedOut)
        store.receive(answer: answer(transport.sent[0].id))
        XCTAssertFalse(store.current!.timedOut)
        XCTAssertNotNil(store.current?.answer)
    }

    func testRetryAsksAgain() async throws {
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

    func testRoomVoiceAnswersOnlyWithTheFlag() {
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
}
