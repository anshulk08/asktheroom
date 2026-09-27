import XCTest
@testable import AskTheRoom

final class ModelsTests: XCTestCase {
    func testSampleSnapshotDecodesEveryStatus() throws {
        let snap = try XCTUnwrap(Wire.decode(Snapshot.self, from: Data(MockData.sampleSnapshotJSON.utf8)))
        XCTAssertEqual(snap.v, 1)
        XCTAssertEqual(snap.tableSize, TablePoint(x: 90, y: 60))
        XCTAssertEqual(snap.laser, LaserState(on: true, target: "keys"))
        XCTAssertEqual(snap.entities.count, 12)
        XCTAssertEqual(Set(snap.entities.map(\.status)), [.visible, .held, .under, .inside, .gone, .lost])

        let keys = try XCTUnwrap(snap.entity(named: "keys"))
        XCTAssertEqual(keys.drawPoint, TablePoint(x: 70.4, y: 38.1))
        XCTAssertEqual(keys.parent, "box")

        let phone = try XCTUnwrap(snap.entity(named: "phone"))
        XCTAssertEqual(phone.edge, .left)
        XCTAssertNil(phone.r)
        XCTAssertEqual(phone.drawPoint, TablePoint(x: 3, y: 30))

        XCTAssertTrue(try XCTUnwrap(snap.entity(named: "glasses")).isUncertain)
        XCTAssertTrue(try XCTUnwrap(snap.entity(named: "remote")).isInHand)
        XCTAssertEqual(try XCTUnwrap(snap.entity(named: "thing:9")).maybeSameAs, [MaybeSame(name: "thing:4", score: 0.62)])
    }

    func testDisplayNames() {
        XCTAssertEqual(Entity.displayName(for: "pill_bottle"), "pill bottle")
        XCTAssertEqual(Entity.displayName(for: "thing:7", aliases: ["my charger"]), "my charger")
        XCTAssertEqual(Entity.displayName(for: "thing:9"), "unnamed object 9")
    }

    private func thing(_ fields: String) throws -> Entity {
        let json = #"{"e":[{"n":"thing:7","k":"t","s":"V"\#(fields)}]}"#
        return try XCTUnwrap(Wire.decode(Snapshot.self, from: Data(json.utf8))?.entities.first)
    }

    func testTaughtNameIsThePersons() throws {
        let e = try thing(#","a":["my charger"],"g":"phone charger","gc":0.9"#)
        XCTAssertTrue(e.hasTaughtName)
        XCTAssertFalse(e.isHedged)
        XCTAssertEqual(e.displayName, "my charger")
        XCTAssertEqual(e.phrase, "my charger")
    }

    func testGrokNameIsHedged() throws {
        let e = try thing(#","a":["tape roll"],"as":"grok""#)
        XCTAssertFalse(e.hasTaughtName)
        XCTAssertEqual(e.displayName, "tape roll?")
        XCTAssertEqual(e.phrase, "what looks like a tape roll")
        XCTAssertFalse(e.isNameless)
    }

    func testConfidentGuessIsHedged() throws {
        let e = try thing(#","g":"apple","gc":0.5"#)
        XCTAssertEqual(e.displayName, "apple?")
        XCTAssertEqual(e.phrase, "what looks like an apple")
        XCTAssertEqual(e.thingNumber, "7")
        // Older bridges send no `gc`: 0.6 is enough.
        XCTAssertEqual(try thing(#","g":"cup""#).displayName, "cup?")
    }

    func testWeakOrMissingGuessStaysUnnamed() throws {
        for fields in [#","g":"cup","gc":0.3"#, "", #","g":"","gc":0.9"#] {
            let e = try thing(fields)
            XCTAssertFalse(e.isHedged, fields)
            XCTAssertTrue(e.isNameless, fields)
            XCTAssertEqual(e.displayName, "unnamed object 7", fields)
        }
    }

    func testNamedTargetsIgnoreGuesses() throws {
        let json = #"{"e":[{"n":"keys","k":"t","s":"V","g":"coins","gc":0.9}]}"#
        let keys = try XCTUnwrap(Wire.decode(Snapshot.self, from: Data(json.utf8))?.entities.first)
        XCTAssertFalse(keys.isHedged)
        XCTAssertEqual(keys.displayName, "keys")
    }

    func testChainFollowsNestingAndStopsAtHands() {
        let json = """
        {"e":[
          {"n":"keys","k":"t","s":"U","p":"notebook"},
          {"n":"notebook","k":"v","s":"I","p":"box"},
          {"n":"box","k":"c","s":"H","p":"hand:1"}
        ]}
        """
        let snap = Wire.decode(Snapshot.self, from: Data(json.utf8))!
        XCTAssertEqual(snap.chain(from: "keys").map(\.name), ["keys", "notebook", "box"])
        XCTAssertEqual(snap.tableSize, Snapshot.defaultTable)
    }

    func testChainSurvivesCyclesAndMissingParents() {
        let json = """
        {"e":[{"n":"a","k":"t","s":"I","p":"b"},{"n":"b","k":"c","s":"I","p":"a"},{"n":"c","k":"t","s":"I","p":"nowhere"}]}
        """
        let snap = Wire.decode(Snapshot.self, from: Data(json.utf8))!
        XCTAssertEqual(snap.chain(from: "a").map(\.name), ["a", "b"])
        XCTAssertEqual(snap.chain(from: "c").map(\.name), ["c"])
    }

    func testUnknownValuesDecodeInsteadOfFailing() throws {
        let json = #"{"e":[{"n":"keys","k":"z","s":"Q","edge":"left"}]}"#
        let keys = try XCTUnwrap(Wire.decode(Snapshot.self, from: Data(json.utf8))?.entities.first)
        XCTAssertEqual(keys.kind, .unrecognized)
        XCTAssertEqual(keys.status, .unrecognized)
    }

    func testAnswerActionsAreAnOpenString() throws {
        let raw = #"{"id":17,"ok":true,"text":"Your keys are inside the box.","point_at":"keys","action":"trace","target":[70.4,38.1],"ms":412}"#
        let answer = try XCTUnwrap(Wire.decode(Answer.self, from: Data(raw.utf8)))
        XCTAssertEqual(answer.laserAction, .other("trace"))
        XCTAssertEqual(answer.target, TablePoint(x: 70.4, y: 38.1))
        XCTAssertEqual(LaserAction("sweep:right"), .sweep(.right))
        XCTAssertEqual(LaserAction("sweep:sideways"), .other("sweep:sideways"))
        XCTAssertEqual(LaserAction("circle"), .circle)
    }

    func testAnswerWithoutIDOrActionDecodes() throws {
        let raw = #"{"id":null,"ok":false,"text":"The room isn't running right now."}"#
        let answer = try XCTUnwrap(Wire.decode(Answer.self, from: Data(raw.utf8)))
        XCTAssertNil(answer.id)
        XCTAssertFalse(answer.succeeded)
        XCTAssertNil(answer.laserAction)
    }

    /// The two examples in PROTOCOL.md 6a.
    func testAnswersThePhoneDidntAskFor() throws {
        let room = try XCTUnwrap(Wire.decode(Answer.self, from: Data("""
            {"id": null, "src": "voice", "q": "where are my keys", "ok": true, "text": "Your keys are inside the box.",
             "point_at": "keys", "action": "point", "target": [70.4, 38.1], "ms": null}
            """.utf8)))
        XCTAssertTrue(room.isRoomAnswer)
        XCTAssertFalse(room.isNotice)
        XCTAssertEqual(room.q, "where are my keys")

        let notice = try XCTUnwrap(Wire.decode(Answer.self, from: Data("""
            {"id": null, "src": "notice", "nid": 12, "kind": "reminder", "ok": true,
             "text": "It's 9 and the pill bottle hasn't been picked up yet.", "point_at": "pill_bottle", "action": "point",
             "target": [30.2, 12.0], "ms": null}
            """.utf8)))
        XCTAssertTrue(notice.isNotice)
        XCTAssertFalse(notice.isRoomAnswer)
        XCTAssertEqual(notice.nid, 12)
        XCTAssertEqual(notice.kind, "reminder")

        let mine = try XCTUnwrap(Wire.decode(Answer.self, from: Data(#"{"id": 3, "src": "voice", "text": "x"}"#.utf8)))
        XCTAssertFalse(mine.isRoomAnswer, "an answer with an id is the phone's own")
    }

    func testMalformedJSONReturnsNil() {
        XCTAssertNil(Wire.decode(Snapshot.self, from: Data("{\"e\":[".utf8)))
        XCTAssertNil(Wire.decode(Answer.self, from: Data("not json".utf8)))
    }

    func testStatus() throws {
        let raw = #"{"app":"down","fps":0}"#
        let status = try XCTUnwrap(Wire.decode(RigStatus.self, from: Data(raw.utf8)))
        XCTAssertFalse(status.appIsUp)
    }

    func testQuestionFitsInOneWrite() throws {
        let short = try XCTUnwrap(Question(id: 17, q: "where are my keys?").encoded())
        XCTAssertEqual(String(decoding: short, as: UTF8.self), #"{"id":17,"q":"where are my keys?"}"#)

        let long = try XCTUnwrap(Question(id: 65535, q: String(repeating: "é", count: 300)).encoded())
        XCTAssertLessThanOrEqual(long.count, Question.maxBytes)
        XCTAssertNotNil(Wire.decode(Question.self, from: long))
    }

    // MARK: the rig's speaker and the helper's voice (PROTOCOL.md 5a, 8)

    func testRigSpeaksOnlyWhenTheAppIsUpAndTheSpeakerIsOn() throws {
        func status(_ json: String) throws -> RigStatus {
            try XCTUnwrap(Wire.decode(RigStatus.self, from: Data(json.utf8)))
        }
        XCTAssertTrue(try status(#"{"app":"up","fps":12.0,"online":true,"cal":true,"laser_cal":false,"gk":true,"spk":true}"#).rigSpeaks)
        XCTAssertFalse(try status(#"{"app":"up","spk":false}"#).rigSpeaks)
        XCTAssertFalse(try status(#"{"app":"down","spk":true}"#).rigSpeaks)
        XCTAssertFalse(try status(#"{"app":"up","fps":12.0}"#).rigSpeaks)          // an older rig
    }

    func testVoiceSettingsAreOneShortWriteTheRigReads() throws {
        let data = try XCTUnwrap(VoiceSettings(voice: .init(e: "rigVoice", v: "ara", s: 1.1)).encoded())
        XCTAssertEqual(String(decoding: data, as: UTF8.self), #"{"voice":{"e":"rigVoice","s":1.1,"v":"ara"}}"#)
        XCTAssertLessThanOrEqual(data.count, Question.maxBytes)
        XCTAssertNil(VoiceSettings(voice: .init(e: "grok", v: String(repeating: "x", count: 300), s: 1)).encoded())
    }

    // MARK: where the person sits (the "view" key and the orient write)

    func testViewDecodesFromTheSample() throws {
        let snap = try XCTUnwrap(Wire.decode(Snapshot.self, from: Data(MockData.sampleSnapshotJSON.utf8)))
        let view = try XCTUnwrap(snap.view)
        XCTAssertEqual(view, ViewInfo(front: .right, outline: true, sides: [.right: "couch"]))
        XCTAssertEqual(view.name(at: .bottom), "couch", "the couch is where the person sits")
        XCTAssertNil(view.name(at: .top))
        XCTAssertEqual(view.viewerEdge(of: .right), .bottom)
    }

    func testOlderRigsSendNoView() throws {
        let snap = try XCTUnwrap(Wire.decode(Snapshot.self, from: Data(#"{"v":1,"table":[90,60],"e":[]}"#.utf8)))
        XCTAssertNil(snap.view)
    }

    func testViewIsTolerant() throws {
        let odd = #"{"e":[],"view":{"f":"diagonal","s":{"upstairs":"x","left":"window"}}}"#
        let view = try XCTUnwrap(Wire.decode(Snapshot.self, from: Data(odd.utf8))?.view)
        XCTAssertEqual(view.front, .bottom, "an unknown side reads as the camera's own")
        XCTAssertFalse(view.outline)
        XCTAssertEqual(view.sides, [.left: "window"])

        let bare = try XCTUnwrap(Wire.decode(Snapshot.self, from: Data(#"{"e":[],"view":{"f":"top"}}"#.utf8))?.view)
        XCTAssertEqual(bare, ViewInfo(front: .top))
    }

    /// The saved map keeps its view.
    func testViewSurvivesSavingTheMap() throws {
        let snap = MockData.sampleSnapshot
        let again = try JSONDecoder().decode(Snapshot.self, from: try JSONEncoder().encode(snap))
        XCTAssertEqual(again, snap)
        XCTAssertNotNil(again.view)
    }

    /// The rig's camera to viewer table, per front, written out in full.
    private let rigTable: [Side: [Side: Edge]] = [
        .bottom: [.left: .left, .right: .right, .top: .top, .bottom: .bottom],
        .top: [.left: .right, .right: .left, .top: .bottom, .bottom: .top],
        .right: [.right: .bottom, .left: .top, .bottom: .left, .top: .right],
        .left: [.left: .bottom, .right: .top, .top: .left, .bottom: .right],
    ]

    func testCameraSidesTurnLikeTheRigsTable() {
        for front in Side.allCases {
            for side in Side.allCases {
                XCTAssertEqual(Seat.viewerEdge(of: side, front: front), rigTable[front]![side]!, "\(side) sitting at \(front)")
            }
        }
    }

    func testViewerEdgesTurnBackToCameraSides() {
        for front in Side.allCases {
            for (side, edge) in rigTable[front]! {
                XCTAssertEqual(Seat.cameraSide(at: edge, front: front), side, "\(edge) sitting at \(front)")
            }
        }
    }

    func testTurningIsARoundTripAndTheSeatIsAlwaysAtTheBottom() {
        for front in Side.allCases {
            XCTAssertEqual(Seat.viewerEdge(of: front, front: front), .bottom)
            XCTAssertEqual(Seat.cameraSide(at: .bottom, front: front), front)
            for edge in Edge.allCases {
                XCTAssertEqual(Seat.viewerEdge(of: Seat.cameraSide(at: edge, front: front), front: front), edge)
                XCTAssertEqual(Seat.cameraSide(at: Seat.viewerEdge(of: edge, front: front), front: front), edge)
            }
            XCTAssertEqual(Set(Edge.allCases.map { Seat.cameraSide(at: $0, front: front) }), Set(Side.allCases))
        }
    }

    func testOrientIsOneShortWriteTheRigReads() throws {
        let data = try XCTUnwrap(OrientSettings(front: .right).encoded())
        XCTAssertEqual(String(decoding: data, as: UTF8.self), #"{"orient":{"front":"right"}}"#)
        XCTAssertEqual(OrientSettings(front: .left).front, .left)
    }

    /// The reset must carry an explicit null: a missing key isn't a reset.
    func testResetSendsAnExplicitNull() throws {
        let data = try XCTUnwrap(OrientSettings.reset.encoded())
        XCTAssertEqual(String(decoding: data, as: UTF8.self), #"{"orient":{"front":null}}"#)
        XCTAssertEqual(data.count, 25)
        XCTAssertNil(OrientSettings.reset.front)
        let parsed = try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: [String: Any]])
        XCTAssertTrue(parsed["orient"]?["front"] is NSNull)
    }

    private func clearSeat() {
        UserDefaults.standard.removeObject(forKey: Seat.savedKey)
        UserDefaults.standard.removeObject(forKey: Seat.resetKey)
    }

    func testSavedSeatIsACameraSideAndGoesOnEveryConnect() {
        clearSeat()
        defer { clearSeat() }
        XCTAssertNil(Seat.savedOrient, "never chosen: the rig's default applies")
        XCTAssertNil(Seat.orientForConnect())
        Seat.choose(.top)
        XCTAssertEqual(Seat.saved, .top)
        XCTAssertEqual(Seat.savedOrient, OrientSettings(front: .top))
        XCTAssertEqual(Seat.orientForConnect(), OrientSettings(front: .top))
        XCTAssertEqual(Seat.orientForConnect(), OrientSettings(front: .top), "again on the next connect")
    }

    /// Back to the default: a reset now and on the next connect, then nothing.
    func testResetGoesOutOnOneConnectThenNothing() {
        clearSeat()
        defer { clearSeat() }
        Seat.choose(.left)
        Seat.choose(nil)
        XCTAssertNil(Seat.saved)
        XCTAssertEqual(Seat.savedOrient, .reset, "sent when picked")
        XCTAssertEqual(Seat.orientForConnect(), .reset, "and on the next connect")
        XCTAssertNil(Seat.orientForConnect(), "then forgotten")
        XCTAssertNil(Seat.savedOrient)
        Seat.choose(nil)
        Seat.choose(.bottom)
        XCTAssertEqual(Seat.orientForConnect(), OrientSettings(front: .bottom), "a new seat cancels an owed reset")
    }

    /// `s` is keyed by camera side, like `f`: it lands wherever that side is on the turned map.
    func testSideNamesAreKeyedByCameraSide() throws {
        func view(_ f: String) throws -> ViewInfo {
            let json = #"{"e":[],"view":{"f":"\#(f)","o":true,"s":{"right":"couch"}}}"#
            return try XCTUnwrap(Wire.decode(Snapshot.self, from: Data(json.utf8))?.view)
        }
        XCTAssertEqual(try view("right").name(at: .bottom), "couch", "on the couch: 'You · Couch'")
        XCTAssertEqual(try view("top").name(at: .left), "couch")
        XCTAssertEqual(try view("left").name(at: .top), "couch")
        XCTAssertEqual(try view("bottom").name(at: .right), "couch")
        for f in ["right", "top", "left", "bottom"] {
            XCTAssertEqual(Edge.allCases.compactMap { try? view(f).name(at: $0) }, ["couch"], f)
        }
    }
}
