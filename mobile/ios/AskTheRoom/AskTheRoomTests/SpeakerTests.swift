import AVFoundation
import XCTest
@testable import AskTheRoom

final class SpeakerTests: XCTestCase {
    func testRigVoiceRequestMatchesTheRigsCall() throws {
        let request = try XCTUnwrap(ElevenLabs.request(text: "Your keys are inside the box.", voiceID: "abc123", key: "k"))
        XCTAssertEqual(request.url?.absoluteString,
                       "https://api.elevenlabs.io/v1/text-to-speech/abc123?output_format=mp3_44100_128")
        XCTAssertEqual(request.httpMethod, "POST")
        XCTAssertEqual(request.value(forHTTPHeaderField: "xi-api-key"), "k")
        let body = try JSONSerialization.jsonObject(with: XCTUnwrap(request.httpBody)) as? [String: String]
        XCTAssertEqual(body, ["text": "Your keys are inside the box.", "model_id": "eleven_flash_v2_5"])
        XCTAssertEqual(request.timeoutInterval, ElevenLabs.timeout)
    }

    func testBuiltInVoiceWithoutKeyOrVoiceID() {
        XCTAssertNil(ElevenLabs.request(text: "Hi", voiceID: "abc123", key: nil))
        XCTAssertNil(ElevenLabs.request(text: "Hi", voiceID: "abc123", key: ""))
        XCTAssertNil(ElevenLabs.request(text: "Hi", voiceID: "", key: "k"))
    }

    func testGrokRequestAsksForTheChosenVoiceSizedForTheRoute() throws {
        let headphones = VoiceRoute(kind: .ears, name: "AirPods")
        let request = try XCTUnwrap(Grok.request(text: "Your keys are inside the box.", voice: "leo",
                                                 speed: 1.2, route: headphones, key: "k"))
        XCTAssertEqual(request.url?.absoluteString, "https://api.x.ai/v1/tts")
        XCTAssertEqual(request.httpMethod, "POST")
        XCTAssertEqual(request.value(forHTTPHeaderField: "Authorization"), "Bearer k")
        XCTAssertEqual(request.timeoutInterval, Grok.timeout)
        let body = try XCTUnwrap(JSONSerialization.jsonObject(with: XCTUnwrap(request.httpBody)) as? [String: Any])
        XCTAssertEqual(body["text"] as? String, "Your keys are inside the box.")
        XCTAssertEqual(body["voice_id"] as? String, "leo")
        XCTAssertEqual(body["language"] as? String, "en")
        XCTAssertEqual(body["speed"] as? Double, 1.2)
        let format = try XCTUnwrap(body["output_format"] as? [String: Any])
        XCTAssertEqual(format["codec"] as? String, "mp3")
        XCTAssertEqual(format["sample_rate"] as? Int, 44_100)
        XCTAssertEqual(format["bit_rate"] as? Int, 128_000)
    }

    func testGrokNeedsAKeyAndDefaultsToEve() throws {
        let phone = VoiceRoute(kind: .phone, name: "iPhone")
        XCTAssertNil(Grok.request(text: "Hi", voice: "eve", speed: 1, route: phone, key: nil))
        XCTAssertNil(Grok.request(text: "Hi", voice: "eve", speed: 1, route: phone, key: ""))
        let request = try XCTUnwrap(Grok.request(text: "Hi", voice: "", speed: 1, route: phone, key: "k"))
        let body = try XCTUnwrap(JSONSerialization.jsonObject(with: XCTUnwrap(request.httpBody)) as? [String: Any])
        XCTAssertEqual(body["voice_id"] as? String, "eve")
    }

    func testRoutesMapToProfiles() {
        XCTAssertEqual(VoiceRoute.kind(of: .builtInSpeaker), .phone)
        XCTAssertEqual(VoiceRoute.kind(of: .builtInReceiver), .phone)
        XCTAssertEqual(VoiceRoute.kind(of: .headphones), .ears)
        XCTAssertEqual(VoiceRoute.kind(of: .bluetoothA2DP), .ears)
        XCTAssertEqual(VoiceRoute.kind(of: .bluetoothLE), .ears)
        XCTAssertEqual(VoiceRoute.kind(of: .bluetoothHFP), .headset)
        XCTAssertEqual(VoiceRoute.kind(of: .airPlay), .room)
        XCTAssertEqual(VoiceRoute.kind(of: .carAudio), .room)
        XCTAssertEqual(VoiceRoute.kind(of: .HDMI), .room)
    }

    func testRouteSlowsSpeakersAndCapsQualityForHeadsets() {
        let phone = VoiceRoute(kind: .phone, name: "iPhone")
        let room = VoiceRoute(kind: .room, name: "Living Room")
        let headset = VoiceRoute(kind: .headset, name: "Headset")
        XCTAssertEqual(phone.speed(1), 0.92)
        XCTAssertEqual(room.speed(1), 0.9)
        XCTAssertEqual(headset.speed(1), 1)
        XCTAssertEqual(headset.sampleRate, 16_000)
        // Never outside what Grok accepts.
        XCTAssertEqual(room.speed(0.7), 0.7)
        XCTAssertEqual(VoiceRoute(kind: .ears, name: "").speed(2), 1.5)
    }

    func testVoiceListParsesAndFallsBack() {
        let json = #"{"voices":[{"voice_id":"ara","name":"Ara","language":"en"},{"voice_id":"rex"}]}"#
        XCTAssertEqual(Grok.voices(from: Data(json.utf8)).map(\.id), ["ara", "rex"])
        XCTAssertEqual(Grok.voices(from: Data(json.utf8)).last?.name, "Rex")
        XCTAssertEqual(Grok.voices(from: Data("nope".utf8)), [])
        XCTAssertEqual(Grok.knownVoices.first?.id, Grok.defaultVoice)
    }
}
