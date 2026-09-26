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
}
