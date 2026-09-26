import AVFoundation

/// Reads answers aloud on the phone. Off by default: the rig already speaks in the room,
/// so this is for when the phone is used away from the table.
@MainActor
final class Speaker {
    static let enabledKey = "readAnswersAloud"
    static let shared = Speaker()

    private let synthesizer = AVSpeechSynthesizer()

    func say(_ text: String) {
        guard UserDefaults.standard.bool(forKey: Self.enabledKey), !text.isEmpty else { return }
        // Plays with the silent switch on, like a spoken answer should.
        try? AVAudioSession.sharedInstance().setCategory(.playback, mode: .spokenAudio, options: .duckOthers)
        try? AVAudioSession.sharedInstance().setActive(true)
        synthesizer.stopSpeaking(at: .immediate)
        let utterance = AVSpeechUtterance(string: text)
        utterance.rate = AVSpeechUtteranceDefaultSpeechRate * 0.9
        synthesizer.speak(utterance)
    }
}
