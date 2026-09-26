import AVFoundation
import Security

/// Reads answers aloud on the phone, in the rig's own voice when it can. The rig speaks with
/// ElevenLabs (`voice/tts.py` on main: `ELEVENLABS_VOICE_ID`, model `eleven_flash_v2_5`) and
/// falls back to Piper offline; the phone does the same with the iPhone's built-in voice.
/// Off by default: the rig already speaks in the room, so this is for away from the table.
@MainActor
final class Speaker {
    static let enabledKey = "readAnswersAloud"
    static let voiceIDKey = "elevenLabsVoiceID"
    static let shared = Speaker()

    /// Which voice the last answer used, for helper settings.
    enum Engine: Equatable { case rigVoice, builtIn }
    private(set) var lastEngine: Engine?

    private let synthesizer = AVSpeechSynthesizer()
    private var player: AVAudioPlayer?
    private var fetch: Task<Void, Never>?

    /// The ElevenLabs key lives in the Keychain, entered in helper settings; never in the repo.
    static var apiKey: String? {
        get { Keychain.read(account: "elevenlabs") }
        set { Keychain.write(newValue, account: "elevenlabs") }
    }

    static var voiceID: String {
        UserDefaults.standard.string(forKey: voiceIDKey)?.trimmingCharacters(in: .whitespaces) ?? ""
    }

    func say(_ text: String) {
        guard UserDefaults.standard.bool(forKey: Self.enabledKey) else { return }
        speak(text)
    }

    /// Speaks now, whatever the setting (helper settings' "Try the voice").
    func speak(_ text: String) {
        let text = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty else { return }
        stop()
        // Plays with the silent switch on, like a spoken answer should.
        try? AVAudioSession.sharedInstance().setCategory(.playback, mode: .spokenAudio, options: .duckOthers)
        try? AVAudioSession.sharedInstance().setActive(true)

        guard let request = ElevenLabs.request(text: text, voiceID: Self.voiceID, key: Self.apiKey) else {
            speakBuiltIn(text)
            return
        }
        fetch = Task { [weak self] in
            let audio = try? await URLSession.shared.data(for: request)
            guard let self, !Task.isCancelled else { return }
            if let (data, response) = audio, (response as? HTTPURLResponse)?.statusCode == 200,
               let player = try? AVAudioPlayer(data: data), player.play() {
                self.player = player
                self.lastEngine = .rigVoice
            } else {
                // No network, a bad key or voice: say it anyway, like the rig's Piper fallback.
                self.speakBuiltIn(text)
            }
        }
    }

    func stop() {
        fetch?.cancel()
        player?.stop()
        synthesizer.stopSpeaking(at: .immediate)
    }

    private func speakBuiltIn(_ text: String) {
        let utterance = AVSpeechUtterance(string: text)
        utterance.rate = AVSpeechUtteranceDefaultSpeechRate * 0.9
        synthesizer.speak(utterance)
        lastEngine = .builtIn
    }
}

/// The same text-to-speech call the rig makes, asking for MP3 so `AVAudioPlayer` can play it.
enum ElevenLabs {
    /// Same model as the rig's `tts.elevenlabs_model`.
    static let model = "eleven_flash_v2_5"
    static let outputFormat = "mp3_44100_128"
    /// Past this the built-in voice answers instead; a late answer is worse than a different voice.
    static let timeout: TimeInterval = 3

    static func request(text: String, voiceID: String, key: String?) -> URLRequest? {
        guard let key, !key.isEmpty, !voiceID.isEmpty,
              let id = voiceID.addingPercentEncoding(withAllowedCharacters: .urlPathAllowed),
              var url = URLComponents(string: "https://api.elevenlabs.io/v1/text-to-speech/\(id)") else { return nil }
        url.queryItems = [URLQueryItem(name: "output_format", value: outputFormat)]
        guard let resolved = url.url else { return nil }
        var request = URLRequest(url: resolved, timeoutInterval: timeout)
        request.httpMethod = "POST"
        request.setValue(key, forHTTPHeaderField: "xi-api-key")
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.setValue("audio/mpeg", forHTTPHeaderField: "Accept")
        request.httpBody = try? JSONSerialization.data(withJSONObject: ["text": text, "model_id": model], options: .sortedKeys)
        return request
    }
}

/// A generic password in the Keychain, this device only.
enum Keychain {
    private static let service = "com.asktheroom.app"

    static func read(account: String) -> String? {
        let query: [String: Any] = [kSecClass as String: kSecClassGenericPassword,
                                    kSecAttrService as String: service,
                                    kSecAttrAccount as String: account,
                                    kSecReturnData as String: true]
        var item: CFTypeRef?
        guard SecItemCopyMatching(query as CFDictionary, &item) == errSecSuccess,
              let data = item as? Data else { return nil }
        return String(data: data, encoding: .utf8)
    }

    static func write(_ value: String?, account: String) {
        let query: [String: Any] = [kSecClass as String: kSecClassGenericPassword,
                                    kSecAttrService as String: service,
                                    kSecAttrAccount as String: account]
        SecItemDelete(query as CFDictionary)
        guard let value, !value.isEmpty else { return }
        var add = query
        add[kSecValueData as String] = Data(value.utf8)
        add[kSecAttrAccessible as String] = kSecAttrAccessibleWhenUnlockedThisDeviceOnly
        SecItemAdd(add as CFDictionary, nil)
    }
}
