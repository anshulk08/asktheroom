import AVFoundation
import Security

/// Reads answers aloud on the phone. Grok's voice by default; the rig's ElevenLabs voice
/// (`voice/tts.py` on main) or the iPhone's own voice if the helper picks them. Any cloud voice
/// falls back to the iPhone's when there's no key, no internet or no reply in time, like the
/// rig's Piper fallback. Each answer is matched to where the sound is going (`VoiceRoute`).
/// Off by default: the rig already speaks in the room, so this is for away from the table.
@MainActor
final class Speaker {
    static let enabledKey = "readAnswersAloud"
    static let engineKey = "voiceEngine"
    static let grokVoiceKey = "grokVoice"
    static let speedKey = "voiceSpeed"
    static let voiceIDKey = "elevenLabsVoiceID"
    static let shared = Speaker()

    enum Engine: String, CaseIterable, Identifiable {
        case grok, rigVoice, builtIn
        var id: String { rawValue }
        var title: String {
            switch self {
            case .grok: return "Grok"
            case .rigVoice: return "Same as the rig"
            case .builtIn: return "iPhone"
            }
        }
    }

    /// Which voice the last answer used, for helper settings.
    private(set) var lastEngine: Engine?

    private let synthesizer = AVSpeechSynthesizer()
    private var player: AVAudioPlayer?
    private var fetch: Task<Void, Never>?

    private init() {
        // Headphones or a Bluetooth speaker went away mid-answer: stop rather than carry on out
        // loud through the phone, as Apple asks. A new device just gets the next answer.
        NotificationCenter.default.addObserver(forName: AVAudioSession.routeChangeNotification,
                                               object: nil, queue: .main) { note in
            let raw = note.userInfo?[AVAudioSessionRouteChangeReasonKey] as? UInt
            guard raw.flatMap(AVAudioSession.RouteChangeReason.init) == .oldDeviceUnavailable else { return }
            MainActor.assumeIsolated { Speaker.shared.stop() }
        }
    }

    /// Keys live in the Keychain, entered in helper settings; never in the repo.
    static var grokKey: String? {
        get { Keychain.read(account: "xai") }
        set { Keychain.write(newValue, account: "xai") }
    }

    static var apiKey: String? {
        get { Keychain.read(account: "elevenlabs") }
        set { Keychain.write(newValue, account: "elevenlabs") }
    }

    static var engine: Engine {
        UserDefaults.standard.string(forKey: engineKey).flatMap(Engine.init) ?? .grok
    }

    static var grokVoice: String {
        let voice = UserDefaults.standard.string(forKey: grokVoiceKey)?.trimmingCharacters(in: .whitespaces) ?? ""
        return voice.isEmpty ? Grok.defaultVoice : voice
    }

    /// The helper's speed, 0.7 to 1.5, before the route adjusts it.
    static var speed: Double {
        let speed = UserDefaults.standard.object(forKey: speedKey) as? Double ?? 1
        return min(max(speed, Grok.speeds.lowerBound), Grok.speeds.upperBound)
    }

    /// The voice settings as the rig takes them, so its speaker sounds like this phone would.
    static var voiceSettings: VoiceSettings {
        VoiceSettings(voice: .init(e: engine.rawValue, v: grokVoice, s: (speed * 10).rounded() / 10))
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
        // Plays with the silent switch on, like a spoken answer should; never the earpiece.
        try? AVAudioSession.sharedInstance().setCategory(.playback, mode: .spokenAudio, options: .duckOthers)
        try? AVAudioSession.sharedInstance().setActive(true)
        let route = VoiceRoute.current

        let request: URLRequest?
        var playbackRate: Float = 1
        switch Self.engine {
        case .grok:
            request = Grok.request(text: text, voice: Self.grokVoice, speed: Self.speed, route: route, key: Self.grokKey)
        case .rigVoice:
            request = ElevenLabs.request(text: text, voiceID: Self.voiceID, key: Self.apiKey)
            // ElevenLabs sounds like the rig at its own pace; stretch it for the route instead.
            playbackRate = Float(route.speed(Self.speed))
        case .builtIn:
            request = nil
        }
        guard let request else {
            speakBuiltIn(text, route: route)
            return
        }
        let engine = Self.engine
        fetch = Task { [weak self] in
            let audio = try? await URLSession.shared.data(for: request)
            guard let self, !Task.isCancelled else { return }
            if let (data, response) = audio, (response as? HTTPURLResponse)?.statusCode == 200,
               let player = try? AVAudioPlayer(data: data) {
                player.enableRate = playbackRate != 1
                player.rate = playbackRate
                if player.play() {
                    self.player = player
                    self.lastEngine = engine
                    return
                }
            }
            // No network, a bad key or voice: say it anyway.
            self.speakBuiltIn(text, route: route)
        }
    }

    func stop() {
        fetch?.cancel()
        player?.stop()
        synthesizer.stopSpeaking(at: .immediate)
    }

    private func speakBuiltIn(_ text: String, route: VoiceRoute) {
        let utterance = AVSpeechUtterance(string: text)
        let rate = AVSpeechUtteranceDefaultSpeechRate * 0.9 * Float(route.speed(Self.speed))
        utterance.rate = min(max(rate, AVSpeechUtteranceMinimumSpeechRate), AVSpeechUtteranceMaximumSpeechRate)
        synthesizer.speak(utterance)
        lastEngine = .builtIn
    }
}

/// Where the sound is going, and how to speak there. Read fresh for every answer.
struct VoiceRoute: Equatable {
    enum Kind: Equatable {
        /// The phone's own speaker, at arm's length.
        case phone
        /// Headphones, AirPods, Bluetooth music: close to the ear, full quality.
        case ears
        /// A Bluetooth headset on its call profile, which carries 16 kHz at best.
        case headset
        /// AirPlay, a TV, the car, speakers on a cable: heard across a room.
        case room
    }

    let kind: Kind
    /// The device's own name ("AirPods Pro"), for helper settings.
    var name: String

    /// Slower where the sound has further to travel.
    var speedFactor: Double {
        switch kind {
        case .phone: return 0.92
        case .ears, .headset: return 1
        case .room: return 0.9
        }
    }

    /// No more quality than the output can carry.
    var sampleRate: Int {
        switch kind {
        case .phone: return 24_000
        case .ears: return 44_100
        case .headset: return 16_000
        case .room: return 48_000
        }
    }

    var bitRate: Int {
        switch kind {
        case .phone: return 64_000
        case .ears: return 128_000
        case .headset: return 32_000
        case .room: return 192_000
        }
    }

    /// The helper's speed adjusted for this output, kept in the range Grok accepts.
    func speed(_ chosen: Double) -> Double {
        let speed = (chosen * speedFactor * 100).rounded() / 100
        return min(max(speed, Grok.speeds.lowerBound), Grok.speeds.upperBound)
    }

    var summary: String {
        switch kind {
        case .phone: return "iPhone speaker: a little slower"
        case .ears: return "\(name): full quality"
        case .headset: return "\(name): call quality"
        case .room: return "\(name): slower, for a room"
        }
    }

    static func kind(of port: AVAudioSession.Port) -> Kind {
        switch port {
        case .headphones, .bluetoothA2DP, .bluetoothLE: return .ears
        case .bluetoothHFP: return .headset
        case .airPlay, .HDMI, .carAudio, .lineOut, .usbAudio, .displayPort: return .room
        default: return .phone
        }
    }

    static var current: VoiceRoute {
        guard let output = AVAudioSession.sharedInstance().currentRoute.outputs.first else {
            return VoiceRoute(kind: .phone, name: "iPhone")
        }
        return VoiceRoute(kind: kind(of: output.portType), name: output.portName)
    }
}

/// xAI's text-to-speech: one request per answer, MP3 back, which `AVAudioPlayer` plays as is.
/// Answers are a sentence or two, so streaming wouldn't start them noticeably sooner.
enum Grok {
    static let endpoint = URL(string: "https://api.x.ai/v1/tts")!
    static let voicesEndpoint = URL(string: "https://api.x.ai/v1/tts/voices")!
    static let defaultVoice = "eve"
    static let speeds = 0.7...1.5
    /// Same as the rig's voice: a late answer is worse than a different voice.
    static let timeout = ElevenLabs.timeout

    struct Voice: Identifiable, Hashable, Decodable {
        let id: String
        let name: String

        enum CodingKeys: String, CodingKey { case id = "voice_id", name }

        init(id: String, name: String) {
            self.id = id
            self.name = name
        }

        init(from decoder: Decoder) throws {
            let c = try decoder.container(keyedBy: CodingKeys.self)
            id = try c.decode(String.self, forKey: .id)
            name = try c.decodeIfPresent(String.self, forKey: .name) ?? id.capitalized
        }
    }

    /// Shown until the helper's key fetches the list; all 28 from `GET /v1/tts/voices` on Sep 26 2026.
    static let knownVoices = ["altair", "ara", "atlas", "aurora", "carina", "castor", "celeste", "cosmo", "eve",
                              "helios", "helix", "iris", "kepler", "leo", "liora", "lumen", "luna", "lux", "naksh",
                              "orion", "perseus", "rex", "rigel", "sal", "sirius", "ursa", "zagan", "zenith"]
        .map { Voice(id: $0, name: $0.capitalized) }

    static func request(text: String, voice: String, speed: Double, route: VoiceRoute, key: String?) -> URLRequest? {
        guard let key, !key.isEmpty else { return nil }
        var request = URLRequest(url: endpoint, timeoutInterval: timeout)
        request.httpMethod = "POST"
        request.setValue("Bearer \(key)", forHTTPHeaderField: "Authorization")
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        let body: [String: Any] = [
            "text": text,
            "voice_id": voice.isEmpty ? defaultVoice : voice,
            "language": "en",
            "speed": route.speed(speed),
            "output_format": ["codec": "mp3", "sample_rate": route.sampleRate, "bit_rate": route.bitRate],
        ]
        request.httpBody = try? JSONSerialization.data(withJSONObject: body, options: .sortedKeys)
        return request
    }

    static func voicesRequest(key: String?) -> URLRequest? {
        guard let key, !key.isEmpty else { return nil }
        var request = URLRequest(url: voicesEndpoint, timeoutInterval: 10)
        request.setValue("Bearer \(key)", forHTTPHeaderField: "Authorization")
        return request
    }

    static func voices(from data: Data) -> [Voice] {
        struct List: Decodable { let voices: [Voice] }
        return (try? JSONDecoder().decode(List.self, from: data).voices) ?? []
    }

    /// Every voice the key can use, or the known ones if the list can't be fetched.
    static func fetchVoices(key: String?) async -> [Voice] {
        guard let request = voicesRequest(key: key),
              let (data, response) = try? await URLSession.shared.data(for: request),
              (response as? HTTPURLResponse)?.statusCode == 200 else { return knownVoices }
        let voices = voices(from: data)
        return voices.isEmpty ? knownVoices : voices.sorted { $0.name < $1.name }
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
