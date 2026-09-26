import AVFoundation
import Observation
import Speech

/// Hold-to-talk dictation (spec section 5): on-device `SFSpeechRecognizer` only, live partial
/// text, and it finishes on release or after 1.2 s of silence. Without on-device recognition
/// or permission the mic is hidden and typing still works.
@MainActor
@Observable
final class Dictation {
    static let silenceTimeout: Duration = .milliseconds(1200)

    private(set) var isListening = false
    private(set) var denied = false

    private let recognizer = SFSpeechRecognizer(locale: Locale(identifier: "en-US"))
    private let engine = AVAudioEngine()
    private var request: SFSpeechAudioBufferRecognitionRequest?
    private var task: SFSpeechRecognitionTask?
    private var silenceTask: Task<Void, Never>?
    private var text = ""
    private var releasedEarly = false
    private var onPartial: (String) -> Void = { _ in }
    private var onFinish: (String) -> Void = { _ in }

    var isAvailable: Bool {
        guard !denied, let recognizer else { return false }
        return recognizer.supportsOnDeviceRecognition
    }

    func start(onPartial: @escaping (String) -> Void, onFinish: @escaping (String) -> Void) async {
        guard isAvailable, !isListening else { return }
        releasedEarly = false
        self.onPartial = onPartial
        self.onFinish = onFinish

        guard await Self.authorize() else {
            denied = true
            return
        }
        // Let go while the permission prompts were up: don't start listening.
        guard !releasedEarly else { return }

        do {
            try begin()
        } catch {
            teardown()
        }
    }

    /// Mic released: send what was heard.
    func stop() {
        guard isListening else {
            releasedEarly = true
            return
        }
        finish()
    }

    private func begin() throws {
        let session = AVAudioSession.sharedInstance()
        try session.setCategory(.record, mode: .measurement, options: .duckOthers)
        try session.setActive(true, options: .notifyOthersOnDeactivation)

        let request = SFSpeechAudioBufferRecognitionRequest()
        request.shouldReportPartialResults = true
        request.requiresOnDeviceRecognition = true
        request.taskHint = .search
        self.request = request

        Self.feed(engine.inputNode, into: request)
        engine.prepare()
        try engine.start()

        text = ""
        isListening = true
        task = recognizer?.recognitionTask(with: request, resultHandler: Self.resultHandler(for: self))
    }

    private func heard(_ heard: String?, isFinal: Bool) {
        guard isListening else { return }
        if let heard {
            text = heard
            onPartial(heard)
            restartSilenceTimer()
        }
        if isFinal { finish() }
    }

    // The audio tap and recognition results arrive on background queues, so these closures
    // are built outside the main actor and hop back to it.

    private nonisolated static func feed(_ input: AVAudioInputNode, into request: SFSpeechAudioBufferRecognitionRequest) {
        input.installTap(onBus: 0, bufferSize: 1024, format: input.outputFormat(forBus: 0)) { buffer, _ in
            request.append(buffer)
        }
    }

    private nonisolated static func resultHandler(for dictation: Dictation) -> (SFSpeechRecognitionResult?, Error?) -> Void {
        { [weak dictation] result, error in
            let text = result?.bestTranscription.formattedString
            let done = (result?.isFinal ?? false) || error != nil
            Task { @MainActor in dictation?.heard(text, isFinal: done) }
        }
    }

    private func restartSilenceTimer() {
        silenceTask?.cancel()
        silenceTask = Task { [weak self] in
            try? await Task.sleep(for: Self.silenceTimeout)
            guard !Task.isCancelled else { return }
            self?.finish()
        }
    }

    private func finish() {
        guard isListening else { return }
        let heard = text
        teardown()
        onFinish(heard)
    }

    private func teardown() {
        silenceTask?.cancel()
        if engine.isRunning { engine.stop() }
        engine.inputNode.removeTap(onBus: 0)
        request?.endAudio()
        task?.cancel()
        request = nil
        task = nil
        isListening = false
        try? AVAudioSession.sharedInstance().setActive(false, options: .notifyOthersOnDeactivation)
    }

    private static func authorize() async -> Bool {
        let speech = await withCheckedContinuation { continuation in
            SFSpeechRecognizer.requestAuthorization { continuation.resume(returning: $0 == .authorized) }
        }
        guard speech else { return false }
        return await AVAudioApplication.requestRecordPermission()
    }
}
