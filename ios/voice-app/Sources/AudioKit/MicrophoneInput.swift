import AVFoundation
import Foundation
import Speech

/// `VoiceInput` from the microphone, transcribed by the Speech framework.
///
/// On-device recognition is requested whenever the device supports it, so
/// that in the intended configuration nothing spoken to this app leaves
/// the phone — the audio is transcribed locally and the transcript is
/// answered by a model that is also local. When on-device recognition is
/// unavailable the class still works, but it reports that fact rather
/// than quietly shipping audio to a server.
final class MicrophoneInput: NSObject, VoiceInput {

    weak var delegate: VoiceInputDelegate?
    private(set) var isListening = false

    /// Silence after speech that ends an utterance. Long enough to think
    /// mid-sentence, short enough that the app doesn't feel deaf.
    private let endOfSpeechSilence: TimeInterval = 1.3

    private let recognizer = SFSpeechRecognizer(locale: Locale(identifier: "en-US"))
    private let audioEngine = AVAudioEngine()

    private var request: SFSpeechAudioBufferRecognitionRequest?
    private var task: SFSpeechRecognitionTask?
    private var silenceTimer: Timer?

    private var latestTranscript = ""
    private var didFinalize = false
    private var usesOnDeviceRecognition = false

    // MARK: - VoiceInput

    func prepare() async -> VoiceInputAvailability {
        guard let recognizer else {
            return .unavailable("No speech recogniser for this locale")
        }

        let speechStatus = await withCheckedContinuation { continuation in
            SFSpeechRecognizer.requestAuthorization { continuation.resume(returning: $0) }
        }
        guard speechStatus == .authorized else {
            return .denied("Speech recognition permission was declined")
        }

        let micGranted = await withCheckedContinuation { continuation in
            AVAudioApplication.requestRecordPermission { continuation.resume(returning: $0) }
        }
        guard micGranted else {
            return .denied("Microphone permission was declined")
        }

        guard recognizer.isAvailable else {
            return .unavailable("Speech recogniser is not available right now")
        }

        usesOnDeviceRecognition = OnDeviceRecognition.isAvailable(on: recognizer)
        return .ready(onDevice: usesOnDeviceRecognition)
    }

    func start() throws {
        guard !isListening else { return }
        guard let recognizer, recognizer.isAvailable else {
            throw VoiceInputError.recogniserUnavailable
        }

        try AudioSessionCoordinator.configure()

        let request = SFSpeechAudioBufferRecognitionRequest()
        request.shouldReportPartialResults = true
        request.requiresOnDeviceRecognition = usesOnDeviceRecognition
        request.taskHint = .dictation
        self.request = request

        latestTranscript = ""
        didFinalize = false

        let input = audioEngine.inputNode
        let format = input.outputFormat(forBus: 0)
        input.installTap(onBus: 0, bufferSize: 1024, format: format) { [weak self] buffer, _ in
            self?.request?.append(buffer)
            self?.reportLevel(of: buffer)
        }

        audioEngine.prepare()
        try audioEngine.start()
        isListening = true

        task = recognizer.recognitionTask(with: request) { [weak self] result, error in
            guard let self else { return }

            if let result {
                let text = result.bestTranscription.formattedString
                if !text.isEmpty {
                    self.latestTranscript = text
                    DispatchQueue.main.async { self.delegate?.voiceInputDidUpdatePartial(text) }
                    self.restartSilenceTimer()
                }
                if result.isFinal {
                    self.finishUtterance()
                    return
                }
            }

            if let error {
                // A recogniser that times out after the user has already
                // said something is not a failure — it is the end of the
                // utterance, and the transcript stands.
                if !self.latestTranscript.isEmpty {
                    self.finishUtterance()
                    return
                }
                // Live audio can't be replayed, so unlike the file input
                // this attempt is lost — but the local path is marked
                // unusable so the next one goes over the network.
                if self.usesOnDeviceRecognition {
                    OnDeviceRecognition.markUnusable()
                    self.usesOnDeviceRecognition = false
                    DispatchQueue.main.async {
                        self.delegate?.voiceInputDidChangeAvailability(.ready(onDevice: false))
                    }
                }
                DispatchQueue.main.async { self.delegate?.voiceInputDidFail(error) }
                self.teardown()
            }
        }
    }

    func stop() {
        guard isListening else { return }
        silenceTimer?.invalidate()
        silenceTimer = nil
        teardownAudio()
        request?.endAudio()

        // The final result usually follows within a few hundred
        // milliseconds. If it doesn't, the last partial is what was said.
        DispatchQueue.main.asyncAfter(deadline: .now() + 1.0) { [weak self] in
            self?.finishUtterance()
        }
    }

    // MARK: - Internals

    private func restartSilenceTimer() {
        DispatchQueue.main.async { [weak self] in
            guard let self else { return }
            self.silenceTimer?.invalidate()
            self.silenceTimer = Timer.scheduledTimer(
                withTimeInterval: self.endOfSpeechSilence,
                repeats: false
            ) { [weak self] _ in
                self?.stop()
            }
        }
    }

    private func finishUtterance() {
        guard !didFinalize else { return }
        didFinalize = true

        let transcript = latestTranscript.trimmingCharacters(in: .whitespacesAndNewlines)
        teardown()
        DispatchQueue.main.async { [weak self] in
            self?.delegate?.voiceInputDidFinalize(transcript)
        }
    }

    private func teardownAudio() {
        if audioEngine.isRunning {
            audioEngine.stop()
        }
        audioEngine.inputNode.removeTap(onBus: 0)
        isListening = false
    }

    private func teardown() {
        silenceTimer?.invalidate()
        silenceTimer = nil
        teardownAudio()
        task?.cancel()
        task = nil
        request = nil
    }

    /// RMS of the buffer, mapped onto 0…1 for the meter ring.
    private func reportLevel(of buffer: AVAudioPCMBuffer) {
        guard let channel = buffer.floatChannelData?[0] else { return }
        let count = Int(buffer.frameLength)
        guard count > 0 else { return }

        var sum: Float = 0
        for index in 0..<count {
            let sample = channel[index]
            sum += sample * sample
        }
        let rms = sqrt(sum / Float(count))
        // -50 dBFS is a quiet room, 0 dBFS is clipping.
        let decibels = 20 * log10(max(rms, 1e-7))
        let level = max(0, min(1, (decibels + 50) / 50))

        DispatchQueue.main.async { [weak self] in
            self?.delegate?.voiceInputDidUpdateLevel(level)
        }
    }
}

enum VoiceInputError: LocalizedError {
    case recogniserUnavailable
    case missingAudioFile(String)

    var errorDescription: String? {
        switch self {
        case .recogniserUnavailable:
            return "The speech recogniser is unavailable"
        case .missingAudioFile(let name):
            return "Bundled audio file not found: \(name)"
        }
    }
}
