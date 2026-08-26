import SwiftUI

/// Composition root.
///
/// The only place where a concrete backend, a concrete microphone, and a
/// concrete synthesizer are named. Everything downstream sees protocols,
/// which is what keeps a Pie upgrade confined to `PieKit/` and an audio
/// change confined to `AudioKit/`.
final class AppComposition: ObservableObject {

    let controller: ConversationController
    /// Exposed so benchmark mode can drive the backend without the
    /// listen/speak loop in the way.
    let backend: ConversationBackend

    private let microphone = MicrophoneInput()
    private let sample = AudioFileInput(resources: AppComposition.sampleResources)

    /// A bundled utterance is optional — the app is fully usable without
    /// one, it just can't be driven hands-off.
    var hasSampleRecording: Bool { sample.url != nil }

    @Published var inputSource: InputSource = .microphone {
        didSet {
            guard oldValue != inputSource else { return }
            controller.useInput(inputSource == .microphone ? microphone : sample)
        }
    }

    /// Turns 2 and 3 are follow-ups on purpose: they are only coherent if
    /// the backend kept the conversation, so pressing "Sample" three
    /// times exercises the KV-snapshot path without anyone speaking.
    static let sampleResources = [
        "sample-question-1",
        "sample-question-2",
        "sample-question-3",
    ]

    init() {
        // Restore the ladder rung chosen on a previous launch before the
        // controller reads the engine description.
        PieRuntimeConfig.restoreSelection()

        let engine = PieEngine()
        backend = engine
        controller = ConversationController(
            backend: engine,
            input: microphone,
            output: SpokenOutput()
        )
    }
}

@main
struct PieVoiceApp: App {

    @StateObject private var composition = AppComposition()

    var body: some Scene {
        WindowGroup {
            VoiceChatView(
                controller: composition.controller,
                inputSource: Binding(
                    get: { composition.inputSource },
                    set: { composition.inputSource = $0 }
                ),
                hasSampleRecording: composition.hasSampleRecording
            )
            .onAppear {
                    if BenchmarkRunner.isEnabled {
                        // Benchmark mode drives the backend directly; the
                        // normal warm-up/listen path would race it.
                        BenchmarkRunner.run(backend: composition.backend)
                    } else {
                        composition.controller.bootstrap()
                    }
                }
        }
    }
}
