import SwiftUI

/// Where utterances come from. Both cases are `VoiceInput`
/// implementations; picking between them at runtime is the modularity
/// claim made visible in the UI.
enum InputSource: String, CaseIterable, Identifiable {
    case microphone = "Mic"
    case sample = "Sample"

    var id: String { rawValue }
}

struct VoiceChatView: View {

    @ObservedObject var controller: ConversationController
    @Binding var inputSource: InputSource
    let hasSampleRecording: Bool

    @State private var typed = ""
    @State private var showsKeyboardEntry = false

    var body: some View {
        VStack(spacing: 0) {
            header
            Divider()

            TranscriptView(
                turns: controller.turns,
                partialTranscript: controller.partialTranscript,
                isListening: controller.state == .listening
            )

            Divider()
            footer
        }
        .background(Color(.systemBackground))
    }

    // MARK: - Header

    private var header: some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack(alignment: .firstTextBaseline) {
                Text("Pie Voice")
                    .font(.title2.bold())
                Spacer()
                privacyBadge
            }
            Text(controller.engineDescription)
                .font(.caption2)
                .foregroundStyle(.secondary)
        }
        .padding(.horizontal, 16)
        .padding(.top, 8)
        .padding(.bottom, 10)
    }

    /// States plainly whether the transcription is local. If it isn't,
    /// saying so matters more than the badge looking tidy.
    private var privacyBadge: some View {
        Group {
            switch controller.availability {
            case .ready(let onDevice):
                Label(
                    onDevice ? "on-device speech" : "cloud speech",
                    systemImage: onDevice ? "lock.fill" : "cloud"
                )
                .foregroundStyle(onDevice ? Color.green : Color.orange)
            case .denied, .unavailable:
                Label("speech unavailable", systemImage: "exclamationmark.triangle")
                    .foregroundStyle(.orange)
            case .none:
                Label("checking…", systemImage: "hourglass")
                    .foregroundStyle(.secondary)
            }
        }
        .font(.caption2)
        .labelStyle(.titleAndIcon)
    }

    // MARK: - Footer

    private var footer: some View {
        VStack(spacing: 12) {
            statusLine

            MicOrb(
                state: controller.state,
                level: controller.inputLevel,
                action: controller.toggleListening
            )

            controls

            if showsKeyboardEntry {
                keyboardEntry
            }
        }
        .padding(.horizontal, 16)
        .padding(.top, 12)
        .padding(.bottom, 8)
    }

    private var statusLine: some View {
        Group {
            switch controller.state {
            case .cold:
                Label("starting the engine and loading the model…", systemImage: "hourglass")
            case .idle:
                Text(controller.turns.isEmpty ? "ready" : "ready — ask a follow-up")
            case .listening:
                Text("listening…")
            case .thinking:
                Text("generating…")
            case .speaking:
                Text("speaking — tap to interrupt")
            case .failed(let message):
                Label(message, systemImage: "exclamationmark.triangle")
                    .foregroundStyle(.orange)
            }
        }
        .font(.caption)
        .foregroundStyle(.secondary)
        .multilineTextAlignment(.center)
        .frame(maxWidth: .infinity)
    }

    private var controls: some View {
        HStack(spacing: 10) {
            // A button-styled toggle rather than a switch: at phone width
            // a switch plus its label crowds the row off the screen.
            Toggle(isOn: $controller.handsFree) {
                Label("hands-free", systemImage: "infinity")
                    .font(.caption)
            }
            .toggleStyle(.button)

            Spacer(minLength: 0)

            if hasSampleRecording {
                Picker("Input", selection: $inputSource) {
                    ForEach(InputSource.allCases) { source in
                        Text(source.rawValue).tag(source)
                    }
                }
                .pickerStyle(.segmented)
                .frame(width: 118)
                .disabled(controller.state.isBusy)
            }

            Button {
                showsKeyboardEntry.toggle()
            } label: {
                Image(systemName: "keyboard")
            }
            .accessibilityLabel("Type instead of speaking")

            Button {
                controller.startOver()
            } label: {
                Image(systemName: "arrow.counterclockwise")
            }
            .disabled(controller.turns.isEmpty)
            .accessibilityLabel("Start a new conversation")
        }
        .buttonStyle(.bordered)
        .controlSize(.small)
    }

    /// Same pipeline, keyboard instead of microphone — the fallback when
    /// speech recognition is unavailable, and the way the app is driven
    /// in an automated run.
    private var keyboardEntry: some View {
        HStack(spacing: 8) {
            TextField("type an utterance", text: $typed)
                .textFieldStyle(.roundedBorder)
                .submitLabel(.send)
                .onSubmit(send)
                .disabled(controller.state.isBusy || controller.state == .cold)

            Button("Send", action: send)
                .buttonStyle(.borderedProminent)
                .controlSize(.small)
                .disabled(typed.isEmpty || controller.state.isBusy || controller.state == .cold)
        }
    }

    private func send() {
        let text = typed
        typed = ""
        controller.submit(typed: text)
    }
}
