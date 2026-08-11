import SwiftUI

// C ABI from libpie_ios_shim.a (ios/pie-shim in the Pie fork).
@_silgen_name("pie_ios_run_stream")
func pie_ios_run_stream(
    _ configPath: UnsafePointer<CChar>,
    _ wasmPath: UnsafePointer<CChar>,
    _ manifestPath: UnsafePointer<CChar>,
    _ inferletId: UnsafePointer<CChar>,
    _ inputJson: UnsafePointer<CChar>,
    _ cb: @convention(c) (UnsafePointer<CChar>?, UnsafeMutableRawPointer?) -> Void,
    _ ctx: UnsafeMutableRawPointer?
) -> UnsafeMutablePointer<CChar>?

@_silgen_name("pie_ios_free")
func pie_ios_free(_ s: UnsafeMutablePointer<CChar>?)

/// Stream sink for the C callback (single generation at a time).
final class Stream: ObservableObject {
    static let shared = Stream()
    @Published var text = ""
    func append(_ s: String) {
        DispatchQueue.main.async { self.text += s }
    }
    func reset() {
        DispatchQueue.main.async { self.text = "" }
    }
}

@main
struct PieDemoApp: App {
    var body: some Scene {
        WindowGroup { ContentView() }
    }
}

struct ContentView: View {
    @ObservedObject private var stream = Stream.shared
    @State private var prompt = "Write a haiku about pie."
    @State private var running = false
    @State private var status = "engine cold — first run boots Pie and loads Qwen3-0.6B"

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("Pie on iOS")
                .font(.largeTitle.bold())
            Text("Qwen3-0.6B Q4_K_M · ggml (CPU) · inferlets on wasmtime Pulley")
                .font(.caption)
                .foregroundStyle(.secondary)

            HStack {
                TextField("prompt", text: $prompt)
                    .textFieldStyle(.roundedBorder)
                    .disabled(running)
                Button("Generate") {
                    run(
                        inferlet: "text-completion@0.2.15",
                        wasm: "text_completion",
                        manifest: "text-completion-Pie",
                        input: #"{"prompt": "\#(jsonEscape(prompt))", "max_tokens": 96}"#
                    )
                }
                .buttonStyle(.borderedProminent)
                .disabled(running || prompt.isEmpty)
            }

            HStack {
                Button("Watermarked haiku") {
                    run(
                        inferlet: "marketing-tab2-watermark@0.1.0",
                        wasm: "marketing_tab2_watermark",
                        manifest: "tab2-Pie",
                        input: #"{"prompt": "\#(jsonEscape(prompt))", "max_tokens": 48}"#
                    )
                }
                .buttonStyle(.bordered)
                .disabled(running)

                Button("helloworld") {
                    run(
                        inferlet: "helloworld@0.1.0",
                        wasm: "helloworld",
                        manifest: "helloworld-Pie",
                        input: "{}"
                    )
                }
                .buttonStyle(.bordered)
                .disabled(running)
            }

            HStack(spacing: 8) {
                if running { ProgressView().controlSize(.small) }
                Text(status)
                    .font(.caption2)
                    .foregroundStyle(.secondary)
            }

            ScrollView {
                Text(stream.text.isEmpty ? "…" : stream.text)
                    .font(.system(size: 14, design: .monospaced))
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .textSelection(.enabled)
                    .padding(8)
            }
            .background(Color(.secondarySystemBackground))
            .clipShape(RoundedRectangle(cornerRadius: 8))
        }
        .padding()
    }

    private func run(inferlet: String, wasm: String, manifest: String, input: String) {
        running = true
        status = "running \(inferlet)…"
        stream.reset()

        DispatchQueue.global(qos: .userInitiated).async {
            let bundle = Bundle.main.bundlePath
            let configPath = writeConfig(bundlePath: bundle)
            let wasmPath = "\(bundle)/\(wasm).wasm"
            let manifestPath = "\(bundle)/\(manifest).toml"

            let cb: @convention(c) (UnsafePointer<CChar>?, UnsafeMutableRawPointer?) -> Void = { chunk, _ in
                guard let chunk else { return }
                Stream.shared.append(String(cString: chunk))
            }

            let started = Date()
            let result: String
            if let ptr = pie_ios_run_stream(configPath, wasmPath, manifestPath, inferlet, input, cb, nil) {
                result = String(cString: ptr)
                pie_ios_free(ptr)
            } else {
                result = "PIE ERROR: null result"
            }
            let elapsed = String(format: "%.1fs", -started.timeIntervalSinceNow)

            DispatchQueue.main.async {
                if stream.text.isEmpty || result.hasPrefix("PIE ERROR") {
                    stream.text += result
                }
                status = "done in \(elapsed) — engine warm"
                running = false
            }
        }
    }

    private func jsonEscape(_ s: String) -> String {
        s.replacingOccurrences(of: "\\", with: "\\\\")
            .replacingOccurrences(of: "\"", with: "\\\"")
            .replacingOccurrences(of: "\n", with: "\\n")
    }

    /// Engine config: real Qwen3 GGUF through the portable (ggml CPU) driver.
    private func writeConfig(bundlePath: String) -> String {
        let config = """
        [server]
        host = "127.0.0.1"
        port = 8093

        [auth]
        enabled = false

        [runtime]
        allow_fs = true
        allow_network = true

        [[model]]
        name = "default"
        hf_repo = "\(bundlePath)/qwen3-gguf/Qwen3-0.6B-Q4_K_M.gguf"

        [model.driver]
        type = "portable"
        device = ["cpu"]
        """
        let path = NSTemporaryDirectory() + "pie-config.toml"
        try? config.write(toFile: path, atomically: true, encoding: .utf8)
        return path
    }
}
