import SwiftUI

// C ABI from libpie_ios_shim.a (ios/pie-shim in the Pie fork).
@_silgen_name("pie_ios_run")
func pie_ios_run(
    _ configPath: UnsafePointer<CChar>,
    _ wasmPath: UnsafePointer<CChar>,
    _ manifestPath: UnsafePointer<CChar>,
    _ inferletId: UnsafePointer<CChar>,
    _ inputJson: UnsafePointer<CChar>
) -> UnsafeMutablePointer<CChar>?

@_silgen_name("pie_ios_free")
func pie_ios_free(_ s: UnsafeMutablePointer<CChar>?)

@main
struct PieDemoApp: App {
    var body: some Scene {
        WindowGroup { ContentView() }
    }
}

struct ContentView: View {
    @State private var output = "Pie engine idle.\nTap a button to boot the engine and run an inferlet."
    @State private var running = false

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("Pie on iOS")
                .font(.largeTitle.bold())
            Text("programmable LLM serving · wasmtime (Pulley) · dummy driver")
                .font(.caption)
                .foregroundStyle(.secondary)

            HStack {
                Button("Run helloworld") {
                    run(
                        inferlet: "helloworld@0.1.0",
                        wasm: "helloworld",
                        manifest: "helloworld-Pie",
                        input: "{}"
                    )
                }
                .buttonStyle(.borderedProminent)
                .disabled(running)

                Button("Run generation") {
                    run(
                        inferlet: "marketing-tab2-watermark@0.1.0",
                        wasm: "marketing_tab2_watermark",
                        manifest: "tab2-Pie",
                        input: #"{"prompt": "Write a haiku about Rust.", "max_tokens": 16}"#
                    )
                }
                .buttonStyle(.bordered)
                .disabled(running)
            }

            if running {
                ProgressView("engine running…")
                    .font(.caption)
            }

            ScrollView {
                Text(output)
                    .font(.system(size: 11, design: .monospaced))
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .textSelection(.enabled)
            }
            .background(Color(.secondarySystemBackground))
            .clipShape(RoundedRectangle(cornerRadius: 8))
        }
        .padding()
    }

    private func run(inferlet: String, wasm: String, manifest: String, input: String) {
        running = true
        output += "\n\n=== \(inferlet) ===\n"

        DispatchQueue.global(qos: .userInitiated).async {
            let bundle = Bundle.main.bundlePath
            let configPath = writeConfig(bundlePath: bundle)
            let wasmPath = "\(bundle)/\(wasm).wasm"
            let manifestPath = "\(bundle)/\(manifest).toml"

            let result: String
            if let ptr = pie_ios_run(configPath, wasmPath, manifestPath, inferlet, input) {
                result = String(cString: ptr)
                pie_ios_free(ptr)
            } else {
                result = "PIE ERROR: null result"
            }

            DispatchQueue.main.async {
                output += result
                running = false
            }
        }
    }

    /// Write the engine config into tmp with absolute sandbox paths.
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
        hf_repo = "\(bundlePath)/qwen3-tok"

        [model.driver]
        type = "dummy"
        device = ["cpu"]

        [model.driver.options]
        vocab_size = 151936
        arch_name = "qwen3"
        """
        let path = NSTemporaryDirectory() + "pie-config.toml"
        try? config.write(toFile: path, atomically: true, encoding: .utf8)
        return path
    }
}
