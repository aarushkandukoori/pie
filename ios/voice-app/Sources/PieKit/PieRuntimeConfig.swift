import Foundation

/// Everything version-specific about the Pie side of the app.
///
/// This is the upgrade seam. When a new Pie release changes the engine
/// config schema, the model layout, or an inferlet's id/manifest, this
/// file is the only one that should need editing — `AudioKit`, the
/// conversation controller, and the views know nothing about Pie.
enum PieRuntimeConfig {

    /// Which inferlet serves a conversational turn, and where its
    /// artifacts sit inside the app bundle.
    struct Inferlet {
        /// `name@version`, matching the inferlet's Pie.toml.
        let id: String
        /// Bundle-relative wasm filename, without extension.
        let wasmName: String
        /// Bundle-relative manifest filename, without extension.
        let manifestName: String

        var wasmPath: String { "\(Bundle.main.bundlePath)/\(wasmName).wasm" }
        var manifestPath: String { "\(Bundle.main.bundlePath)/\(manifestName).toml" }
    }

    /// The conversational inferlet: one invocation per spoken turn,
    /// carrying KV state across turns in a named snapshot.
    static let voiceChat = Inferlet(
        id: "voice-chat@0.1.0",
        wasmName: "voice_chat",
        manifestName: "voice-chat-Pie"
    )

    /// KV snapshot name the conversation lives under, inside the engine.
    static let sessionName = "ios-voice-session"

    /// Model + driver, as the engine reports them. Shown in the UI so a
    /// screenshot says what actually ran.
    /// One rung of the model ladder. The app ships whichever GGUFs are in
    /// the bundle; `available` reports the ones actually present, so a
    /// build with only the 0.6B behaves exactly as before.
    struct Model: Equatable {
        let label: String       // shown in the UI
        let fileName: String    // inside qwen3-gguf/
        /// Documents first, then the app bundle. Bundling every rung of the
        /// ladder would mean a multi-gigabyte app; pushing the larger models
        /// into the container keeps the shipped app small and lets a
        /// benchmark run swap models without a rebuild.
        var path: String {
            let documents = FileManager.default
                .urls(for: .documentDirectory, in: .userDomainMask)[0]
                .appendingPathComponent("qwen3-gguf/\(fileName)").path
            if FileManager.default.fileExists(atPath: documents) { return documents }
            return "\(Bundle.main.bundlePath)/qwen3-gguf/\(fileName)"
        }
        var isPresent: Bool { FileManager.default.fileExists(atPath: path) }
    }

    /// Ordered smallest first — the ladder used for device benchmarking.
    static let ladder: [Model] = [
        Model(label: "Qwen3-0.6B Q4_K_M", fileName: "Qwen3-0.6B-Q4_K_M.gguf"),
        Model(label: "Qwen3-1.7B Q4_K_M", fileName: "Qwen3-1.7B-Q4_K_M.gguf"),
        Model(label: "Qwen3-4B Q4_K_M",   fileName: "Qwen3-4B-Q4_K_M.gguf"),
        Model(label: "Qwen3-8B Q4_K_M",   fileName: "Qwen3-8B-Q4_K_M.gguf"),
    ]

    static var available: [Model] { ladder.filter(\.isPresent) }

    /// The model the engine boots with. The engine loads weights once per
    /// process, so switching models requires a relaunch — the UI says so
    /// rather than pretending it is live-swappable.
    private(set) static var selected: Model =
        available.first ?? ladder[0]

    /// Persisted across launches so a relaunch comes back on the chosen rung.
    /// The key is launch-argument friendly (no dots): passing
    /// `-PieModel Qwen3-4B-Q4_K_M.gguf` selects a rung for one run, which is
    /// how the benchmark driver pins each model.
    static let modelDefaultsKey = "PieModel"

    static func select(_ model: Model) {
        selected = model
        UserDefaults.standard.set(model.fileName, forKey: modelDefaultsKey)
    }

    /// What was requested, whether or not it could be honoured.
    static var requestedModelFile: String? {
        UserDefaults.standard.string(forKey: modelDefaultsKey)
    }

    static func restoreSelection() {
        guard let saved = requestedModelFile,
              let match = available.first(where: { $0.fileName == saved })
        else { return }
        selected = match
    }

    static var modelDescription: String { selected.label }
    static let driverDescription = "ggml (CPU)"
    static let runtimeDescription = "wasmtime Pulley"

    /// Engine config TOML, written to a temp file for the shim to load.
    ///
    /// Mirrors what `pie run` would read from disk. The port is
    /// irrelevant — the shim rewrites it to 0 and lets the OS pick — but
    /// the config schema requires the section.
    static func writeEngineConfig() throws -> String {
        let toml = """
        [server]
        host = "127.0.0.1"
        port = 8093

        [auth]
        enabled = false

        [runtime]
        allow_fs = true
        allow_network = false

        [[model]]
        name = "default"
        hf_repo = "\(selected.path)"

        [model.driver]
        type = "portable"
        device = ["cpu"]
        """
        let path = NSTemporaryDirectory() + "pie-voice-config.toml"
        try toml.write(toFile: path, atomically: true, encoding: .utf8)
        return path
    }

    /// Generation settings for a spoken turn. Short replies matter more
    /// here than in a text UI: every extra token is another second of
    /// someone waiting to be talked at.
    static let maxTokensPerTurn = 110
    static let temperature = 0.7
    static let topP = 0.95
}
