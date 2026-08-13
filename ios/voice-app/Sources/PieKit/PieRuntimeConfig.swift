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
    static let modelDescription = "Qwen3-0.6B Q4_K_M"
    static let driverDescription = "ggml (CPU)"
    static let runtimeDescription = "wasmtime Pulley"

    /// Engine config TOML, written to a temp file for the shim to load.
    ///
    /// Mirrors what `pie run` would read from disk. The port is
    /// irrelevant — the shim rewrites it to 0 and lets the OS pick — but
    /// the config schema requires the section.
    static func writeEngineConfig() throws -> String {
        let bundle = Bundle.main.bundlePath
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
        hf_repo = "\(bundle)/qwen3-gguf/Qwen3-0.6B-Q4_K_M.gguf"

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
