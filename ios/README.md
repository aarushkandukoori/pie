# Pie on iOS (work in progress)

Runs the Pie runtime — wasmtime executing inferlets — inside an iOS app.

- `pie-shim/`: C-ABI staticlib embedding the engine (`pie_ios_run` /
  `pie_ios_free`), boot path mirrors `pie run`. Build with
  `cargo build --target aarch64-apple-ios-sim` (or `aarch64-apple-ios`).
- `demo-app/`: minimal SwiftUI shell + hand-assembled .app bundle for the
  Simulator. See `build-app.sh` (expects a Qwen3 `tokenizer.json` under
  `target/qwen3-tok/` and the helloworld + marketing-tab2-watermark
  inferlets built for `wasm32-wasip2`).

Status: engine + WebSocket control plane + wasm inferlets (Pulley
interpreter, no JIT) + dummy driver run end-to-end in the iOS Simulator.
iOS-specific changes so far: Pulley engine target (runtime/src/bootstrap.rs),
file-backed mmap fallback for POSIX shmem (driver/bridge/src/ipc/posix.rs).
Next: ggml-based driver (CPU, then Metal) for real model inference.
