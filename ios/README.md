# Pie on iOS (work in progress)

Runs the Pie runtime — wasmtime executing inferlets — inside an iOS app.

- `pie-shim/`: C-ABI staticlib embedding the engine (`pie_ios_run` /
  `pie_ios_free`), boot path mirrors `pie run`. Build with
  `cargo build --target aarch64-apple-ios-sim` (or `aarch64-apple-ios`).
- `demo-app/`: minimal SwiftUI shell + hand-assembled .app bundle for the
  Simulator. See `build-app.sh` (expects a Qwen3 `tokenizer.json` under
  `target/qwen3-tok/` and the helloworld + marketing-tab2-watermark
  inferlets built for `wasm32-wasip2`).

Status: REAL MODEL INFERENCE WORKS in the iOS Simulator — Qwen3-0.6B
Q4_K_M GGUF through the portable (ggml, CPU) driver, streaming tokens
into a SwiftUI chat view via the text-completion inferlet running under
wasmtime Pulley. Build the shim with --release: ggml at -O0 is unusably
slow.

iOS-specific changes so far:
- Pulley engine target (runtime/src/bootstrap.rs)
- file-backed mmap fallback for POSIX shmem (driver/bridge/src/ipc/posix.rs)
- iOS cross-compile support for the portable driver's CMake build
  (server/build.rs: SDK sysroot defines + ios system-libs arm)

Next: ggml Metal backend on iOS, physical-device deployment, Android
(llama.cpp Vulkan or ggml), durable-inferlet migration demo.
