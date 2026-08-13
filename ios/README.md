# Pie on iOS (work in progress)

Runs the Pie runtime — wasmtime executing inferlets — inside an iOS app.

- `pie-shim/`: C-ABI staticlib embedding the engine (`pie_ios_run` /
  `pie_ios_run_stream` / `pie_ios_free`), boot path mirrors `pie run`.
  Build with `cargo build --target aarch64-apple-ios-sim` (or
  `aarch64-apple-ios`).
- `demo-app/`: minimal SwiftUI shell + hand-assembled .app bundle for the
  Simulator. See `build-app.sh` (expects a Qwen3 `tokenizer.json` under
  `target/qwen3-tok/` and the helloworld + marketing-tab2-watermark
  inferlets built for `wasm32-wasip2`).
- `voice-app/`: **a voice assistant you hold a conversation with** —
  speech in, model reply spoken back, all on the device. Layered so the
  Pie-facing code, the audio code, and the UI can each be replaced
  independently; see `voice-app/README.md`. Uses the `voice-chat`
  inferlet, which carries the conversation's KV state across turns in a
  named snapshot.

Status: REAL MODEL INFERENCE WORKS in the iOS Simulator — Qwen3-0.6B
Q4_K_M GGUF through the portable (ggml, CPU) driver, streaming tokens
into a SwiftUI chat view via inferlets running under wasmtime Pulley.
Build the shim with --release: ggml at -O0 is unusably slow.

iOS-specific changes so far:
- Pulley engine target (runtime/src/bootstrap.rs)
- file-backed mmap fallback for POSIX shmem (driver/bridge/src/ipc/posix.rs)
- iOS cross-compile support for the portable driver's CMake build
  (server/build.rs: SDK sysroot defines + ios system-libs arm)

Two things worth flagging to maintainers, found while building the
voice app:

1. `Context::take` fails on the portable driver with
   `take: insufficient GPU pages (got 0, need 1)`, so a multi-turn
   session has to resume with `Context::open` and delete the old
   snapshot by hand before re-saving. `save` also refuses to overwrite an
   existing name.
2. `Generator` truncates the stop token instead of appending it, so a
   context saved right after generation ends on an unterminated
   assistant turn. Inferlets that persist a conversation need an explicit
   `Context::seal()` before `save()`, which is easy to miss —
   `demo-persistent-kv` has the same gap.

Next: ggml Metal backend on iOS, physical-device deployment, Android
(llama.cpp Vulkan or ggml), durable-inferlet migration demo.
