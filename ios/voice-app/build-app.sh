#!/bin/bash
# Assemble PieVoice.app for the iOS Simulator: compile the Swift sources,
# link the Rust staticlib (Pie engine + ggml portable driver), and bundle
# the inferlet, its manifest, the model, and the sample recordings.
#
# Run from the repo root:  bash ios/voice-app/build-app.sh
#
# Prerequisites (same as ios/demo-app):
#   - ios/pie-shim built --release for aarch64-apple-ios-sim
#   - inferlets/voice-chat built for wasm32-wasip2 --release
#   - target/qwen3-gguf/Qwen3-0.6B-Q4_K_M.gguf present
set -euo pipefail

PIE=.
APP_NAME=PieVoice
APPDIR=target/$APP_NAME.app
SRC=ios/voice-app
SHIMLIB=$PIE/ios/pie-shim/target/aarch64-apple-ios-sim/release
# Matches what the shim staticlib was built against; linking an older
# deployment target against it only produces ld warnings.
IOS_TARGET=${IOS_TARGET:-arm64-apple-ios26.0-simulator}

if [ ! -f "$SHIMLIB/libpie_ios_shim.a" ]; then
  echo "missing $SHIMLIB/libpie_ios_shim.a" >&2
  echo "build it with: (cd ios/pie-shim && cargo build --release --target aarch64-apple-ios-sim)" >&2
  exit 1
fi

rm -rf "$APPDIR"
mkdir -p "$APPDIR"

# ── Swift sources ──────────────────────────────────────────────────────
# Order does not matter to swiftc; the grouping mirrors the layering.
SOURCES=(
  "$SRC/Sources/PieKit/PieBridge.swift"
  "$SRC/Sources/PieKit/PieRuntimeConfig.swift"
  "$SRC/Sources/PieKit/PieEngine.swift"
  "$SRC/Sources/AudioKit/VoiceIO.swift"
  "$SRC/Sources/AudioKit/SentenceChunker.swift"
  "$SRC/Sources/AudioKit/SpokenOutput.swift"
  "$SRC/Sources/AudioKit/MicrophoneInput.swift"
  "$SRC/Sources/AudioKit/AudioFileInput.swift"
  "$SRC/Sources/Conversation/ConversationBackend.swift"
  "$SRC/Sources/Conversation/Turn.swift"
  "$SRC/Sources/Conversation/ConversationController.swift"
  "$SRC/Sources/UI/MicOrb.swift"
  "$SRC/Sources/UI/TranscriptView.swift"
  "$SRC/Sources/UI/VoiceChatView.swift"
  "$SRC/Sources/VoiceApp.swift"
)

xcrun -sdk iphonesimulator swiftc \
  -target "$IOS_TARGET" \
  -parse-as-library \
  -O \
  "${SOURCES[@]}" \
  -L "$SHIMLIB" -lpie_ios_shim \
  -framework Security -framework CoreFoundation -framework SystemConfiguration \
  -framework Accelerate -framework AVFoundation -framework Speech \
  -lresolv -lc++ \
  -Xlinker -U -Xlinker _SCDynamicStoreCopyComputerName \
  -o "$APPDIR/$APP_NAME"

cp "$SRC/Info.plist" "$APPDIR/Info.plist"

# ── Inferlet + manifest ────────────────────────────────────────────────
cp "$PIE/inferlets/voice-chat/target/wasm32-wasip2/release/voice_chat.wasm" "$APPDIR/"
cp "$PIE/inferlets/voice-chat/Pie.toml" "$APPDIR/voice-chat-Pie.toml"

# ── Model ──────────────────────────────────────────────────────────────
MODEL=${MODEL:-target/qwen3-gguf/Qwen3-0.6B-Q4_K_M.gguf}
if [ ! -f "$MODEL" ]; then
  echo "missing model $MODEL (override with MODEL=/path/to.gguf)" >&2
  exit 1
fi
mkdir -p "$APPDIR/qwen3-gguf"
cp "$MODEL" "$APPDIR/qwen3-gguf/Qwen3-0.6B-Q4_K_M.gguf"

# ── Sample recordings ──────────────────────────────────────────────────
# Synthesised rather than checked in: a few KB of generated speech beats
# committing audio, and it keeps the utterances editable in one place.
# 16 kHz mono LPCM is what the recogniser wants.
bash "$SRC/make-samples.sh" "$APPDIR"

# Ad-hoc sign (sufficient for the simulator)
codesign --force --sign - "$APPDIR"

echo "BUILT: $APPDIR"
du -sh "$APPDIR"
