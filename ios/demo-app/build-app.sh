#!/bin/bash
# Assemble PieDemo.app for the iOS Simulator: compile the SwiftUI shell,
# link the Rust staticlib (Pie engine + dummy driver), bundle inferlets,
# manifests, and the tokenizer.
set -euo pipefail

# Run from the repo root: bash ios/demo-app/build-app.sh
PIE=.
APPDIR=target/PieDemo.app
SHIMLIB=$PIE/ios/pie-shim/target/aarch64-apple-ios-sim/debug

rm -rf "$APPDIR"
mkdir -p "$APPDIR"

xcrun -sdk iphonesimulator swiftc \
  -target arm64-apple-ios26.0-simulator \
  -parse-as-library \
  "ios/demo-app/main.swift" \
  -L "$SHIMLIB" -lpie_ios_shim \
  -framework Security -framework CoreFoundation -framework SystemConfiguration \
  -lresolv \
  -Xlinker -U -Xlinker _SCDynamicStoreCopyComputerName \
  -o "$APPDIR/PieDemo"

cp "ios/demo-app/Info.plist" "$APPDIR/Info.plist"

# Inferlets + manifests
cp "$PIE/inferlets/helloworld/target/wasm32-wasip2/release/helloworld.wasm" "$APPDIR/"
cp "$PIE/inferlets/helloworld/Pie.toml" "$APPDIR/helloworld-Pie.toml"
cp "$PIE/inferlets/marketing-tab2-watermark/target/wasm32-wasip2/release/marketing_tab2_watermark.wasm" "$APPDIR/"
cp "$PIE/inferlets/marketing-tab2-watermark/Pie.toml" "$APPDIR/tab2-Pie.toml"

# Tokenizer snapshot for the dummy driver's runtime tokenizer
mkdir -p "$APPDIR/qwen3-tok"
cp "target/qwen3-tok/tokenizer.json" "$APPDIR/qwen3-tok/"

# Ad-hoc sign (sufficient for the simulator)
codesign --force --sign - "$APPDIR"

echo "BUILT: $APPDIR"
du -sh "$APPDIR"
