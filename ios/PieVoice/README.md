# PieVoice on a physical iPhone

Two ways to get the app onto a device. Both end with the full app — engine,
model, and inferlet run on the phone; nothing needs a network.

## A. Build and run with Xcode (recommended — 10 minutes)

Requires: Xcode 16+, a free Apple ID, an iPhone on iOS 17+.

```bash
# from the repo root
rustup target add aarch64-apple-ios wasm32-wasip2
(cd ios/pie-shim && cargo build --release --target aarch64-apple-ios)
(cd inferlets/voice-chat && cargo build --release --target wasm32-wasip2)
# put the model at target/qwen3-gguf/Qwen3-0.6B-Q4_K_M.gguf, then:
cp inferlets/voice-chat/target/wasm32-wasip2/release/voice_chat.wasm ios/PieVoice/Resources/
cp inferlets/voice-chat/Pie.toml ios/PieVoice/Resources/voice-chat-Pie.toml
mkdir -p ios/PieVoice/Resources/qwen3-gguf
cp target/qwen3-gguf/Qwen3-0.6B-Q4_K_M.gguf ios/PieVoice/Resources/qwen3-gguf/
bash ios/voice-app/make-samples.sh ios/PieVoice/Resources
brew install xcodegen && (cd ios/PieVoice && xcodegen)
open ios/PieVoice/PieVoice.xcodeproj
```

In Xcode: Signing & Capabilities → select your team (a free Apple ID works;
Xcode creates the certificate), plug in the iPhone, press Run. With a free
Apple ID the install expires after 7 days — re-run to refresh.

## B. Sideload the prebuilt .ipa

An unsigned `PieVoice.ipa` is attached to the releases of
[aarushkandukoori/pie-ios](https://github.com/aarushkandukoori/pie-ios/releases).
Install it with [AltStore](https://altstore.io) or
[Sideloadly](https://sideloadly.io), which re-sign it with your own Apple ID
on install. Same 7-day refresh rule for free accounts.

## Notes for device runs

- The simulator build cannot use on-device speech recognition (missing
  assets); a real iPhone can — the header badge should show
  "on-device speech" once the recognizer assets download.
- Decode speed on the CPU driver will differ from the published
  Simulator-on-Mac numbers in either direction; the Metal driver is the
  performance milestone.
- First launch pays the full model load (~400 MB from flash) — expect a
  noticeably longer warm-up than a relaunch.
- TestFlight distribution needs an Apple Developer Program membership and is
  planned once the Metal driver lands; the project page tracks it.
