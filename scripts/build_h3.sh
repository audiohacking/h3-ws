#!/bin/bash
# Build the native h3.c Metal binary into third_party/h3.c/h3
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SRC="$ROOT/third_party/h3.c"
if [[ ! -f "$SRC/Makefile" ]]; then
  echo "h3.c sources missing at $SRC (vendored tree expected)."
  exit 1
fi
if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "h3.c targets Apple Silicon macOS."
  exit 1
fi
# Metal4 / MTLMathModeSafe / MPSGraph SDPA need the macOS 26+ SDK (Xcode 26+).
SDKROOT="$(xcrun --sdk macosx --show-sdk-path 2>/dev/null || true)"
METAL_HDR="${SDKROOT}/System/Library/Frameworks/Metal.framework/Headers/MTLDevice.h"
if [[ -z "$SDKROOT" || ! -f "$METAL_HDR" ]] || ! grep -q 'MTLGPUFamilyMetal4' "$METAL_HDR"; then
  echo "h3.c needs the macOS 26+ Metal SDK (Xcode 26+)."
  echo "  SDKROOT=${SDKROOT:-unset}"
  xcodebuild -version 2>/dev/null || true
  exit 1
fi
jobs="$(sysctl -n hw.ncpu 2>/dev/null || echo 8)"
PATCH="$ROOT/patches/h3-prefer-H3_AV.patch"
if [[ -f "$PATCH" ]] && ! grep -q 'getenv("H3_AV")' "$SRC/h3_ffmpeg.c"; then
  echo "Applying H3_AV media-shim patch…"
  patch -p1 -d "$SRC" < "$PATCH"
fi
LORA_DIR="$ROOT/patches/h3-lora"
if [[ -f "$LORA_DIR/h3_lora.c" ]]; then
  cp -f "$LORA_DIR/h3_lora.c" "$LORA_DIR/h3_lora.h" "$SRC/"
  if ! grep -q 'h3_lora.h' "$SRC/h3_weights.c"; then
    echo "Applying DiT LoRA fuse patch…"
    patch -p1 -d "$SRC" < "$LORA_DIR/engine.patch"
  fi
fi
echo "Building h3.c with make -j${jobs} …"
make -C "$SRC" -j"$jobs"
if [[ -x "$SRC/h3" ]]; then
  echo "OK: $SRC/h3"
  "$SRC/h3" --help | head -n 20 || true
else
  echo "Build finished but $SRC/h3 is missing."
  exit 1
fi
