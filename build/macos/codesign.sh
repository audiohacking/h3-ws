#!/bin/bash
# Code signing script for H3-WS macOS application (ported from AceForge).
# Default identity "-" = ad-hoc signing (no Apple Developer ID required).
# Prevents the "app is damaged" Gatekeeper quarantine failure.

set -euo pipefail

APP_PATH="${1:-dist/H3-WS.app}"
SIGNING_IDENTITY="${MACOS_SIGNING_IDENTITY:--}"
ENTITLEMENTS_PATH="build/macos/entitlements.plist"

echo "=================================================="
echo "H3-WS macOS Code Signing"
echo "=================================================="
echo "App path: $APP_PATH"
echo "Signing identity: $SIGNING_IDENTITY"
echo "Entitlements: $ENTITLEMENTS_PATH"
echo ""

if [ ! -d "$APP_PATH" ]; then
    echo "Error: App bundle not found at $APP_PATH"
    exit 1
fi

if [ ! -f "$ENTITLEMENTS_PATH" ]; then
    echo "Error: Entitlements file not found at $ENTITLEMENTS_PATH"
    exit 1
fi

sign_binary() {
    local target="$1"
    echo "Signing: $target"
    local cmd
    if [ "$SIGNING_IDENTITY" = "-" ]; then
        cmd=(
            xcrun codesign
            --sign "$SIGNING_IDENTITY"
            --force
            --options runtime
            --entitlements "$ENTITLEMENTS_PATH"
            --deep
            "$target"
        )
    else
        cmd=(
            xcrun codesign
            --sign "$SIGNING_IDENTITY"
            --force
            --options runtime
            --entitlements "$ENTITLEMENTS_PATH"
            --deep
            --timestamp
            "$target"
        )
    fi
    if "${cmd[@]}"; then
        echo "OK: $target"
        return 0
    fi
    echo "FAIL: $target"
    return 1
}

echo "Step 1: Signing frameworks and libraries..."
find "$APP_PATH/Contents" -type f \( -name "*.dylib" -o -name "*.so" \) -print0 | while IFS= read -r -d '' lib; do
    sign_binary "$lib" || true
done

if [ -d "$APP_PATH/Contents/Frameworks" ]; then
    find "$APP_PATH/Contents/Frameworks" -type f -perm -111 -print0 | while IFS= read -r -d '' binary; do
        sign_binary "$binary" || true
    done
fi

# Native Metal binary shipped beside shaders
if [ -x "$APP_PATH/Contents/Resources/third_party/h3.c/h3" ]; then
    sign_binary "$APP_PATH/Contents/Resources/third_party/h3.c/h3" || true
fi
# PyInstaller may place h3 under MacOS/_internal depending on version
find "$APP_PATH/Contents" -type f -name "h3" -perm -111 -print0 | while IFS= read -r -d '' h3bin; do
    sign_binary "$h3bin" || true
done

echo ""
echo "Step 2: Signing main executables..."
for exe in "$APP_PATH/Contents/MacOS"/*; do
    if [ -f "$exe" ] && [ -x "$exe" ]; then
        sign_binary "$exe"
    fi
done

echo ""
echo "Step 3: Signing the app bundle..."
if sign_binary "$APP_PATH"; then
    echo ""
    echo "=================================================="
    echo "Code signing completed"
    echo "=================================================="
    xcrun codesign --verify --deep --strict --verbose=2 "$APP_PATH" 2>&1 || true
    xcrun codesign -dv --verbose=4 "$APP_PATH" 2>&1 || true
    exit 0
fi

echo "Code signing failed"
exit 1
