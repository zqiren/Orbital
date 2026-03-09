#!/usr/bin/env bash
set -euo pipefail

echo "=== Orbital macOS Build ==="

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

# Verify we're on macOS
if [[ "$(uname)" != "Darwin" ]]; then
    echo "ERROR: This script must run on macOS"
    exit 1
fi

# 1. Build React SPA
echo "[1/5] Building React SPA..."
cd web && npm run build && cd ..

# 2. Generate .icns if missing
if [[ ! -f "assets/icon.icns" ]]; then
    echo "[2/5] Generating macOS icon..."
    mkdir -p /tmp/orbital_icon.iconset
    sips -z 16 16     assets/icon.png --out /tmp/orbital_icon.iconset/icon_16x16.png
    sips -z 32 32     assets/icon.png --out /tmp/orbital_icon.iconset/icon_16x16@2x.png
    sips -z 32 32     assets/icon.png --out /tmp/orbital_icon.iconset/icon_32x32.png
    sips -z 64 64     assets/icon.png --out /tmp/orbital_icon.iconset/icon_32x32@2x.png
    sips -z 128 128   assets/icon.png --out /tmp/orbital_icon.iconset/icon_128x128.png
    sips -z 256 256   assets/icon.png --out /tmp/orbital_icon.iconset/icon_128x128@2x.png
    sips -z 256 256   assets/icon.png --out /tmp/orbital_icon.iconset/icon_256x256.png
    sips -z 512 512   assets/icon.png --out /tmp/orbital_icon.iconset/icon_256x256@2x.png
    sips -z 512 512   assets/icon.png --out /tmp/orbital_icon.iconset/icon_512x512.png
    sips -z 1024 1024 assets/icon.png --out /tmp/orbital_icon.iconset/icon_512x512@2x.png
    iconutil -c icns /tmp/orbital_icon.iconset -o assets/icon.icns
    rm -rf /tmp/orbital_icon.iconset
else
    echo "[2/5] macOS icon already exists, skipping..."
fi

# 3. Run PyInstaller with macOS spec
echo "[3/5] Running PyInstaller..."
pyinstaller agent_os/desktop/agentos-macos.spec --distpath dist/ --noconfirm

# 4. Copy SPA and assets into .app bundle
echo "[4/5] Copying SPA and assets..."
APP_RESOURCES="dist/Orbital.app/Contents/Resources"
APP_MACOS="dist/Orbital.app/Contents/MacOS"
cp -r web/dist "$APP_RESOURCES/web"
mkdir -p "$APP_RESOURCES/assets"
cp assets/icon.png "$APP_RESOURCES/assets/"
cp assets/icon.icns "$APP_RESOURCES/assets/"

# 5. Create DMG
echo "[5/5] Creating DMG..."
DMG_NAME="Orbital-1.0.0-macOS.dmg"

# Clean previous DMG
rm -f "dist/$DMG_NAME"

# Check for create-dmg (brew install create-dmg)
if command -v create-dmg &>/dev/null; then
    create-dmg \
        --volname "Orbital" \
        --volicon "assets/icon.icns" \
        --window-pos 200 120 \
        --window-size 600 400 \
        --icon-size 100 \
        --icon "Orbital.app" 150 185 \
        --app-drop-link 450 185 \
        --hide-extension "Orbital.app" \
        --no-internet-enable \
        "dist/$DMG_NAME" \
        "dist/Orbital.app"
else
    echo "  create-dmg not found, using hdiutil fallback..."
    # Simple DMG without fancy layout
    mkdir -p dist/dmg_staging
    cp -r dist/Orbital.app dist/dmg_staging/
    ln -sf /Applications dist/dmg_staging/Applications
    hdiutil create -volname "Orbital" -srcfolder dist/dmg_staging -ov -format UDZO "dist/$DMG_NAME"
    rm -rf dist/dmg_staging
fi

echo ""
echo "=== Build complete ==="
echo "App:       dist/Orbital.app"
echo "DMG:       dist/$DMG_NAME"

# --- Code signing (uncomment when Apple Developer account is ready) ---
# DEVELOPER_ID="Developer ID Application: Your Name (TEAM_ID)"
# echo "[signing] Signing .app bundle..."
# codesign --force --deep --sign "$DEVELOPER_ID" dist/Orbital.app
# echo "[signing] Notarizing DMG..."
# xcrun notarytool submit "dist/$DMG_NAME" --keychain-profile "orbital-notarize" --wait
# xcrun stapler staple "dist/$DMG_NAME"
# echo "[signing] Signed and notarized."
