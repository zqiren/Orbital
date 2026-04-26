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
#    Uses Pillow instead of sips to preserve the alpha channel
#    (sips strips transparency, breaking macOS squircle mask).
if [[ ! -f "assets/icon.icns" ]]; then
    echo "[2/5] Generating macOS icon..."
    python3 -c "
from PIL import Image
import os, subprocess, shutil
src = Image.open('assets/icon.png').convert('RGBA')
iconset = '/tmp/orbital_icon.iconset'
os.makedirs(iconset, exist_ok=True)
for name, sz in [
    ('icon_16x16.png',16),('icon_16x16@2x.png',32),
    ('icon_32x32.png',32),('icon_32x32@2x.png',64),
    ('icon_128x128.png',128),('icon_128x128@2x.png',256),
    ('icon_256x256.png',256),('icon_256x256@2x.png',512),
    ('icon_512x512.png',512),('icon_512x512@2x.png',1024),
]:
    src.resize((sz,sz), Image.LANCZOS).save(os.path.join(iconset,name), format='PNG')
subprocess.run(['iconutil','-c','icns',iconset,'-o','assets/icon.icns'], check=True)
shutil.rmtree(iconset)
"
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

# 4b. Re-sign bundle AFTER SPA/assets are copied in.
# PyInstaller ad-hoc signs the .app during BUNDLE, but step 4 adds new files
# that aren't in _CodeSignature/CodeResources. macOS Sequoia's Finder
# validates the seal when drag-installing to /Applications and skips items
# whose hashes don't match — surfacing as "some items had to be skipped".
echo "[4b/5] Re-signing bundle ad-hoc after asset copy..."
codesign --force --deep --sign - dist/Orbital.app
codesign --verify --deep --strict dist/Orbital.app

# 5. Create DMG
echo "[5/5] Creating DMG..."
DMG_NAME="Orbital-1.0.0-macOS.dmg"

# Strip user-writable extended attributes (quarantine etc.).
# com.apple.provenance is restricted and can't be stripped — harmless.
xattr -cr dist/Orbital.app

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
    # Use ditto (not cp -r) so symlinks inside the bundle stay as symlinks.
    # cp -r dereferences them, bloating the DMG ~2x (bundle has ~2500 symlinks).
    mkdir -p dist/dmg_staging
    ditto dist/Orbital.app dist/dmg_staging/Orbital.app
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
