#!/usr/bin/env bash
set -euo pipefail

echo "=== Orbital Desktop Build ==="

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

# 0. Stamp the runtime version (spec 046 §7): agent_os/_version.py is the only
#    authoritative version source inside the frozen bundle. Derives from
#    pyproject.toml — no new manual bump location.
APP_VERSION="$(python -c "import tomllib; print(tomllib.load(open('pyproject.toml','rb'))['project']['version'])")"
printf '__version__ = "%s"\n' "$APP_VERSION" > agent_os/_version.py
echo "[0/4] Stamped agent_os/_version.py = $APP_VERSION"

# 1. Build React SPA
echo "[1/4] Building React SPA..."
cd web && npm run build && cd ..

# 2. Run PyInstaller
echo "[2/4] Running PyInstaller..."
pyinstaller agent_os/desktop/agentos.spec --distpath dist/ --noconfirm

# 3. Copy SPA and assets
echo "[3/4] Copying SPA and assets..."
cp -r web/dist dist/Orbital/web
mkdir -p dist/Orbital/assets
cp assets/icon.png dist/Orbital/assets/
cp assets/icon.ico dist/Orbital/assets/

# 3b. Bundle Chromium so first launch needs no CDN download (alongside the exe)
echo "[3b/4] Staging bundled Chromium..."
bash scripts/stage-browsers.sh dist/Orbital/browsers.tar.gz

# 4. Build installer (if iscc is available)
if command -v iscc &>/dev/null; then
    echo "[4/4] Building installer with Inno Setup..."
    # WebView2 Evergreen bootstrapper — referenced by the .iss [Files]
    # section; the installer runs it when the runtime is missing.
    WV2="installer/MicrosoftEdgeWebView2Setup.exe"
    if [ ! -f "$WV2" ]; then
        echo "Downloading WebView2 Evergreen bootstrapper..."
        curl -fSL -o "$WV2" "https://go.microsoft.com/fwlink/p/?LinkId=2124703"
    fi
    iscc installer/agentos-setup.iss
    echo "Installer: installer/Output/Orbital-Setup-0.10.0.exe"
else
    echo "[4/4] Skipping installer (iscc not found on PATH)"
fi

echo "=== Build complete ==="
echo "Binary: dist/Orbital/Orbital.exe"
