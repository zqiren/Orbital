#!/usr/bin/env bash
set -euo pipefail

echo "=== Orbital Desktop Build ==="

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

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

# 4. Build installer (if iscc is available)
if command -v iscc &>/dev/null; then
    echo "[4/4] Building installer with Inno Setup..."
    iscc installer/agentos-setup.iss
    echo "Installer: installer/Output/Orbital-Setup-1.0.0.exe"
else
    echo "[4/4] Skipping installer (iscc not found on PATH)"
fi

echo "=== Build complete ==="
echo "Binary: dist/Orbital/Orbital.exe"
