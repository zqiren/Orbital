#!/usr/bin/env bash
set -uo pipefail
#
# Inside-out Developer ID signing for the Orbital .app (notarization-ready).
#
# Apple explicitly says NOT to use `codesign --deep` for distribution: it does
# not reliably re-sign nested binaries with a Developer ID cert, a secure
# timestamp, and the hardened runtime — which makes notarization reject the
# archive ("binary is not signed with a valid Developer ID certificate"). The
# correct approach is to sign every nested Mach-O individually (leaves first),
# then seal the app bundle last with the entitlements.
#
# Usage:
#   ORBITAL_SIGN_IDENTITY="Developer ID Application: Name (TEAMID)" \
#     bash scripts/sign-macos-app.sh dist/Orbital.app
#
APP="${1:?usage: sign-macos-app.sh <path-to-.app>}"
ID="${ORBITAL_SIGN_IDENTITY:?ORBITAL_SIGN_IDENTITY required}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ENT="${ORBITAL_ENTITLEMENTS:-$ROOT/agent_os/desktop/Orbital.entitlements}"

echo "Inside-out signing $APP"
echo "  identity:     $ID"
echo "  entitlements: $ENT"

signed=0
fail=0
# Sign every Mach-O file in the bundle (dylibs, .so, embedded executables like
# the Python interpreter, ripgrep, and the patchright node driver). --timestamp
# contacts Apple's TSA, so one transient network blip gets one retry.
while IFS= read -r -d '' f; do
  if file -b "$f" 2>/dev/null | grep -q 'Mach-O'; then
    if codesign --force --timestamp --options runtime --sign "$ID" "$f" 2>/dev/null \
       || codesign --force --timestamp --options runtime --sign "$ID" "$f"; then
      signed=$((signed + 1))
    else
      echo "  FAILED to sign: $f" >&2
      fail=$((fail + 1))
    fi
  fi
done < <(find "$APP" -type f -print0)
echo "Signed $signed nested Mach-O binaries (failures: $fail)"

# Seal the app bundle itself (signs the main executable + builds CodeResources)
# with the hardened-runtime entitlements.
codesign --force --timestamp --options runtime \
  --entitlements "$ENT" --sign "$ID" "$APP"

echo "Verifying..."
codesign --verify --deep --strict --verbose=2 "$APP"
echo "Main executable seal:"
codesign -dvv "$APP" 2>&1 | grep -E '^Authority=|^TeamIdentifier=|^Timestamp=|flags='

if [[ "$fail" -gt 0 ]]; then
  echo "ERROR: $fail binaries failed to sign — notarization would reject the archive." >&2
  exit 1
fi
