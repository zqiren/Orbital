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
#
# CRITICAL: pass --entitlements to EVERY nested Mach-O, not just the bundle seal
# below. Entitlements are per-binary and do NOT inherit from the app seal. The
# patchright `node` driver runs V8, which JITs; under the hardened runtime JIT
# requires com.apple.security.cs.allow-jit. Sealing only the main executable
# with entitlements (the old behaviour) left `node` hardened-but-JIT-less, so it
# died with SIGTRAP ("Failed to reserve virtual memory for CodeRange") and the
# entire browser subsystem was broken on the first notarized build (v0.6.6).
# These entitlements are all valid for Developer ID notarization; harmless on
# dylibs (ignored at runtime), essential on node/python.
while IFS= read -r -d '' f; do
  if file -b "$f" 2>/dev/null | grep -q 'Mach-O'; then
    if codesign --force --timestamp --options runtime --entitlements "$ENT" --sign "$ID" "$f" 2>/dev/null \
       || codesign --force --timestamp --options runtime --entitlements "$ENT" --sign "$ID" "$f"; then
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

# Regression guard for the v0.6.6 SIGTRAP bug: the bundled patchright `node`
# driver runs V8 (JIT) and MUST carry com.apple.security.cs.allow-jit, or it
# crashes on launch and every browser action fails. Fail the build here rather
# than ship another browser-broken installer. Checks every node copy in the
# bundle (PyInstaller stages patchright under both Frameworks/ and Resources/).
echo "Verifying JIT entitlement on the patchright node driver(s)..."
node_found=0
while IFS= read -r -d '' node_bin; do
  node_found=$((node_found + 1))
  if ! codesign -d --entitlements - --xml "$node_bin" 2>/dev/null | grep -q 'com.apple.security.cs.allow-jit'; then
    echo "ERROR: $node_bin lacks com.apple.security.cs.allow-jit — V8 will SIGTRAP and the browser tool is dead. Did nested-binary signing drop --entitlements?" >&2
    exit 1
  fi
  echo "  OK: allow-jit present on $node_bin"
done < <(find "$APP" -type f -path '*/patchright/driver/node' -print0)
if [[ "$node_found" -eq 0 ]]; then
  echo "WARNING: no patchright node driver found in bundle — browser tool may be unbundled." >&2
fi

if [[ "$fail" -gt 0 ]]; then
  echo "ERROR: $fail binaries failed to sign — notarization would reject the archive." >&2
  exit 1
fi
