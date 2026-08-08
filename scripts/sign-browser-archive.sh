#!/usr/bin/env bash
set -uo pipefail
#
# Re-sign the bundled Chromium archive (browsers.tar.gz) for notarization.
#
# Why this exists: notarytool scans INSIDE tar.gz archives. Keeping the browser
# payload as an "opaque" tarball hid it from our own inside-out signer, but not
# from Apple — every notarization of a bundled-browser DMG came back Invalid
# (44 issues, all under browsers.tar.gz: invalid signature / no secure
# timestamp / no hardened runtime on the Chromium binaries; patchright patches
# the binaries, which is what breaks Google's original signatures).
#
# Fix: unpack, sign every Mach-O with the Developer ID identity + hardened
# runtime + secure timestamp + our permissive entitlements (same superset
# sign-macos-app.sh applies bundle-wide; Chromium's V8 needs allow-jit — see
# the v0.6.6 SIGTRAP regression), reseal the nested bundles inside-out, and
# repack. This is the same thing every Electron app does to ship Chromium
# through notarization.
#
# Usage:
#   ORBITAL_SIGN_IDENTITY="Developer ID Application: Name (TEAMID)" \
#     bash scripts/sign-browser-archive.sh <path-to-browsers.tar.gz>
#
ARCHIVE="${1:?usage: sign-browser-archive.sh <browsers.tar.gz>}"
ID="${ORBITAL_SIGN_IDENTITY:?ORBITAL_SIGN_IDENTITY required}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ENT="${ORBITAL_ENTITLEMENTS:-$ROOT/agent_os/desktop/Orbital.entitlements}"

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

echo "Re-signing bundled browser archive: $ARCHIVE"
echo "  identity:     $ID"
echo "  entitlements: $ENT"

tar -xzf "$ARCHIVE" -C "$WORK"

# 1. File pass: sign every LOOSE Mach-O (dylibs, Libraries/, bare Helpers/
# binaries like chrome_crashpad_handler, chrome-headless-shell, ffmpeg).
# Bundle main executables are deliberately SKIPPED here: codesign treats
# signing `Foo.app/Contents/MacOS/Foo` (or a framework's Versions/V/Foo
# binary) as signing the enclosing bundle and validates its subcomponents —
# which are not signed yet at this point. Those binaries are signed by the
# bundle-seal pass below instead. --timestamp contacts Apple's TSA; one
# transient network blip gets one retry.
signed=0
fail=0
while IFS= read -r -d '' f; do
  file -b "$f" 2>/dev/null | grep -q 'Mach-O' || continue
  case "$f" in
    */Contents/MacOS/*) continue ;;  # bundle main executable — sealed in pass 2
  esac
  parent="$(dirname "$f")"
  if [[ "$parent" == *.framework/Versions/* && "$(basename "$(dirname "$parent")")" == "Versions" ]]; then
    continue  # framework main binary (Versions/V/Name) — sealed in pass 2
  fi
  if codesign --force --timestamp --options runtime --entitlements "$ENT" --sign "$ID" "$f" 2>/dev/null \
     || codesign --force --timestamp --options runtime --entitlements "$ENT" --sign "$ID" "$f"; then
    signed=$((signed + 1))
  else
    echo "  FAILED to sign: $f" >&2
    fail=$((fail + 1))
  fi
done < <(find "$WORK" -type f -print0)
echo "Signed $signed Mach-O files in browser archive (failures: $fail)"
if [[ "$fail" -gt 0 ]]; then
  echo "ERROR: $fail browser binaries failed to sign — notarization would reject the archive." >&2
  exit 1
fi

# 2. Bundle pass: reseal nested bundles deepest-first (helper .apps, then the
# framework, then the outer Chrome .app). Sealing a bundle re-signs its main
# executable and rebuilds _CodeSignature/CodeResources; --deep is deliberately
# not used (Apple: unreliable for distribution signing).
while IFS= read -r bundle; do
  if ! codesign --force --timestamp --options runtime --entitlements "$ENT" --sign "$ID" "$bundle"; then
    echo "ERROR: failed to seal bundle: $bundle" >&2
    exit 1
  fi
done < <(find "$WORK" \( -name '*.app' -o -name '*.framework' \) -type d | awk '{ print gsub(/\//, "/"), $0 }' | sort -rn | cut -d' ' -f2-)

# 3. Verify: every Chrome .app must pass strict deep verification, and the
# Chromium executables must carry allow-jit (V8 under hardened runtime).
while IFS= read -r app; do
  echo "Verifying bundle seal: $app"
  codesign --verify --deep --strict "$app" || { echo "ERROR: seal verification failed for $app" >&2; exit 1; }
done < <(find "$WORK" -name '*.app' -maxdepth 4 -type d)

while IFS= read -r -d '' bin; do
  if ! codesign -d --entitlements - --xml "$bin" 2>/dev/null | grep -q 'com.apple.security.cs.allow-jit'; then
    echo "ERROR: $bin lacks com.apple.security.cs.allow-jit — Chromium would SIGTRAP under the hardened runtime (v0.6.6 class regression)." >&2
    exit 1
  fi
  echo "  OK: allow-jit present on $(basename "$bin")"
done < <(find "$WORK" -type f \( -name 'chrome-headless-shell' -o -name 'Google Chrome for Testing' \) -print0)

# 4. Repack in place (same layout: paths relative to the browsers dir).
rm -f "$ARCHIVE"
tar -czf "$ARCHIVE" -C "$WORK" .
echo "Repacked signed browser archive: $ARCHIVE ($(du -h "$ARCHIVE" | cut -f1))"
