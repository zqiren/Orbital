#!/usr/bin/env bash
# Download a Chromium build via the patchright driver and pack it into a
# tarball for the installers to bundle. Installers ship this so a fresh
# machine (notably behind the Great Firewall) never needs to fetch ~170MB
# of Chromium from a throttled CDN at first launch.
#
# tar.gz (not zip): Chromium's macOS framework uses relative symlinks
# (Versions/Current -> A) that Python's zipfile drops on extract but
# tarfile restores faithfully, along with the executable bits.
#
# Usage: bash scripts/stage-browsers.sh <output-tar-gz-path>
set -euo pipefail

OUT="${1:?usage: stage-browsers.sh <output-tar-gz-path>}"
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

PY="${PYTHON:-python3}"
STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT

echo "Downloading Chromium into staging dir $STAGE ..."
PLAYWRIGHT_BROWSERS_PATH="$STAGE" "$PY" -m patchright install chromium

# Tar the whole browsers dir (contains chromium-<rev>/ + its
# INSTALLATION_COMPLETE marker), paths relative to the browsers dir so it
# extracts straight into PLAYWRIGHT_BROWSERS_PATH.
mkdir -p "$(dirname "$OUT")"
rm -f "$OUT"
tar -czf "$OUT" -C "$STAGE" .
echo "Staged browser archive: $OUT ($(du -h "$OUT" | cut -f1))"
