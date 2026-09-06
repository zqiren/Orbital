#!/usr/bin/env bash
# release-aliases.sh — give a GitHub release version-less installer names.
#
# The README's download buttons point at
#   https://github.com/zqiren/Orbital/releases/latest/download/Orbital-Setup.exe
#   https://github.com/zqiren/Orbital/releases/latest/download/Orbital-macOS.dmg
# which only resolve when the latest release carries assets with exactly those
# names. Our installers are versioned (Orbital-Setup-X.Y.Z.exe,
# Orbital-X.Y.Z-macOS.dmg), so after uploading them run this once per release:
#
#   bash scripts/release-aliases.sh            # latest release
#   bash scripts/release-aliases.sh v0.11.0    # a specific tag
#
# It downloads each versioned asset and re-uploads a copy under the stable
# name. Existing aliases are replaced (--clobber). Needs `gh` logged in.
set -euo pipefail

REPO="zqiren/Orbital"
TAG="${1:-$(gh release view --repo "$REPO" --json tagName -q .tagName)}"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

assets="$(gh release view "$TAG" --repo "$REPO" --json assets -q '.assets[].name')"
alias_for() {
  case "$1" in
    Orbital-Setup-*.exe) echo "Orbital-Setup.exe" ;;
    Orbital-*-macOS.dmg) echo "Orbital-macOS.dmg" ;;
    *) echo "" ;;
  esac
}

echo "Release $TAG"
uploaded=0
while IFS= read -r name; do
  alias="$(alias_for "$name")"
  [[ -z "$alias" || "$alias" == "$name" ]] && continue
  echo "  $name -> $alias"
  gh release download "$TAG" --repo "$REPO" --pattern "$name" --dir "$WORK" --clobber
  mv "$WORK/$name" "$WORK/$alias"
  gh release upload "$TAG" "$WORK/$alias" --repo "$REPO" --clobber
  rm -f "$WORK/$alias"
  uploaded=$((uploaded + 1))
done <<< "$assets"

if [[ "$uploaded" -eq 0 ]]; then echo "No versioned installer assets found on $TAG; nothing uploaded."; exit 1; fi
echo "Done. Verify:"
for a in Orbital-Setup.exe Orbital-macOS.dmg; do
  echo "  curl -sIL https://github.com/$REPO/releases/latest/download/$a | grep -i '^content-length'"
done
