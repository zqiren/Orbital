# Vendored ripgrep binaries

Orbital ships ripgrep with the installer so the `grep` tool works on
machines that don't have ripgrep installed system-wide.

## Current version

ripgrep **14.1.1** (https://github.com/BurntSushi/ripgrep/releases/tag/14.1.1)

## Layout

- `rg.exe` — Windows x86_64
- `macos-arm64/rg` — macOS Apple Silicon
- `macos-x86_64/rg` — macOS Intel

Linux bundling is a separate task.

## How to update (macOS)

Run from the repo root, substituting the target version:

```bash
RG_VERSION="14.1.1"
TMP=$(mktemp -d) && cd "$TMP"

curl -fsSL "https://github.com/BurntSushi/ripgrep/releases/download/${RG_VERSION}/ripgrep-${RG_VERSION}-aarch64-apple-darwin.tar.gz" | tar xz
curl -fsSL "https://github.com/BurntSushi/ripgrep/releases/download/${RG_VERSION}/ripgrep-${RG_VERSION}-x86_64-apple-darwin.tar.gz" | tar xz

cd -
cp "$TMP/ripgrep-${RG_VERSION}-aarch64-apple-darwin/rg"  agent_os/vendor/rg/macos-arm64/rg
cp "$TMP/ripgrep-${RG_VERSION}-x86_64-apple-darwin/rg"   agent_os/vendor/rg/macos-x86_64/rg
chmod +x agent_os/vendor/rg/macos-arm64/rg agent_os/vendor/rg/macos-x86_64/rg

# Ad-hoc code sign so Gatekeeper accepts the bundled binary
codesign --sign - --force agent_os/vendor/rg/macos-arm64/rg
codesign --sign - --force agent_os/vendor/rg/macos-x86_64/rg

# Sanity check
file agent_os/vendor/rg/macos-arm64/rg   # Mach-O 64-bit executable arm64
file agent_os/vendor/rg/macos-x86_64/rg  # Mach-O 64-bit executable x86_64
codesign -dvv agent_os/vendor/rg/macos-arm64/rg   # Signature=adhoc
```

## Why ad-hoc signed

The bundled `rg` is unsigned off the release tarball. macOS may refuse
to execute an unsigned binary launched from a signed parent app, or
raise a second Gatekeeper dialog. An ad-hoc signature (`codesign --sign -`)
satisfies the minimum signature requirement without needing an Apple
Developer account.

## PyInstaller

The spec files bundle these via `datas=` (not `binaries=`). Using
`binaries=` causes PyInstaller to rewrite/analyse the binary, which
breaks ripgrep. See `TASK/TASK-bundle-ripgrep-macos.md` Risk 3.

Execute bits and quarantine xattrs are handled at daemon startup —
see `_prepare_bundled_ripgrep` in `agent_os/desktop/main.py`.
