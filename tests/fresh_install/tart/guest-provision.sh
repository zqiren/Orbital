#!/bin/bash
# Runs INSIDE the golden VM (as admin) once. Installs what a user would have
# before trying Orbital: Node, the Codex and DeepSeek Harness CLIs, plus the
# test's own Playwright WebKit. Installs NO Orbital and creates no Orbital
# data; the last block proves it.
set -euo pipefail

CODEX_VERSION=${CODEX_VERSION:-latest}
DSH_VERSION=${DSH_VERSION:-0.2.2-rc.1}       # npm dist-tags on @deepseek-ai/dsh are not reliable; pin
PLAYWRIGHT_VERSION=${PLAYWRIGHT_VERSION:-1.60.0}

export HOMEBREW_NO_AUTO_UPDATE=1 HOMEBREW_NO_INSTALL_CLEANUP=1 HOMEBREW_NO_ENV_HINTS=1
eval "$(/opt/homebrew/bin/brew shellenv)"

# Node >= 24.9 (dsh's engines floor). Homebrew's node@24, linked onto PATH.
if ! node -e 'process.exit(+process.versions.node.split(".")[0] >= 24 ? 0 : 1)' 2>/dev/null; then
  brew install node@24
  brew link --overwrite --force node@24
fi
node --version
npm --version

# Worker CLIs, the way their READMEs say: global npm installs into
# /opt/homebrew/bin (on Orbital's auto-detect list and on the login PATH).
# npm 11 holds back install scripts unless allowed; allow dsh's own (its
# spawn helper and node-pty), exactly as npm's warning tells the user to.
DSH_SCRIPTS=@deepseek-ai/dsh-subprocess-local,koffi,node-pty,protobufjs
npm install -g "@openai/codex@${CODEX_VERSION}"
npm install -g --allow-scripts="$DSH_SCRIPTS" "@deepseek-ai/dsh@${DSH_VERSION}"
codex --version
dsh -V

# Warm the test's npm cache + Playwright WebKit (lives in ~/Library/Caches/ms-playwright).
mkdir -p /tmp/fi-warm && cd /tmp/fi-warm
cat > package.json <<EOF
{"name":"fi-warm","private":true,"dependencies":{"playwright":"${PLAYWRIGHT_VERSION}"}}
EOF
npm install --no-audit --no-fund
npx playwright install webkit
cd / && rm -rf /tmp/fi-warm

# Keep the session awake for long scenario runs.
sudo pmset -a sleep 0 displaysleep 0 disksleep 0 >/dev/null 2>&1 || true
defaults -currentHost write com.apple.screensaver idleTime 0 || true

# The image must carry nothing of Orbital and no worker logins. The version
# probes above may have created empty CLI homes; a new user's have none.
rm -rf "$HOME/.codex" "$HOME/.dsh"
for p in /Applications/Orbital.app "$HOME/Library/Application Support/Orbital" \
         "$HOME/Library/Logs/Orbital" "$HOME/orbital" "$HOME/.orbital" "$HOME/.codex" "$HOME/.dsh"; do
  if [ -e "$p" ]; then echo "golden image is not clean: $p exists" >&2; exit 1; fi
done
security show-keychain-info ~/Library/Keychains/login.keychain-db 2>&1 | head -1 || true
echo "golden provision OK"
