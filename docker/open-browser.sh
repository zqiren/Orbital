#!/bin/bash
# Target of $BROWSER: opens the links the daemon or a CLI login launches on the
# virtual desktop, so OAuth loopback callbacks stay inside the container.
chrome=$(ls -d /opt/ms-playwright/chromium-*/chrome-linux*/chrome 2>/dev/null | head -1)
exec "$chrome" --no-sandbox --no-first-run --no-default-browser-check \
  --password-store=basic --user-data-dir="$HOME/.orbital/oauth-browser" "$@" >/dev/null 2>&1
