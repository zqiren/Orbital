# Re-capturing the README screenshots

The README image set goes stale every time the UI moves. `web/scripts/capture-readme-shots.mjs`
re-shoots it against a running daemon, in both languages, at a fixed size and
scroll position — so the set stays internally consistent instead of drifting a
little with each hand-taken replacement.

**The rig never points at your real Orbital.** The packaged app's data contains
private projects, and its daemon holds both port 8000 and the singleton PID file
at `$HOME/orbital/daemon.pid`. Capture runs against throwaway daemons with an
overridden `HOME`, serving a filtered clone of the demo-safe projects.

## 1. Build the SPA

```bash
cd web && npx tsc -b && npm run build && cd ..
```

The capture daemon serves `web/dist` via `AGENT_OS_SPA_DIR`, so a stale build
means stale screenshots.

## 2. Stage demo data

Clone only the publish-safe projects — **Quick Tasks**, **Orbital-marketing**,
**Hn-daily** — into a scratch directory, rewrite each `workspace` path in
`projects.json` to the clone, and copy `settings.json` with `llm.api_key` nulled.

Two scrub passes are needed, and they are not interchangeable:

- **Token replacement** across `.md` / `.json` / `.jsonl` in the clone. Do *not*
  delete matching lines: `orbital/tool-results/*.json` are single-line JSON
  documents, so dropping the line empties the file and breaks every parse
  downstream (this bit once — 410 files).
- **Block removal** in the six memory files that actually get screenshotted
  (`CONTEXT.md`, `DECISIONS.md`, `LESSONS.md`, `PROJECT_STATE.md`,
  `SESSION_LOG.md`, `DECISIONS_ARCHIVE.md`). Renaming an employer is not enough —
  the surrounding narrative is still a disclosure. Drop the whole `##` section
  when it is substantially about it.

Then check the rendered result, not just the grep: the Workbench reads its cards
from `PROJECT_STATE.md`, so real names of third parties can surface there even
after the memory files look clean.

## 3. Start the capture daemon

```bash
cd "$SCRATCH"
HOME="$SCRATCH/home" \
AGENT_OS_SPA_DIR="$REPO/web/dist" \
AGENT_OS_API_KEY=sk-demo-capture-rig \
PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring \
PYTHONPATH="$REPO" \
python3 -m uvicorn agent_os.api.app:create_app --factory --port 8765 --host 127.0.0.1
```

`orbital-data` resolves relative to the working directory, so `cd` first.
`AGENT_OS_API_KEY` takes precedence over the keychain in `ApiKeyStore`, which is
what marks the daemon as past setup *and* keeps it from ever reading your real
key.

## 4. Start the wizard daemon

The setup wizard only renders while `llm.api_key_set` is false, so it needs a
second daemon with an empty data dir, on port 8766, with **no**
`AGENT_OS_API_KEY`. It also needs a keyring that can actually store a value —
the wizard advances only once `set_api_key` reads its own write back, which the
null backend can never satisfy. Use a small in-process backend
(`PYTHON_KEYRING_BACKEND=shotkeyring.InMemoryKeyring`) rather than the real
Keychain.

Restart this daemon before every run: capturing `connect-accounts` sets a key,
and the daemon then holds it for the rest of its life. The script checks for
this and refuses rather than silently shooting the wrong screen.

## 5. Shoot

```bash
node web/scripts/capture-readme-shots.mjs                 # -> docs/screenshots/
LOCALE=zh node web/scripts/capture-readme-shots.mjs       # -> docs/screenshots/zh/
ONLY=files,workbench node web/scripts/capture-readme-shots.mjs
```

WebKit at 1200×800, `deviceScaleFactor: 2`. WebKit is deliberate — the desktop
app ships on pywebview → WKWebView, and Chromium would render a browser users
never see. The zh pass drives `localStorage['orbital.locale']`, and resolves
on-screen text through the app's own catalog (`web/src/i18n/strings.ts`) so it
keeps working when a string changes.

Shots marked `needsAgent` are skipped: their subject only exists while a real
agent run is in flight (a genuinely running queue item, a streaming reply).
Capture those during a live run — see below.

## 6. Review

Screenshots pass silently when they are wrong — a collapsed tree, a settings
pane scrolled to the wrong section, a surface that rendered its empty state.
Build a contact sheet and look at the whole pass at once rather than trusting
the `ok` lines.

Two failure modes to look for specifically:

- **Leaked context behind an overlay.** Trigger detail only covers the top of
  the pane; on the Chat tab the session list and message body stay visible
  underneath. Shoot those over the Files tab.
- **Empty-looking surfaces.** The calendar opens scrolled to midnight and the
  current week only projects slots from today forward — on a Sunday that is one
  lonely block. The script advances a week and scrolls to working hours.

## 7. Video

Same rig, `recordVideo` on the browser context, then `ffmpeg` to mp4/gif. A live
agent run needs a real key in `AGENT_OS_API_KEY` on the capture daemon — which
means real spend, and non-deterministic output, so expect several takes.
