# AgentOS — Development Guidelines

## Project Overview

AgentOS is an AI agent orchestration platform with a Python backend (FastAPI) and React/TypeScript frontend (Vite). The backend manages agent lifecycles, sessions, and sub-agent delegation. The frontend provides a chat UI with @mention routing, global/project settings, and real-time WebSocket updates.

## Architecture

- **Backend**: `agent_os/` — FastAPI app at `agent_os/api/app.py`, daemon via uvicorn
- **Frontend**: `web/` — React + TypeScript + Vite, dev server on port 5173
- **Tests**: `tests/unit/`, `tests/platform/` — pytest with pytest-asyncio

## Testing Requirements

**Every feature or bug fix MUST be verified through real daemon testing before committing.**

### 1. Unit Tests First

Run the unit test suite to catch regressions:

```bash
python -m pytest tests/unit/ tests/platform/ -q --ignore=tests/platform/test_consumer3_wiring.py
```

Expected: 629+ passed, 0 failures. The `test_consumer3_wiring.py` failures are pre-existing (sandbox user not configured).

### 2. TypeScript Check (for frontend changes)

```bash
cd web && npx tsc --noEmit
```

Must produce zero errors.

### 3. Daemon Integration Test (MANDATORY)

After unit tests pass, restart the daemon with new code and test the actual behavior:

```bash
# Use the restart script:
bash scripts/restart-daemon.sh

# Or manually:
python -m uvicorn agent_os.api.app:create_app --factory --port 8000
```

Then verify the change works end-to-end:
- For backend changes: use curl or a test script to hit the API endpoints
- For sub-agent changes: inject a message with `target` and verify the response
- For frontend changes: start Vite with `--host` and print the QR code so the user can test on mobile (see Frontend QR Code section below)
- For chat/message changes: check `/api/v2/agents/{pid}/chat` to verify message shape in session

### 4. What Counts as "Tested"

- API endpoint returns expected response codes and body
- Messages appear in session JSONL with correct `role`, `source`, `content`
- Frontend renders the change correctly (no collapsed sections where bubbles should be, etc.)
- No 400/500 errors in daemon logs

## Daemon Management

Use `scripts/restart-daemon.sh` to restart the daemon with fresh code:

```bash
bash scripts/restart-daemon.sh          # restart on default port 8000
bash scripts/restart-daemon.sh 8321     # restart on custom port
```

The script kills any existing daemon, starts a new one, and verifies it's responding.

## Key Conventions

- Branch: `feature/web-ui` — all work happens here
- Backend entry point: `python -m uvicorn agent_os.api.app:create_app --factory --port 8000`
- Frontend dev server: `cd web && npx vite --host 127.0.0.1 --port 5173`
- Sub-agent pipe mode: spawns `claude -p <msg> --output-format stream-json --verbose` per message
- Session continuity: `--resume <session_id>` flag for multi-turn sub-agent conversations

## Frontend QR Code (for mobile testing)

After making frontend changes, ALWAYS set up mobile testing so the user can test on their phone:

1. **Restart the daemon** connected to the production cloud relay on Railway:
   ```bash
   AGENT_OS_RELAY_URL=https://agentos-relay-production.up.railway.app bash scripts/restart-daemon.sh
   ```

2. **Start Vite dev server** bound to all interfaces:
   ```bash
   cd web && npx vite --host 0.0.0.0 --port 5173
   ```

3. **Print a QR code** for the frontend LAN URL so the user can scan it:
   ```bash
   PYTHONIOENCODING=utf-8 python -c "
   import io, sys, qrcode
   qr = qrcode.QRCode(border=1)
   qr.add_data('http://<LAN-IP>:5173')
   qr.make()
   f = io.StringIO()
   qr.print_ascii(out=f, invert=True)
   sys.stdout.buffer.write(f.getvalue().encode('utf-8'))
   "
   ```

The cloud relay is deployed on Railway at `agentos-relay-production.up.railway.app`. Do NOT start a local relay — always use the production deployment. The user tests UI changes on a real mobile device via LAN. Always provide the QR code before claiming frontend work is done.

## Relay Deployment Rule

After any relay redeployment to Railway, ALWAYS restart the local daemon. The relay stores device registrations and tunnel state in-memory — a redeploy wipes all of it. The daemon will hold a zombie WebSocket connection to the dead relay process indefinitely. Restart forces re-registration and fresh tunnel.

## Services & Testing

After making code changes to the daemon or backend services, ALWAYS restart the service before testing. Never test against a running instance with stale code.

## Behavioral Rules

- When asked to only investigate or diagnose, do NOT make code edits (including debug logging) unless explicitly asked. Report findings only.
- If a requested task file or resource doesn't exist, STOP and ask the user for clarification. Do not infer a different task and start executing it without confirmation.

## Debugging Guidelines

- When a fix attempt fails twice for the same issue, stop and present the user with a summary of what was tried, what failed, and ask for direction before continuing. Do not keep iterating on the same approach.
- When fixing a bug that involves cached state (config objects, agent handles, running instances), always consider that in-memory references may be stale. Restart or refresh all dependent components after the fix.
- For bug fixes, use the `test-gated-bugfix` skill to enforce red-green-refactor workflow with automatic stop after 3 failed attempts.

## Agent Coordination Rules

- When spawning parallel agents, always define file scope boundaries (`allowed_files`, `forbidden_files`) in each task description. Use the `coordinated-agents` skill.
- The coordinator must run the full test suite after merging all sub-agent work.
- Sub-agents must not commit — only the coordinator commits.
- Sub-agents must not modify files outside their defined scope.

## Project-Specific Knowledge

- This project uses Python (backend/daemon) and TypeScript (frontend/desktop). The Kimi/Moonshot API requires temperature=1 and uses api.moonshot.cn as the base URL. When configuring LLM providers, check for hardcoded defaults in ALL code paths (backend, frontend, agent loop).

## React Anti-Patterns to Avoid

- **Never rely on closure variables mutated inside `setState` updaters.** React 19 batching makes the read timing unpredictable — the updater may be deferred to the render phase, so a local variable set inside the updater can still be `false` when read outside it. Use `flushSync` if you need synchronous state computation, or restructure to avoid cross-boundary communication entirely. See the `toggleDirectory` fix in `FileExplorer.tsx` for a concrete example.

## Release Process

Cutting an Orbital release means producing platform-specific installers (`.exe` for Windows, `.dmg` for macOS), tagging the SHA, and publishing to GitHub Releases. PyInstaller cannot cross-compile, so each platform is built on its native machine.

### Pre-flight (run once, on either platform)

1. **Confirm clean working tree:**
   ```bash
   git status      # must be clean
   git rev-parse HEAD
   ```

2. **Bump version strings.** Seven locations carry a version, and they are currently out of sync (`0.0.0` / `0.1.0` / `1.0.0`) — they have never been kept aligned. Update every one to the new `{X.Y.Z}`:
   - `pyproject.toml` — `version` field (line 7, currently `0.1.0`)
   - `web/package.json` — `version` field (line 4, currently `0.0.0`)
   - `agent_os/desktop/agentos-macos.spec` — `CFBundleShortVersionString` and `CFBundleVersion` (lines 95–96, currently `1.0.0`)
   - `installer/agentos-setup.iss` — `AppVersion` (line 6) AND `OutputBaseFilename` (line 11), both currently `1.0.0`
   - `scripts/build-desktop.sh` — the version inside the echoed installer-path string (line 28)
   - `scripts/build-macos.sh` — `DMG_NAME` (line 69, currently `Orbital-1.0.0-macOS.dmg`)
   - `agent_os/desktop/agentos.spec` — no version to bump (Windows spec carries none)

   Commit with message `chore: bump version to v{X.Y.Z}`.

3. **Verify clean build from cold state** (catches `.tsbuildinfo`-cached failures like the v0.5.1 regression):
   ```bash
   rm -rf web/node_modules web/dist web/.tsbuildinfo
   cd web && npm ci && npx tsc --noEmit && npm run build && cd ..
   python -m pytest tests/unit/ -q
   ```
   All four commands must exit zero before proceeding. **Do not skip — local builds can pass on stale caches while fresh checkouts fail.**

---

### Windows build

**Machine requirement:** Windows 10/11 with Python 3.x, Node.js, Git Bash (or WSL), and Inno Setup installed (with `iscc` on PATH).

**Steps:**

1. From the repo root in Git Bash:
   ```bash
   bash scripts/build-desktop.sh
   ```
   This is the Windows build path — runs PyInstaller against `agent_os/desktop/agentos.spec`, copies the React SPA into `dist/Orbital/web/`, and invokes Inno Setup automatically if `iscc` is on PATH.

2. Verify outputs exist:
   - PyInstaller bundle at `dist/Orbital/`, entrypoint `dist/Orbital/Orbital.exe`
   - Bundle includes `agent_os/vendor/rg/rg.exe` (Windows ripgrep — required for the `grep` tool to work on user machines)
   - Installer at `installer/Output/Orbital-Setup-{X.Y.Z}.exe`

3. If `iscc` was not on PATH during step 1, run Inno Setup manually now:
   ```bash
   iscc installer/agentos-setup.iss
   ```

4. Smoke test on a clean Windows VM (or fresh user account):
   - Install via the `.exe`
   - Launch Orbital
   - Create a test project
   - Verify the SmartScreen warning is the only "scary" dialog (expected — installer is unsigned)
   - Run an agent with a prompt that exercises `grep` to confirm ripgrep is bundled correctly

**Output filename convention:** `Orbital-Setup-{X.Y.Z}.exe` (controlled by `OutputBaseFilename` in the .iss — bump as part of pre-flight step 2).

---

### macOS build

**Machine requirement:** Apple Silicon Mac (M1 or later), macOS 13+ (Ventura), Python 3.x, Node.js, `create-dmg` (or `hdiutil` fallback).

**Note on architecture:** `agentos-macos.spec` does not set `target_arch`, so PyInstaller produces a binary matching the host machine — **arm64 only when built on Apple Silicon, x86_64 only when built on Intel.** It is *not* universal. Confirm after build with:
```bash
file dist/Orbital.app/Contents/MacOS/Orbital
```
If a universal binary is needed later, that is a spec change (`target_arch='universal2'`) plus a Python install with universal wheels — defer until there is real demand.

**Steps:**

1. From the repo root:
   ```bash
   bash scripts/build-macos.sh
   ```
   The script handles ad-hoc signing, xattr stripping, and DMG packaging — see `## macOS Build Notes` below for the gotchas it works around.

2. Verify outputs exist:
   ```bash
   ls -lh dist/Orbital-{version}-macOS.dmg
   ls -lh dist/Orbital.app/Contents/MacOS/Orbital
   ```

3. **Sanity-check that platform-specific assets are bundled** (these have caused regressions before):
   ```bash
   # ripgrep for grep tool — both archs are vendored at agent_os/vendor/rg/macos-{arm64,x86_64}/
   # and copied into the .app by the spec's datas list. Runtime selects via platform.machine().
   ls dist/Orbital.app/Contents/Resources/agent_os/vendor/rg/macos-arm64/
   ls dist/Orbital.app/Contents/Resources/agent_os/vendor/rg/macos-x86_64/

   # Patchright driver for browser automation
   ls dist/Orbital.app/Contents/Resources/patchright/driver/ | head -3

   # App Nap suppression code
   grep -l "beginActivity\|app_nap" agent_os/platform/macos/provider.py

   # Window close intercept
   grep -l "miniaturize\|windowShouldClose" agent_os/desktop/main.py
   ```

4. Smoke test on a clean Mac (or new user account):
   - Mount the `.dmg`, drag to Applications
   - First launch: expect Gatekeeper warning ("cannot be opened because Apple cannot check it for malicious software"). User must Right-click → Open → Open to bypass. Document this in release notes.
   - Verify agent runs, grep tool works, browser automation launches.

**Output filename convention:** `Orbital-{X.Y.Z}-macOS.dmg`

**Code signing:** Not currently configured. `scripts/build-macos.sh` contains commented-out Developer-ID `codesign` and `notarytool` placeholders for future use. The script *does* already perform ad-hoc signing (mandatory after asset copy — see `## macOS Build Notes` below). Defer Developer-ID signing + notarization until a signed certificate is available.

---

### Tag and publish (run once, after both platform builds succeed)

1. **Tag from the SHA that was built:**
   ```bash
   git tag -a v{X.Y.Z} -m "v{X.Y.Z}"
   git push origin v{X.Y.Z}
   ```

2. **Create GitHub Release** tied to the tag:
   - Title: `v{X.Y.Z}`
   - Upload both `Orbital-Setup-{X.Y.Z}.exe` and `Orbital-{X.Y.Z}-macOS.dmg` as release assets
   - Write release notes covering: user-visible changes, known issues (including the unsigned-installer warnings on both platforms), and install instructions

3. **Update the README install links** if they reference a specific version rather than `/releases/latest`.

---

### Post-release verification

Within 24 hours of publishing:

- Download both installers from the public Releases page (not local artifacts) on a fresh machine each
- Run through the smoke test in each platform section
- If a regression is found: do **not** delete or modify the release; cut a v{X.Y.Z+1} patch instead

---

### Hotfix workflow

If a bug is reported on a tagged version while main has moved ahead:

1. `git checkout -b hotfix/v{X.Y.Z+1} v{X.Y.Z}`
2. Apply the fix
3. Tag, build, publish v{X.Y.Z+1} via the steps above
4. Cherry-pick the fix back to main
5. Delete the hotfix branch

---

### Things this runbook does NOT cover (intentional)

- CI / GitHub Actions automation — manual builds only for now
- Code signing (Windows or macOS)
- Auto-update mechanisms
- Homebrew cask, winget, or other package-manager distribution
- Notarization (macOS) — gated on signing first

If any of these become priorities, add them as a separate section rather than inlining into the existing flow.

## macOS Build Notes (`scripts/build-macos.sh`)

- **Re-sign the bundle AFTER copying SPA/assets in.** PyInstaller ad-hoc signs the `.app` during its BUNDLE step. Steps that copy new files into `Contents/Resources/` (web SPA, icons) happen *after* that signing, so the new files aren't in `_CodeSignature/CodeResources` and the seal is broken (`codesign --verify` reports "a sealed resource is missing or invalid"). On macOS Sequoia+, Finder validates the seal when drag-installing from a DMG into `/Applications` and skips items whose hashes don't match — surfacing as **"The operation can't be completed because some items had to be skipped."** The app still launches fine from elsewhere (e.g. `~/Desktop`) because Gatekeeper doesn't re-validate there. Fix: `codesign --force --deep --sign - dist/Orbital.app` after asset copy, before DMG creation.
- **Use `ditto`, not `cp -r`, when staging the `.app` for DMG packaging.** The bundle contains ~2,500 symlinks (dyld framework versioning like `Current -> A`); `cp -r` dereferences them into real copies, bloating the DMG ~2x (587 MB staging → 247 MB DMG vs. 274 MB → 122 MB with `ditto`).
- **`com.apple.quarantine` xattrs should be stripped** (`xattr -cr dist/Orbital.app`) to avoid Gatekeeper nags. `com.apple.provenance` is a restricted kernel-added xattr on every Sequoia-built binary — it cannot be stripped by userspace and is **not** the cause of drag-install failures (the broken seal is).
- The bundle is ad-hoc signed, so first launch triggers Gatekeeper. Users must right-click → **Open** once, or we need a Developer ID + notarization (stubs already in the script).
- `create-dmg` (from Homebrew) produces nicer DMGs but isn't required; the `hdiutil` fallback path is what CI/users without brew hit — keep it correct.

## Known Issues

- `tests/platform/test_consumer3_wiring.py` — pre-existing failures (Windows sandbox user not configured)
- `tests/test_e2e.py`, `tests/test_user_stories.py`, `tests/test_wiring.py` — require real LLM API key and platform setup
- Daemon on Windows: use `tasklist | grep python` / `taskkill /F /PID <pid>` if `pkill` doesn't work

## Robustness Testing

- Daemon runs at localhost:8000, must be started before tests
- Each test project gets its own temp workspace, never reuse
- Evidence goes to evidence/ at project root
- All fixes need regression tests (TEST RULE 1, no exceptions)
- Read ACTIVE-decisions.md before any architectural code changes
- Never modify ROBUSTNESS-batch-*.md, they are specs not code
- If pass criteria seem wrong, report to lead, do not change the spec
- After each fix, reload the daemon for the debugged code
