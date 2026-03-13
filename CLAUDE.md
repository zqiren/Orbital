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
