# REPRODUCTION-suite — Sub-Agent Dispatch Round-Trip Bug

**Date:** 2026-05-11
**Purpose:** Repeatable reproductions that verify (a) whether the bug is currently present, and (b) whether a fix candidate has resolved it. Two suites: a Windows daemon-level repro (the only one that exhibits the bug today) and a cross-platform minimal FastAPI suite (sanity baseline, all green).

---

## Suite 1 — Windows daemon repro (definitive, run on the failing host)

Lifted verbatim from `DIAGNOSIS-dispatch-roundtrip-bug.md` §5 with light annotation. Must be run on Windows 10/11 with Python 3.13, claude-agent-sdk 0.1.48, claude-code CLI 2.1.138.

### Pre-conditions

- Branch checked out: `worktree-agent-a9e24fbde848229ba` (spec 3 alone) **or** `test/full-integration` (all specs merged). Either reproduces.
- `orbital-data/settings.json` present in the daemon's cwd with a working management LLM config (Moonshot/Kimi recommended per `CLAUDE.md`). The management LLM is irrelevant to the bug — any working setup boots the daemon.
- No claude-code session in the parent process tree of the daemon (start daemon from a fresh PowerShell or `cmd`, not from inside Claude Code).

### Steps

```bash
# 1. Boot daemon WITHOUT inherited Claude Code env vars
nohup env -u CLAUDECODE -u CLAUDE_CODE_ENTRYPOINT \
  python -m uvicorn agent_os.api.app:create_app --factory --port 8000 --host 0.0.0.0 \
  > /tmp/daemon.log 2>&1 &
sleep 4
curl -s http://127.0.0.1:8000/api/v2/settings >/dev/null   # smoke

# 2. Seed a workspace with orbital/PROJECT_STATE.md (presence triggers spec-3 prompt rendering)
mkdir -p /tmp/repro/orbital
echo "Status: smoke test." > /tmp/repro/orbital/PROJECT_STATE.md

# 3. Create project (substitute a real Windows workspace path)
PROJ=$(curl -s -X POST http://127.0.0.1:8000/api/v2/projects \
  -H "Content-Type: application/json" \
  -d '{"name":"repro","workspace":"C:\\repro","model":"kimi-k2.5","api_key":"",
       "provider":"moonshot","sdk":"openai","autonomy":"hands_off",
       "agent_slug":"built-in","disabled_sub_agents":[]}')
PID=$(echo "$PROJ" | python -c "import json,sys;print(json.load(sys.stdin)['project_id'])")

# 4. Dispatch and time
time curl -s -m 90 -X POST "http://127.0.0.1:8000/api/v2/agents/$PID/inject" \
  -H "Content-Type: application/json" \
  -d '{"content":"hi","target":"claude-code"}'
```

### Pass / fail criteria

| State | Expected response | Time | tasklist |
|---|---|---|---|
| **FAIL** (current bug behavior) | `{"status":"Error: agent 'claude-code' not running for project 'proj_…'"}` | **71–72 s** | No new `claude.exe` |
| **PASS** (after a successful fix) | `{"status":"Message sent to claude-code. Transcript: …"}` | **5–10 s** | New `claude.exe` pid appears |

### Sanity baseline (parent branch — should always PASS)

```bash
cd /d/orbital-public                          # 63140eb (parent, no specs)
# Re-run steps 1-4. Expected: ~5 s response, "Message sent…", new claude.exe spawned.
```

---

## Suite 2 — Cross-platform minimal FastAPI repro (sanity baseline)

Built and validated on macOS during this investigation. **All five scenarios pass on macOS;** they remain useful as:
- a smoke test that the SDK + CLI install is functional on the host
- a baseline to confirm a fix candidate doesn't regress other SDK call shapes
- a starting point if the bug ever needs to be reproduced cross-platform in the future

### Files (located under `/tmp/orbital-tier4/` during the investigation; copy or recreate to verify)

| Path | Role |
|---|---|
| `probe_common.py` | Shared helpers; strips Claude Code env vars and renders the real orbital sub-agent prompt via `agent_os.agent.sub_agent_prompt.render_sub_agent_prompt`. |
| `scenario_a_asyncio.py` | Scenario A — `asyncio.run` + bare-str `system_prompt`. Baseline. |
| `fastapi_app.py` | Scenarios B–E — single FastAPI app with `/dispatch?mode={barestr,preset,stderr,barestr_canusetool,preset_canusetool}`. |
| `tier3_instrument.py` | Tier 3 SDK monkey-patch instrumentation (see below). |
| `workspace/orbital/PROJECT_STATE.md` | Seeded workspace mimicking orbital's directory shape. |

### Launch

```bash
# All scenarios: strip Claude Code env vars (the SDK at subprocess_cli.py:346-348
# spreads os.environ over options.env, so the strip MUST be at the process level)
ENV_STRIP="env -u CLAUDECODE -u CLAUDE_CODE_ENTRYPOINT -u CLAUDE_CODE_SESSION_ID \
           -u CLAUDE_CODE_EXECPATH -u CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS -u CLAUDE_EFFORT"

# Scenario A: standalone asyncio
cd /tmp/orbital-tier4 && $ENV_STRIP python3 scenario_a_asyncio.py

# Scenarios B-E: FastAPI
$ENV_STRIP python3 -m uvicorn fastapi_app:app --port 8765 --host 127.0.0.1 &
sleep 3
curl -s --max-time 90 "http://127.0.0.1:8765/dispatch?mode=barestr"
curl -s --max-time 90 "http://127.0.0.1:8765/dispatch?mode=preset"
curl -s --max-time 90 "http://127.0.0.1:8765/dispatch?mode=stderr"
curl -s --max-time 90 "http://127.0.0.1:8765/dispatch?mode=barestr_canusetool"
```

### Observed results (macOS, claude-agent-sdk 0.1.48, claude CLI 2.1.138)

| Scenario | mode | connect | total | result |
|---|---|---|---|---|
| A | (asyncio.run) | 4.33 s | 6.25 s | OK |
| B | barestr | 4.09 s | 6.77 s | OK |
| C | preset | 4.34 s | 7.09 s | OK |
| D | stderr | 4.39 s | 7.49 s | OK |
| E1 | barestr_canusetool | 4.28 s | 6.10 s | OK |

### Use as a fix-verification gate

Run the full suite both before and after applying Option 1 (`FIX-PROPOSAL.md`). All five scenarios must continue to pass. If a fix regresses any of them on macOS, it's a sign the fix has unintended SDK-API interaction beyond the Windows-specific path it was supposed to address.

---

## Suite 3 — Tier 3 instrumentation deployment (for the next debug pass on Windows)

The monkey-patch harness lives at `/tmp/orbital-tier4/tier3_instrument.py`. To use it on Windows:

1. Copy `tier3_instrument.py` to the orbital repo root on the Windows machine.
2. Add to `agent_os/api/app.py` as the **first import** in the module (before `from fastapi import …`):
   ```python
   import sys, os
   sys.path.insert(0, os.path.dirname(__file__))   # or absolute path to where you copied it
   import tier3_instrument  # noqa: F401
   ```
3. Restart daemon (`bash scripts/restart-daemon.sh`).
4. Run one failing dispatch (Suite 1 step 4).
5. `grep "\[T3 " /tmp/daemon.log` or `Get-Content orbital-data\logs\daemon.log | Select-String "\[T3 "` to capture the trace.

### Interpretation cheat-sheet

| Last trace line before the 60s silence | Diagnosis | Action |
|---|---|---|
| `[T3 … SUBPROC.spawn] PRE open_process …` | Spawn is blocking. Windows asyncio loop can't handle the subprocess. | **W2 confirmed.** Option 1 will not help. Pivot to forcing `WindowsProactorEventLoopPolicy` in the daemon, or to a thread-pool wrapper for the SDK call. |
| `[T3 … SUBPROC.spawn] POST pid=…` and then `[T3 … TRANSPORT.write] POST` and then 60 s silence ending in `[T3 … QUERY.ctrl_req] EXC TimeoutError` | claude.exe spawned, stdin write succeeded, CLI never emitted `control_response` on stdout. **CLI-side stall.** | **W1 confirmed.** Apply Option 1 from `FIX-PROPOSAL.md`. |
| `[T3 … SUBPROC.spawn] EXC OSError(...)` | Spawn raised. Likely env or path issue. | Read the exception message; not the hang we were debugging. |
| Trace stops before `[T3 … BOOT]` line ever appears | Instrumentation didn't load. | Verify the import is the very first line of the daemon's entry module, before any `claude_agent_sdk` import. |

### Removal

The instrumentation is self-contained; just remove the `import tier3_instrument` line and restart the daemon. No SDK files are modified.

---

## Cleanup invariants

These should hold after any reproduction session, success or failure:

- `git diff` on every involved branch returns empty (no committed instrumentation).
- `agent_os/daemon_v2/sub_agent_manager.py` has no `[DBG]` prints from prior diagnosis runs.
- `claude_agent_sdk/` site-packages directory is untouched.
- All test daemon processes terminated (Windows: `taskkill /F /IM python.exe` for orphans; macOS: `pgrep -f uvicorn` then `kill`).
- `/tmp/orbital-tier4/` cleaned up only after a fix has been verified on Windows. Until then, keep it as the cross-platform sanity baseline.
