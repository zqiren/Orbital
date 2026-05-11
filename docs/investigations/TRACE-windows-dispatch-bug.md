# TRACE — Windows Dispatch Round-Trip Bug (Tier 3)

**Investigation:** `TASK-windows-tier3-trace-capture.md`
**Date:** 2026-05-12
**Host:** Windows 10 (MINGW64_NT-10.0-19045), Python 3.13, claude-agent-sdk 0.1.48, claude-code 2.1.138.
**Branch:** `test/full-integration` @ `a55a28c` (specs 1+2+3+4 merged).
**Status:** Bug confirmed reproducing; Tier 3 trace captured; classification W1; recommendation: apply `FIX-PROPOSAL.md` Option 1 as written.

---

## 1. Phase 1 result — bug reproduces on current branch

Un-instrumented dispatch against the integrated branch:

| Probe | Value |
|---|---|
| Project | `proj_fcc375d3401b` name=`phase1-baseline` workspace=`D:\repro-smoke` |
| Dispatch | `POST /api/v2/agents/{pid}/inject` with `{"content":"What is the current state of this project? Be concise.","target":"claude-code"}` |
| Elapsed | **72 seconds** |
| Response | `{"status":"Error: agent 'claude-code' not running for project 'proj_fcc375d3401b'"}` |
| New `claude.exe` visible in `tasklist`? | No (only lingering PID 21404 from prior runs) |

Bug reproduces with the same 60-72s hang + "not running" pattern observed in the deep diagnosis. **Phase 1 gate passes — proceed to Tier 3 instrumentation.**

---

## 2. Phase 2 trace — instrumented dispatch

`tier3_instrument.py` monkey-patches `SubprocessCLITransport.connect`/`write`, `anyio.open_process` (during connect), and `Query.start`/`initialize`/`_send_control_request`. Loaded via single import in `agent_os/api/app.py`. All trace lines flushed to stderr with monotonic timestamps prefixed `[T3 <ts> <name>]`.

Daemon was restarted under env `-u CLAUDECODE -u CLAUDE_CODE_ENTRYPOINT`. Six `BOOT` lines confirmed all five SDK patch points engaged. Dispatch fired against a fresh project (`proj_3fd8b0f00940`), same workspace.

**Full captured trace (verbatim):**

```
[T3 0.000 BOOT] tier3_instrument loaded (monotonic t0 captured)
[T3 0.617 BOOT] patched _sub_cli.anyio.open_process
[T3 0.617 BOOT] patched SubprocessCLITransport.connect
[T3 0.617 BOOT] patched SubprocessCLITransport.write
[T3 0.617 BOOT] patched Query.start / Query.initialize / Query._send_control_request
[T3 0.617 BOOT] tier3 patches complete
[T3 20.892 TRANSPORT.connect] ENTER
[T3 20.892 SUBPROC.spawn] PRE open_process argv0='C:\\Users\\qiren\\AppData\\Roaming\\npm\\claude.CMD' argv_len=2 kwargs=['stdout', 'stderr']
[T3 20.900 SUBPROC.spawn] POST pid=21800
[T3 21.019 SUBPROC.spawn] PRE open_process argv0='C:\\Users\\qiren\\AppData\\Roaming\\npm\\claude.CMD' argv_len=14 kwargs=['stdin', 'stdout', 'stderr', 'cwd', 'env', 'user']
[T3 21.026 SUBPROC.spawn] POST pid=5912
[T3 21.026 TRANSPORT.connect] EXIT pid=5912
[T3 21.027 QUERY.start] ENTER
[T3 21.027 QUERY.start] EXIT
[T3 21.027 QUERY.initialize] ENTER
[T3 21.027 QUERY.ctrl_req] ENTER subtype=initialize timeout=60.0
[T3 21.027 TRANSPORT.write] ENTER size=113
[T3 21.027 TRANSPORT.write] POST
[T3 81.040 QUERY.ctrl_req] EXC subtype=initialize Exception: Exception('Control request timeout: initialize')
[T3 81.040 QUERY.initialize] EXC Exception: Exception('Control request timeout: initialize')
```

**Trace observations (annotated):**

- `t=20.892` — first event after daemon boot. The ~20s gap is the orbital-side work *before* SDK invocation (sub-agent inheritance: `setup_engine.check_all()`, `render_sub_agent_prompt`, `ensure_memory_md`, `_maybe_emit_claudemd_warning`, then `_resolve_transport`, `CLIAdapter` construction, lock acquisition). Matches prior instrumented runs.
- `t=20.892 → 20.900` — version-probe spawn of `claude.CMD` with 2 argv (likely `claude.CMD --version`). 8ms. Returns PID 21800. Process exits on its own; not the SDK session.
- `t=21.019 → 21.026` — **actual SDK session spawn.** `claude.CMD` with 14 argv (full options: `cwd`, `env`, etc.), stdin/stdout/stderr piped. **Returns PID 5912 in 7ms** — `claude.exe` **DID spawn successfully**.
- `t=21.026` — `TRANSPORT.connect EXIT pid=5912` confirms anyio's `open_process` and the SDK transport's connect both completed.
- `t=21.027` — `Query.start` / `Query.initialize` enter; `_send_control_request` enters with `subtype=initialize` and 60.0s timeout.
- `t=21.027` — **`TRANSPORT.write` 113 bytes (the `control_request {"subtype":"initialize",...}` JSON) ENTER and POST in <1ms**. The stdin write *succeeded immediately*.
- `t=21.027 → 81.040` — **60.013s of silence.** No further `[T3 ...]` event. claude.exe was alive (PID 5912 in the process tree during this window) but emitted nothing on its stdout that the SDK would interpret as a `control_response`.
- `t=81.040` — exactly at the 60.0s timeout, `_send_control_request` raises `Exception('Control request timeout: initialize')`. The exception propagates up through `Query.initialize` and ultimately back to orbital's auto-start path, which returns the "agent not running" string on the retry-`send()`.

Total dispatch time on the curl side: 70s (matches setup ~10s + 60s SDK timeout).

> Side note: prior un-instrumented attempts reported "no new `claude.exe` spawned" via `tasklist`. The Tier 3 trace shows claude.exe DID spawn (PID 5912) — it likely exited quickly after the SDK closed stdin on timeout, so `tasklist` polls between dispatches caught it gone. The trace's evidence is authoritative; the prior observation was a polling artifact, not the symptom it appeared to be.

---

## 3. Phase 3 classification

Per the interpretation table in `TASK-windows-tier3-trace-capture.md` §"Phase 3":

| Captured pattern | Maps to |
|---|---|
| `[T3 21.026 SUBPROC.spawn] POST pid=5912` (spawn succeeded) | row 2 condition 1 ✅ |
| `[T3 21.027 TRANSPORT.write] POST` (stdin write succeeded) | row 2 condition 2 ✅ |
| 60.013s silence (21.027 → 81.040) | row 2 condition 3 ✅ |
| `[T3 81.040 QUERY.ctrl_req] EXC subtype=initialize Exception('Control request timeout: initialize')` | row 2 condition 4 ✅ (TimeoutError-equivalent: claude-agent-sdk wraps the timeout in a plain `Exception` rather than `asyncio.TimeoutError`, but the semantic is identical — 60s ctrl-request timeout exceeded) |

**Classification: W1 — CLI-side stall.** claude.exe spawned and accepted stdin, but never wrote `control_response` to its stdout under whatever combination of CLI args / env / context orbital is dispatching with.

Pattern matches row 2 exactly; not row 1 (W2), not row 3 (spawn EXC), not row 4 (no BOOT). Not novel.

---

## 4. Fix recommendation

**Tier 3 trace confirms W1; apply `FIX-PROPOSAL.md` Option 1 as written. No deviations.**

The fix changes `agent_os/agent/transports/sdk_transport.py` to pass `system_prompt` as the SDK's preset/append dict (`{"type": "preset", "preset": "claude_code", "append": <text>}`) instead of a plain `str`. Per `claude_agent_sdk/_internal/transport/subprocess_cli.py:170-180`, the dict form maps to `--append-system-prompt <text>` (APPEND); the plain string form maps to `--system-prompt <text>` (REPLACE — wipes claude-code's default system prompt). The W1 hypothesis is that claude.exe with REPLACE semantics on Windows never finishes its bootstrap to the point of responding to the SDK's `control_request initialize`.

The trace does not directly prove "switching to APPEND restores the response" — proof requires applying the fix and re-running. But it does eliminate the alternative classification (W2 / Python-side spawn block), which would have required a different fix entirely (Windows event loop policy or thread executor). Confidence on Option 1 is now bounded by the verification step, not by the failure-mode hypothesis: the spawn, write, and timeout markers all confirm we're in CLI-side stall territory.

---

## 5. Reproduction commands

These are the exact commands executed to produce the trace. Run from a shell where the daemon is launched with `CLAUDECODE` / `CLAUDE_CODE_ENTRYPOINT` stripped (the daemon child's env must not flag it as "running inside Claude Code", though Tier 2 already ruled that out as the cause — same hygiene applies here for cleanliness).

```bash
# --- Pre: integrated worktree on test/full-integration @ a55a28c
cd D:\orbital-public\.claude\worktrees\test-full-integration
git rev-parse --short HEAD    # → a55a28c

# --- Stage Tier 3 harness in the worktree root (so the relative import in app.py resolves)
cp /tmp/orbital-tier4/tier3_instrument.py ./tier3_instrument.py
# (or rebuild from this file — full source committed at docs/investigations/TRACE-windows-dispatch-bug.md
#  appendix is NOT included; see /tmp/orbital-tier4/tier3_instrument.py for the canonical copy)

# --- Add ONE import block as the FIRST executable statements in agent_os/api/app.py
#     (after the module docstring, before all other imports):
#
#       import sys as _t3_sys, os as _t3_os
#       _t3_sys.path.insert(0,
#           _t3_os.path.dirname(_t3_os.path.dirname(_t3_os.path.dirname(__file__))))
#       import tier3_instrument  # noqa: F401

# --- Boot daemon with CLAUDECODE stripped (daemon stderr goes to /tmp/daemon-phase2.log)
nohup env -u CLAUDECODE -u CLAUDE_CODE_ENTRYPOINT \
  python -m uvicorn agent_os.api.app:create_app --factory --port 8000 --host 0.0.0.0 \
  > /tmp/daemon-phase2.log 2>&1 &

# --- Wait for ready
for i in $(seq 1 20); do
  curl -s -m 1 http://127.0.0.1:8000/api/v2/settings >/dev/null 2>&1 && break
  sleep 1
done

# --- Confirm 6x [T3 ... BOOT] lines visible in daemon stderr
grep -E '\[T3 .* BOOT\]' /tmp/daemon-phase2.log

# --- Create a project pointing at D:\repro-smoke (seeded with orbital/PROJECT_STATE.md etc.)
PROJ=$(curl -s -m 10 -X POST http://127.0.0.1:8000/api/v2/projects \
  -H "Content-Type: application/json" \
  -d '{"name":"phase2-traced","workspace":"D:\\repro-smoke",
       "model":"kimi-k2.5","api_key":"","base_url":"https://api.moonshot.cn/v1",
       "provider":"moonshot","sdk":"openai","autonomy":"hands_off",
       "agent_slug":"built-in","disabled_sub_agents":[]}')
PID=$(echo "$PROJ" | python -c "import json,sys;print(json.load(sys.stdin)['project_id'])")

# --- Fire the failing dispatch; expect ~70-72s elapsed, "not running" response
time curl -s -m 90 -X POST "http://127.0.0.1:8000/api/v2/agents/$PID/inject" \
  -H "Content-Type: application/json" \
  -d '{"content":"What is the current state of this project? Be concise.","target":"claude-code"}'

# --- Capture full trace
grep -E '\[T3 ' /tmp/daemon-phase2.log
```

**Expected trace pattern (must match the verbatim trace in §2 to count as a confirmed W1 reproduction).**

### Cleanup (run after capture)

```bash
# Stop daemon
for pid in $(netstat -ano 2>/dev/null | grep ':8000 ' | grep LISTEN | awk '{print $NF}'); do
  taskkill /F /PID "$pid"
done

# Remove instrumentation
cd D:\orbital-public\.claude\worktrees\test-full-integration
# Revert app.py (remove the four-line Tier 3 block)
git checkout -- agent_os/api/app.py
rm tier3_instrument.py
git diff --quiet && echo "clean" || echo "WARNING: residual diff present"
```

Canonical copy of `tier3_instrument.py` is preserved at `/tmp/orbital-tier4/tier3_instrument.py` for future Tier 3 runs.
