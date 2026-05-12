# DIAGNOSIS — Sub-Agent Dispatch Round-Trip Bug

**Investigation:** TASK-investigate-dispatch-roundtrip-bug.md
**Date:** 2026-05-11
**Status:** Diagnosis complete. Bug is real and reproducible. Not fixed (per spec).
**Tester branch state:** all four spec branches present; investigation done on `worktree-agent-a9e24fbde848229ba` (spec 3 alone) and `feature/render-chat-variant-a` (parent baseline).

---

## 1. Bisect result

| Branch tested | Commit | Dispatch result | Spawn? |
|---|---|---|---|
| `feature/render-chat-variant-a` (parent, no specs) | `63140eb` | ✅ **PASS** — 5s round-trip; full response with `LSE-2026-04` and `HMAC` content from `DECISIONS.md` | new claude.exe spawned |
| `worktree-agent-af380bbefe331d1fb` (spec 2 alone) | tip `43e0ef6` | ✅ **PASS** — 5s round-trip; "Message sent to claude-code" ack | new claude.exe spawned |
| `worktree-agent-a9e24fbde848229ba` (spec 3 alone) | **`cbfb8ca`** | ❌ **FAIL** — 71-72s hang, returns `"Error: agent 'claude-code' not running for project '...'"` | **no new claude.exe spawn** |
| `integration/subagents-exposure` (specs 2+3 auto-merged) | `7d35b9f` | ❌ **FAIL** — same 71s hang, same error | no new claude.exe spawn |

**Verdict: the regression is introduced by spec 3's commit `cbfb8ca` ("feat(sub-agents): inherit project context via system prompt + lazy MEMORY.md").** The auto-merge into `integration/subagents-exposure` is innocent (merge interaction ruled out — Lead 2 in the task spec). Spec 2 alone is innocent.

All four specs sit on the `feature/render-chat-variant-a` head (`63140eb`); each branch produces 1 to 4 commits on top of `b445986` (a slightly earlier point), but the bisect was decisive at the branch granularity — no sub-commit bisect needed because spec 3 has only a single commit.

---

## 2. Loop-prevention finding (Tier 2)

**The bug is NOT environmental loop-prevention.** Tested explicitly:

- The daemon was started with `env -u CLAUDECODE -u CLAUDE_CODE_ENTRYPOINT python -m uvicorn ...` (both Claude-Code env vars stripped). Dispatch on spec 3 STILL fails with the same 71s hang.
- A standalone Python probe (`/tmp/sdk-probe/probe_orbital.py`) running with the SAME `claude-agent-sdk` (v0.1.48), the SAME rendered orbital system_prompt (2080 chars), the SAME workspace, with `env={"CLAUDECODE": ""}` mimicking orbital's subprocess env — **WORKS**. Returns `PONG-XYZ` from claude-code in ~8s, with tool_use blocks (`Read PROJECT_STATE.md`, `Read SESSION_LOG.md`) firing as expected.

So `claude-agent-sdk` is **not refusing** to start because the daemon's parent process is Claude Code. The SDK's spawn path itself works fine when called in isolation. The bug is inside orbital's orchestration around the SDK call.

Lead 1 from the task spec (Claude Code SDK loop-prevention) is **conclusively ruled out.**

---

## 3. Code-level root cause (Tier 3)

### Hang location: `await adapter.start(config)` inside `_start_from_registry`

I added temporary `print(f"[DBG] ...", file=sys.stderr, flush=True)` instrumentation around every step in `agent_os/daemon_v2/sub_agent_manager.py:_start_from_registry()` and re-ran a failing dispatch. Trace from a 72s-hang dispatch (UNIX timestamps):

```
[DBG] start_from_registry ENTER ... t=1778429694.395
[DBG] before registry.get t=1778429694.395
[DBG] after registry.get manifest=True t=1778429694.395
[DBG] before get_adapter_config t=1778429694.395
[DBG] before configure_network t=1778429694.411
[DBG] before check_all t=1778429694.412
[DBG] after check_all t=1778429696.035       ← check_all() took 1.6s; not the cause
[DBG] before ensure_memory_md t=1778429696.035
[DBG] after ensure_memory_md t=1778429696.035
[DBG] before render_prompt t=1778429696.035
[DBG] before claudemd_warning t=1778429696.036
[DBG] after claudemd_warning t=1778429696.037
[DBG] before resolve_transport t=1778429696.037
[DBG] after resolve_transport t=1778429696.424 transport_type=SDKTransport
[DBG] before CLIAdapter ctor t=1778429696.424
[DBG] after CLIAdapter ctor t=1778429696.424
[DBG] got lock obj t=1778429696.424
[DBG] inside lock t=1778429696.424
[DBG] before adapter.start t=1778429696.424   ← HANGS HERE
                                              (no further output for 60+ seconds)
```

`await adapter.start(config)` → `CLIAdapter.start()` → `transport.start()` → `ClaudeSDKClient.connect()` → `Query.start()` then `Query.initialize()`.

`Query.initialize()` sends a `control_request` of subtype `initialize` and awaits the response with a hard floor of **60 seconds**:

```python
# claude_agent_sdk/client.py:155
initialize_timeout = max(initialize_timeout_ms / 1000.0, 60.0)
```

```python
# claude_agent_sdk/_internal/query.py:158-160
response = await self._send_control_request(
    request, timeout=self._initialize_timeout
)
```

The 60s timeout matches the 71-72s end-to-end (12s for setup before adapter.start + 60s SDK initialize timeout = 72s).

### What does NOT happen during the hang

- **`claude.exe` is never spawned.** The pre-existing claude.exe (from earlier test runs) stays at unchanged memory; no new process appears in `tasklist | grep claude.exe`.
- **No daemon log entries** (after `Daemon file logging enabled at orbital-data\logs\daemon.log` startup banner) — no exceptions, no warnings, just silence for 60s, then dispatch handler returns `"not running"`.

### What I CAN'T pinpoint conclusively within scope

The hang is inside **`ClaudeSDKClient.connect()` → `transport.connect()` → `anyio.open_process(...)`** — the call to spawn claude.exe. But:

- My standalone probe with the same SDK + same options + same env spawns claude.exe in ~5s (not 60s).
- Orbital's spec 3 path doesn't spawn claude.exe at all in 60s.

The asymmetry is real but not fully explained by spec 3's surface diff. Spec 3's only relevant SDK-path change in `sdk_transport.py` is:

```python
# Spec 3:
options_kwargs: dict = dict(
    cwd=workspace, permission_mode="default",
    can_use_tool=self._handle_permission, cli_path=command or None, env=sdk_env,
)
if self._system_prompt is not None:
    options_kwargs["system_prompt"] = self._system_prompt
options = ClaudeAgentOptions(**options_kwargs)
```

vs. parent which never set `system_prompt` at all. Passing `system_prompt=<plain str>` does NOT itself break the SDK (probe verified — the same plain string works in isolation).

Hypotheses I could not narrow further within the diagnosis budget:
- **Hypothesis A:** Some asyncio task-group / context interaction inside FastAPI/uvicorn's event loop affects how `anyio.open_process()` behaves when called transitively via the SDK, in a way that doesn't repro under a plain `asyncio.run()`. The SDK uses anyio task groups internally (`Query.start` → `anyio.create_task_group()` at `_internal/query.py:165`); orbital's dispatch is invoked from within an HTTP request handler whose surrounding task context differs from a plain probe.
- **Hypothesis B:** The argv length of `claude` invocation when `--system-prompt <2080-char-text>` is passed via SDK exceeds some Windows console limit on the SDK's spawn path specifically (different from when invoked directly by my probe). Spec 3 is the only branch passing `--system-prompt` via SDK.
- **Hypothesis C:** Spec 3's earlier work — `setup_engine.check_all()` invocation, `ensure_memory_md` filesystem activity, or `_maybe_emit_claudemd_warning` WS broadcast — leaves event-loop state (a pending callback or unfinished task) that interferes with the subsequent SDK spawn. Tracing showed each of these sub-steps completes in milliseconds, but a stale anyio task group could still be at fault.

Of these, **(A) is the most plausible** because:
1. The standalone probe works — meaning the SDK-as-API isn't broken.
2. The orbital flow with the SAME SDK call doesn't even reach claude.exe spawn — meaning something in the calling context blocks the spawn.
3. Both check_all() and the inheritance block run inside the same async function as adapter.start() — Python `print()` confirms each completes in milliseconds, but there's no easy way to confirm no anyio leak without deeper instrumentation.

**Recommendation: try Hypothesis B as the cheapest test before deeper task-group debugging** — switch SDKTransport to write the prompt to a temp file and pass `extra_args={"append-system-prompt-file": <path>}` (the `extra_args` field on `ClaudeAgentOptions` allows passing arbitrary CLI flags) instead of `system_prompt=<str>`. PipeTransport already uses this exact pattern; mirroring it on the SDK side is the smallest delta from spec 3. If that fixes dispatch, B was correct. If it still fails, A is more likely.

---

## 4. Fix proposal

**Concrete recommendation, minimum-delta:**

In `agent_os/agent/transports/sdk_transport.py`, replace this block:

```python
options_kwargs: dict = dict(
    cwd=workspace, permission_mode="default",
    can_use_tool=self._handle_permission, cli_path=command or None, env=sdk_env,
)
if self._system_prompt is not None:
    options_kwargs["system_prompt"] = self._system_prompt
options = ClaudeAgentOptions(**options_kwargs)
```

with one of these two alternatives, in priority order:

**Option 1 (recommended — preset/append dict):** the SDK accepts a `{"type": "preset", "preset": "claude_code", "append": "<text>"}` dict, which maps to `--append-system-prompt <text>` (instead of `--system-prompt <text>`, which is REPLACE semantics). This preserves Claude Code's default system prompt and only appends the orbital inheritance. Code:

```python
if self._system_prompt is not None:
    options_kwargs["system_prompt"] = {
        "type": "preset", "preset": "claude_code",
        "append": self._system_prompt,
    }
```

**Option 2 (file-based, mirroring PipeTransport):** write the prompt to a temp file under `{workspace}/orbital/.tmp/` and pass `extra_args={"append-system-prompt-file": <abs-path>}`:

```python
extra_args = dict(options_kwargs.get("extra_args") or {})
if self._system_prompt is not None:
    path = self._write_temp_prompt_file()  # mirror PipeTransport's helper
    extra_args["append-system-prompt-file"] = path
options_kwargs["extra_args"] = extra_args
```

Either option:
- ✅ Restores the Claude Code default system prompt (so the agent has its real tool/behavioral framing).
- ✅ Appends the orbital inheritance template instead of replacing it.
- ✅ Aligns SDK transport with the (already-working) PipeTransport semantics. Spec 3 split implementations between SDK/Pipe and chose plain-string `system_prompt` for SDK — that string-shape triggers REPLACE semantics in `claude-agent-sdk` (verified at `_internal/transport/subprocess_cli.py:170-180`). Whether that REPLACE alone is the operational cause or merely correlated with the hang, this fix removes a spec-level deviation between transports.

**Both options are diagnosis-grade hypotheses, not verified fixes.** The spec instructed "do NOT attempt to fix" — implementation + verification is a separate dispatch.

**Risk note for the fix dispatch:** if neither option resolves the dispatch round-trip, the cause is Hypothesis A (asyncio context interaction) and deeper instrumentation is required — try `anyio.run_in_subprocess` for the SDK call, or replace SDKTransport with PipeTransport for claude-code on this code path (PipeTransport works on parent + spec 2 alone with the same project shape).

---

## 5. Reproduction steps

Exact, verified to reproduce on this machine (Windows 10, Python 3.13, claude-agent-sdk 0.1.48, claude-code 2.1.138):

```bash
cd /d/orbital-public/.claude/worktrees/agent-a9e24fbde848229ba    # spec 3 alone
git rev-parse HEAD                                                # confirm cbfb8ca

# Provide settings.json so daemon has Kimi/Moonshot for management agent
mkdir -p orbital-data
# (Copy or write a settings.json with provider=moonshot, model=kimi-k2.5,
#  base_url=https://api.moonshot.cn/v1, plus a valid api_key. The bug is
#  in the sub-agent dispatch path — the management LLM choice is
#  irrelevant; any provider that can boot the daemon is fine.)

# Boot daemon WITHOUT inherited Claude Code env vars (rules out loop-prevention)
nohup env -u CLAUDECODE -u CLAUDE_CODE_ENTRYPOINT \
  python -m uvicorn agent_os.api.app:create_app --factory --port 8000 --host 0.0.0.0 \
  > /tmp/daemon.log 2>&1 &
sleep 4
curl -s http://127.0.0.1:8000/api/v2/settings >/dev/null   # confirm up

# Seed a project workspace with orbital/PROJECT_STATE.md (any content; presence
# of orbital/ is what makes spec 3 render a non-None system_prompt)
mkdir -p /tmp/repro/orbital
echo "Status: 60% complete on widget-foo." > /tmp/repro/orbital/PROJECT_STATE.md

# Create project
PROJ=$(curl -s -X POST http://127.0.0.1:8000/api/v2/projects \
  -H "Content-Type: application/json" \
  -d '{"name":"repro","workspace":"C:\\repro","model":"kimi-k2.5","api_key":"",
       "provider":"moonshot","sdk":"openai","autonomy":"hands_off",
       "agent_slug":"built-in","disabled_sub_agents":[]}')
PID=$(echo "$PROJ" | python -c "import json,sys;print(json.load(sys.stdin)['project_id'])")

# Dispatch and time it
time curl -s -m 90 -X POST "http://127.0.0.1:8000/api/v2/agents/$PID/inject" \
  -H "Content-Type: application/json" \
  -d '{"content":"hi","target":"claude-code"}'
# Expected: ~71-72 second hang, then:
#   {"status":"Error: agent 'claude-code' not running for project 'proj_XXX'"}
# Concurrently, `tasklist | grep claude.exe` shows no NEW claude.exe.
# Concurrently, orbital-data/logs/daemon.log has no application-level entries
# beyond the daemon-start banner.

# Compare against parent branch (works):
cd /d/orbital-public                                              # 63140eb (parent)
# Re-do the daemon start + project create + dispatch sequence above.
# Result: ~5 second response with "Message sent to claude-code. Transcript: ...".
# tasklist shows a new claude.exe spawned. orbital-data/logs/daemon.log has
# Sandbox account / Sleep prevention / network config entries.
```

**Standalone SDK probe (proves SDK is fine in isolation):**

```bash
mkdir -p C:\Users\qiren\AppData\Local\Temp\sdk-probe
mkdir -p C:\Users\qiren\AppData\Local\Temp\repro\orbital  # any seeded workspace
# Write probe.py (see /tmp/sdk-probe/probe_orbital.py for content) which
# instantiates ClaudeSDKClient with system_prompt=<orbital-rendered-text>
# and dispatches "Reply with PONG-XYZ".
env -u CLAUDECODE -u CLAUDE_CODE_ENTRYPOINT \
  python C:\Users\qiren\AppData\Local\Temp\sdk-probe\probe_orbital.py
# Expected: CONNECTED ~5s, query() returns, tool_use Read PROJECT_STATE.md fires,
# text PONG-XYZ-equivalent response, total ~8s.
```

**Instrumentation reproduction** (to find hang location yourself):

Apply the following Edit to `agent_os/daemon_v2/sub_agent_manager.py:_start_from_registry()` (insert `print(f"[DBG] before adapter.start t={time.time():.3f}", file=sys.stderr, flush=True)` immediately before `await adapter.start(config)`, and `[DBG] after adapter.start ...` immediately after). Restart daemon, dispatch, and observe via `grep "\[DBG\]" /tmp/daemon.log` that `before adapter.start` prints but `after adapter.start` never does. Remove instrumentation when done.

---

## 6. Implications for the four specs

| Spec | Implementation status (per smoke test report) | Blocked on this bug? | Action |
|---|---|---|---|
| 1 (ACP cleanup) | ✅ correctly implemented; smoke test verified the rejection path | ❌ no — doesn't depend on dispatch round-trip | Ready to merge as-is. |
| 2 (settings page) | ✅ correctly implemented; settings persistence + UI confirmed | Partially — "dispatch uses opus model" sub-check is BLOCKED | Ready to merge: the only blocked sub-check is downstream verification, not implementation. Re-verify after dispatch fix. |
| 3 (worker memory inheritance) | ✅ MEMORY.md lazy creation + system_prompt rendering + CLAUDE.md banner all correctly implemented | ✅ **yes — this spec authored the regression** | **Spec 3's SDKTransport changes need revision** (per fix proposal §4). Implementation otherwise correct. Re-verify dispatch round-trip after revision. |
| 4 (memory viewer) | ✅ correctly implemented; GET/PUT API + UI confirmed | Partially — "next dispatch references new memory entry" sub-check is BLOCKED | Ready to merge: same as spec 2; downstream sub-check only. |

**None of the four specs needs implementation revision EXCEPT spec 3's SDK transport `system_prompt` shape.** The MEMORY.md lifecycle, prompt renderer, CLAUDE.md detection, and Pipe transport injection in spec 3 are all correctly implemented and aren't part of the regression. The fix is narrow: change how spec 3 hands the rendered prompt to `claude-agent-sdk` (Option 1 or 2 in §4 above).

---

## Cleanup performed

- Temporary `print()` instrumentation in `sub_agent_manager.py` was applied during diagnosis and **restored** to spec 3's original (`cbfb8ca`) state. Verified clean: `git diff --stat agent_os/daemon_v2/sub_agent_manager.py` returns nothing on `worktree-agent-a9e24fbde848229ba`.
- All test daemon instances stopped (`netstat -ano | grep ':8000 '` returns no listeners).
- Standalone probe artifacts at `/tmp/sdk-probe/probe_orbital.py` and `rendered_prompt.txt` left in place for future fix-verification (the probe demonstrates the SDK works in isolation; useful as a regression baseline).
- Test workspaces under `/tmp/orbital-smoke-spec3` and `/tmp/orbital-smoke-spec3-fresh` unchanged (preserved for the fix dispatch).
- One orphan claude.exe process (PID 21404, ~500 MB) lingers from earlier failed attempts — `taskkill /F /PID 21404` if it bothers the user.
- No code committed; all branches are at their pre-investigation HEADs.

---

## Confidence summary

| Conclusion | Confidence |
|---|---|
| Regression introduced by spec 3 alone, not the auto-merge | High — bisect was decisive at branch granularity |
| Hang is in `await adapter.start(config)` → SDK `Query.initialize()` 60s timeout | High — instrumented trace + matching 60s timeout floor in SDK source |
| Bug is NOT environmental (loop-prevention via `CLAUDECODE`) | High — verified by `env -u CLAUDECODE` + working standalone probe |
| Exact triggering interaction in spec 3's code | Medium — narrowed to either string-shape `system_prompt` (REPLACE semantic) or asyncio context interaction; plain-string SDK call works in isolation |
| Recommended fix (Option 1: preset/append dict) resolves the hang | Medium — it removes the only spec-level asymmetry between SDK and Pipe transports, but the standalone probe with plain-string also worked, so the fix may not be load-bearing on its own. The fix dispatch should verify, not assume. |
