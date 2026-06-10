# DEEP-DIAGNOSIS — Sub-Agent Dispatch Round-Trip Bug

**Date:** 2026-05-11
**Investigator host:** macOS arm64 (Darwin 24.6.0), Python 3.13.7, claude-agent-sdk 0.1.48, claude-code CLI 2.1.138.
**Builds on:** `DIAGNOSIS-dispatch-roundtrip-bug.md` (commit `7bc9ef4` on `feature/render-chat-variant-a`; not present on `test/full-integration`).
**Status:** **Investigation incomplete.** Platform-agnostic tiers executed; Windows-side tiers (1 & 2) deferred by agreement. Strong evidence the bug is Windows-specific; root cause not definitively isolated. Next decisive step is running the included Tier 3 instrumentation on the Windows machine where the hang reproduces.

---

## Scope reduction

Per agreement with the requester (macOS environment cannot reproduce the Windows-only manual smoke), the following tiers from the task spec were re-scoped:

| Tier | Spec status | Executed here |
|---|---|---|
| 1 — bisect within spec 3 by sub-step | **Skipped.** Requires reproducing the hang, which the prior diagnosis verified only on Windows. | No |
| 2 — asyncio context capture (orbital vs probe) | **Skipped.** Same reason — both contexts must be running on the failing host to produce a meaningful diff. | No |
| 3 — SDK-internal trace via monkey-patch | **Built and smoke-tested.** Ready to deploy on Windows. | Partial — harness verified working; Windows execution pending |
| 4 — minimal FastAPI repro outside orbital | **Fully executed.** Five distinct scenarios run on macOS. | Yes |
| 5 — classification + fix proposal | Drives this document and `FIX-PROPOSAL.md`. | Yes |

---

## Tier 4 — minimal FastAPI repro on macOS

### Setup

- Workspace: `/tmp/orbital-tier4/workspace/` with `orbital/PROJECT_STATE.md` seeded (mirrors the diagnosis's repro env).
- Real orbital prompt rendered via `agent_os.agent.sub_agent_prompt.render_sub_agent_prompt` — **1756 characters**, matching the spec's "~2080 char" ballpark within ±20%.
- All scenarios launched with `env -u CLAUDECODE -u CLAUDE_CODE_ENTRYPOINT -u CLAUDE_CODE_SESSION_ID -u CLAUDE_CODE_EXECPATH -u CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS -u CLAUDE_EFFORT python3 …` to neutralize Claude Code loop-prevention.
- A defensive `os.environ.pop(…)` in `/tmp/orbital-tier4/probe_common.py` re-strips inside the Python process (the SDK at `subprocess_cli.py:346-348` re-spreads `os.environ` over the user-supplied `options.env`, so the strip must be at both layers).

### Scenarios run

| # | Mode | Calling context | `system_prompt` shape | `can_use_tool` | `stderr` | Result |
|---|---|---|---|---|---|---|
| A | `scenario_a_asyncio.py` | `asyncio.run(main())` | bare-str (1756 chars) | — | callback | ✅ **6.25s** total (4.33s connect) |
| B | `?mode=barestr` | FastAPI / uvicorn | bare-str | — | — | ✅ **6.77s** (4.09s connect) |
| C | `?mode=preset` | FastAPI / uvicorn | `{"type":"preset","preset":"claude_code","append":<str>}` (Option 1) | — | — | ✅ **7.09s** (4.34s connect) |
| D | `?mode=stderr` | FastAPI / uvicorn | bare-str | — | callback | ✅ **7.49s** (4.39s connect) |
| E1 | `?mode=barestr_canusetool` | FastAPI / uvicorn | bare-str | callback | callback | ✅ **6.10s** (4.28s connect) — **closest analogue to orbital's actual SDK call shape** |

**Decisive result:** None of the five scenarios reproduces the 60-72 s hang documented in the prior diagnosis. Scenario E1 in particular replicates orbital's exact `ClaudeAgentOptions` shape (bare-str `system_prompt` + `can_use_tool` callback) inside a FastAPI/uvicorn request handler and completes in 6.1 s end-to-end.

### What Tier 4 rules out

The combination "asyncio task context of a FastAPI/uvicorn request handler + bare-string `system_prompt` + `can_use_tool` permission callback" — i.e. all the orbital-specific surface that the prior diagnosis flagged — is **not sufficient on its own** to produce the hang. Specifically:

1. **Hypothesis A from prior diagnosis (asyncio context interaction under uvicorn)** is not falsified outright, but the minimal FastAPI repro shows that *generic* FastAPI/uvicorn context does not break the SDK on this host. If A is the cause, the trigger must be more specific — e.g. uvicorn's `loop="asyncio"` (default) vs an explicit `loop="uvloop"`, or a particular event-loop policy that orbital sets, or interaction with other long-lived async tasks the orbital daemon holds open (WebSocket manager, message bus). My test does not run those.
2. **The `--system-prompt` (replace) vs `--append-system-prompt` (append) CLI semantic difference** does not, by itself on this host, change whether the SDK hangs. Both shapes succeed in 6.0–7.1 s. So if Option 1 from the prior diagnosis is the correct fix, it must be the correct fix *only on Windows*. (It could still be the correct fix — just not for a reason this macOS evidence demonstrates.)
3. **Stderr-pipe blocking under uvicorn** (one strong candidate from the SDK-internals research) does not manifest here — Scenario B succeeds without a stderr callback, meaning uvicorn's inherited stderr fd accepts whatever claude-code writes during init.

### What Tier 4 still admits as possible

- Windows-only subprocess/IO machinery in `anyio.open_process` (Selector vs Proactor event-loop interaction; pipe buffer differences; child-process stdio handle inheritance).
- Windows-only behavior in `claude.exe` itself when invoked with `--system-prompt <long text>` (the REPLACE flag) under inherited environment from uvicorn — possibly the CLI takes a slow path that does not complete within 60 s.
- An orbital-specific concurrent async task (WebSocket pump, message-bus subscription, periodic timer) that runs in the daemon's event loop alongside `_start_from_registry` and that my 50-line FastAPI app does not have. This is plausible but speculative without Tier 3 trace data from Windows.

---

## Tier 3 — SDK-internal instrumentation (built, smoke-tested, pending Windows execution)

### Harness

`/tmp/orbital-tier4/tier3_instrument.py` — a monkey-patch module that wraps the four SDK call sites identified during research:

| Tracepoint | Wraps | Logs |
|---|---|---|
| `SUBPROC.connect` enter/exit | `SubprocessCLITransport.connect` | both ends of the spawn sequence |
| `SUBPROC.spawn` PRE/POST/EXC | `anyio.open_process` (replaced for the duration of `connect` only) | argv summary, env subset, returned pid; any exception |
| `QUERY.start` enter/exit/EXC | `Query.start` | reader task-group entry/exit |
| `QUERY.init` enter/exit/EXC | `Query.initialize` | full initialize handshake bracket |
| `QUERY.ctrl_req` PRE/POST/EXC | `Query._send_control_request` | subtype, request_id, timeout |
| `TRANSPORT.write` PRE/POST/EXC | `self.transport.write` | byte count + first 200 chars of payload |

All traces go to stderr with `flush=True` and monotonic timestamps, so the last line emitted before a 60 s silence pinpoints which SDK await is blocking.

### Smoke test result (macOS)

```
[T3 …345 BOOT] claude_agent_sdk version=0.1.48
[T3 …357 SUBPROC.connect] enter
[T3 …357 SUBPROC.spawn] PRE open_process argv0~='claude -v' (2 argv)             ← version check
[T3 …360 SUBPROC.spawn] POST pid=65083                                            ← 3 ms
[T3 …723 SUBPROC.spawn] PRE open_process argv0~='claude --output-format stream-json...' (12 argv)
[T3 …728 SUBPROC.spawn] POST pid=65090                                            ← 5 ms
[T3 …729 SUBPROC.connect] exit
[T3 …729 QUERY.start] enter
[T3 …729 QUERY.start] exit
[T3 …729 QUERY.init] enter
[T3 …729 QUERY.ctrl_req] PRE subtype=initialize request_id=req_1_… timeout=60.0
[T3 …729 TRANSPORT.write] PRE bytes=113 preview='{"type":"control_request",...}'
[T3 …729 TRANSPORT.write] POST                                                    ← stdin write returned immediately
[T3 …654.034 QUERY.ctrl_req] POST                                                 ← waited 4.3 s for claude.exe to emit control_response
[T3 …654.034 QUERY.init] exit
```

So on macOS the entire handshake is: spawn (8 ms total) → write to stdin (sub-ms) → **wait 4.3 s for control_response on stdout** → done.

### Predicted Windows trace (if hang behaves as prior diagnosis described)

Same sequence up through `TRANSPORT.write POST` — then 60 s silence — then either `QUERY.ctrl_req EXC TimeoutError` or `QUERY.init EXC`. This would pinpoint the hang to **the wait-for-control-response phase**, after a successful spawn and stdin write. If this prediction matches, the bug is "claude.exe is alive but never emits its initialize control_response within 60 s" — a CLI-side stall, not a Python-side stall. The most parsimonious cause is the `--system-prompt` (REPLACE) code path in the CLI's initialize sequence under Windows, which Option 1 would side-step by switching to `--append-system-prompt`.

If the Windows trace instead shows the hang BEFORE `SUBPROC.spawn POST` (i.e., `anyio.open_process` never returns), the cause is Python-side: Windows asyncio subprocess support failing under uvicorn's event loop. This would NOT be fixed by Option 1 and would require a different remediation path (e.g., switching the SDK transport to `subprocess.run` via a thread executor, or forcing ProactorEventLoop).

The instrumentation is small, removable, and platform-portable. **Deploying it to the Windows daemon is the highest-value next investigation step.**

---

## Cross-tier classification

Per the classification table from the task spec:

| Pattern | Status | Action |
|---|---|---|
| Tier 1 isolates to single orbital sub-step + Tier 2 shows context asymmetry | **Cannot conclude** — Tiers 1 & 2 not run. | — |
| Tier 1 shows interaction effect + Tier 3 shows SDK fails inside its own task group | **Cannot conclude.** | — |
| Tier 4 reproduces in minimal FastAPI app | **No.** Five scenarios all pass on macOS. | This rules out a generic FastAPI/asyncio explanation. |
| None of the tiers reproduce / isolate | **Yes, on this host.** | Escalate to Tier 3 on Windows; do not ship inheritance until resolved. |

**Final classification:** **Investigation insufficient to declare wiring-bug vs SDK-bug with certainty.** The evidence we DO have:
- Spec 3 introduces the regression. (Pre-existing bisect; not re-verified here.)
- The minimal cross-platform repro (FastAPI + orbital's exact `ClaudeAgentOptions` shape) is **clean** on macOS.
- Therefore the proximate trigger is Windows-specific and/or orbital-specific in a way not covered by the bare FastAPI test.

---

## Open hypotheses, ranked by remaining plausibility

| # | Hypothesis | Plausibility after Tier 4 macOS evidence | Falsifiable by |
|---|---|---|---|
| **W1** | Windows claude.exe under `--system-prompt` (REPLACE) takes >60 s on the CLI side to emit `control_response`, but under `--append-system-prompt` (APPEND) it's fast. | **High.** Matches the symptom (60 s timeout exactly), explains why PipeTransport works (uses `--append-system-prompt-file`), explains why the standalone Windows probe with bare-str + `asyncio.run` succeeds (different ambient context — possibly different inherited stderr / handle inheritance changes the CLI's slow path). | Apply Option 1 on Windows; run repro. If <10 s success → W1 confirmed. |
| **W2** | Windows `anyio.open_process` under uvicorn's event loop fails or hangs differently than under `asyncio.run`, because of Selector vs Proactor loop policy. | **Medium.** Consistent with the prior diagnosis's "no new claude.exe spawned" observation, IF that observation was real (Tier 3 trace will confirm or deny). | Tier 3 trace on Windows: if `SUBPROC.spawn POST` never fires, W2 is confirmed. |
| **W3** | Orbital-specific concurrent async task (WebSocket pump, message bus, periodic check) leaks an unfinished coroutine into the event loop that interferes with the SDK's anyio task group. | **Low-medium.** Not falsified by anything yet, but my FastAPI test had none of those and still didn't repro — so this hypothesis is necessary if Windows-only factors are insufficient. | Build a "FastAPI + orbital's daemon startup" repro on Windows that mimics the daemon's other long-lived tasks, OR delete those tasks from a scratch worktree and re-test. |
| **W4** | argv length limit on Windows console for `--system-prompt <1756 char>` invocation via the SDK's spawn. | **Low.** A 1756-char argv element is well under Windows's 32,767 `CreateProcess` cap. Possible if claude.exe's own argv parsing has a tighter limit, but no evidence. | Tier 3 trace: if `SUBPROC.spawn` raises a `ValueError`/`OSError`, W4 is in play. |

W1 is the best bet, and Option 1 is its direct remediation. **But this is still inference, not proof.** Without Windows Tier 3 evidence, the fix proposal cannot be 100% confident.

---

## What `DEEP-DIAGNOSIS` cannot tell you

- The exact line of the hang on Windows.
- Whether the prior diagnosis's "no new claude.exe in tasklist" observation is literally true or an artifact of tasklist snapshot timing (claude.exe may have spawned briefly, written nothing, and exited or stalled silently).
- Whether Option 1 fixes the bug. The macOS evidence shows Option 1 works there, but the macOS test also shows bare-str works there, so this is non-informative for Windows.

The honest path forward is the Tier 3 trace on Windows, then the fix.
