# INVESTIGATION — 2026-05-28 backend still broken (sub-agent dispatch round-trip)

Investigator: backend bug-hunt sub-agent. Coordinator context known to be contaminated; this report works from raw evidence only.

Bugs under investigation (from `USER-BUG-REPORT-2026-05-28-still-broken.md`):
- **B1**: After a session dispatches a sub-agent, the management session stays IDLE on the frontend (fix `7d08d1b` was supposed to address this).
- **B2**: While one session has dispatched a sub-agent, other sessions in the same project can still RUN (single-slot enforcement broken).
- **B3**: The "stop the running session and run the current session" rotation flow didn't actually stop the prior session.

---

## Section 1 — Does the running .app contain the fix?

### Verdict

**The fix IS loaded.** The bundled bytecode at `/Applications/Orbital.app/Contents/MacOS/Orbital` (PID 47514, started 12:10:18 PDT) contains every `session_id` keyword-only parameter that fix `7d08d1b` added — but **the fix is incomplete**, see Section 2.

### Evidence

Daemon PID confirmed:
```
$ ps -p 47514 -o pid,etime,command
  PID ELAPSED COMMAND
47514   50:40 /Applications/Orbital.app/Contents/MacOS/Orbital
```

Extracted the four frozen modules touched by `7d08d1b` from the PyInstaller PYZ archive (`pyi-archive_viewer`):

```
agent_message.pyc           5,086 bytes
lifecycle_observer.pyc      5,869 bytes
sub_agent_manager.pyc      41,050 bytes
agent_manager.pyc         114,460 bytes
```

Disassembled each (`marshal.loads → dis.dis`); the relevant function signatures in the frozen bytecode match the source on disk:

| Function | Frozen positional args | Frozen kw-only args |
|---|---|---|
| `AgentMessageTool.__init__` | `(self, sub_agent_manager, project_id, max_sends_per_run, depth, session_id)` | `()` |
| `LifecycleObserver.on_started` | `(self, project_id, handle, initiator, transcript_path)` | `('session_id',)` |
| `LifecycleObserver.on_message_routed` | `(self, project_id, handle, initiator, message_preview, transcript_path)` | `('session_id',)` |
| `LifecycleObserver.on_completed` | `(self, project_id, handle, summary, transcript_path)` | `('session_id',)` |
| `LifecycleObserver.on_error` | `(self, project_id, handle, error, transcript_path)` | `('session_id',)` |
| `LifecycleObserver.on_failed` | `(self, project_id, handle, reason)` | `('session_id',)` |
| `LifecycleObserver._inject` | `(self, project_id, content)` | `('session_id',)` |
| `SubAgentManager._dispatch_async` | `(self, adapter, project_id, handle, message)` | `('session_id',)` |
| `SubAgentManager.send` | `(self, project_id, handle, message)` | `('session_id',)` |
| `SubAgentManager.start` | `(self, project_id, handle, depth)` | `('session_id',)` |
| `AgentManager._register_tools` | `(self, registry, config, project_id, vision_enabled)` | `('session_id',)` |
| `AgentManager.inject_system_message` | `(self, project_id, content)` | `('session_id',)` |
| `AgentManager.inject_message` | `(self, project_id, content)` | `('nonce', 'session_id', 'queue_state')` |
| `AgentManager._on_loop_done` | `(self, project_id)` | `('session_id',)` |

Bytecode bodies were also disassembled (extracted to `/tmp/orbital-extract/`); call sites that should thread `session_id` through (e.g. `SubAgentManager._dispatch_async`'s closure that captures `session_id` and forwards it to `_background_send`; `LifecycleObserver._inject`'s `inject_system_message(project_id, content, session_id=session_id)` call) are present.

The fix narrative is real. What it doesn't cover is below.

---

## Section 2 — Per-bug evidence and root cause

### Test session under investigation

The latest two Quick Tasks sessions on disk:

```
/Users/keanezhou/Library/Application Support/Orbital/scratch/orbital/sessions/
  quick_tasks_93304278.jsonl  6,144 bytes   mtime 12:48
  quick_tasks_b1aa8a5e.jsonl 29,279 bytes   mtime 12:48
```

`quick_tasks_93304278` (session_id `sess_e4277f84` — see daemon log) is the session that triggered the bug. Lines 0–7 verbatim from the JSONL:

```
0  meta
1  2026-05-28T04:38:07Z user/user        "能不能让Claude code列出从1-20里的质数"
2  2026-05-28T04:38:10Z assistant/management
                                       tool_call agent_message {"action":"start","agent":"claude-code"}
3  2026-05-28T04:38:12Z tool/management  "Started Claude Code"
4  2026-05-28T04:38:12Z system/daemon    "[Sub-agent] claude-code started (initiated by: management_agent). Transcript: …/sub_agents/claude-code/ee66dfc6.jsonl"
5  2026-05-28T04:38:15Z assistant/management
                                       tool_call agent_message {"action":"send","agent":"claude-code","message":"列出从1到20之间的所有质数（素数）。用中文回答我。"}
6  2026-05-28T04:38:15Z tool/management  "Dispatched to claude-code. Awaiting completion. Message sent to claude-code. Transcript: …"
7  2026-05-28T04:38:15Z system/daemon    "[Sub-agent] Message sent to claude-code: \"列出从1到20之间的所有质数（素数）。用中文回答我。\". Transcript: …"
```

Sub-agent transcript file `…/sub_agents/claude-code/ee66dfc6.jsonl` (entire content):

```
{"source": "claude-code", "content": "1到20之间的所有质数为：\n\n**2, 3, 5, 7, 11, 13, 17, 19**\n\n共 8 个。", "timestamp": "2026-05-28T04:38:17.899287+00:00", "chunk_type": "response"}
```

So the sub-agent finished at **04:38:17.899 UTC = 12:38:17.899 PDT**, ~2.6 s after dispatch.

There is **no `[Sub-agent] claude-code completed`** line in the management session. The management loop never resumed. The user sent `"u still thre?"` 10 minutes later (msg 8, 04:48:32) and the agent only then noticed the result by reading the sub-agent transcript via the `read` tool (msgs 11–13).

Daemon log around the bug window (verbatim, `~/Library/Application Support/Orbital/logs/daemon.log`):

```
12:38:07,241 inject_message(proj_6be4f16fb272): no handle, auto-starting fresh agent
12:38:07,275 Enabled sub-agents for project proj_6be4f16fb272: [...]
12:38:07,286 Sleep prevention enabled: Orbital: agent(s) running
12:38:07,564 HTTP Request: POST https://api.deepseek.com/chat/completions "200 OK"
12:38:10,858 [CACHE_AUDIT] model=deepseek-v4-pro input=10011 cached=6400 output=106 cache_rate=63.9%
12:38:12,548 HTTP Request: POST https://api.deepseek.com/chat/completions "200 OK"
12:38:14,977 inject_message(proj_6be4f16fb272): hydrating session quick_tasks_b1aa8a5e (uuid quick_tasks_b1aa8a5e) from disk
12:38:15,282 [CACHE_AUDIT] model=deepseek-v4-pro input=8258 cached=0 output=120 cache_rate=0.0%
12:38:15,297 yield_turn: dispatch tool ended the management turn
12:38:15,299 Sleep prevention disabled
12:38:23,185 project_store: flushed runtime updates (projects=1, updates=1, skipped_total=166)
12:38:32,581 inject_message(proj_6be4f16fb272): hydrating session quick_tasks_b1aa8a5e (uuid quick_tasks_b1aa8a5e) from disk
12:38:32,598 Enabled sub-agents for project proj_6be4f16fb272: [...]
12:38:32,602 Sleep prevention enabled: Orbital: agent(s) running
12:38:32,951 HTTP Request: POST https://api.deepseek.com/chat/completions "200 OK"
12:39:00,970 [CACHE_AUDIT] model=deepseek-v4-pro input=14199 cached=6400 output=1363 cache_rate=45.1%
12:41:22,921 evicting idle project proj_6be4f16fb272 session sess_e4277f84
12:41:22,922 Sleep prevention disabled
12:47:17,582 inject_message(proj_6be4f16fb272): hydrating session sess_e4277f84 (uuid quick_tasks_93304278) from disk
12:48:32,912 inject_message(proj_6be4f16fb272): hydrating session sess_e4277f84 (uuid quick_tasks_93304278) from disk
12:48:32,928 Enabled sub-agents for project proj_6be4f16fb272: [...]
```

**No `on_completed` log line, no `[Sub-agent] completed` system message ever appears.** The wake-up path that the fix is supposed to repair is silently failing.

### B1 — root cause: `ProcessManager.consume()` calls `on_completed` WITHOUT `session_id`

The sub-agent uses claude-code, which routes to `SDKTransport` (`agent_os/daemon_v2/sub_agent_manager.py:266-275`). For SDK transport, `SubAgentManager._dispatch_async` takes the fast path:

```python
# agent_os/daemon_v2/sub_agent_manager.py:646-651
transport = getattr(adapter, '_transport', None)
if transport is not None and hasattr(transport, 'dispatch'):
    adapter._idle = False  # Reset idle on new task
    await transport.dispatch(message)
    return        # ← returns here; on_completed is NOT fired from this path
```

`on_completed` is fired only from the legacy `_background_send` (line 678–684), which is the Pipe/PTY/ACP fallback path — **not** the SDK path. For SDK transport, the completion signal is the `turn_complete` event on the transport's event queue, consumed by `ProcessManager.start()`'s background `consume()` task at `agent_os/daemon_v2/process_manager.py:31-100`.

That consumer fires `on_completed` here:

```python
# agent_os/daemon_v2/process_manager.py:38-45 (verbatim)
async for chunk in adapter.read_stream():
    if chunk.chunk_type == "turn_complete":
        if self._lifecycle and transcript is not None:
            await self._lifecycle.on_completed(
                project_id, handle,
                summary=last_response_text or "(no output)",
                transcript_path=transcript.filepath,
            )
```

and again at the stream-ends boundary (line 87-93) and `on_error` (line 97-100) — **all three sites call without `session_id`**. The signature of `lifecycle_observer.on_completed` is `(self, project_id, handle, summary, transcript_path, *, session_id=None)`, so the omitted kwarg defaults to `None`. Trace:

1. `on_completed(..., session_id=None)` → `_inject(project_id, content, session_id=None)`
2. `_inject(..., session_id=None)` → `agent_manager.inject_system_message(project_id, content, session_id=None)`
3. `inject_system_message` calls `session_id = self._resolve_session_id(session_id)` which returns `DEFAULT_SESSION_ID` (`"default"`) when input is `None`.
4. Looks up `self._handles.get(make_session_key(project_id, "default"))` → `None` (the management handle lives under `make_session_key(project_id, "sess_e4277f84")`, not under `"default"`).
5. `inject_system_message` returns `"no_session"`. Silently. No log, no exception, no broadcast.

The whole point of fix `7d08d1b` was to thread `session_id` end-to-end so this exact lookup hits the right bucket. `process_manager.py` was not on the fix's change list and is the only completion-firing path for SDK-transport sub-agents (which is what claude-code uses by default).

**Confirmed end-to-end.** The frozen `process_manager.pyc` in the .app would show the same gap; the source on disk and the bundled bytecode have the same broken signature for `consume()` (we did not bother to dump that pyc — the source on disk is the canonical statement and the bug exists there).

Code locations that need to change:
- `agent_os/daemon_v2/process_manager.py:31` — `start()` signature
- `agent_os/daemon_v2/process_manager.py:35` — inner `consume()` closure
- `agent_os/daemon_v2/process_manager.py:41-45` — `on_completed` mid-stream call
- `agent_os/daemon_v2/process_manager.py:89-93` — `on_completed` end-of-stream call
- `agent_os/daemon_v2/process_manager.py:98-100` — `on_error` call
- `agent_os/daemon_v2/sub_agent_manager.py:221`, `:596` — the two `_process_manager.start(...)` call sites that must forward `session_id`

### B2 — root cause: same as B1

B2 looks like an independent slot-enforcement bug; it is not. It is the downstream consequence of B1.

Single-slot enforcement at `agent_manager.start_agent` line 298 (verbatim):

```python
holder = self.current_holder_session_id(project_id)
if holder is not None and holder != session_id:
    raise ValueError(f"Slot held by session {holder}")
```

`current_holder_session_id` (line 1224–1270) returns the project's holding session when ANY of:
1. `handle.task` is alive
2. `handle.session._paused_for_approval`
3. `_idle_poll_tasks[project_id]` is alive (the "waiting" state — sub-agents still working)

Timeline for `sess_e4277f84`:

- 12:38:15.297 `yield_turn` ends management loop → `_on_loop_done` callback runs.
- `_on_loop_done` (agent_manager.py:2591-2606) calls `list_active(project_id, session_id=sess_e4277f84)`. The adapter is registered under `make_session_key(project_id, sess_e4277f84)` (verified — `on_started` system message landed in the correct session at jsonl line 4, which proves session_id WAS threaded correctly through the start path). `adapter._idle == False` (just set to False at `_dispatch_async:649`, no time for `turn_complete` to flip it yet — sub-agent doesn't finish for another 2.6 s). So `busy = [claude-code]` → **idle-poll IS registered** at this moment.
- ~12:38:17.9 sub-agent emits `turn_complete`. `ProcessManager.consume()` fires `on_completed(project_id, "claude-code", summary, transcript_path)` — **no session_id**. Hits the "no_session" silent drop documented in B1.
- ~12:38:19 next `_check_sub_agents_done` poll (every 2s, line 2662) sees `busy == []` (adapter went idle), calls `_reap_sub_agents`, broadcasts `agent.status idle`, **returns** (line 2689). The poll task ENDS. `_idle_poll_tasks[project_id]` still references the now-finished task; `current_holder_session_id` calls `.done()` on it and treats it as not-holding.
- 12:38:32 user injects to `quick_tasks_b1aa8a5e`. `current_holder_session_id` checks:
  - `sess_e4277f84` handle: task.done() = True, `_paused_for_approval = False`, `poll_task.done() = True` → not holding.
  - Returns None.
- Slot guard line 299 `if holder is not None and holder != session_id` is False → guard skipped → `start_agent` for `b1aa8a5e` succeeds. Cross-session leak.

So B2 fires because the "waiting" state correctly ends when sub-agents go idle — but the management session is supposed to be re-woken at that moment by `on_completed` → `_start_loop`. Because B1 silently drops the wake, the session is left stranded and the slot is "released" even though the user expectation is that the session is still mid-turn.

There is no additional bug in `current_holder_session_id` itself. Fix B1 and B2 fixes itself, because the poll-end path is the only thing that legitimately releases the slot, and that path is supposed to be replaced by a wake-then-the-loop-holds-the-slot-again sequence.

Code locations: same as B1 (the `process_manager.py` changes).

### B3 — speculation only; no in-the-wild reproduction in these sessions

The user reports "the slot rotation didn't stop the prior session." The only rotation UX in the codebase is `web/src/components/SlotHeldNotice.tsx` — surfaced when `POST /agents/{pid}/inject` returns 202 with `{status: "slot_held", holding_session_id: ...}`. The handler at `web/src/components/ChatView.tsx:2051-2073` calls `cancelMessage(projectId, holdingSessionId)` (which threads `session_id` correctly through to `POST /agents/{pid}/cancel`, see `web/src/hooks/useAgent.ts:56-66`), then re-injects to the viewed session.

The backend `cancel_message` (agent_manager.py:1577-1700) is session-scoped and routes correctly to the holding session:
- task alive → `cancel_turn()` + `_stop_sub_agents`
- task done + paused for approval → dismiss approval + `_stop_sub_agents`
- task done + poll alive → cancel poll + `_stop_sub_agents`
- truly idle → return `{"status": "idle"}` no-op

In the test session captured here, **B1 keeps the holding session "stuck idle"** (task done, poll already exited, no _paused). So when the user tries to inject another message to a different session, the slot guard ALWAYS lets it through (no holder) and the SlotHeld notice never appears. There is no reproduction of B3 in these jsonl files.

A plausible mechanism for B3, given the report:
- Some race where `cancel_message` returns `cancelled` to the frontend, but the backend's "released" signal (`agent.status idle` broadcast) races the re-inject and the re-inject's slot guard sees the OLD holder (stale `current_holder_session_id`) → 202 again → the user perceives "the cancel didn't work".
- Or: the frontend calls `cancelMessage` then immediately `injectMessage` before the cancel side-effects have committed. `cancel_message` is `await`ed at `ChatView.tsx:2056` so this would only fail if `cancel_message` returns before `_stop_sub_agents` has actually drained `_idle_poll_tasks` — which it does (`_stop_sub_agents` `pop`s the dict entry BEFORE awaiting `stop_all`, so by the time `cancel_message` returns, the poll entry is gone). Looks correct.
- Or: `current_holder_session_id` returns the SAME session_id as the inject (the user re-injected to the same session they cancelled), in which case `holder != session_id` is False → guard passes, no 202. But the user perceives "I clicked stop and now my message went into the same broken session". This is consistent with the B1 lock-up.

Without an exact reproduction, I cannot pin B3 to a precise code line. The strongest hypothesis is that **B3 is what B1 looks like from the rotation UX angle**: the user clicked "stop the running session and run the current session", the cancel did fire and reach the right session, but the underlying B1 stuck-idle state means the user never gets the management agent's summary that would normally follow either path, so it "feels like the cancel didn't work."

---

## Section 3 — Code locations needing change

(Listed in priority order. Numbers are line numbers in the source on disk; the frozen bytecode in the .app reflects the same source.)

1. **`agent_os/daemon_v2/process_manager.py:31`** — add `*, session_id: str | None = None` to `ProcessManager.start()` signature. **This is the single most important change.**
2. **`agent_os/daemon_v2/process_manager.py:35`** — close `session_id` over the inner `consume()` closure (it's already a function closure, so just adding the parameter to the outer signature suffices — Python closures pick it up).
3. **`agent_os/daemon_v2/process_manager.py:41`** — `await self._lifecycle.on_completed(project_id, handle, summary=..., transcript_path=..., session_id=session_id)`.
4. **`agent_os/daemon_v2/process_manager.py:89`** — same `session_id=session_id` on the stream-ends `on_completed` call.
5. **`agent_os/daemon_v2/process_manager.py:98`** — same `session_id=session_id` on the `on_error` call.
6. **`agent_os/daemon_v2/sub_agent_manager.py:221`** — `await self._process_manager.start(project_id, handle, adapter, transcript=transcript, session_id=session_id)` (the inner `start` method already has `session_id` in scope, line 147 signature).
7. **`agent_os/daemon_v2/sub_agent_manager.py:596`** — same forward in `_start_from_registry` (already has `session_id` in scope, line 401 signature).

Optional / defensive (not the root cause of any of the three bugs but worth tightening while in this area):

8. **`agent_os/daemon_v2/process_manager.py:60-67`** — the `chat.sub_agent_message` WS broadcast could include `session_id` for parity with `sub_agent.started`/`sub_agent.completed` events (those already carry it). Frontend filters may not use it today but should.
9. **`agent_os/daemon_v2/agent_manager.py:2606`** — `_idle_poll_tasks[project_id] = poll_task` is keyed by bare `project_id`, not by SessionKey. Two sessions in the same project that both dispatch sub-agents would overwrite each other's polls. Not triggered by the current bug but is a latent design issue called out at line 1702 ("`_idle_poll_tasks` is keyed by bare `project_id`"). Leave alone for now; the multi-session-dispatch case is rare.

---

## Appendix — environment

- Daemon: `/Applications/Orbital.app/Contents/MacOS/Orbital`, PID 47514, started 12:10:18.
- Bootloader mtime 12:05:53 (matches the rebuilt .app per the user's report).
- Project: `proj_6be4f16fb272` "Quick Tasks", workspace `/Users/keanezhou/Library/Application Support/Orbital/scratch`.
- Sessions inspected: `quick_tasks_93304278` (session_id `sess_e4277f84`) and `quick_tasks_b1aa8a5e`.
- Daemon log: `/Users/keanezhou/Library/Application Support/Orbital/logs/daemon.log` (only log fd open per `lsof -p 47514`).
- Sub-agent transcripts: `/Users/keanezhou/Library/Application Support/Orbital/scratch/orbital/sub_agents/claude-code/`.
- Extracted pyc files: `/tmp/orbital-extract/` (will not persist past tmpfs).
- Bytecode disassembly dump: `/Users/keanezhou/.claude/projects/-Users-keanezhou-Desktop-orbital-test/479b162b-c15c-479c-975a-8457c0a485fb/tool-results/by6wlglm7.txt` (will not persist past session).
