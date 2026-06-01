# FINDINGS — Reap confirmation (cross-task `disconnect()` swallow)

**Date:** 2026-05-31
**Type:** confirmation, runtime evidence. **No fix implemented.**
**Verdict (one line):** **HYPOTHESIS CONFIRMED** — `SDKTransport.stop()` swallows a
cross-task `RuntimeError` from the SDK's `Query.close()`, skipping `transport.close()`
(`process.terminate()`), leaving the claude subprocess alive.

---

## 1. Decision-matrix row that fired

> **Row 1:** `disconnect() raised: RuntimeError … exit cancel scope in a different task` →
> subprocess **alive** → **Hypothesis CONFIRMED.**

A control run (same-task `stop()`) fired the contrasting row — `disconnect() returned
cleanly` → subprocess **gone** — proving the cross-task exit is the *specific* trigger, not a
general "disconnect never terminates" failure.

---

## 2. `[REAP_DEBUG]` output — verbatim

### Repro: cross-task (connect in task A, stop in task B) — matches the daemon

```
[repro] task A: connected. transport.is_alive()=True
[repro] task A: turn_complete seen=True
[repro] SDK subprocess pid = 13381
[repro] ps BEFORE: 13381 13376 S 00:03 /Users/keanezhou/.claude/local/node_modules/.bin/claude --output-format stream-json --verbose --append-system-prompt
[repro] transport.is_alive() BEFORE = True

2026-05-31 19:03:20,350 WARNING agent_os.agent.transports.sdk_transport: [REAP_DEBUG] stop() entered: client=True alive=True
2026-05-31 19:03:20,352 WARNING agent_os.agent.transports.sdk_transport: [REAP_DEBUG] disconnect() raised: RuntimeError('Attempted to exit cancel scope in a different task than it was entered in')
Traceback (most recent call last):
  File ".../agent_os/agent/transports/sdk_transport.py", line 288, in stop
    await self._client.disconnect()
  File ".../claude_agent_sdk/client.py", line 487, in disconnect
    await self._query.close()
  File ".../claude_agent_sdk/_internal/query.py", line 666, in close
    await self._tg.__aexit__(None, None, None)
  File ".../anyio/_backends/_asyncio.py", line 794, in __aexit__
    return self.cancel_scope.__exit__(exc_type, exc_val, exc_tb)
  File ".../anyio/_backends/_asyncio.py", line 461, in __exit__
    raise RuntimeError(
RuntimeError: Attempted to exit cancel scope in a different task than it was entered in

2026-05-31 19:03:20,352 WARNING agent_os.agent.transports.sdk_transport: [REAP_DEBUG] stop() returning: alive=False
```

Interpretation: the RuntimeError is raised at `query.py:666` (`await self._tg.__aexit__`),
which is **before** `query.py:668` (`await self.transport.close()`) — the call that runs
`process.terminate()`. The `with suppress(anyio.get_cancelled_exc_class())` wrapping the
`__aexit__` does **not** catch it (a `RuntimeError` is not a `CancelledError`). It propagates
into `SDKTransport.stop()`'s `except Exception: pass`, is swallowed, and `stop()` returns
normally with `alive=False`. Terminate never runs.

### Control: same-task (connect + stop in one task)

```
[control] subprocess pid=14143 BEFORE: 14143 14135 S /Users/.../claude --output-format stream-json --verbose --append-...
[REAP_DEBUG] stop() entered: client=True alive=True
[REAP_DEBUG] disconnect() returned cleanly
[REAP_DEBUG] stop() returning: alive=False
[control] ps AFTER: (pid 14143 not found)
[control] >>> subprocess 14143 STILL ALIVE after same-task stop()? False
```

When `stop()` runs in the same task that entered the scope, `__aexit__` succeeds,
`transport.close()` is reached, and the claude subprocess is terminated.

---

## 3. `ps` / `pgrep` evidence — before vs. after `stop()`

**Cross-task repro** (`python` pid 13376 plays the daemon role):

| | SDK subprocess (claude) | result |
|---|---|---|
| BEFORE stop() | `13381 13376 S … claude --output-format stream-json …` | alive (PPID = python) |
| AFTER stop()  | `13381 13376 S … claude --output-format stream-json …` | **STILL ALIVE** (PPID unchanged, not reparented) |
| `child_pids(python)` AFTER | `['13381']` | still a live child |

`>>> SDK subprocess 13381 STILL ALIVE AFTER stop()? True`
(Killed afterward by the repro's own cleanup.)

**Control:** subprocess `14143` → `(pid 14143 not found)` after same-task stop → killed.

**Production corroboration (untouched):** the real daemon's orphan is identical in shape —
`58761 42328 S 1:06:02 … claude --output-format stream-json …` (PPID 42328 = the running
Orbital daemon, not reparented, not a zombie). Same end-state the repro produces.

---

## 4. Transport shape actually used

**SDK transport (persistent), confirmed.** All three processes (repro 13381, control 14143,
production 58761) launch claude with `--output-format stream-json --verbose
--append-system-prompt …` — the persistent SDK shape, **not** the one-shot `subprocess.run`
PipeTransport. `transport.is_alive()` returned `True` while idle-but-alive (SDK
`is_alive()==self._alive`), matching the leak condition. Pipe transport is not implicated
(its `is_alive()` is already `False` post-`subprocess.run`).

**Build staleness eliminated:** the repro ran against the **source tree**
(`python3` + `PYTHONPATH=<repo>`), and the `[REAP_DEBUG] stop() entered` line appeared in the
output — proving the instrumented `SDKTransport.stop()` was the code actually executing, not a
stale packaged copy.

---

## 5. Verdict & scope notes

- **Verdict: CONFIRMED.** The cross-task cancel-scope `RuntimeError` is raised inside
  `Query.close()` before `transport.close()`, swallowed by `SDKTransport.stop()`, and the
  claude subprocess survives. The control proves the cross-task exit (not disconnect itself)
  is the trigger.
- **Fidelity note (deviation from "run the full daemon"):** I reproduced at the
  `SDKTransport` level rather than booting the whole daemon-from-source. Justification: the
  daemon's orchestration reaching `stop_all` in a task *other* than the one that called
  `connect()` is already established (prior trace), and `CLIAdapter.stop()`/
  `SubAgentManager.stop()` are thin pass-throughs to `transport.stop()`. The focused repro
  exercises the exact code under test (real `SDKTransport.stop()`, real `claude_agent_sdk`,
  real claude subprocess) under the identical cross-task condition, with a control that
  isolates the mechanism — and the production orphan (PID 58761) corroborates the same
  end-state. Build staleness, the other stated risk, is independently eliminated above.
- **No fix proposed** (per task). The cause is now confirmed; the fix spec is a separate,
  gated task.

---

## Artifacts / local state (NOT committed)

- `agent_os/agent/transports/sdk_transport.py` — `[REAP_DEBUG]` instrumentation in `stop()`
  + `import traceback`. **Kept local, uncommitted** (removal decided after review, per task).
- `scripts/reap_repro.py` — cross-task repro (untracked).
- `/tmp/reap_control.py` — same-task control.
- Repro/control subprocesses (13381, 14143) were cleaned up. The production orphan **58761
  was NOT touched.**
