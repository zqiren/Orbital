# Investigation: Cancellation Architecture & Sub-Agent Lifecycle Gaps

**Branch:** `investigation/b1-b2-b5` (post-merge of `feature/2026-4-22-tasks` + `feature-workspace-file-structure-fix`)
**Method:** Read-only research across `agent_os/`, `web/src/`, tests, git history, and the orbital-marketing daemon log at `~/Library/Application Support/Orbital/logs/daemon.log`.
**Status:** No source files modified. No tests run.

This document answers the open design questions raised by the prior B1/B2/B5 investigation before fix specs are drafted. Citations are file:line throughout. Where an answer cannot be derived from the codebase, the response is "**Cannot determine from codebase**" rather than speculation.

## Headline findings (non-obvious results worth surfacing)

1. **The `asyncio.shield` motivation at `agent_manager.py:1206` is not recoverable from history** — the entire daemon was added in a single squashed `Initial release` commit (`9a6c882`, 2026-03-08). There is no pre-shield diff, no PR, no issue link, and no comment near the call. Any plan that drops the shield must justify it from current-code analysis alone, and must update **all four** identical sites (1086, 1168, 1206, 1325) in lockstep.

2. **The cli_adapter cancellation pattern is a good *partial* template for `loop.py`**, but a literal mirror leaves four Moonshot-specific gaps the template doesn't cover: partial-stream persistence into the JSONL, mid-stream tool-call boundaries, output-token billing on cancelled generations, and `httpx` connection-pool lifecycle.

3. **`AgentLoop.run()` captures the Session by value at construction (`loop.py:91`) and never re-reads `handle.session`**, so the loop survives Session swap by `new_session` only because the agent_manager waits for the old loop to die first — that is the load-bearing invariant.

4. **`loop.py:643` fires the session-end follow-up as `asyncio.create_task(...)` with no strong reference**, exactly reproducing the B5 fire-and-forget anti-pattern at a separate site that `AgentLoop.cancel()` would not catch.

5. **`SubAgentManager` already exposes `list_active`, `stop`, and `stop_all`** (sub_agent_manager.py:517, 424, 500). The watchdog at `agent_manager.py:1525-1532` does **not** call any of them — it just broadcasts `idle`. The "fix" here is using existing methods, not writing new ones. But `stop_all` itself has concurrency gaps: `send()` is lock-free and doesn't check `_stopping`; `stop_all` has no `wait_for` budget; the per-project lock is dropped between snapshot and per-handle stop.

6. **The Stop button already calls `POST /agents/{pid}/stop`, not `/inject`** (`web/src/hooks/useAgent.ts:35-40`). This contradicts the original B3 framing. The actual gap is product-shaped, not protocol-shaped: the only existing endpoint is a full teardown, and there is no verb meaning "interrupt this turn but keep the agent alive."

---

## Section 1 — asyncio.shield motivation at agent_manager.py:1206

### 1.1 Origin commit

**Direct answer:** The shield at `agent_os/daemon_v2/agent_manager.py:1206` (and three siblings at 1086, 1168, 1325) was added in commit `9a6c882` — the squashed initial public-release commit. No pre-shield diff, no linked PR/issue, no introducing commit message specific to the shield is recoverable.

**Evidence:**

```
$ git blame -L 1200,1215 agent_os/daemon_v2/agent_manager.py
^9a6c882 (Orbital 2026-03-08 18:08:07 +0800 1206)
    await asyncio.wait_for(asyncio.shield(handle.task), timeout=10.0)

$ git log --all --diff-filter=A --format='%H %ad %s' -- agent_os/daemon_v2/agent_manager.py
9a6c8827a85e587fd985e5067482add10773a45f  Sun Mar 8 18:08:07 2026 +0800
    Initial release: Orbital v0.1.0
```

The full commit message is `Initial release: Orbital v0.1.0 / An operating system for AI agents...` — no `#NNN`, no `PR-`, no `fixes`, no `closes`. The file is added wholesale at `+@@ 0,0 +1,1300`. `git log --all -S "asyncio.shield" -- agent_os/daemon_v2/agent_manager.py` returns only this commit.

**Implication for the proposed fix:** The plan to drop `asyncio.shield` cannot lean on commit-history reasoning. Whatever original-author intent existed was discarded by the public-release squash. Any justification must come from current-code analysis only.

---

### 1.2 All `asyncio.shield` usages in repo

**Direct answer:** Four call sites, all in `agent_os/daemon_v2/agent_manager.py`. Zero in tests. Zero elsewhere in `agent_os/`.

**Evidence:**

```
$ grep -rn "asyncio.shield" agent_os/ tests/
agent_os/daemon_v2/agent_manager.py:1086:  await asyncio.wait_for(asyncio.shield(handle.task), timeout=5.0)
agent_os/daemon_v2/agent_manager.py:1168:  await asyncio.wait_for(asyncio.shield(handle.task), timeout=5.0)
agent_os/daemon_v2/agent_manager.py:1206:  await asyncio.wait_for(asyncio.shield(handle.task), timeout=10.0)
agent_os/daemon_v2/agent_manager.py:1325:  await asyncio.wait_for(asyncio.shield(handle.task), timeout=10.0)
```

- `1086` (`approve`): drain previous loop task before recording an approval.
- `1168` (`deny`): same shape, deny path.
- `1206` (`new_session`): after `handle.session.stop()`, drain loop before pre-flush + session swap.
- `1325` (`stop_session`): after `handle.session.stop()`, drain loop before broadcasting `stopped`.

**Pattern:** every shield wraps the same object (`handle.task` = the `AgentLoop.run()` task), always after a cooperative-stop signal (`session.stop()` at lines 1204/1320), and always inside `asyncio.wait_for(...)` with a bounded timeout (5s or 10s). All four catch `(asyncio.TimeoutError, Exception)` and log a warning.

**Implication for the proposed fix:** Any change must update all four sites coherently — fixing only line 1206 leaves three identical risks behind. The shield is a uniform invariant, not a one-off.

---

### 1.3 Failure mode the shield guards against

**Direct answer:** **Cannot determine the original motivation from history or comments.** The semantic reading of the current code is clear, but the originating bug ticket / commit context is lost in the squash. There are no `# why shield` annotations and no docstrings explain it.

**Evidence (semantic, current code only):** The pattern at `agent_manager.py:1202-1208` is:

```python
1204:  handle.session.stop()                                  # cooperative signal
1205:  try:
1206:      await asyncio.wait_for(asyncio.shield(handle.task), timeout=10.0)
1207:  except (asyncio.TimeoutError, Exception):
1208:      logger.warning("new_session(%s): loop did not stop gracefully", project_id)
```

Without `shield`, when an *outer* request task is cancelled (FastAPI client disconnect, daemon shutdown, an outer `wait_for` higher up), `wait_for` would propagate cancellation **into** `handle.task`, killing the loop mid-iteration — likely between `_stream_response` and tool-result append, which would corrupt session state (orphaned tool calls without results, half-written JSONL). The shield converts "outer cancel + inner timeout" into "outer cancel waits, inner times out gracefully."

The candidates "ensuring cleanup completes despite caller cancellation" and "surviving an outer wait_for timeout" both fit; "partial-state corruption mid-rotation" is plausible for line 1206 specifically (it precedes session swap + workspace pre-flush).

**Implication for the proposed fix:** The proposed removal makes an inferential assumption that no outer cancellation can ever reach this stack. That assumption is not validated by code or tests — the fix spec must independently audit (a) FastAPI route handlers for `wait_for` wrappers around these calls, (b) the daemon shutdown handler in `app.py`, and (c) any caller that might cancel an approve/deny request mid-flight (e.g., a frontend that aborts a fetch).

---

### 1.4 Tests that pin shield-protected behavior

**Direct answer:** No test references `shield` directly (`grep -rn "shield" tests/` → empty). Tests pin the *observable effects* (loop drains, session created, status broadcast in order) rather than the mechanism.

**Evidence:**
- `tests/integration/test_new_session.py:97-104` `test_new_session_returns_ok_when_no_handle` — covers the no-handle short-circuit.
- `tests/integration/test_new_session.py:104-174` `test_new_session_creates_fresh_session` (line 165: `await mgr.new_session(project_id)`) — drives the full path with a mock loop task.
- `tests/integration/test_new_session.py:240-290` `test_broadcasts_new_session_then_idle` — pins the status-broadcast ordering that depends on the shield keeping the loop alive long enough.
- `tests/integration/test_new_session.py:297-351` `test_new_session_does_not_touch_other_project` — cross-project isolation.
- `tests/regression/test_new_session_feedback.py:58-85` — same observable-status pin.
- `tests/integration/test_stop_button_live.py:114` `test_stop_during_active_turn_returns_within_3s` — pins the stop_session path (line 1325 shield) under live-daemon conditions: must return within 5s with no `forcing idle` watchdog hit.
- `tests/integration/test_stop_button_live.py:170,202` — additional stop-path coverage.

The 4-22 branch's regression tests (`tests/regression/test_cli_adapter_stop_and_send.py`, 17 tests) target the **adapter-level** `_inflight_send` cancel — they are independent of `agent_manager`'s shield.

**Implication for the proposed fix:** Removing the shield will not fail any test by symbol. It will only fail tests if (1) a test injects an outer cancellation into the approve/deny/new_session/stop call (none currently do) or (2) removing it exposes a state-corruption symptom an integration test catches. CI is not protective here. The integration tests in `test_stop_button_live.py` are the only ones that exercise shield-line-1325 path under realistic concurrency; those should be re-run after any shield change.

---

## Section 2 — cli_adapter.py cancellation pattern (around line 320)

### 2.1 Reproduce the full pattern

**Direct answer:** Pattern lives at `agent_os/agent/adapters/cli_adapter.py:79` (state), `:168-171` (task creation), `:316-329` (cancel invocation in `stop()`), `:172-194` (exception handling), `:332-339` (post-cancel teardown order).

**Evidence:**

*Task creation/storage* — `cli_adapter.py:79` (state declaration) and `:164-171` (creation under `_send_lock`):

```python
79:  self._inflight_send: asyncio.Task | None = None  # active transport.send() task; cancelled on stop()

164:  async with self._send_lock:
165:      if self._transport:
166:          self._idle = False
167:          self._pending_response = True
168:          self._inflight_send = asyncio.create_task(
169:              self._transport.send(message),
170:              name=f"send-{id(self)}",
171:          )
```

*Cancel invocation* — `cli_adapter.py:316-329` (inside `stop()`):

```python
316:  async def stop(self) -> None:
317:      # Cancel any in-flight send BEFORE tearing down the transport so the
318:      # awaiting task releases _send_lock and doesn't wedge on a transport
319:      # that is about to disappear. Bounded 2s wait — stop() must be fast.
320:      if self._inflight_send is not None and not self._inflight_send.done():
321:          self._inflight_send.cancel()
322:          try:
323:              await asyncio.wait_for(self._inflight_send, timeout=2.0)
324:          except (asyncio.CancelledError, asyncio.TimeoutError):
325:              pass
326:          except Exception:
327:              logger.exception("inflight send raised during stop cancellation")
```

*Exception handling* — `cli_adapter.py:172-195`:

```python
172:  try:
173:      response = await self._inflight_send
174:      if response is not None: ...
178:      self._idle = True
179:  except asyncio.CancelledError:
180:      # Propagate cancellation — caller (typically stop()) is
181:      # tearing down. Do NOT log at ERROR; cleanup in finally.
182:      raise
183:  except Exception:
184:      logger.error("transport.send() raised for adapter %s", self.handle, exc_info=True)
185:      self._idle = True
186:      raise
192:  finally:
193:      self._inflight_send = None
194:      self._pending_response = False
```

*Post-cancel cleanup:* `_inflight_send = None` and `_pending_response = False` in the `finally` block (193-194), then `await self._transport.stop()` at line 332, then `_idle = False` at 333. **Order matters:** cancel-and-await *first*, transport stop *second*.

*Surrounding loop reaction:* the consumer is `SubAgentManager.dispatch()` / its background sender task (`_background_send_task`, line 78). When CancelledError propagates, normal-shutdown callers swallow it; transport-level `Exception` sets `self._broken = True` and broadcasts `LifecycleObserver.on_failed` — see commit `1782a8b` body ("AdapterBrokenError raised on reuse of broken adapters").

**Implication for the proposed fix:** All five facets of the pattern (storage / creation / cancel / exception / cleanup) are clean and well-tested. Mirroring is mechanically straightforward.

---

### 2.2 Minimum viable mirror for `loop.py` (Moonshot stream)

**Direct answer:** The general mechanics translate; four Moonshot-specific concerns do **not**.

**Evidence:**

*General parts that translate:*
- Strong-ref the in-flight task: `self._inflight_stream: asyncio.Task | None`.
- Cancel-and-await with bounded timeout (`wait_for(task, timeout=N)`).
- `try / except CancelledError: raise` in the awaiting code; reset state in `finally`.
- Order: cancel inflight stream **before** any client teardown.

*CLI-specific parts that do NOT translate:*
- Subprocess + ConPTY/PTY teardown at `_stop_via_provider` / `_stop_fallback` (cli_adapter.py:341-364): no equivalent for `httpx.AsyncClient`.
- `stdin` close, `process.terminate`/`kill`, process-group signaling: irrelevant for an HTTP POST.
- `_master_fd` close: N/A.

The Moonshot stream lives at `loop.py:118-128` as a bare `async for chunk in self._provider.stream(...)` — it is **not currently wrapped in a task at all**. The minimum viable mirror is to extract that body into a coroutine launched as a named task, with a sibling cancel method on `AgentLoop`.

*Moonshot-specific concerns NOT covered by the cli_adapter template:*

1. **Partial response in StreamAccumulator (`loop.py:124-128`)** — cli_adapter's transport returns a single response object atomically. Loop's accumulator is incremental; cancellation mid-stream drops partially-accumulated tokens without persisting them to the session JSONL.
2. **Mid-stream tool-call boundaries** — Moonshot streams interleave content deltas and `tool_calls` deltas. Cancelling between the tool-call header chunk and the arguments-complete chunk leaves a malformed tool call in `accumulator` that, if persisted, triggers an "orphaned tool result" cleanup later.
3. **Output-token billing already consumed** — Moonshot bills on tokens *generated*, not delivered. Cancelling mid-stream still costs money. The cost-tracking path (`_on_cost_update`, line 103) needs to know about the partial generation. cli_adapter has no billing surface.
4. **HTTP connection lifecycle** — `httpx.AsyncClient.stream(...)` requires the response context manager to be exited; raw cancellation may leak the connection into the pool. Closer analogue is the SDK transport's `_bg_task` cancel-then-`disconnect()` (`sdk_transport.py:247-266`), not cli_adapter.

**Implication for the proposed fix:** A correct port additionally needs to (a) persist accumulated content with a `cancelled=true` marker before discarding the task, (b) debit partial-token cost against the cost tracker, (c) ensure the `httpx` stream context exits cleanly (use `async with` *inside* the wrapped task, not outside), and (d) decide whether `provider.stream()` owns sub-tasks needing explicit teardown. The cli_adapter pattern is necessary but not sufficient.

---

### 2.3 Subtle invariants

| Invariant | Status | Evidence |
|---|---|---|
| `_inflight_send` always cleared before next dispatch | **Yes** | Set to `None` in `finally` at line 193; pinned by `tests/regression/test_cli_adapter_stop_and_send.py:244-286 test_inflight_send_cleared_in_all_paths` |
| Ordering with other state | **Yes** | `_send_lock` (line 76) serializes concurrent `send()`. `_pending_response` paired with `_inflight_send`. `_broken` flag (line 77) gates reuse. `stop()` order: cancel-inflight (320-325) → `_transport.stop()` (332) → `_idle = False` (333). Reversing would let the awaiting send observe a torn-down transport. |
| Concurrent `cancel()` from two callers safe | **Uncertain** | Guard `if self._inflight_send is not None and not self._inflight_send.done()` (line 320) is not atomic with `.cancel()` and `wait_for`. Two concurrent `stop()` callers (rapid double-click on Stop, or rotation racing manual stop) both pass the guard. `.cancel()` is idempotent (safe), but `self._transport.stop()` (332) is unguarded — two callers call it twice; safety depends on the transport. **No test covers concurrent `stop()`.** |
| CancelledError re-entry / child tasks | **Likely no** | `transport.send()` for SDK transport spawns its own `_bg_task` (sdk_transport.py:148); when cancelled, that bg task is cancelled in `transport.stop()` (sdk_transport.py:247-260), not by the cli_adapter cancel. So cli_adapter does **not** directly cancel grand-child tasks — it relies on the transport's `stop()` to do that. |

**Implication for the proposed fix:** A literal mirror in `loop.py` will leave Moonshot-specific gaps and concurrency-unsafe spots around concurrent cancel (Stop double-click). A correct port additionally guards `provider.stream()`'s connection cleanup explicitly and decides whether multi-cancel is benign or programming-error.

---

## Section 3 — AgentLoop state machine

### 3.1 All `session.is_stopped()` call sites in the daemon

**Direct answer:** Six call sites total — one in `loop.py`, four in `agent_manager.py`, one definitional. All are between-iteration / post-iteration polling — none attempt to interrupt an awaiting coroutine.

**Evidence:**

1. `agent_os/agent/loop.py:555` — between iterations, after the tool-execution batch:
   ```python
   if self._session.is_stopped():
       self._session.resolve_pending_tool_calls()
       break
   ```

2. `agent_os/daemon_v2/agent_manager.py:926` — `inject_message` "zombie" branch:
   ```python
   # Case 2: loop idle — append message and hot-resume
   if handle is not None and handle.session.is_stopped():
       del self._handles[project_id]
       handle = None
   ```

3. `agent_os/daemon_v2/agent_manager.py:1027` — `get_run_status`, observation only:
   ```python
   if handle.session.is_stopped():
       return "stopped"
   ```

4. `agent_os/daemon_v2/agent_manager.py:1426` — inside `_on_loop_done`, runs after the task finishes:
   ```python
   if handle.session.is_stopped():
       self._handles.pop(project_id, None)
       self._ws.broadcast(project_id, {... "status": "stopped"})
       return
   ```

5. `agent_os/daemon_v2/agent_manager.py:1507` — sub-agent watchdog (`_check_sub_agents_done`), polled every 2s between watchdog iterations:
   ```python
   handle = self._handles.get(project_id)
   if handle is None or handle.session.is_stopped():
       return
   ```

6. `agent_os/agent/session.py:309` — definition (not a call site).

**Implication for the proposed fix:** Every consumer treats `is_stopped()` as a *cooperative checkpoint* polled at safe boundaries. Adding `AgentLoop.cancel()` that cancels the inflight LLM stream task is **complementary**, not conflicting: the cancellation breaks the `_stream_response` await, and on re-entry the existing `is_stopped()` check at `loop.py:555` will exit the outer loop *if* `cancel()` also calls `session.stop()`. So `cancel()` should set `_stopped = True` *and* cancel the inflight task — both layers cooperate.

---

### 3.2 `Session.stop()` semantics

**Direct answer:** `stop()` is a flag-only setter — does **not** cancel any task; the Session holds **no** task reference. Contract is **cooperative**. No sub-states beyond binary `_stopped` plus orthogonal `_paused` and `_paused_for_approval`.

**Evidence:**

`agent_os/agent/session.py:303-310`:
```python
def stop(self) -> None:
    self._stopped = True

def is_stopped(self) -> bool:
    return self._stopped
```

`session.py:45-48`:
```python
self._paused: bool = False
self._stopped: bool = False
self._paused_for_approval: bool = False
```

`grep -n "_task\|_loop_task\|_inflight\|asyncio.Task" agent_os/agent/session.py` returned **zero matches** — no task reference is ever stored on a Session. The only `Lock` is `threading.Lock` for JSONL writes (`session.py:36-37`), not an asyncio primitive.

**No docstring** declares a contract, but the empirical contract is shown by every caller in `agent_manager.py` immediately following `session.stop()` with `await asyncio.wait_for(asyncio.shield(handle.task), timeout=10.0)` — i.e., callers know the flag alone won't end the loop. Examples: `agent_manager.py:1204-1208` (`new_session`) and `agent_manager.py:1320-1327` (`stop_agent`).

**Implication for the proposed fix:** Because `Session.stop()` is purely advisory and the loop only checks it between LLM/tool turns, an in-flight LLM stream (which can run 30–120s) cannot be interrupted today — `stop_agent`'s 10s shield/wait will routinely time out for slow streams. `AgentLoop.cancel()` must own the missing capability: (a) call `session.stop()` so post-iteration checks still trip, and (b) `task.cancel()` the inflight `_stream_response` to actually break the await. Session needs no new state — keep it the cooperative flag holder; put task ownership on the AgentLoop.

---

### 3.3 Session ↔ AgentLoop ↔ asyncio.Task wiring

**Direct answer:** Loop coroutine wrapped at two sites (`agent_manager.py:600-603` start_agent, `:1375-1377` _start_loop). Strong-ref lives on `ProjectHandle.task`. `inject_message` does NOT push messages into a queue the loop awaits — it appends to `session._queue` (a plain `list[tuple]`), polled at iteration top. There is no queue/event the loop blocks on. `new_session` swaps `handle.session` but the AgentLoop captured `self._session = session` at construction (`loop.py:91`) and never re-reads `handle.session`. After rotation, the live AgentLoop still references the *old* Session — benign today only because rotation waits for the old loop to die first.

**Evidence:**

Task creation in `start_agent` (`agent_manager.py:600-603`):
```python
task = asyncio.create_task(loop.run(initial_message, initial_nonce=initial_nonce))
task.add_done_callback(self._on_loop_done(project_id))
project_handle.task = task
```

And `_start_loop` hot-resume (`agent_manager.py:1375-1377`):
```python
task = asyncio.create_task(handle.loop.run())
task.add_done_callback(self._on_loop_done(project_id))
handle.task = task
```

Strong-ref location: `agent_manager.py:52`: `task: asyncio.Task | None`.

`inject_message` queue path (`agent_manager.py:823-827`):
```python
if handle.task is not None and not handle.task.done():
    # Case 1: loop running — queue for next iteration
    logger.info("inject_message(%s): loop running, queuing message", project_id)
    handle.session.queue_message(content, nonce=nonce)
    return "queued"
```

Loop drains queue at iteration head (`loop.py:202-218`):
```python
while True:
    queued = self._session.pop_queued_messages()
    if queued:
        for q_item in queued:
            ...
            self._session.append(q_msg)
```

`Session.queue_message` is just a list append (`session.py:261-263`) — no `asyncio.Event`, no `Queue.put`. The loop never `await`s on the queue.

`new_session` rotation (`agent_manager.py:1267-1268`):
```python
handle.session = new_session
handle.task = None
```

AgentLoop's stale-reference capture (`loop.py:91`):
```python
self._session = session
```

`grep -rn "self._session = " agent_os/agent/loop.py` shows only this single assignment — there is **no** mechanism for the loop to pick up the new Session after `handle.session` is swapped.

**Implication for the proposed fix:** `AgentLoop.cancel()` should cancel the loop's *own* task. Two options:
- **(a)** AgentLoop captures `asyncio.current_task()` at the top of `run()` into `self._task`, so `cancel()` can call `self._task.cancel()`.
- **(b)** agent_manager (already holds `handle.task`) does the cancel; AgentLoop exposes a coroutine `cancel()` wrapper that calls `session.stop() + task.cancel()`.

Either way, `cancel()` must clear `_paused_for_approval` too — otherwise the post-cancel cleanup at `loop.py:633-634` (`if not self._session._paused_for_approval: resolve_pending_tool_calls()`) will leak orphan tool_calls.

---

### 3.4 Call-site map for hypothetical `AgentLoop.cancel()`

**Direct answer:** Six places currently do some combination of `session.stop()` + `wait_for(shield(task))` or pop handles. Three would directly use `cancel()`; three are observers that don't need integration.

**Evidence and per-site analysis:**

**1. `new_session` rotation pre-cleanup — `agent_manager.py:1203-1208`** (use `cancel`):
```python
if handle.task is not None and not handle.task.done():
    handle.session.stop()
    try:
        await asyncio.wait_for(asyncio.shield(handle.task), timeout=10.0)
    except (asyncio.TimeoutError, Exception):
        logger.warning("new_session(%s): loop did not stop gracefully", project_id)
```
Replacement: replace `session.stop()` with `await handle.loop.cancel()` and keep the `wait_for(shield(...))` as a final reaper. **Ordering note:** still must `await` the task to finish before swapping `handle.session = new_session` at line 1267, otherwise the dying loop can write to the new session.

**2. `inject_message` zombie cleanup branch — `agent_manager.py:925-928`** (no change):
```python
if handle is not None and handle.session.is_stopped():
    del self._handles[project_id]
    handle = None
```
This runs only after the loop has already exited (otherwise case 1 at 823 would have queued the message). `cancel()` is **not** needed — the task is already done.

**3. `stop_agent` IPC handler — `agent_manager.py:1320-1327`** (use `cancel`, primary fix target):
```python
handle.session.stop()
if handle.task is not None and not handle.task.done():
    try:
        await asyncio.wait_for(asyncio.shield(handle.task), timeout=10.0)
    except (asyncio.TimeoutError, Exception):
        pass
```
This hangs for up to 10s if the LLM is mid-stream. Replace `handle.session.stop()` with `await handle.loop.cancel()`. Ordering: `_handles.pop(project_id, None)` at line 1336 must stay *after* the wait so `_on_loop_done` can observe `is_stopped()` and broadcast correctly.

**4. `_on_loop_done` cleanup — `agent_manager.py:1402-1433`** (no change, consumer):
```python
def callback(task: asyncio.Task):
    try:
        exc = task.exception()
    except asyncio.CancelledError:
        self._ws.broadcast(project_id, {"type": "agent.status", ... "status": "stopped"})
        return
    ...
    if handle.session.is_stopped():
        self._handles.pop(project_id, None)
```
This is a *consumer* of cancellation. With the new `cancel()`, the `CancelledError` arm fires (it already broadcasts `"stopped"` correctly). No change needed unless distinguishing user-initiated cancel from generic `CancelledError`.

**5. Watchdog "forcing idle" — `agent_manager.py:1525-1532`** (no change):
```python
logger.warning("Sub-agent poll timeout for project %s, forcing idle", project_id)
self._ws.broadcast(project_id, {... "status": "idle"})
```
Observational/UI only; does NOT stop the management loop (already idle by this point — watchdog only runs after `_on_loop_done`). No `cancel()` integration needed *here* — but see Section 4.3 for the orthogonal sub-agent stop gap at this site.

**6. AgentManager shutdown path — `agent_manager.py:198-205`** (transitive improvement):
```python
for pid in list(self._handles.keys()):
    stop_tasks.append(self.stop_agent(pid))
if stop_tasks:
    try:
        await asyncio.wait_for(asyncio.gather(*stop_tasks, return_exceptions=True), timeout=timeout)
```
Improves transitively once `stop_agent` uses `cancel()` — currently shutdown blocks `min(10s_per_agent, timeout)` per stuck stream.

**Ordering-concern summary:** In all sites, the order must be: (1) call `cancel()` (sets flag, cancels task), (2) `await` the task to actually finish (reap any cleanup in the loop's `finally`), (3) THEN mutate `_handles[pid]`, `handle.task = None`, or swap `handle.session`. Inverting (3) before (2) risks the dying loop writing to a freshly-allocated handle/session.

---

### 3.5 Additional finding: `loop.py:643-644` session-end follow-up bypasses `cancel()`

(Not in the user's question list — added because it's an orphan path that `AgentLoop.cancel()` would not catch.)

**Direct answer:** The session-end callback at `loop.py:643-644` is fired **fire-and-forget** via `asyncio.create_task` inside the loop's `finally` block. Not awaited. Not stored. `AgentLoop.cancel()` would not reach it.

**Evidence — `agent_os/agent/loop.py:633-645`:**

```python
finally:
    if not self._session._paused_for_approval:
        self._session.resolve_pending_tool_calls()
    # Drain any remaining deferred messages
    for msg in self._session.pop_deferred_messages():
        self._session.append(msg)
    # Trigger session-end callback (e.g. workspace file generation)
    # Fire-and-forget so idle broadcasts immediately.
    # Skip if LLM failed — the provider is unreachable so the
    # session-end LLM call would also fail.
    if self._on_session_end is not None and not self._llm_failed:
        asyncio.create_task(self._on_session_end())
    self._running = False
```

The created task has no name, no strong reference stored, no done-callback, and is not tracked in `self._handles` or `_idle_poll_tasks`. This is exactly the B5 anti-pattern at a fresh site.

The gating condition is `not self._llm_failed` — a user-initiated cancel does **not** set `_llm_failed`, so cancelling will still kick off a session-end LLM call in the background.

**Implication for the proposed fix:** This is a separate orphan path that `AgentLoop.cancel()` cannot reach by simply cancelling the loop task — `create_task` detaches the child task lifetime from the parent. Three options:
- **(a)** Skip the session-end follow-up when `is_stopped()` is true: `if self._on_session_end is not None and not self._llm_failed and not self._session.is_stopped():`
- **(b)** Track the spawned task on `self._session_end_task` and have `cancel()` cancel it too.
- **(c)** Note: `new_session` already calls `run_session_end_routine` synchronously with a 30s timeout (`agent_manager.py:1222-1230`), so the fire-and-forget path partially duplicates that work — option (a) is consistent with the precedent that explicit `stop`/`new_session` flows manage their own session-end semantics. **Recommend (a) as the lowest-risk fix.**

---

## Section 4 — sub_agent_manager API surface

### 4.1 Lifecycle-relevant public methods

**Direct answer — internal state declaration** at `agent_os/daemon_v2/sub_agent_manager.py:38-41`:

```python
self._adapters: dict[str, dict[str, object]] = {}        # project_id -> {handle -> adapter}
self._transcripts: dict[tuple[str, str], object] = {}    # (project_id, handle) -> SubAgentTranscript
self._lifecycle_locks: dict[str, asyncio.Lock] = {}      # project_id -> lock
self._stopping: set[str] = set()                         # project_ids currently in stop_all
```

**Public methods, all in `sub_agent_manager.py`:**

| Method | Signature | Async | Purpose | Line |
|---|---|---|---|---|
| `start` | `async def start(self, project_id, handle, depth=0) -> str` | yes | Create adapter, register with process_manager. | 51 |
| `send` | `async def send(self, project_id, handle, message) -> str` | yes | Dispatch message to adapter, non-blocking. | 290 |
| `_dispatch_async` | `async def _dispatch_async(self, adapter, project_id, handle, message) -> None` | yes | Internal: route to transport.dispatch or background sender. | 317 |
| `stop` | `async def stop(self, project_id, handle) -> str` | yes | Stop one adapter, deregister. | 424 |
| `status` | `def status(self, project_id, handle) -> str` | no | `'running'\|'idle'\|'stopped'\|'unknown'`. | 437 |
| `get_pending_sub_agent_approval` | `def get_pending_sub_agent_approval(self, project_id) -> dict \| None` | no | First pending sub-agent approval. | 449 |
| `resolve_sub_agent_approval` | `async def resolve_sub_agent_approval(self, project_id, tool_call_id, approved) -> bool` | yes | Route approval to sub-agent transport. | 476 |
| `update_sub_agent_autonomy` | `def update_sub_agent_autonomy(self, project_id, preset) -> None` | no | Push autonomy preset to all SDK sub-agent transports. | 492 |
| `stop_all` | `async def stop_all(self, project_id) -> None` | yes | Stop all sub-agents for a project. | 500 |
| `list_active` | `def list_active(self, project_id) -> list[dict]` | no | `[{'handle','display_name','status'}, ...]` for alive adapters. | 517 |

**Specific answers to the question's checklist:**
- `list_active(project_id)`: **exists** (`sub_agent_manager.py:517`). Iterates `self._adapters[project_id]`, filters by `adapter.is_alive()`, returns dicts.
- `stop(project_id, handle)`: **exists** (`:424`). Acquires per-project lifecycle lock, pops adapter, awaits `adapter.stop()` and `process_manager.stop()`.
- `stop_all(project_id)`: **exists** (`:500`). See 4.2 for concurrency analysis.
- Internal dict structure: see above.

**Implication for the proposed fix:** All the methods the prior B5 investigation suggested adding (`list_active`, `stop`, `stop_all`) are **already present**. A new `cancel_message` verb can layer on top of them without inventing new collection methods. However, `stop_all` is the wrong primitive for a chat-level Stop — see 4.2.

---

### 4.2 Existing `stop_all` — concurrency review (the user's question is moot, here's the actual analysis)

**Direct answer:** `stop_all` is already implemented (`sub_agent_manager.py:500-515`). Three concurrency gaps exist that any stop-pathway fix must account for.

**Existing behavior:**
- Sets `self._stopping.add(project_id)` (rejects new `start()` / `_start_from_registry()` invocations on lines 53-54 and 179-180).
- Acquires `_lifecycle_locks[project_id]`, snapshots `list(adapters.keys())` while holding it, then **releases** the lock before iterating per-handle `stop()` calls.
- Each `stop()` re-acquires the lock (lines 426-427) for its pop-and-await sequence.
- In `finally`, removes the project from `_stopping` and pops its lock.

**Concurrency concerns:**

| Concern | Evidence | Severity |
|---|---|---|
| New `dispatch()` arriving mid-`stop_all` | `send()` (line 290) does **not** check `self._stopping` and does **not** acquire the lifecycle lock — it reads `self._adapters[project_id]` lock-free. A `send()` landing between snapshot and per-handle stop dispatches to an adapter about to be popped. The `_stopping` flag protects starts, not sends. | High |
| A sub-agent's `stop()` blocks longer than expected | `stop_all` has **no timeout** on `await self.stop(project_id, handle)` (line 510). A wedged `adapter.stop()` blocks `stop_all`, which `agent_manager.stop_agent()` awaits at line 1307 and `new_session()` at line 1214. **No `wait_for` shield.** | High |
| Two concurrent `stop_all` callers (rotation + watchdog) | Lock released between snapshot and per-handle stop, so callers interleave: A snapshots `[h1,h2]`, B snapshots `[]`. A pops h1; B sees empty and returns immediately. The `_lifecycle_locks.pop(project_id, None)` in `finally` (515) of A can drop the lock entry while B is between `_get_lock` and `async with`, creating a fresh `Lock()` on the next `_get_lock`. `_stopping.discard` runs even if A is still working — A may still be inside `stop()` when `_stopping` is cleared, allowing fresh `start()` to slip in. | Medium |

**Implication for the proposed fix:** `stop_all` is fine as a **project-shutdown** primitive but is **not** the right primitive for a Stop button on the management agent's chat. Stop-button semantics need to cancel the loop's current LLM call and dispatch a cancellation marker into the conversation — not tear down all sub-agents. If a B3 fix routes Stop through `stop_all`, the three gaps above also need patching.

---

### 4.3 Watchdog "forcing idle" path

**Direct answer:** The watchdog at `_check_sub_agents_done` (`agent_manager.py:1501-1532`) does **not** call `stop` or `stop_all` today — it only polls `list_active` and broadcasts `agent.status: idle`. Adding a `stop_all` call is safe w.r.t. circular imports and async deadlock; the only real concern is bounding the call.

**Evidence — `agent_manager.py:1501-1532`:**

```python
async def _check_sub_agents_done(self, project_id: str) -> None:
    """Poll until sub-agents finish, then broadcast idle."""
    for _ in range(self._MAX_IDLE_POLLS):
        await asyncio.sleep(2.0)
        handle = self._handles.get(project_id)
        if handle is None or handle.session.is_stopped():
            return
        if handle.task is not None and not handle.task.done():
            return
        active = self._sub_agent_manager.list_active(project_id)
        busy = [a for a in active if a.get("status") != "idle"]
        if not busy:
            self._ws.broadcast(project_id, {... "status": "idle" ...})
            return
    logger.warning("Sub-agent poll timeout for project %s, forcing idle", project_id)
    self._ws.broadcast(project_id, {... "status": "idle" ...})
```

**Risk analysis:**
- **Circular import risk:** None. `agent_manager.py` does not `import` `SubAgentManager` (verified — `grep "from agent_os.daemon_v2.sub_agent_manager" agent_os/daemon_v2/agent_manager.py` returns no matches). Reference is held via constructor injection (`self._sub_agent_manager`, `agent_manager.py:69`).
- **Async deadlock risk:** Low. Watchdog runs as its own asyncio task (`asyncio.ensure_future(self._check_sub_agents_done(project_id))`, `agent_manager.py:1483-1484`), holds **no** locks of its own. If it called `stop_all`, that acquires `_lifecycle_locks[project_id]`. Concurrent `stop_agent()` (which also calls `stop_all` at line 1307) serializes on the same lock — bounded wait, not deadlock.
- **Re-entrance:** The watchdog can fire while already in flight only weakly. `agent_manager.py:1486` stores the task in `self._idle_poll_tasks[project_id]` without checking whether a previous task is still running; a second `_on_loop_done` callback could overwrite the entry, leaving the prior watchdog orphaned but still polling. Worst behavior: duplicate `agent.status: idle` broadcasts.

**Implication for the proposed fix:** A Stop verb that delegates to `stop_all` from the watchdog path is **safe** w.r.t. deadlock but should add (a) a `wait_for` timeout around `stop_all`, (b) explicit cancellation of any prior `_idle_poll_tasks[project_id]` before scheduling a new one, and (c) a check on `_stopping` before broadcasting `idle` to avoid lying about state mid-shutdown.

---

## Section 5 — B3 Stop button IPC surface (CONDITIONAL — skip unless instructed otherwise)

### 5.1 Verbs the frontend can send to the daemon

**Direct answer:** All daemon HTTP routes are FastAPI routers in `agent_os/api/routes/`. The single WebSocket endpoint at `/ws` (`app.py:340-371`) accepts only one client→server message type: `"subscribe"` (`:357-362`). All agent control flows are HTTP.

**Routes relevant to agent lifecycle**, all in `agent_os/api/routes/agents_v2.py`:

| Verb | Path | Purpose | Line |
|---|---|---|---|
| POST | `/agents/start` | Start agent for a project | 508 |
| POST | `/agents/{project_id}/inject` | Send user message to management or sub-agent | 578 |
| GET | `/agents/{project_id}/run-status` | Read run status string | 649 |
| GET | `/agents/{project_id}/pending-approval` | Recover pending approval card | 656 |
| POST | `/agents/{project_id}/stop` | **Tear down agent + sub-agents for project** | 670 |
| POST | `/agents/{project_id}/new-session` | Archive session and start fresh | 679 |
| POST | `/agents/{project_id}/approve` | Approve pending tool call | 685 |
| POST | `/agents/{project_id}/deny` | Deny pending tool call | 711 |
| GET | `/agents/{project_id}/chat` | Read chat history | 804 |

**WebSocket (`app.py:340`):**
```python
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    ...
    data = await websocket.receive_json()
    ...
    ws_manager.subscribe(websocket, project_ids)
```

The WebSocket is server-push-only after the initial `subscribe` handshake — **not** a viable channel for a "cancel" verb without protocol changes.

**Specific answer — is there a verb other than `inject_message` that the Stop button could route to today?**

Yes. `POST /agents/{project_id}/stop` exists (`agents_v2.py:670-676`) and calls `_agent_manager.stop_agent(project_id)` (`agent_manager.py:1295-1341`). However, this is a **full teardown**: calls `stop_all` on sub-agents (1307), closes browser pages (1311-1314), tears down platform sandbox (1317-1318), calls `handle.session.stop()` (1320), waits up to 10s for the loop task (1322-1327), broadcasts `"status": "stopped"`, pops the handle, writes daemon state. After this call the agent is gone — the next user message goes through `/agents/start` again.

**Implication for the proposed fix:** The daemon already has both verbs but the gap is in the middle: there is **no verb that means "interrupt the current LLM/tool turn but keep the agent and session alive."** B3's `cancel_message` would fill this gap.

---

### 5.2 Stop button frontend handler

**Direct answer:** **Surprise finding — the Stop button already calls `POST /api/v2/agents/{pid}/stop` (the destructive teardown verb), NOT `/inject`.** The earlier B3 framing — "Stop and Send both route through `inject_message`" — is contradicted by the current frontend code.

**Evidence:**

*Render site* — `web/src/components/ChatView.tsx:1153-1162`:
```tsx
{(agentStatus === 'running' || agentStatus === 'waiting') ? (
  <>
    <button
      type="button"
      onClick={() => stopAgent(projectId)}
      onTouchEnd={(e) => { e.preventDefault(); stopAgent(projectId); }}
      className="...text-red-500..."
    >
      <Square size={18} />
    </button>
```

*Handler* — `web/src/hooks/useAgent.ts:35-40`:
```ts
const stopAgent = useCallback(async (projectId: string) => {
  return api<ActionResult>(
    `/api/v2/agents/${encodeURIComponent(projectId)}/stop`,
    { method: 'POST' },
  );
}, []);
```

*Exact request:* `POST /api/v2/agents/{projectId}/stop`, no body, no `Content-Type`. Response: `{ "status": "stopping" }` (`agents_v2.py:676`).

A second Stop control exists in `ProjectDetail.tsx:54` (`onClick={onStopAgent}`) — same hook.

**Implication for the proposed fix:** This contradicts the prior B3 framing. The orbital-marketing incident's `inject_message: loop idle, delivering message and resuming` log entries at 15:16:59+ around the user's Stop press were probably **not** caused by the Stop button — they were caused by something else (e.g., a follow-up `Send`, or the watchdog). The actual user-visible failure of Stop in the incident was different: `/stop` *did* fire, but at that moment the orphaned management loop was unreachable from `_handles[project_id]` (which pointed at the auto-spawned new agent — see B2). So `stop_agent` cleanly killed the wrong handle. The full-teardown semantic is correct for "End session" but wrong for "interrupt this turn." Either way, a new `cancel_message` verb is genuinely needed for cancel-this-turn semantics; the existing `/stop` verb is not the bug, it's the wrong tool for the use case.

---

### 5.3 Wiring required for a new `cancel_message` verb

| Layer | File:line | Change |
|---|---|---|
| Daemon HTTP route | `agent_os/api/routes/agents_v2.py:670` (next to existing `/stop`) | Add `@router.post("/agents/{project_id}/cancel")` handler that calls `_agent_manager.cancel_message(project_id)` and returns `{"status": "cancelled"}`. WebSocket dispatch is not viable — current WS only handles `"subscribe"` (`app.py:357-362`) — verb must be HTTP. |
| `agent_manager` method | `agent_os/daemon_v2/agent_manager.py:1295` (alongside `stop_agent`) | Add `async def cancel_message(self, project_id) -> None`: cancel `handle.task` if it is the in-flight LLM/tool turn (without calling `session.stop()`), append a cancellation marker via `agent/session.py:365-372` shape, broadcast `"status": "idle"` (or new `"cancelled"`), and **do not** call `stop_all` — sub-agents survive. |
| `AgentLoop.cancel()` invocation | `agent_os/agent/loop.py:69` (`class AgentLoop`) — **no `cancel()` method exists today** | `grep "def cancel\|def stop"` against `agent/loop.py` returns nothing. Current cancellation goes via `Session.stop()`. A real mid-stream `AgentLoop.cancel()` is non-trivial: the loop awaits `self._stream_response(...)` (`loop.py:118-130`) and a provider-side abort hook is not in evidence. **Flag:** the cancel verb may only be able to interrupt at well-defined yield points (between tool calls / between chunks) unless cancellation is plumbed into `LLMProvider.stream` (see Section 2.2 — this is the cli_adapter-template gap). |
| WebSocket broadcast for "cancelled" | `agent_os/api/ws.py:83` (`broadcast` method); existing pattern at `agent_manager.py:1329-1334` | Mirror existing status-broadcast shape with `"status": "cancelled"` (or reuse `"idle"` to avoid frontend churn). Sync, queued — no new infra needed. |
| Frontend handler | `web/src/hooks/useAgent.ts:35-40` (replace or add alongside `stopAgent`); button render at `ChatView.tsx:1153-1162` | Add `cancelMessage(projectId) → POST /api/v2/agents/{projectId}/cancel`. Wire the existing `<Square>` button's `onClick` to `cancelMessage` instead of `stopAgent`. Keep `stopAgent` reachable from a more deliberate "End session" affordance (e.g. `ProjectDetail.tsx:54`). |

**Layers currently missing or unclear (flagged):**

1. **`AgentLoop.cancel()` does not exist.** Closest analog is `Session.stop()`, polled at `loop.py:555`. A cancel verb that aborts an in-flight LLM stream needs either (a) a new `loop.cancel()` that cancels the underlying `asyncio.Task` running `loop.run()` (already available at `handle.task` — `agent_manager.py:1206-1208` shows the pattern), or (b) a cooperative cancellation flag the loop checks between tool calls. The fix spec must pick one explicitly.
2. **No WebSocket inbound message types beyond `"subscribe"`** (`app.py:357-362`). If Stop must feel "instant" on mobile (sub-100ms) without HTTP RTT, that requires a protocol extension; otherwise stick with HTTP.
3. **Stop button is currently wired to `/stop` (full teardown).** The fix spec must decide product semantics: (a) repurpose the button's onClick to `cancelMessage` and find a new home for full teardown, or (b) introduce a separate "End session" icon. Decide before coding.

---

## Cross-cutting observations

**1. The cancellation gap is concentrated in one component.** Three of five sections converge on `loop.py` lacking a way to interrupt `await self._provider.stream(...)`. S2 says the cli_adapter template covers the mechanics but not Moonshot specifics; S3 says no `AgentLoop.cancel()` exists; S5 says even adding the verb is gated on solving this. The single highest-leverage change is adding cancellable streaming in `LLMProvider.stream` + `AgentLoop.cancel()` — once that exists, the rest is wiring.

**2. Two fire-and-forget `asyncio.create_task(...)` patterns remain on this branch.** B5 fixed `sdk_transport.py:140` (now strong-ref + cancel-and-await). But `loop.py:643` still does fire-and-forget for the session-end follow-up. Same anti-pattern, different site. See 3.5.

**3. `stop_all` already exists; the watchdog just doesn't call it.** S4 closes a question opened by the B5 investigation — the prescribed `stop_all(project_id)` method is already in the tree. The fix is a one-line `await self._sub_agent_manager.stop_all(project_id)` in the watchdog path, plus a `wait_for` budget.

**4. The B3 frame was wrong about the Stop button.** It already calls `/stop`, not `/inject`. The orbital-marketing incident's "Stop did nothing" symptom was a B2 problem (handle pointing at wrong loop), not a B3 problem (no cancel verb). The right fix is still adding `cancel_message`, but for a different reason than originally argued — to provide cancel-this-turn semantics distinct from end-session semantics, not to provide a cancel pathway where none existed.

**5. The shield invariant is undocumented.** S1 cannot recover the original motivation. Any plan that drops shield must either (a) prove from current code that no outer cancellation can reach these stacks, or (b) keep the shield and find a different intervention point. "Drop the shield" is not a safe default.
