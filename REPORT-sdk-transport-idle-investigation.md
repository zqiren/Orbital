# SDK Transport Idle State Investigation

## 1. Completion Event Shape

**`ResultMessage` marks end-of-turn, but does NOT emit a TransportEvent in the normal case.**

When Claude Code finishes a task, `_consume_response_background()` (`sdk_transport.py:137-159`) iterates `self._client.receive_response()`:

1. Zero or more `AssistantMessage` -> converted via `_message_to_events()` to `"message"` and `"tool_use"` TransportEvents
2. `ResultMessage` -> captures `session_id`; only emits an `"error"` event if `msg.is_error` is True (lines 292-299)
3. **Async for loop exhausts** -> `_consume_response_background()` returns

The actual completion signal is **the end of the `async for` loop** -- not an explicit event. In normal (non-error) completion, `ResultMessage` is consumed silently. No "done" or "idle" TransportEvent is emitted.

After completion, the `_event_queue` simply goes empty. `read_stream()` (`sdk_transport.py:187-192`) continues looping with 0.5s timeouts, yielding nothing.

**Code locations:**
- `sdk_transport.py:137-159` -- `_consume_response_background()`
- `sdk_transport.py:269-301` -- `_message_to_events()` (handles AssistantMessage, ResultMessage)

## 2. Idle vs Terminated

The SDK transport is persistent -- it stays alive between messages.

| State | Transport behavior | `_event_queue` | `self._alive` |
|-------|-------------------|----------------|---------------|
| Streaming response | `_consume_response_background` running, events flowing | Non-empty | True |
| Finished, waiting for next message | `_consume_response_background` returned, queue drained | Empty | True |
| Waiting for approval | `_handle_permission` blocked on future; `_consume_response_background` blocked at `receive_response()` | Empty (after approval_request consumed) | True |
| Sub-agent died | Transport stopped | N/A | False |

**There is no explicit "idle" signal.** The queue just goes silent. `read_stream()` continues timing out every 0.5s in a spin loop.

## 3. Dispatch Resumption

`_dispatch_async()` (`sub_agent_manager.py:316-318`) calls `transport.dispatch(message)` directly, **bypassing `adapter.send()`**:

```python
if transport is not None and hasattr(transport, 'dispatch'):
    await transport.dispatch(message)  # adapter._idle is NEVER touched
    return
```

`dispatch()` (`sdk_transport.py:126-135`):
- Calls `self._client.query(message)`
- Creates background task `_consume_response_background()`
- Returns immediately (fire-and-forget)

**No "un-idle" event is emitted.** The adapter's idle state is only managed in `adapter.send()` (lines 147-155), which is never called for SDK transport dispatch.

## 4. Approval Interaction

When Claude Code needs permission (`_handle_permission()`, `sdk_transport.py:226-267`):

1. Creates a future: `self._pending_approvals[request_id] = future`
2. Emits `TransportEvent(event_type="permission_request", ...)` to `_event_queue` (line 244)
3. **Blocks**: `approved = await future` (line 258)
4. During the block, `_consume_response_background()` is also blocked inside the SDK's `receive_response()` iterator
5. `read_stream()` yields the `permission_request` event, then goes silent (0.5s timeouts)

**The `permission_request` event is the last event before silence.** No distinct "paused" or "waiting_for_approval" event exists.

When approved via `respond_to_permission()`:
- `future.set_result(approved)` unblocks `_handle_permission()`
- `receive_response()` resumes yielding messages
- Events flow again through `_event_queue`

## 5. All `is_idle()` / `status()` Consumers

### Idle state readers

| File | Line | Usage |
|------|------|-------|
| `sub_agent_manager.py` | 350 | `status()`: returns `"idle"` if `adapter.is_idle()` |
| `sub_agent_manager.py` | 423 | `list_active()`: maps idle to `"idle"` status string |
| `agent_manager.py` | ~1273 | `_check_sub_agents_done()`: polls `list_active()` for busy sub-agents |
| `agent_manager.py` | ~844 | `get_run_status()`: determines project-level status |
| `agents_v2.py` | ~613 | REST `/run-status` endpoint |
| `smoke_v5.py` | ~155 | Test helper `wait_for_idle()` |

### Idle state writers (CLIAdapter)

| Line | Trigger | Value |
|------|---------|-------|
| 51 | `__init__` | `False` |
| 147 | Before `transport.send()` | `False` |
| 155 | After `transport.send()` completes | `True` |
| 158 | Before PTY/pipe write | `False` |
| 201 | After non-transport `read_stream()` ends | `True` |
| 268, 274 | After `stop()` | `False` |

**For SDK transport via `dispatch()`: NONE of the `_idle = True` writers execute.**

## 6. Recommended Idle State Machine

```
                  dispatch()
    IDLE --------------------------> NOT_IDLE
     ^                                  |
     |                                  |
     |   ResultMessage received         |  permission_request emitted
     |   (response complete)            |
     |                                  v
     +---------------------------- WAITING_APPROVAL
                 approval resolved
```

### Implementation approach

1. Emit a `"turn_complete"` TransportEvent at the end of `_consume_response_background()` when `got_result` is True
2. In ProcessManager `consume()`, when `chunk_type == "turn_complete"`: signal idle
3. In `_dispatch_async()`, set `adapter._idle = False` before calling `transport.dispatch()`
4. `is_idle()` should also check `_pending_approvals` -- if any are pending, return False
