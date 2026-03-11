# TASK-FIX-sdk-transport-rate-limit: Handle Unknown SDK Message Types Gracefully

*Created: 2026-03-11*
*Status: INVESTIGATED*
*Severity: Medium (sub-agent response lost/delayed on rate limit events)*
*Complexity: Low*
*Ref: Daemon monitoring session 2026-03-11, Issue 1*
*Depends on: Nothing*

---

## One-Sentence Summary

The SDK transport's `receive_response()` loop crashes on unknown message types (like `rate_limit_event`), losing the current response and requiring a flush on the next send.

## What Happens

```
SDKTransport: background response consumption failed: Unknown message type: rate_limit_event
SDKTransport: background response ended without ResultMessage; will flush on next send
```

The SDK's `receive_response()` async iterator raises an exception when it encounters a message type it doesn't recognize. Our code wraps the entire `async for` loop in a try/except, so ANY unknown message type aborts the entire response stream. The `ResultMessage` (which contains session_id and completion status) is never received.

## Current Code (sdk_transport.py)

The transport only imports and handles two message types:
```python
from claude_agent_sdk.types import AssistantMessage, ResultMessage, TextBlock, ToolUseBlock
```

The `_message_to_events()` method (line 258) handles `AssistantMessage` and `ResultMessage` via isinstance checks. Unknown types return an empty event list — this is fine.

The REAL problem is in the `async for` loop (lines 92, 138): the SDK itself raises an exception when iterating over a message it can't deserialize into a known type. This exception propagates out of our for loop, terminating response consumption.

## SDK Analysis

The `claude_agent_sdk.types` module exports a `StreamEvent` type that we don't import. Available types include:
- `AssistantMessage` (handled)
- `ResultMessage` (handled)
- `StreamEvent` (NOT handled — likely wraps rate_limit_event and others)
- `SystemMessage`, `UserMessage` (not relevant for receive)
- `ControlResponse`, `ControlErrorResponse` (control channel)

The SDK does NOT export a `RateLimitEvent` type directly. The error "Unknown message type: rate_limit_event" suggests the SDK's internal deserializer encounters a JSON message with `type: "rate_limit_event"` and can't map it to any known class.

## Root Cause Analysis

### Root Cause 1: SDK deserializer raises on unknown types (MOST LIKELY)
The SDK's internal message parser has a strict type map. When the Claude Code subprocess emits a `rate_limit_event` JSON message, the SDK can't find a matching dataclass and raises `ValueError("Unknown message type: rate_limit_event")`. This is a defect in the SDK — unknown types should be skipped, not raise.

**Evidence:** The error message format matches a simple `raise ValueError(f"Unknown message type: {msg_type}")` pattern.

### Root Cause 2: Anthropic API added rate_limit_event to streaming protocol
The Anthropic API streaming protocol was updated to include `rate_limit_event` messages (containing rate limit headers like `requests-remaining`, `tokens-remaining`). Claude Code forwards these through its subprocess output. The SDK version we're using predates this addition.

**Evidence:** The Anthropic API docs mention rate limit events in SSE streams.

### Root Cause 3: Exception handling granularity too coarse
Even if the SDK raises on unknown types, our code wraps the entire response loop in a single try/except. A per-iteration try/except would let us skip unknown messages and continue consuming the stream until we get the `ResultMessage`.

**Evidence:** The code structure at lines 92-108 and 138-150 shows the entire `async for` is wrapped, not individual iterations.

## Proposed Fix

### Option A: Per-iteration exception handling (recommended, no SDK changes needed)

```python
async for msg in self._client.receive_response():
    try:
        if isinstance(msg, ResultMessage):
            got_result = True
        events = self._message_to_events(msg)
        for event in events:
            await self._event_queue.put(event)
    except Exception as e:
        logger.debug("SDKTransport: skipping unknown message: %s", e)
        continue
```

This won't work if the SDK raises DURING iteration (inside `__anext__`), not during processing. Need to verify.

### Option B: Wrap the async iterator to catch per-yield exceptions

```python
async def _safe_receive(self):
    """Wrap receive_response() to skip messages that fail deserialization."""
    try:
        async for msg in self._client.receive_response():
            yield msg
    except Exception as e:
        if "Unknown message type" in str(e):
            logger.debug("SDKTransport: skipping unknown message type: %s", e)
            # Cannot continue iteration after async generator raises
            return
        raise
```

This also won't work — once an async generator raises, iteration stops.

### Option C: Monkey-patch or upgrade the SDK (best long-term fix)

1. Check if a newer version of `claude-agent-sdk` handles `rate_limit_event`
2. If not, file an issue / contribute a fix to skip unknown types
3. As interim, pin the SDK version and accept the flush-on-next-send recovery

### Option D: Pre-filter at subprocess level (if using pipe-based SDK)

If the SDK reads from a subprocess pipe, we could intercept the JSON stream and filter out `rate_limit_event` messages before the SDK sees them. This is fragile and not recommended.

## Recommendation

**Short-term:** Option A with verification. Test whether the exception occurs inside `__anext__` (can't catch per-iteration) or during message processing (can catch). If inside `__anext__`, accept the current flush recovery and upgrade the SDK when available.

**Long-term:** Upgrade `claude-agent-sdk` to a version that handles rate limit events, or contribute the fix upstream.

## Impact Assessment

- **Current impact:** Sub-agent response for the affected turn may be lost or delayed. The `_needs_flush = True` recovery path handles this — the next `send()` drains stale messages first. The response is recovered, just delayed by one turn.
- **Frequency:** Depends on API rate limiting. With the Moonshot/Kimi API proxy, rate limits may trigger more frequently than with direct Anthropic API access.
- **User-visible:** Sub-agent may appear to "skip" a response, then respond normally on the next interaction.

## Tests

```
Test: test_unknown_message_type_does_not_crash_send
Setup: Mock SDK client to yield [AssistantMessage, <unknown>, ResultMessage]
Assert: send() returns the text from AssistantMessage
Assert: got_result is True (ResultMessage was received)
Assert: No exception raised

Test: test_rate_limit_event_triggers_flush_recovery
Setup: Mock SDK to raise on rate_limit_event mid-stream
Assert: _needs_flush is True after send()
Assert: Next send() calls _flush_stale_messages() first

Test: test_flush_recovers_result_message
Setup: Mock flush to find a ResultMessage in the buffer
Assert: session_id is captured
Assert: _needs_flush reset to False
```

## Cleanup

- Remove ALL debug prints, console.logs, and temporary code
- Search for: `print(f"[DEBUG`, `console.log("[DEBUG`, `TODO`, `HACK`, `TEMP`, `FIXME`
- Remove any you find
