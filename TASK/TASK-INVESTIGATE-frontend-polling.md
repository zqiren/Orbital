# TASK-INVESTIGATE-frontend-polling: Frontend Polling vs WebSocket Analysis

*Created: 2026-03-11*
*Status: INVESTIGATED — NOT A BUG*
*Severity: Low (intentional resilience pattern)*
*Complexity: N/A*
*Ref: Daemon monitoring session 2026-03-11, Issue 3*

---

## One-Sentence Summary

The frontend's run-status and pending-approval polling at 5-second intervals is an intentional WebSocket fallback mechanism, not a bug — no fix needed.

## What Was Observed

```
GET /agents/{pid}/run-status    (periodic)
GET /agents/{pid}/pending-approval  (periodic)
```

The daemon log shows periodic REST requests for run-status and pending-approval even though a WebSocket connection is active.

## Investigation Findings

### Polling is a deliberate fallback, not the primary mechanism

**Primary delivery:** WebSocket (`/ws`) pushes `agent.status`, `approval.request`, and `approval.resolved` events in real-time. The frontend subscribes to all project IDs on connect.

**Fallback polling:** `ChatView.tsx` polls run-status every **5 seconds** (not 1-2s as initially estimated) ONLY when `agentStatus === 'running'` or `agentStatus === 'pending_approval'`. Polling stops when status changes to idle/error.

### Polling entry points (ChatView.tsx)

1. **Status poll (lines 333-380):** 5-second interval while running/pending_approval. Detects if agent appears stuck with no stream activity. Also checks for new approvals in pending_approval state.

2. **Mount check (lines 309-331):** One-time check when ChatView mounts to hydrate initial approval state.

3. **Status change handler (lines 261-285):** Triggered when agentStatus changes to pending_approval — fetches the approval details once.

### Why both run-status AND pending-approval?

- `run-status` returns the agent lifecycle state (running, idle, error, pending_approval)
- `pending-approval` returns the specific tool call details needed to render the approval UI
- They serve different purposes: status detection vs approval data

### WebSocket resilience features

- Automatic reconnection with exponential backoff (1s → 30s max)
- REST hydration on reconnect (App.tsx, lines 151-169)
- Status override custom events to handle stale WebSocket state
- Heartbeat ping/pong mechanism

## Conclusion

**No fix needed.** The polling pattern is a well-designed resilience layer:
- 5-second interval is reasonable (not excessive)
- Only active during running/pending states (not always-on)
- Provides recovery from missed WebSocket events (tunnel drops, relay restarts, etc.)
- The two endpoints serve different data needs

### Minor optimization opportunities (low priority)

1. **Combine into single endpoint:** A single `/agents/{pid}/status` that returns both run status and pending approval data could halve the request volume. Low priority since the endpoints are lightweight.

2. **Conditional polling:** Skip the poll if a WebSocket status event was received within the last 5 seconds (the poll is redundant if WebSocket is delivering reliably). Minor optimization.

3. **ETag/If-None-Match:** Return 304 Not Modified when status hasn't changed, reducing response payload size. Negligible benefit for small JSON responses.

None of these are worth implementing unless the daemon is under significant load.
