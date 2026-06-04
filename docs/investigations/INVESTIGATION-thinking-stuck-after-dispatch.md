# INVESTIGATION: thinking-spinner stuck after sub-agent dispatch completes

**Date:** 2026-05-28
**Branch:** `fix/rotation-by-session-id` (commit `cd3325d`, the session_id audit pass)
**Build under test:** `/Applications/Orbital.app` rebuilt at 14:52 (web assets `index-B4J6eT2k.js`)
**Reporter:** smoke testing the audit fix end-to-end
**Severity:** P1 — visible UI hang after every successful sub-agent dispatch on this branch
**Status:** ROOT CAUSE HYPOTHESIZED, NOT YET CONFIRMED with DevTools trace.

## Observed behavior

1. After `Cmd+R` to refresh the SPA, user typed a message into session `sess_e4277f84` ("能不能让Claude code列出从1-20里的质数").
2. The "thinking" indicator showed.
3. **The daemon completed the entire round-trip correctly** — verified via JSONL on disk:
   - `15:09:12` Management agent emitted the dispatch tool call (`agent_message → send claude-code`). `yield_turn` ended the management turn.
   - `15:09:14` Claude-code sub-agent finished with response: `"40到60之间的质数有：- 41 - 43 - 47 - 53 - 59 共5个。"`
     (yes, the user actually said 1–20 but the agent dispatched 40–60 — model creativity, unrelated to this bug)
   - `15:09:16` `lifecycle.on_completed` fired, injected system message into `sess_e4277f84`.
   - `15:09:19` Management restart generated final assistant message: `"已派发，预计结果：**41, 43, 47, 53, 59** 这 5 个质数..."`. All four messages carry the correct `session_id` and `session_uuid`.
4. **In the UI:** thinking spinner never cleared, no sub-agent message capsule appeared, the final assistant reply never rendered. Session APPEARS hung from the user's view.
5. The fix from the audit pass (commit `cd3325d`) is correctly compiled into the bundle — all four backend broadcasts (`chat.sub_agent_message`, `approval.request`, `agent.activity`, `sub_agent.completed`) carry `session_id="sess_e4277f84"`, confirmed by reading the on-disk JSONL paths cited in the daemon log.

## What was verified by reading actual code

- `agent.status` is broadcast at four key transitions: `running` on `_start_loop`, `waiting` on yield+busy in `_on_loop_done`, `running` again when `lifecycle.on_completed → inject_system_message → _start_loop` restarts the management loop, and `idle` when the final `_on_loop_done` sees no busy sub-agents.
- All four `agent.status` broadcasts carry `session_id` (via the `_broadcast` helper at `agent_manager.py:135` which `setdefault`s it from `_resolve_session_id`).
- `App.tsx:202-204` handles `agent.status` for ALL projects, keyed by `project_id` only — no `session_id` filter, so the event reaches state regardless of session.
- `ChatView` derives `viewingHolder` from `(sessionId !== undefined) && (holderSessionId === null ? agentIsActive : sessionId === holderSessionId)` (`ChatView.tsx:419-421`).
- `ChatView.tsx:748-797` is the effect that handles `agentStatus` transitions. The idle branch (line 750) gates **all** UI cleanup behind `if (!viewing) return;` — including the thinking-spinner clear and the catch-up `fetchLatestMessage()`.
- `fetchHolder` (`ChatView.tsx:592-602`) runs on every `agentStatus` change (line 608-610). After the final loop ends, `current_holder_session_id` returns `null` (loop task done, poll gone), so `holderSessionId` flips to `null`.

## Hypothesis A (leading): running→idle race collapses `viewingHolder` before the idle effect can clear thinking

### Sequence

```
T0: daemon broadcasts agent.status: idle  (final loop ended)
T1: App.tsx setAgentStatuses({proj: 'idle'})
T2: ChatView re-renders with agentStatus='idle'
T2a: agentIsActive = false  (idle is not in the active set)
T2b: viewingHolder = (sessionId !== undefined) && (holderSessionId === null ? agentIsActive : sessionId === holderSessionId)
     At this render holderSessionId is STILL 'sess_e4277f84' (last fetched on running),
     so viewingHolder = (sessionId === holderSessionId) = TRUE.
T2c: viewingHolderRef.current = true
T3: Effects run (source order):
    Effect 1 (line 608, fetchHolder): triggers REST GET /run-status → async
    Effect 2 (line 748, status handler): reads viewingHolderRef.current = true
                                          → enters idle branch
                                          → setShowThinking(false)  ✓
                                          → fetchLatestMessage() ✓
T4: Effect 1's fetch resolves → setHolderSessionId(null)
T5: ChatView re-renders → viewingHolder = (null === null ? false : ...) = false
    But the spinner is already cleared. No problem.
```

So under this clean sequence Hypothesis A would NOT reproduce the bug.

### Where the race actually occurs

The race only fires if a **prior** render has already collapsed `viewingHolder` to false before the idle event arrives. There are at least three plausible triggers:

**Trigger 1 — late holder fetch overwrites mid-flight:**
Between the `agent.status: waiting → running` transitions (lifecycle restart), `fetchHolder` runs twice in quick succession because both transitions hit the effect deps. If the `waiting`-triggered fetch resolves *after* the `running` re-fetch, it could write a stale value. Less likely the `null` though, since the daemon was reporting the holder correctly throughout.

**Trigger 2 — out-of-order `agent.status` events:**
Both the `_check_sub_agents_done` poll AND the second `_on_loop_done` can broadcast `idle`. If one arrives just before the lifecycle-driven `_start_loop` broadcasts `running`, the sequence becomes `running → waiting → idle → running → idle`. Each `agentStatus='idle'` carries a fetchHolder. The intermediate idle could flip `holderSessionId` to `null` and clear `wasRunningRef` (no — actually the inner idle branch sets `wasRunningRef = false` after fetching). Then the second `running` re-arms `wasRunningRef`. Then the final `idle` runs the effect with `viewingHolderRef.current = ???`. This needs DevTools to confirm.

**Trigger 3 — `viewingHolderRef.current` mutated by an earlier render:**
The ref is updated synchronously on every render (`viewingHolderRef.current = viewingHolder;`). If between the daemon emitting `idle` and React batching the effect, a different re-render (e.g. from a `useSessions` refresh triggered by `agent.status`) recomputes `viewingHolder` as false because of a transient `holderSessionId=null` write, the ref will be `false` when the effect reads it.

## Hypothesis B: `agent.status: idle` is never broadcast at the end of the lifecycle-restart loop

### Why it's plausible

The `lifecycle.on_completed → inject_system_message → _start_loop` chain restarts the management agent's loop. That loop finishes and `_on_loop_done` runs. The "busy sub-agents" check (`agent_manager.py:2608-2623`) uses `list_active` — and the sub-agent that just finished may still be in `_adapters` (idle, not yet reaped).

- `list_active` returns it with `status="idle"`.
- `busy = [a for a in active if a.get("status") != "idle"]` → empty.
- Else branch fires reap + idle broadcast.

So under the normal path `idle` does get sent. **But** if `list_active` returns the sub-agent with a status OTHER than `"idle"` at this moment (e.g. `"completed"` or some intermediate state), `busy` is non-empty and the manager broadcasts `waiting` again instead of `idle`. The UI stays stuck.

This warrants checking `sub_agent_manager.list_active` and the adapter's `status()` returns for a freshly-completed SDK sub-agent.

## Hypothesis C: WebSocket dropped between `running` and the final events

The user already hit one WS disconnect earlier in this session (the original "stuck" state that `Cmd+R` cleared). The desktop app may be losing the WS again under specific timing — possibly when sleep prevention toggles, or when the system briefly suspends. If the WS drops AFTER the dispatch starts but BEFORE the final `idle` broadcast, the UI permanently holds the last-seen `running` status.

The reconnect logic only re-fetches on certain triggers — it does not aggressively re-fetch `agent.status` on every reconnect (we should verify this).

## Hypothesis D: `wasRunningRef` is reset on an intermediate event

`wasRunningRef.current` controls whether `fetchLatestMessage()` fires on `idle`. It is set to `true` only on a `running` transition with `viewing=true`. If a fast `running → idle` happens while viewing was briefly false, then `wasRunningRef` is never set, and the subsequent `running → idle` (the real one) sees `wasRunningRef=false` and skips the catch-up fetch.

This explains the "final message never appears" half of the bug, but not the "thinking spinner stuck" half (which is gated by `setShowThinking(false)` independent of `wasRunningRef`).

## Hypothesis E: The new session_id filter blocks events when sessionIdRef is briefly undefined

The audit added (in `ChatView.tsx`):

```typescript
if (e.session_id && e.session_id !== sessionIdRef.current) return;
```

To `handleApprovalRequest`, `handleApprovalResolved`, `handleActivity`, `handleSubAgentMessage`. **NOT** to `handleStreamDelta` or any agent.status handler (those live in `App.tsx`).

If at mount-time `sessionId` prop is `undefined` and a WS event arrives carrying `session_id="sess_e4277f84"`, the filter evaluates: truthy `e.session_id` AND `"sess_e4277f84" !== undefined` → **drop**.

This would suppress sub-agent message capsules even though they belong to the (about-to-be-viewed) session. The thinking spinner is set by `agentStatus` (App.tsx) — App.tsx does not have the new filter — so the spinner *would* still appear. But the sub-agent capsule would not.

This hypothesis matches the "no sub-agent message appeared, no final reply" half of the bug. It does NOT match the "thinking spinner stays" half if the user was already mounted with `sessionId` set before sending.

Mitigating factor: `sessionId` should be set whenever the user is actively viewing a session that has loaded history. So this race window is small.

## What we cannot determine without DevTools

- Whether `chat.stream_delta` events fired at all (deepseek-v4-pro may not stream short replies — would need provider config inspection or a network-tab capture).
- The actual sequence of `agent.status` values seen by the React tree.
- Whether `holderSessionId` ever flipped to `null` mid-sequence.
- Whether the WS connection stayed up through the full round-trip.

## Recommendation: instrument before fixing

A "shotgun" fix to ChatView risks introducing a different bug. Recommended order of operations:

1. **Instrument** `ChatView.tsx` with a single log statement at each effect entry capturing `agentStatus`, `holderSessionId`, `viewingHolder`, `wasRunningRef.current`, and the current timestamp.
2. **Reproduce** the bug with DevTools console open and the daemon log tailing.
3. Compare actual event sequence with hypotheses. Most likely we land on a refined A or B+C combination.
4. Pick the targeted fix below that the trace supports.

## Design alternatives for the fix

These are not mutually exclusive — the trace decides which combination is needed.

### Option 1 — make idle handling viewing-agnostic for cleanup-only state

Smallest change:

```typescript
if (agentStatus === 'idle' || agentStatus === 'error') {
  setIsCancelling(false);
  setShowThinking(false);          // ALWAYS clear on idle, regardless of viewing
  if (wasRunningRef.current) {
    wasRunningRef.current = false;
    fetchLatestMessage();           // ALWAYS catch up on idle, regardless of viewing
  }
  if (!viewing) return;
  const finalStatus = agentStatus === 'idle' ? 'completed' : 'error';
  setItems((prev) => finalizeLiveCapsule(prev, finalStatus));
}
```

- ✓ Fixes Hypothesis A and D regardless of which underlying race triggered them.
- ✗ `fetchLatestMessage` would fire for non-viewed sessions, costing a REST call per session-switch. Acceptable — it's idempotent.
- ✗ Doesn't address Hypothesis B (daemon didn't broadcast idle in the first place).

### Option 2 — defer the holder fetch until AFTER the status-handler effect runs

Move `fetchHolder` out of the `agentStatus` effect dep list. Trigger it from the status handler effect's idle/running branches instead, AFTER `setShowThinking` and `fetchLatestMessage` have run. This guarantees `viewingHolderRef.current` is whatever it was at the start of the transition.

- ✓ Removes the race entirely.
- ✗ Holder refresh becomes slightly slower (one effect tick later).
- ✗ Requires moving logic around in a file that already has 9 useEffects.

### Option 3 — make `viewingHolder` resilient to transient `holderSessionId=null`

When `holderSessionId` flips from a concrete value to `null`, hold the previous value for one render tick before letting `viewingHolder` recompute as false. This is a small `useRef` plus a coalesced `setTimeout(0)`.

- ✓ Prevents the flicker.
- ✗ Adds state-machine subtlety to a value that is supposed to be derived.
- ✗ Won't help if Trigger 2 (out-of-order events) is the real cause.

### Option 4 — daemon emits `agent.status: idle` with a `final: true` flag

The daemon emits ONE authoritative idle event with `final: true` after the entire dispatch round-trip has wound down (lifecycle.on_completed→restart→loop end→reap). The frontend treats `final: true` as "always run cleanup, ignore viewing gates."

- ✓ Most robust against frontend races.
- ✗ Largest change: requires daemon-side coordination state to know when the round-trip is "actually done." Already tricky given lifecycle is async.

### Option 5 — drop the new `session_id` filter on `chat.sub_agent_message` (revert that part of the audit)

If Hypothesis E turns out to be the cause (sub-agent capsules suppressed), then the filter additive was too aggressive. Falling back to `viewingHolder` only would restore the prior shape.

- ✓ Fixes Hypothesis E.
- ✗ Loses the cross-session safety the audit was designed to add.
- ✗ Better to fix the race that makes `sessionIdRef.current` briefly stale than to drop the filter.

## Concrete next steps

1. **Add minimal logging** to `ChatView.tsx` (2 console.debug calls). Rebuild SPA only — no daemon rebuild needed.
2. **Reproduce** the dispatch + stuck case with DevTools console + Network (WS frames) open.
3. **Compare** the captured trace to the five hypotheses.
4. **Choose** Option 1 (likely sufficient) or Option 2 (more principled) based on trace.
5. **Audit** whether `viewingHolder`-gated cleanup elsewhere in `ChatView.tsx` (lines 772–796) suffers the same race — and apply the same pattern.

## What this investigation explicitly does NOT do

- Does NOT modify any code. The user requested investigation + design only.
- Does NOT commit to a single hypothesis. The bug fits multiple patterns; instrument and confirm before fixing.
- Does NOT roll back any of the audit changes. The audit fix is correct on the backend; this is a separate frontend race that is exposed (not caused) by the audit's making `current_holder_session_id` more accurate.

## Related artifacts

- Audit fix commit: `cd3325d`
- Full audit table: `REPORT-session-id-audit.md`
- Static test that would catch a backend regression: `tests/unit/test_session_id_threading_static.py`
- Daemon log around the event: `~/Library/Application Support/Orbital/logs/daemon.log` lines around `2026-05-28 15:09:02`–`2026-05-28 15:09:27`
- Session JSONL with full round-trip: `/Users/keanezhou/Library/Application Support/Orbital/scratch/orbital/sessions/quick_tasks_93304278.jsonl`
- Sub-agent transcript: `/Users/keanezhou/Library/Application Support/Orbital/scratch/orbital/sub_agents/claude-code/20e6b919.jsonl`
