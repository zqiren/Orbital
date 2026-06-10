# FINDINGS — pending-approval 500 (investigation only, no fixes applied)

Date: 2026-06-03 · Branch: fix/rotation-by-session-id
Reproduced empirically against an isolated daemon; load-bearing Phase C/D claims verified by direct code reading (not inferred from "nothing broke").

## HEADLINE (severity)

- **Phase D verdict — FAIL-CLOSED.** The 500 is confined to a READ/recovery endpoint. The actual
  approval GATE is a separate code path that does **not** consult `get_pending_sub_agent_approval`
  and cannot be affected by its 500. A gated action does **not** proceed ungated. This is **not** the
  catastrophic fail-open case. (Verified at `loop.py:721-793`.)
- **Phase C blast radius — real liveness/observability hole, not a safety hole.** In a *degraded*
  state (the primary WS approval event was missed) **and** during the brief window where the SPA polls
  with no `session_id`, the REST recovery backstop 500s and the frontend **silently swallows it**. A
  genuinely-pending approval can then **fail to surface**: the agent stays correctly blocked, but the
  human is never shown the card and the agent waits indefinitely on a decision no one saw.
- **Net severity: not drop-everything (no ungated execution), but more than a footnote.** It is a
  liveness + observability defect on Orbital's core supervision path, plus a sibling functional bug
  (@mention REST → misleading 404). Fix the **class** (Phase E), not the single endpoint.

## Phase A — Reproduce & characterize the trigger

**Reproduced deterministically** (isolated daemon, fresh project):
- `GET /api/v2/agents/{pid}/pending-approval` (no `session_id`) → **HTTP 500** "Internal Server Error";
  daemon logs `ValueError: sub-agent session_id is required … got None`.
- `GET …/pending-approval?session_id=nope_123` → **HTTP 200** `{"pending": false}`.

A1. Trigger shape: a **session-less poll** triggers it — but only when BOTH (a) the main-agent store
returns no pending approval AND (b) `_sub_agent_manager is not None`, so the handler falls through to
the sub-agent branch (`agents_v2.py:866-870`). It is not specific to a real sub-agent approval flow;
any session-less poll in that state 500s.

A2. Frequency: the SPA calls `fetchPendingApproval` from **six** sites (mount, WS-reconnect,
`agentStatus→'pending_approval'`, the `/run-status` poll seeing `pending_approval`, a 5s steady-state
poll while running, and session-switch remount — `ChatView.tsx:987,999,942,1037,1053`). `session_id`
is omitted whenever `sessionIdRef.current` is `undefined` (the transient mount/route-resolution
window — `ChatView.tsx:692`: `const qs = viewed ? \`?session_id=…\` : ''`). So the 500 fires in normal
operation during that window, repeatedly under the timers — exactly why it showed up as "noise."

## Phase B — Contract: caller or endpoint wrong?

B1. `session_id` is declared **OPTIONAL**: `agents_v2.py:852-856`
`async def get_pending_approval(project_id, session_id: str | None = Query(default=None, …))`. The
docstring even describes the omitted case as degraded-but-safe ("silently miss approvals"), i.e. it
was meant to return nothing, not crash.

B2. **The endpoint is wrong, the SPA is right.** The SPA omitting `session_id` during early mount is a
legitimate transient state the contract allows. The bug: the sub-agent branch calls
`_resolve_session_id` (`sub_agent_manager.py:82-98`), whose `_required()` policy **hard-raises** on
`None`. That invariant ("a sub-agent always has a parent session") is correct for sub-agent *lifecycle*
calls but **inapplicable to a REST lookup** where `None` legitimately means "session not yet known."
The sibling lookup `resolve_sub_agent_approval` already handles `None` by scanning all slates;
`get_pending_sub_agent_approval` was left on the hard-raise.

## Phase C — Blast radius (how approvals reach the user)

Two surfacing paths:
- **WS `approval.request` — PRIMARY, carries the FULL payload.** Emitted at `autonomy.py:103-133`
  (main agent) and `process_manager.py:81-93` (sub-agent). Frontend `handleApprovalRequest`
  (`ChatView.tsx:1186-1221`) renders the card **directly from the WS event** — no REST call needed.
- **REST `GET /pending-approval` — REDUNDANT backstop.** Recovers approvals missed due to WS drops,
  tab backgrounding, reloads, or relay-tunnel reconnects. Code comment is explicit: "fallback — the
  WebSocket handler … remains the primary path" (`ChatView.tsx:1049-1050`).

C2. **Redundant, not authoritative** — *when the WS path works.* The card appears from WS before any
poll fires.

C3. **The hole.** If the WS event is missed (the exact case the backstop exists for) AND the recovery
poll fires while `session_id` is unresolved → the poll 500s → frontend `.catch(() => {/* best effort */})`
(`ChatView.tsx:730-732`) **silently discards** it (no state, no retry, no UI). The pending approval is
**never surfaced**. The agent remains correctly blocked, but invisibly — the human is unaware an agent
is waiting. So a real pending approval **can fail to surface**, in a narrow but real degraded window.

## Phase D — Fail-open vs fail-closed (verified, not inferred)

The GATE is agent-side and independent of the 500 read path. Verified at `loop.py:721-793`:
- `loop.py:721-726`: `should_intercept` raises → `should_intercept = True` (**error ⇒ gate**). The
  tool executes only when `should_intercept` is `False`.
- `loop.py:761-787`: `on_intercept` raises → append tool result `"… The tool was not executed."` +
  `continue` (**skip execution**).
- `loop.py:788-793`: normal intercept → `session.pause()` + break (loop exits, tool not executed).
- `loop.py:735-759`: queue-running → block signal, exit (tool not executed).
- Registration (`autonomy.py:103-133`) stores the pending keyed by `tool_call_id` (not session); the
  interceptor is constructed with an already-resolved `session_id` (`agent_manager.py:559-563`, minted
  non-None earlier), so the **write/gate path has no `session_id=None` exposure**.
- Sub-agents run `Autonomy.HANDS_OFF` and gate via an `asyncio.Future` in the SDK transport; cancel/stop
  resolve it to `PermissionResultDeny` — also fail-closed.

There is **no** "approval lookup returned None ⇒ allow" branch, no timeout that releases the action, and
no exception handler that lets the tool through. The tool executes only on an explicit not-intercepted
decision. **Verdict: FAIL-CLOSED.** The 500 cannot cause ungated execution.

## Phase E — Shared root with seam-3 sub-agent plumbing

**ONE shared root: sub-agent `session_id` propagation is incomplete after seam-3** (commit 433912a
turned `_resolve_session_id` from `session_id or DEFAULT` into a hard-raise; it updated
`resolve_sub_agent_approval` to scan-on-None but **missed** the siblings).

Symptoms of the one root:
1. **`get_pending_sub_agent_approval(None)` → 500** (`sub_agent_manager.py:841`). *Reproduced.*
2. **@mention REST inject → misleading 404** (`agents_v2.py:771,776`): the handler calls
   `_sub_agent_manager.send(project_id, target, content)` / `.start(project_id, target)` **without**
   forwarding `req.session_id` (which is available), so `send`/`start` hit `_resolve_session_id(None)`
   → raise → caught by `except Exception: raise HTTPException(404, …)`. Pre-seam-3 this silently routed
   to the "default" sentinel; seam-3 turned it into a crash. *Code-traced and confirmed by reading;
   NOT empirically reproduced here — recommend reproducing before fixing.*

NOT the same root:
- `QueueSnapshot.chat_session_id` (`types.ts`) — stale TS type for a field the backend stopped sending;
  never read; no runtime effect. Separate cleanup.
- The deferred "sub-agent reasoning path drops reasoning" (`process_manager.py`/`sub_agent_manager.py`)
  is about `reasoning_content`, not `session_id`; the session_id stamping there was already fixed
  (cd3325d). Distinct concern.

**Scope implication:** fix the class — wherever `sub_agent_manager` is reached from a REST/recovery
context with a legitimately-`None` session (lookup semantics, not spawn/send/stop), apply the
scan-on-None policy that `resolve_sub_agent_approval` already uses, and/or thread `req.session_id`
through the @mention inject path. Do not patch only the one endpoint.

## STOP-AND-SURFACE — gating answers recorded

- (D) **FAIL-CLOSED** — no ungated execution (verified `loop.py:721-793`). Not a drop-everything defect.
- (C) Blast radius = a **liveness/observability** hole: in a WS-missed + session-less-poll window, a
  real pending approval can be **silently not surfaced**; agent stuck, human unaware.
- (B) Endpoint wrong (hard-raise on an optional param); SPA correct.
- (E) ONE shared root (incomplete seam-3 sub-agent session_id propagation); 2 symptoms (500 +
  @mention 404); queue type staleness is unrelated.
- **Severity: P1-liveness on the core approval path + a functional @mention 404 — fix the class.** Not
  critical-fail-open, but not cosmetic.

No fixes applied.
