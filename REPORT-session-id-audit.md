# REPORT: session_id Threading Audit

**Date:** 2026-05-28
**Branch:** `fix/rotation-by-session-id`
**Scope:** Every code path in the Orbital daemon and frontend that touches
`session_id`, `SessionKey`, `DEFAULT_SESSION_ID`, or any method / broadcast /
endpoint that should be session-scoped.

## Method

Four parallel audit agents read-only-scanned independent file domains:

| Agent | Scope |
|---|---|
| A | `agent_os/daemon_v2/agent_manager.py` (2,709 LoC) |
| B | `sub_agent_manager.py` + `process_manager.py` + `lifecycle_observer.py` + `message_router.py` |
| C | `agents_v2.py` (REST) + `ws.py` + transports + `agent_message.py` tool |
| D | `web/src/` (all frontend) |

Findings were then spot-verified by reading actual file contents at the cited line
numbers. The table below is what survived verification.

---

## Audit Table

### Backend — `agent_manager.py`

| File:Line | Method/Context | session_id source | Status | Action |
|---|---|---|---|---|
| `agent_manager.py:101` | `_idle_poll_tasks` declaration | `dict[str, asyncio.Task]` keyed by bare `project_id` | ❌ BARE KEY | Change to `dict[SessionKey, asyncio.Task]` |
| `agent_manager.py:1160` | `inject_message()` waiting-state check | `_idle_poll_tasks.get(project_id)` | ❌ BARE KEY | Lookup by `make_session_key(project_id, session_id)` |
| `agent_manager.py:1241–1245` | `current_holder_session_id` docstring | references `_idle_poll_tasks[project_id]` | ❌ STALE DOC | Update doc to reflect SessionKey |
| `agent_manager.py:1267` | `current_holder_session_id` Cond 3 | `_idle_poll_tasks.get(project_id)` | ❌ BARE KEY | Lookup by `make_session_key(pid, sid)` (sid is loop var) |
| `agent_manager.py:1347` | `get_run_status()` waiting check | `_idle_poll_tasks.get(project_id)` | ❌ BARE KEY | Lookup by `make_session_key(project_id, session_id)` |
| `agent_manager.py:1652` | `cancel_message()` waiting check | `_idle_poll_tasks.get(project_id)` | ❌ BARE KEY | Lookup by `make_session_key(project_id, session_id)` |
| `agent_manager.py:1702–1706` | `_stop_sub_agents()` doc + pop | doc says "bare project_id"; code pops bare key | ❌ BARE KEY | Pop by `make_session_key(project_id, session_id)`; rewrite doc |
| `agent_manager.py:1954` | `list_sessions()` waiting check | `_idle_poll_tasks.get(project_id)` | ❌ BARE KEY | Lookup by `make_session_key(pid, sid)` (sid is loop var) |
| `agent_manager.py:2606` | `_on_loop_done()` poll registration | `_idle_poll_tasks[project_id] = poll_task` | ❌ BARE KEY | Register under `make_session_key(project_id, session_id)` |
| `agent_manager.py:_resolve_session_id(None)` callers | various | resolves to `DEFAULT_SESSION_ID` | ✅ ACCEPTABLE | Each caller documented; resolve at session-key boundary, not before. No bug found — all callers that get None receive it from a public API where None means "default session" by contract. |

### Backend — `process_manager.py`

| File:Line | Method/Context | session_id source | Status | Action |
|---|---|---|---|---|
| `process_manager.py:31–42` | `start()` signature | accepts `session_id` kwarg | ✅ | (Round-2 fix verified) |
| `process_manager.py:51–56` | `on_completed` (turn_complete) | closure | ✅ | Verified |
| `process_manager.py:72–78` | `chat.sub_agent_message` broadcast | **omits session_id** | ❌ BROADCAST MISSING | Add `"session_id": session_id` |
| `process_manager.py:82–91` | `approval.request` broadcast | **omits session_id** | ❌ BROADCAST MISSING | Add `"session_id": session_id` |
| `process_manager.py:100–105` | `on_completed` (stream end) | closure | ✅ | Verified |
| `process_manager.py:110–113` | `on_error` | closure | ✅ | Verified |

### Backend — `sub_agent_manager.py`

| File:Line | Method/Context | session_id source | Status | Action |
|---|---|---|---|---|
| `sub_agent_manager.py:147–228` | `start()` | param + resolved | ✅ | |
| `sub_agent_manager.py:379–385` | `workspace_claudemd_warning` broadcast | **omits session_id** | ❌ BROADCAST MISSING | Thread session_id through; include in payload |
| `sub_agent_manager.py:401–602` | `_start_from_registry()` | resolved param | ✅ | |
| `sub_agent_manager.py:604–636` | `send()` | param + resolved | ✅ | |
| `sub_agent_manager.py:638–710` | `_dispatch_async()` | param + closure | ✅ | (signature) |
| `sub_agent_manager.py:671–677` | `chat.sub_agent_message` (fallback path) broadcast | **omits session_id** | ❌ BROADCAST MISSING | Add `"session_id": session_id` |
| `sub_agent_manager.py:782–809` | `get_pending_sub_agent_approval()` | param + resolved | ✅ method-side | (Caller in `agents_v2.py:857` does NOT pass session_id — see below) |
| `sub_agent_manager.py:811–828` | `resolve_sub_agent_approval()` | param + resolved | ✅ method-side | (Callers in `agents_v2.py:1294, 1319` do NOT pass session_id — see below) |

### Backend — `lifecycle_observer.py`

| File:Line | Method/Context | session_id source | Status | Action |
|---|---|---|---|---|
| `lifecycle_observer.py:25–37` | `on_started` | param | ✅ | Broadcast includes session_id |
| `lifecycle_observer.py:39–48` | `on_message_routed` | param | ✅ method-side | (Caller in `agents_v2.py:794` does NOT pass session_id — see below) |
| `lifecycle_observer.py:50–63` | `on_completed` | param | ✅ | Broadcast includes session_id |
| `lifecycle_observer.py:65–77` | `on_error` | param | ✅ | Broadcast includes session_id |
| `lifecycle_observer.py:79–96` | `on_failed` | param | ✅ | Broadcast includes session_id |

### Backend — REST endpoints (`agents_v2.py`)

| File:Line | Endpoint/Method | session_id source | Status | Action |
|---|---|---|---|---|
| `agents_v2.py:643–713` | `POST /agents/{pid}/start` → `start_agent()` | body `req.session_id` | ✅ | |
| `agents_v2.py:716–835` | `POST /agents/{pid}/inject` → `inject_message` | body `req.session_id` | ✅ for management; ❌ for sub-agent dispatch (see next row) |
| `agents_v2.py:794–799` | `on_message_routed` call (sub-agent path) | **omitted** | ❌ MISSING | Add `session_id=req.session_id` kwarg |
| `agents_v2.py:849–860` | `GET /agents/{pid}/pending-approval` | URL has no session_id | ❌ ENDPOINT | Add `session_id` query param; thread to both `_agent_manager.get_pending_approval` and `_sub_agent_manager.get_pending_sub_agent_approval` |
| `agents_v2.py:1284–1307` | `POST /agents/{pid}/approve` | body req.session_id (mgr path) ✅; sub-agent path ❌ | ❌ PARTIAL | Pass `session_id=req.session_id` to `resolve_sub_agent_approval` at L1294 |
| `agents_v2.py:1301–1306` | `approval.resolved` broadcast | **omits session_id** | ❌ BROADCAST MISSING | Add `"session_id": req.session_id` |
| `agents_v2.py:1310–1332` | `POST /agents/{pid}/deny` | body req.session_id (mgr path) ✅; sub-agent path ❌ | ❌ PARTIAL | Pass `session_id=req.session_id` to `resolve_sub_agent_approval` at L1319 |
| `agents_v2.py:1326–1331` | `approval.resolved` (deny) broadcast | **omits session_id** | ❌ BROADCAST MISSING | Add `"session_id": req.session_id` |

### Backend — Tools and transports

| File:Line | Component | session_id source | Status | Action |
|---|---|---|---|---|
| `agent_message.py:66–125` | `AgentMessageTool` all sub_agent_manager calls | `self.session_id` (tool ctor) | ✅ | Verified — Round-1 fix held |
| `agent/transports/sdk_transport.py` | no lifecycle hooks | (events flow through ProcessManager) | ✅ | Stream-based — ProcessManager threads session_id correctly |
| `api/ws.py` | `broadcast()` | project_id only | ⚠️ STRUCTURAL | WS sub layer is project-scoped by design. Session filter happens client-side; backend just needs to put `session_id` in payloads so the client can filter. Out of scope for this audit (no broadcast routing change needed). |

### Frontend — `web/src/types.ts` and `ChatView.tsx`

| File:Line | Component/Context | session_id source | Status | Action |
|---|---|---|---|---|
| `types.ts:196–205` | `ActivityEvent` typedef | none | ❌ MISSING | Add optional `session_id?: string` |
| `types.ts:214–223` | `ApprovalRequestEvent` typedef | none | ❌ MISSING | Add optional `session_id?: string` |
| `types.ts:225–230` | `ApprovalResolvedEvent` typedef | none | ❌ MISSING | Add optional `session_id?: string` |
| `types.ts:232–238` | `SubAgentMessageEvent` typedef | none | ❌ MISSING | Add optional `session_id?: string` |
| `ChatView.tsx:538–578` | `fetchPendingApproval()` | URL has no session_id | ❌ NO SEND | Append `?session_id=` query param |
| `ChatView.tsx:895–939` | `handleStreamDelta` | viewingHolderRef | ✅ | Correctly project-scoped (stream is implicit holder) |
| `ChatView.tsx:941–983` | `handleActivity` | project_id + viewingHolderRef | ⚠️ DEPENDS ON BACKEND | Once event carries session_id, filter on it directly |
| `ChatView.tsx:985–1018` | `handleApprovalRequest` | project_id + viewingHolderRef | ⚠️ DEPENDS ON BACKEND | Once event carries session_id, filter on it directly |
| `ChatView.tsx:1034–1053` | `handleSubAgentMessage` | project_id + viewingHolderRef | ⚠️ DEPENDS ON BACKEND | Once event carries session_id, filter on it directly |

### Frontend — Hooks and REST callers (already correct)

| File:Line | Call | Status |
|---|---|---|
| `useAgent.ts:56–66` | `cancelMessage` POST | ✅ session_id in body |
| `useAgent.ts:80–104` | `injectMessage` POST | ✅ session_id in body |
| `useAgent.ts:106–128` | `approveToolCall` POST | ✅ session_id in body |
| `useAgent.ts:130–145` | `denyToolCall` POST | ✅ session_id in body |
| `useChatHistory.ts:31–47` | `loadHistory` GET | ✅ session_id in URL |
| `ChatView.tsx:461–495` | `fetchLatestMessage` GET | ✅ session_id in URL |

---

## Summary of fixes (this commit)

**Backend (Python)**
1. `agent_manager.py`: change `_idle_poll_tasks` key from `str` → `SessionKey`; update all 7 read/write sites (lines 101, 1160, 1267, 1347, 1652, 1706, 1954, 2606). Update docstrings (lines 1241–1245, 1702–1705).
2. `process_manager.py`: add `"session_id": session_id` to two broadcasts (chat.sub_agent_message L72; approval.request L82).
3. `sub_agent_manager.py`: add session_id to `chat.sub_agent_message` broadcast in `_dispatch_async` (L671) and `workspace_claudemd_warning` broadcast in `_maybe_emit_claudemd_warning` (L379) (thread session_id from caller).
4. `agents_v2.py`: pass `session_id=req.session_id` to `on_message_routed` (L794), `resolve_sub_agent_approval` (L1294, L1319); accept `session_id` query param in `GET pending-approval` (L849); include `session_id` in `approval.resolved` broadcasts (L1301, L1326).

**Frontend (TypeScript)**
5. `types.ts`: add optional `session_id?: string` to `ActivityEvent`, `ApprovalRequestEvent`, `ApprovalResolvedEvent`, `SubAgentMessageEvent`.
6. `ChatView.tsx`: filter handlers on `event.session_id` when present (additive over existing holder-based filter); pass `session_id` query param on `fetchPendingApproval`.

**Tests**
7. New file `tests/unit/test_session_id_threading_static.py`: static-analysis tests that grep/AST-walk the source tree and assert (a) every WebSocket broadcast for session-scoped event types carries `session_id`, (b) every `lifecycle_observer.on_*` call site at REST/router level passes `session_id` kwarg, (c) `_idle_poll_tasks` is keyed by `SessionKey` (declaration type matches).

---

## Critical findings (severity-ordered)

### 1. `_idle_poll_tasks` keyed by bare `project_id` (architectural)

7 call sites. Under the single-slot invariant this is *latently* correct (at most one entry per project). It becomes a real bug as soon as the slot enforcement races, or any code path registers two polls under the same project. Fixing it removes the latent footgun and aligns with the rest of the `_handles` SessionKey discipline. **The docstring at `current_holder_session_id:1241` explicitly describes condition 3 as session-scoped — the code does not match the doc.**

### 2. `process_manager.py` broadcasts (`approval.request`, `chat.sub_agent_message`) omit session_id

Discovered in Round 3. The events still flow, but the frontend cannot reliably attribute them to a session — it has to fall back to `viewingHolder` comparison which fails open when holder is null (during fetch race).

### 3. REST endpoint `GET pending-approval` ignores session

The recovery endpoint that mobile clients hit to re-fetch missed approvals defaults to `DEFAULT_SESSION_ID`. In a non-default session, the approval card never shows up after WS reconnect.

### 4. `POST /approve` and `POST /deny` sub-agent paths drop session_id

The management-agent path correctly passes `req.session_id`. The fallback sub-agent path silently drops it, so the approval resolves on `DEFAULT_SESSION_ID` and the real sub-agent's transport is never unblocked.

### 5. Frontend event typedefs lack `session_id`

Without `session_id` in the typedef, even if backend started sending it, TypeScript would discard it. Three event types affected: `ActivityEvent`, `ApprovalRequestEvent`, `SubAgentMessageEvent`.

---

## What was NOT changed

- Single-slot semantics: unchanged.
- `_resolve_session_id(None)` itself: kept. Audit confirmed it is only reached at public API boundaries where `None` is documented to mean "default session". The bug pattern was always callers passing `None` *through* internal paths — now fixed at the call sites.
- WebSocket subscription routing (`ws.py`): broadcasts remain project-scoped; clients filter by `session_id` from payload. Re-routing the WS layer per session is a larger refactor and out of scope.
- Backend `AgentMessageTool`, transports, and `lifecycle_observer` signatures: already correct from rounds 1–2.

## Tests

Two test surfaces:
- **Static analysis** (`test_session_id_threading_static.py`): regex/AST guards that fail CI if a future change reintroduces any of the patterns this audit eliminated.
- **Runtime regression** (extends existing tests): exercise multi-session sub-agent dispatch and verify adapter lookup, completion routing, approval routing all key by the dispatching session — without mocking `process_manager` or `sub_agent_manager`.
