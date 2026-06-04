# INVESTIGATION — Sub-agent approval not rendering (R3-2)

Date: 2026-05-28
Branch: fix/rotation-by-session-id
Daemon binary: `/Applications/Orbital.app` PID 39197 (build `bf3c97c` on top of `7d08d1b`)
Scope: investigate-only; no code modified.

---

## What I looked at

- User report: `docs/investigations/USER-BUG-REPORT-2026-05-28-round3-slot-and-approval.md`.
- Latest Quick Tasks sessions (project `proj_6be4f16fb272`, autonomy = `check_in`):
  - `quick_tasks_b1aa8a5e.jsonl` (HN-post conversation, management `write` approval at line 17/18).
  - `quick_tasks_93304278.jsonl` (sess `sess_e4277f84`, two `claude-code` dispatches).
- Sub-agent transcripts: `ee66dfc6.jsonl`, `f9d99379.jsonl`.
- Daemon log `/Users/keanezhou/Library/Application Support/Orbital/logs/daemon.log` over the 13:55 test window.
- Live processes: orbital PID 39197, claude-code child PID 40917 (started 13:55:20, ELAPSED 07:36 at observation, state `S`).
- Backend code: `agent_os/agent/transports/sdk_transport.py`, `agent_os/daemon_v2/process_manager.py`, `agent_os/daemon_v2/sub_agent_manager.py`, `agent_os/daemon_v2/autonomy.py`, `agent_os/daemon_v2/agent_manager.py`, `agent_os/api/routes/agents_v2.py`.
- Frontend: `web/src/components/ChatView.tsx` (handlers + mount), `web/src/App.tsx`.

---

## Approval flow (so the hypotheses below have something to anchor to)

1. claude-code (spawned by the SDK as `--permission-prompt-tool stdio`) requests permission for a tool. The python-side `claude_agent_sdk` invokes the `can_use_tool` callback wired in `agent_os/agent/transports/sdk_transport.py:82` → `SDKTransport._handle_permission` (`sdk_transport.py:310`).
2. `_handle_permission` (a) tries `should_auto_approve(tool_name, self._autonomy)` (`sdk_transport.py:317`, defined in `agent_os/agent/transports/tool_risk.py:60`). Under CHECK_IN autonomy (the Quick Tasks project setting per `projects.json:8`), only READ tools are silently allowed; Edit/Write/Bash fall through. (b) If not auto-approved, it stores a future on `self._pending_approvals` and a metadata dict on `self._pending_approval_data` (`sdk_transport.py:325-331`), then puts a `permission_request` `TransportEvent` onto `self._event_queue` (`sdk_transport.py:334-343`) and `await`s the future.
3. The SDK transport's `read_stream()` (`sdk_transport.py:246-253`) drains `self._event_queue` — but only when a consumer is iterating it. The consumer is `ProcessManager.start.consume()` (`agent_os/daemon_v2/process_manager.py:45-114`), launched by `SubAgentManager.start_agent` for non-ACP/non-Pipe transports (`sub_agent_manager.py:595-596`).
4. The consumer translates the event into an `OutputChunk` of `chunk_type="approval_request"` via `transport_event_to_chunk` (`agent_os/agent/transports/base.py:52-67`). On that chunk it does TWO things (`process_manager.py:80-91`):

       if chunk.chunk_type == "approval_request":
           metadata = getattr(chunk, 'metadata', {}) or {}
           self._ws.broadcast(project_id, {
               "type": "approval.request",
               "project_id": project_id,
               "what": f"Sub-agent {handle} requests approval",
               "tool_name": metadata.get("tool_name", ""),
               "tool_call_id": metadata.get("request_id", ""),
               "tool_args": metadata.get("tool_input", {}),
               "source": handle,
               "recent_activity": [],
           })

   Note: NO `session_id` field; NO write to the management session JSONL.
5. The browser's `ChatView.handleApprovalRequest` (`web/src/components/ChatView.tsx:985-1018`) filters: `if (e.project_id !== projectId) return;` then `if (!viewingHolderRef.current) return;`. `viewingHolder` is computed at `ChatView.tsx:419-421`:

       const viewingHolder =
         sessionId !== undefined &&
         (holderSessionId === null ? agentIsActive : sessionId === holderSessionId);

   `holderSessionId` comes from `GET /api/v2/agents/{pid}/run-status` (`agents_v2.py:837-846`), which calls `AgentManager.current_holder_session_id` (`agent_manager.py:1224-1270`). The holder is the management-loop session whose Condition 1/2/3 is true; Condition 3 specifically keeps the slot held while sub-agents are still working (`_idle_poll_tasks[project_id]` alive — set at `agent_manager.py:2603-2606`).
6. If the gate passes, `setApprovals` stores the card; the render path at `ChatView.tsx:1979-1996` materialises it via `<ApprovalCard … />`.
7. On user click, `resolve_approval` hits the REST endpoint, which calls `SubAgentManager.resolve_sub_agent_approval` (`sub_agent_manager.py:811-828`); it walks the SessionKey's adapters, finds the one whose transport has the pending request_id, and calls `transport.respond_to_permission` — which completes the future from step 2.

---

## What the evidence does and does not show

- The two `claude-code` dispatches in `sess_e4277f84` (lines 6, 21 of `quick_tasks_93304278.jsonl`) asked for prime numbers and responded with pure text. `f9d99379.jsonl` and `ee66dfc6.jsonl` contain exactly one chunk each, both `chunk_type: "response"`. **Neither dispatch contained a tool-use that would have triggered a permission request.** So the user's "approval wasn't rendered" complaint cannot be diagnosed from these two transcripts alone — there is no evidence claude-code actually emitted a permission_request during the captured window.
- The management agent's `write` tool DID produce a working approval in `quick_tasks_b1aa8a5e.jsonl` line 17 (`"Writing multi-agent-2026.md"` → line 18 `"DISMISSED: User cancelled the session while this approval was pending."`). The DENY entry in `approval_history.jsonl` at `2026-05-28T04:48:31.237097+00:00` (`"deny_reason": "User cancelled while approval was pending"`) corroborates it. So the **management-agent approval path is intact end-to-end**: backend autonomy.py broadcast → frontend render → user dismissal → backend deny resolution.
- PID 40917 (claude-code, child of 39197) started Thu May 28 13:55:20, has lived 7m36s in state `S`, stdout/fd4 still piped to orbital. Its only transcript chunk is at 13:55:28. **The process is idle, awaiting next `query()` from the SDK** — this is normal SDK keep-alive behaviour, not a stuck permission prompt. There is no observable "claude-code is wedged waiting for stdio" right now.
- The daemon log is silent on `approval`, `permission`, `pending`, `stdio` for the entire 13:53–13:58 test window. The SDKTransport logs `auto-approved %s` at DEBUG level only (`sdk_transport.py:318`); default log level wouldn't capture it. So we cannot confirm whether auto-approval ran.
- `projects.json` confirms Quick Tasks autonomy = `check_in`. Under CHECK_IN, `tool_risk.should_auto_approve` returns True only for tools categorised as READ (Read/Glob/Grep/LS/Search/Explore/WebSearch/WebFetch/AskUser per `tool_risk.py:23-35`). Edit/Write/Bash/MultiEdit/NotebookEdit/Agent fall through and MUST raise an approval.

---

## Three potential root causes, ranked by likelihood

### 1. (Most likely) Approval IS broadcast, but the frontend's `viewingHolder` gate drops it because the user switched away from (or never landed on) the management session that initiated the dispatch.

**Hypothesis.** When a sub-agent under session A requests a tool permission, the broadcast carries `project_id` only and no `session_id`. The frontend at `ChatView.tsx:990` drops the event unless the *currently viewed* session is the slot holder. The user dispatched from one session, then either switched session, opened a fresh `quick_tasks` chat, or the slot holder changed (Condition 3 brittleness — see below) before the click-receiving render — and `viewingHolder` evaluated to false.

**Mechanism.**
- Sub-agent's tool call enters `_handle_permission` (`sdk_transport.py:310`). Not auto-approved (Edit/Write/Bash). Event queued.
- `process_manager.consume()` broadcasts `{"type":"approval.request", "project_id":..., "what":..., "tool_call_id":..., "source": handle, "recent_activity":[]}` (`process_manager.py:82-91`). **No `session_id` is included.**
- Frontend `handleApprovalRequest` (`ChatView.tsx:985`):

      if (e.project_id !== projectId) return;
      if (!viewingHolderRef.current) return;

  `viewingHolderRef.current` is recomputed from `sessionId === holderSessionId`. If the viewed session does not match the holder, the event is silently dropped.
- `holderSessionId` comes from `current_holder_session_id` (`agent_manager.py:1224`). That function returns a session ID only if Condition 1, 2, or 3 holds. After `yield_turn: dispatch tool ended the management turn` (which the daemon log shows at 13:55:24,952), Condition 1 (task running) drops; Condition 2 (paused_for_approval) is for management approvals, not sub-agent ones; Condition 3 (`_idle_poll_tasks[project_id]` alive) is the only remaining anchor. If `_check_sub_agents_done` finishes its poll before the sub-agent emits the permission_request, `_idle_poll_tasks[project_id]` gets popped (`agent_manager.py:1706`), `current_holder_session_id` returns None, the frontend's holder fetch sets `holderSessionId=null`, and `viewingHolder` collapses to `agentIsActive` (which is also false once status flips to `idle`).
- The REST recovery path `GET /pending-approval` (`agents_v2.py:849`) DOES check sub-agent pending approvals via `get_pending_sub_agent_approval` (`sub_agent_manager.py:782-809`), which walks `_adapters.get(make_session_key(project_id, session_id))`. **But it requires a session_id at the python layer (`_resolve_session_id` defaults to `DEFAULT_SESSION_ID`), and the route does not pass one** (`agents_v2.py:855-857`). If the sub-agent adapter lives under SessionKey(proj, sess_e4277f84) but recovery resolves to DEFAULT_SESSION_ID, the lookup returns `{}` and recovery never finds the pending approval either.

**Supporting evidence.**
- `process_manager.py:82-91` payload literally has no `session_id` field.
- `ChatView.tsx:990` is the unconditional gate.
- `quick_tasks_b1aa8a5e.jsonl` line 17 (HN session, different session from the prime-number test) shows that a management `write` approval was rendered AND dismissed — proving the FE pipeline works WHEN the viewed session is the holder. The user's report comes from a different sub-agent path that doesn't have the same session-holder coincidence.
- `current_holder_session_id`'s reliance on `_idle_poll_tasks` (`agent_manager.py:1267-1268`) is the only thread keeping the slot held after the management `yield_turn` fires — and the management agent yields BEFORE the sub-agent has done any work (`13:55:24,952` yield in the log; sub-agent finished at `13:55:28`). If `_check_sub_agents_done` clears the poll task between those events, the holder evaporates.

**Ruling-out evidence.** None — I have nothing in the captured window that shows a sub-agent tool requiring approval (both transcripts are text-only). I cannot directly confirm the broadcast went out and was dropped; I can only confirm the gate exists and the broadcast omits session_id.

**Diagnostic to confirm.** Re-run the scenario with one extra step: dispatch a sub-agent and explicitly ask it to run `Bash` (e.g., `claude-code: please run \`ls /tmp\``). When permission is requested:
- In a network tab / WS sniff, confirm an `approval.request` event arrives with `source: "claude-code"`, `tool_name: "Bash"`, and **no `session_id` field**.
- In the React devtools, inspect `viewingHolder` and `holderSessionId` at that moment.
- Hit `GET /api/v2/agents/{proj_id}/pending-approval` directly and observe whether it returns `{"pending": true, ...}` or `{"pending": false}`. If the WS event arrives but `viewingHolder` is false AND the REST endpoint returns `pending: false`, both halves of the hypothesis are confirmed.

---

### 2. The approval IS broadcast and the FE handler accepts it, but the consumer task that converts permission_request → `approval.request` is silently dead because the SDK background task crashed earlier in the session.

**Hypothesis.** The `SDKTransport._consume_response_background` (or the prior `receive_response` iterator) raised an exception that ended without a `ResultMessage`, leaving `_needs_flush=True`. After the next `send()` triggers `_flush_stale_messages` (`sdk_transport.py:222-244`), that flush consumes any queued events — including the `permission_request` — without forwarding them to `process_manager.consume()`. Alternatively, the consumer task in `process_manager.consume()` exited via the `except Exception` arm (`process_manager.py:108-113`) on an earlier chunk and the new `permission_request` event has no consumer.

**Mechanism.**
- The daemon log contains the precedent: at `2026-05-28 11:19:54,281 ERROR claude_agent_sdk._internal.query: Fatal error in message reader: Command failed with exit code 143` and the matching `SDKTransport: background response consumption failed` and `will flush on next send`. This is the exact failure mode that sets `_needs_flush=True`.
- `_flush_stale_messages` iterates `receive_response()` until a `ResultMessage` and `break`s (`sdk_transport.py:236-240`). Items put on `self._event_queue` by the next `_handle_permission` call would in principle still reach `read_stream`, but the consumer in `process_manager.consume()` lives across the whole adapter lifetime — IF it died on a prior exception, its task is gone and `_event_queue` fills up forever.
- `process_manager.start.consume()` catches `Exception as e` and calls `on_error` but does NOT restart itself (`process_manager.py:108-113`). One bad chunk → consumer is dead for the rest of the adapter's life → all subsequent `permission_request` events drop into `_event_queue` and never reach the WS broadcaster.
- Critically: the consumer-died case is recoverable on the SDK side (the `respond_to_permission` API still works — the future is still in `_pending_approvals`) but invisible to the user because no broadcast ever happens. The REST `/pending-approval` endpoint WOULD find it via `get_pending_sub_agent_approval` — provided the SessionKey is resolved correctly (see hypothesis 1's tail).

**Supporting evidence.**
- Daemon log shows the exit-143 / `will flush on next send` event has happened in this exact codebase recently (`2026-05-28 11:19:54`).
- `process_manager.consume()` has no restart logic and no fault-tolerance: a single exception nukes the consumer for the adapter's lifetime.
- claude-code PID 40917 has been alive in state `S` for >7 minutes despite the management agent yielding — fully consistent with an SDK keep-alive that has no consumer draining its events.

**Ruling-out evidence.** No `SDKTransport: background response consumption failed` lines in the 13:53–13:58 test window (only the older one at 11:19). No `[Sub-agent] claude-code completed` lines for f9d99379 in the daemon log either, which is unusual — though the sub-agent's "completed" system message DOES appear in the session JSONL at `quick_tasks_93304278.jsonl:24` at `13:55:28`, so the lifecycle path fired at least once. The consumer task was alive at 13:55:28; it could have died after.

**Diagnostic to confirm.** With PID 40917 still alive, send it a tool request from a fresh dispatch. Check whether `_event_queue` accumulates (look at `id(transport._event_queue)`'s `qsize()` from a debug shell, or attach a probe). Simultaneously, watch the daemon log for any SDK-warning lines between the dispatch and the now-missing broadcast. If queue grows but no broadcast lands, the consumer is dead.

---

### 3. The approval was auto-approved on the wrong autonomy path — either CHECK_IN incorrectly classified the tool as READ, or `_autonomy` on the live SDKTransport was `None` and the tool fell through `should_auto_approve` differently than expected.

**Hypothesis.** `SDKTransport._handle_permission` (`sdk_transport.py:316-319`):

    if self._autonomy is not None and should_auto_approve(tool_name, self._autonomy):
        return PermissionResultAllow()

Either (a) the project's CHECK_IN autonomy maps a tool the user thought was risky (e.g. `WebFetch`, `WebSearch`, `AskUser`) into READ and approves it without surfacing; or (b) the autonomy passed to the SDKTransport was `None` (start path failed to load project autonomy), so the `is not None` guard short-circuits — but then `_handle_permission` would proceed to the queue-and-await branch, NOT auto-approve. Path (a) is the realistic one.

**Mechanism.**
- `sub_agent_manager.start_agent` reads `project.get("autonomy", "check_in")` and constructs `Autonomy(autonomy_str)` (`sub_agent_manager.py:458-465`). Default and fallback are both CHECK_IN.
- `should_auto_approve(tool_name, CHECK_IN)` returns True iff `classify_tool(tool_name) == READ`. The READ set (`tool_risk.py:23-35`) includes `WebSearch`, `WebFetch`, `Explore`, `AskUser`. If the user expected `WebSearch` or `WebFetch` to prompt, it never would under CHECK_IN.
- Any unknown tool (e.g. an MCP tool) classifies as `REQUIRES_APPROVAL` (`tool_risk.py:51-58`) — that case is fine, would prompt.
- Edit/Write/Bash all fall outside READ — those DO prompt.

**Supporting evidence.** The tool_risk table at `tool_risk.py:23-50` is explicit. CHECK_IN auto-approves WebFetch/WebSearch — surprising to a user expecting "any net access prompts."

**Ruling-out evidence.** The user's bug specifies the approval *should have appeared* and did not. They are reporting an absence. Without knowing which tool claude-code tried to invoke, we cannot disprove that it was a READ-classified one auto-approved silently. The session transcripts in `quick_tasks_93304278.jsonl` show tools that returned text only with no Bash/Edit, and the user's complaint is for a fresh dispatch. If their fresh dispatch asked claude-code to "go look something up," `WebFetch` would auto-approve.

**Diagnostic to confirm.** Re-run the scenario, and as part of the user prompt, force a known non-READ tool: "claude-code: edit `/tmp/test.txt` and add the word hi". Edit is in the WRITE category — under CHECK_IN this MUST prompt. If even an Edit dispatch produces no card, hypothesis 1 or 2 is the cause. If it prompts correctly, then the original bug was an auto-approved READ-classified tool and the "missing approval" was actually a correctly-silent CHECK_IN auto-approval.

---

## Cross-cutting note: the session_id omission and the recovery endpoint

Independent of which hypothesis is right, two related defects exist in the current code (no fix proposed, just noting):

- `process_manager.py:82-91` broadcasts `approval.request` without a `session_id`. The frontend's correct-session gate depends entirely on inferring it via the holder fetch, which depends on Condition 3 of `current_holder_session_id` — a transient signal that can drop between the management yield and the sub-agent's permission request.
- The REST recovery endpoint at `agents_v2.py:855-857` calls `_sub_agent_manager.get_pending_sub_agent_approval(project_id)` with no `session_id` argument. `get_pending_sub_agent_approval` internally calls `_resolve_session_id(None)` which returns `DEFAULT_SESSION_ID`. If the sub-agent's adapter slate was created under any non-default `session_id` (which is the multi-session norm after the b48b689 / 7d08d1b changes), the lookup returns `None`. Recovery is broken for non-default sessions even when the broadcast worked.

Both observations point at the same architectural gap: the sub-agent approval path was originally written assuming a single (project, default-session) world, and the multi-session SessionKey refactor wired sub-agent storage into SessionKey buckets without back-filling session_id into the approval broadcast or the recovery endpoint.
