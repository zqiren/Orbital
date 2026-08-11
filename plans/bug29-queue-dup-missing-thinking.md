# Bug #29 Investigation — Duplicate messages + missing intermediate thinking on queue-message entry

**Investigated:** 2026-07-23 · claude-code dispatch · branch `feature/triggers-signals-links`
**Verdict:** Both defects are real, pre-existing (not introduced by this branch), and share one
trigger: **a user message sent while the viewed session's agent is busy** — i.e. exactly the
messages that get *queued* instead of delivered immediately. D1 is a backend double-broadcast;
D2 is a frontend state-architecture collision. They fire simultaneously because the same send
takes both code paths at once.

---

## 0. Which "queue" this is

Orbital has three message-queueing paths; the bug lives in the second and third, which are the
ones a chat user actually hits:

| Path | Entry point | When |
|---|---|---|
| Autonomous task queue | Queue tab → `QueueComposer` → `agent_os/queue/dispatcher.py` | Items run in **fresh sessions**; the chat composer is *replaced* by `ComposerDisabledPrompt` while it drains (`ChatView.tsx:3169`), so the user cannot type into chat at all. Not the repro surface. |
| **Same-session queue** (`session._queue`) | Chat composer while the viewed session's loop is mid-turn → `inject_message` Case 1 (`agent_manager.py:1758-1767`) → `queue_message` | Returns `queued_same_session`; drained at `loop.py:667` (mid-run) or `_on_loop_done` (`agent_manager.py:4228`). |
| **Wait-state queue** | Chat composer while the agent is in "waiting" (dispatched a sub-agent, idle-poll alive) → `inject_message` wait-state branch (`agent_manager.py:1891-1911`) | Returns `{"status": "queued"}`; the message is appended to the session immediately, surfaced when the sub-agent finishes. |

The reported symptom pairing ("duplicates" + "tool calls, **sub-agent activity markers**,
thinking indicators disappear") matches typing follow-up messages while the agent is running
tools / waiting on sub-agents — the wait-state and same-session paths.

---

## D1 — Duplicate message bubbles

### Root cause (confirmed, deterministic): double `chat.user_message` broadcast on the wait-state queue path

When a user message arrives while the session is in the waiting state, the backend emits the
same message to the frontend **twice**, once with the dedup nonce and once without it:

1. `agent_os/daemon_v2/agent_manager.py:1893-1901` — the wait-state branch builds `user_msg`
   (including `nonce` when provided) and calls `handle.session.append(user_msg)`.
   `session.append` fires the `on_append` observer (wired at `agent_manager.py:806`), which is
   `ActivityTranslator.on_message` → the **canonical echo**:
   `agent_os/daemon_v2/activity_translator.py:204-212` broadcasts `chat.user_message` **with**
   `nonce: message.get("nonce", "")` and the session_id.

2. `agent_os/daemon_v2/agent_manager.py:1902-1906` — the same branch then broadcasts a **second,
   hand-rolled** `chat.user_message`:

   ```python
   self._broadcast(project_id, {
       "type": "chat.user_message",
       "project_id": project_id,
       "content": content,
   }, session_id=session_id)
   ```

   This payload has **no `nonce` field**. `_broadcast` stamps `session_id` into it
   (`agent_manager.py:404`), so it passes the frontend's session gate.

On the frontend, `handleUserMessage` (`web/src/components/ChatView.tsx:1577-1619`) has exactly
one dedup mechanism — the nonce:

```ts
if (e.nonce && localNoncesRef.current.has(e.nonce)) { ...skip... }   // :1595
if (!e.session_id || e.session_id !== sessionIdRef.current) return;  // :1605
setItems(append user_message bubble)                                 // :1607-1617
```

- Echo #1 (with nonce): correctly skipped on the sending client (its optimistic bubble from
  `handleSend`, `ChatView.tsx:2345-2356`, already renders the message).
- Echo #2 (no nonce): `e.nonce` is falsy → the nonce check is bypassed → session matches →
  **a second bubble is appended**. Every client viewing the session gets the extra copy — the
  sender shows *optimistic bubble + echo #2*, any other client shows *echo #1 + echo #2*.

The duplicate persists for the entire busy window: it is only cleaned up when the next full
history reseed runs, and that reseed is gated on a running→idle transition
(`ChatView.tsx:1193-1222` — the `wasRunningRef` catch-up), which does not fire while the agent
is waiting on sub-agents or hot-resuming through queued messages (see D2 §"why the staleness
persists"). So "each message shows twice" for the whole processing stretch — exactly as
reported.

No other `chat.user_message` producer has this problem: the only two producers in the backend
are the translator (`activity_translator.py:206`) and this wait-state branch
(`agent_manager.py:1903`) — verified by grep; all other inject paths rely solely on the
translator echo.

### Secondary dedup gaps (same family, lower priority)

- `handleUserMessage`'s non-own-echo append (`ChatView.tsx:1607`) has **no idempotency key at
  all** (no event id, no content/timestamp check). A relay redelivery of the same
  `chat.user_message` (a known real phenomenon — the code at `ChatView.tsx:1595-1599` explicitly
  dedups *own* relay retries by nonce TTL) duplicates the bubble on any non-sending client
  (e.g. the phone viewing while the desktop sends). `handleSubAgentMessage`
  (`ChatView.tsx:1547-1575`) and `handleAgentNotify` (`:1621`) have the same unconditional
  append. This is the likely amplifier when duplicates are seen on mobile via the relay.
- Same-session drains that happen at `_on_loop_done` broadcast `chat.pending_dispatched` per
  nonce (`agent_manager.py:4247-4253`), but drains that happen **mid-run** at the top of the
  loop (`agent_os/agent/loop.py:665-688`) broadcast nothing — the FE "Waiting for the current
  response to finish." line (`pendingInputs` overlay) is never cleared on that path and orphans
  under an already-answered bubble. Similarly, the cross-session batch dispatcher `_fire`
  broadcasts `pending_dispatched` **only for `batch[0].nonce`**
  (`agent_manager.py:2256-2262`) — entries 2..N orphan. An orphaned "waiting" line under a
  message whose answer already streamed reads as a stuck/duplicated message and compounds the
  perceived D1.

### Recommended fix (D1)

1. **Delete the hand-rolled broadcast at `agent_manager.py:1902-1906`.** The
   `session.append(user_msg)` one line above already produces the canonical translator echo
   with nonce, timestamp, and session_id. Every other inject path (Case 1b step 8, Case 2,
   loop drains) relies on the translator echo alone; the wait-state branch is the odd one out.
   (If any consumer is found to depend on the second event's exact shape, the alternative is to
   add `"nonce": nonce` and the session timestamp to it — but removal is the right call.)
2. **Per-nonce `chat.pending_dispatched` everywhere a queued message drains:** iterate all
   nonces in `_fire` (`agent_manager.py:2256`), and emit the broadcast for the mid-run drain at
   `loop.py:667` (the loop has no WS access — thread a `on_queue_drained(nonces)` callback onto
   the session/loop the same way `on_append` is wired, and fault-isolate it so a broadcast
   failure can never break the drain).
3. **(Hardening, optional)** give `chat.user_message` / `chat.sub_agent_message` appends a
   cheap idempotency guard for relay retries — e.g. skip when an identical
   `(content, timestamp)` user bubble already sits in the trailing few items. Do **not** dedup
   on content alone (legitimately repeated "ok" messages must survive).

### Regression test (D1)

Backend: drive `inject_message` on a session whose `_idle_poll_tasks` entry is alive (mock
poll task) with a WS-manager spy → assert exactly **one** `chat.user_message` broadcast and
that it carries the nonce. Frontend (vitest): emit a nonce-less `chat.user_message` for the
viewed session after an optimistic send of the same content → assert a single bubble
(this locks the contract from the FE side too).

---

## D2 — Intermediate thinking / tool markers disappear

### Root cause (confirmed, deterministic): mid-turn history reseed wipes the live-only overlay

ChatView uses a transform-once architecture (`FE-1`):

- `rawMessages` holds **persisted** rows fetched from `/chat`; `historyItems` is a `useMemo`
  transform of it (`ChatView.tsx:619-622`).
- The **seed effect** (`ChatView.tsx:642-656`) does `setItems(historyItems)` — a **wholesale
  replace** of the rendered list — every time `historyItems` changes (i.e. every time
  `rawMessages` changes).
- All *intermediate* content of an in-flight turn lives **only in `items`**, appended by live
  WS handlers and never mirrored into `rawMessages`: tool rows via `handleActivity` →
  `appendToLiveCapsule` (`ChatView.tsx:107-148`, `:1476-1491`), streamed reasoning via
  `appendLiveReasoning` (`:158-222`), sub-agent bubbles via `handleSubAgentMessage` (`:1565`),
  finals via the `is_final` branch (`:1383-1399`).

`handleSend` pushes the optimistic user bubble into **both** `items` and `rawMessages`
(`ChatView.tsx:2345-2371`; same pattern in `handlePendingEnqueued` at `:1750-1760` and
`reconcilePending` at `:997-1017`). The comment says this is so the reseed "reproduces this
bubble instead of stomping it" — the bubble was made reseed-proof, but the *reseed itself* was
not made live-tail-proof.

**Sequence when a message is sent mid-turn (the queue paths):**

1. Agent is streaming — `items` = seeded history + live capsule (tool rows, reasoning) +
   sub-agent bubbles + earlier finals of this processing chain.
2. User sends → `setRawMessages(prev => [...prev, bubble])` (`:2362`).
3. `historyItems` recomputes → seed effect fires → `setItems(historyItems)` →
   **everything not backed by `rawMessages` is erased**: the live capsule with every tool
   call/thinking row, live sub-agent markers, and any finals from earlier turns of the chain.
4. What remains: old history + user bubbles. Subsequent finals re-append via `is_final` —
   hence *"only final responses remain visible."*

**Why the wipe persists across the whole queue-processing window (the aggravator):**
`rawMessages` is only refreshed by `refreshRawMessages()` (`:754-770`), which is gated on an
`agentStatus` transition to `idle`/`error` while `wasRunningRef` is set (`:1193-1222`). During
queue processing that transition never happens mid-chain:

- Same-session drains hot-resume directly: `_on_loop_done`'s queued branch
  (`agent_manager.py:4228-4257`) appends + `_start_loop`s and **returns before the idle
  broadcast** (`:4295-4300` is never reached); `_start_loop` then broadcasts `running`
  (`:4064-4069`). Status goes running→running.
- The wait-state broadcasts `waiting` (`:4269-4274`), which also doesn't trigger the catch-up.

So each queued-message entry wipes the live tail, and nothing back-fills `rawMessages` until
the entire chain finally idles — at which point the full refetch (`:1218`) restores persisted
capsules (collapsed). During processing the chat is user bubbles + finals only — "nearly
unreadable," as reported.

The direct-send path is unaffected because its `rawMessages` push happens while the agent is
idle: the live tail is empty and the previous turn was already reconciled, so the reseed is a
no-op visually. This is precisely why the defect *only* manifests on queue-message entry.

### Recommended fix (D2)

**Primary (frontend, XS-S): skip the wholesale reseed while the viewed session's turn is in
flight.** In the seed effect (`ChatView.tsx:642`), early-return when the viewed session is
actively executing (viewed == holder && status running/waiting — read via refs, not closures,
per the React-19 batching rule in CLAUDE.md). The optimistic bubble still renders (handleSend
appends it to `items` directly), and the end-of-chain idle catch-up (`refreshRawMessages` →
new `historyItems` → reseed) restores canonical server truth exactly as today. Net effect: the
live capsule, sub-agent markers and prior finals survive queued sends.

Details to get right:

- Keep the `historyItems.length === 0` gate and the empty-session `setItems([])` path
  (`:1088`) unchanged.
- The "Load earlier" prepend (`:1141`) also mutates `rawMessages`; mid-turn it would now be
  deferred until idle. Acceptable — or detect prepends (length grew at the head) and allow
  them.
- On remount mid-turn (tab switch back), `reconcilePending`'s bubble re-append (`:997`) also
  won't reseed; queued-but-undrained bubbles would not render until idle. If that matters,
  allow the reseed when `items` contains no live capsule (`cap:live:` prefix check) — the
  stomp is only harmful when a live tail exists.

**Secondary (backend, optional but recommended): give the FE a mid-chain catch-up signal.**
Since hot-resume never emits `idle`, the FE can instead refresh raw history on
`chat.pending_dispatched` (turn boundary for a drained queued message — debounced). Combined
with D1-fix #2 (per-nonce dispatched broadcasts from every drain site) this closes the
staleness window generally, so even long chains reconcile turn by turn instead of only at the
end.

### Regression test (D2)

Vitest: seed a session, emit `agent.activity` tool events (live capsule appears), then
simulate a send that returns `queued_same_session` → assert the capsule's tool rows are still
rendered after the send. A second test for the wait-state (`{"status":"queued"}`) shape.
This is the test that today's suite lacks — nothing covers "live capsule survives a mid-turn
send" (checked: no such case in `ChatView.test.tsx`).

---

## Risk assessment

| Change | Risk | Mitigation |
|---|---|---|
| Remove `agent_manager.py:1902-1906` broadcast | A consumer silently depending on the nonce-less event (relay push tiers, mobile). The translator echo still fires with a superset of the fields, so parity should hold. | Grep relay `_should_push` / mobile handling for `chat.user_message` field assumptions; daemon smoke with two WS clients before/after. |
| Per-nonce `pending_dispatched` from `loop.py` drain | The loop must not gain a hard WS dependency; a broadcast raising inside the drain would kill the turn. | Callback injected like `on_append`, wrapped in try/except; no-op default preserves current behavior for tests. |
| Skip reseed while turn in flight | (1) A mid-turn `locale` change or prepend won't re-render until idle. (2) If the idle catch-up fetch fails, `items` keeps the live tail — which is *better* than today's wipe, but means stale history persists until the next successful fetch. (3) Must read holder/status via refs inside the effect (React 19 batching — see the `toggleDirectory` precedent in CLAUDE.md). | Keep the skip narrowly scoped (viewed==holder && running/waiting && a `cap:live:` capsule or non-empty stream exists); everything else reseeds as today. |
| FE idempotency guard on user/sub-agent echoes | Over-aggressive dedup could swallow a genuinely repeated identical message. | Key on `(content, timestamp)` pair or a server event id, never content alone; bound the scan to the trailing window. |

Cross-cutting: all fixes sit in shared paths (wait-state branch, translator, seed effect), so
they are provider-neutral by construction — consistent with the #23/#24 requirement. Per
CLAUDE.md, verification needs unit + vitest + a real daemon smoke (dispatch a sub-agent, type
two follow-ups while it runs, confirm single bubbles and surviving tool markers), and a QR
mobile pass since the relay-retry amplifier only shows on the phone.

---

## File/line index

- `agent_os/daemon_v2/agent_manager.py:1891-1911` — wait-state queue branch; **:1902-1906 is the D1 double-broadcast**
- `agent_os/daemon_v2/activity_translator.py:204-212` — canonical `chat.user_message` echo (with nonce)
- `agent_os/daemon_v2/agent_manager.py:389-405` — `_broadcast` stamps `session_id` (why echo #2 passes the FE gate)
- `agent_os/daemon_v2/agent_manager.py:4228-4257` — `_on_loop_done` queued-drain: hot-resume without idle broadcast (D2 staleness); per-nonce `pending_dispatched` here only
- `agent_os/daemon_v2/agent_manager.py:2194-2274` — cross-session pending dispatch; `:2256-2262` broadcasts only `batch[0].nonce`
- `agent_os/agent/loop.py:665-688` — mid-run same-session drain; no `pending_dispatched`
- `agent_os/daemon_v2/agent_manager.py:1758-1767` — inject Case 1 (`queued_same_session`)
- `web/src/components/ChatView.tsx:642-656` — seed effect (the D2 wipe)
- `web/src/components/ChatView.tsx:2345-2371` — optimistic push into `items` + `rawMessages`
- `web/src/components/ChatView.tsx:1577-1619` — `handleUserMessage` nonce-only dedup (D1 render site)
- `web/src/components/ChatView.tsx:1193-1222` — idle-gated catch-up (`refreshRawMessages`)
- `web/src/components/ChatView.tsx:107-222` — live capsule/reasoning helpers (live-only state)
