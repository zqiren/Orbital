# Holistic Investigation: 3 Blocking Bugs (settings nav + queue subsystem)

**Branch:** check/phase-1b-foundation (latest code) · **Running daemon:** new build, PID 20325 @ :8000
**Project under test:** orbital-marketing (`proj_e1164e981494`)
**Status:** Diagnosis complete via 3 parallel read-only agents. **No code changed.**
**Date:** 2026-05-25

---

## Executive summary

| Bug | Domain | Top root cause | Shares cause? |
|---|---|---|---|
| **1** | Frontend routing | `handleSelectProject` updates `route` but never clears the `showGlobalSettings` overlay flag; content panels are gated on `!showGlobalSettings` | independent |
| **2** | Queue subsystem | Empty/agentless queue persists as `state:"running"`; `/queue/stop` 409s because no dispatcher exists; frontend swallows the error | **shared with Bug 3** |
| **3** | Queue subsystem | Same phantom `RUNNING`: `auto_idle_if_empty()` only runs inside the dispatcher loop, which exists only while an agent runs — so an agentless queue reports "running" forever and nothing advances | **shared with Bug 2** |

**The headline finding: Bugs 2 and 3 are the same defect seen from two angles** — the persistent `QueueStore` state is *decoupled from dispatcher/agent liveness*. A fresh `queue.json` defaults to `RUNNING` and is only ever demoted to `IDLE` from inside the dispatcher's run loop (`auto_idle_if_empty()`), and the dispatcher only exists after an agent is started. So with no agent running, the queue claims "running" forever (→ Bug 3: nothing advances), the UI renders Stop/Pause as live controls, and `/queue/stop` rejects with **409 "No active dispatcher"** (→ Bug 2: can't stop/pause). One reconciliation fix resolves both.

**Live evidence (both queue agents, independently):**
- `GET /api/v2/projects/proj_e1164e981494/queue` → `{"version":1,"state":"running","items":[]}`
- `GET /api/v2/agents/proj_e1164e981494/run-status` → `{"status":"idle","current_holder_session_id":null}`
- `daemon.log` since the 13:01 restart contains **zero** "dispatcher" lines → no dispatcher ever ran.
- `queue.json` mtime 13:01:19 = the default-`RUNNING` auto-save on first read.

---

## Bug 1 — Global Settings won't dismiss when selecting a project

**Symptom:** With Global Settings open, clicking a project in the sidebar does not switch to the project view; the settings overlay lingers.

**How it works:** Navigation is pure React state (no URL router, `web/src/route.ts:5-9`). Two independent state pieces govern the main panel:
- `route` — `App.tsx:63` `useState<Route>({name:'list'})`.
- `showGlobalSettings` — a **separate overlay boolean**, `App.tsx:64-65`.

Every content panel is gated on `!showGlobalSettings` (`App.tsx:479` create, `:489` project, `:541` list, `:560` blocked); the overlay renders when `showGlobalSettings` is truthy (`App.tsx:473`). The overlay's *only* writers are: open via Sidebar's bottom "Settings" button → `onSettings` → `setShowGlobalSettings(true)` (`App.tsx:444-447`); close via `GlobalSettings` `onBack` → `setShowGlobalSettings(false)` (`App.tsx:475`). Project-click goes Sidebar (`Sidebar.tsx:99/137` → `:61-63`) → App `handleSelectProject` (`App.tsx:356-359`) which sets `route` + `mobileView` only. (The header gear `SettingsIcon.tsx` is the *per-project* `route.settings` opener — a red herring.)

### Candidate root causes
1. **HIGH — `handleSelectProject` never clears `showGlobalSettings`.** `App.tsx:356-359` mutates only `route`; the project panel (`App.tsx:489`, guarded `!showGlobalSettings`) stays suppressed while the overlay (`App.tsx:473`) keeps winning. *Solution:* add `setShowGlobalSettings(false)` in `handleSelectProject` (and `handleNewProject` `App.tsx:361-364` + the SW `navigate` handler `App.tsx:339-342`, which have the same exposure). *Impact:* very low; just flips an already-relevant flag. If only `handleSelectProject` is fixed, New-Project/SW nav can still strand the user.
2. **MED — Structural: the overlay flag globally gates all route panels.** The `!showGlobalSettings &&` guards (`App.tsx:479/489/541/560`) mean *every* route change must remember to clear the flag. *Solution:* either a `useEffect` keyed on `route` that resets `showGlobalSettings`, or fold global settings into the `Route` union (`route.ts:11-15`, e.g. `{name:'settings'}`) so one state decides the panel. *Impact:* the effect is small but adds an implicit reset (verify it doesn't fight the open path, which sets the flag without changing route); folding into `Route` is cleanest long-term but touches `route.ts`, all guard sites, `Sidebar` active-state, and the already-modified `route.test.tsx`/`Sidebar.test.tsx`.
3. **LOW — (ruled out) wrong/stale sidebar callback.** Verified the wiring is correct (`App.tsx:442` `onSelectProject={handleSelectProject}`); the route *does* change, the panel is merely suppressed. No change warranted; documented to close the hypothesis.

**Confirm:** the only `setShowGlobalSettings(false)` in the codebase is `App.tsx:475` (overlay Back). Runtime: open settings → click project → React DevTools shows `route.name==='project'` while `showGlobalSettings` stays `true`.

---

## Bugs 2 & 3 — Queue subsystem (shared root cause)

**Bug 2 symptom:** On an empty queue, the **Stop** control (Queue tab) and the **Pause** control (Chat tab) silently do nothing.
**Bug 3 symptom:** The queue shows "Running" but never advances/executes items.

### The two queues (the seam where the bug lives)
- **Persistent `QueueStore`** — backs the Queue tab; file-backed `queue.json` per project (`agent_os/queue/store.py:39-118`). **Defaults to `RUNNING`** (`agent_os/queue/models.py:85`).
- **In-loop hot-resume queue** `session._queue` — drained inside the loop (`agent_os/agent/loop.py:273`).

Advancement of the persistent queue is the **`QueueDispatcher`** (`agent_os/queue/dispatcher.py`): `_run()` loops `auto_idle_if_empty()` → `next_queued()` → `_dispatch_one()` → `_await_and_handle()` → rotate session + mark item DONE/BLOCKED. The dispatcher is created **only** by `_ensure_dispatcher` (`agent_manager.py:2183-2217`), called from **exactly one place** — `start_agent` (`agent_manager.py:837`). (The comment one line above, `:836` "Phase 1 — passive; Phase 2+ wires advancement", is **stale/misleading** — advancement *is* wired on `:837`.)

### Shared root cause (HIGH) — phantom `RUNNING` decoupled from dispatcher/agent liveness
`QueueState` defaults to `RUNNING` (`models.py:85`); the only thing that demotes an empty queue to `IDLE` is `auto_idle_if_empty()`, called **only inside the dispatcher `_run` loop** (`dispatcher.py:448,655,719,739,801`). With no agent running there is no dispatcher, so `queue.json` stays `RUNNING` forever. The UI reads `snapshot.state` verbatim (`QueueHeader.tsx:38/53`, `useQueue.ts:28-31`) → shows "Running" + Stop. `GET /queue` returns the raw state without reconciling (`agents_v2.py:966-969`, `store.py:341-343`).
- **→ Bug 3:** "running" with no dispatcher behind it = nothing advances.
- **→ Bug 2:** `/queue/stop` requires a live dispatcher and 409s otherwise: `agents_v2.py:1056-1067` (`get_dispatcher` None → `HTTPException(409, "No active dispatcher; start the agent first")`).

The codebase already half-knows this: `start_queue` special-cases "RUNNING && has_handle" and notes the "state-without-handle … defaulted to RUNNING but no agent started" hazard (`agents_v2.py:1128-1133`).

*Shared solution:* reconcile the displayed/effective queue state with liveness on the read path — run `auto_idle_if_empty()` (or compute an effective `idle` when there are no queueable items / no dispatcher) inside `get_queue` (`agents_v2.py:966-969`) or `store.snapshot()` (`store.py:341-343`); and/or default `QueueState.state` to `IDLE` (`models.py:85`). *Impact:* read-side reconciliation is low-risk; changing the persisted default interacts with `start_queue`'s no-op check and daemon-restart resume — regression-test those.

### Bug 2 — additional causes
2. **MED — `/queue/stop` hard-409s instead of treating "nothing to stop" as success.** Stopping an empty/stopped queue is a no-op success, not a client error. *Solution:* in `stop_queue` (`agents_v2.py:1056-1067`), when no dispatcher, set persisted state to PAUSED/IDLE via `_resolve_queue_store`, broadcast `queue.state_changed`, return 200. *Impact:* makes the controls idempotent; pick PAUSED vs IDLE carefully so it doesn't fight auto-idle.
3. **LOW — frontend swallows the 409 (silent no-op).** `QueueHeader.tsx:67` does `void onStop()` (discards rejection); `useQueue.stopQueue` never routes errors to its `error` state (`useQueue.ts:105-108`, error only set in `refresh()` `:32`); `ComposerDisabledPrompt.tsx:22-30` awaits with no `catch`. *Solution:* catch + surface the error (toast/inline) and gate the Stop button on effective non-idle state. *Impact:* observability only; pair with the HIGH fix.

### Bug 3 — additional causes
2. **MED — `_dispatch_one` handoff swallowed by the in-loop queue.** When a dispatcher *does* exist: auto-start launches `loop.run(None)` (`agent_manager.py:816`) *then* starts the dispatcher (`:837`). If that first run is still alive when the dispatcher injects the item, `inject_message` takes **Case 1** (`agent_manager.py:1139-1143`) and dumps the wrapped item into `session._queue`, returning `"queued"` — not a dedicated run. The loop may consume it as an ordinary message and exit text-only, leaving `_exit_reason="text"` → treated as a contract violation rather than a clean advance. *(Amplified by prior-session findings: `loop.run()` has no re-entrancy guard and loops can survive `new_session` via `asyncio.shield`.)* *Solution:* in `_dispatch_one` (`dispatcher.py:504-567`) ensure the loop is idle before injecting (gate on `get_run_status==idle`/`task.done()`), and/or have `inject_message` distinguish queue-item injections so they're never folded into `session._queue`. *Impact:* medium — touches the dispatch/inject contract and loop lifecycle; sequence against the auto-start `run(None)` and `_on_loop_done` hot-resume.
3. **LOW — stale "Phase 1 passive" comment (`agent_manager.py:836`) misdirects diagnosis.** Advancement is actually wired on the next line. *Solution:* fix/remove the comment; audit other passive-era assumptions (e.g. UI trusting `state` verbatim). *Impact:* docs only.

**Confirm:** live `GET /queue`=`running`/`items:[]` + `run-status`=`idle` + zero "dispatcher" lines in `daemon.log` + `queue.json` mtime = restart time. `auto_idle_if_empty` appears only on the dispatcher path (grep). Optional non-paid check: `POST /queue/stop` → expect `409 "No active dispatcher; start the agent first"`.

---

## Recommended fix sequencing (when authorized — no code yet)
1. **Queue (fixes Bug 2 + Bug 3 together):** read-path state reconciliation so an empty/agentless queue reports `idle` (HIGH). Then make `/queue/stop` no-op-succeed without a dispatcher (Bug 2 MED) and surface errors in the UI (Bug 2 LOW). Then close the `_dispatch_one`/`inject_message` Case-1 race for real items (Bug 3 MED).
2. **Settings (Bug 1):** add `setShowGlobalSettings(false)` to `handleSelectProject` (+ `handleNewProject` + SW nav) as the minimal fix; consider folding settings into `Route` as the durable fix.
3. Update the stale dispatcher comment (`agent_manager.py:836`).

Each should land with a regression test (queue: agentless-RUNNING reconciles to idle + stop succeeds; settings: selecting a project from the settings overlay renders the project).
