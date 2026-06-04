# Seam-3 Migration Completeness — Findings

**Date:** 2026-06-03 · **Branch:** `fix/rotation-by-session-id` · **Mode:** investigation only (no fixes applied)
**Method:** inline scouting → 31-agent fan-out workflow (Phase B enumerate / Phase C trace+adversarial-verify / Phase D test reconcile / Phase E trace) → every NEW claim re-verified by hand → runtime reproduction of the @mention 404.

---

## HEADLINE — Phase C fail-open verdict

> **The seam-3 session-resolution machinery is FAIL-CLOSED across every gating path. ZERO fail-open is caused by a session resolution raising, returning `None`, or an approval lookup erroring.** The earlier "fail-closed" verdict — previously verified for ONE gate — **holds codebase-wide** across all 7 gating/approval/permission paths traced and adversarially re-verified.

The broader "every gating path" sweep (wider than the seam-3 question) **incidentally surfaced 2 fail-open paths, but NEITHER is seam-3-induced and neither is triggered by session resolution**:

| # | Path | Trigger | Seam-3? | Status |
|---|------|---------|---------|--------|
| FO-1 | Pre-compaction memory-flush executor (`loop.py:1000-1023`) executes flush `tool_calls` via the tool registry with **no interceptor** | context >80% + flush LLM returns a tool call | **No** (compaction path, no session-id involved) | **Real but LATENT** — the flush calls `complete()` with **no tools** (`openai_compat.py:448` → API gets `tools=None`), so a conformant provider returns no `tool_calls` and nothing executes. Structural defense-in-depth gap; one provider quirk or refactor from triggering. **MEDIUM, pre-existing.** |
| FO-2 | Sub-agent SDK `HANDS_OFF` auto-approve (`sdk_transport.py:398-400` via `tool_risk.py:68-69`); ACP transport unconditional auto-approve (`acp_transport.py:74-93`) | any sub-agent tool requiring permission | **No** (hardcoded `HANDS_OFF` at `sub_agent_manager.py:484`) | **By design** ("dispatching a sub-agent IS the user's approval", documented at `sub_agent_manager.py:476-481`). ACP path is an MVP stub (logged WARNING) and is **not used for claude-code** (routed to SDK). **INFO, pre-existing.** |

**Crucially**, the FO-2 verifier independently re-confirmed that every seam-3 resolution failure mode on that gate is fail-CLOSED: `get_pending_sub_agent_approval(None)` raises → 500 on a read-only endpoint that never touches the permission Future; `resolve_sub_agent_approval(None)` scans → 404, Future untouched; `respond_to_permission` only resolves the Future inside `if tool_call_id in pending`, so any miss is a no-op leaving the tool blocked; the management `approve()` path resolves a wrong/None session to an absent handle → 404, tool never executes. **A 500/404/return-None leaves the gated action BLOCKED — that is fail-closed, not fail-open.**

The two seam-3 fail-open *classes* the brief named — approval gates and **slot integrity** — are both clean: `start_agent(session_id=None)` **mints a fresh uuid** (`agent_manager.py:375`) *before* the cross-session slot guard (`:392`), so `None` is rejected by the guard, never bypasses it.

---

## Phase A — the migration invariant (the classifier)

Resolution of `None` is **caller-class specific**, decided at the call site via `models.resolve_session_id(session_id, *, on_none)` (non-`None` always passes through; only `on_none` varies):

| Context class | Correct `None` behavior | Resolver |
|---|---|---|
| **Lifecycle / spawn / send / stop** | a non-`None` session is a true invariant → **hard-raise / mint** is correct | sub-agent `_resolve_session_id` (raise); `start_agent`/`_sid_inject` (mint) |
| **Lookup / recovery / read** (REST polls, recovery backstops, status reads) | `None` is a legitimate "not yet known" → **graceful** (empty / scan / holder), **never a raise** | `_sid_read` (→ holder → None) |
| **Routing / broadcast** | must key on canonical `session_uuid`; **never** F1/`"default"` | `_resolve_session_id` passthrough-None (→ dropped by strict frontend routing) |

**Classification:** CORRECT · UN-MIGRATED (routes on F1/`"default"`, or fails to forward a known id → wrong-session/404) · OVER-MIGRATED (hard-raises/crashes in a legitimate-`None` context → 500/404) · UNKNOWN.

---

## Phase B — the closed inventory

**257 session-identity sites classified** across 8 backend areas (239) + frontend (18).

| Area | Sites |
|---|---|
| AgentManager (`agent_manager.py`) | 52 |
| SubAgentManager (`sub_agent_manager.py`) | 22 |
| REST routes (`agents_v2.py`) | 26 |
| Queue + dispatcher (`queue/`) | 26 |
| Agent loop + Session (`loop.py`, `session.py`, …) | 22 |
| Sub-agent transports + adapters | 31 |
| Event stamping & lifecycle routing | 39 |
| WS/broadcast routing + misc | 21 |
| Frontend (`web/src`, 15 production files) | 18 |

**Closure: CONFIRMED CLOSED (file-level proven).** Grep census: 28 backend files (666 matches) + 32 frontend files / 15 production (476 matches) = **1142 total matches**. Set-difference of matched-files vs coverage-list = **∅** (zero uncovered files). `api/ws.py` and `daemon_v2/message_router.py` legitimately carry **no** session routing (project-keyed). **No live `"default"`/F1 routing sentinel survives** — the only live `sid == "default"` is `queue/store.py:104` (the sanctioned F1→uuid remap detector); all other `"default"`/`DEFAULT_SESSION_ID` occurrences are comments/docstrings or unrelated (browser namespace, SDK permission-mode). `viewingHolder` is fully retired (comment-only in `ChatView.tsx`/`chatTransform.ts`; strict `session_id` routing complete).

> **Honesty note on site-level closure:** the targeted `agents_v2` enumerator returned 26 sites but **missed** the delete-route `stop_agent(pid)` defect (Root C below); it was caught by the Phase D cross-check and then hand-verified. File-level closure is proven; the multi-angle sweep was what closed the one site-level gap — which is the point of the fan-out.

### Non-CORRECT sites (the actionable inventory)

| Status | Site | Sev | Note |
|---|---|---|---|
| UN-MIGRATED | `agents_v2.py:771` `inject_message` @mention `send()` | **critical** | Root A · omits `req.session_id` → 404 |
| UN-MIGRATED | `agents_v2.py:776` @mention auto-start `start()` | **critical** | Root A (same branch) |
| UN-MIGRATED | `agents_v2.py:777` @mention retry-`send()` | **critical** | Root A (same branch) |
| UN-MIGRATED | `agents_v2.py:783-790` @mention ack broadcast stamps `req.session_id` (None) | high | Root A (frontend drops the ack) |
| UN-MIGRATED | `agents_v2.py:796-802` @mention lifecycle transcript marker → `(pid, None)` | medium | Root A |
| OVER-MIGRATED | `sub_agent_manager.py:841` `get_pending_sub_agent_approval` hard-raises on None | **critical** | Root B · the 500 |
| OVER-MIGRATED | `agents_v2.py:868-870` `GET /pending-approval` forwards None into the hard-raise resolver, no try/except | **critical** | Root B (route side) |
| UN-MIGRATED | `agents_v2.py:601` (`delete_old`) + `:621` (`delete_project`) `stop_agent(pid)` | high | **Root C (NEW)** · `is_running` holder-aware but `stop_agent` passthrough-misses → KeyError/orphan |
| UN-MIGRATED | `queue/dispatcher.py:812-814` corrective-turn `inject_system_message` omits in-scope `session_id` | high | **Root D (NEW)** · on the rotation path |
| OVER-MIGRATED | `agent_manager.py:1001-1002` `inject_system_message` passthrough-None → returns `"no_session"` (drops msg) | medium | **Root D (NEW, callee side)** |
| UNKNOWN (benign) | `agent_manager.py:1845`, `:2808`; `sub_agent_manager.py:823`, `:900`, `:942` | low | read/affect methods; `None` degrades gracefully. Candidates for `_sid_read` tightening, not bugs |

---

## Phase C — fail-open re-check (7 gates, trace + adversarial verify)

| Gate | Trace | Adversarial verifier | Seam-3 verdict |
|---|---|---|---|
| main-agent intercept | fail-closed | found FO-1 (out-of-scope compaction path) | **fail-closed** — `should_intercept` reads no session state; `except → True` (`loop.py:725`); `on_intercept` raise → "tool was not executed" + continue |
| sub-agent HANDS_OFF gate | mixed | found FO-2 (by-design) | **fail-closed** for all resolution failures; ungated path is the by-design HANDS_OFF auto-approve |
| queue-running block | fail-closed | no fail-open | **fail-closed** |
| approval lookup / approve-deny | fail-closed | no fail-open | **fail-closed** — None/error → 404/500/return-None, action stays blocked; `has_result_for` guards double-resolution |
| timeout / auto-release / bypass | fail-closed | no fail-open | **fail-closed** — bypass-all/window require prior human action; inject-while-paused AUTO-DENIES |
| single-slot guard | fail-closed | no fail-open | **fail-closed** — `None` mints a distinct uuid, guard rejects (`agent_manager.py:375→392`) |
| credential / request_access / browser-write | fail-closed | no fail-open | **fail-closed** — `request_credential`→True unconditionally, checked before any bypass |

**Seam-3 fail-open count: 0.** (Sweep-wide fail-open count: 2, both pre-existing / non-seam-3 — FO-1 latent, FO-2 by-design.)

---

## Phase D — reconciliation of the 89 deferred reds

Authoritative current snapshot: **89 failed, 1086 passed** (`tests/integration` + `tests/regression`; +1 Windows-only collection error `test_acl_teardown_revoke.py` — `ctypes.windll` on macOS, unrelated).

| Cluster | Failed | Real bugs |
|---|---|---|
| subagent-lifecycle | 20 | 0 (all stale fixture) |
| approval-gate | 18 | **6 reds → 1 root** (the `get_pending_sub_agent_approval` 500; these tests are **correct**, not stale) |
| cancel-stop-terminate | 15 | 0 stale + flagged Root C (delete routes) |
| send-nonblocking-broadcast | 17 | 0 (all stale) |
| queue-slot-lifecycle | 9 | 0 (incl. `test_session_id_none_does_not_bypass_slot_guard` = **stale fixture**: unstubbed MagicMock project name → TypeError before the guard; guard NOT bypassed) |
| autonomy-misc | 9 | 0 (all stale) |

**Verdict:** the ACTIVE-doc's "asserted benign" hypothesis is **confirmed** — ≈83 of the 89 are stale `"default"`-premise / no-session-id fixtures. The **one** real regression hiding in the noise is Root B (the 6 `TestGetPendingSubAgentApproval` reds correctly catch the over-migrated 500). No *other* hidden real regression surfaced in the reds; the additional real bugs (Roots C, D) were found by code-read in the same sweep, outside test coverage. The 6 get_pending tests must NOT be "fixed" to pass — they pass once Root B is fixed.

---

## Phase E — @mention 404 reproduction (runtime)

**REPRODUCED** against an isolated in-process daemon (`TestClient` on `create_app(data_dir=tmp)`, throwaway temp dirs — no shared Orbital data, no port). `POST /api/v2/agents/{pid}/inject` with `target` set → **`404 {"detail":"No active session for project"}`**.

- **Fires unconditionally:** `send()`'s first body line is `_resolve_session_id(None)` (`sub_agent_manager.py:634`), *above* the lock and the "agent not running" check — so the 404 fires even for a running sub-agent; the auto-start retry branch is unreachable.
- **404 even WITH client-forwarded `session_id`:** confirmed in the repro — the defect is the **route** not forwarding `req.session_id` (`agents_v2.py:771/776/777`), available and correctly used at `:786/:801`. The fix is server-side.
- The brief's "one shared root" holds for the @mention; the investigation as a whole shows **one shared *theme*** (incomplete migration) with **distinct fix-points** (Roots A–D).

---

## Ranked fix list (grouped by root — NO fixes applied)

1. **[CRITICAL] Root A — @mention 404 (UN-MIGRATED).** `agents_v2.py:771,776,777`: forward `session_id=req.session_id` into `send()`/`start()`. Also resolve the ack stamp (`:786`) and lifecycle marker (`:801`) to the same session. *Reproduced at runtime.*
2. **[CRITICAL] Root B — pending-approval 500 (OVER-MIGRATED).** `sub_agent_manager.py:841`: make `get_pending_sub_agent_approval` tolerate `None` by **scanning all project slates**, mirroring its fixed sibling `resolve_sub_agent_approval` (`:879`). (`agents_v2.py:868` then stops 500-ing.) *Confirmed by 6 correct tests.*
3. **[HIGH] Root C — delete running project → 500 + orphaned loop (UN-MIGRATED, NEW).** `agents_v2.py:601,621` call `stop_agent(pid)` with no session_id; `stop_agent` (`:1871`) is passthrough-None and raises `KeyError` while `is_running` (`:1346`) is holder-aware. Fix: forward the holder (`current_holder_session_id(pid)`) **or** give `stop_agent`'s None-policy `_sid_read`. *Hand-verified.*
4. **[HIGH/MED] Root D — dropped corrective-turn / lifecycle notifications (UN-MIGRATED caller + OVER-MIGRATED callee, NEW).** `dispatcher.py:812` forward the in-scope `session_id`; and/or change `inject_system_message`'s None-policy (`agent_manager.py:1000`) from passthrough to `_sid_read`/`_sid_inject` so it doesn't return `"no_session"` and drop the message. *Hand-verified.*
5. **[MED · PRE-EXISTING · NON-SEAM-3] FO-1 — compaction flush executor bypasses interceptor.** `loop.py:1000-1023` executes flush `tool_calls` ungated. Latent (flush exposes no tools), but the prompt asks for tool use. Route flush tool calls through the interceptor, or assert no tool execution in the flush path.
6. **[INFO · PRE-EXISTING · NON-SEAM-3] FO-2 — sub-agent HANDS_OFF auto-approve + ACP MVP stub.** Documented design; ACP unused for claude-code. Flag for awareness; gate ACP behind a real approval UI before ACP ships.
7. **[CLEANUP] Stale docs/comments.** `agent_manager.py:389` ("None → DEFAULT_SESSION_ID" — code mints), `_broadcast` docstring (stamps None, not a sentinel), `inject_system_message` docstring ("default session"). Optionally tighten the 5 UNKNOWN-low read methods to `_sid_read`.
8. **[TEST DEBT] ≈83 stale-fixture reds.** Rewrite to the NEW contract (uuid handle keys, explicit `session_id`, assert the hard-raise where intended). **Do NOT** revive `(pid,"default")` to get green. The 6 `TestGetPendingSubAgentApproval` reds are **real** (Root B) — leave them red until Root B lands.

**Distinct roots: 4 seam-3 production bugs (A critical, B critical, C high, D high/med) + 2 pre-existing non-seam-3 fail-opens (1 latent, 1 by-design) + test debt + doc cleanup.** Larger than the two known symptoms — as the brief predicted.
