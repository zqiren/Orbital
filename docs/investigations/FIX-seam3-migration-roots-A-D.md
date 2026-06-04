# Fix — seam-3 migration Roots A–D (PR notes)

**Date:** 2026-06-03 · **Branch:** `fix/rotation-by-session-id` · Companion to `INVESTIGATION-seam3-migration-completeness-FINDINGS.md`.

Four production bugs from the incompletely-applied seam-3 session-id migration. One theme (uneven
migration), four distinct fix-points. Red-green per the test-gated workflow.

## Roots fixed (fix sites)

| Root | Sev | Fix site(s) | Fix |
|---|---|---|---|
| **A — @mention inject 404** | critical | `api/routes/agents_v2.py:771,776,777` (send/start/retry), `:786` (ack), `:801` (lifecycle) | Resolve `mention_session_id = req.session_id or session.session_uuid` once and forward it to `send()`/`start()` and stamp it on the ack + lifecycle marker (was dropped → sub-agent resolver hard-raised on None → caught → 404). |
| **B — pending-approval 500** | critical | `daemon_v2/sub_agent_manager.py:834` (`get_pending_sub_agent_approval`) | Tolerate `None` by **scanning all project slates** (mirrors the already-fixed sibling `resolve_sub_agent_approval`). Resolver-semantics fix — **no** try/except swallow. The route `agents_v2.py:868` stops 500-ing. |
| **C — delete running project → 500 + orphan** | high | `api/routes/agents_v2.py:601` (bulk), `:621` (`delete_project`) | Forward the holder: `stop_agent(pid, session_id=current_holder_session_id(pid))` (`is_running` is holder-aware but `stop_agent` is passthrough-None). **Caller-side** — `stop_agent` None-policy unchanged. |
| **D — dropped corrective-turn / lifecycle msg** | high/med | `queue/dispatcher.py:812` | Forward the in-scope `session_id` to `inject_system_message` (was omitted → passthrough-None → `"no_session"` drop). **Caller-side** — `inject_system_message` None-policy unchanged. |

## FO-1 (not fixed — owner-confirmed by-design)

Added a one-line scope comment at `agent/loop.py` (pre-compaction memory-flush executor): documents
that the flush is intentionally not interceptor-gated and scoped to project-state persistence only.
No behavior change.

## Tests

- **Regression (fail-before / pass-after):**
  - Root A — `tests/regression/test_mention_forwards_session_id.py` (3) ✓
  - Root C — `tests/regression/test_delete_running_project_stops_loop.py` (2) ✓
  - Root D — `tests/regression/test_dispatcher_corrective_turn_session_id.py` (1) ✓
  - Root B — the **existing** `tests/regression/test_subagent_pending_approval.py::TestGetPendingSubAgentApproval` (6) — red before, green after, **UNEDITED**.
- **Integration** — `tests/integration/test_seam3_mention_and_recovery_journey.py` (2: @mention journey to a running sub-agent; session-less recovery poll = 200 not 500) ✓
- **Smoke** (isolated in-process daemon, throwaway temp dirs) — `tests/smoke/test_seam3_fixes_smoke.py` (4: A 2xx / B 200 {pending:false} / C clean delete + not running / D delivered-with-session vs dropped-without) ✓
- **Self-inflicted regression fixed:** `tests/regression/test_corrective_turn_on_text_only_exit.py` — its `inject_system_message` test double didn't accept `session_id`; updated to mirror the real signature (`*, session_id=None`). 3/3 green.

## Verification

- `tests/unit tests/platform` (excl. live-sandbox hazard files): **1705 passed, 2 failed** — the 2 are the documented pre-existing env-fails (`test_consumer2_wiring::test_echo_with_null_provider`, `test_pty_reconciliation::...accepts_use_pty`; the 3rd is in an excluded file). No regression.
- `tests/integration tests/regression tests/smoke`: **1106 passed, 83 failed** (down from the 89 baseline — the 6 Root B reds now pass). None of the 83 are my tests; they are the pre-existing stale-`"default"` fixtures.
- Constraints honored: shared callees `stop_agent`/`inject_system_message` None-policy unchanged; the 6 approval tests unedited; no swallowing try/except for B; no F1/`"default"` routing introduced. No production probes/logs added.

## Open follow-ups (NOT done here, by scope)

- The **~83 stale-fixture reds** rewrite — gated on the cancel/stop cluster vs Root C re-check (that cluster overlaps Root C and must be re-verified before any bulk "stale" marking).
- Doc/comment cleanup (e.g. `agent_manager.py:389` "None → DEFAULT_SESSION_ID" stale comment; `_broadcast`/`inject_system_message` "default session" docstrings).
- `test_browser_integration::test_wait_for_text` is a full-suite timing flake (passes in isolation), unrelated to these changes.
