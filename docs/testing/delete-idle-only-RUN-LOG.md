# RUN-LOG — Delete Is Idle-Only (2026-06-08)

Branch: `fix/delete-idle-only` (off `fix/overnight-ui-bugs-and-queue-test`). Commit: `04aef4c`.

## What shipped
Queue-item delete is now idle-only (server-side reject-running + idle session-JSONL
cleanup, TOCTOU-safe ordering); the frontend disables (not hides) the running item's
delete control and surfaces a server 409. Session-delete already satisfied the model
and was left unchanged. Sub-agent JSONLs are intentionally not deleted (handle-keyed,
no safe per-session mapping — decided with the user).

## Automated test evidence (Part 3 — all green)
- Backend, after the TOCTOU-reorder fix-up:
  - `tests/regression/test_delete_queue_item_reject_running.py` — RED-before/GREEN-after (verified by stashing the route: old code returned 200 + removed the record).
  - `tests/regression/test_delete_session_reject_running.py` — regression lock (already-green; behavior pre-existed at agent_manager.py:2063).
  - `tests/integration/test_delete_queue_item_idle_cleans_jsonl.py` — real TestClient → route → store → filesystem: session JSONL unlinked, queue record gone, unrelated session preserved, sub-agent transcript left in place, no-attempts + multi-distinct-session cases.
  - `tests/integration/test_delete_idle_only_journey.py` — real dispatcher: item rotates to RUNNING → delete 409 with item intact + no phantom stall → pause releases slot → idle delete removes JSONL+record → resume advances remaining items to DONE, no orphan.
  - Adjacency set (session/trigger delete + queue phase1/phase2 + the 4 new files): **25 passed**.
- Frontend: `npx tsc --noEmit` clean; `vitest` **62 passed** (both QueueItemCard suites + useQueue hook).
- Playwright `web/e2e/queue-delete-idle-only.spec.ts` (375×667): asserts the running item's delete control is disabled and an idle item's is enabled. Compiles/loads under Playwright; **skips at runtime** in this environment (the packaged Orbital.app holds the singleton `~/orbital/daemon.pid`, so the harness's isolated daemon can't boot — same documented skip path as the existing `sub-agent-status-stop.spec.ts`).
- Broader suite: 78 pre-existing regression failures (shared `agent_manager.py:1919` fixture KeyError cluster + v5/nonblocking/subagent-limits) — confirmed identical with the change stashed. **Zero new failures from this change.**

## Part 4 — live smoke test: RAN 2026-06-08 (after the user quit the installed app) — ALL PASS

Setup: isolated dev daemon on :8000 (`PYTHON_KEYRING_BACKEND=null`, `AGENT_OS_API_KEY`=live MiniMax key), project `proj_4daccc2f3871` / workspace `/tmp/smoke-delete-ws`, provider minimax/MiniMax-M2.5, hands_off. Onboarding wrote PROJECT_STATE.md (needed a one-message nudge — the model's first turn only greeted; not a delete-path issue). Driver: `/tmp/smoke_driver.py`; evidence: `/tmp/smoke-evidence/summary.json`. Queue: item A = `sleep 25` (long RUNNING window), B/C = quick `echo` tasks. The onboarding session `smoke-delete_2b7964f6.jsonl` served as the unrelated session.

**Step 2 — reject deleting a RUNNING item (evidence):** caught A running with `holder == A.attempts[-1].session_id` (smoke-delete_08fbad82). `DELETE` → **HTTP 409** `{"detail":"Cannot delete a running queue item. Stop or pause it first, then delete."}`. ZERO mutation confirmed: item still present, attempt outcome unchanged (`null`, not CANCELLED), holder unchanged (still A's session). Daemon log line 19: `DELETE ... 409 Conflict`.

**Step 3 — pause → idle:** `queue/stop` → run-status idle, holder `None`; A's stored `item.state` stayed `"running"` (stale flag) — exercising the liveness-vs-stored-flag distinction live.

**Step 4 — idle delete cleans, scoped (evidence):** `DELETE` the paused item → **HTTP 200** `{"status":"removed"}`. Bound session JSONL `smoke-delete_08fbad82.jsonl` GONE; queue record gone; unrelated `smoke-delete_2b7964f6.jsonl` SURVIVED; no `sub_agents/` dir (none spawned). Daemon log line 24: `DELETE ... 200 OK`. (This is the stale-RUNNING-flag-but-not-holder case: delete correctly allowed by liveness.)

**Step 5 — resume drains, no orphan/stall (evidence):** `queue/resume` → B and C both reached `done`, queue auto-idled, final run-status idle / holder `None`. No `reclaim` WARNING in the daemon log; no ERROR/Traceback/500; no leaked `sleep 25` process (item A's shell was cleaned up on pause).

Result: `step2(reject-running)=PASS step4(idle-clean)=PASS step5(drain)=PASS`. Daemon torn down afterward (port 8000 + `~/orbital/daemon.pid` freed for the installed app).
