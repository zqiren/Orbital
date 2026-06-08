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

## Part 4 — live smoke test: DEFERRED (with rationale)
The live smoke test needs a dev daemon on port 8000, but the user's **installed Orbital.app**
currently owns port 8000 and the singleton `~/orbital/daemon.pid`, and the user explicitly
asked to keep that field clear. Starting a second daemon would re-create the documented
mutual-exclusion conflict and disrupt their running app. The in-process integration
journey test (`test_delete_idle_only_journey.py`) exercises the same step sequence the
smoke test would (running-delete rejected → pause → idle delete cleans JSONL+record →
resume drains, no orphan/stall) against a real dispatcher, route, store, and filesystem.

To run the live smoke test later (when the installed app is closed):
1. Quit Orbital.app (frees port 8000 + `~/orbital/daemon.pid`).
2. `PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring AGENT_OS_API_KEY=<key> bash scripts/restart-daemon.sh` (prepend `.venv/bin` to PATH — the script calls bare `python`).
3. Create a project, start a short task; while RUNNING, DELETE the queue item → expect 409 + clear message, session keeps running, slot stays held, daemon log shows no mutation (step-2 evidence).
4. Stop/pause → idle; delete → session JSONL gone from `{workspace}/orbital/sessions/`, queue record gone, an unrelated session's JSONL still present, sub-agent transcripts under `{workspace}/orbital/sub_agents/{handle}/` left in place (step-4 evidence).
5. Multi-item queue: confirm a running item can't be deleted but the queue keeps draining; no `reclaim` WARNING during normal operation.
