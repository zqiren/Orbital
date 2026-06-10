# RUN-LOG — Cold-Start Workspace Scan

**Branch:** `feature/context-md-cold-start` (11 commits off `812ff00`)
**Spec:** `TASK-context-md-cold-start` · **Plan:** `docs/superpowers/plans/2026-06-08-context-md-cold-start.md`
**Date:** 2026-06-08

## What shipped (Tasks 1-11)

| # | Change | Files |
|---|---|---|
| 1 | `find_ripgrep` public alias | `agent_os/agent/tools/grep_tool.py` |
| 2 | Deterministic gitignore-respecting self-bounding skeleton walker | `agent_os/agent/workspace_scan.py` |
| 3 | Cold-start prompt branch (3-stage scan) gated on `PromptContext.cold_start` | `agent_os/agent/prompt_builder.py` |
| 4 | Thread `cold_start` + skeleton injection + `origin="cold_start"` through `start_agent` | `agent_os/daemon_v2/agent_manager.py` |
| 5 | `POST /agents/{id}/cold-start-scan` | `agent_os/api/routes/agents_v2.py` |
| 6 | `is_empty_workspace` on project payload | `agent_os/api/routes/agents_v2.py` |
| 7 | Guard `instructions → project_goals.md` sync (existence check) | `agent_os/api/routes/agents_v2.py` |
| 8 | Consent card + `coldStartScan` hook + `is_empty_workspace` type | `web/src/components/ColdStartCard.tsx`, `useAgent.ts`, `types.ts` |
| 9 | Mount card in ChatView empty state | `web/src/components/ChatView.tsx` |
| 10 | Integration journey | `tests/integration/test_cold_start_scan.py` |
| 11 | Playwright @375×667 | `web/e2e/cold-start-card.spec.ts` |

## Verified — AUTOMATED (green)

- Backend unit suite: **1624 passed, 2 skipped** (no regressions).
- New regression: walker, cold-start prompt, `is_empty_workspace`, sync-guard — **13 passed**.
- Integration (`is_empty_workspace`, scan-starts-session, gate-flip-given-files) — **3 passed**.
- Frontend: `tsc --noEmit` zero errors; ColdStartCard + ChatView + ChatTab — **45 passed**.
- Playwright @375×667 (card on imported, absent on empty, Skip dismisses) — **3 passed**.
- Pre-existing unrelated failure `test_v5_05_prompt_sub_agent_section` confirmed identical at branch base `812ff00`.

## DEFERRED — needs the live daemon (your smoke)

The one leg automated tests can't cover: **the agent actually producing the files from a real conversation** (the flagged finalize risk). Steps:

1. Open the UI (QR → `http://10.64.33.65:5173`, or desktop `http://localhost:5173`).
2. Create a project pointing at a **real non-empty folder**, entering a **real model + API key** (the daemon was started with a dummy global key).
3. Open the project → confirm the **"Scan this workspace?"** card appears (not the plain empty state).
4. Click **Scan** → the agent should greet, read selectively, and **propose** State + draft Goals (writing nothing yet).
5. Reply with an affirmative ("looks good") → confirm the agent writes:
   - `orbital/instructions/project_goals.md` (prompt off-switch), AND
   - `orbital/PROJECT_STATE.md` (via `checkpoint_state` → session-end routine; the `is_onboarding_complete` gate).
   - `orbital/CONTEXT.md` should also be written/updated.
6. **WATCH THE FLAGGED RISK:** if `PROJECT_STATE.md` is NOT written after confirmation (session-end routine returned empty state on a thin transcript), the fallback is to have the Stage-3 prompt instruct a direct `write` of `PROJECT_STATE.md` too. Note it if so.
7. Confirm the card does NOT appear for an **empty** folder (normal onboarding).

Daemon log: `/tmp/orbital-daemon-smoke.log` · Vite log: `/tmp/orbital-vite-smoke.log`.
