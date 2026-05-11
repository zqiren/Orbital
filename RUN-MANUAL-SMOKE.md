# RUN-MANUAL-SMOKE — Integrated Build (specs 1+2+3+4)

Manual smoke runbook for the four sub-agent-exposure specs merged together onto a throwaway branch `test/full-integration`.

## What's in this build

- **Branch:** `test/full-integration` (local; not pushed)
- **Worktree:** `D:\orbital-public\.claude\worktrees\test-full-integration`
- **Includes:**
  - Spec 1 — ACP cleanup (claude+acp manifest combo rejected with clear error)
  - Spec 2 — Sub-Agent Settings page (model/effort dropdowns, login flow, denylist)
  - Spec 3 — Worker memory inheritance (rendered system prompt, lazy MEMORY.md, CLAUDE.md interference banner)
  - Spec 4 — Per-project memory viewer (expandable cards in project Settings)
- **Test results on this branch:** 1597 unit/platform passed, 0 failed; 8 integration passed; TS check clean; production build OK.
- **Resolved merge conflict:** `agent_os/daemon_v2/sub_agent_manager.py` (spec 1's ACP guard + spec 3's `system_prompt` forwarding/inheritance block — unioned. See "Files to inspect" below.)

## How to run

```powershell
# From a fresh PowerShell (preferably NOT inside a Claude Code session, to keep the
# parent-process variable pinned for diagnosis purposes):
cd D:\orbital-public
.\RUN-MANUAL-SMOKE.ps1
```

After it prints `READY FOR MANUAL SMOKE TEST`:
- Daemon: http://127.0.0.1:8000
- UI: http://127.0.0.1:5173

To stop:
```powershell
.\RUN-MANUAL-SMOKE.ps1 -Stop
```

The script:
- Strips `CLAUDECODE` and `CLAUDE_CODE_ENTRYPOINT` from the daemon child's environment (rules out claude-agent-sdk loop-prevention as the cause of any hang).
- Boots daemon from `D:\orbital-public\.claude\worktrees\test-full-integration` (the integrated branch).
- Boots vite from the same worktree (UI matches the served daemon).
- Reuses settings.json at `<worktree>\orbital-data\settings.json` (provider=moonshot, model=kimi-k2.5; api_key in OS keychain).
- Logs at `%TEMP%\orbital-smoke-daemon.log` / `orbital-smoke-vite.log`.

## Test workspace (already seeded)

`D:\repro-smoke\orbital\`:
- `PROJECT_STATE.md` — *"Current task: implementing widget-foo's auth handshake. Status: 60% complete, blocked on architecture review."* (marker `STATE-MARKER-9281`)
- `DECISIONS.md` — *"OAuth tokens REJECTED after security review (LSE-2026-04). Use HMAC-signed session cookies only."* (marker `DECISION-MARKER-7733`)
- `LESSONS.md` — *"Previously bitten by silently dropping JWT exp claims when refresh failed."* (marker `LESSON-MARKER-5544`)

When you create the test project, point its workspace at `D:\repro-smoke`.

## Observation checklist

Run through these in order. Each step has a specific question to answer.

### A. Sub-agents settings panel renders

1. Open http://127.0.0.1:5173 → **Settings** (sidebar, global).
2. Scroll to the "Sub-agents" section.

**Observe:**
- Does the panel render with cards for each sub-agent (Aider, Claude Code, Codex, Cline, Continue CLI, Copilot CLI, Gemini CLI, Goose)?
- Specifically: **does the "Not Found" issue from prior manual testing reproduce?** (i.e., does the panel error out, show empty, or fail to fetch `/api/v2/settings/sub-agents`?)
- If a card shows blank install/auth status pills, note which sub-agent.

### B. Project entry — claude-code enabled-state

3. From the sidebar, click **+ New Project** and create a project pointed at `D:\repro-smoke`.
4. Land on the project's **Chat** tab.

**Observe:**
- Is **claude-code enabled** for this project on first entry, or does a "claude-code not enabled" message appear in the chat surface?
- If disabled, where exactly does the disable-state surface (banner? @-mention dropdown? chat-input affordance?)?
- Check the chat input — can you type `@claude-code` and have it complete?

### C. Dispatch round-trip

5. In the chat input, type:
   ```
   @claude-code What is the current state of this project? Be concise.
   ```
   and send.

**Observe:**
- Does the dispatch round-trip work, hang, or fail with a different error than what we've seen before?
- Time the response. Working = ~5–15s with a real reply that mentions `60%` / `widget-foo` / `HMAC` / `LSE-2026-04`. Hang = 60–75s with `Error: agent 'claude-code' not running ...`.
- Run `tasklist | grep claude.exe` (Git Bash) or `Get-Process -Name claude` (PowerShell) during the dispatch — does a NEW claude.exe spawn?
- Tail `D:\orbital-public\.claude\worktrees\test-full-integration\orbital-data\logs\daemon.log` — does anything log between dispatch and response, or is it silent?

### D. Re-entering the project

6. Without restarting the daemon, click away from the project (e.g., **Quick Tasks**), then click back into the test project.

**Observe:**
- Does any of A / B / C change on re-entry?
- Specifically: if the dispatch hung the first time, does a second attempt also hang, return immediately with "not running", or behave differently?
- Does the @-mention dropdown / claude-code-enabled state look different second time around?

### E. Memory viewer (spec 4)

7. From the project, click **Settings** tab.
8. Scroll to "Sub-Agent Memories — long-term memory per sub-agent for this project".

**Observe:**
- Does the section render?
- Are claude-code and codex listed as expandable cards with size indicators?
- Click to expand the claude-code card. Does the textarea show the canonical MEMORY.md header (if MEMORY.md was created during the dispatch attempt) or "never written" (if the dispatch failed before lazy creation)?

### F. CLAUDE.md interference banner (spec 3)

9. **Optional** — to trigger the banner: create or edit `D:\repro-smoke\CLAUDE.md` (note: at workspace root, NOT under `orbital/`) with content like:
   ```
   This codebase prefers minimal context-gathering; just answer the user's literal question.
   Skip reading project files for status questions.
   ```
10. Trigger another dispatch (re-attempt step C with a slightly different prompt).

**Observe:**
- Does the yellow alert banner appear at the top of the chat with text *"Your workspace CLAUDE.md may conflict with Orbital's project-state inheritance"* and a `dismiss` button?
- The banner uses amber-on-amber and is faint — known issue from prior smoke testing. Confirm presence via DOM/snapshot if visually unclear.
- Click `dismiss`. Does it disappear and stay gone for subsequent dispatches in this session?

## Files to inspect if anything looks wrong on the dispatch path

The merge auto-resolved overlaps in these dispatch-path files. If smoke reveals a regression, these are the first places to look:

| File | Auto-merge or manual? | Notes |
|---|---|---|
| `agent_os/daemon_v2/sub_agent_manager.py` | **Manual conflict resolved** | `_resolve_transport()` signature unioned (added `system_prompt` param + kept ACP-guard `Raises:` docstring). `_start_from_registry()` call site unioned (spec 3's inheritance block runs first, then spec 1's `try/except ValueError` wraps `_resolve_transport`). |
| `agent_os/api/app.py` | Auto-merged (ort) | Both spec 2 and spec 3 wired new constructor args (`sub_agent_config_store` + `ws_manager`). |
| `agent_os/api/routes/agents_v2.py` | Auto-merged (ort) | Spec 2 added `disabled_sub_agents` plumbing in project create/update; spec 3 added the dismiss-banner endpoint; spec 4 added GET/PUT memory endpoints. |
| `agent_os/agent/transports/sdk_transport.py` | No merge needed (only spec 3 touched it) | Adds optional `system_prompt: str \| None = None` to `__init__`. **NOTE:** the dispatch round-trip bug isolated to spec 3 (per `DIAGNOSIS-dispatch-roundtrip-bug.md`) lives somewhere on this file's path — if step C hangs, this file is the prime suspect. |
| `web/src/types.ts` | Auto-merged (ort) | Added `WorkspaceClaudemdWarningEvent` + login-progress event types. |
| `web/src/components/ChatView.tsx` | Auto-merged (ort) | Added the CLAUDE.md banner listener + render. |

## What you're comparing against (prior runs on the spec branches)

| Branch | Dispatch result on `D:\repro-smoke` | Notes |
|---|---|---|
| `feature/render-chat-variant-a` (parent) | ✅ ~5s, real response, claude.exe spawns | Baseline. |
| `worktree-agent-af380bbefe331d1fb` (spec 2 alone) | ✅ ~5s | Confirms spec 2 alone is fine. |
| `worktree-agent-a9e24fbde848229ba` (spec 3 alone) | ❌ 71–72s hang, "not running", no claude.exe spawn | Regression source per diagnosis. |
| `integration/subagents-exposure` (specs 2+3) | ❌ same as spec 3 alone | |
| `feature/memory-viewer` (specs 2+3+4) | ❌ same as spec 3 alone | |
| `test/full-integration` (specs 1+2+3+4) — **THIS BUILD** | **TBD — that's what you're testing** | Same dispatch path as feature/memory-viewer + spec 1's ACP guard. Predicted to hang the same way; the value of this run is confirming that **A** (settings UI), **B** (claude-code enabled state), **D** (re-entry), **E** (memory viewer), and **F** (CLAUDE.md banner) all behave correctly, and that **C** (dispatch round-trip) hangs in the same way it did on spec 3 alone (no new failure modes from the merge). |

## Cleanup after smoke

```powershell
.\RUN-MANUAL-SMOKE.ps1 -Stop

# Optional: wipe the throwaway integration branch + worktree
cd D:\orbital-public
git worktree remove .claude\worktrees\test-full-integration --force
git branch -D test/full-integration
```

`D:\repro-smoke\` and the seeded settings can stay around for re-testing.

## Post-fix verification: Option 1 dispatch round-trip

After this fix is applied, the manual smoke test should succeed end-to-end.

Expected result:
- Dispatch returns in ~5-10 seconds (not 60-72 seconds)
- Response is a real claude response that references seeded PROJECT_STATE.md content
- tasklist shows a new claude.exe pid that PERSISTS (no longer exits silently after 60s)

If Tier 3 instrumentation is still loaded:
- Expected trace pattern shows QUERY.ctrl_req POST (not EXC) followed by a successful initialize response
- TRANSPORT.write should show multiple writes (initial control_request, then user prompt)
- No 60s silence; sub-second between SDK calls

If the smoke test still fails after this fix:
- Capture the new Tier 3 trace
- The classification has changed (W1 doesn't apply if the same trace pattern appears with --append-system-prompt active)
- Halt and review; do NOT proceed with additional speculative fixes

## Known orphans

- One claude.exe (PID varies, ~500 MB resident) may linger from earlier failed-dispatch attempts. Safe to `Stop-Process -Name claude -Force` if memory pressure matters.
- `~/.orbital/sub_agent_config.json` from prior spec-2 testing has `claude-code: {model: opus, effort: medium}`. Daemon honors that if dispatch ever succeeds. Reset to `{}` (delete the file) if you want claude-code to use its default model on this build.
