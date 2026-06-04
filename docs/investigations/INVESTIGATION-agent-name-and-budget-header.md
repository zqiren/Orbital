# INVESTIGATION — agent name not in chat + inconsistent budget/model header

**Date:** 2026-05-29
**Branch:** `fix/rotation-by-session-id`
**Scope:** read-only. No code changed. Two issues + proposed fixes.

---

## Issue 1 — The chat never shows the configured agent name

### Root cause
`ChatMessage.tsx` hardcodes the literal `'agent'` for management messages; `project.agent_name` is never threaded into the chat render path.

`web/src/components/ChatMessage.tsx:104-114`:
```tsx
const isSubAgent =
  message.type === 'sub_agent_message' ||
  (message.type === 'agent_message' && message.source && message.source !== 'management' && message.source !== 'user');
const senderLabel = isSubAgent && message.source ? message.source : 'agent';
…
<span className="text-secondary">{senderLabel}</span>
```
A management message has `source === 'management'` → `isSubAgent` false → `senderLabel = 'agent'`, always.

`chatTransform` passes the backend `source` straight onto the `agent_message` item; the backend writes management assistant turns with `source = "management"`. `agent_name` is used only for the LLM **prompt identity** (`prompt_builder.py:261`: `name = context.agent_name or context.project_name or "Agent"`) — it is never attached to a message or sent per-message to the FE.

### Where `agent_name` is / isn't used (FE)
```
types.ts:37/58/71      Project / Create / Update interfaces
ProjectDetail.tsx:69   header — only as a FALLBACK for the model name
SettingsView.tsx:61,226  edit form
CreateProject.tsx:84   create form
```
`ChatView` / `ChatMessage` never receive it — even though `ChatView` already holds the full `project` object (`App.tsx:490/498/516` pass `project={selectedProject}`).

### Proposed fix (FE-only, ~5 lines)
1. `ChatView` already has `project` → pass `project.agent_name` to `ChatMessage` (add optional `agentName?: string` prop).
2. In `ChatMessage`, for the management row use `agentName || 'agent'`:
```tsx
const senderLabel = isSubAgent && message.source
  ? message.source
  : (agentName && agentName.trim() ? agentName : 'agent');
```
Sub-agent rows keep their handle (claude-code, …); only the management label changes. No backend change — `agent_name` is already on the project the FE holds.

**Optional polish:** show agent_name initials in the management avatar (today the ◐ glyph), mirroring the new sub-agent badges.

---

## Issue 2 — The model+budget header shows inconsistently

`web/src/components/ProjectDetail.tsx:68-75` (verified verbatim):
```tsx
const modelName = project.model || project.agent_name || '';
const cost = `$${(project.budget_spent_usd ?? 0).toFixed(2)}`;
const label = modelName ? `${modelName} · ${cost}` : cost;
```
`selectedProject = projects.find(p => p.project_id === route.projectId)` (`App.tsx:412-413`); the `projects` array is owned by `useProjects`.

### Root cause — four independent vectors

**(2a) The header reflects a projects list that is only fetched at coarse moments — never after a turn.** Verified `listProjects()` call sites in `App.tsx`: **140, 155** (bootstrap/setup gate), **254** (mount effect, dep `[listProjects]` → once), **348** (after create-project), **386** (a connection/bootstrap handler). The two registered WS handlers are **only** `agent.status` (`:260`) and `agent.status_summary` (`:261`); `handleAgentStatus` (`:204`) writes `agentStatuses` and **does not** refresh `projects`. `refreshProject(...)` (which exists in `useProjects`) is **never called** in `App.tsx`. There is **no per-turn, no idle, and no interval** refresh of the projects list.

→ `budget_spent_usd` and `model` in the header are frozen at whatever the list held at the last mount / setup / create / reconnect. Spend that accrues during a session is invisible until one of those coarse events re-runs `listProjects()`. **This is the "sometimes I see it, sometimes I don't": you see the current value only if a remount/reconnect happened after the spend was persisted; otherwise you see the stale (often $0.00) value.**

**(2b) No backend push for cost.** `agent_manager.py` `on_cost_update`:
```python
def on_cost_update(delta_usd, total_spent_usd):
    self._project_store.update_runtime(project_id, {"budget_spent_usd": round(total_spent_usd, 6)})
```
Writes the store only — **no WS broadcast**. No `project.update` / cost event exists in `agent_os`. So even a correctly-wired FE can't learn the new spend without re-fetching.

**(2c) The model portion is conditional.** When `project.model` is empty (e.g. a project on the global/default LLM with no per-project model pinned — typical for scratch "Quick Tasks"), `modelName` falls back to `agent_name`, and if that's blank too the label collapses to a bare cost (`$0.00`). So whether the "model · $X" shape appears at all depends on whether a model string is set — another "sometimes shows" axis. (Using `agent_name` as the model fallback is also a category error — an identity, not a model.)

**(2d) Runtime spend can reset across daemon restarts.** Both the list and single-get handlers read `budget_spent_usd` from `runtime.budget_spent_usd` (`agents_v2.py`). `runtime` is per-daemon-process. `agent_manager` does seed the loop's `budget_spent_usd` from a persisted value (`persisted_spend`), so cumulative spend may survive a restart — but there's a window after restart, before the project loop is constructed, where the list reports `0.0`. Worth confirming whether `project_store.update_runtime` persists to disk and is reloaded into the list response on cold start.

### Proposed fixes (pick per appetite)

**(A) FE-only freshness — refresh the project on turn-end.** Call `refreshProject(projectId)` (already in `useProjects`, merges one project into the list) on the `agent.status === 'idle'` transition. `App.tsx` already tracks idle (`selectedStatus === 'idle'` at `:86`) and ChatView has the idle effect — wire one `refreshProject` there. One REST call per turn-end; header then shows post-turn spend. No backend change.

**(B) Live push.** Backend: broadcast `project.runtime_update` `{project_id, budget_spent_usd}` from `on_cost_update`. Frontend: add `on('project.runtime_update', …)` in `App.tsx` merging into `projects` via `setProjects`. Live cost during a run, no polling. Larger (new event type, backend+FE) but the "right" fix.

**(C) Model fallback.** When `project.model` is empty, fall back to the **global default** model (`/api/v2/settings` `llm.model`), not to `agent_name`/blank, so the label is meaningful for scratch/global-LLM projects. (Or surface the model from the latest assistant message.) At minimum stop using `agent_name` as the model fallback.

**(D) Persistence (optional/confirm).** Ensure `runtime.budget_spent_usd` persists and is reflected in the list response immediately after daemon start, so a restart doesn't briefly show `$0.00` for a project that has real cumulative spend.

### Recommended minimal combination
- **Issue 1:** thread `agent_name` → `ChatMessage` label (FE-only).
- **Issue 2:** (A) `refreshProject(projectId)` on idle + (C) global-model fallback. Add (B) if you want the cost to tick up live during a run.

Consistent with the earlier `REPORT-frontend-rendering-comprehensive.md` (FE-A5 "budget never pushed", FE-A6 "header model/name placeholder") — same root causes, now pinned to exact call sites.

---

## Files referenced
- `web/src/components/ChatMessage.tsx:104-122`
- `web/src/utils/chatTransform.ts` (assistant → agent_message branch, `source: msg.source`)
- `web/src/components/ProjectDetail.tsx:68-75`
- `web/src/App.tsx` — `listProjects` at :140/:155/:254/:348/:386; WS `on(...)` only `agent.status`/`agent.status_summary` (:260-261); `selectedProject` :412-413
- `web/src/hooks/useProjects.ts` — `listProjects`, `refreshProject` (defined, never called by App), `updateProject`
- `agent_os/daemon_v2/agent_manager.py` — `on_cost_update` (store write, no broadcast); seeds `persisted_spend`
- `agent_os/api/routes/agents_v2.py` — GET projects / GET project: `budget_spent_usd` from `runtime`
- `agent_os/agent/prompt_builder.py:261` — `agent_name` used only for prompt identity

No code modified.
