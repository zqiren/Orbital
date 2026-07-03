# Feature Backlog

> Managed with Orbital, dogfooded on its own repo. This file is **tracked in git** —
> it's the shared, reviewable roadmap. Orbital's runtime state lives in the
> gitignored `orbital-data/`, not here.

## Ground rules for Orbital

- **Read freely.** You may read any file in this repo to understand the codebase.
- **Write directly only here.** `BACKLOG.md` is yours to update without asking.
- **Production code = propose, don't push.** Any change under `agent_os/`, `web/`,
  `scripts/`, `installer/`, etc. must be surfaced as a diff for human approval first.
  Never auto-commit.
- These are *conventions*, not a sandbox. The enforced control is the project's
  **`supervised`** autonomy preset (approve every tool call) + git diff review.

## Now (in progress)

| # | Feature | Notes |
|---|---------|-------|
| 9 | **Sub-agent fanout — parallel native worker sessions** | 2026-07-04 · **In implementation** on `feature/backlog-9-subagent-fanout`. All scope decisions + §7 open questions resolved (spec §0/§0.5): native worker sessions, `fanout` batch tool, wait-for-all join with activity watchdog (10 min stall / 60 min cap), utility-model workers, fanout progress card + nested drill-in UX (workers read-only, claude/codex resumable). Spec at [`BACKLOG/specs/009-…md`](BACKLOG/specs/009-subagent-fanout-parallel-workers.md). |

## Next (queued)

| # | Feature | Notes |
|---|---------|-------|
|   |         |       |

## Later (someday / ideas)

| # | Feature | Notes |
|---|---------|-------|
| 1 | **Cloud hosting for project files** | 2026-06-26 · user-flagged "very important". Implementation spec at [`BACKLOG/specs/001-…md`](BACKLOG/specs/001-cloud-hosting-for-project-files.md); see Spec 1 note below. |
| 7 | **Document generation support (docx / pptx / xlsx)** | 2026-07-03 · Three implementation tiers (skill-only / native tools + preview / in-app render) with code paths cited, WorkBuddy posture cross-validated. Implementation spec at [`BACKLOG/specs/007-…md`](BACKLOG/specs/007-document-generation-support.md); see Spec 7 note below. |
| 8 | **Language toggle during onboarding** | 2026-07-03 · Frontend-only. The i18n runtime is already wired — `LocaleProvider` wraps `<App>` at `web/src/main.tsx:14`, all wizard strings already have en+zh translations (`strings.ts:689-716`), the existing dropdown primitive is at `GlobalSettings.tsx:79-93`. The gap is the in-wizard UI affordance: a SetupWizard currently always boots to English (`LocaleContext.tsx:18-26` falls back to `DEFAULT_LOCALE = 'en'`). Tiered approach: A1 = new wizard step / **A2 = header widget (recommended)** / B1 = `navigator.language` default / B2 = daemon-side persistence (skip for v1). A2 alone is XS (~50 LOC); A2+B1 is XS (~55 LOC). Implementation spec at [`BACKLOG/specs/008-…md`](BACKLOG/specs/008-language-toggle-during-onboarding.md); see Spec 8 note below. |
| 10 | **Group projects — shared project state + merge engine** | 2026-07-03 · Strategy note. Group projects need a **merge/reconciliation engine** (not worktrees per se — members sync via the Spec 001 backend). Merge-engine decisions resolved (three-way + recency tiebreaker, per-change latest-edit ordering, archive losers, per-file-class policy, git-optional); first consumer is fanout phase 2. Blocked behind Spec 001's open questions. Spec at [`BACKLOG/specs/010-…md`](BACKLOG/specs/010-group-projects-shared-state.md); see Spec 10 note below. |

## Done

| # | Feature | Shipped |
|---|---------|---------|
| 2 | **Clickable local paths in chat → drawer** | 2026-07-01 · shipped in claude-code session. |
| 3 | **Render HTML files in the file viewer** | 2026-07-01 · shipped in claude-code session. |
| 4 | **Suppress CLAUDE.md conflict banner** | 2026-07-01 · shipped in claude-code session. |
| 5 | **Sub-agent session context continuity across dispatch paths** | 2026-07-01 · shipped in claude-code session. |

## Specs

> Detailed explorations for non-trivial backlog entries. Each spec has a status tag
> and a pointer back to its Kanban row. Status → table mapping: `idea` → Later ·
> `exploring` / `shaped` → Next · `ready` → Now · shipped → Done.

### Spec 1 — Cloud hosting for project files
**Status:** idea · **Added:** 2026-06-26

User said: *"one very important features. cloud hosting for project files."* — user-flagged "very important".

The detailed implementation spec (code paths cited, `ProjectStorageBackend` interface sketch, sync semantics per file category, sandbox + credential + offline analysis, dependencies, effort estimate, and 6 open questions for the user) lives at:
- **[`BACKLOG/specs/001-cloud-hosting-for-project-files.md`](BACKLOG/specs/001-cloud-hosting-for-project-files.md)**

→ Awaiting user answers on the 6 open questions in the spec before promoting from `idea`.

### Spec 2 — Clickable local paths in chat → drawer
**Status:** shipped · **Added:** 2026-06-26 · **Decisions resolved:** 2026-07-01 · **Shipped:** 2026-07-01

User said: *"i want the link presented by agent to be clickable - the local path. i want to invoke a side bar to visualize them"*

**Resolved decisions (spec §0):**
- Side panel → **right-hand drawer** (desktop) / **bottom-sheet** (mobile). The `files`-tab-switch alternative is not built.
- **No new agent tool** — the `MarkdownContent` link renderer is made kind-aware: workspace-relative links to previewable artifacts (`.html`, images, `.csv`, `.md`, `.json`) render as **cards** that open the drawer; source-file paths render as inline **chips**. A `present_artifact` tool is deferred until we outgrow the convention.
- Linkify **high-signal surfaces only** (markdown links, inline-code, tool-call activity rows, attachment chips) — **no plain prose** in v1.
- Read-only preview, replace-on-click, toast on missing file.
- Effort **M (~2-3 days)**, frontend-only.

Spec: **[`BACKLOG/specs/002-clickable-local-paths-in-chat.md`](BACKLOG/specs/002-clickable-local-paths-in-chat.md)**

→ Shipped 2026-07-01 in claude-code session.

### Spec 3 — Render HTML files in the file viewer
**Status:** shipped · **Added:** 2026-06-26 · **Shipped:** 2026-07-01

User said: *"i want to be able to see html in the files."*

The detailed implementation spec (file-preview code paths, current `.html` returns raw text from the file API, rendering mode options — sandboxed iframe vs raw vs code-view, asset resolution, security model, 8 open questions for the user, effort estimate) lives at:
- **[`BACKLOG/specs/003-html-rendering-in-files.md`](BACKLOG/specs/003-html-rendering-in-files.md)**

→ Shipped 2026-07-01 in claude-code session.

### Spec 4 — Suppress CLAUDE.md conflict banner
**Status:** shipped · **Added:** 2026-06-27 · **Shipped:** 2026-07-01

User said: *"there is a sign above the chat interface ... that said ... investigate into it and propose solutions on how to suppress it"*. Bug investigation only — no code change.

The detailed investigation (root cause, exact files + line numbers — `agent_os/agent/sub_agent_prompt.py:214-260` scanner, `agent_os/agent/sub_agent_manager.py:524-583` WS broadcast, `web/src/.../ClaudemdWarningBanner.tsx` rendered from `ChatView.tsx:2032`; quoted "bypass" matches at `CLAUDE.md:329` and `CLAUDE.md:492`; data flow from disk → WS → component; 5 suppression options with effort/risk/trade-offs; recommended approach) lives at:
- **[`BACKLOG/specs/004-suppress-claudemd-banner.md`](BACKLOG/specs/004-suppress-claudemd-banner.md)**

→ Shipped 2026-07-01 in claude-code session.

### Spec 5 — Sub-agent session context continuity across dispatch paths
**Status:** shipped · **Added:** 2026-06-29 · **Decisions resolved:** 2026-07-01 · **Shipped:** 2026-07-01

User said (translated from Chinese): *"In Orbital, if you first dispatch a sub-agent through the management agent, and then later call the sub-agent via `@`-syntax, the sub-agent's session is not the same — the `@`-initiated session does NOT carry forward the context from the earlier management-agent-dispatched session."*

**Resolved decisions (spec §0):**
- **Single-session model:** a sub-agent thread is one continuous thread per `(project, chat_session, handle)`; both dispatch paths into the same chat session share it. The chat session is the boundary (lifetime coupled to it).
- **Cross-session re-attach → no** (a new chat session starts sub-agents fresh). This **drops §4c and the lock re-keying** — the heaviest part of the original plan.
- **Reset via a thin flag:** `agent_message(send, …, fresh=True)` lets the management agent deliberately start a fresh sub-agent thread (unrelated task, or reset an overlong thread). Not a new tool.
- Known gap: provider-level resume is SDK/codex-only; pipe transport keeps a continuous transcript but a cold LLM thread (deferred).
- Chosen build scope **4a + 4b + 4d**, ≈ **2 days** backend, no frontend required.

Spec: **[`BACKLOG/specs/005-subagent-session-context-continuity.md`](BACKLOG/specs/005-subagent-session-context-continuity.md)**

→ Shipped 2026-07-01 in claude-code session.

### Spec 7 — Document generation support (docx / pptx / xlsx)
**Status:** idea · **Added:** 2026-07-03

User said (translated from Chinese): *"调研一下，orbital应该怎么支持做文档，比如说docx，ppt，excel的能力"*. User added: *"顺便看一下我电脑里的workbuddy是怎么做的"*.

The detailed implementation spec (tool-layer text-only analysis with `read.py:88-89` / `write.py:49` / `edit.py:51` cited; files API classifier chain at `files_v2.py:49-192`; agent manifest survey; frontend preview pipeline at `FilePreview.tsx` + `web/src/types.ts:635`; pyproject dependency posture; three implementation tiers from XS (~50-100 LOC, skill-only) to M (~400-700 LOC, native read/write + preview) to L/XL (~1500-3000 LOC, in-app render via mammoth/SheetJS); effort estimates calibrated against shipped specs 003/005; **WorkBuddy reference** validating Tier A strongly, half-validating Tier B, fully validating Tier C's `docx-preview` + SheetJS + custom pptx-preview stack; 8 original open questions + 2 surfaced by WorkBuddy) lives at:
- **[`BACKLOG/specs/007-document-generation-support.md`](BACKLOG/specs/007-document-generation-support.md)**

→ Awaiting user answers on the open questions (format priority, library choice, read-vs-write split, preview depth, installer-weight tolerance) plus the WorkBuddy-surfaced questions (skill-vs-tool generation, client-side-only render) before promoting from `idea`.

### Spec 8 — Language toggle during onboarding
**Status:** idea · **Added:** 2026-07-03

User said: *"we need the language toggle during onboarding."*

The detailed implementation spec (cold-start problem: English-non-speaker's first screen is English; current-state trace through `LocaleContext.tsx:18-26` → `locales.ts:5-18` → `LocaleContext.tsx:28-49` → `main.tsx:12-20` → `SetupWizard.tsx:254-361`; existing `GlobalSettings.tsx:79-93` dropdown primitive ready to drop in; `navigator.language` is currently unread; runtime is already client-side only via `localStorage['orbital.locale']`; tiered approach A1 / A2 / B1 / B2 / B3 with **A2 + B1 recommended**, A1 = S, A2 = XS, B2 = S-M on top; no new libraries, no backend changes, no `react-intl`/`i18next`/FormatJS; 8 open questions covering layout, default-locale detection, daemon persistence, language list expansion, and partial-translation fallback) lives at:
- **[`BACKLOG/specs/008-language-toggle-during-onboarding.md`](BACKLOG/specs/008-language-toggle-during-onboarding.md)**

→ Awaiting user answers on the 8 open questions in the spec (layout choice A1 vs A2, position of widget, `navigator.language` fallback yes/no, daemon persistence yes/no, language list expansion, partial-translation strategy, discovery-vs-clutter tradeoff, first-run-detection-coupling to API-key) before promoting from `idea`.

### Spec 9 — Sub-agent fanout: parallel native worker sessions
**Status:** ready (in implementation) · **Added:** 2026-07-03 · **All decisions resolved:** 2026-07-04

User said: *"implement sub agent fanout - meaning that dispatching multiple parallel sessions for sub tasks … this is not about sub agents such as claude code and codex."*

**Resolved decisions (spec §0):**
- **Workers = native Orbital sessions** — in-process `AgentLoop` per sub-task with the project's provider + restricted tools, wrapped in the existing sub-agent adapter interface (plumbing worker-agnostic; CLI handles inherit parallelism later).
- **Initiation = management-agent tool** — one thin `fanout(tasks=[…])` batch-dispatch tool; one tool call = one `yield_turn`, so the loop's sibling-cancellation behavior needs no change.
- **Join = wait-for-all** — a `JoinGroup` in `SubAgentManager` gathers all worker completions (with timeout + partial-failure reporting) and injects **one** summary via the existing `inject_system_message` funnel.
- **Isolation (resolved 2026-07-03 in the sequencing/strategy discussion):** shared workspace + per-task file scopes with in-process write-guards in v1; worktree + **separable merge engine** as phase 2 (spec §5a addendum; merge-engine decisions recorded in Spec 10). Endorsed sequencing: fanout v1 → worktree manager + merge engine → multi-session execution.

The detailed spec (three blockers with citations — `yield_turn` single-dispatch at `loop.py:1121-1125`, first-completion-owns-restart at `lifecycle_observer.py:59-92`, one-thread-per-handle keying at `sub_agent_manager.py:77-82`; what already works — non-blocking dispatch, 5-adapter concurrency, defer-safe push-back, multi-aware dispatcher slot-hold; work packages W1–W5; Claude Code worktree-convention trade-off analysis; 8 open questions; L effort ≈ 1250–1900 LOC / 1.5–2.5 weeks) lives at:
- **[`BACKLOG/specs/009-subagent-fanout-parallel-workers.md`](BACKLOG/specs/009-subagent-fanout-parallel-workers.md)**

→ Implementation started 2026-07-04 on `feature/backlog-9-subagent-fanout` (off `feature/backlog-2-3-4-5` @ `78c05bd`). All §7 questions resolved with the user — see spec §0.5 (toolset incl. web search/fetch; cap 5; utility-model workers; activity-watchdog timeout; nested drill-in UX; approve-the-batch; fanout progress card, workers not in chips row).

### Spec 10 — Group projects: shared project state + merge engine
**Status:** idea (strategy note) · **Added:** 2026-07-03

User said: *"we will want to support group projects in future, where all members should be able to use the same project state files and contribute back to the project."*

**Framing agreed 2026-07-03:** group projects don't require worktrees per se — members sync across machines via the **Spec 001 backend**; the strategic shared asset is a **merge/reconciliation engine** (pluggable policy) whose consumers, in order, are: fanout worker dispatch (Spec 9 phase 2) → CLI sub-agent dispatch → worktree-per-session → group-project sync reconciliation.

**Merge-engine decisions resolved:** three-way merge with last-writer-wins as the conflict *tiebreaker* only (never whole-file LWW); ordering by **per-change latest-edit time** (resolves the user's session-start-vs-latest-edit question — order changes, not sessions; git-native form = auto-commit per turn, order by commit time, buying attribution + revert); losing diffs archived + surfaced, never deleted; policy per file class (metadata may LWW/regenerate, user work product gets three-way + archive); git-optional engine (copy-dir fallback is first-class — Orbital projects are often not git repos). **Rejected:** merge-when-all-sessions-idle (idle ≠ done; long-lived session blocks all merges; breaks the shared-blackboard model).

Spec: **[`BACKLOG/specs/010-group-projects-shared-state.md`](BACKLOG/specs/010-group-projects-shared-state.md)**

→ Parked as `idea`: merge-boundary question open (turn-end auto-commit vs session close vs explicit action); transport/membership/identity blocked behind Spec 001's 6 open questions.
