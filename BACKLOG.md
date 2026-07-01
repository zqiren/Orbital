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
|   |         |       |

## Next (queued)

| # | Feature | Notes |
|---|---------|-------|
| 2 | **Clickable local paths in chat → drawer** | 2026-06-26 · **shaped 2026-07-01**. Agent's chat-output paths become clickable; click opens a right-hand drawer (mobile bottom-sheet) to preview the file. Artifacts (HTML/reports) render as cards via markdown-link convention — no new tool. Spec at [`BACKLOG/specs/002-…md`](BACKLOG/specs/002-clickable-local-paths-in-chat.md); see Spec 2 note below. |
| 5 | **Sub-agent session context continuity across dispatch paths** | 2026-06-29 · **shaped 2026-07-01**. Single-session model: dispatch via `agent_message` and `@<handle>` share one thread per `(project, chat_session, handle)`. Scope = 4a+4b+4d (`fresh=True` reset flag); cross-session re-attach dropped. Spec at [`BACKLOG/specs/005-…md`](BACKLOG/specs/005-subagent-session-context-continuity.md); see Spec 5 note below. |

## Later (someday / ideas)

| # | Feature | Notes |
|---|---------|-------|
| 1 | **Cloud hosting for project files** | 2026-06-26 · user-flagged "very important". Implementation spec at [`BACKLOG/specs/001-…md`](BACKLOG/specs/001-cloud-hosting-for-project-files.md); see Spec 1 note below. |
| 3 | **Render HTML files in the file viewer** | 2026-06-26 · show HTML files rendered (not raw text). Implementation spec at [`BACKLOG/specs/003-…md`](BACKLOG/specs/003-html-rendering-in-files.md); see Spec 3 note below. |
| 4 | **Suppress CLAUDE.md conflict banner** | 2026-06-27 · bug: a banner ("may conflict with Orbital's project-state inheritance. Sub-agents may behave unexpectedly. · matched: \"bypass\"") fires above chat because the daemon does a naked substring scan of workspace CLAUDE.md for conflict keywords and this workspace's own CLAUDE.md mentions "bypass" in two unrelated macOS-Gatekeeper release-runbook steps. Implementation spec at [`BACKLOG/specs/004-…md`](BACKLOG/specs/004-suppress-claudemd-banner.md); see Spec 4 note below. |

## Done

| # | Feature | Shipped |
|---|---------|---------|
|   |         |         |

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
**Status:** shaped · **Added:** 2026-06-26 · **Decisions resolved:** 2026-07-01

User said: *"i want the link presented by agent to be clickable - the local path. i want to invoke a side bar to visualize them"*

**Resolved decisions (spec §0):**
- Side panel → **right-hand drawer** (desktop) / **bottom-sheet** (mobile). The `files`-tab-switch alternative is not built.
- **No new agent tool** — the `MarkdownContent` link renderer is made kind-aware: workspace-relative links to previewable artifacts (`.html`, images, `.csv`, `.md`, `.json`) render as **cards** that open the drawer; source-file paths render as inline **chips**. A `present_artifact` tool is deferred until we outgrow the convention.
- Linkify **high-signal surfaces only** (markdown links, inline-code, tool-call activity rows, attachment chips) — **no plain prose** in v1.
- Read-only preview, replace-on-click, toast on missing file.
- Effort **M (~2-3 days)**, frontend-only.

Spec: **[`BACKLOG/specs/002-clickable-local-paths-in-chat.md`](BACKLOG/specs/002-clickable-local-paths-in-chat.md)**

→ Ready to build (Next). One non-blocking open question remains (persist `previewPath` across reload — deferred).

### Spec 3 — Render HTML files in the file viewer
**Status:** idea · **Added:** 2026-06-26

User said: *"i want to be able to see html in the files."*

The detailed implementation spec (file-preview code paths, current `.html` returns raw text from the file API, rendering mode options — sandboxed iframe vs raw vs code-view, asset resolution, security model, 8 open questions for the user, effort estimate) lives at:
- **[`BACKLOG/specs/003-html-rendering-in-files.md`](BACKLOG/specs/003-html-rendering-in-files.md)**

→ Awaiting user answers on the 8 open questions in the spec before promoting from `idea`.

### Spec 4 — Suppress CLAUDE.md conflict banner
**Status:** idea · **Added:** 2026-06-27

User said: *"there is a sign above the chat interface ... that said ... investigate into it and propose solutions on how to suppress it"*. Bug investigation only — no code change.

The detailed investigation (root cause, exact files + line numbers — `agent_os/agent/sub_agent_prompt.py:214-260` scanner, `agent_os/agent/sub_agent_manager.py:524-583` WS broadcast, `web/src/.../ClaudemdWarningBanner.tsx` rendered from `ChatView.tsx:2032`; quoted "bypass" matches at `CLAUDE.md:329` and `CLAUDE.md:492`; data flow from disk → WS → component; 5 suppression options with effort/risk/trade-offs; recommended approach) lives at:
- **[`BACKLOG/specs/004-suppress-claudemd-banner.md`](BACKLOG/specs/004-suppress-claudemd-banner.md)**

→ Awaiting user to pick a suppression option (or combination) from the spec before promoting from `idea`.

### Spec 5 — Sub-agent session context continuity across dispatch paths
**Status:** shaped · **Added:** 2026-06-29 · **Decisions resolved:** 2026-07-01

User said (translated from Chinese): *"In Orbital, if you first dispatch a sub-agent through the management agent, and then later call the sub-agent via `@`-syntax, the sub-agent's session is not the same — the `@`-initiated session does NOT carry forward the context from the earlier management-agent-dispatched session."*

**Resolved decisions (spec §0):**
- **Single-session model:** a sub-agent thread is one continuous thread per `(project, chat_session, handle)`; both dispatch paths into the same chat session share it. The chat session is the boundary (lifetime coupled to it).
- **Cross-session re-attach → no** (a new chat session starts sub-agents fresh). This **drops §4c and the lock re-keying** — the heaviest part of the original plan.
- **Reset via a thin flag:** `agent_message(send, …, fresh=True)` lets the management agent deliberately start a fresh sub-agent thread (unrelated task, or reset an overlong thread). Not a new tool.
- Known gap: provider-level resume is SDK/codex-only; pipe transport keeps a continuous transcript but a cold LLM thread (deferred).
- Chosen build scope **4a + 4b + 4d**, ≈ **2 days** backend, no frontend required.

Spec: **[`BACKLOG/specs/005-subagent-session-context-continuity.md`](BACKLOG/specs/005-subagent-session-context-continuity.md)**

→ Ready to build (Next). No blocking open questions remain.
