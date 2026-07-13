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
| 8 | **Language toggle during onboarding** | 2026-07-13 · decisions resolved (all defaults): A2 header widget + B1 `navigator.language` fallback, no daemon persistence. XS ~½ day, frontend-only. **Hard prerequisite for 17.** See Spec 8. |
| 17 | **First-launch onboarding guide — Tier 1** | 2026-07-13 · decisions resolved (3rd pass): upgraded single provider screen — fixed preset order for everyone (no locale logic), **default = DeepSeek, never Custom**, **region toggle defaults to China** (CN-first launch audience); per-provider "Get your API key" console links; static "no CN endpoint" captions replace the locale-triggered hint band; official CN endpoints only. S ~2-3 days, frontend + `providers.json`. Depends on Spec 8. See Spec 17. |
| 15 | **`AGENTS.md` at workspace root** | 2026-07-13 · decisions resolved (all defaults + content pivot): **memory-system manual for external agents** — positioning / when-to-read / when-to-update per memory file. Full read/write on all memory files (agents are collaborators, not hostile — user 2026-07-13); hands off only machine-managed runtime dirs (sessions/ledger/tool-results/transcripts). XS ~½ day, docs-only. See Spec 15. |
| 7 | **Document generation — Tier A** | 2026-07-13 · decisions resolved: Tier A skill-only — bundle python-docx/openpyxl/python-pptx (~5 MB `[office]` extra) + doc-gen skill + prompt hint; no new tools, no UI, no pandoc/LibreOffice. XS ~½ day. Tier B only on evidence. See Spec 7. |

## Next (queued)

| # | Feature | Notes |
|---|---------|-------|
| 16 | **Cursor + Grok as sub-agents via an official-ACP-SDK transport** | 2026-07-13 · Investigation complete. No Python "cursor-sdk"/"grok-sdk" exists, but both vendors ship **first-party ACP server modes** (Cursor: `agent acp`, [cursor.com/docs/cli/acp](https://cursor.com/docs/cli/acp), with `session/load` resume + `session/request_permission` + agent/plan/ask modes; Grok Build: `grok agent stdio` — ACP is xAI's declared integration standard). Recommended design: **one new `acp_sdk_transport.py` on the official `agent-client-protocol` PyPI SDK** (v0.11.0), replacing the stale homegrown `acp_transport.py` — which is **dead code** (233 lines, no manifest routes to it) — plus `cursor.yaml` + `grok.yaml` manifests; same seam later absorbs gemini-cli. Effort ≈ **3–6 dev-days** (~300–500 LOC transport; hardest part = mapping `session/request_permission` onto Orbital approvals, replacing the old auto-approve). Dependency vetted and acceptable: Apache-2.0, sole dep pydantic (already shipped), pure-Python 66KB → PyInstaller-trivial, official org, active (pushed 2026-07-10); pre-1.0 → pin exact version à la `SUPPORTED_CODEX_VERSION`. Cursor's TS-only `@cursor/sdk` rejected for v1 (Node sidecar). Neither CLI installed locally yet; Grok needs a paid plan. Spec at [`BACKLOG/specs/016-…md`](BACKLOG/specs/016-cursor-grok-acp-sdk-transport.md). |
| 14 | **Use open-connector as an MCP server source for connectors** | 2026-07-06 · Can Orbital consume `oomol-lab/open-connector` (840+ providers, 8,300+ Actions, exposes `/mcp`) as a Tier-0 MCP source under Spec 11's connector substrate? Sub-agent verdict: **consumable today** via the existing Custom MCP server form at `ConnectorSettings.tsx` + `POST /api/v2/connectors/custom` (`agent_os/api/routes/connectors.py:64-72`) — zero Orbital code changes. But `/mcp` exposes only 4 meta-tools (`list_apps`, `search_actions`, `get_action_guide`, `execute_action`), not direct action passthrough, so deeper integration needs an agent-loop extension. Integration shapes (a–d) compared: standalone sidecar / catalog import at build / user-installed Tier-0 / hybrid. Awaiting user answers on deployment model (local-first posture), OAuth app-registration responsibility, curated-vs-full-catalog posture. **2026-07-07 research addendum** (spec §"2026-07-07"): broker landscape + OAuth-app cost + connector roadmap. Verdicts — **Nango is not the fit** (BYO OAuth app per provider; moves tokens off keychain); **Composio** is the only managed broker that owns verified OAuth apps (removes per-service registration) but stores tokens on its SaaS cloud (breaks local-first); **first-party is best for the free/sensitive tier** since OAuth registration is ~free (real money = Google restricted-scope CASA $500–4.5k/yr + WeChat 企业认证 ¥300/$99; real barrier = a Chinese entity for CN platforms). **China workspace suite (Feishu/DingTalk/WeChat) is first-party regardless** — even China-origin open-connector ships only notification bots for them. Connector tiers + launch order recorded. Spec at [`BACKLOG/specs/014-…md`](BACKLOG/specs/014-open-connector-as-mcp-source.md). |

## Later (someday / ideas)

| # | Feature | Notes |
|---|---------|-------|
| 1 | **Cloud hosting for project files** | 2026-06-26 · user-flagged "very important". Implementation spec at [`BACKLOG/specs/001-…md`](BACKLOG/specs/001-cloud-hosting-for-project-files.md); see Spec 1 note below. |
| 10 | **Group projects — shared project state + merge engine** | 2026-07-03 · Strategy note. Group projects need a **merge/reconciliation engine** (not worktrees per se — members sync via the Spec 001 backend). Merge-engine decisions resolved (three-way + recency tiebreaker, per-change latest-edit ordering, archive losers, per-file-class policy, git-optional); first consumer is fanout phase 2. Blocked behind Spec 001's open questions. Spec at [`BACKLOG/specs/010-…md`](BACKLOG/specs/010-group-projects-shared-state.md); see Spec 10 note below. |

## Done

| # | Feature | Shipped |
|---|---------|---------|
| 2 | **Clickable local paths in chat → drawer** | 2026-07-01 · shipped in claude-code session. |
| 3 | **Render HTML files in the file viewer** | 2026-07-01 · shipped in claude-code session. |
| 4 | **Suppress CLAUDE.md conflict banner** | 2026-07-01 · shipped in claude-code session. |
| 5 | **Sub-agent session context continuity across dispatch paths** | 2026-07-01 · shipped in claude-code session. |
| 9 | **Sub-agent fanout: parallel native worker sessions** | shipped — confirmed by user 2026-07-13. |
| 12 | **Cross-project scope for Quick Tasks (the global lens)** | shipped — confirmed by user 2026-07-13. |

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
**Status:** ready (Tier A) · **Added:** 2026-07-03 · **Decisions resolved:** 2026-07-13

User said (translated from Chinese): *"调研一下，orbital应该怎么支持做文档，比如说docx，ppt，excel的能力"*. User added: *"顺便看一下我电脑里的workbuddy是怎么做的"*.

The detailed implementation spec (tool-layer text-only analysis with `read.py:88-89` / `write.py:49` / `edit.py:51` cited; files API classifier chain at `files_v2.py:49-192`; agent manifest survey; frontend preview pipeline at `FilePreview.tsx` + `web/src/types.ts:635`; pyproject dependency posture; three implementation tiers from XS (~50-100 LOC, skill-only) to M (~400-700 LOC, native read/write + preview) to L/XL (~1500-3000 LOC, in-app render via mammoth/SheetJS); effort estimates calibrated against shipped specs 003/005; **WorkBuddy reference** validating Tier A strongly, half-validating Tier B, fully validating Tier C's `docx-preview` + SheetJS + custom pptx-preview stack; 8 original open questions + 2 surfaced by WorkBuddy) lives at:
- **[`BACKLOG/specs/007-document-generation-support.md`](BACKLOG/specs/007-document-generation-support.md)**

→ Decisions resolved 2026-07-13 (all defaults → **Tier A**): bundle the three pure-Python libs as an `[office]` extra + doc-gen skill + prompt hint; no new tools, no UI, no pandoc/LibreOffice; read via the same libs through shell. Tier B/C only if real usage shows Tier A falling short. See spec "Resolved decisions". **Follow-up (2026-07-13, from implementation):** packaged installers don't yet carry the `office` extra — the sandboxed shell runs the *system* python, so bundling into the frozen app wouldn't serve it anyway; the prompt hint degrades gracefully (on-demand `pip install`, pypi.org is network-allowlisted). Deciding whether/how installers should pre-provision these libs is an open design question for a future pass.

### Spec 8 — Language toggle during onboarding
**Status:** ready · **Added:** 2026-07-03 · **Decisions resolved:** 2026-07-13

User said: *"we need the language toggle during onboarding."*

The detailed implementation spec (cold-start problem: English-non-speaker's first screen is English; current-state trace through `LocaleContext.tsx:18-26` → `locales.ts:5-18` → `LocaleContext.tsx:28-49` → `main.tsx:12-20` → `SetupWizard.tsx:254-361`; existing `GlobalSettings.tsx:79-93` dropdown primitive ready to drop in; `navigator.language` is currently unread; runtime is already client-side only via `localStorage['orbital.locale']`; tiered approach A1 / A2 / B1 / B2 / B3 with **A2 + B1 recommended**, A1 = S, A2 = XS, B2 = S-M on top; no new libraries, no backend changes, no `react-intl`/`i18next`/FormatJS; 8 open questions covering layout, default-locale detection, daemon persistence, language list expansion, and partial-translation fallback) lives at:
- **[`BACKLOG/specs/008-language-toggle-during-onboarding.md`](BACKLOG/specs/008-language-toggle-during-onboarding.md)**

→ Decisions resolved 2026-07-13 (all defaults): **A2 header widget** (top-right, Globe icon + select, every step) + **B1 `navigator.language` fallback**; no daemon persistence (B2 skipped); languages stay en+zh. Hard prerequisite for Spec 17 — ship first or same branch. See spec §8.

### Spec 9 — Sub-agent fanout: parallel native worker sessions
**Status:** shipped · **Added:** 2026-07-03 · **All decisions resolved:** 2026-07-04 · **Shipped:** confirmed by user 2026-07-13

User said: *"implement sub agent fanout - meaning that dispatching multiple parallel sessions for sub tasks … this is not about sub agents such as claude code and codex."*

**Resolved decisions (spec §0):**
- **Workers = native Orbital sessions** — in-process `AgentLoop` per sub-task with the project's provider + restricted tools, wrapped in the existing sub-agent adapter interface (plumbing worker-agnostic; CLI handles inherit parallelism later).
- **Initiation = management-agent tool** — one thin `fanout(tasks=[…])` batch-dispatch tool; one tool call = one `yield_turn`, so the loop's sibling-cancellation behavior needs no change.
- **Join = wait-for-all** — a `JoinGroup` in `SubAgentManager` gathers all worker completions (with timeout + partial-failure reporting) and injects **one** summary via the existing `inject_system_message` funnel.
- **Isolation (resolved 2026-07-03 in the sequencing/strategy discussion):** shared workspace + per-task file scopes with in-process write-guards in v1; worktree + **separable merge engine** as phase 2 (spec §5a addendum; merge-engine decisions recorded in Spec 10). Endorsed sequencing: fanout v1 → worktree manager + merge engine → multi-session execution.

The detailed spec (three blockers with citations — `yield_turn` single-dispatch at `loop.py:1121-1125`, first-completion-owns-restart at `lifecycle_observer.py:59-92`, one-thread-per-handle keying at `sub_agent_manager.py:77-82`; what already works — non-blocking dispatch, 5-adapter concurrency, defer-safe push-back, multi-aware dispatcher slot-hold; work packages W1–W5; Claude Code worktree-convention trade-off analysis; 8 open questions; L effort ≈ 1250–1900 LOC / 1.5–2.5 weeks) lives at:
- **[`BACKLOG/specs/009-subagent-fanout-parallel-workers.md`](BACKLOG/specs/009-subagent-fanout-parallel-workers.md)**

→ Implementation started 2026-07-04 on `feature/backlog-9-subagent-fanout` (off `feature/backlog-2-3-4-5` @ `78c05bd`). All §7 questions resolved with the user — see spec §0.5 (toolset incl. web search/fetch; cap 5; utility-model workers; activity-watchdog timeout; nested drill-in UX; approve-the-batch; fanout progress card, workers not in chips row). **Shipped — confirmed by user 2026-07-13.**

### Spec 10 — Group projects: shared project state + merge engine
**Status:** idea (strategy note) · **Added:** 2026-07-03

User said: *"we will want to support group projects in future, where all members should be able to use the same project state files and contribute back to the project."*

**Framing agreed 2026-07-03:** group projects don't require worktrees per se — members sync across machines via the **Spec 001 backend**; the strategic shared asset is a **merge/reconciliation engine** (pluggable policy) whose consumers, in order, are: fanout worker dispatch (Spec 9 phase 2) → CLI sub-agent dispatch → worktree-per-session → group-project sync reconciliation.

**Merge-engine decisions resolved:** three-way merge with last-writer-wins as the conflict *tiebreaker* only (never whole-file LWW); ordering by **per-change latest-edit time** (resolves the user's session-start-vs-latest-edit question — order changes, not sessions; git-native form = auto-commit per turn, order by commit time, buying attribution + revert); losing diffs archived + surfaced, never deleted; policy per file class (metadata may LWW/regenerate, user work product gets three-way + archive); git-optional engine (copy-dir fallback is first-class — Orbital projects are often not git repos). **Rejected:** merge-when-all-sessions-idle (idle ≠ done; long-lived session blocks all merges; breaks the shared-blackboard model).

Spec: **[`BACKLOG/specs/010-group-projects-shared-state.md`](BACKLOG/specs/010-group-projects-shared-state.md)**

→ Parked as `idea`: merge-boundary question open (turn-end auto-commit vs session close vs explicit action); transport/membership/identity blocked behind Spec 001's 6 open questions.

### Spec 11 — Connectors + the workspace layer
**Status:** shaped · **Added:** 2026-07-04 · **Design aligned:** 2026-07-04

User said: *"we need to enable connector for services such as gmail and calendar, and there are many of them, we need to have one place to host it and a way to scale the addition of these services."* Follow-ups: calendar deserves a UI (*"a place where human and agent can collaborate"*), connectors at onboarding, sandbox wizard step removed, settings need an index.

**Resolved decisions (spec §0):**
- **Substrate = MCP client in the daemon** — connectors are catalog *manifests*, not per-service code; tools reflect namespaced into the existing `ToolRegistry`; adding a Tier-1 connector ≈ a ~20-line manifest + icon + smoke test (friction model in spec §3, incl. Tier-0 user-added custom MCP servers and the Google OAuth verification caveat).
- **Auth global, enable per-project** (project = trust boundary; tokens in keychain `UserCredentialStore`); writes gated by autonomy, unknown tools fail closed.
- **Two-tier surface rule:** email = tools only; **calendar = first surface**, one component mounted as global calendar + per-project lens (external calendar stays source of truth; project linkage = Orbital metadata).
- **IA: two-zone sidebar** (Workspace zone: Calendar, promoted Quick Tasks · Projects zone unchanged); connectors get *no* sidebar surface — Global Settings section + Project Settings toggles.
- **Onboarding:** wizard → `api_key → connect your accounts` (featured-but-skippable, merged with browser sign-in); sandbox step deleted (macOS vestigial, **Windows setup moves to the installer** via the existing `--setup-sandbox` elevated path).
- **Settings index:** generalize the one-off `settingsAnchor: 'budget'` into a named-section scrollspy rail on both settings surfaces (rejected paged settings — preserves single-form-save).

Spec: **[`BACKLOG/specs/011-connectors-and-workspace-layer.md`](BACKLOG/specs/011-connectors-and-workspace-layer.md)**

→ `shaped`, phased A (connector core, L) / B (calendar surface, M-L) / C (onboarding + installer + settings index, M). Awaiting §7 answers — biggest: Google OAuth app strategy (Q1, likely decides launch set), calendar sync mechanism (Q2), remote-only vs local MCP servers (Q4).

### Spec 12 — Cross-project scope for Quick Tasks (the global lens)
**Status:** shipped · **Added:** 2026-07-04 · **Design aligned:** 2026-07-04 · **Shipped:** confirmed by user 2026-07-13

User said: *"sometimes he needs an overview of all projects or maybe even full access of the computer … projects are assets, but there are different ways to look at assets — work in progress or historical reference."* (Corporate-trainer scenario: search all client projects for prior deck style/content.)

**Resolved decisions (spec §0):**
- **Principle: every asset has two lenses** — WIP (project agent, read-write) vs reference (cross-project, read-only). Quick Tasks is the *agent* of the global lens.
- **Quick Tasks gets opt-in all-projects READ-ONLY scope** — reads resolve across all project workspaces, Write/Edit stay in scratch; agreed scope-down from the user's original "full access" phrasing.
- **Full computer access** = separate loudly-consented opt-in, deferred; **dispatch-to-project-agents** deferred to post-fanout (composes with Spec 9 machinery).
- **Prerequisite fix ships regardless:** per-project portal scoping — today `_portal_paths` is a process-global dict on the shared provider, so project A's *shell* can already read+write project B once both started (`agent_manager.py:514` + `macos/provider.py:194-198`). Real isolation leak.
- Other projects' `orbital/` runtime dirs excluded from reference reads (work product only, not agent internals).

Spec: **[`BACKLOG/specs/012-cross-project-scope-quick-tasks.md`](BACKLOG/specs/012-cross-project-scope-quick-tasks.md)**

→ Shipped — confirmed by user 2026-07-13.

### Spec 14 — Use open-connector as an MCP server source for connectors
**Status:** exploring · **Added:** 2026-07-06

User said: *"do you think we can use this for connector in orbital? https://github.com/oomol-lab/open-connector"* — pointing at `oomol-lab/open-connector`, a self-hostable MCP server (840+ providers, 8,300+ prebuilt Actions, `/mcp` endpoint).

**Sub-agent verdict (exploration complete, awaiting user shape decisions):**
- **Yes** as an opt-in Tier-0 source *today* — the existing "Custom MCP server" form at `ConnectorSettings.tsx` plus `POST /api/v2/connectors/custom` (`agent_os/api/routes/connectors.py:64-72`) already handle the integration. Zero Orbital code changes needed for users to point at their own open-connector.
- **Architectural caveat:** `/mcp` exposes only 4 meta-tools (`list_apps`, `search_actions`, `get_action_guide`, `execute_action`) — not direct action passthrough — so any deeper integration than Tier-0 requires an agent-loop extension to call `search_actions` → `get_action_guide` → execute over HTTP.
- **Not a replacement** for Spec 11's curated Google catalog: open-connector just pushes the Google OAuth app-verification problem one layer down (each user / org still needs their own registered Google OAuth client).

Four integration shapes compared in the spec (a) standalone Docker/Node sidecar / (b) catalog import at build time / (c) user-installed Tier-0 (zero-code path already works via Spec 11's existing custom-server surface) / (d) hybrid curated + import + Tier-0 fallback — with decision matrix per user persona (casual / power / enterprise). License (Apache-2.0), catalog freshness model, Cloudflare-deploy posture, and the `oo CLI` mid-July 2026 timing note all captured.

The detailed spec (Orbital-side code paths at `agent_os/connectors/manager.py:232` / `agent_os/connectors/mcp_client.py:19-29` / `agent_os/connectors/manifest.py:116-119`; open-connector catalog format + OAuth handling; four integration shapes with effort estimates; cross-cutting risks — third-party dependency, OAuth app-registration responsibility, curated-vs-full posture, license obligations on embedding; 5+ open questions for the user) lives at:
- **[`BACKLOG/specs/014-open-connector-as-mcp-source.md`](BACKLOG/specs/014-open-connector-as-mcp-source.md)**

→ Awaiting user answers on the open questions (deployment model: self-host on user machine vs Cloudflare sidecar vs OOMOL hosted; OAuth-registration responsibility; curated vs full catalog posture; Tier-0 docs guide vs first-class integration path; Apache-2.0 embedding obligations) before promoting from `exploring`.

### Spec 15 — `AGENTS.md` at workspace root (grep-discoverable onboarding for external agents)
**Status:** ready · **Added:** 2026-07-13 · **Pivot** 2026-07-13 from "sub-agent onboarding" → "external agentic-tool onboarding" · **Decisions resolved:** 2026-07-13
User said: *"got another idea. i think we should have an sub agent instruction.md so that when other agents visit our workspace. they are able to catch up with the work seamlessly"* — clarified: *"not really for sub agents. more like for other agents. for example, for codex app to open our project folder and will be able to pick up the project seamlessly. the key will be that to ensure claude code, codex app is able to identify from grep that this file is the one to read before everything. this should be simple task right?"* — i.e. **external agentic tools** (Claude Code, Codex App, Cursor, Copilot, ...) opening the project folder, with grep-discoverability as the core requirement.
[Spec at `BACKLOG/specs/015-agents-md-at-workspace-root.md`](BACKLOG/specs/015-agents-md-at-workspace-root.md) covers: 5 design axes (file name — `AGENTS.md` community convention vs `INSTRUCTION.md` vs reuse existing `CLAUDE.md`; location — workspace root, tracked in git; multiplicity — one file at root for MVP; maintenance — hand-curated vs auto-synced from `orbital/PROJECT_STATE.md`; interaction with existing `CLAUDE.md` / `ONBOARDING.md` at workspace root), an MVP content skeleton (project summary + current focus + in-progress pointers + conventions + how-to-recover-context + land-mine warnings + write-scope), the relationship to the existing `agent_os/agent/sub_agent_prompt.py` onboarding machinery (complementary, not redundant — that one only runs for Orbital-dispatched sub-agents), effort = **XS** (author the file + commit it, ~½ day), and 5 open questions for the user.
→ Decisions resolved 2026-07-13 (all defaults + content pivot): `AGENTS.md` at root, committed, hand-curated, cross-references `CLAUDE.md`. **Content = memory-system manual for external agents** (per memory file: positioning, when to read, when/how to update). **Full read/write on all memory files** — external agents are collaborators, not hostile (user); hands off only machine-managed runtime state (`orbital/sessions|ledger|tool-results` + transcript JSONLs). See spec "Resolved decisions".

### Spec 16 — Cursor + Grok as sub-agents via an official-ACP-SDK transport
**Status:** shaped · **Added:** 2026-07-13 · **Investigation complete:** 2026-07-13

User said: *"can you help to investigate some new cli tools such as cursor and grok. and is there a way we can integrate them as sub agents like claude code and codex?"* — then corrected the bar: *"pipe and acp transport is actually stale … please check again if there is any official ways to plug them in"*, and asked for the replacement cost + dependency assessment.

**Resolved direction (spec §Verdict):**
- **Both integrable through one seam: ACP.** No Python agent-SDK exists for either, but both ship first-party ACP server modes — Cursor `agent acp` (binary renamed `cursor-agent` → `agent`; `session/new` + `session/load` resume, `session/request_permission`, agent/plan/ask modes, MCP), Grok Build `grok agent stdio` (xAI's declared standard for external tools).
- **Build one `acp_sdk_transport.py` on the official `agent-client-protocol` PyPI SDK** (v0.11.0, official `agentclientprotocol` org, Apache-2.0), replacing homegrown `acp_transport.py` — confirmed **dead code**: no manifest routes to `transport: acp` (all ship sdk / codex-appserver / pty), so no behavior migration; ~660 test lines refactor, dummy-agent fixture reusable.
- **Cost ≈ 3–6 dev-days:** ~300–500-LOC transport (SDK removes the JSON-RPC plumbing that makes `codex_transport.py` 855 lines) + the genuinely new parts: `session/request_permission` → Orbital approvals (replaces old MVP auto-approve), `session/load` resume under the fail-closed `_determine_resume` contract, exact-version pin à la `SUPPORTED_CODEX_VERSION`; then `cursor.yaml` + `grok.yaml` manifests, `_resolve_transport` registration, live verification.
- **Dependency accepted:** sole transitive dep pydantic>=2.7 (already shipped, 2.12.5), pure-Python 66KB wheel (PyInstaller-trivial, no signing implications), Python `>=3.10,<3.15` fits our `>=3.11`, actively maintained (pushed 2026-07-10); pre-1.0 risk mitigated by pinning.
- **Rejected for v1:** Cursor's official `@cursor/sdk` (TypeScript-only → Node sidecar; revisit only for Cursor cloud-VM agents). Cursor's Claude-shaped `-p --output-format stream-json` pipe path recorded as fallback only.

Spec: **[`BACKLOG/specs/016-cursor-grok-acp-sdk-transport.md`](BACKLOG/specs/016-cursor-grok-acp-sdk-transport.md)**

→ `shaped`. Before implementation: install both CLIs (neither on this machine; Grok needs a paid SuperGrok/X Premium+ plan or API key) and run the spec's live-verification checklist — biggest unknowns: Grok `session/load` support, per-agent permission-request semantics, on-disk resume sources for `resume_source_exists`.

---

### Spec 17 — First-launch onboarding guide
**Status:** ready (Tier 1) · **Added:** 2026-07-13 · **Decisions resolved:** 2026-07-13
User said: *"another idea. i think we need an onboarding guide when first launched. i saw a lot of confused user. especially on the china vs global api address and provider selection. we will also need the language toggle during the onboarding."*

**The pain points the user named** (each grounded by code citations in the spec): (F1) free-form API base URL at `LLMProviderSettings.tsx:163-166` with no region hint — users paste the wrong region; (F2) provider picker exposes only first-party IDs (Anthropic, OpenAI, Google, etc.) — no CN providers surfaced, no regional auto-suggest; (F3) cold-start screen is always English (covered by Spec 8, not re-spec'd here). The current SetupWizard is gated by `App.tsx:194-200` on `appState === 'unconfigured'`, runs two steps (`api_key` → `connect accounts`) and has no welcome / language / region step at all. `web/src/data/providers.json` defaults are global-only — no CN endpoint ever suggested.

**Three tiered approaches proposed** (each with effort + file:line citations): **Tier 0 (XS, ~1 day)** — rewrite the existing welcome copy in `SetupWizard.tsx:27-254` to explain what Orbital is in 4-6 sentences; add a region-aware hint band; surface one provider recommendation line. **Tier 1 (M, ~1 wk)** — add a `pick provider` step to the wizard with regional presets (Anthropic CN / OpenAI CN-mirror / Zhipu GLM / DeepSeek / Moonshot Kimi / Alibaba Qwen / Google Gemini / OpenRouter); suggest known CN endpoints per provider; consume [Spec 8](BACKLOG/specs/008-language-toggle-during-onboarding.md)'s language toggle (A2 widget) as a header affordance. **Tier 2 (L, ~2-3 wks)** — full multi-step guided tour: welcome → language → provider+region → API key (with explanation of what the key is) → first project → first chat; one-paragraph explanation + "learn more" link per step.

[Spec at `BACKLOG/specs/017-onboarding-guide-first-launch.md`](BACKLOG/specs/017-onboarding-guide-first-launch.md) covers: current-state citations, the three tiers with file:line diff plans and effort estimates, a provider-preset table (who offers a CN endpoint, who doesn't), the Spec 8 dependency wiring, a Tier-vs-resolves-confusion matrix, and **10 open questions** (which tier, regional preset list, official-CN-endpoint knowledge responsibility, language-toggle placement, force-once vs resumable, IA — modal vs dedicated route vs sidebar, skip-tour button + re-trigger mechanism, integration with [Spec 11](BACKLOG/specs/011-proactive-connectors-onboarding.md)'s `connect accounts` step, post-onboarding `?help` in-app guide, telemetry for where users actually get stuck).

→ Decisions resolved 2026-07-13 (3rd pass): **Tier 1 minimal** — upgraded single provider screen (Tier 2 tour deferred). Q8 promoted into scope — per-provider `console_url` + "Get your API key →" link with one-line how-to (the user's reframed priority). **Fixed preset order for everyone** (CN-friendly first; no locale logic), **default = DeepSeek, never Custom**, **region toggle defaults to China** (CN-first launch audience; region-locked keys mean the default decides who sees a first-try 401). Static "no mainland-China endpoint" captions on OpenAI/Anthropic/Google replace the locale-triggered hint band. Official CN endpoints only; Test Connection prominent but not gating; Spec 8 (A2 widget) is a hard prerequisite. See spec §9.
