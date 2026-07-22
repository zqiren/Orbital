# Feature: Simplify New Project Creation

**Status:** spec-written · **Date:** 2026-07-21 · **Effort:** S (~1 day frontend-heavy, optional XS backend)

## Purpose

The user finds the current "New Project" flow too troublesome — too many steps
and too much information demanded up front. The goal is to minimize the
decisions required to get from "New Project" click to a working agent, deferring
everything that already has a sensible default or is editable later in Project
Settings.

The evidence that a minimal payload is enough already exists in the codebase:
the daemon auto-creates the "Quick Tasks" scratch project with nothing but
`name`, `agent_name`, `workspace`, `autonomy`, and empty `model`/`api_key`
(`agent_os/api/app.py:90-106`), and it works because runtime LLM config falls
back to global settings (`agent_os/daemon_v2/agent_manager.py:1431-1457`).

## Current Flow

`web/src/components/CreateProject.tsx` renders a full-page form with **7
sections**; the user must consider all of them before the single "Deploy"
button at the bottom:

| # | Field | Required? | Notes |
|---|-------|-----------|-------|
| 1 | Project Name | **Required** (non-empty, `CreateProject.tsx:67`) | Sanitized server-side (`project_store.py:170`) |
| 2 | Agent Name | Optional | Server defaults it to project name anyway (`project_store.py:172`) — near-zero value as a creation-time question |
| 3 | Workspace Path | **Required**, absolute path regex (`CreateProject.tsx:47-49`) | Browse button → `FolderPickerModal`. Backend 400s if the directory doesn't already exist (`agents_v2.py:460-461`); **neither the picker nor the backend can create a folder**, so users must leave the app to `mkdir` first |
| 4 | Instructions | Optional | 6-row textarea invites an essay at creation time; fully editable later |
| 5 | LLM Provider | Read-only info card | `LLMProviderSettings mode="wizard"` (`LLMProviderSettings.tsx:571-605`) just states "using global defaults" or warns if no API key — informational, but visually another section to parse |
| 6 | Autonomy | Pre-selected `hands_off` | 3-card selector (`CreateProject.tsx:25-45`) forces reading three descriptions to confirm the default |
| 7 | Budget Limit | Optional | Number input + hint text |

The form submits `model: ''` and `api_key: ''` hardcoded
(`CreateProject.tsx:84-85`) — the project always inherits global LLM config at
creation. `CreateProjectRequest` (`agents_v2.py:46-70`) nevertheless declares
`model` and `api_key` as **required** fields (empty string allowed), a vestige
that forces every client to send placeholder values.

Net decision load: **2 truly required inputs** (name + pre-existing workspace
path) buried in 7 sections of optional/informational chrome, plus an
out-of-app detour whenever the target folder doesn't exist yet.

## Proposed Simplification

### Option A — Minimal one-decision form
Workspace folder picker only. Name auto-derived from the folder basename,
shown as editable text. Create button. Everything else defaulted invisibly.
- Pro: absolute minimum. Con: no escape hatch for users who *do* want to set
  instructions/autonomy up front; power users forced into a second trip to
  Settings.

### Option B — Two-stage (quick create → in-project onboarding)
Stage 1 is Option A; stage 2 surfaces a dismissible "finish setting up" card
inside the project (instructions, autonomy, budget).
- Pro: nicest narrative. Con: new surface to build and localize; the project
  Settings tab already *is* stage 2 — this duplicates it.

### Option C — Smart defaults + progressive disclosure (**recommended**)
Keep one form, but only two visible controls; collapse the rest:

1. **Workspace** — picker + path input, unchanged validation. On selection,
   **auto-fill Project Name from the folder basename** (editable; stop
   auto-filling once the user types their own). Order flipped: workspace
   first, since name derives from it.
2. **Project Name** — pre-filled, editable, still required.
3. **Everything else moves under a collapsed "Advanced options" disclosure**
   (same pattern as the `mode="project"` collapsible header already in
   `LLMProviderSettings.tsx:617`): Agent Name, Instructions, Autonomy cards,
   Budget. All keep current defaults (`hands_off`, empty, unset).
4. **LLM info card** shrinks to a single conditional line: render *only* the
   warning variant when `!api_key_set` (the "using global defaults" happy-path
   card disappears — it asks the user to read a paragraph to learn "nothing
   needed here"). Keep the warning: hiding it would let users create projects
   whose agent can't start.
5. Optional backend nicety (XS): allow the workspace leaf directory to be
   created on demand — either a "New folder" affordance in `FolderPickerModal`
   or a `create_workspace: bool` flag on the endpoint that `os.mkdir`s the
   leaf when the parent exists. This removes the last out-of-app detour.
   Ship-separable from the form change.

Result: the default path is **pick folder → (name pre-filled) → Create** — one
real decision, matching the scratch-project precedent.

### Backend cleanup (bundled, XS)
Make `model` and `api_key` optional with default `""` in
`CreateProjectRequest` (`agents_v2.py:49-50`) and in
`web/src/types.ts:84-97`'s `ProjectCreateRequest`, so the frontend stops
sending placeholder empties. Behavior-neutral: the endpoint already treats
empty string as "inherit global" (BYOK dedup at `agents_v2.py:462-469`,
runtime fallback in `agent_manager.py:1435-1440`).

## Files Touched

- `web/src/components/CreateProject.tsx` — reorder fields, name auto-derivation,
  Advanced disclosure, conditional LLM warning. Main change.
- `web/src/types.ts` — `ProjectCreateRequest.model` / `.api_key` → optional.
- `agent_os/api/routes/agents_v2.py` — `CreateProjectRequest` optional
  `model`/`api_key` (default `""`); optional `create_workspace` flag if the
  folder-creation nicety is taken.
- `web/src/components/LLMProviderSettings.tsx` — wizard mode: render only the
  not-configured warning (happy-path card removed) *or* leave untouched and
  gate rendering from CreateProject.
- `web/src/components/FolderPickerModal.tsx` + platform browse route — only if
  the "New folder" affordance option is chosen instead of the endpoint flag.
- `web/src/i18n/strings.ts` — new keys (`createProject.advanced`, derived-name
  hint); run `node web/scripts/check-i18n.mjs`; ship English-first per i18n
  policy.
- Tests: Vitest for name-derivation + advanced-disclosure logic
  (`web/src/components/CreateProject.test.tsx`, new); backend unit test for
  optional `model`/`api_key` and (if added) `create_workspace`.

## Risks

- **agent_name uniqueness collisions.** `project_store.create_project` raises
  `ValueError` → HTTP 409 when the (defaulted) agent name is taken
  (`project_store.py:175-177`). Auto-derived names ("repo", "project") make
  collisions likelier. Mitigation: surface the 409 inline on the name field
  (error path already exists) — do **not** silently suffix, since the name is
  user-visible identity.
- **Backward compatibility: none at rest.** Changes are creation-time only;
  existing project records, `ProjectUpdate`, and the scratch-project path are
  untouched. Making required request fields optional is loosening-only for API
  clients.
- **Hidden ≠ gone.** Every field moved into Advanced stays editable in the
  project Settings tab (all covered by `ProjectUpdate`, `agents_v2.py:78+`).
  Verify Settings parity before removing anything from the form.
- **Onboarding expectations.** The first-run `SetupWizard` (api_key → accounts)
  is a separate surface and is not touched; Spec 17's Tier-1 provider
  onboarding also lands in the wizard, not this form — no overlap. The only
  interaction is the `api_key_set` warning, which this spec keeps.
- **Auto-create workspace safety.** If the `create_workspace` flag is taken,
  create only the leaf under an existing parent (`os.mkdir`, not
  `makedirs`) so a typo'd path fails loudly instead of materializing a deep
  wrong tree.
- **i18n/layout.** New strings need catalog keys; zh strings can overflow the
  disclosure header — run the EN/ZH screenshot pass per CLAUDE.md when
  touching layout.

## Implementation Notes

- Name derivation: `basename(workspace).trim()` with the existing server-side
  sanitizer as the source of truth; client mirrors it loosely and lets the
  server's sanitized value win (response is the created project).
- Track "user has manually edited name" with a boolean; folder re-selection
  overwrites the name only while it's untouched.
- The Autonomy cards move into Advanced *unchanged* — `hands_off` remains the
  backend default (`agents_v2.py:476`), so omitting the field entirely is
  also safe.
- Budget currency resolution (`agents_v2.py:509-517`) only runs when a limit
  is supplied — hiding the field changes nothing.
- Daemon smoke test per CLAUDE.md: restart daemon, create a project through
  the new form with only folder+name, confirm 201, confirm agent starts and
  inherits global provider/model; then a second create in the same folder
  name to exercise the 409 inline error.
- Follow-up candidate (out of scope): recent-workspaces shortcut list on the
  picker already exists (`FolderPickerModal.tsx` recentPaths) — surfacing it
  directly in the create form could make repeat creation one click.
