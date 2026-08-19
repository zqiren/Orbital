# AGENTS.md — read this first

Onboarding for any agentic tool landing in this project — Claude Code, Codex
App, Cursor, Copilot, or otherwise. Read this before touching anything.

## What this project is

Orbital is an AI agent orchestration platform: a management agent that
delegates work to sub-agents (Claude Code, Codex, Gemini CLI, ...) and keeps
a persistent memory of state, decisions, and lessons across sessions instead
of starting cold every time. This repo is **dual-purpose** — it is both
Orbital's own product source (`agent_os/` Python daemon, `web/` React
frontend) *and* a live dogfooding workspace: an Orbital instance manages
this project's own feature backlog using the same memory system described
below. When you land here, you may be touching product code, or you may be
extending the memory that runs the dogfooding loop itself.

## The memory system (read this section fully — it's the core of this file)

Orbital's memory lives under `orbital/`. Each file has a distinct job:

- **`orbital/PROJECT_STATE.md`** — what's true *right now*: current focus,
  in-progress work, blockers, next steps. Read first, every session. It's
  overwritten in place, never appended to — if you change the focus, replace
  the stale lines rather than adding a new dated entry.
- **`orbital/DECISIONS.md`** — durable, settled decisions with their
  reasoning. Read before re-litigating anything that sounds already decided.
  When a new decision lands, append an entry (or supersede/replace an old
  one if it contradicts — never leave two conflicting entries standing).
- **`orbital/LESSONS.md`** — numbered, hard-won gotchas and playbooks from
  past failures. Append a new entry whenever you recover from a non-obvious
  mistake or find a workaround worth remembering next time.
- **`orbital/INDEX.md`** — a navigation map of everything else in the repo
  (one-line-per-path, no dates or decisions). Update it when files move or
  a new area is added; it's the thing that keeps the rest of this list
  honest.
- **`orbital/instructions/`** — standing goals (mission, scope, rules,
  preferences) for whichever agent is operating this workspace. Read to
  understand *why* the rules below exist; edit when the mission changes.
- **`orbital/skills/`** — reusable multi-step procedures captured once a
  workflow repeats (e.g. `learning-capture`, `process-capture`,
  `task-planning`, `efficient-execution`). Check here before inventing an
  approach from scratch; add a new skill dir when a workflow proves durable.
- **`orbital/sub_agents/*/MEMORY.md`** — private, per-agent memory (e.g.
  `orbital/sub_agents/claude-code/MEMORY.md`). If you *are* that agent,
  this is yours to read and append to across dispatches.
- **`BACKLOG.md` + `BACKLOG/specs/`** — the roadmap. Both are **gitignored**
  and local-only, so a fresh clone has neither. `BACKLOG.md` is the curated
  Kanban + spec index; `BACKLOG/specs/` holds the rough implementation specs
  (one per non-trivial backlog item — often carrying the resolved decisions
  for that feature) that `BACKLOG.md` links to.

## Write posture

Agents here are collaborators, not intruders on probation — full read/write
on every file listed above, no append-only games, no policing. Update what
needs updating; the guidance above on *how* each file wants to be edited
(overwrite vs. append vs. supersede) is a courtesy to the next reader, not a
permission gate.

The one hands-off zone: **machine-managed runtime state** —
`orbital/sessions/`, `orbital/ledger/`, `orbital/tool-results/`, and
sub-agent transcript `.jsonl` files. These are process-owned, not a trust
boundary; nobody hand-edits them, agent or human.

Production code (`agent_os/`, `web/`, `scripts/`, `installer/`) is different:
propose a diff for human review, don't push directly — see `BACKLOG.md`'s
ground rules.

## How to recover context (read in order)

1. `README.md` — what Orbital is and why.
2. This file's memory map (above).
3. `orbital/PROJECT_STATE.md` — current focus.
4. `orbital/LESSONS.md` — recent entries especially.
5. `BACKLOG.md` — what's queued next.

**Modifying `agent_os/` or `web/`?** Read `CLAUDE.md` next — it carries the
testing gates, daemon-restart rules, and release process for this codebase.

## Land-mines (distilled from `orbital/LESSONS.md`)

- **Read what already exists before designing a template.** Don't impose a
  schema on `BACKLOG.md` or the memory files without reading their current,
  hand-built shape first — they usually already have one.
- **The cloud relay is control, not storage.** It proxies approvals and
  events; it is not a file/project storage backend. Don't conflate the two
  when scoping "cloud hosting" work.
- **Check a sub-agent's framing matches the actual question before you
  summarize its report.** A thorough answer to the wrong question still
  reads as a good answer unless you explicitly flag the mismatch.
- **Always `tools/list` an external MCP server before estimating connector
  integration effort.** Some servers expose only meta-tools (search/guide/
  execute), not direct action passthrough — this changes the effort
  estimate by an order of magnitude.
