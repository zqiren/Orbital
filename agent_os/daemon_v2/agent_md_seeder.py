# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""`AGENTS.md` seeder — writes the onboarding signpost at a new project's
workspace **root** (not under ``orbital/``).

The workspace root is where an external agentic tool (Claude Code, Codex,
Cursor, Copilot) lands when a human opens the project outside Orbital. Orbital
already renders an equivalent pointer list in-process at dispatch time
(``agent_os/agent/sub_agent_prompt.py::render_sub_agent_prompt``); this module
is the on-disk half of that same onboarding contract, for tools Orbital did not
dispatch.

Contract mirrors ``default_skills_installer.py``: resolve → guard → write →
return a ``{"status": ...}`` dict, with scratch projects skipped entirely.

Two constraints worth stating because they are not visible from the code:

* The seed runs **once, at project creation** — never on agent start. A user
  who deletes the file has deleted it for good, which is the intended
  behaviour (unlike skills, there is nothing to reconcile).
* The template must stay **committable**: no timestamps, no session ids, no
  absolute paths. A user's workspace is often their own git repo, and a seeded
  file carrying machine-local paths would produce a diff that breaks on any
  other checkout.
"""

import logging
import os

logger = logging.getLogger(__name__)

AGENT_MD_FILENAME = "AGENTS.md"

# Substitution is `str.format`, so the template must contain **no literal
# braces** beyond the named fields below.
AGENT_MD_TEMPLATE = """# AGENTS.md — read this first

Onboarding for any agentic tool landing in this project — Claude Code, Codex,
Cursor, Copilot, or otherwise. Read this before touching anything.

Orbital generated this file when the project was created, and never rewrites
it. Edit it freely; your changes will not be overwritten. Where it disagrees
with a hand-authored `CLAUDE.md` or anything under `orbital/instructions/`,
those win.

## This project

- **Project:** {project_name}
- **Orbital agent:** {agent_name}
- **Workspace:** the directory this file sits in.

## The memory system

Orbital is an agent orchestration platform: a management agent works on this
project across many sessions and keeps what it learns on disk instead of
starting cold each time. That memory lives in `orbital/`, beside this file. It
is as much yours as Orbital's — read it to recover context, and update it when
you learn something worth keeping.

These files accumulate as the project runs. One that does not exist yet simply
means nothing has been recorded there; it is not a broken install. In a
brand-new project this file may be the only one present.

- **`orbital/PROJECT_STATE.md`** — what is true *right now*: current focus,
  work in progress, blockers, next steps. Read it first, every session.
  Overwrite stale lines in place rather than appending a dated entry.
- **`orbital/DECISIONS.md`** — settled decisions and the reasoning behind
  them. Read before re-litigating anything that sounds already decided. Append
  when a decision lands; supersede the old entry outright if they conflict —
  never leave two contradictory ones standing.
- **`orbital/LESSONS.md`** — hard-won gotchas and playbooks from past
  failures. Append whenever you recover from a non-obvious mistake or find a
  workaround worth remembering next time.
- **`orbital/INDEX.md`** — the navigation map: one line per path. Start here
  when you do not know where something lives, and update it when files move or
  a new area appears.
- **`orbital/instructions/`** — standing goals, scope, and the user's own
  directives for whoever operates this workspace. Read these to understand
  *why* the conventions here exist.
- **`orbital/skills/`** — reusable multi-step procedures, captured once a
  workflow has repeated. Check here before inventing an approach from scratch.
- **`orbital/sub_agents/<slug>/MEMORY.md`** — private memory for one
  sub-agent. If you are that agent, this file is yours to read and append to
  across dispatches.

`orbital/INDEX.md` describes the layout as it actually is today. Where this
file's map has drifted from it, believe INDEX.md.

## Write posture

Full read/write on everything listed above — no append-only games, no asking
permission first. Update what needs updating. The guidance on *how* each file
wants to be edited (overwrite vs. append vs. supersede) is a courtesy to the
next reader, not a gate.

The one hands-off zone is machine-managed runtime state, which nobody
hand-edits — agent or human:

- `orbital/sessions/` — session transcripts
- `orbital/ledger/` — cost and usage records
- `orbital/tool-results/` — captured tool output
- `orbital/sub_agents/*/*.jsonl` — sub-agent dispatch transcripts

Reading those while debugging is fine; editing them corrupts state Orbital
depends on.

## Recovering context

1. `orbital/PROJECT_STATE.md` — where things stand.
2. `orbital/DECISIONS.md` — what is already settled.
3. `orbital/LESSONS.md` — recent entries especially.
4. `orbital/INDEX.md` — where everything lives.
"""


def seed_project_agent_md(project_store, project_id: str) -> dict:
    """Write ``{workspace}/AGENTS.md`` for *project_id* if it is not there yet.

    Contract (see ``BACKLOG/specs/030-agents-md-seeding-on-project-creation.md``):

    * Scratch (Quick Tasks) projects → ``{"status": "skipped_scratch"}``, no
      side effects. The skip lives here rather than at the call site so a
      future third create path inherits it — same posture as
      ``install_default_skills``.
    * Target already exists → ``{"status": "skipped_exists"}``. Checked with
      ``os.path.exists`` (not ``isfile``) so a *directory* of that name also
      short-circuits instead of raising. On case-insensitive filesystems this
      additionally means an existing ``agents.md`` blocks the seed, which is
      correct — to the OS they are the same file. On Linux they are not, and a
      second file differing only in case is the accepted outcome.
    * Write failure (read-only workspace, permissions) → ERROR log and
      ``{"status": "write_failed"}``; never raises, because a missing signpost
      must not fail project creation.

    Raises:
        ValueError: the project does not exist.
    """
    project = project_store.get_project(project_id)
    if project is None:
        raise ValueError(f"project {project_id!r} not found")

    if project.get("is_scratch"):
        return {"status": "skipped_scratch"}

    workspace = project.get("workspace", "")
    if not workspace:
        logger.warning(
            "cannot seed %s — project %s has no workspace",
            AGENT_MD_FILENAME, project_id,
        )
        return {"status": "no_workspace"}

    dest = os.path.join(workspace, AGENT_MD_FILENAME)
    if os.path.exists(dest):
        return {"status": "skipped_exists", "path": dest}

    content = AGENT_MD_TEMPLATE.format(
        project_name=project.get("name") or "(unnamed)",
        agent_name=project.get("agent_name") or project.get("name") or "(unnamed)",
    )

    try:
        # newline="\n" so the seeded file is byte-identical on every platform —
        # it is expected to land in the user's own git repo.
        with open(dest, "w", encoding="utf-8", newline="\n") as f:
            f.write(content)
    except OSError:
        logger.error(
            "failed to seed %s into project %s", AGENT_MD_FILENAME, project_id,
            exc_info=True,
        )
        return {"status": "write_failed", "path": dest}

    logger.info("seeded %s into project %s", AGENT_MD_FILENAME, project_id)
    return {"status": "ok", "path": dest}


# Prior revisions of AGENT_MD_TEMPLATE (verbatim). When the live template
# changes, append the outgoing text here so ``reseed_project_agent_md`` can
# still recognize a file seeded from it as UNEDITED and safely refresh it.
# The current template is not listed — a file matching it needs no rewrite.
_HISTORICAL_TEMPLATES: tuple[str, ...] = ()


def reseed_project_agent_md(project_store, project_id: str) -> dict:
    """Hash-guarded ``AGENTS.md`` refresh at pin time (spec 074 §3.6).

    The guard is MANDATORY: the seeded file invites the user to edit it
    ("your changes will not be overwritten"), so an unconditional rewrite is
    a data-loss bug. Rewrite ONLY when the current content matches a version
    Orbital itself seeded (the live template rendering, or a listed
    historical one) — i.e. the file is provably unedited. Anything else is
    left alone and logged.

    Statuses:

    * ``skipped_scratch`` / ``no_workspace`` — same guards as the seeder.
    * ``seeded`` — no file existed (e.g. pre-spec-030 project, or the user
      deleted it); the current template was written. Codex's only context
      channel is this file, so a pin must ensure it exists.
    * ``unchanged`` — file already matches the current template; no write.
    * ``reseeded`` — file matched a HISTORICAL seeded version verbatim;
      refreshed to the current template.
    * ``skipped_user_modified`` — content matches no seeded version; the
      file is the user's now. Left untouched, logged.
    * ``read_failed`` / ``write_failed`` — I/O errors; never raises.

    Raises:
        ValueError: the project does not exist.
    """
    project = project_store.get_project(project_id)
    if project is None:
        raise ValueError(f"project {project_id!r} not found")

    if project.get("is_scratch"):
        return {"status": "skipped_scratch"}

    workspace = project.get("workspace", "")
    if not workspace:
        return {"status": "no_workspace"}

    fields = {
        "project_name": project.get("name") or "(unnamed)",
        "agent_name": project.get("agent_name") or project.get("name") or "(unnamed)",
    }
    current = AGENT_MD_TEMPLATE.format(**fields)
    dest = os.path.join(workspace, AGENT_MD_FILENAME)

    if not os.path.exists(dest):
        try:
            with open(dest, "w", encoding="utf-8", newline="\n") as f:
                f.write(current)
        except OSError:
            logger.error(
                "failed to reseed %s into project %s", AGENT_MD_FILENAME,
                project_id, exc_info=True,
            )
            return {"status": "write_failed", "path": dest}
        logger.info(
            "reseed: %s was missing for project %s — seeded", AGENT_MD_FILENAME,
            project_id,
        )
        return {"status": "seeded", "path": dest}

    try:
        with open(dest, "r", encoding="utf-8", newline="") as f:
            existing = f.read()
    except OSError:
        return {"status": "read_failed", "path": dest}

    # Normalize line endings for the comparison only (a CRLF checkout of an
    # otherwise-unedited seed still counts as unedited); writes stay "\n".
    normalized = existing.replace("\r\n", "\n")
    if normalized == current:
        return {"status": "unchanged", "path": dest}

    for tmpl in _HISTORICAL_TEMPLATES:
        try:
            if normalized == tmpl.format(**fields):
                break
        except (KeyError, IndexError):  # a historical template with other fields
            continue
    else:
        logger.info(
            "reseed: %s for project %s was edited by the user — leaving it "
            "untouched", AGENT_MD_FILENAME, project_id,
        )
        return {"status": "skipped_user_modified", "path": dest}

    try:
        with open(dest, "w", encoding="utf-8", newline="\n") as f:
            f.write(current)
    except OSError:
        logger.error(
            "failed to reseed %s into project %s", AGENT_MD_FILENAME,
            project_id, exc_info=True,
        )
        return {"status": "write_failed", "path": dest}
    logger.info(
        "reseed: refreshed unedited %s for project %s to the current "
        "template", AGENT_MD_FILENAME, project_id,
    )
    return {"status": "reseeded", "path": dest}
