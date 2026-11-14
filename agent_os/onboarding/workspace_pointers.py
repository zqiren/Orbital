# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""What a folder already says about itself, for a project with no goals yet.

Batch 1114 removed the import's background read: an imported project gets
its goals at the end of its first run, like a new one. Two things lean on
the files the folder already holds instead, read by the daemon, with no
model call:

* :func:`pointers` — the paths Orbital's own agent is pointed at on its first
  run in a goal-less project (one tail row, :func:`pointer_row`): the root
  ``CLAUDE.md``, a user-written ``AGENTS.md`` (not the one Orbital seeds),
  ``README*``, and the memory folder Claude Code keeps for this workspace.
  Workers (Claude Code, Codex …) load their own instruction files and memory
  natively and get nothing. Codex keeps no per-project memory file (its
  memories are global, under ``~/.codex/memories``), so there is nothing of
  Codex's to point at.
* :func:`doc_head` — the head of the first of ``CLAUDE.md``, ``AGENTS.md``
  (user-written), ``README*`` that exists, for the Workbench router's block
  of a project with no goals and no state.

Read-only; never raises; blocking (it lists the folder and may probe Claude
Code's project dirs) — call it off the event loop.
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

CLAUDE_MD = "CLAUDE.md"
AGENTS_MD = "AGENTS.md"
README_MAX = 3
# The row's ``_meta.event``: the loop checks for it so the row is written
# once per session.
POINTER_EVENT = "workspace_pointers"

POINTER_HEADER = (
    "This folder already has notes from earlier work. Read the ones that "
    "matter for the user's request before you start:"
)


def _is_file(path: str) -> bool:
    try:
        return os.path.isfile(path)
    except OSError:
        return False


def _user_agents_md(workspace: str) -> str | None:
    path = os.path.join(workspace, AGENTS_MD)
    if not _is_file(path):
        return None
    from agent_os.daemon_v2.agent_md_seeder import is_seeded_agent_md
    return None if is_seeded_agent_md(path) else path


def readmes(workspace: str) -> list[str]:
    """The folder's ``README*`` files (any case), ``README.md`` first."""
    try:
        names = [e.name for e in os.scandir(workspace)
                 if e.name.upper().startswith("README") and e.is_file()]
    except OSError:
        return []
    names.sort(key=lambda n: (n.upper() != "README.MD", len(n), n.lower()))
    return names[:README_MAX]


def claude_memory_dirs(workspace: str) -> list[str]:
    """Claude Code's memory folders for this workspace that hold a note
    (``~/.claude/projects/<dir>/memory``)."""
    from agent_os.onboarding.history_paths import claude_dir, history_home
    from agent_os.onboarding.import_scanner import claude_project_dirs_for
    from agent_os.onboarding.sources.claude_code import MEMORY_DIRNAME
    projects = os.path.join(claude_dir(history_home()), "projects")
    out = []
    for d in claude_project_dirs_for(projects, workspace):
        mem = os.path.join(d, MEMORY_DIRNAME)
        try:
            if any(e.is_file() and e.name.lower().endswith(".md") for e in os.scandir(mem)):
                out.append(mem)
        except OSError:
            continue
    return sorted(out)


def pointers(workspace: str) -> list[str]:
    """The paths to name, in order: workspace-relative for files in the
    folder, absolute for Claude Code's memory folder (outside the workspace;
    the agent's read tool takes it as is). ``[]`` when there is nothing, or
    on any failure."""
    if not workspace or not os.path.isdir(workspace):
        return []
    out: list[str] = []
    try:
        if _is_file(os.path.join(workspace, CLAUDE_MD)):
            out.append(CLAUDE_MD)
        if _user_agents_md(workspace):
            out.append(AGENTS_MD)
        out.extend(readmes(workspace))
        out.extend(d.replace("\\", "/").rstrip("/") + "/" for d in claude_memory_dirs(workspace))
    except Exception:
        logger.warning("workspace pointers: could not list %s", workspace, exc_info=True)
    return out


def pointer_row(paths: list[str]) -> str | None:
    """The one tail row's text, or None when there is nothing to name."""
    if not paths:
        return None
    return POINTER_HEADER + "\n" + "\n".join(f"- {p}" for p in paths)


def doc_head(workspace: str, cap: int) -> tuple[str, str] | None:
    """``(file name, head)`` of the first of ``CLAUDE.md``, a user-written
    ``AGENTS.md``, ``README*`` that has text: at most ``cap`` characters,
    cut at a line end when one is near. None when there is none."""
    if not workspace or not os.path.isdir(workspace):
        return None
    try:
        names = []
        if _is_file(os.path.join(workspace, CLAUDE_MD)):
            names.append(CLAUDE_MD)
        if _user_agents_md(workspace):
            names.append(AGENTS_MD)
        names.extend(readmes(workspace))
        for name in names:
            try:
                with open(os.path.join(workspace, name), encoding="utf-8",
                          errors="replace") as fh:
                    text = fh.read(cap * 4)
            except OSError:
                continue
            text = text.strip()
            if not text:
                continue
            if len(text) > cap:
                cut = text.rfind("\n", 0, cap)
                text = text[:cut if cut >= cap // 2 else cap].rstrip()
            return name, text
    except Exception:
        logger.warning("workspace doc head: could not read %s", workspace, exc_info=True)
    return None
