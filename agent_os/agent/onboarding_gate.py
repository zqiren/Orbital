# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""The onboarding gate: onboarding ends with a proposal and waits for the go.

From the moment a run CREATES ``orbital/instructions/project_goals.md`` until
the user's next message in that session, the daemon refuses every tool that
acts on the world (shell, writes outside ``orbital/``, worker dispatch,
automations, notifications, browser actions, connector tools). Reads,
searches, listings and memory writes under ``orbital/`` still run, so the
model can finish INDEX.md / PROJECT_STATE.md / ASKS.md and reply with its
proposed first step. The user's reply ("开始", "交给 Codex", "先改成 500 字")
is the go.

The flag is daemon state, not prompt state: ``Session.onboarding_gate``, set
by the loop's tool-dispatch path (``AgentLoop.run``) and cleared by
``Session.append`` on a real user row. The prompt is rebuilt every call and
never mentions it, so the cache prefix is untouched. A project whose goals
file already exists never arms it (an overwrite is not a creation), and
unattended runs (automation sessions, queue runs) are exempt.

Pure functions only; the loop owns the state.
"""

from __future__ import annotations

import os

from agent_os.agent.project_paths import ProjectPaths
from agent_os.agent.tools._path_utils import resolve_safe

# One line, returned as the refused call's tool result.
REFUSAL = (
    "Onboarding is finishing: reply to the user with your proposed first step "
    "(2–3 lines) and who does it, then stop. Tools resume after the user's reply."
)

# A run that is refused this many times ends (the text it wrote stays).
MAX_REFUSALS_PER_RUN = 2

# Tools that only look. Everything not listed here or handled below is
# refused, so a new tool or a connector (MCP) tool fails closed.
_READ_ONLY_TOOLS = frozenset({
    "read", "glob", "grep",
    "list_sessions", "read_session",
    "list_triggers", "calendar_read",
    # Memory, not the world: user-level memory file / background tidy of the
    # orbital/ memory files.
    "remember_about_user", "checkpoint_state",
})
_FILE_TOOLS = frozenset({"write", "edit"})
_AGENT_MESSAGE_READS = frozenset({"list", "status"})
_BROWSER_READS = frozenset({"search", "fetch"})


def _resolved_path(workspace: str, tc: dict) -> str | None:
    args = tc.get("arguments") or {}
    path = args.get("path") if isinstance(args, dict) else None
    if not workspace or not isinstance(path, str) or not path.strip():
        return None
    return resolve_safe(workspace, path.strip())


def _under_orbital(workspace: str, resolved: str) -> bool:
    orbital = os.path.realpath(ProjectPaths(workspace).orbital_dir)
    return resolved.startswith(orbital + os.sep)


def allowed(workspace: str, tc: dict) -> bool:
    """True when ``tc`` may run while the gate is up."""
    name = tc.get("name") or ""
    args = tc.get("arguments") or {}
    if not isinstance(args, dict):
        args = {}
    if name in _READ_ONLY_TOOLS:
        return True
    if name in _FILE_TOOLS:
        resolved = _resolved_path(workspace, tc)
        return resolved is not None and _under_orbital(workspace, resolved)
    if name == "agent_message":
        return args.get("action") in _AGENT_MESSAGE_READS
    if name == "browser":
        return args.get("action") in _BROWSER_READS
    return False


def creates_goals(workspace: str, tc: dict) -> bool:
    """True when ``tc`` is a ``write`` that would create the goals file.

    Checked before the call runs: an overwrite of an existing goals file
    (an established project updating its directive) is not a creation.
    """
    if tc.get("name") != "write":
        return False
    resolved = _resolved_path(workspace, tc)
    if resolved is None:
        return False
    goals = os.path.realpath(ProjectPaths(workspace).project_goals)
    return resolved == goals and not os.path.exists(goals)
