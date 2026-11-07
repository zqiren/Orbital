# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Map Claude Code tool names to risk categories for autonomy-based approval."""

from __future__ import annotations

from agent_os.agent.prompt_builder import Autonomy

# ---------------------------------------------------------------------------
# Risk categories
# ---------------------------------------------------------------------------
READ = "read"
WRITE = "write"
SHELL = "shell"
DELEGATE = "delegate"
REQUIRES_APPROVAL = "requires_approval"

# ---------------------------------------------------------------------------
# Tool -> category mapping
# ---------------------------------------------------------------------------
_TOOL_CATEGORY: dict[str, str] = {
    # read / observation tools
    "Read":        READ,
    "Glob":        READ,
    "Grep":        READ,
    "LS":          READ,
    "Search":      READ,
    "Explore":     READ,
    "TaskGet":     READ,
    "TaskList":    READ,
    "TaskOutput":  READ,
    "WebSearch":   READ,
    "WebFetch":    READ,
    "AskUser":     READ,
    # write / mutation tools
    "Edit":         WRITE,
    "Write":        WRITE,
    "MultiEdit":    WRITE,
    "NotebookEdit": WRITE,
    "TodoWrite":    WRITE,
    "TaskCreate":   WRITE,
    "TaskUpdate":   WRITE,
    # shell tools
    "Bash":        SHELL,
    "ExecuteBash": SHELL,
    # delegation tools
    "Agent":       DELEGATE,
}


# Spec 161: Orbital's own dispatch tool, served to the top-of-session worker
# over the daemon's MCP endpoint. Approved by NAME under check-in: the user
# put this worker in the seat, and handing work on is what the seat does.
# Claude Code's native ``Agent`` (the DELEGATE class) keeps prompting.
_APPROVED_BY_NAME = frozenset({"mcp__orbital__dispatch"})


def classify_tool(tool_name: str) -> str:
    """Return the risk category for *tool_name*.

    Unrecognised tool names are classified as ``requires_approval``.
    """
    return _TOOL_CATEGORY.get(tool_name, REQUIRES_APPROVAL)


def should_auto_approve(tool_name: str, autonomy: Autonomy) -> bool:
    """Decide whether *tool_name* can be auto-approved under *autonomy*.

    Policy:
        HANDS_OFF  — auto-approve everything.
        CHECK_IN   — auto-approve ``read`` tools and ``mcp__orbital__dispatch``.
        SUPERVISED — never auto-approve (surface every request).
    """
    if autonomy is Autonomy.HANDS_OFF:
        return True
    if autonomy is Autonomy.SUPERVISED:
        return False
    # CHECK_IN: read tools, plus Orbital's own dispatch tool by name
    if tool_name in _APPROVED_BY_NAME:
        return True
    return classify_tool(tool_name) == READ
