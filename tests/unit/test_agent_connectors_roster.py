# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Agent-held connectors route tasks: the agents that can hand off work learn
which agent holds which connector.

- The manager's sub-agent roster: one short line under an agent with known
  connectors (names only, capped, sorted), in the roster that already sits
  in the semi-stable block, so the cached prefix moves only when the set does.
- The ``dispatch`` tool's agent list and the seat note (what a pinned worker
  sees) carry the same names.
- An agent with nothing known adds nothing: byte-identical to before.
"""

from __future__ import annotations

import pytest

from agent_os.agent.prompt_builder import Autonomy, PromptBuilder, PromptContext
from agent_os.daemon_v2 import agent_connectors as ac
from agent_os.daemon_v2.dispatch_tools import DispatchContext, seat_note, tool_description

AGENTS = [
    {"handle": "claude-code", "display_name": "Claude Code", "type": "cli"},
    {"handle": "codex", "display_name": "Codex", "type": "cli"},
]


@pytest.fixture(autouse=True)
def _registry(tmp_path):
    ac.configure(str(tmp_path / "data"))
    yield
    ac.configure(None)


def _ctx(tmp_path):
    return PromptContext(workspace=str(tmp_path), model="m", autonomy=Autonomy.CHECK_IN,
                         enabled_agents=[dict(a) for a in AGENTS], tool_names=[],
                         os_type="macos", datetime_now="2026-11-17T00:00:00")


def _roster(tmp_path):
    return PromptBuilder(workspace=None)._sub_agents(_ctx(tmp_path))


def _claude():
    ac.record("claude-code", [
        {"name": "claude.ai Gmail", "status": "pending"},
        {"name": "figma", "status": "needs-auth"},
        {"name": "playwright", "status": "connected"},
        {"name": "orbital", "status": "connected"}])


# ------------------------------------------------------------------ manager

def test_nothing_known_leaves_the_roster_byte_identical(tmp_path):
    before = _roster(tmp_path)
    ac.configure(None)
    assert _roster(tmp_path) == before
    assert "connectors" not in before


def test_the_roster_names_an_agents_own_connectors(tmp_path):
    before = _roster(tmp_path)
    _claude()
    after = _roster(tmp_path)
    lines = after.split("\n")
    i = lines.index("- **claude-code** (Claude Code): cli agent.")
    assert lines[i + 1] == ("  Its own connectors (only it can use them; hand it tasks "
                            "that need one): figma, Gmail, playwright")
    # Only that line was added; codex has nothing known and is unchanged.
    assert after.replace(lines[i + 1] + "\n", "", 1) == before


def test_the_roster_is_capped_and_deterministic(tmp_path):
    ac.record("codex", [{"name": f"s{i:02d}", "status": "connected"} for i in range(12)])
    first = _roster(tmp_path)
    assert "s00, s01, s02, s03, s04, s05, s06, s07 and 4 more" in first
    assert _roster(tmp_path) == first


def test_the_roster_moves_only_when_the_set_changes(tmp_path):
    _claude()
    first = _roster(tmp_path)
    # A status change that keeps the names keeps the bytes.
    ac.record("claude-code", [
        {"name": "claude.ai Gmail", "status": "connected"},
        {"name": "figma", "status": "connected"},
        {"name": "playwright", "status": "connected"}])
    assert _roster(tmp_path) == first


# ------------------------------------------------------- pinned worker's tool

def _dctx(handle="codex"):
    return DispatchContext("p1", "s1", handle, agents=("claude-code", "codex"))


def test_dispatch_description_unchanged_with_nothing_known():
    assert tool_description(_dctx()).endswith(" Agents: claude-code.")
    assert seat_note(_dctx()) == (
        "Other agents in this project: claude-code. When the user asks for one "
        "of them, hand it the task with the `dispatch` tool of the `orbital` MCP "
        "server. Never run that agent's CLI from your shell: its reply would "
        "never reach the user.")


def test_dispatch_description_names_the_other_agents_connectors():
    _claude()
    desc = tool_description(_dctx())
    assert desc.endswith(" Agents: claude-code (its own connectors: figma, Gmail, "
                         "playwright).")
    note = seat_note(_dctx())
    assert note.startswith("Other agents in this project: claude-code (its own "
                           "connectors: figma, Gmail, playwright).")
    assert "or for a task that needs one of their own connectors" in note


def test_a_worker_never_lists_its_own_connectors():
    _claude()
    assert "figma" not in tool_description(_dctx("claude-code"))
    assert seat_note(_dctx("claude-code")).startswith(
        "Other agents in this project: codex. When the user asks")
