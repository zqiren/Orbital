# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Tests for the ``fresh=True`` reset flag on agent_message(send) — BACKLOG
005 §4d.

The default send resumes the sub-agent's prior conversation in the chat
session (§4a/§4b make resume the default); ``fresh=True`` is the deliberate
opt-out that starts a clean thread. This test pins the tool-side plumbing: the
``fresh`` flag is exposed in the schema and threaded into
``SubAgentManager.send`` (default false when omitted).
"""

from __future__ import annotations

import asyncio

from agent_os.agent.tools.agent_message import AgentMessageTool


class _RecordingManager:
    """Captures the kwargs each send() receives."""

    def __init__(self):
        self.sends = []

    async def send(self, project_id, agent, message, *, session_id=None,
                   depth=0, fresh=False):
        self.sends.append({
            "project_id": project_id, "agent": agent, "message": message,
            "session_id": session_id, "depth": depth, "fresh": fresh,
        })
        return f"Message sent to {agent}. Transcript: /tmp/{agent}.jsonl"


def _tool(mgr):
    return AgentMessageTool(
        sub_agent_manager=mgr, project_id="proj", session_id="sess",
    )


def test_fresh_is_in_the_tool_schema():
    tool = _tool(_RecordingManager())
    props = tool.parameters["properties"]
    assert "fresh" in props
    assert props["fresh"]["type"] == "boolean"
    # fresh is NOT required — it must default to resume.
    assert "fresh" not in tool.parameters["required"]


def test_send_defaults_to_resume_when_fresh_omitted():
    mgr = _RecordingManager()
    tool = _tool(mgr)
    result = asyncio.run(tool.execute(action="send", agent="researcher",
                                      message="continue the analysis"))
    assert result.meta == {"yield_turn": True}
    assert len(mgr.sends) == 1
    # Omitting fresh resumes the prior thread (fresh=False).
    assert mgr.sends[0]["fresh"] is False


def test_send_threads_fresh_true_into_manager():
    mgr = _RecordingManager()
    tool = _tool(mgr)
    asyncio.run(tool.execute(action="send", agent="researcher",
                             message="unrelated new task", fresh=True))
    assert len(mgr.sends) == 1
    assert mgr.sends[0]["fresh"] is True


def test_non_send_actions_ignore_fresh():
    """list/status/stop don't take fresh — passing it must not break them."""
    class _Mgr(_RecordingManager):
        def list_active(self, project_id, *, session_id=None):
            return []

    mgr = _Mgr()
    tool = _tool(mgr)
    # Should not raise even though fresh is passed to a non-send action.
    result = asyncio.run(tool.execute(action="list", agent="", fresh=True))
    assert result.content == "[]"
    assert mgr.sends == []
