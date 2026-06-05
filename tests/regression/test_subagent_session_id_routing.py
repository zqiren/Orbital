# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Sub-agent lifecycle is routed under the management session's SessionKey.

The pre-fix bug: AgentMessageTool called sub_agent_manager.start/send/stop/etc.
without threading the management session's ``session_id``, so the adapter was
always registered under SessionKey(project, "default"). When the management
session was non-default (e.g. ``sess_82924192``):

  - ``_on_loop_done(session_id=sess_X)`` → ``list_active(project, sess_X)``
    looked in the wrong bucket → empty → no idle-poll → status reported
    ``idle`` instead of ``waiting``.
  - ``stop_all(project, sess_X)`` (eviction / reap / cancel) targeted the wrong
    bucket → no-op → sub-agent processes leaked.
  - ``on_completed`` injected a system message into the wrong session
    (or "no_session") → the management agent's loop was never resumed with
    the result.

The fix: every sub-agent lifecycle call carries the management session's
``session_id`` end-to-end (tool → manager → observer → ``inject_system_message``).
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from agent_os.agent.tools.agent_message import AgentMessageTool
from agent_os.daemon_v2.lifecycle_observer import LifecycleObserver


def _tool(session_id: str | None = "sess_X") -> tuple[AgentMessageTool, MagicMock]:
    mgr = MagicMock()
    mgr.start = AsyncMock(return_value="Started claude-code")
    mgr.send = AsyncMock(return_value="Message sent to claude-code. Transcript: /x")
    mgr.stop = AsyncMock(return_value="Stopped claude-code")
    mgr.status = MagicMock(return_value="running")
    mgr.list_active = MagicMock(return_value=[])
    tool = AgentMessageTool(
        sub_agent_manager=mgr, project_id="proj",
        depth=0, session_id=session_id,
    )
    return tool, mgr


# --- AgentMessageTool threads session_id to every sub_agent_manager.* call --

@pytest.mark.asyncio
async def test_send_threads_session_id():
    tool, mgr = _tool("sess_X")
    await tool.execute(action="send", agent="claude-code", message="go")
    mgr.send.assert_awaited_once()
    assert mgr.send.await_args.kwargs.get("session_id") == "sess_X"


@pytest.mark.asyncio
async def test_spawn_via_send_threads_session_id():
    """start left the tool surface (send spawns-on-demand,
    TASK-collapse-dispatch-to-send); the spawn's session threading now rides
    the send call — the manager forwards session_id to its internal start()."""
    tool, mgr = _tool("sess_X")
    result = await tool.execute(action="start", agent="claude-code")
    assert "unknown action" in result.content.lower()
    mgr.start.assert_not_called()
    await tool.execute(action="send", agent="claude-code", message="go")
    assert mgr.send.await_args.kwargs.get("session_id") == "sess_X"


@pytest.mark.asyncio
async def test_stop_threads_session_id():
    tool, mgr = _tool("sess_X")
    await tool.execute(action="stop", agent="claude-code")
    mgr.stop.assert_awaited_once()
    assert mgr.stop.await_args.kwargs.get("session_id") == "sess_X"


@pytest.mark.asyncio
async def test_status_threads_session_id():
    tool, mgr = _tool("sess_X")
    await tool.execute(action="status", agent="claude-code")
    mgr.status.assert_called_once()
    assert mgr.status.call_args.kwargs.get("session_id") == "sess_X"


@pytest.mark.asyncio
async def test_list_threads_session_id():
    tool, mgr = _tool("sess_X")
    await tool.execute(action="list", agent="claude-code")
    mgr.list_active.assert_called_once()
    assert mgr.list_active.call_args.kwargs.get("session_id") == "sess_X"


# --- LifecycleObserver threads session_id to inject_system_message ---------

def _obs() -> tuple[LifecycleObserver, MagicMock]:
    agent_mgr = MagicMock()
    agent_mgr.inject_system_message = AsyncMock(return_value="delivered")
    obs = LifecycleObserver(agent_manager=agent_mgr, ws_manager=MagicMock())
    return obs, agent_mgr


@pytest.mark.asyncio
async def test_on_started_threads_session_id():
    obs, agent_mgr = _obs()
    await obs.on_started("proj", "claude-code", initiator="management_agent",
                         transcript_path="/x", session_id="sess_X")
    agent_mgr.inject_system_message.assert_awaited_once()
    assert agent_mgr.inject_system_message.await_args.kwargs.get("session_id") == "sess_X"


@pytest.mark.asyncio
async def test_on_message_routed_threads_session_id():
    obs, agent_mgr = _obs()
    await obs.on_message_routed("proj", "claude-code", initiator="management_agent",
                                message_preview="task", transcript_path="/x",
                                session_id="sess_X")
    agent_mgr.inject_system_message.assert_awaited_once()
    assert agent_mgr.inject_system_message.await_args.kwargs.get("session_id") == "sess_X"


@pytest.mark.asyncio
async def test_on_completed_threads_session_id():
    """The critical push path: completion must land in the management session."""
    obs, agent_mgr = _obs()
    await obs.on_completed("proj", "claude-code", summary="Found 4 products",
                           transcript_path="/x", session_id="sess_X")
    agent_mgr.inject_system_message.assert_awaited_once()
    assert agent_mgr.inject_system_message.await_args.kwargs.get("session_id") == "sess_X"


@pytest.mark.asyncio
async def test_on_error_threads_session_id():
    obs, agent_mgr = _obs()
    await obs.on_error("proj", "claude-code", error="boom",
                       transcript_path="/x", session_id="sess_X")
    agent_mgr.inject_system_message.assert_awaited_once()
    assert agent_mgr.inject_system_message.await_args.kwargs.get("session_id") == "sess_X"


# --- Backward compat: omitting session_id falls back to default behaviour ---

@pytest.mark.asyncio
async def test_tool_without_session_id_omits_kwarg():
    """A tool constructed without session_id passes None — sub_agent_manager
    resolves it to DEFAULT_SESSION_ID internally (back-compat preserved)."""
    tool, mgr = _tool(session_id=None)
    await tool.execute(action="send", agent="claude-code", message="go")
    mgr.send.assert_awaited_once()
    assert mgr.send.await_args.kwargs.get("session_id") is None


@pytest.mark.asyncio
async def test_observer_without_session_id_omits_kwarg():
    obs, agent_mgr = _obs()
    await obs.on_completed("proj", "claude-code", summary="ok", transcript_path="/x")
    agent_mgr.inject_system_message.assert_awaited_once()
    assert agent_mgr.inject_system_message.await_args.kwargs.get("session_id") is None
