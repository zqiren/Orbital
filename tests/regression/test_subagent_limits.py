# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Regression tests for sub-agent depth and breadth limits.

Verifies that:
- Sub-agent delegation chains deeper than MAX_DEPTH (3) are blocked
- More than MAX_CONCURRENT_SUBAGENTS (5) per project are blocked
- Depth counter increments when spawning sub-agents
- Breadth slots are freed when a sub-agent is stopped
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent_os.agent.tools.agent_message import AgentMessageTool, MAX_DEPTH
from agent_os.daemon_v2.sub_agent_manager import SubAgentManager, MAX_CONCURRENT_SUBAGENTS


def _make_tool(depth=0, max_sends=10):
    """Create an AgentMessageTool with a mock SubAgentManager."""
    mgr = MagicMock()
    mgr.send = AsyncMock(return_value="sent")
    mgr.start = AsyncMock(return_value="started")
    mgr.stop = AsyncMock(return_value="stopped")
    mgr.list_active = MagicMock(return_value=[])
    mgr.status = MagicMock(return_value="running")
    tool = AgentMessageTool(
        sub_agent_manager=mgr,
        project_id="proj-1",
        max_sends_per_run=max_sends,
        depth=depth,
    )
    return tool, mgr


def _make_mock_adapter():
    """CLIAdapter-like mock."""
    adapter = MagicMock()
    adapter.is_alive = MagicMock(return_value=True)
    adapter.is_idle = MagicMock(return_value=False)
    adapter.stop = AsyncMock()
    adapter.start = AsyncMock()
    adapter._last_response = None
    return adapter


def _make_sub_agent_manager():
    """SubAgentManager with mock process_manager."""
    pm = MagicMock()
    pm.start = AsyncMock()
    pm.stop = AsyncMock()
    return SubAgentManager(process_manager=pm)


# Explicit parent session id: the "default" sentinel was retired in 433912a
# (seam 3 / D1); SubAgentManager hard-raises on session_id=None.
SID = "sess-limits-0001"


def _register_adapter(mgr, project_id, handle, adapter):
    """Directly inject an adapter into SubAgentManager for testing."""
    sk = (project_id, SID)
    if sk not in mgr._adapters:
        mgr._adapters[sk] = {}
    mgr._adapters[sk][handle] = adapter


# ---------------------------------------------------------------------------
# Depth limit tests
# ---------------------------------------------------------------------------

class TestDepthLimit:
    """Sub-agent delegation chains deeper than MAX_DEPTH are blocked."""

    @pytest.mark.asyncio
    async def test_max_depth_blocks_spawn(self):
        """An agent at depth=MAX_DEPTH cannot dispatch (send spawns-on-demand,
        so the depth gate now lives on send — TASK-collapse-dispatch-to-send)."""
        tool, mgr = _make_tool(depth=MAX_DEPTH)
        result = await tool.execute(action="send", agent="child-agent",
                                    message="go")
        assert "depth limit" in result.content.lower()
        assert str(MAX_DEPTH) in result.content
        mgr.send.assert_not_called()

    @pytest.mark.asyncio
    async def test_depth_counter_increments(self):
        """An agent at depth=1 passes depth=2 through send (the manager
        forwards it to its internal spawn-on-demand start)."""
        tool, mgr = _make_tool(depth=1)
        await tool.execute(action="send", agent="child-agent", message="go")
        mgr.send.assert_called_once()
        assert mgr.send.call_args.kwargs.get("depth") == 2

    @pytest.mark.asyncio
    async def test_depth_zero_allows_spawn(self):
        """Primary agent (depth=0) dispatches (and thereby spawns) normally."""
        tool, mgr = _make_tool(depth=0)
        result = await tool.execute(action="send", agent="child-agent",
                                    message="go")
        assert "depth limit" not in result.content.lower()
        mgr.send.assert_called_once()

    @pytest.mark.asyncio
    async def test_depth_just_below_limit_allows_spawn(self):
        """Agent at depth=MAX_DEPTH-1 can still dispatch/spawn."""
        tool, mgr = _make_tool(depth=MAX_DEPTH - 1)
        result = await tool.execute(action="send", agent="child-agent",
                                    message="go")
        assert "depth limit" not in result.content.lower()
        mgr.send.assert_called_once()

    @pytest.mark.asyncio
    async def test_depth_limit_gates_send(self):
        """NEW INVARIANT (TASK-collapse-dispatch-to-send): send spawns-on-
        demand, so the depth gate applies to send. An at-limit agent could
        never have a deeper sub-agent running, so gating every send is
        equivalent to the old start-only gate — without the spawn loophole."""
        tool, mgr = _make_tool(depth=MAX_DEPTH)
        result = await tool.execute(action="send", agent="a", message="hello")
        assert "depth limit" in result.content.lower()
        mgr.send.assert_not_called()

    @pytest.mark.asyncio
    async def test_depth_does_not_affect_stop(self):
        """Depth limit only applies to start, not stop."""
        tool, mgr = _make_tool(depth=MAX_DEPTH)
        result = await tool.execute(action="stop", agent="a")
        assert "depth" not in result.content.lower()
        mgr.stop.assert_called_once()


# ---------------------------------------------------------------------------
# Breadth limit tests
# ---------------------------------------------------------------------------

class TestBreadthLimit:
    """Concurrent sub-agent count per project is capped at MAX_CONCURRENT_SUBAGENTS."""

    @pytest.mark.asyncio
    async def test_max_breadth_blocks_spawn(self):
        """Starting a sub-agent when MAX_CONCURRENT_SUBAGENTS are active returns error."""
        mgr = _make_sub_agent_manager()
        # Register MAX_CONCURRENT_SUBAGENTS adapters
        for i in range(MAX_CONCURRENT_SUBAGENTS):
            _register_adapter(mgr, "proj-1", f"agent-{i}", _make_mock_adapter())

        # Attempt to start one more via the legacy path
        mgr._adapter_configs["agent-extra"] = MagicMock()
        result = await mgr.start("proj-1", "agent-extra", session_id=SID)
        assert "limit" in result.lower() or "concurrent" in result.lower()
        # Should not have been added
        assert "agent-extra" not in mgr._adapters.get(("proj-1", SID), {})

    @pytest.mark.asyncio
    async def test_breadth_freed_on_completion(self):
        """After stopping one sub-agent, a new one can be started."""
        mgr = _make_sub_agent_manager()
        # Register MAX_CONCURRENT_SUBAGENTS adapters
        for i in range(MAX_CONCURRENT_SUBAGENTS):
            _register_adapter(mgr, "proj-1", f"agent-{i}", _make_mock_adapter())

        # Stop one
        await mgr.stop("proj-1", "agent-0", session_id=SID)
        # Now count should be MAX-1, new start should pass breadth check
        count = len(mgr._adapters.get(("proj-1", SID), {}))
        assert count == MAX_CONCURRENT_SUBAGENTS - 1

    @pytest.mark.asyncio
    async def test_breadth_under_limit_allows_spawn(self):
        """Starting a sub-agent when under the limit succeeds (breadth check passes)."""
        mgr = _make_sub_agent_manager()
        # Register fewer than limit
        for i in range(MAX_CONCURRENT_SUBAGENTS - 1):
            _register_adapter(mgr, "proj-1", f"agent-{i}", _make_mock_adapter())

        # Attempt to start one more — should pass breadth check
        # (may fail for other reasons like missing config, but not breadth)
        mgr._adapter_configs["agent-new"] = MagicMock()
        result = await mgr.start("proj-1", "agent-new", session_id=SID)
        assert "concurrent" not in result.lower()

    @pytest.mark.asyncio
    async def test_breadth_is_per_project(self):
        """Breadth limit is per-project, not global."""
        mgr = _make_sub_agent_manager()
        # Fill project-1 to the limit
        for i in range(MAX_CONCURRENT_SUBAGENTS):
            _register_adapter(mgr, "proj-1", f"agent-{i}", _make_mock_adapter())

        # project-2 should still allow starts (breadth check passes)
        mgr._adapter_configs["agent-a"] = MagicMock()
        result = await mgr.start("proj-2", "agent-a", session_id=SID)
        assert "concurrent" not in result.lower()
