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


def _register_adapter(mgr, project_id, handle, adapter):
    """Directly inject an adapter into SubAgentManager for testing."""
    if project_id not in mgr._adapters:
        mgr._adapters[project_id] = {}
    mgr._adapters[project_id][handle] = adapter


# ---------------------------------------------------------------------------
# Depth limit tests
# ---------------------------------------------------------------------------

class TestDepthLimit:
    """Sub-agent delegation chains deeper than MAX_DEPTH are blocked."""

    @pytest.mark.asyncio
    async def test_max_depth_blocks_spawn(self):
        """An agent at depth=MAX_DEPTH cannot start sub-agents."""
        tool, mgr = _make_tool(depth=MAX_DEPTH)
        result = await tool.execute(action="start", agent="child-agent")
        assert "depth limit" in result.content.lower()
        assert str(MAX_DEPTH) in result.content
        mgr.start.assert_not_called()

    @pytest.mark.asyncio
    async def test_depth_counter_increments(self):
        """An agent at depth=1 passes depth=2 to SubAgentManager.start()."""
        tool, mgr = _make_tool(depth=1)
        await tool.execute(action="start", agent="child-agent")
        mgr.start.assert_called_once()
        call_kwargs = mgr.start.call_args
        # depth=2 should be passed through
        assert call_kwargs[1].get("depth") == 2 or (
            len(call_kwargs[0]) >= 3 and call_kwargs[0][2] == 2
        )

    @pytest.mark.asyncio
    async def test_depth_zero_allows_spawn(self):
        """Primary agent (depth=0) can start sub-agents normally."""
        tool, mgr = _make_tool(depth=0)
        result = await tool.execute(action="start", agent="child-agent")
        assert "Error" not in result.content or "depth" not in result.content.lower()
        mgr.start.assert_called_once()

    @pytest.mark.asyncio
    async def test_depth_just_below_limit_allows_spawn(self):
        """Agent at depth=MAX_DEPTH-1 can still start sub-agents."""
        tool, mgr = _make_tool(depth=MAX_DEPTH - 1)
        result = await tool.execute(action="start", agent="child-agent")
        assert "depth limit" not in result.content.lower()
        mgr.start.assert_called_once()

    @pytest.mark.asyncio
    async def test_depth_does_not_affect_send(self):
        """Depth limit only applies to start, not send."""
        tool, mgr = _make_tool(depth=MAX_DEPTH)
        result = await tool.execute(action="send", agent="a", message="hello")
        assert "depth" not in result.content.lower()
        mgr.send.assert_called_once()

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
        result = await mgr.start("proj-1", "agent-extra")
        assert "limit" in result.lower() or "concurrent" in result.lower()
        # Should not have been added
        assert "agent-extra" not in mgr._adapters.get("proj-1", {})

    @pytest.mark.asyncio
    async def test_breadth_freed_on_completion(self):
        """After stopping one sub-agent, a new one can be started."""
        mgr = _make_sub_agent_manager()
        # Register MAX_CONCURRENT_SUBAGENTS adapters
        for i in range(MAX_CONCURRENT_SUBAGENTS):
            _register_adapter(mgr, "proj-1", f"agent-{i}", _make_mock_adapter())

        # Stop one
        await mgr.stop("proj-1", "agent-0")
        # Now count should be MAX-1, new start should pass breadth check
        count = len(mgr._adapters.get("proj-1", {}))
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
        result = await mgr.start("proj-1", "agent-new")
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
        result = await mgr.start("proj-2", "agent-a")
        assert "concurrent" not in result.lower()
