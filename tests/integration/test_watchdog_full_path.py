# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Integration test for TASK-cancel-arch-03: watchdog full-path verification.

Tests the AgentManager watchdog wired to a realistic SubAgentManager stub that
simulates a wedged sub-agent (one that ignores stop signals). Verifies end-to-end
that:
  - idle broadcast is received after the watchdog fires
  - stop_all is invoked
  - No orphan adapter remains after stop_all completes

Uses the inline construction pattern (no subprocess/live daemon required) so this
test runs in CI without additional setup. The pattern follows test_global_settings_api.py.
"""

from __future__ import annotations

import asyncio
import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent_os.daemon_v2.agent_manager import AgentManager, ProjectHandle
from agent_os.daemon_v2.sub_agent_manager import SubAgentManager


# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #

PROJECT_ID = "integ-proj-watchdog"


def _make_handle(task_done: bool = True) -> ProjectHandle:
    """Build a minimal ProjectHandle with a still-active session."""
    mock_session = MagicMock()
    mock_session.is_stopped.return_value = False

    mock_task = MagicMock(spec=asyncio.Task)
    mock_task.done.return_value = task_done

    return ProjectHandle(
        session=mock_session,
        loop=MagicMock(),
        provider=MagicMock(),
        registry=MagicMock(),
        context_manager=MagicMock(),
        interceptor=MagicMock(),
        task=mock_task,
    )


class _WedgedSubAgentManager:
    """Minimal SubAgentManager stand-in where sub-agents never go idle.

    Tracks stop_all calls. The internal _adapters dict is maintained so
    we can assert no orphan adapters remain after stop_all.
    """

    def __init__(self):
        self.stop_all_calls: list[str] = []
        # Simulate one adapter registered for the project
        self._adapters: dict[str, dict] = {
            PROJECT_ID: {"handle-001": MagicMock()}
        }

    def list_active(self, project_id: str) -> list[dict]:
        """Always report a running sub-agent (wedged — never goes idle)."""
        adapters = self._adapters.get(project_id, {})
        return [
            {"handle": h, "display_name": "wedged-helper", "status": "running"}
            for h in adapters
        ]

    async def stop_all(self, project_id: str) -> None:
        """Record the call and remove the adapters (simulates a successful stop)."""
        self.stop_all_calls.append(project_id)
        self._adapters.pop(project_id, None)


# ------------------------------------------------------------------ #
# Integration test
# ------------------------------------------------------------------ #


@pytest.mark.asyncio
async def test_watchdog_full_path_stops_wedged_subagents():
    """End-to-end: AgentManager watchdog fires, calls stop_all on a wedged
    SubAgentManager, and broadcasts idle.

    Scenario:
      - One sub-agent registered and permanently 'running' (wedged)
      - _MAX_IDLE_POLLS patched to 1 (instant timeout)
      - asyncio.sleep patched to no-op

    Asserts:
      - agent.status: idle broadcast received
      - stop_all called with correct project_id
      - No orphan adapter remaining in sub_agent_manager._adapters
    """
    broadcast_messages: list[dict] = []

    mock_ws = MagicMock()
    mock_ws.broadcast.side_effect = lambda _pid, msg: broadcast_messages.append(msg)

    wedged_sam = _WedgedSubAgentManager()

    manager = AgentManager(
        project_store=MagicMock(),
        ws_manager=mock_ws,
        sub_agent_manager=wedged_sam,
        activity_translator=MagicMock(),
        process_manager=MagicMock(),
    )

    # Register a project handle so the watchdog doesn't short-circuit
    manager._handles[PROJECT_ID] = _make_handle(task_done=True)

    # Run the watchdog with patched poll count and sleep
    with (
        patch.object(type(manager), "_MAX_IDLE_POLLS", new=1),
        patch("agent_os.daemon_v2.agent_manager.asyncio.sleep", new=AsyncMock()),
    ):
        await manager._check_sub_agents_done(PROJECT_ID)

    # 1. idle broadcast must have been sent
    idle_broadcasts = [m for m in broadcast_messages if m.get("status") == "idle"]
    assert len(idle_broadcasts) == 1, (
        f"Expected exactly 1 idle broadcast, got {len(idle_broadcasts)}: {broadcast_messages}"
    )

    # 2. stop_all must have been invoked with the correct project_id
    assert wedged_sam.stop_all_calls == [PROJECT_ID], (
        f"stop_all not called correctly: {wedged_sam.stop_all_calls}"
    )

    # 3. No orphan adapters remaining
    remaining = wedged_sam._adapters.get(PROJECT_ID, {})
    assert not remaining, f"Orphan adapters still present: {remaining}"


@pytest.mark.asyncio
async def test_watchdog_full_path_no_stop_when_subagents_idle():
    """When sub-agents are already idle on first poll, stop_all is NOT called
    and the idle broadcast fires via the normal happy path."""

    broadcast_messages: list[dict] = []

    mock_ws = MagicMock()
    mock_ws.broadcast.side_effect = lambda _pid, msg: broadcast_messages.append(msg)

    mock_sam = MagicMock()
    mock_sam.stop_all = AsyncMock()
    mock_sam.list_active.return_value = [
        {"handle": "h1", "status": "idle"}
    ]

    manager = AgentManager(
        project_store=MagicMock(),
        ws_manager=mock_ws,
        sub_agent_manager=mock_sam,
        activity_translator=MagicMock(),
        process_manager=MagicMock(),
    )
    manager._handles[PROJECT_ID] = _make_handle(task_done=True)

    with (
        patch.object(type(manager), "_MAX_IDLE_POLLS", new=1),
        patch("agent_os.daemon_v2.agent_manager.asyncio.sleep", new=AsyncMock()),
    ):
        await manager._check_sub_agents_done(PROJECT_ID)

    # stop_all NOT called
    mock_sam.stop_all.assert_not_called()

    # idle broadcast fired
    idle_broadcasts = [m for m in broadcast_messages if m.get("status") == "idle"]
    assert len(idle_broadcasts) == 1
