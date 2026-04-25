# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Regression tests for TASK-cancel-arch-03: watchdog stops sub-agents before
broadcasting idle.

Previously, when _check_sub_agents_done exceeded _MAX_IDLE_POLLS, it would
broadcast agent.status: idle WITHOUT stopping the still-running sub-agents.
Orphan claude CLI processes accumulated in the background.

Fix: before broadcasting idle, call
    await asyncio.wait_for(self._sub_agent_manager.stop_all(project_id), timeout=10.0)
with graceful handling of TimeoutError and other exceptions.
"""

from __future__ import annotations

import asyncio
import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent_os.daemon_v2.agent_manager import AgentManager, ProjectHandle


# ------------------------------------------------------------------ #
# Helpers / factories
# ------------------------------------------------------------------ #

PROJECT_ID = "proj-watchdog-test"


def _make_handle(task_done: bool = True) -> ProjectHandle:
    """Build a minimal ProjectHandle with a stopped-looking session."""
    mock_session = MagicMock()
    mock_session.is_stopped.return_value = False  # session still active

    mock_task = MagicMock(spec=asyncio.Task)
    mock_task.done.return_value = task_done  # task is done → no new loop running

    return ProjectHandle(
        session=mock_session,
        loop=MagicMock(),
        provider=MagicMock(),
        registry=MagicMock(),
        context_manager=MagicMock(),
        interceptor=MagicMock(),
        task=mock_task,
    )


def _make_manager(sub_agent_status: str = "running") -> tuple[AgentManager, MagicMock, MagicMock]:
    """Construct a minimal AgentManager for watchdog testing.

    Returns (manager, mock_ws, mock_sub_agent_manager).
    """
    mock_ws = MagicMock()
    mock_ws.broadcast = MagicMock()

    mock_sam = MagicMock()
    mock_sam.stop_all = AsyncMock()

    # list_active returns one agent (busy or idle depending on argument)
    mock_sam.list_active = MagicMock(
        return_value=[{"handle": "h1", "display_name": "helper", "status": sub_agent_status}]
    )

    manager = AgentManager(
        project_store=MagicMock(),
        ws_manager=mock_ws,
        sub_agent_manager=mock_sam,
        activity_translator=MagicMock(),
        process_manager=MagicMock(),
    )
    # Populate _handles so the watchdog doesn't short-circuit on "handle is None"
    manager._handles[PROJECT_ID] = _make_handle(task_done=True)

    return manager, mock_ws, mock_sam


async def _run_watchdog_fast(manager: AgentManager, project_id: str) -> None:
    """Run _check_sub_agents_done with _MAX_IDLE_POLLS patched to 1 and
    asyncio.sleep patched to a no-op so the test finishes instantly."""
    with (
        patch.object(type(manager), "_MAX_IDLE_POLLS", new=1),
        patch("agent_os.daemon_v2.agent_manager.asyncio.sleep", new=AsyncMock()),
    ):
        await manager._check_sub_agents_done(project_id)


# ------------------------------------------------------------------ #
# Test 1: watchdog calls stop_all on timeout
# ------------------------------------------------------------------ #


@pytest.mark.asyncio
async def test_watchdog_calls_stop_all_on_timeout():
    """When sub-agents are never idle, the watchdog must call stop_all exactly
    once (with the correct project_id) BEFORE broadcasting idle.

    This test FAILS on pre-fix code because stop_all is never called.
    """
    manager, mock_ws, mock_sam = _make_manager(sub_agent_status="running")

    # Track call order: stop_all before broadcast
    call_order: list[str] = []

    async def _recording_stop_all(pid: str) -> None:
        call_order.append("stop_all")

    mock_sam.stop_all.side_effect = _recording_stop_all
    mock_ws.broadcast.side_effect = lambda *_a, **_kw: call_order.append("broadcast")

    await _run_watchdog_fast(manager, PROJECT_ID)

    # stop_all must have been called exactly once with the right project_id
    mock_sam.stop_all.assert_called_once_with(PROJECT_ID)

    # broadcast must have fired
    mock_ws.broadcast.assert_called_once()
    broadcast_payload = mock_ws.broadcast.call_args[0][1]
    assert broadcast_payload["status"] == "idle"

    # stop_all must precede broadcast
    assert call_order == ["stop_all", "broadcast"], (
        f"Expected stop_all before broadcast, got: {call_order}"
    )


# ------------------------------------------------------------------ #
# Test 2: no stop_all when sub-agents are already idle
# ------------------------------------------------------------------ #


@pytest.mark.asyncio
async def test_watchdog_no_stop_call_when_subagents_idle():
    """When sub-agents report idle on first poll, the watchdog broadcasts idle
    immediately WITHOUT calling stop_all.

    Verifies the happy path is unaffected by the fix.
    """
    manager, mock_ws, mock_sam = _make_manager(sub_agent_status="idle")

    await _run_watchdog_fast(manager, PROJECT_ID)

    # stop_all must NOT have been called
    mock_sam.stop_all.assert_not_called()

    # broadcast must have fired
    mock_ws.broadcast.assert_called_once()
    broadcast_payload = mock_ws.broadcast.call_args[0][1]
    assert broadcast_payload["status"] == "idle"


# ------------------------------------------------------------------ #
# Test 3: watchdog proceeds when stop_all hangs (TimeoutError)
# ------------------------------------------------------------------ #


@pytest.mark.asyncio
async def test_watchdog_proceeds_on_stop_all_timeout(caplog):
    """When stop_all hangs and exceeds the 10s budget, the watchdog must still
    broadcast idle and log an ERROR.

    Fails on pre-fix code: stop_all is never called, so the asyncio.TimeoutError
    path cannot be exercised.
    """
    manager, mock_ws, mock_sam = _make_manager(sub_agent_status="running")

    # Patch wait_for to immediately raise TimeoutError (simulates 10s elapsed)
    async def _immediate_timeout(coro, timeout):  # noqa: ARG001
        coro.close()  # clean up the coroutine object
        raise asyncio.TimeoutError

    # We patch the module-level asyncio.wait_for rather than the specific call
    # because there's no clean way to target one call site. Today the watchdog
    # block is the only wait_for in _check_sub_agents_done; a future maintainer
    # adding another wait_for here would need to revisit this stub.
    with caplog.at_level(logging.ERROR, logger="agent_os.daemon_v2.agent_manager"):
        with patch("agent_os.daemon_v2.agent_manager.asyncio.wait_for", new=_immediate_timeout):
            await _run_watchdog_fast(manager, PROJECT_ID)

    # idle broadcast must still fire
    mock_ws.broadcast.assert_called_once()
    assert mock_ws.broadcast.call_args[0][1]["status"] == "idle"

    # ERROR log must have been emitted
    error_records = [r for r in caplog.records if r.levelno >= logging.ERROR]
    assert any(
        "timed out" in r.message.lower() or "time" in r.message.lower()
        for r in error_records
    ), f"Expected ERROR log about timeout, got: {[r.message for r in error_records]}"


# ------------------------------------------------------------------ #
# Test 4: watchdog proceeds when stop_all raises an exception
# ------------------------------------------------------------------ #


@pytest.mark.asyncio
async def test_watchdog_proceeds_on_stop_all_exception(caplog):
    """When stop_all raises a RuntimeError, the watchdog must still broadcast
    idle and log the exception.

    Fails on pre-fix code: stop_all is never called, so no exception path runs.
    """
    manager, mock_ws, mock_sam = _make_manager(sub_agent_status="running")

    mock_sam.stop_all.side_effect = RuntimeError("simulated wedge")

    with caplog.at_level(logging.ERROR, logger="agent_os.daemon_v2.agent_manager"):
        await _run_watchdog_fast(manager, PROJECT_ID)

    # idle broadcast must still fire
    mock_ws.broadcast.assert_called_once()
    assert mock_ws.broadcast.call_args[0][1]["status"] == "idle"

    # Exception must have been logged (logger.exception emits at ERROR level)
    error_records = [r for r in caplog.records if r.levelno >= logging.ERROR]
    assert error_records, "Expected an ERROR/EXCEPTION log, got none"
