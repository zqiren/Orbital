# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Idle project handles are evicted after EVICTION_IDLE_TIMEOUT.

Under the multi-session model there is no reason to keep idle runtime
resources warm — the measured cold-restart cost is ~140ms, invisible behind
the LLM round-trip (see TASK/REPORT-restart-costs.md). The eviction sweep
removes the handle (closing browser pages, killing sub-agents, releasing the
sandbox) once a session has been idle long enough; the next message cold-starts
from the JSONL on disk.

The sweep MUST NOT evict a session that is running, paused for approval, or
waiting on sub-agents — those states carry in-memory state that a cold restart
would lose.
"""

from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent_os.daemon_v2.agent_manager import AgentManager


def _manager() -> AgentManager:
    return AgentManager(
        project_store=MagicMock(),
        ws_manager=MagicMock(),
        sub_agent_manager=MagicMock(),
        activity_translator=MagicMock(),
        process_manager=MagicMock(),
    )


def _handle(last_activity: float) -> MagicMock:
    """A handle whose runtime state reads as 'idle' via the real get_run_status."""
    h = MagicMock()
    h.session.is_stopped.return_value = False
    h.session._paused_for_approval = False
    h.task = MagicMock()
    h.task.done.return_value = True  # loop finished → idle
    h.last_activity = last_activity
    return h


@pytest.mark.asyncio
async def test_idle_handle_evicted_after_timeout():
    """A handle idle longer than EVICTION_IDLE_TIMEOUT is evicted via stop_agent."""
    mgr = _manager()
    sk = ("proj_idle", "default")
    mgr._handles[sk] = _handle(time.time() - 9999)

    async def _pop(pid, *, session_id=None):
        mgr._handles.pop((pid, session_id), None)

    mgr.stop_agent = AsyncMock(side_effect=_pop)

    evicted = await mgr._evict_idle_once()

    assert sk in evicted
    mgr.stop_agent.assert_awaited_once_with("proj_idle", session_id="default")
    assert sk not in mgr._handles


@pytest.mark.asyncio
async def test_recently_active_handle_not_evicted():
    """A handle whose last_activity is recent is left alone."""
    mgr = _manager()
    sk = ("proj_fresh", "default")
    mgr._handles[sk] = _handle(time.time())
    mgr.stop_agent = AsyncMock()

    evicted = await mgr._evict_idle_once()

    assert sk not in evicted
    mgr.stop_agent.assert_not_awaited()
    assert sk in mgr._handles


@pytest.mark.asyncio
async def test_running_handle_not_evicted_regardless_of_idle_time():
    """A long-idle handle that is still running is never evicted."""
    mgr = _manager()
    sk = ("proj_run", "default")
    h = _handle(time.time() - 9999)
    h.task.done.return_value = False  # loop active → running
    mgr._handles[sk] = h
    mgr.stop_agent = AsyncMock()

    evicted = await mgr._evict_idle_once()

    assert sk not in evicted
    mgr.stop_agent.assert_not_awaited()
    assert sk in mgr._handles


@pytest.mark.asyncio
async def test_pending_approval_handle_not_evicted():
    """A long-idle handle paused for approval is never evicted — the approval
    state is in-memory only and a cold restart would lose it."""
    mgr = _manager()
    sk = ("proj_appr", "default")
    h = _handle(time.time() - 9999)
    h.session._paused_for_approval = True  # → pending_approval
    mgr._handles[sk] = h
    mgr.stop_agent = AsyncMock()

    evicted = await mgr._evict_idle_once()

    assert sk not in evicted
    mgr.stop_agent.assert_not_awaited()
    assert sk in mgr._handles


@pytest.mark.asyncio
async def test_eviction_failure_does_not_abort_sweep():
    """If stop_agent raises for one project, the sweep still evicts the others."""
    mgr = _manager()
    sk_bad = ("proj_bad", "default")
    sk_good = ("proj_good", "default")
    mgr._handles[sk_bad] = _handle(time.time() - 9999)
    mgr._handles[sk_good] = _handle(time.time() - 9999)

    async def _maybe_fail(pid, *, session_id=None):
        if pid == "proj_bad":
            raise RuntimeError("teardown blew up")
        mgr._handles.pop((pid, session_id), None)

    mgr.stop_agent = AsyncMock(side_effect=_maybe_fail)

    # Should not raise.
    await mgr._evict_idle_once()

    assert sk_good not in mgr._handles
    assert mgr.stop_agent.await_count == 2


@pytest.mark.asyncio
async def test_eviction_sweep_loop_fires_and_evicts():
    """The background sweep started by start_eviction() actually evicts an
    idle handle (covers the _eviction_sweep loop + start_eviction wiring, not
    just the _evict_idle_once decision logic)."""
    mgr = _manager()
    # Squeeze the policy intervals for a fast test — instance overrides of the
    # class-level defaults; the production values (180s / 60s) are unchanged.
    mgr.EVICTION_IDLE_TIMEOUT = 0.01
    mgr.EVICTION_SWEEP_INTERVAL = 0.02

    sk = ("proj_sweep", "default")
    mgr._handles[sk] = _handle(time.time() - 9999)

    async def _pop(pid, *, session_id=None):
        mgr._handles.pop((pid, session_id), None)

    mgr.stop_agent = AsyncMock(side_effect=_pop)

    mgr.start_eviction()
    assert mgr._eviction_task is not None and not mgr._eviction_task.done()

    # Give the loop a few sweep intervals to fire.
    for _ in range(50):
        if sk not in mgr._handles:
            break
        await asyncio.sleep(0.02)

    assert sk not in mgr._handles, "sweep loop should have evicted the idle handle"
    mgr.stop_agent.assert_awaited_with("proj_sweep", session_id="default")

    # shutdown() must cancel the sweep task.
    mgr._sub_agent_manager.stop_all = AsyncMock()
    await mgr.shutdown(timeout=1.0)
    assert mgr._eviction_task is None


@pytest.mark.asyncio
async def test_start_eviction_is_idempotent():
    """Calling start_eviction twice does not spawn a second sweep task."""
    mgr = _manager()
    mgr.EVICTION_SWEEP_INTERVAL = 0.02
    mgr.start_eviction()
    first = mgr._eviction_task
    mgr.start_eviction()
    assert mgr._eviction_task is first
    first.cancel()
    try:
        await first
    except asyncio.CancelledError:
        pass
