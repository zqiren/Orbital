# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""User messages sent during the "waiting" state are queued, not loop-starting.

After the management agent dispatches a sub-agent and yields, it is in the
"waiting" state (idle-poll alive, sub-agent working). A user message then must
be appended to the session WITHOUT starting a loop — `on_completed` owns the
restart and will surface the message when it resumes. Without a live idle-poll
(not waiting), the existing hot-resume behaviour is preserved.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent_os.daemon_v2.agent_manager import AgentManager


def _manager() -> AgentManager:
    mgr = AgentManager(
        project_store=MagicMock(), ws_manager=MagicMock(),
        sub_agent_manager=MagicMock(), activity_translator=MagicMock(),
        process_manager=MagicMock(),
    )
    mgr._ws.broadcast = MagicMock()
    mgr._start_loop = AsyncMock()
    return mgr


def _idle_handle() -> MagicMock:
    h = MagicMock()
    h.session.is_stopped.return_value = False
    h.session._paused_for_approval = False
    h.session.pop_queued_messages.return_value = []
    h.interceptor = None
    h.task = MagicMock()
    h.task.done.return_value = True  # management loop finished (idle)
    return h


@pytest.mark.asyncio
async def test_user_message_queued_during_waiting():
    """idle-poll alive → message appended, NO loop started, status 'queued'."""
    mgr = _manager()
    sk = ("proj", "sess-qdw-1")  # explicit sid: "default" retired in 433912a (seam 3 / D1)
    handle = _idle_handle()
    mgr._handles[sk] = handle

    async def _never_done():
        await asyncio.sleep(9999)
    poll = asyncio.ensure_future(_never_done())
    mgr._idle_poll_tasks[sk] = poll  # waiting state (SessionKey-keyed, agent_manager.py:108)
    try:
        result = await mgr.inject_message("proj", "also check pricing", session_id="sess-qdw-1")

        # Message appended to the session (the agent sees it on resume).
        assert handle.session.append.called
        appended = handle.session.append.call_args[0][0]
        assert appended["role"] == "user" and appended["content"] == "also check pricing"
        # No loop started — on_completed owns the restart.
        mgr._start_loop.assert_not_awaited()
        # Status indicates the message was queued.
        status = result.get("status") if isinstance(result, dict) else result
        assert status == "queued"
    finally:
        poll.cancel()
        try:
            await poll
        except asyncio.CancelledError:
            pass


@pytest.mark.asyncio
async def test_user_message_starts_loop_when_not_waiting():
    """No idle-poll → existing hot-resume behaviour (loop started)."""
    mgr = _manager()
    sk = ("proj", "sess-qdw-1")  # explicit sid: "default" retired in 433912a (seam 3 / D1)
    handle = _idle_handle()
    mgr._handles[sk] = handle
    # No _idle_poll_tasks entry → not waiting.

    result = await mgr.inject_message("proj", "next task", session_id="sess-qdw-1")

    mgr._start_loop.assert_awaited_once()
    status = result.get("status") if isinstance(result, dict) else result
    assert status == "delivered"
