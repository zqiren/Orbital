# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Regression (guard against over-rescue): a GENUINE verdict-less stall — a
text exit with NO pending continuation (``_loop_exit_path != "yield_turn"``) —
still terminates as a contract violation.

This proves the session-level fix narrows the violation path by
continuation-presence, not disables it: a turn that ends with no verdict and
no pending continuation gets one corrective turn and then BLOCKs. Passes
before AND after the fix.
"""

import pytest
from unittest.mock import MagicMock, AsyncMock

from agent_os.queue.dispatcher import QueueDispatcher
from agent_os.queue.models import AttemptOutcome, ItemState


def _dispatcher_stall():
    """A QueueDispatcher (via ``__new__``) wired to drive ``_await_and_handle``
    into a verdict-less stall (``text_complete``) with no pending continuation:
    corrective turn on the first text exit, BLOCKED on the second."""
    d = QueueDispatcher.__new__(QueueDispatcher)

    loop_obj = MagicMock()
    loop_obj.get_completion_state = MagicMock(return_value=("text", None, None))
    loop_obj._loop_exit_path = "text_complete"
    loop_obj._exit_reason = "text"

    sub_mgr = MagicMock()
    # No running sub-agent — and for a non-yield exit path this should never
    # even be consulted.
    sub_mgr.list_active = MagicMock(return_value=[])

    agent_manager = MagicMock()
    agent_manager.get_loop = MagicMock(return_value=loop_obj)
    agent_manager.get_sub_agent_manager = MagicMock(return_value=sub_mgr)
    agent_manager.inject_system_message = AsyncMock(return_value="delivered")

    d._agent_manager = agent_manager
    d._project_id = "proj_q"
    d._stop_generation = 0
    d._shutting_down = False
    d._max_runtime_seconds = 999
    d._wait_for_loop_done = AsyncMock(return_value=None)

    store = MagicMock()
    store.auto_idle_if_empty.return_value = False
    d._store = store

    d._idle_event = MagicMock()
    d._broadcast_advance = MagicMock()
    d._broadcast_state_changed = MagicMock()
    d._log_attempt_close = MagicMock()
    return d, agent_manager, store


@pytest.mark.asyncio
async def test_text_stall_without_continuation_still_blocks():
    d, agent_manager, store = _dispatcher_stall()
    item = MagicMock()
    item.id = "item1"

    await d._await_and_handle(item, "proj_q_sess")

    # One corrective turn fired (the existing CHANGE-2 behavior).
    agent_manager.inject_system_message.assert_awaited_once()

    # Then the second text exit closes the attempt BLOCKED + advances the queue.
    blocked_closes = [
        c for c in store.close_latest_attempt.call_args_list
        if c.kwargs.get("outcome") == AttemptOutcome.BLOCKED
    ]
    assert blocked_closes, "stall must close the attempt BLOCKED (contract violation)"
    store.set_item_state.assert_any_call("item1", ItemState.BLOCKED)
    d._broadcast_advance.assert_any_call("item1", "blocked")
