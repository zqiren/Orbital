# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Regression (Root D / seam 3): the dispatcher's corrective-turn injection on
the rotation path must carry the item's session id.

When a queue item exits text-only, the dispatcher gives the agent ONE corrective
turn by calling ``inject_system_message``. ``_await_and_handle`` HAS the item's
concrete ``session_id`` in scope (it routes every other loop read by it), but the
corrective inject dropped it. ``inject_system_message`` is passthrough-None, so a
session-less call keys ``(pid, None)`` → handle-miss → returns ``"no_session"`` and
the corrective system message is SILENTLY DROPPED — the agent never gets its
corrective turn. Fix is caller-side: forward the in-scope ``session_id``. The
shared callee ``inject_system_message``'s None-policy is deliberately NOT changed.
"""

import pytest
from unittest.mock import MagicMock, AsyncMock

from agent_os.queue.dispatcher import QueueDispatcher


def _dispatcher_reaching_corrective(inject_mock):
    """A QueueDispatcher wired (via __new__, bypassing __init__) to drive
    _await_and_handle into the text-only corrective-turn branch once, then
    complete on the second loop iteration."""
    d = QueueDispatcher.__new__(QueueDispatcher)
    loop_obj = MagicMock()
    # iter1 → text-only (triggers corrective inject); iter2 → complete (returns).
    loop_obj.get_completion_state = MagicMock(
        side_effect=[("text", None, None), ("complete", "ok", None)],
    )
    agent_manager = MagicMock()
    agent_manager.get_loop = MagicMock(return_value=loop_obj)
    agent_manager.inject_system_message = inject_mock
    d._agent_manager = agent_manager
    d._project_id = "proj_q"
    d._stop_generation = 0
    d._max_runtime_seconds = 999
    d._wait_for_loop_done = AsyncMock(return_value=None)
    store = MagicMock()
    store.auto_idle_if_empty.return_value = False
    d._store = store
    d._idle_event = MagicMock()
    d._broadcast_advance = MagicMock()
    d._broadcast_state_changed = MagicMock()
    d._log_attempt_close = MagicMock()
    return d


@pytest.mark.asyncio
async def test_corrective_turn_inject_carries_item_session_id():
    inject = AsyncMock(return_value="delivered")
    d = _dispatcher_reaching_corrective(inject)
    item = MagicMock()
    item.id = "item1"

    await d._await_and_handle(item, "proj_q_sess_FRESH")

    inject.assert_awaited_once()
    # The corrective system message must target the item's session, not None —
    # otherwise inject_system_message handle-misses and drops it ("no_session").
    assert inject.await_args.kwargs.get("session_id") == "proj_q_sess_FRESH"
