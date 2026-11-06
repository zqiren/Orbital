# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Spec 161 §5.9: an automation's item closes on the seat-holder's
follow-up, not on the turn that only dispatched.

A queue item assigned to a worker settles on that worker's terminal event
(2026-09-05). When the worker handed the job to a child, its first terminal
is "I dispatched" — settling there would close the item before anything was
done, and NOT settling would wake the Orbital loop for a verdict (two seats).
So while a child is running the settlement is deferred and the wake stays
suppressed; the follow-up turn (delivered with ``initiator="queue_item"``)
settles the item with the real outcome, exactly once.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from agent_os.daemon_v2.lifecycle_observer import LifecycleObserver

P, S, PARENT, CHILD = "p1", "s1", "codex", "claude-code"


def _observer():
    am = MagicMock()
    am.inject_system_message = AsyncMock(return_value="suppressed")
    running = {"value": True}
    am.has_running_children = MagicMock(
        side_effect=lambda pid, sid, handle: running["value"] and handle == PARENT)
    obs = LifecycleObserver(am, MagicMock())
    obs.queue_terminal_hook = MagicMock(return_value=True)
    obs.worker_terminal_hook = MagicMock()
    obs.set_dispatch_initiator(P, PARENT, "queue_item", session_id=S)
    return obs, am, running


@pytest.mark.asyncio
async def test_dispatch_turn_with_a_running_child_neither_settles_nor_wakes():
    obs, am, _ = _observer()
    await obs.on_completed(P, PARENT, "Dispatched to claude-code.", "t.jsonl", session_id=S)
    obs.queue_terminal_hook.assert_not_called()
    assert am.inject_system_message.call_args.kwargs["meta"]["suppress_wake"] is True


@pytest.mark.asyncio
async def test_follow_up_settles_the_item_once():
    obs, am, running = _observer()
    await obs.on_completed(P, PARENT, "Dispatched to claude-code.", "t.jsonl", session_id=S)
    # The child finishes; the delivery re-dispatches the parent as queue_item.
    running["value"] = False
    obs.set_dispatch_initiator(P, PARENT, "queue_item", session_id=S)
    await obs.on_completed(P, PARENT, "Done: report written.", "t.jsonl", session_id=S)
    obs.queue_terminal_hook.assert_called_once()
    assert obs.queue_terminal_hook.call_args.kwargs["kind"] == "completed"
    assert obs.queue_terminal_hook.call_args.kwargs["summary"] == "Done: report written."
    assert am.inject_system_message.call_args.kwargs["meta"]["suppress_wake"] is True


@pytest.mark.asyncio
async def test_an_error_while_a_child_runs_is_deferred_too():
    obs, _, _ = _observer()
    await obs.on_error(P, PARENT, "rate limited", "t.jsonl", session_id=S)
    obs.queue_terminal_hook.assert_not_called()


@pytest.mark.asyncio
async def test_without_children_an_assigned_item_settles_as_before():
    obs, _, running = _observer()
    running["value"] = False
    await obs.on_completed(P, PARENT, "Done.", "t.jsonl", session_id=S)
    obs.queue_terminal_hook.assert_called_once()
