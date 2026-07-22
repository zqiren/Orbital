# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Regression tests for backlog #23 D1: deferred terminal sub-agent events
must wake the management loop.

A terminal lifecycle event (sub-agent error/failed/stopped) injected while
the management turn is in flight is deferred (``Session.defer_message``) for
safe insertion after the current tool batch. ``_on_loop_done`` already drains
the deferred buffer when the turn ends, but it only *appends* the messages —
it never restarts the loop, unlike the sibling queued-user-message path a few
lines below it. A fast mid-turn sub-agent failure therefore sits silently in
session history until the user happens to send another message; the
management agent never wakes to process it (backlog #23, diagnosed live via
a codex 400 the user never saw addressed).

Fix: when the drained deferred batch includes a terminal sub-agent event of
kind error/failed/stopped, ``_on_loop_done`` triggers the same hot-resume
path used for queued user messages.
"""

from unittest.mock import MagicMock, patch

import pytest

from agent_os.daemon_v2.agent_manager import AgentManager
from agent_os.daemon_v2.lifecycle_observer import LifecycleObserver


# Explicit session id: the "default" sentinel is retired (seam 3 / D1); bare
# project-level calls no longer resolve to a planted handle.
SID = "sess-wake-deferred-0001"


@pytest.fixture
def manager():
    ws = MagicMock()
    ws.broadcast = MagicMock()
    project_store = MagicMock()
    sub_agent_manager = MagicMock()
    sub_agent_manager.list_active = MagicMock(return_value=[])
    activity_translator = MagicMock()
    process_manager = MagicMock()
    mgr = AgentManager(
        project_store=project_store,
        ws_manager=ws,
        sub_agent_manager=sub_agent_manager,
        activity_translator=activity_translator,
        process_manager=process_manager,
    )
    return mgr


def _handle_with_deferred(manager, deferred_messages, *, queued=None):
    session = MagicMock()
    session.is_stopped.return_value = False
    session._paused_for_approval = False
    session.pop_deferred_messages.return_value = deferred_messages
    session.pop_queued_messages.return_value = queued or []
    session.append = MagicMock()

    handle = MagicMock()
    handle.session = session
    handle.loop = MagicMock(last_llm_error=None)

    task = MagicMock()
    task.exception.return_value = None
    handle.task = task

    manager._handles[("proj_test", SID)] = handle
    return handle, task


def _terminal_marker(content: str, kind: str) -> dict:
    """Shape a deferred message the way LifecycleObserver.on_error/on_failed/
    on_user_stopped actually stamp it (``_meta.event`` + ``_meta.kind``)."""
    return {
        "role": "system",
        "content": content,
        "source": "daemon",
        "timestamp": "2026-07-22T00:00:00+00:00",
        "_meta": {"event": "sub_agent_terminal", "kind": kind},
    }


def _run_callback(manager, task):
    callback = manager._on_loop_done("proj_test", session_id=SID)
    mock_future = MagicMock()
    with patch("asyncio.ensure_future", return_value=mock_future) as mock_ensure:
        callback(task)
        if mock_ensure.call_args:
            coro = mock_ensure.call_args[0][0]
            coro.close()
    return mock_ensure


@pytest.mark.parametrize("kind", ["error", "failed", "stopped"])
def test_deferred_negative_terminal_wakes_the_loop(manager, kind):
    """error/failed/stopped deferred mid-turn must trigger the same
    hot-resume path as a queued user message."""
    handle, task = _handle_with_deferred(
        manager, [_terminal_marker(f"[Sub-agent] cursor {kind}", kind)])

    mock_ensure = _run_callback(manager, task)

    # The event content reaches the model: appended to session history...
    handle.session.append.assert_called_once()
    appended = handle.session.append.call_args[0][0]
    assert appended["content"] == f"[Sub-agent] cursor {kind}"
    # ...and the loop is woken (hot-resume), not left idle.
    mock_ensure.assert_called_once()


def test_deferred_completed_event_does_not_wake_by_itself(manager):
    """Scope guard: only the negative terminals (error/failed/stopped) wake
    the loop per the backlog row — a plain completed marker must not, even
    though the same gap plausibly exists there (noted, not fixed)."""
    handle, task = _handle_with_deferred(
        manager, [_terminal_marker("[Sub-agent] cursor completed. Summary: ok",
                                   "completed")])

    mock_ensure = _run_callback(manager, task)

    handle.session.append.assert_called_once()
    mock_ensure.assert_not_called()


def test_idle_path_unchanged_when_nothing_deferred(manager):
    """No regression: an ordinary idle turn-end (no deferred, no queued,
    no busy sub-agents) still broadcasts idle and does not spuriously wake."""
    handle, task = _handle_with_deferred(manager, [])

    mock_ensure = _run_callback(manager, task)

    handle.session.append.assert_not_called()
    mock_ensure.assert_not_called()
    manager._ws.broadcast.assert_called()
    call_args = manager._ws.broadcast.call_args[0]
    assert call_args[1]["status"] == "idle"


def test_queued_user_message_still_wakes_without_any_deferred(manager):
    """Regression guard: the pre-existing queued-user-message resume path is
    untouched by the D1 fix when there is nothing deferred at all."""
    handle, task = _handle_with_deferred(
        manager, [], queued=[("go ahead", None)])

    mock_ensure = _run_callback(manager, task)

    handle.session.append.assert_called_once()
    appended = handle.session.append.call_args[0][0]
    assert appended["role"] == "user"
    assert appended["content"] == "go ahead"
    mock_ensure.assert_called_once()


def test_deferred_terminal_not_reprocessed_on_a_second_idle_turn_end(manager):
    """No double-processing: pop_deferred_messages is destructive (already
    clears on pop), so a SECOND _on_loop_done invocation for the same handle
    — simulating the woken turn ending idle with nothing newly deferred —
    must not re-wake the loop."""
    handle, task = _handle_with_deferred(
        manager, [_terminal_marker("[Sub-agent] cursor error", "error")])

    first_ensure = _run_callback(manager, task)
    first_ensure.assert_called_once()

    # Simulate the resumed turn ending idle: the deferred buffer is now
    # empty (already popped) and nothing new is queued.
    handle.session.pop_deferred_messages.return_value = []
    handle.session.pop_queued_messages.return_value = []
    handle.session.append.reset_mock()

    second_task = MagicMock()
    second_task.exception.return_value = None
    handle.task = second_task

    second_ensure = _run_callback(manager, second_task)

    handle.session.append.assert_not_called()
    second_ensure.assert_not_called()
