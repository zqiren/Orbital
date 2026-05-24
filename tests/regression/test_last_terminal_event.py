# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Regression: per-session ``last_terminal_event`` field is recorded on
``error`` / ``stopped`` / ``new_session`` broadcasts and cleared on the
next successful ``/inject``.

Before this fix: ``error``, ``stopped``, ``new_session`` were transient
WebSocket events only. If the frontend missed the broadcast (WS down,
page reload, user navigated away), the supervisor had no way to recover
the fact that a session terminated abnormally. Polling
``/api/v2/projects/{pid}/sessions`` returned status ``idle`` after an
error — silently masking the failure.

After this fix: ``list_sessions`` includes ``last_terminal_event`` on
each entry. The field shape:
```
{"type": "error"|"stopped"|"new_session", "timestamp": <iso>, "details": <str|null>}
```
The field is cleared when the session transitions out of the terminal
state via a fresh ``/inject``.

See: ``TASK/TASK-state-model-alignment-fixes.md`` §2.4.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent_os.daemon_v2.agent_manager import AgentManager, ProjectHandle


def _make_manager():
    ws = MagicMock()
    ws.broadcast = MagicMock()
    project_store = MagicMock()
    sub_agent_mgr = MagicMock()
    sub_agent_mgr.list_active = MagicMock(return_value=[])
    sub_agent_mgr.stop_all = AsyncMock()
    activity_translator = MagicMock()
    process_manager = MagicMock()
    mgr = AgentManager(
        project_store=project_store,
        ws_manager=ws,
        sub_agent_manager=sub_agent_mgr,
        activity_translator=activity_translator,
        process_manager=process_manager,
    )
    return mgr, ws


def _make_idle_handle(mgr, project_id, session_id="default"):
    """Plant a handle with a DONE task and no paused-for-approval."""
    session = MagicMock()
    session.is_stopped = MagicMock(return_value=False)
    session._paused_for_approval = False
    session.session_id = f"{session_id}-uuid"

    async def _done_immediately():
        return None
    task = asyncio.get_event_loop().create_task(_done_immediately())

    handle = ProjectHandle(
        session=session,
        loop=MagicMock(),
        provider=MagicMock(),
        registry=MagicMock(),
        context_manager=MagicMock(),
        interceptor=MagicMock(),
        task=task,
    )
    mgr._handles[(project_id, session_id)] = handle
    return handle, task


def _make_error_handle(mgr, project_id, session_id="default"):
    """Plant a handle whose task crashed with an exception."""
    session = MagicMock()
    session.is_stopped = MagicMock(return_value=False)
    session._paused_for_approval = False
    session.session_id = f"{session_id}-uuid"

    async def _raise():
        raise RuntimeError("LLM provider 500: connection reset")
    task = asyncio.get_event_loop().create_task(_raise())

    handle = ProjectHandle(
        session=session,
        loop=MagicMock(),
        provider=MagicMock(),
        registry=MagicMock(),
        context_manager=MagicMock(),
        interceptor=MagicMock(),
        task=task,
    )
    mgr._handles[(project_id, session_id)] = handle
    return handle, task


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_error_event_records_last_terminal_event():
    """When _on_loop_done fires the error broadcast, the same event is
    recorded into last_terminal_event keyed by (project_id, session_id).
    Subsequent list_sessions() returns the field on the matching entry."""
    mgr, ws = _make_manager()
    pid = "proj_terminal_error"
    handle, task = _make_error_handle(mgr, pid)

    # Wait for the task to crash, then fire the done-callback synchronously.
    try:
        await task
    except RuntimeError:
        pass
    cb = mgr._on_loop_done(pid, session_id="default")
    cb(task)

    # last_terminal_event recorded for this session.
    sessions = mgr.list_sessions(pid)
    assert len(sessions) == 1
    entry = sessions[0]
    assert entry["session_id"] == "default"
    assert "last_terminal_event" in entry, (
        "list_sessions must include last_terminal_event field"
    )
    evt = entry["last_terminal_event"]
    assert evt is not None
    assert evt["type"] == "error"
    assert "timestamp" in evt
    assert "details" in evt
    assert "LLM provider 500" in (evt["details"] or "")


@pytest.mark.asyncio
async def test_stopped_event_records_last_terminal_event():
    """stop_agent's stopped broadcast must also record last_terminal_event."""
    mgr, ws = _make_manager()
    pid = "proj_terminal_stop"
    handle, task = _make_idle_handle(mgr, pid)
    await task  # ensure task.done()

    # last_terminal_event keyed by SessionKey persists beyond handle pop;
    # we expose it via a direct accessor for this assertion.
    await mgr.stop_agent(pid)

    # Handle was popped — list_sessions is empty for the stopped session.
    assert mgr.list_sessions(pid) == [], (
        "Stopped session must not appear in list_sessions"
    )
    # But last_terminal_event for that key persists.
    evt = mgr.get_last_terminal_event(pid, session_id="default")
    assert evt is not None, (
        "stop_agent must record last_terminal_event for the stopped session"
    )
    assert evt["type"] == "stopped"
    assert "timestamp" in evt


@pytest.mark.asyncio
async def test_new_session_event_records_last_terminal_event():
    """new_session broadcast must record last_terminal_event."""
    mgr, ws = _make_manager()
    pid = "proj_terminal_rotate"
    handle, task = _make_idle_handle(mgr, pid)
    await task

    # new_session is a complex method (LLM call for summary, etc.). Drive
    # the broadcast directly to test only the terminal-event recording.
    mgr._broadcast_new_session_terminal_event(pid, session_id="default")

    sessions = mgr.list_sessions(pid)
    assert len(sessions) == 1
    evt = sessions[0]["last_terminal_event"]
    assert evt is not None
    assert evt["type"] == "new_session"


@pytest.mark.asyncio
async def test_terminal_event_cleared_on_next_inject():
    """A fresh /inject after an error must clear last_terminal_event."""
    mgr, ws = _make_manager()
    pid = "proj_terminal_clear"
    handle, task = _make_error_handle(mgr, pid)
    try:
        await task
    except RuntimeError:
        pass
    cb = mgr._on_loop_done(pid, session_id="default")
    cb(task)

    # Confirm event was recorded.
    assert mgr.get_last_terminal_event(pid, session_id="default") is not None

    # Drive inject; need to mock _start_loop because we don't have a real
    # provider/loop wired up for this manager.
    mgr._start_loop = AsyncMock()
    # Also need build_agent_config to not blow up (Case 3 path uses it);
    # but our handle exists, so we hit Case 2 (hot-resume).
    handle.interceptor.deactivate_bypass_all = MagicMock()

    await mgr.inject_message(pid, "try again please")

    # last_terminal_event should now be cleared.
    evt = mgr.get_last_terminal_event(pid, session_id="default")
    assert evt is None, (
        f"Inject must clear last_terminal_event after error; got {evt}"
    )


@pytest.mark.asyncio
async def test_list_sessions_includes_null_when_no_terminal_event():
    """Sessions with no terminal event must return last_terminal_event=None."""
    mgr, _ = _make_manager()
    pid = "proj_no_event"
    _make_idle_handle(mgr, pid)
    sessions = mgr.list_sessions(pid)
    assert len(sessions) == 1
    assert "last_terminal_event" in sessions[0]
    assert sessions[0]["last_terminal_event"] is None, (
        "Fresh session must have last_terminal_event=None"
    )
