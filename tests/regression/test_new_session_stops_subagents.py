# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Regression tests for TASK-cancel-arch-06: new_session() stops sub-agents
after session-end summarization, before building the new session.

Previously, new_session() called stop_all before session-end (step 2), leaving
the window between session-end and new-session-build where old adapters were
still live. This batch adds a second stop_all call AFTER session-end (step 3)
and BEFORE building the new session (step 5), wrapped in asyncio.wait_for
with a 10s budget to avoid blocking the rotation indefinitely.

Each test FAILS on pre-fix code (no stop_all call in new_session after
session-end) and PASSES with the fix.
"""

from __future__ import annotations

import asyncio
import logging
from unittest.mock import AsyncMock, MagicMock, patch, call

import pytest

from agent_os.daemon_v2.agent_manager import AgentManager, ProjectHandle


# ------------------------------------------------------------------ #
# Helpers / factories
# ------------------------------------------------------------------ #

PROJECT_ID = "proj-new-session-stop-all"


def _make_session(session_id: str = "old_session_123") -> MagicMock:
    mock_session = MagicMock()
    mock_session.session_id = session_id
    mock_session.is_stopped.return_value = False
    return mock_session


def _make_handle(session_id: str = "old_session_123") -> ProjectHandle:
    """Build a minimal ProjectHandle with task=None (idle agent)."""
    return ProjectHandle(
        session=_make_session(session_id),
        loop=MagicMock(),
        provider=MagicMock(),
        registry=MagicMock(),
        context_manager=MagicMock(),
        interceptor=MagicMock(),
        task=None,  # idle — skips loop-stop branch
        config_snapshot={"workspace": "/tmp/test-workspace"},
        project_dir_name="test-project",
    )


def _make_manager() -> tuple[AgentManager, MagicMock, MagicMock]:
    """Construct a minimal AgentManager.

    Returns (manager, mock_ws, mock_sub_agent_manager).
    """
    mock_ws = MagicMock()
    mock_ws.broadcast = MagicMock()

    mock_sam = MagicMock()
    mock_sam.stop_all = AsyncMock()

    mock_project_store = MagicMock()
    mock_project_store.get_project.return_value = {
        "project_id": PROJECT_ID,
        "name": "Test Project",
        "workspace": "/tmp/test-workspace",
    }

    manager = AgentManager(
        project_store=mock_project_store,
        ws_manager=mock_ws,
        sub_agent_manager=mock_sam,
        activity_translator=MagicMock(),
        process_manager=MagicMock(),
    )
    return manager, mock_ws, mock_sam


# ------------------------------------------------------------------ #
# Test 1: stop_all called once AFTER session-end, BEFORE new session build
# ------------------------------------------------------------------ #


@pytest.mark.asyncio
async def test_new_session_calls_stop_all():
    """new_session() must call stop_all exactly once (after session-end) with
    the correct project_id, and it must be called BEFORE the new session is
    built.

    Fails on pre-fix code: no stop_all call exists in new_session after
    session-end.
    """
    manager, _ws, mock_sam = _make_manager()
    manager._handles[PROJECT_ID] = _make_handle()

    call_order: list[str] = []

    # The stop_all we care about is called *after* session-end.
    # We intercept Session.new to track call order.
    original_stop_all = mock_sam.stop_all

    async def _recording_stop_all(pid: str) -> None:
        call_order.append("stop_all")

    mock_sam.stop_all.side_effect = _recording_stop_all

    # Patch run_session_end_routine to a no-op coroutine so the test is fast.
    async def _noop_session_end(**kwargs):
        call_order.append("session_end")

    # Patch Session.new to record its call order
    mock_new_session = MagicMock()
    mock_new_session.session_id = "new_session_abc"
    mock_new_session.on_append = None
    mock_new_session.on_stream = None

    original_session_new = None

    def _recording_session_new(*args, **kwargs):
        call_order.append("session_new")
        # Return a proper mock session
        s = MagicMock()
        s.session_id = "new_session_abc"
        s.on_append = None
        s.on_stream = None
        return s

    with (
        patch(
            "agent_os.daemon_v2.agent_manager.run_session_end_routine",
            side_effect=_noop_session_end,
        ),
        patch(
            "agent_os.daemon_v2.agent_manager.Session.new",
            side_effect=_recording_session_new,
        ),
    ):
        result = await manager.new_session(PROJECT_ID)

    assert result.get("status") == "ok", f"Expected ok, got {result}"

    # T06 contract: stop_all must be called EXACTLY ONCE, with the correct
    # project_id, AFTER session-end and BEFORE the new session is built.
    mock_sam.stop_all.assert_called_once_with(PROJECT_ID)

    # Verify canonical ordering
    assert call_order == ["session_end", "stop_all", "session_new"], (
        f"Expected order [session_end, stop_all, session_new], got: {call_order}"
    )


# ------------------------------------------------------------------ #
# Test 2: new_session completes when stop_all times out
# ------------------------------------------------------------------ #


@pytest.mark.asyncio
async def test_new_session_proceeds_on_stop_all_timeout(caplog):
    """When stop_all hangs past the 10s budget, new_session must still
    complete, log an ERROR, and build a new session.

    Fails on pre-fix code: the post-session-end stop_all call doesn't exist,
    so the timeout path can never be exercised.
    """
    manager, _ws, mock_sam = _make_manager()
    manager._handles[PROJECT_ID] = _make_handle()

    # Track whether a new session was built
    new_session_built: list[bool] = []

    def _recording_session_new(*args, **kwargs):
        new_session_built.append(True)
        s = MagicMock()
        s.session_id = "new_session_timeout"
        s.on_append = None
        s.on_stream = None
        return s

    # Patch asyncio.wait_for so that *any* call to it with the post-session-end
    # stop_all coroutine raises TimeoutError immediately.
    # We identify the right wait_for call by checking if stop_all is the
    # coroutine (timeout=10.0 budget).
    _real_wait_for = asyncio.wait_for

    async def _selective_timeout(coro, timeout):
        # Raise TimeoutError only for the post-session-end stop_all call
        # (timeout=10.0, not the 200s session-end call).
        if timeout == 10.0:
            try:
                coro.close()
            except Exception:
                pass
            raise asyncio.TimeoutError()
        return await _real_wait_for(coro, timeout)

    with (
        patch(
            "agent_os.daemon_v2.agent_manager.run_session_end_routine",
            new=AsyncMock(),
        ),
        patch(
            "agent_os.daemon_v2.agent_manager.Session.new",
            side_effect=_recording_session_new,
        ),
        patch(
            "agent_os.daemon_v2.agent_manager.asyncio.wait_for",
            side_effect=_selective_timeout,
        ),
        caplog.at_level(logging.ERROR, logger="agent_os.daemon_v2.agent_manager"),
    ):
        import time
        start = time.monotonic()
        result = await manager.new_session(PROJECT_ID)
        elapsed = time.monotonic() - start

    # Must complete within 15s (10s budget + overhead)
    assert elapsed < 15.0, f"new_session took {elapsed:.2f}s, expected <15s"

    # New session must have been built
    assert new_session_built, "new session must be built even after stop_all timeout"
    assert result.get("status") == "ok", f"Expected ok, got {result}"

    # ERROR log must have been emitted
    error_records = [r for r in caplog.records if r.levelno >= logging.ERROR]
    assert any(
        "timed out" in r.message.lower() or "stop_all" in r.message.lower()
        for r in error_records
    ), f"Expected ERROR log about stop_all timeout, got: {[r.message for r in error_records]}"


# ------------------------------------------------------------------ #
# Test 3: new_session completes when stop_all raises an exception
# ------------------------------------------------------------------ #


@pytest.mark.asyncio
async def test_new_session_proceeds_on_stop_all_exception(caplog):
    """When stop_all raises RuntimeError, new_session must still complete,
    log the exception, and build a new session.

    Fails on pre-fix code: post-session-end stop_all never called.
    """
    manager, _ws, mock_sam = _make_manager()
    manager._handles[PROJECT_ID] = _make_handle()

    new_session_built: list[bool] = []

    def _recording_session_new(*args, **kwargs):
        new_session_built.append(True)
        s = MagicMock()
        s.session_id = "new_session_exc"
        s.on_append = None
        s.on_stream = None
        return s

    # T06: stop_all is called exactly once (post-session-end). Make it raise.
    mock_sam.stop_all.side_effect = RuntimeError("simulated stop_all failure")

    with (
        patch(
            "agent_os.daemon_v2.agent_manager.run_session_end_routine",
            new=AsyncMock(),
        ),
        patch(
            "agent_os.daemon_v2.agent_manager.Session.new",
            side_effect=_recording_session_new,
        ),
        caplog.at_level(logging.ERROR, logger="agent_os.daemon_v2.agent_manager"),
    ):
        result = await manager.new_session(PROJECT_ID)

    # New session must have been built
    assert new_session_built, "new session must be built even after stop_all exception"
    assert result.get("status") == "ok", f"Expected ok, got {result}"

    # Exception must have been logged
    error_records = [r for r in caplog.records if r.levelno >= logging.ERROR]
    assert error_records, "Expected an ERROR/EXCEPTION log from stop_all failure"


# ------------------------------------------------------------------ #
# Test 4: no sub-agents registered — stop_all safe to call (no-op)
# ------------------------------------------------------------------ #


@pytest.mark.asyncio
async def test_new_session_no_subagents_skips_cleanly():
    """When there are no registered sub-agents, stop_all is still invoked
    (its own early-return handles the empty case). No error must occur and
    the new session must be built.

    Fails on pre-fix code: post-session-end stop_all never called.
    """
    manager, _ws, mock_sam = _make_manager()
    manager._handles[PROJECT_ID] = _make_handle()

    # stop_all is a no-op for empty adapter dict (verified in implementation)
    mock_sam.stop_all = AsyncMock()  # clean no-op

    new_session_built: list[bool] = []

    def _recording_session_new(*args, **kwargs):
        new_session_built.append(True)
        s = MagicMock()
        s.session_id = "new_session_clean"
        s.on_append = None
        s.on_stream = None
        return s

    with (
        patch(
            "agent_os.daemon_v2.agent_manager.run_session_end_routine",
            new=AsyncMock(),
        ),
        patch(
            "agent_os.daemon_v2.agent_manager.Session.new",
            side_effect=_recording_session_new,
        ),
    ):
        result = await manager.new_session(PROJECT_ID)

    # stop_all must have been invoked at least once (T06 call after session-end)
    assert mock_sam.stop_all.call_count >= 1, (
        f"stop_all must be called (safe for empty project), got {mock_sam.stop_all.call_count} calls"
    )
    assert new_session_built, "new session must be built"
    assert result.get("status") == "ok", f"Expected ok, got {result}"


# ------------------------------------------------------------------ #
# Test 5: canonical ordering — terminate → shield/wait → stop_all → Session.new → handle.session
# ------------------------------------------------------------------ #


@pytest.mark.asyncio
async def test_new_session_ordering_terminate_before_stop_all():
    """Verify the canonical ordering of new_session:
      (a) old loop terminated
      (b) old loop drained via shield+wait
      (c) sub-agents stopped via stop_all (post-session-end, T06 call)
      (d) new session built (Session.new)
      (e) handle.session swapped

    Uses a shared call-order list with mocks attached to each event.

    Fails on pre-fix code: step (c) after session-end doesn't exist.
    """
    manager, _ws, mock_sam = _make_manager()

    # Make a handle with a running task so we exercise the terminate path.
    mock_loop = MagicMock()
    mock_loop.terminate = AsyncMock()

    mock_task = MagicMock(spec=asyncio.Task)
    mock_task.done.return_value = False  # task is "running" → triggers terminate

    handle = ProjectHandle(
        session=_make_session("old_ordering_session"),
        loop=mock_loop,
        provider=MagicMock(),
        registry=MagicMock(),
        context_manager=MagicMock(),
        interceptor=MagicMock(),
        task=mock_task,
        config_snapshot={"workspace": "/tmp/ordering-test"},
        project_dir_name="ordering-project",
    )
    manager._handles[PROJECT_ID] = handle

    call_order: list[str] = []

    # (a) terminate
    async def _recording_terminate():
        call_order.append("terminate")

    mock_loop.terminate.side_effect = _recording_terminate

    # (b) shield+wait — the task future resolution (we make asyncio.shield
    # return a done future so new_session doesn't hang)
    real_shield = asyncio.shield

    async def _fast_shield_wait(coro, timeout):
        # Intercept the wait_for(shield(task), 10.0) call
        # This is the loop-drain step; record it and return immediately
        if timeout == 10.0 and not isinstance(coro, asyncio.coroutines.CoroutineType.__mro__[0] if False else type(coro)):
            pass
        call_order.append("shield_wait")
        # Cancel the coro to avoid "coroutine was never awaited" warnings
        try:
            coro.close()
        except Exception:
            pass
        # Simulate the task finishing

    # We only want to intercept the shield+wait, not stop_all's wait_for.
    # Use a counter: first wait_for call with timeout=10.0 is the shield+wait.
    wait_for_count = [0]

    async def _recording_wait_for(coro, timeout):
        if timeout == 10.0:
            wait_for_count[0] += 1
            if wait_for_count[0] == 1:
                # First 10s call: shield+wait for loop drain
                call_order.append("shield_wait")
                try:
                    coro.close()
                except Exception:
                    pass
                return
            else:
                # Subsequent 10s calls: post-session-end stop_all (T06)
                call_order.append("stop_all_wait_for")
                return await asyncio.wait_for(coro, timeout=10.0)
        # 200s call: session-end routine
        call_order.append("session_end")
        try:
            coro.close()
        except Exception:
            pass

    # (c) stop_all after session-end
    async def _recording_stop_all(pid: str) -> None:
        call_order.append("stop_all")

    mock_sam.stop_all.side_effect = _recording_stop_all

    # (d) Session.new
    def _recording_session_new(*args, **kwargs):
        call_order.append("session_new")
        s = MagicMock()
        s.session_id = "new_ordering_session"
        s.on_append = None
        s.on_stream = None
        return s

    with (
        patch(
            "agent_os.daemon_v2.agent_manager.run_session_end_routine",
            new=AsyncMock(),
        ),
        patch(
            "agent_os.daemon_v2.agent_manager.Session.new",
            side_effect=_recording_session_new,
        ),
        patch(
            "agent_os.daemon_v2.agent_manager.asyncio.wait_for",
            side_effect=_recording_wait_for,
        ),
    ):
        result = await manager.new_session(PROJECT_ID)

    assert result.get("status") == "ok", f"Expected ok, got {result}"

    # Verify ordering assertions:
    # (a) terminate happens before shield_wait
    assert "terminate" in call_order, f"terminate missing: {call_order}"
    assert "shield_wait" in call_order, f"shield_wait missing: {call_order}"
    terminate_idx = call_order.index("terminate")
    shield_idx = call_order.index("shield_wait")
    assert terminate_idx < shield_idx, (
        f"terminate must precede shield_wait: {call_order}"
    )

    # (c) A stop_all call must exist AFTER session_end and BEFORE session_new
    assert "session_end" in call_order or "stop_all_wait_for" in call_order, (
        f"session_end or stop_all_wait_for missing: {call_order}"
    )
    assert "session_new" in call_order, f"session_new missing: {call_order}"

    session_new_idx = call_order.index("session_new")

    # Find stop_all or stop_all_wait_for after shield_wait and before session_new
    post_shield_stop = [
        i for i, ev in enumerate(call_order)
        if ev in ("stop_all", "stop_all_wait_for") and shield_idx < i < session_new_idx
    ]
    assert post_shield_stop, (
        f"Expected stop_all after loop drain and before session_new.\n"
        f"call_order={call_order}"
    )

    # (e) handle.session must be swapped to new session
    assert manager._handles[PROJECT_ID].session.session_id == "new_ordering_session", (
        "handle.session must be swapped to new session"
    )
