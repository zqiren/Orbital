# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Regression: defense-in-depth tests for the new_session-during-stream hang.

The hang regression documented in BLOCKER-K3b-rebase-test_new_session
_terminates_old_loop.md was incidentally resolved by PR #27's Stop button
fix (the ``continue -> break`` change in loop.py's CancelledError
handlers). This file locks in the resulting invariants so the bug can't
silently re-appear under a future refactor:

  1. ``mgr.new_session()`` on an in-flight loop must exit within ~3s
     (the loop cannot re-enter the while body after cancel_turn fires
     from within terminate()).
  2. ``mgr.stop_agent()`` on an agent that was started via the full
     start_agent path must drain the queue dispatcher task — the
     ``_ensure_dispatcher`` -> ``_dispatchers[project_id]`` task
     introduced in variant-a (PR #26) must not leak.

These tests are the §5.1 §3+§4 regression requirements from
TASK-hang-fix-new-session-terminate.md.
"""

from __future__ import annotations

import asyncio
import json
import tempfile
import time
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent_os.agent.loop import AgentLoop
from agent_os.agent.providers.types import StreamChunk, TokenUsage
from agent_os.agent.session import Session
from agent_os.agent.tools.base import ToolResult
from agent_os.daemon_v2.agent_manager import AgentManager, ProjectHandle


# ---------------------------------------------------------------------------
# Fake collaborators (mirrors test_terminate_paths.py)
# ---------------------------------------------------------------------------


class _SlowStreamProvider:
    """Yields one chunk then hangs forever."""

    def __init__(self):
        self.chunks_yielded = 0
        self.call_count = 0

    @property
    def model(self):
        return "fake-slow"

    def update_api_key(self, key):
        return None

    async def stream(self, context, tools=None):
        self.call_count += 1
        yield StreamChunk(
            text="partial ",
            usage=TokenUsage(input_tokens=10, output_tokens=2),
        )
        self.chunks_yielded += 1
        await asyncio.Event().wait()


class _NoOpRegistry:
    def reset_run_state(self):
        pass

    def schemas(self):
        return []

    def is_async(self, name):
        return False

    def execute(self, name, args):
        return ToolResult(content="ok")

    async def execute_async(self, name, args):
        return ToolResult(content="ok")


class _PassthroughContextManager:
    def __init__(self, session):
        self._session = session
        self.model_context_limit = 200_000

    def prepare(self):
        return list(self._session.get_messages())

    def should_compact(self):
        return False


def _make_manager_with_loop(workspace, project_id, provider=None):
    """Construct an AgentManager + ProjectHandle with a real AgentLoop."""
    project_store = MagicMock()
    project_store.get_project.return_value = {
        "project_id": project_id,
        "name": "Terminate Leak Test",
        "workspace": workspace,
        "model": "fake-slow",
        "api_key": "sk-test",
        "provider": "custom",
        "sdk": "openai",
    }
    ws = MagicMock()
    ws.broadcast = MagicMock()
    sub_agent_mgr = MagicMock()
    sub_agent_mgr.list_active = MagicMock(return_value=[])
    sub_agent_mgr.stop = AsyncMock()
    sub_agent_mgr.stop_all = AsyncMock()
    activity_translator = MagicMock()
    process_manager = MagicMock()
    process_manager.set_session = MagicMock()

    mgr = AgentManager(
        project_store=project_store,
        ws_manager=ws,
        sub_agent_manager=sub_agent_mgr,
        activity_translator=activity_translator,
        process_manager=process_manager,
    )

    session = Session.new(f"sess_{project_id}", workspace)
    if provider is None:
        provider = _SlowStreamProvider()
    cm = _PassthroughContextManager(session)
    loop_obj = AgentLoop(
        session=session,
        provider=provider,
        tool_registry=_NoOpRegistry(),
        context_manager=cm,
    )
    handle = ProjectHandle(
        session=session,
        loop=loop_obj,
        provider=provider,
        registry=_NoOpRegistry(),
        context_manager=cm,
        interceptor=None,
        task=None,
        config_snapshot={"workspace": workspace, "model": "fake-slow"},
    )
    mgr._handles[(project_id, "default")] = handle
    return mgr, ws, handle


def _read_session_messages(filepath):
    msgs = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            msgs.append(json.loads(line))
    return msgs


# ---------------------------------------------------------------------------
# Test 1: new_session terminates the old loop quickly with no orphan tasks
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_new_session_terminates_in_under_3s_without_leak():
    """Direct mgr.new_session() on an in-flight loop returns in < 3s and
    leaves no orphan asyncio tasks behind.

    Before the loop.py continue->break fix (PR #27): this would hang on
    the inner ``asyncio.wait_for(asyncio.shield(handle.task), timeout=10.0)``
    inside new_session, because the loop's CancelledError handler did
    ``continue`` and re-streamed iteration N+1 before session.stop()
    was set inside terminate(). The 10s shield wait completed, but the
    leaked old run_task continued to run, manifesting as a pytest-loop-
    teardown hang.

    After the fix: the CancelledError handler breaks immediately, the
    loop exits via _on_loop_done, and new_session's shield wait returns
    promptly. The leak is closed at the source.
    """
    with tempfile.TemporaryDirectory() as workspace:
        project_id = "proj_terminate_no_leak"
        provider = _SlowStreamProvider()
        mgr, ws, handle = _make_manager_with_loop(workspace, project_id, provider)

        loop_obj = handle.loop
        run_task = asyncio.create_task(loop_obj.run(initial_message="hi"))
        handle.task = run_task
        run_task.add_done_callback(
            mgr._on_loop_done(project_id, session_id="default")
        )

        try:
            # Wait for stream to begin (loop is mid-stream).
            deadline = time.monotonic() + 2.0
            while provider.chunks_yielded < 1 and time.monotonic() < deadline:
                await asyncio.sleep(0.01)
            assert provider.chunks_yielded >= 1, "Loop did not start streaming"

            old_session_path = handle.session._filepath
            tasks_before = set(asyncio.all_tasks())

            # Direct method invocation — same path the HTTP route delegates to.
            t0 = time.monotonic()
            result = await asyncio.wait_for(
                mgr.new_session(project_id), timeout=3.0,
            )
            elapsed = time.monotonic() - t0

            # ---- A1: timing ----
            assert elapsed < 3.0, (
                f"new_session must complete in < 3s; got {elapsed:.2f}s. "
                f"This indicates the loop is hanging on the shield wait "
                f"because CancelledError handler is re-entering the while "
                f"body instead of breaking."
            )

            # ---- A2: response shape ----
            assert result.get("status") == "ok", f"Got: {result}"

            # ---- A3: old loop is done ----
            # The break-on-cancel path means the old run_task exited via
            # break -> finally -> task completion, with no need to re-cancel.
            try:
                await asyncio.wait_for(run_task, timeout=2.0)
            except asyncio.CancelledError:
                pass
            assert run_task.done(), "Old loop run_task must be done after new_session"

            # ---- A4: only ONE LLM stream call happened ----
            # Critical break-on-cancel invariant. Under continue-on-cancel
            # the terminate() path would have triggered iteration N+1
            # (call_count == 2) before session.stop() landed.
            assert provider.call_count == 1, (
                f"Loop must not start a second LLM call between cancel_turn "
                f"and session.stop() inside terminate(). Got "
                f"call_count={provider.call_count} — the continue->break "
                f"invariant in loop.py has regressed."
            )

            # ---- A5: cancellation marker on old session ----
            old_msgs = _read_session_messages(old_session_path)
            markers = [m for m in old_msgs if m.get("cancelled_by_user") is True]
            assert len(markers) == 1, (
                f"Old session JSONL must contain exactly 1 cancellation "
                f"marker, got {len(markers)}"
            )

            # ---- A6: new session exists and is fresh ----
            new_handle = mgr._handles.get((project_id, "default"))
            assert new_handle is not None, "New handle must exist after new_session"
            assert new_handle.session._filepath != old_session_path, (
                "New session must have a different JSONL file"
            )

            # ---- A7: no orphan asyncio tasks ----
            await asyncio.sleep(0.05)  # let scheduler drain any done-tasks
            tasks_after = set(asyncio.all_tasks())
            new_tasks = tasks_after - tasks_before
            new_tasks.discard(asyncio.current_task())
            still_running = {t for t in new_tasks if not t.done()}
            assert len(still_running) == 0, (
                f"Orphan asyncio tasks still running after new_session: "
                f"{[t.get_name() for t in still_running]}"
            )
        finally:
            if not run_task.done():
                run_task.cancel()
                try:
                    await run_task
                except (asyncio.CancelledError, Exception):
                    pass


# ---------------------------------------------------------------------------
# Test 2: stop_agent drains the queue dispatcher task (variant-a feature)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_stop_agent_drains_dispatcher_task():
    """The variant-a feature (_ensure_dispatcher -> _dispatchers[pid] task)
    must be fully drained by stop_agent.

    PR #26 added a long-lived per-project queue dispatcher task on
    start_agent. stop_agent owns the cleanup: it must pop the dispatcher
    from _dispatchers and await its shutdown(). If that cleanup is ever
    refactored away, the dispatcher task lingers in the event loop after
    stop_agent returns, and pytest-loop teardown will see the leak.

    This test directly verifies:
      - mgr._dispatchers contains the project after we wire one in
      - stop_agent removes the entry from _dispatchers
      - the dispatcher's internal task is done after stop_agent returns
    """
    with tempfile.TemporaryDirectory() as workspace:
        project_id = "proj_dispatcher_cleanup"
        provider = _SlowStreamProvider()
        mgr, ws, handle = _make_manager_with_loop(workspace, project_id, provider)

        # Stand up the queue dispatcher via the same code path start_agent
        # uses. This is the leak-prone object.
        await mgr._ensure_dispatcher(project_id, workspace)
        assert project_id in mgr._dispatchers, (
            "Sanity: dispatcher must be registered after _ensure_dispatcher"
        )
        dispatcher = mgr._dispatchers[project_id]
        # Dispatcher started its background task during .start()
        dispatch_task = getattr(dispatcher, "_task", None)
        assert dispatch_task is not None, (
            "Dispatcher must own a background task after start"
        )
        assert not dispatch_task.done(), (
            "Sanity: dispatcher task should be alive before stop_agent"
        )

        # Start the loop and let it begin streaming so stop_agent has a
        # real cancellation path to exercise.
        loop_obj = handle.loop
        run_task = asyncio.create_task(loop_obj.run(initial_message="hi"))
        handle.task = run_task
        run_task.add_done_callback(
            mgr._on_loop_done(project_id, session_id="default")
        )
        deadline = time.monotonic() + 2.0
        while provider.chunks_yielded < 1 and time.monotonic() < deadline:
            await asyncio.sleep(0.01)
        assert provider.chunks_yielded >= 1

        try:
            tasks_before = set(asyncio.all_tasks())
            await asyncio.wait_for(mgr.stop_agent(project_id), timeout=5.0)

            # ---- A1: dispatcher entry removed ----
            assert project_id not in mgr._dispatchers, (
                "stop_agent must pop the dispatcher entry from "
                "_dispatchers; got entries: "
                f"{list(mgr._dispatchers.keys())}"
            )

            # ---- A2: dispatcher's internal task is done ----
            assert dispatch_task.done(), (
                "Dispatcher background task must be drained by "
                "dispatcher.shutdown() inside stop_agent"
            )

            # ---- A3: no orphan tasks survive stop_agent ----
            await asyncio.sleep(0.05)
            tasks_after = set(asyncio.all_tasks())
            new_tasks = tasks_after - tasks_before
            new_tasks.discard(asyncio.current_task())
            still_running = {t for t in new_tasks if not t.done()}
            assert len(still_running) == 0, (
                f"Orphan asyncio tasks still running after stop_agent: "
                f"{[t.get_name() for t in still_running]}"
            )
        finally:
            if not run_task.done():
                run_task.cancel()
                try:
                    await run_task
                except (asyncio.CancelledError, Exception):
                    pass
