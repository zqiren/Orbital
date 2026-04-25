# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Integration test: HTTP POST /api/v2/agents/{pid}/cancel wires to
AgentManager.cancel_message() which calls AgentLoop.cancel_turn() (T05).

Drives the FastAPI route directly via fastapi.testclient.TestClient so the
real HTTP path executes (request → route handler → AgentManager.cancel_message
→ AgentLoop.cancel_turn → cancellation marker written to JSONL).

Pattern follows test_terminate_paths.py: real Session, real AgentLoop,
real provider yielding chunks then sleeping. Loop runs as task in test event
loop; HTTP call dispatched via asyncio.to_thread. cancel_turn() persists the
JSONL marker SYNCHRONOUSLY before its cross-loop await — so the marker
assertion holds regardless of whether the wait_for completes cleanly.

Tests:
  1. test_cancel_via_http_no_agent — 200 {"status": "no_agent"} for unknown pid.
  2. test_cancel_via_http_during_stream — real AgentLoop in flight; /cancel
     within 5s; session JSONL contains cancellation marker; agent task
     still alive in _handles; sub-agents untouched; ws broadcasts idle.
  3. test_cancel_via_http_idle_loop — 200 {"status": "idle"} when task.done().
  4. test_cancel_then_send_continues — handle still in _handles after cancel,
     session not stopped, stop_all not called.
  5. test_cancel_versus_stop_endpoint_isolation — /cancel keeps handle;
     /stop pops handle.
"""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
import time
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from agent_os.agent.loop import AgentLoop
from agent_os.agent.providers.types import StreamChunk, TokenUsage
from agent_os.agent.session import Session
from agent_os.agent.tools.base import ToolResult
from agent_os.api.routes import agents_v2
from agent_os.daemon_v2.agent_manager import AgentManager, ProjectHandle


# ---------------------------------------------------------------------------
# Fake collaborators (mirrors test_terminate_paths.py)
# ---------------------------------------------------------------------------


class _SlowStreamProvider:
    """Yields one chunk then hangs forever — simulates a long LLM call."""

    def __init__(self):
        self.chunks_yielded = 0

    @property
    def model(self):
        return "fake-slow"

    async def stream(self, context, tools=None):
        yield StreamChunk(text="partial output ",
                          usage=TokenUsage(input_tokens=10, output_tokens=2))
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_agent_manager_with_running_loop(workspace, project_id, project_name,
                                          provider=None):
    """Build AgentManager + ProjectHandle with a real AgentLoop.

    Mirrors test_terminate_paths._make_agent_manager_with_running_loop.
    """
    from agent_os.daemon_v2.project_store import project_dir_name as _pdn

    project_store = MagicMock()
    project_store.get_project.return_value = {
        "project_id": project_id,
        "name": project_name,
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

    dir_name = _pdn(project_name, project_id)
    session = Session.new(f"sess_{project_id}", workspace, project_dir_name=dir_name)

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
        project_dir_name=dir_name,
    )
    mgr._handles[project_id] = handle
    return mgr, ws, project_store, handle


def _make_manager_mock_only():
    """Minimal AgentManager with mocked collaborators for HTTP-only tests."""
    project_store = MagicMock()
    ws = MagicMock()
    ws.broadcast = MagicMock()
    sub_agent_mgr = MagicMock()
    sub_agent_mgr.stop_all = AsyncMock()
    sub_agent_mgr.list_active = MagicMock(return_value=[])
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
    return mgr, ws, sub_agent_mgr, project_store


def _inject_mock_handle(mgr, project_id):
    """Inject a fake handle (mocked AgentLoop) into _handles."""
    loop = MagicMock()
    loop.cancel_turn = AsyncMock()
    session = MagicMock()
    session.is_stopped = MagicMock(return_value=False)
    session.stop = MagicMock()

    async def _never_done():
        await asyncio.sleep(9999)

    task = asyncio.get_event_loop().create_task(_never_done())

    handle = ProjectHandle(
        session=session,
        loop=loop,
        provider=MagicMock(),
        registry=MagicMock(),
        context_manager=MagicMock(),
        interceptor=MagicMock(),
        task=task,
    )
    mgr._handles[project_id] = handle
    return handle, task


def _build_test_client(mgr, project_store=None):
    """Wire the test HTTP client with the AgentManager."""
    if project_store is None:
        project_store = MagicMock()
    saved = {k: getattr(agents_v2, k)
             for k in ["_project_store", "_agent_manager", "_ws_manager", "_sub_agent_manager"]}
    sub_agent_mgr = MagicMock()
    sub_agent_mgr.resolve_sub_agent_approval = AsyncMock(return_value=False)
    agents_v2.configure(
        project_store=project_store,
        agent_manager=mgr,
        ws_manager=MagicMock(),
        sub_agent_manager=sub_agent_mgr,
    )
    app = FastAPI()
    app.include_router(agents_v2.router)
    client = TestClient(app)
    return client, saved


def _restore(saved):
    for k, v in saved.items():
        setattr(agents_v2, k, v)


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
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cancel_via_http_no_agent():
    """POST /api/v2/agents/{pid}/cancel on unknown project returns no_agent."""
    mgr, ws, _, _ = _make_manager_mock_only()
    client, saved = _build_test_client(mgr)
    try:
        resp = client.post("/api/v2/agents/does-not-exist/cancel")
        assert resp.status_code == 200
        assert resp.json() == {"status": "no_agent"}
        ws.broadcast.assert_not_called()
    finally:
        client.close()
        _restore(saved)


@pytest.mark.asyncio
async def test_cancel_via_http_during_stream():
    """End-to-end: cancel_message on a real in-flight AgentLoop turn.

    DEVIATION FROM HTTP PATH (documented per spec Option B): drives
    agent_manager.cancel_message() directly rather than through TestClient.post.

    Reason: TestClient (Starlette 1.0.0 + anyio 4.13 + Python 3.13) starts
    a fresh event loop in a worker thread for each request via
    anyio.from_thread.start_blocking_portal(). That portal loop then
    awaits AgentLoop._inflight_stream — a Task created in the parent
    test loop. asyncio.wait_for(<cross-loop-task>, ...) raises
    "Task got Future attached to a different loop" but the resulting
    coroutine never returns to the portal cleanly, hanging the worker
    thread indefinitely (verified at 8s, 15s, 30s pytest timeouts).

    The HTTP routing layer is covered by the 4 sibling integration tests
    (no_agent, idle_loop, then_send_continues, vs_stop_isolation) which
    drive the actual /api/v2/agents/{pid}/cancel route via TestClient.
    The route handler is a 1-line passthrough (return await
    _agent_manager.cancel_message(project_id)), so direct invocation here
    exercises identical behaviour for the in-flight-stream path.

    Asserts:
    - returns {"status": "cancelled"} within 5s
    - session JSONL contains cancellation marker (cancelled_by_user=True)
    - agent task still alive in _handles (not popped)
    - sub-agent stop_all NOT called (this is /cancel, not /stop)
    - ws.broadcast called with agent.status: idle
    """
    with tempfile.TemporaryDirectory() as workspace:
        project_id = "proj_cancel_stream"
        project_name = "Cancel Stream Test"
        provider = _SlowStreamProvider()

        mgr, ws, project_store, handle = _make_agent_manager_with_running_loop(
            workspace, project_id, project_name, provider=provider,
        )

        # Start the real AgentLoop in this event loop.
        loop_obj = handle.loop
        run_task = asyncio.create_task(loop_obj.run(initial_message="hi"))
        handle.task = run_task

        # Wait for the stream to begin (provider yields one chunk then hangs).
        deadline = time.monotonic() + 2.0
        while provider.chunks_yielded < 1 and time.monotonic() < deadline:
            await asyncio.sleep(0.01)
        assert provider.chunks_yielded >= 1, "Loop did not start streaming within 2s"

        session_path = handle.session._filepath

        try:
            started = time.monotonic()
            # Direct method invocation — same path the HTTP route delegates to.
            result = await asyncio.wait_for(
                mgr.cancel_message(project_id), timeout=5.0,
            )
            elapsed = time.monotonic() - started

            assert elapsed < 5.0, f"cancel_message took too long: {elapsed:.2f}s"
            assert result == {"status": "cancelled"}, f"Got: {result}"

            # ws.broadcast must have included an agent.status: idle event
            idle_calls = [
                call for call in ws.broadcast.call_args_list
                if (call.args
                    and isinstance(call.args[1], dict)
                    and call.args[1].get("type") == "agent.status"
                    and call.args[1].get("status") == "idle")
            ]
            assert len(idle_calls) >= 1, (
                f"Expected agent.status:idle broadcast, got: "
                f"{[c.args for c in ws.broadcast.call_args_list]}"
            )

            # Session JSONL must contain the cancellation marker.
            msgs = _read_session_messages(session_path)
            markers = [m for m in msgs if m.get("cancelled_by_user") is True]
            assert len(markers) == 1, (
                f"Session must contain exactly one cancellation marker, "
                f"got {len(markers)}: {[m.get('content') for m in msgs]}"
            )

            # Handle still in _handles — agent alive, not torn down.
            assert project_id in mgr._handles
            assert mgr._handles[project_id] is handle

            # /cancel must NOT call stop_all (that's /stop's job).
            mgr._sub_agent_manager.stop_all.assert_not_awaited()
        finally:
            # The cancel-only path leaves the loop running (by design — that's
            # what /cancel means). To tear down the test cleanly, mark the
            # session stopped so the loop's CancelledError handler propagates
            # instead of `continue`-ing into iteration N+1 (loop.py line 306).
            if not run_task.done():
                handle.session.stop()
                run_task.cancel()
            try:
                await asyncio.wait_for(run_task, timeout=3.0)
            except (asyncio.CancelledError, asyncio.TimeoutError, Exception):
                pass


@pytest.mark.asyncio
async def test_cancel_via_http_idle_loop():
    """POST /api/v2/agents/{pid}/cancel when task is done returns idle."""
    mgr, ws, _, _ = _make_manager_mock_only()
    pid = "proj-idle-http"
    handle, task = _inject_mock_handle(mgr, pid)

    task.cancel()
    try:
        await task
    except (asyncio.CancelledError, Exception):
        pass
    assert task.done()

    client, saved = _build_test_client(mgr)
    try:
        resp = client.post(f"/api/v2/agents/{pid}/cancel")
        assert resp.status_code == 200
        assert resp.json() == {"status": "idle"}
        handle.loop.cancel_turn.assert_not_awaited()
        ws.broadcast.assert_not_called()
    finally:
        client.close()
        _restore(saved)


@pytest.mark.asyncio
async def test_cancel_then_send_continues():
    """After /cancel, handle stays in _handles, session not stopped, stop_all not called."""
    mgr, ws, sub_agent_mgr, _ = _make_manager_mock_only()
    pid = "proj-cancel-continue"
    handle, task = _inject_mock_handle(mgr, pid)
    client, saved = _build_test_client(mgr)
    try:
        resp = client.post(f"/api/v2/agents/{pid}/cancel")
        assert resp.status_code == 200
        assert resp.json()["status"] == "cancelled"

        # Agent handle is still alive, same loop object.
        assert pid in mgr._handles
        assert mgr._handles[pid] is handle

        # Session not stopped.
        handle.session.stop.assert_not_called()

        # stop_all not called.
        sub_agent_mgr.stop_all.assert_not_awaited()
    finally:
        client.close()
        _restore(saved)
        task.cancel()
        try:
            await task
        except (asyncio.CancelledError, Exception):
            pass


@pytest.mark.asyncio
async def test_cancel_versus_stop_endpoint_isolation():
    """/cancel keeps handle alive; /stop pops the handle.

    Single AgentManager with two project handles — one hit with /cancel,
    one with /stop. Verifies the two routes have isolated effects.
    """
    mgr, ws, _, _ = _make_manager_mock_only()
    pid_cancel = "proj-iso-cancel"
    pid_stop = "proj-iso-stop"

    handle_cancel, task_cancel = _inject_mock_handle(mgr, pid_cancel)
    handle_stop, task_stop = _inject_mock_handle(mgr, pid_stop)

    client, saved = _build_test_client(mgr)
    try:
        # /cancel
        resp_cancel = client.post(f"/api/v2/agents/{pid_cancel}/cancel")
        assert resp_cancel.status_code == 200
        assert resp_cancel.json()["status"] == "cancelled"
        assert pid_cancel in mgr._handles

        # /stop — mark target task done first so stop_agent's shield-wait
        # doesn't hang.
        task_stop.cancel()
        try:
            await task_stop
        except (asyncio.CancelledError, Exception):
            pass

        resp_stop = client.post(f"/api/v2/agents/{pid_stop}/stop")
        assert resp_stop.status_code == 200

        # /stop'd agent: handle popped.
        assert pid_stop not in mgr._handles
        # /cancel'd agent: handle still alive.
        assert pid_cancel in mgr._handles
    finally:
        client.close()
        _restore(saved)
        for t in (task_cancel, task_stop):
            if not t.done():
                t.cancel()
            try:
                await t
            except (asyncio.CancelledError, Exception):
                pass
