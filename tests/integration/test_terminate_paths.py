# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Integration test: HTTP routes that previously called Session.stop()
now call AgentLoop.terminate(), interrupting an in-flight LLM stream.

Drives the FastAPI route directly via fastapi.testclient.TestClient so the
real HTTP path executes (request → route handler → AgentManager →
AgentLoop.terminate → CancelledError → loop exits via is_stopped()).

These tests verify B2/RC-C: the management agent's LLM stream now has a
cancellation pathway. Before the fix, Session.stop() was purely a flag
setter; an in-flight LLM stream would block the loop indefinitely until
the shield-wait timed out at 10s.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import tempfile
import threading
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
# Helpers — mostly copied from the regression test
# ---------------------------------------------------------------------------


class _SlowStreamProvider:
    """Provider whose stream() yields one chunk then hangs forever."""

    def __init__(self):
        self.chunks_yielded = 0

    @property
    def model(self):
        return "fake-slow"

    async def stream(self, context, tools=None):
        # One chunk so cancel_turn has partial content to debit.
        yield StreamChunk(text="partial output ",
                          usage=TokenUsage(input_tokens=10, output_tokens=2))
        self.chunks_yielded += 1
        # Hang forever — simulates a 30s+ LLM call.
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


def _make_agent_manager_with_running_loop(workspace, project_id, project_name,
                                          provider=None):
    """Construct an AgentManager with a ProjectHandle whose AgentLoop is
    actively running (in the same event loop as the test) and is awaiting
    a long-running stream."""
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

    # Build session, loop, provider.
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
    mgr._handles[project_id] = handle
    return mgr, ws, project_store, handle


def _build_test_client(mgr, project_store):
    saved = {
        "_project_store": agents_v2._project_store,
        "_agent_manager": agents_v2._agent_manager,
        "_ws_manager": agents_v2._ws_manager,
        "_sub_agent_manager": agents_v2._sub_agent_manager,
    }
    sub_agent_mgr = MagicMock()
    # The approve/deny route falls through to this on KeyError; make the
    # async method awaitable so the test exercises the right branches.
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


def _restore_agents_v2(saved):
    agents_v2._project_store = saved["_project_store"]
    agents_v2._agent_manager = saved["_agent_manager"]
    agents_v2._ws_manager = saved["_ws_manager"]
    agents_v2._sub_agent_manager = saved["_sub_agent_manager"]


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
async def test_stop_agent_terminates_via_loop(caplog):
    """POST /api/v2/agents/{pid}/stop with an in-flight slow LLM stream
    must terminate the loop within 5s and produce no
    'loop did not stop gracefully' warning.
    """
    with tempfile.TemporaryDirectory() as workspace:
        project_id = "proj_stop_terminate"
        project_name = "Stop Terminate Test"
        provider = _SlowStreamProvider()

        mgr, ws, project_store, handle = _make_agent_manager_with_running_loop(
            workspace, project_id, project_name, provider=provider,
        )

        # Start the loop running in the test's event loop. The TestClient's
        # synchronous .post() call below is dispatched from a worker thread,
        # so its handler runs on the same anyio event loop the test owns.
        loop_obj = handle.loop
        run_task = asyncio.create_task(loop_obj.run(initial_message="hi"))
        handle.task = run_task

        # Wait for the stream to begin
        deadline = time.monotonic() + 2.0
        while provider.chunks_yielded < 1 and time.monotonic() < deadline:
            await asyncio.sleep(0.01)
        assert provider.chunks_yielded >= 1, (
            "Provider stream did not produce a chunk within 2s — "
            "loop did not start as expected"
        )

        # Hit the HTTP /stop endpoint via TestClient.
        client, saved = _build_test_client(mgr, project_store)
        try:
            with caplog.at_level(logging.WARNING,
                                  logger="agent_os.daemon_v2.agent_manager"):
                started = time.monotonic()
                # Run the synchronous client.post in a thread so we don't
                # block the event loop.
                response = await asyncio.to_thread(
                    client.post,
                    f"/api/v2/agents/{project_id}/stop",
                )
                elapsed = time.monotonic() - started

            assert response.status_code == 200, (
                f"Expected HTTP 200, got {response.status_code}: {response.text}"
            )
            assert response.json().get("status") == "stopping"
            assert elapsed < 5.0, (
                f"/stop must return within 5s, took {elapsed:.2f}s"
            )

            # The "stopped" agent.status WS broadcast must have fired.
            stopped_calls = [
                call for call in ws.broadcast.call_args_list
                if (call.args
                    and isinstance(call.args[1], dict)
                    and call.args[1].get("type") == "agent.status"
                    and call.args[1].get("status") == "stopped")
            ]
            assert len(stopped_calls) >= 1, (
                f"Expected agent.status:stopped broadcast, got: "
                f"{[c.args for c in ws.broadcast.call_args_list]}"
            )

            # No "loop did not stop gracefully" warning.
            graceful_warnings = [
                r for r in caplog.records
                if "loop did not stop gracefully" in r.message
            ]
            # stop_agent path doesn't emit that exact phrase, so this is
            # belt-and-braces. The real signal is that elapsed < 5s.
            assert len(graceful_warnings) == 0, (
                f"Unexpected 'did not stop gracefully' warnings: "
                f"{[r.message for r in graceful_warnings]}"
            )

            # The loop task should be done (cancelled or returned).
            try:
                await asyncio.wait_for(run_task, timeout=2.0)
            except asyncio.CancelledError:
                pass
            assert run_task.done(), (
                "Loop task must be done after /stop"
            )
        finally:
            client.close()
            _restore_agents_v2(saved)
            if not run_task.done():
                run_task.cancel()
                try:
                    await run_task
                except (asyncio.CancelledError, Exception):
                    pass


@pytest.mark.asyncio
async def test_new_session_terminates_old_loop():
    """POST /api/v2/agents/{pid}/new-session with an in-flight slow stream:
    - old session JSONL contains a cancellation marker
    - new session JSONL is fresh and exists
    - no orphan loop task remains in asyncio.all_tasks
    """
    with tempfile.TemporaryDirectory() as workspace:
        project_id = "proj_newsess_terminate"
        project_name = "New Session Terminate Test"
        provider = _SlowStreamProvider()
        mgr, ws, project_store, handle = _make_agent_manager_with_running_loop(
            workspace, project_id, project_name, provider=provider,
        )

        loop_obj = handle.loop
        run_task = asyncio.create_task(loop_obj.run(initial_message="hi"))
        handle.task = run_task

        # Wait for the stream to begin
        deadline = time.monotonic() + 2.0
        while provider.chunks_yielded < 1 and time.monotonic() < deadline:
            await asyncio.sleep(0.01)
        assert provider.chunks_yielded >= 1

        old_session_path = handle.session._filepath

        client, saved = _build_test_client(mgr, project_store)
        try:
            tasks_before = set(asyncio.all_tasks())
            response = await asyncio.to_thread(
                client.post,
                f"/api/v2/agents/{project_id}/new-session",
            )
            assert response.status_code == 200, (
                f"Expected 200, got {response.status_code}: {response.text}"
            )

            body = response.json()
            assert body.get("status") in ("ok", "no_active_session"), (
                f"Unexpected status: {body}"
            )

            # The old run_task should no longer be running.
            try:
                await asyncio.wait_for(run_task, timeout=2.0)
            except asyncio.CancelledError:
                pass
            assert run_task.done()

            # Old session JSONL must contain the cancellation marker.
            old_msgs = _read_session_messages(old_session_path)
            markers = [m for m in old_msgs if m.get("cancelled_by_user") is True]
            assert len(markers) == 1, (
                f"Old session must contain exactly one cancellation marker, "
                f"got {len(markers)}: "
                f"{[m.get('content') for m in old_msgs]}"
            )

            # New session JSONL must exist and be the new handle's session.
            new_handle = mgr._handles.get(project_id)
            assert new_handle is not None
            new_session_path = new_handle.session._filepath
            assert os.path.isfile(new_session_path), (
                f"New session JSONL not found at {new_session_path}"
            )
            assert new_session_path != old_session_path

            # No orphan loop tasks remain (let scheduler drain first)
            await asyncio.sleep(0.05)
            tasks_after = set(asyncio.all_tasks())
            new_tasks = tasks_after - tasks_before
            new_tasks.discard(asyncio.current_task())
            # The TestClient runs the POST in a worker thread that bridges
            # back to the event loop via anyio.to_thread; some plumbing
            # tasks may exist briefly. Filter out anything that's done().
            still_running = {t for t in new_tasks if not t.done()}
            assert len(still_running) == 0, (
                f"Orphan tasks still running after new-session: "
                f"{[t.get_name() for t in still_running]}"
            )
        finally:
            client.close()
            _restore_agents_v2(saved)
            if not run_task.done():
                run_task.cancel()
                try:
                    await run_task
                except (asyncio.CancelledError, Exception):
                    pass


@pytest.mark.asyncio
async def test_approve_during_stream_drains_cleanly(caplog):
    """When an approval is pending and the loop is paused, approving via
    HTTP must complete within 5s and not emit the shield-wait warning.

    NOTE: This path doesn't currently call session.stop()/terminate() —
    it only awaits a shield on the existing loop task. The test ensures
    that the change in T04 (replacing session.stop in two sites) did NOT
    accidentally introduce a regression in the approve path.
    """
    with tempfile.TemporaryDirectory() as workspace:
        project_id = "proj_approve_drain"
        project_name = "Approve Drain Test"

        # Use a provider that returns a normal text response (no tool calls).
        class _SimpleProvider:
            @property
            def model(self):
                return "fake-simple"

            async def stream(self, context, tools=None):
                yield StreamChunk(text="hello")
                yield StreamChunk(text="",
                                   is_final=True,
                                   usage=TokenUsage(input_tokens=5,
                                                    output_tokens=1))

        mgr, ws, project_store, handle = _make_agent_manager_with_running_loop(
            workspace, project_id, project_name, provider=_SimpleProvider(),
        )

        # Configure interceptor on the handle so approve has a target.
        # We use a minimal stub. Approve will short-circuit because
        # there's no real pending approval — we expect a 404 via the
        # KeyError path in the route handler. That still exercises the
        # approve handler's code paths up to the not-found branch.
        from unittest.mock import MagicMock
        interceptor_stub = MagicMock()
        interceptor_stub.get_pending.return_value = None
        interceptor_stub._pending_approvals = {}
        handle.interceptor = interceptor_stub

        # Start a finite loop run (it will exit on its own because the
        # provider returns no tool calls).
        loop_obj = handle.loop
        run_task = asyncio.create_task(loop_obj.run(initial_message="hi"))
        handle.task = run_task

        try:
            await asyncio.wait_for(run_task, timeout=3.0)
        except asyncio.TimeoutError:
            pass

        client, saved = _build_test_client(mgr, project_store)
        try:
            with caplog.at_level(logging.WARNING,
                                  logger="agent_os.daemon_v2.agent_manager"):
                started = time.monotonic()
                response = await asyncio.to_thread(
                    client.post,
                    f"/api/v2/agents/{project_id}/approve",
                    json={"tool_call_id": "tc_no_such_id"},
                )
                elapsed = time.monotonic() - started

            # 404 from the not-found branch — that's fine, we're testing
            # the timing/no-warnings property.
            assert response.status_code in (200, 404), (
                f"Unexpected status {response.status_code}: {response.text}"
            )
            assert elapsed < 5.0, (
                f"/approve must return within 5s, took {elapsed:.2f}s"
            )

            # No shield-wait warning.
            shield_warnings = [
                r for r in caplog.records
                if "loop did not stop gracefully" in r.message
            ]
            assert len(shield_warnings) == 0, (
                f"Unexpected shield-wait warnings: "
                f"{[r.message for r in shield_warnings]}"
            )
        finally:
            client.close()
            _restore_agents_v2(saved)
            if not run_task.done():
                run_task.cancel()
                try:
                    await run_task
                except (asyncio.CancelledError, Exception):
                    pass
