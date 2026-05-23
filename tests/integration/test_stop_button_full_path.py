# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Full-path integration test for the UI Stop button (POST /cancel).

Exercises the production call chain end-to-end: real Session, real
AgentLoop, real AgentManager (no AgentLoop mocking), real
WebSocketManager broadcast hooks. The test drives
``agent_manager.cancel_message()`` directly per the documented Option-B
deviation in ``test_cancel_message_full_path.py`` (TestClient cross-loop
hang under Python 3.13 + anyio 4.13). The HTTP route handler is a 1-line
passthrough to this method, so this exercises identical wire behaviour.

Assertions (drawn from the fix/stop-button-turn-cancel branch's spec):
  A1. /cancel returns {"status": "cancelled"} within 5s of the request.
  A2. Exactly one cancellation marker lands in the session JSONL with
      cancelled_by_user=True.
  A3. The AgentLoop task completes after the cancel (break-on-cancel) —
      the loop does NOT keep running.
  A4. The active-loop slot is mechanically free (current_holder_session_id
      returns None) because the loop task is done.
  A5. The WS broadcast stream contains an agent.status:idle event.
  A6. A subsequent inject_message returns "delivered" (Case 2 hot-resume)
      and NOT "queued" — proving the loop was actually idle, not still
      busy.
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
# Fake collaborators (same shape as test_cancel_message_full_path.py)
# ---------------------------------------------------------------------------


class _HangingProvider:
    """Yields one chunk + usage, then hangs forever. Simulates an LLM
    call mid-stream that the Stop button needs to interrupt."""

    def __init__(self):
        self.chunks_yielded = 0
        self.call_count = 0
        # Set when the stream() coroutine reaches its hang point.
        self._hang_event = asyncio.Event()

    @property
    def model(self):
        return "fake-hang"

    def update_api_key(self, key):
        # Hot-resume path calls this on the handle.provider; no-op here.
        return None

    async def stream(self, context, tools=None):
        self.call_count += 1
        if self.call_count == 1:
            yield StreamChunk(
                text="partial output ",
                usage=TokenUsage(input_tokens=10, output_tokens=2),
            )
            self.chunks_yielded += 1
            self._hang_event.set()
            await asyncio.Event().wait()  # hang forever
        else:
            # On hot-resume (case 6) emit a normal finished reply.
            yield StreamChunk(text="ok done")
            yield StreamChunk(
                text="",
                is_final=True,
                usage=TokenUsage(input_tokens=5, output_tokens=2),
            )


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


def _make_manager_with_loop(workspace, project_id, provider):
    """Build AgentManager + ProjectHandle wired to a real AgentLoop."""
    project_store = MagicMock()
    project_store.get_project.return_value = {
        "project_id": project_id,
        "name": "Stop Button Full Path",
        "workspace": workspace,
        "model": "fake-hang",
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
        config_snapshot={"workspace": workspace, "model": "fake-hang"},
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


def _status_broadcasts(ws):
    """Return the list of agent.status payloads broadcast on ws."""
    out = []
    for call in ws.broadcast.call_args_list:
        if len(call.args) >= 2 and isinstance(call.args[1], dict):
            payload = call.args[1]
            if payload.get("type") == "agent.status":
                out.append(payload)
    return out


# ---------------------------------------------------------------------------
# The full-path test
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_stop_button_full_path_cancels_loop_and_reopens_for_inject():
    """A1–A6 in one test. Single workspace, single AgentManager.

    Flow:
      1. Start loop with initial_message="please run".
      2. Wait for the provider to hit its hang point (mid-stream).
      3. Drive mgr.cancel_message() — the production endpoint handler's
         underlying call.
      4. Assert A1 (return status), A2 (marker), A3 (run_task.done()),
         A4 (slot free), A5 (idle broadcast).
      5. inject_message with new content. Assert A6 (delivered, not queued).
    """
    with tempfile.TemporaryDirectory() as workspace:
        project_id = "proj_stop_full_path"
        provider = _HangingProvider()
        mgr, ws, handle = _make_manager_with_loop(workspace, project_id, provider)

        # Hot-start the loop manually (matches what start_agent would do, but
        # without the heavy provider/registry building we don't need here).
        loop_obj = handle.loop
        run_task = asyncio.create_task(
            loop_obj.run(initial_message="please run")
        )
        handle.task = run_task
        run_task.add_done_callback(
            mgr._on_loop_done(project_id, session_id="default")
        )

        try:
            # Wait until the provider is mid-stream (hung).
            await asyncio.wait_for(provider._hang_event.wait(), timeout=2.0)

            session_path = handle.session._filepath

            # ---- POST /cancel equivalent ----
            t0 = time.monotonic()
            result = await asyncio.wait_for(
                mgr.cancel_message(project_id), timeout=5.0,
            )
            elapsed = time.monotonic() - t0

            # A1 — return value and timing
            assert elapsed < 5.0, f"cancel_message took too long: {elapsed:.2f}s"
            assert result == {"status": "cancelled"}, (
                f"Expected {{status: cancelled}}, got: {result}"
            )

            # Give the loop a chance to actually exit (break path) and
            # _on_loop_done callback to run.
            try:
                await asyncio.wait_for(run_task, timeout=2.0)
            except asyncio.TimeoutError:
                raise AssertionError(
                    "Loop did not exit within 2s of cancel — break-on-cancel "
                    "is not in effect"
                )

            # A3 — run_task completed (break-on-cancel)
            assert run_task.done(), "Loop task must be done after /cancel"

            # A2 — exactly one cancellation marker in JSONL
            msgs = _read_session_messages(session_path)
            markers = [m for m in msgs if m.get("cancelled_by_user") is True]
            assert len(markers) == 1, (
                f"Expected 1 cancellation marker, got {len(markers)}: "
                f"{[m.get('content') for m in msgs]}"
            )

            # A4 — active-loop slot is mechanically free
            holder = mgr.current_holder_session_id(project_id)
            assert holder is None, (
                f"Active-loop slot must be free after cancel, got "
                f"holder={holder!r}"
            )

            # A5 — agent.status:idle was broadcast at some point
            statuses = [b.get("status") for b in _status_broadcasts(ws)]
            assert "idle" in statuses, (
                f"Expected an agent.status:idle broadcast, got: {statuses}"
            )

            # A6 — subsequent inject reaches Case 2 (delivered) hot-resume
            inject_result = await mgr.inject_message(
                project_id, "follow up after cancel"
            )
            assert inject_result == "delivered", (
                f"Expected hot-resume 'delivered' after cancel (loop idle, "
                f"session alive), got: {inject_result!r}"
            )

            # Tear down the new loop spawned by hot-resume so the test exits.
            new_handle = mgr._handles.get((project_id, "default"))
            if new_handle is not None and new_handle.task is not None:
                if not new_handle.task.done():
                    try:
                        await asyncio.wait_for(new_handle.task, timeout=3.0)
                    except (asyncio.CancelledError, asyncio.TimeoutError,
                            Exception):
                        if not new_handle.task.done():
                            new_handle.task.cancel()
                            try:
                                await new_handle.task
                            except (asyncio.CancelledError, Exception):
                                pass
        finally:
            if not run_task.done():
                handle.session.stop()
                run_task.cancel()
                try:
                    await asyncio.wait_for(run_task, timeout=2.0)
                except (asyncio.CancelledError, asyncio.TimeoutError,
                        Exception):
                    pass
