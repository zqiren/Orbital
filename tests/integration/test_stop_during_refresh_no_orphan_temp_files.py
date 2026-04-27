# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Integration test: Stop sent during a state refresh → cancel propagates →
no orphan .tmp files left in workspace.

Scenario:
  - Agent loop is running; turn-count trigger fires at turn 15.
  - Refresh routine (run_session_end_routine) is artificially slowed with a
    sleep to create a window for the Stop to land mid-refresh.
  - loop.terminate() is called while the refresh is in-flight.
  - After the stop propagates:
    * No .tmp files remain in the workspace's orbital/ directory.
    * The WS event log shows the refresh terminal state is "failed", "skipped",
      or the refresh was cancelled (CancelledError) — but NOT stuck on "in_progress".
  - The loop exits cleanly (no deadlock, no stuck in-progress state).

This test FAILS on main (no refresh machinery, no _refresh_task cancel pathway)
and PASSES on the branch.
"""

from __future__ import annotations

import asyncio
import glob
import os
import tempfile
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent_os.agent import workspace_files as wsf_module
from agent_os.agent.loop import AgentLoop, COOLDOWN_TURNS
from agent_os.agent.project_paths import ProjectPaths
from agent_os.agent.providers.types import LLMResponse, TokenUsage
from agent_os.agent.session import Session
from agent_os.agent.tools.base import ToolResult
from agent_os.agent.workspace_files import WorkspaceFileManager


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _tool_response(n: int) -> LLMResponse:
    tc = [{"id": f"tc_{n}", "function": {"name": "read", "arguments": f'{{"n": {n}}}'}}]
    return LLMResponse(
        text="",
        tool_calls=tc,
        raw_message={"role": "assistant", "content": "", "tool_calls": tc},
        has_tool_calls=True,
        finish_reason="tool_calls",
        status_text=None,
        usage=TokenUsage(input_tokens=20, output_tokens=5),
    )


def _text_response(text: str = "Done.") -> LLMResponse:
    return LLMResponse(
        text=text,
        tool_calls=[],
        raw_message={"role": "assistant", "content": text},
        has_tool_calls=False,
        finish_reason="stop",
        status_text=None,
        usage=TokenUsage(input_tokens=10, output_tokens=5),
    )


def _build_ws_mock():
    ws = MagicMock()
    ws.broadcast = MagicMock()
    return ws


def _lifecycle_events(ws_mock):
    return [
        call.args[1]
        for call in ws_mock.broadcast.call_args_list
        if (len(call.args) >= 2
            and isinstance(call.args[1], dict)
            and call.args[1].get("type") == "state_refresh.lifecycle")
    ]


def _find_tmp_files(workspace: str) -> list[str]:
    """Find any .tmp files left in the workspace's orbital/ dir."""
    pp = ProjectPaths(workspace)
    orbital = pp.orbital_dir
    if not os.path.exists(orbital):
        return []
    return glob.glob(os.path.join(orbital, "*.tmp"))


class _TrackingContextManager:
    def __init__(self, session):
        self.model_context_limit = 128_000
        self.should_compact = MagicMock(return_value=False)
        self.usage_percentage = 0.0
        self._last_checkpoint_turn: int = 0
        self._last_checkpoint_ts: str = ""
        self._turns_since_last_update: int = 0

    def prepare(self):
        return [{"role": "system", "content": "You are helpful."}]


class _UniqueRegistry:
    def __init__(self):
        self._n = 0

    def reset_run_state(self):
        pass

    def schemas(self):
        return []

    def is_async(self, name):
        return False

    def execute(self, name, args):
        self._n += 1
        return ToolResult(content=f"r_{self._n}")

    async def execute_async(self, name, args):
        return ToolResult(content="ok")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_stop_during_refresh_leaves_no_tmp_files():
    """Stop mid-refresh → refresh task cancelled → no orphan .tmp files.

    The refresh routine is slowed with a sleep. terminate() is called while
    the sleep is active. After the loop exits, scan orbital/ for .tmp files.
    """
    total_turns = COOLDOWN_TURNS + 5  # enough to trigger turn-count at turn 15

    with tempfile.TemporaryDirectory() as workspace:
        pp = ProjectPaths(workspace)
        os.makedirs(pp.orbital_dir, exist_ok=True)

        project_id = "proj_stop_refresh"
        ws = _build_ws_mock()
        session = Session.new("sess-stop-refresh", workspace)
        session.is_paused = MagicMock(return_value=False)
        session._paused_for_approval = False
        session.pending_tool_calls = set()
        session.pop_queued_messages = MagicMock(return_value=[])
        session.resolve_pending_tool_calls = MagicMock()
        session.pop_deferred_messages = MagicMock(return_value=[])

        # Use a real is_stopped to allow terminate() to take effect
        _stopped = {"v": False}
        session.is_stopped = MagicMock(side_effect=lambda: _stopped["v"])

        def _stop():
            _stopped["v"] = True

        session.stop = MagicMock(side_effect=_stop)

        wfm = WorkspaceFileManager(workspace)
        wfm.ensure_dir()

        ctx = _TrackingContextManager(session)
        registry = _UniqueRegistry()

        refresh_started = asyncio.Event()
        terminate_during_refresh = asyncio.Event()

        from datetime import datetime, timezone

        async def slow_refresh_callback(trigger_name: str) -> None:
            ts = datetime.now(timezone.utc).isoformat()
            ws.broadcast(project_id, {
                "type": "state_refresh.lifecycle",
                "project_id": project_id,
                "status": "in_progress",
                "trigger": trigger_name,
                "timestamp": ts,
            })
            try:
                # Write a .tmp file to simulate mid-write state, then sleep
                tmp_path = os.path.join(pp.orbital_dir, "PROJECT_STATE.md.tmp")
                with open(tmp_path, "w") as f:
                    f.write("# partial write\n")

                refresh_started.set()
                # Slow sleep — gives the test coroutine time to call terminate()
                await asyncio.sleep(100)  # will be cancelled

                # Clean up tmp file on success (would not reach here if cancelled)
                if os.path.exists(tmp_path):
                    os.replace(tmp_path, os.path.join(pp.orbital_dir, "PROJECT_STATE.md"))

                ws.broadcast(project_id, {
                    "type": "state_refresh.lifecycle",
                    "project_id": project_id,
                    "status": "done",
                    "trigger": trigger_name,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                })
            except asyncio.CancelledError:
                # Clean up orphan tmp file on cancellation — this is the expected path
                tmp_path = os.path.join(pp.orbital_dir, "PROJECT_STATE.md.tmp")
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
                ws.broadcast(project_id, {
                    "type": "state_refresh.lifecycle",
                    "project_id": project_id,
                    "status": "failed",
                    "trigger": trigger_name,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                })
                raise
            except Exception:
                tmp_path = os.path.join(pp.orbital_dir, "PROJECT_STATE.md.tmp")
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
                ws.broadcast(project_id, {
                    "type": "state_refresh.lifecycle",
                    "project_id": project_id,
                    "status": "failed",
                    "trigger": trigger_name,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                })
                raise

        loop = AgentLoop(
            session=session,
            provider=MagicMock(),
            tool_registry=registry,
            context_manager=ctx,
            on_session_end_refresh=slow_refresh_callback,
            max_iterations=total_turns + 1,
        )

        call_n = {"n": 0}

        async def mock_stream(context, tool_schemas):
            call_n["n"] += 1
            if call_n["n"] > total_turns:
                return _text_response("Done.")
            return _tool_response(call_n["n"])

        loop._stream_response = mock_stream

        loop_task = asyncio.create_task(loop.run("Start task"))

        # Wait for the refresh to start (event set when refresh enters sleep)
        await asyncio.wait_for(refresh_started.wait(), timeout=15.0)

        # Call terminate() to cancel the in-flight refresh
        await loop.terminate()

        # Wait for the loop task to exit
        try:
            await asyncio.wait_for(loop_task, timeout=5.0)
        except (asyncio.TimeoutError, asyncio.CancelledError, Exception):
            pass

        # ---- Assertions ----

        # 1. No orphan .tmp files in the workspace
        tmp_files = _find_tmp_files(workspace)
        assert len(tmp_files) == 0, (
            f"No .tmp files should remain after Stop + cancel.\n"
            f"Found: {tmp_files}"
        )

        # 2. WS event log must not be stuck on in_progress (must have a terminal state)
        events = _lifecycle_events(ws)
        statuses = [e.get("status") for e in events]
        terminal_statuses = [s for s in statuses if s in ("done", "failed", "skipped")]
        assert len(terminal_statuses) >= 1, (
            f"WS events must have at least one terminal state (done/failed/skipped).\n"
            f"Statuses seen: {statuses}"
        )

        # 3. No event stuck at in_progress as the last status
        if statuses:
            assert statuses[-1] != "in_progress", (
                f"Last status must not be 'in_progress' (refresh must terminate).\n"
                f"Statuses: {statuses}"
            )


@pytest.mark.asyncio
async def test_stop_before_refresh_no_tmp_files():
    """Stop arrives before any refresh fires — no .tmp files, loop exits cleanly."""
    with tempfile.TemporaryDirectory() as workspace:
        pp = ProjectPaths(workspace)
        os.makedirs(pp.orbital_dir, exist_ok=True)

        project_id = "proj_stop_before"
        ws = _build_ws_mock()
        session = Session.new("sess-stop-before", workspace)
        session.is_paused = MagicMock(return_value=False)
        session._paused_for_approval = False
        session.pending_tool_calls = set()
        session.pop_queued_messages = MagicMock(return_value=[])
        session.resolve_pending_tool_calls = MagicMock()
        session.pop_deferred_messages = MagicMock(return_value=[])

        _stopped = {"v": False}
        session.is_stopped = MagicMock(side_effect=lambda: _stopped["v"])

        def _stop():
            _stopped["v"] = True

        session.stop = MagicMock(side_effect=_stop)

        wfm = WorkspaceFileManager(workspace)
        wfm.ensure_dir()

        ctx = _TrackingContextManager(session)
        registry = _UniqueRegistry()

        stream_started = asyncio.Event()

        async def refresh_callback(trigger_name: str) -> None:
            pass  # should not be called

        loop = AgentLoop(
            session=session,
            provider=MagicMock(),
            tool_registry=registry,
            context_manager=ctx,
            on_session_end_refresh=refresh_callback,
            max_iterations=100,  # no natural exit — we'll terminate early
        )

        call_n = {"n": 0}

        async def slow_stream(context, tool_schemas):
            call_n["n"] += 1
            stream_started.set()
            await asyncio.sleep(0.05)
            return _tool_response(call_n["n"])

        loop._stream_response = slow_stream

        loop_task = asyncio.create_task(loop.run("Start"))

        # Wait for stream to begin, then immediately terminate
        await asyncio.wait_for(stream_started.wait(), timeout=5.0)
        await loop.terminate()

        try:
            await asyncio.wait_for(loop_task, timeout=5.0)
        except (asyncio.TimeoutError, asyncio.CancelledError, Exception):
            pass

        # No .tmp files should exist
        tmp_files = _find_tmp_files(workspace)
        assert len(tmp_files) == 0, (
            f"No .tmp files expected when stopped before any refresh.\n"
            f"Found: {tmp_files}"
        )

        # No lifecycle events (refresh never fired)
        events = _lifecycle_events(ws)
        assert len(events) == 0, (
            f"No lifecycle events expected (refresh never fired).\n"
            f"Events: {events}"
        )


@pytest.mark.asyncio
async def test_cancel_propagates_to_refresh_task():
    """terminate() cancels the _refresh_task when it is in-flight.

    Verifies that loop._refresh_task receives CancelledError when terminate()
    is called during a refresh.
    """
    with tempfile.TemporaryDirectory() as workspace:
        pp = ProjectPaths(workspace)
        os.makedirs(pp.orbital_dir, exist_ok=True)

        project_id = "proj_cancel_propagate"
        ws = _build_ws_mock()
        session = Session.new("sess-cancel-prop", workspace)
        session.is_paused = MagicMock(return_value=False)
        session._paused_for_approval = False
        session.pending_tool_calls = set()
        session.pop_queued_messages = MagicMock(return_value=[])
        session.resolve_pending_tool_calls = MagicMock()
        session.pop_deferred_messages = MagicMock(return_value=[])

        _stopped = {"v": False}
        session.is_stopped = MagicMock(side_effect=lambda: _stopped["v"])

        def _stop():
            _stopped["v"] = True

        session.stop = MagicMock(side_effect=_stop)

        wfm = WorkspaceFileManager(workspace)
        wfm.ensure_dir()

        ctx = _TrackingContextManager(session)
        registry = _UniqueRegistry()

        refresh_started = asyncio.Event()
        refresh_cancelled = asyncio.Event()

        async def slow_refresh_callback(trigger_name: str) -> None:
            refresh_started.set()
            try:
                await asyncio.sleep(100)  # Will be cancelled
            except asyncio.CancelledError:
                refresh_cancelled.set()
                raise

        loop = AgentLoop(
            session=session,
            provider=MagicMock(),
            tool_registry=registry,
            context_manager=ctx,
            on_session_end_refresh=slow_refresh_callback,
            max_iterations=COOLDOWN_TURNS + 5,
        )

        call_n = {"n": 0}

        async def mock_stream(context, tool_schemas):
            call_n["n"] += 1
            return _tool_response(call_n["n"])

        loop._stream_response = mock_stream

        loop_task = asyncio.create_task(loop.run("go"))

        # Wait for refresh to start
        await asyncio.wait_for(refresh_started.wait(), timeout=15.0)

        # Terminate — should cancel the refresh task
        await loop.terminate()

        try:
            await asyncio.wait_for(loop_task, timeout=5.0)
        except (asyncio.TimeoutError, asyncio.CancelledError, Exception):
            pass

        # Refresh coroutine must have received CancelledError
        assert refresh_cancelled.is_set(), (
            "terminate() must propagate CancelledError to the in-flight refresh task"
        )

        # _refresh_task must be None or done (not stuck pending)
        assert loop._refresh_task is None or loop._refresh_task.done(), (
            "After terminate(), _refresh_task must be None or done"
        )
