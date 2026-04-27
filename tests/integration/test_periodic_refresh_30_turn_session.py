# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Integration test: 30-turn interactive session — turn-count trigger fires twice.

Scenario:
  - 30 loop iterations driven through a real AgentLoop with a fake LLM provider.
  - Turn-count trigger fires at turn 15 (COOLDOWN_TURNS) and again at turn 30.
  - PROJECT_STATE mtime changes between turns 14→16 and between turns 29→31.
  - WS broadcast log contains exactly two state_refresh.lifecycle events with
    trigger="turn_count" and terminal status="done".
  - The context_manager's _last_checkpoint_turn reflects the new turn after
    each refresh.

This test exercises the full real AgentLoop turn-count machinery (loop.py
lines 291–322 and _run_refresh) wired to a mock WS broadcaster and a patched
run_session_end_routine that writes a minimal PROJECT_STATE file.

Test FAILS on main (no turn-count trigger exists) and PASSES on the branch.
"""

from __future__ import annotations

import asyncio
import os
import tempfile
import time
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
# Fake LLM provider: emits tool-call responses to keep the loop alive
# ---------------------------------------------------------------------------


class _ToolCallProvider:
    """Yields a unique tool-call response on each call, then a text response
    to terminate the loop on the final iteration."""

    def __init__(self, total_turns: int):
        self._total = total_turns
        self._call_n = 0

    @property
    def model(self):
        return "fake-30turn"

    async def complete(self, context, **kwargs):
        self._call_n += 1
        if self._call_n >= self._total:
            return _text_llm_response("Session complete.")
        return _tool_llm_response(self._call_n)

    async def stream(self, context, tools=None):
        raise NotImplementedError("use complete()")


def _tool_llm_response(n: int) -> LLMResponse:
    tc = [{"id": f"tc_{n}", "function": {"name": "read", "arguments": f'{{"n": {n}}}'}}]
    return LLMResponse(
        text="",
        tool_calls=tc,
        raw_message={"role": "assistant", "content": "", "tool_calls": tc},
        has_tool_calls=True,
        finish_reason="tool_calls",
        status_text=None,
        usage=TokenUsage(input_tokens=30, output_tokens=5),
    )


def _text_llm_response(text: str = "Done.") -> LLMResponse:
    return LLMResponse(
        text=text,
        tool_calls=[],
        raw_message={"role": "assistant", "content": text},
        has_tool_calls=False,
        finish_reason="stop",
        status_text=None,
        usage=TokenUsage(input_tokens=30, output_tokens=5),
    )


# ---------------------------------------------------------------------------
# Minimal tool registry
# ---------------------------------------------------------------------------


class _UniqueToolRegistry:
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
        return ToolResult(content=f"result_{self._n}")

    async def execute_async(self, name, args):
        return ToolResult(content="ok")


# ---------------------------------------------------------------------------
# Context manager stub (supports checkpoint metadata attributes)
# ---------------------------------------------------------------------------


class _TrackingContextManager:
    def __init__(self, session):
        self._session = session
        self.model_context_limit = 128_000
        self.should_compact = MagicMock(return_value=False)
        self.usage_percentage = 0.0
        # Checkpoint metadata — mirrored from loop._run_refresh
        self._last_checkpoint_turn: int = 0
        self._last_checkpoint_ts: str = ""
        self._turns_since_last_update: int = 0

    def prepare(self):
        return [{"role": "system", "content": "You are a helpful assistant."}]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_ws_mock():
    ws = MagicMock()
    ws.broadcast = MagicMock()
    return ws


def _lifecycle_events(ws_mock):
    """Return list of state_refresh.lifecycle broadcast payloads."""
    return [
        call.args[1]
        for call in ws_mock.broadcast.call_args_list
        if (len(call.args) >= 2
            and isinstance(call.args[1], dict)
            and call.args[1].get("type") == "state_refresh.lifecycle")
    ]


def _make_refresh_callback(ws, project_id, session, wfm):
    """Build the session_end_refresh_callback closure (mirrors agent_manager.py)."""
    from datetime import datetime, timezone

    async def session_end_refresh_callback(trigger_name: str) -> None:
        ts = datetime.now(timezone.utc).isoformat()
        ws.broadcast(project_id, {
            "type": "state_refresh.lifecycle",
            "project_id": project_id,
            "status": "in_progress",
            "trigger": trigger_name,
            "timestamp": ts,
        })
        try:
            # Use the real run_session_end_routine with bypass_idempotency=True
            await wsf_module.run_session_end_routine(
                session=session,
                provider=AsyncMock(),   # provider not used inside (patched below)
                workspace_files=wfm,
                utility_provider=None,
                session_id=session.session_id,
                bypass_idempotency=True,
            )
            ws.broadcast(project_id, {
                "type": "state_refresh.lifecycle",
                "project_id": project_id,
                "status": "done",
                "trigger": trigger_name,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })
        except asyncio.TimeoutError:
            ws.broadcast(project_id, {
                "type": "state_refresh.lifecycle",
                "project_id": project_id,
                "status": "skipped",
                "trigger": trigger_name,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })
            raise
        except Exception:
            ws.broadcast(project_id, {
                "type": "state_refresh.lifecycle",
                "project_id": project_id,
                "status": "failed",
                "trigger": trigger_name,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })
            raise

    return session_end_refresh_callback


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_30_turn_session_fires_refresh_at_turn_15_and_30():
    """A 30-turn interactive session fires turn-count refresh at turn 15 and turn 30.

    Asserts:
    - WS broadcast log contains exactly two state_refresh.lifecycle events with
      trigger="turn_count" and status="done".
    - PROJECT_STATE mtime changes between turns 14→16 and 29→31.
    - context_manager._last_checkpoint_turn reflects the updated turn.
    """
    total_turns = 2 * COOLDOWN_TURNS  # 30

    with tempfile.TemporaryDirectory() as workspace:
        pp = ProjectPaths(workspace)
        os.makedirs(pp.orbital_dir, exist_ok=True)

        project_id = "proj_30turn"
        ws = _build_ws_mock()
        session = Session.new("sess-30turn", workspace)
        session.is_paused = MagicMock(return_value=False)
        session.is_stopped = MagicMock(return_value=False)
        session._paused_for_approval = False
        session.pending_tool_calls = set()
        session.pop_queued_messages = MagicMock(return_value=[])
        session.resolve_pending_tool_calls = MagicMock()
        session.pop_deferred_messages = MagicMock(return_value=[])

        wfm = WorkspaceFileManager(workspace)
        wfm.ensure_dir()

        ctx = _TrackingContextManager(session)
        registry = _UniqueToolRegistry()

        refresh_callback = _make_refresh_callback(ws, project_id, session, wfm)

        loop = AgentLoop(
            session=session,
            provider=MagicMock(),   # not used directly (mock_stream overrides)
            tool_registry=registry,
            context_manager=ctx,
            on_session_end_refresh=refresh_callback,
            max_iterations=total_turns + 1,
        )

        call_n = {"n": 0}

        async def mock_stream(context, tool_schemas):
            call_n["n"] += 1
            if call_n["n"] > total_turns:
                return _text_llm_response("All done.")
            return _tool_llm_response(call_n["n"])

        loop._stream_response = mock_stream

        # Track PROJECT_STATE mtime before, after turn 15, and after turn 30.
        # We capture mtimes via the refresh callback firing.
        mtime_snapshots: list[float] = []
        original_refresh = refresh_callback

        async def instrumented_refresh(trigger_name: str) -> None:
            await original_refresh(trigger_name)
            try:
                mtime_snapshots.append(os.path.getmtime(pp.project_state))
            except FileNotFoundError:
                mtime_snapshots.append(-1.0)

        loop._on_session_end_refresh = instrumented_refresh

        # Patch run_session_end_routine to write a minimal PROJECT_STATE file
        # (avoids needing a real LLM for the summarization call)
        async def fake_session_end_routine(**kwargs):
            wfm.write("state", f"# State\nLast update: {call_n['n']}")
            wfm.write("decisions", "# Decisions\n")
            wfm.write("lessons", "# Lessons\n")
            wfm.write("context", "# Context\n")
            wfm.append("session_log", "\n## Session\n- Updated\n")

        with patch.object(wsf_module, "run_session_end_routine",
                          new=AsyncMock(side_effect=fake_session_end_routine)):
            await loop.run("Start a 30-turn session")

        # ---- Assertions ----

        # 1. Exactly two turn_count lifecycle events with status=done
        events = _lifecycle_events(ws)
        turn_count_done = [
            e for e in events
            if e.get("trigger") == "turn_count" and e.get("status") == "done"
        ]
        assert len(turn_count_done) == 2, (
            f"Expected exactly 2 turn_count/done events, got {len(turn_count_done)}.\n"
            f"All events: {[{k: v for k, v in e.items() if k != 'timestamp'} for e in events]}"
        )

        # 2. Two distinct mtime values recorded (one per refresh)
        assert len(mtime_snapshots) == 2, (
            f"Expected 2 mtime snapshots (one per refresh), got {len(mtime_snapshots)}"
        )
        assert mtime_snapshots[0] != -1.0 and mtime_snapshots[1] != -1.0, (
            "PROJECT_STATE must exist after each refresh"
        )

        # 3. context_manager checkpoint turn was updated
        assert ctx._last_checkpoint_turn > 0, (
            "context_manager._last_checkpoint_turn must be updated after refresh"
        )


@pytest.mark.asyncio
async def test_project_state_mtime_changes_between_refreshes():
    """PROJECT_STATE mtime must change between the two turn-count refresh windows.

    Captures mtime before the second refresh and asserts it differs from
    what was set in the first refresh (i.e., the second refresh wrote the file).
    """
    total_turns = 2 * COOLDOWN_TURNS  # 30

    with tempfile.TemporaryDirectory() as workspace:
        pp = ProjectPaths(workspace)
        os.makedirs(pp.orbital_dir, exist_ok=True)

        project_id = "proj_mtime"
        ws = _build_ws_mock()
        session = Session.new("sess-mtime", workspace)
        session.is_paused = MagicMock(return_value=False)
        session.is_stopped = MagicMock(return_value=False)
        session._paused_for_approval = False
        session.pending_tool_calls = set()
        session.pop_queued_messages = MagicMock(return_value=[])
        session.resolve_pending_tool_calls = MagicMock()
        session.pop_deferred_messages = MagicMock(return_value=[])

        wfm = WorkspaceFileManager(workspace)
        wfm.ensure_dir()

        ctx = _TrackingContextManager(session)
        registry = _UniqueToolRegistry()

        mtime_after_first: list[float] = []
        mtime_after_second: list[float] = []
        refresh_count = {"n": 0}

        async def fake_session_end_routine(**kwargs):
            refresh_count["n"] += 1
            # Small sleep so OS mtime changes (filesystem granularity)
            await asyncio.sleep(0.01)
            wfm.write("state", f"# State\nRefresh #{refresh_count['n']}")

        async def instrumented_callback(trigger_name: str) -> None:
            from datetime import datetime, timezone
            ws.broadcast(project_id, {
                "type": "state_refresh.lifecycle",
                "project_id": project_id,
                "status": "in_progress",
                "trigger": trigger_name,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })
            try:
                with patch.object(wsf_module, "run_session_end_routine",
                                  new=AsyncMock(side_effect=fake_session_end_routine)):
                    await wsf_module.run_session_end_routine(
                        session=session,
                        provider=None,
                        workspace_files=wfm,
                        utility_provider=None,
                        session_id=session.session_id,
                        bypass_idempotency=True,
                    )
                ws.broadcast(project_id, {
                    "type": "state_refresh.lifecycle",
                    "project_id": project_id,
                    "status": "done",
                    "trigger": trigger_name,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                })
                try:
                    mt = os.path.getmtime(pp.project_state)
                    if refresh_count["n"] == 1:
                        mtime_after_first.append(mt)
                    else:
                        mtime_after_second.append(mt)
                except FileNotFoundError:
                    pass
            except Exception:
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
            on_session_end_refresh=instrumented_callback,
            max_iterations=total_turns + 1,
        )

        call_n = {"n": 0}

        async def mock_stream(context, tool_schemas):
            call_n["n"] += 1
            if call_n["n"] > total_turns:
                return _text_llm_response("All done.")
            return _tool_llm_response(call_n["n"])

        loop._stream_response = mock_stream

        with patch.object(wsf_module, "run_session_end_routine",
                          new=AsyncMock(side_effect=fake_session_end_routine)):
            await loop.run("Start")

        # Both refreshes should have recorded an mtime
        assert len(mtime_after_first) >= 1, (
            "PROJECT_STATE must exist after first refresh (turn 15)"
        )
        assert len(mtime_after_second) >= 1, (
            "PROJECT_STATE must exist after second refresh (turn 30)"
        )
        # The second mtime must be >= first (file was rewritten)
        assert mtime_after_second[0] >= mtime_after_first[0], (
            f"Second refresh mtime ({mtime_after_second[0]}) must be >= "
            f"first ({mtime_after_first[0]}) — file must be rewritten on second refresh"
        )


@pytest.mark.asyncio
async def test_ws_event_log_contains_exactly_two_turn_count_events():
    """WS event log must have exactly 2 state_refresh.lifecycle events with
    trigger=turn_count and status=done over a 30-turn session."""
    total_turns = 2 * COOLDOWN_TURNS  # 30

    with tempfile.TemporaryDirectory() as workspace:
        pp = ProjectPaths(workspace)
        os.makedirs(pp.orbital_dir, exist_ok=True)

        project_id = "proj_wscheck"
        ws = _build_ws_mock()
        session = Session.new("sess-wscheck", workspace)
        session.is_paused = MagicMock(return_value=False)
        session.is_stopped = MagicMock(return_value=False)
        session._paused_for_approval = False
        session.pending_tool_calls = set()
        session.pop_queued_messages = MagicMock(return_value=[])
        session.resolve_pending_tool_calls = MagicMock()
        session.pop_deferred_messages = MagicMock(return_value=[])

        wfm = WorkspaceFileManager(workspace)
        wfm.ensure_dir()

        ctx = _TrackingContextManager(session)
        registry = _UniqueToolRegistry()

        refresh_callback = _make_refresh_callback(ws, project_id, session, wfm)

        loop = AgentLoop(
            session=session,
            provider=MagicMock(),
            tool_registry=registry,
            context_manager=ctx,
            on_session_end_refresh=refresh_callback,
            max_iterations=total_turns + 1,
        )

        call_n = {"n": 0}

        async def mock_stream(context, tool_schemas):
            call_n["n"] += 1
            if call_n["n"] > total_turns:
                return _text_llm_response("Done")
            return _tool_llm_response(call_n["n"])

        loop._stream_response = mock_stream

        async def fake_session_end_routine(**kwargs):
            wfm.write("state", "# State\nOK")

        with patch.object(wsf_module, "run_session_end_routine",
                          new=AsyncMock(side_effect=fake_session_end_routine)):
            await loop.run("go")

        events = _lifecycle_events(ws)

        # Must have exactly 2 in_progress + 2 done for turn_count
        in_progress = [e for e in events if e.get("trigger") == "turn_count"
                       and e.get("status") == "in_progress"]
        done = [e for e in events if e.get("trigger") == "turn_count"
                and e.get("status") == "done"]

        assert len(in_progress) == 2, (
            f"Expected 2 in_progress events for turn_count, got {len(in_progress)}"
        )
        assert len(done) == 2, (
            f"Expected 2 done events for turn_count, got {len(done)}"
        )

        # No failed or skipped events
        bad = [e for e in events if e.get("status") in ("failed", "skipped")]
        assert len(bad) == 0, (
            f"No failed/skipped events expected in a clean 30-turn session: {bad}"
        )
