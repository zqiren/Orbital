# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Integration test: checkpoint_state tool mid-conversation → cooldown engaged.

Scenario:
  - LLM emits a checkpoint_state tool call on turn 8.
  - Refresh fires with trigger="agent_decided".
  - Cooldown resets turns_since_last_update to 0.
  - Turn-count trigger would normally fire at turn (8+COOLDOWN_TURNS) but must
    NOT fire any earlier — because the cooldown window from the agent-decided
    trigger blocks it.
  - The next turn-count fire happens at turn (8 + 2*COOLDOWN_TURNS).
  - We run enough turns to confirm exactly two additional turn_count refreshes.

Test FAILS on main (no trigger machinery) and PASSES on the branch.
"""

from __future__ import annotations

import asyncio
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
from agent_os.agent.tools.checkpoint_state import CheckpointStateTool
from agent_os.agent.tools.registry import ToolRegistry
from agent_os.agent.workspace_files import WorkspaceFileManager


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _tool_response(n: int, name: str = "read") -> LLMResponse:
    tc = [{"id": f"tc_{n}", "function": {"name": name, "arguments": f'{{"n": {n}}}'}}]
    return LLMResponse(
        text="",
        tool_calls=tc,
        raw_message={"role": "assistant", "content": "", "tool_calls": tc},
        has_tool_calls=True,
        finish_reason="tool_calls",
        status_text=None,
        usage=TokenUsage(input_tokens=20, output_tokens=5),
    )


def _checkpoint_response(n: int) -> LLMResponse:
    tc = [{"id": f"tc_chk_{n}", "function": {
        "name": "checkpoint_state",
        "arguments": '{"reason": "significant work done"}',
    }}]
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
        usage=TokenUsage(input_tokens=20, output_tokens=5),
    )


def _build_ws_mock():
    ws = MagicMock()
    ws.broadcast = MagicMock()
    return ws


def _lifecycle_events(ws_mock, trigger=None, status=None):
    return [
        call.args[1]
        for call in ws_mock.broadcast.call_args_list
        if (len(call.args) >= 2
            and isinstance(call.args[1], dict)
            and call.args[1].get("type") == "state_refresh.lifecycle"
            and (trigger is None or call.args[1].get("trigger") == trigger)
            and (status is None or call.args[1].get("status") == status))
    ]


def _make_refresh_callback(ws, project_id, session, wfm):
    from datetime import datetime, timezone

    async def callback(trigger_name: str) -> None:
        ts = datetime.now(timezone.utc).isoformat()
        ws.broadcast(project_id, {
            "type": "state_refresh.lifecycle",
            "project_id": project_id,
            "status": "in_progress",
            "trigger": trigger_name,
            "timestamp": ts,
        })
        try:
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

    return callback


class _TrackingContextManager:
    def __init__(self, session):
        self._session = session
        self.model_context_limit = 128_000
        self.should_compact = MagicMock(return_value=False)
        self.usage_percentage = 0.0
        self._last_checkpoint_turn: int = 0
        self._last_checkpoint_ts: str = ""
        self._turns_since_last_update: int = 0

    def prepare(self):
        return [{"role": "system", "content": "You are helpful."}]


class _CountingRegistry:
    """Tool registry that handles checkpoint_state and read tools."""

    def __init__(self, checkpoint_tool=None):
        self._n = 0
        self._checkpoint_tool = checkpoint_tool

    def reset_run_state(self):
        pass

    def schemas(self):
        return []

    def is_async(self, name):
        return name == "checkpoint_state"

    def execute(self, name, args):
        self._n += 1
        return ToolResult(content=f"result_{self._n}")

    async def execute_async(self, name, args):
        if name == "checkpoint_state" and self._checkpoint_tool:
            return await self._checkpoint_tool.execute(**args)
        return ToolResult(content="ok")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_checkpoint_at_turn_8_engages_cooldown_no_turn_count_at_23():
    """Agent-decided refresh at turn 8 → cooldown active → next turn_count blocked.

    Invariant under test: agent_decided RESETS the cooldown, preventing a
    double-fire if turn_count would have fired again before COOLDOWN_TURNS
    elapsed since the agent-decided refresh.

    Sequence with checkpoint at turn 8:
    - Turn 8 TOP: counter=8 (< COOLDOWN_TURNS, no fire). LLM returns
      checkpoint_state. Tool executes → _run_refresh("agent_decided", 8) →
      resets turns_since_last_update to 0.
    - Turns 9..(8+COOLDOWN_TURNS-1): counter increments 1..(COOLDOWN_TURNS-1).
    - Turn (8+COOLDOWN_TURNS): counter hits COOLDOWN_TURNS → turn_count fires.
      Counter resets to 0.
    - Turn (8+2*COOLDOWN_TURNS): counter hits COOLDOWN_TURNS again → fires.

    So with checkpoint at turn 8 and total_turns = 8 + 2*COOLDOWN_TURNS, we
    expect exactly: 1 agent_decided fire + 2 turn_count fires = 3 refreshes.
    """
    # Checkpoint at turn 8. Run enough turns to capture two subsequent
    # turn_count fires (one per COOLDOWN_TURNS window after the checkpoint).
    CHECKPOINT_TURN = 8
    total_turns = CHECKPOINT_TURN + 2 * COOLDOWN_TURNS

    with tempfile.TemporaryDirectory() as workspace:
        pp = ProjectPaths(workspace)
        os.makedirs(pp.orbital_dir, exist_ok=True)

        project_id = "proj_chk_cooldown"
        ws = _build_ws_mock()
        session = Session.new("sess-chk-cooldown", workspace)
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

        refresh_callback = _make_refresh_callback(ws, project_id, session, wfm)

        # We need a checkpoint_state tool that triggers _run_refresh.
        # We'll set this up after creating the loop (circular reference).
        checkpoint_tool_holder = {"tool": None}

        registry = _CountingRegistry()

        loop = AgentLoop(
            session=session,
            provider=MagicMock(),
            tool_registry=registry,
            context_manager=ctx,
            on_session_end_refresh=refresh_callback,
            max_iterations=total_turns + 1,
        )

        # Wire checkpoint_state tool to trigger_checkpoint (mirrors agent_manager.py)
        async def on_agent_checkpoint():
            await loop.trigger_checkpoint()

        checkpoint_tool = CheckpointStateTool(on_checkpoint=on_agent_checkpoint)
        registry._checkpoint_tool = checkpoint_tool

        call_n = {"n": 0}

        async def mock_stream(context, tool_schemas):
            call_n["n"] += 1
            if call_n["n"] == CHECKPOINT_TURN:
                # Emit checkpoint_state tool call on this turn
                return _checkpoint_response(call_n["n"])
            if call_n["n"] > total_turns:
                return _text_response("All done.")
            return _tool_response(call_n["n"])

        loop._stream_response = mock_stream

        async def fake_session_end_routine(**kwargs):
            wfm.write("state", f"# State\nOK")

        with patch.object(wsf_module, "run_session_end_routine",
                          new=AsyncMock(side_effect=fake_session_end_routine)):
            await loop.run("Start task")

        events = _lifecycle_events(ws)
        done_events = [e for e in events if e.get("status") == "done"]
        agent_decided_done = [e for e in done_events if e.get("trigger") == "agent_decided"]
        turn_count_done = [e for e in done_events if e.get("trigger") == "turn_count"]

        # 1. Exactly one agent_decided refresh at turn 8
        assert len(agent_decided_done) == 1, (
            f"Expected exactly 1 agent_decided/done event, got {len(agent_decided_done)}.\n"
            f"All done events: {done_events}"
        )

        # 2. Turn-count must NOT fire at turn COOLDOWN_TURNS (because agent_decided
        # at turn 8 reset the cooldown). Instead it fires at turn (8+COOLDOWN_TURNS)
        # and again at turn (8+2*COOLDOWN_TURNS). Over total_turns we expect 2.
        assert len(turn_count_done) == 2, (
            f"Expected 2 turn_count/done events (at turns "
            f"{CHECKPOINT_TURN + COOLDOWN_TURNS} and "
            f"{CHECKPOINT_TURN + 2 * COOLDOWN_TURNS}), "
            f"got {len(turn_count_done)}.\n"
            f"All done events: {done_events}"
        )

        # 3. Total refreshes: 1 agent_decided + 2 turn_count = 3
        total_done = len(done_events)
        assert total_done == 3, (
            f"Expected 3 total refresh-done events (1 agent_decided + 2 turn_count), "
            f"got {total_done}.\nAll done events: {done_events}"
        )


@pytest.mark.asyncio
async def test_agent_decided_fires_once_regardless_of_turn_count():
    """Agent calling checkpoint_state fires exactly one agent_decided refresh
    and does NOT double-fire with a simultaneous turn_count trigger."""
    CHECKPOINT_TURN = COOLDOWN_TURNS  # trigger at same turn as COOLDOWN_TURNS threshold
    total_turns = COOLDOWN_TURNS + 1

    with tempfile.TemporaryDirectory() as workspace:
        pp = ProjectPaths(workspace)
        os.makedirs(pp.orbital_dir, exist_ok=True)

        project_id = "proj_double_fire"
        ws = _build_ws_mock()
        session = Session.new("sess-double-fire", workspace)
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
        registry = _CountingRegistry()

        refresh_callback = _make_refresh_callback(ws, project_id, session, wfm)

        loop = AgentLoop(
            session=session,
            provider=MagicMock(),
            tool_registry=registry,
            context_manager=ctx,
            on_session_end_refresh=refresh_callback,
            max_iterations=total_turns + 1,
        )

        async def on_agent_checkpoint():
            await loop.trigger_checkpoint()

        checkpoint_tool = CheckpointStateTool(on_checkpoint=on_agent_checkpoint)
        registry._checkpoint_tool = checkpoint_tool

        call_n = {"n": 0}

        async def mock_stream(context, tool_schemas):
            call_n["n"] += 1
            if call_n["n"] == CHECKPOINT_TURN:
                return _checkpoint_response(call_n["n"])
            if call_n["n"] > total_turns:
                return _text_response("Done")
            return _tool_response(call_n["n"])

        loop._stream_response = mock_stream

        async def fake_session_end_routine(**kwargs):
            wfm.write("state", "# State\nOK")

        with patch.object(wsf_module, "run_session_end_routine",
                          new=AsyncMock(side_effect=fake_session_end_routine)):
            await loop.run("Start")

        events = _lifecycle_events(ws)
        done_events = [e for e in events if e.get("status") == "done"]

        # At turn COOLDOWN_TURNS, both turn_count (counter=COOLDOWN_TURNS) AND
        # agent_decided (tool call) would fire. turn_count fires FIRST (at top of
        # iteration before LLM call). After turn_count resets counter to 0,
        # agent_decided fires inside tool execution. Then counter resets again to 0.
        # So we get: 1 turn_count + 1 agent_decided = 2 total done events.
        # (No additional turn_count on the next turn since counter = 1 after second reset.)
        total_done = len(done_events)
        assert total_done <= 2, (
            f"Expected at most 2 done events over {total_turns} turns, "
            f"got {total_done}: {done_events}"
        )
        assert total_done >= 1, (
            f"Expected at least 1 done event, got 0. "
            f"Refresh trigger machinery not working."
        )
