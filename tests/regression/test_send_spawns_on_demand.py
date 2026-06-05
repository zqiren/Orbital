# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""One dispatch verb: ``send`` spawns-on-demand; ``start`` leaves the tool surface.

Contract (TASK-collapse-dispatch-to-send):
    The management agent has exactly one dispatch verb. A ``send`` to a
    not-running sub-agent spawns it (via the unchanged SubAgentManager.start
    function) and dispatches in the same tool call. The ``start`` tool action
    no longer exists, so the spawned-but-untasked strand state
    (INVESTIGATION-orchestrator-start-without-send) is unreachable by
    construction.

Interplay hazards (both mandatory):
    H1 — a spawn-on-demand send consumes exactly ONE unit of the send budget
         and yields the turn exactly like a send to a running agent.
    H2 — the internal spawn must NOT double-announce: on_started's session
         injection is suppressed (announce=False -> inject=False); the
         piece-2 resume/honesty clause rides the dispatch ack instead.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent_os.agent.tools.agent_message import AgentMessageTool, MAX_DEPTH
from agent_os.daemon_v2.lifecycle_observer import LifecycleObserver
from agent_os.daemon_v2.models import make_session_key
from agent_os.daemon_v2.sub_agent_manager import SubAgentManager

pytestmark = pytest.mark.asyncio


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _IdleAdapter:
    """Minimal adapter the spawn side-effect registers."""

    def __init__(self):
        self._transport = None
        self._last_response = None
        self._background_send_task = None
        self._broken = False
        self._resume_status = ("fresh", "first_spawn")

    async def send(self, message):
        self._last_response = None


def _manager_with_spawnable(handle="claude-code", start_result=None):
    """SubAgentManager whose .start is a mock that registers an adapter,
    mimicking a successful spawn (the real start function is exercised by
    the live integration test)."""
    pm = MagicMock()
    pm._ws = MagicMock()
    pm.note_turn_closed = MagicMock()
    mgr = SubAgentManager(process_manager=pm)
    adapter = _IdleAdapter()

    async def _spawn(project_id, h, depth=0, *, session_id=None, announce=True):
        sk = make_session_key(project_id, mgr._resolve_session_id(session_id))
        mgr._adapters.setdefault(sk, {})[h] = adapter
        return start_result or f"Started Claude Code (fresh session — first spawn)"

    mgr.start = AsyncMock(side_effect=_spawn)
    transcript = MagicMock()
    transcript.filepath = "/tmp/t.jsonl"
    mgr._transcripts[("proj_x", handle)] = transcript
    return mgr, adapter


# ---------------------------------------------------------------------------
# Manager level: send spawns-on-demand
# ---------------------------------------------------------------------------


class TestSendSpawnsOnDemand:

    async def test_send_to_not_running_spawns_then_dispatches(self):
        mgr, adapter = _manager_with_spawnable()
        result = await mgr.send("proj_x", "claude-code", "do the task",
                                session_id="sess_x")
        # Spawned exactly once, with announcement suppressed (H2 contract)
        # and the dispatch's session identity threaded through.
        mgr.start.assert_awaited_once()
        kwargs = mgr.start.await_args.kwargs
        assert kwargs.get("session_id") == "sess_x"
        assert kwargs.get("announce") is False, (
            "internal spawn must suppress the standalone on_started "
            "injection (H2) — the dispatch ack is the one announcement"
        )
        # Dispatched, not errored.
        assert not result.startswith("Error"), result
        assert "Message sent to claude-code" in result

    async def test_dispatch_ack_carries_resume_clause(self):
        """Piece-2 honesty rides the dispatch ack now that start is internal."""
        mgr, adapter = _manager_with_spawnable()
        result = await mgr.send("proj_x", "claude-code", "go",
                                session_id="sess_x")
        assert "fresh session — first spawn" in result

    async def test_send_to_running_agent_does_not_spawn(self):
        mgr, adapter = _manager_with_spawnable()
        sk = make_session_key("proj_x", "sess_x")
        mgr._adapters[sk] = {"claude-code": adapter}
        result = await mgr.send("proj_x", "claude-code", "go",
                                session_id="sess_x")
        mgr.start.assert_not_awaited()
        assert "Message sent to claude-code" in result
        # No spawn -> no resume clause on the ack (status was reported when
        # the thread actually spawned).
        assert "first spawn" not in result

    async def test_spawn_failure_surfaces_as_error(self):
        pm = MagicMock()
        mgr = SubAgentManager(process_manager=pm)
        mgr.start = AsyncMock(return_value="Error: unknown agent 'nope'")
        result = await mgr.send("proj_x", "nope", "go", session_id="sess_x")
        assert result.startswith("Error")

    async def test_depth_forwarded_to_spawn(self):
        mgr, adapter = _manager_with_spawnable()
        await mgr.send("proj_x", "claude-code", "go", session_id="sess_x",
                       depth=2)
        assert mgr.start.await_args.kwargs.get("depth") == 2 or (
            len(mgr.start.await_args.args) >= 3
            and mgr.start.await_args.args[2] == 2
        )


# ---------------------------------------------------------------------------
# Tool level: start is gone; H1 budget/yield semantics
# ---------------------------------------------------------------------------


def _tool(mgr=None, depth=0, max_sends=10):
    mgr = mgr or MagicMock()
    tool = AgentMessageTool(
        sub_agent_manager=mgr, project_id="proj_x",
        depth=depth, session_id="sess_x", max_sends_per_run=max_sends,
    )
    return tool, mgr


class TestStartActionRemoved:

    async def test_start_action_is_unknown(self):
        tool, mgr = _tool()
        result = await tool.execute(action="start", agent="claude-code")
        assert "unknown action" in result.content.lower()
        mgr.start.assert_not_called()

    async def test_action_enum_no_longer_offers_start(self):
        tool, _ = _tool()
        assert "start" not in tool.parameters["properties"]["action"]["enum"]
        assert "send" in tool.parameters["properties"]["action"]["enum"]

    async def test_prompt_no_longer_advertises_start(self):
        from agent_os.agent.prompt_builder import Autonomy, PromptBuilder, PromptContext

        ctx = PromptContext(
            workspace="/tmp/w", model="m", autonomy=Autonomy.HANDS_OFF,
            enabled_agents=[
                {"handle": "claude-code", "display_name": "Claude Code"},
            ],
            tool_names=["agent_message"], os_type="macos",
            datetime_now="2026-06-05T00:00:00Z",
        )
        section = PromptBuilder()._sub_agents(ctx)
        assert 'action="start"' not in section
        assert 'action="send"' in section
        assert "spawns the agent automatically" in section


class TestH1DispatchBudget:

    async def test_spawn_on_demand_send_counts_as_one_dispatch(self):
        """H1: one send-with-internal-spawn = one budget unit + one yield."""
        mgr, adapter = _manager_with_spawnable()
        tool, _ = _tool(mgr=mgr, max_sends=2)
        result = await tool.execute(action="send", agent="claude-code",
                                    message="go")
        assert tool._send_count == 1
        assert (result.meta or {}).get("yield_turn") is True
        assert result.content.startswith("Dispatched to claude-code")
        # Second send still within budget; third trips the limit — proving
        # the internal spawn did not silently consume a unit.
        await tool.execute(action="send", agent="claude-code", message="more")
        result3 = await tool.execute(action="send", agent="claude-code",
                                     message="again")
        assert "send limit reached" in result3.content

    async def test_depth_limit_enforced_on_send_spawn(self):
        """The MAX_DEPTH guard moved from start to send: a max-depth agent
        cannot spawn deeper via send."""
        mgr = MagicMock()
        mgr.send = AsyncMock()
        tool, _ = _tool(mgr=mgr, depth=MAX_DEPTH)
        result = await tool.execute(action="send", agent="claude-code",
                                    message="go")
        assert "depth limit" in result.content
        mgr.send.assert_not_awaited()


# ---------------------------------------------------------------------------
# H2: no double-announce
# ---------------------------------------------------------------------------


class TestH2NoDoubleAnnounce:

    async def test_on_started_inject_false_broadcasts_without_injection(self):
        mock_am = MagicMock()
        mock_am.inject_system_message = AsyncMock()
        ws = MagicMock()
        observer = LifecycleObserver(agent_manager=mock_am, ws_manager=ws)
        await observer.on_started("proj_x", "claude-code",
                                  initiator="management_agent",
                                  transcript_path="/t", session_id="sess_x",
                                  inject=False)
        mock_am.inject_system_message.assert_not_awaited()
        assert any(c.args[1].get("type") == "sub_agent.started"
                   for c in ws.broadcast.call_args_list), (
            "WS broadcast must survive inject=False — the frontend status "
            "chip still needs the event"
        )

    async def test_start_wrapper_threads_announce_to_registry_path(self):
        """start(announce=False) must reach _start_from_registry."""
        pm = MagicMock()
        mgr = SubAgentManager(process_manager=pm, registry=MagicMock(),
                              setup_engine=MagicMock())
        mgr._start_from_registry = AsyncMock(return_value="Started X")
        await mgr.start("proj_x", "claude-code", session_id="sess_x",
                        announce=False)
        assert mgr._start_from_registry.await_args.kwargs.get(
            "announce") is False
