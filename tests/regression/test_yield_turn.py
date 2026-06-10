# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""yield_turn: dispatching a sub-agent ends the management agent's turn.

When the management agent delivers a task to a sub-agent (agent_message
action="send"), the tool returns an ack result carrying meta {"yield_turn":
True}. The loop appends the result (pairing the tool_call_id) and then ends the
turn — the LLM never gets a chance to busy-poll the sub-agent (which previously
tripped the ping-pong guard, see REPORT-dispatch-yield-and-push.md and the
Quick Tasks investigation).

Synchronous queries (status/list/stop) and a bare start (a task-less launch —
the prompt's start→send pattern delivers the task via send) do NOT yield.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from agent_os.agent.providers.types import StreamChunk, LLMResponse, TokenUsage
from agent_os.agent.tools.base import ToolResult
from agent_os.agent.tools.agent_message import AgentMessageTool
from agent_os.agent.prompt_builder import PromptContext, Autonomy
from agent_os.agent.session import Session
from agent_os.agent.loop import AgentLoop
from agent_os.agent.context import ContextManager


# --- minimal loop harness (mirrors tests/unit/test_component_a.py) ----------

def _ctx(ws: str) -> PromptContext:
    return PromptContext(
        workspace=ws, model="test-model", autonomy=Autonomy.HANDS_OFF,
        enabled_agents=[], tool_names=["agent_message"], os_type="linux",
        datetime_now="2026-01-01T00:00:00", context_usage_pct=0.0,
    )


class _Builder:
    def build(self, context):
        return ("sys", "suffix", "runtime")


def _text(text: str) -> LLMResponse:
    return LLMResponse(raw_message={"role": "assistant", "content": text},
                       text=text, tool_calls=[], has_tool_calls=False,
                       finish_reason="stop", status_text=None,
                       usage=TokenUsage(10, 5))


def _tool(tc_id: str, name: str, args: str = "{}") -> LLMResponse:
    tc = {"id": tc_id, "type": "function",
          "function": {"name": name, "arguments": args}}
    return LLMResponse(raw_message={"role": "assistant", "content": None,
                                    "tool_calls": [tc]},
                       text="", tool_calls=[tc], has_tool_calls=True,
                       finish_reason="tool_calls", status_text=None,
                       usage=TokenUsage(10, 5))


class _Provider:
    def __init__(self, responses):
        self._responses = list(responses)
        self._idx = 0
        self.stream_call_count = 0

    async def stream(self, messages, tools=None):
        self.stream_call_count += 1
        idx = self._idx
        self._idx += 1
        if idx < len(self._responses):
            resp = self._responses[idx]
            if resp.text:
                yield StreamChunk(text=resp.text)
            for i, tc in enumerate(resp.tool_calls):
                d = dict(tc)
                d["index"] = i
                yield StreamChunk(tool_calls_delta=[d])
            yield StreamChunk(is_final=True, usage=resp.usage)
        else:
            yield StreamChunk(text="default", is_final=True, usage=TokenUsage(1, 1))


class _Registry:
    def __init__(self, results):
        self._results = results  # name -> ToolResult

    def schemas(self):
        return [{"type": "function", "function": {"name": n}} for n in self._results]

    def is_async(self, name):
        return False

    def execute(self, name, arguments):
        return self._results.get(name, ToolResult(content=f"result of {name}"))

    def tool_names(self):
        return list(self._results.keys())

    def reset_run_state(self):
        pass


def _loop(session, provider, registry, ws):
    return AgentLoop(session, provider, registry, ContextManager(session, _Builder(), _ctx(ws)))


# --- loop-level tests -------------------------------------------------------

@pytest.mark.asyncio
async def test_dispatch_yields_the_turn(tmp_path):
    """A tool result with meta yield_turn ends the turn after ONE LLM call."""
    session = Session.new("y1", str(tmp_path))
    provider = _Provider([_tool("abc123", "agent_message", '{"action":"send"}')])
    registry = _Registry({"agent_message": ToolResult(
        content="Dispatched to claude-code. Awaiting completion.",
        meta={"yield_turn": True})})
    loop = _loop(session, provider, registry, str(tmp_path))

    await loop.run(initial_message="delegate this")

    # Only one LLM call — the loop yielded instead of looping back to poll.
    assert provider.stream_call_count == 1
    # The dispatch result was appended (tool_call_id paired) and is not pending.
    msgs = session.get_messages()
    tool_results = [m for m in msgs if m.get("role") == "tool" and m.get("tool_call_id") == "abc123"]
    assert len(tool_results) == 1
    assert "Dispatched to claude-code" in tool_results[0]["content"]
    assert "abc123" not in session.pending_tool_calls


@pytest.mark.asyncio
async def test_non_yield_tool_result_continues_the_loop(tmp_path):
    """A tool result with NO yield_turn meta does not end the turn."""
    session = Session.new("y2", str(tmp_path))
    provider = _Provider([
        _tool("s1", "agent_message", '{"action":"status"}'),
        _text("the sub-agent is still running"),
    ])
    registry = _Registry({"agent_message": ToolResult(content="running")})  # no meta
    loop = _loop(session, provider, registry, str(tmp_path))

    await loop.run(initial_message="check status")

    # Two LLM calls — the loop continued after the non-yield result.
    assert provider.stream_call_count == 2


@pytest.mark.asyncio
async def test_tool_call_id_survives_yield(tmp_path):
    """The dispatch tool_call_id is paired (not CANCELLED) across the yield."""
    session = Session.new("y3", str(tmp_path))
    provider = _Provider([_tool("xyz789", "agent_message", '{"action":"send"}')])
    registry = _Registry({"agent_message": ToolResult(
        content="Dispatched to codex. Awaiting completion.",
        meta={"yield_turn": True})})
    loop = _loop(session, provider, registry, str(tmp_path))

    await loop.run(initial_message="go")

    tr = [m for m in session.get_messages()
          if m.get("role") == "tool" and m.get("tool_call_id") == "xyz789"]
    assert len(tr) == 1
    assert "Dispatched to codex" in tr[0]["content"]
    assert "CANCELLED" not in tr[0]["content"]
    assert "xyz789" not in session.pending_tool_calls


# --- tool-level tests (Step 1) ----------------------------------------------

def _tool_with_mock_mgr():
    mgr = MagicMock()
    mgr.start = AsyncMock(return_value="Started claude-code")
    mgr.send = AsyncMock(return_value="Message sent to claude-code. Transcript: /x")
    mgr.stop = AsyncMock(return_value="Stopped claude-code")
    mgr.status = MagicMock(return_value="running")
    mgr.list_active = MagicMock(return_value=[])
    return AgentMessageTool(sub_agent_manager=mgr, project_id="p")


@pytest.mark.asyncio
async def test_send_returns_yield_meta():
    tool = _tool_with_mock_mgr()
    result = await tool.execute(action="send", agent="claude-code", message="investigate X")
    assert result.meta is not None and result.meta.get("yield_turn") is True
    assert result.content  # non-empty ack the LLM reads on resume


@pytest.mark.asyncio
async def test_status_does_not_yield():
    tool = _tool_with_mock_mgr()
    result = await tool.execute(action="status", agent="claude-code")
    assert not (result.meta and result.meta.get("yield_turn"))


@pytest.mark.asyncio
async def test_list_and_stop_do_not_yield():
    tool = _tool_with_mock_mgr()
    for action in ("list", "stop"):
        result = await tool.execute(action=action, agent="claude-code")
        assert not (result.meta and result.meta.get("yield_turn")), action


@pytest.mark.asyncio
async def test_start_action_rejected_without_yield():
    """The start action no longer exists (send spawns-on-demand,
    TASK-collapse-dispatch-to-send) — a stray start must be rejected as an
    unknown action and must not yield. The task-less-start strand state this
    test used to guard against is now unreachable by construction."""
    tool = _tool_with_mock_mgr()
    result = await tool.execute(action="start", agent="claude-code")
    assert "unknown action" in result.content.lower()
    assert not (result.meta and result.meta.get("yield_turn"))
