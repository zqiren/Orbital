# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Unit tests: the loop's empty-final-response guard.

A turn may only end as ``text_complete`` when the model actually produced a
user-visible message. Before this guard, a response with NO tool calls and
whitespace-only text (e.g. MiniMax-M3 dying mid-think and returning only
reasoning_content) terminated the turn silently — the UI showed a thinking
capsule and then nothing (BACKLOG-m3-reasoning-only-silent-stall.md).

Contract pinned here:
- Empty/whitespace-only text + no tool calls → the loop persists the partial
  assistant message (KEEPING reasoning_content — it demonstrably helps the
  model resume), appends a system nudge, and re-asks the model.
- The nudge is bounded per run; on exhaustion the loop exits via the
  declared ``empty_response`` path with ``_llm_failed`` set — never silently.
- The guard keys on the turn contract (empty message + no handoff), NOT on
  finish_reason or any provider identity — provider-neutral by design.
- Normal text turns are unaffected.
"""

import pytest

from agent_os.agent.session import Session
from agent_os.agent.loop import AgentLoop
from agent_os.agent.context import ContextManager
from agent_os.agent.providers.types import StreamChunk, TokenUsage
from agent_os.agent.tools.base import ToolResult
from agent_os.agent.prompt_builder import PromptContext, Autonomy


def _make_base_prompt_context(workspace: str) -> PromptContext:
    return PromptContext(
        workspace=workspace,
        model="test-model",
        autonomy=Autonomy.HANDS_OFF,
        enabled_agents=[],
        tool_names=[],
        os_type="linux",
        datetime_now="2026-01-01T00:00:00",
        context_usage_pct=0.0,
    )


class MockPromptBuilder:
    def build(self, context: PromptContext) -> tuple[str, str, str]:
        return ("cached-system-prefix", "semi-stable-suffix", "dynamic-runtime")


class SimpleToolRegistry:
    def schemas(self) -> list[dict]:
        return []

    def is_async(self, name: str) -> bool:
        return False

    def execute(self, name: str, arguments: dict) -> ToolResult:
        return ToolResult(content="ok")

    async def execute_async(self, name: str, arguments: dict) -> ToolResult:
        return ToolResult(content="ok")

    def tool_names(self) -> list[str]:
        return []

    def reset_run_state(self) -> None:
        pass


class ScriptedProvider:
    """Provider that replays a scripted sequence of streams, then repeats
    the last entry. Each script entry is (text, reasoning, finish_reason)."""

    provider = "minimax"
    model = "MiniMax-M3"
    sdk = "openai"

    def __init__(self, script: list[tuple[str, str, str | None]]):
        self._script = script
        self.calls = 0

    async def stream(self, messages, tools=None):
        entry = self._script[min(self.calls, len(self._script) - 1)]
        self.calls += 1
        text, reasoning, finish_reason = entry
        yield StreamChunk(text=text, reasoning_content=reasoning)
        yield StreamChunk(
            is_final=True,
            usage=TokenUsage(input_tokens=100, output_tokens=50),
            finish_reason=finish_reason,
        )


def _make_loop(tmp_path, provider, session_id: str) -> tuple[AgentLoop, Session]:
    session = Session.new(session_id, str(tmp_path))
    context_mgr = ContextManager(
        session, MockPromptBuilder(), _make_base_prompt_context(str(tmp_path))
    )
    loop = AgentLoop(
        session, provider, SimpleToolRegistry(), context_mgr,
        project_dir=str(tmp_path),
        max_iterations=10,
    )
    return loop, session


def _system_messages(session: Session) -> list[str]:
    return [
        m.get("content") or ""
        for m in session.get_messages()
        if m.get("role") == "system"
    ]


@pytest.mark.asyncio
async def test_reasoning_only_response_is_nudged_then_recovers(tmp_path):
    """First response dies mid-think (whitespace text, reasoning only, cut by
    'length') → the loop must NOT end the turn; it nudges and the second,
    healthy response ends the turn normally."""
    provider = ScriptedProvider([
        ("\n\n", "thinking that was cut mid-sent", "length"),
        ("recovered answer", "", "stop"),
    ])
    loop, session = _make_loop(tmp_path, provider, "guard_recover")
    await loop.run(initial_message="do the thing")

    assert provider.calls == 2
    # The healthy answer ended the turn normally.
    assistants = [m for m in session.get_messages() if m.get("role") == "assistant"]
    assert assistants[-1]["content"] == "recovered answer"
    assert loop._loop_exit_path == "text_complete"
    # A nudge was recorded for the model to see.
    assert any("no visible text" in s or "truncated" in s
               for s in _system_messages(session))


@pytest.mark.asyncio
async def test_partial_reasoning_is_preserved_for_resumption(tmp_path):
    """The partial assistant message must keep its reasoning_content — the
    observed manual-'continue' recoveries show the model resumes from it."""
    provider = ScriptedProvider([
        ("\n\n", "thinking that was cut mid-sent", "length"),
        ("recovered answer", "", "stop"),
    ])
    loop, session = _make_loop(tmp_path, provider, "guard_reasoning")
    await loop.run(initial_message="do the thing")

    partials = [
        m for m in session.get_messages()
        if m.get("role") == "assistant"
        and m.get("reasoning_content") == "thinking that was cut mid-sent"
    ]
    assert len(partials) == 1


@pytest.mark.asyncio
async def test_persistent_empty_responses_exit_declared_not_silent(tmp_path):
    """When every retry comes back empty, the loop must end via the declared
    empty_response exit — bounded (initial + 2 nudges = 3 calls), flagged as
    an LLM failure, and with a system row explaining the give-up."""
    provider = ScriptedProvider([
        ("", "", None),  # empty forever — no reasoning, no finish_reason:
    ])                   # the guard keys on the contract, not the metadata
    loop, session = _make_loop(tmp_path, provider, "guard_exhaust")
    await loop.run(initial_message="do the thing")

    assert provider.calls == 3
    assert loop._loop_exit_path == "empty_response"
    assert loop._llm_failed is True
    assert any("empty" in s.lower() for s in _system_messages(session))


@pytest.mark.asyncio
async def test_normal_text_turn_is_unaffected(tmp_path):
    """Regression pin: a healthy text response ends the turn on the first
    call with no nudge rows."""
    provider = ScriptedProvider([("hello there", "", "stop")])
    loop, session = _make_loop(tmp_path, provider, "guard_normal")
    await loop.run(initial_message="hi")

    assert provider.calls == 1
    assert loop._loop_exit_path == "text_complete"
    assert not any("no visible text" in s for s in _system_messages(session))


@pytest.mark.asyncio
async def test_response_shape_diag_includes_finish_reason(tmp_path, caplog):
    """Observability: the per-iteration response_shape diagnostic must carry
    the real finish_reason so truncations are visible in the daemon log."""
    import json as _json
    import logging

    provider = ScriptedProvider([("hello there", "", "stop")])
    loop, _session = _make_loop(tmp_path, provider, "guard_diag")
    with caplog.at_level(logging.INFO, logger="agent.diag"):
        await loop.run(initial_message="hi")

    shapes = []
    for record in caplog.records:
        msg = record.getMessage()
        if '"response_shape"' in msg:
            shapes.append(_json.loads(msg[msg.index("{"):]))
    assert shapes, "no response_shape diagnostic emitted"
    assert shapes[-1].get("finish_reason") == "stop"
