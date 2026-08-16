# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Unit tests: the agent loop appends one token-ledger event per management
LLM response (Budget Piece 1, Task 2).

These exercise the *real* loop code path (AgentLoop.run) against a mock
provider, writing a real ledger file under a tmp project_dir, then read the
JSONL back. Budget Piece 2 deleted the legacy on_cost_update / budget_spent_usd
accumulator — the ledger is now the SOLE spend record, which these pin.
"""

import json
import os
from unittest.mock import MagicMock, patch

import pytest

from agent_os.agent.session import Session, persist_user_row
from agent_os.agent.loop import AgentLoop
from agent_os.agent.context import ContextManager
from agent_os.agent.providers.types import (
    StreamChunk,
    LLMResponse,
    TokenUsage,
)
from agent_os.agent.tools.base import ToolResult
from agent_os.agent.prompt_builder import PromptContext, Autonomy
from agent_os.budget.ledger import ledger_path


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


class TextProvider:
    """Single-turn provider with provider/model/sdk identity attributes."""

    def __init__(self, provider: str, model: str, sdk: str,
                 input_tokens: int = 100, output_tokens: int = 50,
                 cache_read_tokens: int = 0, cache_write_tokens: int = 0):
        self.provider = provider
        self.model = model
        self.sdk = sdk
        self._usage = TokenUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_read_tokens=cache_read_tokens,
            cache_write_tokens=cache_write_tokens,
        )

    async def stream(self, messages, tools=None):
        yield StreamChunk(text="hello from " + self.model)
        yield StreamChunk(is_final=True, usage=self._usage)


def _read_ledger(project_dir: str) -> list[dict]:
    p = ledger_path(project_dir)
    if not os.path.isfile(p):
        return []
    with open(p, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


@pytest.mark.asyncio
async def test_one_ledger_line_per_response(tmp_path):
    """A single management text response writes exactly one ledger line with
    the active provider's identity and disjoint (openai_compat) token fields."""
    session = Session.new("led_one", str(tmp_path))
    provider = TextProvider(
        "moonshot", "kimi-k2.5", "openai",
        input_tokens=100, output_tokens=50, cache_read_tokens=20,
    )
    builder = MockPromptBuilder()
    ctx = _make_base_prompt_context(str(tmp_path))
    context_mgr = ContextManager(session, builder, ctx)
    registry = SimpleToolRegistry()

    loop = AgentLoop(
        session, provider, registry, context_mgr,
        project_dir=str(tmp_path),
        max_iterations=5,
    )
    persist_user_row(loop._session, "hi")
    await loop.run()

    events = _read_ledger(str(tmp_path))
    assert len(events) == 1, events
    ev = events[0]
    assert ev["source"] == "management"
    assert ev["provider"] == "moonshot"
    assert ev["model"] == "kimi-k2.5"
    # openai_compat: uncached_input = input - cache_read = 100 - 20 = 80
    assert ev["uncached_input"] == 80
    assert ev["cache_read"] == 20
    assert ev["cache_write"] == 0
    assert ev["output"] == 50
    assert ev["session_id"] == session.session_id


@pytest.mark.asyncio
async def test_ledger_is_the_sole_spend_record(tmp_path):
    """Budget Piece 2: the legacy on_cost_update / budget_spent_usd accumulator
    is GONE. The token ledger is the SOLE spend record — one line per response,
    and the loop carries no dollar-accumulator attribute or constructor param."""
    session = Session.new("led_cost", str(tmp_path))
    provider = TextProvider("moonshot", "kimi-k2.5", "openai",
                            input_tokens=1000, output_tokens=500)
    builder = MockPromptBuilder()
    ctx = _make_base_prompt_context(str(tmp_path))
    context_mgr = ContextManager(session, builder, ctx)
    registry = SimpleToolRegistry()

    # The deleted legacy params no longer exist.
    with pytest.raises(TypeError):
        AgentLoop(
            session, provider, registry, context_mgr,
            project_dir=str(tmp_path),
            on_cost_update=lambda d, t: None,
            max_iterations=5,
        )

    loop = AgentLoop(
        session, provider, registry, context_mgr,
        project_dir=str(tmp_path),
        max_iterations=5,
    )
    persist_user_row(loop._session, "hi")
    await loop.run()

    # The ledger is the sole spend record: exactly one line written.
    assert len(_read_ledger(str(tmp_path))) == 1
    # No dollar-accumulator attribute on the loop.
    assert not hasattr(loop, "_budget_spent_usd")
    assert not hasattr(loop, "_on_cost_update")


@pytest.mark.asyncio
async def test_no_project_dir_no_ledger_no_crash(tmp_path):
    """When project_dir is not supplied, the loop must run normally and simply
    not write a ledger (no crash). Guards the optional-plumbing contract."""
    session = Session.new("led_none", str(tmp_path))
    provider = TextProvider("moonshot", "kimi-k2.5", "openai")
    builder = MockPromptBuilder()
    ctx = _make_base_prompt_context(str(tmp_path))
    context_mgr = ContextManager(session, builder, ctx)
    registry = SimpleToolRegistry()

    loop = AgentLoop(session, provider, registry, context_mgr, max_iterations=5)
    persist_user_row(loop._session, "hi")
    await loop.run()

    # No ledger file written.
    assert _read_ledger(str(tmp_path)) == []
    # Loop completed normally (a text-only assistant message exists).
    assert any(m.get("role") == "assistant" for m in session.get_messages())


@pytest.mark.asyncio
async def test_ledger_append_failure_does_not_break_loop(tmp_path, monkeypatch):
    """If the ledger append itself raises, the loop must still complete
    normally (resilience contract). The legacy cost path is gone — the ledger is
    the only spend writer, and its failure is swallowed."""
    session = Session.new("led_resilient", str(tmp_path))
    provider = TextProvider("moonshot", "kimi-k2.5", "openai",
                            input_tokens=100, output_tokens=50)
    builder = MockPromptBuilder()
    ctx = _make_base_prompt_context(str(tmp_path))
    context_mgr = ContextManager(session, builder, ctx)
    registry = SimpleToolRegistry()

    import agent_os.agent.loop as loop_mod

    def boom(*args, **kwargs):
        raise OSError("disk on fire")

    # Patch the symbol the loop actually calls.
    monkeypatch.setattr(loop_mod, "append_event", boom)

    loop = AgentLoop(
        session, provider, registry, context_mgr,
        project_dir=str(tmp_path),
        max_iterations=5,
    )
    # Must not raise even though append_event blows up.
    persist_user_row(loop._session, "hi")
    await loop.run()

    assert any(m.get("role") == "assistant" for m in session.get_messages())


# ---------------------------------------------------------------------------
# Pre-compaction memory flush: a real management-LLM response served via
# flush_llm.complete() (utility provider when configured, else the primary).
# It must also append a ledger line, attributed to the provider that actually
# served it. Trigger pattern mirrors tests/regression/test_precompaction_flush.py.
# ---------------------------------------------------------------------------

def _llm_response(text: str, usage, tool_calls=None) -> LLMResponse:
    tcs = tool_calls or []
    return LLMResponse(
        raw_message={"role": "assistant", "content": text,
                     **({"tool_calls": tcs} if tcs else {})},
        text=text,
        tool_calls=tcs,
        has_tool_calls=bool(tcs),
        finish_reason="tool_calls" if tcs else "stop",
        status_text=None,
        usage=usage,
    )


class UtilityCompleteProvider:
    """Utility provider serving the flush via complete(); distinct identity."""

    def __init__(self, usage):
        self.provider = "moonshot"
        self.model = "kimi-utility"
        self.sdk = "openai"
        self._usage = usage
        self.complete_calls = 0

    async def complete(self, messages, tools=None):
        self.complete_calls += 1
        return _llm_response("<silent>", self._usage)


def _flush_scenario_loop(tmp_path, session, utility):
    """Build a loop whose first iteration tool-calls, then compaction fires
    (flush turn via `utility`), then a final text ends the run."""
    primary = TextProvider("moonshot", "kimi-k2.5", "openai")

    context_manager = MagicMock()
    context_manager.prepare.return_value = [{"role": "system", "content": "t"}]
    context_manager.model_context_limit = 128_000
    compact_calls = {"n": 0}

    def should_compact_once():
        compact_calls["n"] += 1
        return compact_calls["n"] == 1

    context_manager.should_compact = MagicMock(side_effect=should_compact_once)

    registry = MagicMock()
    registry.schemas.return_value = []
    registry.is_async.return_value = False
    registry.execute.return_value = ToolResult(content="file contents")
    registry.reset_run_state = MagicMock()

    loop = AgentLoop(
        session, primary, registry, context_manager,
        utility_provider=utility,
        project_dir=str(tmp_path),
        max_iterations=10,
    )

    tc_list = [{"id": "tc_f1", "function": {"name": "read", "arguments": "{}"}}]
    responses = iter([
        _llm_response("", TokenUsage(input_tokens=100, output_tokens=50),
                      tool_calls=tc_list),
        _llm_response("Done.", TokenUsage(input_tokens=80, output_tokens=30)),
    ])

    async def mock_stream(context, tool_schemas):
        return next(responses)

    loop._stream_response = mock_stream
    return loop


@pytest.mark.asyncio
async def test_flush_completion_emits_ledger_line(tmp_path):
    """The pre-compaction flush response appends one ledger line attributed to
    the provider that actually served it (the utility provider), in between the
    two main-path lines from the primary."""
    session = Session.new("led_flush", str(tmp_path))
    utility = UtilityCompleteProvider(
        TokenUsage(input_tokens=500, output_tokens=20, cache_read_tokens=100),
    )
    loop = _flush_scenario_loop(tmp_path, session, utility)

    async def mock_compact_run(sess, prov, utility_provider=None):
        pass

    with patch("agent_os.agent.compaction.run", new=mock_compact_run):
        persist_user_row(loop._session, "do the task")
        await loop.run()

    assert utility.complete_calls == 1
    events = _read_ledger(str(tmp_path))
    assert len(events) == 3, events

    # Lines 1 and 3: main-path responses served by the primary.
    assert events[0]["model"] == "kimi-k2.5"
    assert events[0]["uncached_input"] == 100
    assert events[0]["output"] == 50
    assert events[2]["model"] == "kimi-k2.5"
    assert events[2]["uncached_input"] == 80
    assert events[2]["output"] == 30

    # Line 2: the flush, attributed to the utility provider that served it.
    assert events[1]["provider"] == "moonshot"
    assert events[1]["model"] == "kimi-utility"
    assert events[1]["source"] == "management"
    # openai_compat: uncached = 500 - 100 cache_read.
    assert events[1]["uncached_input"] == 400
    assert events[1]["cache_read"] == 100
    assert events[1]["output"] == 20


@pytest.mark.asyncio
async def test_flush_without_usage_emits_nothing(tmp_path):
    """If the flush response carries no usage, no flush ledger line is written
    (only the two main-path lines), and the loop completes normally."""
    session = Session.new("led_flush_nousage", str(tmp_path))
    utility = UtilityCompleteProvider(None)  # usage=None on the flush response
    loop = _flush_scenario_loop(tmp_path, session, utility)

    async def mock_compact_run(sess, prov, utility_provider=None):
        pass

    with patch("agent_os.agent.compaction.run", new=mock_compact_run):
        persist_user_row(loop._session, "do the task")
        await loop.run()

    assert utility.complete_calls == 1
    events = _read_ledger(str(tmp_path))
    assert len(events) == 2, events
    assert all(e["model"] == "kimi-k2.5" for e in events)
