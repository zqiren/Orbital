# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Regression: the token ledger attributes each response to the provider/model
that ACTUALLY served it — post-rotation, never the project's configured model.

The agent loop auto-rotates providers on transient LLM errors. If the ledger
read the project's configured (primary) model, every rotated call would be
mis-attributed. This test simulates a rotation mid-run and asserts the ledger
lines before and after rotation carry different ``provider``/``model`` values
matching what served each call.

Scenario:
  iteration 1 — primary succeeds with a tool call  -> ledger line #1 = PRIMARY
  iteration 2 — primary raises a transient (503) error
                 loop rotates to the fallback provider
                 fallback succeeds with a final text -> ledger line #2 = FALLBACK
"""

import json
import os

import pytest

from agent_os.agent.session import Session
from agent_os.agent.loop import AgentLoop
from agent_os.agent.context import ContextManager
from agent_os.agent.providers.types import (
    StreamChunk,
    TokenUsage,
    LLMError,
)
from agent_os.agent.tools.base import ToolResult
from agent_os.agent.prompt_builder import PromptContext, Autonomy
from agent_os.budget.ledger import ledger_path


def _make_base_prompt_context(workspace: str) -> PromptContext:
    return PromptContext(
        workspace=workspace,
        model="primary-model",  # the project's "configured" model
        autonomy=Autonomy.HANDS_OFF,
        enabled_agents=[],
        tool_names=["noop"],
        os_type="linux",
        datetime_now="2026-01-01T00:00:00",
        context_usage_pct=0.0,
    )


class MockPromptBuilder:
    def build(self, context: PromptContext) -> tuple[str, str, str]:
        return ("cached-system-prefix", "semi-stable-suffix", "dynamic-runtime")


class NoopToolRegistry:
    def schemas(self) -> list[dict]:
        return [{"type": "function", "function": {"name": "noop"}}]

    def is_async(self, name: str) -> bool:
        return False

    def execute(self, name: str, arguments: dict) -> ToolResult:
        return ToolResult(content="ok")

    async def execute_async(self, name: str, arguments: dict) -> ToolResult:
        return ToolResult(content="ok")

    def tool_names(self) -> list[str]:
        return ["noop"]

    def reset_run_state(self) -> None:
        pass


class PrimaryToolThenError:
    """Primary: first call returns a tool-call response (continues the loop),
    second call raises a transient 503 (forces rotation)."""

    def __init__(self):
        self.provider = "moonshot"
        self.model = "primary-model"
        self.sdk = "openai"
        self._calls = 0

    async def stream(self, messages, tools=None):
        self._calls += 1
        if self._calls == 1:
            tc = {
                "id": "call_1",
                "type": "function",
                "function": {"name": "noop", "arguments": "{}"},
                "index": 0,
            }
            yield StreamChunk(tool_calls_delta=[tc])
            yield StreamChunk(
                is_final=True,
                usage=TokenUsage(input_tokens=200, output_tokens=40, cache_read_tokens=0),
            )
            return
        # Second call: transient failure -> loop rotates.
        raise LLMError("service unavailable", status_code=503)


class FallbackSuccess:
    """Fallback: succeeds with a final text response; distinct identity."""

    def __init__(self):
        self.provider = "anthropic"
        self.model = "fallback-model"
        self.sdk = "anthropic"
        self._calls = 0

    async def stream(self, messages, tools=None):
        self._calls += 1
        yield StreamChunk(text="done from fallback")
        yield StreamChunk(
            is_final=True,
            usage=TokenUsage(
                input_tokens=300, output_tokens=70,
                cache_read_tokens=10, cache_write_tokens=5,
            ),
        )


def _read_ledger(project_dir: str) -> list[dict]:
    p = ledger_path(project_dir)
    if not os.path.isfile(p):
        return []
    with open(p, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


@pytest.mark.asyncio
async def test_ledger_attributes_post_rotation_provider(tmp_path):
    session = Session.new("led_rot", str(tmp_path))
    primary = PrimaryToolThenError()
    fallback = FallbackSuccess()

    builder = MockPromptBuilder()
    ctx = _make_base_prompt_context(str(tmp_path))
    context_mgr = ContextManager(session, builder, ctx)
    registry = NoopToolRegistry()

    loop = AgentLoop(
        session, primary, registry, context_mgr,
        fallback_providers=[fallback],
        project_dir=str(tmp_path),
        max_iterations=10,
    )
    await loop.run(initial_message="go")

    # The loop must have rotated and completed via the fallback.
    assert not loop._llm_failed
    assert fallback._calls >= 1

    events = _read_ledger(str(tmp_path))
    assert len(events) == 2, events

    before, after = events[0], events[1]

    # Line #1 attributed to the PRIMARY that actually served iteration 1.
    assert before["provider"] == "moonshot"
    assert before["model"] == "primary-model"
    assert before["uncached_input"] == 200  # openai_compat, cache_read=0
    assert before["output"] == 40

    # Line #2 attributed to the FALLBACK that actually served post-rotation.
    assert after["provider"] == "anthropic"
    assert after["model"] == "fallback-model"
    # anthropic additive semantics: uncached_input copied through as-is.
    assert after["uncached_input"] == 300
    assert after["cache_read"] == 10
    assert after["cache_write"] == 5
    assert after["output"] == 70

    # The two lines carry DIFFERENT provider/model — proves post-rotation
    # attribution, not the configured-model mis-attribution.
    assert before["provider"] != after["provider"]
    assert before["model"] != after["model"]
