# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Regression tests for Part 4D — per-session cache-hit telemetry.

The TASK's on-judgment CONTEXT.md update is an explicit empirical bet on the
prefix cache. It must not ship without the ability to measure cache impact at
session granularity. Per-call cache audit already exists in the provider; this
test pins the NEW session-level aggregation + session-end log line.
"""

from __future__ import annotations

import logging

import pytest

from agent_os.agent.providers.types import LLMResponse, TokenUsage
from agent_os.agent.session import Session
from agent_os.agent.loop import AgentLoop
from agent_os.agent.context import ContextManager
from agent_os.agent.prompt_builder import PromptContext, Autonomy
from agent_os.agent.providers.types import StreamChunk
from agent_os.agent.tools.base import ToolResult


def _base_ctx(workspace: str) -> PromptContext:
    return PromptContext(
        workspace=workspace, model="test-model", autonomy=Autonomy.HANDS_OFF,
        enabled_agents=[], tool_names=["read"], os_type="linux",
        datetime_now="2026-01-01T00:00:00", context_usage_pct=0.0,
    )


class _MockPromptBuilder:
    def build(self, context):
        return ("cached", "semi", "dynamic")


class _CacheProvider:
    """Streams a single text response carrying cache_read_tokens."""

    def __init__(self, input_tok, output_tok, cache_read_tok):
        self._usage = TokenUsage(
            input_tokens=input_tok, output_tokens=output_tok,
            cache_read_tokens=cache_read_tok,
        )

    async def stream(self, messages, tools=None):
        yield StreamChunk(text="all done")
        yield StreamChunk(is_final=True, usage=self._usage)


class _MockToolRegistry:
    def schemas(self):
        return []

    def is_async(self, name):
        return False

    def reset_run_state(self):
        pass


@pytest.mark.asyncio
async def test_loop_accumulates_session_cache_tokens(tmp_path):
    session = Session.new("cachesess", str(tmp_path))
    cm = ContextManager(session, _MockPromptBuilder(), _base_ctx(str(tmp_path)))
    loop = AgentLoop(
        session,
        _CacheProvider(input_tok=1000, output_tok=20, cache_read_tok=800),
        _MockToolRegistry(),
        cm,
    )

    await loop.run(initial_message="hi")

    assert loop._prompt_input_tokens_total == 1000
    assert loop._cache_read_tokens_total == 800


@pytest.mark.asyncio
async def test_loop_logs_session_cache_summary_at_run_end(tmp_path, caplog):
    session = Session.new("cachelog", str(tmp_path))
    cm = ContextManager(session, _MockPromptBuilder(), _base_ctx(str(tmp_path)))
    loop = AgentLoop(
        session,
        _CacheProvider(input_tok=1000, output_tok=20, cache_read_tok=750),
        _MockToolRegistry(),
        cm,
    )

    with caplog.at_level(logging.INFO, logger="orbital.cache_audit"):
        await loop.run(initial_message="hi")

    summary_lines = [r.getMessage() for r in caplog.records
                     if "CACHE_SESSION" in r.getMessage()]
    assert summary_lines, "expected a [CACHE_SESSION] aggregate log line at run end"
    # 750 / 1000 = 75.0%
    assert "75.0%" in summary_lines[-1]
