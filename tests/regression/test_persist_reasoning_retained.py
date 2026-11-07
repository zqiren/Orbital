# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Regression (RULE 1): reasoning_content must survive persist.

Model reasoning was moved out of message ``content`` into a dedicated
``reasoning_content`` field. The text-only assistant persist path in
``loop.py`` built a fresh ``{"role","content","source"}`` dict that DROPPED
reasoning. (A second such path, the pre-compaction memory-flush text-only
branch, was removed with the flush turn on 2026-10-21.)

The path must carry ``reasoning_content`` through to the session so a worked
turn that produced little/no visible answer still renders a reasoning capsule
(the locked product decision: silent-vanish is unacceptable).

These tests drive a real ``AgentLoop`` with a stub provider whose stream emits
reasoning, then inspect what gets appended to the session.
"""

import asyncio

import pytest

from agent_os.agent.providers.types import (
    StreamChunk,
    LLMResponse,
    TokenUsage,
)
from agent_os.agent.tools.base import ToolResult
from agent_os.agent.prompt_builder import PromptContext, Autonomy
from agent_os.agent.session import Session, persist_user_row
from agent_os.agent.loop import AgentLoop, _EMPTY_RESPONSE_NUDGE_LIMIT
from agent_os.agent.context import ContextManager


def _ctx(workspace: str) -> PromptContext:
    return PromptContext(
        workspace=workspace,
        model="test-model",
        autonomy=Autonomy.HANDS_OFF,
        enabled_agents=[],
        tool_names=["write"],
        os_type="linux",
        datetime_now="2026-01-01T00:00:00",
        context_usage_pct=0.0,
    )


class _Builder:
    def build(self, context):
        return ("cached-prefix", "semi-stable", "dynamic")


class _Registry:
    def schemas(self):
        return [{"type": "function", "function": {"name": "write"}}]

    def is_async(self, name):
        return False

    def execute(self, name, arguments):
        return ToolResult(content="ok")

    def tool_names(self):
        return ["write"]

    def reset_run_state(self):
        pass


def _text_only_stream_with_reasoning(reasoning: str, answer: str):
    """A streamed text-only turn: reasoning deltas (empty text) then answer."""
    async def gen(messages, tools=None):
        # Reasoning-only deltas — empty text, non-empty reasoning_content.
        yield StreamChunk(text="", reasoning_content=reasoning)
        if answer:
            yield StreamChunk(text=answer)
        yield StreamChunk(is_final=True, usage=TokenUsage(10, 5))
    return gen


@pytest.mark.asyncio
async def test_text_only_turn_retains_reasoning(tmp_path):
    """loop.py ~609-615: a streamed text-only assistant turn that reasoned must
    persist reasoning_content (not drop it onto a fresh content-only dict)."""
    session = Session.new("reasontextsess", str(tmp_path))

    class _Provider:
        def __init__(self):
            self._gen = _text_only_stream_with_reasoning(
                "let me think about this carefully", "Final answer."
            )

        def stream(self, messages, tools=None):
            return self._gen(messages, tools)

    loop = AgentLoop(
        session, _Provider(), _Registry(),
        ContextManager(session, _Builder(), _ctx(str(tmp_path))),
        memory_record_enabled=False,
    )
    persist_user_row(loop._session, "please answer")
    await loop.run()

    assistant = [m for m in session.get_messages() if m["role"] == "assistant"]
    assert len(assistant) == 1, assistant
    msg = assistant[0]
    assert msg.get("content") == "Final answer."
    assert msg.get("reasoning_content") == "let me think about this carefully", (
        "text-only persist dropped reasoning_content: %r" % msg
    )


@pytest.mark.asyncio
async def test_text_only_turn_with_no_answer_retains_reasoning(tmp_path):
    """THE LANDMINE (loop.py ~609-615): a turn that reasoned but produced NO
    visible answer text must still persist reasoning_content. Otherwise the row
    is content-empty + reasoning-dropped + no tool_calls and renders as nothing
    (silent-vanish)."""
    session = Session.new("reasononlysess", str(tmp_path))

    class _Provider:
        def __init__(self):
            self._gen = _text_only_stream_with_reasoning(
                "thought hard but nothing to say", ""
            )

        def stream(self, messages, tools=None):
            return self._gen(messages, tools)

    loop = AgentLoop(
        session, _Provider(), _Registry(),
        ContextManager(session, _Builder(), _ctx(str(tmp_path))),
        memory_record_enabled=False,
    )
    persist_user_row(loop._session, "please answer")
    await loop.run()

    # Each empty reply is persisted as a partial row before the nudge, so the
    # stub's reasoning-only reply lands once per attempt.
    assistant = [m for m in session.get_messages() if m["role"] == "assistant"]
    assert len(assistant) == _EMPTY_RESPONSE_NUDGE_LIMIT + 1, assistant
    for msg in assistant:
        assert msg.get("reasoning_content") == "thought hard but nothing to say", (
            "reasoning-only persist dropped reasoning_content (silent-vanish): %r" % msg
        )
