# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Regression (RULE 1): reasoning_content must survive persist.

Model reasoning was moved out of message ``content`` into a dedicated
``reasoning_content`` field. Two backend persist paths in ``loop.py`` built a
fresh ``{"role","content","source"}`` dict that DROPPED reasoning:

  - the text-only assistant turn (``loop.py`` ~609-615), and
  - the pre-compaction memory-flush text-only branch (``loop.py`` ~1022-1026).

Both paths must carry ``reasoning_content`` through to the session so a worked
turn that produced little/no visible answer still renders a reasoning capsule
(the locked product decision: silent-vanish is unacceptable).

These tests drive a real ``AgentLoop`` with a stub provider whose stream/complete
emit reasoning, then inspect what gets appended to the session.
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
from agent_os.agent.loop import AgentLoop
from agent_os.agent.context import ContextManager
from agent_os.agent import compaction as compaction_mod


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
    )
    persist_user_row(loop._session, "please answer")
    await loop.run()

    assistant = [m for m in session.get_messages() if m["role"] == "assistant"]
    assert len(assistant) == 1, assistant
    msg = assistant[0]
    assert msg.get("reasoning_content") == "thought hard but nothing to say", (
        "reasoning-only persist dropped reasoning_content (silent-vanish): %r" % msg
    )


@pytest.mark.asyncio
async def test_compaction_flush_text_only_retains_reasoning(tmp_path, monkeypatch):
    """loop.py ~1022-1026: the pre-compaction memory-flush text-only branch
    builds a fresh dict that drops reasoning_content. The flush response is a
    text-only LLMResponse whose raw_message carries reasoning_content; it must
    survive persist.

    We drive: turn 1 = a tool call (loop continues), then force should_compact()
    True so the flush branch fires with a reasoning-bearing text-only flush
    response. compaction_mod.run / inject_reorientation are stubbed so only the
    flush-persist behaviour is exercised; turn 2 is a plain text turn to exit.
    """
    session = Session.new("flushreasonsess", str(tmp_path))

    flush_reasoning = "deciding what state to checkpoint"

    class _Provider:
        """stream() drives the main loop; complete() serves the flush turn."""

        def __init__(self):
            self._stream_idx = 0

        def stream(self, messages, tools=None):
            idx = self._stream_idx
            self._stream_idx += 1

            async def gen():
                if idx == 0:
                    # Turn 1: a tool call so the loop continues to the
                    # bottom-of-loop compaction check.
                    yield StreamChunk(tool_calls_delta=[{
                        "index": 0,
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "write", "arguments": "{}"},
                    }])
                    yield StreamChunk(is_final=True, usage=TokenUsage(10, 5))
                else:
                    # Turn 2 (after compaction): plain text to exit.
                    yield StreamChunk(text="done")
                    yield StreamChunk(is_final=True, usage=TokenUsage(10, 5))

            return gen()

        async def complete(self, messages, tools=None, *, disable_reasoning=False):
            # The pre-compaction memory flush. Text-only response WITH reasoning.
            return LLMResponse(
                raw_message={
                    "role": "assistant",
                    "content": "Saved state to PROJECT_STATE.md",
                    "reasoning_content": flush_reasoning,
                },
                text="Saved state to PROJECT_STATE.md",
                tool_calls=[],
                has_tool_calls=False,
                finish_reason="stop",
                status_text=None,
                usage=TokenUsage(10, 5),
            )

    context_mgr = ContextManager(session, _Builder(), _ctx(str(tmp_path)))

    # Force compaction to fire exactly once (after the first tool turn).
    compact_calls = {"n": 0}
    orig_should = context_mgr.should_compact

    def _should_compact():
        compact_calls["n"] += 1
        return compact_calls["n"] == 1

    monkeypatch.setattr(context_mgr, "should_compact", _should_compact)

    # Isolate the flush-persist behaviour: stub the heavy compaction machinery.
    async def _noop_run(*args, **kwargs):
        return None

    monkeypatch.setattr(compaction_mod, "run", _noop_run)
    monkeypatch.setattr(compaction_mod, "inject_reorientation", lambda *a, **k: None)

    loop = AgentLoop(session, _Provider(), _Registry(), context_mgr)
    persist_user_row(loop._session, "work then compact")
    await loop.run()

    # The flush turn's text-only assistant message must carry reasoning.
    flush_msgs = [
        m for m in session.get_messages()
        if m["role"] == "assistant"
        and m.get("content") == "Saved state to PROJECT_STATE.md"
    ]
    assert len(flush_msgs) == 1, [m for m in session.get_messages()]
    assert flush_msgs[0].get("reasoning_content") == flush_reasoning, (
        "compaction-flush persist dropped reasoning_content: %r" % flush_msgs[0]
    )
