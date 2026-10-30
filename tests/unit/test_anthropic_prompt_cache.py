# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Anthropic-protocol caching needs explicit breakpoints; without them the
manager's hit rate was 0% (87 real turns, 4.8M tokens, none read).

Three markers: the static prompt, the end of the leading system rows, and the
last block before the per-call runtime tail. The tail changes on every call, so
a marker after it (top-level automatic caching) would write and never read.
"""

import json

import pytest

from agent_os.agent.context import _RUNTIME_MARKER, _RUNTIME_SEP
from agent_os.agent.providers.anthropic_adapter import translate_messages_to_anthropic

SYS = [{"role": "system", "content": "static prompt"},
       {"role": "system", "content": "semi stable"},
       {"role": "system", "content": "layer one"}]
TAIL = f"{_RUNTIME_MARKER}\ntime=1"


def _markers(t):
    out = []
    for b in t["system"]:
        if "cache_control" in b:
            out.append(("system", b["text"]))
    for m in t["messages"]:
        for b in (m["content"] if isinstance(m["content"], list) else []):
            if "cache_control" in b:
                out.append((m["role"], b.get("text") or b.get("tool_use_id") or b.get("id")))
    return out


def test_marks_the_static_prompt_the_end_of_system_and_the_last_row_before_the_tail():
    msgs = SYS + [{"role": "user", "content": "hello" + _RUNTIME_SEP + TAIL}]
    t = translate_messages_to_anthropic(msgs, cache=True)
    assert _markers(t) == [("system", "static prompt"), ("system", "layer one"), ("user", "hello")]
    assert t["messages"][-1]["content"][-1]["text"].endswith(TAIL)
    assert "cache_control" not in t["messages"][-1]["content"][-1]


def test_a_tool_result_tail_marks_the_result_not_the_runtime_row():
    msgs = SYS + [
        {"role": "user", "content": "go"},
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "t1", "type": "function", "function": {"name": "read", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "t1", "content": "file text"},
        {"role": "user", "content": TAIL},
    ]
    t = translate_messages_to_anthropic(msgs, cache=True)
    assert _markers(t)[-1] == ("user", "t1")
    assert t["messages"][-1]["content"] == TAIL


def test_the_marked_prefix_of_one_call_is_byte_identical_in_the_next():
    """What caching actually depends on."""
    def render(t):
        blocks = [("sys", b["text"]) for b in t["system"]]
        for m in t["messages"]:
            c = m["content"]
            for b in ([{"type": "text", "text": c}] if isinstance(c, str) else c):
                blocks.append((m["role"], json.dumps({k: v for k, v in b.items()
                                                      if k != "cache_control"}, sort_keys=True)))
        return blocks

    first = translate_messages_to_anthropic(
        SYS + [{"role": "user", "content": "hello" + _RUNTIME_SEP + TAIL}], cache=True)
    second = translate_messages_to_anthropic(
        SYS + [{"role": "user", "content": "hello"},
               {"role": "assistant", "content": "hi"},
               {"role": "user", "content": "more" + _RUNTIME_SEP + f"{_RUNTIME_MARKER}\ntime=2"}],
        cache=True)
    cached_through = render(first)[:-1]  # everything up to the runtime block
    assert render(second)[:len(cached_through)] == cached_through


def test_a_later_system_row_stays_behind_the_system_markers():
    msgs = SYS + [{"role": "user", "content": "a"},
                  {"role": "system", "content": "loop nudge"},
                  {"role": "user", "content": "b"}]
    t = translate_messages_to_anthropic(msgs, cache=True)
    assert [b["text"] for b in t["system"]][-1] == "loop nudge"
    assert "cache_control" not in t["system"][-1]
    assert sum("cache_control" in b for b in t["system"]) == 2


def test_never_more_than_four_markers_and_never_on_thinking_or_empty_text():
    msgs = SYS + [{"role": "user", "content": [
        {"type": "text", "text": "q"}, {"type": "text", "text": ""}]}]
    t = translate_messages_to_anthropic(msgs, cache=True)
    assert len(_markers(t)) <= 4
    assert _markers(t)[-1] == ("user", "q")


def test_a_short_conversation_with_one_system_row_has_one_system_marker():
    t = translate_messages_to_anthropic(
        [SYS[0], {"role": "user", "content": "hi"}], cache=True)
    assert _markers(t) == [("system", "static prompt"), ("user", "hi")]


def test_without_cache_the_shape_is_unchanged():
    t = translate_messages_to_anthropic(SYS + [{"role": "user", "content": "hi"}])
    assert t["system"] == "static prompt\nsemi stable\nlayer one"
    assert t["messages"] == [{"role": "user", "content": "hi"}]


@pytest.mark.asyncio
async def test_the_stream_path_sends_the_markers():
    from unittest.mock import AsyncMock, MagicMock
    from agent_os.agent.providers.openai_compat import LLMProvider

    p = LLMProvider("claude-x", "k", "https://example.com", sdk="anthropic")

    async def _empty():
        return
        yield

    p._anthropic_client = MagicMock()
    p._anthropic_client.messages.create = AsyncMock(return_value=_empty())
    async for _ in p.stream(SYS + [{"role": "user", "content": "hi"}]):
        pass
    sent = p._anthropic_client.messages.create.call_args.kwargs
    assert sent["system"][0]["cache_control"] == {"type": "ephemeral"}


def test_the_whole_prompt_size_counts_cached_tokens_on_the_anthropic_protocol():
    """Run limits and the cache-rate log read this; with caching on, Anthropic's
    input_tokens alone shrinks to the few uncached tokens."""
    from types import SimpleNamespace
    from agent_os.agent.loop import _whole_prompt_tokens
    from agent_os.agent.providers.types import TokenUsage

    usage = TokenUsage(input_tokens=14, output_tokens=5,
                       cache_read_tokens=11037, cache_write_tokens=15)
    assert _whole_prompt_tokens(SimpleNamespace(sdk="anthropic"), usage) == 11066
    assert _whole_prompt_tokens(SimpleNamespace(sdk="openai"), usage) == 14
