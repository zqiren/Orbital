# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Strict OpenAI-compatible upstreams reject a replayed assistant tool call
whose ``function.arguments`` is not valid JSON. DeepSeek via OpenCode Go
answers every request carrying one with 400 "Assistant tool call
function.arguments must be valid JSON" (probed live 2026-09-19; the same
history with ``"{}"`` answered 200).

Such a row gets persisted when a stream dies mid tool call: session
orbital-marketing_82b446b8 holds a GLM ``write`` call cut off at 1,275 chars
(no final usage chunk). The write tool already failed on it and said so in
the paired tool result, but every later turn on a strict model 400'd — the
session was unusable on DeepSeek while lenient models (GLM) kept working.

The repair lives at the wire boundary, like the empty-content and contiguity
repairs: unparseable arguments go out as ``"{}"`` (the Anthropic adapter's
existing fallback for the same case); the persisted session is untouched.
"""

import json

from agent_os.agent.providers.openai_compat import LLMProvider

TRUNCATED = '{"content":"# 朱啸虎说模型是水电丨那什么才是留得下来的？\\n\\n问题在于，agent 记不住'


def _provider():
    return LLMProvider("deepseek-v4-flash", "key",
                       base_url="https://opencode.ai/zen/go/v1", sdk="openai")


def _history(args):
    return [
        {"role": "user", "content": "好的"},
        {"role": "assistant", "content": "[STATUS: 按特工宇宙风格改写 v2]",
         "tool_calls": [{"id": "call_1", "type": "function",
                         "function": {"name": "write", "arguments": args}}]},
        {"role": "tool", "tool_call_id": "call_1",
         "content": "Error: could not write to ?: [Errno 21] Is a directory"},
        {"role": "user", "content": "continue"},
    ]


def _sent_args(out):
    return [tc["function"]["arguments"]
            for m in out if m.get("role") == "assistant"
            for tc in m.get("tool_calls") or []]


def test_truncated_arguments_go_out_as_valid_json():
    out = _provider()._prepare_messages_openai(_history(TRUNCATED))
    (args,) = _sent_args(out)
    assert json.loads(args) == {}


def test_valid_arguments_are_sent_byte_identical():
    raw = '{"path":"a.md", "content":"x"}'
    out = _provider()._prepare_messages_openai(_history(raw))
    assert _sent_args(out) == [raw]


def test_empty_and_missing_arguments_become_an_empty_object():
    for args in ("", None, "   "):
        out = _provider()._prepare_messages_openai(_history(args))
        assert _sent_args(out) == ["{}"], args


def test_non_object_json_is_left_alone():
    # Valid JSON is the upstream's contract; shape is the tool's business.
    out = _provider()._prepare_messages_openai(_history("[]"))
    assert _sent_args(out) == ["[]"]


def test_persisted_history_is_not_mutated():
    history = _history(TRUNCATED)
    _provider()._prepare_messages_openai(history)
    assert history[1]["tool_calls"][0]["function"]["arguments"] == TRUNCATED


def test_pairing_and_other_fields_survive_the_repair():
    out = _provider()._prepare_messages_openai(_history(TRUNCATED))
    tc = out[1]["tool_calls"][0]
    assert tc["id"] == "call_1" and tc["type"] == "function"
    assert tc["function"]["name"] == "write"
    assert out[2] == {"role": "tool", "tool_call_id": "call_1",
                      "content": "Error: could not write to ?: [Errno 21] Is a directory"}
