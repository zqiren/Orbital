# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Strict OpenAI-compatible providers (MiniMax api.minimaxi.com) reject any
chat request that has empty/blank content or no non-system turn, returning
400 "invalid params, chat content is empty (2013)".

A fresh agent's very first onboarding turn is system-prompt-driven: the
prepared payload contains ONLY role:"system" messages and no user/assistant
turn (see ContextManager.prepare + PromptBuilder._onboarding_or_directive).
Lenient providers answer that; MiniMax 400s.

The fix sanitizes outbound messages at the openai_compat wire boundary
(_prepare_messages_openai → _ensure_chat_content) WITHOUT mutating the
persisted session: blank user/system content is placeholdered, and a minimal
user kickoff turn is appended when there is no user/assistant message at all.
"""

from types import SimpleNamespace

import pytest

from agent_os.agent.providers.openai_compat import (
    LLMProvider,
    _ensure_chat_content,
    _content_is_blank,
)


def _provider():
    return LLMProvider("MiniMax-M2.5", "key", base_url="https://api.minimaxi.com/v1", sdk="openai")


# --- pure helper unit tests -------------------------------------------------


def test_content_is_blank_cases():
    assert _content_is_blank(None) is True
    assert _content_is_blank("") is True
    assert _content_is_blank("   \n\t") is True
    assert _content_is_blank([]) is True
    assert _content_is_blank("hi") is False
    assert _content_is_blank([{"type": "text", "text": "x"}]) is False


def test_system_only_payload_gets_user_kickoff():
    # The exact shape ContextManager.prepare() produces on a fresh project
    # start with no initial_message: system messages only, no chat turn.
    sysonly = [
        {"role": "system", "content": "## ONBOARDING MODE\n..."},
        {"role": "system", "content": "[Project Instructions]\n..."},
    ]
    out = _ensure_chat_content(sysonly)
    chat_turns = [m for m in out if m.get("role") in ("user", "assistant")]
    assert len(chat_turns) == 1
    assert chat_turns[0]["role"] == "user"
    assert chat_turns[0]["content"].strip() != ""
    # No message in the outbound payload has blank content.
    assert all(not _content_is_blank(m.get("content")) for m in out)


def test_blank_user_content_placeholdered():
    out = _ensure_chat_content([
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "   "},
    ])
    assert all(not _content_is_blank(m.get("content")) for m in out)
    # A real chat turn already exists → no kickoff appended.
    assert sum(1 for m in out if m.get("role") == "user") == 1


def test_assistant_tool_call_only_content_preserved():
    # An assistant turn carrying only tool_calls legitimately has content
    # None/"" — it must NOT be placeholdered (that would be a forbidden hack
    # and could confuse providers), and its presence counts as a chat turn so
    # no kickoff is appended.
    tc = {"id": "call_1", "type": "function",
          "function": {"name": "read", "arguments": "{}"}}
    msgs = [
        {"role": "system", "content": "sys"},
        {"role": "assistant", "content": None, "tool_calls": [tc]},
        {"role": "tool", "tool_call_id": "call_1", "content": "ok"},
    ]
    out = _ensure_chat_content(msgs)
    asst = [m for m in out if m.get("role") == "assistant"][0]
    assert asst["content"] is None
    assert not any(m.get("role") == "user" for m in out)


def test_empty_list_gets_kickoff():
    out = _ensure_chat_content([])
    assert out and out[0]["role"] == "user"
    assert out[0]["content"].strip() != ""


def test_persisted_session_not_mutated():
    original = [{"role": "system", "content": "onboard"}]
    snapshot = [dict(m) for m in original]
    _ensure_chat_content(original)
    assert original == snapshot  # input list/dicts untouched


# --- provider-level integration: the actual wire payload --------------------


@pytest.mark.asyncio
async def test_first_turn_payload_has_no_empty_content():
    """End-to-end through the provider: capture the messages the OpenAI SDK
    would receive for the fresh-onboarding (system-only) case and assert the
    MiniMax-rejected condition cannot occur."""
    captured = {}

    class _Completions:
        async def create(self, **kwargs):
            captured["messages"] = kwargs["messages"]

            async def _gen():
                yield SimpleNamespace(
                    choices=[SimpleNamespace(
                        delta=SimpleNamespace(content="hi", reasoning_content=None, tool_calls=None),
                        finish_reason="stop",
                    )],
                    usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1),
                )

            return _gen()

    prov = _provider()
    prov._openai_client = SimpleNamespace(chat=SimpleNamespace(completions=_Completions()))

    # System-only context — exactly what prepare() returns on a fresh start.
    sysonly = [
        {"role": "system", "content": "## ONBOARDING MODE\nGreet the user."},
    ]
    async for _ in prov.stream(sysonly):
        pass

    sent = captured["messages"]
    # 1. At least one user/assistant turn exists.
    assert any(m.get("role") in ("user", "assistant") for m in sent)
    # 2. No message carries blank content (the MiniMax 2013 trigger), except
    #    a legitimate assistant tool_calls-only turn (none here).
    for m in sent:
        if m.get("role") == "assistant" and m.get("tool_calls"):
            continue
        assert not _content_is_blank(m.get("content")), f"blank content in {m!r}"
