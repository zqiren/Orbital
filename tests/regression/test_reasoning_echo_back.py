# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Regression tests for per-model reasoning echo-back policy.

Reproduces the bug we hit on Orbital-marketing with deepseek-v4-pro:
a text-only assistant turn was persisted without `reasoning_content`,
DeepSeek 400'd on the next request because v4 requires every assistant
turn to echo reasoning_content back.

The policy lookup table per echo_back value:
  required  – ensure the field is present on assistant msgs (default to "")
  forbidden – strip the field from assistant msgs (legacy deepseek-reasoner)
  none      – strip defensively (model produces no reasoning to echo)
  optional  – passthrough (current behavior)
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent_os.agent.providers.openai_compat import (
    LLMProvider,
    _apply_reasoning_policy,
)
from agent_os.config.provider_registry import ReasoningInfo


# ── helpers ──────────────────────────────────────────────────────────────


def _info(echo_back: str, field: str | None = "reasoning_content") -> ReasoningInfo:
    return ReasoningInfo(
        supported=field is not None,
        field=field,
        echo_back=echo_back,
        enable="auto",
    )


# ── _apply_reasoning_policy: pure function tests ─────────────────────────


class TestApplyReasoningPolicy:
    """Unit tests for the per-message normalizer."""

    def test_required_adds_empty_field_when_missing(self):
        """deepseek-v4-pro: assistant msg without reasoning_content gets one."""
        msg = {"role": "assistant", "content": "hello"}
        out = _apply_reasoning_policy(msg, _info("required"))
        assert out["reasoning_content"] == ""

    def test_required_preserves_existing_field(self):
        """When already present, never overwrite."""
        msg = {"role": "assistant", "content": "hello", "reasoning_content": "thinking..."}
        out = _apply_reasoning_policy(msg, _info("required"))
        assert out["reasoning_content"] == "thinking..."

    def test_forbidden_strips_field(self):
        """Legacy deepseek-reasoner: 400s if reasoning_content is sent."""
        msg = {"role": "assistant", "content": "hello", "reasoning_content": "leak"}
        out = _apply_reasoning_policy(msg, _info("forbidden"))
        assert "reasoning_content" not in out

    def test_none_strips_field_defensively(self):
        """Model has no reasoning support; defensive strip on outbound."""
        msg = {"role": "assistant", "content": "hi", "reasoning_content": "stale"}
        out = _apply_reasoning_policy(msg, _info("none"))
        assert "reasoning_content" not in out

    def test_optional_passthrough_with_field(self):
        msg = {"role": "assistant", "content": "hi", "reasoning_content": "trace"}
        out = _apply_reasoning_policy(msg, _info("optional"))
        assert out["reasoning_content"] == "trace"

    def test_optional_passthrough_without_field(self):
        msg = {"role": "assistant", "content": "hi"}
        out = _apply_reasoning_policy(msg, _info("optional"))
        assert "reasoning_content" not in out

    def test_user_message_never_modified(self):
        """Policy targets assistant role only."""
        msg = {"role": "user", "content": "hi", "reasoning_content": "weird"}
        out = _apply_reasoning_policy(msg, _info("forbidden"))
        # Even with forbidden, user-role messages aren't policed (they shouldn't carry the field anyway,
        # but if they do, it's out of scope).
        assert out["reasoning_content"] == "weird"

    def test_tool_message_never_modified(self):
        msg = {"role": "tool", "tool_call_id": "t1", "content": "result"}
        out = _apply_reasoning_policy(msg, _info("required"))
        assert "reasoning_content" not in out

    def test_no_reasoning_info_passthrough(self):
        """When the LLMProvider was constructed without ReasoningInfo, behave as before."""
        msg = {"role": "assistant", "content": "hi"}
        out = _apply_reasoning_policy(msg, None)
        assert out == msg

    def test_field_none_passthrough(self):
        """Field unset → nothing to do regardless of echo_back."""
        msg = {"role": "assistant", "content": "hi"}
        info = ReasoningInfo(supported=False, field=None, echo_back="required", enable="model_only")
        out = _apply_reasoning_policy(msg, info)
        assert out == msg

    def test_alternate_field_name_reasoning(self):
        """Provider-specific field other than reasoning_content (e.g. OpenRouter/Groq)."""
        msg = {"role": "assistant", "content": "hi"}
        out = _apply_reasoning_policy(msg, _info("required", field="reasoning"))
        assert out["reasoning"] == ""

    def test_alternate_field_name_thinking(self):
        """Anthropic via Anthropic SDK uses thinking blocks (out of OpenAI-compat scope, but
        verify the policy treats the field name uniformly)."""
        msg = {"role": "assistant", "content": "hi"}
        out = _apply_reasoning_policy(msg, _info("required", field="thinking"))
        assert out["thinking"] == ""

    def test_does_not_mutate_input(self):
        """Policy must return a new dict when changes are needed; never mutate caller's data."""
        msg = {"role": "assistant", "content": "hi"}
        snapshot = dict(msg)
        _apply_reasoning_policy(msg, _info("required"))
        assert msg == snapshot, "input must not be mutated"


# ── _prepare_messages_openai integration ─────────────────────────────────


class TestPrepareMessagesIntegration:
    """End-to-end: provider with reasoning info applies policy after stripping."""

    def test_required_fixes_missing_field_on_assistant_messages(self):
        provider = LLMProvider(
            model="deepseek-v4-pro",
            api_key="fake",
            base_url="https://api.deepseek.com",
            reasoning=_info("required"),
        )
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "hi", "timestamp": "t1"},
            {"role": "assistant", "content": "first", "reasoning_content": "thought 1", "timestamp": "t2"},
            {"role": "assistant", "content": "second", "timestamp": "t3"},  # missing — the bug
            {"role": "user", "content": "again", "timestamp": "t4"},
        ]
        prepared = provider._prepare_messages_openai(messages)
        assistants = [m for m in prepared if m["role"] == "assistant"]
        assert len(assistants) == 2
        assert assistants[0]["reasoning_content"] == "thought 1"
        assert assistants[1]["reasoning_content"] == ""
        # Internal fields still stripped
        for m in prepared:
            assert "timestamp" not in m

    def test_forbidden_strips_field_for_legacy_reasoner(self):
        provider = LLMProvider(
            model="deepseek-reasoner",
            api_key="fake",
            base_url="https://api.deepseek.com",
            reasoning=_info("forbidden"),
        )
        messages = [
            {"role": "assistant", "content": "x", "reasoning_content": "leak"},
        ]
        prepared = provider._prepare_messages_openai(messages)
        assert "reasoning_content" not in prepared[0]

    def test_optional_unchanged(self):
        provider = LLMProvider(
            model="kimi-k2.5",
            api_key="fake",
            base_url="https://api.moonshot.cn/v1",
            reasoning=_info("optional"),
        )
        messages = [
            {"role": "assistant", "content": "with", "reasoning_content": "trace"},
            {"role": "assistant", "content": "without"},
        ]
        prepared = provider._prepare_messages_openai(messages)
        assert prepared[0]["reasoning_content"] == "trace"
        assert "reasoning_content" not in prepared[1]

    def test_no_reasoning_info_keeps_existing_behavior(self):
        """LLMProvider without reasoning param continues to behave as before (denylist passthrough)."""
        provider = LLMProvider(
            model="anything",
            api_key="fake",
            base_url="http://localhost:1234",
        )
        messages = [
            {"role": "assistant", "content": "x", "reasoning_content": "kept"},
        ]
        prepared = provider._prepare_messages_openai(messages)
        assert prepared[0]["reasoning_content"] == "kept"


# ── Regression: replay the actual failing session ────────────────────────


class TestOrbitalMarketingRegression:
    """Reproduce the production bug: replay the exact assistant-message sequence
    from orbital-marketing_2599e313.jsonl and confirm every assistant message
    carries reasoning_content after _prepare_messages_openai when echo_back=required.
    """

    def test_replay_failing_session_shape(self, tmp_path: Path):
        """Construct the message sequence we observed on disk and verify the fix."""
        # Mirrors the structure we read from the live JSONL: rows #1, #3 had
        # reasoning_content; row #6 (text-only final after tool use) didn't.
        observed_rows = [
            {"role": "user", "content": "翻译", "nonce": "n1", "timestamp": "t0"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"id": "tc1", "type": "function", "function": {"name": "write", "arguments": "{}"}}],
                "reasoning_content": "thinking step 1",
                "timestamp": "t1",
                "_status": "Using write",
            },
            {"role": "tool", "tool_call_id": "tc1", "content": "ok", "timestamp": "t2"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"id": "tc2", "type": "function", "function": {"name": "read", "arguments": "{}"}}],
                "reasoning_content": "thinking step 2",
                "timestamp": "t3",
                "_status": "Using read",
            },
            {"role": "tool", "tool_call_id": "tc2", "content": "data", "timestamp": "t4"},
            # Row #6 in the real session: text-only final, NO reasoning_content (the bug)
            {"role": "assistant", "content": "Done.", "timestamp": "t5"},
            # Next user turn that would have triggered the 400
            {"role": "user", "content": "继续", "nonce": "n2", "timestamp": "t6"},
        ]
        provider = LLMProvider(
            model="deepseek-v4-pro",
            api_key="fake",
            base_url="https://api.deepseek.com",
            reasoning=_info("required"),
        )
        prepared = provider._prepare_messages_openai(observed_rows)

        # Every assistant message must now have reasoning_content (the contract)
        assistants = [m for m in prepared if m["role"] == "assistant"]
        assert len(assistants) == 3
        for i, m in enumerate(assistants):
            assert "reasoning_content" in m, f"assistant[{i}] missing reasoning_content"

        # Verify the previously-missing one is the empty-string default
        assert assistants[0]["reasoning_content"] == "thinking step 1"
        assert assistants[1]["reasoning_content"] == "thinking step 2"
        assert assistants[2]["reasoning_content"] == ""  # the row that was breaking deepseek

        # Sanity: tool/user/system messages untouched
        users = [m for m in prepared if m["role"] == "user"]
        for m in users:
            assert "reasoning_content" not in m
