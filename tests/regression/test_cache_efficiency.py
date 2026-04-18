# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Regression tests for cache efficiency improvements.

Verifies:
1. Dynamic content (runtime, context_budget) is NOT in the first system message
2. Dynamic content IS in the last system message
3. Non-spec fields are stripped before API call
4. Cache audit logging is present
5. Context audit logging is present
"""

import logging
from unittest.mock import MagicMock

import pytest

from agent_os.agent.context import ContextManager
from agent_os.agent.prompt_builder import Autonomy, PromptContext
from agent_os.agent.session import Session
from agent_os.agent.providers.openai_compat import (
    LLMProvider,
    _strip_to_spec,
    _log_cache_audit,
    ORBITAL_INTERNAL_FIELDS,
)
from agent_os.agent.providers.types import TokenUsage


# ── helpers ──────────────────────────────────────────────────────────────


class _StubPromptBuilder:
    """Returns a known cached_prefix, semi_stable, and truly_dynamic for testing."""

    def __init__(self, cached="Cached system prompt.",
                 semi_stable="Semi-stable session content.",
                 dynamic="Current time: 2026-04-16T12:00:00\nContext usage: ~25%."):
        self._cached = cached
        self._semi_stable = semi_stable
        self._dynamic = dynamic

    def build(self, context):
        return (self._cached, self._semi_stable, self._dynamic)


def _make_base_ctx(workspace: str) -> PromptContext:
    return PromptContext(
        workspace=workspace,
        model="test-model",
        autonomy=Autonomy.HANDS_OFF,
        enabled_agents=[],
        tool_names=["read"],
        os_type="linux",
        datetime_now="2026-01-01T00:00:00",
        context_usage_pct=0.0,
    )


def _make_context_manager(tmp_path, prompt_builder=None):
    workspace = str(tmp_path)
    session = Session.new("cache-test", workspace)
    session.append({"role": "user", "content": "Hello", "source": "user"})
    ctx_mgr = ContextManager(
        session,
        prompt_builder or _StubPromptBuilder(),
        _make_base_ctx(workspace),
    )
    return ctx_mgr


# ── Change 1: Dynamic content relocated to end ──────────────────────────


class TestDynamicContentPosition:
    """Verify dynamic_suffix is appended at the end, not concatenated into first system msg."""

    def test_first_system_message_is_static_only(self, tmp_path):
        """First system message should contain only cached_prefix, no dynamic content."""
        ctx_mgr = _make_context_manager(tmp_path)
        messages = ctx_mgr.prepare()

        system_messages = [m for m in messages if m["role"] == "system"]
        assert len(system_messages) >= 2, "Expected at least 2 system messages (static + runtime)"

        first_system = system_messages[0]["content"]
        assert first_system == "Cached system prompt.", (
            f"First system message should be exactly the cached prefix, got: {first_system[:100]}"
        )
        assert "Current time:" not in first_system, "Dynamic timestamp found in cacheable prefix"
        assert "Context usage:" not in first_system, "Dynamic context budget found in cacheable prefix"

    def test_dynamic_content_in_last_system_message(self, tmp_path):
        """Last system message should contain the dynamic_suffix."""
        ctx_mgr = _make_context_manager(tmp_path)
        messages = ctx_mgr.prepare()

        system_messages = [m for m in messages if m["role"] == "system"]
        last_system = system_messages[-1]["content"]

        assert "Current time:" in last_system, "Dynamic timestamp missing from runtime context"
        assert "Context usage:" in last_system, "Dynamic context budget missing from runtime context"

    def test_dynamic_suffix_after_conversation_history(self, tmp_path):
        """The dynamic system message must appear AFTER the conversation history."""
        ctx_mgr = _make_context_manager(tmp_path)
        messages = ctx_mgr.prepare()

        # Find the last user/assistant/tool message index
        last_history_idx = -1
        for i, m in enumerate(messages):
            if m["role"] in ("user", "assistant", "tool"):
                last_history_idx = i

        # Find the dynamic suffix message index
        dynamic_idx = -1
        for i, m in enumerate(messages):
            if m["role"] == "system" and "Current time:" in m.get("content", ""):
                dynamic_idx = i

        assert last_history_idx >= 0, "No conversation history found"
        assert dynamic_idx >= 0, "No dynamic suffix message found"
        assert dynamic_idx > last_history_idx, (
            f"Dynamic suffix (idx={dynamic_idx}) should come after history (idx={last_history_idx})"
        )

    def test_message_order_matches_spec(self, tmp_path):
        """Full message order: static system → layers → history → dynamic runtime."""
        ctx_mgr = _make_context_manager(tmp_path)
        messages = ctx_mgr.prepare()

        roles = [m["role"] for m in messages]

        # First message is system (cached prefix)
        assert roles[0] == "system"
        assert messages[0]["content"] == "Cached system prompt."

        # Last message is system (dynamic suffix)
        assert roles[-1] == "system"
        assert "Current time:" in messages[-1]["content"]

        # Conversation history sits between them
        history_indices = [i for i, r in enumerate(roles) if r in ("user", "assistant", "tool")]
        if history_indices:
            assert all(0 < i < len(messages) - 1 for i in history_indices), (
                "History messages should be between first and last system messages"
            )

    def test_empty_dynamic_suffix_not_appended(self, tmp_path):
        """If truly_dynamic is empty, no extra system message should be appended at end."""
        ctx_mgr = _make_context_manager(tmp_path, _StubPromptBuilder(semi_stable="", dynamic=""))
        messages = ctx_mgr.prepare()

        system_messages = [m for m in messages if m["role"] == "system"]
        # Only the cached prefix system message
        assert len(system_messages) == 1
        assert system_messages[0]["content"] == "Cached system prompt."


    def test_semi_stable_before_history(self, tmp_path):
        """Semi-stable content must appear BEFORE conversation history."""
        ctx_mgr = _make_context_manager(tmp_path)
        messages = ctx_mgr.prepare()

        # Find semi-stable message
        semi_idx = -1
        for i, m in enumerate(messages):
            if m["role"] == "system" and "Semi-stable" in m.get("content", ""):
                semi_idx = i
                break

        # Find first history message
        history_idx = -1
        for i, m in enumerate(messages):
            if m["role"] in ("user", "assistant", "tool"):
                history_idx = i
                break

        assert semi_idx >= 0, "No semi-stable message found"
        assert history_idx >= 0, "No history message found"
        assert semi_idx < history_idx, (
            f"Semi-stable (idx={semi_idx}) must come before history (idx={history_idx})"
        )

    def test_three_part_message_structure(self, tmp_path):
        """Message order: static prefix -> semi-stable -> layers -> history -> truly dynamic."""
        ctx_mgr = _make_context_manager(tmp_path)
        messages = ctx_mgr.prepare()

        # First system message is cached prefix
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "Cached system prompt."

        # Second system message is semi-stable
        assert messages[1]["role"] == "system"
        assert messages[1]["content"] == "Semi-stable session content."

        # Last message is truly dynamic
        assert messages[-1]["role"] == "system"
        assert "Current time:" in messages[-1]["content"]
        assert "Context usage:" in messages[-1]["content"]

        # Semi-stable should NOT contain dynamic markers
        assert "Current time:" not in messages[1]["content"]
        assert "Context usage:" not in messages[1]["content"]


# ── Change 2: Non-spec field stripping ──────────────────────────────────


class TestFieldStripping:
    """Verify non-OpenAI-spec fields are stripped before API call."""

    def test_strip_to_spec_removes_internal_fields(self):
        """All Orbital-internal bookkeeping fields must be stripped before wire send."""
        msg = {
            "role": "user",
            "content": "hi",
            "timestamp": "2026-04-18T10:00:00Z",
            "session_id": "sess_abc",
            "source": "user",
            "_meta": {"image_info": {}},
            "_stubbed": True,
            "_status": "Using read",
            "_activity_descriptions": {"call_1": "Reading file"},
            "nonce": "abc123",
            "target": "@claudecode",
        }
        stripped = _strip_to_spec(msg)
        for internal_field in [
            "timestamp", "session_id", "source", "_meta", "_stubbed",
            "_status", "_activity_descriptions", "nonce", "target",
        ]:
            assert internal_field not in stripped, f"{internal_field} should be stripped"
        assert stripped["role"] == "user"
        assert stripped["content"] == "hi"

    def test_strip_to_spec_preserves_provider_native_fields(self):
        """Provider-native fields (including future extensions) must pass through."""
        msg = {
            "role": "assistant",
            "content": "hello",
            "tool_calls": [{"id": "t1", "type": "function"}],
            "reasoning_content": "thinking...",           # Moonshot/DeepSeek
            "reasoning": {"effort": "high"},              # OpenAI o1/o3
            "x_future_provider_extension": "preserved",   # Forward-compat
        }
        stripped = _strip_to_spec(msg)
        assert stripped["reasoning_content"] == "thinking..."
        assert stripped["reasoning"] == {"effort": "high"}
        assert stripped["x_future_provider_extension"] == "preserved"
        assert stripped["role"] == "assistant"
        assert stripped["content"] == "hello"
        assert stripped["tool_calls"] == [{"id": "t1", "type": "function"}]

    def test_strip_to_spec_passes_unknown_fields(self):
        """Regression guard: fields not in ORBITAL_INTERNAL_FIELDS must pass through
        unchanged. This test guards against reverting to an allowlist approach,
        which silently breaks provider extensions (see ORBITAL_INTERNAL_FIELDS
        docstring).
        """
        msg = {
            "role": "assistant",
            "content": "test",
            "this_field_does_not_exist_yet": "value",
            "some_future_provider_key": {"nested": True},
        }
        stripped = _strip_to_spec(msg)
        assert stripped["this_field_does_not_exist_yet"] == "value"
        assert stripped["some_future_provider_key"] == {"nested": True}

    def test_strip_to_spec_handles_tool_results(self):
        """Tool result messages keep only spec fields."""
        msg = {
            "role": "tool",
            "tool_call_id": "tc_42",
            "content": "result data",
            "timestamp": "2026-04-16T10:00:05",
            "_meta": {"snapshot_stats": {"refs": 10}},
        }
        stripped = _strip_to_spec(msg)
        assert set(stripped.keys()) == {"role", "tool_call_id", "content"}

    def test_prepare_messages_openai_strips_fields(self):
        """_prepare_messages_openai should strip non-spec fields from all messages."""
        provider = LLMProvider(
            model="test-model",
            api_key="fake-key",
            base_url="http://localhost:1234",
        )
        input_messages = [
            {
                "role": "system",
                "content": "You are helpful.",
                "source": "daemon",
            },
            {
                "role": "user",
                "content": "Hello",
                "timestamp": "2026-04-16T10:00:00",
                "session_id": "sess_123",
                "source": "management",
                "_meta": {"foo": "bar"},
            },
            {
                "role": "assistant",
                "content": "Hi there",
                "timestamp": "2026-04-16T10:00:01",
                "session_id": "sess_123",
                "_stubbed": False,
            },
        ]
        prepared = provider._prepare_messages_openai(input_messages)

        for msg in prepared:
            leaked_internal = set(msg.keys()) & ORBITAL_INTERNAL_FIELDS
            assert not leaked_internal, f"Orbital-internal fields leaked to wire: {leaked_internal}"


# ── Change 3: Cache audit logging ───────────────────────────────────────


class TestCacheAuditLogging:
    """Verify [CACHE_AUDIT] log entries."""

    def test_cache_audit_logged(self, caplog):
        """_log_cache_audit should produce a [CACHE_AUDIT] info log."""
        with caplog.at_level(logging.INFO, logger="orbital.cache_audit"):
            usage = TokenUsage(input_tokens=10000, output_tokens=500, cache_read_tokens=8500)
            _log_cache_audit("test-model", usage)

        cache_logs = [r for r in caplog.records if "[CACHE_AUDIT]" in r.message]
        assert len(cache_logs) == 1, "Expected exactly one CACHE_AUDIT log entry"

        log_msg = cache_logs[0].message
        assert "model=test-model" in log_msg
        assert "input=10000" in log_msg
        assert "cached=8500" in log_msg
        assert "output=500" in log_msg
        assert "cache_rate=85.0%" in log_msg

    def test_cache_audit_zero_input_no_divide_by_zero(self, caplog):
        """Cache rate should be 0.0% when input_tokens is zero."""
        with caplog.at_level(logging.INFO, logger="orbital.cache_audit"):
            usage = TokenUsage(input_tokens=0, output_tokens=0, cache_read_tokens=0)
            _log_cache_audit("test-model", usage)

        cache_logs = [r for r in caplog.records if "[CACHE_AUDIT]" in r.message]
        assert len(cache_logs) == 1
        assert "cache_rate=0.0%" in cache_logs[0].message


# ── Change 4: Context audit logging ─────────────────────────────────────


class TestContextAuditLogging:
    """Verify [CONTEXT_AUDIT] debug log entries from prepare()."""

    def test_context_audit_logged_on_prepare(self, tmp_path, caplog):
        """prepare() should emit a CONTEXT_AUDIT debug log."""
        with caplog.at_level(logging.DEBUG, logger="agent_os.agent.context"):
            ctx_mgr = _make_context_manager(tmp_path)
            ctx_mgr.prepare()

        context_logs = [r for r in caplog.records if "[CONTEXT_AUDIT]" in r.message]
        assert len(context_logs) == 1, "Expected exactly one CONTEXT_AUDIT log entry"

        log_msg = context_logs[0].message
        assert "messages=" in log_msg
        assert "system_msgs=" in log_msg
        assert "history_msgs=" in log_msg
        assert "total_chars=" in log_msg
