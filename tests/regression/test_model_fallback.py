# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Regression tests: model fallback with provider rotation on transient LLM errors.

When the primary LLM returns transient errors (429, 502, 503, timeout),
the agent loop rotates to fallback models instead of dying after 3 retries.
Non-recoverable errors (401, 403, 400) abort immediately. Failed providers
re-enter the candidate list after a 60-second cooldown.
"""

import json
import time
from unittest.mock import patch

import pytest

from agent_os.agent.session import Session
from agent_os.agent.loop import AgentLoop
from agent_os.agent.context import ContextManager
from agent_os.agent.providers.types import (
    StreamChunk,
    LLMResponse,
    TokenUsage,
    ContextOverflowError,
    LLMError,
    ErrorCategory,
)
from agent_os.agent.tools.base import ToolResult
from agent_os.agent.prompt_builder import PromptContext, Autonomy


# ---------------------------------------------------------------------------
# Helpers (reuse patterns from test_browser_screenshot_repetition.py)
# ---------------------------------------------------------------------------

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


def _make_text_response(text: str) -> LLMResponse:
    return LLMResponse(
        raw_message={"role": "assistant", "content": text},
        text=text,
        tool_calls=[],
        has_tool_calls=False,
        finish_reason="stop",
        status_text=None,
        usage=TokenUsage(input_tokens=100, output_tokens=50),
    )


class ErrorThenSuccessProvider:
    """Provider that raises LLMError for the first N calls, then returns text."""

    def __init__(self, model: str, error: LLMError, fail_count: int):
        self.model = model
        self._error = error
        self._fail_count = fail_count
        self._call_count = 0
        self.calls = []

    async def stream(self, messages, tools=None):
        self._call_count += 1
        self.calls.append(self._call_count)
        if self._call_count <= self._fail_count:
            raise self._error
        yield StreamChunk(text="Done from " + self.model)
        yield StreamChunk(
            is_final=True,
            usage=TokenUsage(input_tokens=100, output_tokens=50),
        )


class AlwaysErrorProvider:
    """Provider that always raises LLMError (async generator that raises before yielding)."""

    def __init__(self, model: str, error: LLMError):
        self.model = model
        self._error = error
        self._call_count = 0

    async def stream(self, messages, tools=None):
        self._call_count += 1
        if True:  # noqa: SIM108 — ensure this is an async generator
            raise self._error
        yield  # unreachable, but makes Python treat this as an async generator


class SuccessProvider:
    """Provider that always succeeds with a text response."""

    def __init__(self, model: str):
        self.model = model
        self._call_count = 0

    async def stream(self, messages, tools=None):
        self._call_count += 1
        yield StreamChunk(text="Done from " + self.model)
        yield StreamChunk(
            is_final=True,
            usage=TokenUsage(input_tokens=1000, output_tokens=500),
        )


class SimpleToolRegistry:
    """Minimal tool registry with no tools."""
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


# ---------------------------------------------------------------------------
# Tests: Error Classification
# ---------------------------------------------------------------------------

class TestErrorClassification:
    """Verify LLMError self-classifies based on status_code and message."""

    def test_429_is_transient(self):
        err = LLMError("rate limited", status_code=429)
        assert err.category == ErrorCategory.TRANSIENT

    def test_502_is_transient(self):
        err = LLMError("bad gateway", status_code=502)
        assert err.category == ErrorCategory.TRANSIENT

    def test_503_is_transient(self):
        err = LLMError("service unavailable", status_code=503)
        assert err.category == ErrorCategory.TRANSIENT

    def test_timeout_message_is_transient(self):
        err = LLMError("Request timed out")
        assert err.category == ErrorCategory.TRANSIENT

    def test_connection_failed_is_transient(self):
        err = LLMError("Connection failed")
        assert err.category == ErrorCategory.TRANSIENT

    def test_dns_failure_is_transient(self):
        err = LLMError("DNS resolution failed")
        assert err.category == ErrorCategory.TRANSIENT

    def test_401_is_abort(self):
        err = LLMError("unauthorized", status_code=401)
        assert err.category == ErrorCategory.ABORT

    def test_403_is_abort(self):
        err = LLMError("forbidden", status_code=403)
        assert err.category == ErrorCategory.ABORT

    def test_400_is_abort(self):
        err = LLMError("bad request", status_code=400)
        assert err.category == ErrorCategory.ABORT

    def test_500_is_retry(self):
        err = LLMError("internal server error", status_code=500)
        assert err.category == ErrorCategory.RETRY

    def test_unknown_status_is_retry(self):
        err = LLMError("something weird", status_code=418)
        assert err.category == ErrorCategory.RETRY

    def test_no_status_no_keyword_is_retry(self):
        err = LLMError("unknown error")
        assert err.category == ErrorCategory.RETRY


# ---------------------------------------------------------------------------
# Tests: Fallback Rotation
# ---------------------------------------------------------------------------

class TestFallbackRotation:

    @pytest.mark.asyncio
    async def test_fallback_rotates_on_transient_error(self, tmp_path):
        """Primary returns 503, loop rotates to fallback, completes successfully."""
        session = Session.new("fb_503", str(tmp_path))

        primary = AlwaysErrorProvider(
            "primary-model",
            LLMError("service unavailable", status_code=503),
        )
        fallback = SuccessProvider("fallback-model")

        builder = MockPromptBuilder()
        ctx = _make_base_prompt_context(str(tmp_path))
        context_mgr = ContextManager(session, builder, ctx)
        registry = SimpleToolRegistry()

        loop = AgentLoop(
            session, primary, registry, context_mgr,
            fallback_providers=[fallback],
            max_iterations=10,
        )
        await loop.run(initial_message="hello")

        # Loop should NOT have failed
        assert not loop._llm_failed

        # Fallback was called
        assert fallback._call_count >= 1

        # System message about switching should be present
        msgs = session.get_messages()
        system_msgs = [
            m for m in msgs
            if m["role"] == "system" and "Switching to" in m.get("content", "")
        ]
        assert len(system_msgs) >= 1
        assert "fallback-model" in system_msgs[0]["content"]

    @pytest.mark.asyncio
    async def test_fallback_aborts_on_auth_error(self, tmp_path):
        """Primary returns 401, loop aborts immediately without trying fallback."""
        session = Session.new("fb_401", str(tmp_path))

        primary = AlwaysErrorProvider(
            "primary-model",
            LLMError("unauthorized", status_code=401),
        )
        fallback = SuccessProvider("fallback-model")

        builder = MockPromptBuilder()
        ctx = _make_base_prompt_context(str(tmp_path))
        context_mgr = ContextManager(session, builder, ctx)
        registry = SimpleToolRegistry()

        loop = AgentLoop(
            session, primary, registry, context_mgr,
            fallback_providers=[fallback],
            max_iterations=10,
        )
        await loop.run(initial_message="hello")

        # Loop should have failed
        assert loop._llm_failed

        # Fallback was never called
        assert fallback._call_count == 0

        # System message about non-recoverable error
        msgs = session.get_messages()
        system_msgs = [
            m for m in msgs
            if m["role"] == "system" and "non-recoverable" in m.get("content", "")
        ]
        assert len(system_msgs) >= 1

    @pytest.mark.asyncio
    async def test_fallback_cooldown_reentry(self, tmp_path):
        """Primary fails with 503, rotates to fallback. After cooldown expires,
        primary re-enters the candidate list."""
        session = Session.new("fb_cooldown", str(tmp_path))

        # Primary fails once, then succeeds
        primary = ErrorThenSuccessProvider(
            "primary-model",
            LLMError("service unavailable", status_code=503),
            fail_count=1,
        )
        # Fallback also fails (to force waiting for cooldown expiry)
        fallback = AlwaysErrorProvider(
            "fallback-model",
            LLMError("also unavailable", status_code=503),
        )

        builder = MockPromptBuilder()
        ctx = _make_base_prompt_context(str(tmp_path))
        context_mgr = ContextManager(session, builder, ctx)
        registry = SimpleToolRegistry()

        # Patch time.monotonic to simulate cooldown expiry
        real_monotonic = time.monotonic
        call_count = [0]

        def fast_monotonic():
            call_count[0] += 1
            # After a few calls, jump forward past cooldown
            if call_count[0] > 4:
                return real_monotonic() + 120  # 120s in the future
            return real_monotonic()

        with patch("agent_os.agent.loop.time") as mock_time:
            mock_time.monotonic = fast_monotonic

            loop = AgentLoop(
                session, primary, registry, context_mgr,
                fallback_providers=[fallback],
                max_iterations=10,
            )
            await loop.run(initial_message="hello")

        # Primary was called at least twice (first fail, then success after cooldown)
        assert primary._call_count >= 2

    @pytest.mark.asyncio
    async def test_no_fallback_config_preserves_existing_behavior(self, tmp_path):
        """No fallback chain configured, primary returns 503 3x, loop stops."""
        session = Session.new("fb_none", str(tmp_path))

        primary = AlwaysErrorProvider(
            "primary-model",
            LLMError("service unavailable", status_code=503),
        )

        builder = MockPromptBuilder()
        ctx = _make_base_prompt_context(str(tmp_path))
        context_mgr = ContextManager(session, builder, ctx)
        registry = SimpleToolRegistry()

        loop = AgentLoop(
            session, primary, registry, context_mgr,
            fallback_providers=[],  # no fallbacks
            max_iterations=10,
        )
        await loop.run(initial_message="hello")

        # Loop should have failed
        assert loop._llm_failed

        # System message about retries exhausted
        msgs = session.get_messages()
        system_msgs = [
            m for m in msgs
            if m["role"] == "system" and "retries" in m.get("content", "").lower()
        ]
        assert len(system_msgs) >= 1

    @pytest.mark.asyncio
    async def test_fallback_cost_tracking(self, tmp_path):
        """Cost tracking accumulates regardless of which provider responds."""
        session = Session.new("fb_cost", str(tmp_path))

        primary = AlwaysErrorProvider(
            "primary-model",
            LLMError("service unavailable", status_code=503),
        )
        # Fallback returns 1000 input + 500 output tokens
        fallback = SuccessProvider("fallback-model")

        builder = MockPromptBuilder()
        ctx = _make_base_prompt_context(str(tmp_path))
        context_mgr = ContextManager(session, builder, ctx)
        registry = SimpleToolRegistry()

        loop = AgentLoop(
            session, primary, registry, context_mgr,
            fallback_providers=[fallback],
            budget_limit_usd=100.0,
            max_iterations=10,
        )
        await loop.run(initial_message="hello")

        # Cost should be > 0 (tracked from fallback's response)
        assert loop._budget_spent_usd > 0


# ---------------------------------------------------------------------------
# Tests: Config Models
# ---------------------------------------------------------------------------

class TestFallbackConfig:

    def test_agent_config_default_empty_list(self):
        from agent_os.daemon_v2.models import AgentConfig
        config = AgentConfig(workspace="/tmp", model="gpt-4", api_key="sk-test")
        assert config.llm_fallback_models == []

    def test_fallback_entry_defaults(self):
        from agent_os.daemon_v2.models import FallbackModelEntry
        entry = FallbackModelEntry(model="gpt-4-mini")
        assert entry.sdk == "openai"
        assert entry.provider == "custom"
        assert entry.api_key == ""

    def test_global_settings_has_fallback(self):
        from agent_os.daemon_v2.settings_store import GlobalLLMSettings
        settings = GlobalLLMSettings()
        assert settings.fallback_models == []
