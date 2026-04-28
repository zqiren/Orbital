# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Regression tests for `disable_reasoning=` on LLMProvider.complete().

Background: the session-end refresh routine times out on thinking-mode
models because the LLM spends 90+ seconds thinking before emitting JSON.
For models whose ``reasoning.enable`` starts with ``param:``, we can
disable thinking via ``extra_body=`` on the OpenAI SDK call.

These tests verify the off-switch mapping table by capturing the kwargs
the SDK is invoked with via AsyncMock.
"""

from __future__ import annotations

import logging
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent_os.agent.providers.openai_compat import LLMProvider
from agent_os.config.provider_registry import ReasoningInfo


# ── helpers ──────────────────────────────────────────────────────────────


def _make_provider(model: str, reasoning: ReasoningInfo | None) -> LLMProvider:
    """Build a provider whose openai client is a MagicMock with an AsyncMock
    on chat.completions.create returning a minimal-but-valid completion."""
    provider = LLMProvider(
        model=model,
        api_key="fake",
        base_url="https://example.invalid/v1",
        reasoning=reasoning,
    )
    provider._openai_client = MagicMock()
    mock_create = AsyncMock(
        return_value=MagicMock(
            choices=[
                MagicMock(
                    message=MagicMock(
                        content="{}",
                        tool_calls=None,
                        # model_dump returns the wire-format dict
                        model_dump=MagicMock(return_value={"role": "assistant", "content": "{}"}),
                    ),
                    finish_reason="stop",
                )
            ],
            usage=MagicMock(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        )
    )
    provider._openai_client.chat.completions.create = mock_create
    return provider


def _info(enable: str, field: str | None = "reasoning_content") -> ReasoningInfo:
    return ReasoningInfo(
        supported=True,
        field=field,
        echo_back="required",
        enable=enable,
    )


# ── disable_reasoning=True: per-model mapping ────────────────────────────


class TestDisableReasoningParamMapping:
    """Each known ``param:`` spec maps to its documented off-switch."""

    @pytest.mark.asyncio
    async def test_deepseek_v4_thinking_type(self):
        """DeepSeek v4: enable=param:thinking.type=enabled
        → extra_body={"thinking": {"type": "disabled"}}"""
        provider = _make_provider(
            "deepseek-v4-pro",
            _info("param:thinking.type=enabled"),
        )
        await provider.complete(
            [{"role": "user", "content": "x"}],
            disable_reasoning=True,
        )
        kwargs = provider._openai_client.chat.completions.create.call_args.kwargs
        assert kwargs.get("extra_body") == {"thinking": {"type": "disabled"}}

    @pytest.mark.asyncio
    async def test_qwen_enable_thinking(self):
        """Qwen: enable=param:enable_thinking=true
        → extra_body={"enable_thinking": False}"""
        provider = _make_provider(
            "qwen3-coder",
            _info("param:enable_thinking=true"),
        )
        await provider.complete(
            [{"role": "user", "content": "x"}],
            disable_reasoning=True,
        )
        kwargs = provider._openai_client.chat.completions.create.call_args.kwargs
        assert kwargs.get("extra_body") == {"enable_thinking": False}

    @pytest.mark.asyncio
    async def test_openai_o_series_reasoning_effort(self):
        """OpenAI o-series: enable=param:reasoning_effort=medium
        → extra_body={"reasoning_effort": "minimal"}"""
        provider = _make_provider(
            "o3-mini",
            _info("param:reasoning_effort=medium"),
        )
        await provider.complete(
            [{"role": "user", "content": "x"}],
            disable_reasoning=True,
        )
        kwargs = provider._openai_client.chat.completions.create.call_args.kwargs
        assert kwargs.get("extra_body") == {"reasoning_effort": "minimal"}

    @pytest.mark.asyncio
    async def test_openrouter_reasoning_max_tokens(self):
        """OpenRouter: enable=param:reasoning.max_tokens=N
        → extra_body={"reasoning": {"enabled": False}}"""
        provider = _make_provider(
            "openrouter/some-model",
            _info("param:reasoning.max_tokens=4096"),
        )
        await provider.complete(
            [{"role": "user", "content": "x"}],
            disable_reasoning=True,
        )
        kwargs = provider._openai_client.chat.completions.create.call_args.kwargs
        assert kwargs.get("extra_body") == {"reasoning": {"enabled": False}}

    @pytest.mark.asyncio
    async def test_openrouter_reasoning_enabled_true(self):
        """enable=param:reasoning.enabled=true → reasoning.enabled=False"""
        provider = _make_provider(
            "openrouter/another-model",
            _info("param:reasoning.enabled=true"),
        )
        await provider.complete(
            [{"role": "user", "content": "x"}],
            disable_reasoning=True,
        )
        kwargs = provider._openai_client.chat.completions.create.call_args.kwargs
        assert kwargs.get("extra_body") == {"reasoning": {"enabled": False}}

    @pytest.mark.asyncio
    async def test_openrouter_reasoning_effort(self):
        """enable=param:reasoning.effort=high → reasoning.enabled=False"""
        provider = _make_provider(
            "openrouter/yet-another-model",
            _info("param:reasoning.effort=high"),
        )
        await provider.complete(
            [{"role": "user", "content": "x"}],
            disable_reasoning=True,
        )
        kwargs = provider._openai_client.chat.completions.create.call_args.kwargs
        assert kwargs.get("extra_body") == {"reasoning": {"enabled": False}}


# ── disable_reasoning=True: locked-on / unknown / None reasoning ─────────


class TestDisableReasoningEdgeCases:
    """Cases where no extra_body should be sent."""

    @pytest.mark.asyncio
    async def test_locked_on_auto_logs_warning_no_extra_body(self, caplog):
        """enable=auto → no extra_body, warning logged."""
        provider = _make_provider(
            "some-locked-on-model",
            _info("auto"),
        )
        with caplog.at_level(logging.WARNING, logger="agent_os.agent.providers.openai_compat"):
            await provider.complete(
                [{"role": "user", "content": "x"}],
                disable_reasoning=True,
            )
        kwargs = provider._openai_client.chat.completions.create.call_args.kwargs
        assert "extra_body" not in kwargs
        # Warning should mention the model and the enable value
        warning_msgs = [r.message for r in caplog.records if r.levelno == logging.WARNING]
        assert any("some-locked-on-model" in m for m in warning_msgs)
        assert any("auto" in m for m in warning_msgs)

    @pytest.mark.asyncio
    async def test_locked_on_model_only_logs_warning_no_extra_body(self, caplog):
        """enable=model_only → no extra_body, warning logged."""
        provider = _make_provider(
            "another-locked-model",
            _info("model_only"),
        )
        with caplog.at_level(logging.WARNING, logger="agent_os.agent.providers.openai_compat"):
            await provider.complete(
                [{"role": "user", "content": "x"}],
                disable_reasoning=True,
            )
        kwargs = provider._openai_client.chat.completions.create.call_args.kwargs
        assert "extra_body" not in kwargs
        warning_msgs = [r.message for r in caplog.records if r.levelno == logging.WARNING]
        assert any("another-locked-model" in m for m in warning_msgs)
        assert any("model_only" in m for m in warning_msgs)

    @pytest.mark.asyncio
    async def test_unknown_param_logs_warning_no_extra_body(self, caplog):
        """Unrecognized param:foo=bar → no extra_body, warning logged."""
        provider = _make_provider(
            "novel-model",
            _info("param:foo=bar"),
        )
        with caplog.at_level(logging.WARNING, logger="agent_os.agent.providers.openai_compat"):
            await provider.complete(
                [{"role": "user", "content": "x"}],
                disable_reasoning=True,
            )
        kwargs = provider._openai_client.chat.completions.create.call_args.kwargs
        assert "extra_body" not in kwargs
        warning_msgs = [r.message for r in caplog.records if r.levelno == logging.WARNING]
        assert any("novel-model" in m for m in warning_msgs)
        assert any("param:foo=bar" in m for m in warning_msgs)

    @pytest.mark.asyncio
    async def test_disable_reasoning_false_default_no_extra_body(self):
        """disable_reasoning=False (default) → never sends extra_body even when
        the model has a param-style enable."""
        provider = _make_provider(
            "deepseek-v4-pro",
            _info("param:thinking.type=enabled"),
        )
        # Default behavior — do not pass disable_reasoning
        await provider.complete([{"role": "user", "content": "x"}])
        kwargs = provider._openai_client.chat.completions.create.call_args.kwargs
        assert "extra_body" not in kwargs

        # Explicit False matches default
        await provider.complete(
            [{"role": "user", "content": "x"}],
            disable_reasoning=False,
        )
        kwargs2 = provider._openai_client.chat.completions.create.call_args.kwargs
        assert "extra_body" not in kwargs2

    @pytest.mark.asyncio
    async def test_reasoning_none_no_extra_body(self):
        """self.reasoning is None → no extra_body regardless of disable_reasoning."""
        provider = _make_provider("plain-model", reasoning=None)
        await provider.complete(
            [{"role": "user", "content": "x"}],
            disable_reasoning=True,
        )
        kwargs = provider._openai_client.chat.completions.create.call_args.kwargs
        assert "extra_body" not in kwargs
