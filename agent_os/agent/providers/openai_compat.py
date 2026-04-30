# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Dual-SDK LLM client: OpenAI SDK or Anthropic SDK.

Routing logic:
  - sdk == "openai"    -> openai.AsyncOpenAI(base_url, api_key)
  - sdk == "anthropic" -> anthropic.AsyncAnthropic(api_key)
"""

from __future__ import annotations

import asyncio
import logging
import re
from typing import AsyncIterator

import openai

from agent_os.agent.providers.types import (
    TokenUsage,
    StreamChunk,
    LLMResponse,
    ContextOverflowError,
    LLMError,
)

_cache_logger = logging.getLogger("orbital.cache_audit")


def _extract_status(text: str | None) -> str | None:
    if not text:
        return None
    match = re.search(r'\[STATUS:\s*(.+?)\]', text)
    return match.group(1).strip() if match else None


def _extract_cache_read_tokens(usage_obj) -> int:
    """Extract cached token count — handles Anthropic, OpenAI, and Kimi field names."""
    for attr in ("cache_read_input_tokens", "prompt_cache_hit_tokens", "cached_tokens"):
        val = getattr(usage_obj, attr, None)
        if isinstance(val, (int, float)) and val > 0:
            return int(val)
    return 0


def _make_token_usage(usage_obj) -> TokenUsage:
    return TokenUsage(
        input_tokens=usage_obj.prompt_tokens,
        output_tokens=usage_obj.completion_tokens,
        cache_read_tokens=_extract_cache_read_tokens(usage_obj),
    )


def _log_cache_audit(model: str, usage: TokenUsage) -> None:
    """Log cache efficiency metrics after each LLM call."""
    input_tokens = usage.input_tokens
    cached_tokens = usage.cache_read_tokens
    output_tokens = usage.output_tokens
    cache_rate = cached_tokens / input_tokens if input_tokens > 0 else 0.0
    _cache_logger.info(
        "[CACHE_AUDIT] model=%s input=%d cached=%d output=%d cache_rate=%.1f%%",
        model, input_tokens, cached_tokens, output_tokens, cache_rate * 100,
    )


def _classify_error(exc: Exception) -> None:
    """Map an API/network exception to ContextOverflowError or LLMError."""
    status_code = getattr(exc, "status_code", None)
    message = getattr(exc, "message", str(exc))

    if isinstance(exc, (asyncio.TimeoutError, TimeoutError)):
        raise LLMError("Request timed out") from exc

    if status_code == 400 and ("context_length" in str(exc).lower() or "context length" in str(message).lower()):
        raise ContextOverflowError(str(message)) from exc

    if status_code is not None:
        raise LLMError(str(message), status_code=status_code) from exc

    if isinstance(exc, (ConnectionError, OSError)):
        raise LLMError("Connection failed") from exc

    raise LLMError(str(message)) from exc


def _flatten_multimodal_content(content):
    """Convert list content blocks to a plain string (for non-vision or OpenAI tool results)."""
    if isinstance(content, str):
        return content
    parts = []
    for block in content:
        if block.get("type") == "text":
            parts.append(block.get("text", ""))
        elif block.get("type") == "image_url":
            parts.append("[image omitted]")
        elif block.get("type") == "image":
            parts.append("[image omitted]")
    return "\n".join(parts) if parts else ""


# Orbital-internal bookkeeping fields that must not be sent to LLM providers.
# These are added by Orbital during session management and would bust provider
# prefix caches if sent on the wire.
#
# ARCHITECTURAL NOTE: This is a DENYLIST, not an allowlist, on purpose.
# Provider extensions like Moonshot/DeepSeek `reasoning_content` or future
# provider-specific fields must pass through untouched. Replacing this with
# an allowlist has historically caused silent breakage on multi-turn
# tool-calling with reasoning models (see TASK-strip-spec-fix.md).
# Do not "simplify" back to an allowlist without reading that spec.
ORBITAL_INTERNAL_FIELDS = {
    "timestamp",               # ISO-8601 append time (session.py)
    "session_id",              # Owning session ID (session.py)
    "source",                  # user / management / sub-agent slug (loop.py, session.py)
    "_meta",                   # Tool-result metadata (session.py)
    "_stubbed",                # Placeholder tool-result marker (session.py)
    "_status",                 # Human-readable "Using X" label for UI (loop.py)
    "_activity_descriptions",  # Tool_call_id → description map for UI (activity_translator.py)
    "nonce",                   # Dedup key on injected user messages (loop.py)
    "target",                  # Sub-agent @mention routing target (agent_manager.py)
}


def _strip_to_spec(message: dict) -> dict:
    """Strip Orbital-internal fields from a message before sending to provider.

    Preserves all provider-native fields including provider-specific extensions
    like `reasoning_content` (Moonshot/DeepSeek), `reasoning` (OpenAI o1/o3),
    and any future extensions. Only explicitly-enumerated Orbital bookkeeping
    fields are removed.
    """
    return {k: v for k, v in message.items() if k not in ORBITAL_INTERNAL_FIELDS}


def _build_reasoning_off_switch(model: str, reasoning) -> dict | None:
    """Compute the ``extra_body`` payload that disables reasoning for the model.

    Returns:
        - dict to pass via ``extra_body=`` when reasoning can be disabled
        - None when:
            - ``reasoning`` is None
            - ``reasoning.enable`` does NOT start with ``param:`` (locked-on
              model — auto / model_only). A WARNING is logged for these cases
              so callers know the request will go out with reasoning still on.
            - the ``param:...`` value is unrecognized (also logs a WARNING)

    Mapping (prefix match on the ``enable`` string):
        param:thinking.type=enabled        -> {"thinking": {"type": "disabled"}}
        param:enable_thinking=true         -> {"enable_thinking": False}
        param:reasoning_effort=...         -> {"reasoning_effort": "minimal"}
        param:reasoning.max_tokens=...     -> {"reasoning": {"enabled": False}}
        param:reasoning.enabled=true       -> {"reasoning": {"enabled": False}}
        param:reasoning.effort=...         -> {"reasoning": {"enabled": False}}
    """
    logger = logging.getLogger(__name__)
    if reasoning is None:
        return None

    enable = getattr(reasoning, "enable", "") or ""
    if not enable.startswith("param:"):
        logger.warning(
            "disable_reasoning requested but model %s is locked-on "
            "(enable=%r); request will be sent with reasoning still active",
            model, enable,
        )
        return None

    spec = enable[len("param:"):]

    if spec.startswith("thinking.type=enabled"):
        return {"thinking": {"type": "disabled"}}
    if spec.startswith("enable_thinking=true"):
        return {"enable_thinking": False}
    if spec.startswith("reasoning_effort="):
        return {"reasoning_effort": "minimal"}
    if spec.startswith("reasoning.max_tokens="):
        return {"reasoning": {"enabled": False}}
    if spec.startswith("reasoning.enabled=true"):
        return {"reasoning": {"enabled": False}}
    if spec.startswith("reasoning.effort="):
        return {"reasoning": {"enabled": False}}

    logger.warning(
        "disable_reasoning requested but model %s has unrecognized "
        "enable=%r; request will be sent with reasoning still active",
        model, enable,
    )
    return None


def _apply_reasoning_policy(message: dict, reasoning) -> dict:
    """Enforce per-model echo_back contract on outbound assistant messages.

    Targets `role: assistant` only — user/tool/system rows are passthrough.

      required  – ensure `field` is present (default to "") so providers like
                  deepseek-v4-pro that 400 on missing thinking-history don't
                  reject multi-turn requests after a text-only final.
      forbidden – strip `field` so legacy deepseek-reasoner doesn't 400 on
                  reasoning_content being sent back as input.
      none      – defensive strip (model produces no reasoning to echo).
      optional  – passthrough (current denylist behavior).

    `reasoning` is a ReasoningInfo (frozen dataclass). When None or when
    `field` is unset, this is a no-op for backward compat.
    """
    if reasoning is None or message.get("role") != "assistant":
        return message
    field = reasoning.field
    if not field:
        return message
    echo = reasoning.echo_back
    if echo == "required":
        if field not in message:
            new_msg = dict(message)
            new_msg[field] = ""
            return new_msg
    elif echo in ("forbidden", "none"):
        if field in message:
            new_msg = dict(message)
            new_msg.pop(field)
            return new_msg
    return message


class LLMProvider:
    """Dual-SDK LLM client: OpenAI or Anthropic."""

    def __init__(self, model: str, api_key: str, base_url: str | None = None, sdk: str = "openai",
                 max_output: int = 16384, capabilities=None, reasoning=None,
                 provider: str = "unknown"):
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self.sdk = sdk  # "openai" or "anthropic"
        self.provider = provider  # vendor key (e.g. "moonshot", "anthropic", "openai") for forensic logging
        self.max_output = max_output
        self.capabilities = capabilities
        self.reasoning = reasoning  # ReasoningInfo | None — echo-back contract per model

        if sdk == "anthropic":
            import anthropic
            self._anthropic_client = anthropic.AsyncAnthropic(api_key=api_key)
            self._openai_client = None
        else:
            self._openai_client = openai.AsyncOpenAI(base_url=base_url, api_key=api_key)
            self._anthropic_client = None

    def update_api_key(self, new_key: str) -> None:
        """Hot-swap the API key, reconstructing the underlying client."""
        if new_key == self.api_key:
            return
        self.api_key = new_key
        if self.sdk == "anthropic":
            import anthropic
            self._anthropic_client = anthropic.AsyncAnthropic(api_key=new_key)
        else:
            self._openai_client = openai.AsyncOpenAI(base_url=self.base_url, api_key=new_key)

    def _prepare_messages_openai(self, messages: list) -> list:
        """Prepare messages for OpenAI SDK: strip internal fields, handle multimodal
        content, then enforce the model's reasoning echo-back contract."""
        result = []
        has_vision = self.capabilities and self.capabilities.vision
        for msg in messages:
            content = msg.get("content")
            if isinstance(content, list) and not has_vision:
                msg = dict(msg)
                msg["content"] = _flatten_multimodal_content(content)
            stripped = _strip_to_spec(msg)
            result.append(_apply_reasoning_policy(stripped, getattr(self, "reasoning", None)))
        return result

    async def stream(self, messages, tools=None) -> AsyncIterator[StreamChunk]:
        """Stream completion, yielding StreamChunk objects.

        The final chunk has is_final=True and usage populated.
        """
        if self.sdk == "anthropic":
            async for chunk in self._stream_anthropic(messages, tools):
                yield chunk
        else:
            async for chunk in self._stream_openai(messages, tools):
                yield chunk

    async def _stream_openai(self, messages, tools=None) -> AsyncIterator[StreamChunk]:
        """OpenAI SDK streaming path."""
        messages = self._prepare_messages_openai(messages)
        try:
            response_iter = await self._openai_client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=tools or None,
                stream=True,
                stream_options={"include_usage": True},
            )
        except (ContextOverflowError, LLMError):
            raise
        except Exception as exc:
            # 400-error safety net: if provider rejects multimodal tool content,
            # retry once with all image blocks flattened to text descriptions.
            status_code = getattr(exc, "status_code", None)
            has_multimodal = any(
                isinstance(m.get("content"), list) for m in messages
            )
            if status_code == 400 and has_multimodal:
                logger = logging.getLogger(__name__)
                logger.warning(
                    "Provider rejected multimodal tool content, falling back to text-only."
                )
                flat_messages = []
                for msg in messages:
                    if isinstance(msg.get("content"), list):
                        msg = dict(msg)
                        msg["content"] = _flatten_multimodal_content(msg["content"])
                    flat_messages.append(msg)
                try:
                    response_iter = await self._openai_client.chat.completions.create(
                        model=self.model,
                        messages=flat_messages,
                        tools=tools or None,
                        stream=True,
                        stream_options={"include_usage": True},
                    )
                except (ContextOverflowError, LLMError):
                    raise
                except Exception as retry_exc:
                    _classify_error(retry_exc)
            else:
                _classify_error(exc)

        async for chunk in response_iter:
            if not chunk.choices:
                if chunk.usage is not None:
                    usage = _make_token_usage(chunk.usage)
                    _log_cache_audit(self.model, usage)
                    yield StreamChunk(
                        is_final=True,
                        usage=usage,
                    )
                continue

            choice = chunk.choices[0]
            delta = choice.delta
            text = delta.content or ""
            tc_delta = delta.tool_calls or []
            reasoning = getattr(delta, "reasoning_content", None) or ""

            if choice.finish_reason is not None and chunk.usage is not None:
                usage = _make_token_usage(chunk.usage)
                _log_cache_audit(self.model, usage)
                yield StreamChunk(
                    text=text,
                    tool_calls_delta=tc_delta,
                    is_final=True,
                    usage=usage,
                    reasoning_content=reasoning,
                )
            elif chunk.usage is not None and not text and not tc_delta and not reasoning:
                usage = _make_token_usage(chunk.usage)
                _log_cache_audit(self.model, usage)
                yield StreamChunk(
                    is_final=True,
                    usage=usage,
                )
            else:
                yield StreamChunk(text=text, tool_calls_delta=tc_delta, reasoning_content=reasoning)

    async def _stream_anthropic(self, messages, tools=None) -> AsyncIterator[StreamChunk]:
        """Anthropic SDK streaming path with adapter translation."""
        messages = [_strip_to_spec(m) for m in messages]
        from agent_os.agent.providers.anthropic_adapter import (
            translate_messages_to_anthropic,
            translate_stream_event,
            StreamState,
        )

        translated = translate_messages_to_anthropic(messages, tools)

        kwargs: dict = {
            "model": self.model,
            "messages": translated["messages"],
            "max_tokens": self.max_output,
            "stream": True,
        }
        if translated["system"]:
            kwargs["system"] = translated["system"]
        if translated["tools"]:
            kwargs["tools"] = translated["tools"]

        try:
            stream = await self._anthropic_client.messages.create(**kwargs)
        except (ContextOverflowError, LLMError):
            raise
        except Exception as exc:
            self._classify_anthropic_error(exc)

        state = StreamState()

        try:
            async for event in stream:
                chunk = translate_stream_event(event, state)
                if chunk is not None:
                    yield chunk
        except (ContextOverflowError, LLMError):
            raise
        except Exception as exc:
            self._classify_anthropic_error(exc)

    async def complete(self, messages, tools=None, *, disable_reasoning: bool = False) -> LLMResponse:
        """Non-streaming completion. Returns LLMResponse directly.

        ``disable_reasoning``: when True AND ``self.reasoning.enable`` starts
        with ``"param:"``, send the matching off-switch via ``extra_body=``
        on the OpenAI SDK call. No-op for the Anthropic SDK path.
        """
        if self.sdk == "anthropic":
            return await self._complete_anthropic(messages, tools)
        return await self._complete_openai(messages, tools, disable_reasoning=disable_reasoning)

    async def _complete_openai(self, messages, tools=None, *, disable_reasoning: bool = False) -> LLMResponse:
        """OpenAI SDK non-streaming path."""
        messages = self._prepare_messages_openai(messages)
        extra_kwargs: dict = {}
        if disable_reasoning:
            off_switch = _build_reasoning_off_switch(self.model, self.reasoning)
            if off_switch is not None:
                extra_kwargs["extra_body"] = off_switch
        try:
            response = await self._openai_client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=tools or None,
                stream=False,
                **extra_kwargs,
            )
        except (ContextOverflowError, LLMError):
            raise
        except Exception as exc:
            _classify_error(exc)

        choice = response.choices[0]
        message = choice.message
        raw_message = message.model_dump()
        text = message.content
        tool_calls = message.tool_calls
        if tool_calls is None:
            tool_calls = []
        finish_reason = choice.finish_reason
        usage = _make_token_usage(response.usage)
        _log_cache_audit(self.model, usage)
        status_text = _extract_status(text)

        return LLMResponse(
            raw_message=raw_message,
            text=text,
            tool_calls=tool_calls,
            has_tool_calls=bool(tool_calls),
            finish_reason=finish_reason,
            status_text=status_text,
            usage=usage,
        )

    async def _complete_anthropic(self, messages, tools=None) -> LLMResponse:
        """Anthropic SDK non-streaming path with adapter translation."""
        messages = [_strip_to_spec(m) for m in messages]
        from agent_os.agent.providers.anthropic_adapter import (
            translate_messages_to_anthropic,
            translate_response_to_openai,
        )

        translated = translate_messages_to_anthropic(messages, tools)

        kwargs: dict = {
            "model": self.model,
            "messages": translated["messages"],
            "max_tokens": self.max_output,
        }
        if translated["system"]:
            kwargs["system"] = translated["system"]
        if translated["tools"]:
            kwargs["tools"] = translated["tools"]

        try:
            response = await self._anthropic_client.messages.create(**kwargs)
        except (ContextOverflowError, LLMError):
            raise
        except Exception as exc:
            self._classify_anthropic_error(exc)

        result = translate_response_to_openai(response)
        status_text = _extract_status(result["text"])

        return LLMResponse(
            raw_message=result["raw_message"],
            text=result["text"],
            tool_calls=result["tool_calls"],
            has_tool_calls=result["has_tool_calls"],
            finish_reason=result["finish_reason"],
            status_text=status_text,
            usage=result["usage"],
        )

    @staticmethod
    def _classify_anthropic_error(exc: Exception) -> None:
        """Map Anthropic SDK exceptions to ContextOverflowError or LLMError."""
        # Handle timeouts first
        if isinstance(exc, (asyncio.TimeoutError, TimeoutError)):
            raise LLMError("Request timed out") from exc

        # Try to import anthropic for type-specific handling
        try:
            import anthropic as anthropic_mod
        except ImportError:
            _classify_error(exc)
            return

        if isinstance(exc, anthropic_mod.APIStatusError):
            status_code = exc.status_code
            message = str(exc.message) if hasattr(exc, "message") else str(exc)

            if status_code == 400 and ("context_length" in message.lower() or "context length" in message.lower()):
                raise ContextOverflowError(message) from exc

            raise LLMError(message, status_code=status_code) from exc

        if isinstance(exc, anthropic_mod.APIError):
            message = str(exc.message) if hasattr(exc, "message") else str(exc)
            raise LLMError(message) from exc

        if isinstance(exc, (ConnectionError, OSError)):
            raise LLMError("Connection failed") from exc

        raise LLMError(str(exc)) from exc
