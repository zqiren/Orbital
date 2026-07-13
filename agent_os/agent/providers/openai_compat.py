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
from agent_os.agent.providers.think_splitter import InlineThinkSplitter

_cache_logger = logging.getLogger("orbital.cache_audit")


def _extract_status(text: str | None) -> str | None:
    if not text:
        return None
    match = re.search(r'\[STATUS:\s*(.+?)\]', text)
    return match.group(1).strip() if match else None


def _extract_cache_read_tokens(usage_obj) -> int:
    """Extract cached token count — handles Anthropic, OpenAI, and Kimi field names.

    Probe order (first positive hit wins):
    1. Top-level: cache_read_input_tokens, prompt_cache_hit_tokens, cached_tokens
       (Anthropic / Kimi / older OpenAI-compat servers).
    2. Nested: usage.prompt_tokens_details.cached_tokens
       (official OpenAI API — cache hits live here, not at top level).

    The nested probe handles three shapes without raising:
    - prompt_tokens_details absent or None  → skip
    - prompt_tokens_details is an attribute-access object → getattr
    - prompt_tokens_details is a plain dict  → dict.get
    """
    for attr in ("cache_read_input_tokens", "prompt_cache_hit_tokens", "cached_tokens"):
        val = getattr(usage_obj, attr, None)
        if isinstance(val, (int, float)) and val > 0:
            return int(val)
    # Fallback: official OpenAI nested path
    details = getattr(usage_obj, "prompt_tokens_details", None)
    if details is None:
        return 0
    if isinstance(details, dict):
        val = details.get("cached_tokens")
    else:
        val = getattr(details, "cached_tokens", None)
    if isinstance(val, (int, float)) and val > 0:
        return int(val)
    return 0


def _make_token_usage(usage_obj) -> TokenUsage:
    # Some OpenAI-compatible endpoints (observed: MiniMax api.minimaxi.com)
    # return a usage object whose prompt_tokens / completion_tokens are None.
    # Coerce to 0 so downstream int math (_log_cache_audit, cumulative token
    # tracking, budget) never hits "'>' not supported between NoneType and int",
    # which previously crashed the whole turn mid-stream.
    # cache_write_tokens is always 0 for OpenAI-compat providers: none of them
    # expose a cache-write field (cached_tokens/prompt_cache_hit_tokens are
    # read-hit counts only).
    return TokenUsage(
        input_tokens=getattr(usage_obj, "prompt_tokens", 0) or 0,
        output_tokens=getattr(usage_obj, "completion_tokens", 0) or 0,
        cache_read_tokens=_extract_cache_read_tokens(usage_obj),
        cache_write_tokens=0,
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
    "session_id",              # F1 user-facing chat id (session.py)
    "session_uuid",            # F2 JSONL stem (session.py)
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

    payload = _reasoning_off_payload(spec)
    if payload is not None:
        return payload

    logger.warning(
        "disable_reasoning requested but model %s has unrecognized "
        "enable=%r; request will be sent with reasoning still active",
        model, enable,
    )
    return None


def _reasoning_off_payload(spec: str) -> dict | None:
    """Pure ``param:`` spec → reasoning-off ``extra_body`` mapping (no logging).

    Shared by ``_build_reasoning_off_switch`` (which adds the WARNING logs) and
    ``LLMProvider.reasoning_locked_on`` (which must probe silently).
    """
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
    return None


def _content_is_blank(content) -> bool:
    """True when a message's content is empty/whitespace-only.

    A list (multimodal) with any block is treated as non-blank — image-only
    turns are legitimate. None and "" / "   " are blank.
    """
    if content is None:
        return True
    if isinstance(content, list):
        return len(content) == 0
    if isinstance(content, str):
        return content.strip() == ""
    return False


# Sent when an OpenAI-compatible request would otherwise carry zero non-system
# turns. The onboarding flow is driven entirely by the system prompt (see
# PromptBuilder._onboarding_or_directive), so a fresh agent's very first request
# is system-only. Lenient providers answer that; strict ones (MiniMax
# api.minimaxi.com → 400 "chat content is empty (2013)") require at least one
# user/assistant message with real content. A minimal user kickoff gives them a
# turn to respond to without changing the persisted session history.
_KICKOFF_CONTENT = "Begin."


def _ensure_chat_content(messages: list) -> list:
    """Guarantee the outbound OpenAI-style chat array is acceptable to strict
    providers (e.g. MiniMax) that reject empty content.

    Two defects are repaired, both at the provider boundary only (the persisted
    session is never mutated):

    1. A user/system message whose content is blank ("" / whitespace / None)
       gets a single-space placeholder. Assistant messages are left alone —
       an assistant turn carrying only ``tool_calls`` legitimately has
       content None/"" and must stay that way.
    2. If, after step 1, there is no user OR assistant message at all (the
       system-prompt-only onboarding kickoff), a minimal user turn is
       appended so the provider has a conversational turn to answer.
    """
    if not messages:
        return [{"role": "user", "content": _KICKOFF_CONTENT}]

    repaired: list = []
    has_chat_turn = False
    for msg in messages:
        role = msg.get("role")
        if role in ("user", "assistant"):
            has_chat_turn = True
        if role in ("system", "user") and _content_is_blank(msg.get("content")):
            # Must be genuinely non-blank: MiniMax strips whitespace before its
            # emptiness check, so a single space would still trip error 2013.
            msg = dict(msg)
            msg["content"] = "(no content)"
        repaired.append(msg)

    if not has_chat_turn:
        repaired.append({"role": "user", "content": _KICKOFF_CONTENT})
    return repaired


def _ensure_tool_result_contiguity(messages: list) -> list:
    """Guarantee every tool-result block is contiguous for strict providers.

    A ``role:"tool"`` result must immediately follow the assistant message that
    holds the matching ``tool_calls`` (or another tool result of the same
    batch). A persisted turn can violate this when a non-tool message is written
    between siblings — e.g. a turn-yielding dispatch injects a ``system`` echo
    after its own result, before the cancelled siblings are appended:
    ``tool(_1) → system(echo) → tool(_2) → tool(_3)``. MiniMax rejects that with
    400 "tool call result does not follow tool call (2013)".

    This reorders ONLY: for each assistant ``tool_calls`` message, the matched
    tool results are pulled up to be contiguous (keeping their relative order),
    and any non-tool message interleaved *within* the block is relocated to just
    after the block (keeping its relative order). Nothing is fabricated,
    dropped, or merged; messages after the block's last tool result are left in
    place; the input list and its dicts are never mutated.

    Reorder-only by contract: this relies on ``ContextManager._validate_tool_results``
    having already run (it does, on every send path, before message prep) to inject
    missing results and drop true orphans/duplicate tool_call_ids. This helper cannot
    repair a tool result with no live owner — it only moves existing rows. Keep this
    pass *after* that validation; moving it ahead could re-expose the 400.
    """
    n = len(messages)
    consumed = [False] * n
    out: list = []
    i = 0
    while i < n:
        if consumed[i]:
            i += 1
            continue
        msg = messages[i]
        out.append(msg)
        consumed[i] = True

        tool_calls = msg.get("role") == "assistant" and msg.get("tool_calls")
        if tool_calls:
            ids = {
                tc.get("id")
                for tc in tool_calls
                if isinstance(tc, dict) and tc.get("id")
            }
            if ids:
                tool_results: list = []   # matched results, in encounter order
                displaced: list = []      # non-tool rows before the last match
                pending: list = []        # non-tool rows since the last match
                seen: set = set()
                j = i + 1
                while j < n and seen != ids:
                    m = messages[j]
                    # A new tool_calls turn bounds this block — stop scanning.
                    if m.get("role") == "assistant" and m.get("tool_calls"):
                        break
                    tcid = m.get("tool_call_id")
                    if m.get("role") == "tool" and tcid in ids and tcid not in seen:
                        tool_results.append(j)
                        seen.add(tcid)
                        displaced.extend(pending)
                        pending = []
                    else:
                        pending.append(j)
                    j += 1
                # Emit matched results contiguously, then the interleaved rows.
                # `pending` (rows after the last matched result) stays in place —
                # the outer loop appends it in original order.
                for idx in tool_results + displaced:
                    out.append(messages[idx])
                    consumed[idx] = True
        i += 1
    return out


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
            # base_url must be threaded through: a custom/self-hosted Anthropic-
            # format router (e.g. co.yes.vg) is reached only when the SDK client
            # points at it. Omitting it silently sends the router's key to the
            # default api.anthropic.com → 401 "Invalid API key". None keeps the
            # SDK default endpoint.
            self._anthropic_client = anthropic.AsyncAnthropic(api_key=api_key, base_url=base_url)
            self._openai_client = None
        else:
            self._openai_client = openai.AsyncOpenAI(base_url=base_url, api_key=api_key)
            self._anthropic_client = None

    @property
    def reasoning_locked_on(self) -> bool:
        """True when the model always reasons and no request param can turn it
        off (``enable`` is auto/model_only, or an unrecognized ``param:`` spec).

        ``disable_reasoning=True`` is a no-op for these models — callers doing
        long utility calls (e.g. the session-end consolidation merge) use this
        to budget timeouts realistically instead of retrying a ladder the model
        can never beat (MiniMax-M3 incident, 2026-07-09).
        """
        r = self.reasoning
        if r is None or not getattr(r, "supported", False):
            return False
        enable = getattr(r, "enable", "") or ""
        if not enable.startswith("param:"):
            return True
        return _reasoning_off_payload(enable[len("param:"):]) is None

    def _inline_think_mode(self) -> bool:
        """True when this model emits reasoning inline as ``<think>…</think>``
        within ``content`` (ReasoningInfo.field is None + supported). Such
        models need the think segments peeled into ``reasoning_content``; models
        with a named reasoning delta field are left untouched."""
        r = self.reasoning
        return bool(
            r is not None
            and getattr(r, "supported", False)
            and getattr(r, "field", "sentinel") is None
        )

    def _inline_think_tag(self) -> str:
        """The reasoning tag name for this model's inline-think content. Sourced
        from ReasoningInfo.inline_tag (provider config); defaults to ``think``
        for models that don't declare a namespaced tag (e.g. MiniMax-M3 uses
        ``mm:think``)."""
        return getattr(self.reasoning, "inline_tag", "think") or "think"

    def update_api_key(self, new_key: str) -> None:
        """Hot-swap the API key, reconstructing the underlying client."""
        if new_key == self.api_key:
            return
        self.api_key = new_key
        if self.sdk == "anthropic":
            import anthropic
            self._anthropic_client = anthropic.AsyncAnthropic(api_key=new_key, base_url=self.base_url)
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
        # Final pass: strict OpenAI-compatible providers (MiniMax) reject any
        # message with empty content and reject system-only requests. Repair
        # both here, at the wire boundary, so the persisted session is untouched.
        repaired = _ensure_chat_content(result)
        # Then guarantee every tool-result block is contiguous (strict providers
        # 400 on a tool result that does not immediately follow its tool_calls).
        return _ensure_tool_result_contiguity(repaired)

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

        # Inline-think models (e.g. MiniMax M-series) stream reasoning inside
        # <think>…</think> within `content`. Peel it into reasoning_content as
        # deltas arrive (tags may span chunk boundaries). None for other models.
        splitter = (
            InlineThinkSplitter(tag=self._inline_think_tag())
            if self._inline_think_mode() else None
        )

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

            if splitter is not None and text:
                visible, think = splitter.feed(text)
                text = visible
                reasoning = reasoning + think

            if choice.finish_reason is not None and chunk.usage is not None:
                usage = _make_token_usage(chunk.usage)
                _log_cache_audit(self.model, usage)
                yield StreamChunk(
                    text=text,
                    tool_calls_delta=tc_delta,
                    is_final=True,
                    usage=usage,
                    reasoning_content=reasoning,
                    finish_reason=choice.finish_reason,
                )
            elif chunk.usage is not None and not text and not tc_delta and not reasoning:
                usage = _make_token_usage(chunk.usage)
                _log_cache_audit(self.model, usage)
                yield StreamChunk(
                    is_final=True,
                    usage=usage,
                )
            else:
                yield StreamChunk(
                    text=text,
                    tool_calls_delta=tc_delta,
                    reasoning_content=reasoning,
                    finish_reason=choice.finish_reason,
                )

        # Flush any buffered inline-think remainder (an unclosed <think> block,
        # or a trailing partial tag) once the upstream stream is exhausted.
        if splitter is not None:
            visible, think = splitter.flush()
            if visible or think:
                yield StreamChunk(text=visible, reasoning_content=think)

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

        # Inline-think models: peel <think>…</think> out of content into
        # reasoning_content so the persisted message keeps a clean answer.
        if self._inline_think_mode() and isinstance(text, str) and text:
            splitter = InlineThinkSplitter(tag=self._inline_think_tag())
            visible, think = splitter.feed(text)
            v2, t2 = splitter.flush()
            text = visible + v2
            extracted = think + t2
            raw_message["content"] = text
            if extracted:
                raw_message["reasoning_content"] = (
                    (raw_message.get("reasoning_content") or "") + extracted
                )

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
