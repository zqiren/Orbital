# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Bidirectional format translation between OpenAI and Anthropic APIs.

OpenAI format is the internal lingua franca for Agent OS. All messages in
the session, context manager, and tool layer use OpenAI format. This adapter
translates outbound (OpenAI -> Anthropic API) and inbound (Anthropic API ->
OpenAI) so that Component C can route to Anthropic transparently.

Functions:
    translate_messages_to_anthropic  -- outbound: messages + tools
    translate_response_to_openai     -- inbound: complete() response
    translate_stream_event           -- inbound: stream() event
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from agent_os.agent.providers.types import TokenUsage, StreamChunk


# ---------------------------------------------------------------------------
# Content block translation helpers
# ---------------------------------------------------------------------------

def _translate_content_to_anthropic(content) -> str | list:
    """Translate content (str or list of blocks) from OpenAI format to Anthropic format.

    - str content passes through as-is
    - list content: image_url blocks are converted to Anthropic image blocks
    """
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return str(content)

    anthropic_blocks = []
    for block in content:
        block_type = block.get("type", "")
        if block_type == "text":
            anthropic_blocks.append({"type": "text", "text": block.get("text", "")})
        elif block_type == "image_url":
            # OpenAI format: {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}
            data_url = block.get("image_url", {}).get("url", "")
            media_type, b64_data = _parse_data_url(data_url)
            anthropic_blocks.append({
                "type": "image",
                "source": {"type": "base64", "media_type": media_type, "data": b64_data},
            })
        elif block_type == "image":
            # Already in Anthropic format
            anthropic_blocks.append(block)
        else:
            # Unknown block type — pass through
            anthropic_blocks.append(block)
    return anthropic_blocks


def _parse_data_url(data_url: str) -> tuple[str, str]:
    """Parse a data URL into (media_type, base64_data)."""
    # Format: data:image/png;base64,iVBOR...
    if data_url.startswith("data:"):
        header, _, data = data_url.partition(",")
        media_type = header.split(":")[1].split(";")[0] if ":" in header else "image/png"
        return media_type, data
    # Not a data URL — return as-is (might be a regular URL)
    return "image/png", data_url


# ---------------------------------------------------------------------------
# Outbound: OpenAI format -> Anthropic API format
# ---------------------------------------------------------------------------

def translate_messages_to_anthropic(
    messages: list[dict],
    tools: list[dict] | None = None,
) -> dict:
    """Convert OpenAI-format messages and tool schemas to Anthropic API format.

    Returns a dict with keys:
        system   -- str or None (extracted from system-role messages)
        messages -- list of Anthropic-format message dicts
        tools    -- list of Anthropic-format tool schemas
    """
    system_parts: list[str] = []
    anthropic_messages: list[dict] = []

    i = 0
    while i < len(messages):
        msg = messages[i]
        role = msg.get("role", "")

        # 1. System message extraction
        if role == "system":
            content = msg.get("content", "")
            if content:
                system_parts.append(content)
            i += 1
            continue

        # 4. Assistant messages with tool_calls -> content blocks
        if role == "assistant":
            anthropic_messages.append(_translate_assistant_message(msg))
            i += 1
            continue

        # 3. Tool result messages -> user message with tool_result blocks
        if role == "tool":
            # Collect consecutive tool results
            tool_result_blocks: list[dict] = []
            while i < len(messages) and messages[i].get("role") == "tool":
                tool_msg = messages[i]
                raw_content = tool_msg.get("content", "")
                translated = _translate_content_to_anthropic(raw_content)
                tool_result_blocks.append({
                    "type": "tool_result",
                    "tool_use_id": tool_msg.get("tool_call_id", ""),
                    "content": translated,
                })
                i += 1
            anthropic_messages.append({
                "role": "user",
                "content": tool_result_blocks,
            })
            continue

        # User messages: translate multimodal content if present
        if role == "user":
            raw_content = msg.get("content", "")
            translated = _translate_content_to_anthropic(raw_content)
            anthropic_messages.append({
                "role": "user",
                "content": translated,
            })
            i += 1
            continue

        # Unknown role — pass through as-is
        anthropic_messages.append(msg)
        i += 1

    # 2. Tool schema translation
    anthropic_tools: list[dict] = []
    if tools:
        for tool in tools:
            func = tool.get("function", {})
            anthropic_tools.append({
                "name": func.get("name", ""),
                "description": func.get("description", ""),
                "input_schema": func.get("parameters", {}),
            })

    return {
        "system": "\n".join(system_parts) if system_parts else None,
        "messages": anthropic_messages,
        "tools": anthropic_tools,
    }


def _translate_assistant_message(msg: dict) -> dict:
    """Translate an OpenAI assistant message to Anthropic content blocks.

    OpenAI format:
        {"role": "assistant", "content": "text", "tool_calls": [...]}
    Anthropic format:
        {"role": "assistant", "content": [{"type": "text", ...}, {"type": "tool_use", ...}]}
    """
    content_blocks: list[dict] = []
    text = msg.get("content")
    tool_calls = msg.get("tool_calls")

    if text:
        content_blocks.append({"type": "text", "text": text})

    if tool_calls:
        for tc in tool_calls:
            func = tc.get("function", {})
            arguments_str = func.get("arguments", "{}")
            try:
                input_dict = json.loads(arguments_str)
            except (json.JSONDecodeError, TypeError):
                input_dict = {}
            content_blocks.append({
                "type": "tool_use",
                "id": tc.get("id", ""),
                "name": func.get("name", ""),
                "input": input_dict,
            })

    # If no content and no tool_calls, use empty content blocks
    if not content_blocks:
        content_blocks.append({"type": "text", "text": ""})

    return {
        "role": "assistant",
        "content": content_blocks,
    }


# ---------------------------------------------------------------------------
# Inbound: Anthropic API response -> OpenAI format
# ---------------------------------------------------------------------------

_STOP_REASON_MAP = {
    "end_turn": "stop",
    "tool_use": "tool_calls",
    "max_tokens": "length",
}


def translate_response_to_openai(response) -> dict:
    """Convert a native Anthropic Messages API response to OpenAI-format dict.

    The returned dict has keys compatible with LLMResponse construction:
        raw_message, text, tool_calls, has_tool_calls, finish_reason,
        status_text (always None here; caller extracts), usage.
    """
    # Extract text and tool_use blocks from response.content
    text_parts: list[str] = []
    tool_calls: list[dict] = []

    for block in getattr(response, "content", []):
        block_type = getattr(block, "type", "")
        if block_type == "text":
            text_parts.append(getattr(block, "text", ""))
        elif block_type == "tool_use":
            tool_calls.append({
                "id": getattr(block, "id", ""),
                "type": "function",
                "function": {
                    "name": getattr(block, "name", ""),
                    "arguments": json.dumps(getattr(block, "input", {})),
                },
            })

    text = "".join(text_parts) if text_parts else None

    # Build raw_message in OpenAI format
    raw_message: dict = {"role": "assistant", "content": text}
    if tool_calls:
        raw_message["tool_calls"] = tool_calls

    # Finish reason mapping
    stop_reason = getattr(response, "stop_reason", "end_turn") or "end_turn"
    finish_reason = _STOP_REASON_MAP.get(stop_reason, stop_reason)

    # Usage mapping
    usage_obj = getattr(response, "usage", None)
    if usage_obj is not None:
        usage = TokenUsage(
            input_tokens=getattr(usage_obj, "input_tokens", 0),
            output_tokens=getattr(usage_obj, "output_tokens", 0),
            cache_read_tokens=getattr(usage_obj, "cache_read_input_tokens", 0) or 0,
        )
    else:
        usage = TokenUsage(0, 0)

    return {
        "raw_message": raw_message,
        "text": text,
        "tool_calls": tool_calls,
        "has_tool_calls": bool(tool_calls),
        "finish_reason": finish_reason,
        "usage": usage,
    }


# ---------------------------------------------------------------------------
# Inbound streaming: Anthropic stream events -> StreamChunk
# ---------------------------------------------------------------------------

@dataclass
class StreamState:
    """Tracks state across Anthropic streaming events.

    Anthropic streams are stateful: content_block_start tells you the block
    type and index, then deltas arrive referencing that block. This class
    tracks the active blocks so translate_stream_event can produce correct
    StreamChunk objects.
    """
    # Map block index -> {"type": str, "id": str, "name": str, "first_delta": bool}
    blocks: dict[int, dict] = field(default_factory=dict)
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    stop_reason: str | None = None


def translate_stream_event(event, state: StreamState) -> StreamChunk | None:
    """Convert an Anthropic streaming event to a StreamChunk, or None.

    Args:
        event: A native Anthropic streaming event object.
        state: Mutable StreamState tracking cross-event state.

    Returns:
        StreamChunk with translated data, or None for setup-only events.
    """
    event_type = getattr(event, "type", "")

    # message_start: extract initial usage, no chunk emitted
    if event_type == "message_start":
        message = getattr(event, "message", None)
        if message:
            usage = getattr(message, "usage", None)
            if usage:
                state.input_tokens = getattr(usage, "input_tokens", 0)
                state.output_tokens = getattr(usage, "output_tokens", 0)
                state.cache_read_tokens = (
                    getattr(usage, "cache_read_input_tokens", 0) or 0
                )
        return None

    # content_block_start: register block type/index
    if event_type == "content_block_start":
        index = getattr(event, "index", 0)
        content_block = getattr(event, "content_block", None)
        block_type = getattr(content_block, "type", "text") if content_block else "text"
        block_info: dict = {"type": block_type, "first_delta": True}
        if block_type == "tool_use" and content_block is not None:
            block_info["id"] = getattr(content_block, "id", "")
            block_info["name"] = getattr(content_block, "name", "")
        state.blocks[index] = block_info
        return None

    # content_block_delta: emit text or tool_calls_delta
    if event_type == "content_block_delta":
        index = getattr(event, "index", 0)
        delta = getattr(event, "delta", None)
        if delta is None:
            return None

        delta_type = getattr(delta, "type", "")
        block_info = state.blocks.get(index, {})

        # Text delta
        if delta_type == "text_delta":
            text = getattr(delta, "text", "")
            return StreamChunk(text=text)

        # Tool use input JSON delta
        if delta_type == "input_json_delta":
            partial_json = getattr(delta, "partial_json", "")
            tc_delta: dict = {
                "index": index,
                "type": "function",
                "function": {"arguments": partial_json},
            }
            # Include id and name only on first delta for this block
            if block_info.get("first_delta", False):
                tc_delta["id"] = block_info.get("id", "")
                tc_delta["function"]["name"] = block_info.get("name", "")
                block_info["first_delta"] = False
            return StreamChunk(tool_calls_delta=[tc_delta])

        return None

    # content_block_stop: finalize block, no chunk emitted
    if event_type == "content_block_stop":
        return None

    # message_delta: extract stop_reason and usage delta
    if event_type == "message_delta":
        delta = getattr(event, "delta", None)
        if delta:
            state.stop_reason = getattr(delta, "stop_reason", None)
        usage = getattr(event, "usage", None)
        if usage:
            state.output_tokens += getattr(usage, "output_tokens", 0)
        return None

    # message_stop: emit final chunk with accumulated usage
    if event_type == "message_stop":
        return StreamChunk(
            is_final=True,
            usage=TokenUsage(
                input_tokens=state.input_tokens,
                output_tokens=state.output_tokens,
                cache_read_tokens=state.cache_read_tokens,
            ),
        )

    # Unknown event type — ignore
    return None
