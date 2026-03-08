# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Tests for multimodal content handling across the provider pipeline."""

import json
import pytest

from agent_os.agent.providers.anthropic_adapter import (
    translate_messages_to_anthropic,
    _translate_content_to_anthropic,
    _parse_data_url,
)
from agent_os.agent.providers.openai_compat import _flatten_multimodal_content
from agent_os.agent.tools.base import ToolResult
from agent_os.agent.tool_result_filters import dispatch_prefilter


# --- Data URL parsing ---

class TestParseDataUrl:
    def test_parse_png_data_url(self):
        url = "data:image/png;base64,iVBORw0KGgo="
        media, data = _parse_data_url(url)
        assert media == "image/png"
        assert data == "iVBORw0KGgo="

    def test_parse_jpeg_data_url(self):
        url = "data:image/jpeg;base64,/9j/4AAQ="
        media, data = _parse_data_url(url)
        assert media == "image/jpeg"
        assert data == "/9j/4AAQ="

    def test_non_data_url_returns_default(self):
        url = "https://example.com/image.png"
        media, data = _parse_data_url(url)
        assert media == "image/png"
        assert data == url


# --- Content translation ---

class TestContentTranslation:
    def test_string_passes_through(self):
        assert _translate_content_to_anthropic("hello") == "hello"

    def test_text_blocks_translated(self):
        blocks = [{"type": "text", "text": "hello world"}]
        result = _translate_content_to_anthropic(blocks)
        assert result == [{"type": "text", "text": "hello world"}]

    def test_image_url_to_anthropic_image(self):
        blocks = [
            {"type": "text", "text": "Screenshot"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,ABC123"}},
        ]
        result = _translate_content_to_anthropic(blocks)
        assert len(result) == 2
        assert result[0] == {"type": "text", "text": "Screenshot"}
        assert result[1]["type"] == "image"
        assert result[1]["source"]["type"] == "base64"
        assert result[1]["source"]["media_type"] == "image/png"
        assert result[1]["source"]["data"] == "ABC123"

    def test_anthropic_image_passes_through(self):
        """Blocks already in Anthropic format should pass through."""
        blocks = [
            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "XYZ"}},
        ]
        result = _translate_content_to_anthropic(blocks)
        assert result == blocks


# --- Anthropic adapter tool result handling ---

class TestAnthropicAdapterToolResults:
    def test_string_tool_result_unchanged(self):
        messages = [
            {"role": "assistant", "content": "I'll check.", "tool_calls": [
                {"id": "tc1", "type": "function", "function": {"name": "read", "arguments": "{}"}}
            ]},
            {"role": "tool", "content": "file content here", "tool_call_id": "tc1"},
        ]
        result = translate_messages_to_anthropic(messages)
        tool_result_msg = result["messages"][-1]
        assert tool_result_msg["role"] == "user"
        blocks = tool_result_msg["content"]
        assert blocks[0]["type"] == "tool_result"
        assert blocks[0]["content"] == "file content here"

    def test_multimodal_tool_result_translated(self):
        multimodal_content = [
            {"type": "text", "text": "Screenshot of page"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,IMG"}},
        ]
        messages = [
            {"role": "assistant", "content": "Taking screenshot", "tool_calls": [
                {"id": "tc2", "type": "function", "function": {"name": "browser", "arguments": "{}"}}
            ]},
            {"role": "tool", "content": multimodal_content, "tool_call_id": "tc2"},
        ]
        result = translate_messages_to_anthropic(messages)
        tool_result_msg = result["messages"][-1]
        blocks = tool_result_msg["content"]
        assert blocks[0]["type"] == "tool_result"
        tool_content = blocks[0]["content"]
        # Should be translated to Anthropic format
        assert isinstance(tool_content, list)
        assert tool_content[0]["type"] == "text"
        assert tool_content[1]["type"] == "image"
        assert tool_content[1]["source"]["data"] == "IMG"


# --- Anthropic adapter user message handling ---

class TestAnthropicAdapterUserMessages:
    def test_string_user_message(self):
        messages = [{"role": "user", "content": "hello"}]
        result = translate_messages_to_anthropic(messages)
        assert result["messages"][0]["content"] == "hello"

    def test_multimodal_user_message(self):
        content = [
            {"type": "text", "text": "What is this?"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,ABC"}},
        ]
        messages = [{"role": "user", "content": content}]
        result = translate_messages_to_anthropic(messages)
        user_content = result["messages"][0]["content"]
        assert isinstance(user_content, list)
        assert user_content[0]["type"] == "text"
        assert user_content[1]["type"] == "image"


# --- OpenAI message preparation ---

class TestFlattenMultimodal:
    def test_string_passes_through(self):
        assert _flatten_multimodal_content("hello") == "hello"

    def test_list_with_text_blocks(self):
        blocks = [
            {"type": "text", "text": "Line 1"},
            {"type": "text", "text": "Line 2"},
        ]
        result = _flatten_multimodal_content(blocks)
        assert result == "Line 1\nLine 2"

    def test_list_with_image_blocks_omitted(self):
        blocks = [
            {"type": "text", "text": "Screenshot"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,ABC"}},
        ]
        result = _flatten_multimodal_content(blocks)
        assert "Screenshot" in result
        assert "[image omitted]" in result


# --- ToolResult accepts both types ---

class TestToolResult:
    def test_string_content(self):
        r = ToolResult(content="hello")
        assert r.content == "hello"

    def test_list_content(self):
        blocks = [{"type": "text", "text": "hello"}]
        r = ToolResult(content=blocks)
        assert isinstance(r.content, list)
        assert r.content[0]["type"] == "text"

    def test_with_meta(self):
        r = ToolResult(content="ok", meta={"key": "val"})
        assert r.meta == {"key": "val"}


# --- Tool result pre-filter ---

class TestPrefilterMultimodal:
    def test_list_content_passes_through(self):
        blocks = [{"type": "text", "text": "hello"}, {"type": "image_url", "image_url": {"url": "..."}}]
        result = dispatch_prefilter("browser", {"action": "screenshot"}, blocks)
        assert result is blocks  # exact same object, not modified

    def test_empty_content_passes_through(self):
        assert dispatch_prefilter("browser", {}, "") == ""
        assert dispatch_prefilter("browser", {}, []) == []


# --- JSONL serialization ---

class TestJsonlSerialization:
    def test_string_content_serializes(self):
        msg = {"role": "tool", "content": "hello", "tool_call_id": "tc1"}
        line = json.dumps(msg, ensure_ascii=False)
        loaded = json.loads(line)
        assert loaded["content"] == "hello"

    def test_list_content_serializes(self):
        blocks = [
            {"type": "text", "text": "Screenshot"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,ABC"}},
        ]
        msg = {"role": "tool", "content": blocks, "tool_call_id": "tc1"}
        line = json.dumps(msg, ensure_ascii=False)
        loaded = json.loads(line)
        assert isinstance(loaded["content"], list)
        assert loaded["content"][0]["type"] == "text"
        assert loaded["content"][1]["type"] == "image_url"
