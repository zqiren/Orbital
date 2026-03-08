# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Unit tests for provider vision handling — image block preservation and flattening."""

import pytest

from agent_os.agent.providers.openai_compat import LLMProvider, _flatten_multimodal_content
from agent_os.agent.providers.anthropic_adapter import _translate_content_to_anthropic


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_multimodal_tool_result():
    """Return a tool message with multimodal content (text + image)."""
    return {
        "role": "tool",
        "tool_call_id": "call_123",
        "content": [
            {"type": "text", "text": "Image file: photo.png, 100x100, 5KB, image/png"},
            {
                "type": "image_url",
                "image_url": {
                    "url": "data:image/png;base64,iVBORw0KGgoAAAANS",
                    "detail": "low",
                },
            },
        ],
    }


def _make_vision_capabilities():
    """Return a capabilities object with vision=True."""
    class Caps:
        vision = True
    return Caps()


def _make_no_vision_capabilities():
    """Return a capabilities object with vision=False."""
    class Caps:
        vision = False
    return Caps()


# ---------------------------------------------------------------------------
# OpenAI provider tests
# ---------------------------------------------------------------------------

class TestVisionModelPreservesImages:
    def test_vision_model_preserves_image_blocks_in_tool_result(self):
        """Vision-capable model should keep multimodal content in tool results."""
        provider = LLMProvider.__new__(LLMProvider)
        provider.capabilities = _make_vision_capabilities()

        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Read the image."},
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": "call_123", "type": "function", "function": {"name": "read", "arguments": '{"path":"photo.png"}'}}
            ]},
            _make_multimodal_tool_result(),
        ]

        result = provider._prepare_messages_openai(messages)

        tool_msg = [m for m in result if m.get("role") == "tool"][0]
        assert isinstance(tool_msg["content"], list), "Vision model should preserve list content"
        assert tool_msg["content"][1]["type"] == "image_url"

    def test_vision_model_preserves_user_multimodal(self):
        """Vision model should also preserve multimodal user messages."""
        provider = LLMProvider.__new__(LLMProvider)
        provider.capabilities = _make_vision_capabilities()

        messages = [
            {"role": "user", "content": [
                {"type": "text", "text": "What's in this image?"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc123"}},
            ]},
        ]

        result = provider._prepare_messages_openai(messages)

        assert isinstance(result[0]["content"], list)


class TestNonVisionModelFlattens:
    def test_non_vision_model_flattens_image_blocks(self):
        """Non-vision model should flatten image blocks to text descriptions."""
        provider = LLMProvider.__new__(LLMProvider)
        provider.capabilities = _make_no_vision_capabilities()

        messages = [_make_multimodal_tool_result()]

        result = provider._prepare_messages_openai(messages)

        tool_msg = result[0]
        assert isinstance(tool_msg["content"], str), "Non-vision should flatten to string"
        assert "[image omitted]" in tool_msg["content"]
        assert "photo.png" in tool_msg["content"]

    def test_no_capabilities_flattens(self):
        """Provider with no capabilities should flatten (safe default)."""
        provider = LLMProvider.__new__(LLMProvider)
        provider.capabilities = None

        messages = [_make_multimodal_tool_result()]

        result = provider._prepare_messages_openai(messages)

        assert isinstance(result[0]["content"], str)

    def test_string_content_passes_through(self):
        """String tool results should pass through unchanged regardless of vision."""
        provider = LLMProvider.__new__(LLMProvider)
        provider.capabilities = _make_no_vision_capabilities()

        messages = [{"role": "tool", "tool_call_id": "x", "content": "File content here"}]

        result = provider._prepare_messages_openai(messages)

        assert result[0]["content"] == "File content here"


# ---------------------------------------------------------------------------
# Flatten function tests
# ---------------------------------------------------------------------------

class TestFlattenMultimodalContent:
    def test_flatten_image_url(self):
        content = [
            {"type": "text", "text": "Image file: photo.png"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}},
        ]
        result = _flatten_multimodal_content(content)
        assert "[image omitted]" in result
        assert "photo.png" in result

    def test_flatten_anthropic_image(self):
        content = [
            {"type": "text", "text": "Screenshot"},
            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "abc"}},
        ]
        result = _flatten_multimodal_content(content)
        assert "[image omitted]" in result

    def test_flatten_string_passthrough(self):
        assert _flatten_multimodal_content("hello") == "hello"

    def test_flatten_empty_list(self):
        assert _flatten_multimodal_content([]) == ""


# ---------------------------------------------------------------------------
# Anthropic adapter tests
# ---------------------------------------------------------------------------

class TestAnthropicAdapterConvertsImageFormat:
    def test_image_url_to_anthropic_image(self):
        """OpenAI image_url format should convert to Anthropic image source format."""
        content = [
            {"type": "text", "text": "Image file: photo.png"},
            {
                "type": "image_url",
                "image_url": {"url": "data:image/jpeg;base64,/9j/4AAQSkZJRg=="},
            },
        ]

        result = _translate_content_to_anthropic(content)

        assert isinstance(result, list)
        assert result[0] == {"type": "text", "text": "Image file: photo.png"}
        assert result[1]["type"] == "image"
        assert result[1]["source"]["type"] == "base64"
        assert result[1]["source"]["media_type"] == "image/jpeg"
        assert result[1]["source"]["data"] == "/9j/4AAQSkZJRg=="

    def test_string_content_passes_through(self):
        assert _translate_content_to_anthropic("hello") == "hello"

    def test_already_anthropic_format_passes_through(self):
        content = [
            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "abc"}},
        ]
        result = _translate_content_to_anthropic(content)
        assert result[0]["type"] == "image"
