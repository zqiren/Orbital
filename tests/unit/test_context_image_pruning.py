# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Unit tests for context image pruning and token estimation."""

import json

import pytest

from agent_os.agent.context import ContextManager
from agent_os.agent.token_utils import estimate_message_tokens


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_multimodal_tool_msg(image_path="/workspace/photo.png", tool_call_id="call_img"):
    """Create a tool result message with multimodal image content."""
    return {
        "role": "tool",
        "tool_call_id": tool_call_id,
        "content": [
            {"type": "text", "text": f"Image file: photo.png, 100x100, 5KB, image/png"},
            {
                "type": "image_url",
                "image_url": {
                    "url": "data:image/png;base64," + "A" * 1000,
                    "detail": "low",
                },
            },
        ],
        "_meta": {"image_path": image_path, "mime": "image/png", "size": 5000},
    }


def _make_assistant_msg(content="I can see a red square in the image."):
    return {"role": "assistant", "content": content}


def _make_user_msg(content="What color was that image?"):
    return {"role": "user", "content": content}


def _make_conversation_with_image_at_turn(total_assistant_turns, image_at_turn=1):
    """Build a message sequence where image is at a specific assistant turn.

    Returns messages list. image_at_turn=1 means the image is in the first
    assistant turn. total_assistant_turns is the total number of assistant
    messages.
    """
    messages = []
    for t in range(1, total_assistant_turns + 1):
        if t == image_at_turn:
            # Assistant calls tool, tool returns image
            messages.append({
                "role": "assistant",
                "content": None,
                "tool_calls": [{"id": f"call_{t}", "type": "function",
                               "function": {"name": "read", "arguments": '{"path":"photo.png"}'}}],
            })
            messages.append(_make_multimodal_tool_msg(tool_call_id=f"call_{t}"))
            messages.append(_make_assistant_msg("I see a red square."))
        else:
            messages.append(_make_user_msg(f"Question {t}"))
            messages.append(_make_assistant_msg(f"Answer {t}"))
    return messages


# ---------------------------------------------------------------------------
# Image pruning tests
# ---------------------------------------------------------------------------

class TestImagePruning:
    def test_current_turn_image_preserved(self):
        """Image in the latest assistant turn should NOT be pruned."""
        # Only 1 assistant turn, image is in it → turns_ago = 0
        messages = [
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": "call_1", "type": "function",
                 "function": {"name": "read", "arguments": '{"path":"photo.png"}'}}
            ]},
            _make_multimodal_tool_msg(tool_call_id="call_1"),
        ]

        result = ContextManager._prune_old_tool_results(None, messages)

        tool_msg = [m for m in result if m.get("role") == "tool"][0]
        assert isinstance(tool_msg["content"], list), "Current-turn image should be preserved as list"

    def test_image_pruned_after_one_turn(self):
        """Image from 2+ assistant turns ago should be replaced with text reference."""
        # 3 assistant turns, image is in turn 1 → turns_ago = 2
        messages = _make_conversation_with_image_at_turn(
            total_assistant_turns=3, image_at_turn=1
        )

        result = ContextManager._prune_old_tool_results(None, messages)

        tool_msg = [m for m in result if m.get("role") == "tool"][0]
        assert isinstance(tool_msg["content"], str), "Old image should be pruned to string"
        assert "[Image:" in tool_msg["content"]

    def test_pruned_reference_includes_path(self):
        """Pruned image reference should include the file path from meta."""
        messages = _make_conversation_with_image_at_turn(
            total_assistant_turns=3, image_at_turn=1
        )

        result = ContextManager._prune_old_tool_results(None, messages)

        tool_msg = [m for m in result if m.get("role") == "tool"][0]
        assert "/workspace/photo.png" in tool_msg["content"]

    def test_pruned_reference_includes_text_metadata(self):
        """Pruned image reference should include text from the original text block."""
        messages = _make_conversation_with_image_at_turn(
            total_assistant_turns=3, image_at_turn=1
        )

        result = ContextManager._prune_old_tool_results(None, messages)

        tool_msg = [m for m in result if m.get("role") == "tool"][0]
        assert "photo.png" in tool_msg["content"]  # from text metadata block

    def test_no_base64_in_pruned_context(self):
        """Pruned context should NOT contain any base64 image data."""
        messages = _make_conversation_with_image_at_turn(
            total_assistant_turns=3, image_at_turn=1
        )

        result = ContextManager._prune_old_tool_results(None, messages)

        serialized = json.dumps(result)
        assert "AAAAAAA" not in serialized, "Base64 data should not survive pruning"

    def test_image_at_turn_boundary_preserved(self):
        """Image exactly 1 assistant-turn ago should still be preserved."""
        # [assistant+tc, tool_result, assistant_desc] = 2 assistant msgs
        # Tool sits after assistant turn 1, total = 2 turns, turns_ago = 2-1 = 1
        # 1 is NOT > 1, so image should be preserved
        messages = [
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": "call_1", "type": "function",
                 "function": {"name": "read", "arguments": '{"path":"photo.png"}'}}
            ]},
            _make_multimodal_tool_msg(tool_call_id="call_1"),
            _make_assistant_msg("I see a red square."),
        ]

        result = ContextManager._prune_old_tool_results(None, messages)

        tool_msg = [m for m in result if m.get("role") == "tool"][0]
        assert isinstance(tool_msg["content"], list), "Image 1 turn ago should be preserved"

    def test_screenshot_meta_also_gets_path_in_ref(self):
        """Image with screenshot_path meta (from BrowserTool) should include path."""
        messages = [
            _make_assistant_msg("Turn 1"),
            _make_assistant_msg("Turn 2"),
            _make_assistant_msg("Turn 3"),
        ]
        # Insert a tool result with screenshot_path meta between turns
        tool_msg = {
            "role": "tool",
            "tool_call_id": "call_ss",
            "content": [
                {"type": "text", "text": "Screenshot of page"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc", "detail": "low"}},
            ],
            "_meta": {"screenshot_path": "/workspace/.screenshots/page.png"},
        }
        # Place at the beginning (before all assistant turns)
        messages.insert(0, tool_msg)

        result = ContextManager._prune_old_tool_results(None, messages)

        pruned_tool = [m for m in result if m.get("role") == "tool"][0]
        assert isinstance(pruned_tool["content"], str)
        assert "/workspace/.screenshots/page.png" in pruned_tool["content"]


# ---------------------------------------------------------------------------
# Token estimation tests
# ---------------------------------------------------------------------------

class TestTokenEstimation:
    def test_text_message_estimation(self):
        msg = {"role": "user", "content": "Hello world"}
        tokens = estimate_message_tokens(msg)
        # len(json.dumps(msg)) / 4 ≈ 10-15 tokens
        assert 5 < tokens < 30

    def test_image_message_counts_85_per_image(self):
        """Image blocks should count as exactly 85 tokens, not len(base64)/4."""
        b64_data = "A" * 100_000  # 100KB of base64 = would be ~25K tokens naively
        msg = {
            "role": "tool",
            "tool_call_id": "call_1",
            "content": [
                {"type": "text", "text": "Image file: photo.png"},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_data}"}},
            ],
        }

        tokens = estimate_message_tokens(msg)

        # Should be: 85 (image) + ~5 (text "Image file: photo.png" / 4) + overhead
        # NOT: ~25,000 from the base64 data
        assert tokens < 200, f"Token estimation should be ~100, not {tokens}"
        assert tokens > 80, f"Token estimation should include 85 for image, got {tokens}"

    def test_multiple_images(self):
        msg = {
            "role": "tool",
            "tool_call_id": "call_1",
            "content": [
                {"type": "text", "text": "Two images"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64," + "A" * 50000}},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64," + "B" * 50000}},
            ],
        }

        tokens = estimate_message_tokens(msg)

        # 85 * 2 = 170 for images + text + overhead
        assert 170 < tokens < 300

    def test_string_message_unchanged(self):
        """String messages should still use len(json.dumps)/4."""
        msg = {"role": "user", "content": "x" * 400}
        tokens = estimate_message_tokens(msg)
        expected = len(json.dumps(msg)) / 4
        assert tokens == expected
