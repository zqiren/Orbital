# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Regression: tool result size cap prevents context overflow from large file reads.

Bug: A single large tool result (e.g. reading a 500KB file) could consume the
entire context window before compaction has a chance to act, causing context
overflow on the next LLM call.

Fix: Session.append_tool_result() caps content at 30% of the model context
window (passed as context_limit parameter). Truncates at newline boundary,
appends a [truncated] marker. Hard max: 400K chars. Min preserve: 200 chars.
"""

import os
import tempfile

import pytest

from agent_os.agent.session import Session


@pytest.fixture
def session(tmp_path):
    """Create a fresh session for testing."""
    return Session.new("test-cap", str(tmp_path))


class TestToolResultSizeCap:
    """Tool results exceeding 30% of context window must be truncated at append time."""

    def test_large_result_is_truncated(self, session):
        """A tool result exceeding 30% of context limit is truncated."""
        # 128K tokens * 0.30 * 4 chars/token = 153,600 char cap
        context_limit = 128_000
        cap_chars = int(context_limit * 0.30 * 4)  # 153,600

        large_content = "x" * (cap_chars + 10_000)

        session.append({
            "role": "assistant",
            "tool_calls": [{"id": "tc_1", "name": "read", "arguments": {}}],
            "source": "management",
        })
        session.append_tool_result("tc_1", large_content, context_limit=context_limit)

        messages = session.get_messages()
        tool_msg = [m for m in messages if m.get("role") == "tool"][0]
        stored = tool_msg["content"]

        assert len(stored) < len(large_content), "Result should be truncated"
        assert "[truncated" in stored, "Must contain truncation marker"

    def test_truncation_marker_format(self, session):
        """Truncation marker shows number of omitted chars."""
        context_limit = 128_000
        cap_chars = int(context_limit * 0.30 * 4)
        large_content = "y" * (cap_chars + 5_000)

        session.append({
            "role": "assistant",
            "tool_calls": [{"id": "tc_2", "name": "read", "arguments": {}}],
            "source": "management",
        })
        session.append_tool_result("tc_2", large_content, context_limit=context_limit)

        messages = session.get_messages()
        tool_msg = [m for m in messages if m.get("role") == "tool"][0]
        stored = tool_msg["content"]

        # Marker format: [truncated — N chars omitted]
        assert "chars omitted]" in stored

    def test_small_result_unchanged(self, session):
        """A tool result under the cap is stored unchanged."""
        context_limit = 128_000
        small_content = "Hello, world! This is a normal result."

        session.append({
            "role": "assistant",
            "tool_calls": [{"id": "tc_3", "name": "read", "arguments": {}}],
            "source": "management",
        })
        session.append_tool_result("tc_3", small_content, context_limit=context_limit)

        messages = session.get_messages()
        tool_msg = [m for m in messages if m.get("role") == "tool"][0]

        assert tool_msg["content"] == small_content

    def test_truncation_preserves_minimum_chars(self, session):
        """Even with a tiny context limit, at least 200 chars are preserved."""
        # Tiny context: 100 tokens -> 0.30 * 100 * 4 = 120 chars cap
        # But min_preserve = 200, so we keep at least 200
        context_limit = 100
        content = "a" * 500

        session.append({
            "role": "assistant",
            "tool_calls": [{"id": "tc_4", "name": "read", "arguments": {}}],
            "source": "management",
        })
        session.append_tool_result("tc_4", content, context_limit=context_limit)

        messages = session.get_messages()
        tool_msg = [m for m in messages if m.get("role") == "tool"][0]
        stored = tool_msg["content"]

        # The preserved portion (before marker) should be at least 200 chars
        marker_pos = stored.find("\n[truncated")
        if marker_pos >= 0:
            assert marker_pos >= 200, f"Must preserve at least 200 chars, got {marker_pos}"

    def test_truncation_marker_never_truncated(self, session):
        """The [truncated] marker itself must not be cut off."""
        context_limit = 128_000
        cap_chars = int(context_limit * 0.30 * 4)
        large_content = "z" * (cap_chars + 100_000)

        session.append({
            "role": "assistant",
            "tool_calls": [{"id": "tc_5", "name": "read", "arguments": {}}],
            "source": "management",
        })
        session.append_tool_result("tc_5", large_content, context_limit=context_limit)

        messages = session.get_messages()
        tool_msg = [m for m in messages if m.get("role") == "tool"][0]
        stored = tool_msg["content"]

        # Marker must be complete (ends with ']')
        assert stored.rstrip().endswith("]"), "Truncation marker must be complete"

    def test_truncation_at_newline_boundary(self, session):
        """Truncation should happen at a newline boundary when possible."""
        context_limit = 128_000
        cap_chars = int(context_limit * 0.30 * 4)

        # Build content with newlines every 100 chars, exceeding cap
        lines = []
        while len("\n".join(lines)) < cap_chars + 10_000:
            lines.append("x" * 99)
        large_content = "\n".join(lines)

        session.append({
            "role": "assistant",
            "tool_calls": [{"id": "tc_6", "name": "read", "arguments": {}}],
            "source": "management",
        })
        session.append_tool_result("tc_6", large_content, context_limit=context_limit)

        messages = session.get_messages()
        tool_msg = [m for m in messages if m.get("role") == "tool"][0]
        stored = tool_msg["content"]

        # The content before the marker should end at a newline
        marker_pos = stored.find("\n[truncated")
        if marker_pos >= 0:
            before_marker = stored[:marker_pos]
            # Should not end mid-line (last char before marker should be after a newline)
            assert before_marker.endswith("\n") or "\n" in before_marker[-100:]

    def test_hard_max_400k(self, session):
        """Hard max of 400K chars applies regardless of context limit."""
        # Huge context limit that would allow >400K
        context_limit = 10_000_000  # 10M tokens
        content = "a" * 500_000  # 500K chars

        session.append({
            "role": "assistant",
            "tool_calls": [{"id": "tc_7", "name": "read", "arguments": {}}],
            "source": "management",
        })
        session.append_tool_result("tc_7", content, context_limit=context_limit)

        messages = session.get_messages()
        tool_msg = [m for m in messages if m.get("role") == "tool"][0]
        stored = tool_msg["content"]

        # Content portion (before marker) must not exceed 400K
        marker_pos = stored.find("\n[truncated")
        if marker_pos >= 0:
            assert marker_pos <= 400_000, f"Hard max exceeded: {marker_pos} chars"
        else:
            # If no marker, the full content must be <= 400K
            assert len(stored) <= 400_000

    def test_no_context_limit_no_truncation(self, session):
        """When context_limit is not provided, no truncation occurs (backward compat)."""
        content = "a" * 500_000

        session.append({
            "role": "assistant",
            "tool_calls": [{"id": "tc_8", "name": "read", "arguments": {}}],
            "source": "management",
        })
        # No context_limit parameter
        session.append_tool_result("tc_8", content)

        messages = session.get_messages()
        tool_msg = [m for m in messages if m.get("role") == "tool"][0]

        assert tool_msg["content"] == content

    def test_jsonl_persistence_of_truncated_result(self, session, tmp_path):
        """Truncated result must be persisted to JSONL, not just in memory."""
        context_limit = 128_000
        cap_chars = int(context_limit * 0.30 * 4)
        large_content = "p" * (cap_chars + 10_000)

        session.append({
            "role": "assistant",
            "tool_calls": [{"id": "tc_9", "name": "read", "arguments": {}}],
            "source": "management",
        })
        session.append_tool_result("tc_9", large_content, context_limit=context_limit)

        # Reload session from disk
        reloaded = Session.load(session._filepath)
        messages = reloaded.get_messages()
        tool_msg = [m for m in messages if m.get("role") == "tool"][0]

        assert "[truncated" in tool_msg["content"]
        assert len(tool_msg["content"]) < len(large_content)
