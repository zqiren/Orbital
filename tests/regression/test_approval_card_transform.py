# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Regression tests for approval card rendering from chat history and transcripts.

Bug 2: chatTransform.ts previously skipped ALL system messages, including ones
with _meta.approval_request=true. After page reload, stored approval events
were invisible.

Bug 3: Sub-agent transcript entries with chunk_type="approval_request" were
rendered as inert sub_agent_message text instead of actionable approval cards.

These tests validate the data shapes that the frontend transform function
processes. Since chatTransform is TypeScript, we test the *contracts* here:
the input shapes the backend produces and the expected output classification.
"""

import json
import pytest


def _classify_message(msg: dict) -> str:
    """Pure-Python reimplementation of the chatTransform classification logic.

    This mirrors the TypeScript transformChatHistory() after the bug fixes,
    returning the DisplayItem type that should be produced for each message.
    """
    if msg.get("_compaction"):
        return "skip"

    role = msg.get("role")

    if role == "system":
        meta = msg.get("_meta") or {}
        if meta.get("approval_request"):
            return "approval_card"
        return "skip"

    if role == "user":
        return "user_message"

    if role == "agent":
        if msg.get("chunk_type") == "approval_request":
            return "approval_card"
        content = (msg.get("content") or "").strip()
        if content and content != "(no response)":
            return "sub_agent_message"
        return "skip"

    if role == "assistant":
        if msg.get("tool_calls"):
            return "activity_block"
        if msg.get("content"):
            return "agent_message"
        return "skip"

    if role == "tool":
        return "skip"

    return "skip"


class TestBug2SystemApprovalNotSkipped:
    """Bug 2: System messages with _meta.approval_request should produce approval_card."""

    def test_system_approval_request_produces_approval_card(self):
        """A system message stored by useChatHistory.handleApprovalRequest
        must be classified as approval_card, not skipped."""
        msg = {
            "role": "system",
            "content": "Permission requested: Edit file.txt",
            "source": "management",
            "timestamp": "2026-03-13T10:00:00Z",
            "_meta": {
                "approval_request": True,
                "tool_call_id": "tc_123",
                "tool_name": "edit",
                "tool_args": {"file_path": "/tmp/file.txt", "content": "hello"},
            },
        }
        assert _classify_message(msg) == "approval_card"

    def test_system_approval_with_resolution_produces_approval_card(self):
        """A resolved approval stored in history should still be approval_card
        (the resolution field tells the card to render as resolved)."""
        msg = {
            "role": "system",
            "content": "Permission requested: shell command",
            "source": "management",
            "timestamp": "2026-03-13T10:01:00Z",
            "_meta": {
                "approval_request": True,
                "tool_call_id": "tc_456",
                "tool_name": "shell",
                "tool_args": {"command": "rm -rf /tmp/test"},
                "resolution": "denied",
            },
        }
        assert _classify_message(msg) == "approval_card"

    def test_plain_system_message_still_skipped(self):
        """Regular system messages (status summaries, etc.) should still be skipped."""
        msg = {
            "role": "system",
            "content": "Agent completed 5 tasks",
            "source": "management",
            "timestamp": "2026-03-13T10:02:00Z",
        }
        assert _classify_message(msg) == "skip"

    def test_system_message_with_empty_meta_still_skipped(self):
        """System messages with _meta but no approval_request flag are skipped."""
        msg = {
            "role": "system",
            "content": "Status update",
            "source": "management",
            "timestamp": "2026-03-13T10:03:00Z",
            "_meta": {"some_other_field": True},
        }
        assert _classify_message(msg) == "skip"

    def test_compacted_system_approval_still_skipped(self):
        """Compacted messages should always be skipped, even with approval_request."""
        msg = {
            "role": "system",
            "content": "Old approval",
            "source": "management",
            "timestamp": "2026-03-13T09:00:00Z",
            "_compaction": True,
            "_meta": {"approval_request": True},
        }
        assert _classify_message(msg) == "skip"


class TestBug3TranscriptApprovalRequest:
    """Bug 3: Agent messages with chunk_type=approval_request should produce approval_card."""

    def test_agent_approval_request_produces_approval_card(self):
        """A transcript entry with chunk_type=approval_request loaded as role=agent
        must be classified as approval_card, not sub_agent_message."""
        msg = {
            "role": "agent",
            "content": "Sub-agent coder requests approval",
            "source": "coder",
            "timestamp": "2026-03-13T10:05:00Z",
            "chunk_type": "approval_request",
        }
        assert _classify_message(msg) == "approval_card"

    def test_agent_response_still_produces_sub_agent_message(self):
        """Normal agent transcript entries (response/message) should still
        produce sub_agent_message."""
        msg = {
            "role": "agent",
            "content": "I've completed the refactoring.",
            "source": "coder",
            "timestamp": "2026-03-13T10:06:00Z",
            "chunk_type": "response",
        }
        assert _classify_message(msg) == "sub_agent_message"

    def test_agent_without_chunk_type_produces_sub_agent_message(self):
        """Agent messages without chunk_type still produce sub_agent_message."""
        msg = {
            "role": "agent",
            "content": "Working on the task...",
            "source": "coder",
            "timestamp": "2026-03-13T10:07:00Z",
        }
        assert _classify_message(msg) == "sub_agent_message"

    def test_agent_approval_with_meta_produces_approval_card(self):
        """If the backend eventually stores metadata on transcript entries,
        the card should still be classified correctly."""
        msg = {
            "role": "agent",
            "content": "Sub-agent coder requests approval",
            "source": "coder",
            "timestamp": "2026-03-13T10:08:00Z",
            "chunk_type": "approval_request",
            "_meta": {
                "tool_name": "shell",
                "tool_call_id": "tc_789",
                "tool_args": {"command": "npm install"},
            },
        }
        assert _classify_message(msg) == "approval_card"

    def test_empty_agent_message_skipped(self):
        """Agent messages with empty content are skipped."""
        msg = {
            "role": "agent",
            "content": "",
            "source": "coder",
            "timestamp": "2026-03-13T10:09:00Z",
            "chunk_type": "response",
        }
        assert _classify_message(msg) == "skip"

    def test_no_response_agent_message_skipped(self):
        """Agent messages with '(no response)' content are skipped."""
        msg = {
            "role": "agent",
            "content": "(no response)",
            "source": "coder",
            "timestamp": "2026-03-13T10:10:00Z",
        }
        assert _classify_message(msg) == "skip"


class TestApprovalCardDataShape:
    """Validate the data shape that the approval_card DisplayItem should contain."""

    def test_system_approval_card_fields(self):
        """System approval messages must provide all fields needed by ApprovalCard."""
        msg = {
            "role": "system",
            "content": "Permission requested: Edit web/index.ts",
            "source": "management",
            "timestamp": "2026-03-13T10:00:00Z",
            "_meta": {
                "approval_request": True,
                "tool_call_id": "tc_abc",
                "tool_name": "edit",
                "tool_args": {"file_path": "web/index.ts"},
                "reasoning": "Need to fix a bug in the config",
            },
        }
        assert _classify_message(msg) == "approval_card"
        meta = msg["_meta"]
        # These are the fields ApprovalCard expects
        assert meta["tool_call_id"]
        assert meta["tool_name"]
        assert isinstance(meta["tool_args"], dict)

    def test_transcript_approval_card_limited_metadata(self):
        """Transcript-loaded approval entries may lack tool metadata since
        ProcessManager only writes source, content, timestamp, chunk_type.
        The card should still render (with empty tool info)."""
        msg = {
            "role": "agent",
            "content": "Sub-agent coder requests approval",
            "source": "coder",
            "timestamp": "2026-03-13T10:05:00Z",
            "chunk_type": "approval_request",
            # No _meta — this is the normal case for transcript entries
        }
        assert _classify_message(msg) == "approval_card"
        # The transform will use empty defaults for missing metadata
        assert msg.get("_meta") is None  # confirms metadata is absent


class TestFullTransformPipeline:
    """End-to-end test of the classify function on a realistic message sequence."""

    def test_mixed_history_with_approvals(self):
        """A realistic chat history should produce the correct item types."""
        messages = [
            {"role": "user", "content": "Fix the bug", "source": "user",
             "timestamp": "2026-03-13T10:00:00Z"},
            {"role": "assistant", "content": "I'll look into it.", "source": "main",
             "timestamp": "2026-03-13T10:00:01Z"},
            {"role": "system", "content": "Agent status update", "source": "management",
             "timestamp": "2026-03-13T10:00:02Z"},
            {"role": "system", "content": "Permission requested: shell",
             "source": "management", "timestamp": "2026-03-13T10:00:03Z",
             "_meta": {"approval_request": True, "tool_call_id": "tc_1",
                       "tool_name": "shell", "tool_args": {"command": "npm test"}}},
            {"role": "agent", "content": "Running tests...", "source": "coder",
             "timestamp": "2026-03-13T10:00:04Z", "chunk_type": "response"},
            {"role": "agent", "content": "Sub-agent coder requests approval",
             "source": "coder", "timestamp": "2026-03-13T10:00:05Z",
             "chunk_type": "approval_request"},
            {"role": "assistant", "content": "All done.", "source": "main",
             "timestamp": "2026-03-13T10:00:06Z"},
        ]

        types = [_classify_message(m) for m in messages]
        assert types == [
            "user_message",       # user message
            "agent_message",      # assistant text
            "skip",               # plain system message
            "approval_card",      # Bug 2 fix: system approval -> card
            "sub_agent_message",  # normal agent transcript
            "approval_card",      # Bug 3 fix: agent approval_request -> card
            "agent_message",      # assistant final response
        ]
