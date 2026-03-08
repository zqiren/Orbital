# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Regression: orphaned tool results stripped from LLM context.

Bug: When daemon crashes during approval pause, CANCELLED tool results
end up far from the assistant message that issued them. Strict LLMs
(Moonshot/Kimi) reject the conversation with HTTP 400.

Fix 1: ContextManager._validate_tool_results() Pass 2 strips any
role:"tool" message whose tool_call_id doesn't match a tool_calls entry
in the immediately preceding assistant message block.

Fix 2: Session.heal_orphaned_tool_calls() inserts CANCELLED results
adjacent to the original assistant message on session load, and rewrites
the JSONL atomically.
"""

import json
import os
import shutil

import pytest

from agent_os.agent.context import ContextManager
from agent_os.agent.session import Session


# ---------------------------------------------------------------------------
# Fix 1 tests: ContextManager orphan stripping
# ---------------------------------------------------------------------------


class TestOrphanedToolResultStripping:
    """Pass 2 of _validate_tool_results strips orphaned tool messages."""

    def test_orphaned_tool_result_at_end_stripped(self):
        """A tool result far from its assistant message is dropped."""
        messages = [
            {"role": "assistant", "tool_calls": [{"id": "tc_1", "function": {"name": "shell"}}], "content": None},
            {"role": "tool", "tool_call_id": "tc_1", "content": "ok"},
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi there"},
            # Orphaned: tc_1 was already resolved, this is a stale CANCELLED
            {"role": "tool", "tool_call_id": "tc_1", "content": "CANCELLED"},
        ]
        result = ContextManager._validate_tool_results(messages)
        tool_msgs = [m for m in result if m.get("role") == "tool"]
        assert len(tool_msgs) == 1
        assert tool_msgs[0]["content"] == "ok"

    def test_correctly_positioned_results_preserved(self):
        """Normal session with properly paired tool_calls and results."""
        messages = [
            {"role": "assistant", "tool_calls": [
                {"id": "tc_1", "function": {"name": "shell"}},
                {"id": "tc_2", "function": {"name": "read"}},
            ], "content": None},
            {"role": "tool", "tool_call_id": "tc_1", "content": "output 1"},
            {"role": "tool", "tool_call_id": "tc_2", "content": "output 2"},
            {"role": "user", "content": "thanks"},
            {"role": "assistant", "content": "done"},
        ]
        result = ContextManager._validate_tool_results(messages)
        assert result == messages

    def test_mixed_orphaned_and_valid(self):
        """Some valid, one orphaned result injected between blocks."""
        messages = [
            # Block 1: valid
            {"role": "assistant", "tool_calls": [{"id": "tc_1"}], "content": None},
            {"role": "tool", "tool_call_id": "tc_1", "content": "ok"},
            # Orphaned: not preceded by any assistant with tool_calls
            {"role": "tool", "tool_call_id": "tc_orphan", "content": "CANCELLED"},
            # Block 2: valid
            {"role": "assistant", "tool_calls": [{"id": "tc_2"}], "content": None},
            {"role": "tool", "tool_call_id": "tc_2", "content": "ok2"},
        ]
        result = ContextManager._validate_tool_results(messages)
        tool_msgs = [m for m in result if m.get("role") == "tool"]
        assert len(tool_msgs) == 2
        assert {m["tool_call_id"] for m in tool_msgs} == {"tc_1", "tc_2"}

    def test_interaction_with_missing_result_injection(self):
        """Missing result gets synthetic injection, orphan gets stripped."""
        messages = [
            # Block with missing result for tc_2
            {"role": "assistant", "tool_calls": [
                {"id": "tc_1"},
                {"id": "tc_2"},
            ], "content": None},
            {"role": "tool", "tool_call_id": "tc_1", "content": "ok"},
            # tc_2 result is missing — Pass 1 should inject synthetic
            {"role": "user", "content": "next"},
            {"role": "assistant", "content": "thinking"},
            # Orphaned result for tc_2 appended at end (the bug scenario)
            {"role": "tool", "tool_call_id": "tc_2", "content": "CANCELLED"},
        ]
        result = ContextManager._validate_tool_results(messages)

        # tc_1 result should be preserved
        tc1_results = [m for m in result if m.get("role") == "tool" and m.get("tool_call_id") == "tc_1"]
        assert len(tc1_results) == 1

        # tc_2 should have synthetic injection (from Pass 1), orphan stripped (Pass 2)
        tc2_results = [m for m in result if m.get("role") == "tool" and m.get("tool_call_id") == "tc_2"]
        assert len(tc2_results) == 1
        assert "lost" in tc2_results[0]["content"].lower() or "error" in tc2_results[0]["content"].lower()

        # The orphaned CANCELLED at end should be gone
        last_tool = [m for m in result if m.get("role") == "tool"][-1]
        assert last_tool["tool_call_id"] != "tc_2" or "CANCELLED" not in last_tool["content"]

    def test_orphan_after_text_only_assistant(self):
        """Tool result after text-only assistant is orphaned."""
        messages = [
            {"role": "assistant", "content": "I'll help you"},
            {"role": "tool", "tool_call_id": "tc_stale", "content": "CANCELLED"},
        ]
        result = ContextManager._validate_tool_results(messages)
        tool_msgs = [m for m in result if m.get("role") == "tool"]
        assert len(tool_msgs) == 0

    def test_orphan_after_user_message(self):
        """Tool result after a user message is orphaned."""
        messages = [
            {"role": "assistant", "tool_calls": [{"id": "tc_1"}], "content": None},
            {"role": "tool", "tool_call_id": "tc_1", "content": "ok"},
            {"role": "user", "content": "next question"},
            {"role": "tool", "tool_call_id": "tc_1", "content": "CANCELLED: duplicate"},
        ]
        result = ContextManager._validate_tool_results(messages)
        tool_msgs = [m for m in result if m.get("role") == "tool"]
        assert len(tool_msgs) == 1
        assert tool_msgs[0]["content"] == "ok"


# ---------------------------------------------------------------------------
# Fix 2 tests: Session startup recovery
# ---------------------------------------------------------------------------


def _write_jsonl(filepath: str, messages: list[dict]) -> None:
    """Write messages to JSONL file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        for msg in messages:
            f.write(json.dumps(msg, ensure_ascii=False) + "\n")


class TestSessionStartupRecovery:
    """heal_orphaned_tool_calls() inserts CANCELLED adjacent on load."""

    def test_orphaned_tool_calls_healed_on_load(self, tmp_path):
        """Unresolved tool call gets CANCELLED inserted adjacent to assistant msg."""
        filepath = str(tmp_path / "test_session.jsonl")
        messages = [
            {"role": "user", "content": "do something"},
            {"role": "assistant", "tool_calls": [
                {"id": "tc_1", "function": {"name": "shell"}},
                {"id": "tc_2", "function": {"name": "read"}},
            ], "content": None},
            {"role": "tool", "tool_call_id": "tc_1", "content": "ok"},
            # tc_2 is missing — simulates crash before result was written
            {"role": "user", "content": "what happened?"},
            {"role": "assistant", "content": "Let me check."},
        ]
        _write_jsonl(filepath, messages)

        session = Session.load(filepath)

        # pending_tool_calls should be empty after healing
        assert len(session.pending_tool_calls) == 0

        # tc_2 CANCELLED should be inserted right after tc_1 result
        healed_msgs = session.get_messages()
        tc2_results = [m for m in healed_msgs if m.get("role") == "tool" and m.get("tool_call_id") == "tc_2"]
        assert len(tc2_results) == 1
        assert "CANCELLED" in tc2_results[0]["content"]
        assert "session interrupted" in tc2_results[0]["content"]

        # Verify position: tc_2 result should be after tc_1 result, before user message
        tc1_idx = next(i for i, m in enumerate(healed_msgs) if m.get("tool_call_id") == "tc_1")
        tc2_idx = next(i for i, m in enumerate(healed_msgs) if m.get("tool_call_id") == "tc_2")
        user_idx = next(i for i, m in enumerate(healed_msgs) if m.get("role") == "user" and "what" in m.get("content", ""))
        assert tc2_idx == tc1_idx + 1
        assert tc2_idx < user_idx

    def test_no_orphans_no_rewrite(self, tmp_path):
        """Clean session: JSONL is not rewritten (check content hash)."""
        filepath = str(tmp_path / "clean_session.jsonl")
        messages = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "tool_calls": [{"id": "tc_1"}], "content": None},
            {"role": "tool", "tool_call_id": "tc_1", "content": "done"},
            {"role": "assistant", "content": "complete"},
        ]
        _write_jsonl(filepath, messages)

        # Record file content before load
        with open(filepath, "r", encoding="utf-8") as f:
            content_before = f.read()

        session = Session.load(filepath)
        assert len(session.pending_tool_calls) == 0

        # File should not be rewritten
        with open(filepath, "r", encoding="utf-8") as f:
            content_after = f.read()
        assert content_before == content_after

    def test_multiple_orphaned_batches_healed(self, tmp_path):
        """Two separate assistant messages with unresolved tool calls."""
        filepath = str(tmp_path / "multi_orphan.jsonl")
        messages = [
            # Batch 1: tc_a resolved, tc_b orphaned
            {"role": "assistant", "tool_calls": [
                {"id": "tc_a"},
                {"id": "tc_b"},
            ], "content": None},
            {"role": "tool", "tool_call_id": "tc_a", "content": "ok"},
            {"role": "user", "content": "continue"},
            # Batch 2: tc_c orphaned
            {"role": "assistant", "tool_calls": [{"id": "tc_c"}], "content": None},
            {"role": "user", "content": "still waiting"},
        ]
        _write_jsonl(filepath, messages)

        session = Session.load(filepath)
        assert len(session.pending_tool_calls) == 0

        healed_msgs = session.get_messages()
        # tc_b should be healed adjacent to batch 1
        tc_b_results = [m for m in healed_msgs if m.get("tool_call_id") == "tc_b"]
        assert len(tc_b_results) == 1
        assert "CANCELLED" in tc_b_results[0]["content"]

        # tc_c should be healed adjacent to batch 2
        tc_c_results = [m for m in healed_msgs if m.get("tool_call_id") == "tc_c"]
        assert len(tc_c_results) == 1
        assert "CANCELLED" in tc_c_results[0]["content"]

    def test_healing_preserves_message_order(self, tmp_path):
        """After healing, every assistant with tool_calls is followed by correct tool results."""
        filepath = str(tmp_path / "order_test.jsonl")
        messages = [
            {"role": "user", "content": "start"},
            {"role": "assistant", "tool_calls": [
                {"id": "tc_1"},
                {"id": "tc_2"},
                {"id": "tc_3"},
            ], "content": None},
            {"role": "tool", "tool_call_id": "tc_1", "content": "ok"},
            # tc_2 and tc_3 are orphaned
            {"role": "user", "content": "next"},
            {"role": "assistant", "content": "done"},
        ]
        _write_jsonl(filepath, messages)

        session = Session.load(filepath)
        healed_msgs = session.get_messages()

        # Find the assistant with tool_calls
        for i, msg in enumerate(healed_msgs):
            if msg.get("role") == "assistant" and msg.get("tool_calls"):
                tc_ids = {tc["id"] for tc in msg["tool_calls"]}
                # All results should immediately follow
                results = []
                j = i + 1
                while j < len(healed_msgs) and healed_msgs[j].get("role") == "tool":
                    results.append(healed_msgs[j]["tool_call_id"])
                    j += 1
                assert set(results) == tc_ids, \
                    f"Expected tool results {tc_ids} immediately after assistant, got {results}"

    def test_healed_jsonl_persists_to_disk(self, tmp_path):
        """JSONL file is rewritten with healed messages."""
        filepath = str(tmp_path / "persist_test.jsonl")
        messages = [
            {"role": "assistant", "tool_calls": [{"id": "tc_1"}], "content": None},
            {"role": "user", "content": "hello"},
        ]
        _write_jsonl(filepath, messages)

        Session.load(filepath)

        # Reload from disk — should have the CANCELLED result
        reloaded = Session.load(filepath)
        tc1_results = [m for m in reloaded.get_messages() if m.get("tool_call_id") == "tc_1"]
        assert len(tc1_results) == 1
        assert "CANCELLED" in tc1_results[0]["content"]
        # And no more pending
        assert len(reloaded.pending_tool_calls) == 0
