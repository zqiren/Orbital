# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Regression: tool result truncation prevents stale context domination (CS-10).

Bug: After a browser-heavy turn (7 snapshots ≈ 49K tokens), subsequent turns
have degraded context quality — the sliding window is dominated by stale
HTML/JSON noise, causing the LLM to go idle instead of acting.

Fix: truncate_consumed_tool_results() replaces consumed tool results with
compact stubs after the LLM responds, saving full content to disk.
"""

import json
import os

import pytest

from agent_os.agent.session import Session
from agent_os.agent.tool_result_lifecycle import truncate_consumed_tool_results


@pytest.fixture
def workspace(tmp_path):
    return str(tmp_path)


@pytest.fixture
def session(workspace):
    return Session.new("cs10-test", workspace)


def _add_tool_call_and_result(session, call_id, tool_name, arguments, content):
    """Helper: add an assistant tool_call message and its tool result."""
    session.append({
        "role": "assistant",
        "tool_calls": [{
            "id": call_id,
            "type": "function",
            "function": {
                "name": tool_name,
                "arguments": json.dumps(arguments),
            },
        }],
        "source": "management",
    })
    session.append_tool_result(call_id, content)


class TestToolResultTruncationCS10:
    """Reproduce CS-10: 7 large tool results must become stubs after LLM responds."""

    def test_seven_large_results_become_stubs(self, session):
        """7 browser snapshot results (≥5K tokens each) are replaced with stubs."""
        # Simulate 7 browser snapshot tool calls with large content
        large_content = "A" * 20_000  # ~5K tokens

        for i in range(7):
            _add_tool_call_and_result(
                session,
                call_id=f"tc_snap_{i}",
                tool_name="browser",
                arguments={"action": "snapshot", "url": f"https://example.com/page{i}"},
                content=large_content,
            )

        # Simulate LLM responding (this is when truncation fires)
        assistant_text = "I analyzed all 7 pages. Here is a summary of the content."
        truncate_consumed_tool_results(session, assistant_text, iteration=1)

        # Verify all 7 tool results are now stubs
        messages = session.get_messages()
        tool_msgs = [m for m in messages if m.get("role") == "tool"]
        assert len(tool_msgs) == 7

        for msg in tool_msgs:
            assert msg.get("_stubbed") is True
            content = msg["content"]
            assert content.startswith("[Tool:")
            assert "Agent summary:" in content
            assert assistant_text[:100] in content
            # Stub must be much smaller than original
            assert len(content) < 1000

    def test_stubs_contain_metadata(self, session):
        """Stubs include tool name, target URL, token count, and disk path."""
        _add_tool_call_and_result(
            session,
            call_id="tc_meta",
            tool_name="browser",
            arguments={"action": "snapshot", "url": "https://figma.com/design"},
            content="X" * 10_000,
        )

        truncate_consumed_tool_results(session, "Analyzed the design page.", iteration=2)

        messages = session.get_messages()
        tool_msg = [m for m in messages if m.get("role") == "tool"][0]
        stub = tool_msg["content"]

        assert "Tool: browser" in stub
        assert "Target: https://figma.com/design" in stub
        assert "Original: 2500 tokens" in stub
        assert "Full result:" in stub
        assert ".json" in stub

    def test_stubs_contain_agent_summary(self, session):
        """Stubs contain the agent's response text as summary."""
        _add_tool_call_and_result(
            session, "tc_sum", "browser",
            {"action": "snapshot", "url": "https://example.com"},
            "Y" * 5_000,
        )

        summary = "The page shows a login form with email and password fields."
        truncate_consumed_tool_results(session, summary, iteration=1)

        messages = session.get_messages()
        tool_msg = [m for m in messages if m.get("role") == "tool"][0]
        assert summary in tool_msg["content"]

    def test_full_content_not_in_session_after_truncation(self, session):
        """Original large content must NOT remain in session messages."""
        original = "UNIQUE_MARKER_CONTENT_" + "Z" * 10_000
        _add_tool_call_and_result(
            session, "tc_gone", "browser",
            {"action": "snapshot", "url": "https://example.com"},
            original,
        )

        truncate_consumed_tool_results(session, "Done.", iteration=1)

        # Verify original content is gone
        raw_session = json.dumps(session.get_messages())
        assert "UNIQUE_MARKER_CONTENT_" not in raw_session

    def test_small_results_not_stubbed(self, session):
        """Results under 500 chars should NOT be stubbed."""
        small_content = "Exit code: 0\nHello world"
        _add_tool_call_and_result(
            session, "tc_small", "shell",
            {"command": "echo hello"},
            small_content,
        )

        truncate_consumed_tool_results(session, "Command succeeded.", iteration=1)

        messages = session.get_messages()
        tool_msg = [m for m in messages if m.get("role") == "tool"][0]
        assert tool_msg["content"] == small_content
        assert not tool_msg.get("_stubbed")

    def test_already_stubbed_not_re_processed(self, session):
        """Stubs with _stubbed=True are not re-processed on next truncation."""
        _add_tool_call_and_result(
            session, "tc_once", "browser",
            {"action": "snapshot", "url": "https://example.com"},
            "W" * 5_000,
        )

        truncate_consumed_tool_results(session, "First response.", iteration=1)

        # Get the stub content
        messages = session.get_messages()
        tool_msg = [m for m in messages if m.get("role") == "tool"][0]
        first_stub = tool_msg["content"]

        # Second truncation should NOT change the stub
        truncate_consumed_tool_results(session, "Second response.", iteration=2)

        messages = session.get_messages()
        tool_msg = [m for m in messages if m.get("role") == "tool"][0]
        assert tool_msg["content"] == first_stub

    def test_multi_tool_calls_share_same_summary(self, session):
        """Multiple tool calls in one turn all get the same agent summary."""
        # Single assistant message with 3 tool calls
        session.append({
            "role": "assistant",
            "tool_calls": [
                {"id": f"tc_multi_{i}", "type": "function",
                 "function": {"name": "read", "arguments": json.dumps({"path": f"file{i}.txt"})}}
                for i in range(3)
            ],
            "source": "management",
        })
        for i in range(3):
            session.append_tool_result(f"tc_multi_{i}", "Q" * 2_000)

        summary = "I read all three files and compared their contents."
        truncate_consumed_tool_results(session, summary, iteration=1)

        messages = session.get_messages()
        tool_msgs = [m for m in messages if m.get("role") == "tool"]
        assert len(tool_msgs) == 3

        for msg in tool_msgs:
            assert msg.get("_stubbed") is True
            assert summary in msg["content"]

    def test_token_savings_cs10_scenario(self, session):
        """CS-10 scenario: 49K tokens reduced to ~2.1K tokens in stubs."""
        # 7 snapshots × ~7K tokens each = ~49K tokens
        for i in range(7):
            _add_tool_call_and_result(
                session, f"tc_big_{i}", "browser",
                {"action": "snapshot", "url": f"https://example.com/p{i}"},
                "B" * 28_000,  # ~7K tokens
            )

        original_total = sum(
            len(m.get("content", ""))
            for m in session.get_messages()
            if m.get("role") == "tool"
        )
        assert original_total > 180_000  # ~49K tokens * 4 chars/token

        truncate_consumed_tool_results(
            session, "Summary of all pages.", iteration=1,
        )

        stub_total = sum(
            len(m.get("content", ""))
            for m in session.get_messages()
            if m.get("role") == "tool"
        )
        # Should achieve >90% reduction
        assert stub_total < original_total * 0.10

    def test_jsonl_persistence_after_truncation(self, session, workspace):
        """Stubs must persist to JSONL file, not just in memory."""
        _add_tool_call_and_result(
            session, "tc_persist", "browser",
            {"action": "snapshot", "url": "https://example.com"},
            "P" * 8_000,
        )

        truncate_consumed_tool_results(session, "Persisted.", iteration=1)

        # Reload session from disk
        reloaded = Session.load(session._filepath)
        tool_msgs = [m for m in reloaded.get_messages() if m.get("role") == "tool"]
        assert len(tool_msgs) == 1
        assert tool_msgs[0].get("_stubbed") is True
        assert tool_msgs[0]["content"].startswith("[Tool:")
