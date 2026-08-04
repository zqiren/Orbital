# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Regression: CS-10 must not be "fixed" by blanket-stubbing tool results.

Original bug (CS-10): after a browser-heavy turn (7 snapshots ≈ 49K tokens),
subsequent turns had degraded context quality — the sliding window was
dominated by stale HTML/JSON noise.

Original fix, now REVERTED: truncate_consumed_tool_results() replaced EVERY
consumed tool result over 500 chars with a stub after every LLM response. That
traded one bug for a worse one — the agent lost documents mid-turn while still
reasoning over them (measured: a 7,282-token listing read once, then reasoned
about in detail 20 messages later from a stub, wrongly), and the stub was a
same-shaped object that read like a successful retrieval, so the model's own
history suppressed the re-read.

Current contract: archive_and_supersede_tool_results() exports large results to
disk unconditionally and rewrites history ONLY where a later fetch of the same
target supersedes an earlier copy. Context pressure is compaction's job.
"""

import json
import os

import pytest

from agent_os.agent.session import Session
from agent_os.agent.tool_result_lifecycle import (
    archive_and_supersede_tool_results,
)


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
    """The CS-10 workload must keep its content and still be archived."""

    def test_seven_distinct_snapshots_stay_live(self, session):
        """7 browser snapshots of 7 different URLs: nothing is stubbed."""
        large_content = "A" * 20_000  # ~5K tokens

        for i in range(7):
            _add_tool_call_and_result(
                session,
                call_id=f"tc_snap_{i}",
                tool_name="browser",
                arguments={"action": "snapshot", "url": f"https://example.com/page{i}"},
                content=large_content,
            )

        # This is where the old blanket truncation fired.
        archive_and_supersede_tool_results(session, iteration=1)

        messages = session.get_messages()
        tool_msgs = [m for m in messages if m.get("role") == "tool"]
        assert len(tool_msgs) == 7

        for msg in tool_msgs:
            assert not msg.get("_stubbed")
            assert msg["content"] == large_content

    def test_repeated_snapshot_supersedes_the_prior_copy(self, session):
        """Re-snapshotting one URL stubs the earlier copy and keeps the newest."""
        for i in range(7):
            _add_tool_call_and_result(
                session,
                call_id=f"tc_same_{i}",
                tool_name="browser",
                arguments={"action": "snapshot", "url": "https://example.com/live"},
                content=f"STATE_{i}_" + "A" * 20_000,
            )

        archive_and_supersede_tool_results(session, iteration=1)

        tool_msgs = [m for m in session.get_messages() if m.get("role") == "tool"]
        live = [m for m in tool_msgs if not m.get("_stubbed")]
        assert len(live) == 1
        assert live[0]["content"].startswith("STATE_6_")
        assert len([m for m in tool_msgs if m.get("_stubbed")]) == 6

    def test_superseded_stubs_contain_metadata(self, session):
        """Superseded stubs include tool name, target URL, token count, disk path."""
        for i in range(2):
            _add_tool_call_and_result(
                session,
                call_id=f"tc_meta_{i}",
                tool_name="browser",
                arguments={"action": "snapshot", "url": "https://figma.com/design"},
                content="X" * 10_000,
            )

        archive_and_supersede_tool_results(session, iteration=2)

        tool_msgs = [m for m in session.get_messages() if m.get("role") == "tool"]
        stub = tool_msgs[0]["content"]

        assert "Tool: browser" in stub
        assert "Target: https://figma.com/design" in stub
        assert "Original: 2500 tokens" in stub
        assert "Full result:" in stub
        assert ".json" in stub

    def test_stubs_do_not_embed_narration(self, session):
        """Stubs must NOT embed the agent's narration as a faux content summary."""
        narration = "The page shows a login form with email and password fields."
        _add_tool_call_and_result(
            session, "tc_sum_1", "browser",
            {"action": "snapshot", "url": "https://example.com"},
            "Y" * 5_000,
        )
        session.append({
            "role": "assistant", "content": narration, "source": "management",
        })
        _add_tool_call_and_result(
            session, "tc_sum_2", "browser",
            {"action": "snapshot", "url": "https://example.com"},
            "Z" * 5_000,
        )

        archive_and_supersede_tool_results(session, iteration=1)

        content = [
            m for m in session.get_messages() if m.get("role") == "tool"
        ][0]["content"]
        assert narration not in content
        assert "Agent summary:" not in content
        assert "NOT the content" in content
        assert "re-read" in content.lower()

    def test_full_content_survives_a_single_fetch(self, session):
        """The one thing the old behaviour got wrong: content must NOT vanish."""
        original = "UNIQUE_MARKER_CONTENT_" + "Z" * 10_000
        _add_tool_call_and_result(
            session, "tc_gone", "browser",
            {"action": "snapshot", "url": "https://example.com"},
            original,
        )

        archive_and_supersede_tool_results(session, iteration=1)

        raw_session = json.dumps(session.get_messages())
        assert "UNIQUE_MARKER_CONTENT_" in raw_session

    def test_small_results_not_stubbed(self, session):
        """Results under 500 chars are never touched."""
        small_content = "Exit code: 0\nHello world"
        _add_tool_call_and_result(
            session, "tc_small", "shell",
            {"command": "echo hello"},
            small_content,
        )

        archive_and_supersede_tool_results(session, iteration=1)

        messages = session.get_messages()
        tool_msg = [m for m in messages if m.get("role") == "tool"][0]
        assert tool_msg["content"] == small_content
        assert not tool_msg.get("_stubbed")

    def test_already_stubbed_not_re_processed(self, session):
        """Stubs with _stubbed=True are not re-processed on later invocations."""
        for i in range(2):
            _add_tool_call_and_result(
                session, f"tc_once_{i}", "browser",
                {"action": "snapshot", "url": "https://example.com"},
                "W" * 5_000,
            )

        archive_and_supersede_tool_results(session, iteration=1)

        tool_msgs = [m for m in session.get_messages() if m.get("role") == "tool"]
        first_stub = tool_msgs[0]["content"]

        archive_and_supersede_tool_results(session, iteration=2)

        tool_msgs = [m for m in session.get_messages() if m.get("role") == "tool"]
        assert tool_msgs[0]["content"] == first_stub
        assert not tool_msgs[1].get("_stubbed")

    def test_multi_tool_calls_on_distinct_paths_all_stay_live(self, session):
        """3 reads of 3 different paths in one turn: none supersedes another."""
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

        archive_and_supersede_tool_results(session, iteration=1)

        messages = session.get_messages()
        tool_msgs = [m for m in messages if m.get("role") == "tool"]
        assert len(tool_msgs) == 3

        for msg in tool_msgs:
            assert not msg.get("_stubbed")
            assert msg["content"] == "Q" * 2_000

    def test_repeated_target_token_savings(self, session):
        """Supersession still reclaims the duplicated copies of one target."""
        for i in range(7):
            _add_tool_call_and_result(
                session, f"tc_big_{i}", "browser",
                {"action": "snapshot", "url": "https://example.com/p"},
                "B" * 28_000,  # ~7K tokens
            )

        original_total = sum(
            len(m.get("content", ""))
            for m in session.get_messages()
            if m.get("role") == "tool"
        )
        assert original_total > 180_000  # ~49K tokens * 4 chars/token

        archive_and_supersede_tool_results(session, iteration=1)

        stub_total = sum(
            len(m.get("content", ""))
            for m in session.get_messages()
            if m.get("role") == "tool"
        )
        # 6 of 7 copies collapse to stubs; the newest is kept in full.
        assert stub_total < original_total * 0.20
        assert stub_total > 28_000

    def test_jsonl_persistence_after_supersession(self, session, workspace):
        """Supersession stubs must persist to JSONL, not just in memory."""
        for i in range(2):
            _add_tool_call_and_result(
                session, f"tc_persist_{i}", "browser",
                {"action": "snapshot", "url": "https://example.com"},
                "P" * 8_000,
            )

        archive_and_supersede_tool_results(session, iteration=1)

        reloaded = Session.load(session._filepath)
        tool_msgs = [m for m in reloaded.get_messages() if m.get("role") == "tool"]
        assert len(tool_msgs) == 2
        assert tool_msgs[0].get("_stubbed") is True
        assert tool_msgs[0]["content"].startswith("[SUPERSEDED")
        assert not tool_msgs[1].get("_stubbed")

    def test_large_results_archived_without_being_stubbed(self, session, workspace):
        """The disk corpus is preserved even though history is not rewritten."""
        _add_tool_call_and_result(
            session, "tc_archive", "browser",
            {"action": "snapshot", "url": "https://example.com/solo"},
            "C" * 9_000,
        )

        archive_and_supersede_tool_results(session, iteration=3)

        path = os.path.join(
            workspace, "orbital", "tool-results", "cs10-test",
            "turn_3_call_tc_archive.json",
        )
        assert os.path.exists(path)
        with open(path, "r", encoding="utf-8") as f:
            assert json.load(f)["content"] == "C" * 9_000
