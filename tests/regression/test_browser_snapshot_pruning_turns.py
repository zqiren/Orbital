# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Regression test for P-06: browser snapshot pruning uses LLM turns, not message count.

The old _prune_old_tool_results used raw message index distance
(len(messages) - i) as "turns_ago".  In browser-heavy multi-turn sessions
the message count grows fast (2+ messages per LLM call), so browser
snapshots were pruned after just a few browser calls even within the same
logical turn.  The fix counts actual assistant messages (LLM turns) so that
the 2-turn TTL for snapshots and 5-turn TTL for general results correspond
to real LLM turns.
"""

import pytest
from agent_os.agent.context import ContextManager


def _make_messages(assistant_count: int, tool_per_assistant: int = 1,
                   browser_snapshot: bool = True):
    """Build a realistic message sequence with assistant + tool result pairs.

    Each assistant message is followed by `tool_per_assistant` tool results.
    Browser tool results carry _meta with snapshot_stats to trigger the
    browser-specific pruning path.
    """
    msgs = []
    for turn in range(assistant_count):
        tc_ids = [f"tc_{turn}_{j}" for j in range(tool_per_assistant)]
        msgs.append({
            "role": "assistant",
            "content": f"Turn {turn} response",
            "tool_calls": [{"id": tid, "function": {"name": "browser"}} for tid in tc_ids],
        })
        for j, tid in enumerate(tc_ids):
            meta = {}
            if browser_snapshot:
                meta = {
                    "snapshot_stats": {"refs": 42},
                    "url": f"https://example.com/page-{turn}-{j}",
                }
            msgs.append({
                "role": "tool",
                "tool_call_id": tid,
                "content": "X" * 2000,  # large enough to trigger pruning
                "_meta": meta,
            })
    return msgs


class TestBrowserSnapshotPruningUsesLLMTurns:
    """Verify that browser snapshot pruning counts LLM turns, not messages."""

    def test_snapshots_within_2_turns_are_preserved(self):
        """Browser snapshots from the last 2 LLM turns must keep full content."""
        # 3 LLM turns, each with 1 browser snapshot → 6 messages total.
        # Old code: message at index 0 has turns_ago=6 → pruned (wrong).
        # New code: turn 1's snapshot is 2 LLM turns ago → preserved.
        msgs = _make_messages(assistant_count=3, tool_per_assistant=1,
                              browser_snapshot=True)

        result = ContextManager._prune_old_tool_results(None, msgs)
        # The tool result from turn index 2 (the 3rd assistant, 0-based) is
        # 0 LLM turns ago.  Turn index 1 is 1 turn ago.  Both should keep
        # full content.
        tool_results = [m for m in result if m.get("role") == "tool"]
        assert len(tool_results) == 3

        # Last 2 turns' snapshots should NOT be pruned
        assert "X" * 2000 in tool_results[2]["content"]  # turn 2 (0 ago)
        assert "X" * 2000 in tool_results[1]["content"]  # turn 1 (1 ago)

    def test_snapshots_older_than_2_turns_are_pruned(self):
        """Browser snapshots from 3+ LLM turns ago should be pruned."""
        msgs = _make_messages(assistant_count=5, tool_per_assistant=1,
                              browser_snapshot=True)
        result = ContextManager._prune_old_tool_results(None, msgs)
        tool_results = [m for m in result if m.get("role") == "tool"]

        # Turn 0 is 4 turns ago → pruned (>2)
        assert "[Snapshot of" in tool_results[0]["content"]
        # Turn 1 is 3 turns ago → pruned (>2)
        assert "[Snapshot of" in tool_results[1]["content"]
        # Turn 2 is 2 turns ago → borderline, NOT pruned (2 is not >2)
        assert "X" * 2000 in tool_results[2]["content"]
        # Turn 3 is 1 turn ago → preserved
        assert "X" * 2000 in tool_results[3]["content"]
        # Turn 4 is 0 turns ago → preserved
        assert "X" * 2000 in tool_results[4]["content"]

    def test_many_tool_calls_per_turn_not_over_pruned(self):
        """Multiple tool calls in one LLM turn should all share the same
        turn count and not be over-pruned."""
        # 2 turns, each with 5 browser tool calls → 12 messages.
        # Old code: 12 messages, tool at index 1 has turns_ago=11 → pruned.
        # New code: all tool results from turn 1 are 1 LLM turn ago → preserved.
        msgs = _make_messages(assistant_count=2, tool_per_assistant=5,
                              browser_snapshot=True)
        assert len(msgs) == 12  # 2 * (1 assistant + 5 tools)

        result = ContextManager._prune_old_tool_results(None, msgs)
        tool_results = [m for m in result if m.get("role") == "tool"]
        assert len(tool_results) == 10

        # All tool results from the last turn (turn 1, 0 LLM turns ago)
        # should keep full content.
        for tr in tool_results[5:]:
            assert "X" * 2000 in tr["content"], "Recent turn's results should not be pruned"

        # All tool results from turn 0 (1 LLM turn ago) should also keep
        # full content (within 2-turn TTL).
        for tr in tool_results[:5]:
            assert "X" * 2000 in tr["content"], "1-turn-ago results should not be pruned"

    def test_general_pruning_uses_turn_count(self):
        """Non-browser tool results older than 5 LLM turns should be pruned."""
        msgs = _make_messages(assistant_count=8, tool_per_assistant=1,
                              browser_snapshot=False)
        result = ContextManager._prune_old_tool_results(None, msgs)
        tool_results = [m for m in result if m.get("role") == "tool"]

        # Turn 0 is 7 turns ago → pruned (>5)
        assert "[Truncated]" in tool_results[0]["content"]
        # Turn 1 is 6 turns ago → pruned (>5)
        assert "[Truncated]" in tool_results[1]["content"]
        # Turn 2 is 5 turns ago → borderline, NOT pruned (5 is not >5)
        assert "X" * 2000 in tool_results[2]["content"]
        # Turn 3 is 4 turns ago → preserved
        assert "X" * 2000 in tool_results[3]["content"]

    def test_old_message_index_would_over_prune(self):
        """Demonstrate the specific scenario that caused P-06.

        Simulate a multi-turn browser session where turn 2 has 8 browser
        calls. Under the old code, the early browser calls in turn 2 would
        have been pruned because their message index was far from the end.
        """
        # Turn 1: 3 browser calls (already completed)
        msgs = _make_messages(assistant_count=3, tool_per_assistant=1,
                              browser_snapshot=True)
        # User message for turn 2
        msgs.append({"role": "user", "content": "Now create positioning.md"})
        # Turn 2: 8 more browser calls
        for turn in range(3, 11):
            tid = f"tc_{turn}_0"
            msgs.append({
                "role": "assistant",
                "content": f"Browsing page {turn}",
                "tool_calls": [{"id": tid, "function": {"name": "browser"}}],
            })
            msgs.append({
                "role": "tool",
                "tool_call_id": tid,
                "content": f"Page {turn} data: " + "Y" * 2000,
                "_meta": {
                    "snapshot_stats": {"refs": 30},
                    "url": f"https://competitor.com/{turn}",
                },
            })

        # Total: 3*(1+1) + 1 + 8*(1+1) = 23 messages.
        assert len(msgs) == 23

        result = ContextManager._prune_old_tool_results(None, msgs)
        tool_results = [m for m in result if m.get("role") == "tool"]

        # Under the old code, tool results from early in turn 2 would be
        # pruned because turns_ago = 23 - index >> 2.
        # Under the fix, they should be preserved if within 2 LLM turns.
        # The last assistant message is turn 10 (0-indexed from the batch).
        # Tool results from turn 9 (1 ago) and turn 10 (0 ago) should be
        # preserved.
        preserved_count = sum(
            1 for tr in tool_results if "Y" * 2000 in tr["content"]
        )
        # At minimum, the last 2 turns' results should be preserved
        assert preserved_count >= 2, (
            f"Expected at least 2 preserved browser results, got {preserved_count}"
        )
