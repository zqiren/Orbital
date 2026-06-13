# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Tests for read_sub_agent_summary.

`read_sub_agent_summary` turns a sub-agent's own JSONL transcript into a
**list of per-turn** summary dicts (one element per `turn_complete`-delimited
turn), each shaped `{tools_used, tool_rows, total_duration_seconds, response,
chunk_count, duration_seconds}`.

Contract (TASK-subagent-last-message-display):
- A transcript is split on `chunk_type="turn_complete"` boundary rows.
- Each completed turn = the chunks BEFORE a boundary (the boundary closes it).
- Trailing chunks after the last boundary = an in-flight turn → NOT returned.
- A transcript with ZERO boundaries (legacy flat file) = ONE turn (whole file).
- Boundary rows are excluded from `chunk_count` and duration tallies.
- Missing / empty / unparseable transcript → None.
"""

from __future__ import annotations

import json

from agent_os.daemon_v2.sub_agent_transcript import read_sub_agent_summary


def _write_transcript(path, entries):
    with open(path, "w", encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps(e) + "\n")


def _boundary(ts):
    """A turn_complete delimiter row (empty content — must not render)."""
    return {"source": "claude-code", "content": "", "chunk_type": "turn_complete",
            "timestamp": ts}


# ---------------------------------------------------------------------------
# Single-turn (legacy flat, zero boundaries) — list of exactly one element
# ---------------------------------------------------------------------------

def test_summary_reader_parses_tools_response_and_duration(tmp_path):
    p = tmp_path / "claude-code" / "abc.jsonl"
    p.parent.mkdir(parents=True)
    _write_transcript(p, [
        {"source": "claude-code", "content": "[Using tool: Write]", "chunk_type": "tool_activity",
         "timestamp": "2026-05-28T12:00:00+00:00"},
        {"source": "claude-code", "content": "[Using tool: Bash]", "chunk_type": "tool_activity",
         "timestamp": "2026-05-28T12:00:02+00:00"},
        {"source": "claude-code", "content": "[Using tool: Read]", "chunk_type": "tool_activity",
         "timestamp": "2026-05-28T12:00:03+00:00"},
        {"source": "claude-code", "content": "The file already exists with the correct primes.",
         "chunk_type": "response", "timestamp": "2026-05-28T12:00:04+00:00"},
    ])

    summary = read_sub_agent_summary(str(p))
    assert isinstance(summary, list)
    assert len(summary) == 1
    turn = summary[0]
    assert turn["tools_used"] == ["Write", "Bash", "Read"]
    assert turn["response"] == "The file already exists with the correct primes."
    assert turn["total_duration_seconds"] > 0  # 4 seconds between first and last
    assert turn["chunk_count"] == 4


def test_summary_reader_per_tool_rows_with_timing(tmp_path):
    """Per-tool rows in chronological order, each with a positive duration."""
    p = tmp_path / "run.jsonl"
    _write_transcript(p, [
        {"content": "[Using tool: Write]", "chunk_type": "tool_activity", "timestamp": "2026-05-28T12:00:00+00:00"},
        {"content": "[Using tool: Bash]",  "chunk_type": "tool_activity", "timestamp": "2026-05-28T12:00:02+00:00"},
        {"content": "[Using tool: Read]",  "chunk_type": "tool_activity", "timestamp": "2026-05-28T12:00:03+00:00"},
        {"content": "done", "chunk_type": "response", "timestamp": "2026-05-28T12:00:07+00:00"},
    ])
    summary = read_sub_agent_summary(str(p))
    assert isinstance(summary, list) and len(summary) == 1
    rows = summary[0]["tool_rows"]
    assert [r["name"] for r in rows] == ["Write", "Bash", "Read"]  # chronological
    assert rows[0]["duration_seconds"] == 2.0
    assert rows[1]["duration_seconds"] == 1.0
    assert rows[2]["duration_seconds"] == 4.0
    assert all(r["duration_seconds"] > 0 for r in rows)
    assert summary[0]["total_duration_seconds"] == 7.0  # 12:00:00 → 12:00:07


def test_summary_reader_dedupes_repeated_tools_in_first_seen_order(tmp_path):
    p = tmp_path / "t.jsonl"
    _write_transcript(p, [
        {"content": "[Using tool: Read]", "chunk_type": "tool_activity", "timestamp": "2026-05-28T12:00:00+00:00"},
        {"content": "[Using tool: Read]", "chunk_type": "tool_activity", "timestamp": "2026-05-28T12:00:01+00:00"},
        {"content": "[Using tool: Bash]", "chunk_type": "tool_activity", "timestamp": "2026-05-28T12:00:02+00:00"},
        {"content": "done", "chunk_type": "response", "timestamp": "2026-05-28T12:00:02+00:00"},
    ])
    summary = read_sub_agent_summary(str(p))
    assert summary[0]["tools_used"] == ["Read", "Bash"]


def test_summary_reader_last_response_wins_within_a_turn(tmp_path):
    """Within ONE turn (no boundary), the last response chunk wins."""
    p = tmp_path / "t.jsonl"
    _write_transcript(p, [
        {"content": "first", "chunk_type": "response", "timestamp": "2026-05-28T12:00:00+00:00"},
        {"content": "[Using tool: Read]", "chunk_type": "tool_activity", "timestamp": "2026-05-28T12:00:01+00:00"},
        {"content": "final answer", "chunk_type": "response", "timestamp": "2026-05-28T12:00:02+00:00"},
    ])
    summary = read_sub_agent_summary(str(p))
    assert len(summary) == 1
    assert summary[0]["response"] == "final answer"


# ---------------------------------------------------------------------------
# Missing / empty transcript → None (unchanged)
# ---------------------------------------------------------------------------

def test_summary_reader_missing_file_returns_none():
    assert read_sub_agent_summary("/nonexistent/path/xyz.jsonl") is None
    assert read_sub_agent_summary("") is None


def test_summary_reader_empty_file_returns_none(tmp_path):
    p = tmp_path / "empty.jsonl"
    p.write_text("", encoding="utf-8")
    assert read_sub_agent_summary(str(p)) is None


def test_summary_reader_only_blank_lines_returns_none(tmp_path):
    p = tmp_path / "blank.jsonl"
    p.write_text("\n\n  \n", encoding="utf-8")
    assert read_sub_agent_summary(str(p)) is None


def test_summary_reader_skips_malformed_lines(tmp_path):
    p = tmp_path / "mixed.jsonl"
    with open(p, "w", encoding="utf-8") as f:
        f.write("not json at all\n")
        f.write(json.dumps({"content": "[Using tool: Glob]", "chunk_type": "tool_activity",
                             "timestamp": "2026-05-28T12:00:00+00:00"}) + "\n")
        f.write("{ broken\n")
        f.write(json.dumps({"content": "ok", "chunk_type": "response",
                             "timestamp": "2026-05-28T12:00:01+00:00"}) + "\n")
    summary = read_sub_agent_summary(str(p))
    assert isinstance(summary, list) and len(summary) == 1
    assert summary[0]["tools_used"] == ["Glob"]
    assert summary[0]["response"] == "ok"
    assert summary[0]["chunk_count"] == 2  # only the 2 valid JSON lines


# ---------------------------------------------------------------------------
# Multi-turn — the heart of the task (R1–R4 from TASK Step 1)
# ---------------------------------------------------------------------------

def test_R1_two_turns_each_keeps_its_own_final_response_no_aliasing(tmp_path):
    """Two turn_complete-delimited turns → 2 elements, each its own response."""
    p = tmp_path / "two-turns.jsonl"
    _write_transcript(p, [
        {"content": "[Using tool: Read]", "chunk_type": "tool_activity", "timestamp": "2026-06-13T10:00:00+00:00"},
        {"content": "TURN-1", "chunk_type": "response", "timestamp": "2026-06-13T10:00:01+00:00"},
        _boundary("2026-06-13T10:00:02+00:00"),
        {"content": "TURN-2", "chunk_type": "response", "timestamp": "2026-06-13T10:00:05+00:00"},
        _boundary("2026-06-13T10:00:06+00:00"),
    ])
    summary = read_sub_agent_summary(str(p))
    assert isinstance(summary, list)
    assert len(summary) == 2
    assert summary[0]["response"] == "TURN-1"
    assert summary[1]["response"] == "TURN-2"
    # Boundary rows excluded from chunk_count.
    assert summary[0]["chunk_count"] == 2   # Read + TURN-1 (not the boundary)
    assert summary[1]["chunk_count"] == 1   # TURN-2 (not the boundary)


def test_R2_errored_turn_has_empty_response_not_aliased(tmp_path):
    """Turn 1 succeeds; turn 2 errors with no final response → turn 2 response
    is empty, NEVER turn 1's text aliased in (DIAGNOSIS Q3/C1)."""
    p = tmp_path / "errored.jsonl"
    _write_transcript(p, [
        {"content": "TURN-1 fixed auth.py", "chunk_type": "response", "timestamp": "2026-06-13T10:00:00+00:00"},
        _boundary("2026-06-13T10:00:01+00:00"),
        {"content": "[Using tool: Bash]", "chunk_type": "tool_activity", "timestamp": "2026-06-13T10:00:02+00:00"},
        {"content": "boom", "chunk_type": "error", "timestamp": "2026-06-13T10:00:03+00:00"},
        _boundary("2026-06-13T10:00:04+00:00"),
    ])
    summary = read_sub_agent_summary(str(p))
    assert len(summary) == 2
    assert summary[0]["response"] == "TURN-1 fixed auth.py"
    assert summary[1]["response"] == ""                    # errored turn: no response chunk
    assert summary[1]["response"] != "TURN-1 fixed auth.py"  # no cross-turn aliasing
    assert summary[1]["tools_used"] == ["Bash"]            # its own tools preserved


def test_R3_tool_only_turn_has_empty_response_no_neighbor_aliasing(tmp_path):
    """A turn with tool rows but no response chunk → empty response, tools kept,
    neighbor's text NOT aliased in (DIAGNOSIS Q3/C4)."""
    p = tmp_path / "toolonly.jsonl"
    _write_transcript(p, [
        {"content": "[Using tool: Write]", "chunk_type": "tool_activity", "timestamp": "2026-06-13T10:00:00+00:00"},
        _boundary("2026-06-13T10:00:01+00:00"),
        {"content": "real answer", "chunk_type": "response", "timestamp": "2026-06-13T10:00:02+00:00"},
        _boundary("2026-06-13T10:00:03+00:00"),
    ])
    summary = read_sub_agent_summary(str(p))
    assert len(summary) == 2
    assert summary[0]["response"] == ""               # tool-only turn
    assert summary[0]["tools_used"] == ["Write"]
    assert summary[1]["response"] == "real answer"


def test_R4_legacy_flat_transcript_zero_boundaries_is_one_turn(tmp_path):
    """Retroactivity: a pre-boundary flat transcript → a 1-element list."""
    p = tmp_path / "legacy.jsonl"
    _write_transcript(p, [
        {"content": "[Using tool: Read]", "chunk_type": "tool_activity", "timestamp": "2026-06-13T10:00:00+00:00"},
        {"content": "legacy answer", "chunk_type": "response", "timestamp": "2026-06-13T10:00:01+00:00"},
    ])
    summary = read_sub_agent_summary(str(p))
    assert isinstance(summary, list)
    assert len(summary) == 1
    assert summary[0]["response"] == "legacy answer"
    assert summary[0]["tools_used"] == ["Read"]


def test_trailing_inflight_turn_after_last_boundary_is_dropped(tmp_path):
    """A new transcript with one completed turn (boundary) + trailing in-flight
    chunks (no closing boundary) → only the completed turn is returned."""
    p = tmp_path / "inflight.jsonl"
    _write_transcript(p, [
        {"content": "DONE-1", "chunk_type": "response", "timestamp": "2026-06-13T10:00:00+00:00"},
        _boundary("2026-06-13T10:00:01+00:00"),
        {"content": "[Using tool: Bash]", "chunk_type": "tool_activity", "timestamp": "2026-06-13T10:00:02+00:00"},
        {"content": "still working...", "chunk_type": "response", "timestamp": "2026-06-13T10:00:03+00:00"},
    ])
    summary = read_sub_agent_summary(str(p))
    assert len(summary) == 1                       # in-flight 2nd turn not yet a slice
    assert summary[0]["response"] == "DONE-1"
