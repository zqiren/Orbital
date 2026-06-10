# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Tests for read_sub_agent_summary + the /chat interleave helper.

Covers Part 2 of TASK-capsule-collapse-and-subagent-transcript.md:
the server-side summary reader that turns a sub-agent's own JSONL transcript
into a compact {tools_used, duration_seconds, response, chunk_count} dict,
and the helper that inlines it into the management /chat response.
"""

from __future__ import annotations

import json

from agent_os.daemon_v2.sub_agent_transcript import read_sub_agent_summary
from agent_os.api.routes.agents_v2 import _interleave_sub_agent_summaries


def _write_transcript(path, entries):
    with open(path, "w", encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps(e) + "\n")


# ---------------------------------------------------------------------------
# Test 3 — summary reader parses transcript correctly
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
    assert summary is not None
    assert summary["tools_used"] == ["Write", "Bash", "Read"]
    assert summary["response"] == "The file already exists with the correct primes."
    assert summary["total_duration_seconds"] > 0  # 4 seconds between first and last
    assert summary["chunk_count"] == 4


def test_summary_reader_per_tool_rows_with_timing(tmp_path):
    """Test 5: per-tool rows in chronological order, each with a positive duration."""
    p = tmp_path / "run.jsonl"
    _write_transcript(p, [
        {"content": "[Using tool: Write]", "chunk_type": "tool_activity", "timestamp": "2026-05-28T12:00:00+00:00"},
        {"content": "[Using tool: Bash]",  "chunk_type": "tool_activity", "timestamp": "2026-05-28T12:00:02+00:00"},
        {"content": "[Using tool: Read]",  "chunk_type": "tool_activity", "timestamp": "2026-05-28T12:00:03+00:00"},
        {"content": "done", "chunk_type": "response", "timestamp": "2026-05-28T12:00:07+00:00"},
    ])
    summary = read_sub_agent_summary(str(p))
    assert summary is not None
    rows = summary["tool_rows"]
    assert [r["name"] for r in rows] == ["Write", "Bash", "Read"]  # chronological
    # Write→Bash gap = 2s, Bash→Read = 1s, Read→response = 4s — all positive.
    assert rows[0]["duration_seconds"] == 2.0
    assert rows[1]["duration_seconds"] == 1.0
    assert rows[2]["duration_seconds"] == 4.0
    assert all(r["duration_seconds"] > 0 for r in rows)
    assert summary["total_duration_seconds"] == 7.0  # 12:00:00 → 12:00:07


def test_summary_reader_dedupes_repeated_tools_in_first_seen_order(tmp_path):
    p = tmp_path / "t.jsonl"
    _write_transcript(p, [
        {"content": "[Using tool: Read]", "chunk_type": "tool_activity", "timestamp": "2026-05-28T12:00:00+00:00"},
        {"content": "[Using tool: Read]", "chunk_type": "tool_activity", "timestamp": "2026-05-28T12:00:01+00:00"},
        {"content": "[Using tool: Bash]", "chunk_type": "tool_activity", "timestamp": "2026-05-28T12:00:02+00:00"},
        {"content": "done", "chunk_type": "response", "timestamp": "2026-05-28T12:00:02+00:00"},
    ])
    summary = read_sub_agent_summary(str(p))
    assert summary["tools_used"] == ["Read", "Bash"]


def test_summary_reader_last_response_wins(tmp_path):
    p = tmp_path / "t.jsonl"
    _write_transcript(p, [
        {"content": "first", "chunk_type": "response", "timestamp": "2026-05-28T12:00:00+00:00"},
        {"content": "[Using tool: Read]", "chunk_type": "tool_activity", "timestamp": "2026-05-28T12:00:01+00:00"},
        {"content": "final answer", "chunk_type": "response", "timestamp": "2026-05-28T12:00:02+00:00"},
    ])
    summary = read_sub_agent_summary(str(p))
    assert summary["response"] == "final answer"


# ---------------------------------------------------------------------------
# Test 4 — missing transcript returns None
# ---------------------------------------------------------------------------

def test_summary_reader_missing_file_returns_none():
    assert read_sub_agent_summary("/nonexistent/path/xyz.jsonl") is None
    assert read_sub_agent_summary("") is None


# ---------------------------------------------------------------------------
# Test 5 — empty transcript returns None
# ---------------------------------------------------------------------------

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
    assert summary is not None
    assert summary["tools_used"] == ["Glob"]
    assert summary["response"] == "ok"
    assert summary["chunk_count"] == 2  # only the 2 valid JSON lines


# ---------------------------------------------------------------------------
# Interleave helper — inlines a source=sub_agent row after the started marker
# ---------------------------------------------------------------------------

def test_interleave_injects_sub_agent_row_after_started_marker(tmp_path):
    transcript = tmp_path / "claude-code" / "run1.jsonl"
    transcript.parent.mkdir(parents=True)
    _write_transcript(transcript, [
        {"content": "[Using tool: Bash]", "chunk_type": "tool_activity", "timestamp": "2026-05-28T12:00:00+00:00"},
        {"content": "listed files", "chunk_type": "response", "timestamp": "2026-05-28T12:00:01+00:00"},
    ])
    messages = [
        {"role": "user", "content": "list files", "timestamp": "2026-05-28T11:59:00+00:00"},
        {"role": "system",
         "content": f"[Sub-agent] claude-code started (initiated by: management_agent). Transcript: {transcript}",
         "source": "daemon", "timestamp": "2026-05-28T11:59:30+00:00", "session_id": "sess_x"},
        {"role": "system",
         "content": f"[Sub-agent] claude-code completed. Summary: listed files. Transcript: {transcript}",
         "source": "daemon", "timestamp": "2026-05-28T12:00:05+00:00"},
    ]

    out = _interleave_sub_agent_summaries(messages)
    # A synthetic sub_agent row was inserted right after the started marker.
    assert len(out) == 4
    injected = out[2]
    assert injected["source"] == "sub_agent"
    assert injected["sub_agent_handle"] == "claude-code"
    assert [r["name"] for r in injected["sub_agent_tool_rows"]] == ["Bash"]
    assert injected["content"] == "listed files"
    assert injected["session_id"] == "sess_x"  # inherits the marker's session
    # The completed marker is untouched and still present.
    assert out[3]["role"] == "system" and "completed" in out[3]["content"]


def test_interleave_noop_when_transcript_missing(tmp_path):
    messages = [
        {"role": "system",
         "content": "[Sub-agent] claude-code started. Transcript: /nope/missing.jsonl",
         "source": "daemon", "timestamp": "2026-05-28T11:59:30+00:00"},
    ]
    out = _interleave_sub_agent_summaries(messages)
    assert out == messages  # unchanged — best-effort, no crash, no injection


def test_interleave_ignores_non_sub_agent_system_messages():
    messages = [
        {"role": "system", "content": "Repetitive action detected.", "source": "daemon",
         "timestamp": "2026-05-28T11:59:30+00:00"},
        {"role": "assistant", "content": "hi", "source": "management",
         "timestamp": "2026-05-28T11:59:31+00:00"},
    ]
    out = _interleave_sub_agent_summaries(messages)
    assert out == messages
