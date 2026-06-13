# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Tests for `_interleave_sub_agent_summaries` (the /chat display transform).

Contract (TASK-subagent-last-message-display):
- Anchor on the **message_routed** marker — the per-dispatch system line
  `[Sub-agent] Message sent to {handle}: "…". Transcript: {path}` (or the
  `User sent @{handle}: …` variant). It fires once per dispatch, in order, and
  carries the transcript path. The `started` marker is NOT an anchor (it is
  suppressed on the real dispatch path — DIAGNOSIS Q3/C3).
- The i-th dispatch for a transcript pairs with the i-th per-turn slice from
  `read_sub_agent_summary`.
- A slice WITH a response → a full `source="sub_agent"` bubble.
- A slice with NO response (errored / interrupted / tool-only) → NO synthetic
  bubble; the existing terminal-marker one-liner stands. NEVER alias another
  turn's text in (DIAGNOSIS Q3 honest-degradation).
- More dispatches than completed slices (last dispatch in-flight) → bubbles for
  the completed slices only (TASK decision 4).
- Display-only: never mutates/persists; missing transcript is a no-op.
"""

from __future__ import annotations

import json

from agent_os.api.routes.agents_v2 import _interleave_sub_agent_summaries


def _write_transcript(path, entries):
    with open(path, "w", encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps(e) + "\n")


def _boundary(ts):
    return {"source": "claude-code", "content": "", "chunk_type": "turn_complete", "timestamp": ts}


def _routed(handle, preview, path, ts, *, user=False, session_id=None):
    if user:
        content = f'[Sub-agent] User sent @{handle}: "{preview}". Transcript: {path}'
    else:
        content = f'[Sub-agent] Message sent to {handle}: "{preview}". Transcript: {path}'
    row = {"role": "system", "content": content, "source": "daemon", "timestamp": ts}
    if session_id is not None:
        row["session_id"] = session_id
    return row


def _subs(out):
    return [m for m in out if m.get("source") == "sub_agent"]


# ---------------------------------------------------------------------------
# Single dispatch — bubble injected after the message_routed marker
# ---------------------------------------------------------------------------

def test_injects_bubble_after_message_routed_marker(tmp_path):
    transcript = tmp_path / "claude-code" / "run1.jsonl"
    transcript.parent.mkdir(parents=True)
    _write_transcript(transcript, [
        {"content": "[Using tool: Bash]", "chunk_type": "tool_activity", "timestamp": "2026-06-13T12:00:00+00:00"},
        {"content": "listed files", "chunk_type": "response", "timestamp": "2026-06-13T12:00:01+00:00"},
        _boundary("2026-06-13T12:00:02+00:00"),
    ])
    messages = [
        {"role": "user", "content": "list files", "timestamp": "2026-06-13T11:59:00+00:00"},
        _routed("claude-code", "list files", transcript, "2026-06-13T11:59:30+00:00", session_id="sess_x"),
    ]
    out = _interleave_sub_agent_summaries(messages)
    subs = _subs(out)
    assert len(subs) == 1
    inj = subs[0]
    assert inj["sub_agent_handle"] == "claude-code"
    assert inj["content"] == "listed files"
    assert [r["name"] for r in inj["sub_agent_tool_rows"]] == ["Bash"]
    assert inj["session_id"] == "sess_x"          # inherits the marker's session
    # Injected immediately AFTER the marker (index 2, right after the routed marker at 1).
    assert out.index(inj) == 2


def test_anchor_also_matches_user_mention_variant(tmp_path):
    transcript = tmp_path / "run.jsonl"
    _write_transcript(transcript, [
        {"content": "did the thing", "chunk_type": "response", "timestamp": "2026-06-13T12:00:01+00:00"},
        _boundary("2026-06-13T12:00:02+00:00"),
    ])
    messages = [_routed("claude-code", "do the thing", transcript, "2026-06-13T11:59:30+00:00", user=True)]
    out = _interleave_sub_agent_summaries(messages)
    subs = _subs(out)
    assert len(subs) == 1
    assert subs[0]["content"] == "did the thing"


def test_anchor_handles_newline_in_preview_tail_anchored_path(tmp_path):
    """The user-message preview can contain newlines; the path regex must be
    tail-anchored (re.DOTALL) and still extract the transcript path."""
    transcript = tmp_path / "run.jsonl"
    _write_transcript(transcript, [
        {"content": "ok", "chunk_type": "response", "timestamp": "2026-06-13T12:00:01+00:00"},
        _boundary("2026-06-13T12:00:02+00:00"),
    ])
    messages = [_routed("claude-code", "line one\nline two\nline three", transcript, "2026-06-13T11:59:30+00:00")]
    out = _interleave_sub_agent_summaries(messages)
    assert len(_subs(out)) == 1
    assert _subs(out)[0]["content"] == "ok"


# ---------------------------------------------------------------------------
# Multi-turn — one bubble per dispatch, each its own content (no aliasing)
# ---------------------------------------------------------------------------

def test_two_successful_dispatches_render_two_distinct_bubbles(tmp_path):
    """The adversarial-recheck happy path: @handle X … later @handle Y →
    two bubbles, each with its own turn's text."""
    transcript = tmp_path / "claude-code" / "run.jsonl"
    transcript.parent.mkdir(parents=True)
    _write_transcript(transcript, [
        {"content": "created hello.txt", "chunk_type": "response", "timestamp": "2026-06-13T10:00:01+00:00"},
        _boundary("2026-06-13T10:00:02+00:00"),
        {"content": "created bye.txt", "chunk_type": "response", "timestamp": "2026-06-13T10:01:01+00:00"},
        _boundary("2026-06-13T10:01:02+00:00"),
    ])
    messages = [
        _routed("claude-code", "create hello.txt", transcript, "2026-06-13T10:00:00+00:00"),
        _routed("claude-code", "now create bye.txt", transcript, "2026-06-13T10:01:00+00:00"),
    ]
    out = _interleave_sub_agent_summaries(messages)
    subs = _subs(out)
    assert len(subs) == 2
    assert subs[0]["content"] == "created hello.txt"
    assert subs[1]["content"] == "created bye.txt"
    assert subs[0]["content"] != subs[1]["content"]   # NO aliasing


def test_R5_errored_second_turn_degrades_no_aliased_bubble(tmp_path):
    """Two dispatches; turn 2 errored (no final response). Turn 1 → full bubble.
    Turn 2 → NO synthetic bubble (honest one-liner from the existing terminal
    marker stands); turn-1 text is NOT aliased into a second bubble."""
    transcript = tmp_path / "claude-code" / "run.jsonl"
    transcript.parent.mkdir(parents=True)
    _write_transcript(transcript, [
        {"content": "TURN-1 done", "chunk_type": "response", "timestamp": "2026-06-13T10:00:01+00:00"},
        _boundary("2026-06-13T10:00:02+00:00"),
        {"content": "[Using tool: Bash]", "chunk_type": "tool_activity", "timestamp": "2026-06-13T10:01:01+00:00"},
        {"content": "boom", "chunk_type": "error", "timestamp": "2026-06-13T10:01:02+00:00"},
        _boundary("2026-06-13T10:01:03+00:00"),
    ])
    messages = [
        _routed("claude-code", "do X", transcript, "2026-06-13T10:00:00+00:00"),
        {"role": "system",
         "content": "[Sub-agent] claude-code completed. Summary: TURN-1 done. Transcript: " + str(transcript),
         "source": "daemon", "timestamp": "2026-06-13T10:00:03+00:00"},
        _routed("claude-code", "now fix Y", transcript, "2026-06-13T10:01:00+00:00"),
        {"role": "system",
         "content": "[Sub-agent] claude-code stopped with error: boom. Transcript: " + str(transcript),
         "source": "daemon", "timestamp": "2026-06-13T10:01:04+00:00"},
    ]
    out = _interleave_sub_agent_summaries(messages)
    subs = _subs(out)
    assert len(subs) == 1                              # only turn 1 gets a bubble
    assert subs[0]["content"] == "TURN-1 done"
    # Anti-aliasing in the NON-terminal position is pinned separately by
    # test_skipped_middle_turn_does_not_shift_later_dispatch_slices.
    # The errored terminal one-liner is preserved untouched in the stream.
    assert any("stopped with error" in (m.get("content") or "") for m in out)


def test_inflight_trailing_dispatch_gets_no_bubble(tmp_path):
    """Two message_routed markers but only one completed turn (2nd in-flight) →
    one bubble for the first; the trailing dispatch waits for the next reload."""
    transcript = tmp_path / "claude-code" / "run.jsonl"
    transcript.parent.mkdir(parents=True)
    _write_transcript(transcript, [
        {"content": "first done", "chunk_type": "response", "timestamp": "2026-06-13T10:00:01+00:00"},
        _boundary("2026-06-13T10:00:02+00:00"),
        # 2nd dispatch dispatched but still running — content, no closing boundary
        {"content": "working on it", "chunk_type": "response", "timestamp": "2026-06-13T10:01:01+00:00"},
    ])
    messages = [
        _routed("claude-code", "do X", transcript, "2026-06-13T10:00:00+00:00"),
        _routed("claude-code", "do Y", transcript, "2026-06-13T10:01:00+00:00"),
    ]
    out = _interleave_sub_agent_summaries(messages)
    subs = _subs(out)
    assert len(subs) == 1
    assert subs[0]["content"] == "first done"


# ---------------------------------------------------------------------------
# Isolation / robustness — no-op cases (display-only, best-effort)
# ---------------------------------------------------------------------------

def test_noop_when_transcript_missing():
    messages = [
        _routed("claude-code", "hi", "/nope/missing.jsonl", "2026-06-13T11:59:30+00:00"),
    ]
    out = _interleave_sub_agent_summaries(messages)
    assert out == messages          # unchanged — best-effort, no crash, no injection


def test_ignores_non_sub_agent_system_messages():
    messages = [
        {"role": "system", "content": "Repetitive action detected.", "source": "daemon",
         "timestamp": "2026-06-13T11:59:30+00:00"},
        {"role": "assistant", "content": "hi", "source": "management", "timestamp": "2026-06-13T11:59:31+00:00"},
    ]
    out = _interleave_sub_agent_summaries(messages)
    assert out == messages


def test_skipped_middle_turn_does_not_shift_later_dispatch_slices(tmp_path):
    """A skipped (errored/empty) turn in the MIDDLE must not shift later
    dispatches: dispatch 3 still pairs with turn 3, not turn 2. Guards the
    unconditional index-advance that keeps pairing aligned past a skip (review
    finding: middle-skip alignment was untested — the natural 'advance only on
    emit' refactor passes every other test but fails this)."""
    t = tmp_path / "claude-code" / "run.jsonl"
    t.parent.mkdir(parents=True)
    _write_transcript(t, [
        {"content": "TURN-1 ok", "chunk_type": "response", "timestamp": "2026-06-13T10:00:01+00:00"},
        _boundary("2026-06-13T10:00:02+00:00"),
        # turn 2 errored: tool + error, NO response chunk
        {"content": "[Using tool: Bash]", "chunk_type": "tool_activity", "timestamp": "2026-06-13T10:01:01+00:00"},
        {"content": "boom", "chunk_type": "error", "timestamp": "2026-06-13T10:01:02+00:00"},
        _boundary("2026-06-13T10:01:03+00:00"),
        {"content": "TURN-3 ok", "chunk_type": "response", "timestamp": "2026-06-13T10:02:01+00:00"},
        _boundary("2026-06-13T10:02:02+00:00"),
    ])
    messages = [
        _routed("claude-code", "do 1", t, "2026-06-13T10:00:00+00:00"),
        _routed("claude-code", "do 2", t, "2026-06-13T10:01:00+00:00"),
        _routed("claude-code", "do 3", t, "2026-06-13T10:02:00+00:00"),
    ]
    subs = _subs(_interleave_sub_agent_summaries(messages))
    assert [s["content"] for s in subs] == ["TURN-1 ok", "TURN-3 ok"]  # turn 3 ↔ dispatch 3


def test_two_distinct_transcript_paths_have_isolated_indexing(tmp_path):
    """One management session can dispatch to multiple handles (@frontend,
    @backend). Each transcript path keeps its OWN dispatch index — no cross-path
    drift (review finding: per-path index isolation was untested)."""
    txA = tmp_path / "agentA" / "run.jsonl"; txA.parent.mkdir(parents=True)
    txB = tmp_path / "agentB" / "run.jsonl"; txB.parent.mkdir(parents=True)
    _write_transcript(txA, [
        {"content": "A-done", "chunk_type": "response", "timestamp": "2026-06-13T10:00:01+00:00"},
        _boundary("2026-06-13T10:00:02+00:00"),
    ])
    _write_transcript(txB, [
        {"content": "B-done", "chunk_type": "response", "timestamp": "2026-06-13T10:01:01+00:00"},
        _boundary("2026-06-13T10:01:02+00:00"),
    ])
    messages = [
        _routed("agentA", "do A", txA, "2026-06-13T10:00:00+00:00"),
        _routed("agentB", "do B", txB, "2026-06-13T10:01:00+00:00"),
    ]
    subs = _subs(_interleave_sub_agent_summaries(messages))
    assert len(subs) == 2
    assert {s["sub_agent_handle"]: s["content"] for s in subs} == {"agentA": "A-done", "agentB": "B-done"}


def test_started_marker_is_not_an_anchor(tmp_path):
    """Regression guard: the (now-dead) started marker must NOT trigger a bubble
    — only message_routed does. Prevents double-injection if both ever appear."""
    transcript = tmp_path / "run.jsonl"
    _write_transcript(transcript, [
        {"content": "x", "chunk_type": "response", "timestamp": "2026-06-13T12:00:01+00:00"},
        _boundary("2026-06-13T12:00:02+00:00"),
    ])
    messages = [
        {"role": "system",
         "content": f"[Sub-agent] claude-code started (initiated by: management_agent). Transcript: {transcript}",
         "source": "daemon", "timestamp": "2026-06-13T11:59:30+00:00"},
    ]
    out = _interleave_sub_agent_summaries(messages)
    assert _subs(out) == []         # started marker alone injects nothing
