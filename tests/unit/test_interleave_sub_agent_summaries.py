# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Tests for `_interleave_sub_agent_summaries` (the /chat display transform).

Contract (TASK-dispatch-id-pairing, replacing TASK-subagent-last-message-display's
positional pairing):

- Sub-agent transcripts are PERSISTENT per-(workspace, handle) JSONL files that
  span many chat sessions. The old contract paired "the i-th message_routed
  marker in the CURRENT session's messages" with "the i-th turn-slice in the
  WHOLE transcript file" — correct only for the transcript's very FIRST
  session; any later session mispaired against stale early turns.
- The new contract is an explicit identity join: each message_routed marker
  carries ``_meta.dispatch_id`` + ``_meta.transcript_path`` (stamped at
  dispatch time — see LifecycleObserver.on_message_routed / SubAgentManager.
  send()). Each transcript turn carries the ``dispatch_id`` of the boundary
  row that closed it (read_sub_agent_summary). The renderer looks up the
  slice whose ``dispatch_id`` matches the marker's — no counting, no session
  scope needed.
- A slice WITH a response → a full `source="sub_agent"` bubble.
- A slice with NO response (errored / interrupted / tool-only) → NO synthetic
  bubble; the existing terminal-marker one-liner stands. NEVER alias another
  turn's text in.
- No match (id not found / turn still in flight / no `_meta` at all — a
  legacy pre-migration marker) → NO bubble. Marker text is untouched either
  way; there is no positional or timestamp fallback.
- Join hygiene / idempotent join: TWO markers carrying the SAME dispatch_id
  (e.g. the known pre-existing @mention double-marker — send()'s internal
  "management_agent"-flavored marker plus the API route's own
  "user_mention"-flavored marker for the same physical turn) must render
  exactly ONE bubble, at the FIRST marker carrying that id. The renderer is
  not a second copy-machine for a turn just because two markers point at
  it — later markers with an already-rendered id fall back to their plain
  one-liner.
- Display-only: never mutates/persists; missing transcript is a no-op.

The prose-parsing regex (``_SUB_AGENT_DISPATCH_RE``) is kept in
``agent_os/api/routes/agents_v2.py`` only because the (separate) legacy-data
migration script imports its shape to backfill dispatch_ids into old
transcripts. This module's runtime path no longer uses it at all.
"""

from __future__ import annotations

import json

from agent_os.api.routes.agents_v2 import _interleave_sub_agent_summaries


def _write_transcript(path, entries):
    with open(path, "w", encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps(e) + "\n")


def _boundary(ts, dispatch_id=None):
    row = {"source": "claude-code", "content": "", "chunk_type": "turn_complete", "timestamp": ts}
    if dispatch_id is not None:
        row["dispatch_id"] = dispatch_id
    return row


def _routed(handle, preview, path, ts, *, dispatch_id=None, user=False, session_id=None):
    """Build a message_routed marker. When ``dispatch_id`` is given, the
    marker carries the structured ``_meta`` the renderer joins against —
    exactly what LifecycleObserver.on_message_routed now stamps. Omitting it
    reproduces a legacy (pre-migration) marker that has no ``_meta`` at all."""
    if user:
        content = f'[Sub-agent] User sent @{handle}: "{preview}". Transcript: {path}'
    else:
        content = f'[Sub-agent] Message sent to {handle}: "{preview}". Transcript: {path}'
    row = {"role": "system", "content": content, "source": "daemon", "timestamp": ts}
    if session_id is not None:
        row["session_id"] = session_id
    if dispatch_id is not None:
        row["_meta"] = {"dispatch_id": dispatch_id, "handle": handle, "transcript_path": str(path)}
    return row


def _subs(out):
    return [m for m in out if m.get("source") == "sub_agent"]


# ---------------------------------------------------------------------------
# The production bug this task fixes: a transcript spans many chat sessions;
# a LATER session's marker must pair with ITS OWN turn, not the file's i-th.
# ---------------------------------------------------------------------------

def test_later_session_marker_pairs_by_id_not_by_position(tmp_path):
    """Reproduces the production shape: turns 1-2 (dispatch_ids A1/A2) belong
    to an EARLIER chat session and are not present in the current message
    list at all — only turn 3 (dispatch_id B1) was dispatched from the
    session being rendered now. Under the old positional code this would
    pair the session's lone marker (index 0 within ITS OWN message list)
    with slices[0] == turn 1 (A1's text) — a weeks-stale turn. The id join
    must select turn 3 regardless of its position in the file."""
    transcript = tmp_path / "claude-code" / "2f1d5b86.jsonl"
    transcript.parent.mkdir(parents=True)
    _write_transcript(transcript, [
        {"content": "turn one text (old session)", "chunk_type": "response",
         "timestamp": "2026-06-29T10:00:00+00:00"},
        _boundary("2026-06-29T10:00:01+00:00", "sessOLD1:aaaaaaaa"),
        {"content": "turn two text (old session)", "chunk_type": "response",
         "timestamp": "2026-06-30T10:00:00+00:00"},
        _boundary("2026-06-30T10:00:01+00:00", "sessOLD1:bbbbbbbb"),
        {"content": "turn three text (THIS session)", "chunk_type": "response",
         "timestamp": "2026-07-13T10:00:00+00:00"},
        _boundary("2026-07-13T10:00:01+00:00", "sessNEW:cccccccc"),
    ])
    # The LATER session's message list contains only ITS OWN marker — the
    # earlier session's markers live in a different session JSONL entirely.
    messages = [
        _routed("claude-code", "do turn three", transcript, "2026-07-13T09:59:00+00:00",
                dispatch_id="sessNEW:cccccccc", session_id="sessNEW"),
    ]
    out = _interleave_sub_agent_summaries(messages)
    subs = _subs(out)
    assert len(subs) == 1
    assert subs[0]["content"] == "turn three text (THIS session)"
    assert subs[0]["content"] != "turn one text (old session)"   # the old bug


# ---------------------------------------------------------------------------
# Single dispatch — bubble injected after the message_routed marker
# ---------------------------------------------------------------------------

def test_injects_bubble_after_message_routed_marker(tmp_path):
    transcript = tmp_path / "claude-code" / "run1.jsonl"
    transcript.parent.mkdir(parents=True)
    _write_transcript(transcript, [
        {"content": "[Using tool: Bash]", "chunk_type": "tool_activity", "timestamp": "2026-06-13T12:00:00+00:00"},
        {"content": "listed files", "chunk_type": "response", "timestamp": "2026-06-13T12:00:01+00:00"},
        _boundary("2026-06-13T12:00:02+00:00", "sess_x:d1"),
    ])
    messages = [
        {"role": "user", "content": "list files", "timestamp": "2026-06-13T11:59:00+00:00"},
        _routed("claude-code", "list files", transcript, "2026-06-13T11:59:30+00:00",
                dispatch_id="sess_x:d1", session_id="sess_x"),
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


def test_anchor_matches_user_mention_variant_too(tmp_path):
    """The renderer no longer regex-parses marker text at all — it only
    looks at ``_meta`` — so the human-readable prose variant (management vs.
    direct @mention) is irrelevant to pairing. Confirms the id join doesn't
    accidentally depend on the content string's shape."""
    transcript = tmp_path / "run.jsonl"
    _write_transcript(transcript, [
        {"content": "did the thing", "chunk_type": "response", "timestamp": "2026-06-13T12:00:01+00:00"},
        _boundary("2026-06-13T12:00:02+00:00", "d1"),
    ])
    messages = [_routed("claude-code", "do the thing", transcript, "2026-06-13T11:59:30+00:00",
                        dispatch_id="d1", user=True)]
    out = _interleave_sub_agent_summaries(messages)
    subs = _subs(out)
    assert len(subs) == 1
    assert subs[0]["content"] == "did the thing"


# ---------------------------------------------------------------------------
# Multi-turn — one bubble per dispatch, each its own content (no aliasing)
# ---------------------------------------------------------------------------

def test_two_successful_dispatches_render_two_distinct_bubbles(tmp_path):
    """The adversarial-recheck happy path: @handle X … later @handle Y →
    two bubbles, each with its own turn's text, joined by distinct ids."""
    transcript = tmp_path / "claude-code" / "run.jsonl"
    transcript.parent.mkdir(parents=True)
    _write_transcript(transcript, [
        {"content": "created hello.txt", "chunk_type": "response", "timestamp": "2026-06-13T10:00:01+00:00"},
        _boundary("2026-06-13T10:00:02+00:00", "d1"),
        {"content": "created bye.txt", "chunk_type": "response", "timestamp": "2026-06-13T10:01:01+00:00"},
        _boundary("2026-06-13T10:01:02+00:00", "d2"),
    ])
    messages = [
        _routed("claude-code", "create hello.txt", transcript, "2026-06-13T10:00:00+00:00", dispatch_id="d1"),
        _routed("claude-code", "now create bye.txt", transcript, "2026-06-13T10:01:00+00:00", dispatch_id="d2"),
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
        _boundary("2026-06-13T10:00:02+00:00", "d1"),
        {"content": "[Using tool: Bash]", "chunk_type": "tool_activity", "timestamp": "2026-06-13T10:01:01+00:00"},
        {"content": "boom", "chunk_type": "error", "timestamp": "2026-06-13T10:01:02+00:00"},
        _boundary("2026-06-13T10:01:03+00:00", "d2"),
    ])
    messages = [
        _routed("claude-code", "do X", transcript, "2026-06-13T10:00:00+00:00", dispatch_id="d1"),
        {"role": "system",
         "content": "[Sub-agent] claude-code completed. Summary: TURN-1 done. Transcript: " + str(transcript),
         "source": "daemon", "timestamp": "2026-06-13T10:00:03+00:00"},
        _routed("claude-code", "now fix Y", transcript, "2026-06-13T10:01:00+00:00", dispatch_id="d2"),
        {"role": "system",
         "content": "[Sub-agent] claude-code stopped with error: boom. Transcript: " + str(transcript),
         "source": "daemon", "timestamp": "2026-06-13T10:01:04+00:00"},
    ]
    out = _interleave_sub_agent_summaries(messages)
    subs = _subs(out)
    assert len(subs) == 1                              # only turn 1 gets a bubble
    assert subs[0]["content"] == "TURN-1 done"
    # The errored terminal one-liner is preserved untouched in the stream.
    assert any("stopped with error" in (m.get("content") or "") for m in out)


def test_inflight_trailing_dispatch_gets_no_bubble(tmp_path):
    """Two message_routed markers but only one completed turn (2nd in-flight,
    its dispatch_id has no boundary yet) → one bubble for the first; the
    trailing dispatch waits for the next reload."""
    transcript = tmp_path / "claude-code" / "run.jsonl"
    transcript.parent.mkdir(parents=True)
    _write_transcript(transcript, [
        {"content": "first done", "chunk_type": "response", "timestamp": "2026-06-13T10:00:01+00:00"},
        _boundary("2026-06-13T10:00:02+00:00", "d1"),
        # 2nd dispatch (d2) dispatched but still running — content, no closing boundary
        {"content": "working on it", "chunk_type": "response", "timestamp": "2026-06-13T10:01:01+00:00"},
    ])
    messages = [
        _routed("claude-code", "do X", transcript, "2026-06-13T10:00:00+00:00", dispatch_id="d1"),
        _routed("claude-code", "do Y", transcript, "2026-06-13T10:01:00+00:00", dispatch_id="d2"),
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
        _routed("claude-code", "hi", "/nope/missing.jsonl", "2026-06-13T11:59:30+00:00", dispatch_id="d1"),
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


def test_legacy_marker_with_no_meta_gets_no_bubble(tmp_path):
    """A pre-migration marker (written before this task shipped) has no
    ``_meta`` at all. Contract: no bubble, one-line marker untouched — there
    is deliberately NO positional or timestamp fallback. (A separate,
    later migration script backfills ids into old transcripts; until then
    these markers just show their one-liner.)"""
    transcript = tmp_path / "claude-code" / "run.jsonl"
    transcript.parent.mkdir(parents=True)
    _write_transcript(transcript, [
        {"content": "some response", "chunk_type": "response", "timestamp": "2026-06-13T10:00:01+00:00"},
        _boundary("2026-06-13T10:00:02+00:00"),  # legacy boundary, no dispatch_id either
    ])
    messages = [
        _routed("claude-code", "do X", transcript, "2026-06-13T10:00:00+00:00"),  # no dispatch_id -> no _meta
    ]
    out = _interleave_sub_agent_summaries(messages)
    assert _subs(out) == []
    assert out == messages   # untouched


def test_dispatch_id_not_found_in_slices_gets_no_bubble(tmp_path):
    """A marker carries a well-formed _meta, but no turn in the transcript
    closed with that id (e.g. the id was for a turn that never completed
    and the process died) → no bubble, no crash."""
    transcript = tmp_path / "run.jsonl"
    _write_transcript(transcript, [
        {"content": "other turn", "chunk_type": "response", "timestamp": "2026-06-13T10:00:01+00:00"},
        _boundary("2026-06-13T10:00:02+00:00", "d-other"),
    ])
    messages = [
        _routed("claude-code", "do X", transcript, "2026-06-13T10:00:00+00:00", dispatch_id="d-missing"),
    ]
    out = _interleave_sub_agent_summaries(messages)
    assert _subs(out) == []


def test_skipped_middle_turn_does_not_confuse_id_lookup(tmp_path):
    """A skipped (errored/empty) turn in the MIDDLE must not affect later
    dispatches: dispatch 3 (id d3) still pairs with turn 3, regardless of
    turn 2 being empty. The id join makes this trivially correct — no index
    to shift — but it's worth pinning as a regression."""
    t = tmp_path / "claude-code" / "run.jsonl"
    t.parent.mkdir(parents=True)
    _write_transcript(t, [
        {"content": "TURN-1 ok", "chunk_type": "response", "timestamp": "2026-06-13T10:00:01+00:00"},
        _boundary("2026-06-13T10:00:02+00:00", "d1"),
        # turn 2 errored: tool + error, NO response chunk
        {"content": "[Using tool: Bash]", "chunk_type": "tool_activity", "timestamp": "2026-06-13T10:01:01+00:00"},
        {"content": "boom", "chunk_type": "error", "timestamp": "2026-06-13T10:01:02+00:00"},
        _boundary("2026-06-13T10:01:03+00:00", "d2"),
        {"content": "TURN-3 ok", "chunk_type": "response", "timestamp": "2026-06-13T10:02:01+00:00"},
        _boundary("2026-06-13T10:02:02+00:00", "d3"),
    ])
    messages = [
        _routed("claude-code", "do 1", t, "2026-06-13T10:00:00+00:00", dispatch_id="d1"),
        _routed("claude-code", "do 2", t, "2026-06-13T10:01:00+00:00", dispatch_id="d2"),
        _routed("claude-code", "do 3", t, "2026-06-13T10:02:00+00:00", dispatch_id="d3"),
    ]
    subs = _subs(_interleave_sub_agent_summaries(messages))
    assert [s["content"] for s in subs] == ["TURN-1 ok", "TURN-3 ok"]  # turn 3 ↔ dispatch 3


def test_two_distinct_transcript_paths_have_isolated_ids(tmp_path):
    """One management session can dispatch to multiple handles (@frontend,
    @backend). The SAME dispatch_id string reused across two different
    transcript paths must not cross-wire — each path's slices are looked up
    (and matched) independently, scoped by transcript_path first."""
    txA = tmp_path / "agentA" / "run.jsonl"; txA.parent.mkdir(parents=True)
    txB = tmp_path / "agentB" / "run.jsonl"; txB.parent.mkdir(parents=True)
    _write_transcript(txA, [
        {"content": "A-done", "chunk_type": "response", "timestamp": "2026-06-13T10:00:01+00:00"},
        _boundary("2026-06-13T10:00:02+00:00", "SAMEID"),
    ])
    _write_transcript(txB, [
        {"content": "B-done", "chunk_type": "response", "timestamp": "2026-06-13T10:01:01+00:00"},
        _boundary("2026-06-13T10:01:02+00:00", "SAMEID"),
    ])
    messages = [
        _routed("agentA", "do A", txA, "2026-06-13T10:00:00+00:00", dispatch_id="SAMEID"),
        _routed("agentB", "do B", txB, "2026-06-13T10:01:00+00:00", dispatch_id="SAMEID"),
    ]
    subs = _subs(_interleave_sub_agent_summaries(messages))
    assert len(subs) == 2
    assert {s["sub_agent_handle"]: s["content"] for s in subs} == {"agentA": "A-done", "agentB": "B-done"}


def test_two_markers_with_same_dispatch_id_render_exactly_one_bubble(tmp_path):
    """Join hygiene: the known pre-existing @mention double-marker (send()'s
    internal 'management_agent' marker + the API route's own 'user_mention'
    marker for the SAME physical dispatch) both carry the identical
    dispatch_id. The renderer must not turn one turn into two bubbles — only
    the FIRST marker gets the bubble; the second renders as its plain
    one-liner (still present in ``out``, just with no sub_agent message
    following it)."""
    transcript = tmp_path / "claude-code" / "run.jsonl"
    transcript.parent.mkdir(parents=True)
    _write_transcript(transcript, [
        {"content": "the one true response", "chunk_type": "response", "timestamp": "2026-06-13T10:00:01+00:00"},
        _boundary("2026-06-13T10:00:02+00:00", "dupe-id"),
    ])
    messages = [
        _routed("claude-code", "do X", transcript, "2026-06-13T10:00:00+00:00",
                dispatch_id="dupe-id"),
        _routed("claude-code", "do X", transcript, "2026-06-13T10:00:00+00:00",
                dispatch_id="dupe-id", user=True),
    ]
    out = _interleave_sub_agent_summaries(messages)
    subs = _subs(out)
    assert len(subs) == 1
    assert subs[0]["content"] == "the one true response"
    # The bubble is attached right after the FIRST marker (index 1, since the
    # first marker sits at index 0); the second marker (now at index 2) is
    # followed by nothing.
    assert out.index(subs[0]) == 1
    assert out[-1] is messages[-1]   # the second marker's one-liner stands alone


def test_started_marker_is_not_an_anchor(tmp_path):
    """Regression guard: the (now-dead) started marker must NOT trigger a
    bubble — it carries no ``_meta`` (LifecycleObserver.on_started never
    stamps one), so it's excluded by the same "no _meta -> no bubble" rule."""
    transcript = tmp_path / "run.jsonl"
    _write_transcript(transcript, [
        {"content": "x", "chunk_type": "response", "timestamp": "2026-06-13T12:00:01+00:00"},
        _boundary("2026-06-13T12:00:02+00:00", "d1"),
    ])
    messages = [
        {"role": "system",
         "content": f"[Sub-agent] claude-code started (initiated by: management_agent). Transcript: {transcript}",
         "source": "daemon", "timestamp": "2026-06-13T11:59:30+00:00"},
    ]
    out = _interleave_sub_agent_summaries(messages)
    assert _subs(out) == []         # started marker alone injects nothing
