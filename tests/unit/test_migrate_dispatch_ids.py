# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Tests for scripts/migrate_dispatch_ids.py — the one-shot legacy-data
backfill for TASK-dispatch-id-pairing (commit 6da5f98).

``scripts/`` is not a package (no ``__init__.py``, not part of
``[tool.setuptools.packages.find]``), so the module under test is loaded via
``importlib`` from its file path rather than a normal import.

Fixture shapes mirror exactly what the daemon writes (verified against
``LifecycleObserver.on_message_routed`` / ``agent_manager.inject_system_message``
for session records, and ``ProcessManager._append_turn_boundary`` for
transcript boundary rows — see ``tests/unit/test_interleave_sub_agent_summaries.py``
for the same conventions):

- Session record: ``{"role": "system", "content": "[Sub-agent] ...
  Transcript: <path>", "source": "daemon", "timestamp": "...", "_meta": {...}}``
  — legacy markers omit ``_meta`` (or carry one without ``dispatch_id``).
- Transcript boundary row: ``{"source": <handle>, "content": "",
  "chunk_type": "turn_complete", "timestamp": "...", "dispatch_id": "..."}``
  — legacy boundaries omit ``dispatch_id``.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest

_SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "migrate_dispatch_ids.py"
_spec = importlib.util.spec_from_file_location("migrate_dispatch_ids", _SCRIPT_PATH)
migrate_dispatch_ids = importlib.util.module_from_spec(_spec)
sys.modules["migrate_dispatch_ids"] = migrate_dispatch_ids
_spec.loader.exec_module(migrate_dispatch_ids)

from agent_os.api.routes.agents_v2 import _interleave_sub_agent_summaries


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------

def _write_jsonl(path, records):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _read_raw(path):
    with open(path, "rb") as f:
        return f.read()


def _marker(handle, transcript_path, ts, *, meta=None, user=True):
    if user:
        content = f'[Sub-agent] User sent @{handle}: "hello". Transcript: {transcript_path}'
    else:
        content = f'[Sub-agent] Message sent to {handle}: "hello". Transcript: {transcript_path}'
    rec = {"role": "system", "content": content, "source": "daemon", "timestamp": ts}
    if meta is not None:
        rec["_meta"] = meta
    return rec


def _response(text, ts):
    return {"content": text, "chunk_type": "response", "timestamp": ts}


def _boundary(ts, dispatch_id=None):
    row = {"source": "claude-code", "content": "", "chunk_type": "turn_complete", "timestamp": ts}
    if dispatch_id is not None:
        row["dispatch_id"] = dispatch_id
    return row


def _sessions_dir(workspace):
    return os.path.join(workspace, "orbital", "sessions")


def _transcript_path(workspace, handle, name="abc12345.jsonl"):
    return os.path.join(workspace, "orbital", "sub_agents", handle, name)


# ---------------------------------------------------------------------------
# (a) happy path — 2 session files, 3 turns, renderer compatibility
# ---------------------------------------------------------------------------

def test_happy_path_stamps_and_renders_correct_bubbles(tmp_path):
    workspace = str(tmp_path)
    handle = "claude-code"
    transcript = _transcript_path(workspace, handle)

    session1 = os.path.join(_sessions_dir(workspace), "session1.jsonl")
    session2 = os.path.join(_sessions_dir(workspace), "session2.jsonl")

    _write_jsonl(session1, [
        _marker(handle, transcript, "2026-01-01T10:00:00+00:00"),
        _marker(handle, transcript, "2026-01-01T10:01:00+00:00"),
    ])
    _write_jsonl(session2, [
        _marker(handle, transcript, "2026-01-01T10:02:00+00:00"),
    ])
    _write_jsonl(transcript, [
        _response("response to turn 1", "2026-01-01T10:00:04+00:00"),
        _boundary("2026-01-01T10:00:05+00:00"),
        _response("response to turn 2", "2026-01-01T10:01:04+00:00"),
        _boundary("2026-01-01T10:01:05+00:00"),
        _response("response to turn 3", "2026-01-01T10:02:04+00:00"),
        _boundary("2026-01-01T10:02:05+00:00"),
    ])

    result = migrate_dispatch_ids.migrate_workspace(workspace, dry_run=False)

    assert result.legacy_markers_found == 3
    assert result.already_migrated_markers == 0
    assert result.total_pairs_stamped == 3
    assert len(result.transcripts) == 1
    assert result.transcripts[0].status == "stamped"

    # Boundaries all stamped, ids traceable to the correct session file stem.
    transcript_records = _read_jsonl(transcript)
    boundaries = [r for r in transcript_records if r.get("chunk_type") == "turn_complete"]
    assert len(boundaries) == 3
    assert boundaries[0]["dispatch_id"].startswith("session1:")
    assert boundaries[1]["dispatch_id"].startswith("session1:")
    assert boundaries[2]["dispatch_id"].startswith("session2:")
    assert len({b["dispatch_id"] for b in boundaries}) == 3  # all distinct

    # Markers stamped with matching _meta.
    s1_records = _read_jsonl(session1)
    s2_records = _read_jsonl(session2)
    assert s1_records[0]["_meta"]["dispatch_id"] == boundaries[0]["dispatch_id"]
    assert s1_records[0]["_meta"]["handle"] == handle
    assert s1_records[0]["_meta"]["transcript_path"] == transcript
    assert s1_records[1]["_meta"]["dispatch_id"] == boundaries[1]["dispatch_id"]
    assert s2_records[0]["_meta"]["dispatch_id"] == boundaries[2]["dispatch_id"]

    # Renderer compatibility: run the REAL read path over the migrated data.
    out1 = _interleave_sub_agent_summaries(s1_records)
    subs1 = [m for m in out1 if m.get("source") == "sub_agent"]
    assert [m["content"] for m in subs1] == ["response to turn 1", "response to turn 2"]

    out2 = _interleave_sub_agent_summaries(s2_records)
    subs2 = [m for m in out2 if m.get("source") == "sub_agent"]
    assert [m["content"] for m in subs2] == ["response to turn 3"]


# ---------------------------------------------------------------------------
# (b) idempotency
# ---------------------------------------------------------------------------

def test_second_run_is_a_noop(tmp_path):
    workspace = str(tmp_path)
    handle = "claude-code"
    transcript = _transcript_path(workspace, handle)
    session1 = os.path.join(_sessions_dir(workspace), "session1.jsonl")

    _write_jsonl(session1, [_marker(handle, transcript, "2026-01-01T10:00:00+00:00")])
    _write_jsonl(transcript, [
        _response("only turn", "2026-01-01T10:00:04+00:00"),
        _boundary("2026-01-01T10:00:05+00:00"),
    ])

    first = migrate_dispatch_ids.migrate_workspace(workspace, dry_run=False)
    assert first.total_pairs_stamped == 1

    bytes_after_first_session = _read_raw(session1)
    bytes_after_first_transcript = _read_raw(transcript)

    second = migrate_dispatch_ids.migrate_workspace(workspace, dry_run=False)
    assert second.total_pairs_stamped == 0
    assert second.legacy_markers_found == 0
    assert second.already_migrated_markers == 1
    assert second.transcripts == []

    assert _read_raw(session1) == bytes_after_first_session
    assert _read_raw(transcript) == bytes_after_first_transcript


# ---------------------------------------------------------------------------
# (c) count mismatch → nothing stamped for that transcript
# ---------------------------------------------------------------------------

def test_count_mismatch_leaves_transcript_untouched(tmp_path):
    workspace = str(tmp_path)
    handle = "claude-code"
    transcript = _transcript_path(workspace, handle)
    session1 = os.path.join(_sessions_dir(workspace), "session1.jsonl")

    # 2 legacy markers, but only 1 turn boundary in the transcript.
    _write_jsonl(session1, [
        _marker(handle, transcript, "2026-01-01T10:00:00+00:00"),
        _marker(handle, transcript, "2026-01-01T10:01:00+00:00"),
    ])
    _write_jsonl(transcript, [
        _response("only turn", "2026-01-01T10:00:04+00:00"),
        _boundary("2026-01-01T10:00:05+00:00"),
    ])

    before_session = _read_raw(session1)
    before_transcript = _read_raw(transcript)

    result = migrate_dispatch_ids.migrate_workspace(workspace, dry_run=False)

    assert result.total_pairs_stamped == 0
    assert len(result.transcripts) == 1
    report = result.transcripts[0]
    assert report.status == "skipped"
    assert "count mismatch" in report.reason

    assert _read_raw(session1) == before_session
    assert _read_raw(transcript) == before_transcript


# ---------------------------------------------------------------------------
# (d) timestamp-order violation → fail closed
# ---------------------------------------------------------------------------

def test_timestamp_order_violation_fails_closed(tmp_path):
    workspace = str(tmp_path)
    handle = "claude-code"
    transcript = _transcript_path(workspace, handle)
    session1 = os.path.join(_sessions_dir(workspace), "session1.jsonl")

    # Marker fires AFTER the boundary that (supposedly) closed its turn.
    _write_jsonl(session1, [
        _marker(handle, transcript, "2026-01-01T10:00:10+00:00"),
    ])
    _write_jsonl(transcript, [
        _response("only turn", "2026-01-01T10:00:00+00:00"),
        _boundary("2026-01-01T10:00:05+00:00"),
    ])

    before_session = _read_raw(session1)
    before_transcript = _read_raw(transcript)

    result = migrate_dispatch_ids.migrate_workspace(workspace, dry_run=False)

    assert result.total_pairs_stamped == 0
    report = result.transcripts[0]
    assert report.status == "skipped"
    assert "after its turn's boundary timestamp" in report.reason

    assert _read_raw(session1) == before_session
    assert _read_raw(transcript) == before_transcript


# ---------------------------------------------------------------------------
# (e) mixed-era transcript
# ---------------------------------------------------------------------------

def test_mixed_era_transcript_only_stamps_unstamped_prefix(tmp_path):
    workspace = str(tmp_path)
    handle = "claude-code"
    transcript = _transcript_path(workspace, handle)
    session1 = os.path.join(_sessions_dir(workspace), "session1.jsonl")

    # turn 1 is legacy (unstamped); turn 2 was already dispatched by the new
    # code (post-migration in production terms) and already carries an id.
    _write_jsonl(session1, [
        _marker(handle, transcript, "2026-01-01T10:00:00+00:00"),
    ])
    _write_jsonl(transcript, [
        _response("legacy turn", "2026-01-01T10:00:04+00:00"),
        _boundary("2026-01-01T10:00:05+00:00"),
        _response("already-stamped turn", "2026-01-01T10:01:04+00:00"),
        _boundary("2026-01-01T10:01:05+00:00", dispatch_id="newSess:eeeeeeee"),
    ])

    result = migrate_dispatch_ids.migrate_workspace(workspace, dry_run=False)

    assert result.total_pairs_stamped == 1
    report = result.transcripts[0]
    assert report.status == "stamped"
    assert report.unstamped_turns == 1

    boundaries = [r for r in _read_jsonl(transcript) if r.get("chunk_type") == "turn_complete"]
    assert boundaries[0]["dispatch_id"].startswith("session1:")
    assert boundaries[1]["dispatch_id"] == "newSess:eeeeeeee"  # untouched

    s1_records = _read_jsonl(session1)
    assert s1_records[0]["_meta"]["dispatch_id"] == boundaries[0]["dispatch_id"]


def test_mixed_era_unstamped_after_stamped_fails_closed(tmp_path):
    """An unstamped turn appearing AFTER an already-stamped one is an
    inconsistency (dispatch_id stamping ships as a one-way deploy) — must
    fail closed rather than guess."""
    workspace = str(tmp_path)
    handle = "claude-code"
    transcript = _transcript_path(workspace, handle)
    session1 = os.path.join(_sessions_dir(workspace), "session1.jsonl")

    _write_jsonl(session1, [
        _marker(handle, transcript, "2026-01-01T10:01:00+00:00"),
    ])
    _write_jsonl(transcript, [
        _response("already-stamped turn", "2026-01-01T10:00:04+00:00"),
        _boundary("2026-01-01T10:00:05+00:00", dispatch_id="newSess:eeeeeeee"),
        _response("legacy turn out of order", "2026-01-01T10:01:04+00:00"),
        _boundary("2026-01-01T10:01:05+00:00"),
    ])

    before_transcript = _read_raw(transcript)
    result = migrate_dispatch_ids.migrate_workspace(workspace, dry_run=False)

    assert result.total_pairs_stamped == 0
    report = result.transcripts[0]
    assert report.status == "skipped"
    assert "inconsistent interleaving" in report.reason
    assert _read_raw(transcript) == before_transcript


# ---------------------------------------------------------------------------
# (f) dry-run writes nothing
# ---------------------------------------------------------------------------

def test_dry_run_writes_nothing(tmp_path):
    workspace = str(tmp_path)
    handle = "claude-code"
    transcript = _transcript_path(workspace, handle)
    session1 = os.path.join(_sessions_dir(workspace), "session1.jsonl")

    _write_jsonl(session1, [_marker(handle, transcript, "2026-01-01T10:00:00+00:00")])
    _write_jsonl(transcript, [
        _response("only turn", "2026-01-01T10:00:04+00:00"),
        _boundary("2026-01-01T10:00:05+00:00"),
    ])

    before_session = _read_raw(session1)
    before_transcript = _read_raw(transcript)

    result = migrate_dispatch_ids.migrate_workspace(workspace, dry_run=True)

    # Analysis still finds the reconcilable pair...
    assert result.total_pairs_stamped == 1
    assert result.transcripts[0].status == "stamped"
    # ...but nothing is written.
    assert _read_raw(session1) == before_session
    assert _read_raw(transcript) == before_transcript


# ---------------------------------------------------------------------------
# (g) flat legacy transcript untouched
# ---------------------------------------------------------------------------

def test_flat_legacy_transcript_left_untouched(tmp_path):
    workspace = str(tmp_path)
    handle = "claude-code"
    transcript = _transcript_path(workspace, handle)
    session1 = os.path.join(_sessions_dir(workspace), "session1.jsonl")

    _write_jsonl(session1, [_marker(handle, transcript, "2026-01-01T10:00:00+00:00")])
    # No turn_complete boundary at all — a flat pre-instrumentation transcript.
    _write_jsonl(transcript, [
        _response("whole file is one turn", "2026-01-01T10:00:04+00:00"),
    ])

    before_session = _read_raw(session1)
    before_transcript = _read_raw(transcript)

    result = migrate_dispatch_ids.migrate_workspace(workspace, dry_run=False)

    assert result.total_pairs_stamped == 0
    report = result.transcripts[0]
    assert report.status == "skipped"
    assert "flat legacy transcript" in report.reason

    assert _read_raw(session1) == before_session
    assert _read_raw(transcript) == before_transcript


# ---------------------------------------------------------------------------
# (h) _meta merge preserves existing keys
# ---------------------------------------------------------------------------

def test_meta_merge_preserves_existing_keys(tmp_path):
    workspace = str(tmp_path)
    handle = "claude-code"
    transcript = _transcript_path(workspace, handle)
    session1 = os.path.join(_sessions_dir(workspace), "session1.jsonl")

    # A marker that already carries _meta for some OTHER reason (no
    # dispatch_id yet) — the migration must not clobber the other key.
    _write_jsonl(session1, [
        _marker(handle, transcript, "2026-01-01T10:00:00+00:00", meta={"yield_turn": True}),
    ])
    _write_jsonl(transcript, [
        _response("only turn", "2026-01-01T10:00:04+00:00"),
        _boundary("2026-01-01T10:00:05+00:00"),
    ])

    result = migrate_dispatch_ids.migrate_workspace(workspace, dry_run=False)
    assert result.total_pairs_stamped == 1

    s1_records = _read_jsonl(session1)
    meta = s1_records[0]["_meta"]
    assert meta["yield_turn"] is True
    assert meta["dispatch_id"].startswith("session1:")
    assert meta["handle"] == handle
    assert meta["transcript_path"] == transcript


# ---------------------------------------------------------------------------
# CLI plumbing
# ---------------------------------------------------------------------------

def test_resolve_workspaces_combines_direct_and_projects_json(tmp_path):
    projects_json = tmp_path / "projects.json"
    projects_json.write_text(json.dumps({
        "proj-a": {"workspace": "/ws/a"},
        "proj-b": {"workspace": "/ws/b"},
    }))
    resolved = migrate_dispatch_ids._resolve_workspaces(
        ["/ws/a", "/ws/c"], str(projects_json),
    )
    assert resolved == ["/ws/a", "/ws/c", "/ws/b"]


def test_main_errors_without_any_workspace_source(capsys):
    with pytest.raises(SystemExit):
        migrate_dispatch_ids.main([])
    captured = capsys.readouterr()
    assert "no workspaces given" in captured.err
