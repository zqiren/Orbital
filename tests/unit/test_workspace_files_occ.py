# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Unit tests for OCC protection of metadata files in run_session_end_routine.

Track C2 / dispatch §3.2 — verifies the Optimistic Concurrency Control
pattern that protects PROJECT_STATE.md, DECISIONS.md, SESSION_LOG.md,
LESSONS.md, and CONTEXT.md from being clobbered when the user (or any
other writer) edits one of them while the session-end LLM call is in
flight.

Pattern under test:
  1. capture st_mtime_ns of each file BEFORE the LLM call (or before
     the read inside the SESSION_LOG asyncio.Lock)
  2. compute new content (LLM call for the four overwrite files,
     read+append for SESSION_LOG)
  3. re-stat just before the atomic write
  4. if mtimes match: write via WorkspaceFileManager.write (tmp+rename)
  5. if mtimes differ: abort with a structured WARNING containing
     project_id, file_path, baseline_mtime, observed_mtime,
     cache_thrash_telemetry=True

Tests cover: per-file abort on user mid-edit, per-file write on clean
baseline, the special SESSION_LOG lock+OCC interaction, multi-file
isolation (one abort does not block the others), and structured log
fields. An integration-flavored test exercises two concurrent
session-end routines hitting SESSION_LOG.md to verify the lock keeps
them from corrupting the file.
"""

import asyncio
import json
import logging
import os
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent_os.agent import workspace_files as wsf_module
from agent_os.agent.workspace_files import (
    WorkspaceFileManager,
    run_session_end_routine,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_session(messages=None, session_id="sess_occ"):
    session = MagicMock()
    session.session_id = session_id
    session.get_messages.return_value = messages or [
        {"role": "user", "content": "Hello"},
    ]
    return session


def _mock_provider(response_text):
    provider = AsyncMock()
    resp = MagicMock()
    resp.text = response_text
    provider.complete.return_value = resp
    return provider


def _valid_llm_response(tag="x"):
    return json.dumps({
        "project_state": f"# State\nstate-{tag}",
        "decisions": f"## 2026-05-20: Decision {tag}\n**Chose:** A\n**Reason:** R",
        "session_log_entry": f"## Session {tag} -- today\n- Did thing",
        "lessons": f"1. Lesson {tag}\n",
        "context": f"- Person {tag}",
    })


@pytest.fixture(autouse=True)
def _reset_module_state():
    """Reset module-level idempotency set and SESSION_LOG lock map
    between tests so cases don't leak state into each other."""
    wsf_module._completed_session_ends.clear()
    wsf_module._session_log_locks.clear()
    yield
    wsf_module._completed_session_ends.clear()
    wsf_module._session_log_locks.clear()


def _bump_mtime_by(path: str, delta_seconds: float = 5.0) -> int:
    """Mutate a file in-place so its st_mtime_ns advances.

    Writes one byte then resets the file but using os.utime to a fixed
    later timestamp — simulates a user editing the file between when
    the session-end routine captured the baseline and the post-LLM
    write. Returns the new st_mtime_ns.
    """
    st = os.stat(path)
    new_atime = st.st_atime + delta_seconds
    new_mtime = st.st_mtime + delta_seconds
    os.utime(path, (new_atime, new_mtime))
    return os.stat(path).st_mtime_ns


# ---------------------------------------------------------------------------
# 1. test_clean_baseline_writes_all_five_files
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_clean_baseline_writes_all_five_files(tmp_path):
    """When no file changes during the LLM call, all 5 files are written."""
    ws = WorkspaceFileManager(str(tmp_path))

    # Pre-seed all 5 files so we have a definite baseline mtime.
    ws.write("state", "before-state")
    ws.write("decisions", "## old\n**Chose:** old")
    ws.write("session_log", "## Session old -- old\n- old")
    ws.write("lessons", "1. old lesson")
    ws.write("context", "- old person")

    session = _mock_session(session_id="sess_clean")
    provider = _mock_provider(_valid_llm_response("clean"))

    await run_session_end_routine(
        session, provider, ws,
        session_uuid="sess_clean", project_id="proj_clean",
    )

    assert ws.read("state") == "# State\nstate-clean"
    # decisions and lessons go through sanity-checks; just verify the
    # marker entry is present (single entry preserved verbatim).
    assert "Decision clean" in ws.read("decisions")
    assert "Lesson clean" in ws.read("lessons")
    # SESSION_LOG is append-style — old entry must remain alongside new
    assert "Session old" in ws.read("session_log")
    assert "Session clean" in ws.read("session_log")
    assert "Person clean" in ws.read("context")


# ---------------------------------------------------------------------------
# 2. test_user_edit_during_llm_aborts_state_write
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_user_edit_during_llm_aborts_state_write(tmp_path, caplog):
    """User touches PROJECT_STATE.md while the LLM is running → abort."""
    ws = WorkspaceFileManager(str(tmp_path))
    ws.write("state", "user-original-state")

    state_path = ws._file_path("state")
    user_edited_bytes = "user-original-state\n## USER ADDED\nhand-written note"

    session = _mock_session(session_id="sess_edit_state")

    # Wedge the user edit into the LLM call itself: when provider.complete
    # is awaited, simulate a user write that bumps the mtime AND changes
    # the on-disk content. This is the canonical "mid-LLM user edit" race.
    async def _llm_side_effect(*args, **kwargs):
        with open(state_path, "w", encoding="utf-8") as f:
            f.write(user_edited_bytes)
        _bump_mtime_by(state_path, delta_seconds=2.0)
        resp = MagicMock()
        resp.text = _valid_llm_response("after_edit")
        return resp

    provider = AsyncMock()
    provider.complete.side_effect = _llm_side_effect

    with caplog.at_level(logging.WARNING, logger="agent_os.agent.workspace_files"):
        await run_session_end_routine(
            session, provider, ws,
            session_uuid="sess_edit_state", project_id="proj_edit",
        )

    # PROJECT_STATE write must be aborted — user's content survives.
    assert ws.read("state") == user_edited_bytes

    # Structured warning was logged
    matching = [r for r in caplog.records if "OCC abort" in r.message and "state" in r.message]
    assert matching, f"expected OCC abort warning for state, got: {[r.message for r in caplog.records]}"
    rec = matching[0]
    # Both human-readable and extra fields carry the structured payload.
    assert getattr(rec, "project_id", None) == "proj_edit"
    assert getattr(rec, "file_path", None) == state_path
    assert getattr(rec, "cache_thrash_telemetry", None) is True
    assert getattr(rec, "baseline_mtime", None) != getattr(rec, "observed_mtime", None)


# ---------------------------------------------------------------------------
# 3. test_one_file_aborts_others_still_write
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_one_file_aborts_others_still_write(tmp_path, caplog):
    """If only LESSONS.md is touched mid-LLM, the other four files still write."""
    ws = WorkspaceFileManager(str(tmp_path))
    ws.write("state", "old-state")
    ws.write("decisions", "## 2026-01-01: Old\n**Chose:** old")
    ws.write("session_log", "## Session old -- 2026-01-01\n- old")
    ws.write("lessons", "1. user-lesson")
    ws.write("context", "- old person")

    lessons_path = ws._file_path("lessons")
    user_lessons = "1. user-lesson\n2. user-additional-lesson-during-llm"

    async def _llm_side_effect(*args, **kwargs):
        with open(lessons_path, "w", encoding="utf-8") as f:
            f.write(user_lessons)
        _bump_mtime_by(lessons_path, delta_seconds=2.0)
        resp = MagicMock()
        resp.text = _valid_llm_response("multi")
        return resp

    provider = AsyncMock()
    provider.complete.side_effect = _llm_side_effect

    session = _mock_session(session_id="sess_multi")

    with caplog.at_level(logging.WARNING, logger="agent_os.agent.workspace_files"):
        await run_session_end_routine(
            session, provider, ws,
            session_uuid="sess_multi", project_id="proj_multi",
        )

    # Lessons abort → user content survives
    assert ws.read("lessons") == user_lessons

    # Other four files write normally
    assert ws.read("state") == "# State\nstate-multi"
    assert "Decision multi" in ws.read("decisions")
    assert "Person multi" in ws.read("context")
    assert "Session multi" in ws.read("session_log")
    # Old session_log entry must also survive (append, not overwrite)
    assert "Session old" in ws.read("session_log")

    # Exactly one OCC abort warning, for lessons
    aborts = [r for r in caplog.records if "OCC abort" in r.message]
    assert len(aborts) == 1
    assert "lessons" in aborts[0].message


# ---------------------------------------------------------------------------
# 4. test_nonexistent_file_baseline_allows_initial_write
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_nonexistent_file_baseline_allows_initial_write(tmp_path):
    """Files that don't exist at baseline write normally (baseline=None)."""
    ws = WorkspaceFileManager(str(tmp_path))
    # No pre-seeding. All 5 files are absent.

    session = _mock_session(session_id="sess_first")
    provider = _mock_provider(_valid_llm_response("first"))

    await run_session_end_routine(
        session, provider, ws,
        session_uuid="sess_first", project_id="proj_first",
    )

    # All five files now exist
    assert ws.exists("state")
    assert ws.exists("decisions")
    assert ws.exists("session_log")
    assert ws.exists("lessons")
    assert ws.exists("context")


# ---------------------------------------------------------------------------
# 5. test_user_creates_file_during_llm_aborts_state_write
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_user_creates_file_during_llm_aborts_state_write(tmp_path, caplog):
    """If a file is absent at baseline but the user creates it before
    the post-LLM write, the abort path catches it."""
    ws = WorkspaceFileManager(str(tmp_path))
    # state is absent at baseline.
    state_path = ws._file_path("state")
    user_content = "user-wrote-this-while-llm-was-running"

    async def _llm_side_effect(*args, **kwargs):
        ws.ensure_dir()
        with open(state_path, "w", encoding="utf-8") as f:
            f.write(user_content)
        resp = MagicMock()
        resp.text = _valid_llm_response("late")
        return resp

    provider = AsyncMock()
    provider.complete.side_effect = _llm_side_effect

    session = _mock_session(session_id="sess_create_race")

    with caplog.at_level(logging.WARNING, logger="agent_os.agent.workspace_files"):
        await run_session_end_routine(
            session, provider, ws,
            session_uuid="sess_create_race", project_id="proj_create",
        )

    # User-created content survived; routine did not clobber it.
    assert ws.read("state") == user_content

    # Warning fired
    aborts = [r for r in caplog.records if "OCC abort" in r.message and "state" in r.message]
    assert aborts, "expected OCC abort warning for state on baseline=None mismatch"


# ---------------------------------------------------------------------------
# 6. test_session_log_user_edit_aborts_append
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_session_log_user_edit_aborts_append(tmp_path, caplog):
    """SESSION_LOG.md user edit between baseline-stat and write aborts cleanly."""
    ws = WorkspaceFileManager(str(tmp_path))
    ws.write("session_log", "## Session old -- 2026-01-01\n- old\n")

    session_log_path = ws._file_path("session_log")

    # Patch _stat_mtime_ns so the SECOND call (the pre-write re-stat
    # inside the lock) returns a value different from the first call
    # (the in-lock baseline). This simulates a user edit landing
    # between the baseline capture and the atomic-write moment.
    original_stat = wsf_module._stat_mtime_ns
    call_count = {"n": 0}

    def fake_stat(path: str):
        real = original_stat(path)
        if path == session_log_path:
            call_count["n"] += 1
            # Second call (the re-stat just before write) returns a
            # different mtime to simulate a user edit landing in between.
            if call_count["n"] == 2:
                return (real or 0) + 1_000_000  # +1 ms
        return real

    user_session_log = "## Session old -- 2026-01-01\n- old\n## USER WROTE\nnotes\n"

    session = _mock_session(session_id="sess_sl_user_edit")
    provider = _mock_provider(_valid_llm_response("sl_user"))

    # Mutate the file content too (so the abort actually preserves user data)
    async def _llm_side_effect(*args, **kwargs):
        with open(session_log_path, "w", encoding="utf-8") as f:
            f.write(user_session_log)
        resp = MagicMock()
        resp.text = _valid_llm_response("sl_user")
        return resp

    provider.complete.side_effect = _llm_side_effect

    with caplog.at_level(logging.WARNING, logger="agent_os.agent.workspace_files"):
        monkeypatch_target = wsf_module
        original = monkeypatch_target._stat_mtime_ns
        monkeypatch_target._stat_mtime_ns = fake_stat
        try:
            await run_session_end_routine(
                session, provider, ws,
                session_uuid="sess_sl_user_edit", project_id="proj_sl_user",
            )
        finally:
            monkeypatch_target._stat_mtime_ns = original

    # User edit must survive.
    assert ws.read("session_log") == user_session_log

    # Structured warning for session_log
    aborts = [r for r in caplog.records if "OCC abort" in r.message and "session_log" in r.message]
    assert aborts, "expected OCC abort for session_log"


# ---------------------------------------------------------------------------
# 7. test_session_log_lock_serializes_concurrent_routines
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_session_log_lock_serializes_concurrent_routines(tmp_path):
    """Two concurrent session-end routines targeting the same SESSION_LOG
    must not corrupt the file. Both entries should land cleanly."""
    ws = WorkspaceFileManager(str(tmp_path))
    ws.write("session_log", "## Session base -- 2026-01-01\n- base\n")

    # Provider with a small await inside so the two routines actually
    # overlap. Without the lock, the second routine's append would
    # read+write-on-stale data and at minimum trigger the OCC abort
    # warning (wasted LLM call); we want zero aborts here.
    async def _delayed_llm(tag):
        async def _impl(*args, **kwargs):
            await asyncio.sleep(0.05)
            resp = MagicMock()
            resp.text = _valid_llm_response(tag)
            return resp
        return _impl

    p1 = AsyncMock()
    p1.complete.side_effect = await _delayed_llm("conc_a")
    p2 = AsyncMock()
    p2.complete.side_effect = await _delayed_llm("conc_b")

    session1 = _mock_session(session_id="sess_a")
    session2 = _mock_session(session_id="sess_b")

    # bypass_idempotency on both so the routines actually run (different
    # session_ids would also work but bypass is more explicit about
    # what's being tested here).
    await asyncio.gather(
        run_session_end_routine(
            session1, p1, ws,
            session_uuid="sess_a", project_id="proj_conc",
            bypass_idempotency=True,
        ),
        run_session_end_routine(
            session2, p2, ws,
            session_uuid="sess_b", project_id="proj_conc",
            bypass_idempotency=True,
        ),
    )

    final = ws.read("session_log") or ""
    # All three entries (base + conc_a + conc_b) present, no corruption
    assert "Session base" in final
    assert "Session conc_a" in final
    assert "Session conc_b" in final

    # File parses as well-formed markdown (no half-written content,
    # no doubled headers, no torn lines)
    # Each "## Session" header marks one entry; there should be exactly 3.
    import re
    headers = re.findall(r'(?m)^## Session ', final)
    assert len(headers) == 3, f"expected 3 session entries, got {len(headers)}:\n{final}"


# ---------------------------------------------------------------------------
# 8. test_session_log_lock_keyed_by_project_id
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_session_log_lock_keyed_by_project_id(tmp_path):
    """Two routines for DIFFERENT projects do not block each other."""
    # Two separate workspaces == two separate SESSION_LOG files == two
    # independent lock entries. The lock helper is exercised directly
    # here rather than through the full routine to keep the test focused.
    lock_a = wsf_module._get_session_log_lock("proj_a")
    lock_b = wsf_module._get_session_log_lock("proj_b")

    assert lock_a is not lock_b
    # Same key returns the same instance
    assert wsf_module._get_session_log_lock("proj_a") is lock_a


# ---------------------------------------------------------------------------
# 9. test_log_includes_all_required_structured_fields
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_log_includes_all_required_structured_fields(tmp_path, caplog):
    """The OCC-abort log record must carry the exact set of structured
    fields specified in dispatch §3.2: project_id, file_path,
    baseline_mtime, observed_mtime, cache_thrash_telemetry=True."""
    ws = WorkspaceFileManager(str(tmp_path))
    ws.write("decisions", "## old decision\n**Chose:** old")
    decisions_path = ws._file_path("decisions")

    async def _llm_side_effect(*args, **kwargs):
        _bump_mtime_by(decisions_path, delta_seconds=2.0)
        resp = MagicMock()
        resp.text = _valid_llm_response("struct")
        return resp

    provider = AsyncMock()
    provider.complete.side_effect = _llm_side_effect
    session = _mock_session(session_id="sess_struct")

    with caplog.at_level(logging.WARNING, logger="agent_os.agent.workspace_files"):
        await run_session_end_routine(
            session, provider, ws,
            session_uuid="sess_struct", project_id="proj_struct_log",
        )

    matching = [r for r in caplog.records if "OCC abort" in r.message and "decisions" in r.message]
    assert matching
    rec = matching[0]

    assert getattr(rec, "project_id", None) == "proj_struct_log"
    assert getattr(rec, "file_path", None) == decisions_path
    assert isinstance(getattr(rec, "baseline_mtime", None), int)
    assert isinstance(getattr(rec, "observed_mtime", None), int)
    assert getattr(rec, "cache_thrash_telemetry", None) is True
    assert rec.levelno == logging.WARNING
