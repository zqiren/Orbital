# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Unit tests for OCC protection of metadata files in run_session_end_routine.

Track C2 / dispatch §3.2 — verifies the Optimistic Concurrency Control
pattern that protects the Layer-1 memory files (PROJECT_STATE.md,
DECISIONS.md, LESSONS.md, INDEX.md) from being clobbered when the user
(or any other writer) edits one of them while the session-end LLM call
is in flight.

Pattern under test:
  1. capture st_mtime_ns of each Layer-1 file BEFORE the LLM call
  2. compute new content (single LLM call returning project_state /
     decisions / lessons / index)
  3. re-stat just before the atomic write
  4. if mtimes match: write via WorkspaceFileManager.write (tmp+rename),
     stamping DECISIONS/LESSONS metadata on the way
  5. if mtimes differ: abort with a structured WARNING containing
     project_id, file_path, baseline_mtime, observed_mtime,
     cache_thrash_telemetry=True

Tests cover: per-file abort on user mid-edit, per-file write on clean
baseline, multi-file isolation (one abort does not block the others),
and structured log fields.

SESSION_LOG.md was removed in the Layer-1 memory redesign (no append
file, no per-project append lock), so the tests that exercised its
special lock + re-stat append path are gone — that feature no longer
exists. INDEX.md replaces the old CONTEXT.md.
"""

import json
import logging
import os
from unittest.mock import AsyncMock, MagicMock

import pytest

from tests.testutils import streamable

from agent_os.agent import workspace_files as wsf_module
from agent_os.agent.workspace_files import (
    WorkspaceFileManager,
    run_session_end_routine,
)



from agent_os.agent import memory_entries as _mem


def _hdr(key: str, content: str) -> str:
    """Expected on-disk form: write() self-heals the <!--format--> header."""
    return _mem.FORMAT_HEADERS[key] + "\n" + content

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
    provider = streamable(AsyncMock())
    resp = MagicMock()
    resp.text = response_text
    provider.complete.return_value = resp
    return provider


def _valid_llm_response(tag="x"):
    """A well-formed session-end JSON payload for the new contract.

    Keys are project_state / decisions / lessons / index (no
    session_log_entry, no "context").
    """
    return json.dumps({
        "project_state": f"# State\nstate-{tag}",
        "decisions": f"## 2026-05-20: Decision {tag}\n**Chose:** A\n**Reason:** R",
        "lessons": f"1. Lesson {tag}\n",
        "index": f"- Person {tag}",
    })


@pytest.fixture(autouse=True)
def _reset_module_state():
    """Reset the module-level idempotency set between tests so cases don't
    leak state into each other.

    (The SESSION_LOG per-project append-lock map is gone with the Layer-1
    redesign — there is no longer a ``_session_log_locks`` to clear.)
    """
    wsf_module._completed_session_ends.clear()
    yield
    wsf_module._completed_session_ends.clear()


def _bump_mtime_by(path: str, delta_seconds: float = 5.0) -> int:
    """Mutate a file's mtime so its st_mtime_ns advances.

    Uses os.utime to a fixed later timestamp — simulates a user editing
    the file between when the session-end routine captured the baseline
    and the post-LLM write. Returns the new st_mtime_ns.
    """
    st = os.stat(path)
    new_atime = st.st_atime + delta_seconds
    new_mtime = st.st_mtime + delta_seconds
    os.utime(path, (new_atime, new_mtime))
    return os.stat(path).st_mtime_ns


# ---------------------------------------------------------------------------
# 1. test_clean_baseline_writes_all_layer1_files
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_clean_baseline_writes_all_layer1_files(tmp_path):
    """When no file changes during the LLM call, all 4 Layer-1 files write."""
    ws = WorkspaceFileManager(str(tmp_path))

    # Pre-seed all 4 Layer-1 files so we have a definite baseline mtime.
    ws.write("state", "before-state")
    ws.write("decisions", "## old\n**Chose:** old")
    ws.write("lessons", "1. old lesson")
    ws.write("index", "- old person")

    session = _mock_session(session_id="sess_clean")
    provider = _mock_provider(_valid_llm_response("clean"))

    await run_session_end_routine(
        session, provider, ws,
        session_uuid="sess_clean", project_id="proj_clean",
    )

    # STATE / INDEX are overwrite scratchpads, written verbatim.
    assert ws.read("state") == _hdr("state", "# State\nstate-clean")
    assert ws.read("index") == _hdr("index", "- Person clean")
    # DECISIONS / LESSONS go through the stamping persist path; just verify
    # the marker entry survives (stamp adds a trailing metadata comment).
    assert "Decision clean" in ws.read("decisions")
    assert "Lesson clean" in ws.read("lessons")


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

    provider = streamable(AsyncMock())
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
    """If only LESSONS.md is touched mid-LLM, the other Layer-1 files still write."""
    ws = WorkspaceFileManager(str(tmp_path))
    ws.write("state", "old-state")
    ws.write("decisions", "## 2026-01-01: Old\n**Chose:** old")
    ws.write("lessons", "1. user-lesson")
    ws.write("index", "- old person")

    lessons_path = ws._file_path("lessons")
    user_lessons = "1. user-lesson\n2. user-additional-lesson-during-llm"

    async def _llm_side_effect(*args, **kwargs):
        with open(lessons_path, "w", encoding="utf-8") as f:
            f.write(user_lessons)
        _bump_mtime_by(lessons_path, delta_seconds=2.0)
        resp = MagicMock()
        resp.text = _valid_llm_response("multi")
        return resp

    provider = streamable(AsyncMock())
    provider.complete.side_effect = _llm_side_effect

    session = _mock_session(session_id="sess_multi")

    with caplog.at_level(logging.WARNING, logger="agent_os.agent.workspace_files"):
        await run_session_end_routine(
            session, provider, ws,
            session_uuid="sess_multi", project_id="proj_multi",
        )

    # Lessons abort → user content survives
    assert ws.read("lessons") == user_lessons

    # Other Layer-1 files write normally
    assert ws.read("state") == _hdr("state", "# State\nstate-multi")
    assert "Decision multi" in ws.read("decisions")
    assert ws.read("index") == _hdr("index", "- Person multi")

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
    # No pre-seeding. All Layer-1 files are absent. With no marker present
    # either, the no-delta gate still allows the routine to run (the LLM
    # output creates the files for the first time).

    session = _mock_session(session_id="sess_first")
    provider = _mock_provider(_valid_llm_response("first"))

    await run_session_end_routine(
        session, provider, ws,
        session_uuid="sess_first", project_id="proj_first",
    )

    # All four Layer-1 files now exist
    assert ws.exists("state")
    assert ws.exists("decisions")
    assert ws.exists("lessons")
    assert ws.exists("index")


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

    provider = streamable(AsyncMock())
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
# 6. test_log_includes_all_required_structured_fields
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

    provider = streamable(AsyncMock())
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
