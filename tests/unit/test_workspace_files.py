# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""
Unit tests for WorkspaceFileManager and session-end routine.

Covers:
  - File CRUD operations (ensure_dir, read, write, append, read_all, exists)
  - Cold resume context assembly (all files, minimal, truncation)
  - Session-end routine = memory editor + floor (choices by id, bad JSON,
    whole-file replies ignored, utility provider)
"""

import json
import logging
import os
from unittest.mock import AsyncMock, MagicMock

import pytest

from tests.testutils import streamable

from agent_os.agent.workspace_files import (
    FILE_NAMES,
    WorkspaceFileManager,
    run_session_end_routine,
)



from agent_os.agent import memory_entries as _mem


def _hdr(key: str, content: str) -> str:
    """Expected on-disk form: write() self-heals the <!--format--> header."""
    return _mem.FORMAT_HEADERS[key] + "\n" + content

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def ws(tmp_path):
    """Return a WorkspaceFileManager rooted at tmp_path."""
    return WorkspaceFileManager(str(tmp_path))


@pytest.fixture
def ws_dir(tmp_path):
    """Return the orbital directory path (not yet created)."""
    return os.path.join(str(tmp_path), "orbital")


def _mock_session(messages=None, session_id="sess_test123", session_uuid=None):
    """Build a mock session with get_messages() returning the given list.

    Both ``session_id`` (F1) and ``session_uuid`` (F2) are set; the
    workspace-files routine now keys idempotency on ``session_uuid`` per the
    F7 canonical rename. ``session_uuid`` defaults to ``session_id`` so old
    tests that only care about a unique key keep working unchanged.
    """
    session = MagicMock()
    session.session_id = session_id
    session.session_uuid = session_uuid if session_uuid is not None else session_id
    session.get_messages.return_value = messages or []
    return session


def _mock_provider(response_text):
    """Build a mock LLM provider whose complete() returns an object with .text."""
    provider = streamable(AsyncMock())
    resp = MagicMock()
    resp.text = response_text
    provider.complete.return_value = resp
    return provider


# ---------------------------------------------------------------------------
# 1. test_ensure_dir_creates
# ---------------------------------------------------------------------------

def test_ensure_dir_creates(ws, ws_dir):
    """Workspace dir created on first call."""
    assert not os.path.isdir(ws_dir)
    ws.ensure_dir()
    assert os.path.isdir(ws_dir)


# ---------------------------------------------------------------------------
# 2. test_read_nonexistent_returns_none
# ---------------------------------------------------------------------------

def test_read_nonexistent_returns_none(ws):
    """Reading a missing file returns None."""
    assert ws.read("state") is None
    assert ws.read("decisions") is None


# ---------------------------------------------------------------------------
# 3. test_write_and_read
# ---------------------------------------------------------------------------

def test_write_and_read(ws):
    """Write state, read it back, content matches."""
    content = "# Project State\n\nAll good."
    ws.write("state", content)
    assert ws.read("state") == _hdr("state", content)


# ---------------------------------------------------------------------------
# 4. test_append_creates_then_appends
# ---------------------------------------------------------------------------

def test_append_creates_then_appends(ws):
    """Append to nonexistent creates file; second append adds content."""
    ws.append("decisions", "Decision 1\n")
    assert ws.read("decisions") == "Decision 1\n"

    ws.append("decisions", "Decision 2\n")
    assert ws.read("decisions") == "Decision 1\nDecision 2\n"


# ---------------------------------------------------------------------------
# 5. test_read_all_mixed
# ---------------------------------------------------------------------------

def test_read_all_mixed(ws):
    """Some files exist, some don't -- correct dict with None for missing.

    The roster is now the Layer-1 redesign set: state, decisions, lessons,
    index plus the two archives. "session_log"/"context" are gone; read_all
    spans all six keys (archives included).
    """
    ws.write("state", "state content")
    ws.write("lessons", "lessons content")

    result = ws.read_all()

    assert result["state"] == _hdr("state", "state content")
    assert result["lessons"] == _hdr("lessons", "lessons content")
    assert result["decisions"] is None
    # index replaces the retired "context" key.
    assert result["index"] is None
    # read_all now spans the archives too.
    assert result["decisions_archive"] is None
    assert result["lessons_archive"] is None
    assert result["state_archive"] is None
    assert set(result) == {
        "state", "decisions", "lessons",
        "index", "decisions_archive", "lessons_archive", "state_archive",
    }
    assert len(result) == 7


# ---------------------------------------------------------------------------
# 6. test_build_cold_resume_context_all_files
# ---------------------------------------------------------------------------

def test_build_cold_resume_context_all_files(ws):
    """All Layer-1 files exist -- assembled string with section headers in order.

    The Layer-1 redesign retired SESSION_LOG and renamed CONTEXT -> INDEX. The
    cold-resume order is now state, decisions, lessons, index (no session log).
    """
    ws.write("state", "In progress.")
    ws.write("decisions", "Chose X over Y.")
    ws.write("lessons", "Don't do Z.")
    ws.write("index", "src/foo.py — the foo module.")

    ctx = ws.build_cold_resume_context()

    # Check section order (index replaces the old context/session-log tail).
    state_pos = ctx.index("## Project State")
    decisions_pos = ctx.index("## Decisions")
    lessons_pos = ctx.index("## Lessons Learned")
    index_pos = ctx.index("## Project Index")

    assert state_pos < decisions_pos < lessons_pos < index_pos

    # Check content is included
    assert "In progress." in ctx
    assert "Chose X over Y." in ctx
    assert "Don't do Z." in ctx
    assert "src/foo.py — the foo module." in ctx

    # SESSION_LOG was removed — no such section should ever appear.
    assert "Session Log" not in ctx


# ---------------------------------------------------------------------------
# 7. test_build_cold_resume_context_minimal
# ---------------------------------------------------------------------------

def test_build_cold_resume_context_minimal(ws):
    """Only PROJECT_STATE.md exists -- just that section."""
    ws.write("state", "I am the project state.")

    ctx = ws.build_cold_resume_context()

    assert "## Project State" in ctx
    assert "I am the project state." in ctx
    # No other sections
    assert "## Decisions" not in ctx
    assert "## Lessons Learned" not in ctx


# ---------------------------------------------------------------------------
# 8. test_session_log_truncation — DELETED.
#
# SESSION_LOG.md and its last-3-sessions resume truncation are a genuinely
# removed feature in the Layer-1 memory redesign (no "session_log" key, no
# _truncate_session_log, build_cold_resume_context no longer special-cases it).
# Nothing about this test's intent survives the removal, so it is dropped
# rather than re-expressed. The roster/resume-no-session-log invariants are
# covered by tests/regression/test_layer1_memory_redesign.py.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# 9-12. The session-end routine is the memory editor (spec 089 §4.2)
# ---------------------------------------------------------------------------

def _mock_editor_provider(response_text):
    """A provider whose complete() answers once with plain text (no tools)."""
    provider = _mock_provider(response_text)
    provider.complete.return_value.raw_message = {"role": "assistant", "content": response_text}
    provider.complete.return_value.tool_calls = []
    return provider


@pytest.mark.asyncio
async def test_session_end_routine_applies_editor_choices_by_id(tmp_path):
    ws = WorkspaceFileManager(str(tmp_path))
    ws.write("state", "## Now\n- Old launch plan.\n- Current focus.\n")
    old_id = [ln for ln in ws.read("state").split("\n") if "<!--mem id:" in ln][0].split("id:")[1].split()[0]
    provider = _mock_editor_provider(json.dumps({
        "archive": [{"file": "PROJECT_STATE.md", "id": old_id, "pointer": "old launch plan"}],
    }))
    session = _mock_session([], session_id="sess_editor")

    out = await run_session_end_routine(
        session, provider, ws, session_uuid=session.session_uuid, force=True)

    assert out == "edited"
    state = ws.read("state")
    assert "- Old launch plan." not in state and "- Current focus." in state
    assert "[archived " in state and f"id:{old_id}] old launch plan → PROJECT_STATE_ARCHIVE.md" in state
    assert "- Old launch plan." in ws.read("state_archive")
    assert "session_log" not in FILE_NAMES


@pytest.mark.asyncio
async def test_session_end_routine_bad_json(tmp_path, caplog):
    """Unparseable replies: nothing is edited, the floor still runs, and the
    outcome says only the backstop ran."""
    ws = WorkspaceFileManager(str(tmp_path))
    ws.write("state", "original state")
    session = _mock_session([{"role": "user", "content": "Hello"}], session_id="sess_bad_json")
    provider = _mock_editor_provider("This is not JSON at all, sorry!")

    with caplog.at_level(logging.WARNING):
        out = await run_session_end_routine(
            session, provider, ws, session_uuid=session.session_uuid, force=True)

    assert out == "backstop_only"
    assert ws.read("state") == _hdr("state", "original state")
    assert ws.read("decisions") is None
    assert ws.read("lessons") is None
    assert "no usable JSON" in caplog.text


@pytest.mark.asyncio
async def test_session_end_routine_ignores_whole_file_replies(tmp_path):
    """The old contract (complete files back) no longer means anything."""
    ws = WorkspaceFileManager(str(tmp_path))
    ws.write("state", "# State\nDoing well.")
    provider = _mock_editor_provider(json.dumps({
        "project_state": "# State\nsomething else", "decisions": "## x\n", "lessons": "1. y\n",
    }))
    session = _mock_session([{"role": "user", "content": "Go"}], session_id="sess_whole")

    out = await run_session_end_routine(
        session, provider, ws, session_uuid=session.session_uuid, force=True)

    assert out == "no_change"
    assert ws.read("state") == _hdr("state", "# State\nDoing well.")
    assert ws.read("decisions") is None
    assert ws.read("lessons") is None


# ---------------------------------------------------------------------------
# Extra: test exists
# ---------------------------------------------------------------------------

def test_exists(ws):
    """exists() returns True only for files that are present."""
    assert ws.exists("state") is False
    ws.write("state", "content")
    assert ws.exists("state") is True


# ---------------------------------------------------------------------------
# Extra: test_session_end_uses_utility_provider
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_session_end_uses_utility_provider(tmp_path):
    """When utility_provider is given, the editor runs on it, not the main one."""
    ws = WorkspaceFileManager(str(tmp_path))
    ws.write("state", "state")

    session = _mock_session([{"role": "user", "content": "Hi"}], session_id="sess_util_prov")
    main_provider = _mock_editor_provider("should not be called")
    utility_provider = _mock_editor_provider("{}")

    await run_session_end_routine(
        session, main_provider, ws,
        utility_provider=utility_provider,
        session_uuid=session.session_uuid,
        force=True,
    )

    utility_provider.complete.assert_called_once()
    main_provider.complete.assert_not_called()
    assert ws.read("state") == _hdr("state", "state")


# ---------------------------------------------------------------------------
# Extra: test invalid file_key
# ---------------------------------------------------------------------------

def test_read_invalid_key(ws):
    """Invalid file_key raises ValueError."""
    with pytest.raises(ValueError, match="Unknown file_key"):
        ws.read("nonexistent_key")


def test_write_invalid_key(ws):
    """Invalid file_key raises ValueError."""
    with pytest.raises(ValueError, match="Unknown file_key"):
        ws.write("bad_key", "content")


def test_build_cold_resume_empty_workspace(ws):
    """No files exist -- returns empty string."""
    ctx = ws.build_cold_resume_context()
    assert ctx == ""
