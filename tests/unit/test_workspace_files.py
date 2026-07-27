# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""
Unit tests for WorkspaceFileManager and session-end routine.

Covers:
  - File CRUD operations (ensure_dir, read, write, append, read_all, exists)
  - Cold resume context assembly (all files, minimal, truncation)
  - Session summary extraction (_build_session_summary)
  - Session-end routine (happy path, bad JSON, empty optionals)
"""

import asyncio
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
    _build_session_summary,
    _parse_session_end_response,
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
# 9. test_build_session_summary
# ---------------------------------------------------------------------------

def test_build_session_summary():
    """Mock session with messages including tool_calls with write/edit -- correct counts."""
    messages = [
        {"role": "user", "content": "Hello"},
        {
            "role": "assistant",
            "content": "I'll write a file.",
            "tool_calls": [
                {
                    "id": "tc_1",
                    "function": {
                        "name": "write",
                        "arguments": json.dumps({"file_path": "/tmp/foo.py"}),
                    },
                },
                {
                    "id": "tc_2",
                    "function": {
                        "name": "edit",
                        "arguments": json.dumps({"file_path": "/tmp/bar.py"}),
                    },
                },
            ],
        },
        {"role": "tool", "content": "OK", "tool_call_id": "tc_1"},
        {"role": "tool", "content": "OK", "tool_call_id": "tc_2"},
        {"role": "user", "content": "Thanks"},
        {
            "role": "assistant",
            "content": "Done.",
            "tool_calls": [
                {
                    "id": "tc_3",
                    "function": {
                        "name": "read",
                        "arguments": json.dumps({"file_path": "/tmp/baz.py"}),
                    },
                },
            ],
        },
        {"role": "tool", "content": "file content", "tool_call_id": "tc_3"},
    ]
    session = _mock_session(messages, session_id="sess_abc")

    summary = _build_session_summary(session)

    assert summary["session_id"] == "sess_abc"
    assert summary["message_count"] == 7
    assert summary["tool_calls_count"] == 3  # tc_1, tc_2, tc_3
    assert sorted(summary["files_modified"]) == ["/tmp/bar.py", "/tmp/foo.py"]
    assert len(summary["recent_messages"]) > 0


# ---------------------------------------------------------------------------
# 10. test_session_end_routine_writes_files
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_session_end_routine_writes_files(tmp_path):
    """Mock LLM returns valid JSON -- correct Layer-1 files written.

    New session-end contract: JSON keys are project_state, decisions, lessons,
    index (no session_log_entry, no "context"). STATE/INDEX are overwritten;
    DECISIONS/LESSONS are stamped with metadata before write, so assertions on
    those files check substrings, not exact bytes.
    """
    ws = WorkspaceFileManager(str(tmp_path))

    llm_response = json.dumps({
        "project_state": "# Project State\nEverything is great.",
        "decisions": "## 2026-02-15: Chose A\n**Chose:** A\n**Reason:** Better.\n",
        "lessons": "1. **Bad thing.** Do the good thing instead.\n",
        "index": "## People\n- Alice: dev lead\n",
    })

    session = _mock_session([
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there."},
    ], session_id="sess_writes_files")
    provider = _mock_provider(llm_response)

    # Fresh workspace has no cleanup marker -> delta gate passes -> LLM runs.
    await run_session_end_routine(session, provider, ws, session_uuid=session.session_uuid)

    # state is written verbatim (overwrite scratchpad)
    assert ws.read("state") == _hdr("state", "# Project State\nEverything is great.")
    # decisions written (stamped) — title/body survive the metadata stamp
    decisions = ws.read("decisions")
    assert "Chose A" in decisions
    assert "<!--mem id:" in decisions  # stamped by the persist path
    # lessons written (stamped, renumbered contiguously)
    assert "Do the good thing instead." in ws.read("lessons")
    # index written verbatim (navigation map, overwrite) — replaces old "context"
    assert "Alice" in ws.read("index")
    # SESSION_LOG is gone — its key no longer exists on the roster.
    assert "session_log" not in FILE_NAMES


# ---------------------------------------------------------------------------
# 11. test_session_end_routine_bad_json
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_session_end_routine_bad_json(tmp_path, caplog):
    """LLM returns garbage -- no LLM-derived writes, parse warning logged.

    On unparseable JSON the routine logs a parse-failure warning and falls
    through to the deterministic backstop. With a tiny pre-written state the
    backstop trims nothing, so no Layer-1 file is touched. (SESSION_LOG no
    longer exists, so there's nothing for it to skip writing.)
    """
    ws = WorkspaceFileManager(str(tmp_path))
    # Pre-write a state file (well under budget) to ensure it's not modified.
    ws.write("state", "original state")

    session = _mock_session([{"role": "user", "content": "Hello"}], session_id="sess_bad_json")
    provider = _mock_provider("This is not JSON at all, sorry!")

    with caplog.at_level(logging.WARNING):
        await run_session_end_routine(session, provider, ws, session_uuid=session.session_uuid)

    # state should be unchanged (no LLM-derived overwrite happened)
    assert ws.read("state") == _hdr("state", "original state")
    # No durable files created from the garbage response.
    assert ws.read("decisions") is None
    assert ws.read("lessons") is None
    # Parse-failure warning should be logged.
    assert "JSON parse failed" in caplog.text


# ---------------------------------------------------------------------------
# 12. test_session_end_routine_empty_optionals
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_session_end_routine_empty_optionals(tmp_path):
    """LLM returns empty decisions/lessons/index -- only state written.

    A "" (or whitespace-only) field means "preserve the existing file
    unchanged" in the new contract, so empty optionals must not create files.
    SESSION_LOG no longer exists, so state is the only file that gets written.
    """
    ws = WorkspaceFileManager(str(tmp_path))

    llm_response = json.dumps({
        "project_state": "# State\nDoing well.",
        "decisions": "",
        "lessons": "  ",
        "index": "",
    })

    session = _mock_session([{"role": "user", "content": "Go"}], session_id="sess_empty_opt")
    provider = _mock_provider(llm_response)

    await run_session_end_routine(session, provider, ws, session_uuid=session.session_uuid)

    # state written
    assert ws.read("state") == _hdr("state", "# State\nDoing well.")
    # Empty optionals should NOT create files
    assert ws.read("decisions") is None
    assert ws.read("lessons") is None
    assert ws.read("index") is None


# ---------------------------------------------------------------------------
# Extra: test_parse_session_end_response edge cases
# ---------------------------------------------------------------------------

def test_parse_response_with_markdown_fences():
    """JSON wrapped in ```json ... ``` fences is correctly parsed."""
    text = '```json\n{"project_state": "ok"}\n```'
    result = _parse_session_end_response(text)
    assert result == {"project_state": "ok"}


def test_parse_response_none_input():
    """None input returns None."""
    assert _parse_session_end_response(None) is None


def test_parse_response_empty_string():
    """Empty string returns None."""
    assert _parse_session_end_response("") is None


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
    """When utility_provider is given, it is used instead of the main provider."""
    ws = WorkspaceFileManager(str(tmp_path))

    llm_response = json.dumps({
        "project_state": "state",
        "decisions": "",
        "lessons": "",
        "index": "",
    })

    session = _mock_session([{"role": "user", "content": "Hi"}], session_id="sess_util_prov")
    main_provider = _mock_provider("should not be called")
    utility_provider = _mock_provider(llm_response)

    await run_session_end_routine(
        session, main_provider, ws,
        utility_provider=utility_provider,
        session_uuid=session.session_uuid,
    )

    # utility_provider should have been called, not main_provider
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
