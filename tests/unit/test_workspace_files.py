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

from agent_os.agent.workspace_files import (
    WORKSPACE_DIR,
    FILE_NAMES,
    WorkspaceFileManager,
    run_session_end_routine,
    _build_session_summary,
    _parse_session_end_response,
)


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
    return os.path.join(str(tmp_path), WORKSPACE_DIR)


def _mock_session(messages=None, session_id="sess_test123"):
    """Build a mock session with get_messages() returning the given list."""
    session = MagicMock()
    session.session_id = session_id
    session.get_messages.return_value = messages or []
    return session


def _mock_provider(response_text):
    """Build a mock LLM provider whose complete() returns an object with .text."""
    provider = AsyncMock()
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
    assert ws.read("agent") is None


# ---------------------------------------------------------------------------
# 3. test_write_and_read
# ---------------------------------------------------------------------------

def test_write_and_read(ws):
    """Write state, read it back, content matches."""
    content = "# Project State\n\nAll good."
    ws.write("state", content)
    assert ws.read("state") == content


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
    """Some files exist, some don't -- correct dict with None for missing."""
    ws.write("state", "state content")
    ws.write("agent", "agent content")

    result = ws.read_all()

    assert result["state"] == "state content"
    assert result["agent"] == "agent content"
    assert result["decisions"] is None
    assert result["session_log"] is None
    assert result["lessons"] is None
    assert result["context"] is None
    assert len(result) == 6


# ---------------------------------------------------------------------------
# 6. test_build_cold_resume_context_all_files
# ---------------------------------------------------------------------------

def test_build_cold_resume_context_all_files(ws):
    """All 6 files exist -- assembled string with section headers in correct order."""
    ws.write("agent", "Do the thing.")
    ws.write("state", "In progress.")
    ws.write("decisions", "Chose X over Y.")
    ws.write("lessons", "Don't do Z.")
    ws.write("context", "Person: Alice.")
    ws.write("session_log", "## Session s1 -- 2026-02-15\n- Did stuff")

    ctx = ws.build_cold_resume_context()

    # Check section order
    agent_pos = ctx.index("## Agent Directive")
    state_pos = ctx.index("## Project State")
    decisions_pos = ctx.index("## Decisions")
    lessons_pos = ctx.index("## Lessons Learned")
    context_pos = ctx.index("## External Context")
    log_pos = ctx.index("## Session Log (Recent)")

    assert agent_pos < state_pos < decisions_pos < lessons_pos < context_pos < log_pos

    # Check content is included
    assert "Do the thing." in ctx
    assert "In progress." in ctx
    assert "Chose X over Y." in ctx
    assert "Don't do Z." in ctx
    assert "Person: Alice." in ctx
    assert "Did stuff" in ctx


# ---------------------------------------------------------------------------
# 7. test_build_cold_resume_context_minimal
# ---------------------------------------------------------------------------

def test_build_cold_resume_context_minimal(ws):
    """Only AGENT.md exists -- just that section."""
    ws.write("agent", "I am an agent.")

    ctx = ws.build_cold_resume_context()

    assert "## Agent Directive" in ctx
    assert "I am an agent." in ctx
    # No other sections
    assert "## Project State" not in ctx
    assert "## Decisions" not in ctx


# ---------------------------------------------------------------------------
# 8. test_session_log_truncation
# ---------------------------------------------------------------------------

def test_session_log_truncation(ws):
    """SESSION_LOG with many sessions -- only last 3 included in resume context."""
    sessions = []
    for i in range(10):
        sessions.append(f"## Session sess_{i} -- 2026-02-{10+i}\n- Did thing {i}")
    ws.write("session_log", "\n\n".join(sessions))

    ctx = ws.build_cold_resume_context()

    # Only last 3 sessions should be present
    assert "sess_7" in ctx
    assert "sess_8" in ctx
    assert "sess_9" in ctx
    # Earlier sessions should not be present
    assert "sess_0" not in ctx
    assert "sess_6" not in ctx


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
    """Mock LLM returns valid JSON -- correct files written."""
    ws = WorkspaceFileManager(str(tmp_path))

    llm_response = json.dumps({
        "project_state": "# Project State\nEverything is great.",
        "decisions": "## 2026-02-15: Chose A\n**Chose:** A\n**Reason:** Better.",
        "session_log_entry": "## Session sess_x -- 2026-02-15\n- Completed: stuff",
        "lessons": "## Lesson 1\n**Problem:** Bad thing.\n**Fix:** Good thing.",
        "context": "## People\n- Alice: dev lead",
    })

    session = _mock_session([
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there."},
    ], session_id="sess_writes_files")
    provider = _mock_provider(llm_response)

    await run_session_end_routine(session, provider, ws, session_id=session.session_id)

    # state is written (overwritten)
    assert ws.read("state") == "# Project State\nEverything is great."
    # decisions appended
    assert "Chose A" in ws.read("decisions")
    # session_log appended
    assert "sess_x" in ws.read("session_log")
    # lessons appended
    assert "Lesson 1" in ws.read("lessons")
    # context appended
    assert "Alice" in ws.read("context")


# ---------------------------------------------------------------------------
# 11. test_session_end_routine_bad_json
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_session_end_routine_bad_json(tmp_path, caplog):
    """LLM returns garbage -- no files modified, warning logged."""
    ws = WorkspaceFileManager(str(tmp_path))
    # Pre-write a state file to ensure it's not modified
    ws.write("state", "original state")

    session = _mock_session([{"role": "user", "content": "Hello"}], session_id="sess_bad_json")
    provider = _mock_provider("This is not JSON at all, sorry!")

    with caplog.at_level(logging.WARNING):
        await run_session_end_routine(session, provider, ws, session_id=session.session_id)

    # state should be unchanged
    assert ws.read("state") == "original state"
    # No other files created
    assert ws.read("decisions") is None
    assert ws.read("session_log") is None
    # Warning should be logged
    assert "failed to parse LLM response" in caplog.text


# ---------------------------------------------------------------------------
# 12. test_session_end_routine_empty_optionals
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_session_end_routine_empty_optionals(tmp_path):
    """LLM returns empty decisions/lessons/context -- only state + session_log written."""
    ws = WorkspaceFileManager(str(tmp_path))

    llm_response = json.dumps({
        "project_state": "# State\nDoing well.",
        "decisions": "",
        "session_log_entry": "## Session sess_y -- 2026-02-15\n- Done things",
        "lessons": "  ",
        "context": "",
    })

    session = _mock_session([{"role": "user", "content": "Go"}], session_id="sess_empty_opt")
    provider = _mock_provider(llm_response)

    await run_session_end_routine(session, provider, ws, session_id=session.session_id)

    # state written
    assert ws.read("state") == "# State\nDoing well."
    # session_log written
    assert "sess_y" in ws.read("session_log")
    # Empty optionals should NOT create files
    assert ws.read("decisions") is None
    assert ws.read("lessons") is None
    assert ws.read("context") is None


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
        "session_log_entry": "## Session s -- date\n- done",
        "lessons": "",
        "context": "",
    })

    session = _mock_session([{"role": "user", "content": "Hi"}], session_id="sess_util_prov")
    main_provider = _mock_provider("should not be called")
    utility_provider = _mock_provider(llm_response)

    await run_session_end_routine(
        session, main_provider, ws,
        utility_provider=utility_provider,
        session_id=session.session_id,
    )

    # utility_provider should have been called, not main_provider
    utility_provider.complete.assert_called_once()
    main_provider.complete.assert_not_called()

    assert ws.read("state") == "state"


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
