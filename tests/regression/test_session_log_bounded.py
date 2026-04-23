# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Regression: SESSION_LOG.md is bounded at write time, not only read time.

Prior to Task 4, `run_session_end_routine` appended a new entry to
SESSION_LOG.md unconditionally and relied on `build_cold_resume_context`
to truncate at read time. The on-disk file grew monotonically (production
instances reached 15+ entries with duplicates).

This module locks in the new contract:

  - After append, the file is re-read and passed through
    `_truncate_session_log(content, _SESSION_LOG_WRITE_CAP=10)`. If the
    result differs, it is written back atomically.
  - `build_session_end_prompt` includes the LAST 3 entries (the read cap)
    between CONTEXT and `--- THIS SESSION ---`, so the utility LLM has
    cross-session context when writing project_state and lessons.
  - A malformed SESSION_LOG (no `## Session` markers) is preserved —
    `_truncate_session_log` returns unchanged content when parse fails,
    so the append plus unchanged-original is the final file.

Idempotency note: Task 2's guard short-circuits repeat invocations with
the same `session_id`. Tests that call `run_session_end_routine` multiple
times on the same manager either use DISTINCT session_ids OR clear the
completion set via the autouse fixture below (mirrors the pattern from
`tests/regression/test_session_end_idempotency.py`).
"""

import json
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

def _mock_session(session_id="sess_bound"):
    session = MagicMock()
    session.session_id = session_id
    session.get_messages.return_value = [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello"},
    ]
    return session


def _mock_provider(response_text):
    provider = AsyncMock()
    resp = MagicMock()
    resp.text = response_text
    provider.complete.return_value = resp
    return provider


def _llm_response(tag):
    """Distinctive well-formed JSON response with a unique session_log_entry
    tagged by `tag` so on-disk entries are individually identifiable."""
    return json.dumps({
        "project_state": f"# State\nstate-{tag}",
        # Empty => preserve existing; we do not want these writing here.
        "decisions": "",
        "session_log_entry": (
            f"## Session {tag} -- 2026-04-24\n- Completed: work-{tag}"
        ),
        "lessons": "",
        "context": "",
    })


@pytest.fixture(autouse=True)
def _reset_completion_set():
    """Reset the module-level idempotency set between tests.

    Mirrors `tests/regression/test_session_end_idempotency.py`.
    """
    wsf_module._completed_session_ends.clear()
    yield
    wsf_module._completed_session_ends.clear()


# ---------------------------------------------------------------------------
# Test 1: 15 runs => exactly 10 ## Session headers on disk
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_session_log_capped_at_write_time(tmp_path):
    """Run the session-end routine 15 times with distinct session_ids and a
    mocked utility LLM that returns one entry per call. The on-disk
    SESSION_LOG.md must contain exactly 10 `## Session ` headers (the
    `_SESSION_LOG_WRITE_CAP`), not 15."""
    ws = WorkspaceFileManager(str(tmp_path))

    for i in range(1, 16):
        sid = f"s_{i:02d}"
        session = _mock_session(session_id=sid)
        provider = _mock_provider(_llm_response(sid))
        await run_session_end_routine(session, provider, ws, session_id=sid)

    on_disk = ws.read("session_log") or ""
    header_count = on_disk.count("## Session ")
    assert header_count == 10, (
        f"expected exactly 10 ## Session headers after 15 runs; got "
        f"{header_count}. File contents:\n{on_disk}"
    )


# ---------------------------------------------------------------------------
# Test 2: Newest-10 entries are retained, oldest dropped
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_session_log_entries_are_newest_10(tmp_path):
    """Seed SESSION_LOG.md with Session-1 ... Session-15, run one more
    through the routine, and assert the on-disk file keeps Session-7
    through Session-16 (the 10 newest)."""
    ws = WorkspaceFileManager(str(tmp_path))

    # Seed 15 entries.
    seeded_entries = [
        f"## Session Session-{i} -- 2026-04-{i:02d}\n- Completed: seed-{i}"
        for i in range(1, 16)
    ]
    ws.write("session_log", "\n\n".join(seeded_entries))

    # Run a 16th via the routine.
    session = _mock_session(session_id="s_16")
    provider = _mock_provider(_llm_response("Session-16"))
    await run_session_end_routine(session, provider, ws, session_id="s_16")

    on_disk = ws.read("session_log") or ""

    # Oldest 6 (Session-1 .. Session-6) should have been dropped.
    for i in range(1, 7):
        assert f"Session-{i} " not in on_disk, (
            f"expected Session-{i} to be truncated from disk; found it "
            f"in:\n{on_disk}"
        )

    # Newest 10 (Session-7 .. Session-16) should remain.
    for i in range(7, 17):
        assert f"Session-{i} " in on_disk, (
            f"expected Session-{i} to be present on disk; missing from:\n"
            f"{on_disk}"
        )

    # And exactly 10 header markers total.
    assert on_disk.count("## Session ") == 10


# ---------------------------------------------------------------------------
# Test 3: Malformed file (no markers) preserved; new entry appended
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_session_log_malformed_file_preserved(tmp_path):
    """If SESSION_LOG has no `## Session` markers (plain paragraph text),
    `_truncate_session_log` returns content unchanged. The post-append
    state (preamble + new entry) must therefore survive to disk without
    data loss."""
    ws = WorkspaceFileManager(str(tmp_path))

    malformed_preamble = (
        "This is some freeform preamble text with no session markers.\n"
        "A curious human may have edited this file by hand.\n"
    )
    ws.write("session_log", malformed_preamble)

    session = _mock_session(session_id="s_malformed")
    provider = _mock_provider(_llm_response("malformed"))
    await run_session_end_routine(
        session, provider, ws, session_id="s_malformed",
    )

    on_disk = ws.read("session_log") or ""

    # Original malformed preamble survived.
    assert "freeform preamble text" in on_disk, (
        f"malformed preamble was lost; on-disk content:\n{on_disk}"
    )
    assert "A curious human" in on_disk

    # New session entry was appended.
    assert "## Session malformed" in on_disk, (
        f"new session entry missing; on-disk content:\n{on_disk}"
    )


# ---------------------------------------------------------------------------
# Test 4: Prompt includes only the last 3 SESSION_LOG entries
# ---------------------------------------------------------------------------

def test_prompt_includes_last_3_session_log_entries(tmp_path):
    """Seed SESSION_LOG.md with 5 distinct entries. `build_session_end_prompt`
    must include the last 3 verbatim in the existing-files block and must
    NOT include the earlier 2."""
    ws = WorkspaceFileManager(str(tmp_path))

    entries = [
        f"## Session tag_{i} -- 2026-04-{10 + i:02d}\n- marker-line-{i}"
        for i in range(1, 6)
    ]
    ws.write("session_log", "\n\n".join(entries))

    prompt = ws.build_session_end_prompt({
        "message_count": 1,
        "tool_calls_count": 0,
        "files_modified": [],
        "recent_messages": [],
    })

    # The per-entry section header itself appears in the prompt.
    assert "SESSION_LOG.md (last 3 entries):" in prompt

    # Last 3 entries present.
    for i in (3, 4, 5):
        assert f"tag_{i}" in prompt, (
            f"expected tag_{i} in prompt; prompt was:\n{prompt}"
        )
        assert f"marker-line-{i}" in prompt

    # Earlier 2 NOT present.
    for i in (1, 2):
        assert f"tag_{i}" not in prompt, (
            f"did not expect tag_{i} in prompt; prompt was:\n{prompt}"
        )
        assert f"marker-line-{i}" not in prompt
