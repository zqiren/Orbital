# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Unit tests for OCC protection of the Layer-1 files during a memory pass.

Verifies the Optimistic Concurrency Control that protects PROJECT_STATE.md,
DECISIONS.md, LESSONS.md and INDEX.md from being clobbered when the agent, the
user, or any other writer edits one of them while the memory editor's model
calls are in flight (spec 089 §4.2: OCC per file).

Pattern under test:
  1. capture st_mtime_ns of each Layer-1 file BEFORE the editor's first call
  2. the editor returns choices by id
  3. re-stat just before writing each file (no await in between)
  4. unchanged: append the moved entries to the archive, then write the file
  5. changed: skip that file (its archive untouched) with a structured
     WARNING carrying project_id, file_path, baseline_mtime, observed_mtime,
     cache_thrash_telemetry=True — the other files still apply
"""

import json
import logging
import os
from unittest.mock import AsyncMock, MagicMock

import pytest

from tests.testutils import streamable

from agent_os.agent import memory_editor
from agent_os.agent import workspace_files as wsf_module
from agent_os.agent.workspace_files import (
    WorkspaceFileManager,
    run_session_end_routine,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_session(session_id="sess_occ"):
    session = MagicMock()
    session.session_id = session_id
    session.session_uuid = session_id
    session.get_messages.return_value = []
    return session


def _resp(payload: dict):
    resp = MagicMock()
    resp.text = json.dumps(payload)
    resp.raw_message = {"role": "assistant", "content": resp.text}
    resp.tool_calls = []
    return resp


def _seed(ws) -> dict:
    """Four Layer-1 files, each with one entry the editor will archive."""
    ws.write("state", "## Now\n- state entry to archive\n- state entry to keep\n")
    ws.write("decisions", (
        "## Keep A <!--mem id:keep-a created:2026-08-01 touched:2026-08-01-->\nx\n\n"
        "## Keep B <!--mem id:keep-b created:2026-08-01 touched:2026-08-01-->\nx\n\n"
        "## Archive me <!--mem id:dec-go created:2026-08-01 touched:2026-08-01-->\nold\n\n"
    ))
    ws.write("lessons", (
        "1. Keep this. <!--mem id:les-keep created:2026-08-01 touched:2026-08-01-->\n"
        "2. Archive this. <!--mem id:les-go created:2026-08-01 touched:2026-08-01-->\n"
    ))
    ws.write("index", "# INDEX\n- a.md — a\n")
    state_id = ws.read("state").split("id:")[1].split()[0]
    return {"archive": [
        {"file": "PROJECT_STATE.md", "id": state_id, "pointer": "p1"},
        {"file": "DECISIONS.md", "id": "dec-go", "pointer": "p2"},
        {"file": "LESSONS.md", "id": "les-go", "pointer": "p3"},
    ], "index": "# INDEX\n- b.md — b\n"}


def _bump_mtime_by(path: str, delta_seconds: float = 5.0) -> int:
    st = os.stat(path)
    os.utime(path, (st.st_atime + delta_seconds, st.st_mtime + delta_seconds))
    return os.stat(path).st_mtime_ns


def _provider_that(side_effect=None, payload=None):
    provider = streamable(AsyncMock())
    if side_effect is not None:
        provider.complete.side_effect = side_effect
    else:
        provider.complete.return_value = _resp(payload)
    return provider


@pytest.fixture(autouse=True)
def _reset_module_state():
    wsf_module._completed_session_ends.clear()
    memory_editor._RUNNING.clear()
    yield
    wsf_module._completed_session_ends.clear()


async def _run(ws, provider, sid, project_id):
    return await run_session_end_routine(
        _mock_session(sid), provider, ws,
        session_uuid=sid, project_id=project_id, force=True,
    )


# ---------------------------------------------------------------------------
# 1. clean baseline → every chosen edit lands
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_clean_baseline_writes_all_layer1_files(tmp_path):
    ws = WorkspaceFileManager(str(tmp_path))
    choices = _seed(ws)
    out = await _run(ws, _provider_that(payload=choices), "sess_clean", "proj_clean")
    assert out == "edited"
    assert "state entry to archive" not in ws.read("state").split("[archived")[0]
    assert "## Archive me" not in ws.read("decisions")
    assert "Archive this." not in ws.read("lessons")
    assert "- b.md — b" in ws.read("index")


# ---------------------------------------------------------------------------
# 2. an edit during the model call → that file is skipped, content survives
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_user_edit_during_llm_aborts_state_write(tmp_path, caplog):
    ws = WorkspaceFileManager(str(tmp_path))
    choices = _seed(ws)
    state_path = ws._file_path("state")
    user_edited_bytes = ws.read("state") + "## USER ADDED\nhand-written note\n"

    async def _llm_side_effect(*args, **kwargs):
        with open(state_path, "w", encoding="utf-8") as f:
            f.write(user_edited_bytes)
        _bump_mtime_by(state_path, delta_seconds=2.0)
        return _resp(choices)

    with caplog.at_level(logging.WARNING, logger="agent_os.agent.workspace_files"):
        await _run(ws, _provider_that(side_effect=_llm_side_effect), "sess_edit_state", "proj_edit")

    assert ws.read("state") == user_edited_bytes
    assert not ws.exists("state_archive"), "a skipped file must not be half-moved"

    matching = [r for r in caplog.records if "OCC abort" in r.message and "state" in r.message]
    assert matching, [r.message for r in caplog.records]
    rec = matching[0]
    assert getattr(rec, "project_id", None) == "proj_edit"
    assert getattr(rec, "file_path", None) == state_path
    assert getattr(rec, "cache_thrash_telemetry", None) is True
    assert getattr(rec, "baseline_mtime", None) != getattr(rec, "observed_mtime", None)


# ---------------------------------------------------------------------------
# 3. one file aborts, the others still write
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_one_file_aborts_others_still_write(tmp_path, caplog):
    ws = WorkspaceFileManager(str(tmp_path))
    choices = _seed(ws)
    lessons_path = ws._file_path("lessons")
    user_lessons = ws.read("lessons") + "3. user-additional-lesson-during-llm\n"

    async def _llm_side_effect(*args, **kwargs):
        with open(lessons_path, "w", encoding="utf-8") as f:
            f.write(user_lessons)
        _bump_mtime_by(lessons_path, delta_seconds=2.0)
        return _resp(choices)

    with caplog.at_level(logging.WARNING, logger="agent_os.agent.workspace_files"):
        await _run(ws, _provider_that(side_effect=_llm_side_effect), "sess_multi", "proj_multi")

    assert ws.read("lessons") == user_lessons
    assert "## Archive me" not in ws.read("decisions")
    assert "- b.md — b" in ws.read("index")
    aborts = [r for r in caplog.records if "OCC abort" in r.message]
    assert len(aborts) == 1
    assert "lessons" in aborts[0].message


# ---------------------------------------------------------------------------
# 4. a file created during the model call is protected too
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_user_creates_file_during_llm_aborts_index_write(tmp_path, caplog):
    ws = WorkspaceFileManager(str(tmp_path))
    ws.write("state", "## Now\n- a\n")
    index_path = ws._file_path("index")
    user_content = "user-wrote-this-while-the-editor-was-running\n"

    async def _llm_side_effect(*args, **kwargs):
        with open(index_path, "w", encoding="utf-8") as f:
            f.write(user_content)
        return _resp({"index": "# INDEX\n- editor.md — editor\n"})

    with caplog.at_level(logging.WARNING, logger="agent_os.agent.workspace_files"):
        await _run(ws, _provider_that(side_effect=_llm_side_effect), "sess_create_race", "proj_create")

    assert ws.read("index") == user_content
    aborts = [r for r in caplog.records if "OCC abort" in r.message and "index" in r.message]
    assert aborts


# ---------------------------------------------------------------------------
# 5. structured fields on the abort record
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_log_includes_all_required_structured_fields(tmp_path, caplog):
    ws = WorkspaceFileManager(str(tmp_path))
    choices = _seed(ws)
    decisions_path = ws._file_path("decisions")

    async def _llm_side_effect(*args, **kwargs):
        _bump_mtime_by(decisions_path, delta_seconds=2.0)
        return _resp(choices)

    with caplog.at_level(logging.WARNING, logger="agent_os.agent.workspace_files"):
        await _run(ws, _provider_that(side_effect=_llm_side_effect), "sess_struct", "proj_struct_log")

    matching = [r for r in caplog.records if "OCC abort" in r.message and "decisions" in r.message]
    assert matching
    rec = matching[0]
    assert getattr(rec, "project_id", None) == "proj_struct_log"
    assert getattr(rec, "file_path", None) == decisions_path
    assert isinstance(getattr(rec, "baseline_mtime", None), int)
    assert isinstance(getattr(rec, "observed_mtime", None), int)
    assert getattr(rec, "cache_thrash_telemetry", None) is True
    assert rec.levelno == logging.WARNING


# ---------------------------------------------------------------------------
# Truth in the abort message (orbital-marketing, 2026-07-28/29).
#
# Every OCC abort blamed "user mid-edit detected". In all four observed cases
# the observed_mtime was the timestamp of a CONCURRENT CONSOLIDATION PASS's own
# write — no user touched anything. The distinction matters: a real mid-edit is
# the feature working, a self-collision is a bug, and they were indistinguishable
# in the log.
# ---------------------------------------------------------------------------

def test_occ_abort_names_a_concurrent_pass_when_the_daemon_made_the_write(tmp_path, caplog):
    wf = WorkspaceFileManager(str(tmp_path))
    wf.write("decisions", "## 2026-07-28: A\n")
    stale_baseline = 1  # anything that is not the current mtime

    with caplog.at_level(logging.WARNING):
        ok = wsf_module._occ_write_metadata(
            wf, "decisions", "## 2026-07-28: B\n", stale_baseline,
            project_id="proj_x",
        )

    assert ok is False
    msg = caplog.text
    assert "concurrent consolidation pass" in msg.lower(), msg
    assert "user mid-edit" not in msg.lower(), (
        "a daemon-authored write must not be reported as a user edit"
    )


def test_occ_abort_still_names_the_user_for_a_real_outside_edit(tmp_path, caplog):
    wf = WorkspaceFileManager(str(tmp_path))
    wf.write("decisions", "## 2026-07-28: A\n")
    # A writer that is NOT the daemon's write path touches the file. Bump the
    # mtime explicitly: on NTFS the append can land on the same timestamp tick
    # as the daemon write above, which reads as a self-collision.
    path = wf._file_path("decisions")
    with open(path, "a", encoding="utf-8") as f:
        f.write("## hand edit\n")
    st = os.stat(path)
    os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns + 5_000_000))

    with caplog.at_level(logging.WARNING):
        ok = wsf_module._occ_write_metadata(
            wf, "decisions", "## 2026-07-28: B\n", 1, project_id="proj_x",
        )

    assert ok is False
    assert "user mid-edit" in caplog.text.lower(), caplog.text
