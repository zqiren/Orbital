# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Step 3 — a session's JSONL is materialized on the first message, not on
Session.new(). A session is a file on disk; until the user (or dispatcher)
sends a first message there is nothing on disk yet. The session_start meta
record is deferred and flushed as the first physical line on the first write.

Also: list_sessions / _disk_session_entries must skip meta-only logs (a file
with identity metadata but no actual conversation is not a real session yet).
"""

from __future__ import annotations

import json
import os
from unittest.mock import MagicMock

from agent_os.agent.session import Session
from agent_os.agent.project_paths import ProjectPaths
from agent_os.daemon_v2.agent_manager import AgentManager


def _session_path(workspace, uuid):
    return os.path.join(ProjectPaths(workspace).sessions_dir, f"{uuid}.jsonl")


def test_session_new_does_not_create_file(tmp_path):
    ws = str(tmp_path / "ws")
    s = Session.new("proj_abcd1234", ws, session_id="default")
    assert not os.path.exists(_session_path(ws, "proj_abcd1234")), (
        "Session.new must NOT write the JSONL — creation is deferred to the "
        "first message"
    )


def test_first_append_materializes_file_meta_first(tmp_path):
    ws = str(tmp_path / "ws")
    s = Session.new("proj_abcd1234", ws, session_id="default")
    s.append({"role": "user", "content": "hi"})

    path = _session_path(ws, "proj_abcd1234")
    assert os.path.exists(path), "first append must create the file"
    lines = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    assert lines[0]["role"] == "meta", "deferred session_start meta is flushed first"
    assert lines[0]["event"] == "session_start"
    assert lines[1]["role"] == "user"
    assert lines[1]["content"] == "hi"


def test_append_meta_alone_flushes_pending_then_event(tmp_path):
    ws = str(tmp_path / "ws")
    s = Session.new("proj_abcd1234", ws, session_id="default")
    # A meta event (e.g. model_swap) before any user message must still flush
    # the deferred session_start first.
    s.append_meta("model_swap", to="other-model")
    path = _session_path(ws, "proj_abcd1234")
    lines = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    assert [r["event"] for r in lines] == ["session_start", "model_swap"]


def _make_manager(tmp_path):
    mgr = AgentManager(
        project_store=MagicMock(), ws_manager=MagicMock(),
        sub_agent_manager=MagicMock(), activity_translator=MagicMock(),
        process_manager=MagicMock(), platform_provider=None,
        registry=MagicMock(), setup_engine=MagicMock(),
        settings_store=None, credential_store=None,
    )
    mgr._state_file = tmp_path / "daemon-state.json"
    return mgr


def test_disk_entries_skip_meta_only_log(tmp_path):
    ws = tmp_path / "ws"
    sessions = ws / "orbital" / "sessions"
    sessions.mkdir(parents=True)
    # Meta-only file: session_start was written but no message ever sent.
    (sessions / "proj_metaonly.jsonl").write_text(
        json.dumps({"role": "meta", "event": "session_start",
                    "session_id": "default", "session_uuid": "proj_metaonly",
                    "timestamp": "2026-05-25T08:00:00+00:00"}) + "\n",
        encoding="utf-8",
    )
    # Real file: has an actual user message.
    (sessions / "proj_real0000.jsonl").write_text(
        json.dumps({"role": "meta", "event": "session_start",
                    "session_id": "default", "session_uuid": "proj_real0000",
                    "timestamp": "2026-05-25T08:00:00+00:00"}) + "\n" +
        json.dumps({"role": "user", "content": "hi", "session_id": "default",
                    "timestamp": "2026-05-25T08:01:00+00:00"}) + "\n",
        encoding="utf-8",
    )
    mgr = _make_manager(tmp_path)
    mgr._project_store.get_project.return_value = {"workspace": str(ws)}

    out = mgr.list_sessions("proj")
    uuids = {s["session_uuid"] for s in out}
    assert "proj_real0000" in uuids
    assert "proj_metaonly" not in uuids, "meta-only logs are not real sessions"
