# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Phase 4 bounded add: every session carries an ``origin`` tag (chat | queue).

- chat: user-initiated (chat / inject). The project's persistent chat thread is
  resolved among chat-origin sessions (D1's inject(None) funnel).
- queue: minted by the QueueDispatcher per attempt.

The session switcher uses origin for visual distinction; D1's resolver uses it
to find the persistent chat session. Stored on the session_start meta so it
survives reload, and reported by list_sessions (in-memory + disk-only).
"""

from __future__ import annotations

import json
import os
from unittest.mock import MagicMock

import pytest

from agent_os.agent.project_paths import ProjectPaths
from agent_os.agent.session import Session
from agent_os.daemon_v2.agent_manager import AgentManager


def test_session_new_records_origin_and_survives_reload(tmp_path):
    ws = str(tmp_path / "ws")
    s = Session.new("proj_q_1111", ws, origin="queue")
    assert s.origin == "queue"
    # Materialize (first write flushes the session_start meta).
    s.append({"role": "user", "content": "hi"})
    path = os.path.join(ProjectPaths(ws).sessions_dir, "proj_q_1111.jsonl")
    # meta line carries origin
    first = json.loads(open(path).readline())
    assert first["event"] == "session_start"
    assert first["origin"] == "queue"
    # reload recovers it
    assert Session.load(path).origin == "queue"


def test_session_origin_defaults_to_chat(tmp_path):
    ws = str(tmp_path / "ws")
    s = Session.new("proj_c_2222", ws)  # no origin → chat
    assert s.origin == "chat"


def test_legacy_session_without_origin_meta_loads_as_chat(tmp_path):
    """A pre-origin JSONL (no origin on its meta) reads back as chat."""
    sdir = ProjectPaths(str(tmp_path / "ws")).sessions_dir
    os.makedirs(sdir, exist_ok=True)
    p = os.path.join(sdir, "proj_legacy_3333.jsonl")
    rows = [
        {"role": "meta", "event": "session_start", "session_id": "sess_old",
         "session_uuid": "proj_legacy_3333", "timestamp": "2026-01-01T00:00:00+00:00"},
        {"role": "user", "content": "hello", "timestamp": "2026-01-01T00:00:01+00:00"},
    ]
    open(p, "w").write("\n".join(json.dumps(r) for r in rows) + "\n")
    assert Session.load(p).origin == "chat"


def test_list_sessions_reports_origin_for_in_memory_handle(tmp_path):
    mgr = AgentManager(
        project_store=MagicMock(), ws_manager=MagicMock(),
        sub_agent_manager=MagicMock(), activity_translator=MagicMock(),
        process_manager=MagicMock(), platform_provider=None,
        registry=MagicMock(), setup_engine=MagicMock(),
        settings_store=None, credential_store=None,
    )
    mgr._project_store.get_project.return_value = {"workspace": str(tmp_path / "ws"), "name": "P"}
    handle = MagicMock()
    handle.session.is_stopped.return_value = False
    handle.session._paused_for_approval = False
    handle.session.session_uuid = "proj_q_4444"
    handle.session.origin = "queue"
    handle.session.name = None
    handle.session._messages = []
    handle.task = MagicMock(); handle.task.done.return_value = True
    mgr._handles[("p1", "proj_q_4444")] = handle

    entries = mgr.list_sessions("p1")
    mine = [e for e in entries if e["session_id"] == "proj_q_4444"]
    assert mine and mine[0].get("origin") == "queue"
