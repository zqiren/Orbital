# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Phase 1 (seam 3): hydrate-on-inject must adopt the session's *uuid* as the
routing identity, not the file's meta F1 id.

Root cause (REPORT-streaming-status-frontend.md): the frontend addresses a
disk-only session by its session_uuid (the JSONL filename stem), but on inject
the daemon hydrated the file and ran the loop under the file's *meta F1* id
(e.g. ``sess_939e452f`` / ``"default"``), keying the handle and the holder
under F1. Viewed (uuid) != holder (F1) -> live events dropped.

The fix: hydrate adopts ``loaded.session_uuid`` so the handle is keyed by the
same id the frontend is viewing, and ``current_holder_session_id`` returns it.
"""

from __future__ import annotations

import json
import os
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent_os.agent.project_paths import ProjectPaths
from agent_os.daemon_v2.agent_manager import AgentManager


def _make_manager(tmp_path, workspace):
    mgr = AgentManager(
        project_store=MagicMock(), ws_manager=MagicMock(),
        sub_agent_manager=MagicMock(), activity_translator=MagicMock(),
        process_manager=MagicMock(), platform_provider=None,
        registry=MagicMock(), setup_engine=MagicMock(),
        settings_store=None, credential_store=None,
    )
    mgr._state_file = tmp_path / "daemon-state.json"
    mgr._project_store.get_project.return_value = {"workspace": str(workspace), "name": "Proj"}
    # The hydrate branch builds a config before start_agent; stub it so the test
    # exercises id routing, not config assembly.
    mgr._build_agent_config_from_project = MagicMock(return_value=MagicMock())
    return mgr


def _write_session(workspace, uuid, f1):
    sdir = ProjectPaths(str(workspace)).sessions_dir
    os.makedirs(sdir, exist_ok=True)
    rows = [
        {"role": "meta", "event": "session_start", "session_id": f1,
         "session_uuid": uuid, "provider": "deepseek", "model": "deepseek-chat",
         "sdk": "openai", "fallback_models": [], "timestamp": "2026-05-26T00:00:00+00:00"},
        {"role": "user", "content": "Remember the number 7.", "session_id": f1,
         "session_uuid": uuid, "timestamp": "2026-05-26T00:00:01+00:00"},
    ]
    p = os.path.join(sdir, f"{uuid}.jsonl")
    with open(p, "w", encoding="utf-8") as f:
        f.write("\n".join(json.dumps(r) for r in rows) + "\n")
    return p


@pytest.mark.asyncio
async def test_hydrate_addressed_by_uuid_starts_loop_under_uuid(tmp_path):
    """Inject addressed by the uuid -> hydrate -> start_agent receives the uuid
    as session_id (NOT the meta F1)."""
    ws = tmp_path / "ws"
    _write_session(ws, uuid="proj_aaaa1111", f1="sess_abc")
    mgr = _make_manager(tmp_path, ws)
    mgr.start_agent = AsyncMock()

    await mgr.inject_message("p1", "hello", session_id="proj_aaaa1111")

    assert mgr.start_agent.await_count == 1
    passed = mgr.start_agent.await_args.kwargs["session_id"]
    assert passed == "proj_aaaa1111", (
        f"hydrate must adopt the uuid, got {passed!r} (the meta F1 leaked through)"
    )


@pytest.mark.asyncio
async def test_hydrate_keys_handle_and_holder_by_uuid(tmp_path):
    """After hydrate, the handle is keyed by the uuid and
    current_holder_session_id returns the uuid (== the id the frontend views)."""
    ws = tmp_path / "ws"
    _write_session(ws, uuid="proj_aaaa1111", f1="sess_abc")
    mgr = _make_manager(tmp_path, ws)

    def fake_start(project_id, config, **kw):
        sid = kw["session_id"]
        h = MagicMock()
        h.task = MagicMock()
        h.task.done.return_value = False
        h.session._paused_for_approval = False
        mgr._handles[(project_id, sid)] = h

    mgr.start_agent = AsyncMock(side_effect=fake_start)

    await mgr.inject_message("p1", "hello", session_id="proj_aaaa1111")

    assert ("p1", "proj_aaaa1111") in mgr._handles, "handle must be keyed by the uuid"
    assert ("p1", "sess_abc") not in mgr._handles, "handle must NOT be keyed by the meta F1"
    assert mgr.current_holder_session_id("p1") == "proj_aaaa1111"
