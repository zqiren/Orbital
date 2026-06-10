# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Phase 1 (seam 3) end-to-end gate: inject a disk session by its uuid, drive a
turn, and assert the whole live path is keyed by the uuid the frontend views.

This is the in-process analog of the live stop-and-verify gate: it exercises
inject -> hydrate -> start_agent -> handle/holder -> run-status -> the event
stamping closure, all through the public FastAPI surface, with the LLM/loop
stubbed (no provider, no network). It proves the bug is fixed:

  - the handle and current_holder_session_id are keyed by the uuid (not meta F1)
  - GET run-status reports `running` for the project (holder-aware)
  - a streamed chunk is stamped with the viewed uuid
"""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from starlette.testclient import TestClient

from agent_os.agent.project_paths import ProjectPaths
from agent_os.daemon_v2.agent_manager import AgentManager
from agent_os.api.routes import agents_v2


SEED_UUID = "proj_seed_aaaa1111"   # JSONL filename stem == what the sidebar/FE addresses
SEED_F1 = "sess_legacy_meta"        # the (non-unique) meta F1 that must NOT become the routing key


def _seed_disk_session(workspace: str):
    sdir = ProjectPaths(workspace).sessions_dir
    os.makedirs(sdir, exist_ok=True)
    rows = [
        {"role": "meta", "event": "session_start", "session_id": SEED_F1,
         "session_uuid": SEED_UUID, "provider": "custom", "model": "gpt-4",
         "sdk": "openai", "fallback_models": [], "timestamp": "2026-05-26T00:00:00+00:00"},
        {"role": "user", "content": "Remember the number 7.", "session_id": SEED_F1,
         "session_uuid": SEED_UUID, "timestamp": "2026-05-26T00:00:01+00:00"},
    ]
    with open(os.path.join(sdir, f"{SEED_UUID}.jsonl"), "w", encoding="utf-8") as f:
        f.write("\n".join(json.dumps(r) for r in rows) + "\n")


def _make_manager(workspace: str):
    project_store = MagicMock()
    project_store.get_project.return_value = {
        "workspace": workspace, "name": "Proj", "autonomy": "hands_off",
        "model": "gpt-4", "api_key": "sk-test", "sdk": "openai",
        "provider": "custom", "enabled_sub_agents": [],
    }
    project_store.update_runtime = MagicMock()
    ws = MagicMock(); ws.broadcast = MagicMock()
    sub_agent_mgr = MagicMock()
    sub_agent_mgr.list_active = MagicMock(return_value=[])
    sub_agent_mgr.stop_all = AsyncMock()
    activity_translator = MagicMock()
    mgr = AgentManager(
        project_store=project_store, ws_manager=ws,
        sub_agent_manager=sub_agent_mgr, activity_translator=activity_translator,
        process_manager=MagicMock(), platform_provider=None,
        registry=MagicMock(), setup_engine=MagicMock(),
        settings_store=None, credential_store=None,
    )
    return mgr, project_store, sub_agent_mgr


def _client(mgr, project_store, sub_agent_mgr):
    saved = {
        "_project_store": agents_v2._project_store,
        "_agent_manager": agents_v2._agent_manager,
        "_ws_manager": agents_v2._ws_manager,
        "_sub_agent_manager": agents_v2._sub_agent_manager,
    }
    agents_v2._project_store = project_store
    agents_v2._agent_manager = mgr
    agents_v2._ws_manager = mgr._ws
    agents_v2._sub_agent_manager = sub_agent_mgr
    app = FastAPI(); app.include_router(agents_v2.router)
    client = TestClient(app)

    def restore():
        agents_v2._project_store = saved["_project_store"]
        agents_v2._agent_manager = saved["_agent_manager"]
        agents_v2._ws_manager = saved["_ws_manager"]
        agents_v2._sub_agent_manager = saved["_sub_agent_manager"]
        client.close()

    return client, restore


@pytest.mark.asyncio
async def test_inject_by_uuid_routes_everything_under_uuid():
    """Inject is driven IN-PROCESS (same event loop) so the turn's background
    task is deterministically observable; the run-status HTTP endpoint is then
    exercised over the shared AgentManager via TestClient."""
    project_id = "proj_gate"
    with tempfile.TemporaryDirectory() as workspace:
        _seed_disk_session(workspace)
        mgr, project_store, sub_agent_mgr = _make_manager(workspace)

        async def fake_loop_run(initial_message=None, initial_nonce=None):
            # Keep the task alive (the turn is "running") until we cancel it.
            await asyncio.sleep(60)

        with patch("agent_os.daemon_v2.agent_manager.LLMProvider"), \
             patch("agent_os.daemon_v2.agent_manager.ToolRegistry") as MockReg, \
             patch("agent_os.daemon_v2.agent_manager.PromptBuilder"), \
             patch("agent_os.daemon_v2.agent_manager.ContextManager"), \
             patch("agent_os.daemon_v2.agent_manager.AutonomyInterceptor"), \
             patch("agent_os.daemon_v2.agent_manager.AgentLoop") as MockLoop, \
             patch("agent_os.daemon_v2.agent_manager.install_default_skills"):

            mock_reg = MagicMock(); mock_reg.tool_names.return_value = ["read"]
            MockReg.return_value = mock_reg

            def make_loop(*args, **kwargs):
                inst = MagicMock()
                inst.run = AsyncMock(side_effect=fake_loop_run)
                return inst
            MockLoop.side_effect = make_loop

            # Frontend addresses the disk session by its UUID.
            result = await mgr.inject_message(
                project_id, "What number?", session_id=SEED_UUID,
            )
            assert result == "started", result
            await asyncio.sleep(0.05)  # let the run task schedule

            try:
                # 1) handle + holder keyed by the UUID, never the meta F1.
                assert (project_id, SEED_UUID) in mgr._handles
                assert (project_id, SEED_F1) not in mgr._handles
                assert mgr.current_holder_session_id(project_id) == SEED_UUID

                # 2) the hydrated session continues history (not a fresh fork).
                sess = mgr._handles[(project_id, SEED_UUID)].session
                assert sess.session_uuid == SEED_UUID
                assert any(m.get("content") == "Remember the number 7."
                           for m in sess.get_messages())

                # 3) run-status is holder-aware, both in-process and over HTTP:
                #    the project reports running and names the viewed uuid.
                assert mgr.get_run_status(project_id) == "running"
                client, restore = _client(mgr, project_store, sub_agent_mgr)
                try:
                    rs = client.get(f"/api/v2/agents/{project_id}/run-status")
                    assert rs.status_code == 200
                    body = rs.json()
                    assert body["status"] == "running", body
                    assert body["current_holder_session_id"] == SEED_UUID, body
                finally:
                    restore()

                # 4) a streamed chunk is stamped with the viewed uuid (the
                #    on_stream closure captured the canonical id at start_agent).
                chunk = MagicMock(); chunk.text = "hi"; chunk.is_final = False
                sess.on_stream(chunk)
                assert mgr._activity_translator.on_stream_chunk.call_count >= 1
                kw = mgr._activity_translator.on_stream_chunk.call_args.kwargs
                assert kw.get("session_id") == SEED_UUID, (
                    f"stream chunk stamped with {kw.get('session_id')!r}, expected the viewed uuid"
                )
            finally:
                # Cancel the long-lived fake loop task so the test exits clean.
                h = mgr._handles.get((project_id, SEED_UUID))
                if h is not None and h.task is not None:
                    h.task.cancel()
                    try:
                        await h.task
                    except (asyncio.CancelledError, Exception):
                        pass
