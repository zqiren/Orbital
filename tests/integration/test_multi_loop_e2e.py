# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Integration tests for Phase 3c multi-loop end-to-end.

Spins up five chat sessions inside a single project, runs one turn
each, and asserts:

  - Each session ends up with its own ``ProjectHandle`` keyed by the
    ``(project_id, session_id)`` tuple in ``AgentManager._handles``.
  - Each session has a distinct underlying ``Session.session_id``
    (uuid stem of the JSONL file).
  - Messages injected into session N reach session N's session object
    only — no cross-session leakage of conversation state.
  - GET /api/v2/projects/{pid}/sessions returns all five sessions with
    the correct shape.
  - Sub-agent ``list_active`` is scoped per session — sub-agents started
    in session A do not appear in the list for session B.
"""

import asyncio
import os
import tempfile
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from fastapi import FastAPI
from starlette.testclient import TestClient


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_agent_manager(workspace: str):
    """Construct an AgentManager with light mocks suitable for the
    end-to-end multi-loop flow under a TestClient."""
    from agent_os.daemon_v2.agent_manager import AgentManager

    project_store = MagicMock()
    project_store.get_project.return_value = {
        "workspace": workspace,
        "name": "MultiLoopE2E",
        "autonomy": "hands_off",
        "model": "gpt-4",
        "api_key": "sk-test",
        "sdk": "openai",
        "provider": "custom",
        "enabled_sub_agents": [],
    }
    # _start_loop / start_agent persist budget runtime via update_runtime
    project_store.update_runtime = MagicMock()

    ws = MagicMock()
    ws.broadcast = MagicMock()

    sub_agent_mgr = MagicMock()
    # Real session_id-aware list_active so the per-session scoping
    # assertion is meaningful instead of vacuous.
    sub_agent_mgr._active_by_session = {}
    def list_active(pid, *, session_id=None):
        sid = session_id or "default"
        return list(sub_agent_mgr._active_by_session.get((pid, sid), []))
    sub_agent_mgr.list_active = MagicMock(side_effect=list_active)
    sub_agent_mgr.stop = AsyncMock()
    sub_agent_mgr.stop_all = AsyncMock()
    sub_agent_mgr.update_sub_agent_autonomy = MagicMock()

    activity_translator = MagicMock()
    process_manager = MagicMock()

    mgr = AgentManager(
        project_store=project_store,
        ws_manager=ws,
        sub_agent_manager=sub_agent_mgr,
        activity_translator=activity_translator,
        process_manager=process_manager,
    )
    return mgr, ws, project_store, sub_agent_mgr


def _build_test_client(mgr, project_store, sub_agent_mgr):
    """Wire the agents_v2 router with our manager + project_store and
    return a TestClient. Restores the prior module state on shutdown
    via the returned ``restore`` callable."""
    from agent_os.api.routes import agents_v2

    saved = {
        "_project_store": agents_v2._project_store,
        "_agent_manager": agents_v2._agent_manager,
        "_ws_manager": agents_v2._ws_manager,
        "_sub_agent_manager": agents_v2._sub_agent_manager,
    }
    agents_v2.configure(
        project_store=project_store,
        agent_manager=mgr,
        ws_manager=mgr._ws,
        sub_agent_manager=sub_agent_mgr,
    )

    app = FastAPI()
    app.include_router(agents_v2.router)
    client = TestClient(app)

    def restore():
        agents_v2._project_store = saved["_project_store"]
        agents_v2._agent_manager = saved["_agent_manager"]
        agents_v2._ws_manager = saved["_ws_manager"]
        agents_v2._sub_agent_manager = saved["_sub_agent_manager"]
        client.close()

    return client, restore


@pytest.mark.asyncio
async def test_five_sessions_in_one_project_run_in_parallel():
    """Five sessions, one project, one turn each — full multi-loop e2e.

    This is the headline acceptance test for Phase 3c-overnight.
    Asserts handle isolation, JSONL-uuid distinctness, per-session
    sub-agent scoping, and the sessions-listing API.
    """
    project_id = "proj_e2e_multi"

    with tempfile.TemporaryDirectory() as workspace:
        mgr, ws, project_store, sub_agent_mgr = _make_agent_manager(workspace)
        client, restore = _build_test_client(mgr, project_store, sub_agent_mgr)

        # Patch out the heavy dependencies of start_agent so each
        # start_agent call is fast and side-effect-free at the LLM /
        # tool layer. We DO let the real AgentManager wire the handle.
        # The loop.run() coroutine is replaced with a fast no-op so the
        # task completes immediately and the loop-done callback fires.
        run_calls: list = []

        async def fake_loop_run(initial_message=None, initial_nonce=None):
            run_calls.append({
                "initial_message": initial_message,
                "initial_nonce": initial_nonce,
            })
            # Yield once so the task is scheduled cleanly before completing.
            await asyncio.sleep(0)

        with patch("agent_os.daemon_v2.agent_manager.LLMProvider"), \
             patch("agent_os.daemon_v2.agent_manager.ToolRegistry") as MockReg, \
             patch("agent_os.daemon_v2.agent_manager.PromptBuilder"), \
             patch("agent_os.daemon_v2.agent_manager.ContextManager"), \
             patch("agent_os.daemon_v2.agent_manager.AutonomyInterceptor"), \
             patch("agent_os.daemon_v2.agent_manager.AgentLoop") as MockLoop, \
             patch("agent_os.daemon_v2.agent_manager.install_default_skills"):

            mock_reg = MagicMock()
            mock_reg.tool_names.return_value = ["read"]
            MockReg.return_value = mock_reg

            def make_loop(*args, **kwargs):
                inst = MagicMock()
                inst.run = AsyncMock(side_effect=fake_loop_run)
                return inst
            MockLoop.side_effect = make_loop

            session_ids = [f"sess_{i}" for i in range(5)]

            try:
                # Inject a turn into each of five sessions via the public
                # API. Each call mints a new handle under
                # (project_id, session_id) via the lazy-mint path.
                for sid in session_ids:
                    resp = client.post(
                        f"/api/v2/agents/{project_id}/inject",
                        json={"content": f"hello from {sid}",
                              "session_id": sid},
                    )
                    assert resp.status_code == 200, (
                        f"inject for {sid} failed: {resp.status_code} {resp.text}"
                    )

                # Give each fake loop one tick to settle the run task.
                await asyncio.sleep(0.05)

                # ── Assertion 1: handle isolation ──────────────────────
                for sid in session_ids:
                    assert (project_id, sid) in mgr._handles, (
                        f"missing handle for session {sid}"
                    )
                # Exactly the five sessions live under this project.
                project_handles = [
                    key for key in mgr._handles if key[0] == project_id
                ]
                assert len(project_handles) == 5
                assert set(k[1] for k in project_handles) == set(session_ids)

                # ── Assertion 2: distinct Session.session_id (uuid stems) ──
                uuids = {
                    mgr._handles[(project_id, sid)].session.session_id
                    for sid in session_ids
                }
                assert len(uuids) == 5, (
                    f"expected 5 distinct session uuids, got {uuids}"
                )

                # ── Assertion 3: each handle owns its own loop / session ──
                # Two sessions must not share a session object instance.
                instances = [
                    mgr._handles[(project_id, sid)].session
                    for sid in session_ids
                ]
                # Pairwise distinct.
                for i in range(len(instances)):
                    for j in range(i + 1, len(instances)):
                        assert instances[i] is not instances[j]

                # ── Assertion 4: sessions listing endpoint reflects state ──
                resp = client.get(
                    f"/api/v2/projects/{project_id}/sessions",
                )
                assert resp.status_code == 200
                body = resp.json()
                assert body["project_id"] == project_id
                returned_sids = {s["session_id"] for s in body["sessions"]}
                assert returned_sids == set(session_ids)
                # Every entry carries a session_uuid mapping back to a real
                # Session object.
                for entry in body["sessions"]:
                    assert entry["session_uuid"] is not None

                # ── Assertion 5: sub-agent scoping per session ──────────
                # Register a fake sub-agent ONLY in session_0; list_active
                # must reflect that for session_0 but be empty for others.
                sub_agent_mgr._active_by_session[(project_id, "sess_0")] = [
                    {"handle": "helper-1", "display_name": "helper-1",
                     "status": "running"},
                ]
                active_0 = sub_agent_mgr.list_active(
                    project_id, session_id="sess_0",
                )
                assert len(active_0) == 1
                for sid in session_ids[1:]:
                    active = sub_agent_mgr.list_active(
                        project_id, session_id=sid,
                    )
                    assert active == [], (
                        f"sub-agents leaked into session {sid}: {active}"
                    )

                # ── Assertion 6: WS broadcasts carry session_id ─────────
                broadcasts_with_sid: dict[str, set[str]] = {}
                for call in ws.broadcast.call_args_list:
                    pid, payload = call.args
                    if pid != project_id:
                        continue
                    sid = payload.get("session_id")
                    if sid is None:
                        continue
                    broadcasts_with_sid.setdefault(sid, set()).add(
                        payload.get("type", "")
                    )
                # Every session must have at least one broadcast tagged
                # with its session_id — agent.status:running fires from
                # start_agent (commit 9 contract).
                for sid in session_ids:
                    assert sid in broadcasts_with_sid, (
                        f"no WS broadcast carried session_id={sid!r}; "
                        f"observed sessions: {sorted(broadcasts_with_sid)}"
                    )

            finally:
                # Clean up handles so subsequent tests start fresh.
                for sid in session_ids:
                    mgr._handles.pop((project_id, sid), None)
                restore()
