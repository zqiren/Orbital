# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Spec 095: the project list's dot must light while a worker runs with no
management turn around it (a pinned dispatch, a queue item assigned to a
worker).

Four pieces, all tested here:
- ``SubAgentManager.has_running_sub_agents(pid)``: True iff some worker in ANY
  session of the project has an open turn. ``background-running`` does not
  count (a detached job would otherwise hold the dot green indefinitely).
- ``GET /agents/{pid}/run-status`` carries it as the additive
  ``sub_agents_running`` field. ``status`` keeps its manager-only meaning.
- ``LifecycleObserver.on_message_routed`` broadcasts ``sub_agent.dispatched``
  for every dispatch, so a queued prompt re-dispatched to an already-warm
  worker (no ``sub_agent.started``, no route ack) still tells the frontend
  to refetch.
- ``AgentManager.delete_session`` refuses while a worker in that session is
  still working. The run-status guard alone read a pinned run as idle, and
  the teardown that followed killed the worker.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent_os.daemon_v2.background_work import BackgroundWorkRegistry
from agent_os.daemon_v2.lifecycle_observer import LifecycleObserver
from agent_os.daemon_v2.models import make_session_key
from agent_os.daemon_v2.sub_agent_manager import SubAgentManager

PID = "proj_flag"
SID_A = "sess_flag_a"
SID_B = "sess_flag_b"


def _manager(*, background_live: bool = False) -> SubAgentManager:
    pm = MagicMock()
    registry = MagicMock(spec=BackgroundWorkRegistry)
    registry.three_state_supported = True
    registry.has_live.return_value = background_live
    pm.background_work = registry
    return SubAgentManager(process_manager=pm)


def _adapter(*, alive: bool = True, idle: bool = False,
             supports_background: bool = False):
    adapter = MagicMock()
    adapter.is_alive = MagicMock(return_value=alive)
    adapter.is_idle = MagicMock(return_value=idle)
    adapter.display_name = "worker"
    adapter._transport = MagicMock()
    adapter._transport.supports_background_status = supports_background
    return adapter


def _plant(mgr: SubAgentManager, pid: str, sid: str, handle: str, adapter):
    mgr._adapters.setdefault(make_session_key(pid, sid), {})[handle] = adapter


# ---------------------------------------------------------------------------
# SubAgentManager.has_running_sub_agents
# ---------------------------------------------------------------------------

class TestHasRunningSubAgents:
    def test_false_with_no_adapters(self):
        assert _manager().has_running_sub_agents(PID) is False

    def test_true_for_open_turn_in_any_session(self):
        mgr = _manager()
        _plant(mgr, PID, SID_A, "codex", _adapter(idle=True))
        _plant(mgr, PID, SID_B, "claude-code", _adapter(idle=False))
        assert mgr.has_running_sub_agents(PID) is True

    def test_false_when_every_worker_is_idle(self):
        mgr = _manager()
        _plant(mgr, PID, SID_A, "codex", _adapter(idle=True))
        _plant(mgr, PID, SID_B, "claude-code", _adapter(idle=True))
        assert mgr.has_running_sub_agents(PID) is False

    def test_background_running_does_not_count(self):
        mgr = _manager(background_live=True)
        adapter = _adapter(idle=True, supports_background=True)
        _plant(mgr, PID, SID_A, "claude-code", adapter)
        # Precondition: the adapter really reads background-running.
        assert mgr.list_active(PID, session_id=SID_A)[0]["status"] == (
            "background-running")
        assert mgr.has_running_sub_agents(PID) is False

    def test_dead_adapter_does_not_count_and_is_evicted(self):
        mgr = _manager()
        _plant(mgr, PID, SID_A, "codex", _adapter(alive=False, idle=False))
        assert mgr.has_running_sub_agents(PID) is False
        assert make_session_key(PID, SID_A) not in mgr._adapters

    def test_other_projects_do_not_count(self):
        mgr = _manager()
        _plant(mgr, "proj_other", SID_A, "codex", _adapter(idle=False))
        assert mgr.has_running_sub_agents(PID) is False
        assert mgr.has_running_sub_agents("proj_other") is True


# ---------------------------------------------------------------------------
# GET /agents/{pid}/run-status → sub_agents_running
# ---------------------------------------------------------------------------

def _run_status_client(sub_agent_manager):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from agent_os.api.routes import agents_v2

    agent_manager = MagicMock()
    agent_manager.get_run_status.return_value = "idle"
    agent_manager.current_holder_session_id.return_value = None
    agent_manager.list_pending.return_value = []
    agent_manager.get_last_terminal_event.return_value = None
    agents_v2.configure(
        project_store=MagicMock(),
        agent_manager=agent_manager,
        ws_manager=MagicMock(),
        sub_agent_manager=sub_agent_manager,
        setup_engine=MagicMock(),
        settings_store=MagicMock(),
        credential_store=MagicMock(),
    )
    app = FastAPI()
    app.include_router(agents_v2.router)
    return TestClient(app)


class TestRunStatusField:
    def test_false_when_no_worker_runs(self):
        client = _run_status_client(_manager())
        body = client.get(f"/api/v2/agents/{PID}/run-status").json()
        assert body["sub_agents_running"] is False
        assert body["status"] == "idle"

    def test_true_with_a_running_worker_and_status_stays_idle(self):
        mgr = _manager()
        _plant(mgr, PID, SID_A, "codex", _adapter(idle=False))
        client = _run_status_client(mgr)
        body = client.get(f"/api/v2/agents/{PID}/run-status").json()
        assert body["sub_agents_running"] is True
        # The manager-only status vocabulary is untouched (spec 095 §4.3).
        assert body["status"] == "idle"
        assert body["current_holder_session_id"] is None

    def test_false_when_no_sub_agent_manager_is_wired(self):
        client = _run_status_client(None)
        body = client.get(f"/api/v2/agents/{PID}/run-status").json()
        assert body["sub_agents_running"] is False


# ---------------------------------------------------------------------------
# LifecycleObserver.on_message_routed → sub_agent.dispatched
# ---------------------------------------------------------------------------

class TestDispatchBroadcast:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("initiator", ["user_pinned", "queue_item", "agent"])
    async def test_every_dispatch_broadcasts(self, initiator):
        agent_manager = MagicMock()
        agent_manager.inject_system_message = AsyncMock()
        ws = MagicMock()
        observer = LifecycleObserver(agent_manager, ws)

        await observer.on_message_routed(
            PID, "codex", initiator=initiator, message_preview="do the thing",
            transcript_path="/t.jsonl", session_id=SID_A,
            dispatch_id=f"{SID_A}:abcd1234",
        )

        events = [c.args[1] for c in ws.broadcast.call_args_list]
        dispatched = [e for e in events if e["type"] == "sub_agent.dispatched"]
        assert dispatched == [{
            "type": "sub_agent.dispatched",
            "project_id": PID,
            "session_id": SID_A,
            "handle": "codex",
            "initiator": initiator,
        }]
        assert ws.broadcast.call_args_list[0].args[0] == PID
        # The marker row still lands exactly as before.
        agent_manager.inject_system_message.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_broadcast_survives_a_failed_inject(self):
        agent_manager = MagicMock()
        agent_manager.inject_system_message = AsyncMock(
            side_effect=RuntimeError("session locked"))
        ws = MagicMock()
        observer = LifecycleObserver(agent_manager, ws)

        await observer.on_message_routed(
            PID, "codex", initiator="user_pinned", message_preview="x",
            transcript_path="/t.jsonl", session_id=SID_A,
        )

        types = [c.args[1]["type"] for c in ws.broadcast.call_args_list]
        assert types == ["sub_agent.dispatched"]


# ---------------------------------------------------------------------------
# AgentManager.delete_session worker guard (spec 095 §8)
# ---------------------------------------------------------------------------

def _delete_fixture(tmp_path, sub_agent_manager):
    from agent_os.agent.project_paths import ProjectPaths
    from agent_os.daemon_v2.agent_manager import AgentManager
    from agent_os.daemon_v2.project_store import ProjectStore

    data_dir = tmp_path / "data"
    workspace = tmp_path / "ws"
    data_dir.mkdir()
    workspace.mkdir()
    store = ProjectStore(data_dir=str(data_dir))
    pid = store.create_project({
        "name": "Delete Guard", "workspace": str(workspace),
        "model": "gpt-4", "api_key": "sk-test",
    })
    sessions_dir = Path(ProjectPaths(str(workspace)).sessions_dir)
    os.makedirs(sessions_dir, exist_ok=True)
    jsonl = sessions_dir / f"{SID_A}.jsonl"
    rows = [
        {"role": "meta", "event": "session_start", "session_id": SID_A,
         "session_uuid": SID_A, "timestamp": "2026-09-19T08:00:00+00:00"},
        {"role": "user", "content": "hi", "session_id": SID_A,
         "timestamp": "2026-09-19T08:00:00+00:00"},
    ]
    jsonl.write_text("\n".join(json.dumps(r) for r in rows) + "\n",
                     encoding="utf-8")
    mgr = AgentManager(
        project_store=store, ws_manager=MagicMock(),
        sub_agent_manager=sub_agent_manager,
        activity_translator=MagicMock(), process_manager=MagicMock(),
        platform_provider=None, registry=MagicMock(),
        setup_engine=MagicMock(), settings_store=None, credential_store=None,
    )
    mgr.stop_agent = AsyncMock()
    return mgr, pid, jsonl


class TestDeleteSessionWorkerGuard:
    @pytest.mark.asyncio
    async def test_refuses_while_a_worker_runs(self, tmp_path):
        sam = _manager()
        sam.stop_all = AsyncMock()
        mgr, pid, jsonl = _delete_fixture(tmp_path, sam)
        _plant(sam, pid, SID_A, "codex", _adapter(idle=False))
        # Precondition: the manager itself reads idle — the old guard passed.
        assert mgr.get_run_status(pid, session_id=SID_A) == "idle"

        with pytest.raises(RuntimeError):
            await mgr.delete_session(pid, SID_A)

        assert jsonl.exists()
        mgr.stop_agent.assert_not_awaited()
        sam.stop_all.assert_not_awaited()
        assert make_session_key(pid, SID_A) in sam._adapters

    @pytest.mark.asyncio
    async def test_refuses_while_background_work_is_live(self, tmp_path):
        sam = _manager(background_live=True)
        mgr, pid, jsonl = _delete_fixture(tmp_path, sam)
        _plant(sam, pid, SID_A, "claude-code",
               _adapter(idle=True, supports_background=True))

        with pytest.raises(RuntimeError):
            await mgr.delete_session(pid, SID_A)
        assert jsonl.exists()

    @pytest.mark.asyncio
    async def test_idle_worker_does_not_block_delete(self, tmp_path):
        sam = _manager()
        mgr, pid, jsonl = _delete_fixture(tmp_path, sam)
        _plant(sam, pid, SID_A, "codex", _adapter(idle=True))

        result = await mgr.delete_session(pid, SID_A)

        assert result == {"status": "deleted", "session_id": SID_A}
        assert not jsonl.exists()

    @pytest.mark.asyncio
    async def test_worker_in_another_session_does_not_block(self, tmp_path):
        sam = _manager()
        mgr, pid, jsonl = _delete_fixture(tmp_path, sam)
        _plant(sam, pid, SID_B, "codex", _adapter(idle=False))

        await mgr.delete_session(pid, SID_A)
        assert not jsonl.exists()

    def test_route_maps_the_refusal_to_409(self, tmp_path):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from agent_os.api.routes import agents_v2

        sam = _manager()
        mgr, pid, jsonl = _delete_fixture(tmp_path, sam)
        _plant(sam, pid, SID_A, "codex", _adapter(idle=False))
        agents_v2.configure(
            project_store=mgr._project_store, agent_manager=mgr,
            ws_manager=MagicMock(), sub_agent_manager=sam,
            setup_engine=MagicMock(), settings_store=MagicMock(),
            credential_store=MagicMock(),
        )
        app = FastAPI()
        app.include_router(agents_v2.router)
        resp = TestClient(app).request(
            "DELETE", f"/api/v2/agents/{pid}/sessions/{SID_A}")

        assert resp.status_code == 409, resp.text
        assert jsonl.exists()
