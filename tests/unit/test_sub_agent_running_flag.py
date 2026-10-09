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


# ---------------------------------------------------------------------------
# Spec 102: SubAgentManager.running_session_ids — the per-session form of the
# same signal, feeding the session-list row glyph.
# ---------------------------------------------------------------------------

class TestRunningSessionIds:
    def test_empty_with_no_adapters(self):
        assert _manager().running_session_ids(PID) == set()

    def test_only_sessions_with_an_open_turn(self):
        mgr = _manager()
        _plant(mgr, PID, SID_A, "codex", _adapter(idle=True))
        _plant(mgr, PID, SID_B, "claude-code", _adapter(idle=False))
        assert mgr.running_session_ids(PID) == {SID_B}

    def test_background_running_is_excluded(self):
        mgr = _manager(background_live=True)
        _plant(mgr, PID, SID_A, "claude-code",
               _adapter(idle=True, supports_background=True))
        assert mgr.list_active(PID, session_id=SID_A)[0]["status"] == (
            "background-running")
        assert mgr.running_session_ids(PID) == set()

    def test_dead_adapter_is_excluded_and_evicted(self):
        mgr = _manager()
        _plant(mgr, PID, SID_A, "codex", _adapter(alive=False, idle=False))
        assert mgr.running_session_ids(PID) == set()
        assert make_session_key(PID, SID_A) not in mgr._adapters

    def test_other_projects_are_excluded(self):
        mgr = _manager()
        _plant(mgr, "proj_other", SID_A, "codex", _adapter(idle=False))
        _plant(mgr, PID, SID_B, "codex", _adapter(idle=False))
        assert mgr.running_session_ids(PID) == {SID_B}
        assert mgr.running_session_ids("proj_other") == {SID_A}

    def test_has_running_sub_agents_agrees(self):
        mgr = _manager()
        assert mgr.has_running_sub_agents(PID) is False
        _plant(mgr, PID, SID_A, "codex", _adapter(idle=False))
        assert mgr.has_running_sub_agents(PID) is (
            bool(mgr.running_session_ids(PID)))
        assert mgr.has_running_sub_agents(PID) is True


# ---------------------------------------------------------------------------
# Spec 102: GET /projects/{pid}/sessions → worker_running, via the REAL
# inject-route target branch (seam-3 key-shape check: the adapter slate key
# send() resolves must be the same F1 id the list entry exposes as
# ``session_id``).
# ---------------------------------------------------------------------------

def _fake_check_all_factory(installed_slugs):
    from agent_os.agents.setup_types import AgentSetupStatus

    def _fake():
        statuses = [AgentSetupStatus(
            slug="built-in", name="Built-in",
            installed=True, binary_path=None, version=None,
            dependencies_met=True, missing_dependencies=[],
            credentials_configured=True, missing_credentials=[],
            setup_actions=[],
        )]
        for slug in installed_slugs:
            statuses.append(AgentSetupStatus(
                slug=slug, name=slug, installed=True,
                binary_path="/fake/" + slug, version="1.0.0",
                dependencies_met=True, missing_dependencies=[],
                credentials_configured=True, missing_credentials=[],
                setup_actions=[],
            ))
        return statuses

    return _fake


class _FakeTransport:
    def __init__(self):
        self.dispatched = None

    async def dispatch(self, message):
        self.dispatched = message


class _FakeAdapter:
    """A live worker with an open turn and no subprocess behind it."""

    def __init__(self, handle):
        self._transport = _FakeTransport()
        self.display_name = handle

    def is_alive(self):
        return True

    def is_idle(self):
        return False


@pytest.fixture
def real_app_client(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("USERPROFILE", str(tmp_path / "home"))
    os.makedirs(str(tmp_path / "home"), exist_ok=True)
    from fastapi.testclient import TestClient
    from agent_os.api.app import create_app
    app = create_app(data_dir=str(tmp_path / "data"))
    from agent_os.api.routes import agents_v2
    agents_v2._setup_engine.check_all = _fake_check_all_factory(["claude-code"])
    return TestClient(app)


def _write_user_only_log(ws: str, stem: str, content: str) -> None:
    from agent_os.agent.session import Session
    s = Session.new(stem, ws)
    s.append({"role": "user", "content": content, "source": "user"})


class TestSessionsRouteWorkerRunning:
    def test_true_only_on_the_dispatching_session(
            self, real_app_client, tmp_path, monkeypatch):
        from agent_os.api.routes import agents_v2
        client = real_app_client
        ws = str(tmp_path / "ws_wr")
        os.makedirs(ws, exist_ok=True)
        resp = client.post("/api/v2/projects", json={
            "name": "wr", "workspace": ws,
            "model": "gpt-4", "api_key": "test-key",
        })
        assert resp.status_code == 201, resp.text
        pid = resp.json()["project_id"]
        _write_user_only_log(ws, "wr_sess_pinned001", "earlier")
        _write_user_only_log(ws, "wr_sess_other0002", "other chat")

        sam = agents_v2._sub_agent_manager
        assert isinstance(sam, SubAgentManager)
        # Spawn-on-demand stub: the REAL send() computes the slate key and
        # calls start() with the session id it resolved; register the live
        # worker under exactly that id, never one built by the test.
        started: list[str] = []

        async def _fake_start(project_id, handle, *, session_id=None, **kw):
            started.append(session_id)
            sam._adapters.setdefault(
                make_session_key(project_id, session_id), {},
            )[handle] = _FakeAdapter(handle)
            return f"Started {handle}"

        monkeypatch.setattr(sam, "start", _fake_start)
        am = agents_v2._agent_manager
        monkeypatch.setattr(am, "start_agent", AsyncMock())
        monkeypatch.setattr(am, "_start_loop", AsyncMock())

        resp = client.post(f"/api/v2/agents/{pid}/inject", json={
            "content": "write the essay", "target": "claude-code",
            "session_id": "wr_sess_pinned001",
        })
        assert resp.status_code == 200, resp.text
        assert started == ["wr_sess_pinned001"]

        listed = client.get(f"/api/v2/projects/{pid}/sessions").json()["sessions"]
        by_id = {s["session_id"]: s for s in listed}
        assert set(by_id) >= {"wr_sess_pinned001", "wr_sess_other0002"}
        assert by_id["wr_sess_pinned001"]["worker_running"] is True
        # Manager-only status vocabulary untouched (spec 095/102 §4).
        assert by_id["wr_sess_pinned001"]["status"] == "idle"
        assert by_id["wr_sess_other0002"]["worker_running"] is False
        assert all("worker_running" in s for s in listed)

    def test_false_everywhere_without_a_sub_agent_manager(self, tmp_path):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from agent_os.api.routes import agents_v2

        agent_manager = MagicMock()
        agent_manager.list_sessions.return_value = [
            {"session_id": SID_A, "status": "idle", "session_uuid": SID_A},
        ]
        agents_v2.configure(
            project_store=MagicMock(), agent_manager=agent_manager,
            ws_manager=MagicMock(), sub_agent_manager=None,
            setup_engine=MagicMock(), settings_store=MagicMock(),
            credential_store=MagicMock(),
        )
        app = FastAPI()
        app.include_router(agents_v2.router)
        body = TestClient(app).get(f"/api/v2/projects/{PID}/sessions").json()
        assert body["sessions"][0]["worker_running"] is False
