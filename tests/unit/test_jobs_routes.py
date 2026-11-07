# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Spec 134 Part B §3.7: the job list / cancel routes, and deleting a session
or project cancels its running jobs first (never rmtree-ing a running job)."""

import os
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException

from agent_os.api.routes import agents_v2
from agent_os.daemon_v2.jobs import JobRegistry

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="POSIX shells")

PROJECT = "proj_routes"
SID = "sess-routes-1"


async def _inject(*a, **k):
    return "delivered"


@pytest.fixture
def env(monkeypatch, tmp_path):
    registry = JobRegistry(inject=_inject, broadcast=lambda *a: None,
                           workspaces=lambda: [], poll_interval_s=3600)
    ws = tmp_path / "ws"
    ws.mkdir()
    store = MagicMock()
    store.get_project = MagicMock(return_value={"project_id": PROJECT, "workspace": str(ws)})
    store.list_projects = MagicMock(return_value=[
        {"project_id": PROJECT, "workspace": str(ws), "name": "p"}])
    store.delete_project = MagicMock()
    manager = MagicMock()
    manager._job_registry = registry
    manager.is_running = MagicMock(return_value=False)
    manager.stop_all_sessions = AsyncMock(return_value=[])
    manager.shutdown_dispatcher = AsyncMock()
    manager.delete_session = AsyncMock(return_value={"status": "deleted", "session_id": SID})
    monkeypatch.setattr(agents_v2, "_agent_manager", manager)
    monkeypatch.setattr(agents_v2, "_project_store", store)
    yield SimpleNamespace(registry=registry, ws=ws, manager=manager, store=store)
    registry.stop_all()


def _start(env, command="sleep 30", session=SID):
    group = env.registry.open_group(PROJECT, session, str(env.ws))
    return env.registry.spawn_job(group, command, ["/bin/sh", "-c", command])


async def test_list_jobs(env):
    job = _start(env)
    body = await agents_v2.list_project_jobs(PROJECT)
    [row] = body["jobs"]
    assert row["job_id"] == job.id and row["state"] == "running"
    assert row["output"] == f"orbital/jobs/{job.id}/output.log"
    assert (await agents_v2.list_project_jobs(PROJECT, session_id="other"))["jobs"] == []


async def test_cancel_route_cancels_and_404s_an_unknown_job(env):
    job = _start(env)
    body = await agents_v2.cancel_project_job(PROJECT, job.id)
    assert body["job"]["state"] == "cancelled"
    with pytest.raises(HTTPException) as e:
        await agents_v2.cancel_project_job(PROJECT, "job_nope")
    assert e.value.status_code == 404


async def test_deleting_a_session_cancels_its_jobs_first(env):
    mine = _start(env)
    other = _start(env, session="sess-other")
    await agents_v2.delete_session(PROJECT, SID)
    assert mine.state == "cancelled"
    assert other.state == "running"
    env.manager.delete_session.assert_awaited_once()


async def test_deleting_a_project_cancels_its_jobs_before_removing_files(env, monkeypatch):
    job = _start(env)
    seen = {}
    real = agents_v2._cleanup_project_files

    def cleanup(workspace):
        seen["state_at_rmtree"] = job.state
        seen["running_dir"] = env.registry.is_running_dir(job.dir)
        real(workspace)
    monkeypatch.setattr(agents_v2, "_cleanup_project_files", cleanup)
    await agents_v2.delete_project(PROJECT)
    assert seen == {"state_at_rmtree": "cancelled", "running_dir": False}
    assert not os.path.exists(job.dir)


async def test_bulk_delete_cancels_jobs_too(env):
    job = _start(env)
    await agents_v2.bulk_delete_projects(agents_v2.BulkDeleteRequest(project_ids=[PROJECT]))
    assert job.state == "cancelled"


async def test_routes_without_a_registry(env, monkeypatch):
    env.manager._job_registry = None
    assert (await agents_v2.list_project_jobs(PROJECT))["jobs"] == []
    with pytest.raises(HTTPException) as e:
        await agents_v2.cancel_project_job(PROJECT, "job_x")
    assert e.value.status_code == 404
