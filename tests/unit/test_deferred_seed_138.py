# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Spec 138 "Import writes nothing until first use": a deferred-seed project
gets its AGENTS.md and ``orbital/`` only when a session is opened or a
message is sent, exactly once; startup does not touch it."""

from __future__ import annotations

import asyncio
import os
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from agent_os.api.routes import agents_v2
from agent_os.daemon_v2.project_store import ProjectStore
from agent_os.onboarding.deferred_seed import ensure_project_seeded


@pytest.fixture
def store(tmp_path):
    return ProjectStore(str(tmp_path / "data"))


def _deferred(store, tmp_path, name="imp"):
    ws = tmp_path / f"ws-{name}"
    ws.mkdir()
    pid = store.create_project({"name": name, "workspace": str(ws), "deferred_seed": True})
    return pid, ws


def test_ensure_seeds_once_and_clears_the_flag(store, tmp_path):
    pid, ws = _deferred(store, tmp_path)
    assert not (ws / "AGENTS.md").exists()
    assert ensure_project_seeded(store, pid) is True
    assert (ws / "AGENTS.md").is_file()
    # 1109 Task B: the skills step ran (reconciled) but bundles nothing.
    assert store.get_project(pid)["default_skills_reconciled"] is True
    assert not (ws / "orbital" / "skills").exists()
    assert store.get_project(pid)["deferred_seed"] is False
    (ws / "AGENTS.md").unlink()
    assert ensure_project_seeded(store, pid) is False  # never again
    assert not (ws / "AGENTS.md").exists()


def test_ensure_is_a_noop_for_ordinary_projects(store, tmp_path):
    ws = tmp_path / "plain"
    ws.mkdir()
    pid = store.create_project({"name": "plain", "workspace": str(ws)})
    assert ensure_project_seeded(store, pid) is False
    assert not (ws / "AGENTS.md").exists()
    assert ensure_project_seeded(store, "proj_missing") is False


def test_existing_agents_md_is_kept(store, tmp_path):
    pid, ws = _deferred(store, tmp_path)
    (ws / "AGENTS.md").write_text("mine")
    ensure_project_seeded(store, pid)
    assert (ws / "AGENTS.md").read_text() == "mine"


@pytest.fixture
def api(store):
    manager = MagicMock()
    manager._ensure_dispatcher = AsyncMock()
    manager.list_sessions = MagicMock(return_value=[])
    manager.materialize_imported_session = MagicMock(return_value="unknown")
    agents_v2.configure(project_store=store, agent_manager=manager, ws_manager=MagicMock())
    app = FastAPI()
    app.include_router(agents_v2.router)
    return TestClient(app), manager


def test_opening_an_imported_session_seeds(api, store, tmp_path):
    client, manager = api
    pid, ws = _deferred(store, tmp_path)
    r = client.get(f"/api/v2/agents/{pid}/chat",
                   params={"session_id": "imp_codex_0123456789abcdef"})
    assert r.status_code == 200
    assert (ws / "AGENTS.md").is_file()
    assert store.get_project(pid)["deferred_seed"] is False
    manager._ensure_dispatcher.assert_awaited_once_with(pid, str(ws))


def test_reading_an_ordinary_chat_does_not_seed(api, store, tmp_path):
    client, _ = api
    pid, ws = _deferred(store, tmp_path)
    client.get(f"/api/v2/agents/{pid}/chat", params={"session_id": "abc_12345678"})
    client.get(f"/api/v2/agents/{pid}/chat")
    assert not (ws / "AGENTS.md").exists()
    assert not (ws / "orbital").exists()


def test_queueing_work_seeds(api, store, tmp_path):
    client, _ = api
    pid, ws = _deferred(store, tmp_path)
    client.post(f"/api/v2/projects/{pid}/queue/items", json={"content": "do it"})
    assert (ws / "AGENTS.md").is_file()


def test_inject_route_seeds_before_anything_else(api, store, tmp_path, monkeypatch):
    client, manager = api
    pid, ws = _deferred(store, tmp_path)
    seen = {}

    async def stop_after_seed(project_id, session_id):
        seen["seeded"] = (ws / "AGENTS.md").is_file()
        raise RuntimeError("stop here")

    monkeypatch.setattr(agents_v2, "_materialize_if_imported", stop_after_seed)
    with pytest.raises(RuntimeError):
        client.post(f"/api/v2/agents/{pid}/inject", json={"content": "hi"})
    assert seen == {"seeded": True}


def test_startup_dispatchers_skip_deferred_projects(store, tmp_path):
    from agent_os.daemon_v2.agent_manager import AgentManager

    pid, _ = _deferred(store, tmp_path)
    ws2 = tmp_path / "ordinary"
    ws2.mkdir()
    pid2 = store.create_project({"name": "ordinary", "workspace": str(ws2)})
    fake = MagicMock()
    fake._project_store = store
    fake._ensure_dispatcher = AsyncMock()
    asyncio.run(AgentManager.start_all_dispatchers(fake))
    started = [c.args[0] for c in fake._ensure_dispatcher.await_args_list]
    assert started == [pid2]
    assert pid not in started


def test_queue_of_an_unseeded_project_reads_idle(api, store, tmp_path):
    """A deferred project has no dispatcher yet, so nothing is running. The
    queue must read idle — "running" locks the chat composer behind
    "Pause queue to start chatting" on every freshly imported project."""
    from agent_os.queue.store import QueueStore
    client, manager = api
    pid, ws = _deferred(store, tmp_path)
    manager.get_queue_store = MagicMock(
        side_effect=lambda p, workspace: QueueStore(os.path.join(workspace, "orbital", "queue.json")))
    r = client.get(f"/api/v2/projects/{pid}/queue")
    assert r.status_code == 200
    assert r.json()["state"] == "idle"
    assert r.json()["items"] == []
    assert not (ws / "orbital").exists()
