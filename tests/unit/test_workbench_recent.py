# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""One-page Welcome + Workbench landing (2026-11-23): the Workbench's
最近 / Recent section rides ``GET /api/v2/workbench/feed`` as one additive
field, ``recent``, computed from the session lists the feed already reads.

The rules: newest first across projects (imported sessions included), never a
session that is in the same response's ``rows``, no Memory session, no
automation run, nothing from a project excluded from the Workbench, capped by
``?recent=`` (default 10). No I/O beyond what the feed already does.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone

import httpx
import pytest
from fastapi import FastAPI

from agent_os.api.routes import workbench as wb
from agent_os.daemon_v2.checked_store import CheckedStore

NOW = datetime(2026, 11, 23, 12, 0, 0, tzinfo=timezone.utc)


class FakeStore:
    def __init__(self, projects):
        self._p = {p["project_id"]: p for p in projects}

    def list_projects(self):
        return list(self._p.values())

    def get_project(self, pid):
        return self._p.get(pid)


class FakeManager:
    def __init__(self, sessions):
        self.sessions = sessions
        self.calls = []

    def list_sessions(self, pid, include_memory=False):
        self.calls.append(pid)
        return [dict(r) for r in self.sessions.get(pid, [])]


class FakeSubAgents:
    def running_session_ids(self, pid):
        return set()


def _project(tmp_path, pid, name=None, **extra):
    ws = tmp_path / pid
    ws.mkdir(parents=True, exist_ok=True)
    return {"project_id": pid, "name": name or f"Name {pid}", "workspace": str(ws),
            "timezone": "UTC", **extra}


def _row(sid, at, **extra):
    """A resting session the user already checked (so it is not in ``rows``)."""
    return {"session_id": sid, "session_uuid": sid, "status": "idle", "name": f"title {sid}",
            "pinned_target": None, "last_user_at": None, "last_reply_at": None,
            "last_activity_at": at, "origin": "chat", **extra}


def _client(tmp_path, projects, sessions):
    store = CheckedStore.in_dir(str(tmp_path / "data"), now_fn=lambda: NOW)
    manager = FakeManager(sessions)
    wb.configure(FakeStore(projects), manager, None, now_fn=lambda: NOW,
                 checked_store=store, sub_agent_manager=FakeSubAgents())
    app = FastAPI()
    app.include_router(wb.router)
    client = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t")
    return client, manager


@pytest.fixture
def world(tmp_path):
    a = _project(tmp_path, "proj_a", "Website")
    b = _project(tmp_path, "proj_b", "Desktop app")
    x = _project(tmp_path, "proj_x", workbench_exclude_global=True)
    q = _project(tmp_path, "proj_q", "Quick Tasks", is_scratch=True)
    sessions = {
        "proj_a": [
            _row("a_old", "2026-11-10T08:00:00+00:00"),
            _row("a_new", "2026-11-22T08:00:00+00:00", pinned_target="codex"),
            # An automation run and the Memory session never show.
            _row("a_trig", "2026-11-23T07:00:00+00:00", trigger_type="schedule"),
            _row("a_mem", "2026-11-23T07:30:00+00:00", origin="memory"),
            # Never active: nothing to sort by.
            _row("a_empty", None),
        ],
        "proj_b": [
            # An imported native session not opened in Orbital yet.
            _row("imp_claude-code_abc", "2026-11-21T08:00:00Z", origin="imported",
                 pinned_target="claude-code",
                 imported={"agent": "claude-code", "native_id": "abc",
                           "materialized": False, "size_bytes": 10}),
            # Running: it is in rows, so it is not in recent.
            _row("b_run", "2026-11-23T11:00:00+00:00", status="running"),
        ],
        "proj_x": [_row("x_hidden", "2026-11-23T11:30:00+00:00")],
        "proj_q": [_row("q_task", "2026-11-20T08:00:00+00:00")],
    }
    return [a, b, x, q], sessions


async def _get(client, query=""):
    r = await client.get("/api/v2/workbench/feed" + query)
    assert r.status_code == 200
    return r.json()


@pytest.mark.asyncio
async def test_recent_is_newest_first_across_projects(tmp_path, world):
    projects, sessions = world
    client, _ = _client(tmp_path, projects, sessions)
    body = await _get(client)
    recent = body["recent"]
    assert [r["session_id"] for r in recent] == [
        "a_new", "imp_claude-code_abc", "q_task", "a_old"]
    new = recent[0]
    assert new == {
        "project_id": "proj_a", "project_name": "Website", "session_id": "a_new",
        "session_uuid": "a_new", "title": "title a_new", "agent": "codex",
        "updated_at": "2026-11-22T08:00:00+00:00", "imported": False,
    }
    imported = recent[1]
    assert imported["agent"] == "claude-code" and imported["imported"] is True
    assert imported["project_name"] == "Desktop app"
    # Quick Tasks is a project like the others here; Orbital's own session = null agent.
    assert recent[2]["project_id"] == "proj_q" and recent[2]["agent"] is None


@pytest.mark.asyncio
async def test_a_session_in_rows_is_never_in_recent(tmp_path, world):
    projects, sessions = world
    client, _ = _client(tmp_path, projects, sessions)
    body = await _get(client)
    in_rows = {(r["project_id"], r["session_id"]) for r in body["rows"]}
    assert ("proj_b", "b_run") in in_rows
    in_recent = {(r["project_id"], r["session_id"]) for r in body["recent"]}
    assert not (in_rows & in_recent)


@pytest.mark.asyncio
async def test_recent_leaves_out_runs_memory_hidden_projects_and_inactive(tmp_path, world):
    projects, sessions = world
    client, _ = _client(tmp_path, projects, sessions)
    ids = {r["session_id"] for r in (await _get(client))["recent"]}
    assert not ids & {"a_trig", "a_mem", "a_empty", "x_hidden"}


@pytest.mark.asyncio
async def test_recent_cap_and_zero(tmp_path, world):
    projects, sessions = world
    client, _ = _client(tmp_path, projects, sessions)
    assert [r["session_id"] for r in (await _get(client, "?recent=2"))["recent"]] == [
        "a_new", "imp_claude-code_abc"]
    assert (await _get(client, "?recent=0"))["recent"] == []
    r = await client.get("/api/v2/workbench/feed?recent=51")
    assert r.status_code == 422


@pytest.mark.asyncio
async def test_default_cap_is_ten(tmp_path):
    p = _project(tmp_path, "proj_a")
    rows = [_row(f"s{i:02d}", f"2026-11-{i + 1:02d}T08:00:00+00:00") for i in range(14)]
    client, _ = _client(tmp_path, [p], {"proj_a": rows})
    recent = (await _get(client))["recent"]
    assert len(recent) == 10
    assert recent[0]["session_id"] == "s13"


@pytest.mark.asyncio
async def test_recent_reads_no_session_list_twice_and_writes_nothing(tmp_path, world):
    projects, sessions = world
    client, manager = _client(tmp_path, projects, sessions)
    before = {p["project_id"]: sorted(os.listdir(p["workspace"])) for p in projects}
    await _get(client)
    # One list per Workbench project, the same calls the feed made before.
    assert sorted(manager.calls) == ["proj_a", "proj_b", "proj_q"]
    after = {p["project_id"]: sorted(os.listdir(p["workspace"])) for p in projects}
    assert before == after


@pytest.mark.asyncio
async def test_offset_timestamps_sort_by_instant(tmp_path):
    p = _project(tmp_path, "proj_a")
    rows = [_row("utc", "2026-11-22T09:00:00+00:00"),
            # 08:30 UTC written in +08:00: older than 09:00 UTC.
            _row("cst", "2026-11-22T16:30:00+08:00")]
    client, _ = _client(tmp_path, [p], {"proj_a": rows})
    assert [r["session_id"] for r in (await _get(client))["recent"]] == ["utc", "cst"]
