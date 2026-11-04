# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Spec 154 §5.4 — ``POST /api/v2/workbench/route``: one model call on send,
a lenient parse, and Quick Tasks on every failure. The provider is faked."""

from __future__ import annotations

import asyncio
import os
from datetime import datetime, timezone
from types import SimpleNamespace

import httpx
import pytest
from fastapi import FastAPI

from agent_os.api.routes import workbench as wb

NOW = datetime(2026, 11, 4, 12, 0, 0, tzinfo=timezone.utc)


class FakeStore:
    def __init__(self, projects):
        self._p = {p["project_id"]: p for p in projects}

    def list_projects(self):
        return list(self._p.values())

    def get_project(self, pid):
        return self._p.get(pid)

    def find_scratch_project(self):
        return next((p for p in self._p.values() if p.get("is_scratch")), None)


class FakeProvider:
    def __init__(self, reply=None, *, exc=None, delay=0.0):
        self.reply, self.exc, self.delay = reply, exc, delay
        self.calls = []

    async def complete(self, messages, tools=None, *, disable_reasoning=False):
        self.calls.append((messages, disable_reasoning))
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.exc:
            raise self.exc
        return SimpleNamespace(text=self.reply, usage=None)


class FakeManager:
    def __init__(self, provider, *, ready=True, sessions=None):
        self.provider, self.ready = provider, ready
        self.sessions = sessions or {}
        self.config_for = []

    def manager_ready(self, pid):
        return self.ready

    def _build_agent_config_from_project(self, pid):
        self.config_for.append(pid)
        return SimpleNamespace(project_id=pid)

    def _build_llm_providers(self, config):
        return ("main", [], self.provider, None)

    def list_sessions(self, pid, include_memory=False):
        return list(self.sessions.get(pid, []))


def _project(tmp_path, pid, *, goals=None, state=None, **extra):
    ws = tmp_path / pid
    (ws / "orbital" / "instructions").mkdir(parents=True, exist_ok=True)
    if goals is not None:
        (ws / "orbital" / "instructions" / "project_goals.md").write_text(goals, encoding="utf-8")
    if state is not None:
        (ws / "orbital" / "PROJECT_STATE.md").write_text(state, encoding="utf-8")
    return {"project_id": pid, "name": f"name-{pid}", "workspace": str(ws), **extra}


@pytest.fixture
def projects(tmp_path):
    return [
        _project(tmp_path, "proj_site", goals="Marketing website and downloads."),
        _project(tmp_path, "proj_app", goals="The desktop app.", state="# State\nshipping v0.20"),
        _project(tmp_path, "proj_hidden", workbench_exclude_global=True),
        _project(tmp_path, "proj_scratch", is_scratch=True),
    ]


def _client(projects, manager):
    wb.configure(FakeStore(projects), manager, None, now_fn=lambda: NOW)
    app = FastAPI()
    app.include_router(wb.router)
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t")


async def _route(client, text="update the download page"):
    r = await client.post("/api/v2/workbench/route", json={"text": text})
    assert r.status_code == 200
    return r.json()


@pytest.mark.asyncio
async def test_routes_to_the_model_pick_with_quick_tasks_credential(projects):
    prov = FakeProvider('{"project": "proj_site"}')
    mgr = FakeManager(prov, sessions={"proj_site": [
        {"name": "old", "last_activity_at": "2026-11-01"},
        {"name": "newest", "last_activity_at": "2026-11-03"},
    ]})
    body = await _route(_client(projects, mgr))
    assert body == {"project_id": "proj_site", "by": "model"}
    assert mgr.config_for == ["proj_scratch"]          # Quick Tasks' credential
    (messages, disable_reasoning), = prov.calls
    assert disable_reasoning is True
    system, user = messages
    # Candidates only: no Quick Tasks, no excluded project; task last.
    assert "proj_site" in system["content"] and "proj_app" in system["content"]
    assert "proj_hidden" not in system["content"] and "proj_scratch" not in system["content"]
    assert user["role"] == "user" and user["content"].endswith("update the download page")
    assert system["content"].index("newest") < system["content"].index("old")


@pytest.mark.parametrize("reply", [
    '```json\n{"project": "proj_app"}\n```',
    'Sure! Here is my answer: {"project": "proj_app"} — hope it helps.',
    '{"thinking": "x"} {"project": "proj_app"}',
])
@pytest.mark.asyncio
async def test_lenient_parse(projects, reply):
    body = await _route(_client(projects, FakeManager(FakeProvider(reply))))
    assert body == {"project_id": "proj_app", "by": "model"}


@pytest.mark.parametrize("reply, reason", [
    ('{"project": null}', "model chose no project"),
    ('{"project": "proj_hidden"}', "unknown project id"),
    ('{"project": "proj_scratch"}', "unknown project id"),
    ("I think the app project.", "no JSON object"),
    ("", "empty reply"),
])
@pytest.mark.asyncio
async def test_unmatched_replies_fall_back_to_quick_tasks(projects, reply, reason):
    body = await _route(_client(projects, FakeManager(FakeProvider(reply))))
    assert body["project_id"] == "proj_scratch" and body["by"] == "fallback"
    assert reason in body["reason"]


@pytest.mark.asyncio
async def test_keyless_falls_back_without_a_call(projects):
    prov = FakeProvider('{"project": "proj_site"}')
    body = await _route(_client(projects, FakeManager(prov, ready=False)))
    assert body["by"] == "fallback" and body["project_id"] == "proj_scratch"
    assert prov.calls == []


@pytest.mark.asyncio
async def test_model_error_falls_back(projects):
    body = await _route(_client(projects, FakeManager(FakeProvider(exc=RuntimeError("boom")))))
    assert body["by"] == "fallback" and "RuntimeError" in body["reason"]


@pytest.mark.asyncio
async def test_timeout_falls_back(projects, monkeypatch):
    monkeypatch.setattr(wb, "ROUTE_TIMEOUT_S", 0.05)
    body = await _route(_client(projects, FakeManager(FakeProvider("{}", delay=1.0))))
    assert body["by"] == "fallback" and "timed out" in body["reason"]


@pytest.mark.asyncio
async def test_no_candidates_or_empty_text_falls_back(tmp_path):
    only_scratch = [_project(tmp_path, "proj_scratch", is_scratch=True)]
    prov = FakeProvider('{"project": "x"}')
    body = await _route(_client(only_scratch, FakeManager(prov)))
    assert body["by"] == "fallback" and prov.calls == []
    body = await _route(_client(only_scratch, FakeManager(prov)), text="   ")
    assert body["by"] == "fallback"


def test_project_block_caps(tmp_path):
    long_goals = "g" * 5000
    state = "\n".join(f"line {i} " + "s" * 30 for i in range(100))
    p = _project(tmp_path, "proj_big", goals=long_goals, state=state)
    block = wb.project_block(p, ["a", "b", "c", "d"])
    goals_part = block.split("Goals:\n", 1)[1].split("\n", 1)[0]
    assert len(goals_part) == wb.GOALS_CAP
    state_part = block.split("State (head):\n", 1)[1].split("\nLatest sessions", 1)[0]
    assert len(state_part) <= wb.STATE_CAP
    assert len(state_part.splitlines()) <= wb.STATE_LINES
    assert block.endswith("Latest sessions: a | b | c")
    assert block.startswith("## Project id: proj_big")


def test_project_block_missing_files(tmp_path):
    p = _project(tmp_path, "proj_bare")
    block = wb.project_block(p, [])
    assert "Goals:" not in block and "State" not in block
    assert os.path.isdir(p["workspace"])


@pytest.mark.asyncio
async def test_model_route_reports_the_call_tokens(projects):
    """The routing call's token use rides on the answer (additive), so a smoke
    or the coordinator can measure the cost of one route without daemon logs."""
    class UsageProvider(FakeProvider):
        async def complete(self, messages, tools=None, *, disable_reasoning=False):
            self.calls.append((messages, disable_reasoning))
            return SimpleNamespace(text='{"project": "proj_app"}',
                                   usage=SimpleNamespace(input_tokens=812, output_tokens=9))
    body = await _route(_client(projects, FakeManager(UsageProvider())))
    assert body == {"project_id": "proj_app", "by": "model",
                    "input_tokens": 812, "output_tokens": 9}
