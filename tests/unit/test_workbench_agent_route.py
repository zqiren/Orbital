# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""One-page Welcome + Workbench landing §5 (2026-11-23): a Workbench task
sent WITHOUT an API key is routed by the first signed-in agent that can
answer a one-off question without leaving a session behind (Claude Code over
the SDK, Codex over the app-server). One prompt and one parser shared with
the keyed route; a timeout or failure lands in Quick Tasks, as a keyless
send did before. The sub-agent manager is faked; the one-off ask itself is
covered in ``test_ask_once.py``.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import httpx
import pytest
from fastapi import FastAPI

from agent_os.api.routes import workbench as wb


class FakeStore:
    def __init__(self, projects):
        self._p = {p["project_id"]: p for p in projects}

    def list_projects(self):
        return list(self._p.values())

    def get_project(self, pid):
        return self._p.get(pid)

    def find_scratch_project(self):
        return next((p for p in self._p.values() if p.get("is_scratch")), None)


class KeylessManager:
    """No key anywhere; ``available`` = the ready workers, in registry order."""

    def __init__(self, available):
        self.available = list(available)
        self.asked_for = []

    def manager_ready(self, pid):
        return False

    def available_worker_slugs(self, pid, project=None, wait=True):
        self.asked_for.append(pid)
        return list(self.available)

    def list_sessions(self, pid, include_memory=False):
        return []


class FakeSubAgents:
    def __init__(self, reply='{"project": "proj_site"}', *, exc=None, delay=0.0,
                 capable=("claude-code", "codex")):
        self.reply, self.exc, self.delay = reply, exc, delay
        self.capable = set(capable)
        self.asks = []

    def can_ask_once(self, slug):
        return slug in self.capable

    async def ask_once(self, slug, prompt, *, cwd):
        self.asks.append((slug, prompt, cwd))
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.exc:
            raise self.exc
        return self.reply

    def running_session_ids(self, pid):
        return set()


def _project(tmp_path, pid, *, goals=None, **extra):
    ws = tmp_path / pid
    (ws / "orbital" / "instructions").mkdir(parents=True, exist_ok=True)
    if goals is not None:
        (ws / "orbital" / "instructions" / "project_goals.md").write_text(goals, encoding="utf-8")
    return {"project_id": pid, "name": f"name-{pid}", "workspace": str(ws), **extra}


@pytest.fixture
def projects(tmp_path):
    return [
        _project(tmp_path, "proj_site", goals="Marketing website and downloads."),
        _project(tmp_path, "proj_app", goals="The desktop app."),
        _project(tmp_path, "proj_scratch", is_scratch=True),
    ]


def _client(projects, manager, subs):
    wb.configure(FakeStore(projects), manager, None, sub_agent_manager=subs)
    app = FastAPI()
    app.include_router(wb.router)
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t")


async def _route(client, text="update the download page"):
    r = await client.post("/api/v2/workbench/route", json={"text": text})
    assert r.status_code == 200
    return r.json()


@pytest.mark.asyncio
async def test_keyless_routes_through_the_first_ready_agent(projects):
    subs = FakeSubAgents()
    mgr = KeylessManager(["kimi-code", "claude-code", "codex"])
    body = await _route(_client(projects, mgr, subs))
    assert body == {"project_id": "proj_site", "by": "agent", "agent": "claude-code"}
    (slug, prompt, cwd), = subs.asks
    assert slug == "claude-code"
    # The router's readiness is Quick Tasks' (where a keyless send lands).
    assert mgr.asked_for == ["proj_scratch"]
    # Neutral folder: never a project's own workspace.
    assert all(cwd != p["workspace"] for p in projects)


@pytest.mark.asyncio
async def test_the_prompt_is_the_keyed_route_messages_as_one_text(projects):
    subs = FakeSubAgents()
    await _route(_client(projects, KeylessManager(["codex"]), subs), "fix the download button")
    (_slug, prompt, _cwd), = subs.asks
    names = [[], []]
    blocks = [wb.project_block(p, n) for p, n in zip(projects[:2], names)]
    messages = wb.build_route_messages(blocks, "fix the download button")
    assert prompt == wb.route_prompt(messages)
    assert prompt.startswith(wb.ROUTE_INSTRUCTION)
    assert prompt.endswith("Task:\nfix the download button")
    assert "## Project id: proj_site" in prompt and "proj_scratch" not in prompt


@pytest.mark.asyncio
async def test_same_prompt_for_either_agent(projects):
    a, b = FakeSubAgents(), FakeSubAgents()
    await _route(_client(projects, KeylessManager(["claude-code"]), a))
    await _route(_client(projects, KeylessManager(["codex"]), b))
    assert a.asks[0][1] == b.asks[0][1]
    assert (a.asks[0][0], b.asks[0][0]) == ("claude-code", "codex")


@pytest.mark.asyncio
async def test_an_agent_that_cannot_answer_once_is_skipped(projects):
    subs = FakeSubAgents(capable=("codex",))
    body = await _route(_client(projects, KeylessManager(["claude-code", "codex"]), subs))
    assert body["agent"] == "codex"


@pytest.mark.asyncio
async def test_no_capable_agent_keeps_the_quick_tasks_fallback(projects):
    subs = FakeSubAgents(capable=())
    body = await _route(_client(projects, KeylessManager(["kimi-code"]), subs))
    assert body == {"project_id": "proj_scratch", "by": "fallback", "reason": "no API key"}
    assert subs.asks == []


@pytest.mark.asyncio
async def test_no_sub_agent_manager_keeps_the_quick_tasks_fallback(projects):
    body = await _route(_client(projects, KeylessManager(["claude-code"]), None))
    assert body == {"project_id": "proj_scratch", "by": "fallback", "reason": "no API key"}


@pytest.mark.asyncio
async def test_timeout_lands_in_quick_tasks(projects, monkeypatch):
    monkeypatch.setattr(wb, "AGENT_ROUTE_BUDGET_S", 0.05)
    subs = FakeSubAgents(delay=1.0)
    body = await _route(_client(projects, KeylessManager(["claude-code"]), subs))
    assert body == {"project_id": "proj_scratch", "by": "fallback", "reason": "agent timeout"}
    assert len(subs.asks) == 1  # one attempt, no retry


def test_agent_route_budget_is_15_seconds():
    assert wb.AGENT_ROUTE_BUDGET_S == 15.0


@pytest.mark.asyncio
@pytest.mark.parametrize("subs", [
    FakeSubAgents(exc=RuntimeError("cli died")),
    FakeSubAgents(reply="I think the site project"),
    FakeSubAgents(reply='{"project": "proj_nope"}'),
    FakeSubAgents(reply=None),
])
async def test_a_failed_or_unusable_answer_lands_in_quick_tasks(projects, subs):
    body = await _route(_client(projects, KeylessManager(["claude-code"]), subs))
    assert body == {"project_id": "proj_scratch", "by": "fallback", "reason": "agent failed"}
    assert len(subs.asks) == 1


@pytest.mark.asyncio
async def test_the_agent_choosing_no_project_goes_to_quick_tasks(projects):
    subs = FakeSubAgents(reply='```json\n{"project": null}\n```')
    body = await _route(_client(projects, KeylessManager(["claude-code"]), subs))
    assert body == {"project_id": "proj_scratch", "by": "fallback",
                    "reason": "model chose no project"}


@pytest.mark.asyncio
async def test_a_keyed_install_never_asks_an_agent(projects):
    class Keyed(KeylessManager):
        def manager_ready(self, pid):
            return True

        def _build_agent_config_from_project(self, pid):
            return SimpleNamespace()

        def _build_llm_providers(self, config):
            class P:
                async def complete(self, messages, tools=None):
                    return SimpleNamespace(text='{"project": "proj_app"}', usage=None)
            return ("main", [], P(), None)

    subs = FakeSubAgents()
    body = await _route(_client(projects, Keyed(["claude-code"]), subs))
    assert body == {"project_id": "proj_app", "by": "model"}
    assert subs.asks == []


@pytest.mark.asyncio
async def test_tasks_route_stamps_routed_by_agent(projects, monkeypatch):
    seen = {}

    async def fake_spawn(pid, content, **kwargs):
        seen.update(pid=pid, **kwargs)
        return {"session_id": "s1", "agent": "claude-code", "inject": {"status": "started"}}

    async def no_refusal(pid, agent):
        return None

    monkeypatch.setattr(wb, "_spawn_seeded", fake_spawn)
    monkeypatch.setattr(wb, "_refuse_keyless_manager", no_refusal)
    client = _client(projects, KeylessManager(["claude-code"]), FakeSubAgents())
    r = await client.post("/api/v2/workbench/tasks", json={"text": "update the download page"})
    assert r.status_code == 200, r.text
    assert r.json()["by"] == "agent" and r.json()["project_id"] == "proj_site"
    assert seen["routed_by"] == "agent"
