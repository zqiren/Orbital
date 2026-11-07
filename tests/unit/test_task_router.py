# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Spec 154 §5.4 — ``POST /api/v2/workbench/route``: one model call on send
and a lenient parse. The provider is faked.

2026-11-07: a call that fails (error, timeout, empty or unusable reply) is
retried once inside one overall budget, then answers a failure shape with no
project (the Workbench asks the user to pick one) instead of silently landing
in Quick Tasks. A deliberate ``{"project": null}`` and a keyless install still
go to Quick Tasks."""

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
    """``reply``/``exc``/``delay`` for every call, or ``script``: one
    ``(reply, exc, delay)`` per call, in order."""

    def __init__(self, reply=None, *, exc=None, delay=0.0, script=None):
        self.reply, self.exc, self.delay = reply, exc, delay
        self.script = list(script) if script else None
        self.calls = []
        self.sdk_retries = None

    def set_sdk_max_retries(self, n):
        self.sdk_retries = n

    async def complete(self, messages, tools=None, *, disable_reasoning=False):
        self.calls.append((messages, disable_reasoning))
        reply, exc, delay = (self.script.pop(0) if self.script
                             else (self.reply, self.exc, self.delay))
        if delay:
            await asyncio.sleep(delay)
        if exc:
            raise exc
        return SimpleNamespace(text=reply, usage=None)


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


@pytest.fixture(autouse=True)
def _no_retry_pause(monkeypatch):
    monkeypatch.setattr(wb, "ROUTE_RETRY_DELAY_S", 0)


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
    # Thinking stays at the model's default (2026-11-07): turning it off
    # saved 1-2 s at best and broke routing whenever a provider's model
    # stopped accepting the catalog's off-switch (glm-5.2 on OpenCode Go).
    assert disable_reasoning is False
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


@pytest.mark.asyncio
async def test_model_choosing_no_project_goes_to_quick_tasks_without_retry(projects):
    prov = FakeProvider('{"project": null}')
    body = await _route(_client(projects, FakeManager(prov)))
    assert body["project_id"] == "proj_scratch" and body["by"] == "fallback"
    assert "model chose no project" in body["reason"]
    assert len(prov.calls) == 1


@pytest.mark.parametrize("reply, code", [
    ('{"project": "proj_hidden"}', "unknown_project"),
    ('{"project": "proj_scratch"}', "unknown_project"),
    ("I think the app project.", "no_json"),
    ("", "empty_reply"),
])
@pytest.mark.asyncio
async def test_unusable_replies_retry_once_then_fail_without_a_project(projects, reply, code):
    prov = FakeProvider(reply)
    body = await _route(_client(projects, FakeManager(prov)))
    assert body == {"project_id": None, "by": "failed", "reason": code, "attempts": 2}
    assert len(prov.calls) == 2


@pytest.mark.asyncio
async def test_keyless_falls_back_without_a_call(projects):
    prov = FakeProvider('{"project": "proj_site"}')
    body = await _route(_client(projects, FakeManager(prov, ready=False)))
    assert body["by"] == "fallback" and body["project_id"] == "proj_scratch"
    assert prov.calls == []


@pytest.mark.asyncio
async def test_model_error_retries_once_then_fails(projects, monkeypatch):
    monkeypatch.setattr(wb, "ROUTE_RETRY_DELAY_S", 0)
    prov = FakeProvider(exc=RuntimeError("boom"))
    body = await _route(_client(projects, FakeManager(prov)))
    assert body == {"project_id": None, "by": "failed", "reason": "call_failed", "attempts": 2}
    assert len(prov.calls) == 2


@pytest.mark.asyncio
async def test_a_failed_first_call_is_retried_and_the_retry_routes(projects, monkeypatch):
    """The 11-06 case: a 429 (here any call error) on the first call."""
    monkeypatch.setattr(wb, "ROUTE_RETRY_DELAY_S", 0)
    prov = FakeProvider(script=[(None, RuntimeError("429 rate limited"), 0),
                                ('{"project": "proj_app"}', None, 0)])
    body = await _route(_client(projects, FakeManager(prov)))
    assert body == {"project_id": "proj_app", "by": "model"}
    assert len(prov.calls) == 2


@pytest.mark.asyncio
async def test_the_route_call_has_no_sdk_retries(projects):
    """The SDK's own backoff (max_retries=2) silently ate the budget on
    11-06; our single retry replaces it."""
    prov = FakeProvider('{"project": "proj_app"}')
    await _route(_client(projects, FakeManager(prov)))
    assert prov.sdk_retries == 0


@pytest.mark.asyncio
async def test_timeout_uses_the_whole_budget_without_a_retry(projects, monkeypatch):
    """A call that hangs gets the whole budget; no retry starts after it,
    since too little budget is left."""
    monkeypatch.setattr(wb, "ROUTE_BUDGET_S", 0.1)
    monkeypatch.setattr(wb, "ROUTE_RETRY_MIN_S", 0.05)
    prov = FakeProvider("{}", delay=1.0)
    body = await _route(_client(projects, FakeManager(prov)))
    assert body == {"project_id": None, "by": "failed", "reason": "timeout", "attempts": 1}
    assert len(prov.calls) == 1


@pytest.mark.asyncio
async def test_retry_only_starts_with_enough_budget_left(projects, monkeypatch):
    """A slow failure that leaves less than the retry minimum: no retry."""
    monkeypatch.setattr(wb, "ROUTE_BUDGET_S", 0.3)
    monkeypatch.setattr(wb, "ROUTE_RETRY_MIN_S", 0.25)
    monkeypatch.setattr(wb, "ROUTE_RETRY_DELAY_S", 0)
    prov = FakeProvider(script=[(None, RuntimeError("502"), 0.1),
                                ('{"project": "proj_app"}', None, 0)])
    body = await _route(_client(projects, FakeManager(prov)))
    assert body["by"] == "failed" and body["attempts"] == 1
    assert len(prov.calls) == 1


@pytest.mark.asyncio
async def test_the_retry_is_bounded_by_what_is_left_of_the_budget(projects, monkeypatch):
    monkeypatch.setattr(wb, "ROUTE_BUDGET_S", 0.3)
    monkeypatch.setattr(wb, "ROUTE_RETRY_MIN_S", 0.05)
    monkeypatch.setattr(wb, "ROUTE_RETRY_DELAY_S", 0)
    prov = FakeProvider(script=[("", None, 0), ("{}", None, 5.0)])
    loop = asyncio.get_running_loop()
    started = loop.time()
    body = await _route(_client(projects, FakeManager(prov)))
    assert loop.time() - started < 1.0
    assert body == {"project_id": None, "by": "failed", "reason": "timeout", "attempts": 2}


def test_route_budget_is_45_seconds():
    assert wb.ROUTE_BUDGET_S == 45.0


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


FORMAT_HEADER = (
    "<!--format PROJECT_STATE is what is true NOW: current focus, in-progress work, "
    "blockers, next steps. One entry = one `- ` bullet line that stands on its own; "
    + "optional detail goes on indented lines under it. " * 30 + "-->"
)


def test_project_block_state_head_is_real_state(tmp_path):
    """The format header (~1,259 chars on every PROJECT_STATE.md) used to
    fill the whole 1,200-char head; it and the mem-id comments are dropped
    before the cap, so the head is the project's actual state."""
    assert len(FORMAT_HEADER) > wb.STATE_CAP
    state = (
        FORMAT_HEADER + "\n# PROJECT_STATE\n\n## Now\n"
        "- Recording Orbital app issues and features into BACKLOG.md\n"
        "  <!--mem id:852134 created:2026-11-04 touched:2026-11-04-->\n"
        "  - detail: sidebar error badges are project-level only\n"
        "- Next: triage the session-list items\n"
        "  <!--mem id:852135 created:2026-11-05 touched:2026-11-06-->\n"
    )
    p = _project(tmp_path, "proj_log", goals="Feature log.", state=state)
    block = wb.project_block(p, [])
    head = block.split("State (head):\n", 1)[1]
    assert "<!--" not in head and "format" not in head
    assert "Recording Orbital app issues and features into BACKLOG.md" in head
    assert "sidebar error badges" in head
    assert "Next: triage the session-list items" in head


def test_project_block_state_of_only_a_header_is_left_out(tmp_path):
    p = _project(tmp_path, "proj_new", goals="New.", state=FORMAT_HEADER + "\n# PROJECT_STATE\n")
    block = wb.project_block(p, [])
    assert "format" not in block


@pytest.mark.asyncio
async def test_recent_sessions_leave_out_memory_and_trigger_runs(projects):
    prov = FakeProvider('{"project": "proj_site"}')
    mgr = FakeManager(prov, sessions={"proj_site": [
        {"name": "Memory", "origin": "memory", "last_activity_at": "2026-11-06"},
        {"name": "[Triggered by schedule: nightly]", "trigger_type": "schedule",
         "last_activity_at": "2026-11-05"},
        {"name": "watch run", "trigger_type": "file_watch", "last_activity_at": "2026-11-05"},
        {"name": "fix the download link", "origin": "chat", "last_activity_at": "2026-11-04"},
    ]})
    await _route(_client(projects, mgr))
    system = prov.calls[0][0][0]["content"]
    assert "fix the download link" in system
    assert "Memory" not in system.split("proj_site", 1)[1].split("## Project id", 1)[0]
    assert "Triggered by" not in system and "watch run" not in system


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


@pytest.mark.parametrize("sdk", ["openai", "anthropic"])
def test_llm_provider_sdk_retries_can_be_turned_off(sdk):
    from agent_os.agent.providers.openai_compat import LLMProvider
    p = LLMProvider("m", "sk-test", "http://127.0.0.1:9", sdk=sdk)
    client = p._openai_client if sdk == "openai" else p._anthropic_client
    assert client.max_retries == 2
    p.set_sdk_max_retries(0)
    client = p._openai_client if sdk == "openai" else p._anthropic_client
    assert client.max_retries == 0
    p.update_api_key("sk-other")
    client = p._openai_client if sdk == "openai" else p._anthropic_client
    assert client.max_retries == 0
