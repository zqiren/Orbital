# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Spec 162 — ``POST /api/v2/workbench/tasks``: the Workbench composer for
scripts. One call routes (when no project is given), mints a session and
sends the text through the inject route, in the composer's order.

Both routers are mounted over an in-process ASGI transport; the agent manager,
the sub-agent manager, the setup engine and the routing provider are faked.
The endpoint calls the real ``/agents/{pid}/new-session`` and
``/agents/{pid}/inject`` handlers, so a targeted send goes through the real
pinned-dispatch branch (persist the user row, then ``send()``).
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock

import httpx
import pytest
from fastapi import FastAPI

from agent_os.api.routes import agents_v2
from agent_os.api.routes import workbench as wb

NOW = datetime(2026, 11, 9, 12, 0, 0, tzinfo=timezone.utc)


class FakeStore:
    def __init__(self, projects):
        self._p = {p["project_id"]: p for p in projects}
        self.updates = []

    def list_projects(self):
        return list(self._p.values())

    def get_project(self, pid):
        return self._p.get(pid)

    def update_project(self, pid, updates):
        self.updates.append((pid, updates))
        self._p[pid].update(updates)

    def find_scratch_project(self):
        return next((p for p in self._p.values() if p.get("is_scratch")), None)


class FakeProvider:
    def __init__(self, reply=None, *, exc=None):
        self.reply, self.exc = reply, exc
        self.calls = []

    def set_sdk_max_retries(self, n):
        pass

    async def complete(self, messages, tools=None, *, disable_reasoning=False):
        self.calls.append(messages)
        if self.exc:
            raise self.exc
        return SimpleNamespace(text=self.reply, usage=self.usage)

    usage = None


class FakeManager:
    """Records every call in ``log`` so the order can be asserted."""

    def __init__(self, *, provider=None, ready=True, pinned=None, inject_result="started"):
        self.provider, self.ready = provider, ready
        self.pinned = pinned or {}           # pid -> pinned_target
        self.inject_result = inject_result
        self.log = []
        self.injected = []                   # (pid, content, session_id)
        self.persisted = []                  # (pid, session_id, row)
        self._n = 0

    # routing
    def manager_ready(self, pid):
        return self.ready

    def _build_agent_config_from_project(self, pid):
        return SimpleNamespace(project_id=pid)

    def _build_llm_providers(self, config):
        self.log.append(("route", config.project_id))
        return ("main", [], self.provider, None)

    def list_sessions(self, pid, include_memory=False):
        return []

    # new-session / inject
    async def new_session(self, project_id, *, session_id=None):
        self._n += 1
        sid = f"s_{self._n}"
        self.log.append(("mint", project_id))
        return {"status": "ok", "session_id": sid, "session_uuid": sid,
                "pinned_target": self.pinned.get(project_id)}

    async def inject_message(self, project_id, content, *, session_id=None, nonce=None):
        self.log.append(("inject", project_id))
        self.injected.append((project_id, content, session_id))
        return self.inject_result

    async def persist_mention_message(self, project_id, session_id, user_msg):
        self.log.append(("persist", project_id))
        self.persisted.append((project_id, session_id, user_msg))
        return session_id

    def note_worker_asks_baseline(self, project_id, session_id, workspace):
        pass

    # The pin switcher's list (``available_workers`` on the project row).
    workers = ["claude-code", "codex"]

    def available_worker_slugs(self, project_id, *, project=None):
        return list(self.workers)


class FakeSubAgents:
    def __init__(self, result="Message sent to {handle}."):
        self.result = result
        self.sent = []

    async def send(self, project_id, handle, message, **kwargs):
        self.sent.append((project_id, handle, message, kwargs))
        return self.result.format(handle=handle)

    def last_start_failure_meta(self, project_id, handle):
        return {}


def _project(tmp_path, pid, **extra):
    ws = tmp_path / pid
    (ws / "orbital").mkdir(parents=True, exist_ok=True)
    return {"project_id": pid, "name": f"name-{pid}", "workspace": str(ws), **extra}


@pytest.fixture
def projects(tmp_path):
    return [
        _project(tmp_path, "proj_app"),
        _project(tmp_path, "proj_site"),
        _project(tmp_path, "proj_hidden", workbench_exclude_global=True),
        _project(tmp_path, "proj_scratch", is_scratch=True),
    ]


@pytest.fixture
def rig(projects, monkeypatch):
    """``rig(manager=…)`` → (client, store, manager, sub_agents)."""
    monkeypatch.setattr(wb, "ROUTE_RETRY_DELAY_S", 0)

    def make(manager=None, *, sub_agents=None, project_list=None):
        store = FakeStore(project_list if project_list is not None else projects)
        mgr = manager or FakeManager()
        subs = sub_agents or FakeSubAgents()
        ws = MagicMock()
        agents_v2.configure(store, mgr, ws, sub_agent_manager=subs)
        monkeypatch.setattr(agents_v2, "_pinned_consolidation", MagicMock())
        wb.configure(store, mgr, None, now_fn=lambda: NOW, ws_manager=ws)
        app = FastAPI()
        app.include_router(wb.router)
        app.include_router(agents_v2.router)
        client = httpx.AsyncClient(transport=httpx.ASGITransport(app=app),
                                   base_url="http://t")
        return client, store, mgr, subs

    yield make
    agents_v2.configure(None, None, None)


async def _post(client, body):
    async with client:
        return await client.post("/api/v2/workbench/tasks", json=body)


# --------------------------------------------------------------------------
# project_id given
# --------------------------------------------------------------------------

async def test_given_project_skips_routing_and_sends_to_the_manager(rig):
    prov = FakeProvider('{"project": "proj_site"}')
    client, store, mgr, subs = rig(FakeManager(provider=prov))
    r = await _post(client, {"text": "Fix #123: login 500s", "project_id": "proj_app"})
    assert r.status_code == 200, r.text
    assert r.json() == {"project_id": "proj_app", "session_id": "s_1",
                        "agent": "orbital", "by": "given",
                        "inject": {"status": "started"}}
    assert prov.calls == []                                   # no routing call
    assert mgr.log == [("mint", "proj_app"), ("inject", "proj_app")]
    assert mgr.injected == [("proj_app", "Fix #123: login 500s", "s_1")]
    assert subs.sent == []
    assert store.updates == []                                # the pin never moves


async def test_omitted_agent_uses_the_minted_pinned_target(rig):
    client, store, mgr, subs = rig(FakeManager(pinned={"proj_app": "codex"}))
    r = await _post(client, {"text": "do it", "project_id": "proj_app"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["agent"] == "codex"
    assert body["inject"] == {"status": "Message sent to codex."}
    assert mgr.log == [("mint", "proj_app"), ("persist", "proj_app")]
    (pid, sid, row), = mgr.persisted
    assert (pid, sid) == ("proj_app", "s_1")
    assert row["role"] == "user" and row["content"] == "do it" and row["target"] == "codex"
    (_pid, handle, message, kwargs), = subs.sent
    assert handle == "codex" and message == "do it"
    assert kwargs["session_id"] == "s_1" and kwargs["initiator"] == "user_pinned"
    assert store.updates == []


async def test_agent_orbital_sends_to_the_manager_even_when_the_project_is_pinned(rig):
    client, _store, mgr, subs = rig(FakeManager(pinned={"proj_app": "codex"}))
    r = await _post(client, {"text": "plan it", "project_id": "proj_app", "agent": "orbital"})
    assert r.status_code == 200, r.text
    assert r.json()["agent"] == "orbital"
    assert mgr.injected == [("proj_app", "plan it", "s_1")]
    assert subs.sent == []


async def test_named_worker_is_a_pinned_dispatch(rig):
    client, store, mgr, subs = rig()
    r = await _post(client, {"text": "review it", "project_id": "proj_app",
                             "agent": "claude-code"})
    assert r.status_code == 200, r.text
    assert r.json()["agent"] == "claude-code"
    (_pid, handle, _msg, _kw), = subs.sent
    assert handle == "claude-code"
    assert mgr.injected == []
    assert store.updates == []


async def test_queued_inject_body_passes_through(rig):
    client, *_ = rig(FakeManager(inject_result="queued"))
    r = await _post(client, {"text": "later", "project_id": "proj_app"})
    assert r.status_code == 200
    assert r.json()["inject"] == {"status": "queued"}


async def test_quick_tasks_can_be_named(rig):
    client, _store, mgr, _ = rig()
    r = await _post(client, {"text": "x", "project_id": "proj_scratch"})
    assert r.status_code == 200, r.text
    assert r.json()["project_id"] == "proj_scratch"


# --------------------------------------------------------------------------
# Errors before any mint
# --------------------------------------------------------------------------

async def test_unknown_project_404_before_mint(rig):
    client, _store, mgr, _ = rig()
    r = await _post(client, {"text": "x", "project_id": "nope"})
    assert r.status_code == 404
    assert mgr.log == []


async def test_excluded_project_404_before_mint(rig):
    client, _store, mgr, _ = rig()
    r = await _post(client, {"text": "x", "project_id": "proj_hidden"})
    assert r.status_code == 404
    assert mgr.log == []


@pytest.mark.parametrize("agent", ["kimi", "not-an-agent", "built-in"])
async def test_unavailable_agent_400_before_mint(rig, agent):
    client, _store, mgr, _ = rig()
    r = await _post(client, {"text": "x", "project_id": "proj_app", "agent": agent})
    assert r.status_code == 400, r.text
    assert agent in r.json()["detail"]
    assert mgr.log == []


@pytest.mark.parametrize("text", ["", "   \n "])
async def test_empty_text_422(rig, text):
    client, _store, mgr, _ = rig()
    r = await _post(client, {"text": text, "project_id": "proj_app"})
    assert r.status_code == 422
    assert mgr.log == []


async def test_missing_text_422(rig):
    client, _store, mgr, _ = rig()
    r = await _post(client, {"project_id": "proj_app"})
    assert r.status_code == 422
    assert mgr.log == []


async def test_corrupted_text_422_before_mint(rig):
    client, _store, mgr, _ = rig()
    r = await _post(client, {"text": "bad \ufffd byte", "project_id": "proj_app"})
    assert r.status_code == 422
    assert mgr.log == []


async def test_invalid_attachment_400_before_mint(rig):
    client, _store, mgr, _ = rig()
    r = await _post(client, {"text": "x", "project_id": "proj_app", "attachments": [
        {"path": "uploads/missing.png", "mime": "image/png", "size": 10}]})
    assert r.status_code == 400, r.text
    assert mgr.log == []


# --------------------------------------------------------------------------
# Attachments
# --------------------------------------------------------------------------

async def test_attachments_reach_the_inject_route_unchanged(rig, projects):
    ws = projects[0]["workspace"]
    os.makedirs(os.path.join(ws, "uploads"), exist_ok=True)
    with open(os.path.join(ws, "uploads", "shot.png"), "wb") as f:
        f.write(b"x" * 10)
    client, _store, mgr, _ = rig()
    r = await _post(client, {"text": "see the shot", "project_id": "proj_app", "attachments": [
        {"path": "uploads/shot.png", "mime": "image/png", "size": 10}]})
    assert r.status_code == 200, r.text
    (_pid, content, _sid), = mgr.injected
    assert content.startswith("<attached_files>")
    assert "uploads/shot.png" in content
    assert content.endswith("see the shot")


# --------------------------------------------------------------------------
# project_id omitted: routing exactly as the composer's 自动
# --------------------------------------------------------------------------

async def test_omitted_project_routes_then_mints_then_injects(rig):
    prov = FakeProvider('{"project": "proj_site"}')
    client, _store, mgr, _ = rig(FakeManager(provider=prov))
    r = await _post(client, {"text": "update the download page"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["project_id"] == "proj_site" and body["by"] == "model"
    assert [e[0] for e in mgr.log] == ["route", "mint", "inject"]
    assert mgr.log[1:] == [("mint", "proj_site"), ("inject", "proj_site")]
    # The same routing call the composer's /route makes.
    (messages,) = prov.calls
    assert messages == wb.build_route_messages(
        [wb.project_block(p, []) for p in wb.route_candidates(_store.list_projects())],
        "update the download page")


async def test_routed_token_use_passes_through_like_the_route(rig):
    prov = FakeProvider('{"project": "proj_app"}')
    prov.usage = SimpleNamespace(input_tokens=6100, output_tokens=12)
    client, *_ = rig(FakeManager(provider=prov))
    r = await _post(client, {"text": "x"})
    body = r.json()
    assert (body["input_tokens"], body["output_tokens"]) == (6100, 12)
    assert "reason" not in body


async def test_null_project_id_routes_too(rig):
    prov = FakeProvider('{"project": "proj_app"}')
    client, *_ = rig(FakeManager(provider=prov))
    r = await _post(client, {"text": "x", "project_id": None})
    assert r.status_code == 200
    assert r.json()["by"] == "model" and r.json()["project_id"] == "proj_app"


async def test_keyless_lands_in_quick_tasks(rig):
    client, _store, mgr, _ = rig(FakeManager(ready=False))
    r = await _post(client, {"text": "anything"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["project_id"] == "proj_scratch"
    assert body["by"] == "fallback"
    assert body["reason"] == "no API key"
    assert mgr.injected == [("proj_scratch", "anything", "s_1")]


async def test_routing_failure_sends_nothing(rig):
    prov = FakeProvider(exc=RuntimeError("boom"))
    client, _store, mgr, _ = rig(FakeManager(provider=prov))
    r = await _post(client, {"text": "anything"})
    assert r.status_code == 502, r.text
    detail = r.json()["detail"]
    assert detail["by"] == "failed" and detail["reason"] == "call_failed"
    assert [e[0] for e in mgr.log] == ["route"]


async def test_no_quick_tasks_and_nothing_to_route_404(rig, tmp_path):
    client, _store, mgr, _ = rig(project_list=[_project(tmp_path, "only_hidden",
                                                        workbench_exclude_global=True)])
    r = await _post(client, {"text": "anything"})
    assert r.status_code == 404
    assert mgr.log == []


async def test_routing_sees_attached_filenames_like_the_composer(rig, projects):
    ws = projects[1]["workspace"]
    os.makedirs(os.path.join(ws, "uploads"), exist_ok=True)
    with open(os.path.join(ws, "uploads", "brief.pdf"), "wb") as f:
        f.write(b"x" * 4)
    prov = FakeProvider('{"project": "proj_site"}')
    client, *_ = rig(FakeManager(provider=prov))
    r = await _post(client, {"text": "read this", "attachments": [
        {"path": "uploads/brief.pdf", "mime": "application/pdf", "size": 4}]})
    assert r.status_code == 200, r.text
    (messages,) = prov.calls
    assert messages[-1]["content"].endswith(
        "read this\n\n<attached_files>\n- brief.pdf\n</attached_files>")


# --------------------------------------------------------------------------
# An inject that fails after the mint returns its error (no cleanup).
# --------------------------------------------------------------------------

async def test_failed_pinned_dispatch_returns_the_inject_error(rig):
    client, _store, mgr, subs = rig(sub_agents=FakeSubAgents("Error: codex is not signed in"))
    r = await _post(client, {"text": "x", "project_id": "proj_app", "agent": "codex"})
    assert r.status_code == 404
    assert "not signed in" in r.json()["detail"]
    assert [e[0] for e in mgr.log][:2] == ["mint", "persist"]


# --------------------------------------------------------------------------
# The composer's /route answers exactly as before (one routing function).
# --------------------------------------------------------------------------

async def test_route_endpoint_unchanged(rig):
    prov = FakeProvider('{"project": "proj_site"}')
    client, _store, mgr, _ = rig(FakeManager(provider=prov))
    async with client:
        r = await client.post("/api/v2/workbench/route", json={"text": "x"})
    assert r.json() == {"project_id": "proj_site", "by": "model"}
    assert [e[0] for e in mgr.log] == ["route"]
