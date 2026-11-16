# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Task sources: the REST routes, the Workbench send path and the feed mark."""

from __future__ import annotations

import re
from datetime import datetime, timezone

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from agent_os.api.routes import sources as routes
from agent_os.sources import core as core_mod
from agent_os.sources.adapters import Adapter
from agent_os.sources.core import SendError, SourcesCore
from agent_os.sources.model import Verified, VerifyError
from agent_os.sources.registry import AdapterRegistry
from agent_os.sources.secrets import MemorySecretStore

SECRET = "123456789:AAH-route-secret"


class Tg(Adapter):
    async def verify(self, ctx):
        if ctx.secret("token") == "bad":
            raise VerifyError("rejected", "token", "Telegram rejected this token")
        return Verified(label="@mira_orbital_bot")


@pytest.fixture
def client(tmp_path):
    # Only the fake telegram adapter exists here: no route test can reach a
    # real service whichever adapters this build ships.
    from tests.unit.test_sources_pipeline import MissingRegistry
    core = SourcesCore(str(tmp_path),
                       registry=MissingRegistry({"telegram": Tg()},
                                                missing=("slack", "discord", "feishu", "dingtalk",
                                                         "wecom", "gitlab", "jira", "email",
                                                         "calendar")),
                       secret_store=MemorySecretStore())
    core.load()
    routes.configure(core)
    app = FastAPI()
    app.include_router(routes.router)
    yield TestClient(app)
    routes.configure(None)


def test_catalog_route(client):
    body = client.get("/api/v2/sources/catalog").json()
    types = [t["type"] for t in body["types"]]
    assert types[:12] == ["github", "gitlab", "jira", "email", "calendar", "website",
                          "telegram", "slack", "discord", "feishu", "dingtalk", "wecom"]
    assert body["platform"] in ("darwin", "win32", "linux")


def test_create_get_patch_delete(client):
    r = client.post("/api/v2/sources", json={"type": "telegram", "kinds": {"dm": {}},
                                             "secrets": {"token": SECRET}})
    assert r.status_code == 201, r.text
    src = r.json()
    assert SECRET not in r.text
    assert src["label"] == "@mira_orbital_bot" and src["kinds"] == {"dm": {}}
    sid = src["id"]
    assert client.get("/api/v2/sources").json()["sources"][0]["id"] == sid
    r = client.patch(f"/api/v2/sources/{sid}", json={"on": False, "name": "Ops bot",
                                                     "instruction": "先复现"})
    assert r.status_code == 200
    assert r.json()["on"] is False and r.json()["display_name"] == "Ops bot"
    assert r.json()["instruction"] == "先复现"
    assert client.get(f"/api/v2/sources/{sid}").json()["history"] == []
    assert client.delete(f"/api/v2/sources/{sid}").json() == {"status": "deleted", "id": sid}
    assert client.get(f"/api/v2/sources/{sid}").status_code == 404


def test_verify_failure_shape(client):
    r = client.post("/api/v2/sources", json={"type": "telegram", "secrets": {"token": "bad"}})
    assert r.status_code == 400
    assert r.json()["detail"] == {"code": "verify_failed", "field": "token", "reason": "rejected",
                                  "message": "Telegram rejected this token"}
    assert client.get("/api/v2/sources").json()["sources"] == []


def test_errors(client):
    assert client.post("/api/v2/sources", json={"type": "nope"}).json()["detail"]["code"] == "unknown_type"
    r = client.post("/api/v2/sources", json={"type": "discord", "secrets": {"token": "x"}})
    assert r.status_code == 409 and r.json()["detail"]["code"] == "type_unavailable"
    r = client.post("/api/v2/sources", json={"type": "telegram", "kinds": {"bogus": {}},
                                             "secrets": {"token": "x"}})
    assert r.status_code == 422 and r.json()["detail"]["field"] == "kinds"
    assert client.get("/api/v2/sources/src_ffffffffffff").status_code == 404
    assert client.get("/api/v2/sources/../etc").status_code == 404


def test_register_route_arrives_off_and_new(client):
    r = client.post("/api/v2/sources/register", json={
        "type": "command", "name": "HN", "command": {"run": "echo []", "interval_s": 10}})
    assert r.status_code == 201, r.text
    src = r.json()["source"]
    assert src["on"] is False and src["is_new"] is True
    assert src["registered_by"]["by"] == "outside"
    assert src["command"] == {"run": "echo []", "cwd": None, "interval_s": 60}


def test_skill_text_is_served(client):
    r = client.get("/api/v2/sources/skill")
    assert r.status_code == 200 and "name: orbital-sources" in r.text
    assert "/api/v2/sources/register" in r.text


# ── the send path (spec 162's handler) ─────────────────────────────────────

async def test_sender_rides_create_task_with_project_and_agent_on_auto(monkeypatch):
    from agent_os.api.routes import workbench
    seen = []

    async def fake_create_task(req):
        seen.append(req)
        return {"project_id": "proj_a", "session_id": "s1", "agent": "orbital", "by": "model",
                "inject": {}}
    monkeypatch.setattr(workbench, "create_task", fake_create_task)
    send = routes.workbench_sender()
    out = await send("title\n\nbody", None)
    assert out["session_id"] == "s1"
    assert seen[0].project_id is None and seen[0].agent is None
    assert seen[0].text == "title\n\nbody"


@pytest.mark.parametrize("exc,reason", [
    (HTTPException(502, {"project_id": None, "by": "failed"}), "routing_failed"),
    (HTTPException(400, {"code": "missing_api_key", "message": "x"}), "missing_api_key"),
    (HTTPException(404, "No project"), "no_project"),
    (HTTPException(400, "Agent not available"), "send_failed"),
])
async def test_sender_maps_refusals(monkeypatch, exc, reason):
    from agent_os.api.routes import workbench

    async def fake_create_task(req):
        raise exc
    monkeypatch.setattr(workbench, "create_task", fake_create_task)
    with pytest.raises(SendError) as e:
        await routes.workbench_sender()("t", None)
    assert e.value.reason == reason


# ── the Workbench feed row carries the source mark ─────────────────────────

def test_feed_rows_carry_the_source_mark(tmp_path, monkeypatch):
    from agent_os.api.routes import workbench
    monkeypatch.setattr(core_mod, "_installed", None)
    project = {"project_id": "proj_a", "workspace": str(tmp_path)}
    row = {"session_id": "sess-1", "status": "running", "worker_running": False}
    now = datetime(2026, 11, 16, tzinfo=timezone.utc)
    assert workbench.feed_rows(project, [row], now)[0]["source"] is None

    class Fake:
        def mark_for_session(self, sid):
            return {"source_id": "src_1", "type": "github", "name": "GitHub",
                    "arrival_id": "arr_1"} if sid == "sess-1" else None
    monkeypatch.setattr(core_mod, "_installed", Fake())
    assert workbench.feed_rows(project, [row], now)[0]["source"]["type"] == "github"


def test_command_path_for_reveal(client, tmp_path):
    script = tmp_path / "check.py"
    script.write_text("print('[]')")
    r = client.post("/api/v2/sources/register", json={
        "type": "command", "command": {"run": f"python3 '{script}' --since 1h", "cwd": str(tmp_path)}})
    sid = r.json()["source"]["id"]
    core = routes._core
    import asyncio
    assert asyncio.run(core.command_path(sid)) == str(script.resolve())
    r2 = client.post("/api/v2/sources/register", json={"type": "command", "command": {"run": "echo []"}})
    assert asyncio.run(core.command_path(r2.json()["source"]["id"])) is None
    assert client.post(f"/api/v2/sources/{r2.json()['source']['id']}/reveal").json()["detail"]["code"] == "no_path"
    assert client.post(f"/api/v2/sources/{sid}/reveal",
                       headers={"X-Via-Relay": "true"}).status_code == 403


async def test_sender_names_the_session_from_the_item_title_only(monkeypatch):
    """fe 2026-11-16: the card read "127.0.0.1 is down [Task source: …" — the
    session was named from the whole first message."""
    from agent_os.api.routes import agents_v2, workbench
    renamed = []

    async def fake_create_task(req):
        return {"project_id": req.project_id or "proj_a", "session_id": "s9", "by": "model"}

    async def fake_patch(pid, sid, req):
        renamed.append((pid, sid, req.name))
        return {"status": "patched"}
    monkeypatch.setattr(workbench, "create_task", fake_create_task)
    monkeypatch.setattr(agents_v2, "patch_session", fake_patch)
    send = routes.workbench_sender()
    await send("127.0.0.1 is down\n\n[Task source: Website check · 127.0.0.1 · Site is down]\n…", None)
    await send("#899 Cart total not updating after removing an item from the basket\n\nbody", "proj_b")
    assert renamed[0] == ("proj_a", "s9", "127.0.0.1 is down")
    assert renamed[1][:2] == ("proj_b", "s9")
    assert renamed[1][2].startswith("#899 Cart total") and "\n" not in renamed[1][2]
    assert len(renamed[1][2]) <= 51


async def test_a_failed_rename_does_not_fail_the_send(monkeypatch):
    from agent_os.api.routes import agents_v2, workbench

    async def fake_create_task(req):
        return {"project_id": "proj_a", "session_id": "s9", "by": "model"}

    async def broken_patch(pid, sid, req):
        raise HTTPException(503, "busy")
    monkeypatch.setattr(workbench, "create_task", fake_create_task)
    monkeypatch.setattr(agents_v2, "patch_session", broken_patch)
    out = await routes.workbench_sender()("title\n\nbody", None)
    assert out["session_id"] == "s9"


def test_copy_prompts_carry_the_live_skill_url_and_stay_short(client, monkeypatch):
    monkeypatch.setenv("AGENT_OS_PORT", "8642")
    types = {t["type"]: t for t in client.get("/api/v2/sources/catalog").json()["types"]}
    url = "http://127.0.0.1:8642/api/v2/sources/skill"
    with_prompt = [t for t in types.values() if t["agent_prompt"]]
    assert {t["type"] for t in with_prompt} >= {"telegram", "slack", "email", "discord", "feishu",
                                                  "dingtalk", "wecom", "jira"}
    for t in with_prompt:
        for lang in ("zh", "en"):
            text = t["agent_prompt"][lang]
            assert url in text, (t["type"], lang)
            assert "{skill_url}" not in text
        assert len(t["agent_prompt"]["en"].split()) <= 140, t["type"]
        # zh word estimate: a word is ~2 characters, plus each Latin token
        zh = t["agent_prompt"]["zh"].replace(url, "")
        words = len(re.findall(r"[\u4e00-\u9fff]", zh)) // 2 + len(re.findall(r"[A-Za-z0-9_:\-@./]+", zh))
        assert words <= 140, (t["type"], words)


def test_skill_url_falls_back_to_the_port_the_request_came_in_on(client, monkeypatch):
    monkeypatch.delenv("AGENT_OS_PORT", raising=False)
    body = client.get("/api/v2/sources/catalog").json()
    tg = next(t for t in body["types"] if t["type"] == "telegram")
    # TestClient's server is ("testserver", 80)
    assert "http://127.0.0.1:80/api/v2/sources/skill" in tg["agent_prompt"]["en"]


def test_open_settings_needs_no_source(client, monkeypatch):
    """Calendar access refused at connect (no source exists yet) and a later
    calendar_permission failure both open the same System Settings pane."""
    calls = []
    monkeypatch.setattr(routes.subprocess, "run", lambda argv, **kw: calls.append(argv))
    monkeypatch.setattr(routes.sys, "platform", "darwin")
    r = client.post("/api/v2/sources/open-settings")
    assert r.status_code == 200 and r.json() == {"opened": True}
    assert calls == [["open", routes.CALENDAR_SETTINGS_URL]]
    r = client.post("/api/v2/sources/open-settings", headers={"x-via-relay": "true"})
    assert r.status_code == 403 and r.json()["detail"]["code"] == "local_only"
    monkeypatch.setattr(routes.sys, "platform", "win32")
    assert client.post("/api/v2/sources/open-settings").status_code == 501
    assert len(calls) == 1


def test_a_web_page_cannot_register_or_switch_on_a_check_command(client, monkeypatch):
    """The daemon answers every origin (CORS *), so without this any web page
    open in the user's browser could register a check command and switch it
    on: a shell command run with nobody asked. A browser cannot forge its
    Origin; scripts and agents send none; the app's own page is loopback."""
    monkeypatch.setenv("AGENT_OS_RELAY_URL", "https://relay.example.app")
    body = {"type": "command", "name": "x", "command": {"run": "echo []"}}
    evil = {"origin": "https://evil.example.com"}
    r = client.post("/api/v2/sources/register", json=body, headers=evil)
    assert r.status_code == 403 and r.json()["detail"]["code"] == "origin_refused"
    r = client.post("/api/v2/sources/register", json=body, headers={"origin": "null"})
    assert r.status_code == 403
    # an agent / a script (no Origin)
    sid = client.post("/api/v2/sources/register", json=body).json()["source"]["id"]
    r = client.patch(f"/api/v2/sources/{sid}", json={"on": True}, headers={**evil, "x-via-relay": "true"})
    assert r.status_code == 403
    assert client.get(f"/api/v2/sources/{sid}").json()["is_new"] is True
    # other changes from anywhere still work; the app's own page and the relay may switch it on
    assert client.patch(f"/api/v2/sources/{sid}", json={"name": "y"}, headers=evil).status_code == 200
    for origin in ("http://127.0.0.1:8000", "http://localhost:5173", "https://relay.example.app"):
        r = client.post("/api/v2/sources/register", json=body, headers={"origin": origin})
        assert r.status_code == 201, origin
        nid = r.json()["source"]["id"]
        assert client.patch(f"/api/v2/sources/{nid}", json={"on": True}, headers={"origin": origin}).status_code == 200


@pytest.mark.skipif(__import__("sys").platform == "win32", reason="POSIX shell line")
def test_a_new_check_command_never_runs_until_switched_on(client, tmp_path):
    """Switching it on in the panel is the only way a registered command
    first runs: 再检查 (and every other route) leaves it alone until then."""
    marker = tmp_path / "ran"
    body = {"type": "command", "name": "x", "command": {"run": f"touch {marker}; echo []"}}
    sid = client.post("/api/v2/sources/register", json=body).json()["source"]["id"]
    r = client.post(f"/api/v2/sources/{sid}/check")
    assert r.status_code == 409 and r.json()["detail"]["code"] == "not_on_yet"
    client.patch(f"/api/v2/sources/{sid}", json={"name": "renamed", "instruction": "x"})
    client.post(f"/api/v2/sources/{sid}/reconnect", json={"secrets": {}, "config": {}})
    client.post("/api/v2/sources/choices", json={"source_id": sid, "list": "x", "params": {}})
    assert not marker.exists()
    client.patch(f"/api/v2/sources/{sid}", json={"on": True})
    assert client.post(f"/api/v2/sources/{sid}/check").status_code == 200
    assert marker.exists()
