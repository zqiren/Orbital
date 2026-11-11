# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Spec 138 onboarding routes over a staged history home: the scan status,
the extended importable-projects payload (candidates + groups) and the batch
import (deferred root seed, orbital/ at import since spec 160, group
membership, hidden agents, skip reasons)."""

from __future__ import annotations

import json
import os
import subprocess
import time
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from agent_os.api.routes import agents_v2
from agent_os.api.routes import onboarding as onboarding_routes
from agent_os.daemon_v2.project_store import ProjectStore
from agent_os.onboarding.history import HistoryScanner, clear_git_cache
from agent_os.onboarding.history_paths import HistoryEnv
from agent_os.onboarding.sources import claude_code


def _transcript(path, sid, cwd, first, *, writes=(), ts="2026-10-20T10:00:00Z"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    rows = [{"type": "user", "sessionId": sid, "cwd": cwd, "timestamp": ts,
             "entrypoint": "cli", "message": {"role": "user", "content": first}}]
    for w in writes:
        rows.append({"type": "assistant", "sessionId": sid, "cwd": cwd, "timestamp": ts,
                     "message": {"model": "m", "content": [
                         {"type": "tool_use", "id": "t", "name": "Write",
                          "input": {"file_path": w, "content": "x"}}]}})
        rows.append({"type": "user", "sessionId": sid, "cwd": cwd, "timestamp": ts,
                     "message": {"content": [{"type": "tool_result", "tool_use_id": "t"}]}})
    rows.append({"type": "assistant", "sessionId": sid, "cwd": cwd, "timestamp": ts,
                 "message": {"model": "m", "content": [{"type": "text", "text": "done"}]}})
    with open(path, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")


class _Settings:
    def __init__(self, hidden=()):
        self.hidden = list(hidden)

    def get(self):
        return SimpleNamespace(sub_agents_hidden=self.hidden)


@pytest.fixture
def staged(tmp_path, monkeypatch):
    home = tmp_path / "home"
    repo = home / "code" / "app"
    decks = home / "decks"
    for d in (home / "Desktop", repo, decks):
        d.mkdir(parents=True)
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    clear_git_cache()
    projects = home / ".claude" / "projects"
    _transcript(projects / "app" / "a1.jsonl", "a1", str(repo), "add the settings page")
    deck = str(decks / "Q3.pptx")
    _transcript(projects / "desk" / "d1.jsonl", "d1", str(home / "Desktop"),
                "make the quarterly deck please", writes=[deck])
    _transcript(projects / "desk" / "d2.jsonl", "d2", str(home / "Desktop"),
                "update the slides with numbers", writes=[deck])
    monkeypatch.setenv("ORBITAL_HISTORY_HOME", str(home))
    monkeypatch.delenv("ORBITAL_CODEX_HOME", raising=False)
    return home


@pytest.fixture
def app_env(staged, tmp_path):
    store = ProjectStore(str(tmp_path / "data"))
    settings = _Settings(hidden=["gemini-cli"])
    agents_v2.configure(project_store=store, agent_manager=MagicMock(),
                        ws_manager=MagicMock(), settings_store=settings)
    scanner = HistoryScanner(str(tmp_path / "data"), home=str(staged),
                             sources=[claude_code.SOURCE], env=HistoryEnv.for_home(str(staged)))
    onboarding_routes.configure(project_store=store, scanner=scanner)
    app = FastAPI()
    app.include_router(agents_v2.router)
    app.include_router(onboarding_routes.router)
    app.include_router(onboarding_routes.projects_router)
    yield SimpleNamespace(client=TestClient(app), store=store, scanner=scanner,
                          home=staged, settings=settings)
    onboarding_routes.configure()


def _ready(client):
    deadline = time.time() + 10
    while time.time() < deadline:
        body = client.get("/api/v2/onboarding/history").json()
        if body["status"] == "ready":
            return body
        time.sleep(0.05)
    raise AssertionError("scan never finished")


def test_history_status_starts_the_scan_and_reports_counts(app_env):
    body = _ready(app_env.client)
    assert body == {"status": "ready", "sessions_total": 3, "by_agent": {"claude-code": 3},
                    "notes_by_agent": {"claude-code": 0}, "notes_by_folder": []}


def test_importable_projects_carries_candidates_and_groups(app_env):
    _ready(app_env.client)
    body = app_env.client.get("/api/v2/onboarding/importable-projects").json()
    assert body["status"] == "ready"
    (cand,) = body["candidates"]  # the Desktop is never a project
    assert cand["name"] == "app" and cand["is_git"] is True and cand["minor"] is False
    assert cand["by_agent"] == {"claude-code": 1} and cand["kind"] == "folder"
    (group,) = body["groups"]
    assert group["session_count"] == 2
    assert group["suggested_folder"] == str(app_env.home / "decks")
    assert group["id"].startswith("grp_")
    assert {m["native_id"] for m in group["members"]} == {"d1", "d2"}
    assert {r["kind"] for r in group["reasons"]} == {"same_files"}


def test_import_creates_deferred_projects_with_members_and_hidden_agents(app_env):
    _ready(app_env.client)
    body = app_env.client.get("/api/v2/onboarding/importable-projects").json()
    folder = body["candidates"][0]["path"]
    gid = body["groups"][0]["id"]
    decks = str(app_env.home / "decks")
    r = app_env.client.post("/api/v2/onboarding/import", json={
        "folders": [folder, str(app_env.home / "gone")],
        "groups": [{"id": gid, "workspace": decks}, {"id": "grp_nope", "workspace": decks}],
    })
    assert r.status_code == 200, r.text
    out = r.json()
    assert [(c["name"], c["from"]) for c in out["created"]] == [("app", "folder"), ("decks", "group")]
    assert {(s["path"], s["reason"]) for s in out["skipped"]} == {
        (str(app_env.home / "gone"), "missing"), (decks, "invalid")}
    for c in out["created"]:
        p = app_env.store.get_project(c["project_id"])
        assert p["deferred_seed"] is True
        assert p["disabled_sub_agents"] == ["gemini-cli"]
        # Spec 160 §5.1: orbital/ is written at import; the root AGENTS.md
        # (the only write outside orbital/) still waits for first use.
        assert not os.path.exists(os.path.join(c["workspace"], "AGENTS.md"))
        # 1109 Task B: the skills step runs at import but bundles nothing.
        assert p["default_skills_reconciled"] is True
        assert p["orbital_seeded"] is True
        assert p["import_scan"] == "pending"
    group_project = app_env.store.get_project(out["created"][1]["project_id"])
    assert sorted(m["native_id"] for m in group_project["imported_sessions"]) == ["d1", "d2"]
    assert all(m["agent"] == "claude-code" for m in group_project["imported_sessions"])
    # Imported now: the folder is marked and the group is no longer offered.
    again = app_env.client.get("/api/v2/onboarding/importable-projects").json()
    assert again["candidates"][0]["already_imported"] is True
    assert again["groups"] == []
    r2 = app_env.client.post("/api/v2/onboarding/import", json={"folders": [folder]})
    assert r2.json() == {"created": [], "skipped": [{"path": folder, "reason": "already_imported"}]}


def test_import_name_collision_is_uniquified(app_env, tmp_path):
    _ready(app_env.client)
    other = tmp_path / "elsewhere" / "app"
    other.mkdir(parents=True)
    app_env.client.post("/api/v2/projects", json={"name": "app", "workspace": str(other)})
    folder = app_env.client.get("/api/v2/onboarding/importable-projects").json()["candidates"][0]["path"]
    out = app_env.client.post("/api/v2/onboarding/import", json={"folders": [folder]}).json()
    assert out["created"][0]["name"] == "app-2"


def test_create_project_seeds_disabled_sub_agents_from_hidden(app_env, tmp_path):
    ws = tmp_path / "plain"
    ws.mkdir()
    r = app_env.client.post("/api/v2/projects", json={"name": "plain", "workspace": str(ws)})
    assert r.status_code == 201
    p = app_env.store.get_project(r.json()["project_id"])
    assert p["disabled_sub_agents"] == ["gemini-cli"]
    assert "deferred_seed" not in p
    assert os.path.isfile(ws / "AGENTS.md")  # the normal path still seeds at once
    # An explicit per-project list wins.
    ws2 = tmp_path / "plain2"
    ws2.mkdir()
    r2 = app_env.client.post("/api/v2/projects", json={
        "name": "plain2", "workspace": str(ws2), "disabled_sub_agents": []})
    assert app_env.store.get_project(r2.json()["project_id"])["disabled_sub_agents"] == []


def test_a_folder_with_history_from_after_the_scan_is_listed_when_import_opens(app_env):
    """1111 keyed smoke D28: the history scan ran once per daemon life, so a
    folder the user worked in with Claude Code after the daemon started never
    reached the sidebar Import. Opening Import re-reads what changed."""
    _ready(app_env.client)
    first = app_env.client.get("/api/v2/onboarding/importable-projects").json()
    fresh = app_env.home / "code" / "later"
    fresh.mkdir(parents=True)
    _transcript(app_env.home / ".claude" / "projects" / "later" / "l1.jsonl", "l1",
                str(fresh), "draft the onboarding email", ts="2026-11-10T10:00:00Z")
    assert str(fresh) not in {c["path"] for c in first["candidates"]}

    body = app_env.client.get("/api/v2/onboarding/importable-projects").json()
    assert body["status"] == "ready"
    rows = {c["path"]: c for c in body["candidates"]}
    assert str(fresh) in rows and rows[str(fresh)]["session_count"] == 1
    assert app_env.client.get("/api/v2/onboarding/history").json()["sessions_total"] == 4
