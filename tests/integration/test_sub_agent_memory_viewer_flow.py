# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Integration tests for the project-level Sub-Agent Memory viewer.

Boots a real FastAPI app and exercises the full GET/PUT round trip, plus
verifies that a sub-agent's MEMORY.md persists across enable/disable
cycles (we never delete the file when a sub-agent is denylisted).
"""

from __future__ import annotations

import os

import pytest
from fastapi.testclient import TestClient

from agent_os.agent.project_paths import ProjectPaths
from agent_os.agents.setup_types import AgentSetupStatus


# ---------------------------------------------------------------------------
# Test fixture
# ---------------------------------------------------------------------------


def _fake_check_all(installed_slugs: list[str]):
    def _fake() -> list[AgentSetupStatus]:
        statuses: list[AgentSetupStatus] = [
            AgentSetupStatus(
                slug="built-in", name="Built-in",
                installed=True, binary_path=None, version=None,
                dependencies_met=True, missing_dependencies=[],
                credentials_configured=True, missing_credentials=[],
                setup_actions=[],
            ),
        ]
        for slug in installed_slugs:
            statuses.append(AgentSetupStatus(
                slug=slug,
                name=slug.replace("-", " ").title(),
                installed=True,
                binary_path="/fake/" + slug,
                version="1.0.0",
                dependencies_met=True,
                missing_dependencies=[],
                credentials_configured=True,
                missing_credentials=[],
                setup_actions=[],
            ))
        return statuses

    return _fake


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("USERPROFILE", str(tmp_path / "home"))
    os.makedirs(str(tmp_path / "home"), exist_ok=True)

    from agent_os.api.app import create_app
    app = create_app(data_dir=str(tmp_path / "data"))

    # Pin the setup-engine output so tests don't depend on local state.
    from agent_os.api.routes import agents_v2 as routes_mod
    routes_mod._setup_engine.check_all = _fake_check_all(
        ["claude-code", "codex"]
    )  # type: ignore[assignment]

    with TestClient(app) as c:
        yield c


def _create_project(client: TestClient, ws: str, *, disabled=None) -> str:
    payload: dict = {
        "name": "memflow-" + os.path.basename(ws),
        "workspace": ws,
        "model": "gpt-4",
        "api_key": "k",
    }
    if disabled is not None:
        payload["disabled_sub_agents"] = disabled
    resp = client.post("/api/v2/projects", json=payload)
    assert resp.status_code == 201, resp.text
    return resp.json()["project_id"]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestMemoryViewerFullFlow:

    def test_memory_viewer_full_flow(self, client, tmp_path):
        """Populate MEMORY.md via PUT, GET it back, PUT updated content,
        verify a re-GET reflects the new content.

        We can't dispatch a real sub-agent in the unit test suite (no
        binary), so the PUT endpoint stands in for the daemon's lazy-
        creation path — the spec explicitly authorizes PUT to create the
        file from scratch. The 'next dispatch sees the new memory'
        clause is exercised by reading the file directly from disk and
        confirming it matches what render_sub_agent_prompt would point
        the sub-agent at.
        """
        ws = str(tmp_path / "ws_full")
        os.makedirs(ws, exist_ok=True)
        pid = _create_project(client, ws, disabled=[])

        # 1. Initial GET: MEMORY.md absent for both agents.
        resp = client.get(f"/api/v2/projects/{pid}/sub-agent-memory")
        assert resp.status_code == 200
        data = resp.json()
        assert {e["agent_slug"] for e in data} == {"claude-code", "codex"}
        for e in data:
            assert e["exists"] is False
            assert e["content"] == ""

        # 2. PUT initial content for claude-code.
        resp = client.put(
            f"/api/v2/projects/{pid}/sub-agent-memory/claude-code",
            json={"content": "initial-content-from-ui\n"},
        )
        assert resp.status_code == 200, resp.text

        # 3. Verify the on-disk path matches the path the dispatcher
        # would resolve for this sub-agent (this is the contract the
        # next dispatch relies on).
        expected = os.path.join(
            ProjectPaths(ws).sub_agent_dir("claude-code"), "MEMORY.md",
        )
        assert os.path.isfile(expected)

        # 4. GET again — claude-code now exists, codex still absent.
        resp = client.get(f"/api/v2/projects/{pid}/sub-agent-memory")
        data = resp.json()
        cc = next(e for e in data if e["agent_slug"] == "claude-code")
        cx = next(e for e in data if e["agent_slug"] == "codex")
        assert cc["exists"] is True
        assert "initial-content-from-ui" in cc["content"]
        assert cx["exists"] is False

        # 5. Edit the content via a second PUT.
        resp = client.put(
            f"/api/v2/projects/{pid}/sub-agent-memory/claude-code",
            json={
                "content": (
                    "always check the API contract before "
                    "implementing endpoints\n"
                ),
            },
        )
        assert resp.status_code == 200

        # 6. The next "dispatch" (here: a re-read of MEMORY.md from the
        # exact path render_sub_agent_prompt points at) must see the
        # user-edited content.
        with open(expected, "r", encoding="utf-8") as f:
            disk_content = f.read()
        assert "always check the API contract" in disk_content
        # The previous content has been overwritten (PUT is replace,
        # not append).
        assert "initial-content-from-ui" not in disk_content


class TestMemoryPersistsAcrossDisableReEnable:

    def test_memory_persists_across_disable_re_enable(self, client, tmp_path):
        """Per spec design: previously-enabled sub-agents' memory
        persists on disk; just hidden from the UI when disabled.
        Re-enabling re-surfaces the existing content."""
        ws = str(tmp_path / "ws_persist")
        os.makedirs(ws, exist_ok=True)
        pid = _create_project(client, ws, disabled=[])

        # 1. PUT content for claude-code.
        resp = client.put(
            f"/api/v2/projects/{pid}/sub-agent-memory/claude-code",
            json={"content": "should-survive-disable\n"},
        )
        assert resp.status_code == 200

        # 2. Disable claude-code on the project.
        resp = client.put(
            f"/api/v2/projects/{pid}",
            json={"disabled_sub_agents": ["claude-code"]},
        )
        assert resp.status_code == 200

        # 3. GET memory: claude-code is absent from the list.
        resp = client.get(f"/api/v2/projects/{pid}/sub-agent-memory")
        slugs = [e["agent_slug"] for e in resp.json()]
        assert "claude-code" not in slugs

        # 4. The file MUST still exist on disk.
        path = os.path.join(
            ProjectPaths(ws).sub_agent_dir("claude-code"), "MEMORY.md",
        )
        assert os.path.isfile(path), (
            "MEMORY.md was deleted when sub-agent was disabled — spec "
            "requires persistence across denylist toggles."
        )

        # 5. Re-enable (clear the denylist).
        resp = client.put(
            f"/api/v2/projects/{pid}",
            json={"disabled_sub_agents": []},
        )
        assert resp.status_code == 200

        # 6. claude-code re-surfaces with the original content intact.
        resp = client.get(f"/api/v2/projects/{pid}/sub-agent-memory")
        data = resp.json()
        cc = next(e for e in data if e["agent_slug"] == "claude-code")
        assert cc["exists"] is True
        assert "should-survive-disable" in cc["content"]
