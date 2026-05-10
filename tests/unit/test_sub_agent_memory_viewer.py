# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Regression tests for the project-level Sub-Agent Memory viewer.

Covers the GET /projects/{pid}/sub-agent-memory listing endpoint and the
PUT /projects/{pid}/sub-agent-memory/{slug} editor endpoint:

- GET returns one entry per available (installed - disabled) sub-agent.
- GET reflects exists=true once MEMORY.md has been created.
- GET marks exists=false for available sub-agents that have never been
  dispatched (no MEMORY.md on disk).
- GET excludes sub-agents on the project's disabled denylist.
- PUT writes raw content (no canonical header) to the sub-agent's
  MEMORY.md, creating parent directories on the way.
- PUT enforces the hard 10 KB cap with a 400 response.
- PUT returns a non-blocking warning when content exceeds the 2 KB
  target.
"""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from agent_os.agent.project_paths import ProjectPaths


# ---------------------------------------------------------------------------
# Test fixture: real FastAPI app + setup_engine patched to report a
# deterministic set of installed sub-agents.
# ---------------------------------------------------------------------------


def _fake_check_all_factory(installed_slugs: list[str]):
    """Build a check_all() replacement that reports the given slugs as
    installed. Real ``setup_engine.check_all`` shells out — patch it.
    """
    from agent_os.agents.setup_types import AgentSetupStatus

    def _fake() -> list[AgentSetupStatus]:
        statuses: list[AgentSetupStatus] = []
        # Always include built-in (must be filtered by the route).
        statuses.append(AgentSetupStatus(
            slug="built-in", name="Built-in",
            installed=True, binary_path=None, version=None,
            dependencies_met=True, missing_dependencies=[],
            credentials_configured=True, missing_credentials=[],
            setup_actions=[],
        ))
        # Include claude-code as an "installed but absent" pretender too —
        # tests parameterize via installed_slugs.
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
        # And include codex as "not installed" so we can verify the
        # filter excludes uninstalled even when it's NOT denylisted.
        if "codex" not in installed_slugs:
            statuses.append(AgentSetupStatus(
                slug="codex", name="Codex",
                installed=False, binary_path=None, version=None,
                dependencies_met=False, missing_dependencies=["codex"],
                credentials_configured=False, missing_credentials=[],
                setup_actions=[],
            ))
        return statuses

    return _fake


@pytest.fixture
def app_state(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("USERPROFILE", str(tmp_path / "home"))
    os.makedirs(str(tmp_path / "home"), exist_ok=True)

    from agent_os.api.app import create_app
    app = create_app(data_dir=str(tmp_path / "data"))
    return app, tmp_path


def _make_client_with_installed(app_state, installed_slugs: list[str]):
    """Return a TestClient with setup_engine.check_all() patched to report
    the given installed sub-agent slugs.
    """
    app, _tmp = app_state
    from agent_os.api.routes import agents_v2 as routes_mod

    fake = _fake_check_all_factory(installed_slugs)
    routes_mod._setup_engine.check_all = fake  # type: ignore[assignment]
    return TestClient(app)


def _make_project(client, ws: str, *, disabled: list[str] | None = None) -> str:
    payload: dict = {
        "name": "memtest-" + os.path.basename(ws),
        "workspace": ws,
        "model": "gpt-4",
        "api_key": "test-key",
    }
    if disabled is not None:
        payload["disabled_sub_agents"] = disabled
    resp = client.post("/api/v2/projects", json=payload)
    assert resp.status_code == 201, resp.text
    return resp.json()["project_id"]


# ---------------------------------------------------------------------------
# GET /projects/{pid}/sub-agent-memory
# ---------------------------------------------------------------------------


class TestGetSubAgentMemory:

    def test_get_sub_agent_memory_returns_array_for_enabled_agents(
        self, app_state, tmp_path,
    ):
        """Project with installed claude-code + codex and no denylist:
        GET returns 2-element array."""
        client = _make_client_with_installed(
            app_state, installed_slugs=["claude-code", "codex"],
        )
        ws = str(tmp_path / "ws_two")
        os.makedirs(ws, exist_ok=True)
        pid = _make_project(client, ws, disabled=[])

        resp = client.get(f"/api/v2/projects/{pid}/sub-agent-memory")
        assert resp.status_code == 200, resp.text
        data = resp.json()
        slugs = sorted(e["agent_slug"] for e in data)
        assert slugs == ["claude-code", "codex"]

    def test_get_sub_agent_memory_excludes_disabled(
        self, app_state, tmp_path,
    ):
        """Project with disabled_sub_agents=['codex'] returns only
        claude-code."""
        client = _make_client_with_installed(
            app_state, installed_slugs=["claude-code", "codex"],
        )
        ws = str(tmp_path / "ws_disabled")
        os.makedirs(ws, exist_ok=True)
        pid = _make_project(client, ws, disabled=["codex"])

        resp = client.get(f"/api/v2/projects/{pid}/sub-agent-memory")
        assert resp.status_code == 200
        data = resp.json()
        slugs = [e["agent_slug"] for e in data]
        assert slugs == ["claude-code"]

    def test_get_sub_agent_memory_includes_exists_false_for_uninitialized(
        self, app_state, tmp_path,
    ):
        """Sub-agent enabled but never dispatched → exists=False, content=''."""
        client = _make_client_with_installed(
            app_state, installed_slugs=["claude-code"],
        )
        ws = str(tmp_path / "ws_uninit")
        os.makedirs(ws, exist_ok=True)
        pid = _make_project(client, ws, disabled=[])

        resp = client.get(f"/api/v2/projects/{pid}/sub-agent-memory")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 1
        cc = data[0]
        assert cc["agent_slug"] == "claude-code"
        assert cc["exists"] is False
        assert cc["content"] == ""
        assert cc["last_modified"] is None
        assert cc["size_bytes"] == 0

    def test_get_sub_agent_memory_reads_existing_file(
        self, app_state, tmp_path,
    ):
        """When MEMORY.md exists on disk, GET surfaces its content + size."""
        client = _make_client_with_installed(
            app_state, installed_slugs=["claude-code"],
        )
        ws = str(tmp_path / "ws_existing")
        os.makedirs(ws, exist_ok=True)
        pid = _make_project(client, ws, disabled=[])

        # Pre-create MEMORY.md with known content. Use binary mode so the
        # on-disk byte count is deterministic across platforms (no \n→\r\n
        # translation on Windows).
        path = os.path.join(
            ProjectPaths(ws).sub_agent_dir("claude-code"), "MEMORY.md",
        )
        os.makedirs(os.path.dirname(path), exist_ok=True)
        text = "# pre-existing memory content\n"
        with open(path, "wb") as f:
            f.write(text.encode("utf-8"))

        resp = client.get(f"/api/v2/projects/{pid}/sub-agent-memory")
        data = resp.json()
        cc = next(e for e in data if e["agent_slug"] == "claude-code")
        assert cc["exists"] is True
        assert cc["content"] == text
        assert cc["size_bytes"] == len(text.encode("utf-8"))
        assert isinstance(cc["last_modified"], (int, float))

    def test_get_sub_agent_memory_404_for_unknown_project(
        self, app_state,
    ):
        client = _make_client_with_installed(
            app_state, installed_slugs=["claude-code"],
        )
        resp = client.get("/api/v2/projects/proj_does_not_exist/sub-agent-memory")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# PUT /projects/{pid}/sub-agent-memory/{slug}
# ---------------------------------------------------------------------------


class TestPutSubAgentMemory:

    def test_put_sub_agent_memory_writes_file(self, app_state, tmp_path):
        """PUT writes content; subsequent GET reflects it."""
        client = _make_client_with_installed(
            app_state, installed_slugs=["claude-code"],
        )
        ws = str(tmp_path / "ws_writes")
        os.makedirs(ws, exist_ok=True)
        # Pre-create the parent so we exercise the "file is absent but
        # parent dir exists" path too.
        pid = _make_project(client, ws, disabled=[])

        body = {"content": "remember to always check the API contract\n"}
        resp = client.put(
            f"/api/v2/projects/{pid}/sub-agent-memory/claude-code",
            json=body,
        )
        assert resp.status_code == 200, resp.text

        # Verify the file is on disk with the user's exact content.
        path = os.path.join(
            ProjectPaths(ws).sub_agent_dir("claude-code"), "MEMORY.md",
        )
        assert os.path.isfile(path)
        with open(path, "r", encoding="utf-8") as f:
            actual = f.read()
        assert actual == body["content"]

    def test_put_sub_agent_memory_creates_file_if_absent(
        self, app_state, tmp_path,
    ):
        """PUT creates the file (and parent dir) when MEMORY.md is absent.

        Per spec: writes user content directly without the daemon's canonical
        header. The UI is overwriting from scratch.
        """
        from agent_os.agent.sub_agent_prompt import MEMORY_MD_HEADER

        client = _make_client_with_installed(
            app_state, installed_slugs=["claude-code"],
        )
        ws = str(tmp_path / "ws_create")
        os.makedirs(ws, exist_ok=True)
        pid = _make_project(client, ws, disabled=[])

        path = os.path.join(
            ProjectPaths(ws).sub_agent_dir("claude-code"), "MEMORY.md",
        )
        # Sanity: parent dir does not exist yet.
        assert not os.path.exists(os.path.dirname(path))

        body = {"content": "fresh content from the UI\n"}
        resp = client.put(
            f"/api/v2/projects/{pid}/sub-agent-memory/claude-code",
            json=body,
        )
        assert resp.status_code == 200, resp.text

        assert os.path.isfile(path)
        with open(path, "r", encoding="utf-8") as f:
            actual = f.read()
        assert actual == body["content"]

        # The canonical daemon header MUST NOT have been prepended.
        canonical = MEMORY_MD_HEADER.format(agent_slug="claude-code")
        assert not actual.startswith(canonical)

    def test_put_sub_agent_memory_rejects_oversize_content(
        self, app_state, tmp_path,
    ):
        """11 KB content → 400 response, file unchanged."""
        client = _make_client_with_installed(
            app_state, installed_slugs=["claude-code"],
        )
        ws = str(tmp_path / "ws_oversize")
        os.makedirs(ws, exist_ok=True)
        pid = _make_project(client, ws, disabled=[])

        oversized = "x" * (11 * 1024)
        resp = client.put(
            f"/api/v2/projects/{pid}/sub-agent-memory/claude-code",
            json={"content": oversized},
        )
        assert resp.status_code == 400

        # File must NOT have been written.
        path = os.path.join(
            ProjectPaths(ws).sub_agent_dir("claude-code"), "MEMORY.md",
        )
        assert not os.path.exists(path)

    def test_put_sub_agent_memory_warns_at_2kb_threshold(
        self, app_state, tmp_path,
    ):
        """3 KB content → 200 with warning field."""
        client = _make_client_with_installed(
            app_state, installed_slugs=["claude-code"],
        )
        ws = str(tmp_path / "ws_warn")
        os.makedirs(ws, exist_ok=True)
        pid = _make_project(client, ws, disabled=[])

        threek = "y" * (3 * 1024)
        resp = client.put(
            f"/api/v2/projects/{pid}/sub-agent-memory/claude-code",
            json={"content": threek},
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert "warning" in body
        assert isinstance(body["warning"], str) and body["warning"]
        assert body["size_bytes"] == 3 * 1024

    def test_put_sub_agent_memory_rejects_disabled_subagent(
        self, app_state, tmp_path,
    ):
        """A slug that is on the project's disabled list is rejected with 400."""
        client = _make_client_with_installed(
            app_state, installed_slugs=["claude-code", "codex"],
        )
        ws = str(tmp_path / "ws_reject_disabled")
        os.makedirs(ws, exist_ok=True)
        pid = _make_project(client, ws, disabled=["codex"])

        resp = client.put(
            f"/api/v2/projects/{pid}/sub-agent-memory/codex",
            json={"content": "blocked"},
        )
        assert resp.status_code == 400

    def test_put_sub_agent_memory_rejects_path_traversal_slug(
        self, app_state, tmp_path,
    ):
        """A slug containing path-traversal characters is rejected outright.

        The route returns 400 (slug regex). A subsequent attempt with a
        well-formed but unknown slug also returns 400 (not in available list).
        """
        client = _make_client_with_installed(
            app_state, installed_slugs=["claude-code"],
        )
        ws = str(tmp_path / "ws_traversal")
        os.makedirs(ws, exist_ok=True)
        pid = _make_project(client, ws, disabled=[])

        # Note: FastAPI may return 404 if the path encoding doesn't match
        # the route. We assert "not 200" rather than a specific code.
        resp = client.put(
            f"/api/v2/projects/{pid}/sub-agent-memory/..%2Fevil",
            json={"content": "x"},
        )
        assert resp.status_code != 200

        # An unknown well-formed slug also fails.
        resp = client.put(
            f"/api/v2/projects/{pid}/sub-agent-memory/no-such-agent",
            json={"content": "x"},
        )
        assert resp.status_code == 400

    def test_put_then_get_round_trip(self, app_state, tmp_path):
        """PUT content; GET surfaces it via the listing endpoint."""
        client = _make_client_with_installed(
            app_state, installed_slugs=["claude-code"],
        )
        ws = str(tmp_path / "ws_round_trip")
        os.makedirs(ws, exist_ok=True)
        pid = _make_project(client, ws, disabled=[])

        body = {"content": "user-edited memory\n"}
        resp = client.put(
            f"/api/v2/projects/{pid}/sub-agent-memory/claude-code",
            json=body,
        )
        assert resp.status_code == 200

        resp = client.get(f"/api/v2/projects/{pid}/sub-agent-memory")
        cc = next(e for e in resp.json() if e["agent_slug"] == "claude-code")
        assert cc["exists"] is True
        assert cc["content"] == body["content"]
