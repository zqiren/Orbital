# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Regression tests for project settings ↔ local file sync gaps.

GAP 1: SettingsView must fetch project detail on mount (frontend; tested via API contract).
GAP 4: user_directives.md must not be double-injected into LLM context.
GAP 5: PUT /projects/:id must return disk content in response body.
"""
import os

import pytest
from fastapi.testclient import TestClient

from agent_os.agent.context import ContextManager
from agent_os.api.app import create_app


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def client(tmp_path):
    app = create_app(data_dir=str(tmp_path / "data"))
    return TestClient(app)


@pytest.fixture
def workspace(tmp_path):
    ws = tmp_path / "workspace"
    ws.mkdir()
    return str(ws)


def _create_project(client, workspace, **overrides):
    payload = {
        "name": "TestProj",
        "workspace": workspace,
        "model": "gpt-4",
        "api_key": "sk-test",
        **overrides,
    }
    resp = client.post("/api/v2/projects", json=payload)
    assert resp.status_code == 201
    return resp.json()["project_id"]


# ---------------------------------------------------------------------------
# GAP 4: user_directives.md excluded from _read_instructions
# ---------------------------------------------------------------------------

class TestGap4UserDirectivesExclusion:
    """user_directives.md must NOT be read by _read_instructions (already
    injected by PromptBuilder._standing_rules)."""

    def test_user_directives_excluded_from_read_instructions(self, tmp_path):
        """_read_instructions must skip user_directives.md."""
        instructions_dir = tmp_path / ".agent-os" / "instructions"
        instructions_dir.mkdir(parents=True)
        (instructions_dir / "project_goals.md").write_text("goals content")
        (instructions_dir / "user_directives.md").write_text("directives content")
        (instructions_dir / "custom_rules.md").write_text("custom content")

        result = ContextManager._read_instructions(str(instructions_dir))

        assert result is not None
        assert "custom content" in result
        assert "directives content" not in result
        assert "goals content" not in result

    def test_project_goals_still_excluded(self, tmp_path):
        """project_goals.md must remain excluded (pre-existing behavior)."""
        instructions_dir = tmp_path / ".agent-os" / "instructions"
        instructions_dir.mkdir(parents=True)
        (instructions_dir / "project_goals.md").write_text("goals here")
        (instructions_dir / "other.md").write_text("other content")

        result = ContextManager._read_instructions(str(instructions_dir))

        assert "other content" in result
        assert "goals here" not in result

    def test_other_md_files_still_included(self, tmp_path):
        """Non-excluded .md files must still be read."""
        instructions_dir = tmp_path / ".agent-os" / "instructions"
        instructions_dir.mkdir(parents=True)
        (instructions_dir / "user_directives.md").write_text("directives")
        (instructions_dir / "coding_standards.md").write_text("standards")
        (instructions_dir / "api_rules.md").write_text("api rules")

        result = ContextManager._read_instructions(str(instructions_dir))

        assert "api rules" in result
        assert "standards" in result
        assert "directives" not in result

    def test_empty_after_exclusions_returns_none(self, tmp_path):
        """If only excluded files exist, _read_instructions returns None."""
        instructions_dir = tmp_path / ".agent-os" / "instructions"
        instructions_dir.mkdir(parents=True)
        (instructions_dir / "project_goals.md").write_text("goals")
        (instructions_dir / "user_directives.md").write_text("directives")

        result = ContextManager._read_instructions(str(instructions_dir))

        assert result is None


# ---------------------------------------------------------------------------
# GAP 5: PUT response includes disk content
# ---------------------------------------------------------------------------

class TestGap5PutReturnsDiskContent:
    """PUT /projects/:id must return project_goals_content and
    user_directives_content in the response body."""

    def test_put_returns_project_goals_content(self, client, workspace):
        pid = _create_project(client, workspace)

        resp = client.put(f"/api/v2/projects/{pid}", json={
            "project_goals_content": "# Mission\nBuild auth",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["project_goals_content"] == "# Mission\nBuild auth"

    def test_put_returns_user_directives_content(self, client, workspace):
        pid = _create_project(client, workspace)

        resp = client.put(f"/api/v2/projects/{pid}", json={
            "user_directives_content": "Always write tests\nUse PostgreSQL",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["user_directives_content"] == "Always write tests\nUse PostgreSQL"

    def test_put_returns_both_fields(self, client, workspace):
        pid = _create_project(client, workspace)

        resp = client.put(f"/api/v2/projects/{pid}", json={
            "project_goals_content": "goals",
            "user_directives_content": "rules",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["project_goals_content"] == "goals"
        assert data["user_directives_content"] == "rules"

    def test_put_without_content_returns_empty_strings(self, client, workspace):
        """PUT that only updates metadata should still include disk fields."""
        pid = _create_project(client, workspace)

        resp = client.put(f"/api/v2/projects/{pid}", json={
            "agent_name": "NewName",
        })
        assert resp.status_code == 200
        data = resp.json()
        # No files on disk yet, so empty strings
        assert data["project_goals_content"] == ""
        assert data["user_directives_content"] == ""

    def test_put_preserves_previously_saved_content(self, client, workspace):
        """PUT with only goals should preserve previously saved directives."""
        pid = _create_project(client, workspace)

        # Save directives first
        client.put(f"/api/v2/projects/{pid}", json={
            "user_directives_content": "existing rules",
        })

        # Save goals only — directives should still be present
        resp = client.put(f"/api/v2/projects/{pid}", json={
            "project_goals_content": "new goals",
        })
        data = resp.json()
        assert data["project_goals_content"] == "new goals"
        assert data["user_directives_content"] == "existing rules"


# ---------------------------------------------------------------------------
# GAP 1: List endpoint does NOT include disk content
#         (ensures we don't regress by leaking disk reads into list)
# ---------------------------------------------------------------------------

class TestGap1ListVsDetail:
    """The list endpoint must stay lean (no disk content), while the detail
    endpoint must include disk content."""

    def test_list_does_not_include_disk_content(self, client, workspace):
        pid = _create_project(client, workspace)

        # Write some content
        client.put(f"/api/v2/projects/{pid}", json={
            "project_goals_content": "goals here",
            "user_directives_content": "rules here",
        })

        # List endpoint should NOT have disk content
        resp = client.get("/api/v2/projects")
        assert resp.status_code == 200
        projects = resp.json()
        proj = next(p for p in projects if p["project_id"] == pid)
        assert "project_goals_content" not in proj
        assert "user_directives_content" not in proj

    def test_detail_includes_disk_content(self, client, workspace):
        pid = _create_project(client, workspace)

        client.put(f"/api/v2/projects/{pid}", json={
            "project_goals_content": "goals here",
            "user_directives_content": "rules here",
        })

        # Detail endpoint MUST have disk content
        resp = client.get(f"/api/v2/projects/{pid}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["project_goals_content"] == "goals here"
        assert data["user_directives_content"] == "rules here"
