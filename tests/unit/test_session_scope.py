# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Per-session cross-project scope (Spec 12 §2a/§2c).

Quick Tasks (scratch) sessions carry a scope record
``{"mode": "all"|"selected"|"off", "selected_project_ids": [...]}``. New scratch
sessions default to ``all``; normal projects default to ``off`` (single-root,
unchanged). Changing scope recomputes the read-root set (a live callable, no
agent restart) and re-syncs the sandbox read portals.
"""

import os
from unittest.mock import MagicMock

import pytest

from agent_os.daemon_v2.agent_manager import AgentManager
from agent_os.platform.types import PermissionResult


def _store(projects):
    """A minimal ProjectStore double over a list of project dicts."""
    store = MagicMock()
    by_id = {p["project_id"]: p for p in projects}
    store.get_project.side_effect = lambda pid: by_id.get(pid)
    store.list_projects.side_effect = lambda: list(projects)
    return store


def _manager(projects, provider=None):
    return AgentManager(
        project_store=_store(projects),
        ws_manager=MagicMock(),
        sub_agent_manager=MagicMock(),
        activity_translator=MagicMock(),
        process_manager=MagicMock(),
        platform_provider=provider,
    )


def _scratch(tmp_path):
    ws = str(tmp_path / "scratch")
    os.makedirs(ws, exist_ok=True)
    return {"project_id": "scratch_1", "name": "Quick Tasks", "workspace": ws, "is_scratch": True}


def _proj(tmp_path, pid, name):
    ws = str(tmp_path / name)
    os.makedirs(ws, exist_ok=True)
    return {"project_id": pid, "name": name, "workspace": ws, "is_scratch": False}


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

def test_default_scope_all_for_scratch(tmp_path):
    mgr = _manager([_scratch(tmp_path)])
    scope = mgr.get_session_scope("scratch_1", "sess1")
    assert scope["mode"] == "all"
    assert scope["selected_project_ids"] == []


def test_default_scope_off_for_normal(tmp_path):
    mgr = _manager([_proj(tmp_path, "p1", "Alpha")])
    scope = mgr.get_session_scope("p1", "sess1")
    assert scope["mode"] == "off"


# ---------------------------------------------------------------------------
# Roots recompute live (callable) after a scope change — no agent restart
# ---------------------------------------------------------------------------

def test_roots_recompute_after_scope_change(tmp_path):
    scratch = _scratch(tmp_path)
    a = _proj(tmp_path, "p_a", "Alpha")
    b = _proj(tmp_path, "p_b", "Beta")
    mgr = _manager([scratch, a, b])

    # Default (all): both project workspaces are read roots (self excluded).
    roots, labels = mgr._compute_scope_roots("scratch_1", "sess1")
    assert roots[0] == scratch["workspace"]
    assert a["workspace"] in roots and b["workspace"] in roots
    assert labels[os.path.realpath(a["workspace"])] == "Alpha"

    # Narrow to a single selected project — recompute reflects it immediately.
    mgr.set_session_scope("scratch_1", "sess1", "selected", ["p_a"])
    roots2, _ = mgr._compute_scope_roots("scratch_1", "sess1")
    assert a["workspace"] in roots2
    assert b["workspace"] not in roots2

    # Off → only the scratch workspace.
    mgr.set_session_scope("scratch_1", "sess1", "off", [])
    roots3, _ = mgr._compute_scope_roots("scratch_1", "sess1")
    assert roots3 == [scratch["workspace"]]


def test_normal_project_never_gets_secondary_roots(tmp_path):
    mgr = _manager([_scratch(tmp_path), _proj(tmp_path, "p1", "Alpha")])
    roots, labels = mgr._compute_scope_roots("p1", "sess1")
    assert len(roots) == 1
    assert labels == {}


# ---------------------------------------------------------------------------
# Sandbox read portals grant/revoke on scope change (recording provider)
# ---------------------------------------------------------------------------

def test_scope_change_syncs_read_portals(tmp_path):
    scratch = _scratch(tmp_path)
    a = _proj(tmp_path, "p_a", "Alpha")
    b = _proj(tmp_path, "p_b", "Beta")
    provider = MagicMock()
    provider.grant_folder_access.return_value = PermissionResult(success=True, path="/x")
    provider.revoke_folder_access.return_value = PermissionResult(success=True, path="/x")
    mgr = _manager([scratch, a, b], provider=provider)
    scope_key = os.path.realpath(scratch["workspace"])

    # mode=all → read portals granted for both projects, scoped to scratch ws.
    mgr.set_session_scope("scratch_1", "sess1", "all", [])
    granted = {
        (c.args[0], c.args[1], c.kwargs.get("scope"))
        for c in provider.grant_folder_access.call_args_list
    }
    assert (os.path.realpath(a["workspace"]), "read", scope_key) in granted
    assert (os.path.realpath(b["workspace"]), "read", scope_key) in granted

    # Narrow to selected [p_a] → Beta's portal is revoked (scoped to scratch).
    provider.revoke_folder_access.reset_mock()
    mgr.set_session_scope("scratch_1", "sess1", "selected", ["p_a"])
    revoked = {
        (c.args[0], c.kwargs.get("scope"))
        for c in provider.revoke_folder_access.call_args_list
    }
    assert (os.path.realpath(b["workspace"]), scope_key) in revoked


# ---------------------------------------------------------------------------
# API validation (422 unknown project, 409 non-scratch)
# ---------------------------------------------------------------------------

@pytest.fixture
def client(tmp_path):
    from agent_os.api.app import create_app
    from starlette.testclient import TestClient
    app = create_app(data_dir=str(tmp_path))
    return TestClient(app)


def _scratch_id(client):
    projects = client.get("/api/v2/projects").json()
    projects = projects if isinstance(projects, list) else projects.get("projects", [])
    for p in projects:
        if p.get("is_scratch"):
            return p["project_id"]
    raise AssertionError("no scratch project")


def test_put_scope_unknown_project_422(client):
    sid = _scratch_id(client)
    resp = client.put(
        f"/api/v2/agents/{sid}/sessions/sess1/scope",
        json={"mode": "selected", "selected_project_ids": ["does_not_exist"]},
    )
    assert resp.status_code == 422


def test_put_scope_non_scratch_409(client, tmp_path):
    from agent_os.api.routes import agents_v2
    pid = agents_v2._project_store.create_project({
        "name": "Normal", "workspace": str(tmp_path / "normal"), "autonomy": "hands_off",
    })
    resp = client.put(
        f"/api/v2/agents/{pid}/sessions/sess1/scope",
        json={"mode": "all", "selected_project_ids": []},
    )
    assert resp.status_code == 409


def test_get_scope_defaults_all_for_scratch(client):
    sid = _scratch_id(client)
    resp = client.get(f"/api/v2/agents/{sid}/sessions/sess1/scope")
    assert resp.status_code == 200
    assert resp.json()["mode"] == "all"
