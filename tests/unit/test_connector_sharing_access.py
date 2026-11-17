# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Connectors shared with agents: the per-connector access rule.

Access is stored globally per connector beside the custom servers in
``connectors.json``: default "all agents" (today's behaviour), or "selected
agents" with a list of agent ids. The manager is the agent ``orbital``. An
agent gets a connector when it is connected, enabled for the project, and
allowed for that agent. A file without the key (every existing user) means
"all", and a file the user never touched keeps its old shape.
"""

from __future__ import annotations

import json
import os

import pytest

from agent_os.connectors import load_catalog
from agent_os.connectors.manager import ConnectorError, ConnectorManager
from agent_os.daemon_v2.connector_sharing import MANAGER_AGENT_ID

from tests.unit._mock_mcp_server import build_mock_server, in_memory_opener


def _mgr(tmp_path, catalog=None):
    return ConnectorManager(
        catalog=catalog if catalog is not None else load_catalog(),
        credential_store=None,
        data_dir=str(tmp_path),
        session_opener=in_memory_opener(build_mock_server()),
    )


def _file(tmp_path):
    with open(os.path.join(tmp_path, "connectors.json"), encoding="utf-8") as fh:
        return json.load(fh)


def test_the_manager_is_the_agent_the_pin_switcher_already_calls_orbital():
    assert MANAGER_AGENT_ID == "orbital"


async def test_default_access_is_all_agents(tmp_path):
    mgr = _mgr(tmp_path)
    await mgr.add_custom("Notes", "https://notes/mcp", "none")
    assert mgr.get_access("custom-notes") == {"mode": "all", "agents": []}
    assert mgr.allows("custom-notes", "claude-code")
    assert mgr.allows("custom-notes", MANAGER_AGENT_ID)
    entry = next(e for e in mgr.list_connectors() if e["manifest"].id == "custom-notes")
    assert entry["access"] == {"mode": "all", "agents": []}


async def test_selected_agents_allows_only_those(tmp_path):
    mgr = _mgr(tmp_path)
    await mgr.add_custom("Notes", "https://notes/mcp", "none")
    out = mgr.set_access("custom-notes", "selected", ["codex", "codex", MANAGER_AGENT_ID])
    assert out == {"mode": "selected", "agents": ["codex", MANAGER_AGENT_ID]}
    assert mgr.allows("custom-notes", "codex")
    assert mgr.allows("custom-notes", MANAGER_AGENT_ID)
    assert not mgr.allows("custom-notes", "claude-code")
    # An empty selection is nobody, the manager included.
    mgr.set_access("custom-notes", "selected", [])
    assert not mgr.allows("custom-notes", MANAGER_AGENT_ID)


async def test_all_keeps_the_last_selection_but_ignores_it(tmp_path):
    mgr = _mgr(tmp_path)
    await mgr.add_custom("Notes", "https://notes/mcp", "none")
    mgr.set_access("custom-notes", "all", ["codex"])
    assert mgr.get_access("custom-notes") == {"mode": "all", "agents": ["codex"]}
    assert mgr.allows("custom-notes", "claude-code")


async def test_access_survives_a_restart_and_is_written_beside_custom(tmp_path):
    mgr = _mgr(tmp_path)
    await mgr.add_custom("Notes", "https://notes/mcp", "none")
    mgr.set_access("custom-notes", "selected", ["codex"])
    mgr.set_access("google-calendar", "selected", [MANAGER_AGENT_ID])
    data = _file(tmp_path)
    assert [c["id"] for c in data["custom"]] == ["custom-notes"]
    assert data["access"]["custom-notes"] == {"mode": "selected", "agents": ["codex"]}
    again = _mgr(tmp_path)
    assert again.get_access("custom-notes") == {"mode": "selected", "agents": ["codex"]}
    assert again.get_access("google-calendar") == {"mode": "selected", "agents": [MANAGER_AGENT_ID]}


def test_bad_mode_and_unknown_connector_are_refused(tmp_path):
    mgr = _mgr(tmp_path)
    with pytest.raises(ConnectorError):
        mgr.set_access("nope", "all", [])
    with pytest.raises(ValueError):
        mgr.set_access("google-calendar", "some", [])
    assert not os.path.exists(os.path.join(tmp_path, "connectors.json"))


async def test_a_file_never_touched_keeps_its_old_shape(tmp_path):
    """Upgrade safety: adding a custom server without touching access writes
    exactly the old ``{"custom": [...]}`` shape."""
    mgr = _mgr(tmp_path)
    await mgr.add_custom("Notes", "https://notes/mcp", "none")
    assert set(_file(tmp_path)) == {"custom"}


async def test_an_old_file_opens_with_everything_shared(tmp_path):
    with open(os.path.join(tmp_path, "connectors.json"), "w", encoding="utf-8") as fh:
        json.dump({"custom": [{"id": "custom-notes", "name": "Notes",
                               "auth_provider": "custom-notes", "auth_type": "none",
                               "server_url": "https://notes/mcp"}]}, fh)
    mgr = _mgr(tmp_path)
    assert mgr.get_access("custom-notes") == {"mode": "all", "agents": []}


async def test_malformed_access_entries_are_ignored_not_fatal(tmp_path):
    with open(os.path.join(tmp_path, "connectors.json"), "w", encoding="utf-8") as fh:
        json.dump({"custom": [], "access": {
            "google-calendar": {"mode": "weird", "agents": ["codex"]},
            "google-drive": "selected",
            "gmail": {"mode": "selected", "agents": ["codex", 7]},
        }}, fh)
    mgr = _mgr(tmp_path)
    assert mgr.get_access("google-calendar") == {"mode": "all", "agents": []}
    assert mgr.get_access("google-drive") == {"mode": "all", "agents": []}
    assert mgr.get_access("gmail") == {"mode": "selected", "agents": ["codex"]}


async def test_removing_a_custom_connector_drops_its_access(tmp_path):
    mgr = _mgr(tmp_path)
    await mgr.add_custom("Notes", "https://notes/mcp", "none")
    mgr.set_access("custom-notes", "selected", ["codex"])
    await mgr.remove_custom("custom-notes")
    assert "custom-notes" not in (_file(tmp_path).get("access") or {})
    await mgr.add_custom("Notes", "https://notes/mcp", "none")
    assert mgr.get_access("custom-notes") == {"mode": "all", "agents": []}


async def test_shared_ids_is_enabled_and_connected_and_allowed(tmp_path):
    mgr = _mgr(tmp_path)
    await mgr.add_custom("Notes", "https://notes/mcp", "none")
    await mgr.add_custom("Mail", "https://mail/mcp", "none")
    enabled = ["custom-mail", "google-calendar", "custom-notes", "custom-mail", "gone"]
    # google-calendar is enabled but not connected; "gone" is not in the catalog.
    assert mgr.shared_ids(enabled, "codex") == ("custom-mail", "custom-notes")
    mgr.set_access("custom-mail", "selected", ["claude-code"])
    assert mgr.shared_ids(enabled, "codex") == ("custom-notes",)
    assert mgr.shared_ids(enabled, "claude-code") == ("custom-mail", "custom-notes")
    assert mgr.shared_ids([], "codex") == ()
    assert mgr.shared_ids(None, "codex") == ()
