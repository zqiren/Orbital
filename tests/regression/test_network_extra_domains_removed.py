# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Regression: the dead ``network_extra_domains`` config surface is REMOVED.

TASK-network-config-cleanup — the field was persisted on projects and
AgentConfig but (per the Phase-1 trace) never plumbed into the enforced
``NetworkRules``; every construction site hardcodes ``DEFAULT_ALLOWLIST_DOMAINS``
(plus per-agent manifest ``network_domains``, a different field). Config that
exists, persists, and silently does nothing violates the honesty principle, so
the surface is removed rather than plumbed.

Pins:
  (i)  create/update payloads no longer accept or echo the field — sending it
       is tolerated (ignored), never persisted, never returned;
  (ii) a legacy project record that already CONTAINS the field loads without
       error, works, is not echoed, and the stored field is dropped on the
       next save round-trip;
  (iii) AgentConfig no longer carries the field.
"""

from __future__ import annotations

import json
import os

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def data_dir(tmp_path):
    d = tmp_path / "orbital-data"
    d.mkdir()
    return str(d)


def _make_client(data_dir: str) -> TestClient:
    from agent_os.api.app import create_app
    return TestClient(create_app(data_dir=data_dir))


def _projects_on_disk(data_dir: str) -> dict:
    path = os.path.join(data_dir, "projects.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def test_create_payload_does_not_accept_or_echo_field(data_dir, tmp_path):
    client = _make_client(data_dir)
    ws = tmp_path / "ws1"
    ws.mkdir()

    resp = client.post("/api/v2/projects", json={
        "name": "net-clean",
        "workspace": str(ws),
        "model": "kimi-k2",
        "api_key": "k",
        "network_extra_domains": ["dify.ai", "cloud.dify.ai"],
    })
    assert resp.status_code == 201
    body = resp.json()
    assert "network_extra_domains" not in body

    pid = body["project_id"] if "project_id" in body else body["id"]

    # Not persisted in the store either.
    get_resp = client.get(f"/api/v2/projects/{pid}")
    assert get_resp.status_code == 200
    assert "network_extra_domains" not in get_resp.json()

    from agent_os.daemon_v2.project_store import ProjectStore
    store = ProjectStore(data_dir=data_dir)
    record = store.get_project(pid)
    assert record is not None
    assert "network_extra_domains" not in record


def test_update_payload_does_not_accept_or_echo_field(data_dir, tmp_path):
    client = _make_client(data_dir)
    ws = tmp_path / "ws2"
    ws.mkdir()

    pid = client.post("/api/v2/projects", json={
        "name": "net-clean-upd",
        "workspace": str(ws),
        "model": "kimi-k2",
        "api_key": "k",
    }).json()["project_id"]

    resp = client.put(f"/api/v2/projects/{pid}", json={
        "name": "net-clean-upd",
        "network_extra_domains": ["dify.ai"],
    })
    assert resp.status_code == 200
    assert "network_extra_domains" not in resp.json()

    get_resp = client.get(f"/api/v2/projects/{pid}")
    assert "network_extra_domains" not in get_resp.json()


def test_legacy_record_loads_and_field_dropped_on_next_save(data_dir, tmp_path):
    """A previously persisted project containing the legacy field must load
    without error; the field is not echoed externally; and a save round-trip
    drops it from disk."""
    ws = tmp_path / "ws-legacy"
    ws.mkdir()

    # Plant a legacy projects.json BEFORE the app boots.
    legacy = {
        "proj_legacy00001": {
            "id": "proj_legacy00001",
            "name": "legacy-net",
            "agent_name": "legacy-net",
            "workspace": str(ws),
            "model": "kimi-k2",
            "api_key": "",
            "provider": "custom",
            "sdk": "openai",
            "autonomy": "hands_off",
            "is_scratch": False,
            "network_extra_domains": ["dify.ai", "*.example.com"],
        },
    }
    with open(os.path.join(data_dir, "projects.json"), "w", encoding="utf-8") as f:
        json.dump(legacy, f)

    client = _make_client(data_dir)

    # Loads cleanly and is usable.
    resp = client.get("/api/v2/projects/proj_legacy00001")
    assert resp.status_code == 200
    body = resp.json()
    assert body["name"] == "legacy-net"
    # The legacy key is never read externally.
    assert "network_extra_domains" not in body

    # Save round-trip (a plain rename) drops the stored field.
    resp = client.put("/api/v2/projects/proj_legacy00001", json={"name": "legacy-net"})
    assert resp.status_code == 200

    from agent_os.daemon_v2.project_store import ProjectStore
    store = ProjectStore(data_dir=data_dir)
    record = store.get_project("proj_legacy00001")
    assert record is not None
    assert "network_extra_domains" not in record


def test_agent_config_has_no_network_extra_domains_field():
    from agent_os.daemon_v2.models import AgentConfig

    config = AgentConfig(workspace="/tmp", model="m", api_key="k")
    assert not hasattr(config, "network_extra_domains")
