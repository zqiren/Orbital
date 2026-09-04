# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Spec 079 — the ``agent`` field across the queue and trigger HTTP surface.

Real routes, real ``ProjectStore`` / ``QueueStore``. The interesting behaviour
is the PATCH contract: the field is read by PRESENCE, not by nullness, because
an explicit null is how the picker hands an item or an automation back to
Orbital. Every other optional field on those bodies uses "None means omitted",
so this one is worth pinning on both routes.
"""

from __future__ import annotations

import tempfile
from unittest.mock import AsyncMock, MagicMock

from fastapi import FastAPI
from fastapi.testclient import TestClient

from agent_os.daemon_v2.agent_manager import AgentManager
from agent_os.daemon_v2.project_store import ProjectStore


def _make_client():
    tmpdir = tempfile.mkdtemp()
    store = ProjectStore(data_dir=tmpdir)
    pid = store.create_project({
        "name": "Picker Project",
        "workspace": tmpdir,
        "model": "gpt-4",
        "api_key": "sk-test",
    })
    sub_agent_manager = MagicMock()
    sub_agent_manager.stop_all = AsyncMock()
    manager = AgentManager(
        project_store=store,
        ws_manager=MagicMock(),
        sub_agent_manager=sub_agent_manager,
        activity_translator=MagicMock(),
        process_manager=MagicMock(),
        platform_provider=None,
        registry=MagicMock(),
        setup_engine=MagicMock(),
        settings_store=None,
        credential_store=None,
    )

    from agent_os.api.routes import agents_v2
    app = FastAPI()
    agents_v2.configure(
        project_store=store,
        agent_manager=manager,
        ws_manager=MagicMock(),
        sub_agent_manager=MagicMock(),
        setup_engine=MagicMock(),
        settings_store=MagicMock(),
        credential_store=MagicMock(),
    )
    app.include_router(agents_v2.router)
    return TestClient(app), pid


# ---------------------------------------------------------------------------
# Queue items
# ---------------------------------------------------------------------------


def test_queue_item_round_trips_the_chosen_agent():
    client, pid = _make_client()

    resp = client.post(
        f"/api/v2/projects/{pid}/queue/items",
        json={"content": "write hello.txt", "agent": "codex"},
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["item"]["agent"] == "codex"

    snapshot = client.get(f"/api/v2/projects/{pid}/queue").json()
    assert snapshot["items"][0]["agent"] == "codex"


def test_queue_item_without_an_agent_reads_back_null():
    """A pre-079 client sends no ``agent`` at all; the item must simply be
    Orbital's, not rejected and not defaulted to some handle."""
    client, pid = _make_client()

    resp = client.post(
        f"/api/v2/projects/{pid}/queue/items",
        json={"content": "write hello.txt"},
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["item"]["agent"] is None


def test_patching_only_the_text_leaves_the_agent_in_place():
    client, pid = _make_client()
    item_id = client.post(
        f"/api/v2/projects/{pid}/queue/items",
        json={"content": "write hello.txt", "agent": "codex"},
    ).json()["item"]["id"]

    resp = client.patch(
        f"/api/v2/projects/{pid}/queue/items/{item_id}",
        json={"content": "write hello.md"},
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["item"]["content"] == "write hello.md"
    assert resp.json()["item"]["agent"] == "codex"


def test_patching_agent_null_hands_the_item_back_to_orbital():
    client, pid = _make_client()
    item_id = client.post(
        f"/api/v2/projects/{pid}/queue/items",
        json={"content": "write hello.txt", "agent": "codex"},
    ).json()["item"]["id"]

    resp = client.patch(
        f"/api/v2/projects/{pid}/queue/items/{item_id}",
        json={"agent": None},
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["item"]["agent"] is None


def test_patching_reassigns_the_agent():
    client, pid = _make_client()
    item_id = client.post(
        f"/api/v2/projects/{pid}/queue/items",
        json={"content": "write hello.txt", "agent": "codex"},
    ).json()["item"]["id"]

    resp = client.patch(
        f"/api/v2/projects/{pid}/queue/items/{item_id}",
        json={"agent": "claude-code"},
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["item"]["agent"] == "claude-code"


# ---------------------------------------------------------------------------
# Triggers / automations
# ---------------------------------------------------------------------------


def _create_trigger(client, pid, **extra):
    body = {
        "name": "Nightly",
        "type": "schedule",
        "task": "Do the thing",
        "schedule": {"cron": "0 7 * * *", "human": "Daily at 7am",
                     "timezone": "UTC"},
    }
    body.update(extra)
    return client.post(f"/api/v2/projects/{pid}/triggers", json=body)


def test_trigger_round_trips_the_chosen_agent():
    client, pid = _make_client()

    resp = _create_trigger(client, pid, agent="codex")
    assert resp.status_code == 201, resp.text
    assert resp.json()["agent"] == "codex"

    listed = client.get(f"/api/v2/projects/{pid}/triggers").json()
    assert listed[0]["agent"] == "codex"


def test_trigger_without_an_agent_reads_back_null():
    client, pid = _make_client()
    resp = _create_trigger(client, pid)
    assert resp.status_code == 201, resp.text
    assert resp.json()["agent"] is None


def test_patching_only_the_name_leaves_the_trigger_agent_in_place():
    client, pid = _make_client()
    tid = _create_trigger(client, pid, agent="codex").json()["id"]

    resp = client.patch(
        f"/api/v2/projects/{pid}/triggers/{tid}", json={"name": "Renamed"},
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["name"] == "Renamed"
    assert resp.json()["agent"] == "codex"


def test_patching_trigger_agent_null_hands_it_back_to_orbital():
    """The null must survive the route's drop-the-nulls merge — that merge
    exists to stop a partial update blanking a REQUIRED field, and applying it
    to ``agent`` would make unassigning impossible."""
    client, pid = _make_client()
    tid = _create_trigger(client, pid, agent="codex").json()["id"]

    resp = client.patch(
        f"/api/v2/projects/{pid}/triggers/{tid}", json={"agent": None},
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["agent"] is None


def test_patching_reassigns_the_trigger_agent():
    client, pid = _make_client()
    tid = _create_trigger(client, pid, agent="codex").json()["id"]

    resp = client.patch(
        f"/api/v2/projects/{pid}/triggers/{tid}", json={"agent": "claude-code"},
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["agent"] == "claude-code"
