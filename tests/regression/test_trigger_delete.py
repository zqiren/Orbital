# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""
Regression tests for the trigger DELETE endpoint.

Covers:
1. Deleting a cron trigger removes it and cancels the timer
2. Deleting a file_watch trigger removes it and stops the observer
3. Deleting a non-existent trigger returns 404
"""

import asyncio
import tempfile
from unittest.mock import MagicMock, AsyncMock

import pytest
from fastapi.testclient import TestClient

from agent_os.daemon_v2.project_store import ProjectStore
from agent_os.daemon_v2.trigger_manager import TriggerManager


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_project_store(triggers=None):
    """Create a ProjectStore with a temp directory and one project."""
    tmpdir = tempfile.mkdtemp()
    store = ProjectStore(data_dir=tmpdir)
    pid = store.create_project({
        "name": "Test Project",
        "workspace": tmpdir,
        "model": "gpt-4",
        "api_key": "sk-test",
    })
    if triggers is not None:
        store.update_project(pid, {"triggers": triggers})
    return store, pid


def _make_app(project_store, trigger_manager=None):
    """Create a minimal FastAPI app with trigger routes."""
    from fastapi import FastAPI
    from agent_os.api.routes import agents_v2

    app = FastAPI()
    agents_v2.configure(
        project_store=project_store,
        agent_manager=MagicMock(),
        ws_manager=MagicMock(),
        sub_agent_manager=MagicMock(),
        setup_engine=MagicMock(),
        settings_store=MagicMock(),
        credential_store=MagicMock(),
        trigger_manager=trigger_manager,
    )
    app.include_router(agents_v2.router)
    return app


# ===========================================================================
# DELETE /api/v2/projects/{pid}/triggers/{trigger_id}
# ===========================================================================

class TestDeleteTriggerEndpoint:

    def test_delete_cron_trigger(self):
        """DELETE should remove a cron trigger and call unregister_trigger."""
        triggers = [
            {
                "id": "trg_cron01",
                "name": "Daily backup",
                "enabled": True,
                "type": "schedule",
                "schedule": {"cron": "0 2 * * *", "human": "Every day at 2 AM", "timezone": "UTC"},
                "task": "Backup the database",
                "autonomy": None,
                "last_triggered": None,
                "trigger_count": 0,
                "created_at": "2026-03-01T00:00:00Z",
            },
        ]
        store, pid = _make_project_store(triggers=triggers)
        mock_tm = MagicMock()
        app = _make_app(store, trigger_manager=mock_tm)
        client = TestClient(app)

        resp = client.delete(f"/api/v2/projects/{pid}/triggers/trg_cron01")
        assert resp.status_code == 204

        # Verify trigger was removed from project
        project = store.get_project(pid)
        assert len(project.get("triggers", [])) == 0

        # Verify unregister was called
        mock_tm.unregister_trigger.assert_called_once_with("trg_cron01")

    def test_delete_file_watch_trigger(self):
        """DELETE should remove a file_watch trigger and call unregister_trigger."""
        triggers = [
            {
                "id": "trg_fw001",
                "name": "Photo watcher",
                "enabled": True,
                "type": "file_watch",
                "watch_path": "incoming",
                "patterns": ["*.jpg"],
                "recursive": False,
                "debounce_seconds": 5,
                "task": "Process photos",
                "autonomy": None,
                "last_triggered": None,
                "trigger_count": 0,
                "created_at": "2026-03-01T00:00:00Z",
            },
        ]
        store, pid = _make_project_store(triggers=triggers)
        mock_tm = MagicMock()
        app = _make_app(store, trigger_manager=mock_tm)
        client = TestClient(app)

        resp = client.delete(f"/api/v2/projects/{pid}/triggers/trg_fw001")
        assert resp.status_code == 204

        # Verify trigger was removed
        project = store.get_project(pid)
        assert len(project.get("triggers", [])) == 0

        mock_tm.unregister_trigger.assert_called_once_with("trg_fw001")

    def test_delete_nonexistent_trigger(self):
        """DELETE for a missing trigger_id should return 404."""
        store, pid = _make_project_store(triggers=[])
        app = _make_app(store)
        client = TestClient(app)

        resp = client.delete(f"/api/v2/projects/{pid}/triggers/trg_nope")
        assert resp.status_code == 404
        assert "not found" in resp.json()["detail"].lower()

    def test_delete_nonexistent_project(self):
        """DELETE for a missing project_id should return 404."""
        store, _ = _make_project_store()
        app = _make_app(store)
        client = TestClient(app)

        resp = client.delete("/api/v2/projects/nonexistent/triggers/trg_abc")
        assert resp.status_code == 404
        assert "not found" in resp.json()["detail"].lower()

    def test_delete_preserves_other_triggers(self):
        """Deleting one trigger should not affect others."""
        triggers = [
            {
                "id": "trg_keep",
                "name": "Keep this one",
                "enabled": True,
                "type": "schedule",
                "schedule": {"cron": "0 0 * * *", "human": "Daily", "timezone": "UTC"},
                "task": "Stay alive",
                "autonomy": None,
                "last_triggered": None,
                "trigger_count": 5,
                "created_at": "2026-01-01T00:00:00Z",
            },
            {
                "id": "trg_delete",
                "name": "Delete this one",
                "enabled": True,
                "type": "schedule",
                "schedule": {"cron": "0 12 * * *", "human": "Noon", "timezone": "UTC"},
                "task": "Remove me",
                "autonomy": None,
                "last_triggered": None,
                "trigger_count": 0,
                "created_at": "2026-02-01T00:00:00Z",
            },
        ]
        store, pid = _make_project_store(triggers=triggers)
        mock_tm = MagicMock()
        app = _make_app(store, trigger_manager=mock_tm)
        client = TestClient(app)

        resp = client.delete(f"/api/v2/projects/{pid}/triggers/trg_delete")
        assert resp.status_code == 204

        project = store.get_project(pid)
        remaining = project.get("triggers", [])
        assert len(remaining) == 1
        assert remaining[0]["id"] == "trg_keep"
        assert remaining[0]["trigger_count"] == 5
