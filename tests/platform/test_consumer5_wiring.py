# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Wiring tests for Consumer 5: REST Platform Endpoints.

Tests verify that the following REST endpoints exist and correctly delegate
to PlatformProvider methods:

1. GET  /api/v2/platform/status     -> provider.get_capabilities()
2. POST /api/v2/platform/setup      -> provider.setup()
3. POST /api/v2/platform/teardown   -> provider.teardown()
4. GET  /api/v2/platform/folders    -> provider.get_available_folders()
5. POST /api/v2/platform/folders/grant  -> provider.grant_folder_access()
6. POST /api/v2/platform/folders/revoke -> provider.revoke_folder_access()

Uses FastAPI TestClient. The platform provider is mocked in the app.
"""

import json
from dataclasses import asdict
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent_os.platform.types import (
    FolderInfo,
    PermissionResult,
    PlatformCapabilities,
    SetupResult,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_capabilities(setup_complete=True):
    return PlatformCapabilities(
        platform="windows",
        isolation_method="sandbox_user",
        setup_complete=setup_complete,
        setup_issues=[] if setup_complete else ["Not set up"],
        supports_network_restriction=True,
        supports_folder_access=True,
        sandbox_username="AgentOS-Worker" if setup_complete else None,
    )


def _make_mock_provider():
    """Create a mock platform provider with sensible defaults."""
    provider = MagicMock()
    provider.get_capabilities.return_value = _make_capabilities(setup_complete=True)
    provider.setup = AsyncMock(return_value=SetupResult(success=True))
    provider.teardown = AsyncMock(return_value=SetupResult(success=True))
    provider.get_available_folders.return_value = [
        FolderInfo(path="C:\\Users\\Test\\Desktop", display_name="Desktop",
                   accessible=False, access_note=None),
        FolderInfo(path="C:\\Users\\Test\\Documents", display_name="Documents",
                   accessible=True, access_note="read_write"),
    ]
    provider.grant_folder_access.return_value = PermissionResult(
        success=True, path="C:\\Users\\Test\\Desktop"
    )
    provider.revoke_folder_access.return_value = PermissionResult(
        success=True, path="C:\\Users\\Test\\Desktop"
    )
    return provider


@pytest.fixture
def app_client(tmp_path):
    """Create a TestClient with a mocked platform provider."""
    from agent_os.api.app import create_app
    from starlette.testclient import TestClient

    mock_provider = _make_mock_provider()

    with patch("agent_os.api.app.create_platform_provider", return_value=mock_provider):
        app = create_app(data_dir=str(tmp_path))

    client = TestClient(app)
    client._mock_provider = mock_provider
    return client


@pytest.fixture
def app_client_with_provider(tmp_path):
    """Create a TestClient and return (client, mock_provider) tuple."""
    from agent_os.api.app import create_app
    from starlette.testclient import TestClient

    mock_provider = _make_mock_provider()

    with patch("agent_os.api.app.create_platform_provider", return_value=mock_provider):
        app = create_app(data_dir=str(tmp_path))

    client = TestClient(app)
    return client, mock_provider


# ---------------------------------------------------------------------------
# GET /api/v2/platform/status
# ---------------------------------------------------------------------------


class TestPlatformStatus:
    """GET /api/v2/platform/status -> provider.get_capabilities()."""

    def test_status_returns_200(self, app_client):
        resp = app_client.get("/api/v2/platform/status")
        assert resp.status_code == 200

    def test_status_returns_capabilities(self, app_client):
        resp = app_client.get("/api/v2/platform/status")
        data = resp.json()
        assert data["platform"] == "windows"
        assert data["isolation_method"] == "sandbox_user"
        assert data["setup_complete"] is True
        assert data["supports_network_restriction"] is True
        assert data["supports_folder_access"] is True

    def test_status_calls_get_capabilities(self, app_client_with_provider):
        client, provider = app_client_with_provider
        client.get("/api/v2/platform/status")
        provider.get_capabilities.assert_called()

    def test_status_when_not_setup(self, tmp_path):
        """When setup_complete=False, status returns with that value."""
        from agent_os.api.app import create_app
        from starlette.testclient import TestClient

        mock_provider = _make_mock_provider()
        mock_provider.get_capabilities.return_value = _make_capabilities(
            setup_complete=False
        )

        with patch("agent_os.api.app.create_platform_provider", return_value=mock_provider):
            app = create_app(data_dir=str(tmp_path))
        client = TestClient(app)

        resp = client.get("/api/v2/platform/status")
        assert resp.status_code == 200
        data = resp.json()
        assert data["setup_complete"] is False


# ---------------------------------------------------------------------------
# POST /api/v2/platform/setup
# ---------------------------------------------------------------------------


class TestPlatformSetup:
    """POST /api/v2/platform/setup -> provider.setup()."""

    def test_setup_returns_200_on_success(self, app_client_with_provider):
        client, provider = app_client_with_provider
        resp = client.post("/api/v2/platform/setup")
        assert resp.status_code == 200

    def test_setup_calls_provider(self, app_client_with_provider):
        client, provider = app_client_with_provider
        client.post("/api/v2/platform/setup")
        provider.setup.assert_awaited_once()

    def test_setup_returns_result(self, app_client_with_provider):
        client, provider = app_client_with_provider
        resp = client.post("/api/v2/platform/setup")
        data = resp.json()
        assert data.get("success") is True or data.get("status") == "ok"

    def test_setup_failure_returns_500(self, tmp_path):
        """When setup fails, return 500 with error."""
        from agent_os.api.app import create_app
        from starlette.testclient import TestClient

        mock_provider = _make_mock_provider()
        mock_provider.setup = AsyncMock(return_value=SetupResult(
            success=False, error="Failed to create sandbox user"
        ))

        with patch("agent_os.api.app.create_platform_provider", return_value=mock_provider):
            app = create_app(data_dir=str(tmp_path))
        client = TestClient(app)

        resp = client.post("/api/v2/platform/setup")
        assert resp.status_code == 500 or "error" in resp.json() or "detail" in resp.json()


# ---------------------------------------------------------------------------
# POST /api/v2/platform/teardown
# ---------------------------------------------------------------------------


class TestPlatformTeardown:
    """POST /api/v2/platform/teardown -> provider.teardown()."""

    def test_teardown_returns_200_on_success(self, app_client_with_provider):
        client, provider = app_client_with_provider
        resp = client.post("/api/v2/platform/teardown")
        assert resp.status_code == 200

    def test_teardown_calls_provider(self, app_client_with_provider):
        client, provider = app_client_with_provider
        client.post("/api/v2/platform/teardown")
        provider.teardown.assert_awaited_once()

    def test_teardown_failure_returns_500(self, tmp_path):
        """When teardown fails, return 500."""
        from agent_os.api.app import create_app
        from starlette.testclient import TestClient

        mock_provider = _make_mock_provider()
        mock_provider.teardown = AsyncMock(return_value=SetupResult(
            success=False, error="Cannot remove user"
        ))

        with patch("agent_os.api.app.create_platform_provider", return_value=mock_provider):
            app = create_app(data_dir=str(tmp_path))
        client = TestClient(app)

        resp = client.post("/api/v2/platform/teardown")
        assert resp.status_code == 500 or "error" in resp.json() or "detail" in resp.json()


# ---------------------------------------------------------------------------
# GET /api/v2/platform/folders
# ---------------------------------------------------------------------------


class TestPlatformFolders:
    """GET /api/v2/platform/folders -> provider.get_available_folders()."""

    def test_folders_returns_200(self, app_client):
        resp = app_client.get("/api/v2/platform/folders")
        assert resp.status_code == 200

    def test_folders_returns_list(self, app_client):
        resp = app_client.get("/api/v2/platform/folders")
        data = resp.json()
        folders = data.get("folders", data)
        assert isinstance(folders, list)
        assert len(folders) == 2

    def test_folders_have_expected_fields(self, app_client):
        resp = app_client.get("/api/v2/platform/folders")
        data = resp.json()
        folders = data.get("folders", data)
        folder = folders[0]
        assert "path" in folder
        assert "display_name" in folder
        assert "accessible" in folder

    def test_folders_calls_provider(self, app_client_with_provider):
        client, provider = app_client_with_provider
        client.get("/api/v2/platform/folders")
        provider.get_available_folders.assert_called()


# ---------------------------------------------------------------------------
# POST /api/v2/platform/folders/grant
# ---------------------------------------------------------------------------


class TestPlatformFolderGrant:
    """POST /api/v2/platform/folders/grant -> provider.grant_folder_access()."""

    def test_grant_returns_200_on_success(self, app_client_with_provider):
        client, provider = app_client_with_provider
        resp = client.post("/api/v2/platform/folders/grant", json={
            "path": "C:\\Users\\Test\\Desktop",
            "mode": "read_write",
        })
        assert resp.status_code == 200

    def test_grant_calls_provider(self, app_client_with_provider):
        client, provider = app_client_with_provider
        client.post("/api/v2/platform/folders/grant", json={
            "path": "C:\\Test",
            "mode": "read_only",
        })
        provider.grant_folder_access.assert_called_once_with("C:\\Test", "read_only")

    def test_grant_returns_result(self, app_client_with_provider):
        client, provider = app_client_with_provider
        resp = client.post("/api/v2/platform/folders/grant", json={
            "path": "C:\\Test",
            "mode": "read_write",
        })
        data = resp.json()
        assert data.get("success") is True or data.get("status") == "ok"

    def test_grant_failure_returns_error(self, tmp_path):
        """When grant fails, return error."""
        from agent_os.api.app import create_app
        from starlette.testclient import TestClient

        mock_provider = _make_mock_provider()
        mock_provider.grant_folder_access.return_value = PermissionResult(
            success=False, path="C:\\Protected", error="ACL error"
        )

        with patch("agent_os.api.app.create_platform_provider", return_value=mock_provider):
            app = create_app(data_dir=str(tmp_path))
        client = TestClient(app)

        resp = client.post("/api/v2/platform/folders/grant", json={
            "path": "C:\\Protected",
            "mode": "read_write",
        })
        # Should indicate failure
        data = resp.json()
        assert resp.status_code >= 400 or data.get("success") is False or "error" in data


# ---------------------------------------------------------------------------
# POST /api/v2/platform/folders/revoke
# ---------------------------------------------------------------------------


class TestPlatformFolderRevoke:
    """POST /api/v2/platform/folders/revoke -> provider.revoke_folder_access()."""

    def test_revoke_returns_200_on_success(self, app_client_with_provider):
        client, provider = app_client_with_provider
        resp = client.post("/api/v2/platform/folders/revoke", json={
            "path": "C:\\Users\\Test\\Desktop",
        })
        assert resp.status_code == 200

    def test_revoke_calls_provider(self, app_client_with_provider):
        client, provider = app_client_with_provider
        client.post("/api/v2/platform/folders/revoke", json={
            "path": "C:\\Test",
        })
        provider.revoke_folder_access.assert_called_once_with("C:\\Test")

    def test_revoke_failure_returns_error(self, tmp_path):
        """When revoke fails, return error."""
        from agent_os.api.app import create_app
        from starlette.testclient import TestClient

        mock_provider = _make_mock_provider()
        mock_provider.revoke_folder_access.return_value = PermissionResult(
            success=False, path="C:\\Protected", error="Cannot revoke"
        )

        with patch("agent_os.api.app.create_platform_provider", return_value=mock_provider):
            app = create_app(data_dir=str(tmp_path))
        client = TestClient(app)

        resp = client.post("/api/v2/platform/folders/revoke", json={
            "path": "C:\\Protected",
        })
        data = resp.json()
        assert resp.status_code >= 400 or data.get("success") is False or "error" in data


# ---------------------------------------------------------------------------
# Existing routes still work after platform routes are added
# ---------------------------------------------------------------------------


class TestExistingRoutesNotBroken:
    """Existing agent routes must still work after platform routes are added."""

    def test_list_projects_still_works(self, app_client):
        resp = app_client.get("/api/v2/projects")
        assert resp.status_code == 200

    def test_create_project_still_works(self, app_client, tmp_path):
        resp = app_client.post("/api/v2/projects", json={
            "name": "Test",
            "workspace": str(tmp_path),
            "model": "gpt-4",
            "api_key": "sk-test",
        })
        assert resp.status_code == 201

    def test_no_url_conflicts(self, app_client):
        """Platform routes (/api/v2/platform/...) don't conflict with agent routes (/api/v2/...)."""
        # Agent routes should still 404 correctly
        resp = app_client.post("/api/v2/agents/nonexistent/stop")
        assert resp.status_code == 404

        # Platform status should work
        resp = app_client.get("/api/v2/platform/status")
        assert resp.status_code == 200
