# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Tests for agent_os.relay.device — device identity management."""

import json
from pathlib import Path

import pytest
import httpx

from agent_os.relay.device import get_or_create_device_identity, register_device


class TestGetOrCreateDeviceIdentity:
    def test_creates_device_json_on_first_call(self, tmp_path):
        identity = get_or_create_device_identity(config_dir=tmp_path)

        device_file = tmp_path / "device.json"
        assert device_file.exists()

        data = json.loads(device_file.read_text())
        assert "device_id" in data
        assert "device_secret" in data
        assert data["device_id"].startswith("dev_")
        assert len(data["device_secret"]) == 64  # 32 bytes hex

    def test_second_call_returns_same_values(self, tmp_path):
        first = get_or_create_device_identity(config_dir=tmp_path)
        second = get_or_create_device_identity(config_dir=tmp_path)

        assert first["device_id"] == second["device_id"]
        assert first["device_secret"] == second["device_secret"]

    def test_creates_config_dir_if_missing(self, tmp_path):
        nested = tmp_path / "deep" / "nested" / "dir"
        identity = get_or_create_device_identity(config_dir=nested)

        assert nested.exists()
        assert (nested / "device.json").exists()
        assert identity["device_id"].startswith("dev_")

    def test_file_contains_valid_json(self, tmp_path):
        get_or_create_device_identity(config_dir=tmp_path)
        raw = (tmp_path / "device.json").read_text()
        data = json.loads(raw)  # should not raise
        assert isinstance(data, dict)
        assert set(data.keys()) == {"device_id", "device_secret"}


class TestRegisterDevice:
    @pytest.mark.asyncio
    async def test_register_device_posts_to_relay(self, httpx_mock):
        """register_device sends POST /relay/devices with correct payload."""
        httpx_mock.route(method="POST", url="https://relay.example.com/relay/devices").mock(
            return_value=httpx.Response(200, json={"status": "registered"})
        )

        result = await register_device(
            "https://relay.example.com", "dev_abc123", "secret_xyz"
        )
        assert result == {"status": "registered"}

    @pytest.mark.asyncio
    async def test_register_device_raises_on_error(self, httpx_mock):
        """register_device raises on non-2xx response."""
        httpx_mock.route(method="POST", url="https://relay.example.com/relay/devices").mock(
            return_value=httpx.Response(409, json={"detail": "already exists"})
        )

        with pytest.raises(httpx.HTTPStatusError):
            await register_device(
                "https://relay.example.com", "dev_abc123", "secret_xyz"
            )


@pytest.fixture
def httpx_mock(monkeypatch):
    """Lightweight httpx mock that intercepts AsyncClient requests."""
    return _HttpxMock(monkeypatch)


class _HttpxMock:
    """Minimal httpx mock for unit tests without external dependencies."""

    def __init__(self, monkeypatch):
        self._monkeypatch = monkeypatch
        self._routes = []
        self._installed = False

    def route(self, method="GET", url=None):
        entry = _MockRoute(method=method.upper(), url=url)
        self._routes.append(entry)
        if not self._installed:
            self._install()
        return entry

    def _install(self):
        self._installed = True
        mock_self = self

        class MockAsyncClient:
            def __init__(self, **kwargs):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                pass

            async def post(self, url, **kwargs):
                return mock_self._find("POST", url)

            async def get(self, url, **kwargs):
                return mock_self._find("GET", url)

            async def request(self, method, url, **kwargs):
                return mock_self._find(method.upper(), url)

        self._monkeypatch.setattr(httpx, "AsyncClient", MockAsyncClient)

    def _find(self, method, url):
        for route in self._routes:
            if route.method == method and (route.url is None or route.url == url):
                resp = route.response
                # Attach a request so raise_for_status() works
                if resp._request is None:
                    resp._request = httpx.Request(method, url)
                return resp
        raise RuntimeError(f"No mock for {method} {url}")


class _MockRoute:
    def __init__(self, method, url):
        self.method = method
        self.url = url
        self.response = None

    def mock(self, return_value):
        self.response = return_value
        return self
