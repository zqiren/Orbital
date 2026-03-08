# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Tests for agent_os.relay.client — RelayClient tunnel client."""

import asyncio
import json
import time

import pytest

from agent_os.relay.client import RelayClient


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class FakeProjectStore:
    """Minimal project store for testing push logic."""

    def __init__(self, projects=None):
        self._projects = projects or {}

    def get_project(self, project_id):
        return self._projects.get(project_id)


class FakeWebSocket:
    """Minimal fake WebSocket for testing RelayClient message handling."""

    def __init__(self):
        self.sent: list[dict] = []
        self.incoming: asyncio.Queue = asyncio.Queue()
        self.closed = False

    async def send(self, data: str):
        self.sent.append(json.loads(data))

    async def close(self):
        self.closed = True

    def inject(self, msg: dict):
        """Queue a message that the client will receive."""
        self.incoming.put_nowait(json.dumps(msg))

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            return await asyncio.wait_for(self.incoming.get(), timeout=1.0)
        except asyncio.TimeoutError:
            raise StopAsyncIteration


# ---------------------------------------------------------------------------
# _should_push tests
# ---------------------------------------------------------------------------

class TestShouldPush:
    def setup_method(self):
        self.store = FakeProjectStore({
            "proj_1": {
                "project_id": "proj_1",
                "notification_prefs": {
                    "task_completed": True,
                    "errors": True,
                    "agent_messages": True,
                    "trigger_started": False,
                },
            },
        })
        self.client = RelayClient(
            relay_url="https://relay.example.com",
            device_id="dev_test",
            device_secret="secret",
            project_store=self.store,
        )

    def test_push_for_approval_request(self):
        event = {"type": "approval.request", "tool_args": "rm -rf /tmp/test"}
        result = self.client._should_push(event, "proj_1")
        assert result is not None
        assert result["title"] == "Agent needs approval"
        assert "rm -rf" in result["body"]

    def test_push_for_agent_error(self):
        event = {"type": "agent.status", "status": "error", "reason": "API key invalid"}
        result = self.client._should_push(event, "proj_1")
        assert result is not None
        assert result["title"] == "Agent error"
        assert "API key" in result["body"]

    def test_no_push_for_stream_delta(self):
        event = {"type": "chat.stream_delta", "text": "Hello"}
        result = self.client._should_push(event, "proj_1")
        assert result is None

    def test_no_push_for_agent_activity(self):
        event = {"type": "agent.activity", "category": "file_read"}
        result = self.client._should_push(event, "proj_1")
        assert result is None

    def test_no_push_for_agent_status_running(self):
        event = {"type": "agent.status", "status": "running"}
        result = self.client._should_push(event, "proj_1")
        assert result is None

    def test_push_body_truncated_to_100_chars(self):
        event = {"type": "approval.request", "tool_args": "x" * 200}
        result = self.client._should_push(event, "proj_1")
        assert len(result["body"]) <= 100


class TestShouldPushTiered:
    """Tests for the three-tier push logic."""

    def _make_client(self, prefs=None):
        projects = {}
        if prefs is not None:
            projects["proj_1"] = {
                "project_id": "proj_1",
                "notification_prefs": prefs,
            }
        store = FakeProjectStore(projects)
        return RelayClient(
            relay_url="https://relay.example.com",
            device_id="dev_test",
            device_secret="secret",
            project_store=store,
        )

    # -- Tier 1 --

    def test_approval_request_always_pushes(self):
        client = self._make_client()
        result = client._should_push({"type": "approval.request", "tool_args": "rm"}, "proj_1")
        assert result is not None
        assert result["title"] == "Agent needs approval"

    def test_budget_exhausted_always_pushes(self):
        client = self._make_client()
        result = client._should_push({"type": "budget.exhausted"}, "proj_1")
        assert result is not None
        assert result["title"] == "Budget exhausted"

    # -- Tier 2: agent.notify --

    def test_agent_notify_normal_pushes_when_prefs_enabled(self):
        client = self._make_client({"agent_messages": True, "errors": True, "task_completed": True, "trigger_started": False})
        result = client._should_push(
            {"type": "agent.notify", "title": "Hi", "body": "World", "urgency": "normal"},
            "proj_1",
        )
        assert result is not None
        assert result["title"] == "Hi"
        assert result["body"] == "World"

    def test_agent_notify_returns_none_when_prefs_disabled(self):
        client = self._make_client({"agent_messages": False, "errors": True, "task_completed": True, "trigger_started": False})
        result = client._should_push(
            {"type": "agent.notify", "title": "Hi", "body": "World", "urgency": "normal"},
            "proj_1",
        )
        assert result is None

    def test_agent_notify_high_bypasses_prefs(self):
        client = self._make_client({"agent_messages": False, "errors": True, "task_completed": True, "trigger_started": False})
        result = client._should_push(
            {"type": "agent.notify", "title": "Alert", "body": "Critical", "urgency": "high"},
            "proj_1",
        )
        assert result is not None
        assert result["title"] == "Alert"

    def test_agent_notify_low_never_pushes(self):
        client = self._make_client({"agent_messages": True, "errors": True, "task_completed": True, "trigger_started": False})
        result = client._should_push(
            {"type": "agent.notify", "title": "FYI", "body": "Minor", "urgency": "low"},
            "proj_1",
        )
        assert result is None

    # -- Tier 2: agent.status --

    def test_agent_status_error_pushes_when_prefs_enabled(self):
        client = self._make_client({"errors": True, "agent_messages": True, "task_completed": True, "trigger_started": False})
        result = client._should_push(
            {"type": "agent.status", "status": "error", "reason": "Timeout"},
            "proj_1",
        )
        assert result is not None
        assert result["title"] == "Agent error"

    def test_agent_status_error_returns_none_when_prefs_disabled(self):
        client = self._make_client({"errors": False, "agent_messages": True, "task_completed": True, "trigger_started": False})
        result = client._should_push(
            {"type": "agent.status", "status": "error", "reason": "Timeout"},
            "proj_1",
        )
        assert result is None

    # -- Rate limiting --

    def test_rate_limiting_blocks_6th_notification(self):
        client = self._make_client({"agent_messages": True, "errors": True, "task_completed": True, "trigger_started": False})
        event = {"type": "agent.notify", "title": "Hi", "body": "World", "urgency": "normal"}
        for i in range(5):
            result = client._should_push(event, "proj_1")
            assert result is not None, f"Push {i+1} should succeed"
        # 6th should be rate-limited
        result = client._should_push(event, "proj_1")
        assert result is None

    def test_rate_limiting_does_not_apply_to_approval(self):
        client = self._make_client()
        # Fill up rate limit
        client._push_counts["proj_1"] = [time.time()] * 10
        result = client._should_push({"type": "approval.request", "tool_args": "rm"}, "proj_1")
        assert result is not None

    # -- Trigger source / scheduled label --

    def test_scheduled_approval_has_prefix(self):
        client = self._make_client()
        event = {"type": "approval.request", "what": "Run rm", "trigger_source": "schedule"}
        result = client._should_push(event, "proj_1")
        assert result is not None
        assert result["title"].startswith("(scheduled)")

    def test_scheduled_error_has_prefix(self):
        client = self._make_client({"errors": True, "agent_messages": True, "task_completed": True, "trigger_started": False})
        event = {"type": "agent.status", "status": "error", "reason": "Timeout", "trigger_source": "schedule"}
        result = client._should_push(event, "proj_1")
        assert result["title"].startswith("(scheduled)")

    def test_scheduled_completion_has_prefix(self):
        client = self._make_client({"errors": True, "agent_messages": True, "task_completed": True, "trigger_started": False})
        event = {"type": "agent.status", "status": "idle", "had_activity": True, "trigger_source": "schedule"}
        result = client._should_push(event, "proj_1")
        assert result["title"].startswith("(scheduled)")

    def test_trigger_started_pushes_when_pref_enabled(self):
        client = self._make_client({"errors": True, "agent_messages": True, "task_completed": True, "trigger_started": True})
        event = {"type": "agent.status", "status": "running", "trigger_source": "schedule"}
        result = client._should_push(event, "proj_1")
        assert result is not None
        assert result["title"] == "(scheduled) Run started"

    def test_trigger_started_no_push_when_pref_disabled(self):
        client = self._make_client({"errors": True, "agent_messages": True, "task_completed": True, "trigger_started": False})
        event = {"type": "agent.status", "status": "running", "trigger_source": "schedule"}
        result = client._should_push(event, "proj_1")
        assert result is None

    # -- Approval body uses `what` field --

    def test_approval_body_uses_what_field(self):
        client = self._make_client()
        event = {"type": "approval.request", "what": "Execute: rm -rf /tmp/test", "tool_args": {"path": "/tmp/test"}}
        result = client._should_push(event, "proj_1")
        assert result["body"] == "Execute: rm -rf /tmp/test"

    def test_approval_body_falls_back_to_tool_args(self):
        client = self._make_client()
        event = {"type": "approval.request", "tool_args": "delete file"}
        result = client._should_push(event, "proj_1")
        assert result["body"] == "delete file"


# ---------------------------------------------------------------------------
# forward_event tests
# ---------------------------------------------------------------------------

class TestForwardEvent:
    def setup_method(self):
        self.client = RelayClient(
            relay_url="https://relay.example.com",
            device_id="dev_test",
            device_secret="secret",
        )

    @pytest.mark.asyncio
    async def test_forward_event_sends_envelope(self):
        ws = FakeWebSocket()
        self.client._ws = ws

        await self.client.forward_event("proj_123", {
            "type": "chat.stream_delta",
            "text": "hello",
        })

        assert len(ws.sent) == 1
        msg = ws.sent[0]
        assert msg["type"] == "event.forward"
        assert msg["event"]["project_id"] == "proj_123"
        assert msg["event"]["type"] == "chat.stream_delta"
        assert "push" not in msg

    @pytest.mark.asyncio
    async def test_forward_event_with_push(self):
        ws = FakeWebSocket()
        self.client._ws = ws

        await self.client.forward_event("proj_123", {
            "type": "approval.request",
            "tool_args": "delete file",
        })

        msg = ws.sent[0]
        assert "push" in msg
        assert msg["push"]["title"] == "Agent needs approval"

    @pytest.mark.asyncio
    async def test_forward_event_noop_when_disconnected(self):
        """forward_event does nothing if WebSocket is not connected."""
        self.client._ws = None
        # Should not raise
        await self.client.forward_event("proj_123", {"type": "test"})


# ---------------------------------------------------------------------------
# REST tunnel tests
# ---------------------------------------------------------------------------

class TestTunnelRestRequest:
    def setup_method(self):
        self.client = RelayClient(
            relay_url="https://relay.example.com",
            device_id="dev_test",
            device_secret="secret",
        )

    @pytest.mark.asyncio
    async def test_tunnel_rest_request_proxies_to_daemon(self, monkeypatch):
        """rest.request message triggers local HTTP call and sends rest.response."""
        ws = FakeWebSocket()
        self.client._ws = ws

        # Mock httpx to capture the request
        captured = {}

        class MockAsyncClient:
            def __init__(self, **kwargs):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                pass

            async def request(self, method, url, **kwargs):
                captured["method"] = method
                captured["url"] = url
                captured["headers"] = kwargs.get("headers", {})
                return FakeResponse(200, {"projects": []})

        import httpx
        monkeypatch.setattr(httpx, "AsyncClient", MockAsyncClient)

        await self.client._tunnel_rest_request({
            "request_id": "req_001",
            "method": "GET",
            "path": "/api/v2/projects",
            "headers": {},
            "body": None,
        })

        # Verify local HTTP call was made
        assert captured["method"] == "GET"
        assert captured["url"] == "http://localhost:8000/api/v2/projects"
        assert captured["headers"]["X-Via-Relay"] == "true"

        # Verify rest.response was sent back
        assert len(ws.sent) == 1
        resp = ws.sent[0]
        assert resp["type"] == "rest.response"
        assert resp["request_id"] == "req_001"
        assert resp["status"] == 200

    @pytest.mark.asyncio
    async def test_tunnel_rest_returns_502_on_error(self, monkeypatch):
        """If local daemon request fails, send 502 rest.response."""
        ws = FakeWebSocket()
        self.client._ws = ws

        class MockAsyncClient:
            def __init__(self, **kwargs):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                pass

            async def request(self, method, url, **kwargs):
                raise ConnectionError("daemon down")

        import httpx
        monkeypatch.setattr(httpx, "AsyncClient", MockAsyncClient)

        await self.client._tunnel_rest_request({
            "request_id": "req_002",
            "method": "GET",
            "path": "/api/v2/projects",
            "headers": {},
            "body": None,
        })

        resp = ws.sent[0]
        assert resp["type"] == "rest.response"
        assert resp["request_id"] == "req_002"
        assert resp["status"] == 502


# ---------------------------------------------------------------------------
# Pairing helpers
# ---------------------------------------------------------------------------

class TestPairingCompleteNotification:
    """pairing.complete is a notification, not a request/response.
    It should persist the device via add_paired_device, not go through _resolve_pending.
    """

    def setup_method(self):
        self.client = RelayClient(
            relay_url="https://relay.example.com",
            device_id="dev_test",
            device_secret="secret",
        )

    @pytest.mark.asyncio
    async def test_pairing_complete_persists_device(self, monkeypatch):
        recorded = []
        monkeypatch.setattr(
            "agent_os.api.routes.pairing.add_paired_device",
            lambda phone_id: recorded.append(phone_id),
        )

        await self.client._handle_tunnel_message({
            "type": "pairing.complete",
            "phone_id": "ph_new_123",
            "device_id": "dev_test",
        })

        assert recorded == ["ph_new_123"]

    @pytest.mark.asyncio
    async def test_pairing_complete_without_phone_id_is_noop(self, monkeypatch):
        recorded = []
        monkeypatch.setattr(
            "agent_os.api.routes.pairing.add_paired_device",
            lambda phone_id: recorded.append(phone_id),
        )

        await self.client._handle_tunnel_message({
            "type": "pairing.complete",
            # no phone_id
        })

        assert recorded == []


class TestPairingHelpers:
    def setup_method(self):
        self.client = RelayClient(
            relay_url="https://relay.example.com",
            device_id="dev_test",
            device_secret="secret",
        )

    @pytest.mark.asyncio
    async def test_send_pairing_create_returns_code(self):
        ws = FakeWebSocket()
        self.client._ws = ws

        async def respond():
            """Simulate relay sending back a pairing.code after a short delay."""
            await asyncio.sleep(0.05)
            # Find the request_id from the sent message
            assert len(ws.sent) >= 1
            req_id = ws.sent[0].get("request_id")
            self.client._resolve_pending(req_id, {
                "type": "pairing.code",
                "request_id": req_id,
                "code": "123456",
            })

        task = asyncio.create_task(respond())
        result = await self.client.send_pairing_create(timeout=5.0)
        await task

        assert result["code"] == "123456"

    @pytest.mark.asyncio
    async def test_send_pairing_create_timeout(self):
        ws = FakeWebSocket()
        self.client._ws = ws

        with pytest.raises(asyncio.TimeoutError):
            await self.client.send_pairing_create(timeout=0.1)

    @pytest.mark.asyncio
    async def test_send_pairing_revoke(self):
        ws = FakeWebSocket()
        self.client._ws = ws

        await self.client.send_pairing_revoke("phone_abc")

        assert len(ws.sent) == 1
        msg = ws.sent[0]
        assert msg["type"] == "pairing.revoke"
        assert msg["phone_id"] == "phone_abc"


# ---------------------------------------------------------------------------
# Connection lifecycle
# ---------------------------------------------------------------------------

class TestLifecycle:
    def test_stop_sets_running_false(self):
        client = RelayClient(
            relay_url="https://relay.example.com",
            device_id="dev_test",
            device_secret="secret",
        )
        assert not client._running

    @pytest.mark.asyncio
    async def test_stop_closes_websocket(self):
        client = RelayClient(
            relay_url="https://relay.example.com",
            device_id="dev_test",
            device_secret="secret",
        )
        ws = FakeWebSocket()
        client._ws = ws
        client._running = True

        await client.stop()

        assert not client._running
        assert ws.closed
        assert client._ws is None


# ---------------------------------------------------------------------------
# FakeResponse helper for httpx mocking
# ---------------------------------------------------------------------------

class FakeResponse:
    """Minimal httpx.Response-like object for testing."""

    def __init__(self, status_code, body=None, headers=None):
        self.status_code = status_code
        self._body = body
        self.headers = headers or {}
        self.text = json.dumps(body) if body is not None else ""

    def json(self):
        if self._body is not None:
            return self._body
        raise ValueError("No JSON body")
