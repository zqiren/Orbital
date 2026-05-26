# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Unit tests for Component F — Daemon Assembly.

Tests cover: models, AgentManager, SubAgentManager, MessageRouter,
ProcessManager, AutonomyInterceptor, ActivityTranslator, ProjectStore,
REST endpoints (FastAPI TestClient), and WebSocket protocol.

All Components A-E are mocked.
"""

import asyncio
import json
import os
import time
from dataclasses import dataclass
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

import pytest
import pytest_asyncio

# ---------------------------------------------------------------------------
# models.py tests
# ---------------------------------------------------------------------------


class TestModels:
    """Tests for daemon_v2/models.py data models."""

    def test_agent_status_enum_values(self):
        from agent_os.daemon_v2.models import AgentStatus

        assert AgentStatus.RUNNING == "running"
        assert AgentStatus.IDLE == "idle"
        assert AgentStatus.PAUSED == "paused"
        assert AgentStatus.STOPPED == "stopped"
        assert AgentStatus.ERROR == "error"

    def test_detect_os_returns_string(self):
        from agent_os.daemon_v2.models import detect_os

        result = detect_os()
        assert result in ("windows", "macos", "linux")

    def test_detect_os_windows(self):
        from agent_os.daemon_v2.models import detect_os

        with patch("platform.system", return_value="Windows"):
            assert detect_os() == "windows"

    def test_detect_os_darwin(self):
        from agent_os.daemon_v2.models import detect_os

        with patch("platform.system", return_value="Darwin"):
            assert detect_os() == "macos"

    def test_detect_os_linux(self):
        from agent_os.daemon_v2.models import detect_os

        with patch("platform.system", return_value="Linux"):
            assert detect_os() == "linux"

    def test_agent_config_defaults(self):
        from agent_os.daemon_v2.models import AgentConfig
        from agent_os.agent.prompt_builder import Autonomy

        cfg = AgentConfig(workspace="/tmp/ws", model="gpt-4", api_key="sk-test")
        assert cfg.max_iterations == 0  # 0 = unlimited (token budget is the cap)
        assert cfg.token_budget == 100_000_000
        assert cfg.autonomy == Autonomy.HANDS_OFF
        assert cfg.enabled_agents == []
        assert cfg.base_url is None
        assert cfg.utility_model is None
        assert cfg.search_api_key is None
        assert cfg.project_instructions == ""

    def test_agent_config_full(self):
        from agent_os.daemon_v2.models import AgentConfig
        from agent_os.agent.prompt_builder import Autonomy

        cfg = AgentConfig(
            workspace="/tmp/ws",
            model="gpt-4",
            api_key="sk-test",
            base_url="http://localhost:8080",
            max_iterations=100,
            token_budget=1_000_000,
            utility_model="gpt-3.5",
            search_api_key="tavily-key",
            autonomy=Autonomy.CHECK_IN,
            enabled_agents=["claudecode"],
            project_instructions="Do the thing",
        )
        assert cfg.base_url == "http://localhost:8080"
        assert cfg.autonomy == Autonomy.CHECK_IN
        assert cfg.enabled_agents == ["claudecode"]

    def test_activity_event_fields(self):
        from agent_os.daemon_v2.models import ActivityEvent

        evt = ActivityEvent(
            id="abc",
            project_id="proj_1",
            category="file_read",
            description="Reading foo.py",
            tool_name="read",
            source="management",
            timestamp="2026-02-11T00:00:00Z",
        )
        assert evt.category == "file_read"
        assert evt.source == "management"
        assert evt.tool_name == "read"

    def test_default_session_id_constant(self):
        from agent_os.daemon_v2.models import DEFAULT_SESSION_ID

        # The constant must be a non-empty string; back-compat code paths
        # rely on it as a stable sentinel for the single-loop default.
        assert isinstance(DEFAULT_SESSION_ID, str)
        assert DEFAULT_SESSION_ID == "default"

    def test_make_session_key_with_explicit_session(self):
        from agent_os.daemon_v2.models import make_session_key

        key = make_session_key("proj_1", "sess_abc")
        assert key == ("proj_1", "sess_abc")
        # ``SessionKey`` is a plain tuple alias — concrete value must be a tuple.
        assert isinstance(key, tuple)
        assert len(key) == 2

    def test_make_session_key_default(self):
        from agent_os.daemon_v2.models import DEFAULT_SESSION_ID, make_session_key

        # Both omission and explicit ``None`` collapse to the default sentinel.
        assert make_session_key("proj_1") == ("proj_1", DEFAULT_SESSION_ID)
        assert make_session_key("proj_1", None) == ("proj_1", DEFAULT_SESSION_ID)

    def test_make_session_key_empty_string_treated_as_none(self):
        from agent_os.daemon_v2.models import DEFAULT_SESSION_ID, make_session_key

        # Empty string is falsy — we collapse to the default so callers do not
        # have to special-case "" themselves.
        assert make_session_key("proj_1", "") == ("proj_1", DEFAULT_SESSION_ID)


# ---------------------------------------------------------------------------
# MessageRouter tests
# ---------------------------------------------------------------------------


class TestMessageRouter:
    """Tests for daemon_v2/message_router.py."""

    def test_route_with_mention(self):
        from agent_os.daemon_v2.message_router import MessageRouter

        router = MessageRouter(known_handles={"claudecode", "aider"})
        target, text = router.route("@claudecode analyse this code")
        assert target == "claudecode"
        assert text == "analyse this code"

    def test_route_without_mention(self):
        from agent_os.daemon_v2.message_router import MessageRouter

        router = MessageRouter(known_handles={"claudecode"})
        target, text = router.route("do something please")
        assert target is None
        assert text == "do something please"

    def test_route_case_insensitive(self):
        from agent_os.daemon_v2.message_router import MessageRouter

        router = MessageRouter(known_handles={"claudecode"})
        target, text = router.route("@ClaudeCode do X")
        assert target == "claudecode"
        assert text == "do X"

    def test_route_unknown_handle(self):
        from agent_os.daemon_v2.message_router import MessageRouter

        router = MessageRouter(known_handles={"claudecode"})
        target, text = router.route("@unknown do X")
        assert target is None
        assert text == "@unknown do X"

    def test_route_mention_not_at_start(self):
        from agent_os.daemon_v2.message_router import MessageRouter

        router = MessageRouter(known_handles={"claudecode"})
        target, text = router.route("tell @claudecode to do X")
        assert target is None
        assert text == "tell @claudecode to do X"

    def test_route_empty_message(self):
        from agent_os.daemon_v2.message_router import MessageRouter

        router = MessageRouter(known_handles={"claudecode"})
        target, text = router.route("")
        assert target is None
        assert text == ""

    def test_route_mention_only(self):
        from agent_os.daemon_v2.message_router import MessageRouter

        router = MessageRouter(known_handles={"claudecode"})
        target, text = router.route("@claudecode")
        assert target == "claudecode"
        assert text == ""


# ---------------------------------------------------------------------------
# AutonomyInterceptor tests
# ---------------------------------------------------------------------------


class TestAutonomyInterceptor:
    """Tests for daemon_v2/autonomy.py."""

    def _make_interceptor(self, preset, ws=None, project_id="proj_1"):
        from agent_os.daemon_v2.autonomy import AutonomyInterceptor

        ws = ws or MagicMock()
        return AutonomyInterceptor(preset=preset, ws_manager=ws, project_id=project_id)

    def test_hands_off_only_intercepts_request_access(self):
        from agent_os.agent.prompt_builder import Autonomy

        interceptor = self._make_interceptor(Autonomy.HANDS_OFF)
        # should NOT intercept shell
        assert not interceptor.should_intercept({"id": "1", "name": "shell", "arguments": {}})
        # should NOT intercept write
        assert not interceptor.should_intercept({"id": "2", "name": "write", "arguments": {}})
        # SHOULD intercept request_access
        assert interceptor.should_intercept({"id": "3", "name": "request_access", "arguments": {}})

    def test_check_in_intercepts_shell(self):
        from agent_os.agent.prompt_builder import Autonomy

        interceptor = self._make_interceptor(Autonomy.CHECK_IN)
        assert interceptor.should_intercept({"id": "1", "name": "shell", "arguments": {}})

    def test_check_in_intercepts_request_access(self):
        from agent_os.agent.prompt_builder import Autonomy

        interceptor = self._make_interceptor(Autonomy.CHECK_IN)
        assert interceptor.should_intercept({"id": "1", "name": "request_access", "arguments": {}})

    def test_check_in_does_not_intercept_read(self):
        from agent_os.agent.prompt_builder import Autonomy

        interceptor = self._make_interceptor(Autonomy.CHECK_IN)
        assert not interceptor.should_intercept({"id": "1", "name": "read", "arguments": {}})

    def test_supervised_intercepts_all_except_read(self):
        from agent_os.agent.prompt_builder import Autonomy

        interceptor = self._make_interceptor(Autonomy.SUPERVISED)
        assert interceptor.should_intercept({"id": "1", "name": "shell", "arguments": {}})
        assert interceptor.should_intercept({"id": "2", "name": "write", "arguments": {}})
        assert interceptor.should_intercept({"id": "3", "name": "edit", "arguments": {}})
        assert not interceptor.should_intercept({"id": "4", "name": "read", "arguments": {}})

    def test_bypass_window_skips_recently_approved(self):
        from agent_os.agent.prompt_builder import Autonomy

        interceptor = self._make_interceptor(Autonomy.CHECK_IN)
        tool_call = {"id": "1", "name": "shell", "arguments": {"command": "ls"}}
        # Initially intercepted
        assert interceptor.should_intercept(tool_call)
        # Record approval
        interceptor.record_approval("shell", {"command": "ls"})
        # Same tool+args should now be bypassed
        assert not interceptor.should_intercept(tool_call)

    def test_bypass_window_expires(self):
        from agent_os.agent.prompt_builder import Autonomy

        interceptor = self._make_interceptor(Autonomy.CHECK_IN)
        interceptor._bypass_window = 0  # instant expiry
        interceptor.record_approval("shell", {"command": "ls"})
        tool_call = {"id": "1", "name": "shell", "arguments": {"command": "ls"}}
        # Should be intercepted again after expiry
        assert interceptor.should_intercept(tool_call)

    def test_on_intercept_broadcasts_approval_request(self):
        from agent_os.agent.prompt_builder import Autonomy

        ws = MagicMock()
        interceptor = self._make_interceptor(Autonomy.CHECK_IN, ws=ws, project_id="proj_1")
        tool_call = {"id": "tc_1", "name": "shell", "arguments": {"command": "rm -rf /"}}
        interceptor.on_intercept(tool_call, [{"role": "user", "content": "do it"}])
        ws.broadcast.assert_called_once()
        payload = ws.broadcast.call_args[0][1]
        assert payload["type"] == "approval.request"
        assert payload["project_id"] == "proj_1"
        assert payload["tool_call_id"] == "tc_1"
        assert payload["tool_name"] == "shell"
        assert "tool_args" in payload

    def test_on_intercept_stores_pending(self):
        from agent_os.agent.prompt_builder import Autonomy

        interceptor = self._make_interceptor(Autonomy.CHECK_IN)
        tool_call = {"id": "tc_1", "name": "shell", "arguments": {"command": "ls"}}
        interceptor.on_intercept(tool_call, [])
        assert "tc_1" in interceptor._pending_approvals
        assert interceptor._pending_approvals["tc_1"]["tool_name"] == "shell"

    def test_fail_closed_on_exception(self):
        """If should_intercept raises internally, caller should treat as intercept.
        The interceptor itself may raise; the loop catches it as fail-closed."""
        from agent_os.agent.prompt_builder import Autonomy
        from agent_os.daemon_v2.autonomy import AutonomyInterceptor

        interceptor = self._make_interceptor(Autonomy.CHECK_IN)
        # Monkey-patch to raise
        original = interceptor.should_intercept
        def broken_intercept(tc):
            raise RuntimeError("internal error")
        interceptor.should_intercept = broken_intercept
        # The contract says exceptions propagate — loop treats any exception as DENY
        with pytest.raises(RuntimeError):
            interceptor.should_intercept({"id": "1", "name": "shell", "arguments": {}})

    def test_check_in_does_not_intercept_browser_evaluate(self):
        """evaluate is a read-only JS execution action and should NOT be
        intercepted under CHECK_IN mode (Issue 3: excessive approvals)."""
        from agent_os.agent.prompt_builder import Autonomy

        interceptor = self._make_interceptor(Autonomy.CHECK_IN)
        tool_call = {"id": "1", "name": "browser", "arguments": {"action": "evaluate"}}
        assert not interceptor.should_intercept(tool_call)

    def test_check_in_still_intercepts_browser_write_actions(self):
        """click, type, fill, etc. MUST still be intercepted under CHECK_IN."""
        from agent_os.agent.prompt_builder import Autonomy

        interceptor = self._make_interceptor(Autonomy.CHECK_IN)
        for action in ("click", "type", "fill", "press", "hover", "select", "drag", "upload_file"):
            tool_call = {"id": "1", "name": "browser", "arguments": {"action": action}}
            assert interceptor.should_intercept(tool_call), f"{action} should be intercepted"

    def test_evaluate_not_in_browser_write_actions(self):
        """evaluate must not appear in BROWSER_WRITE_ACTIONS frozenset."""
        from agent_os.daemon_v2.autonomy import BROWSER_WRITE_ACTIONS

        assert "evaluate" not in BROWSER_WRITE_ACTIONS


# ---------------------------------------------------------------------------
# ActivityTranslator tests
# ---------------------------------------------------------------------------


class TestActivityTranslator:
    """Tests for daemon_v2/activity_translator.py."""

    def _make_translator(self, ws=None):
        from agent_os.daemon_v2.activity_translator import ActivityTranslator

        ws = ws or MagicMock()
        return ActivityTranslator(ws), ws

    def test_assistant_tool_call_generates_activity(self):
        translator, ws = self._make_translator()
        msg = {
            "role": "assistant",
            "content": None,
            "source": "management",
            "tool_calls": [
                {"id": "tc1", "type": "function",
                 "function": {"name": "read", "arguments": '{"path": "foo.py"}'}}
            ],
        }
        translator.on_message(msg, "proj_1")
        ws.broadcast.assert_called()
        payload = ws.broadcast.call_args[0][1]
        assert payload["type"] == "agent.activity"
        assert payload["project_id"] == "proj_1"
        assert payload["category"] == "file_read"
        assert payload["source"] == "management"
        # All snake_case
        assert "project_id" in payload
        assert "tool_name" in payload

    def test_tool_result_generates_activity(self):
        translator, ws = self._make_translator()
        msg = {
            "role": "tool",
            "content": "file contents...",
            "tool_call_id": "tc1",
            "source": "management",
        }
        translator.on_message(msg, "proj_1")
        ws.broadcast.assert_called()
        payload = ws.broadcast.call_args[0][1]
        assert payload["type"] == "agent.activity"
        assert payload["category"] == "tool_result"

    def test_agent_role_generates_activity(self):
        translator, ws = self._make_translator()
        msg = {
            "role": "agent",
            "content": "Sub-agent says hello",
            "source": "claudecode",
        }
        translator.on_message(msg, "proj_1")
        ws.broadcast.assert_called()
        payload = ws.broadcast.call_args[0][1]
        assert payload["type"] == "agent.activity"
        assert payload["source"] == "claudecode"

    def test_stream_chunk_generates_delta(self):
        translator, ws = self._make_translator()
        translator.on_stream_chunk(
            type("Chunk", (), {"text": "hello", "is_final": False})(),
            project_id="proj_1",
            source="management",
        )
        ws.broadcast.assert_called()
        payload = ws.broadcast.call_args[0][1]
        assert payload["type"] == "chat.stream_delta"
        assert payload["text"] == "hello"
        assert payload["source"] == "management"
        assert payload["is_final"] is False
        assert payload["project_id"] == "proj_1"

    def test_tool_name_category_mapping(self):
        """Verify tool name to category mapping."""
        translator, ws = self._make_translator()
        mappings = {
            "read": "file_read",
            "write": "file_write",
            "edit": "file_edit",
            "shell": "command_exec",
        }
        for tool_name, expected_category in mappings.items():
            ws.reset_mock()
            msg = {
                "role": "assistant",
                "source": "management",
                "tool_calls": [
                    {"id": f"tc_{tool_name}", "type": "function",
                     "function": {"name": tool_name, "arguments": "{}"}}
                ],
            }
            translator.on_message(msg, "proj_1")
            payload = ws.broadcast.call_args[0][1]
            assert payload["category"] == expected_category, f"Expected {expected_category} for {tool_name}"

    def test_activity_events_have_required_fields(self):
        """All activity events must include: project_id, id, source, timestamp."""
        translator, ws = self._make_translator()
        msg = {
            "role": "assistant",
            "source": "management",
            "tool_calls": [
                {"id": "tc1", "type": "function",
                 "function": {"name": "read", "arguments": '{"path": "x"}'}}
            ],
        }
        translator.on_message(msg, "proj_1")
        payload = ws.broadcast.call_args[0][1]
        assert "id" in payload
        assert "project_id" in payload
        assert "source" in payload
        assert "timestamp" in payload

    def test_all_ws_payloads_use_snake_case(self):
        """Ensure no camelCase keys in WS payloads."""
        translator, ws = self._make_translator()
        msg = {
            "role": "assistant",
            "source": "management",
            "tool_calls": [
                {"id": "tc1", "type": "function",
                 "function": {"name": "shell", "arguments": '{"command": "ls"}'}}
            ],
        }
        translator.on_message(msg, "proj_1")
        payload = ws.broadcast.call_args[0][1]
        for key in payload.keys():
            assert "_" in key or key.islower(), f"Key '{key}' is not snake_case"
            assert key == key.lower(), f"Key '{key}' contains uppercase"


# ---------------------------------------------------------------------------
# ProcessManager tests
# ---------------------------------------------------------------------------


class TestProcessManager:
    """Tests for daemon_v2/process_manager.py."""

    @pytest.mark.asyncio
    async def test_start_creates_consumer_task(self):
        from agent_os.daemon_v2.process_manager import ProcessManager

        ws = MagicMock()
        ws.broadcast = MagicMock()
        translator = MagicMock()
        pm = ProcessManager(ws_manager=ws, activity_translator=translator)

        adapter = AsyncMock()

        # Make read_stream return an async iterator that yields one chunk then stops
        async def mock_stream():
            from agent_os.agent.adapters.base import OutputChunk
            yield OutputChunk(text="hello", chunk_type="response", timestamp="2026-01-01T00:00:00Z")
        adapter.read_stream = mock_stream

        await pm.start("proj_1", "claudecode", adapter)
        key = "proj_1:claudecode"
        assert key in pm._tasks

        # Let the consumer task run
        await asyncio.sleep(0.1)

        # v5: ProcessManager writes to transcript, not session.
        # Verify activity_translator received the message instead.
        translator.on_message.assert_called()
        call_args = translator.on_message.call_args[0][0]
        assert call_args["role"] == "agent"
        assert call_args["source"] == "claudecode"

    @pytest.mark.asyncio
    async def test_stop_cancels_consumer_task(self):
        from agent_os.daemon_v2.process_manager import ProcessManager

        ws = MagicMock()
        translator = MagicMock()
        pm = ProcessManager(ws_manager=ws, activity_translator=translator)

        adapter = AsyncMock()

        async def endless_stream():
            while True:
                await asyncio.sleep(10)
                yield  # never reached
        adapter.read_stream = endless_stream

        await pm.start("proj_1", "claudecode", adapter)
        await pm.stop("proj_1", "claudecode")
        key = "proj_1:claudecode"
        assert key not in pm._tasks or pm._tasks[key].cancelled()


# ---------------------------------------------------------------------------
# SubAgentManager tests
# ---------------------------------------------------------------------------


class TestSubAgentManager:
    """Tests for daemon_v2/sub_agent_manager.py."""

    def _make_manager(self):
        from agent_os.daemon_v2.sub_agent_manager import SubAgentManager
        from agent_os.agent.adapters.base import AdapterConfig

        pm = MagicMock()
        pm.start = AsyncMock()
        pm.stop = AsyncMock()
        configs = {
            "claudecode": AdapterConfig(
                command="claude", workspace="/tmp/ws", approval_patterns=["Approve?"]
            )
        }
        return SubAgentManager(process_manager=pm, adapter_configs=configs), pm

    @pytest.mark.asyncio
    async def test_start_creates_adapter(self):
        mgr, pm = self._make_manager()

        with patch("agent_os.daemon_v2.sub_agent_manager.CLIAdapter") as MockAdapter:
            mock_instance = AsyncMock()
            MockAdapter.return_value = mock_instance

            result = await mgr.start("proj_1", "claudecode")
            assert ("proj_1", "default") in mgr._adapters
            assert "claudecode" in mgr._adapters[("proj_1", "default")]
            mock_instance.start.assert_awaited_once()
            pm.start.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_send_forwards_message(self):
        mgr, pm = self._make_manager()

        with patch("agent_os.daemon_v2.sub_agent_manager.CLIAdapter") as MockAdapter:
            mock_instance = AsyncMock()
            mock_instance._transport = None  # Legacy path: background task calls adapter.send()
            MockAdapter.return_value = mock_instance

            await mgr.start("proj_1", "claudecode")
            result = await mgr.send("proj_1", "claudecode", "hello")
            assert "Message sent to claudecode" in result
            # Background task dispatches asynchronously
            await asyncio.sleep(0.05)
            mock_instance.send.assert_awaited_once_with("hello")

    @pytest.mark.asyncio
    async def test_stop_stops_adapter(self):
        mgr, pm = self._make_manager()

        with patch("agent_os.daemon_v2.sub_agent_manager.CLIAdapter") as MockAdapter:
            mock_instance = AsyncMock()
            MockAdapter.return_value = mock_instance

            await mgr.start("proj_1", "claudecode")
            await mgr.stop("proj_1", "claudecode")
            mock_instance.stop.assert_awaited_once()
            pm.stop.assert_awaited_once()

    def test_status_unknown_when_not_started(self):
        mgr, _ = self._make_manager()
        assert mgr.status("proj_1", "claudecode") == "unknown"

    @pytest.mark.asyncio
    async def test_list_active(self):
        mgr, _ = self._make_manager()

        with patch("agent_os.daemon_v2.sub_agent_manager.CLIAdapter") as MockAdapter:
            mock_instance = AsyncMock()
            mock_instance.is_alive = MagicMock(return_value=True)
            mock_instance.display_name = "Claude Code"
            mock_instance.handle = "claudecode"
            MockAdapter.return_value = mock_instance

            await mgr.start("proj_1", "claudecode")
            active = mgr.list_active("proj_1")
            assert len(active) == 1
            assert active[0]["handle"] == "claudecode"

    @pytest.mark.asyncio
    async def test_concurrent_dispatch_from_two_sessions_is_isolated(self):
        """Sub-agent dispatch from two sessions in one project stays isolated.

        Phase 3c-overnight: each chat session owns its own adapter slate;
        starting the same handle in two sessions yields two distinct
        adapters, list_active scoped to each session returns only that
        session's adapters, and stop_all on one session does not touch
        the other.
        """
        mgr, pm = self._make_manager()

        # Patch CLIAdapter so every constructor call yields a fresh mock.
        adapters_created: list = []

        def make_adapter(*args, **kwargs):
            inst = AsyncMock()
            inst.is_alive = MagicMock(return_value=True)
            inst.is_idle = MagicMock(return_value=False)
            inst.display_name = kwargs.get("display_name", "ClaudeCode")
            inst.handle = kwargs.get("handle", "claudecode")
            adapters_created.append(inst)
            return inst

        with patch(
            "agent_os.daemon_v2.sub_agent_manager.CLIAdapter",
            side_effect=make_adapter,
        ):
            # Concurrent starts in two sessions of the same project.
            results = await asyncio.gather(
                mgr.start("proj_iso", "claudecode", session_id="sess_a"),
                mgr.start("proj_iso", "claudecode", session_id="sess_b"),
            )

        assert all("Started" in r for r in results), f"unexpected start results: {results}"

        # Each session ended up with its own adapter slate.
        assert ("proj_iso", "sess_a") in mgr._adapters
        assert ("proj_iso", "sess_b") in mgr._adapters
        assert "claudecode" in mgr._adapters[("proj_iso", "sess_a")]
        assert "claudecode" in mgr._adapters[("proj_iso", "sess_b")]
        # Two distinct adapter objects were created — not one shared.
        adapter_a = mgr._adapters[("proj_iso", "sess_a")]["claudecode"]
        adapter_b = mgr._adapters[("proj_iso", "sess_b")]["claudecode"]
        assert adapter_a is not adapter_b
        assert adapter_a in adapters_created
        assert adapter_b in adapters_created

        # list_active is scoped to the session.
        active_a = mgr.list_active("proj_iso", session_id="sess_a")
        active_b = mgr.list_active("proj_iso", session_id="sess_b")
        assert len(active_a) == 1 and active_a[0]["handle"] == "claudecode"
        assert len(active_b) == 1 and active_b[0]["handle"] == "claudecode"
        # Default-session list is empty — neither was registered under default.
        assert mgr.list_active("proj_iso") == []

        # stop_all on sess_a tears down only A's adapter slate.
        await mgr.stop_all("proj_iso", session_id="sess_a")
        assert mgr._adapters.get(("proj_iso", "sess_a"), {}) == {}
        # B's slate is untouched.
        assert "claudecode" in mgr._adapters[("proj_iso", "sess_b")]
        adapter_a.stop.assert_awaited()
        adapter_b.stop.assert_not_awaited()

        # update_sub_agent_autonomy targets a single session as well.
        # Use sess_b — only B's transport (if any) should be touched. We
        # set _transport with an update_autonomy spy on both adapters and
        # confirm A is untouched.
        from agent_os.agent.prompt_builder import Autonomy
        spy_a = MagicMock()
        spy_b = MagicMock()
        # A is gone after stop_all, re-attach to verify isolation.
        mgr._adapters.setdefault(("proj_iso", "sess_a"), {})["claudecode"] = adapter_a
        adapter_a._transport = MagicMock()
        adapter_a._transport.update_autonomy = spy_a
        adapter_b._transport = MagicMock()
        adapter_b._transport.update_autonomy = spy_b

        mgr.update_sub_agent_autonomy(
            "proj_iso", Autonomy.HANDS_OFF, session_id="sess_b",
        )
        spy_a.assert_not_called()
        spy_b.assert_called_once_with(Autonomy.HANDS_OFF)


# ---------------------------------------------------------------------------
# AgentManager tests
# ---------------------------------------------------------------------------


class TestAgentManager:
    """Tests for daemon_v2/agent_manager.py."""

    def _make_manager(self):
        from agent_os.daemon_v2.agent_manager import AgentManager

        project_store = MagicMock()
        ws = MagicMock()
        ws.broadcast = MagicMock()
        sub_agent_mgr = MagicMock()
        sub_agent_mgr.list_active = MagicMock(return_value=[])
        sub_agent_mgr.stop = AsyncMock()
        sub_agent_mgr.stop_all = AsyncMock()
        activity_translator = MagicMock()
        process_manager = MagicMock()
        process_manager.set_session = MagicMock()

        mgr = AgentManager(
            project_store=project_store,
            ws_manager=ws,
            sub_agent_manager=sub_agent_mgr,
            activity_translator=activity_translator,
            process_manager=process_manager,
        )
        return mgr, ws, sub_agent_mgr, activity_translator, process_manager

    @pytest.mark.asyncio
    async def test_start_agent_wires_components(self):
        from agent_os.daemon_v2.models import AgentConfig
        from agent_os.agent.prompt_builder import Autonomy

        mgr, ws, _, _, pm = self._make_manager()
        config = AgentConfig(workspace="/tmp/ws", model="gpt-4", api_key="sk-test")

        with patch("agent_os.daemon_v2.agent_manager.LLMProvider"), \
             patch("agent_os.daemon_v2.agent_manager.ToolRegistry") as MockReg, \
             patch("agent_os.daemon_v2.agent_manager.PromptBuilder"), \
             patch("agent_os.daemon_v2.agent_manager.Session") as MockSession, \
             patch("agent_os.daemon_v2.agent_manager.ContextManager"), \
             patch("agent_os.daemon_v2.agent_manager.AgentLoop") as MockLoop, \
             patch("agent_os.daemon_v2.agent_manager.AutonomyInterceptor"):

            mock_session = MagicMock()
            MockSession.new.return_value = mock_session
            mock_reg = MagicMock()
            mock_reg.tool_names.return_value = ["read", "write"]
            MockReg.return_value = mock_reg

            mock_loop = MagicMock()
            mock_loop.run = AsyncMock()
            MockLoop.return_value = mock_loop

            await mgr.start_agent("proj_1", config)

            assert ("proj_1", "default") in mgr._handles
            # Should broadcast running status
            ws.broadcast.assert_called()

    @pytest.mark.asyncio
    async def test_start_agent_twice_raises(self):
        from agent_os.daemon_v2.models import AgentConfig

        mgr, ws, _, _, _ = self._make_manager()
        config = AgentConfig(workspace="/tmp/ws", model="gpt-4", api_key="sk-test")

        with patch("agent_os.daemon_v2.agent_manager.LLMProvider"), \
             patch("agent_os.daemon_v2.agent_manager.ToolRegistry") as MockReg, \
             patch("agent_os.daemon_v2.agent_manager.PromptBuilder"), \
             patch("agent_os.daemon_v2.agent_manager.Session") as MockSession, \
             patch("agent_os.daemon_v2.agent_manager.ContextManager"), \
             patch("agent_os.daemon_v2.agent_manager.AgentLoop") as MockLoop, \
             patch("agent_os.daemon_v2.agent_manager.AutonomyInterceptor"):

            mock_session = MagicMock()
            MockSession.new.return_value = mock_session
            mock_reg = MagicMock()
            mock_reg.tool_names.return_value = ["read"]
            MockReg.return_value = mock_reg
            mock_loop = MagicMock()
            mock_loop.run = AsyncMock()
            MockLoop.return_value = mock_loop

            await mgr.start_agent("proj_1", config)
            with pytest.raises(ValueError, match="already running"):
                await mgr.start_agent("proj_1", config)

    @pytest.mark.asyncio
    async def test_inject_while_running_queues(self):
        from agent_os.daemon_v2.agent_manager import AgentManager

        mgr, ws, _, _, _ = self._make_manager()

        # Simulate a running handle
        mock_session = MagicMock()
        mock_task = MagicMock()
        mock_task.done.return_value = False
        mgr._handles[("proj_1", "default")] = MagicMock(session=mock_session, task=mock_task)

        await mgr.inject_message("proj_1", "hello")
        mock_session.queue_message.assert_called_once_with("hello", nonce=None)

    @pytest.mark.asyncio
    async def test_inject_while_idle_resumes(self):
        mgr, ws, _, _, _ = self._make_manager()

        mock_session = MagicMock()
        mock_session.is_stopped.return_value = False
        mock_session._paused_for_approval = False

        # Task is done (loop idle), no exception
        mock_task = MagicMock()
        mock_task.done.return_value = True
        mock_task.exception.return_value = None

        handle = MagicMock(session=mock_session, task=mock_task, loop=MagicMock())
        mgr._handles[("proj_1", "default")] = handle

        with patch.object(mgr, "_start_loop", new_callable=AsyncMock) as mock_start:
            await mgr.inject_message("proj_1", "continue work")
            mock_session.append.assert_called_once()
            mock_start.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_inject_after_errored_loop_still_delivers(self):
        """Bug 4 fix: message must be appended even if previous loop errored."""
        mgr, ws, _, _, _ = self._make_manager()

        mock_session = MagicMock()
        mock_session.is_stopped.return_value = False
        mock_session._paused_for_approval = False

        # Task is done AND has an exception (previous loop errored)
        mock_task = MagicMock()
        mock_task.done.return_value = True
        mock_task.exception.return_value = RuntimeError("LLM timeout")

        handle = MagicMock(session=mock_session, task=mock_task, loop=MagicMock())
        mgr._handles[("proj_1", "default")] = handle

        with patch.object(mgr, "_start_loop", new_callable=AsyncMock) as mock_start:
            result = await mgr.inject_message("proj_1", "are you there?")
            # Message must be appended to session (not lost)
            mock_session.append.assert_called_once()
            appended = mock_session.append.call_args[0][0]
            assert appended["role"] == "user"
            assert appended["content"] == "are you there?"
            # Loop must be restarted
            mock_start.assert_awaited_once()
            assert result == "delivered"

    @pytest.mark.asyncio
    async def test_inject_after_cancelled_loop_still_delivers(self):
        """Bug 4 fix: message must be appended even if previous task was cancelled."""
        mgr, ws, _, _, _ = self._make_manager()

        mock_session = MagicMock()
        mock_session.is_stopped.return_value = False
        mock_session._paused_for_approval = False

        # Task is done AND .exception() raises CancelledError
        mock_task = MagicMock()
        mock_task.done.return_value = True
        mock_task.exception.side_effect = asyncio.CancelledError()

        handle = MagicMock(session=mock_session, task=mock_task, loop=MagicMock())
        mgr._handles[("proj_1", "default")] = handle

        with patch.object(mgr, "_start_loop", new_callable=AsyncMock) as mock_start:
            result = await mgr.inject_message("proj_1", "hello again")
            mock_session.append.assert_called_once()
            mock_start.assert_awaited_once()
            assert result == "delivered"

    @pytest.mark.asyncio
    async def test_stop_agent_stops_session(self):
        mgr, ws, sub_mgr, _, _ = self._make_manager()

        mock_session = MagicMock()
        mock_task = asyncio.ensure_future(asyncio.sleep(0))
        await asyncio.sleep(0)  # let it complete

        # T04: stop_agent now does `await handle.loop.terminate()` on the
        # alive-task path; the awaitable must come back from an AsyncMock.
        mock_loop = MagicMock(terminate=AsyncMock())
        handle = MagicMock(session=mock_session, task=mock_task,
                           loop=mock_loop)
        mgr._handles[("proj_1", "default")] = handle

        await mgr.stop_agent("proj_1")
        # Task is still pending (one `await asyncio.sleep(0)` is not enough
        # to mark a sleep(0) future done) ⇒ stop_agent takes the alive
        # path and awaits handle.loop.terminate(). The terminate() call is
        # what propagates the stop intent to the session in T04.
        mock_loop.terminate.assert_awaited_once()
        ws.broadcast.assert_called()

    @pytest.mark.asyncio
    async def test_approve_records_and_resumes(self):
        mgr, ws, _, _, _ = self._make_manager()

        mock_session = MagicMock()
        mock_session.has_result_for.return_value = False
        mock_interceptor = MagicMock()
        mock_interceptor.get_pending.return_value = {
            "tool_name": "shell", "tool_args": {"command": "ls"}
        }

        mock_registry = MagicMock()
        mock_registry.execute.return_value = MagicMock(content="output", meta=None)

        mock_task = asyncio.ensure_future(asyncio.sleep(0))
        await asyncio.sleep(0)

        handle = MagicMock(
            session=mock_session, task=mock_task, interceptor=mock_interceptor,
            registry=mock_registry, loop=MagicMock()
        )
        mgr._handles[("proj_1", "default")] = handle

        with patch.object(mgr, "_start_loop", new_callable=AsyncMock):
            await mgr.approve("proj_1", "tc_1")
            mock_interceptor.record_approval.assert_called_once_with("shell", {"command": "ls"})
            mock_session.resume.assert_called_once()

    @pytest.mark.asyncio
    async def test_deny_appends_result_and_resumes(self):
        mgr, ws, _, _, _ = self._make_manager()

        mock_session = MagicMock()
        mock_session.has_result_for.return_value = False
        mock_interceptor = MagicMock()
        mock_interceptor.get_pending.return_value = {
            "tool_name": "shell", "tool_args": {"command": "rm -rf /"}
        }

        mock_task = asyncio.ensure_future(asyncio.sleep(0))
        await asyncio.sleep(0)

        handle = MagicMock(
            session=mock_session, task=mock_task, interceptor=mock_interceptor,
            loop=MagicMock()
        )
        mgr._handles[("proj_1", "default")] = handle

        with patch.object(mgr, "_start_loop", new_callable=AsyncMock):
            await mgr.deny("proj_1", "tc_1", "Not safe")
            mock_session.append_tool_result.assert_called_once()
            call_args = mock_session.append_tool_result.call_args
            assert "tc_1" in call_args[0]
            assert "DENIED" in call_args[0][1]
            mock_session.resume.assert_called_once()

    @pytest.mark.asyncio
    async def test_inject_message_accepts_session_id_kwarg(self):
        """session_id kwarg routes to the matching (project, session) handle."""
        mgr, ws, _, _, _ = self._make_manager()

        # Register a running handle under an explicit session id.
        mock_session = MagicMock()
        mock_task = MagicMock()
        mock_task.done.return_value = False
        mgr._handles[("proj_1", "sess_a")] = MagicMock(
            session=mock_session, task=mock_task,
        )

        # Calling with the matching session_id reaches the queued-path
        # for that exact handle (storage is now SessionKey-keyed).
        result = await mgr.inject_message("proj_1", "hi", session_id="sess_a")
        assert result == "queued"
        mock_session.queue_message.assert_called_once_with("hi", nonce=None)

    def test_synchronous_entry_points_accept_session_id_kwarg(self):
        """Forward-compat: sync entry points all accept session_id kwarg.

        Each entry point must accept the kwarg and behave identically to
        the no-kwarg form at this commit (storage is still project_id-keyed).
        """
        from agent_os.agent.prompt_builder import Autonomy

        mgr, _, _, _, _ = self._make_manager()

        # No handle yet — read paths return the idle / no-handle answer.
        assert mgr.is_running("proj_1") is False
        assert mgr.is_running("proj_1", session_id="sess_a") is False
        assert mgr.get_run_status("proj_1") == "idle"
        assert mgr.get_run_status("proj_1", session_id="sess_a") == "idle"
        assert mgr.get_pending_approval("proj_1") is None
        assert mgr.get_pending_approval("proj_1", session_id="sess_a") is None
        assert mgr.get_session("proj_1") is None
        assert mgr.get_session("proj_1", session_id="sess_a") is None
        # update_autonomy returns False when no agent is running.
        assert mgr.update_autonomy("proj_1", Autonomy.HANDS_OFF) is False
        assert mgr.update_autonomy(
            "proj_1", Autonomy.HANDS_OFF, session_id="sess_a"
        ) is False

    @pytest.mark.asyncio
    async def test_inject_message_mints_new_session_lazily(self):
        """An unknown session_id in inject_message mints a fresh handle.

        Phase 3c lazy-session-minting contract: when inject_message is
        called with a session_id that has no handle in _handles but the
        project itself exists, AgentManager falls through to start_agent
        which registers a new handle under (project_id, session_id).
        Existing handles in the same project are untouched.
        """
        mgr, ws, _, _, _ = self._make_manager()

        # An existing "default" session for proj_lazy.
        existing_session = MagicMock(name="default_session")
        existing_task = MagicMock()
        existing_task.done.return_value = False
        # config_snapshot must be a real dict — _write_state json-serializes it.
        existing_handle = MagicMock(
            session=existing_session, task=existing_task,
            config_snapshot={"workspace": "/tmp/lazy_ws", "model": "gpt-4"},
            started_at="2026-01-01T00:00:00+00:00",
        )
        mgr._handles[("proj_lazy", "default")] = existing_handle

        # Make project_store.get_project return a minimal project dict so
        # start_agent's auto-derive path has data to work with.
        mgr._project_store.get_project.return_value = {
            "workspace": "/tmp/lazy_ws",
            "name": "Lazy",
            "autonomy": "hands_off",
            "model": "gpt-4",
            "api_key": "sk-test",
            "sdk": "openai",
            "provider": "custom",
            "enabled_sub_agents": [],
        }

        # Patch heavy dependencies of start_agent so the test runs in-process.
        with patch("agent_os.daemon_v2.agent_manager.LLMProvider"), \
             patch("agent_os.daemon_v2.agent_manager.ToolRegistry") as MockReg, \
             patch("agent_os.daemon_v2.agent_manager.PromptBuilder"), \
             patch("agent_os.daemon_v2.agent_manager.Session") as MockSession, \
             patch("agent_os.daemon_v2.agent_manager.ContextManager"), \
             patch("agent_os.daemon_v2.agent_manager.AgentLoop") as MockLoop, \
             patch("agent_os.daemon_v2.agent_manager.AutonomyInterceptor"):

            MockSession.new.return_value = MagicMock()
            mock_reg = MagicMock()
            mock_reg.tool_names.return_value = ["read"]
            MockReg.return_value = mock_reg
            mock_loop = MagicMock()
            mock_loop.run = AsyncMock()
            MockLoop.return_value = mock_loop

            result = await mgr.inject_message(
                "proj_lazy", "hello from new session",
                session_id="sess_fresh",
            )

        # The auto-start branch returns "started".
        assert result == "started"
        # The new session handle was minted.
        assert ("proj_lazy", "sess_fresh") in mgr._handles
        # The existing default handle is untouched.
        assert mgr._handles[("proj_lazy", "default")] is existing_handle
        # The existing session must NOT have received the new message.
        existing_session.queue_message.assert_not_called()
        existing_session.append.assert_not_called()

    @pytest.mark.asyncio
    async def test_cancel_message_broadcast_carries_session_id(self):
        """cancel_message broadcasts the agent.status payload with session_id.

        Multi-loop contract: every status broadcast originating in
        AgentManager must carry a session_id discriminator so clients
        can route by session. cancel_message is one of the simplest
        broadcast paths to validate without standing up a full loop.
        """
        mgr, ws, _, _, _ = self._make_manager()

        # Two handles, two sessions, both running.
        for sid in ("sess_a", "sess_b"):
            session = MagicMock()
            session.is_stopped.return_value = False
            session._paused_for_approval = False
            task = MagicMock()
            task.done.return_value = False
            loop = MagicMock()
            loop.cancel_turn = AsyncMock()
            mgr._handles[("proj_ws", sid)] = MagicMock(
                session=session, task=task, loop=loop,
            )

        # Cancel session A only — should produce one broadcast tagged sess_a.
        result = await mgr.cancel_message("proj_ws", session_id="sess_a")
        assert result == {"status": "cancelled"}

        ws.broadcast.assert_called_once()
        broadcast_pid, payload = ws.broadcast.call_args[0]
        assert broadcast_pid == "proj_ws"
        assert payload["type"] == "agent.status"
        assert payload["status"] == "idle"
        # Critical: session_id field is present and matches the cancelled session.
        assert payload["session_id"] == "sess_a"

        # Cancel session B — second broadcast tagged sess_b, A's handle is
        # untouched.
        ws.broadcast.reset_mock()
        result = await mgr.cancel_message("proj_ws", session_id="sess_b")
        assert result == {"status": "cancelled"}
        ws.broadcast.assert_called_once()
        _, payload_b = ws.broadcast.call_args[0]
        assert payload_b["session_id"] == "sess_b"

    @pytest.mark.asyncio
    async def test_two_sessions_one_project_are_isolated(self):
        """Two sessions in one project route to distinct handles.

        Phase 3c-overnight isolation invariant: a message injected into
        session A must reach only session A's session object, never
        session B's. Likewise for read APIs (is_running, get_run_status,
        get_session, get_pending_approval).
        """
        mgr, _, _, _, _ = self._make_manager()

        # Set up two distinct handles for the same project under
        # different session ids. Session A is running, session B is idle.
        session_a = MagicMock(name="session_a")
        task_a = MagicMock()
        task_a.done.return_value = False  # A is running
        handle_a = MagicMock(
            session=session_a, task=task_a,
            interceptor=MagicMock(_pending_approvals={}),
        )
        session_a.is_stopped.return_value = False
        session_a._paused_for_approval = False

        session_b = MagicMock(name="session_b")
        task_b = MagicMock()
        task_b.done.return_value = True  # B is idle
        handle_b = MagicMock(
            session=session_b, task=task_b,
            interceptor=MagicMock(_pending_approvals={}),
        )
        session_b.is_stopped.return_value = False
        session_b._paused_for_approval = False

        mgr._handles[("proj_iso", "sess_a")] = handle_a
        mgr._handles[("proj_iso", "sess_b")] = handle_b

        # --- inject_message routes to correct session ---
        # Injecting into A (running) queues into A only.
        result_a = await mgr.inject_message(
            "proj_iso", "msg-for-a", session_id="sess_a",
        )
        assert result_a == "queued"
        session_a.queue_message.assert_called_once_with("msg-for-a", nonce=None)
        session_b.queue_message.assert_not_called()

        # Injecting into B (idle) hot-resumes B only — append on B's session.
        with patch.object(mgr, "_start_loop", new_callable=AsyncMock) as mock_start:
            result_b = await mgr.inject_message(
                "proj_iso", "msg-for-b", session_id="sess_b",
            )
            assert result_b == "delivered"
            session_b.append.assert_called_once()
            appended = session_b.append.call_args[0][0]
            assert appended["content"] == "msg-for-b"
            # A's session must NOT have received the new append.
            session_a.append.assert_not_called()
            # _start_loop must be called with session_id="sess_b".
            mock_start.assert_awaited_once_with("proj_iso", session_id="sess_b")

        # --- read APIs respect session_id ---
        assert mgr.is_running("proj_iso", session_id="sess_a") is True
        assert mgr.is_running("proj_iso", session_id="sess_b") is False
        assert mgr.get_run_status("proj_iso", session_id="sess_a") == "running"
        assert mgr.get_run_status("proj_iso", session_id="sess_b") == "idle"
        assert mgr.get_session("proj_iso", session_id="sess_a") is session_a
        assert mgr.get_session("proj_iso", session_id="sess_b") is session_b
        # Default-session lookup against this project returns None — neither
        # handle was registered under the default sentinel.
        assert mgr.get_session("proj_iso") is None
        assert mgr.is_running("proj_iso") is False


# ---------------------------------------------------------------------------
# ProjectStore tests
# ---------------------------------------------------------------------------


class TestProjectStore:
    """Tests for daemon_v2/project_store.py using JSON files (no SQLite)."""

    def test_create_and_list(self, tmp_path):
        from agent_os.daemon_v2.project_store import ProjectStore

        store = ProjectStore(data_dir=str(tmp_path))
        pid = store.create_project({
            "name": "Test Project",
            "workspace": str(tmp_path),
            "model": "gpt-4",
            "api_key": "sk-test",
        })
        assert pid is not None
        projects = store.list_projects()
        assert len(projects) == 1
        assert projects[0]["name"] == "Test Project"

    def test_get_project(self, tmp_path):
        from agent_os.daemon_v2.project_store import ProjectStore

        store = ProjectStore(data_dir=str(tmp_path))
        pid = store.create_project({
            "name": "My Project",
            "workspace": str(tmp_path),
            "model": "gpt-4",
            "api_key": "sk-test",
        })
        project = store.get_project(pid)
        assert project is not None
        assert project["name"] == "My Project"

    def test_get_nonexistent_project(self, tmp_path):
        from agent_os.daemon_v2.project_store import ProjectStore

        store = ProjectStore(data_dir=str(tmp_path))
        assert store.get_project("nonexistent") is None

    def test_update_project(self, tmp_path):
        from agent_os.daemon_v2.project_store import ProjectStore

        store = ProjectStore(data_dir=str(tmp_path))
        pid = store.create_project({
            "name": "Old Name",
            "workspace": str(tmp_path),
            "model": "gpt-4",
            "api_key": "sk-test",
        })
        store.update_project(pid, {"name": "New Name"})
        project = store.get_project(pid)
        assert project["name"] == "New Name"

    def test_delete_project(self, tmp_path):
        from agent_os.daemon_v2.project_store import ProjectStore

        store = ProjectStore(data_dir=str(tmp_path))
        pid = store.create_project({
            "name": "To Delete",
            "workspace": str(tmp_path),
            "model": "gpt-4",
            "api_key": "sk-test",
        })
        store.delete_project(pid)
        assert store.get_project(pid) is None
        assert len(store.list_projects()) == 0

    def test_no_sqlite_references(self, tmp_path):
        """ProjectStore must use JSON files, not SQLite."""
        import inspect
        from agent_os.daemon_v2 import project_store as ps_mod

        source = inspect.getsource(ps_mod)
        assert "sqlite" not in source.lower()
        assert "database" not in source.lower()


# ---------------------------------------------------------------------------
# REST API tests (FastAPI TestClient)
# ---------------------------------------------------------------------------


class TestRESTEndpoints:
    """Tests for api/routes/agents_v2.py using FastAPI TestClient."""

    @pytest.fixture
    def app_client(self, tmp_path):
        from agent_os.api.app import create_app

        app = create_app(data_dir=str(tmp_path))
        from starlette.testclient import TestClient
        return TestClient(app)

    def test_create_project(self, app_client, tmp_path):
        resp = app_client.post("/api/v2/projects", json={
            "name": "Test",
            "workspace": str(tmp_path),
            "model": "gpt-4",
            "api_key": "sk-test",
        })
        assert resp.status_code == 201
        data = resp.json()
        assert "project_id" in data

    def test_create_project_bad_workspace(self, app_client):
        resp = app_client.post("/api/v2/projects", json={
            "name": "Test",
            "workspace": "/nonexistent/path/xyz",
            "model": "gpt-4",
            "api_key": "sk-test",
        })
        assert resp.status_code == 400
        assert "detail" in resp.json()

    def test_list_projects(self, app_client, tmp_path):
        app_client.post("/api/v2/projects", json={
            "name": "Project 1",
            "workspace": str(tmp_path),
            "model": "gpt-4",
            "api_key": "sk-test",
        })
        resp = app_client.get("/api/v2/projects")
        assert resp.status_code == 200
        assert len(resp.json()) >= 1

    def test_start_agent_not_found(self, app_client):
        resp = app_client.post("/api/v2/agents/start", json={
            "project_id": "nonexistent",
        })
        assert resp.status_code == 404

    def test_inject_no_session(self, app_client):
        resp = app_client.post("/api/v2/agents/nonexistent/inject", json={
            "content": "hello",
        })
        assert resp.status_code == 404

    def test_stop_route_removed(self, app_client):
        # The user-facing /stop HTTP route was removed; "stop working" is now
        # POST /api/v2/agents/{project_id}/cancel. The internal
        # AgentManager.stop_agent(...) still exists for idle-eviction, daemon
        # shutdown, and project deletion, but is no longer exposed over HTTP.
        # So this now 404s because the route no longer exists (or 405 if the
        # path collides with another method).
        resp = app_client.post("/api/v2/agents/nonexistent/stop")
        assert resp.status_code in (404, 405)

    def test_approve_no_session(self, app_client):
        resp = app_client.post("/api/v2/agents/nonexistent/approve", json={
            "tool_call_id": "tc_1",
        })
        assert resp.status_code == 404

    def test_deny_no_session(self, app_client):
        resp = app_client.post("/api/v2/agents/nonexistent/deny", json={
            "tool_call_id": "tc_1",
            "reason": "no",
        })
        assert resp.status_code == 404

    def test_chat_history_no_project(self, app_client):
        resp = app_client.get("/api/v2/agents/nonexistent/chat")
        assert resp.status_code == 404

    def test_list_sessions_404_for_unknown_project(self, app_client):
        """GET /projects/{pid}/sessions returns 404 for unknown project."""
        resp = app_client.get("/api/v2/projects/nonexistent/sessions")
        assert resp.status_code == 404

    def test_list_sessions_empty_for_idle_project(self, app_client, tmp_path):
        """A project with no active sessions returns an empty list."""
        create_resp = app_client.post("/api/v2/projects", json={
            "name": "EmptySessions",
            "workspace": str(tmp_path),
            "model": "gpt-4",
            "api_key": "sk-test",
        })
        pid = create_resp.json()["project_id"]

        resp = app_client.get(f"/api/v2/projects/{pid}/sessions")
        assert resp.status_code == 200
        body = resp.json()
        assert body == {"project_id": pid, "sessions": []}

    def test_list_sessions_returns_active_and_idle(self, app_client, tmp_path):
        """Sessions endpoint returns running and idle entries with correct shape.

        We inject handles directly into the agent_manager to simulate two
        concurrent sessions in the same project — one running, one idle.
        """
        from agent_os.api.routes import agents_v2

        # Create the project so 404 doesn't fire.
        create_resp = app_client.post("/api/v2/projects", json={
            "name": "MultiSessionList",
            "workspace": str(tmp_path),
            "model": "gpt-4",
            "api_key": "sk-test",
        })
        pid = create_resp.json()["project_id"]

        # Inject two handles directly into the agent manager.
        mgr = agents_v2._agent_manager

        # A real Session carries TWO distinct identity values:
        #   .session_id   = F1, user-facing chat id (e.g. "default")
        #   .session_uuid = F2, JSONL filename stem (e.g. "proj_run_abc123")
        # The endpoint's "session_uuid" field MUST surface F2 (the stem), since
        # consumers build ``{session_uuid}.jsonl`` from it.
        running_session = MagicMock(name="running_session")
        running_session.is_stopped.return_value = False
        running_session._paused_for_approval = False
        running_session.session_id = "default"          # F1
        running_session.session_uuid = "proj_run_abc123"  # F2 (JSONL stem)
        running_task = MagicMock()
        running_task.done.return_value = False
        mgr._handles[(pid, "default")] = MagicMock(
            session=running_session, task=running_task,
        )

        idle_session = MagicMock(name="idle_session")
        idle_session.is_stopped.return_value = False
        idle_session._paused_for_approval = False
        idle_session.session_id = "sess_b"               # F1
        idle_session.session_uuid = "proj_idle_def456"   # F2 (JSONL stem)
        idle_task = MagicMock()
        idle_task.done.return_value = True
        mgr._handles[(pid, "sess_b")] = MagicMock(
            session=idle_session, task=idle_task,
        )

        try:
            resp = app_client.get(f"/api/v2/projects/{pid}/sessions")
            assert resp.status_code == 200
            body = resp.json()
            assert body["project_id"] == pid
            sessions = body["sessions"]
            assert len(sessions) == 2
            # Default session is sorted first.
            assert sessions[0]["session_id"] == "default"
            assert sessions[0]["status"] == "running"
            # session_uuid is F2 (the JSONL stem), not F1.
            assert sessions[0]["session_uuid"] == "proj_run_abc123"
            assert sessions[1]["session_id"] == "sess_b"
            assert sessions[1]["status"] == "idle"
            assert sessions[1]["session_uuid"] == "proj_idle_def456"
        finally:
            mgr._handles.pop((pid, "default"), None)
            mgr._handles.pop((pid, "sess_b"), None)

    def test_chat_history_no_sessions_returns_empty(self, app_client, tmp_path):
        """Chat endpoint returns empty array when project exists but no sessions."""
        resp = app_client.post("/api/v2/projects", json={
            "name": "ChatTest",
            "workspace": str(tmp_path),
            "model": "gpt-4",
            "api_key": "sk-test",
        })
        pid = resp.json()["project_id"]
        resp = app_client.get(f"/api/v2/agents/{pid}/chat")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_chat_history_reads_all_sessions_from_disk(self, app_client, tmp_path):
        """Chat endpoint returns messages from ALL session JSONL files on disk."""
        # Create project
        resp = app_client.post("/api/v2/projects", json={
            "name": "MultiSession",
            "workspace": str(tmp_path),
            "model": "gpt-4",
            "api_key": "sk-test",
        })
        pid = resp.json()["project_id"]

        # Create session files on disk (flat new layout)
        sessions_dir = tmp_path / "orbital" / "sessions"
        sessions_dir.mkdir(parents=True, exist_ok=True)

        # Session 1 (older)
        s1 = sessions_dir / "session_aaa.jsonl"
        s1.write_text(
            json.dumps({"role": "user", "content": "hello from session 1", "session_id": "aaa", "timestamp": "2026-01-01T00:00:00"}) + "\n"
            + json.dumps({"role": "assistant", "content": "hi back", "session_id": "aaa", "timestamp": "2026-01-01T00:01:00"}) + "\n",
            encoding="utf-8",
        )

        # Session 2 (newer) — set mtime later
        import time
        time.sleep(0.05)
        s2 = sessions_dir / "session_bbb.jsonl"
        s2.write_text(
            json.dumps({"role": "user", "content": "hello from session 2", "session_id": "bbb", "timestamp": "2026-01-02T00:00:00"}) + "\n",
            encoding="utf-8",
        )

        resp = app_client.get(f"/api/v2/agents/{pid}/chat")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 3  # 2 from session1 + 1 from session2
        assert data[0]["content"] == "hello from session 1"
        assert data[1]["content"] == "hi back"
        assert data[2]["content"] == "hello from session 2"
        # Session IDs preserved
        assert data[0]["session_id"] == "aaa"
        assert data[2]["session_id"] == "bbb"

    # -- Chat pagination tests (daemon-hardening) --

    def test_chat_history_pagination_tail(self, app_client, tmp_path):
        """limit=3, offset=0 returns last 3 messages."""
        resp = app_client.post("/api/v2/projects", json={
            "name": "PagTail",
            "workspace": str(tmp_path),
            "model": "gpt-4",
            "api_key": "sk-test",
        })
        pid = resp.json()["project_id"]
        sessions_dir = tmp_path / "orbital" / "sessions"
        sessions_dir.mkdir(parents=True, exist_ok=True)
        lines = []
        for i in range(10):
            lines.append(json.dumps({"role": "user", "content": f"msg_{i}",
                                      "session_id": "s1", "timestamp": f"2026-01-01T00:{i:02d}:00"}))
        (sessions_dir / "session_s1.jsonl").write_text("\n".join(lines) + "\n")

        resp = app_client.get(f"/api/v2/agents/{pid}/chat?limit=3&offset=0")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 3
        assert data[0]["content"] == "msg_7"
        assert data[2]["content"] == "msg_9"
        assert resp.headers["X-Total-Count"] == "10"

    def test_chat_history_pagination_offset(self, app_client, tmp_path):
        """limit=3, offset=3 returns messages 4-6 from the end."""
        resp = app_client.post("/api/v2/projects", json={
            "name": "PagOffset",
            "workspace": str(tmp_path),
            "model": "gpt-4",
            "api_key": "sk-test",
        })
        pid = resp.json()["project_id"]
        sessions_dir = tmp_path / "orbital" / "sessions"
        sessions_dir.mkdir(parents=True, exist_ok=True)
        lines = []
        for i in range(10):
            lines.append(json.dumps({"role": "user", "content": f"msg_{i}",
                                      "session_id": "s1", "timestamp": f"2026-01-01T00:{i:02d}:00"}))
        (sessions_dir / "session_s1.jsonl").write_text("\n".join(lines) + "\n")

        resp = app_client.get(f"/api/v2/agents/{pid}/chat?limit=3&offset=3")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 3
        assert data[0]["content"] == "msg_4"
        assert data[2]["content"] == "msg_6"
        assert resp.headers["X-Total-Count"] == "10"

    def test_chat_history_pagination_empty_page(self, app_client, tmp_path):
        """Offset beyond total returns empty list with correct total."""
        resp = app_client.post("/api/v2/projects", json={
            "name": "PagEmpty",
            "workspace": str(tmp_path),
            "model": "gpt-4",
            "api_key": "sk-test",
        })
        pid = resp.json()["project_id"]
        sessions_dir = tmp_path / "orbital" / "sessions"
        sessions_dir.mkdir(parents=True, exist_ok=True)
        lines = [json.dumps({"role": "user", "content": f"msg_{i}",
                              "session_id": "s1", "timestamp": f"2026-01-01T00:{i:02d}:00"})
                 for i in range(5)]
        (sessions_dir / "session_s1.jsonl").write_text("\n".join(lines) + "\n")

        resp = app_client.get(f"/api/v2/agents/{pid}/chat?limit=10&offset=100")
        assert resp.status_code == 200
        assert resp.json() == []
        assert resp.headers["X-Total-Count"] == "5"

    def test_chat_history_no_limit_returns_all(self, app_client, tmp_path):
        """limit=0 (default) returns all messages, backward compatible."""
        resp = app_client.post("/api/v2/projects", json={
            "name": "PagAll",
            "workspace": str(tmp_path),
            "model": "gpt-4",
            "api_key": "sk-test",
        })
        pid = resp.json()["project_id"]
        sessions_dir = tmp_path / "orbital" / "sessions"
        sessions_dir.mkdir(parents=True, exist_ok=True)
        lines = [json.dumps({"role": "user", "content": f"msg_{i}",
                              "session_id": "s1", "timestamp": f"2026-01-01T00:{i:02d}:00"})
                 for i in range(8)]
        (sessions_dir / "session_s1.jsonl").write_text("\n".join(lines) + "\n")

        resp = app_client.get(f"/api/v2/agents/{pid}/chat")
        assert resp.status_code == 200
        assert len(resp.json()) == 8
        assert resp.headers["X-Total-Count"] == "8"

    # -- Bulk delete tests (daemon-hardening) --

    def test_bulk_delete_by_ids(self, app_client, tmp_path):
        """Bulk delete specific projects by ID."""
        pid1 = app_client.post("/api/v2/projects", json={
            "name": "BulkDel1", "workspace": str(tmp_path),
            "model": "gpt-4", "api_key": "sk-test",
        }).json()["project_id"]
        pid2 = app_client.post("/api/v2/projects", json={
            "name": "BulkDel2", "workspace": str(tmp_path),
            "model": "gpt-4", "api_key": "sk-test",
        }).json()["project_id"]

        resp = app_client.request("DELETE", "/api/v2/projects/bulk", json={
            "project_ids": [pid1, pid2],
        })
        assert resp.status_code == 200
        result = resp.json()
        assert result["deleted"] == 2
        assert result["failed"] == 0

        # Verify projects are gone
        projects = app_client.get("/api/v2/projects").json()
        pids = [p["project_id"] for p in projects]
        assert pid1 not in pids
        assert pid2 not in pids

    def test_bulk_delete_by_prefix(self, app_client, tmp_path):
        """Bulk delete projects by name prefix."""
        pid1 = app_client.post("/api/v2/projects", json={
            "name": "test-Alpha", "workspace": str(tmp_path),
            "model": "gpt-4", "api_key": "sk-test",
        }).json()["project_id"]
        pid2 = app_client.post("/api/v2/projects", json={
            "name": "test-Beta", "workspace": str(tmp_path),
            "model": "gpt-4", "api_key": "sk-test",
        }).json()["project_id"]
        pid3 = app_client.post("/api/v2/projects", json={
            "name": "keep-Gamma", "workspace": str(tmp_path),
            "model": "gpt-4", "api_key": "sk-test",
        }).json()["project_id"]

        resp = app_client.request("DELETE", "/api/v2/projects/bulk", json={
            "prefix": "test-",
        })
        assert resp.status_code == 200
        assert resp.json()["deleted"] == 2

        projects = app_client.get("/api/v2/projects").json()
        pids = [p["project_id"] for p in projects]
        assert pid1 not in pids
        assert pid2 not in pids
        assert pid3 in pids

    def test_bulk_delete_protects_scratch(self, app_client, tmp_path):
        """Scratch project survives bulk delete even if it matches filter."""
        projects = app_client.get("/api/v2/projects").json()
        all_ids = [p["project_id"] for p in projects]

        resp = app_client.request("DELETE", "/api/v2/projects/bulk", json={
            "project_ids": all_ids,
        })
        assert resp.status_code == 200

        remaining = app_client.get("/api/v2/projects").json()
        scratch = [p for p in remaining if p.get("is_scratch")]
        assert len(scratch) == 1

    def test_bulk_delete_no_filter_returns_400(self, app_client):
        """Bulk delete without any filter returns 400."""
        resp = app_client.request("DELETE", "/api/v2/projects/bulk", json={})
        assert resp.status_code == 400

    def test_no_v1_routes(self, app_client):
        """Zero /api/v1/ route registrations as JSON API.

        Cannot rely on a 404 response: the SPA static handler now serves
        index.html as a fallback for any unmatched GET path so deep-links work.
        Instead, verify the response isn't a v1-style JSON API response.
        """
        resp = app_client.get("/api/v1/projects")
        ctype = resp.headers.get("content-type", "")
        assert "application/json" not in ctype or resp.status_code == 404, (
            f"/api/v1/projects served as JSON API ({resp.status_code} {ctype}) — "
            "v1 routes are supposed to be gone"
        )

    def test_no_sqlite_in_app(self):
        """No SQLite references in the app module."""
        import inspect
        from agent_os.api import app as app_mod

        source = inspect.getsource(app_mod)
        assert "sqlite" not in source.lower()
        assert "from agent_os.database" not in source
        assert "from database" not in source


# ---------------------------------------------------------------------------
# WebSocket tests
# ---------------------------------------------------------------------------


class TestWebSocket:
    """Tests for api/ws.py WebSocket manager."""

    def test_ws_manager_subscribe_and_broadcast(self):
        from agent_os.api.ws import WebSocketManager

        ws_mgr = WebSocketManager()
        # Create a mock websocket
        mock_ws = MagicMock()
        mock_ws.send_json = MagicMock()

        ws_mgr.connect(mock_ws)
        ws_mgr.subscribe(mock_ws, ["proj_1", "proj_2"])
        ws_mgr.broadcast("proj_1", {"type": "agent.status", "status": "running"})

        mock_ws.send_json.assert_called_once()
        payload = mock_ws.send_json.call_args[0][0]
        assert payload["type"] == "agent.status"

    def test_ws_manager_broadcast_unsubscribed(self):
        from agent_os.api.ws import WebSocketManager

        ws_mgr = WebSocketManager()
        mock_ws = MagicMock()
        mock_ws.send_json = MagicMock()

        ws_mgr.connect(mock_ws)
        ws_mgr.subscribe(mock_ws, ["proj_1"])
        ws_mgr.broadcast("proj_2", {"type": "agent.status", "status": "running"})

        mock_ws.send_json.assert_not_called()

    def test_ws_manager_disconnect(self):
        from agent_os.api.ws import WebSocketManager

        ws_mgr = WebSocketManager()
        mock_ws = MagicMock()
        ws_mgr.connect(mock_ws)
        ws_mgr.subscribe(mock_ws, ["proj_1"])
        ws_mgr.disconnect(mock_ws)
        # broadcast after disconnect should not fail
        ws_mgr.broadcast("proj_1", {"type": "agent.status", "status": "running"})

    def test_ws_payloads_snake_case(self):
        """All WS event payloads use snake_case field names."""
        from agent_os.api.ws import WebSocketManager

        ws_mgr = WebSocketManager()
        mock_ws = MagicMock()
        ws_mgr.connect(mock_ws)
        ws_mgr.subscribe(mock_ws, ["proj_1"])

        payload = {
            "type": "agent.activity",
            "project_id": "proj_1",
            "tool_name": "read",
            "tool_call_id": "tc_1",
            "is_final": False,
        }
        ws_mgr.broadcast("proj_1", payload)
        sent = mock_ws.send_json.call_args[0][0]
        for key in sent.keys():
            # Allow 'type' (no underscore) and other simple lowercase keys
            assert key == key.lower(), f"Key '{key}' has uppercase characters"


# ---------------------------------------------------------------------------
# WebSocket endpoint integration test
# ---------------------------------------------------------------------------


class TestWebSocketEndpoint:
    """Test /ws endpoint using FastAPI TestClient WebSocket support."""

    @pytest.fixture
    def app_client(self, tmp_path):
        from agent_os.api.app import create_app
        from starlette.testclient import TestClient

        app = create_app(data_dir=str(tmp_path))
        return TestClient(app)

    def test_ws_connect_and_subscribe(self, app_client):
        with app_client.websocket_connect("/ws") as ws:
            ws.send_json({"type": "subscribe", "project_ids": ["proj_1"]})
            data = ws.receive_json()
            assert data["type"] == "subscribed"
            assert "proj_1" in data["project_ids"]


# ---------------------------------------------------------------------------
# Skills system tests
# ---------------------------------------------------------------------------


class TestSkillsCopyOnProjectCreation:
    """Test that default skills are copied on project creation."""

    @pytest.fixture
    def app_client(self, tmp_path):
        from agent_os.api.app import create_app
        from starlette.testclient import TestClient

        app = create_app(data_dir=str(tmp_path))
        return TestClient(app)

    def test_create_project_copies_default_skills(self, app_client, tmp_path):
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        resp = app_client.post("/api/v2/projects", json={
            "name": "SkillTest",
            "workspace": str(workspace),
            "model": "gpt-4",
            "api_key": "sk-test",
        })
        assert resp.status_code == 201
        pid = resp.json()["project_id"]
        skills_dir = workspace / "orbital" / "skills"
        assert skills_dir.is_dir()
        # Should have the 4 seed skills
        subdirs = [d.name for d in skills_dir.iterdir() if d.is_dir()]
        assert "learning-capture" in subdirs
        assert "efficient-execution" in subdirs
        assert "process-capture" in subdirs
        assert "task-planning" in subdirs
        # Persistent flag marks the project as reconciled so the installer
        # short-circuits on the next agent start.
        get_resp = app_client.get(f"/api/v2/projects/{pid}")
        assert get_resp.status_code == 200
        assert get_resp.json().get("default_skills_reconciled") is True

    def test_create_project_does_not_overwrite_existing_skills(self, app_client, tmp_path):
        workspace = tmp_path / "workspace2"
        workspace.mkdir()
        # Pre-create a skills directory with custom content
        skills_dir = workspace / "orbital" / "skills"
        skills_dir.mkdir(parents=True)
        custom_skill = skills_dir / "my-custom"
        custom_skill.mkdir()
        (custom_skill / "SKILL.md").write_text("# My Custom\nCustom skill.")

        resp = app_client.post("/api/v2/projects", json={
            "name": "SkillTest2",
            "workspace": str(workspace),
            "model": "gpt-4",
            "api_key": "sk-test",
        })
        assert resp.status_code == 201
        pid = resp.json()["project_id"]
        # The custom skill is preserved; the 4 bundled defaults are added
        # alongside it because they did not previously exist.
        subdirs = {d.name for d in skills_dir.iterdir() if d.is_dir()}
        assert "my-custom" in subdirs
        assert {"efficient-execution", "learning-capture",
                "process-capture", "task-planning"}.issubset(subdirs)
        # Reconciled flag is set on success.
        get_resp = app_client.get(f"/api/v2/projects/{pid}")
        assert get_resp.json().get("default_skills_reconciled") is True

    def test_scratch_project_does_not_get_skills(self, app_client, tmp_path):
        workspace = tmp_path / "workspace3"
        workspace.mkdir()
        resp = app_client.post("/api/v2/projects", json={
            "name": "ScratchTest",
            "workspace": str(workspace),
            "model": "gpt-4",
            "api_key": "sk-test",
            "is_scratch": True,
        })
        assert resp.status_code == 201
        pid = resp.json()["project_id"]
        skills_dir = workspace / "orbital" / "skills"
        assert not skills_dir.exists()
        # Scratch projects stay unreconciled — nothing to reconcile.
        get_resp = app_client.get(f"/api/v2/projects/{pid}")
        assert get_resp.json().get("default_skills_reconciled") is False


class TestSkillsAPIEndpoints:
    """Test skill CRUD API endpoints."""

    @pytest.fixture
    def app_client(self, tmp_path):
        from agent_os.api.app import create_app
        from starlette.testclient import TestClient

        app = create_app(data_dir=str(tmp_path))
        return TestClient(app)

    def _create_project_with_skills(self, app_client, workspace):
        """Helper: create a project and return its ID."""
        resp = app_client.post("/api/v2/projects", json={
            "name": "SkillAPITest",
            "workspace": str(workspace),
            "model": "gpt-4",
            "api_key": "sk-test",
        })
        assert resp.status_code == 201
        return resp.json()["project_id"]

    def test_list_skills_returns_seeded_skills(self, app_client, tmp_path):
        workspace = tmp_path / "ws"
        workspace.mkdir()
        pid = self._create_project_with_skills(app_client, workspace)
        resp = app_client.get(f"/api/v2/projects/{pid}/skills")
        assert resp.status_code == 200
        skills = resp.json()
        names = [s["name"] for s in skills]
        assert len(names) >= 4

    def test_list_skills_project_not_found(self, app_client):
        resp = app_client.get("/api/v2/projects/nonexistent/skills")
        assert resp.status_code == 404

    def test_delete_skill(self, app_client, tmp_path):
        workspace = tmp_path / "ws2"
        workspace.mkdir()
        pid = self._create_project_with_skills(app_client, workspace)
        # Delete one of the seeded skills
        resp = app_client.delete(f"/api/v2/projects/{pid}/skills/learning-capture")
        assert resp.status_code == 200
        assert resp.json()["deleted"] == "learning-capture"
        # Verify it's gone
        resp = app_client.get(f"/api/v2/projects/{pid}/skills")
        names = [s["name"] for s in resp.json()]
        assert "Learning Capture" not in names

    def test_delete_skill_not_found(self, app_client, tmp_path):
        workspace = tmp_path / "ws3"
        workspace.mkdir()
        pid = self._create_project_with_skills(app_client, workspace)
        resp = app_client.delete(f"/api/v2/projects/{pid}/skills/nonexistent-skill")
        assert resp.status_code == 404

    def test_delete_skill_path_traversal_blocked(self, app_client, tmp_path):
        workspace = tmp_path / "ws4"
        workspace.mkdir()
        pid = self._create_project_with_skills(app_client, workspace)
        resp = app_client.delete(f"/api/v2/projects/{pid}/skills/..%2F..%2Fetc")
        # 405 is also a valid block: FastAPI's path-parameter routing rejects
        # the slash-bearing skill_name as an unmatchable DELETE route. Either
        # way the path-traversal attack is blocked — server never invokes the
        # delete handler with a traversed path.
        assert resp.status_code in (400, 404, 405)

    def test_upload_md_skill(self, app_client, tmp_path):
        workspace = tmp_path / "ws5"
        workspace.mkdir()
        pid = self._create_project_with_skills(app_client, workspace)
        skill_content = b"# My Upload\nA skill uploaded via API."
        resp = app_client.post(
            f"/api/v2/projects/{pid}/skills",
            files={"file": ("skill.md", skill_content, "text/markdown")},
        )
        assert resp.status_code == 201
        data = resp.json()
        assert data["name"] == "My Upload"
        assert "description" in data
        # Verify it's in the list
        resp = app_client.get(f"/api/v2/projects/{pid}/skills")
        names = [s["name"] for s in resp.json()]
        assert "My Upload" in names

    def test_upload_md_invalid_no_heading(self, app_client, tmp_path):
        workspace = tmp_path / "ws6"
        workspace.mkdir()
        pid = self._create_project_with_skills(app_client, workspace)
        resp = app_client.post(
            f"/api/v2/projects/{pid}/skills",
            files={"file": ("bad.md", b"No heading here\nJust text.", "text/markdown")},
        )
        assert resp.status_code == 400

    def test_upload_md_duplicate_skill(self, app_client, tmp_path):
        workspace = tmp_path / "ws7"
        workspace.mkdir()
        pid = self._create_project_with_skills(app_client, workspace)
        skill_content = b"# Unique Skill\nA unique skill."
        # Upload once
        resp = app_client.post(
            f"/api/v2/projects/{pid}/skills",
            files={"file": ("skill.md", skill_content, "text/markdown")},
        )
        assert resp.status_code == 201
        # Upload again → 409
        resp = app_client.post(
            f"/api/v2/projects/{pid}/skills",
            files={"file": ("skill.md", skill_content, "text/markdown")},
        )
        assert resp.status_code == 409
