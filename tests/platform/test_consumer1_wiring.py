# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Pre-wiring contract tests for Consumer 1: Daemon / AgentManager.

These tests define what the Wirer must implement. They verify:
1. start_agent() calls provider.get_capabilities() and rejects if setup_complete=False
2. stop_agent() calls provider.stop_process(project_id)
3. start_agent() accepts trigger_source param and stores it
4. Project model accepts budget fields, serializes/deserializes correctly
5. Integration: full start->stop cycle with NullProvider

All tests mock Components A-E. The provider is mocked for unit tests
and NullProvider is used for integration tests.
"""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent_os.platform.types import PlatformCapabilities


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_capabilities(setup_complete: bool = True) -> PlatformCapabilities:
    return PlatformCapabilities(
        platform="windows",
        isolation_method="sandbox_user",
        setup_complete=setup_complete,
        setup_issues=[] if setup_complete else ["Setup not complete"],
        supports_network_restriction=True,
        supports_folder_access=True,
        sandbox_username="AgentOS-Worker" if setup_complete else None,
    )


def _make_agent_manager(platform_provider=None):
    """Create an AgentManager with mocked dependencies and optional platform_provider."""
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
        platform_provider=platform_provider,
    )
    return mgr, ws, sub_agent_mgr, activity_translator, process_manager


def _make_config(**overrides):
    from agent_os.daemon_v2.models import AgentConfig
    defaults = dict(workspace="/tmp/ws", model="gpt-4", api_key="sk-test")
    defaults.update(overrides)
    return AgentConfig(**defaults)


# ---------------------------------------------------------------------------
# Unit tests: start_agent checks capabilities
# ---------------------------------------------------------------------------


class TestStartAgentCapabilities:
    """start_agent() must call get_capabilities() and reject if setup_complete=False."""

    @pytest.mark.asyncio
    async def test_start_agent_calls_get_capabilities(self):
        """start_agent() calls provider.get_capabilities() before proceeding."""
        provider = MagicMock()
        provider.get_capabilities.return_value = _make_capabilities(setup_complete=True)
        provider.stop_process = AsyncMock(return_value=False)

        mgr, ws, _, _, _ = _make_agent_manager(platform_provider=provider)
        config = _make_config()

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

            provider.get_capabilities.assert_called_once()

    @pytest.mark.asyncio
    async def test_start_agent_warns_when_setup_incomplete(self):
        """start_agent() logs warning and proceeds when setup_complete=False."""
        provider = MagicMock()
        provider.get_capabilities.return_value = _make_capabilities(setup_complete=False)
        provider.stop_process = AsyncMock(return_value=False)

        mgr, ws, _, _, _ = _make_agent_manager(platform_provider=provider)
        config = _make_config()

        with patch("agent_os.daemon_v2.agent_manager.LLMProvider"), \
             patch("agent_os.daemon_v2.agent_manager.ToolRegistry") as MockReg, \
             patch("agent_os.daemon_v2.agent_manager.PromptBuilder"), \
             patch("agent_os.daemon_v2.agent_manager.Session") as MockSession, \
             patch("agent_os.daemon_v2.agent_manager.ContextManager"), \
             patch("agent_os.daemon_v2.agent_manager.AgentLoop") as MockLoop, \
             patch("agent_os.daemon_v2.agent_manager.AutonomyInterceptor"), \
             patch("agent_os.daemon_v2.agent_manager.logger") as mock_logger:

            mock_session = MagicMock()
            MockSession.new.return_value = mock_session
            mock_reg = MagicMock()
            mock_reg.tool_names.return_value = ["read"]
            MockReg.return_value = mock_reg
            mock_loop = MagicMock()
            mock_loop.run = AsyncMock()
            MockLoop.return_value = mock_loop

            await mgr.start_agent("proj_1", config)

        provider.get_capabilities.assert_called_once()
        mock_logger.warning.assert_any_call(
            "Sandbox not configured \u2014 agent will run without isolation"
        )

    @pytest.mark.asyncio
    async def test_start_agent_proceeds_when_setup_complete(self):
        """start_agent() proceeds normally when setup_complete=True."""
        provider = MagicMock()
        provider.get_capabilities.return_value = _make_capabilities(setup_complete=True)
        provider.stop_process = AsyncMock(return_value=False)

        mgr, ws, _, _, _ = _make_agent_manager(platform_provider=provider)
        config = _make_config()

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

            assert "proj_1" in mgr._handles
            ws.broadcast.assert_called()


# ---------------------------------------------------------------------------
# Unit tests: stop_agent calls stop_process
# ---------------------------------------------------------------------------


class TestStopAgentCallsStopProcess:
    """stop_agent() must call provider.stop_process(project_id)."""

    @pytest.mark.asyncio
    async def test_stop_agent_calls_stop_process(self):
        """stop_agent() calls provider.stop_process(project_id) during cleanup."""
        provider = MagicMock()
        provider.get_capabilities.return_value = _make_capabilities(setup_complete=True)
        provider.stop_process = AsyncMock(return_value=True)

        mgr, ws, sub_mgr, _, _ = _make_agent_manager(platform_provider=provider)

        # Set up a fake handle
        mock_session = MagicMock()
        mock_task = asyncio.ensure_future(asyncio.sleep(0))
        await asyncio.sleep(0)  # let it complete

        handle = MagicMock(session=mock_session, task=mock_task)
        mgr._handles["proj_1"] = handle

        await mgr.stop_agent("proj_1")

        provider.stop_process.assert_awaited_once_with("proj_1")
        mock_session.stop.assert_called_once()

    @pytest.mark.asyncio
    async def test_stop_agent_ignores_stop_process_false(self):
        """stop_agent() continues normally when stop_process returns False."""
        provider = MagicMock()
        provider.stop_process = AsyncMock(return_value=False)

        mgr, ws, sub_mgr, _, _ = _make_agent_manager(platform_provider=provider)

        mock_session = MagicMock()
        mock_task = asyncio.ensure_future(asyncio.sleep(0))
        await asyncio.sleep(0)

        handle = MagicMock(session=mock_session, task=mock_task)
        mgr._handles["proj_1"] = handle

        await mgr.stop_agent("proj_1")

        # Should still broadcast stopped status
        ws.broadcast.assert_called()
        last_call = ws.broadcast.call_args[0][1]
        assert last_call["status"] == "stopped"


# ---------------------------------------------------------------------------
# Unit tests: trigger_source parameter
# ---------------------------------------------------------------------------


class TestTriggerSource:
    """start_agent() accepts trigger_source param and stores it."""

    @pytest.mark.asyncio
    async def test_start_agent_accepts_trigger_source(self):
        """start_agent() accepts trigger_source and stores it on the handle."""
        provider = MagicMock()
        provider.get_capabilities.return_value = _make_capabilities(setup_complete=True)
        provider.stop_process = AsyncMock(return_value=False)

        mgr, ws, _, _, _ = _make_agent_manager(platform_provider=provider)
        config = _make_config()

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

            await mgr.start_agent("proj_1", config, trigger_source="desktop_app")

            handle = mgr._handles["proj_1"]
            assert hasattr(handle, "trigger_source") or hasattr(handle, "_trigger_source")
            # Check the stored value
            trigger = getattr(handle, "trigger_source", None) or getattr(handle, "_trigger_source", None)
            assert trigger == "desktop_app"

    @pytest.mark.asyncio
    async def test_start_agent_trigger_source_defaults_none(self):
        """trigger_source defaults to None when not provided."""
        provider = MagicMock()
        provider.get_capabilities.return_value = _make_capabilities(setup_complete=True)
        provider.stop_process = AsyncMock(return_value=False)

        mgr, ws, _, _, _ = _make_agent_manager(platform_provider=provider)
        config = _make_config()

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

            handle = mgr._handles["proj_1"]
            trigger = getattr(handle, "trigger_source", None) or getattr(handle, "_trigger_source", None)
            assert trigger is None


# ---------------------------------------------------------------------------
# Unit tests: AgentConfig budget fields
# ---------------------------------------------------------------------------


class TestProjectBudgetFields:
    """Project model accepts budget fields, serializes/deserializes correctly."""

    def test_agent_config_has_token_budget(self):
        """AgentConfig has token_budget with default 500_000."""
        config = _make_config()
        assert config.token_budget == 500_000

    def test_agent_config_custom_token_budget(self):
        """AgentConfig accepts custom token_budget."""
        config = _make_config(token_budget=1_000_000)
        assert config.token_budget == 1_000_000

    def test_agent_config_has_max_iterations(self):
        """AgentConfig has max_iterations with default 50."""
        config = _make_config()
        assert config.max_iterations == 50

    def test_agent_config_custom_max_iterations(self):
        """AgentConfig accepts custom max_iterations."""
        config = _make_config(max_iterations=200)
        assert config.max_iterations == 200

    def test_agent_config_budget_serializes_to_dict(self):
        """AgentConfig budget fields survive dataclass dict conversion."""
        from dataclasses import asdict
        config = _make_config(token_budget=750_000, max_iterations=100)
        d = asdict(config)
        assert d["token_budget"] == 750_000
        assert d["max_iterations"] == 100

    def test_agent_config_budget_roundtrip_json(self):
        """AgentConfig budget fields survive JSON roundtrip."""
        from dataclasses import asdict
        config = _make_config(token_budget=750_000, max_iterations=100)
        d = asdict(config)
        serialized = json.dumps(d)
        restored = json.loads(serialized)
        assert restored["token_budget"] == 750_000
        assert restored["max_iterations"] == 100


# ---------------------------------------------------------------------------
# Integration: full start->stop cycle with NullProvider
# ---------------------------------------------------------------------------


class TestStartStopCycleNullProvider:
    """Integration test: full start -> stop cycle with NullProvider.

    NullProvider returns isolation_method='none', so the capability gate
    should be skipped (no real isolation to enforce). This allows agents
    to run on non-Windows platforms without blocking.
    When no platform_provider is passed (None), the start should also proceed
    without capability checks (backward-compatible dev mode).
    """

    @pytest.mark.asyncio
    async def test_start_with_null_provider_proceeds(self):
        """NullProvider has isolation_method='none', so start should NOT be rejected."""
        from agent_os.platform.null import NullProvider

        provider = NullProvider()
        mgr, ws, _, _, _ = _make_agent_manager(platform_provider=provider)
        config = _make_config()

        # Should NOT raise — NullProvider is treated as dev mode
        await mgr.start_agent("proj_1", config)

    @pytest.mark.asyncio
    async def test_start_without_provider_proceeds(self):
        """When platform_provider=None, start_agent skips capability check (dev mode)."""
        mgr, ws, _, _, _ = _make_agent_manager(platform_provider=None)
        config = _make_config()

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
            assert "proj_1" in mgr._handles

    @pytest.mark.asyncio
    async def test_full_start_stop_with_mock_provider(self):
        """Full start -> stop cycle with a mock provider that has setup_complete=True."""
        provider = MagicMock()
        provider.get_capabilities.return_value = _make_capabilities(setup_complete=True)
        provider.stop_process = AsyncMock(return_value=True)

        mgr, ws, sub_mgr, _, _ = _make_agent_manager(platform_provider=provider)
        config = _make_config()

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

            # Start
            await mgr.start_agent("proj_1", config)
            assert "proj_1" in mgr._handles
            provider.get_capabilities.assert_called_once()

            # Stop
            await mgr.stop_agent("proj_1")
            provider.stop_process.assert_awaited_once_with("proj_1")
            assert "proj_1" not in mgr._handles
