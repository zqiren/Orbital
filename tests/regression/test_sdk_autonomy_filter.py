# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Regression tests for SDK transport autonomy-based permission filtering.

Verifies that SDKTransport auto-approves low-risk tool calls based on the
project's autonomy preset, rather than surfacing every tool call as an
approval request to the user.
"""

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from agent_os.agent.prompt_builder import Autonomy
from agent_os.agent.transports.sdk_transport import SDKTransport
from agent_os.agent.transports.tool_risk import classify_tool, should_auto_approve


# ---------------------------------------------------------------------------
# Tool risk mapper unit tests
# ---------------------------------------------------------------------------

class TestClassifyTool:
    def test_read_tools(self):
        for name in ("Read", "Glob", "Grep", "LS", "Search", "Explore"):
            assert classify_tool(name) == "read", f"{name} should be 'read'"

    def test_write_tools(self):
        for name in ("Edit", "Write", "MultiEdit", "NotebookEdit"):
            assert classify_tool(name) == "write", f"{name} should be 'write'"

    def test_shell_tools(self):
        for name in ("Bash", "ExecuteBash"):
            assert classify_tool(name) == "shell", f"{name} should be 'shell'"

    def test_delegate_tools(self):
        assert classify_tool("Agent") == "delegate"

    def test_unknown_tool_defaults_to_requires_approval(self):
        assert classify_tool("FutureNewTool") == "requires_approval"
        assert classify_tool("") == "requires_approval"
        assert classify_tool("SomeRandomThing") == "requires_approval"


class TestShouldAutoApprove:
    """Verify autonomy preset × tool category matrix."""

    # -- CHECK_IN: auto-approve reads only --
    def test_check_in_approves_read(self):
        assert should_auto_approve("Read", Autonomy.CHECK_IN) is True

    def test_check_in_surfaces_bash(self):
        assert should_auto_approve("Bash", Autonomy.CHECK_IN) is False

    def test_check_in_surfaces_write(self):
        assert should_auto_approve("Edit", Autonomy.CHECK_IN) is False

    def test_check_in_surfaces_delegate(self):
        assert should_auto_approve("Agent", Autonomy.CHECK_IN) is False

    def test_check_in_surfaces_unknown(self):
        assert should_auto_approve("FutureNewTool", Autonomy.CHECK_IN) is False

    # -- HANDS_OFF: auto-approve everything --
    def test_hands_off_approves_read(self):
        assert should_auto_approve("Read", Autonomy.HANDS_OFF) is True

    def test_hands_off_approves_bash(self):
        assert should_auto_approve("Bash", Autonomy.HANDS_OFF) is True

    def test_hands_off_approves_write(self):
        assert should_auto_approve("Edit", Autonomy.HANDS_OFF) is True

    def test_hands_off_approves_delegate(self):
        assert should_auto_approve("Agent", Autonomy.HANDS_OFF) is True

    def test_hands_off_approves_unknown(self):
        assert should_auto_approve("FutureNewTool", Autonomy.HANDS_OFF) is True

    # -- SUPERVISED: surface everything --
    def test_supervised_surfaces_read(self):
        assert should_auto_approve("Read", Autonomy.SUPERVISED) is False

    def test_supervised_surfaces_bash(self):
        assert should_auto_approve("Bash", Autonomy.SUPERVISED) is False

    def test_supervised_surfaces_write(self):
        assert should_auto_approve("Edit", Autonomy.SUPERVISED) is False

    def test_supervised_surfaces_delegate(self):
        assert should_auto_approve("Agent", Autonomy.SUPERVISED) is False

    def test_supervised_surfaces_unknown(self):
        assert should_auto_approve("FutureNewTool", Autonomy.SUPERVISED) is False


# ---------------------------------------------------------------------------
# SDKTransport integration tests
# ---------------------------------------------------------------------------

class TestSDKTransportAutonomyFilter:
    """Test that SDKTransport._handle_permission uses autonomy filtering."""

    @pytest.mark.asyncio
    async def test_read_auto_approved_under_check_in(self):
        """Read tool should be auto-approved under CHECK_IN — no event emitted."""
        transport = SDKTransport(autonomy=Autonomy.CHECK_IN)
        transport._alive = True

        from claude_agent_sdk import PermissionResultAllow
        result = await transport._handle_permission("Read", {"file_path": "/foo"}, None)

        assert isinstance(result, PermissionResultAllow)
        # No permission_request event should have been emitted
        assert transport._event_queue.empty()
        # No pending approvals should exist
        assert len(transport._pending_approvals) == 0

    @pytest.mark.asyncio
    async def test_bash_surfaces_approval_under_check_in(self):
        """Bash tool should surface approval under CHECK_IN — event emitted, blocks on future."""
        transport = SDKTransport(autonomy=Autonomy.CHECK_IN)
        transport._alive = True

        async def approve_shortly():
            await asyncio.sleep(0.05)
            for req_id, future in transport._pending_approvals.items():
                if not future.done():
                    future.set_result(True)
                    break

        from claude_agent_sdk import PermissionResultAllow

        task = asyncio.create_task(approve_shortly())
        result = await transport._handle_permission("Bash", {"command": "ls"}, None)
        await task

        assert isinstance(result, PermissionResultAllow)
        # A permission_request event SHOULD have been emitted
        assert not transport._event_queue.empty()
        event = transport._event_queue.get_nowait()
        assert event.event_type == "permission_request"
        assert event.data["tool_name"] == "Bash"

    @pytest.mark.asyncio
    async def test_unknown_tool_surfaces_under_check_in(self):
        """Unrecognized tool FutureNewTool must surface under CHECK_IN."""
        transport = SDKTransport(autonomy=Autonomy.CHECK_IN)
        transport._alive = True

        async def approve_shortly():
            await asyncio.sleep(0.05)
            for req_id, future in transport._pending_approvals.items():
                if not future.done():
                    future.set_result(True)
                    break

        task = asyncio.create_task(approve_shortly())
        result = await transport._handle_permission("FutureNewTool", {}, None)
        await task

        from claude_agent_sdk import PermissionResultAllow
        assert isinstance(result, PermissionResultAllow)
        assert not transport._event_queue.empty()
        event = transport._event_queue.get_nowait()
        assert event.event_type == "permission_request"
        assert event.data["tool_name"] == "FutureNewTool"

    @pytest.mark.asyncio
    async def test_unknown_tool_surfaces_under_supervised(self):
        """Unrecognized tool must surface under SUPERVISED."""
        transport = SDKTransport(autonomy=Autonomy.SUPERVISED)
        transport._alive = True

        async def approve_shortly():
            await asyncio.sleep(0.05)
            for req_id, future in transport._pending_approvals.items():
                if not future.done():
                    future.set_result(True)
                    break

        task = asyncio.create_task(approve_shortly())
        result = await transport._handle_permission("FutureNewTool", {}, None)
        await task

        assert not transport._event_queue.empty()

    @pytest.mark.asyncio
    async def test_unknown_tool_auto_approved_under_hands_off(self):
        """Under HANDS_OFF, even unknown tools are auto-approved."""
        transport = SDKTransport(autonomy=Autonomy.HANDS_OFF)
        transport._alive = True

        from claude_agent_sdk import PermissionResultAllow
        result = await transport._handle_permission("FutureNewTool", {}, None)

        assert isinstance(result, PermissionResultAllow)
        assert transport._event_queue.empty()

    @pytest.mark.asyncio
    async def test_hands_off_surfaces_nothing(self):
        """HANDS_OFF should auto-approve reads, writes, and shell — no events."""
        transport = SDKTransport(autonomy=Autonomy.HANDS_OFF)
        transport._alive = True

        from claude_agent_sdk import PermissionResultAllow

        for tool in ("Read", "Edit", "Bash", "Agent", "Glob"):
            result = await transport._handle_permission(tool, {}, None)
            assert isinstance(result, PermissionResultAllow), f"{tool} should be auto-approved"

        assert transport._event_queue.empty()

    @pytest.mark.asyncio
    async def test_supervised_surfaces_everything(self):
        """SUPERVISED should surface approval for all tools including reads."""
        transport = SDKTransport(autonomy=Autonomy.SUPERVISED)
        transport._alive = True

        tools = ["Read", "Glob", "Edit", "Bash", "Agent"]
        for tool in tools:
            async def approve_shortly():
                await asyncio.sleep(0.05)
                for req_id, future in transport._pending_approvals.items():
                    if not future.done():
                        future.set_result(True)
                        break

            task = asyncio.create_task(approve_shortly())
            await transport._handle_permission(tool, {}, None)
            await task

        # Every tool should have emitted a permission_request event
        events = []
        while not transport._event_queue.empty():
            events.append(transport._event_queue.get_nowait())
        assert len(events) == len(tools)
        for event in events:
            assert event.event_type == "permission_request"

    @pytest.mark.asyncio
    async def test_no_autonomy_falls_back_to_full_approval_flow(self):
        """When autonomy is None, all tools go through the full approval flow."""
        transport = SDKTransport(autonomy=None)
        transport._alive = True

        async def approve_shortly():
            await asyncio.sleep(0.05)
            for req_id, future in transport._pending_approvals.items():
                if not future.done():
                    future.set_result(True)
                    break

        # Even Read should require approval when autonomy is not set
        task = asyncio.create_task(approve_shortly())
        await transport._handle_permission("Read", {"file_path": "/foo"}, None)
        await task

        assert not transport._event_queue.empty()
        event = transport._event_queue.get_nowait()
        assert event.event_type == "permission_request"


class TestSDKTransportRuntimeUpdate:
    """Test that runtime autonomy updates propagate to SDKTransport."""

    @pytest.mark.asyncio
    async def test_update_autonomy_changes_filtering(self):
        """Changing autonomy at runtime changes filtering on next call."""
        transport = SDKTransport(autonomy=Autonomy.SUPERVISED)
        transport._alive = True

        from claude_agent_sdk import PermissionResultAllow

        # Under SUPERVISED, Read requires approval
        async def approve_shortly():
            await asyncio.sleep(0.05)
            for req_id, future in transport._pending_approvals.items():
                if not future.done():
                    future.set_result(True)
                    break

        task = asyncio.create_task(approve_shortly())
        await transport._handle_permission("Read", {}, None)
        await task

        assert not transport._event_queue.empty()
        transport._event_queue.get_nowait()  # drain

        # Update to CHECK_IN: Read should now be auto-approved
        transport.update_autonomy(Autonomy.CHECK_IN)

        result = await transport._handle_permission("Read", {}, None)
        assert isinstance(result, PermissionResultAllow)
        assert transport._event_queue.empty()  # no event emitted

    @pytest.mark.asyncio
    async def test_update_autonomy_to_hands_off_approves_all(self):
        """Switching to HANDS_OFF at runtime auto-approves everything."""
        transport = SDKTransport(autonomy=Autonomy.CHECK_IN)
        transport._alive = True

        from claude_agent_sdk import PermissionResultAllow

        # Bash requires approval under CHECK_IN
        # Switch to HANDS_OFF
        transport.update_autonomy(Autonomy.HANDS_OFF)

        result = await transport._handle_permission("Bash", {"command": "rm -rf /"}, None)
        assert isinstance(result, PermissionResultAllow)
        assert transport._event_queue.empty()


class TestSubAgentManagerAutonomyWiring:
    """Test that SubAgentManager propagates autonomy to SDK transports."""

    def test_update_sub_agent_autonomy(self):
        """update_sub_agent_autonomy calls transport.update_autonomy on SDK transports."""
        from unittest.mock import MagicMock
        from agent_os.daemon_v2.sub_agent_manager import SubAgentManager

        pm = MagicMock()
        mgr = SubAgentManager(process_manager=pm)

        # Create a mock adapter with an SDK transport that has update_autonomy
        mock_transport = MagicMock()
        mock_transport.update_autonomy = MagicMock()
        mock_adapter = MagicMock()
        mock_adapter._transport = mock_transport
        mgr._adapters["proj_1"] = {"claude-code": mock_adapter}

        mgr.update_sub_agent_autonomy("proj_1", Autonomy.HANDS_OFF)

        mock_transport.update_autonomy.assert_called_once_with(Autonomy.HANDS_OFF)

    def test_update_sub_agent_autonomy_skips_non_sdk_transports(self):
        """Transports without update_autonomy are silently skipped."""
        from unittest.mock import MagicMock
        from agent_os.daemon_v2.sub_agent_manager import SubAgentManager

        pm = MagicMock()
        mgr = SubAgentManager(process_manager=pm)

        # Create a mock adapter with a transport that has NO update_autonomy
        mock_transport = MagicMock(spec=[])  # empty spec = no attributes
        mock_adapter = MagicMock()
        mock_adapter._transport = mock_transport
        mgr._adapters["proj_1"] = {"other-agent": mock_adapter}

        # Should not raise
        mgr.update_sub_agent_autonomy("proj_1", Autonomy.CHECK_IN)

    def test_update_sub_agent_autonomy_no_adapters(self):
        """No-op when no adapters are running for the project."""
        from unittest.mock import MagicMock
        from agent_os.daemon_v2.sub_agent_manager import SubAgentManager

        pm = MagicMock()
        mgr = SubAgentManager(process_manager=pm)

        # Should not raise
        mgr.update_sub_agent_autonomy("nonexistent_project", Autonomy.CHECK_IN)
