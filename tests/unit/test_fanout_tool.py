# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Unit tests for FanoutTool: task-count validation, depth gating, and the
error-vs-yield_turn ToolResult contract mirroring AgentMessageTool's `send`.

Spec 009 (subagent fanout), Task 3 brief.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from agent_os.agent.tools.fanout import MAX_DEPTH, FanoutTool


def _make_sam(dispatch_return=None):
    sam = MagicMock()
    sam.dispatch_fanout = AsyncMock(return_value=dispatch_return)
    return sam


@pytest.fixture
def stub_sam():
    """A SubAgentManager stub whose dispatch_fanout should NEVER be reached
    by the tests that use it — configured to return an obviously-wrong value
    so an accidental call is loud, not silent."""
    return _make_sam(dispatch_return="Error: dispatch_fanout should not have been called")


@pytest.fixture
def stub_sam_ok():
    """A SubAgentManager stub whose dispatch_fanout succeeds, mirroring the
    real success message shape from sub_agent_manager.dispatch_fanout."""
    return _make_sam(
        dispatch_return=(
            "Fanout ab12cd34 dispatched: 2 tasks — a, b. "
            "Results will arrive together when all tasks finish."
        )
    )


class TestTaskCountValidation:
    @pytest.mark.asyncio
    async def test_send_requires_two_tasks(self, stub_sam):
        t = FanoutTool(sub_agent_manager=stub_sam, project_id="p", session_id="s")
        r = await t.execute(tasks=[{"brief": "x", "label": "a"}])
        assert r.content.startswith("Error")
        stub_sam.dispatch_fanout.assert_not_called()

    @pytest.mark.asyncio
    async def test_rejects_more_than_five_tasks(self, stub_sam):
        t = FanoutTool(sub_agent_manager=stub_sam, project_id="p", session_id="s")
        tasks = [{"brief": f"b{i}", "label": f"l{i}"} for i in range(6)]
        r = await t.execute(tasks=tasks)
        assert r.content.startswith("Error")
        stub_sam.dispatch_fanout.assert_not_called()

    @pytest.mark.asyncio
    async def test_rejects_empty_tasks(self, stub_sam):
        t = FanoutTool(sub_agent_manager=stub_sam, project_id="p", session_id="s")
        r = await t.execute(tasks=[])
        assert r.content.startswith("Error")
        stub_sam.dispatch_fanout.assert_not_called()

    @pytest.mark.asyncio
    async def test_rejects_task_missing_brief(self, stub_sam):
        t = FanoutTool(sub_agent_manager=stub_sam, project_id="p", session_id="s")
        r = await t.execute(tasks=[{"label": "a"}, {"brief": "y", "label": "b"}])
        assert r.content.startswith("Error")
        stub_sam.dispatch_fanout.assert_not_called()

    @pytest.mark.asyncio
    async def test_rejects_task_missing_label(self, stub_sam):
        t = FanoutTool(sub_agent_manager=stub_sam, project_id="p", session_id="s")
        r = await t.execute(tasks=[{"brief": "x"}, {"brief": "y", "label": "b"}])
        assert r.content.startswith("Error")
        stub_sam.dispatch_fanout.assert_not_called()


class TestSuccessfulDispatch:
    @pytest.mark.asyncio
    async def test_successful_dispatch_yields_turn(self, stub_sam_ok):
        t = FanoutTool(sub_agent_manager=stub_sam_ok, project_id="p", session_id="s")
        r = await t.execute(tasks=[{"brief": "x", "label": "a"},
                                   {"brief": "y", "label": "b"}])
        assert r.meta and r.meta.get("yield_turn") is True
        assert "dispatched" in r.content.lower()

    @pytest.mark.asyncio
    async def test_dispatch_forwards_project_session_and_depth(self, stub_sam_ok):
        t = FanoutTool(sub_agent_manager=stub_sam_ok, project_id="proj-1",
                        session_id="sess-1", depth=1)
        await t.execute(tasks=[{"brief": "x", "label": "a"},
                                {"brief": "y", "label": "b"}])
        stub_sam_ok.dispatch_fanout.assert_awaited_once()
        args, kwargs = stub_sam_ok.dispatch_fanout.call_args
        assert args[0] == "proj-1"
        assert kwargs["session_id"] == "sess-1"
        assert kwargs["depth"] == 2  # depth + 1, mirrors agent_message's send

    @pytest.mark.asyncio
    async def test_max_runtime_s_forwarded(self, stub_sam_ok):
        t = FanoutTool(sub_agent_manager=stub_sam_ok, project_id="p", session_id="s")
        await t.execute(
            tasks=[{"brief": "x", "label": "a"}, {"brief": "y", "label": "b"}],
            max_runtime_s=120,
        )
        _, kwargs = stub_sam_ok.dispatch_fanout.call_args
        assert kwargs["max_runtime_s"] == 120

    @pytest.mark.asyncio
    async def test_max_runtime_s_defaults_to_3600(self, stub_sam_ok):
        t = FanoutTool(sub_agent_manager=stub_sam_ok, project_id="p", session_id="s")
        await t.execute(tasks=[{"brief": "x", "label": "a"}, {"brief": "y", "label": "b"}])
        _, kwargs = stub_sam_ok.dispatch_fanout.call_args
        assert kwargs["max_runtime_s"] == 3600

    @pytest.mark.asyncio
    async def test_error_string_from_manager_does_not_yield(self, stub_sam):
        """dispatch_fanout returning an Error string (e.g. concurrency cap)
        must surface as a plain ToolResult, NOT yield_turn — mirrors
        AgentMessageTool's failed-dispatch handling."""
        t = FanoutTool(sub_agent_manager=stub_sam, project_id="p", session_id="s")
        r = await t.execute(tasks=[{"brief": "x", "label": "a"},
                                    {"brief": "y", "label": "b"}])
        assert r.content.startswith("Error")
        assert not (r.meta and r.meta.get("yield_turn"))


class TestDepthGate:
    @pytest.mark.asyncio
    async def test_depth_gate(self, stub_sam_ok):
        t = FanoutTool(sub_agent_manager=stub_sam_ok, project_id="p",
                       session_id="s", depth=3)
        r = await t.execute(tasks=[{"brief": "x", "label": "a"},
                                   {"brief": "y", "label": "b"}])
        assert r.content.startswith("Error")

    @pytest.mark.asyncio
    async def test_depth_gate_blocks_before_dispatch(self, stub_sam_ok):
        t = FanoutTool(sub_agent_manager=stub_sam_ok, project_id="p",
                       session_id="s", depth=MAX_DEPTH)
        await t.execute(tasks=[{"brief": "x", "label": "a"},
                                {"brief": "y", "label": "b"}])
        stub_sam_ok.dispatch_fanout.assert_not_called()

    @pytest.mark.asyncio
    async def test_depth_below_max_is_allowed(self, stub_sam_ok):
        t = FanoutTool(sub_agent_manager=stub_sam_ok, project_id="p",
                       session_id="s", depth=MAX_DEPTH - 1)
        r = await t.execute(tasks=[{"brief": "x", "label": "a"},
                                    {"brief": "y", "label": "b"}])
        assert not r.content.startswith("Error")
        stub_sam_ok.dispatch_fanout.assert_awaited_once()


class TestNoSubAgentManager:
    @pytest.mark.asyncio
    async def test_missing_sub_agent_manager_errors(self):
        t = FanoutTool(sub_agent_manager=None, project_id="p", session_id="s")
        r = await t.execute(tasks=[{"brief": "x", "label": "a"},
                                    {"brief": "y", "label": "b"}])
        assert r.content.startswith("Error")


class TestToolSchema:
    def test_schema_shape(self):
        t = FanoutTool()
        schema = t.schema()
        assert schema["function"]["name"] == "fanout"
        params = schema["function"]["parameters"]
        assert params["required"] == ["tasks"]
        tasks_schema = params["properties"]["tasks"]
        assert tasks_schema["minItems"] == 2
        assert tasks_schema["maxItems"] == 5
        assert tasks_schema["items"]["required"] == ["brief", "label"]
        assert "max_runtime_s" in params["properties"]

    def test_is_async(self):
        assert FanoutTool.is_async is True
