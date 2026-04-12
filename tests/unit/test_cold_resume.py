# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Unit tests for cold resume injection (ContextManager) and session-end loop wiring (AgentLoop).

Tests the workspace files integration added by the workspace-files feature:
- Cold resume context injection into ContextManager.prepare()
- on_session_end callback in AgentLoop.run() finally block
"""

import asyncio
import logging
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent_os.agent.providers.types import (
    StreamChunk,
    LLMResponse,
    TokenUsage,
    LLMError,
)
from agent_os.agent.tools.base import ToolResult
from agent_os.agent.prompt_builder import PromptBuilder, PromptContext, Autonomy
from agent_os.agent.session import Session
from agent_os.agent.loop import AgentLoop
from agent_os.agent.context import ContextManager
from agent_os.agent.workspace_files import WorkspaceFileManager


# ---------------------------------------------------------------------------
# Helpers (same patterns as test_component_a.py)
# ---------------------------------------------------------------------------

def _make_base_prompt_context(workspace: str) -> PromptContext:
    return PromptContext(
        workspace=workspace,
        model="test-model",
        autonomy=Autonomy.HANDS_OFF,
        enabled_agents=[],
        tool_names=["read", "write", "shell"],
        os_type="linux",
        datetime_now="2026-01-01T00:00:00",
        context_usage_pct=0.0,
    )


class MockPromptBuilder:
    def build(self, context: PromptContext) -> tuple[str, str]:
        return ("cached-system-prefix", "dynamic-suffix")


def _make_text_response(text: str, input_tok: int = 100, output_tok: int = 50) -> LLMResponse:
    return LLMResponse(
        raw_message={"role": "assistant", "content": text},
        text=text,
        tool_calls=[],
        has_tool_calls=False,
        finish_reason="stop",
        status_text=None,
        usage=TokenUsage(input_tokens=input_tok, output_tokens=output_tok),
    )


class MockProvider:
    def __init__(self, responses=None):
        self._responses = list(responses or [])
        self._call_idx = 0

    async def stream(self, messages, tools=None):
        idx = self._call_idx
        self._call_idx += 1
        if self._responses and idx < len(self._responses):
            resp = self._responses[idx]
            if resp.text:
                yield StreamChunk(text=resp.text)
            yield StreamChunk(is_final=True, usage=resp.usage)
        else:
            yield StreamChunk(text="default", is_final=True,
                              usage=TokenUsage(10, 5))

    async def complete(self, messages, tools=None):
        idx = self._call_idx
        self._call_idx += 1
        if self._responses and idx < len(self._responses):
            return self._responses[idx]
        return _make_text_response("default")


class MockToolRegistry:
    def __init__(self):
        self.execute_calls = []

    def schemas(self):
        return []

    def execute(self, name, arguments):
        self.execute_calls.append((name, arguments))
        return ToolResult(content=f"result of {name}")

    def tool_names(self):
        return []

    def reset_run_state(self):
        pass


# ===========================================================================
# Cold Resume: ContextManager + WorkspaceFileManager
# ===========================================================================

class TestColdResumeInjection:

    def test_cold_resume_injected_on_first_prepare(self, tmp_path):
        """When workspace files exist, first prepare() includes resume context as system message."""
        # Set up workspace files
        wfm = WorkspaceFileManager(str(tmp_path))
        wfm.ensure_dir()
        wfm.write("agent", "# Agent Directive\n\nRefactor the auth module.")
        wfm.write("state", "# Project State\n\nWorking on token refresh.")

        session = Session.new("cold1", str(tmp_path))
        builder = MockPromptBuilder()
        ctx = _make_base_prompt_context(str(tmp_path))
        cm = ContextManager(session, builder, ctx, workspace_files=wfm)

        messages = cm.prepare()

        # Find the resume context message
        resume_msgs = [m for m in messages
                       if m.get("role") == "system"
                       and "WORKSPACE MEMORY" in m.get("content", "")]
        assert len(resume_msgs) == 1
        content = resume_msgs[0]["content"]
        assert "Agent Directive" in content
        assert "Refactor the auth module" in content
        assert "Project State" in content
        assert "token refresh" in content

    def test_cold_resume_not_re_injected(self, tmp_path):
        """Second prepare() call should NOT include resume context again."""
        wfm = WorkspaceFileManager(str(tmp_path))
        wfm.ensure_dir()
        wfm.write("agent", "# Agent Directive\n\nDo stuff.")

        session = Session.new("cold2", str(tmp_path))
        builder = MockPromptBuilder()
        ctx = _make_base_prompt_context(str(tmp_path))
        cm = ContextManager(session, builder, ctx, workspace_files=wfm)

        # First call: should inject
        msgs1 = cm.prepare()
        resume1 = [m for m in msgs1
                    if "WORKSPACE MEMORY" in m.get("content", "")]
        assert len(resume1) == 1

        # Second call: should NOT inject again
        msgs2 = cm.prepare()
        resume2 = [m for m in msgs2
                    if "WORKSPACE MEMORY" in m.get("content", "")]
        assert len(resume2) == 0

    def test_cold_resume_no_files(self, tmp_path):
        """Empty workspace (no files) should produce no injection and no error."""
        wfm = WorkspaceFileManager(str(tmp_path))
        wfm.ensure_dir()
        # No files written

        session = Session.new("cold3", str(tmp_path))
        builder = MockPromptBuilder()
        ctx = _make_base_prompt_context(str(tmp_path))
        cm = ContextManager(session, builder, ctx, workspace_files=wfm)

        messages = cm.prepare()

        resume_msgs = [m for m in messages
                       if "WORKSPACE MEMORY" in m.get("content", "")]
        assert len(resume_msgs) == 0

    def test_cold_resume_partial_files(self, tmp_path):
        """Only AGENT.md + PROJECT_STATE exist. Only those two sections included."""
        wfm = WorkspaceFileManager(str(tmp_path))
        wfm.ensure_dir()
        wfm.write("agent", "# Agent\n\nBuild the widget.")
        wfm.write("state", "# State\n\nWidget is 50% done.")
        # decisions, lessons, context, session_log are NOT written

        session = Session.new("cold4", str(tmp_path))
        builder = MockPromptBuilder()
        ctx = _make_base_prompt_context(str(tmp_path))
        cm = ContextManager(session, builder, ctx, workspace_files=wfm)

        messages = cm.prepare()

        resume_msgs = [m for m in messages
                       if "WORKSPACE MEMORY" in m.get("content", "")]
        assert len(resume_msgs) == 1
        content = resume_msgs[0]["content"]
        assert "Build the widget" in content
        assert "Widget is 50% done" in content
        # Should NOT contain sections for missing files
        assert "Decisions" not in content
        assert "Lessons Learned" not in content
        assert "External Context" not in content
        assert "Session Log" not in content

    def test_context_budget_with_resume(self, tmp_path):
        """Large workspace files should reduce sliding window proportionally."""
        wfm = WorkspaceFileManager(str(tmp_path))
        wfm.ensure_dir()
        # Write a very large agent directive to consume budget
        large_content = "X" * 100_000
        wfm.write("agent", large_content)

        session = Session.new("cold5", str(tmp_path))
        builder = MockPromptBuilder()
        ctx = _make_base_prompt_context(str(tmp_path))

        # Small context limit to make the effect visible
        cm_with = ContextManager(
            session, builder, ctx,
            model_context_limit=50_000, response_reserve=5_000,
            workspace_files=wfm,
        )
        cm_without = ContextManager(
            session, builder, ctx,
            model_context_limit=50_000, response_reserve=5_000,
            workspace_files=None,
        )

        msgs_with = cm_with.prepare()
        msgs_without = cm_without.prepare()

        # The version with large workspace files should have higher usage percentage
        assert cm_with.usage_percentage > cm_without.usage_percentage

    def test_cold_resume_none_workspace_files(self, tmp_path):
        """When workspace_files=None (default), prepare() works normally without injection."""
        session = Session.new("cold6", str(tmp_path))
        builder = MockPromptBuilder()
        ctx = _make_base_prompt_context(str(tmp_path))
        cm = ContextManager(session, builder, ctx)  # workspace_files defaults to None

        messages = cm.prepare()

        resume_msgs = [m for m in messages
                       if "WORKSPACE MEMORY" in m.get("content", "")]
        assert len(resume_msgs) == 0
        # Should still have at least the system prompt
        assert len(messages) >= 1
        assert messages[0]["role"] == "system"


# ===========================================================================
# Session-End: AgentLoop on_session_end callback
# ===========================================================================

class TestSessionEndCallback:

    @pytest.mark.asyncio
    async def test_session_end_triggered_on_graceful_stop(self, tmp_path):
        """Loop exits normally (text response) -> on_session_end fired as background task."""
        session = Session.new("end1", str(tmp_path))
        provider = MockProvider(responses=[_make_text_response("Done.")])
        registry = MockToolRegistry()
        builder = MockPromptBuilder()
        ctx = _make_base_prompt_context(str(tmp_path))
        context_mgr = ContextManager(session, builder, ctx)

        callback = AsyncMock()
        loop = AgentLoop(
            session, provider, registry, context_mgr,
            on_session_end=callback,
        )
        await loop.run(initial_message="hello")

        # on_session_end is now fire-and-forget (asyncio.create_task),
        # so yield control to let the background task execute.
        await asyncio.sleep(0)
        callback.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_session_end_skipped_on_llm_failure(self, tmp_path):
        """LLM fails 3 times -> on_session_end skipped (provider unreachable)."""
        session = Session.new("end2", str(tmp_path))

        class ErrorProvider:
            _call_count = 0

            async def stream(self, messages, tools=None):
                self._call_count += 1
                # Fail 3 times to trigger the "3 retries" exit with system msg
                raise LLMError("Server error", status_code=500)
                yield  # make this an async generator

            async def complete(self, messages, tools=None):
                return _make_text_response("ok")

        provider = ErrorProvider()
        registry = MockToolRegistry()
        builder = MockPromptBuilder()
        ctx = _make_base_prompt_context(str(tmp_path))
        context_mgr = ContextManager(session, builder, ctx)

        session_end = AsyncMock()
        loop = AgentLoop(
            session, provider, registry, context_mgr,
            on_session_end=session_end,
        )
        await loop.run(initial_message="test")

        # session-end is skipped when LLM failed (provider unreachable)
        session_end.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_session_end_failure_logged_not_raised(self, tmp_path):
        """on_session_end callback throws -> fire-and-forget task, no crash in loop."""
        session = Session.new("end3", str(tmp_path))
        provider = MockProvider(responses=[_make_text_response("Done.")])
        registry = MockToolRegistry()
        builder = MockPromptBuilder()
        ctx = _make_base_prompt_context(str(tmp_path))
        context_mgr = ContextManager(session, builder, ctx)

        async def failing_callback():
            raise RuntimeError("Session-end exploded!")

        loop = AgentLoop(
            session, provider, registry, context_mgr,
            on_session_end=failing_callback,
        )

        # Should not raise — callback failure is contained in fire-and-forget task
        await loop.run(initial_message="hello")
        # Let the background task execute and fail silently
        await asyncio.sleep(0)
        assert not loop.is_running

    @pytest.mark.asyncio
    async def test_session_end_not_called_when_none(self, tmp_path):
        """When on_session_end is None (default), loop exits cleanly."""
        session = Session.new("end4", str(tmp_path))
        provider = MockProvider(responses=[_make_text_response("Done.")])
        registry = MockToolRegistry()
        builder = MockPromptBuilder()
        ctx = _make_base_prompt_context(str(tmp_path))
        context_mgr = ContextManager(session, builder, ctx)

        loop = AgentLoop(session, provider, registry, context_mgr)
        # Should not raise — no callback set
        await loop.run(initial_message="hello")
        assert not loop.is_running
