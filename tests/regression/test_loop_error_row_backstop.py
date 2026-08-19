# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""A run that dies on a non-LLMError must still leave a row in the session.

The loop's `except LLMError` branch writes "LLM error (non-recoverable)" /
"LLM error after N retries" rows, so provider failures are visible in chat.
Anything else — a raw SDK exception, a tool-registry fault, a session write
error — propagated straight through to `agent_manager._record_loop_error`,
which logs it and broadcasts a transient `agent.status` frame. If nothing is
listening on that socket at that instant (nobody has the project open, the
run was started by a scheduled trigger at 09:00), the failure leaves no trace
anywhere the user can find later: the session holds the trigger prompt and
nothing after it.

Observed live on 2026-07-29 and 2026-08-09. `openai_compat` now classifies
mid-stream drops (tests/regression/test_midstream_error_classified.py), which
covers the observed cause; this is the backstop for every other cause.
"""

import asyncio

import pytest

from agent_os.agent.context import ContextManager
from agent_os.agent.loop import AgentLoop
from agent_os.agent.prompt_builder import Autonomy, PromptContext
from agent_os.agent.providers.types import LLMError, StreamChunk, TokenUsage
from agent_os.agent.session import Session, persist_user_row
from agent_os.agent.tools.base import ToolResult


class MockPromptBuilder:
    def build(self, context: PromptContext) -> tuple[str, str, str]:
        return ("cached-system-prefix", "semi-stable-suffix", "dynamic-runtime")


class SimpleToolRegistry:
    def schemas(self) -> list[dict]:
        return []

    def is_async(self, name: str) -> bool:
        return False

    def execute(self, name: str, arguments: dict) -> ToolResult:
        return ToolResult(content="ok")

    async def execute_async(self, name: str, arguments: dict) -> ToolResult:
        return ToolResult(content="ok")

    def tool_names(self) -> list[str]:
        return []

    def reset_run_state(self) -> None:
        pass


class ExplodingProvider:
    """Provider whose stream raises something that is NOT an LLMError."""

    provider = "opencode-go"
    model = "deepseek-v4-flash"
    sdk = "openai"

    def __init__(self, exc: BaseException):
        self._exc = exc

    async def stream(self, messages, tools=None):
        raise self._exc
        yield  # pragma: no cover — makes this an async generator


class HealthyProvider:
    provider = "opencode-go"
    model = "deepseek-v4-flash"
    sdk = "openai"

    async def stream(self, messages, tools=None):
        yield StreamChunk(text="all good")
        yield StreamChunk(
            is_final=True,
            usage=TokenUsage(input_tokens=10, output_tokens=5),
            finish_reason="stop",
        )


def _make_loop(tmp_path, provider, session_id: str) -> tuple[AgentLoop, Session]:
    session = Session.new(session_id, str(tmp_path))
    context_mgr = ContextManager(
        session,
        MockPromptBuilder(),
        PromptContext(
            workspace=str(tmp_path), model="test-model",
            autonomy=Autonomy.HANDS_OFF, enabled_agents=[], tool_names=[],
            os_type="linux", datetime_now="2026-01-01T00:00:00",
            context_usage_pct=0.0,
        ),
    )
    loop = AgentLoop(
        session, provider, SimpleToolRegistry(), context_mgr,
        project_dir=str(tmp_path), max_iterations=10,
    )
    return loop, session


def _system_messages(session: Session) -> list[str]:
    return [m.get("content") or "" for m in session.get_messages()
            if m.get("role") == "system"]


@pytest.mark.asyncio
async def test_unexpected_exception_leaves_an_error_row(tmp_path):
    boom = RuntimeError("peer closed connection without sending complete message body")
    loop, session = _make_loop(tmp_path, ExplodingProvider(boom), "backstop_raw")
    persist_user_row(loop._session, "[Triggered by schedule 'Daily scan']\n\nscan it")

    with pytest.raises(RuntimeError):
        await loop.run()

    rows = _system_messages(session)
    assert rows, "run died with no row in the session — the silent-trigger shape"
    assert any("peer closed connection" in r for r in rows)
    assert loop._loop_exit_path == "internal_error"
    assert loop._llm_failed is True


@pytest.mark.asyncio
async def test_error_row_survives_for_the_reader(tmp_path):
    """The row must be persisted, not just held in memory — a trigger run
    nobody was watching is read back off disk hours later."""
    loop, session = _make_loop(
        tmp_path, ExplodingProvider(RuntimeError("boom")), "backstop_disk")
    persist_user_row(loop._session, "scan it")

    with pytest.raises(RuntimeError):
        await loop.run()

    reloaded = Session.load(session._filepath)
    assert any(m.get("role") == "system" and "boom" in (m.get("content") or "")
               for m in reloaded.get_messages())


@pytest.mark.asyncio
async def test_cancellation_writes_no_error_row(tmp_path):
    """cancel_turn()/stop must stay a clean cancel — a user-cancelled turn is
    not a failure and must not be reported as one. The loop absorbs a bare
    cancel via its own branch and exits without raising."""
    loop, session = _make_loop(
        tmp_path, ExplodingProvider(asyncio.CancelledError()), "backstop_cancel")
    persist_user_row(loop._session, "scan it")

    await loop.run()

    assert loop._loop_exit_path == "cancel"
    assert not any("error" in r.lower() or "failed" in r.lower()
                   for r in _system_messages(session))


@pytest.mark.asyncio
async def test_llm_error_keeps_its_own_row(tmp_path):
    """The backstop must not double-write on top of the LLMError branch."""
    loop, session = _make_loop(
        tmp_path, ExplodingProvider(LLMError("bad key", status_code=401)),
        "backstop_llm")
    persist_user_row(loop._session, "scan it")

    await loop.run()  # LLMError ABORT is handled, not re-raised

    rows = _system_messages(session)
    assert len(rows) == 1
    assert "LLM error (non-recoverable)" in rows[0]
    assert loop._loop_exit_path == "llm_error"


@pytest.mark.asyncio
async def test_healthy_run_is_untouched(tmp_path):
    loop, session = _make_loop(tmp_path, HealthyProvider(), "backstop_healthy")
    persist_user_row(loop._session, "scan it")

    await loop.run()

    assert loop._loop_exit_path == "text_complete"
    assert not _system_messages(session)
