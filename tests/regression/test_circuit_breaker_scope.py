# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Regression: a failing tool is never taken away from the agent (spec 171 B3).

Before: two identical errors from one tool added it to ``blocked_tools`` for
the rest of the run and wrote '[SYSTEM] The <tool> tool is currently
unavailable…'. Two ``read``s of one missing path removed ``read``. That
breaker is gone; the observation-aware repetition cap (5 identical
call+result hashes) is the single guard against a loop.
"""

from __future__ import annotations

import json

import pytest

from agent_os.agent.loop import AgentLoop
from agent_os.agent.providers.types import StreamChunk, TokenUsage
from agent_os.agent.session import Session, persist_user_row
from agent_os.agent.tools.base import ToolResult


# --- Minimal loop harness ----------------------------------------------------

class _FakeContextManager:
    def __init__(self, session):
        self._session = session
        self.model_context_limit = 200_000

    def prepare(self):
        return list(self._session.get_messages())

    def should_compact(self):
        return False


class _ToolThenDoneProvider:
    """Emits the same tool call for the first `n_tool_turns` stream() calls,
    then a text-only response so the loop exits cleanly."""

    def __init__(self, tool_name, args, n_tool_turns=2):
        self._tool_name = tool_name
        self._args = args
        self._n = n_tool_turns
        self._calls = 0

    @property
    def model(self):
        return "fake-model"

    async def stream(self, context, tools=None):
        self._calls += 1
        if self._calls <= self._n:
            yield StreamChunk(tool_calls_delta=[{
                "index": 0,
                "id": f"call_{self._calls}",
                "type": "function",
                "function": {
                    "name": self._tool_name,
                    "arguments": json.dumps(self._args),
                },
            }])
            yield StreamChunk(is_final=True, usage=TokenUsage(1, 1))
            return
        yield StreamChunk(text="done")
        yield StreamChunk(is_final=True, usage=TokenUsage(1, 1))


class _ResultRegistry:
    """Returns a fixed ToolResult for every call (or raises if `raises`)."""

    def __init__(self, *, result=None, raises=None):
        self._result = result
        self._raises = raises
        self.executions = []

    def reset_run_state(self):
        pass

    def schemas(self):
        return []

    def is_async(self, name):
        return False

    def execute(self, name, args):
        self.executions.append((name, args))
        if self._raises is not None:
            raise self._raises
        return self._result

    async def execute_async(self, name, args):
        return self.execute(name, args)


class _EditRegistry:
    """Routes the 'edit' tool to a real EditTool over a temp workspace."""

    def __init__(self, workspace):
        from agent_os.agent.tools.edit import EditTool
        self._edit = EditTool(workspace=workspace)
        self.executions = []

    def reset_run_state(self):
        pass

    def schemas(self):
        return []

    def is_async(self, name):
        return False

    def execute(self, name, args):
        self.executions.append((name, args))
        if name == "edit":
            return self._edit.execute(**args)
        return ToolResult(content=f"{name} executed")

    async def execute_async(self, name, args):
        return self.execute(name, args)


def _make_loop(session, provider, registry):
    return AgentLoop(
        session=session,
        provider=provider,
        tool_registry=registry,
        context_manager=_FakeContextManager(session),
    )


def _was_disabled(session, tool_name):
    for m in session.get_messages():
        c = m.get("content")
        if isinstance(c, str) and "is currently unavailable" in c and tool_name in c:
            return True
    return False


def _repetition_stopped(session):
    return any(
        isinstance(m.get("content"), str)
        and ("Repetitive action detected" in m["content"]
             or "Ping-pong pattern detected" in m["content"])
        for m in session.get_messages()
    )


# --- Tests -------------------------------------------------------------------

@pytest.mark.asyncio
async def test_tool_stays_available_after_two_identical_error_results(tmp_path):
    """read of a missing path, three times: every call really executes."""
    session = Session.new("breaker_gone_result", str(tmp_path))
    registry = _ResultRegistry(result=ToolResult(content="Error: file not found: missing.md"))
    provider = _ToolThenDoneProvider("read", {"path": "missing.md"}, n_tool_turns=3)
    loop = _make_loop(session, provider, registry)
    persist_user_row(loop._session, "read the file")
    await loop.run()

    assert len(registry.executions) == 3
    assert not _was_disabled(session, "read")
    assert not _repetition_stopped(session)


@pytest.mark.asyncio
async def test_tool_stays_available_after_two_identical_raised_failures(tmp_path):
    """A tool that raises the same exception twice is still executed again."""
    session = Session.new("breaker_gone_raise", str(tmp_path))
    registry = _ResultRegistry(raises=OSError("disk on fire"))
    provider = _ToolThenDoneProvider("shell", {"command": "ls"}, n_tool_turns=3)
    loop = _make_loop(session, provider, registry)
    persist_user_row(loop._session, "run a command")
    await loop.run()

    assert len(registry.executions) == 3
    assert not _was_disabled(session, "shell")


@pytest.mark.asyncio
async def test_real_edit_not_found_does_not_disable_edit(tmp_path):
    """End-to-end: repeated real edit not-found errors must NOT disable edit."""
    (tmp_path / "doc.md").write_text("the actual content\n", encoding="utf-8")
    session = Session.new("breaker_edit_e2e", str(tmp_path))
    registry = _EditRegistry(str(tmp_path))
    provider = _ToolThenDoneProvider(
        "edit", {"path": "doc.md", "old_text": "not in the file", "new_text": "x"},
        n_tool_turns=3,
    )
    loop = _make_loop(session, provider, registry)
    persist_user_row(loop._session, "edit the doc")
    await loop.run()

    assert len(registry.executions) == 3
    assert not _was_disabled(session, "edit")


@pytest.mark.asyncio
async def test_repetition_cap_still_stops_an_identical_error_loop(tmp_path):
    """The remaining guard ends the run: the same failing call repeated trips
    the repetition detection (its pair check fires on the 4th identical
    call, the 5-hash cap would on the 5th)."""
    session = Session.new("breaker_gone_cap", str(tmp_path))
    registry = _ResultRegistry(result=ToolResult(content="Error: file not found: missing.md"))
    provider = _ToolThenDoneProvider("read", {"path": "missing.md"}, n_tool_turns=50)
    loop = _make_loop(session, provider, registry)
    persist_user_row(loop._session, "read the file")
    await loop.run()

    assert len(registry.executions) <= 5
    assert _repetition_stopped(session)
    assert not _was_disabled(session, "read")
