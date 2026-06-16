# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Regression: the circuit breaker must trip only on genuine tool/infra
failures — never on recoverable input errors the model can correct.

Before: two errors whose content matched → the tool was added to blocked_tools
and the agent saw '[SYSTEM] The <tool> tool is currently unavailable…'. But an
edit 'old_text not found' is recoverable bad INPUT, not a tool outage, so
disabling the tool is the wrong response (the agent should re-read and retry).

Fix: recoverable failures (ToolResult.meta['recoverable'] is True) never count
toward the breaker. Genuine failures (raised exceptions, unmarked errors) still
trip it after two identical occurrences, compared on the full error string.
"""

from __future__ import annotations

import json

import pytest

from agent_os.agent.loop import AgentLoop
from agent_os.agent.providers.types import StreamChunk, TokenUsage
from agent_os.agent.session import Session
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


# --- Tests -------------------------------------------------------------------

@pytest.mark.asyncio
async def test_recoverable_errors_do_not_disable_tool(tmp_path):
    """Two identical recoverable errors must NOT disable the tool."""
    session = Session.new("breaker_recoverable", str(tmp_path))
    registry = _ResultRegistry(result=ToolResult(
        content="Error: old_text not found in x.md",
        meta={"recoverable": True},
    ))
    provider = _ToolThenDoneProvider("edit", {"path": "x.md", "old_text": "a", "new_text": "b"})
    loop = _make_loop(session, provider, registry)
    await loop.run("try an edit")

    assert not _was_disabled(session, "edit")


@pytest.mark.asyncio
async def test_genuine_failures_still_disable_tool(tmp_path):
    """Two identical genuine tool failures (raised) STILL trip the breaker."""
    session = Session.new("breaker_infra", str(tmp_path))
    registry = _ResultRegistry(raises=OSError("disk on fire"))
    provider = _ToolThenDoneProvider("shell", {"command": "ls"})
    loop = _make_loop(session, provider, registry)
    await loop.run("run a command")

    assert _was_disabled(session, "shell")


@pytest.mark.asyncio
async def test_unmarked_error_results_still_disable_tool(tmp_path):
    """An error result WITHOUT a recoverable marker is treated as a failure."""
    session = Session.new("breaker_unmarked", str(tmp_path))
    registry = _ResultRegistry(result=ToolResult(content="Error: backend exploded"))
    provider = _ToolThenDoneProvider("shell", {"command": "ls"})
    loop = _make_loop(session, provider, registry)
    await loop.run("run a command")

    assert _was_disabled(session, "shell")


@pytest.mark.asyncio
async def test_real_edit_not_found_does_not_disable_edit(tmp_path):
    """End-to-end: repeated real edit not-found errors must NOT disable edit."""
    (tmp_path / "doc.md").write_text("the actual content\n", encoding="utf-8")
    session = Session.new("breaker_edit_e2e", str(tmp_path))
    registry = _EditRegistry(str(tmp_path))
    provider = _ToolThenDoneProvider(
        "edit", {"path": "doc.md", "old_text": "not in the file", "new_text": "x"},
    )
    loop = _make_loop(session, provider, registry)
    await loop.run("edit the doc")

    # The edit really ran twice and really failed both times...
    assert len(registry.executions) >= 2
    # ...but the tool was never disabled.
    assert not _was_disabled(session, "edit")


def test_edit_not_found_is_marked_recoverable(tmp_path):
    """EditTool tags its input-validation errors recoverable so the breaker
    can classify them deterministically (not by string-sniffing)."""
    from agent_os.agent.tools.edit import EditTool

    (tmp_path / "doc.md").write_text("hello world\n", encoding="utf-8")
    tool = EditTool(workspace=str(tmp_path))

    not_found = tool.execute(path="doc.md", old_text="absent", new_text="x")
    assert not_found.meta is not None and not_found.meta.get("recoverable") is True

    (tmp_path / "dup.md").write_text("foo foo", encoding="utf-8")
    multi = tool.execute(path="dup.md", old_text="foo", new_text="bar")
    assert multi.meta is not None and multi.meta.get("recoverable") is True

    success = tool.execute(path="doc.md", old_text="hello world", new_text="hi")
    # Successful edits carry no recoverable marker.
    assert success.meta is None or not success.meta.get("recoverable")
