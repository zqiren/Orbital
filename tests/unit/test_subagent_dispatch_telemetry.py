# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Sub-agent dispatch telemetry: nothing counted a delegation.

The daily ping could say how many management turns an install ran but not
whether it ever handed work to a sub-agent — the product's differentiator.
Two spool events close that gap, each at its chokepoint:

- ``subagent_dispatched`` in ``SubAgentManager._dispatch_async`` — the one
  call every prompt crosses on its way to a worker (immediate send, a drained
  queue, fanout).
- ``subagent_failed`` where a dispatch's failure is decided: a spawn that
  never started (``send``), and the observer's negative terminals
  (``on_error`` / ``on_failed``). A deliberate stop is neither.
"""

import json
from types import SimpleNamespace

import pytest

from agent_os import telemetry
from agent_os.daemon_v2.lifecycle_observer import LifecycleObserver
from agent_os.daemon_v2.models import make_session_key
from agent_os.daemon_v2.sub_agent_manager import SubAgentManager


class _AgentManager:
    async def inject_system_message(self, project_id, content, **kwargs):
        pass


class _WS:
    def broadcast(self, project_id, payload):
        pass


class _ProcessManager:
    def set_turn_closed_callback(self, callback):
        pass

    def set_permission_request_callback(self, callback):
        pass

    def set_active_dispatch(self, *args, **kwargs):
        pass

    def clear_dispatch(self, *args, **kwargs):
        pass


class _Transport:
    def __init__(self):
        self.messages = []

    async def dispatch(self, message):
        self.messages.append(message)


@pytest.fixture
def spool_rows(tmp_path):
    telemetry.reset_for_tests()
    telemetry.configure(tmp_path, is_enabled=lambda: True)

    def read():
        path = tmp_path / "telemetry" / "events.jsonl"
        if not path.exists():
            return []
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]

    yield read
    telemetry.reset_for_tests()


def _subagent_rows(rows):
    return [(r["event"], r.get("agent")) for r in rows if r["event"].startswith("subagent_")]


def _manager(observer=None, handle="claude-code"):
    manager = SubAgentManager(_ProcessManager(), lifecycle_observer=observer)
    transport = _Transport()
    manager._adapters[make_session_key("p1", "s1")] = {
        handle: SimpleNamespace(_transport=transport, _broken=False),
    }
    return manager, transport


@pytest.mark.asyncio
async def test_send_counts_one_dispatch(spool_rows):
    manager, transport = _manager()

    result = await manager.send("p1", "claude-code", "do it", session_id="s1")

    assert result.startswith("Message sent")
    assert _subagent_rows(spool_rows()) == [("subagent_dispatched", "claude-code")]
    row = [r for r in spool_rows() if r["event"] == "subagent_dispatched"][0]
    # No content or ids ride along — only the agent handle, bucketed at rollup.
    assert set(row) == {"event", "ts", "agent"}


@pytest.mark.asyncio
async def test_queued_prompt_counts_when_it_drains_not_when_it_queues(spool_rows):
    manager, _ = _manager()
    manager._prompt_active.add(("p1", "s1", "claude-code"))

    result = await manager.send("p1", "claude-code", "later", session_id="s1")
    assert "queued" in result.lower()
    assert _subagent_rows(spool_rows()) == []

    manager._prompt_active.discard(("p1", "s1", "claude-code"))
    await manager._on_prompt_turn_closed(
        "p1", "claude-code", session_id="s1", cause="success")

    assert _subagent_rows(spool_rows()) == [("subagent_dispatched", "claude-code")]


@pytest.mark.asyncio
async def test_spawn_failure_counts_as_failed_without_a_dispatch(spool_rows):
    # No adapter registered and no config → start() returns "Error: ...".
    manager = SubAgentManager(_ProcessManager())

    result = await manager.send("p1", "codex", "do it", session_id="s1")

    assert result.startswith("Error")
    assert _subagent_rows(spool_rows()) == [("subagent_failed", "codex")]


@pytest.mark.asyncio
@pytest.mark.parametrize("terminal", ["on_error", "on_failed"])
async def test_negative_terminals_count_as_failed(spool_rows, terminal):
    observer = LifecycleObserver(_AgentManager(), _WS())

    if terminal == "on_error":
        await observer.on_error("p1", "codex", "boom", "/t.jsonl", session_id="s1")
    else:
        await observer.on_failed("p1", "codex", "background_send_exception",
                                 session_id="s1")

    assert _subagent_rows(spool_rows()) == [("subagent_failed", "codex")]


@pytest.mark.asyncio
async def test_completion_and_interrupt_are_not_failures(spool_rows):
    observer = LifecycleObserver(_AgentManager(), _WS())

    await observer.on_completed("p1", "codex", "done", "/t.jsonl", session_id="s1")
    await observer.on_turn_interrupted("p1", "codex", transcript_path="/t.jsonl",
                                       session_id="s1")

    assert _subagent_rows(spool_rows()) == []
