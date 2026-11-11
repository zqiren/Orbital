# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Spec 172 D: two dispatches in one manager response both deliver.

A dispatch tool (``agent_message`` send/respond, ``fanout``) returns
``meta={"yield_turn": True}`` so the manager cannot busy-poll a worker. The
loop used to honour it at the FIRST such result and break out of the batch, so
every sibling still pending came back ``CANCELLED: This tool call was not
executed.`` (fresh-install run 2026-11-11: codex got its task, dsh's call was
cancelled). Now the whole batch runs first and the turn ends once after it,
under the same ``yield_turn`` exit label the queue dispatcher's continuation
hold keys on. Each worker's completion still wakes the manager through the
existing push path.

Real ``AgentLoop.run()`` with the real ``AgentMessageTool`` / ``FanoutTool``
over a fake sub-agent manager; a scripted LLM emits the batch.
"""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent_os.agent.context import ContextManager
from agent_os.agent.loop import AgentLoop
from agent_os.agent.prompt_builder import Autonomy, PromptContext
from agent_os.agent.providers.types import StreamChunk, TokenUsage
from agent_os.agent.session import Session, persist_user_row
from agent_os.agent.tools.agent_message import AgentMessageTool
from agent_os.agent.tools.base import ToolResult
from agent_os.agent.tools.fanout import FanoutTool
from agent_os.daemon_v2.agent_manager import AgentManager, ProjectHandle
from agent_os.daemon_v2.models import make_session_key

PROJECT = "proj_172"
SID = "sess-172-dispatch-batch"
_USAGE = TokenUsage(input_tokens=10, output_tokens=5)


class _Builder:
    def build(self, context):
        return ("sys", "suffix", "runtime")


def _ctx(ws):
    return PromptContext(workspace=ws, model="m", autonomy=Autonomy.HANDS_OFF,
                         enabled_agents=[], tool_names=[], os_type="linux",
                         datetime_now="2026-01-01T00:00:00", context_usage_pct=0.0)


def _batch(*calls):
    """One assistant response carrying ``calls`` = [(name, args), ...]."""
    return [
        StreamChunk(tool_calls_delta=[{
            "index": i, "id": f"call_{i}", "type": "function",
            "function": {"name": name, "arguments": json.dumps(args)},
        } for i, (name, args) in enumerate(calls)]),
        StreamChunk(is_final=True, usage=_USAGE, finish_reason="tool_calls"),
    ]


def _reply(text):
    return [StreamChunk(text=text),
            StreamChunk(is_final=True, usage=_USAGE, finish_reason="stop")]


class _LLM:
    provider = "test"
    model = "m"
    sdk = "openai"

    def __init__(self, script):
        self.script = script
        self.calls = 0

    def update_api_key(self, _key):
        pass

    async def stream(self, messages, tools=None):
        idx = self.calls
        self.calls += 1
        for chunk in self.script[min(idx, len(self.script) - 1)]:
            yield chunk


class _SubAgents:
    """What SubAgentManager answers: the first send to an idle worker goes
    out, a send to a worker already running queues behind it."""

    def __init__(self):
        self.sent: list[tuple[str, str]] = []
        self.fanouts: list[list] = []
        self._busy: set[str] = set()

    async def send(self, project_id, agent, message, *, session_id=None,
                   depth=1, fresh=False):
        self.sent.append((agent, message))
        if agent in self._busy:
            return f"Message queued for {agent} (position 1). Transcript: /t/{agent}"
        self._busy.add(agent)
        return f"Message sent to {agent}. Transcript: /t/{agent}"

    async def dispatch_fanout(self, project_id, tasks, *, session_id=None,
                              max_runtime_s=None, depth=1):
        self.fanouts.append(tasks)
        return f"Fanout f1 dispatched: {len(tasks)} tasks."

    def list_active(self, project_id, session_id=None):
        return [{"name": a, "status": "running"} for a in self._busy]


class _Registry:
    """The real dispatch tools plus a plain ``read`` and a ``write_file``
    that the interceptor below holds for approval."""

    def __init__(self, sub_agents):
        self.tools = {
            "agent_message": AgentMessageTool(sub_agent_manager=sub_agents,
                                              project_id=PROJECT, session_id=SID),
            "fanout": FanoutTool(sub_agent_manager=sub_agents,
                                 project_id=PROJECT, session_id=SID),
        }
        self.ran: list[str] = []

    def schemas(self):
        return [{"type": "function", "function": {
            "name": n, "description": n,
            "parameters": {"type": "object", "properties": {}}}}
            for n in ("agent_message", "fanout", "read", "write_file")]

    def tool_names(self):
        return ["agent_message", "fanout", "read", "write_file"]

    def is_async(self, name):
        return name in self.tools

    async def execute_async(self, name, arguments):
        self.ran.append(name)
        return await self.tools[name].execute(**arguments)

    def execute(self, name, arguments):
        self.ran.append(name)
        return ToolResult(content=f"{name} ok")

    def reset_run_state(self):
        for tool in self.tools.values():
            reset = getattr(tool, "on_run_start", None)
            if reset:
                reset()


class _Interceptor:
    """Holds ``write_file`` for approval, like the real one under check-in."""

    def __init__(self):
        self.intercepted: list[str] = []

    def should_intercept(self, tc):
        return tc["name"] == "write_file"

    def on_intercept(self, tc, recent, reasoning=None):
        self.intercepted.append(tc["id"])


def _send(agent, message="Reply with the single word PONG"):
    return ("agent_message", {"action": "send", "agent": agent, "message": message})


def _fanout():
    return ("fanout", {"tasks": [{"label": "a", "brief": "do a"},
                                 {"label": "b", "brief": "do b"}]})


def _make(tmp_path, script, *, interceptor=None):
    session = Session.new(SID, str(tmp_path))
    sub_agents = _SubAgents()
    registry = _Registry(sub_agents)
    llm = _LLM(script)
    ctx = ContextManager(session, _Builder(), _ctx(str(tmp_path)))
    loop = AgentLoop(session, llm, registry, ctx, interceptor=interceptor,
                     project_dir=str(tmp_path), max_iterations=10,
                     memory_record_enabled=False)
    persist_user_row(session, "ask both of them")
    return session, loop, llm, registry, sub_agents


def _tool_rows(session):
    return {m["tool_call_id"]: m["content"] for m in session.get_messages()
            if m.get("role") == "tool"}


def _no_cancelled(rows):
    assert not [c for c in rows.values() if "CANCELLED" in c], rows


@pytest.mark.asyncio
async def test_two_sends_to_different_agents_both_deliver_and_the_turn_ends_once(tmp_path):
    session, loop, llm, _, subs = _make(
        tmp_path, [_batch(_send("codex"), _send("dsh")), _reply("never asked")])
    await loop.run()
    assert subs.sent == [("codex", "Reply with the single word PONG"),
                         ("dsh", "Reply with the single word PONG")]
    rows = _tool_rows(session)
    _no_cancelled(rows)
    assert rows["call_0"].startswith("Dispatched to codex")
    assert rows["call_1"].startswith("Dispatched to dsh")
    assert llm.calls == 1                       # no busy-poll round after the batch
    assert loop._loop_exit_path == "yield_turn"  # the dispatcher's hold label
    assert not session.pending_tool_calls


@pytest.mark.asyncio
async def test_two_sends_to_the_same_agent_both_deliver_the_second_queues(tmp_path):
    session, loop, llm, _, subs = _make(
        tmp_path, [_batch(_send("codex", "first"), _send("codex", "second")),
                   _reply("never asked")])
    await loop.run()
    assert subs.sent == [("codex", "first"), ("codex", "second")]
    rows = _tool_rows(session)
    _no_cancelled(rows)
    assert "Message sent to codex" in rows["call_0"]
    assert "Message queued for codex" in rows["call_1"]
    assert llm.calls == 1
    assert loop._loop_exit_path == "yield_turn"


@pytest.mark.asyncio
@pytest.mark.parametrize("order", ["send_first", "read_first"])
async def test_a_send_and_an_ordinary_tool_both_run_then_the_turn_ends(tmp_path, order):
    calls = ([_send("codex"), ("read", {"path": "a.txt"})] if order == "send_first"
             else [("read", {"path": "a.txt"}), _send("codex")])
    session, loop, llm, registry, subs = _make(
        tmp_path, [_batch(*calls), _reply("never asked")])
    await loop.run()
    assert sorted(registry.ran) == ["agent_message", "read"]
    assert subs.sent == [("codex", "Reply with the single word PONG")]
    _no_cancelled(_tool_rows(session))
    assert llm.calls == 1
    assert loop._loop_exit_path == "yield_turn"


@pytest.mark.asyncio
async def test_a_send_then_a_tool_needing_approval_delivers_and_raises_the_card(tmp_path):
    """The send runs; the next call needs approval, so the turn pauses on its
    card exactly as a batch without a dispatch would (the approval path then
    owns the rest). Before, the approval call was CANCELLED unasked."""
    interceptor = _Interceptor()
    session, loop, llm, registry, subs = _make(
        tmp_path, [_batch(_send("codex"), ("write_file", {"path": "x"})),
                   _reply("never asked")], interceptor=interceptor)
    await loop.run()
    assert subs.sent == [("codex", "Reply with the single word PONG")]
    assert interceptor.intercepted == ["call_1"]
    assert "write_file" not in registry.ran     # held, not run
    rows = _tool_rows(session)
    assert rows["call_0"].startswith("Dispatched to codex")
    assert "call_1" not in rows                  # pending its approval, not cancelled
    assert loop._loop_exit_path == "approval_pause"
    assert llm.calls == 1


@pytest.mark.asyncio
async def test_a_tool_needing_approval_then_a_send_is_held_as_before(tmp_path):
    """Unchanged: an approval earlier in the batch pauses it, and the later
    send is told it will be retried after approval (it never went out)."""
    interceptor = _Interceptor()
    session, loop, llm, _, subs = _make(
        tmp_path, [_batch(("write_file", {"path": "x"}), _send("codex")),
                   _reply("never asked")], interceptor=interceptor)
    await loop.run()
    assert subs.sent == []
    assert "retried after approval" in _tool_rows(session)["call_1"]
    assert loop._loop_exit_path == "approval_pause"


@pytest.mark.asyncio
async def test_fanout_and_a_send_both_dispatch(tmp_path):
    session, loop, llm, _, subs = _make(
        tmp_path, [_batch(_fanout(), _send("codex")), _reply("never asked")])
    await loop.run()
    assert len(subs.fanouts) == 1
    assert subs.sent == [("codex", "Reply with the single word PONG")]
    rows = _tool_rows(session)
    _no_cancelled(rows)
    assert rows["call_0"].startswith("Fanout f1 dispatched")
    assert llm.calls == 1
    assert loop._loop_exit_path == "yield_turn"


@pytest.mark.asyncio
async def test_a_failed_send_beside_a_good_one_does_not_keep_the_turn_open_alone(tmp_path):
    """A failed dispatch never yields on its own; beside a delivered one the
    turn still ends after the batch (the manager reads the error when the
    delivered worker's completion wakes it)."""
    session, loop, llm, _, subs = _make(
        tmp_path, [_batch(_send("codex"), ("agent_message", {"action": "send"})),
                   _reply("never asked")])
    await loop.run()
    rows = _tool_rows(session)
    _no_cancelled(rows)
    assert rows["call_1"].startswith("Error: 'agent' parameter is required")
    assert llm.calls == 1
    assert loop._loop_exit_path == "yield_turn"


# ---------------------------------------------------------------------------
# Each completion wakes the manager (the existing push path, unchanged)
# ---------------------------------------------------------------------------


@pytest.fixture
def manager():
    sub_agent_manager = MagicMock()
    sub_agent_manager.list_active = MagicMock(return_value=[])
    ws = MagicMock()
    ws.broadcast = MagicMock()
    return AgentManager(project_store=MagicMock(), ws_manager=ws,
                        sub_agent_manager=sub_agent_manager,
                        activity_translator=MagicMock(),
                        process_manager=MagicMock())


@pytest.mark.asyncio
async def test_each_workers_completion_wakes_the_manager_after_the_batch(manager, tmp_path):
    session, loop, llm, registry, subs = _make(
        tmp_path, [_batch(_send("codex"), _send("dsh")), _reply("never asked")])
    handle = ProjectHandle(session=session, loop=loop, provider=llm,
                           registry=registry,
                           context_manager=loop._context_manager,
                           interceptor=None, task=None)
    manager._handles[make_session_key(PROJECT, SID)] = handle
    wake = AsyncMock()
    manager._start_loop = wake

    task = asyncio.create_task(loop.run())
    task.add_done_callback(manager._on_loop_done(PROJECT, session_id=SID))
    handle.task = task
    await task
    for _ in range(5):
        await asyncio.sleep(0)
    wake.assert_not_called()                     # the turn just ended; nothing new yet
    assert llm.calls == 1

    meta = {"event": "sub_agent_terminal", "kind": "completed"}
    for agent in ("codex", "dsh"):
        await manager.inject_system_message(
            PROJECT, f"[Sub-agent] {agent} completed. Summary: PONG",
            session_id=SID, meta=meta)
    assert wake.await_count == 2
    rows = [m["content"] for m in session.get_messages() if m.get("role") == "system"]
    assert "[Sub-agent] codex completed. Summary: PONG" in rows
    assert "[Sub-agent] dsh completed. Summary: PONG" in rows
