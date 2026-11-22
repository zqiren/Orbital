# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Claude Code's own background agents (the Agent tool, "Async agent launched
successfully") run as background jobs of the worker.

Observed 2026-11-21 on a pinned claude-code session: a background agent's
text was written to the chat as the worker's reply, its model overwrote the
worker's saved model, and the user's next message was held for 30 minutes
because the dispatch stayed open until every background agent finished.

Wire facts (live capture 2026-11-22, claude 2.1.330, SDK 0.2.160): every
message a background agent produces carries ``parent_tool_use_id`` = the
main agent's Agent tool call id; the main agent's own messages carry None.
A ``task_started`` (``task_type='local_agent'``) opens the job and a
``task_notification`` ends it; a shell the background agent starts arrives
as ``local_bash`` with ``data.owned_by_subagent``. The main agent's
follow-up turn after the job ends arrives with no ``query()`` from Orbital.

Contract:
- a parent-labelled message yields no events and does not touch the
  dispatch reply, the model or the context figures;
- the main agent's result closes the dispatch at once; outstanding jobs are
  separate state (``background_agents``) cleared by their notification;
- the follow-up turn is a self turn (``origin="self"``);
- the manager dispatches a new prompt immediately and reports the job as
  ``background-running`` while it runs.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent_os.agent.transports.base import TransportEvent
from agent_os.agent.transports.sdk_transport import HAS_SDK, SDKTransport

pytestmark = [pytest.mark.asyncio,
              pytest.mark.skipif(not HAS_SDK, reason="claude-agent-sdk not installed")]

AGENT_CALL = "toolu_agent_call"


# ---------------------------------------------------------------------------
# SDK message builders (shapes from the live capture)
# ---------------------------------------------------------------------------

def _assistant(*blocks, parent=None, model="claude-main", usage=None):
    from claude_agent_sdk.types import AssistantMessage
    return AssistantMessage(content=list(blocks), model=model,
                            parent_tool_use_id=parent, error=None, usage=usage)


def _text(text):
    from claude_agent_sdk.types import TextBlock
    return TextBlock(text=text)


def _tool_use(tool_id, name, tool_input):
    from claude_agent_sdk.types import ToolUseBlock
    return ToolUseBlock(id=tool_id, name=name, input=tool_input)


def _thinking(text):
    from claude_agent_sdk.types import ThinkingBlock
    return ThinkingBlock(thinking=text, signature="sig")


def _tool_result_msg(tool_use_id, content, parent=None):
    from claude_agent_sdk.types import ToolResultBlock, UserMessage
    return UserMessage(content=[ToolResultBlock(tool_use_id=tool_use_id,
                                                content=content, is_error=False)],
                       parent_tool_use_id=parent)


def _result(*, is_error=False, session_id="sdk-s-1"):
    from claude_agent_sdk.types import ResultMessage
    return ResultMessage(
        subtype="error" if is_error else "success",
        duration_ms=10, duration_api_ms=8, is_error=is_error, num_turns=1,
        session_id=session_id, total_cost_usd=0.0, usage={}, result=None,
        structured_output=None,
    )


def _task_started(task_id, task_type, *, description="Sleep then reply PONG",
                  tool_use_id=AGENT_CALL, owned_by_subagent=False):
    from claude_agent_sdk.types import TaskStartedMessage
    data = {"task_id": task_id, "task_type": task_type, "tool_use_id": tool_use_id}
    if owned_by_subagent:
        data["owned_by_subagent"] = True
    return TaskStartedMessage(
        subtype="task_started", data=data, task_id=task_id,
        description=description, uuid="u-" + task_id, session_id="sdk-s-1",
        tool_use_id=tool_use_id, task_type=task_type,
    )


def _task_notification(task_id, *, status="completed", tool_use_id=AGENT_CALL):
    from claude_agent_sdk.types import TaskNotificationMessage
    return TaskNotificationMessage(
        subtype="task_notification", data={}, task_id=task_id, status=status,
        output_file="", summary="PONG", uuid="n-" + task_id,
        session_id="sdk-s-1", tool_use_id=tool_use_id, usage=None,
    )


def _launch_turn():
    """The main agent's own turn: the Agent call, its ack, a line, result."""
    return [
        _assistant(_tool_use(AGENT_CALL, "Agent", {
            "description": "Sleep then reply PONG",
            "prompt": "Run sleep 120 then reply PONG",
            "run_in_background": True})),
        _task_started("a01", "local_agent"),
        _tool_result_msg(AGENT_CALL, "Async agent launched successfully."),
        _assistant(_text("Started a background agent; ending my turn.")),
        _result(),
    ]


def _subagent_chatter():
    """What the background agent itself streams while it runs."""
    return [
        _assistant(_tool_use("toolu_sub_bash", "Bash", {"command": "sleep 120"}),
                   parent=AGENT_CALL, model="claude-haiku-sub",
                   usage={"input_tokens": 99999, "output_tokens": 1}),
        _tool_result_msg("toolu_sub_bash", "", parent=AGENT_CALL),
        _assistant(_thinking("the sleep is done"), parent=AGENT_CALL,
                   model="claude-haiku-sub"),
        _assistant(_text("PONG"), parent=AGENT_CALL, model="claude-haiku-sub",
                   usage={"input_tokens": 99999, "output_tokens": 1}),
    ]


# ---------------------------------------------------------------------------
# Rig: a live channel, as in test_sdk_transport_persistent_reader
# ---------------------------------------------------------------------------

class _ChannelClient:
    _END = object()

    def __init__(self):
        self._chan: asyncio.Queue = asyncio.Queue()
        self.queries: list = []

    async def query(self, message, *a, **k):
        self.queries.append(message)

    def feed(self, *messages):
        for m in messages:
            self._chan.put_nowait(m)

    def end(self):
        self._chan.put_nowait(self._END)

    async def receive_messages(self):
        while True:
            item = await self._chan.get()
            if item is self._END:
                return
            yield item

    async def disconnect(self):
        pass


def _transport(workspace: str) -> tuple[SDKTransport, _ChannelClient]:
    transport = SDKTransport()
    client = _ChannelClient()
    transport._client = client
    transport._alive = True
    transport._workspace = workspace
    transport._ensure_reader()
    return transport, client


async def _settle(n: int = 30) -> None:
    for _ in range(n):
        await asyncio.sleep(0)


def _drain(transport) -> list[TransportEvent]:
    events = []
    while not transport._event_queue.empty():
        events.append(transport._event_queue.get_nowait())
    return events


def _tcs(events):
    return [(e.data["origin"], e.data["cause"]) for e in events
            if e.event_type == "turn_complete"]


async def _stop(transport):
    with patch("agent_os.agent.transports.process_kill.kill_process_tree", new=AsyncMock()):
        await transport.stop()


# ---------------------------------------------------------------------------
# A — a background agent's messages are not the worker's
# ---------------------------------------------------------------------------

async def test_parent_labelled_messages_yield_no_events():
    transport = SDKTransport()
    for msg in _subagent_chatter():
        assert transport._message_to_events(msg) == [], msg


async def test_parent_labelled_messages_leave_model_and_context_figures_alone():
    transport = SDKTransport(model=None)
    transport._message_to_events(_assistant(
        _text("main"), usage={"input_tokens": 10, "output_tokens": 2}))
    assert transport._last_model == "claude-main"
    assert transport._last_request_tokens == 12
    for msg in _subagent_chatter():
        transport._message_to_events(msg)
    assert transport._last_model == "claude-main"
    assert transport._last_request_tokens == 12


async def test_main_agents_agent_call_and_launch_ack_still_show():
    transport = SDKTransport()
    events = []
    for msg in _launch_turn()[:4]:
        events.extend(transport._message_to_events(msg))
    kinds = [(e.event_type, e.data.get("tool_name") or e.raw_text) for e in events]
    assert ("tool_use", "Agent") in kinds
    assert any(e.event_type == "tool_result"
               and "Async agent launched" in e.data["content"] for e in events)
    assert ("message", "Started a background agent; ending my turn.") in kinds


async def test_subagent_text_is_not_the_dispatch_reply(tmp_path):
    transport, client = _transport(str(tmp_path))
    try:
        waiter = asyncio.ensure_future(transport.send("go"))
        await _settle()
        client.feed(_assistant(_text("main reply")), *_subagent_chatter(), _result())
        text = await asyncio.wait_for(waiter, 2.0)
        assert text == "main reply"
        msgs = [e.raw_text for e in _drain(transport) if e.event_type == "message"]
        assert msgs == ["main reply"]
    finally:
        await _stop(transport)


# ---------------------------------------------------------------------------
# B — the dispatch closes at the main agent's result
# ---------------------------------------------------------------------------

async def test_agent_task_then_main_result_closes_the_dispatch_at_once(tmp_path):
    transport, client = _transport(str(tmp_path))
    try:
        await transport.dispatch("go")
        client.feed(*_launch_turn())
        await _settle()
        assert _tcs(_drain(transport)) == [("dispatch", "success")]
        assert transport.dispatch_pending is False
        assert transport.background_agents == ["Sleep then reply PONG"]
    finally:
        await _stop(transport)


async def test_task_notice_then_followup_is_a_self_turn_and_clears_the_state(tmp_path):
    transport, client = _transport(str(tmp_path))
    try:
        await transport.dispatch("go")
        client.feed(*_launch_turn(), *_subagent_chatter())
        await _settle()
        events = _drain(transport)
        assert _tcs(events) == [("dispatch", "success")]
        assert "PONG" not in [e.raw_text for e in events]
        assert transport.background_agents == ["Sleep then reply PONG"]

        client.feed(_task_notification("a01"),
                    _assistant(_text("The agent finished: PONG.")),
                    _result())
        await _settle()
        events = _drain(transport)
        assert [e.raw_text for e in events if e.event_type == "message"] == [
            "The agent finished: PONG."]
        assert _tcs(events) == [("self", "success")]
        assert transport.background_agents == []
        assert transport._last_model == "claude-main"
    finally:
        await _stop(transport)


async def test_a_shell_the_background_agent_starts_counts_as_its_job(tmp_path):
    """The live capture: the agent's notification can arrive while a shell it
    started still runs (the agent then resumes). That shell is the agent's
    work, so the job stays reported until it ends too."""
    transport, client = _transport(str(tmp_path))
    try:
        await transport.dispatch("go")
        client.feed(*_launch_turn(),
                    _task_started("b4", "local_bash", description="sleep 120",
                                  tool_use_id="toolu_sub_bash",
                                  owned_by_subagent=True),
                    _task_notification("a01"))
        await _settle()
        assert transport.background_agents == ["sleep 120"]
        client.feed(_task_notification("b4", tool_use_id="toolu_sub_bash"))
        await _settle()
        assert transport.background_agents == []
    finally:
        await _stop(transport)


async def test_the_main_agents_own_shell_is_not_a_background_agent(tmp_path):
    """The main agent's run_in_background shell stays the BackgroundWorkRegistry's."""
    transport, client = _transport(str(tmp_path))
    try:
        await transport.dispatch("go")
        client.feed(_task_started("bg1", "local_bash", tool_use_id="toolu_main_bash"),
                    _result())
        await _settle()
        assert _tcs(_drain(transport)) == [("dispatch", "success")]
        assert transport.background_agents == []
    finally:
        await _stop(transport)


async def test_job_state_change_pushes_status(tmp_path):
    transport, client = _transport(str(tmp_path))
    calls: list[list[str]] = []
    transport.set_status_listener(lambda: calls.append(list(transport.background_agents)))
    try:
        await transport.dispatch("go")
        client.feed(*_launch_turn())
        await _settle()
        assert ["Sleep then reply PONG"] in calls
        calls.clear()
        client.feed(_task_notification("a01"))
        await _settle()
        assert calls and calls[-1] == []
    finally:
        await _stop(transport)


async def test_stream_end_and_stop_forget_the_jobs(tmp_path):
    transport, client = _transport(str(tmp_path))
    await transport.dispatch("go")
    client.feed(*_launch_turn())
    await _settle()
    assert transport.background_agents
    client.end()  # the CLI exited: its background agents went with it
    await _settle()
    assert transport.background_agents == []
    await _stop(transport)
    assert transport.background_agents == []


# ---------------------------------------------------------------------------
# Manager: the next message goes straight in; status says background
# ---------------------------------------------------------------------------

class _WS:
    def __init__(self):
        self.events: list[dict] = []

    def broadcast(self, project_id, payload):
        self.events.append(dict(payload))

    def statuses(self):
        return [e for e in self.events if e.get("type") == "sub_agent.status"]


P, S, H = "proj_bg", "proj_bg_sess0001", "claude-code"


async def _until(pred, timeout=3.0):
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while not pred():
        if loop.time() > deadline:
            raise AssertionError("condition not reached")
        await asyncio.sleep(0.01)


async def test_manager_sends_at_once_and_reports_the_background_job(tmp_path):
    from agent_os.agent.adapters.cli_adapter import CLIAdapter
    from agent_os.daemon_v2.process_manager import ProcessManager
    from agent_os.daemon_v2.sub_agent_manager import SubAgentManager

    ws = _WS()
    pm = ProcessManager(ws, MagicMock())
    mgr = SubAgentManager(process_manager=pm)
    transport, client = _transport(str(tmp_path))
    adapter = CLIAdapter(handle=H, display_name="Claude Code", transport=transport)
    adapter._idle = True
    mgr._adapters[(P, S)] = {H: adapter}
    mgr._wire_status(adapter, P, S, H)
    await pm.start(P, H, adapter, session_id=S)
    try:
        first = await mgr.send(P, H, "start a background agent", session_id=S)
        assert first.startswith("Message sent")
        client.feed(*_launch_turn(), *_subagent_chatter())
        await _until(lambda: ws.statuses()[-1]["status"] == "background-running")
        assert mgr.status(P, H, session_id=S) == "background-running"
        assert ws.statuses()[-1]["background_commands"] == ["Sleep then reply PONG"]
        [entry] = mgr.list_active(P, session_id=S)
        assert entry["status"] == "background-running"
        assert mgr.background_commands(P, S, H) == ["Sleep then reply PONG"]

        second = await mgr.send(P, H, "say HELLO", session_id=S)
        assert second.startswith("Message sent"), second
        assert len(client.queries) == 2
        client.feed(_assistant(_text("HELLO")), _result())
        await _until(lambda: ws.statuses()[-1]["status"] == "background-running")

        client.feed(_task_notification("a01"),
                    _assistant(_text("The agent finished: PONG.")), _result())
        await _until(lambda: ws.statuses()[-1]["status"] == "idle")
        replies = [e["content"] for e in ws.events
                   if e.get("type") == "chat.sub_agent_message"]
        assert replies == ["Started a background agent; ending my turn.",
                           "HELLO", "The agent finished: PONG."]
    finally:
        await pm.stop(P, H, session_id=S)
        await _stop(transport)


async def test_worker_is_not_quiescent_while_a_background_agent_runs(tmp_path):
    from agent_os.daemon_v2.agent_manager import AgentManager

    transport, client = _transport(str(tmp_path))
    try:
        await transport.dispatch("go")
        client.feed(*_launch_turn())
        await _settle()
        am = AgentManager.__new__(AgentManager)
        sam = MagicMock()
        sam._prompt_queues = {}
        sam._adapters = {(P, S): {H: MagicMock(_transport=transport)}}
        am._sub_agent_manager = sam
        am._process_manager = MagicMock(background_work=None)
        am.has_running_children = lambda *a: False
        assert transport.dispatch_pending is False
        assert am._worker_quiescent(P, S, H) is False
        client.feed(_task_notification("a01"))
        await _settle()
        assert am._worker_quiescent(P, S, H) is True
    finally:
        await _stop(transport)
