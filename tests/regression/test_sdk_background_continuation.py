# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Regression: SDK sub-agent background-task continuation must be surfaced.

Bug: when a claude-code sub-agent (SDKTransport) kicks off an AWAITED
background task (e.g. a multi-agent "workflow"), the agent's FOREGROUND turn
ends immediately ("started it, will report when done") and the SDK emits a
``ResultMessage``. The dispatch consumer used ``receive_response()`` once,
which returns at that first ``ResultMessage`` and dies — so the CONTINUATION
turn (the task's ``TaskNotification`` → a follow-up ``AssistantMessage`` →
a second ``ResultMessage``), which carries the real result, was stranded on
the SDK channel and dropped. The management session was told "completed" with
only the preamble; the actual result never surfaced.

Spec 129 rewrote the contract: ONE reader lives for the client's life and
reads the SDK's unbounded iterator (``receive_messages()``) whether or not a
dispatch is pending. Spec 129 kept an awaited-task rule (the dispatch stayed
pending across results while a non ``local_bash`` task was outstanding, up to
``CONTINUATION_BUDGET_S`` = 1800 s). Batch 1122 retired it: a pinned user's
next message was held for 30 minutes behind Claude Code's own background
agent. Now the main agent's result closes the dispatch, the task is reported
as background work (``background_agents``), and the continuation — still
read, never dropped — ends as a self-initiated turn (``origin="self"``).

Fire-and-forget ``run_in_background`` shells (``task_type='local_bash'``) are
the BackgroundWorkRegistry's domain and are intentionally NOT awaited — the
turn ends as before so a never-exiting dev server can't pin the dispatch. The
shell's exit still reaches the reader: the CLI's own follow-up turn is read
live and ends as a self-initiated turn (``origin="self"``), no longer
stranded.
"""

from __future__ import annotations

import asyncio

import pytest

from agent_os.agent.transports.base import TransportEvent
from agent_os.agent.transports.sdk_transport import SDKTransport, HAS_SDK

pytestmark = [pytest.mark.asyncio,
              pytest.mark.skipif(not HAS_SDK, reason="claude-agent-sdk not installed")]


# ---------------------------------------------------------------------------
# SDK message builders
# ---------------------------------------------------------------------------

def _assistant(text: str):
    from claude_agent_sdk.types import AssistantMessage, TextBlock
    return AssistantMessage(content=[TextBlock(text=text)], model="claude-x",
                            parent_tool_use_id=None, error=None)


def _result(*, is_error=False, session_id="sdk-s-1"):
    from claude_agent_sdk.types import ResultMessage
    return ResultMessage(
        subtype="error" if is_error else "success",
        duration_ms=10, duration_api_ms=8, is_error=is_error, num_turns=1,
        session_id=session_id, total_cost_usd=0.0, usage={}, result=None,
        structured_output=None,
    )


def _task_started(*, task_id: str, task_type: str):
    from claude_agent_sdk.types import TaskStartedMessage
    return TaskStartedMessage(
        subtype="task_started", data={}, task_id=task_id,
        description="d", uuid="u-" + task_id, session_id="sdk-s-1",
        tool_use_id="tu-" + task_id, task_type=task_type,
    )


def _task_notification(*, task_id: str, status: str = "completed"):
    from claude_agent_sdk.types import TaskNotificationMessage
    return TaskNotificationMessage(
        subtype="task_notification", data={}, task_id=task_id, status=status,
        output_file="", summary="done", uuid="n-" + task_id,
        session_id="sdk-s-1", tool_use_id="tu-" + task_id, usage=None,
    )


class _FakeSDKClient:
    """Mimics ClaudeSDKClient over a persistent message channel.

    ``receive_messages()`` yields every scripted message in order and then
    stays open (a live CLI), exactly the SDK's single unbounded iterator.
    """

    def __init__(self, messages):
        self._messages = list(messages)
        self.release = asyncio.Event()

    async def query(self, *a, **k):
        return None

    async def receive_messages(self):
        for msg in self._messages:
            yield msg
            await asyncio.sleep(0)
        await self.release.wait()

    async def disconnect(self):
        self.release.set()


async def _dispatch_and_drain(messages, *, workspace="") -> list[TransportEvent]:
    transport = SDKTransport()
    client = _FakeSDKClient(messages)
    transport._client = client
    transport._alive = True
    transport._workspace = workspace  # real path → display-only usage ledger is quiet
    await transport.dispatch("go")
    for _ in range(50):
        await asyncio.sleep(0)
    client.release.set()
    transport._reader_task.cancel()
    await asyncio.gather(transport._reader_task, return_exceptions=True)
    events = []
    while not transport._event_queue.empty():
        events.append(transport._event_queue.get_nowait())
    return events


def _texts(events):
    return [e.raw_text for e in events if e.event_type == "message"]


def _turn_completes(events):
    return [e for e in events if e.event_type == "turn_complete"]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

async def test_awaited_workflow_continuation_is_surfaced(tmp_path):
    """A background agent/workflow task's continuation turn — the one
    carrying the real result — must be consumed and surfaced. The dispatch
    closes at the foreground result (one boundary per dispatch); the
    continuation is the worker continuing (a self turn), not a second
    dispatch boundary."""
    messages = [
        _assistant("Workflow started (wf123). Will write results when done."),
        _task_started(task_id="wf123", task_type="workflow"),
        _result(is_error=False),                       # foreground turn ends
        _task_notification(task_id="wf123", status="completed"),
        _assistant("Wrote results to agent_output/report.md"),  # continuation
        _result(is_error=False),                       # continuation turn ends
    ]

    events = await _dispatch_and_drain(messages, workspace=str(tmp_path))
    texts = _texts(events)

    # The real result (continuation turn) must be present — this is the bug.
    assert any("agent_output/report.md" in t for t in texts), (
        f"continuation result was dropped; saw messages={texts!r}")
    # The preamble is also there (foreground turn).
    assert any("Workflow started" in t for t in texts)
    tcs = _turn_completes(events)
    assert [(t.data.get("origin"), t.data.get("cause")) for t in tcs] == [
        ("dispatch", "success"), ("self", "success")]


async def test_fire_and_forget_local_bash_is_not_awaited(tmp_path):
    """A run_in_background shell (task_type='local_bash') is fire-and-forget —
    the BackgroundWorkRegistry's domain. The dispatch must NOT stay pending
    for its continuation (so a never-exiting dev server can't pin the
    dispatch): the foreground turn ends with the dispatch's one turn_complete.
    Spec 129: the continuation is still READ — the CLI runs it as a turn of
    its own when the shell exits — and ends as a self-initiated turn, never
    as a second dispatch boundary and never stranded."""
    messages = [
        _assistant("Started the dev server in the background."),
        _task_started(task_id="bg1", task_type="local_bash"),
        _result(is_error=False),                       # foreground turn ends
        # The shell exits and the CLI speaks unprompted:
        _task_notification(task_id="bg1", status="completed"),
        _assistant("server process exited"),
        _result(is_error=False),
    ]

    events = await _dispatch_and_drain(messages, workspace=str(tmp_path))
    texts = _texts(events)

    assert any("dev server" in t for t in texts)
    assert any("server process exited" in t for t in texts), (
        "the self-initiated continuation must be read, not stranded")
    tcs = _turn_completes(events)
    assert [(t.data.get("origin"), t.data.get("cause")) for t in tcs] == [
        ("dispatch", "success"), ("self", "success")]


async def test_plain_turn_unchanged_single_turn_complete(tmp_path):
    """Control: a normal turn with no background task behaves exactly as before
    — one turn, one success turn_complete."""
    messages = [_assistant("here is the answer"), _result(is_error=False)]

    events = await _dispatch_and_drain(messages, workspace=str(tmp_path))

    assert _texts(events) == ["here is the answer"]
    tcs = _turn_completes(events)
    assert len(tcs) == 1
    assert tcs[0].data.get("cause") == "success"
    assert tcs[0].data.get("origin") == "dispatch"
