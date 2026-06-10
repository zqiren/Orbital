# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Review-correction regression: an interrupted Codex turn must WAKE an
awaiting management session.

The hang class: management agent dispatches Codex, yields its turn awaiting
the result; the user answers the approval with cancel ("Deny & stop"); the
turn ends `interrupted` with NO teardown (the codex process lives on). If
that routes to cause="stopped", ProcessManager emits nothing
(process_manager.py `elif cause == "stopped": pass`) and the management
session is never woken — Piece 3 Part C's silent-hang class (e4da939).

Locked here: cause="interrupted" -> LifecycleObserver.on_turn_interrupted ->
the SAME _inject -> inject_system_message wake path Part C built. And the
contrast: teardown-interrupted (transport._stopping) stays "stopped" and
silent on this channel (on_user_stopped speaks for teardowns — Part D).
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent_os.agent.adapters.cli_adapter import CLIAdapter
from agent_os.agent.transports.codex_transport import CodexTransport
from agent_os.daemon_v2.lifecycle_observer import LifecycleObserver
from agent_os.daemon_v2.process_manager import ProcessManager

PROJ, SID, HANDLE = "proj_wake", "sess_wake_0001", "codex"


def _stack():
    """Real ProcessManager + real LifecycleObserver; the wake terminus
    (AgentManager.inject_system_message — Part C's hydrate-or-append) is the
    one mock, so the assertion is on the actual wake call."""
    agent_manager = MagicMock()
    agent_manager.inject_system_message = AsyncMock(return_value="delivered")
    observer = LifecycleObserver(agent_manager=agent_manager, ws_manager=MagicMock())
    pm = ProcessManager(MagicMock(), MagicMock(), lifecycle_observer=observer)
    transport = CodexTransport()
    transport._thread_id = "T1"
    transport._effective_model = "gpt-5.4-mini"
    transport._alive = True
    adapter = CLIAdapter(handle=HANDLE, display_name="Codex", transport=transport)
    adapter._idle = False  # a dispatch is in flight — the awaiting state
    return agent_manager, pm, adapter, transport


async def _await_until(predicate, *, timeout: float = 2.0,
                       interval: float = 0.01) -> bool:
    """Poll ``predicate`` until true or ``timeout`` elapses. Returns the final
    truthiness so the caller decides what an exhausted deadline means (a wake
    that must arrive vs. a silence that must hold)."""
    loop = asyncio.get_event_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        if predicate():
            return True
        await asyncio.sleep(interval)
    return bool(predicate())


async def _run_turn_end(pm, adapter, transport, *, stopping: bool):
    """Drive a single interrupted turn through the real PM consumer.

    Deterministic drain: after routing turn/completed the consumer sets
    ``_turn_open[key] = False`` (process_manager.py ~line 164) for BOTH the
    wake (cause="interrupted") and the silent (cause="stopped") paths, only
    AFTER it has processed the turn_complete chunk. Waiting on that flag is a
    presence-free signal the event was consumed — no sleep that races the
    0.5s read_stream poll. We cannot set ``_alive=False`` before this drain:
    ``read_stream``'s ``while self._alive`` re-checks before the next
    ``queue.get()``, so an early flip strands the already-queued event and the
    consumer never reaches the lifecycle branch (verified: 50/50 stranded)."""
    key = ProcessManager._key(PROJ, SID, HANDLE)
    transcript = MagicMock()
    transcript.filepath = "/tmp/codex-transcript.jsonl"
    await pm.start(PROJ, HANDLE, adapter, transcript=transcript, session_id=SID)
    try:
        transport._begin_turn()
        transport._stopping = stopping
        await transport._route_server_message(
            {"jsonrpc": "2.0", "method": "turn/completed",
             "params": {"turn": {"id": "U1", "status": "interrupted"}}})
        # Block until the consumer has provably drained the turn_complete
        # chunk (turn flag flips closed), then end the stream cleanly.
        drained = await _await_until(lambda: pm._turn_open.get(key) is False)
        assert drained, "consumer never processed the turn_complete chunk"
        transport._alive = False
    finally:
        await pm.stop(PROJ, HANDLE, session_id=SID)


@pytest.mark.asyncio
async def test_interrupted_while_alive_wakes_awaiting_management_session():
    agent_manager, pm, adapter, transport = _stack()
    await _run_turn_end(pm, adapter, transport, stopping=False)
    agent_manager.inject_system_message.assert_awaited()
    args, kwargs = agent_manager.inject_system_message.await_args
    content = args[1] if len(args) > 1 else kwargs.get("content", "")
    assert "stopped before completing" in content, \
        "the awaiting session must get an honest stopped notice"
    assert "completed. Summary" not in content, "must not fake a completion"
    assert "stopped with error" not in content, "an interruption is not an error"
    assert kwargs.get("session_id") == SID, "wake must land in the awaiting session"
    assert adapter.is_idle() is True, "the turn boundary still flips idle"


@pytest.mark.asyncio
async def test_teardown_interrupted_stays_silent_on_this_channel():
    # stop_for_user's on_user_stopped speaks for teardowns (Part D);
    # a second notice here would double-report.
    agent_manager, pm, adapter, transport = _stack()
    await _run_turn_end(pm, adapter, transport, stopping=True)
    agent_manager.inject_system_message.assert_not_awaited()
