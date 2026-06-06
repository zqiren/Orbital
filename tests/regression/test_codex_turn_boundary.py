# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""TEST RULE 1 (Codex): the busy/idle signal is `turn/completed` ONLY.

The PTY transport this replaces lied: it looked idle the instant output
paused. The regression locked here: a CLIAdapter consuming CodexTransport
events stays BUSY through thread/status/changed (the tempting wrong signal,
emitted at idle/active transitions) and flips idle exactly ONCE, on
turn/completed. Red against an implementation that maps status-changed to
any event that closes the turn; green with the sentinel-only rule.
"""

import asyncio

import pytest

from agent_os.agent.adapters.cli_adapter import CLIAdapter
from agent_os.agent.transports.codex_transport import CodexTransport


def _adapter_with_transport() -> tuple[CLIAdapter, CodexTransport]:
    transport = CodexTransport()
    transport._thread_id = "T1"
    transport._effective_model = "gpt-5.4-mini"
    transport._alive = True
    adapter = CLIAdapter(handle="codex", display_name="Codex", transport=transport)
    return adapter, transport


@pytest.mark.asyncio
async def test_idle_flips_once_on_turn_completed_never_on_status_changed():
    adapter, transport = _adapter_with_transport()
    adapter._idle = False  # manager does this on dispatch

    consumed = asyncio.Event()
    idle_flips: list[bool] = []

    async def consume():
        async for chunk in adapter.read_stream():
            if chunk.chunk_type == "turn_complete":
                idle_flips.append(adapter.is_idle())
                consumed.set()

    task = asyncio.create_task(consume())
    try:
        # The full noise battery BEFORE the genuine boundary — all verbatim
        # method names from the 0.125.0 traces.
        for method in ("thread/status/changed", "thread/tokenUsage/updated",
                       "account/rateLimits/updated"):
            await transport._route_server_message(
                {"jsonrpc": "2.0", "method": method, "params": {}})
        await transport._route_server_message(
            {"jsonrpc": "2.0", "method": "turn/started",
             "params": {"turn": {"id": "U1", "status": "inProgress"}}})
        await transport._route_server_message(
            {"jsonrpc": "2.0", "method": "item/completed",
             "params": {"item": {"type": "commandExecution", "id": "c1",
                                 "command": "echo hi", "status": "completed",
                                 "exitCode": 0}}})
        await asyncio.sleep(0.2)  # give the consumer time to mis-flip
        assert adapter.is_idle() is False, \
            "idle flipped before turn/completed — the PTY lie is back"

        transport._begin_turn()
        transport._turn_id = "U1"
        await transport._route_server_message(
            {"jsonrpc": "2.0", "method": "turn/completed",
             "params": {"turn": {"id": "U1", "status": "completed",
                                 "durationMs": 9841}}})
        await asyncio.wait_for(consumed.wait(), timeout=2.0)
        assert idle_flips == [True], "idle must flip exactly once, on the sentinel"
        assert adapter.is_idle() is True
    finally:
        transport._alive = False  # ends read_stream's while-alive loop
        await asyncio.wait_for(task, timeout=2.0)
