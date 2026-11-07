# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""A task sent to a full-screen CLI has to arrive the way a person types it.

Gemini CLI never ran a dispatched task. It dropped input that came before it
had drawn, and treated text arriving with its newline as a paste, so nothing
was submitted.
"""

from __future__ import annotations

import time

import pytest

from agent_os.agent.transports import pty_transport
from agent_os.agent.transports.pty_transport import PTYTransport

pytestmark = pytest.mark.skipif(
    not pty_transport._HAS_PTY, reason="the pipe fallback has no terminal UI"
)


@pytest.fixture
def writes(monkeypatch):
    recorded: list[tuple[float, bytes]] = []
    monkeypatch.setattr(
        pty_transport.os, "write",
        lambda _fd, data: recorded.append((time.monotonic(), data)) or len(data),
    )
    monkeypatch.setattr(pty_transport, "_SUBMIT_DELAY_S", 0.01)
    return recorded


def _transport() -> PTYTransport:
    transport = PTYTransport()
    transport._master_fd = 99
    return transport


@pytest.mark.asyncio
async def test_enter_is_a_separate_keystroke(writes):
    transport = _transport()
    transport._ready = True
    await transport.send("do the task")
    assert [data for _, data in writes] == [b"do the task", b"\r"]


@pytest.mark.asyncio
async def test_first_message_waits_for_startup_output_to_go_quiet(writes, monkeypatch):
    monkeypatch.setattr(pty_transport, "_READY_QUIET_S", 0.2)
    transport = _transport()
    last_output = time.monotonic()
    transport._last_output_at = last_output
    await transport.send("do the task")
    assert writes[0][0] - last_output >= 0.2


@pytest.mark.asyncio
async def test_a_silent_cli_still_gets_the_message(writes, monkeypatch):
    monkeypatch.setattr(pty_transport, "_READY_TIMEOUT_S", 0.05)
    transport = _transport()
    await transport.send("do the task")
    assert writes[0][1] == b"do the task"


@pytest.mark.asyncio
async def test_later_messages_do_not_wait(writes, monkeypatch):
    monkeypatch.setattr(pty_transport, "_READY_TIMEOUT_S", 0.05)
    transport = _transport()
    await transport.send("first")
    writes.clear()
    started = time.monotonic()
    await transport.send("second")
    assert writes[0][0] - started < 0.05
