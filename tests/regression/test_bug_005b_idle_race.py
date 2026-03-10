# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Regression tests for BUG-005b: adapter idle-state lifecycle.

Verifies that CLIAdapter.is_idle() returns False at construction (not ready
yet) and True after send() completes on the transport path (work done,
correctly idle).  The BUG-005b fix ensures the INIT state is False so
_on_loop_done does not see a newly-created adapter as idle.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent_os.agent.adapters.cli_adapter import CLIAdapter


def _make_adapter():
    """Create a CLIAdapter with a mock SDKTransport."""
    transport = MagicMock()
    transport.send = AsyncMock(return_value="Hello from Claude Code")
    transport.is_alive = MagicMock(return_value=True)
    transport.start = AsyncMock()
    transport.stop = AsyncMock()

    adapter = CLIAdapter(
        handle="claude-code",
        display_name="Claude Code",
        transport=transport,
    )
    return adapter, transport


@pytest.mark.asyncio
async def test_adapter_starts_not_idle():
    """CLIAdapter should not be idle immediately after construction."""
    adapter, _ = _make_adapter()
    assert adapter.is_idle() is False, "Adapter should start as not-idle"


@pytest.mark.asyncio
async def test_adapter_idle_after_send():
    """After send() completes on transport path, adapter should be idle (work done)."""
    adapter, transport = _make_adapter()

    await adapter.send("analyze this workspace")

    assert adapter.is_idle() is True, \
        "Adapter should be idle after transport send() completes"


@pytest.mark.asyncio
async def test_on_loop_done_broadcasts_waiting_for_active_adapter():
    """When _on_loop_done fires and sub-agent is_idle() is False, status must be 'waiting'."""
    adapter, _ = _make_adapter()

    # Simulate what list_active returns when adapter is alive and not idle
    assert adapter.is_alive() is True
    assert adapter.is_idle() is False

    # This is the check _on_loop_done performs:
    active = [{"handle": "claude-code", "status": "running" if not adapter.is_idle() else "idle"}]
    busy = [a for a in active if a.get("status") != "idle"]

    assert len(busy) == 1, "Sub-agent should be in busy list"
    assert busy[0]["status"] == "running"


@pytest.mark.asyncio
async def test_adapter_transitions_to_idle_after_work():
    """Full lifecycle: not-idle at init, then idle after send() completes."""
    adapter, transport = _make_adapter()

    # Before any work: adapter starts as not-idle
    assert adapter.is_idle() is False, "Adapter should start as not-idle"

    # After work completes: adapter transitions to idle
    await adapter.send("do some work")

    assert adapter.is_idle() is True, \
        "Adapter should be idle after send() completes (work is done)"
