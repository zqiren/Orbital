# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Regression tests for PipeTransport.is_alive() fix.

PipeTransport.is_alive() previously returned True unconditionally, causing
the idle poll (_check_sub_agents_done) to loop for up to 5 minutes before
force-transitioning to idle.  The fix tracks a _running flag that is True
only while send() is executing a subprocess, so is_alive() returns False
before and after send().
"""

import asyncio
import sys
import threading
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent_os.agent.transports.pipe_transport import PipeTransport, PipeTransportConfig


# ---------------------------------------------------------------------------
# Test 1: is_alive is False before any send()
# ---------------------------------------------------------------------------

def test_is_alive_false_before_send():
    """PipeTransport.is_alive() must return False before send() is called."""
    transport = PipeTransport()
    assert transport.is_alive() is False, (
        "is_alive() should be False before any send() call"
    )


# ---------------------------------------------------------------------------
# Test 2: is_alive is False after send() completes
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_is_alive_false_after_send():
    """After send() returns, the subprocess is dead — is_alive() must be False."""
    transport = PipeTransport()
    await transport.start(
        command=sys.executable,
        args=["-c", "print('hello')"],
        workspace=".",
    )

    result = await transport.send("test")
    assert "hello" in result
    assert transport.is_alive() is False, (
        "is_alive() should be False after send() completes"
    )


# ---------------------------------------------------------------------------
# Test 3: is_alive is True during send() execution
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_is_alive_true_during_send():
    """While send() is running a subprocess, is_alive() must return True."""
    transport = PipeTransport()
    # Use a command that sleeps briefly so we can observe _running=True
    await transport.start(
        command=sys.executable,
        args=["-c", "import time; time.sleep(2); print('done')"],
        workspace=".",
    )

    observed_alive = []

    async def observe():
        """Poll is_alive() while send() is running."""
        for _ in range(40):
            await asyncio.sleep(0.05)
            observed_alive.append(transport.is_alive())

    # Run send() and observer concurrently
    send_task = asyncio.create_task(transport.send("test"))
    observe_task = asyncio.create_task(observe())

    await asyncio.gather(send_task, observe_task)

    # During the ~2-second sleep, observer should have seen is_alive() == True
    assert any(observed_alive), (
        "is_alive() should have been True at least once during send()"
    )
    # After send() returns, is_alive() should be False
    assert transport.is_alive() is False


# ---------------------------------------------------------------------------
# Test 4: is_alive False after send() timeout
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_is_alive_false_after_timeout():
    """is_alive() must be False even if send() raises TimeoutExpired."""
    transport = PipeTransport()
    await transport.start(
        command=sys.executable,
        args=["-c", "import time; time.sleep(999)"],
        workspace=".",
    )

    # Patch subprocess.run to raise TimeoutExpired
    import subprocess
    with patch("agent_os.agent.transports.pipe_transport.subprocess.run",
               side_effect=subprocess.TimeoutExpired(cmd="test", timeout=1)):
        result = await transport.send("test")

    assert "timed out" in result
    assert transport.is_alive() is False, (
        "is_alive() should be False after a timeout"
    )


# ---------------------------------------------------------------------------
# Test 5: idle poll detects pipe completion immediately
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_idle_poll_detects_pipe_completion():
    """Simulates the _on_loop_done / _check_sub_agents_done path.

    When PipeTransport.is_alive() returns False (send completed), the
    sub_agent_manager.list_active() should return an empty list (no alive
    adapters), so _on_loop_done broadcasts 'idle' immediately instead of
    entering the polling loop.
    """
    # Create a PipeTransport that has already finished (is_alive == False)
    transport = PipeTransport()
    assert transport.is_alive() is False

    # Build a mock CLIAdapter that delegates is_alive to our transport
    adapter = MagicMock()
    adapter.is_alive = MagicMock(side_effect=lambda: transport.is_alive())
    adapter.is_idle = MagicMock(return_value=True)

    # Simulate SubAgentManager.list_active logic
    adapters = {"claude-code": adapter}
    result = []
    for handle, a in adapters.items():
        if a.is_alive():
            result.append({
                "handle": handle,
                "display_name": handle,
                "status": "running" if not a.is_idle() else "idle",
            })

    # With the fix, list_active returns empty → no busy agents → idle broadcast
    assert len(result) == 0, (
        "list_active should return empty list when transport.is_alive() is False"
    )

    # Verify the _on_loop_done check: no busy agents means no polling needed
    busy = [a for a in result if a.get("status") != "idle"]
    assert len(busy) == 0, (
        "No busy sub-agents should exist, so idle should broadcast immediately"
    )


# ---------------------------------------------------------------------------
# Test 6: is_alive True during send, then transitions correctly
# ---------------------------------------------------------------------------

def test_running_flag_lifecycle():
    """Direct unit test of _running flag transitions."""
    transport = PipeTransport()

    # Initially False
    assert transport._running is False
    assert transport.is_alive() is False

    # Simulate entering send
    transport._running = True
    assert transport.is_alive() is True

    # Simulate exiting send
    transport._running = False
    assert transport.is_alive() is False
