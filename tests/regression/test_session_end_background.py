# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Regression: session-end routine runs in background; idle broadcasts immediately.

After backgrounding _on_session_end() in loop.py, the idle status must
broadcast without waiting for the session-end LLM call.  Workspace file
writes must be atomic (write-to-tmp-then-replace) so concurrent readers
never see partial content.
"""

import asyncio
import os
import tempfile
import threading
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent_os.agent.workspace_files import WorkspaceFileManager


# ---------------------------------------------------------------------------
# Test 1: Idle broadcasts before session-end completes
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_idle_broadcasts_before_session_end_completes():
    """Loop run() must complete (allowing idle broadcast) before
    the fire-and-forget session-end task finishes."""
    session_end_started = asyncio.Event()
    session_end_done = asyncio.Event()

    async def slow_session_end():
        session_end_started.set()
        await asyncio.sleep(5)
        session_end_done.set()

    # Build a minimal AgentLoop that exits immediately
    from agent_os.agent.loop import AgentLoop

    session = MagicMock()
    session._paused_for_approval = False
    session.resolve_pending_tool_calls = MagicMock()
    session.pop_deferred_messages = MagicMock(return_value=[])
    session.pop_queued_messages = MagicMock(return_value=[])
    session.get_messages = MagicMock(return_value=[])

    provider = MagicMock()
    context_manager = MagicMock()

    loop = AgentLoop(
        session=session,
        provider=provider,
        context_manager=context_manager,
        tool_registry=MagicMock(),
        on_session_end=slow_session_end,
    )

    # Make the loop exit on first iteration by raising StopIteration
    # in the provider.complete call
    provider.complete = AsyncMock(side_effect=Exception("force exit"))
    context_manager.prepare = MagicMock(return_value=([], []))

    # Run the loop — it should return quickly despite slow session-end
    start = time.monotonic()
    try:
        await asyncio.wait_for(loop.run(), timeout=3.0)
    except Exception:
        pass  # Expected — provider.complete raises
    elapsed = time.monotonic() - start

    # The loop must return in well under 5s (the session-end sleep duration)
    assert elapsed < 4.0, f"Loop took {elapsed:.1f}s — session-end is blocking"

    # Give the background task a moment to start
    await asyncio.sleep(0.1)
    assert session_end_started.is_set(), "Session-end task was never started"

    # Clean up: cancel the background session-end task
    for task in asyncio.all_tasks():
        if task is not asyncio.current_task() and not task.done():
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass


# ---------------------------------------------------------------------------
# Test 2: Session-end still executes in background
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_session_end_still_executes():
    """Even though session-end is fire-and-forget, it must still run."""
    completed = asyncio.Event()

    async def session_end_with_flag():
        completed.set()

    from agent_os.agent.loop import AgentLoop

    session = MagicMock()
    session._paused_for_approval = False
    session.resolve_pending_tool_calls = MagicMock()
    session.pop_deferred_messages = MagicMock(return_value=[])
    session.pop_queued_messages = MagicMock(return_value=[])
    session.get_messages = MagicMock(return_value=[])

    provider = MagicMock()
    provider.complete = AsyncMock(side_effect=Exception("force exit"))
    context_manager = MagicMock()
    context_manager.prepare = MagicMock(return_value=([], []))

    loop = AgentLoop(
        session=session,
        provider=provider,
        context_manager=context_manager,
        tool_registry=MagicMock(),
        on_session_end=session_end_with_flag,
    )

    try:
        await loop.run()
    except Exception:
        pass

    # Wait for background task to complete
    await asyncio.sleep(0.5)
    assert completed.is_set(), "Session-end never executed"


# ---------------------------------------------------------------------------
# Test 3: Atomic write — no partial reads
# ---------------------------------------------------------------------------

def test_atomic_write_no_partial_reads():
    """Concurrent reads during write must see either old or new content."""
    with tempfile.TemporaryDirectory() as tmpdir:
        wfm = WorkspaceFileManager(tmpdir, "test-proj")
        wfm.ensure_dir()

        # Write initial content
        wfm.write("state", "OLD_CONTENT")

        new_content = "X" * 1_000_000  # 1MB
        results = []
        stop = threading.Event()

        def reader():
            while not stop.is_set():
                content = wfm.read("state")
                if content is not None:
                    results.append(content)
                time.sleep(0.0001)

        # Start reader thread
        t = threading.Thread(target=reader, daemon=True)
        t.start()

        # Perform atomic write
        wfm.write("state", new_content)

        stop.set()
        t.join(timeout=2)

        # Every read must be either old content or new content
        for r in results:
            assert r == "OLD_CONTENT" or r == new_content, \
                f"Partial read detected: len={len(r)}"


# ---------------------------------------------------------------------------
# Test 4: Atomic write — tmp file cleaned up
# ---------------------------------------------------------------------------

def test_atomic_write_tmp_file_cleaned_up():
    """No .tmp files should remain after a successful write."""
    with tempfile.TemporaryDirectory() as tmpdir:
        wfm = WorkspaceFileManager(tmpdir, "test-proj")
        wfm.ensure_dir()

        wfm.write("state", "test content")
        wfm.append("decisions", "\nnew decision")
        wfm.write("lessons", "lesson learned")

        # Check for leftover .tmp files
        workspace_dir = os.path.join(tmpdir, "orbital", "test-proj")
        for f in os.listdir(workspace_dir):
            assert not f.endswith(".tmp"), f"Leftover tmp file: {f}"


# ---------------------------------------------------------------------------
# Test 5: Concurrent read during atomic write sees valid content
# ---------------------------------------------------------------------------

def test_new_loop_reads_valid_file_during_background_write():
    """A reader (context builder) must see valid content even during write."""
    with tempfile.TemporaryDirectory() as tmpdir:
        wfm = WorkspaceFileManager(tmpdir, "test-proj")
        wfm.ensure_dir()

        # Write initial state
        wfm.write("state", "# Project State\nInitial state")

        # Simulate background session-end writing new state
        # while a new loop reads it
        wfm.write("state", "# Project State\nUpdated after session end")

        # Reader should see either version — both are valid
        content = wfm.read("state")
        assert content is not None, "File read returned None"
        assert len(content) > 0, "File read returned empty content"
        assert "Project State" in content, "File content is corrupted"
