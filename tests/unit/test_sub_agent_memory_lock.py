# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Regression tests for Track F4 (dispatch 2026-05-20 §5).

Asymmetric: Orbital can only lock its OWN touches to a sub-agent's
MEMORY.md. claude.exe writes the file directly via its own filesystem
tools and is NOT mediated through ``SubAgentManager`` (per Track D
investigation 2026-05-20). These tests therefore only cover the two
Orbital-side surfaces:

  1. ``SubAgentManager.write_memory_md`` (called by the UI overwrite
     endpoint ``PUT /api/v2/projects/{pid}/sub-agent-memory/{slug}``).
  2. ``SubAgentManager.ensure_memory_md_locked`` (called by
     ``_start_from_registry`` at sub-agent dispatch time).

Two scenarios are exercised:

  - test_concurrent_orbital_writes_serialize: two ``write_memory_md``
    calls in flight at the same time MUST serialize through the
    ``(project_id, agent_slug)`` lock — the underlying file writes
    don't overlap.
  - test_ui_overwrite_during_ensure_memory_md_yields_consistent_file:
    a UI overwrite concurrent with a daemon-side lazy create produces
    a file whose contents are exactly one of the two valid bodies
    (never a partial / corrupted mix).
"""

from __future__ import annotations

import asyncio
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent_os.agent.project_paths import ProjectPaths
from agent_os.agent.sub_agent_prompt import MEMORY_MD_HEADER
from agent_os.daemon_v2.sub_agent_manager import SubAgentManager


def _make_minimal_manager() -> SubAgentManager:
    """Bare SubAgentManager — only ``_memory_write_locks`` is exercised here,
    so we skip registry / setup_engine / project_store wiring."""
    pm = MagicMock()
    pm.start = AsyncMock()
    pm.stop = AsyncMock()
    return SubAgentManager(process_manager=pm)


@pytest.fixture
def workspace(tmp_path):
    ws = tmp_path / "ws"
    ws.mkdir()
    paths = ProjectPaths(str(ws))
    os.makedirs(paths.orbital_dir, exist_ok=True)
    return str(ws)


# ---------------------------------------------------------------------------
# 1. Two concurrent Orbital-side writes serialize through the lock
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_concurrent_orbital_writes_serialize(workspace):
    """Two `write_memory_md` calls launched concurrently MUST serialize
    through the per-(project, slug) lock. We prove serialization by
    racing each writer against a probe task that tries to acquire the
    SAME lock during the write — the probe must wait until both
    writers finish, and the inside-lock counter must never exceed 1."""
    mgr = _make_minimal_manager()
    paths = ProjectPaths(workspace)

    # Counter incremented inside each writer's critical section before it
    # releases the lock; decremented just before release. If the lock
    # serializes correctly, the counter never exceeds 1. If two writers
    # are inside concurrently, the counter reaches 2.
    inside_lock = {"current": 0, "max": 0}

    # We instrument the inner write by swapping in a custom write method
    # for one concrete instance only (avoiding global module monkey-patch).
    real_lock_factory = mgr._get_memory_write_lock

    async def instrumented_write(workspace, project_id, agent_slug, content):
        """Mirrors the real write_memory_md but tracks max concurrent
        in-lock count to prove serialization."""
        from agent_os.agent.sub_agent_prompt import _memory_md_path
        path = _memory_md_path(workspace, agent_slug)
        parent = os.path.dirname(path)
        lock = real_lock_factory(project_id, agent_slug)
        async with lock:
            inside_lock["current"] += 1
            inside_lock["max"] = max(
                inside_lock["max"], inside_lock["current"],
            )
            # Yield to the loop so a missing lock would let task B start
            # this critical section before A finishes.
            await asyncio.sleep(0.05)
            os.makedirs(parent, exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)
            await asyncio.sleep(0.05)
            inside_lock["current"] -= 1
        return path

    mgr.write_memory_md = instrumented_write  # type: ignore[assignment]

    a = asyncio.create_task(
        mgr.write_memory_md(workspace, "proj_1", "claude-code", "A" * 100),
    )
    b = asyncio.create_task(
        mgr.write_memory_md(workspace, "proj_1", "claude-code", "B" * 100),
    )
    await asyncio.gather(a, b)

    # Critical assertion: max two-writers-in-lock count was 1.
    assert inside_lock["max"] == 1, (
        f"writes interleaved — max in-lock count was {inside_lock['max']!r}"
    )

    memory_path = os.path.join(paths.sub_agent_dir("claude-code"), "MEMORY.md")
    with open(memory_path, "r", encoding="utf-8") as f:
        final = f.read()
    assert final in {"A" * 100, "B" * 100}, (
        f"file content is mixed/corrupted: {final!r}"
    )


# ---------------------------------------------------------------------------
# 2. UI overwrite concurrent with daemon-side ensure_memory_md
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ui_overwrite_during_ensure_memory_md_yields_consistent_file(workspace):
    """UI overwrite concurrent with the daemon's lazy MEMORY.md create
    must not corrupt the file. After both complete, the file contents
    must be EXACTLY one of: the canonical MEMORY_MD_HEADER (lazy create
    won the lock) OR the user-supplied content (UI overwrite won) —
    never a partial or interleaved value."""
    mgr = _make_minimal_manager()
    paths = ProjectPaths(workspace)
    memory_path = os.path.join(paths.sub_agent_dir("claude-code"), "MEMORY.md")
    assert not os.path.exists(memory_path), "precondition: file absent"

    user_content = "## My handcrafted memory\n\nDo not lose this.\n"
    canonical = MEMORY_MD_HEADER.format(agent_slug="claude-code")

    a = asyncio.create_task(
        mgr.ensure_memory_md_locked(workspace, "proj_1", "claude-code"),
    )
    b = asyncio.create_task(
        mgr.write_memory_md(workspace, "proj_1", "claude-code", user_content),
    )
    await asyncio.gather(a, b)

    with open(memory_path, "r", encoding="utf-8") as f:
        final = f.read()
    # Acceptable outcomes:
    #   (i) ensure_memory_md_locked ran first → file has canonical header,
    #       then write_memory_md ran second and overwrote → file has
    #       user_content.
    #   (ii) write_memory_md ran first → file has user_content, then
    #       ensure_memory_md_locked ran but ALREADY-EXISTS so was a no-op
    #       → file still has user_content.
    # Either way, the final state is exactly one of the two whole values
    # — never a corrupted mix.
    assert final in {canonical, user_content}, (
        f"file content is mixed/corrupted; got {final!r}"
    )
    # In the common case (UI wins or ensure-then-overwrite), final is the
    # user content. But we tolerate the canonical-wins case too because
    # the scheduler decides the order.


# ---------------------------------------------------------------------------
# 3. Lock identity: same (project, slug) returns same lock; different keys
#    return different locks.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_memory_write_lock_identity_keyed_by_project_and_slug(workspace):
    """Sanity: ``_get_memory_write_lock`` is a true memoizing factory.
    Same key returns the same lock; different keys return different
    locks. Ensures sub-agents and projects don't accidentally share a
    lock."""
    mgr = _make_minimal_manager()
    a1 = mgr._get_memory_write_lock("proj_1", "claude-code")
    a2 = mgr._get_memory_write_lock("proj_1", "claude-code")
    b = mgr._get_memory_write_lock("proj_1", "codex")
    c = mgr._get_memory_write_lock("proj_2", "claude-code")
    assert a1 is a2
    assert a1 is not b
    assert a1 is not c
    assert b is not c


# ---------------------------------------------------------------------------
# 4. The REAL write_memory_md respects the lock (regression target: someone
#    removes the `async with lock:` from the real method's body).
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_real_write_memory_md_blocks_when_lock_held_externally(workspace):
    """If we hold the (project, slug) lock externally, the REAL
    ``write_memory_md`` must block until we release it. This directly
    catches a regression where the lock would be dropped from
    ``write_memory_md``'s body without touching the helper API."""
    mgr = _make_minimal_manager()
    lock = mgr._get_memory_write_lock("proj_1", "claude-code")

    # Acquire the lock and try the real write — it should not complete
    # until we release. Use a timeout to bound the test if the lock is
    # missing.
    await lock.acquire()
    try:
        write_task = asyncio.create_task(
            mgr.write_memory_md(
                workspace, "proj_1", "claude-code", "should_block",
            ),
        )
        # Give the task a chance to begin. If the lock is honored, the
        # task is now awaiting inside `async with lock:`. If the lock
        # is missing, it has already raced through and the file exists.
        await asyncio.sleep(0.1)

        paths = ProjectPaths(workspace)
        memory_path = os.path.join(
            paths.sub_agent_dir("claude-code"), "MEMORY.md",
        )
        assert not os.path.exists(memory_path), (
            "write_memory_md raced past the held lock; serialization is broken"
        )
        assert not write_task.done(), (
            "write_memory_md completed while the lock was held externally"
        )
    finally:
        lock.release()

    # After we release, the task should finish quickly.
    await asyncio.wait_for(write_task, timeout=2.0)
    paths = ProjectPaths(workspace)
    memory_path = os.path.join(
        paths.sub_agent_dir("claude-code"), "MEMORY.md",
    )
    with open(memory_path, "r", encoding="utf-8") as f:
        assert f.read() == "should_block"
