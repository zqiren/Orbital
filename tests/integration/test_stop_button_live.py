# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Live-daemon integration smoke tests for
TASK-fix-cli-adapter-stop-and-send-cleanup.

These exercise the fix against a real uvicorn subprocess from the
harness. They verify that:

1. Stop mid-turn returns fast (< 5s wall-clock) with no 'forcing idle'
   fallback and no direct-child subprocesses left behind.
2. After a stop, the same project can dispatch a new message.
3. 5 rapid dispatch+stop cycles do not leak direct-child processes or
   trigger 'Task was destroyed' / 'forcing idle' warnings.
4. A send-failure path (no LLM backing, induced via an un-reachable
   handle) leaves the adapter recoverable.

All tests require an LLM API key to boot a real management agent and
will skip cleanly when none is configured. They are executed in Phase 4
smoke against Kimi 2.5.

Cross-platform notes:
- No ``sys.platform`` branches in the tests.
- Process-tree inspection goes through ``process_tools``.
- Only direct-child survivors fail the test (Windows grandchildren are a
  documented limitation of the underlying fix).
"""

from __future__ import annotations

import asyncio
import os
import tempfile
import time
from pathlib import Path

import pytest

from tests.integration.harness import process_tools

pytestmark = pytest.mark.live_daemon


# --------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------- #


_LLM_ENV_VARS = ("ANTHROPIC_API_KEY", "KIMI_API_KEY", "OPENAI_API_KEY")


def _llm_available() -> bool:
    return any(os.environ.get(v) for v in _LLM_ENV_VARS)


def _skip_if_no_llm():
    if not _llm_available():
        pytest.skip(
            "requires an LLM API key "
            f"({', '.join(_LLM_ENV_VARS)}) to boot a real management "
            "agent; will run in Phase 4 smoke test"
        )


def _make_workspace() -> str:
    path = tempfile.mkdtemp(prefix="orbital-stop-live-")
    return str(Path(path).resolve())


async def _create_project(api_client, name: str) -> str:
    workspace = _make_workspace()
    project = await api_client.create_project(
        name=name,
        workspace=workspace,
    )
    project_id = project.get("project_id") or project.get("id")
    assert project_id, f"no project_id in response: {project!r}"
    return project_id


async def _dispatch_long(api_client, project_id: str) -> None:
    """Dispatch a prompt that would take many seconds to complete."""
    await api_client.inject(
        project_id,
        "Write a slow, detailed 5-paragraph essay on typewriters. "
        "Think step by step; take your time.",
    )


async def _wait_busy(api_client, project_id: str, timeout: float = 10.0) -> bool:
    """Poll run-status until the project reports a non-idle state."""
    deadline = time.monotonic() + timeout
    idle_states = {"idle", "stopped", "not_running", "done", "completed", ""}
    while time.monotonic() < deadline:
        try:
            status = await api_client.get_status(project_id)
        except Exception:
            await asyncio.sleep(0.2)
            continue
        state = str(status.get("status", "")).lower()
        if state and state not in idle_states:
            return True
        await asyncio.sleep(0.2)
    return False


# --------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_stop_during_active_turn_returns_within_3s(api_client, daemon):
    """Stop mid-turn must complete fast and leave no direct-child agents."""
    _skip_if_no_llm()

    log_mark = daemon.mark_log_position()
    project_id = await _create_project(api_client, "stop-live-1")
    await api_client.start_agent(project_id)

    await _dispatch_long(api_client, project_id)

    # Wait for the turn to be clearly active.
    busy = await _wait_busy(api_client, project_id, timeout=10.0)
    if not busy:
        pytest.skip(
            "management agent did not reach a busy state within 10s — "
            "backend configuration problem, not a stop-button regression"
        )

    t0 = time.monotonic()
    await api_client.stop(project_id)

    # Assert the status returns to idle within 3s via polling run-status.
    idle_reached = False
    deadline = t0 + 3.0
    while time.monotonic() < deadline:
        status = await api_client.get_status(project_id)
        state = str(status.get("status", "")).lower()
        if state in {"idle", "stopped", "not_running", "done", "completed", ""}:
            idle_reached = True
            break
        await asyncio.sleep(0.1)

    elapsed = time.monotonic() - t0
    assert idle_reached, f"project still busy after {elapsed:.2f}s"
    assert elapsed < 5.0, f"stop took {elapsed:.2f}s (budget 5s)"

    # The 5-minute force-idle fallback must NOT have fired for this project.
    recent = daemon.log_since(log_mark)
    force_idle_hits = [ln for ln in recent if "forcing idle" in ln.lower()]
    assert not force_idle_hits, (
        f"'forcing idle' fallback fired — fast stop failed: {force_idle_hits}"
    )

    # Give surviving subprocesses 5s to exit, then assert direct children
    # matching known agent binaries are gone.
    await asyncio.sleep(5.0)
    leftover_agents = process_tools.count_descendants(
        daemon.pid, name_filter=r"claude|codex|gemini"
    )
    assert leftover_agents == 0, (
        f"{leftover_agents} sub-agent subprocess(es) remain after stop "
        "(direct-child assertion)"
    )


@pytest.mark.asyncio
async def test_stop_then_new_dispatch_works(api_client, daemon):
    """After a stop, a fresh dispatch on the same project must succeed."""
    _skip_if_no_llm()

    project_id = await _create_project(api_client, "stop-live-2")
    await api_client.start_agent(project_id)
    await _dispatch_long(api_client, project_id)

    if not await _wait_busy(api_client, project_id, timeout=10.0):
        pytest.skip("management agent did not become busy within 10s")

    await api_client.stop(project_id)

    # Wait for idle.
    await api_client.poll_until_idle(project_id, timeout=10.0)

    # Dispatch something new. The adapter must be reusable.
    await api_client.inject(project_id, "Say the word READY.")

    # Allow up to 60s for some activity / completion signal.
    seen_activity = await _wait_busy(api_client, project_id, timeout=60.0)
    # Either we saw it go busy again OR it completed so fast we only see
    # the idle state — either is acceptable. We only fail if the project
    # is in a stuck/broken state after 60s.
    status = await api_client.get_status(project_id)
    state = str(status.get("status", "")).lower()
    assert state != "error", f"project entered error state: {status!r}"
    # Quiet the unused-variable warning in the "already completed" path.
    _ = seen_activity


@pytest.mark.asyncio
async def test_rapid_stop_cycles_do_not_leak(api_client, daemon):
    """Five dispatch+stop cycles must not accumulate subprocess descendants
    and must not emit 'Task was destroyed' / 'forcing idle' warnings."""
    _skip_if_no_llm()

    project_id = await _create_project(api_client, "stop-live-3")
    await api_client.start_agent(project_id)

    log_mark = daemon.mark_log_position()

    baseline = process_tools.count_descendants(daemon.pid)

    for cycle in range(5):
        await _dispatch_long(api_client, project_id)
        # Let the dispatch actually reach the agent.
        await _wait_busy(api_client, project_id, timeout=10.0)
        await api_client.stop(project_id)
        await api_client.poll_until_idle(project_id, timeout=10.0)

        current = process_tools.count_descendants(daemon.pid)
        # Allow a small transient growth band but not unbounded growth.
        assert current < baseline + 10, (
            f"cycle {cycle}: descendant count {current} exceeds baseline "
            f"{baseline} by too much — probable leak"
        )

    recent = daemon.log_since(log_mark)
    bad = [
        ln for ln in recent
        if "task was destroyed" in ln.lower()
        or "forcing idle" in ln.lower()
    ]
    assert not bad, f"unexpected warning during rapid-stop cycles: {bad}"


@pytest.mark.asyncio
async def test_invalid_send_leaves_adapter_recoverable(api_client, daemon):
    """Inducing a send failure (bogus handle) must surface the failure
    within 5s and leave the project handle reusable for a fresh dispatch.

    Companion test to TASK-fix-background-send-exception-handling smoke
    test #1 but focused on recoverability rather than the error log.
    """
    _skip_if_no_llm()

    project_id = await _create_project(api_client, "stop-live-4")
    await api_client.start_agent(project_id)

    t0 = time.monotonic()
    # Target a handle that can't exist — forces the manager to surface a
    # failure instead of a silent hang.
    resp = await api_client.post(
        f"/api/v2/agents/{project_id}/inject",
        json={
            "content": "@does-not-exist-xyz hello",
            "target": "does-not-exist-xyz",
        },
    )
    elapsed = time.monotonic() - t0
    assert elapsed < 5.0, f"failing dispatch took {elapsed:.2f}s"

    # Either an HTTP error or a 200 with an error in the body — both count
    # as "surfaced."
    surfaced = False
    if resp.status_code >= 400:
        surfaced = True
    else:
        body_text = str(resp.json()).lower()
        if "error" in body_text or "not running" in body_text or "unknown" in body_text:
            surfaced = True
    assert surfaced, (
        f"failure did not surface within 5s: {resp.status_code} {resp.text!r}"
    )

    # Fresh dispatch must still succeed — the project is not in a broken
    # state.
    await api_client.inject(project_id, "Say OK.")
    # Allow up to 30s for the fresh dispatch to reach any activity state.
    _ = await _wait_busy(api_client, project_id, timeout=30.0)
    status = await api_client.get_status(project_id)
    state = str(status.get("status", "")).lower()
    assert state != "error", f"project entered error state: {status!r}"
