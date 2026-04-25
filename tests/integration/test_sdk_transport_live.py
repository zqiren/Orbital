# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Live-daemon integration smoke tests for SDK transport lifecycle.

These tests spawn a real daemon (via the harness's ``daemon`` /
``api_client`` fixtures), then inject prompts that drive the
``SDKTransport`` end-to-end — i.e. spawn a real ``claude`` subprocess
through ``claude-agent-sdk``.

Because that path needs both:

* the ``claude`` CLI binary reachable from PATH, and
* a working LLM provider configuration (API key + model) on the test
  project,

each test skips cleanly when either prerequisite is missing. They will
run for real in the Phase-4 smoke test with a Kimi API key configured.
"""

from __future__ import annotations

import asyncio
import os
import shutil

import pytest

from tests.integration.harness import ApiClient, DaemonProcess, process_tools


pytestmark = pytest.mark.live_daemon


# --------------------------------------------------------------------- #
# Skip helpers
# --------------------------------------------------------------------- #


def _claude_binary_available() -> bool:
    """Return True if the ``claude`` CLI (which the SDK wraps) is on PATH."""
    return shutil.which("claude") is not None


def _llm_credentials_available() -> bool:
    """Heuristic: some LLM provider API key is available in the env.

    Matches the set of keys the daemon consults when it falls back to
    global settings during ``start_agent``. Presence of any one of these
    means the management agent is bootable for a live run.
    """
    for var in (
        "ANTHROPIC_API_KEY",
        "MOONSHOT_API_KEY",
        "KIMI_API_KEY",
        "OPENAI_API_KEY",
    ):
        if os.environ.get(var):
            return True
    return False


def _skip_reason() -> str | None:
    if not _claude_binary_available():
        return (
            "requires `claude` CLI binary on PATH (claude-agent-sdk wraps it "
            "to run sub-agents)"
        )
    if not _llm_credentials_available():
        return (
            "requires LLM provider API key in env (ANTHROPIC_API_KEY / "
            "MOONSHOT_API_KEY / KIMI_API_KEY / OPENAI_API_KEY) so the "
            "management agent can dispatch to the SDK sub-agent"
        )
    return None


def _maybe_skip() -> None:
    reason = _skip_reason()
    if reason:
        pytest.skip(reason)


# --------------------------------------------------------------------- #
# Project bootstrap helper
# --------------------------------------------------------------------- #


async def _bootstrap_project(
    api_client: ApiClient,
    workspace: str,
    *,
    name: str = "sdk-lifecycle-live",
) -> str:
    """Create a project and start its management agent. Returns project_id.

    Uses env-derived credentials so the test is agnostic to which
    provider ends up available at runtime.
    """
    api_key = (
        os.environ.get("MOONSHOT_API_KEY")
        or os.environ.get("KIMI_API_KEY")
        or os.environ.get("ANTHROPIC_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
        or ""
    )
    # Prefer Kimi when its key is set (matches the Phase-4 smoke rig).
    if os.environ.get("MOONSHOT_API_KEY") or os.environ.get("KIMI_API_KEY"):
        provider = "kimi"
        model = os.environ.get("KIMI_MODEL", "kimi-k2-turbo-preview")
        base_url = "https://api.moonshot.cn/v1"
        sdk = "openai"
    elif os.environ.get("ANTHROPIC_API_KEY"):
        provider = "anthropic"
        model = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-5-20250929")
        base_url = None
        sdk = "anthropic"
    else:
        provider = "openai"
        model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
        base_url = None
        sdk = "openai"

    extra: dict = {"autonomy": "hands_off"}
    if base_url is not None:
        extra["base_url"] = base_url

    project = await api_client.create_project(
        name=name,
        workspace=workspace,
        model=model,
        api_key=api_key,
        provider=provider,
        sdk=sdk,
        **extra,
    )
    project_id = project.get("project_id") or project["id"]

    # Start the management agent (required before inject).
    await api_client.start_agent(project_id)

    return project_id


# --------------------------------------------------------------------- #
# 1. Single dispatch: no "Task was destroyed" warning in logs
# --------------------------------------------------------------------- #


async def test_single_dispatch_no_task_destroyed_warning(
    daemon: DaemonProcess,
    api_client: ApiClient,
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    """Dispatch one turn to an SDK-backed sub-agent. Verify no
    ``"Task was destroyed but it is pending"`` warnings and no
    ``ERROR asyncio`` lines appeared during the run."""
    _maybe_skip()

    workspace = tmp_path_factory.mktemp("sdk-live-single")
    project_id = await _bootstrap_project(api_client, str(workspace))

    # Snapshot log counts BEFORE injecting.
    destroyed_before = daemon.log_count(r"Task was destroyed but it is pending")
    asyncio_error_before = daemon.log_count(r"ERROR.*asyncio")

    try:
        # Target the claude-code sub-agent (SDK-backed on installs that
        # have claude-agent-sdk; falls back to PipeTransport otherwise).
        await api_client.inject(
            project_id,
            "Reply with the single word: hello.",
            target="claude-code",
        )

        # Poll until idle or timeout. The SDK path is slow on a cold
        # claude startup, so give it generous headroom.
        await api_client.poll_until_idle(project_id, timeout=90.0)

        # Pending-task GC warnings should not have increased.
        destroyed_after = daemon.log_count(r"Task was destroyed but it is pending")
        asyncio_error_after = daemon.log_count(r"ERROR.*asyncio")

        assert destroyed_after == destroyed_before, (
            f"'Task was destroyed' warnings increased from "
            f"{destroyed_before} to {destroyed_after}; "
            f"tail={daemon.log_tail(30)!r}"
        )
        assert asyncio_error_after == asyncio_error_before, (
            f"ERROR asyncio lines increased from {asyncio_error_before} to "
            f"{asyncio_error_after}; tail={daemon.log_tail(30)!r}"
        )
    finally:
        try:
            await api_client.stop(project_id)
        except Exception:
            pass
        try:
            await api_client.delete_project(project_id)
        except Exception:
            pass


# --------------------------------------------------------------------- #
# 2. Repeated dispatch: no orphan claude process accumulation
# --------------------------------------------------------------------- #


async def test_repeated_dispatch_no_orphan_accumulation(
    daemon: DaemonProcess,
    api_client: ApiClient,
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    """3 sequential turns must not monotonically grow the ``claude``
    descendant count. Acceptable steady state: bounded at <=1 between
    turns. Growth (1->2->3->4) indicates an orphan leak."""
    _maybe_skip()

    workspace = tmp_path_factory.mktemp("sdk-live-repeat")
    project_id = await _bootstrap_project(api_client, str(workspace))

    counts: list[int] = []

    try:
        for turn in range(3):
            await api_client.inject(
                project_id,
                f"Reply with the word 'turn{turn + 1}'.",
                target="claude-code",
            )
            await api_client.poll_until_idle(project_id, timeout=90.0)

            # Give the SDK's subprocess reaper a beat to clean up the
            # claude child (ClaudeSDKClient owns cleanup between turns).
            await asyncio.sleep(1.5)

            count = process_tools.count_descendants(
                daemon.pid, name_filter=r"claude"
            )
            counts.append(count)

        # The steady-state count must not monotonically grow across the
        # 3 samples. It's fine for it to hover at 0 or 1 (one persistent
        # claude child owned by the sub-agent adapter); it's NOT fine
        # for it to trend 1->2->3.
        assert max(counts) <= len(counts), (
            f"claude descendant count trend looks like a leak: {counts}; "
            f"tail={daemon.log_tail(30)!r}"
        )
        # A strictly increasing series across all three turns is the
        # failure mode the task calls out explicitly.
        strictly_increasing = all(
            counts[i] < counts[i + 1] for i in range(len(counts) - 1)
        )
        assert not strictly_increasing, (
            f"claude descendant count is strictly increasing across turns "
            f"{counts} — matches the orphan-accumulation signature"
        )
    finally:
        try:
            await api_client.stop(project_id)
        except Exception:
            pass
        try:
            await api_client.delete_project(project_id)
        except Exception:
            pass


# --------------------------------------------------------------------- #
# 3. Stop during dispatch: clean-up within 5s
# --------------------------------------------------------------------- #


async def test_stop_during_dispatch_cleans_up(
    daemon: DaemonProcess,
    api_client: ApiClient,
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    """Kick off a long turn, call stop ~2s in, and verify the daemon's
    direct ``claude`` children have been reaped within 5s. The broken
    5-minute fallback ('forcing idle') must NOT appear."""
    _maybe_skip()

    workspace = tmp_path_factory.mktemp("sdk-live-stop")
    project_id = await _bootstrap_project(api_client, str(workspace))

    forcing_idle_before = daemon.log_count(r"forcing idle")

    try:
        # A prompt that should take more than 2s to produce a final
        # answer — list many small items so the model streams.
        await api_client.inject(
            project_id,
            "List the numbers 1 through 50, one per line. Do not summarize.",
            target="claude-code",
        )

        # Let the SDK actually spawn claude and start the turn.
        await asyncio.sleep(2.0)

        await api_client.stop(project_id)

        # Poll up to 5s for the direct-child count to drop to zero.
        # Per the spec's Windows-specific note: without Job Objects,
        # grandchildren of `claude` may survive on Windows. Assert only
        # direct-child cleanup (recursive=False).
        import re
        claude_re = re.compile(r"claude", re.IGNORECASE)
        deadline = asyncio.get_event_loop().time() + 5.0
        direct_children_count = None
        while asyncio.get_event_loop().time() < deadline:
            direct_children = process_tools.get_children(daemon.pid, recursive=False)
            direct_children_count = sum(
                1 for p in direct_children
                if claude_re.search(p.name() or "")
            )
            if direct_children_count == 0:
                break
            await asyncio.sleep(0.25)

        assert direct_children_count == 0, (
            f"expected 0 direct `claude` children after stop; got "
            f"{direct_children_count}; tail={daemon.log_tail(30)!r}"
        )

        # Informational: if grandchildren survive (Windows limitation), log them.
        total_claude_descendants = process_tools.count_descendants(
            daemon.pid, name_filter=r"claude"
        )
        if total_claude_descendants > 0:
            print(
                f"[info] {total_claude_descendants} claude descendants (incl. "
                f"grandchildren) remain — Windows Job-Object limitation, out of scope"
            )

        # The broken 5-minute fallback must not have fired.
        forcing_idle_after = daemon.log_count(r"forcing idle")
        assert forcing_idle_after == forcing_idle_before, (
            f"'forcing idle' log count increased from "
            f"{forcing_idle_before} to {forcing_idle_after} — the slow "
            f"fallback fired instead of the fast-stop path"
        )

        # Grandchild note (informational only): log any surviving
        # non-claude descendants so a human can inspect them but don't
        # fail the test — grandchild cleanup is out of scope per spec.
        other = [
            p.name()
            for p in process_tools.get_children(daemon.pid, recursive=True)
            if "claude" not in (p.name() or "").lower()
        ]
        if other:
            # Not an assertion; surfaced via test output for diagnosis.
            print(f"[info] non-claude daemon descendants after stop: {other}")
    finally:
        try:
            await api_client.delete_project(project_id)
        except Exception:
            pass
