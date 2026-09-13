# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Spec 085 §4.3 / §5.1 — the MAIN browser's lazy-launch and crash-recovery
path must be single-entry, mirroring the worker path's ``_worker_lock``.

All Playwright interactions are mocked. The first test replays the exact
interleaving from the spec: two projects hit ``ensure_browser()`` for the
first time at once; on the unlocked code the loser's ``self._context = None``
lands AFTER the winner assigned its context, the loser's launch then fails on
the profile lock, and the manager is left wedged (``_context`` None while
``_browser`` is still connected) so every later call fails too.
"""

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, PropertyMock, patch

import pytest

from agent_os.daemon_v2.browser_manager import BrowserManager
from tests.unit.test_browser_manager import (
    _make_mock_context,
    _make_mock_page,
    _make_mock_playwright,
)


def _manager(tmp_path) -> BrowserManager:
    return BrowserManager(profile_dir=str(tmp_path / "profile"), headless=True)


@pytest.mark.asyncio
async def test_concurrent_first_use_launches_once_and_never_wedges(tmp_path):
    """Two projects' first ensure_browser() overlap → exactly one launch,
    both callers get the same live context, and a third call still works."""
    mgr = _manager(tmp_path)
    ctx_a = _make_mock_context(pages=[_make_mock_page()])
    pw = _make_mock_playwright(ctx_a)

    launches = 0

    async def launch_persistent_context(**kw):
        nonlocal launches
        launches += 1
        if launches > 1:
            # Chrome's ProcessSingleton: the second instance exits cleanly
            # while the incumbent keeps the profile lock.
            raise RuntimeError("SingletonLock held")
        await asyncio.sleep(0)
        return ctx_a

    pw.chromium.launch_persistent_context = AsyncMock(
        side_effect=launch_persistent_context
    )

    # The clean-UA probe is the long await that sits BEFORE the old
    # ``self._context = None`` — give the second entrant a slower probe so
    # its reset would land after the first entrant already assigned.
    ua_calls = 0

    async def slow_second_probe():
        nonlocal ua_calls
        ua_calls += 1
        await asyncio.sleep(0.01 if ua_calls == 2 else 0)
        return "Mozilla/5.0 Chrome"

    mgr._get_clean_user_agent = slow_second_probe

    with patch("agent_os.daemon_v2.browser_manager.async_playwright") as mock_ap:
        mock_ap.return_value.start = AsyncMock(return_value=pw)
        results = await asyncio.gather(
            mgr.ensure_browser(), mgr.ensure_browser(), return_exceptions=True
        )

    assert results == [ctx_a, ctx_a], results
    assert launches == 1
    assert mgr._context is ctx_a
    assert mgr._browser is ctx_a.browser

    # Not wedged: a later caller reuses the live context, no relaunch.
    assert await mgr.ensure_browser() is ctx_a
    assert launches == 1


@pytest.mark.asyncio
async def test_launch_failure_leaves_no_partial_state(tmp_path):
    """A failed launch must not leave ``_browser`` set with ``_context`` None."""
    mgr = _manager(tmp_path)
    pw = _make_mock_playwright(None)
    pw.chromium.launch_persistent_context = AsyncMock(
        side_effect=RuntimeError("no browser")
    )
    mgr._get_clean_user_agent = AsyncMock(return_value=None)

    with patch("agent_os.daemon_v2.browser_manager.async_playwright") as mock_ap:
        mock_ap.return_value.start = AsyncMock(return_value=pw)
        with pytest.raises(RuntimeError):
            await mgr.ensure_browser()

    assert mgr._context is None
    assert mgr._browser is None


@pytest.mark.asyncio
async def test_concurrent_crash_recovery_records_one_restart(tmp_path):
    """Two sessions noticing the same crash together → one recovery, one
    restart timestamp (not two, which would trip the 3-in-5-min limiter on
    the very next real crash)."""
    mgr = _manager(tmp_path)
    ctx_old = _make_mock_context()
    ctx_new = _make_mock_context()
    pw = _make_mock_playwright(ctx_old)

    with patch("agent_os.daemon_v2.browser_manager.async_playwright") as mock_ap:
        mock_ap.return_value.start = AsyncMock(return_value=pw)
        await mgr.ensure_browser()
    assert mgr._context is ctx_old

    mgr._browser.is_connected.return_value = False

    async def relaunch(**kw):
        await asyncio.sleep(0)
        return ctx_new

    pw.chromium.launch_persistent_context = AsyncMock(side_effect=relaunch)
    mgr._get_clean_user_agent = AsyncMock(return_value=None)

    results = await asyncio.gather(mgr.ensure_browser(), mgr.ensure_browser())

    assert results == [ctx_new, ctx_new]
    assert len(mgr._restart_timestamps) == 1
    pw.chromium.launch_persistent_context.assert_awaited_once()

    # Direct double entry into _handle_crash is idempotent too.
    mgr._browser.is_connected.return_value = False
    ctx_third = _make_mock_context()
    pw.chromium.launch_persistent_context = AsyncMock(side_effect=_launch_returning(ctx_third))
    await asyncio.gather(mgr._handle_crash(), mgr._handle_crash())
    assert mgr._context is ctx_third
    assert len(mgr._restart_timestamps) == 2
    pw.chromium.launch_persistent_context.assert_awaited_once()


def _launch_returning(value):
    """AsyncMock side_effect that yields to the loop once, then returns value."""
    async def _launch(**kw):
        await asyncio.sleep(0)
        return value
    return _launch


@pytest.mark.asyncio
async def test_concurrent_stale_relaunch_is_single_entry(tmp_path):
    """Sleep/wake: two sessions see the stale CDP connection at once → the
    old context is closed once and one relaunch serves both."""
    mgr = _manager(tmp_path)
    ctx_old = _make_mock_context(pages=[_make_mock_page()])
    pw = _make_mock_playwright(ctx_old)

    with patch("agent_os.daemon_v2.browser_manager.async_playwright") as mock_ap:
        mock_ap.return_value.start = AsyncMock(return_value=pw)
        await mgr.ensure_browser()

    type(mgr._context).pages = PropertyMock(side_effect=Exception("CDP disconnected"))

    ctx_new = _make_mock_context(pages=[_make_mock_page()])
    pw2 = _make_mock_playwright(ctx_new)
    pw2.chromium.launch_persistent_context = AsyncMock(
        side_effect=_launch_returning(ctx_new)
    )
    mgr._get_clean_user_agent = AsyncMock(return_value=None)

    with patch("agent_os.daemon_v2.browser_manager.async_playwright") as mock_ap:
        mock_ap.return_value.start = AsyncMock(return_value=pw2)
        results = await asyncio.gather(mgr.ensure_browser(), mgr.ensure_browser())

    assert results == [ctx_new, ctx_new]
    ctx_old.close.assert_awaited_once()
    pw2.chromium.launch_persistent_context.assert_awaited_once()


@pytest.mark.asyncio
async def test_crash_cleanup_keeps_worker_bookkeeping(tmp_path):
    """§5.3: crash/stale cleanup is about the DEAD main browser — worker
    contexts are alive and their pages, state and choosers must survive."""
    mgr = _manager(tmp_path)
    main_page, worker_page = _make_mock_page(), _make_mock_page()
    mgr._project_pages = {"proj": [main_page], "worker:f1-0": [worker_page]}
    mgr._page_state = {id(main_page): MagicMock(), id(worker_page): MagicMock()}
    mgr._pending_file_choosers = {"proj": object(), "worker:f1-0": object()}
    mgr._context = AsyncMock()
    mgr._browser = MagicMock()
    mgr._playwright = MagicMock()
    mgr._playwright.stop = AsyncMock()

    await mgr._cleanup_stale()

    assert mgr._project_pages == {"worker:f1-0": [worker_page]}
    assert set(mgr._page_state) == {id(worker_page)}
    assert set(mgr._pending_file_choosers) == {"worker:f1-0"}
    assert mgr._context is None and mgr._browser is None and mgr._playwright is None


@pytest.mark.asyncio
async def test_handoff_waits_for_inflight_launch(tmp_path):
    """close_for_handoff during a launch must not tear down under it: the
    launch completes, THEN the context is closed and the lock released."""
    mgr = _manager(tmp_path)
    ctx = _make_mock_context()
    pw = _make_mock_playwright(ctx)
    pw.chromium.launch_persistent_context = AsyncMock(
        side_effect=_launch_returning(ctx)
    )
    mgr._get_clean_user_agent = AsyncMock(return_value=None)

    with patch("agent_os.daemon_v2.browser_manager.async_playwright") as mock_ap:
        mock_ap.return_value.start = AsyncMock(return_value=pw)
        launched, _ = await asyncio.gather(mgr.ensure_browser(), mgr.close_for_handoff())

    assert launched is ctx
    ctx.close.assert_awaited_once()
    assert mgr._context is None
    assert not mgr._main_lock.locked()
