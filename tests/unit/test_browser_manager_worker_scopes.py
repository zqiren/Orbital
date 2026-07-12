# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Worker-scope isolation in BrowserManager — all Playwright mocked."""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent_os.daemon_v2.browser_manager import BrowserManager, WORKER_SCOPE_PREFIX
from tests.unit.test_browser_manager import _make_mock_page, _make_mock_context


def _make_mock_worker_browser():
    """Mock plain-launch Browser producing a fresh mock context per new_context()."""
    browser = MagicMock()
    browser.is_connected = MagicMock(return_value=True)
    browser.close = AsyncMock()
    browser.new_context = AsyncMock(side_effect=lambda **kw: _make_mock_context())
    return browser


@pytest.fixture
def manager(tmp_path):
    m = BrowserManager(profile_dir=str(tmp_path / "profile"))
    # Playwright already "started"; worker browser launches via chromium.launch
    m._playwright = MagicMock()
    worker_browser = _make_mock_worker_browser()
    m._playwright.chromium.launch = AsyncMock(return_value=worker_browser)
    # Main persistent-context path must never be touched by worker scopes
    m._playwright.chromium.launch_persistent_context = AsyncMock(
        side_effect=AssertionError("worker scope must not launch the main browser")
    )
    m._apply_stealth = AsyncMock()
    m._setup_page_handlers = MagicMock()
    return m


@pytest.mark.asyncio
async def test_worker_scope_gets_isolated_context(manager):
    page_a = await manager.get_page("worker:f1-0")
    page_b = await manager.get_page("worker:f1-1")
    assert page_a is not page_b
    # one context per scope
    assert set(manager._worker_contexts) == {"worker:f1-0", "worker:f1-1"}
    assert manager._worker_contexts["worker:f1-0"] is not manager._worker_contexts["worker:f1-1"]
    # single worker-browser launch, two contexts
    manager._playwright.chromium.launch.assert_awaited_once()


@pytest.mark.asyncio
async def test_worker_scope_reuses_existing_page(manager):
    page1 = await manager.get_page("worker:f1-0")
    page2 = await manager.get_page("worker:f1-0")
    assert page1 is page2


@pytest.mark.asyncio
async def test_close_worker_scope_closes_context_and_browser_when_last(manager):
    await manager.get_page("worker:f1-0")
    await manager.get_page("worker:f1-1")
    ctx0 = manager._worker_contexts["worker:f1-0"]

    await manager.close_worker_scope("worker:f1-0")
    ctx0.close.assert_awaited_once()
    assert "worker:f1-0" not in manager._worker_contexts
    assert manager._worker_browser is not None  # f1-1 still alive

    await manager.close_worker_scope("worker:f1-1")
    assert manager._worker_contexts == {}
    assert manager._worker_browser is None  # last context gone → browser closed


@pytest.mark.asyncio
async def test_close_worker_scope_is_idempotent(manager):
    await manager.get_page("worker:f1-0")
    await manager.close_worker_scope("worker:f1-0")
    await manager.close_worker_scope("worker:f1-0")  # must not raise


def test_prefix_constant():
    assert WORKER_SCOPE_PREFIX == "worker:"


@pytest.mark.asyncio
async def test_double_launch_race_closes_loser(manager):
    """Concurrent get_page on different worker scopes must not double-launch.

    Under the lock design, the first caller to enter _get_worker_context holds
    _worker_lock for the whole launch+new_context sequence, so a second,
    concurrent first-use call cannot even start launching until the first has
    already registered its browser and context. Only one launch happens — no
    loser browser is ever created, so there is nothing to close. This is the
    same "single browser, no orphan" invariant the old post-await re-check
    enforced, now true by construction instead of by patch.
    """
    created_browsers = []

    async def mock_launch(**kw):
        browser = _make_mock_worker_browser()
        created_browsers.append(browser)
        await asyncio.sleep(0)  # would have exposed the old race, if any remained
        return browser

    manager._playwright.chromium.launch = AsyncMock(side_effect=mock_launch)

    page_a, page_b = await asyncio.gather(
        manager.get_page("worker:f1-0"),
        manager.get_page("worker:f1-1"),
    )

    assert page_a is not page_b
    manager._playwright.chromium.launch.assert_awaited_once()
    assert len(created_browsers) == 1
    assert manager._worker_browser is created_browsers[0]
    created_browsers[0].close.assert_not_awaited()  # sole browser survives
    assert set(manager._worker_contexts) == {"worker:f1-0", "worker:f1-1"}


@pytest.mark.asyncio
async def test_close_during_inflight_creation_does_not_close_browser(manager):
    """Regression: close of the last scope while a sibling creation is in flight.

    Scope A exists. Scope B's creation parks inside new_context() — B already
    holds _worker_lock, ctx is built but not yet committed to the dict.
    Closing scope A must wait behind the lock rather than racing to tear down
    the browser while B still needs it.
    """
    await manager.get_page("worker:f1-0")

    event = asyncio.Event()
    original_new_context = manager._worker_browser.new_context

    async def parked_new_context(**kw):
        ctx = await original_new_context(**kw)
        await event.wait()  # parks here, still holding _worker_lock
        return ctx

    manager._worker_browser.new_context = AsyncMock(side_effect=parked_new_context)

    get_b_task = asyncio.create_task(manager.get_page("worker:f1-1"))
    await asyncio.sleep(0)  # let B acquire the lock and park inside new_context

    close_a_task = asyncio.create_task(manager.close_worker_scope("worker:f1-0"))
    await asyncio.sleep(0)  # close_a queues behind the lock, must not race ahead

    # close_a is blocked on the lock — guard with a short timeout so a
    # regression that lets it race ahead fails fast instead of hanging.
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(asyncio.shield(close_a_task), timeout=0.05)
    manager._worker_browser.close.assert_not_awaited()

    event.set()  # release B's creation
    page_b = await asyncio.wait_for(get_b_task, timeout=1)
    assert page_b is not None
    assert "worker:f1-1" in manager._worker_contexts

    # close_a can now complete; browser survives because B registered first.
    await asyncio.wait_for(close_a_task, timeout=1)
    assert "worker:f1-0" not in manager._worker_contexts
    assert manager._worker_browser is not None
    manager._worker_browser.close.assert_not_awaited()


@pytest.mark.asyncio
async def test_legit_last_close_overlapping_new_creation(manager):
    """Regression: legitimate last-context close overlaps a new creation in flight.

    Scope A exists. Scope B's creation parks inside new_context() (holding
    _worker_lock, ctx built but not yet committed). Closing scope A waits
    behind the lock; once B commits, the browser must survive (B still needs
    it). Only after B is ALSO closed does the browser go down.
    """
    await manager.get_page("worker:f1-0")
    browser_ref = manager._worker_browser  # save reference before it's cleared

    event = asyncio.Event()
    original_new_context = browser_ref.new_context

    async def parked_new_context(**kw):
        ctx = await original_new_context(**kw)
        await event.wait()  # parks here, still holding _worker_lock
        return ctx

    browser_ref.new_context = AsyncMock(side_effect=parked_new_context)

    get_b_task = asyncio.create_task(manager.get_page("worker:f1-1"))
    await asyncio.sleep(0)  # let B park inside new_context, holding the lock

    close_a_task = asyncio.create_task(manager.close_worker_scope("worker:f1-0"))
    await asyncio.sleep(0)  # close_a queues behind the lock

    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(asyncio.shield(close_a_task), timeout=0.05)
    browser_ref.close.assert_not_awaited()

    event.set()  # release B's creation
    page_b = await asyncio.wait_for(get_b_task, timeout=1)
    assert page_b is not None
    assert "worker:f1-1" in manager._worker_contexts

    await asyncio.wait_for(close_a_task, timeout=1)
    assert set(manager._worker_contexts) == {"worker:f1-1"}
    assert manager._worker_browser is not None  # B still alive
    browser_ref.close.assert_not_awaited()

    # Now close B (actual last scope) → browser closes.
    await manager.close_worker_scope("worker:f1-1")
    assert manager._worker_contexts == {}
    assert manager._worker_browser is None
    browser_ref.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_close_waits_behind_failed_creation_leaves_no_orphans(manager):
    """TDD regression (RED under the counter design, GREEN under the lock).

    Scope A exists. Scope B's creation parks inside new_context() (holding
    _worker_lock). Closing scope A must wait rather than proceed concurrently.
    B's parked creation is then released to FAIL. Under the old counter-based
    code, a close arriving while a sibling creation was in flight could see
    the counter hit zero and never re-fire once that creation failed — the
    worker browser orphaned permanently. Under the lock, close_worker_scope's
    own acquisition (deferred behind B) is what tears the browser down once
    B's failure path declines to (because A's context was still in the dict
    when B's except-block ran) — so nothing is ever left running.
    """
    await manager.get_page("worker:f1-0")
    ctx_a = manager._worker_contexts["worker:f1-0"]
    browser_ref = manager._worker_browser

    created_browsers = [browser_ref]
    event = asyncio.Event()

    async def parked_failing_new_context(**kw):
        await event.wait()  # parks here, still holding _worker_lock
        raise RuntimeError("simulated context creation failure")

    browser_ref.new_context = AsyncMock(side_effect=parked_failing_new_context)

    get_b_task = asyncio.create_task(manager.get_page("worker:f1-1"))
    await asyncio.sleep(0)  # let B acquire the lock and park inside new_context

    close_a_task = asyncio.create_task(manager.close_worker_scope("worker:f1-0"))
    await asyncio.sleep(0)  # close_a queues behind the lock

    # close_a must be blocked behind B's in-flight (about-to-fail) creation.
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(asyncio.shield(close_a_task), timeout=0.05)
    ctx_a.close.assert_not_awaited()

    event.set()  # release B's creation — it raises
    with pytest.raises(RuntimeError, match="simulated context creation failure"):
        await asyncio.wait_for(get_b_task, timeout=1)

    # close_a can now proceed and complete — it, not B, tears down the browser.
    await asyncio.wait_for(close_a_task, timeout=1)

    assert manager._worker_contexts == {}
    assert manager._worker_browser is None
    ctx_a.close.assert_awaited_once()
    for browser in created_browsers:
        browser.close.assert_awaited_once()  # zero orphans
