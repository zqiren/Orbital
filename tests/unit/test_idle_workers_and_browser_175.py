# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Spec 175: idle workers and the agent browser stop instead of running until
the app quits.

1. The idle sweep visits sessions whose workers no manager handle owns (a
   pinned chat). Same timeout, same gates, the workers' own stop. Claude Code
   only for now.
2. The agent browser closes after 15 minutes with no agent browser action and
   no user input through the live view; the sign-in browser is never closed.
"""

from __future__ import annotations

import time
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent_os.daemon_v2.agent_manager import AgentManager
from agent_os.daemon_v2.browser_manager import BrowserManager
from agent_os.daemon_v2.sub_agent_manager import SubAgentManager

TIMEOUT = AgentManager.EVICTION_IDLE_TIMEOUT


def _idle_adapter():
    a = MagicMock()
    a.is_alive.return_value = True
    a.is_idle.return_value = True
    a._transport = None
    return a


def _sam() -> SubAgentManager:
    return SubAgentManager(process_manager=MagicMock())


def _age(sam, sk, seconds):
    sam._last_worker_activity[sk] = time.monotonic() - seconds


# ---------------------------------------------------------------------------
# SubAgentManager: the idle clock and the candidate list
# ---------------------------------------------------------------------------

def test_handleless_claude_code_session_idle_past_timeout_is_a_candidate():
    sam = _sam()
    sk = ("p", "s1")
    sam._adapters[sk] = {"claude-code": _idle_adapter()}
    _age(sam, sk, TIMEOUT + 1)
    assert sam.idle_handleless_sessions({}, TIMEOUT) == [sk]


def test_recent_worker_activity_is_not_a_candidate():
    sam = _sam()
    sk = ("p", "s1")
    sam._adapters[sk] = {"claude-code": _idle_adapter()}
    _age(sam, sk, TIMEOUT - 5)
    assert sam.idle_handleless_sessions({}, TIMEOUT) == []


def test_session_owned_by_a_manager_handle_is_left_to_the_handle_branch():
    sam = _sam()
    sk = ("p", "s1")
    sam._adapters[sk] = {"claude-code": _idle_adapter()}
    _age(sam, sk, TIMEOUT + 1)
    assert sam.idle_handleless_sessions({sk: object()}, TIMEOUT) == []


@pytest.mark.parametrize("slate", [
    {"codex": None},
    {"claude-code": None, "codex": None},
    {"opencode": None},
])
def test_other_agents_keep_todays_behaviour(slate):
    """Only Claude Code's stop-then-resume was verified: a session holding any
    other worker is not a candidate."""
    sam = _sam()
    sk = ("p", "s1")
    sam._adapters[sk] = {h: _idle_adapter() for h in slate}
    _age(sam, sk, TIMEOUT + 100)
    assert sam.idle_handleless_sessions({}, TIMEOUT) == []


def test_unclocked_session_starts_its_clock_instead_of_being_stopped_at_once():
    sam = _sam()
    sk = ("p", "s1")
    sam._adapters[sk] = {"claude-code": _idle_adapter()}
    assert sam.idle_handleless_sessions({}, TIMEOUT) == []
    assert sk in sam._last_worker_activity
    _age(sam, sk, TIMEOUT)
    assert sam.idle_handleless_sessions({}, TIMEOUT) == [sk]


def test_clock_entries_for_sessions_without_workers_are_dropped():
    sam = _sam()
    _age(sam, ("p", "gone"), 5)
    sam.idle_handleless_sessions({}, TIMEOUT)
    assert ("p", "gone") not in sam._last_worker_activity


@pytest.mark.asyncio
async def test_turn_end_resets_the_idle_clock():
    sam = _sam()
    sk = ("p", "s1")
    sam._adapters[sk] = {"claude-code": _idle_adapter()}
    _age(sam, sk, TIMEOUT + 50)
    await sam._on_prompt_turn_closed("p", "claude-code", session_id="s1",
                                     cause="completed")
    assert sam.idle_handleless_sessions({}, TIMEOUT) == []


@pytest.mark.asyncio
async def test_dispatch_resets_the_idle_clock():
    from agent_os.daemon_v2.sub_agent_manager import _QueuedPrompt
    sam = _sam()
    sk = ("p", "s1")
    adapter = _idle_adapter()
    sam._adapters[sk] = {"claude-code": adapter}
    _age(sam, sk, TIMEOUT + 50)
    sam._build_recap = AsyncMock(return_value="")
    sam._dispatch_async = AsyncMock()
    adapter._orbital_first_turn_note = None
    await sam._dispatch_prompt_locked(
        adapter, _QueuedPrompt(message="hi", dispatch_id="d1",
                               transcript_path="t"),
        "p", "s1", "claude-code")
    assert time.monotonic() - sam._last_worker_activity[sk] < 5


# ---------------------------------------------------------------------------
# AgentManager: the sweep stops handle-less idle workers through the gates
# ---------------------------------------------------------------------------

def _manager(sam) -> AgentManager:
    return AgentManager(
        project_store=MagicMock(),
        ws_manager=MagicMock(),
        sub_agent_manager=sam,
        activity_translator=MagicMock(),
        process_manager=MagicMock(),
    )


@pytest.mark.asyncio
async def test_sweep_stops_idle_pinned_claude_code_worker():
    sam = _sam()
    sk = ("p", "s1")
    sam._adapters[sk] = {"claude-code": _idle_adapter()}
    _age(sam, sk, TIMEOUT + 1)
    sam.stop_all = AsyncMock()
    mgr = _manager(sam)

    evicted = await mgr._evict_idle_once()

    assert evicted == [sk]
    sam.stop_all.assert_awaited_once_with("p", session_id="s1")


@pytest.mark.asyncio
async def test_sweep_leaves_a_working_pinned_worker():
    sam = _sam()
    sk = ("p", "s1")
    adapter = _idle_adapter()
    adapter.is_idle.return_value = False  # a turn is open
    sam._adapters[sk] = {"claude-code": adapter}
    _age(sam, sk, TIMEOUT + 1)
    sam.stop_all = AsyncMock()
    mgr = _manager(sam)

    assert await mgr._evict_idle_once() == []
    sam.stop_all.assert_not_awaited()


@pytest.mark.asyncio
async def test_sweep_leaves_a_pinned_worker_with_recent_background_work():
    """The existing background-activity gate (it compares against the same
    one timeout) holds a handle-less session exactly as a managed one."""
    sam = _sam()
    sk = ("p", "s1")
    sam._adapters[sk] = {"claude-code": _idle_adapter()}
    _age(sam, sk, TIMEOUT + 1)
    sam.stop_all = AsyncMock()
    sam.seconds_since_background_activity = MagicMock(return_value=10.0)
    mgr = _manager(sam)

    assert await mgr._evict_idle_once() == []
    sam.stop_all.assert_not_awaited()


@pytest.mark.asyncio
async def test_sweep_leaves_a_pinned_worker_with_live_background_status():
    sam = _sam()
    sk = ("p", "s1")
    sam._adapters[sk] = {"claude-code": _idle_adapter()}
    _age(sam, sk, TIMEOUT + 1)
    sam.stop_all = AsyncMock()
    sam.list_active = MagicMock(return_value=[
        {"handle": "claude-code", "status": "background-running"}])
    mgr = _manager(sam)

    assert await mgr._evict_idle_once() == []
    sam.stop_all.assert_not_awaited()


@pytest.mark.asyncio
async def test_sweep_does_not_double_stop_a_managed_session():
    """A session with a manager handle goes through stop_agent once; the
    handle-less branch does not visit it."""
    sam = _sam()
    sk = ("p", "s1")
    sam._adapters[sk] = {"claude-code": _idle_adapter()}
    _age(sam, sk, TIMEOUT + 1)
    sam.stop_all = AsyncMock()
    mgr = _manager(sam)
    h = MagicMock()
    h.session.is_stopped.return_value = False
    h.session._paused_for_approval = False
    h.task.done.return_value = True
    h.last_activity = time.time() - 9999
    mgr._handles[sk] = h
    mgr.stop_agent = AsyncMock()

    assert await mgr._evict_idle_once() == [sk]
    mgr.stop_agent.assert_awaited_once_with("p", session_id="s1",
                                            keep_browser=True)
    sam.stop_all.assert_not_awaited()


@pytest.mark.asyncio
async def test_one_failed_stop_does_not_abort_the_sweep():
    sam = _sam()
    for sid in ("s1", "s2"):
        sam._adapters[("p", sid)] = {"claude-code": _idle_adapter()}
        _age(sam, ("p", sid), TIMEOUT + 1)

    async def _stop_all(pid, *, session_id=None):
        if session_id == "s1":
            raise RuntimeError("boom")

    sam.stop_all = AsyncMock(side_effect=_stop_all)
    mgr = _manager(sam)

    assert await mgr._evict_idle_once() == [("p", "s2")]
    assert sam.stop_all.await_count == 2


@pytest.mark.asyncio
async def test_sweep_loop_runs_the_browser_idle_check(monkeypatch):
    import asyncio
    sam = _sam()
    mgr = _manager(sam)
    mgr._browser_manager = MagicMock()
    mgr._browser_manager.close_if_idle = AsyncMock(return_value=False)
    monkeypatch.setattr(AgentManager, "EVICTION_SWEEP_INTERVAL", 0.01)
    task = asyncio.create_task(mgr._eviction_sweep())
    for _ in range(100):
        await asyncio.sleep(0.01)
        if mgr._browser_manager.close_if_idle.await_count:
            break
    task.cancel()
    assert mgr._browser_manager.close_if_idle.await_count >= 1


# ---------------------------------------------------------------------------
# BrowserManager: idle close
# ---------------------------------------------------------------------------

def _browser(tmp_path, *, idle_for: float) -> BrowserManager:
    bm = BrowserManager(profile_dir=str(tmp_path / "profile"), headless=True)
    bm._context = MagicMock()
    bm._context.close = AsyncMock()
    bm._browser = MagicMock()
    bm._playwright = MagicMock()
    bm._playwright.stop = AsyncMock()
    bm._last_use = time.monotonic() - idle_for
    return bm


@pytest.mark.asyncio
async def test_browser_closes_after_idle_period(tmp_path):
    bm = _browser(tmp_path, idle_for=BrowserManager.IDLE_CLOSE_AFTER + 1)
    ctx = bm._context
    bm._project_pages["p"] = [MagicMock()]

    assert await bm.close_if_idle() is True
    ctx.close.assert_awaited_once()
    assert bm._context is None and bm._playwright is None
    assert bm._project_pages == {}


@pytest.mark.asyncio
async def test_browser_in_use_is_not_closed(tmp_path):
    bm = _browser(tmp_path, idle_for=BrowserManager.IDLE_CLOSE_AFTER - 60)
    ctx = bm._context
    assert await bm.close_if_idle() is False
    ctx.close.assert_not_awaited()


@pytest.mark.asyncio
async def test_browser_not_closed_while_sign_in_browser_is_open(tmp_path):
    bm = _browser(tmp_path, idle_for=BrowserManager.IDLE_CLOSE_AFTER + 1)
    ctx = bm._context
    bm._warmup_active = True
    assert await bm.close_if_idle() is False
    ctx.close.assert_not_awaited()


@pytest.mark.asyncio
async def test_no_browser_is_a_noop(tmp_path):
    bm = BrowserManager(profile_dir=str(tmp_path / "profile"), headless=True)
    bm._last_use = 0.0
    assert await bm.close_if_idle() is False


@pytest.mark.asyncio
async def test_agent_page_access_stamps_use_worker_scope_does_not(tmp_path):
    bm = _browser(tmp_path, idle_for=BrowserManager.IDLE_CLOSE_AFTER + 1)
    page = MagicMock()
    page.is_closed.return_value = False
    bm._project_pages["p"] = [page]
    bm.ensure_browser = AsyncMock(return_value=bm._context)
    await bm.get_page("p")
    assert await bm.close_if_idle() is False

    bm._last_use = time.monotonic() - BrowserManager.IDLE_CLOSE_AFTER - 1
    bm._get_worker_context = AsyncMock(return_value=MagicMock())
    wpage = MagicMock()
    wpage.is_closed.return_value = False
    bm._project_pages["worker:x"] = [wpage]
    await bm.get_page("worker:x")
    assert await bm.close_if_idle() is True


def test_live_view_user_input_stamps_use():
    from agent_os.api.routes.browser_live import _LiveStream
    bm = MagicMock()
    stream = _LiveStream(MagicMock(), "p", "s1", bm, None)
    stream._enqueue({"method": "Input.dispatchMouseEvent", "params": {},
                     "merge": None}, "mouse.down")
    bm.note_use.assert_called_once_with()


def test_profile_cache_is_capped_at_256_mb_for_both_browsers():
    """Measured (REPORT-1114): with the cap Chrome trims an existing oversized
    HTTP cache AND Code Cache down to it on its own; nothing of ours deletes."""
    assert "--disk-cache-size=268435456" in BrowserManager.CHROME_FLAGS
    # The sign-in browser takes the same list minus --start-minimized.
    assert "--disk-cache-size=268435456" in [
        f for f in BrowserManager.CHROME_FLAGS if f != "--start-minimized"]
