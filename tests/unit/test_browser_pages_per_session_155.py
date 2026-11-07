# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Spec 155 §5.4 (B5–B7): sessions of one project run in parallel, so the
manager browser tool's pages are keyed by (project, session). Two sessions
get different tabs, stopping one leaves the other's tab open, and the live
view mirrors the viewer's session (without one: the project's most recent
page, today's behaviour). The profile — cookies, sign-ins — stays shared.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent_os.agent.tools.browser import BrowserTool
from agent_os.api.routes.browser_live import _LiveStream
from agent_os.daemon_v2.browser_manager import BrowserManager, page_scope
from tests.unit.test_browser_manager import (
    _make_mock_context,
    _make_mock_page,
    _make_mock_playwright,
)

PID = "proj155"


def _manager(tmp_path, pages):
    ctx = _make_mock_context(pages=pages)
    pw = _make_mock_playwright(ctx)
    mgr = BrowserManager(profile_dir=str(tmp_path / "profile"), headless=True)
    patcher = patch("agent_os.daemon_v2.browser_manager.async_playwright")
    mock_ap = patcher.start()
    mock_ap.return_value.start = AsyncMock(return_value=pw)
    return mgr, patcher


def _tool(bm, session_id):
    return BrowserTool(browser_manager=bm, project_id=PID, workspace="/tmp/ws",
                       autonomy_preset="hands_off", session_id=session_id)


@pytest.mark.asyncio
async def test_two_sessions_get_their_own_pages_and_stop_closes_only_its_own(tmp_path):
    page_a, page_b = _make_mock_page(), _make_mock_page()
    bm, patcher = _manager(tmp_path, [page_a, page_b])
    try:
        tool_a, tool_b = _tool(bm, "sess-a"), _tool(bm, "sess-b")
        assert tool_a._scope != tool_b._scope
        got_a = await bm.get_page(tool_a._scope)
        got_b = await bm.get_page(tool_b._scope)
        assert got_a is page_a and got_b is page_b
        assert await bm.get_page(tool_a._scope) is page_a   # A keeps its tab

        await bm.close_session_pages(PID, "sess-a")         # A's Stop
        page_a.close.assert_awaited_once()
        page_b.close.assert_not_awaited()
        assert await bm.get_all_pages(tool_b._scope) == [page_b]
    finally:
        patcher.stop()


@pytest.mark.asyncio
async def test_project_teardown_closes_every_session_page(tmp_path):
    page_a, page_b, other = _make_mock_page(), _make_mock_page(), _make_mock_page()
    bm, patcher = _manager(tmp_path, [page_a, page_b, other])
    try:
        await bm.get_page(page_scope(PID, "sess-a"))
        await bm.get_page(page_scope(PID, "sess-b"))
        await bm.get_page(page_scope("another-project", "sess-c"))
        await bm.close_project_pages(PID)
        page_a.close.assert_awaited_once()
        page_b.close.assert_awaited_once()
        other.close.assert_not_awaited()
    finally:
        patcher.stop()


def test_worker_and_session_less_scopes_are_unchanged():
    assert page_scope(PID) == PID
    assert page_scope("worker:fan1-0", "sess-a") == "worker:fan1-0"
    assert page_scope(PID, "sess-a") != PID


def _stream(bm, session_id):
    return _LiveStream(MagicMock(), PID, session_id, bm, None)


@pytest.mark.asyncio
async def test_the_live_view_mirrors_the_viewers_session(tmp_path):
    page_a, page_b = _make_mock_page(), _make_mock_page()
    bm, patcher = _manager(tmp_path, [page_a, page_b])
    try:
        await bm.get_page(page_scope(PID, "sess-a"))
        await bm.get_page(page_scope(PID, "sess-b"))       # B used last
        assert await _stream(bm, "sess-a")._current_page() is page_a
        assert await _stream(bm, "sess-b")._current_page() is page_b
        # No session named: the project's most recent page, as before.
        assert await _stream(bm, None)._current_page() is page_b
        await bm.get_page(page_scope(PID, "sess-a"))       # A acts again
        assert await _stream(bm, None)._current_page() is page_a
        # A session with no page of its own shows nothing (never B's tab).
        assert await _stream(bm, "sess-c")._current_page() is None
    finally:
        patcher.stop()
