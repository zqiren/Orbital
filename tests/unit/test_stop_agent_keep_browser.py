"""Spec 078: the idle-eviction sweep must not close the project's browser
pages — the workspace panel's live view shows them and the user expects the
page to survive the agent idling out. An explicit stop still closes them."""
import time
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent_os.daemon_v2.agent_manager import AgentManager

SID = "sess-keepbrowser"


def _make_manager():
    project_store = MagicMock()
    ws = MagicMock()
    ws.broadcast = MagicMock()
    sub_agent_mgr = MagicMock()
    sub_agent_mgr.list_active = MagicMock(return_value=[])
    sub_agent_mgr.stop = AsyncMock()
    sub_agent_mgr.stop_all = AsyncMock()
    mgr = AgentManager(
        project_store=project_store,
        ws_manager=ws,
        sub_agent_manager=sub_agent_mgr,
        activity_translator=MagicMock(),
        process_manager=MagicMock(),
    )
    mgr._browser_manager = MagicMock()
    mgr._browser_manager.close_project_pages = AsyncMock()
    return mgr


def _make_handle():
    session = MagicMock()
    session.is_stopped.return_value = False
    session.pop_queued_messages.return_value = []
    session._paused_for_approval = False
    task = MagicMock()
    task.done.return_value = True
    handle = MagicMock()
    handle.session = session
    handle.task = task
    handle.interceptor = None
    handle.last_activity = time.time() - 10_000
    return handle


@pytest.mark.asyncio
async def test_explicit_stop_closes_browser_pages():
    mgr = _make_manager()
    mgr._handles[("proj", SID)] = _make_handle()
    await mgr.stop_agent("proj", session_id=SID)
    mgr._browser_manager.close_project_pages.assert_awaited_once_with("proj")


@pytest.mark.asyncio
async def test_keep_browser_leaves_pages_open():
    mgr = _make_manager()
    mgr._handles[("proj", SID)] = _make_handle()
    await mgr.stop_agent("proj", session_id=SID, keep_browser=True)
    mgr._browser_manager.close_project_pages.assert_not_awaited()


@pytest.mark.asyncio
async def test_idle_eviction_stops_with_keep_browser():
    mgr = _make_manager()
    mgr._handles[("proj", SID)] = _make_handle()
    mgr.get_run_status = MagicMock(return_value="idle")
    mgr._sub_agents_block_eviction = MagicMock(return_value=False)
    mgr.stop_agent = AsyncMock()
    evicted = await mgr._evict_idle_once()
    assert evicted == [("proj", SID)]
    mgr.stop_agent.assert_awaited_once_with("proj", session_id=SID, keep_browser=True)
