# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Spec 155 §5.5 (B8, B9): the queue stays serial over its OWN items, with
chat sessions of the project running beside it.

- An item dispatches while a chat session runs (no busy-slot defer).
- The next item waits while the previous item's session is still live.
- Pause stops the queue's attempt session, never the chat.
- Retry injects into the attempt session by id and never touches (re-keys,
  stops) another session's handle — ``switch_session`` is gone.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from agent_os.daemon_v2.models import make_session_key
from agent_os.queue.dispatcher import QueueDispatcher
from agent_os.queue.models import AttemptRecord, ItemState, QueueRunState
from tests.unit.test_parallel_sessions_155 import PID, _cleanup, _manager, _plant

CHAT = "sess-chat-1111"
ATTEMPT = "sess-attempt-2222"


def _dispatcher(mgr, ws):
    store = mgr.get_queue_store(PID, workspace=ws)
    disp = QueueDispatcher(project_id=PID, store=store, agent_manager=mgr, workspace=ws)
    disp._wait_idle = AsyncMock()
    disp._await_and_handle = AsyncMock()
    return disp, store


@pytest.mark.asyncio
async def test_an_item_dispatches_while_a_chat_session_runs(tmp_path):
    ws = str(tmp_path)
    mgr = _manager(ws)
    _plant(mgr, CHAT, ws)
    mgr.is_onboarding_complete = MagicMock(return_value=True)
    mgr.new_session = AsyncMock(return_value={"session_id": ATTEMPT})
    mgr.inject_message = AsyncMock(return_value="started")
    disp, store = _dispatcher(mgr, ws)
    item = store.add_item("summarise the inbox")
    try:
        await disp._dispatch_one(item)
    finally:
        _cleanup(mgr)
    mgr.inject_message.assert_awaited_once()
    assert mgr.inject_message.await_args.kwargs["session_id"] == ATTEMPT
    disp._wait_idle.assert_not_awaited()
    assert store.load().items[0].attempts[-1].session_id == ATTEMPT


@pytest.mark.asyncio
async def test_the_next_item_waits_while_the_previous_attempt_session_is_live(tmp_path):
    ws = str(tmp_path)
    mgr = _manager(ws)
    mgr.is_onboarding_complete = MagicMock(return_value=True)
    mgr.new_session = AsyncMock(return_value={"session_id": "sess-next"})
    mgr.inject_message = AsyncMock(return_value="started")
    disp, store = _dispatcher(mgr, ws)
    disp._attempt_session_id = ATTEMPT
    _plant(mgr, ATTEMPT, ws, running=False, waiting=True)  # its workers still run
    item = store.add_item("next thing")
    try:
        await disp._dispatch_one(item)
        mgr.new_session.assert_not_awaited()
        disp._wait_idle.assert_awaited_once()
        # Once the attempt is at rest the item goes out.
        mgr._idle_poll_tasks.pop(make_session_key(PID, ATTEMPT)).cancel()
        await disp._dispatch_one(item)
    finally:
        _cleanup(mgr)
    mgr.inject_message.assert_awaited_once()


@pytest.mark.asyncio
async def test_pause_stops_only_the_attempt_session(tmp_path):
    ws = str(tmp_path)
    mgr = _manager(ws)
    chat = _plant(mgr, CHAT, ws)                  # first handle of the project
    attempt = _plant(mgr, ATTEMPT, ws)
    disp, store = _dispatcher(mgr, ws)
    item = store.add_item("long research")
    store.append_attempt(item.id, AttemptRecord(session_id=ATTEMPT))
    store.set_item_state(item.id, ItemState.RUNNING)
    try:
        await disp.stop()
    finally:
        _cleanup(mgr)
    attempt.loop.terminate.assert_awaited_once()
    chat.loop.terminate.assert_not_awaited()
    mgr._sub_agent_manager.stop_all.assert_awaited_once_with(PID, session_id=ATTEMPT)
    assert store.load().state == QueueRunState.PAUSED


@pytest.mark.asyncio
async def test_retry_injects_into_the_attempt_and_never_touches_a_chat_handle(tmp_path):
    ws = str(tmp_path)
    mgr = _manager(ws)
    chat = _plant(mgr, CHAT, ws)                  # a running chat, first handle
    chat_session = chat.session
    mgr.inject_message = AsyncMock(return_value="started")
    disp, store = _dispatcher(mgr, ws)
    item = store.add_item("draft the post")
    store.append_attempt(item.id, AttemptRecord(session_id=ATTEMPT))
    store.set_item_state(item.id, ItemState.BLOCKED)
    try:
        result = await disp.retry_blocked_item(item.id, "use the shorter title", mode="answer")
    finally:
        _cleanup(mgr)
    assert result["session_id"] == ATTEMPT
    mgr.inject_message.assert_awaited_once()
    assert mgr.inject_message.await_args.kwargs["session_id"] == ATTEMPT
    assert mgr.inject_message.await_args.kwargs["queue_state"] == "running"
    # The chat's handle is where it was, the same session, never stopped.
    assert mgr._handles[make_session_key(PID, CHAT)] is chat
    assert chat.session is chat_session
    chat.loop.terminate.assert_not_awaited()
