# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Upload wake semantics (product decision 2026-06-11):

An upload-sourced queue item may wake an IDLE queue that holds no other
queueable item — the drag-drop itself is the user's intent. PAUSED always
gates (pause is a consent boundary). Staged user items keep explicit-start
semantics: idle-with-staged-items stages the upload alongside them.
"""

from __future__ import annotations

import pytest

from agent_os.api.routes import files_v2
from agent_os.queue.models import ItemState, QueueRunState
from agent_os.queue.store import QueueStore


class _FakeDispatcher:
    def __init__(self):
        self.resume_calls = 0
        self.notify_calls = 0

    async def resume(self):
        self.resume_calls += 1
        return {"status": "running"}

    def notify_new_item(self):
        self.notify_calls += 1


class _FakeAgentManager:
    def __init__(self, store, dispatcher, onboarded=True):
        self._store = store
        self._dispatcher = dispatcher
        self._onboarded = onboarded

    def get_queue_store(self, project_id, workspace=""):
        return self._store

    def get_dispatcher(self, project_id):
        return self._dispatcher

    def is_onboarding_complete(self, project_id):
        return self._onboarded


class _FakeProjectStore:
    def __init__(self, workspace):
        self._workspace = workspace

    def get_project(self, project_id):
        return {"project_id": project_id, "workspace": self._workspace}


@pytest.fixture(autouse=True)
def _restore_files_v2_globals():
    """Restore files_v2 module globals after each test to prevent state leakage."""
    saved = (files_v2._project_store, files_v2._agent_manager, files_v2._ws_manager)
    yield
    files_v2._project_store, files_v2._agent_manager, files_v2._ws_manager = saved


def _wire(tmp_path, *, onboarded=True):
    store = QueueStore(tmp_path / "queue.json")
    dispatcher = _FakeDispatcher()
    mgr = _FakeAgentManager(store, dispatcher, onboarded=onboarded)
    files_v2.configure(_FakeProjectStore(str(tmp_path)), agent_manager=mgr)
    return store, dispatcher


@pytest.mark.asyncio
async def test_upload_wakes_idle_empty_queue(tmp_path):
    store, dispatcher = _wire(tmp_path)
    store.set_queue_state(QueueRunState.IDLE)
    await files_v2._notify_upload("proj_x", "uploads/f.txt", "f.txt", 10)
    assert dispatcher.resume_calls == 1
    items = store.load().items
    assert len(items) == 1 and items[0].state == ItemState.QUEUED


@pytest.mark.asyncio
async def test_upload_does_not_wake_paused_queue(tmp_path):
    store, dispatcher = _wire(tmp_path)
    store.set_queue_state(QueueRunState.PAUSED)
    await files_v2._notify_upload("proj_x", "uploads/f.txt", "f.txt", 10)
    assert dispatcher.resume_calls == 0
    assert dispatcher.notify_calls == 1  # staged + dispatcher poked, not started
    assert store.load().state == QueueRunState.PAUSED


@pytest.mark.asyncio
async def test_upload_does_not_wake_idle_queue_with_staged_items(tmp_path):
    store, dispatcher = _wire(tmp_path)
    store.add_item("user staged batch item")
    store.set_queue_state(QueueRunState.IDLE)
    await files_v2._notify_upload("proj_x", "uploads/f.txt", "f.txt", 10)
    assert dispatcher.resume_calls == 0
    assert dispatcher.notify_calls == 1
    assert store.load().state == QueueRunState.IDLE
    assert len(store.load().items) == 2


@pytest.mark.asyncio
async def test_upload_does_not_wake_before_onboarding(tmp_path):
    store, dispatcher = _wire(tmp_path, onboarded=False)
    store.set_queue_state(QueueRunState.IDLE)
    await files_v2._notify_upload("proj_x", "uploads/f.txt", "f.txt", 10)
    assert dispatcher.resume_calls == 0


@pytest.mark.asyncio
async def test_running_queue_just_notifies(tmp_path):
    store, dispatcher = _wire(tmp_path)
    # fresh store defaults to RUNNING
    await files_v2._notify_upload("proj_x", "uploads/f.txt", "f.txt", 10)
    assert dispatcher.resume_calls == 0
    assert dispatcher.notify_calls == 1


@pytest.mark.asyncio
async def test_pause_landing_mid_notify_wins_over_wake(tmp_path):
    """A pause that lands between the should_wake computation and the resume
    must win — an upload never overrides PAUSED. Simulated by flipping the
    store to PAUSED as a side effect of get_dispatcher(), which is called
    between the two."""
    store = QueueStore(tmp_path / "queue.json")
    dispatcher = _FakeDispatcher()

    class _PauseInterleavingManager(_FakeAgentManager):
        def get_dispatcher(self, project_id):
            # Interleaved POST /queue/stop landing mid-notify.
            self._store.set_queue_state(QueueRunState.PAUSED)
            return self._dispatcher

    mgr = _PauseInterleavingManager(store, dispatcher)
    files_v2.configure(_FakeProjectStore(str(tmp_path)), agent_manager=mgr)

    store.set_queue_state(QueueRunState.IDLE)
    await files_v2._notify_upload("proj_x", "uploads/f.txt", "f.txt", 10)
    assert dispatcher.resume_calls == 0
    assert dispatcher.notify_calls == 1  # item must not be silently dropped
    assert store.load().state == QueueRunState.PAUSED


@pytest.mark.asyncio
async def test_dedup_reupload_of_done_item_does_not_wake(tmp_path):
    """Idempotency hit on an already-DONE item must not restart the queue."""
    store, dispatcher = _wire(tmp_path)
    item = store.add_item(
        "x", source="upload", idempotency_key="upload:uploads/f.txt:10",
    )
    store.set_item_state(item.id, ItemState.DONE)
    store.set_queue_state(QueueRunState.IDLE)
    await files_v2._notify_upload("proj_x", "uploads/f.txt", "f.txt", 10)
    assert dispatcher.resume_calls == 0
