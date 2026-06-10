# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Journey: DELETE IS IDLE-ONLY, end to end against the real dispatcher + route.

Storyline:
  1. Enqueue several items; start the real QueueDispatcher.
  2. Item 1 rotates to RUNNING (its loop task is held open by a release event).
  3. DELETE the RUNNING item via the real route → 409. The queue is NOT stalled
     by a phantom, and the item is intact (record present, attempt still open).
  4. Pause the queue (dispatcher.stop): the held loop terminates, the slot is
     released, the item goes idle.
  5. DELETE the now-idle item → 200: its session JSONL AND the record are gone.
  6. Resume: the dispatcher advances the remaining items to completion — no
     orphaned session, no stall.

The agent loop is mocked (no real LLM): a purpose-built scripted AgentManager
drives ``inject_message`` / ``new_session`` / loop tasks with real on-disk
session JSONLs so ``delete_session`` actually unlinks files. ``current_holder_
session_id`` is REAL-shaped — it reports the session whose loop task is still
alive, which is exactly the liveness gate the delete route consults.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from agent_os.queue.dispatcher import QueueDispatcher
from agent_os.queue.models import ItemState, QueueRunState
from agent_os.queue.store import QueueStore


class _ScriptedLoop:
    """Per-session loop double. ``terminate()`` marks a cancelled exit and
    releases the held loop task so it can exit (mirrors the real pause path)."""

    def __init__(self, release_event: asyncio.Event):
        self._exit_reason = "complete"
        self._exit_summary = "did the work"
        self._exit_block_reason = None
        self._queue_state = "running"
        self._release = release_event

    async def terminate(self):
        self._exit_reason = "cancelled"
        self._release.set()


class _JourneyManager:
    """Scripted AgentManager faithful enough to drive the real dispatcher AND
    satisfy the delete route. Sessions materialize as real JSONL files on disk
    so idle delete can unlink them; the active loop task models slot ownership.

    ``hold_first`` keeps item 1's loop task blocked until pause terminates it,
    so the journey can observe a genuinely RUNNING item.
    """

    def __init__(self, sessions_dir: Path, *, hold_first: bool):
        self._sessions_dir = sessions_dir
        self._sessions_dir.mkdir(parents=True, exist_ok=True)
        self._counter = 0
        self._loops: dict[str, _ScriptedLoop] = {}
        self._tasks: dict[str, asyncio.Task] = {}
        self._hold_first = hold_first
        self._first_release = asyncio.Event()
        self._injects = 0

    # -- onboarding / slot ------------------------------------------------
    def is_onboarding_complete(self, project_id):
        return True

    def current_holder_session_id(self, project_id):
        """Holder == the session whose loop task is still alive (real-shaped:
        condition 1 of the production accessor)."""
        for sid, task in self._tasks.items():
            if task is not None and not task.done():
                return sid
        return None

    # -- session lifecycle ------------------------------------------------
    async def new_session(self, project_id, *, session_id=None):
        self._counter += 1
        uuid = f"journey_{self._counter:04d}"
        # Materialize a real session JSONL on disk.
        p = self._sessions_dir / f"{uuid}.jsonl"
        rows = [
            {"role": "meta", "event": "session_start", "session_id": uuid,
             "session_uuid": uuid, "timestamp": "2026-06-08T00:00:00+00:00"},
        ]
        p.write_text("\n".join(json.dumps(r) for r in rows) + "\n",
                     encoding="utf-8")
        return {"status": "ok", "session_id": uuid, "session_uuid": uuid}

    def get_loop(self, project_id, *, session_id=None):
        return self._loops.get(session_id)

    def get_loop_task(self, project_id, *, session_id=None):
        return self._tasks.get(session_id)

    def get_sub_agent_manager(self):
        return None

    async def inject_message(self, project_id, content, *, nonce=None,
                             session_id=None, queue_state="chat"):
        self._injects += 1
        # First dispatched item is held open until pause terminates it.
        hold = self._hold_first and self._injects == 1
        release = self._first_release if hold else asyncio.Event()
        loop = _ScriptedLoop(release)
        self._loops[session_id] = loop

        async def _run():
            if hold:
                # Block until terminate() (pause) releases us.
                await release.wait()
            # Non-held items exit immediately as a clean completion.
            return None

        self._tasks[session_id] = asyncio.create_task(_run())
        return "delivered"

    async def delete_session(self, project_id, session_id):
        p = self._sessions_dir / f"{session_id}.jsonl"
        if not p.exists():
            raise FileNotFoundError(session_id)
        p.unlink()
        # Drop any in-memory handles so the slot is clean.
        self._loops.pop(session_id, None)
        self._tasks.pop(session_id, None)
        return {"status": "deleted", "session_id": session_id}


def _make_app(project_store, agent_manager):
    from agent_os.api.routes import agents_v2
    app = FastAPI()
    agents_v2.configure(
        project_store=project_store,
        agent_manager=agent_manager,
        ws_manager=MagicMock(),
        sub_agent_manager=MagicMock(),
        setup_engine=MagicMock(),
        settings_store=MagicMock(),
        credential_store=MagicMock(),
    )
    app.include_router(agents_v2.router)
    return TestClient(app)


async def _wait_until(predicate, timeout: float = 5.0, interval: float = 0.02):
    deadline = asyncio.get_event_loop().time() + timeout
    while asyncio.get_event_loop().time() < deadline:
        if predicate():
            return True
        await asyncio.sleep(interval)
    return False


@pytest.mark.asyncio
async def test_delete_idle_only_journey(tmp_path):
    workspace = tmp_path / "workspace"
    sessions_dir = workspace / "orbital" / "sessions"
    queue_file = workspace / "orbital" / "queue.json"
    sessions_dir.mkdir(parents=True, exist_ok=True)

    store = QueueStore(queue_file)
    item_a = store.add_item("first item (will run, then idle, then deleted)")
    item_b = store.add_item("second item")
    item_c = store.add_item("third item")

    pid = "proj_journey"
    mgr = _JourneyManager(sessions_dir, hold_first=True)

    # The route resolves the store through agent_manager.get_queue_store and the
    # project through project_store. Wire both to our real store + workspace.
    mgr.get_queue_store = MagicMock(return_value=store)  # type: ignore[attr-defined]
    project_store = MagicMock()
    project_store.get_project = MagicMock(
        return_value={"workspace": str(workspace)}
    )

    dispatcher = QueueDispatcher(
        project_id=pid, store=store, agent_manager=mgr,
        workspace=str(workspace),
    )
    client = _make_app(project_store, mgr)

    try:
        # --- start queue; item A rotates to RUNNING and stays held ---------
        await dispatcher.start()
        dispatcher.notify_new_item()

        ran = await _wait_until(lambda: (
            store.load().items[0].state == ItemState.RUNNING
            and bool(store.load().items[0].attempts)
            and mgr.current_holder_session_id(pid) is not None
        ))
        assert ran, "item A should be RUNNING and hold the slot"

        running_item = store.load().items[0]
        assert running_item.id == item_a.id
        holder_sid = running_item.attempts[-1].session_id
        assert mgr.current_holder_session_id(pid) == holder_sid
        held_jsonl = sessions_dir / f"{holder_sid}.jsonl"
        assert held_jsonl.exists()

        # --- DELETE the RUNNING item → 409, zero mutation -----------------
        resp = client.delete(f"/api/v2/projects/{pid}/queue/items/{item_a.id}")
        assert resp.status_code == 409, resp.text

        after = store.load()
        surviving = [it for it in after.items if it.id == item_a.id]
        assert len(surviving) == 1, "running item must NOT be removed"
        assert surviving[0].attempts[-1].outcome is None, \
            "running item's attempt must stay open (no CANCELLED stamp)"
        assert held_jsonl.exists(), "running item's session JSONL must survive"
        # Queue not stalled by a phantom: head is the genuine RUNNING item,
        # not a zombie, and the slot is still held by it.
        assert after.items[0].id == item_a.id
        assert mgr.current_holder_session_id(pid) == holder_sid

        # --- pause: terminate the held loop → item A goes idle ------------
        await dispatcher.stop()
        released = await _wait_until(
            lambda: mgr.current_holder_session_id(pid) is None
        )
        assert released, "pause must release the slot (no holder)"
        assert store.load().state == QueueRunState.PAUSED

        # --- DELETE the now-idle item → 200, JSONL + record gone ----------
        resp = client.delete(f"/api/v2/projects/{pid}/queue/items/{item_a.id}")
        assert resp.status_code == 200, resp.text
        assert resp.json()["status"] == "removed"
        assert not held_jsonl.exists(), "idle delete must remove session JSONL"
        assert all(it.id != item_a.id for it in store.load().items), \
            "idle delete must remove the record"

        # --- resume: queue advances B and C to completion, no orphan ------
        mgr._hold_first = False  # remaining items run to clean completion
        await dispatcher.resume()
        dispatcher.notify_new_item()

        done = await _wait_until(lambda: all(
            it.state == ItemState.DONE
            for it in store.load().items
            if it.id in (item_b.id, item_c.id)
        ), timeout=10.0)
        if not done:
            final = store.load()
            pytest.fail(
                "queue did not advance to completion after idle delete. Final: "
                + ", ".join(f"{it.id}={it.state.value}" for it in final.items)
            )

        final = store.load()
        remaining_ids = {it.id for it in final.items}
        assert item_a.id not in remaining_ids
        assert remaining_ids == {item_b.id, item_c.id}
        for it in final.items:
            assert it.state == ItemState.DONE, f"{it.id} not DONE: {it.state}"

        # No orphaned holder after completion.
        assert mgr.current_holder_session_id(pid) is None
    finally:
        await dispatcher.shutdown()
