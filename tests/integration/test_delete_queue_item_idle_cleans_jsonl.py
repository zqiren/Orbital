# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Integration: idle queue-item delete cleans up the bound session JSONL.

Real ``AgentManager`` (so ``delete_session`` actually unlinks the file) + real
DELETE route + real ``QueueStore`` on disk.

Locked model: DELETE IS IDLE-ONLY. When an item is not running, deleting it must
remove BOTH the store record AND the bound session JSONL(s) for each distinct
attempt session — while leaving UNRELATED sessions untouched.

Sub-agent JSONLs are intentionally out of scope: ``delete_session`` deliberately
leaves sub-agent transcripts in place (handle-keyed, shared on disk, no
per-session mapping survives restart). This test asserts a sub-agent transcript
is LEFT in place after an idle delete.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

from fastapi import FastAPI
from fastapi.testclient import TestClient

from agent_os.agent.project_paths import ProjectPaths
from agent_os.daemon_v2.agent_manager import AgentManager
from agent_os.daemon_v2.project_store import ProjectStore
from agent_os.queue.models import AttemptRecord, ItemState


def _make_store():
    tmpdir = tempfile.mkdtemp()
    store = ProjectStore(data_dir=tmpdir)
    pid = store.create_project({
        "name": "Idle Delete Project",
        "workspace": tmpdir,
        "model": "gpt-4",
        "api_key": "sk-test",
    })
    project = store.get_project(pid)
    workspace = project["workspace"]
    sessions_dir = Path(ProjectPaths(workspace).sessions_dir)
    os.makedirs(sessions_dir, exist_ok=True)
    return store, pid, workspace, sessions_dir


def _make_manager(store):
    sub_agent_manager = MagicMock()
    sub_agent_manager.stop_all = AsyncMock()
    return AgentManager(
        project_store=store,
        ws_manager=MagicMock(),
        sub_agent_manager=sub_agent_manager,
        activity_translator=MagicMock(),
        process_manager=MagicMock(),
        platform_provider=None,
        registry=MagicMock(),
        setup_engine=MagicMock(),
        settings_store=None,
        credential_store=None,
    )


def _write_session(sessions_dir, uuid, session_id, ts="2026-06-08T08:00:00+00:00"):
    p = sessions_dir / f"{uuid}.jsonl"
    rows = [
        {"role": "meta", "event": "session_start", "session_id": session_id,
         "session_uuid": uuid, "timestamp": ts},
        {"role": "user", "content": "do the thing", "session_id": session_id,
         "timestamp": ts},
        {"role": "assistant", "content": "done", "session_id": session_id,
         "timestamp": ts},
    ]
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return p


def _make_app(store, manager):
    from agent_os.api.routes import agents_v2
    app = FastAPI()
    agents_v2.configure(
        project_store=store,
        agent_manager=manager,
        ws_manager=MagicMock(),
        sub_agent_manager=MagicMock(),
        setup_engine=MagicMock(),
        settings_store=MagicMock(),
        credential_store=MagicMock(),
    )
    app.include_router(agents_v2.router)
    return TestClient(app)


def test_idle_delete_removes_bound_jsonl_and_record_only():
    store, pid, workspace, sessions_dir = _make_store()
    mgr = _make_manager(store)

    # The item's bound session, with a real JSONL on disk.
    bound_uuid = "idle_bound_aaa"
    bound_jsonl = _write_session(sessions_dir, bound_uuid, "sess_bound")

    # An UNRELATED session that must survive the delete.
    other_uuid = "other_keepme_b"
    other_jsonl = _write_session(sessions_dir, other_uuid, "sess_other")

    # A sub-agent transcript that must be LEFT in place (out of scope).
    sub_dir = Path(workspace) / "orbital" / "sub_agents" / "claude-code"
    sub_dir.mkdir(parents=True, exist_ok=True)
    sub_transcript = sub_dir / "tr_bound.jsonl"
    sub_transcript.write_text('{"role":"sub"}\n', encoding="utf-8")

    # Seed an IDLE queue item bound to the session. No live handle exists, so
    # current_holder_session_id is None → not running → idle delete.
    qstore = mgr.get_queue_store(pid, workspace=workspace)
    item = qstore.add_item("the dispatched item")
    qstore.set_item_state(item.id, ItemState.DONE)
    qstore.append_attempt(item.id, AttemptRecord(session_id=bound_uuid))

    assert mgr.current_holder_session_id(pid) is None  # nothing running

    client = _make_app(store, mgr)
    resp = client.delete(f"/api/v2/projects/{pid}/queue/items/{item.id}")
    assert resp.status_code == 200, resp.text
    assert resp.json()["status"] == "removed"

    # Bound session JSONL is gone.
    assert not bound_jsonl.exists(), "bound session JSONL must be deleted"

    # Queue record is gone.
    fresh = qstore.load()
    assert all(it.id != item.id for it in fresh.items)

    # Unrelated session JSONL is untouched.
    assert other_jsonl.exists(), "unrelated session JSONL must be preserved"

    # Sub-agent transcript is LEFT in place (intentionally out of scope).
    assert sub_transcript.exists(), \
        "sub-agent transcript must NOT be deleted by an idle queue-item delete"


def test_idle_delete_no_attempts_just_removes_record():
    """An item that was never dispatched (no attempts) has no session to clean
    up — delete just removes the record and broadcasts."""
    store, pid, workspace, sessions_dir = _make_store()
    mgr = _make_manager(store)

    qstore = mgr.get_queue_store(pid, workspace=workspace)
    item = qstore.add_item("never dispatched")  # state=QUEUED, attempts=[]

    client = _make_app(store, mgr)
    resp = client.delete(f"/api/v2/projects/{pid}/queue/items/{item.id}")
    assert resp.status_code == 200, resp.text

    fresh = qstore.load()
    assert all(it.id != item.id for it in fresh.items)


def test_idle_delete_distinct_attempt_sessions_all_cleaned():
    """An item retried across multiple distinct sessions cleans up each distinct
    session JSONL exactly once."""
    store, pid, workspace, sessions_dir = _make_store()
    mgr = _make_manager(store)

    uuid1 = "retry_one_aaaa"
    uuid2 = "retry_two_bbbb"
    j1 = _write_session(sessions_dir, uuid1, "sess_r1")
    j2 = _write_session(sessions_dir, uuid2, "sess_r2")

    qstore = mgr.get_queue_store(pid, workspace=workspace)
    item = qstore.add_item("retried item")
    qstore.set_item_state(item.id, ItemState.BLOCKED)
    qstore.append_attempt(item.id, AttemptRecord(session_id=uuid1))
    qstore.append_attempt(item.id, AttemptRecord(session_id=uuid2))

    client = _make_app(store, mgr)
    resp = client.delete(f"/api/v2/projects/{pid}/queue/items/{item.id}")
    assert resp.status_code == 200, resp.text

    assert not j1.exists(), "first attempt session JSONL must be deleted"
    assert not j2.exists(), "second attempt session JSONL must be deleted"
