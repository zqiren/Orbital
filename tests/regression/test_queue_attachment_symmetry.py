# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Regression (Step 4): queue items carry file_refs, and a dispatched item must
get the SAME <attached_files> block a user's Send produces.

Before this fix _dispatch_one ignored item.file_refs entirely, so an item
staged with attachments reached the agent with no record of them — asymmetric
with the chat inject endpoint, which prepends the block via format_prefix.
"""

from __future__ import annotations

import asyncio
import os

import pytest

from agent_os.api.routes._attachment_formatter import format_prefix_from_refs
from agent_os.queue.dispatcher import QueueDispatcher
from agent_os.queue.models import ItemState
from agent_os.queue.store import QueueStore


class _FakeSession:
    def __init__(self, sid):
        self.session_id = sid


class _FakeLoop:
    def __init__(self):
        self._exit_reason = "complete"
        self._exit_summary = "ok"
        self._exit_block_reason = None
        self._queue_state = "chat"


class _RecordingManager:
    """Agent already running; records the content the dispatcher injects."""

    def __init__(self):
        self._loop = _FakeLoop()
        self._session_counter = 0
        self._session = _FakeSession("s")
        self._task = None
        self.injected_content: list[str] = []

    def is_onboarding_complete(self, pid):
        return True

    def has_handle(self, pid):
        return True

    def get_session(self, pid):
        return self._session

    def get_loop(self, pid, *, session_id=None):
        return self._loop

    def get_loop_task(self, pid, *, session_id=None):
        return self._task

    def get_run_status(self, pid, *, session_id=None):
        return "idle"

    async def ensure_agent_started(self, pid):
        return False

    async def inject_message(self, pid, content, *, nonce=None,
                             session_id=None, queue_state="chat"):
        self.injected_content.append(content)
        self._loop._exit_reason = "complete"

        async def _instant():
            return None

        self._task = asyncio.create_task(_instant())
        return "delivered"

    async def new_session(self, pid, *, session_id=None):
        self._session_counter += 1
        sid = f"sess_{self._session_counter}"
        self._session = _FakeSession(sid)
        self._loop._exit_reason = "text"
        return {
            "status": "ok",
            "session_id": sid,
            "session_uuid": f"proj_{self._session_counter:08d}",
        }

    def get_sub_agent_manager(self):
        return None


def test_format_prefix_from_refs_builds_attached_files_block(tmp_path):
    f = tmp_path / "report.md"
    f.write_text("hello", encoding="utf-8")
    prefix = format_prefix_from_refs(str(tmp_path), ["report.md"])
    assert prefix.startswith("<attached_files>")
    assert "- report.md (" in prefix
    assert prefix.endswith("\n\n")
    # Empty refs → empty prefix (symmetric with format_prefix([])).
    assert format_prefix_from_refs(str(tmp_path), []) == ""


@pytest.mark.asyncio
async def test_dispatched_item_with_file_refs_gets_attachment_prefix(tmp_path):
    (tmp_path / "data.csv").write_text("a,b,c\n", encoding="utf-8")
    store = QueueStore(tmp_path / "orbital" / "queue.json")
    item = store.add_item("analyze the data", file_refs=["data.csv"])

    mgr = _RecordingManager()
    dispatcher = QueueDispatcher(
        project_id="proj_attach", store=store, agent_manager=mgr,
        workspace=str(tmp_path),
    )

    await dispatcher._dispatch_one(item)

    assert mgr.injected_content, "dispatcher must have injected the item"
    content = mgr.injected_content[0]
    expected_prefix = format_prefix_from_refs(str(tmp_path), ["data.csv"])
    assert expected_prefix and expected_prefix in content, (
        "dispatched item must carry the <attached_files> block"
    )
    # The attachment block precedes the [QUEUE ITEM] contract header.
    assert content.index("<attached_files>") < content.index("[QUEUE ITEM")
    assert store.load().items[0].state == ItemState.DONE
