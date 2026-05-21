# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Regression: dispatcher wraps queue items with a [QUEUE ITEM | id | attempt]
header before injecting them into the agent loop.

Concern 3 of TASK-queue-architecture-amendments.md: the agent must be able to
distinguish queue items (header present) from chat messages (header absent),
so _dispatch_one now prepends a header carrying the item id and the new
attempt's 1-based number.
"""

from __future__ import annotations

import asyncio

import pytest

from agent_os.queue.dispatcher import QueueDispatcher
from agent_os.queue.models import ItemState
from agent_os.queue.store import QueueStore
from tests.integration.test_queue_phase2 import _ScriptedAgentManager


class _CapturingAgentManager(_ScriptedAgentManager):
    """Scripted manager that also records every inject_message argument."""

    def __init__(self, script):
        super().__init__(script)
        self.injected: list[tuple[str, str]] = []

    async def inject_message(self, project_id, content, *, nonce=None, session_id=None):
        self.injected.append((project_id, content))
        return await super().inject_message(project_id, content, nonce=nonce, session_id=session_id)


async def _wait(predicate, timeout=10.0, interval=0.05):
    deadline = asyncio.get_event_loop().time() + timeout
    while asyncio.get_event_loop().time() < deadline:
        if predicate():
            return True
        await asyncio.sleep(interval)
    return False


@pytest.mark.asyncio
async def test_first_attempt_injection_is_wrapped_with_header(tmp_path):
    """A freshly-dispatched item lands as
    '[QUEUE ITEM | id={item.id} | attempt=1]\\n{content}'.
    """
    store = QueueStore(tmp_path / "queue.json")
    item = store.add_item("my content")

    mgr = _CapturingAgentManager([{"reason": "complete", "summary": "done"}])
    dispatcher = QueueDispatcher(
        project_id="proj_header",
        store=store,
        agent_manager=mgr,
    )
    await dispatcher.start()
    dispatcher.notify_new_item()

    ok = await _wait(lambda: len(mgr.injected) >= 1, timeout=5.0)
    if not ok:
        await dispatcher.shutdown()
        pytest.fail("inject_message was never called")

    project_id, content = mgr.injected[0]
    expected = (
        f"[QUEUE ITEM | id={item.id} | attempt=1]\n"
        + QueueDispatcher.HEADER_CONTRACT
        + "my content"
    )
    assert project_id == "proj_header"
    assert content == expected, (
        f"expected exact wrapped content\n  expected: {expected!r}\n"
        f"  actual:   {content!r}"
    )

    await dispatcher.shutdown()


@pytest.mark.asyncio
async def test_header_includes_item_id_and_attempt_number(tmp_path):
    """The header format is load-bearing for the agent contract — assert
    the exact tokens plus the embedded HEADER_CONTRACT block."""
    store = QueueStore(tmp_path / "queue.json")
    item = store.add_item("write a file")

    mgr = _CapturingAgentManager([{"reason": "complete", "summary": "ok"}])
    dispatcher = QueueDispatcher(
        project_id="proj_header_tokens",
        store=store,
        agent_manager=mgr,
    )
    await dispatcher.start()
    dispatcher.notify_new_item()

    await _wait(lambda: store.load().items[0].state == ItemState.DONE,
                timeout=10.0)

    assert len(mgr.injected) == 1
    _, content = mgr.injected[0]
    first_line, _, rest = content.partition("\n")
    assert first_line == f"[QUEUE ITEM | id={item.id} | attempt=1]"
    # Contract block sits between metadata line and user content.
    assert rest.startswith(QueueDispatcher.HEADER_CONTRACT)
    body = rest[len(QueueDispatcher.HEADER_CONTRACT):]
    assert body == "write a file"

    await dispatcher.shutdown()
