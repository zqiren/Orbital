# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Spec 161 §5.1: the one dispatch verb the top-of-session worker gets.

``dispatch`` is a thin wrapper over ``SubAgentManager.send``, the only
dispatch verb, exactly like the manager's ``agent_message``: it refuses the
seat-holder's own slug, sends with ``depth=1`` and ``initiator="worker:<handle>"``
(the routing key that brings the child's result back to the dispatcher), and
passes ``send``'s error strings through unchanged.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from agent_os.daemon_v2.dispatch_tools import DispatchContext, dispatch

CTX = DispatchContext(project_id="p1", session_id="s1", handle="claude-code")


def _sam(ack="Message sent to codex. Transcript: /ws/t.jsonl"):
    sam = MagicMock()
    sam.send = AsyncMock(return_value=ack)
    return sam


@pytest.mark.asyncio
async def test_own_slug_is_refused_and_nothing_is_sent():
    sam = _sam()
    out = await dispatch(sam, CTX, "claude-code", "do it")
    assert out == "You are claude-code; do this task yourself."
    sam.send.assert_not_awaited()


@pytest.mark.asyncio
async def test_send_gets_depth_one_and_the_worker_initiator():
    sam = _sam()
    await dispatch(sam, CTX, "codex", "write hello.txt")
    sam.send.assert_awaited_once_with(
        "p1", "codex", "write hello.txt", session_id="s1", depth=1,
        fresh=False, initiator="worker:claude-code")


@pytest.mark.asyncio
async def test_fresh_is_forwarded():
    sam = _sam()
    await dispatch(sam, CTX, "codex", "start over", fresh=True)
    assert sam.send.call_args.kwargs["fresh"] is True


@pytest.mark.asyncio
async def test_success_text_carries_the_ack_and_tells_the_model_to_end_its_turn():
    sam = _sam("Message sent to codex. Transcript: /ws/t.jsonl")
    out = await dispatch(sam, CTX, "codex", "write hello.txt")
    assert out == (
        "Dispatched to codex. Message sent to codex. Transcript: /ws/t.jsonl "
        "The result will arrive as your next message. End your turn now and wait for it.")


@pytest.mark.asyncio
@pytest.mark.parametrize("err", [
    "Error: unknown agent 'nope'",
    "Error: concurrent sub-agent limit reached (max 5 per project). Stop an existing sub-agent before starting a new one.",
    "Error: project is shutting down, cannot start new agents",
])
async def test_send_errors_pass_through_unchanged(err):
    out = await dispatch(_sam(err), CTX, "codex", "x")
    assert out == err
