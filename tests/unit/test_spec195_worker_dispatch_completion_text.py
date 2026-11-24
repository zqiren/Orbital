# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Spec 195: a worker-dispatched child's reply is folded inside its card.

The chat no longer shows the child's reply as a bubble when another worker
dispatched it (``initiator="worker:<parent>"``), so nothing the daemon writes
for that completion may say the user already saw it: not the text the parent
worker reads as its next turn, and not the row that lands in Orbital's
session. An Orbital dispatch still shows the reply as a bubble under its card,
so its row keeps today's words byte for byte.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from agent_os.daemon_v2.lifecycle_observer import LifecycleObserver

P, S, PARENT, CHILD = "p1", "s1", "claude-code", "codex"

_CLAIMS = ("already see", "already sees", "already seen", "can already see")


def _observer():
    am = MagicMock()
    am.inject_system_message = AsyncMock(return_value="delivered")
    am.has_running_children = MagicMock(return_value=False)
    obs = LifecycleObserver(am, MagicMock())
    obs.worker_terminal_hook = MagicMock()
    return obs, am


def _row(am):
    args, kwargs = am.inject_system_message.call_args
    return args[1], kwargs.get("meta") or {}


@pytest.mark.asyncio
async def test_the_parent_is_not_told_the_user_saw_the_childs_reply():
    obs, _ = _observer()
    obs.set_dispatch_initiator(P, CHILD, f"worker:{PARENT}", session_id=S)
    await obs.on_completed(P, CHILD, "Wrote hello.txt.", "t.jsonl", session_id=S)
    content = obs.worker_terminal_hook.call_args.kwargs["content"]
    assert not any(c in content.lower() for c in _CLAIMS), content
    assert content == (
        "Your dispatched worker codex finished. Its reply is in the conversation "
        "above (labelled assistant). The user has not seen it, so after checking "
        "the work, tell the user what it did or found.")


@pytest.mark.asyncio
async def test_orbitals_row_for_a_worker_dispatched_child_makes_no_such_claim():
    obs, am = _observer()
    obs.set_dispatch_initiator(P, CHILD, f"worker:{PARENT}", session_id=S)
    await obs.on_completed(P, CHILD, "Wrote hello.txt.", "t.jsonl", session_id=S)
    content, meta = _row(am)
    assert not any(c in content.lower() for c in _CLAIMS), content
    assert content.endswith(
        " claude-code dispatched this task and was given its result to report "
        "to the user; the user has not seen the result in chat.")
    # The display split and the stored reply are untouched.
    assert meta["display_content"] == (
        "[Sub-agent] codex completed. Summary: Wrote hello.txt.. Transcript: t.jsonl.")
    assert meta["reply"] == "Wrote hello.txt."
    assert meta["suppress_wake"] is True


@pytest.mark.asyncio
async def test_a_worker_dispatched_child_without_a_reply_gets_the_same_row_note():
    obs, am = _observer()
    obs.set_dispatch_initiator(P, CHILD, f"worker:{PARENT}", session_id=S)
    await obs.on_completed(P, CHILD, "", "t.jsonl", session_id=S)
    content, _ = _row(am)
    assert "tell the user the outcome yourself" not in content
    assert content.endswith("the user has not seen the result in chat.")


@pytest.mark.parametrize("initiator", ["management_agent", "user_pinned", "queue_item"])
@pytest.mark.asyncio
async def test_a_non_worker_dispatch_keeps_todays_row_text(initiator):
    obs, am = _observer()
    obs.set_dispatch_initiator(P, CHILD, initiator, session_id=S)
    await obs.on_completed(P, CHILD, "Wrote hello.txt.", "t.jsonl", session_id=S)
    content, _ = _row(am)
    assert content == (
        "[Sub-agent] codex completed. Summary: Wrote hello.txt.. Transcript: t.jsonl."
        " The user can already see this summary in chat as the sub-agent's "
        "own message — do NOT repeat or re-summarize it. Verify the work if "
        "needed, then continue or reply only with what's new.")
    obs.worker_terminal_hook.assert_not_called()


@pytest.mark.asyncio
async def test_an_orbital_dispatch_without_a_reply_keeps_todays_row_text():
    obs, am = _observer()
    obs.set_dispatch_initiator(P, CHILD, "management_agent", session_id=S)
    await obs.on_completed(P, CHILD, "", "t.jsonl", session_id=S)
    content, _ = _row(am)
    assert content == (
        "[Sub-agent] codex completed. Summary: (no output). Transcript: t.jsonl."
        " The sub-agent produced no final message, so nothing was shown to "
        "the user — briefly tell the user the outcome yourself.")
