# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""A wake that cannot start fails its queue item with the real reason
(2026-09-05).

An item held for a management turn — the verdict on a reviewed worker item, a
continuation — used to depend on that turn being startable at all. With no API
key (or one the provider rejects) ``inject_system_message`` persisted the row,
logged, and returned; nothing was told. The hold then sat out the full runtime
backstop and blocked the item as a *timeout*, which is the one thing that was
not wrong with it.

Now the injection path reports the failure to the dispatcher, which blocks the
item immediately carrying the classifier's own code — the same code chat shows
for the identical failure.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock

from agent_os.daemon_v2.provider_errors import ProviderConfigError
from agent_os.queue.dispatcher import QueueDispatcher
from agent_os.queue.models import AttemptOutcome, AttemptRecord, ItemState
from agent_os.queue.store import QueueStore


def _dispatcher_with_running_item(tmp_path, *, session_id="sess_1", agent="codex"):
    store = QueueStore(tmp_path / "queue.json")
    item = store.add_item("write hello.txt", agent=agent)
    store.set_item_state(item.id, ItemState.RUNNING)
    store.append_attempt(item.id, AttemptRecord(session_id=session_id))

    mgr = MagicMock()
    mgr.get_loop = MagicMock(return_value=None)
    mgr.get_loop_task = MagicMock(return_value=None)
    d = QueueDispatcher("proj", store, mgr, workspace=str(tmp_path))
    d.HOLD_POLL_SEC = 0.01
    return d, store, item


def test_a_wake_that_cannot_start_blocks_the_item_with_its_code(tmp_path):
    d, store, item = _dispatcher_with_running_item(tmp_path)

    assert d.on_wake_failed(
        "sess_1",
        reason="No LLM API key configured for this project or globally",
        code="missing_api_key",
    ) is True

    blocked = store.load().items[0]
    assert blocked.state == ItemState.BLOCKED
    latest = blocked.attempts[-1]
    assert latest.outcome == AttemptOutcome.BLOCKED
    # The code the frontend localizes, and the provider's own words behind it.
    assert latest.block_reason_code == "missing_api_key"
    assert "API key" in latest.block_reason
    # NOT the runtime-cap disposition the backstop would have applied.
    assert latest.block_reason_code != "runtime_cap"
    assert blocked.interrupted_count == 0


def test_it_blocks_an_unassigned_item_too(tmp_path):
    """The manager is the runner there, so a wake it cannot take is even more
    directly the item's problem."""
    d, store, _item = _dispatcher_with_running_item(tmp_path, agent=None)

    assert d.on_wake_failed("sess_1", reason="boom", code="provider_error") is True
    assert store.load().items[0].state == ItemState.BLOCKED


def test_a_wake_failure_in_another_session_is_not_ours(tmp_path):
    d, store, _item = _dispatcher_with_running_item(tmp_path, session_id="sess_1")

    assert d.on_wake_failed("sess_other", reason="boom") is False
    assert store.load().items[0].state == ItemState.RUNNING


def test_a_wake_failure_with_nothing_running_is_a_no_op(tmp_path):
    store = QueueStore(tmp_path / "queue.json")
    store.add_item("not dispatched yet")
    d = QueueDispatcher("proj", store, MagicMock(), workspace=str(tmp_path))

    assert d.on_wake_failed("sess_1", reason="boom") is False
    assert store.load().items[0].state == ItemState.QUEUED


# ---------------------------------------------------------------------------
# The injection path: it is what notices, and it must classify, not guess
# ---------------------------------------------------------------------------


def test_the_manager_classifies_the_failure_before_reporting_it():
    """``_note_wake_failed`` turns the raised exception into the same
    (code, message) pair chat surfaces, and hands it to the project's
    dispatcher."""
    from agent_os.daemon_v2.agent_manager import AgentManager

    mgr = MagicMock(spec=AgentManager)
    dispatcher = MagicMock()
    mgr._dispatchers = {"proj": dispatcher}

    AgentManager._note_wake_failed(
        mgr, "proj", "sess_1",
        ProviderConfigError("missing_api_key", "No LLM API key configured"),
    )

    dispatcher.on_wake_failed.assert_called_once()
    kwargs = dispatcher.on_wake_failed.call_args.kwargs
    assert dispatcher.on_wake_failed.call_args.args[0] == "sess_1"
    assert kwargs["code"] == "missing_api_key"
    assert "API key" in kwargs["reason"]


def test_a_project_without_a_dispatcher_is_simply_skipped():
    from agent_os.daemon_v2.agent_manager import AgentManager

    mgr = MagicMock(spec=AgentManager)
    mgr._dispatchers = {}
    # No raise, no report — a plain chat session has no item to fail.
    AgentManager._note_wake_failed(mgr, "proj", "sess_1", RuntimeError("boom"))


def test_a_reporting_failure_never_breaks_the_injection():
    from agent_os.daemon_v2.agent_manager import AgentManager

    mgr = MagicMock(spec=AgentManager)
    dispatcher = MagicMock()
    dispatcher.on_wake_failed.side_effect = RuntimeError("dispatcher is broken")
    mgr._dispatchers = {"proj": dispatcher}

    AgentManager._note_wake_failed(mgr, "proj", "sess_1", RuntimeError("boom"))


@pytest.mark.asyncio
async def test_a_live_handle_whose_wake_raises_is_reported_not_propagated(tmp_path):
    """The live-handle branch used to let the exception escape into whatever
    injected the row (the lifecycle observer), which swallowed it."""
    from agent_os.daemon_v2.agent_manager import AgentManager
    from agent_os.daemon_v2.models import make_session_key

    mgr = MagicMock(spec=AgentManager)
    session = MagicMock()
    handle = MagicMock()
    handle.task = None
    handle.session = session
    mgr._resolve_session_id = MagicMock(return_value="s1")
    mgr._handles = {make_session_key("proj", "s1"): handle}
    mgr._start_loop = AsyncMock(
        side_effect=ProviderConfigError("invalid_api_key", "provider said no"),
    )
    mgr._note_wake_failed = MagicMock()

    result = await AgentManager.inject_system_message(
        mgr, "proj", "[Sub-agent] codex completed.", session_id="s1",
        meta={"event": "sub_agent_terminal", "kind": "completed"},
    )

    assert result == "persisted"
    # The row landed first — the event is never lost to a failed wake.
    session.append.assert_called_once()
    mgr._note_wake_failed.assert_called_once()
    assert mgr._note_wake_failed.call_args.args[1] == "s1"
