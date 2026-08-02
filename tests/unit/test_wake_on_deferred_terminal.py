# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Regression tests for backlog #23 D1 + #28: deferred terminal sub-agent
events must wake the management loop.

A terminal lifecycle event injected while the management turn is in flight is
deferred (``Session.defer_message``) for safe insertion after the current tool
batch. ``_on_loop_done`` already drains the deferred buffer when the turn
ends, but it only *appends* the messages — it never restarts the loop, unlike
the sibling queued-user-message path a few lines below it. A fast mid-turn
sub-agent terminal therefore sits silently in session history until the user
happens to send another message; the management agent never wakes to process
it (backlog #23, diagnosed live via a codex 400 the user never saw addressed).

Fix: producers stamp ``_meta = {"event": "sub_agent_terminal", "kind": ...}``
onto the injected marker, and ``_on_loop_done`` triggers the same hot-resume
path used for queued user messages when the drained batch carries that tag.

Backlog #23 covered only the negative terminals (error/failed/stopped).
Backlog #28 closes the three siblings that were left silent — ``on_completed``,
``on_turn_interrupted``, and the fanout join summary — and widens the consumer
from a kind list to the tag itself, so a producer that tags a NEW kind is woken
by construction rather than by remembering to edit ``_on_loop_done``. Both the
producer side (does the marker carry the tag, without losing the ``#24``
``display_content`` split?) and the consumer side (does the tag wake the loop?)
are asserted below.
"""

import asyncio
from unittest.mock import MagicMock, patch

import pytest

from agent_os.agent.session import Session
from agent_os.daemon_v2.agent_manager import AgentManager
from agent_os.daemon_v2.fanout import FanoutRegistry, FanoutTask
from agent_os.daemon_v2.lifecycle_observer import LifecycleObserver


# Explicit session id: the "default" sentinel is retired (seam 3 / D1); bare
# project-level calls no longer resolve to a planted handle.
SID = "sess-wake-deferred-0001"


@pytest.fixture
def manager():
    ws = MagicMock()
    ws.broadcast = MagicMock()
    project_store = MagicMock()
    sub_agent_manager = MagicMock()
    sub_agent_manager.list_active = MagicMock(return_value=[])
    activity_translator = MagicMock()
    process_manager = MagicMock()
    mgr = AgentManager(
        project_store=project_store,
        ws_manager=ws,
        sub_agent_manager=sub_agent_manager,
        activity_translator=activity_translator,
        process_manager=process_manager,
    )
    return mgr


def _handle_with_deferred(manager, deferred_messages, *, queued=None):
    session = MagicMock()
    session.is_stopped.return_value = False
    session._paused_for_approval = False
    session.pop_deferred_messages.return_value = deferred_messages
    session.pop_queued_messages.return_value = queued or []
    session.append = MagicMock()

    handle = MagicMock()
    handle.session = session
    handle.loop = MagicMock(last_llm_error=None)

    task = MagicMock()
    task.exception.return_value = None
    handle.task = task

    manager._handles[("proj_test", SID)] = handle
    return handle, task


def _terminal_marker(content: str, kind: str) -> dict:
    """Shape a deferred message the way the LifecycleObserver terminals and
    the fanout join actually stamp it (``_meta.event`` + ``_meta.kind``)."""
    return {
        "role": "system",
        "content": content,
        "source": "daemon",
        "timestamp": "2026-07-22T00:00:00+00:00",
        "_meta": {"event": "sub_agent_terminal", "kind": kind},
    }


class _RecordingManager:
    """Captures every injection exactly as ``inject_system_message`` receives
    it, so producer-side tests assert on the same ``meta`` dict the consumer
    later reads off ``_meta``."""

    def __init__(self):
        self.injections: list[tuple[str, dict]] = []

    async def inject_system_message(self, project_id, content, **kwargs):
        self.injections.append((content, kwargs))


async def _wait_until(cond, *, timeout: float = 2.0, interval: float = 0.005):
    """Poll ``cond`` — the fanout stubs below suspend for real (mirroring
    test_fanout_registry.py), so a group needs more than one loop tick to
    resolve."""
    loop = asyncio.get_event_loop()
    start = loop.time()
    while not cond():
        if loop.time() - start > timeout:
            raise AssertionError(f"condition not met within {timeout}s")
        await asyncio.sleep(interval)


def _run_callback(manager, task):
    callback = manager._on_loop_done("proj_test", session_id=SID)
    mock_future = MagicMock()
    with patch("asyncio.ensure_future", return_value=mock_future) as mock_ensure:
        callback(task)
        if mock_ensure.call_args:
            coro = mock_ensure.call_args[0][0]
            coro.close()
    return mock_ensure


@pytest.mark.parametrize(
    "kind",
    # #23's negative terminals…
    ["error", "failed", "stopped",
     # …and #28's three siblings (completed / interrupted / fanout join).
     "completed", "interrupted", "fanout_join"],
)
def test_deferred_terminal_wakes_the_loop(manager, kind):
    """Any tagged terminal deferred mid-turn must trigger the same hot-resume
    path as a queued user message."""
    handle, task = _handle_with_deferred(
        manager, [_terminal_marker(f"[Sub-agent] cursor {kind}", kind)])

    mock_ensure = _run_callback(manager, task)

    # The event content reaches the model: appended to session history...
    handle.session.append.assert_called_once()
    appended = handle.session.append.call_args[0][0]
    assert appended["content"] == f"[Sub-agent] cursor {kind}"
    # ...and the loop is woken (hot-resume), not left idle.
    mock_ensure.assert_called_once()


def test_untagged_lifecycle_marker_does_not_wake(manager):
    """Scope guard: the TAG is what wakes, not the fact that something was
    deferred. Non-terminal lifecycle chatter (a "started" marker, or the
    pre-#28 join shape that carried only display_content) is still appended
    silently — otherwise every routine notification would restart the loop."""
    started = {
        "role": "system",
        "content": "[Sub-agent] cursor started (initiated by: user).",
        "source": "daemon",
        "timestamp": "2026-07-22T00:00:00+00:00",
    }
    display_only = {
        "role": "system",
        "content": "[Fanout f] 1/1 succeeded.\n- [completed] a (worker:f-0)",
        "source": "daemon",
        "timestamp": "2026-07-22T00:00:00+00:00",
        "_meta": {"display_content": "[Fanout f] 1/1 succeeded."},
    }
    handle, task = _handle_with_deferred(manager, [started, display_only])

    mock_ensure = _run_callback(manager, task)

    assert handle.session.append.call_count == 2
    mock_ensure.assert_not_called()


def test_idle_path_unchanged_when_nothing_deferred(manager):
    """No regression: an ordinary idle turn-end (no deferred, no queued,
    no busy sub-agents) still broadcasts idle and does not spuriously wake."""
    handle, task = _handle_with_deferred(manager, [])

    mock_ensure = _run_callback(manager, task)

    handle.session.append.assert_not_called()
    mock_ensure.assert_not_called()
    manager._ws.broadcast.assert_called()
    call_args = manager._ws.broadcast.call_args[0]
    assert call_args[1]["status"] == "idle"


def test_queued_user_message_still_wakes_without_any_deferred(manager):
    """Regression guard: the pre-existing queued-user-message resume path is
    untouched by the D1 fix when there is nothing deferred at all."""
    handle, task = _handle_with_deferred(
        manager, [], queued=[("go ahead", None)])

    mock_ensure = _run_callback(manager, task)

    handle.session.append.assert_called_once()
    appended = handle.session.append.call_args[0][0]
    assert appended["role"] == "user"
    assert appended["content"] == "go ahead"
    mock_ensure.assert_called_once()


def test_deferred_terminal_not_reprocessed_on_a_second_idle_turn_end(manager):
    """No double-processing: pop_deferred_messages is destructive (already
    clears on pop), so a SECOND _on_loop_done invocation for the same handle
    — simulating the woken turn ending idle with nothing newly deferred —
    must not re-wake the loop."""
    handle, task = _handle_with_deferred(
        manager, [_terminal_marker("[Sub-agent] cursor error", "error")])

    first_ensure = _run_callback(manager, task)
    first_ensure.assert_called_once()

    # Simulate the resumed turn ending idle: the deferred buffer is now
    # empty (already popped) and nothing new is queued.
    handle.session.pop_deferred_messages.return_value = []
    handle.session.pop_queued_messages.return_value = []
    handle.session.append.reset_mock()

    second_task = MagicMock()
    second_task.exception.return_value = None
    handle.task = second_task

    second_ensure = _run_callback(manager, second_task)

    handle.session.append.assert_not_called()
    second_ensure.assert_not_called()


# ---------------------------------------------------------------------------
# Producer side (backlog #28) — the three terminals that shipped untagged.
# Each asserts the wake tag AND that nothing else about the marker moved:
# the content strings are consumed by the chat renderer's parity fixture, and
# `display_content` is #24's user-visible/agent-facing split.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_on_completed_carries_the_wake_tag_and_keeps_display_content():
    """(a) A mid-turn completion used to be appended silently."""
    agent_manager = _RecordingManager()
    observer = LifecycleObserver(agent_manager, MagicMock())

    await observer.on_completed(
        "proj_test", "cursor", "all green", "/tmp/t.jsonl", session_id=SID)

    content, kwargs = agent_manager.injections[0]
    meta = kwargs["meta"]
    assert meta["event"] == "sub_agent_terminal"
    assert meta["kind"] == "completed"
    # The #24 split survives the added keys: display_content is still the
    # clean marker, and the LLM-facing content still carries the guidance.
    assert meta["display_content"] == (
        "[Sub-agent] cursor completed. Summary: all green. "
        "Transcript: /tmp/t.jsonl."
    )
    assert content.startswith(meta["display_content"])
    assert "do NOT repeat or re-summarize" in content


@pytest.mark.asyncio
async def test_on_completed_no_output_variant_also_carries_the_wake_tag():
    """The empty-summary branch takes a different guidance string — and must
    not take a different meta."""
    agent_manager = _RecordingManager()
    observer = LifecycleObserver(agent_manager, MagicMock())

    await observer.on_completed(
        "proj_test", "cursor", "", "/tmp/t.jsonl", session_id=SID)

    content, kwargs = agent_manager.injections[0]
    assert kwargs["meta"]["event"] == "sub_agent_terminal"
    assert kwargs["meta"]["kind"] == "completed"
    assert content.startswith(kwargs["meta"]["display_content"])


@pytest.mark.asyncio
async def test_on_turn_interrupted_carries_the_wake_tag():
    """(b) The silence its own docstring calls the Piece-3 Part-C
    silent-hang class: the management session may be AWAITING this result."""
    agent_manager = _RecordingManager()
    observer = LifecycleObserver(agent_manager, MagicMock())

    await observer.on_turn_interrupted(
        "proj_test", "cursor", "/tmp/t.jsonl", session_id=SID)

    content, kwargs = agent_manager.injections[0]
    assert kwargs["meta"]["event"] == "sub_agent_terminal"
    assert kwargs["meta"]["kind"] == "interrupted"
    # Marker text unchanged (no display split on this one).
    assert content.startswith(
        "[Sub-agent] cursor was stopped before completing its current task")
    assert "display_content" not in kwargs["meta"]


def _fanout_registry(injections):
    """A registry whose collaborators suspend for real — see
    test_fanout_registry.py's module docstring on why that matters."""
    async def inject(project_id, content, **kwargs):
        await asyncio.sleep(0)
        injections.append((content, kwargs))

    async def stop_worker(project_id, handle, session_id=None):
        await asyncio.sleep(0)

    return FanoutRegistry(inject=inject, broadcast=lambda *a, **k: None,
                          stop_worker=stop_worker)


@pytest.mark.asyncio
@pytest.mark.parametrize("worker_kind", ["completed", "error"])
async def test_fanout_join_summary_carries_the_wake_tag(worker_kind):
    """(c) The join summary is the group's terminal event. The all-failed
    group (worker_kind="error") is the case that matters most: nothing else
    will ever wake the owner to report the failure."""
    injections: list[tuple[str, dict]] = []
    registry = _fanout_registry(injections)
    registry.create_group(
        "proj_test", SID,
        [FanoutTask(handle="worker:f-0", label="a", brief="x"),
         FanoutTask(handle="worker:f-1", label="b", brief="y")],
        max_runtime_s=3600,
    )
    for i in (0, 1):
        registry.absorb_terminal(
            "proj_test", f"worker:f-{i}", SID, kind=worker_kind,
            summary=f"s{i}", transcript_path=f"t{i}")

    await _wait_until(lambda: bool(injections))

    content, kwargs = injections[0]
    meta = kwargs["meta"]
    assert meta["event"] == "sub_agent_terminal"
    assert meta["kind"] == "fanout_join"
    succeeded = 2 if worker_kind == "completed" else 0
    assert content.startswith(f"[Fanout f] {succeeded}/2 succeeded.")
    # FROZEN join format + the display split are both untouched.
    assert meta["display_content"].startswith(f"[Fanout f] {succeeded}/2")
    assert "Synthesize these results" not in meta["display_content"]
    assert "Synthesize these results" in content


# ---------------------------------------------------------------------------
# Producer → real defer buffer → consumer, with no marker shapes hand-written.
# ---------------------------------------------------------------------------


def _handle_with_real_session(manager, tmp_path):
    """Plant a handle whose session is a REAL Session (so the defer buffer is
    the production one) and whose loop task reads as still running."""
    filepath = tmp_path / "proj_test_abcd1234.jsonl"
    filepath.write_text("")
    session = Session(str(filepath))
    session.session_id = SID

    task = MagicMock()
    task.done.return_value = False

    handle = MagicMock()
    handle.session = session
    handle.loop = MagicMock(last_llm_error=None)
    handle.task = task

    manager._handles[("proj_test", SID)] = handle
    return session, task


@pytest.mark.asyncio
@pytest.mark.parametrize("event", ["completed", "interrupted"])
async def test_real_defer_path_wakes_for_each_new_producer(
        manager, tmp_path, event):
    """End-to-end over the production defer path: the observer injects while
    the turn is in flight, ``inject_system_message`` defers, and the drain in
    ``_on_loop_done`` both persists the marker and hot-resumes the loop."""
    session, task = _handle_with_real_session(manager, tmp_path)
    observer = LifecycleObserver(manager, MagicMock())

    if event == "completed":
        await observer.on_completed(
            "proj_test", "cursor", "all green", "/tmp/t.jsonl",
            session_id=SID)
    else:
        await observer.on_turn_interrupted(
            "proj_test", "cursor", "/tmp/t.jsonl", session_id=SID)

    # Deferred, not appended — the turn is still in flight.
    assert session.get_messages() == []
    assert len(session._deferred_messages) == 1

    task.exception.return_value = None
    mock_ensure = _run_callback(manager, task)

    contents = [m["content"] for m in session.get_messages()]
    assert any(c.startswith("[Sub-agent] cursor") for c in contents)
    mock_ensure.assert_called_once()


@pytest.mark.asyncio
async def test_real_defer_path_wakes_for_an_all_failed_fanout_join(
        manager, tmp_path):
    """Same end-to-end, driven by the fanout registry wired to the real
    ``inject_system_message`` — every worker failed, so the join summary is
    the only thing that can tell the owner anything."""
    session, task = _handle_with_real_session(manager, tmp_path)

    async def stop_worker(project_id, handle, session_id=None):
        await asyncio.sleep(0)

    registry = FanoutRegistry(
        inject=manager.inject_system_message,
        broadcast=lambda *a, **k: None,
        stop_worker=stop_worker,
    )
    registry.create_group(
        "proj_test", SID,
        [FanoutTask(handle="worker:f-0", label="a", brief="x")],
        max_runtime_s=3600,
    )
    registry.absorb_terminal("proj_test", "worker:f-0", SID, kind="error",
                             summary="ProviderError: 429",
                             transcript_path="t0")

    await _wait_until(lambda: bool(session._deferred_messages))
    assert session.get_messages() == []

    task.exception.return_value = None
    mock_ensure = _run_callback(manager, task)

    contents = [m["content"] for m in session.get_messages()]
    assert any(c.startswith("[Fanout f] 0/1 succeeded.") for c in contents)
    mock_ensure.assert_called_once()
