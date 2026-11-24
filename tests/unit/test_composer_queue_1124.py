# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Composer queue block (2026-11-24).

Queued messages live on the composer, not in the chat, until they are really
sent. The daemon is the authority on each one:

1. A message queued behind this chat's own agent (``session._queue``) keeps
   the words the user typed and its attachments, so a reload rebuilds the row
   and an edit restores its chips. ``send now`` moves it to the front and
   stops the turn; the stop's own drain sends it first.
2. The rows stay truthful: a turn that ends without draining (error, task
   cancelled) does not report its leftover messages as sent; a teardown that
   drops them says they were dropped.
3. A message to an outside agent that waits behind its current task is held
   back: its chat row is written when it is dispatched, not when it is typed.
   It can be removed one at a time, and ``send now`` stops that agent's task
   and delivers it first. Dropped with the queue (the agent stopped), the row
   is still written, followed by the existing "not sent" marker.
"""

from __future__ import annotations

from collections import deque
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent_os.agent.session import Session
from agent_os.daemon_v2.models import make_session_key
from agent_os.daemon_v2.sub_agent_manager import HeldUserRow, SubAgentManager

from tests.unit.test_spec195_parallel_cards_backend import (
    _ProcessManager, _Transport, _observer,
)

P, S = "p1", "s1"


# ---------------------------------------------------------------------------
# 1. Session queue keeps the typed words and attachments
# ---------------------------------------------------------------------------


def _session(tmp_path) -> Session:
    return Session.new(S, str(tmp_path))


def test_queue_meta_is_kept_and_cleared(tmp_path):
    s = _session(tmp_path)
    att = [{"path": "uploads/a.png", "mime": "image/png", "size": 3}]
    s.queue_message("<attached_files>…</attached_files>\n\nhi", nonce="n1",
                    meta={"text": "hi", "attachments": att})
    s.queue_message("plain", nonce="n2")
    meta = s.queued_meta("n1")
    assert meta["text"] == "hi" and meta["attachments"] == att
    # Every queued message records when it queued (the block's row order).
    assert meta["queued_at"] and s.queued_meta("n2")["queued_at"]
    assert s.remove_queued_message("n1") is not None
    assert s.queued_meta("n1") is None
    s.queue_message("again", nonce="n3", meta={"text": "again"})
    s.pop_queued_messages()
    assert s.queued_meta("n3") is None


def test_move_queued_to_front(tmp_path):
    s = _session(tmp_path)
    for n in ("a", "b", "c"):
        s.queue_message(f"msg {n}", nonce=n)
    assert s.move_queued_to_front("c") is True
    assert [n for _c, n in s.list_queued()] == ["c", "a", "b"]
    assert s.move_queued_to_front("zzz") is False
    assert [n for _c, n in s.list_queued()] == ["c", "a", "b"]


# ---------------------------------------------------------------------------
# AgentManager harness (no loop; the handle is a namespace)
# ---------------------------------------------------------------------------


def _agent_manager(tmp_path, *, running: bool):
    from agent_os.daemon_v2.agent_manager import AgentManager

    mgr = AgentManager.__new__(AgentManager)
    mgr._handles = {}
    mgr._broadcasts = []
    mgr._broadcast = lambda pid, payload, session_id=None: mgr._broadcasts.append(payload)
    session = _session(tmp_path)
    task = MagicMock()
    task.done.return_value = not running
    handle = SimpleNamespace(session=session, task=task)
    mgr._handles[make_session_key(P, S)] = handle
    sam = SubAgentManager(_ProcessManager())
    mgr._sub_agent_manager = sam
    return mgr, session


def _events(mgr, kind):
    return [b for b in mgr._broadcasts if b.get("type") == kind]


def test_list_pending_carries_text_and_attachments(tmp_path):
    mgr, session = _agent_manager(tmp_path, running=True)
    att = [{"path": "uploads/a.png", "mime": "image/png", "size": 3}]
    session.queue_message("wire", nonce="n1", meta={"text": "hi", "attachments": att})
    session.queue_message("old style", nonce="n2")
    out = mgr.list_pending(P)
    q1 = out[0].pop("queued_at")
    q2 = out[1].pop("queued_at")
    assert q1 <= q2
    assert out[0] == {"session_id": S, "nonce": "n1", "content": "wire", "position": 1,
                      "kind": "same", "text": "hi", "attachments": att}
    # Without composer meta: the old shape plus only the queued time.
    assert out[1] == {"session_id": S, "nonce": "n2", "content": "old style",
                      "position": 2, "kind": "same"}


@pytest.mark.asyncio
async def test_send_now_while_running_moves_it_first_and_stops_the_turn(tmp_path):
    mgr, session = _agent_manager(tmp_path, running=True)
    for n in ("a", "b", "c"):
        session.queue_message(f"msg {n}", nonce=n)
    mgr.cancel_message = AsyncMock(return_value={"status": "cancelled"})
    res = await mgr.send_queued_now(P, S, nonce="b")
    assert res == {"status": "sent", "sent": True}
    assert [n for _c, n in session.list_queued()] == ["b", "a", "c"]
    mgr.cancel_message.assert_awaited_once_with(P, session_id=S)


@pytest.mark.asyncio
async def test_send_now_of_a_message_already_gone_does_nothing(tmp_path):
    mgr, session = _agent_manager(tmp_path, running=True)
    session.queue_message("msg a", nonce="a")
    mgr.cancel_message = AsyncMock()
    res = await mgr.send_queued_now(P, S, nonce="gone")
    assert res == {"status": "gone", "sent": False}
    mgr.cancel_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_send_now_when_no_turn_runs_delivers_it(tmp_path):
    # e.g. the turn ended on an error and left the queue in place.
    mgr, session = _agent_manager(tmp_path, running=False)
    session.queue_message("msg a", nonce="a")
    session.queue_message("msg b", nonce="b")
    mgr.inject_message = AsyncMock(return_value="delivered")
    res = await mgr.send_queued_now(P, S, nonce="b")
    assert res == {"status": "sent", "sent": True}
    mgr.inject_message.assert_awaited_once_with(P, "msg b", nonce="b", session_id=S)
    # Its row leaves the block; the other stays queued (the new turn drains it).
    assert [e["nonce"] for e in _events(mgr, "chat.pending_dispatched")] == ["b"]
    assert [n for _c, n in session.list_queued()] == ["a"]


def test_leftover_after_an_undrained_end_is_not_reported_sent(tmp_path):
    mgr, session = _agent_manager(tmp_path, running=False)
    session.queue_message("msg a", nonce="a")
    mgr._clear_same_session_pending(P, S)
    assert _events(mgr, "chat.pending_dispatched") == []
    assert _events(mgr, "chat.pending_cancelled") == []
    assert [n for _c, n in session.list_queued()] == ["a"]


def test_leftover_dropped_with_the_handle_is_reported_dropped(tmp_path):
    mgr, session = _agent_manager(tmp_path, running=False)
    session.queue_message("msg a", nonce="a")
    mgr._drop_same_session_pending(P, S)
    assert [e["nonce"] for e in _events(mgr, "chat.pending_cancelled")] == ["a"]
    assert _events(mgr, "chat.pending_dispatched") == []


# ---------------------------------------------------------------------------
# 3. Outside agent: held-back rows
# ---------------------------------------------------------------------------


def _held(text="user msg", nonce="nz"):
    persisted: list[dict] = []

    async def persist(row):
        persisted.append(dict(row))

    row = {"role": "user", "content": text, "target": "codex",
           "timestamp": "2026-01-01T00:00:00+00:00", "nonce": nonce}
    return HeldUserRow(row=row, persist_fn=persist, text=text), persisted


def _sam(observer=None):
    pm = _ProcessManager()
    mgr = SubAgentManager(pm, lifecycle_observer=observer)
    transport = _Transport()
    mgr._adapters[make_session_key(P, S)] = {
        "codex": SimpleNamespace(_transport=transport, _broken=False)}
    return mgr, pm, transport


@pytest.mark.asyncio
async def test_held_row_waits_for_dispatch_then_lands_before_it():
    obs, am, _ws = _observer()
    mgr, pm, transport = _sam(obs)
    await mgr.send(P, "codex", "Orbital's task", session_id=S, dispatch_id="d1")
    held, persisted = _held()
    order: list[str] = []
    real_persist = held.persist_fn

    async def tracking(row):
        order.append("persist")
        await real_persist(row)
    held.persist_fn = tracking
    orig_dispatch = transport.dispatch

    async def tracking_dispatch(message):
        order.append("dispatch")
        await orig_dispatch(message)
    transport.dispatch = tracking_dispatch

    await mgr.send(P, "codex", "user msg", session_id=S, dispatch_id="d2",
                   initiator="user_pinned", held_row=held)
    assert mgr.queued_position(P, S, "codex", "d2") == 1
    assert persisted == []  # not in the chat while it waits
    assert mgr.holds_user_rows(P, S, "codex") is True

    await mgr._on_prompt_turn_closed(P, "codex", session_id=S, cause="completed")
    assert order == ["persist", "dispatch"]
    assert persisted[0]["content"] == "user msg"
    # Stamped when it went out, so it sits after what the agent said before.
    assert persisted[0]["timestamp"] != "2026-01-01T00:00:00+00:00"


@pytest.mark.asyncio
async def test_held_row_of_an_immediate_send_is_written_first():
    mgr, _pm, transport = _sam()
    held, persisted = _held()
    await mgr.send(P, "codex", "user msg", session_id=S, dispatch_id="d1",
                   initiator="user_pinned", held_row=held)
    assert len(persisted) == 1 and held.persisted is True
    assert transport.messages  # and it went out


@pytest.mark.asyncio
async def test_remove_one_queued_prompt():
    mgr, _pm, _t = _sam()
    await mgr.send(P, "codex", "Orbital's task", session_id=S, dispatch_id="d1")
    held2, persisted2 = _held("two")
    held3, _ = _held("three")
    await mgr.send(P, "codex", "two", session_id=S, dispatch_id="d2",
                   initiator="user_pinned", held_row=held2)
    await mgr.send(P, "codex", "three", session_id=S, dispatch_id="d3",
                   initiator="user_pinned", held_row=held3)
    assert await mgr.remove_queued_prompt(P, S, "d2") is True
    assert await mgr.remove_queued_prompt(P, S, "d2") is False
    assert mgr.queued_position(P, S, "codex", "d3") == 1
    assert persisted2 == []  # removed by the user: no row, no marker


@pytest.mark.asyncio
async def test_dropped_held_row_is_written_then_marked():
    obs, am, _ws = _observer()
    mgr, _pm, _t = _sam(obs)
    await mgr.send(P, "codex", "Orbital's task", session_id=S, dispatch_id="d1")
    held, persisted = _held()
    await mgr.send(P, "codex", "user msg", session_id=S, dispatch_id="d2",
                   initiator="user_pinned", held_row=held)
    mgr._kill_confirm_and_release = AsyncMock(return_value="stopped")
    await mgr.stop(P, "codex", session_id=S)
    assert len(persisted) == 1
    kinds = [(c.kwargs.get("meta") or {}).get("kind")
             for c in am.inject_system_message.await_args_list]
    assert "queue_dropped" in kinds


@pytest.mark.asyncio
async def test_send_now_stops_the_task_and_sends_it_first():
    mgr, _pm, _t = _sam()
    await mgr.send(P, "codex", "Orbital's task", session_id=S, dispatch_id="d1")
    h2, _ = _held("two")
    h3, _ = _held("three")
    await mgr.send(P, "codex", "two", session_id=S, dispatch_id="d2",
                   initiator="user_pinned", held_row=h2)
    await mgr.send(P, "codex", "three", session_id=S, dispatch_id="d3",
                   initiator="user_pinned", held_row=h3)
    mgr.stop_for_user = AsyncMock(return_value={"status": "stopped"})
    sent: list[tuple[str, str, object]] = []

    async def fake_send(project_id, handle, message, *, session_id, dispatch_id,
                        initiator, daemon_turn=False, held_row=None):
        sent.append((dispatch_id, message, held_row))
        return "Message sent"
    mgr.send = fake_send
    assert await mgr.send_queued_now(P, S, "d3") is True
    mgr.stop_for_user.assert_awaited_once_with(P, "codex", session_id=S)
    assert [d for d, _m, _h in sent] == ["d3", "d2"]
    assert sent[0][2] is h3  # the held row travels with its prompt
    assert await mgr.send_queued_now(P, S, "nope") is False


@pytest.mark.asyncio
async def test_list_pending_includes_held_worker_rows(tmp_path):
    mgr, _session_ = _agent_manager(tmp_path, running=False)
    sam = mgr._sub_agent_manager
    sam._adapters[make_session_key(P, S)] = {
        "codex": SimpleNamespace(_transport=_Transport(), _broken=False)}
    await sam.send(P, "codex", "Orbital's task", session_id=S, dispatch_id="d1")
    held, _ = _held("wire text")
    held.text = "typed"
    held.attachments = [{"path": "uploads/a.png", "mime": "image/png", "size": 3}]
    await sam.send(P, "codex", "wire text", session_id=S, dispatch_id="d2",
                   initiator="user_pinned", held_row=held)
    out = mgr.list_pending(P)
    assert out == [{"session_id": S, "dispatch_id": "d2", "handle": "codex",
                    "nonce": "nz", "content": "wire text", "text": "typed",
                    "attachments": held.attachments, "position": 1, "kind": "worker",
                    "queued_at": "2026-01-01T00:00:00+00:00"}]


@pytest.mark.asyncio
async def test_cancel_pending_by_dispatch_id(tmp_path):
    mgr, _session_ = _agent_manager(tmp_path, running=False)
    sam = mgr._sub_agent_manager
    sam._adapters[make_session_key(P, S)] = {
        "codex": SimpleNamespace(_transport=_Transport(), _broken=False)}
    await sam.send(P, "codex", "Orbital's task", session_id=S, dispatch_id="d1")
    held, _ = _held()
    await sam.send(P, "codex", "m", session_id=S, dispatch_id="d2",
                   initiator="user_pinned", held_row=held)
    res = await mgr.cancel_pending(P, S, dispatch_id="d2")
    assert res == {"status": "cancelled", "removed": True}
    (ev,) = _events(mgr, "chat.pending_cancelled")
    assert ev["dispatch_id"] == "d2"
    assert (await mgr.cancel_pending(P, S, dispatch_id="d2"))["removed"] is False


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


from tests.unit.test_spec074_routes import (  # noqa: E402
    _make_project, _make_session, _routes_mod, _session_rows, client,  # noqa: F401
)


def _routes_with_busy_codex(monkeypatch, pid, sid):
    """A real SubAgentManager whose codex is busy with Orbital's task."""
    import asyncio

    routes = _routes_mod()
    sam = SubAgentManager(_ProcessManager())
    sam._adapters[make_session_key(pid, sid)] = {
        "codex": SimpleNamespace(_transport=_Transport(), _broken=False)}
    asyncio.run(sam.send(pid, "codex", "Orbital's task", session_id=sid,
                         dispatch_id="d-orbital"))
    monkeypatch.setattr(routes, "_sub_agent_manager", sam)
    monkeypatch.setattr(routes._agent_manager, "_sub_agent_manager", sam)
    monkeypatch.setattr(routes, "_pinned_consolidation", MagicMock())
    return sam


def test_pinned_send_behind_a_task_is_held_out_of_the_chat(client, tmp_path, monkeypatch):
    pid, ws = _make_project(client, tmp_path, "held")
    sid = _make_session(ws)
    _routes_with_busy_codex(monkeypatch, pid, sid)

    body = client.post(f"/api/v2/agents/{pid}/inject", json={
        "content": "wait your turn", "target": "codex", "session_id": sid,
        "nonce": "n-held"}).json()
    assert body["status"] == "queued_behind" and body["held"] is True
    assert not any(r.get("content") == "wait your turn" for r in _session_rows(ws, sid))

    pending = client.get(f"/api/v2/agents/{pid}/pending").json()["pending"]
    (row,) = [p for p in pending if p["kind"] == "worker"]
    assert row["dispatch_id"] == body["dispatch_id"]
    assert row["handle"] == "codex" and row["text"] == "wait your turn"
    assert row["nonce"] == "n-held"

    res = client.post(f"/api/v2/agents/{pid}/pending/cancel", json={
        "session_id": sid, "dispatch_id": body["dispatch_id"]}).json()
    assert res == {"status": "cancelled", "removed": True}
    assert not any(r.get("content") == "wait your turn" for r in _session_rows(ws, sid))
    assert [p for p in client.get(f"/api/v2/agents/{pid}/pending").json()["pending"]
            if p["kind"] == "worker"] == []


def test_pinned_send_that_spawns_keeps_writing_its_row_first(client, tmp_path, monkeypatch):
    routes = _routes_mod()
    sam = MagicMock()
    sam.send = AsyncMock(return_value="Message sent to codex. Transcript: /t")
    sam.queued_position = MagicMock(return_value=None)
    sam.holds_user_rows = MagicMock(return_value=False)
    monkeypatch.setattr(routes, "_sub_agent_manager", sam)
    monkeypatch.setattr(routes, "_pinned_consolidation", MagicMock())
    pid, ws = _make_project(client, tmp_path, "spawn")
    sid = _make_session(ws)
    client.post(f"/api/v2/agents/{pid}/inject", json={
        "content": "start up", "target": "codex", "session_id": sid})
    assert any(r.get("content") == "start up" for r in _session_rows(ws, sid))
    assert "held_row" not in sam.send.await_args.kwargs


def test_send_now_route_for_a_message_no_longer_waiting(client, tmp_path, monkeypatch):
    pid, ws = _make_project(client, tmp_path, "gone")
    sid = _make_session(ws)
    _routes_with_busy_codex(monkeypatch, pid, sid)
    assert client.post(f"/api/v2/agents/{pid}/pending/send-now", json={
        "session_id": sid, "nonce": "nope"}).json() == {"status": "gone", "sent": False}
    assert client.post(f"/api/v2/agents/{pid}/pending/send-now", json={
        "session_id": sid, "dispatch_id": "nope"}).json() == {"status": "gone", "sent": False}
