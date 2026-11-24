# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Spec 195 backend: parallel outside agents shown as cards.

1. The dispatch marker records who started the dispatch (``initiator``), and
   the live worker events carry the dispatch they belong to.
2. The worker ``dispatch`` tool refuses unless the caller's CURRENT turn is a
   top-of-session one, so a worker pinned earlier cannot dispatch after
   Orbital takes it back (cards never nest).
3. A pinned send that waits behind another task returns a structured queued
   status; the ``sub_agent.dispatched`` broadcast and a new
   ``sub_agent.queue_dropped`` broadcast carry the dispatch id that clears it.
4. A worker question is suppressed for a pinned / worker-dispatched turn, and
   now wakes a mid-turn manager like a terminal does.

The "Orbital stays quiet while pinned" mechanism is in
``test_spec195_hold_while_pinned.py`` (real loop tasks).
"""

from __future__ import annotations

import os
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent_os.daemon_v2.dispatch_tools import DispatchContext, dispatch
from agent_os.daemon_v2.lifecycle_observer import LifecycleObserver
from agent_os.daemon_v2.models import make_session_key
from agent_os.daemon_v2.sub_agent_manager import SubAgentManager


P, S = "p1", "s1"


def _observer():
    am = MagicMock()
    am.inject_system_message = AsyncMock(return_value="delivered")
    ws = MagicMock()
    return LifecycleObserver(am, ws), am, ws


def _injected_meta(am) -> list[dict]:
    return [c.kwargs.get("meta") or {} for c in am.inject_system_message.await_args_list]


def _broadcasts(ws, kind: str) -> list[dict]:
    return [c.args[1] for c in ws.broadcast.call_args_list if c.args[1].get("type") == kind]


class _ProcessManager:
    def __init__(self):
        self.turn_closed = None

    def set_turn_closed_callback(self, callback):
        self.turn_closed = callback

    def set_permission_request_callback(self, callback):
        pass

    def set_active_dispatch(self, *args, **kwargs):
        pass

    def clear_dispatch(self, *args, **kwargs):
        pass

    async def stop(self, *args, **kwargs):
        return None


class _Transport:
    def __init__(self):
        self.messages = []

    async def dispatch(self, message):
        self.messages.append(message)


def _manager_with(observer, *handles):
    pm = _ProcessManager()
    mgr = SubAgentManager(pm, lifecycle_observer=observer)
    mgr._adapters[make_session_key(P, S)] = {
        h: SimpleNamespace(_transport=_Transport(), _broken=False) for h in handles}
    return mgr, pm


# ---------------------------------------------------------------------------
# 1. initiator on the marker
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("initiator", [
    "management_agent", "user_pinned", "queue_item", "worker:claude-code"])
async def test_marker_records_its_initiator(initiator):
    obs, am, ws = _observer()
    await obs.on_message_routed(
        P, "codex", initiator=initiator, message_preview="do it",
        transcript_path="/t.jsonl", session_id=S, dispatch_id="d1")
    (meta,) = _injected_meta(am)
    assert meta["initiator"] == initiator
    assert meta["dispatch_id"] == "d1"
    # Unchanged: only the manager's own dispatch stays wake-eligible.
    assert bool(meta.get("suppress_wake")) is (initiator != "management_agent")


@pytest.mark.asyncio
async def test_marker_without_dispatch_id_still_records_initiator():
    obs, am, _ws = _observer()
    await obs.on_message_routed(
        P, "codex", initiator="management_agent", message_preview="x",
        transcript_path="/t.jsonl", session_id=S)
    (meta,) = _injected_meta(am)
    assert meta == {"initiator": "management_agent"}


@pytest.mark.asyncio
async def test_dispatched_broadcast_carries_the_dispatch_id():
    obs, _am, ws = _observer()
    await obs.on_message_routed(
        P, "codex", initiator="user_pinned", message_preview="x",
        transcript_path="/t.jsonl", session_id=S, dispatch_id="d7")
    (event,) = _broadcasts(ws, "sub_agent.dispatched")
    assert event["dispatch_id"] == "d7"
    assert event["preview"] == "x"  # the card title, as the marker row stores it
    assert event["handle"] == "codex"
    assert event["initiator"] == "user_pinned"


def test_dispatch_initiator_names_the_worker_parent():
    obs, _am, _ws = _observer()
    obs.set_dispatch_initiator(P, "codex", "worker:claude-code", session_id=S)
    assert obs.dispatch_initiator(P, "codex", S) == "worker:claude-code"
    obs.set_dispatch_initiator(P, "codex", "user_pinned", session_id=S)
    assert obs.dispatch_initiator(P, "codex", S) == "user_pinned"
    obs.set_dispatch_initiator(P, "codex", "management_agent", session_id=S)
    assert obs.dispatch_initiator(P, "codex", S) == "management_agent"
    assert obs.dispatch_initiator(P, "never-seen", S) is None


# ---------------------------------------------------------------------------
# 1b. live worker events carry dispatch_id + initiator
# ---------------------------------------------------------------------------


def test_live_worker_tool_and_thinking_events_carry_the_dispatch():
    from agent_os.daemon_v2.activity_translator import ActivityTranslator
    ws = MagicMock()
    tr = ActivityTranslator(ws)
    tr.on_worker_chunk("tool_activity", {"tool_name": "Bash", "tool_call_id": "t1",
                                         "arguments": {"command": "ls"}},
                       "", P, session_id=S, handle="codex",
                       dispatch_id="d1", initiator="management_agent")
    tr.on_worker_chunk("thinking", {}, "hmm", P, session_id=S, handle="codex",
                       dispatch_id="d1", initiator="management_agent")
    tr.on_worker_turn_closed(P, session_id=S, handle="codex")
    events = [c.args[1] for c in ws.broadcast.call_args_list]
    tool = [e for e in events if e.get("worker_event") == "tool_call"]
    think = [e for e in events if e.get("type") == "chat.stream_delta" and e.get("worker")]
    assert tool and think
    for e in tool + think:
        assert e["dispatch_id"] == "d1"
        assert e["initiator"] == "management_agent"


def test_live_worker_events_without_a_dispatch_carry_no_ids():
    """Spec 129 self-initiated turns: no dispatch, so nothing routes them to
    a card (they render inline, as today)."""
    from agent_os.daemon_v2.activity_translator import ActivityTranslator
    ws = MagicMock()
    tr = ActivityTranslator(ws)
    tr.on_worker_chunk("tool_activity", {"tool_name": "Bash", "tool_call_id": "t1"},
                       "", P, session_id=S, handle="claude-code")
    events = [c.args[1] for c in ws.broadcast.call_args_list]
    assert events and all("dispatch_id" not in e and "initiator" not in e for e in events)


# ---------------------------------------------------------------------------
# 2. the dispatch tool needs a top-of-session turn NOW
# ---------------------------------------------------------------------------


CTX = DispatchContext(project_id=P, session_id=S, handle="codex",
                      agents=("codex", "claude-code"))


@pytest.mark.asyncio
async def test_pinned_worker_can_dispatch():
    obs, _am, _ws = _observer()
    obs.set_dispatch_initiator(P, "codex", "user_pinned", session_id=S)
    sam = MagicMock()
    sam._lifecycle_observer = obs
    sam.send = AsyncMock(return_value="Message sent to claude-code. Transcript: /t")
    out = await dispatch(sam, CTX, "claude-code", "write a.txt")
    assert out.startswith("Dispatched to claude-code.")
    sam.send.assert_awaited_once()


@pytest.mark.asyncio
async def test_worker_taken_back_by_orbital_cannot_dispatch_through_its_old_tool():
    """The critique's path: Codex was pinned (spawned with the tool), the user
    switched back to Orbital, Orbital dispatched the same warm Codex. Its
    token still lists ``dispatch``; the call is refused."""
    obs, _am, _ws = _observer()
    mgr, _pm = _manager_with(obs, "codex")
    await mgr.send(P, "codex", "pinned chat", session_id=S, initiator="user_pinned")
    await mgr._on_prompt_turn_closed(P, "codex", session_id=S, cause="success")
    await mgr.send(P, "codex", "Orbital's task", session_id=S,
                   initiator="management_agent")

    mgr.send = AsyncMock()
    out = await dispatch(mgr, CTX, "claude-code", "nested task")
    assert out.startswith("Error: this agent cannot hand tasks to other agents")
    mgr.send.assert_not_awaited()


@pytest.mark.asyncio
async def test_worker_dispatched_child_cannot_dispatch():
    obs, _am, _ws = _observer()
    obs.set_dispatch_initiator(P, "codex", "worker:claude-code", session_id=S)
    sam = MagicMock()
    sam._lifecycle_observer = obs
    sam.send = AsyncMock()
    out = await dispatch(sam, CTX, "claude-code", "x")
    assert out.startswith("Error: this agent cannot hand tasks to other agents")
    sam.send.assert_not_awaited()


@pytest.mark.asyncio
async def test_queue_item_worker_can_dispatch():
    obs, _am, _ws = _observer()
    obs.set_dispatch_initiator(P, "codex", "queue_item", session_id=S)
    sam = MagicMock()
    sam._lifecycle_observer = obs
    sam.send = AsyncMock(return_value="Message sent to claude-code. Transcript: /t")
    out = await dispatch(sam, CTX, "claude-code", "x")
    assert out.startswith("Dispatched to")


# ---------------------------------------------------------------------------
# 3. structured queued status and its clear signals
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_queued_position_finds_a_waiting_dispatch():
    obs, _am, _ws = _observer()
    mgr, _pm = _manager_with(obs, "codex")
    await mgr.send(P, "codex", "Orbital's task", session_id=S, dispatch_id="d1")
    assert mgr.queued_position(P, S, "codex", "d1") is None  # running, not waiting
    await mgr.send(P, "codex", "user msg", session_id=S, dispatch_id="d2",
                   initiator="user_pinned")
    await mgr.send(P, "codex", "user msg 2", session_id=S, dispatch_id="d3",
                   initiator="user_pinned")
    assert mgr.queued_position(P, S, "codex", "d2") == 1
    assert mgr.queued_position(P, S, "codex", "d3") == 2
    assert mgr.queued_position(P, S, "codex", "nope") is None
    assert mgr.queued_position(P, S, "claude-code", "d2") is None


@pytest.mark.asyncio
async def test_queue_drop_broadcasts_its_dispatch_id():
    obs, am, ws = _observer()
    await obs.on_queue_dropped(P, "codex", why="stopped by the user",
                               session_id=S, initiator="user_pinned",
                               dispatch_id="d2")
    (event,) = _broadcasts(ws, "sub_agent.queue_dropped")
    assert event == {"type": "sub_agent.queue_dropped", "project_id": P,
                     "session_id": S, "handle": "codex", "dispatch_id": "d2",
                     "why": "stopped by the user"}
    # The durable row is unchanged.
    (meta,) = _injected_meta(am)
    assert meta["kind"] == "queue_dropped"


@pytest.mark.asyncio
async def test_manager_passes_the_dropped_prompts_dispatch_id():
    obs = MagicMock()
    obs.on_queue_dropped = AsyncMock()
    mgr = SubAgentManager(_ProcessManager(), lifecycle_observer=obs)
    dropped = [SimpleNamespace(message="m", initiator="user_pinned", dispatch_id="d9")]
    await mgr._mark_queued_prompts_dropped(dropped, P, "codex", session_id=S, why="w")
    assert obs.on_queue_dropped.await_args.kwargs["dispatch_id"] == "d9"


# ---------------------------------------------------------------------------
# 4. worker questions
# ---------------------------------------------------------------------------


async def _ask(obs):
    await obs.on_interaction_required(
        P, "dsh", interaction_id="i1", kind="question", prompt="Which file?",
        session_id=S)


@pytest.mark.asyncio
async def test_question_from_orbitals_dispatch_stays_wake_eligible():
    obs, am, _ws = _observer()
    obs.set_dispatch_initiator(P, "dsh", "management_agent", session_id=S)
    await _ask(obs)
    (meta,) = _injected_meta(am)
    assert meta["event"] == "interaction_required"
    assert "suppress_wake" not in meta


@pytest.mark.asyncio
@pytest.mark.parametrize("initiator", ["user_pinned", "worker:claude-code"])
async def test_question_from_a_pinned_or_worker_dispatched_turn_is_suppressed(initiator):
    obs, am, _ws = _observer()
    obs.set_dispatch_initiator(P, "dsh", initiator, session_id=S)
    await _ask(obs)
    (meta,) = _injected_meta(am)
    assert meta["suppress_wake"] is True


def test_question_is_a_wake_event():
    from agent_os.daemon_v2 import agent_manager
    assert "interaction_required" in agent_manager._WAKE_EVENTS


# ---------------------------------------------------------------------------
# Routes: the pinned send's queued status, the unpin wake
# ---------------------------------------------------------------------------


from tests.unit.test_spec074_routes import (  # noqa: E402
    _make_project, _make_session, _routes_mod, client,  # noqa: F401 (fixture)
)


class _QueueingSAM:
    """A worker busy with Orbital's task: the user's send waits in its queue
    (the real ``queued_position`` over a real ``_prompt_queues``)."""

    def __init__(self):
        self.mgr = SubAgentManager(_ProcessManager())

    async def send(self, project_id, handle, message, *, session_id, dispatch_id,
                   initiator, **kwargs):
        key = (project_id, session_id, handle)
        from collections import deque
        q = self.mgr._prompt_queues.setdefault(key, deque())
        q.append(SimpleNamespace(dispatch_id=dispatch_id, initiator=initiator))
        return f"Message queued for {handle} (position {len(q)}). Transcript: /t"

    def queued_position(self, *args):
        return self.mgr.queued_position(*args)


def test_pinned_send_waiting_behind_a_task_returns_queued_behind(client, tmp_path, monkeypatch):
    routes = _routes_mod()
    monkeypatch.setattr(routes, "_sub_agent_manager", _QueueingSAM())
    monkeypatch.setattr(routes, "_pinned_consolidation", MagicMock())
    pid, ws = _make_project(client, tmp_path, "q195")
    sid = _make_session(ws)

    first = client.post(f"/api/v2/agents/{pid}/inject", json={
        "content": "hi codex", "target": "codex", "session_id": sid}).json()
    second = client.post(f"/api/v2/agents/{pid}/inject", json={
        "content": "and this", "target": "codex", "session_id": sid}).json()

    assert first["status"] == "queued_behind"
    assert first["handle"] == "codex" and first["position"] == 1
    assert first["dispatch_id"].startswith(f"{sid}:")
    assert second["position"] == 2
    assert second["dispatch_id"] != first["dispatch_id"]


def test_pinned_send_that_went_out_keeps_its_text_and_adds_the_id(client, tmp_path, monkeypatch):
    routes = _routes_mod()
    sam = MagicMock()
    sam.send = AsyncMock(return_value="Message sent to codex. Transcript: /t")
    sam.queued_position = MagicMock(return_value=None)
    monkeypatch.setattr(routes, "_sub_agent_manager", sam)
    monkeypatch.setattr(routes, "_pinned_consolidation", MagicMock())
    pid, ws = _make_project(client, tmp_path, "s195")
    sid = _make_session(ws)

    body = client.post(f"/api/v2/agents/{pid}/inject", json={
        "content": "hi codex", "target": "codex", "session_id": sid}).json()
    assert body["status"] == "Message sent to codex. Transcript: /t"
    assert body["dispatch_id"] == sam.send.await_args.kwargs["dispatch_id"]


def test_unpin_wakes_orbital_and_retarget_does_not(client, tmp_path, monkeypatch):
    routes = _routes_mod()
    monkeypatch.setattr(routes, "_pinned_consolidation", MagicMock())
    pid, ws = _make_project(client, tmp_path, "w195")
    sid = _make_session(ws)
    calls = []

    async def _wake(project_id, session_id):
        calls.append((project_id, session_id))
        return "nothing"

    monkeypatch.setattr(routes._agent_manager, "wake_if_unread", _wake)
    url = f"/api/v2/agents/{pid}/sessions/{sid}"
    client.patch(url, json={"pinned_target": "codex"})
    client.patch(url, json={"pinned_target": "claude-code"})  # retarget
    assert calls == []
    client.patch(url, json={"pinned_target": None})  # back to Orbital
    assert calls == [(pid, sid)]
    client.patch(url, json={"pinned_target": None})  # already Orbital
    assert calls == [(pid, sid)]
