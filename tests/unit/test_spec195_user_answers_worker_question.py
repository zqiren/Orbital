# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Spec 195: a worker's question that is not Orbital's to answer goes to the user.

A worker whose turn the user started (a pinned send) or another worker
started (``worker:<parent>``) can block its turn on a question
(``interaction_required``, ACP). Orbital is not woken for it (the user is
talking to the worker), and the dispatching worker has no way to answer, so
the question rides the approval route verbatim: ``approval.request`` with
``kind: "question"`` live, ``GET /pending-approval`` after a reload, and
``POST /approve`` (``reply_text`` = the answer, ``response_payload`` = an
option id) / ``POST /deny``. The answer goes through
``SubAgentManager.respond_to_interaction``, the same call Orbital's own
``agent_message(respond)`` makes. A question from a turn Orbital dispatched is
Orbital's, unchanged.

Driven through the real app: create_app's routes, SubAgentManager,
LifecycleObserver, ProcessManager consumer and a real ACPSDKTransport's
blocking interaction; only the worker's turn is scripted.
"""

from __future__ import annotations

import asyncio
import json
import os
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from agent_os.agent.adapters.base import OutputChunk
from agent_os.agent.session import Session
from agent_os.agent.transports.acp_sdk_transport import ACPSDKTransport
from agent_os.agent.transports.base import transport_event_to_chunk
from agent_os.daemon_v2.models import make_session_key

H = "dsh"
QUESTION = {"questions": [{"id": "q1", "prompt": "用哪种颜色？", "allowFreeText": True,
                           "options": [{"id": "red", "label": "红"}, {"id": "blue", "label": "蓝"}]}],
            "title": "颜色"}


class _AskingWorker:
    """An ACP worker whose turn asks one question, then answers with what it
    got. The question is the real transport's ``_request_interaction`` (the
    future the turn blocks on); the stream is what ProcessManager reads."""

    def __init__(self, kind="cursor/ask_question", params=None):
        self._transport = ACPSDKTransport()
        self.display_name = "DeepSeek Harness"
        self.kind = kind
        self.params = params if params is not None else QUESTION
        self.answers: list = []
        self._turn_task = None

    def is_alive(self):
        return True

    def is_idle(self):
        return False

    async def _turn(self):
        answer = await self._transport._request_interaction(self.kind, self.params)
        self.answers.append(answer)
        q = self._transport._event_queue
        await q.put(OutputChunk(text=f"GOT {json.dumps(answer, ensure_ascii=False)}",
                                chunk_type="response", timestamp=""))
        await q.put(OutputChunk(text="", chunk_type="turn_complete", timestamp="",
                                metadata={"cause": "success"}))
        await q.put(None)

    async def read_stream(self):
        self._turn_task = asyncio.ensure_future(self._turn())
        q = self._transport._event_queue
        while True:
            item = await q.get()
            if item is None:
                return
            yield item if isinstance(item, OutputChunk) else transport_event_to_chunk(item)


class _Transcript:
    def __init__(self, path):
        self.filepath = path
        self.entries = []

    def append(self, entry):
        self.entries.append(entry)


async def _until(cond, timeout=3.0):
    loop = asyncio.get_running_loop()
    end = loop.time() + timeout
    while not cond():
        if loop.time() > end:
            raise AssertionError("condition not met in time")
        await asyncio.sleep(0.01)


@pytest.fixture
async def env(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("USERPROFILE", str(tmp_path / "home"))
    os.makedirs(str(tmp_path / "home"), exist_ok=True)
    from agent_os.api.app import create_app
    from agent_os.api.routes import agents_v2 as routes
    app = create_app(data_dir=str(tmp_path / "data"))
    from tests.unit.test_spec074_routes import _fake_check_all_factory
    # Never probe the machine's real agent CLIs.
    monkeypatch.setattr(routes._setup_engine, "check_all",
                        _fake_check_all_factory(["claude-code", "codex", H]))
    events: list[dict] = []
    real_broadcast = routes._ws_manager.broadcast

    def _spy(project_id, payload, *a, **kw):
        events.append(payload)
        return real_broadcast(project_id, payload, *a, **kw)

    monkeypatch.setattr(routes._ws_manager, "broadcast", _spy)
    am = routes._agent_manager
    sam = routes._sub_agent_manager
    observer = sam._lifecycle_observer
    # Collaborators that would start real turns: spies, so "Orbital took no
    # turn" and "the parent was not involved" are observable.
    monkeypatch.setattr(am, "start_agent", AsyncMock())
    monkeypatch.setattr(am, "_start_loop", AsyncMock())
    observer.worker_terminal_hook = MagicMock()
    observer.pinned_terminal_hook = MagicMock()

    client = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t")
    ws = str(tmp_path / "ws")
    os.makedirs(ws, exist_ok=True)
    resp = await client.post("/api/v2/projects", json={
        "name": "q195", "workspace": ws, "model": "gpt-4", "api_key": "test-key"})
    assert resp.status_code == 201, resp.text
    pid = resp.json()["project_id"]
    sid = "q195_sess_cafe0001"
    s = Session.new(sid, ws)
    s.append({"role": "user", "content": "hi", "source": "user"})
    s.append({"role": "assistant", "content": "hello", "source": "management"})

    class E:
        pass
    e = E()
    e.client, e.events, e.am, e.sam, e.observer = client, events, am, sam, observer
    e.pid, e.sid, e.ws = pid, sid, ws

    async def ask(initiator, worker=None):
        worker = worker or _AskingWorker()
        sam._adapters.setdefault(make_session_key(pid, sid), {})[H] = worker
        sam._wire_status(worker, pid, sid, H)  # the real status listener
        observer.set_dispatch_initiator(pid, H, initiator, session_id=sid)
        await sam._process_manager.start(
            pid, H, worker, transcript=_Transcript(str(tmp_path / "t.jsonl")),
            session_id=sid)
        await _until(lambda: bool(worker._transport._pending_interactions))
        await _until(lambda: any(r.get("_meta", {}).get("event") == "interaction_required"
                                 for r in e.rows()))
        return worker, next(iter(worker._transport._pending_interactions))

    def rows():
        p = os.path.join(ws, "orbital", "sessions", f"{sid}.jsonl")
        with open(p, "r", encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]

    e.ask, e.rows = ask, rows
    yield e
    await client.aclose()


def _question_events(events):
    return [ev for ev in events if ev.get("type") == "approval.request"
            and ev.get("kind") == "question"]


@pytest.mark.parametrize("initiator", ["worker:claude-code", "user_pinned"])
async def test_the_question_reaches_the_user_and_the_answer_reaches_the_worker(env, initiator):
    worker, qid = await env.ask(initiator)

    # Live: the approval route's event, the asking worker as its source.
    [ev] = _question_events(env.events)
    assert ev["tool_call_id"] == qid and ev["source"] == H
    assert ev["session_id"] == env.sid and ev["project_id"] == env.pid
    assert ev["tool_args"]["prompt"] == "颜色"
    assert ev["tool_args"]["options"] == QUESTION["questions"]

    # Reload: the same card comes back from the approval recovery route.
    rec = (await env.client.get(f"/api/v2/agents/{env.pid}/pending-approval",
                                params={"session_id": env.sid})).json()
    assert rec["pending"] is True and rec["tool_call_id"] == qid
    assert rec["kind"] == "question" and rec["source"] == H
    assert rec["tool_args"]["prompt"] == "颜色"

    # Orbital's row lands without a wake and without telling Orbital to answer.
    [row] = [r for r in env.rows() if r.get("_meta", {}).get("event") == "interaction_required"]
    assert row["_meta"]["suppress_wake"] is True
    assert "agent_message(action=\"respond\"" not in row["content"]
    assert row["_meta"]["display_content"] == f"[Sub-agent] {H} requires input (cursor/ask_question): 颜色"
    assert row["_meta"]["answerer"] == "user"

    # The user's answer.
    r = await env.client.post(f"/api/v2/agents/{env.pid}/approve", json={
        "tool_call_id": qid, "reply_text": "蓝绿色", "session_id": env.sid})
    assert r.status_code == 200, r.text
    await _until(lambda: worker.answers)
    assert worker.answers == [{"outcome": {"outcome": "answered", "answers": [
        {"questionId": "q1", "text": "蓝绿色"}]}}]
    assert any(ev.get("type") == "approval.resolved" and ev.get("tool_call_id") == qid
               for ev in env.events)

    # The worker's turn continued to its end.
    await _until(lambda: any(r.get("_meta", {}).get("kind") == "completed" for r in env.rows()))

    # Nothing pending any more; a second answer is refused and changes nothing.
    rec = (await env.client.get(f"/api/v2/agents/{env.pid}/pending-approval",
                                params={"session_id": env.sid})).json()
    assert rec == {"pending": False}
    again = await env.client.post(f"/api/v2/agents/{env.pid}/approve", json={
        "tool_call_id": qid, "reply_text": "红", "session_id": env.sid})
    assert again.status_code == 404
    assert len(worker.answers) == 1

    # Orbital took no turn.
    env.am.start_agent.assert_not_awaited()
    env.am._start_loop.assert_not_awaited()


async def test_the_dispatching_worker_is_not_asked(env):
    await env.ask("worker:claude-code")
    env.observer.worker_terminal_hook.assert_not_called()


async def test_an_option_answer_rides_response_payload(env):
    worker, qid = await env.ask("user_pinned")
    r = await env.client.post(f"/api/v2/agents/{env.pid}/approve", json={
        "tool_call_id": qid, "response_payload": "blue", "session_id": env.sid})
    assert r.status_code == 200, r.text
    await _until(lambda: worker.answers)
    assert worker.answers == [{"outcome": {"outcome": "answered", "answers": [
        {"questionId": "q1", "selectedOptionIds": ["blue"]}]}}]


async def test_deny_cancels_the_question_and_the_turn_goes_on(env):
    worker, qid = await env.ask("worker:claude-code")
    r = await env.client.post(f"/api/v2/agents/{env.pid}/deny", json={
        "tool_call_id": qid, "reason": "no", "session_id": env.sid})
    assert r.status_code == 200, r.text
    await _until(lambda: worker.answers)
    assert worker.answers == [{"outcome": {"outcome": "cancelled"}}]
    await _until(lambda: any(r.get("_meta", {}).get("kind") == "completed" for r in env.rows()))
    again = await env.client.post(f"/api/v2/agents/{env.pid}/deny", json={
        "tool_call_id": qid, "reason": "no", "session_id": env.sid})
    assert again.status_code == 404


async def test_the_card_tells_while_the_question_is_open(env):
    """The agent's status says it waits on the user (the card's amber dot and
    the worker chip), and stops saying so once answered."""
    worker, qid = await env.ask("user_pinned")
    assert env.sam._awaiting_approval(worker, env.pid, env.sid, H) is True

    def pushed():
        return [ev.get("awaiting_approval") for ev in env.events
                if ev.get("type") == "sub_agent.status" and ev.get("handle") == H]

    # Pushed when the question opens, without any other transition.
    assert pushed()[-1] is True
    await env.client.post(f"/api/v2/agents/{env.pid}/approve", json={
        "tool_call_id": qid, "reply_text": "x", "session_id": env.sid})
    await _until(lambda: worker.answers)
    assert env.sam._awaiting_approval(worker, env.pid, env.sid, H) is False
    assert pushed()[-1] is False


async def test_a_question_from_orbitals_own_dispatch_stays_orbitals(env):
    worker, qid = await env.ask("management_agent")
    assert _question_events(env.events) == []
    [row] = [r for r in env.rows() if r.get("_meta", {}).get("event") == "interaction_required"]
    assert "suppress_wake" not in row["_meta"]
    assert "agent_message(action=\"respond\"" in row["content"]
    assert "display_content" not in row["_meta"] and "answerer" not in row["_meta"]
    rec = (await env.client.get(f"/api/v2/agents/{env.pid}/pending-approval",
                                params={"session_id": env.sid})).json()
    assert rec == {"pending": False}
    # The user's route cannot answer Orbital's question.
    r = await env.client.post(f"/api/v2/agents/{env.pid}/approve", json={
        "tool_call_id": qid, "reply_text": "x", "session_id": env.sid})
    assert r.status_code == 404
    assert worker.answers == []
    # Orbital answers it on the path it always used.
    ok = await env.sam.respond_to_interaction(env.pid, H, qid, session_id=env.sid, text="由 Orbital 回答")
    assert ok is True
    await _until(lambda: worker.answers)
    assert worker.answers[0]["outcome"]["answers"][0]["text"] == "由 Orbital 回答"


async def test_a_one_field_form_takes_the_typed_answer(env):
    """ACP elicitation: a form with one field gets the user's text (it was
    dropped as an empty form before; Orbital's respond gains it too)."""
    params = {"message": "你的名字？", "mode": {
        "mode": "form",
        "requestedSchema": {"type": "object", "properties": {"name": {"type": "string"}}}}}
    worker, qid = await env.ask("user_pinned", _AskingWorker("elicitation/create", params))
    [ev] = _question_events(env.events)
    assert ev["tool_args"]["prompt"] == "你的名字？"
    await env.client.post(f"/api/v2/agents/{env.pid}/approve", json={
        "tool_call_id": qid, "reply_text": "小明", "session_id": env.sid})
    await _until(lambda: worker.answers)
    assert worker.answers == [{"action": "accept", "content": {"name": "小明"}}]
