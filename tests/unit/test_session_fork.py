# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Spec 156 phase A: edit a sent message → fork the session.

The fork is a new session file holding the parent's conversation up to (not
including) the edited user row, re-stamped with the fork's identity; the
parent file is never written; no resume record is copied, so every worker
starts fresh in the fork with the whole kept history as its recap; the route
refuses while the session runs and answers 404 for an unknown session or
message.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from agent_os import telemetry
from agent_os.agent import session as session_mod
from agent_os.agent.session import (
    ForkMessageNotEditable, ForkMessageNotFound, ForkSessionRunning, Session,
    cut_for_fork, fork_session_lines, write_new_session_file,
)
from agent_os.api.routes import agents_v2
from agent_os.daemon_v2.agent_manager import AgentManager, ProjectHandle
from agent_os.daemon_v2.models import make_session_key
from agent_os.daemon_v2.recap import build_recap, source_log_entries
from agent_os.daemon_v2.sub_agent_manager import SubAgentManager
from agent_os.telemetry.rollup import build_ping

PID = "proj156"
PARENT = "proj156_aaaa1111"


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------


def _sessions_dir(ws: str) -> str:
    return os.path.join(ws, "orbital", "sessions")


def _parent_path(ws: str) -> str:
    return os.path.join(_sessions_dir(ws), f"{PARENT}.jsonl")


def _session_files(ws: str) -> list[str]:
    # Session JSONLs only: on Windows the session lock leaves a sibling
    # ``*.jsonl.lock`` sentinel (agent_os/utils/file_lock.py), not a session.
    return sorted(f for f in os.listdir(_sessions_dir(ws)) if f.endswith(".jsonl"))


def _sha(path: str) -> str:
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def _rows(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _user(content, ts, **extra):
    return {"role": "user", "content": content, "timestamp": ts,
            "source": "user", **extra}


def _build_parent(ws: str, *, target: str | None = None,
                  with_compaction: bool = False) -> tuple[Session, list, str]:
    """Six user turns. Returns (parent session, the parent's model view at
    the moment message 3 was sent, message 3's timestamp)."""
    s = Session.new(PARENT, ws, session_id=PARENT, provider="deepseek",
                    model="v4", sdk="openai", fallback_models=["fb"])
    s.set_sub_agent_params("codex", {"model": "gpt-6-astra"})
    s.set_name("Parent chat")
    view_before_3 = None
    ts3 = None
    for i in range(1, 7):
        ts = f"2026-11-09T10:0{i}:00.000001+00:00"
        extra = {"nonce": f"n{i}"}
        if target:
            extra["target"] = target
        if i == 3:
            view_before_3 = [dict(m) for m in s.get_model_messages()]
            ts3 = ts
        s.append(_user(f"message {i}", ts, **extra))
        if i == 1:
            s.append_meta("model_swap", old="a", new="b")
            s.set_sub_agent_thread("codex", session_id="thr-parent", model="m")
        s.append({"role": "assistant", "content": f"reply {i}",
                  "timestamp": ts.replace(":00.", ":30."),
                  "_meta": {"x": i}})
        if i == 2 and with_compaction:
            s._compact({"role": "user", "content": "SUMMARY of 1",
                        "_compaction": True,
                        "timestamp": "2026-11-09T10:02:40.000001+00:00"}, 2)
            s.append({"role": "system", "content": "[Sub-agent] codex dispatched",
                      "timestamp": "2026-11-09T10:02:50.000001+00:00",
                      "_meta": {"dispatch_id": "d-pre", "handle": "codex",
                                "transcript_path": "/ws/orbital/transcripts/codex.jsonl"}})
    s.append_meta("session_kind", kind="x")
    return s, view_before_3, ts3


def _manager(ws: str, sub=None) -> AgentManager:
    if sub is None:
        sub = MagicMock()
        sub.list_active = MagicMock(return_value=[])
    sub.claude_fork_rows = MagicMock(return_value=[])
    mgr = AgentManager(project_store=MagicMock(), ws_manager=MagicMock(),
                       sub_agent_manager=sub, activity_translator=MagicMock(),
                       process_manager=MagicMock())
    mgr._ws.broadcast = MagicMock()
    mgr._project_store.get_project.return_value = {
        "project_id": PID, "name": "proj156", "workspace": ws,
    }
    mgr._platform_provider = None
    return mgr


@pytest.fixture
def emitted(monkeypatch):
    sent: list[str] = []
    monkeypatch.setattr(telemetry, "emit",
                        lambda event, fields=None, **k: sent.append(event))
    monkeypatch.setattr(telemetry, "latch", lambda *a, **k: None)
    return sent


def _strip_ids(rows):
    return [{k: v for k, v in r.items() if k not in ("session_id", "session_uuid")}
            for r in rows]


# ---------------------------------------------------------------------------
# The copy
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fork_copies_the_prefix_restamped_and_leaves_the_parent_untouched(
        tmp_path, emitted):
    ws = str(tmp_path)
    _s, _view, ts3 = _build_parent(ws)
    before = _sha(_parent_path(ws))
    mgr = _manager(ws)

    result = await mgr.fork_session(PID, PARENT, at=ts3,
                                    name="Parent chat · 改自 10:03")

    assert _sha(_parent_path(ws)) == before
    new = result["session_id"]
    assert new != PARENT and result["session_uuid"] == new
    assert result["target"] is None
    assert result["name"] == "Parent chat · 改自 10:03"
    rows = _rows(os.path.join(_sessions_dir(ws), f"{new}.jsonl"))
    meta, body = rows[0], rows[1:]
    assert meta["event"] == "session_start"
    assert meta["session_id"] == meta["session_uuid"] == new
    assert meta["origin"] == "chat"
    assert (meta["provider"], meta["model"], meta["sdk"], meta["fallback_models"]) == (
        "deepseek", "v4", "openai", ["fb"])
    assert meta["name"] == "Parent chat · 改自 10:03"
    assert meta["pinned_target"] is None
    assert meta["sub_agent_params"] == {"codex": {"model": "gpt-6-astra"}}
    assert meta["forked_from"] == {"session_id": PARENT, "at": ts3}
    # Messages 1-2 and their replies, every field but the identity unchanged.
    parent_body = [r for r in _rows(_parent_path(ws)) if r.get("role") != "meta"]
    assert _strip_ids(body) == _strip_ids(parent_body[:4])
    assert [r["content"] for r in body] == ["message 1", "reply 1", "message 2", "reply 2"]
    assert all(r["session_id"] == new and r["session_uuid"] == new for r in body)
    assert body[0]["nonce"] == "n1" and body[1]["_meta"] == {"x": 1}
    # No parent meta row of any kind.
    assert [r for r in body if r.get("role") == "meta"] == []
    assert emitted.count("message_edited") == 1
    assert emitted.count("session_created") == 1


@pytest.mark.asyncio
async def test_fork_holds_no_resume_record_so_every_worker_starts_fresh(
        tmp_path, emitted):
    """The invariant: two Orbital sessions never resume one native worker
    session. The fork's codex starts fresh, its recap the whole prefix."""
    ws = str(tmp_path)
    _s, _view, ts3 = _build_parent(ws)
    mgr = _manager(ws)
    new = (await mgr.fork_session(PID, PARENT, at=ts3))["session_id"]
    fork = Session.load(os.path.join(_sessions_dir(ws), f"{new}.jsonl"))
    assert fork.sub_agent_threads == {} and fork.sub_agent_forks == {}
    assert fork.forked_from == {"session_id": PARENT, "at": ts3}
    assert fork.name is None or fork.name  # loaded without error

    sam = SubAgentManager(process_manager=MagicMock())
    sam._session_resolver = lambda pid, sid: fork
    rec, status, reason = sam._determine_resume(ws, PID, "codex", new)
    assert (rec, status, reason) == (None, "fresh", "first_spawn")
    recap = build_recap(fork.get_messages(), "codex", fresh=True)
    assert "user: message 1" in recap and "assistant: reply 2" in recap
    assert "message 3" not in recap and "reply 3" not in recap


@pytest.mark.asyncio
async def test_fork_model_view_equals_the_parent_view_before_the_edit(
        tmp_path, emitted):
    """Compaction markers count back from themselves; a prefix from row 0
    keeps them valid, so the fork's model view is the parent's, row for row."""
    ws = str(tmp_path)
    _s, view_before_3, ts3 = _build_parent(ws, with_compaction=True)
    mgr = _manager(ws)
    new = (await mgr.fork_session(PID, PARENT, at=ts3))["session_id"]
    fork = Session.load(os.path.join(_sessions_dir(ws), f"{new}.jsonl"))
    assert any(m.get("_compaction") for m in fork.get_messages())
    assert _strip_ids(fork.get_model_messages()) == _strip_ids(view_before_3)


@pytest.mark.asyncio
async def test_pinned_fork_carries_the_edited_rows_target(tmp_path, emitted):
    ws = str(tmp_path)
    _s, _view, ts3 = _build_parent(ws, target="claude-code")
    mgr = _manager(ws)
    result = await mgr.fork_session(PID, PARENT, at=ts3)
    assert result["target"] == "claude-code"
    fork = Session.load(os.path.join(_sessions_dir(ws), f"{result['session_id']}.jsonl"))
    assert fork.pinned_target == "claude-code"


@pytest.mark.asyncio
async def test_editing_the_first_message_forks_an_empty_conversation(
        tmp_path, emitted):
    ws = str(tmp_path)
    _build_parent(ws)
    mgr = _manager(ws)
    result = await mgr.fork_session(PID, PARENT,
                                    at="2026-11-09T10:01:00.000001+00:00")
    rows = _rows(os.path.join(_sessions_dir(ws), f"{result['session_id']}.jsonl"))
    assert [r["event"] for r in rows] == ["session_start"]


# ---------------------------------------------------------------------------
# Refusals
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_unknown_session_and_unknown_message(tmp_path, emitted):
    ws = str(tmp_path)
    _build_parent(ws)
    mgr = _manager(ws)
    with pytest.raises(FileNotFoundError):
        await mgr.fork_session(PID, "proj156_nope0000", at="x")
    with pytest.raises(ForkMessageNotFound):
        await mgr.fork_session(PID, PARENT, at="2026-11-09T10:01:30.000001+00:00")
    assert "message_edited" not in emitted and "session_created" not in emitted
    assert _session_files(ws) == [f"{PARENT}.jsonl"]


@pytest.mark.parametrize("content", [
    "[Triggered by schedule 'daily']\ndo it",
    "[QUEUE ITEM | id: q1 | attempt 1]\nYou are working on a queue item.",
])
def test_automation_and_queue_rows_are_not_editable(content):
    text = json.dumps(_user(content, "T1")) + "\n"
    with pytest.raises(ForkMessageNotEditable):
        cut_for_fork(text, "T1")


@pytest.mark.asyncio
@pytest.mark.parametrize("state", ["running", "paused", "waiting"])
async def test_refused_while_the_manager_is_live(tmp_path, emitted, state):
    ws = str(tmp_path)
    s, _view, ts3 = _build_parent(ws)
    mgr = _manager(ws)
    stopped = asyncio.Event()
    task = (asyncio.get_running_loop().create_task(stopped.wait())
            if state == "running" else None)
    s._paused_for_approval = state == "paused"
    mgr._handles[make_session_key(PID, PARENT)] = ProjectHandle(
        session=s, loop=MagicMock(), provider=None, registry=None,
        context_manager=None, interceptor=None, task=task)
    if state == "waiting":
        mgr._idle_poll_tasks[make_session_key(PID, PARENT)] = (
            asyncio.get_running_loop().create_task(asyncio.sleep(3600)))
    try:
        with pytest.raises(ForkSessionRunning):
            await mgr.fork_session(PID, PARENT, at=ts3)
    finally:
        if task is not None:
            task.cancel()
        for t in mgr._idle_poll_tasks.values():
            t.cancel()
    assert _session_files(ws) == [f"{PARENT}.jsonl"]


@pytest.mark.asyncio
async def test_refused_while_a_worker_turn_runs_not_for_background_work(
        tmp_path, emitted):
    ws = str(tmp_path)
    _s, _view, ts3 = _build_parent(ws)
    sub = MagicMock()
    sub.list_active = MagicMock(return_value=[{"handle": "codex", "status": "running"}])
    mgr = _manager(ws, sub)
    with pytest.raises(ForkSessionRunning):
        await mgr.fork_session(PID, PARENT, at=ts3)
    sub.list_active.return_value = [{"handle": "codex", "status": "background-running"}]
    result = await mgr.fork_session(PID, PARENT, at=ts3)
    assert result["session_id"]


@pytest.mark.asyncio
async def test_an_idle_live_parent_forks(tmp_path, emitted):
    ws = str(tmp_path)
    s, _view, ts3 = _build_parent(ws)
    mgr = _manager(ws)
    mgr._handles[make_session_key(PID, PARENT)] = ProjectHandle(
        session=s, loop=MagicMock(), provider=None, registry=None,
        context_manager=None, interceptor=None, task=None)
    assert (await mgr.fork_session(PID, PARENT, at=ts3))["session_id"]


# ---------------------------------------------------------------------------
# The writer: whole or nothing
# ---------------------------------------------------------------------------


def test_writer_leaves_nothing_when_the_write_fails(tmp_path, monkeypatch):
    sdir = str(tmp_path / "sessions")
    real_fsync = os.fsync

    def boom(fd):
        raise OSError("disk full")

    monkeypatch.setattr(session_mod.os, "fsync", boom)
    with pytest.raises(OSError):
        write_new_session_file(sdir, "proj_x_1", ['{"role":"meta"}\n'])
    monkeypatch.setattr(session_mod.os, "fsync", real_fsync)
    assert os.listdir(sdir) == []


def test_writer_never_replaces_an_existing_log(tmp_path):
    sdir = str(tmp_path)
    path = os.path.join(sdir, "proj_x_1.jsonl")
    with open(path, "w") as fh:
        fh.write("keep\n")
    with pytest.raises(FileExistsError):
        write_new_session_file(sdir, "proj_x_1", ["new\n"])
    assert open(path).read() == "keep\n"


def test_temp_name_is_never_a_session_log(tmp_path, monkeypatch):
    """A crash between write and replace leaves a file the list never shows."""
    seen: list[str] = []
    real_replace = os.replace

    def spy(src, dst):
        seen.append(os.path.basename(src))
        return real_replace(src, dst)

    monkeypatch.setattr(session_mod.os, "replace", spy)
    write_new_session_file(str(tmp_path), "proj_x_1", ["x\n"])
    assert seen and not seen[0].endswith(".jsonl")


# ---------------------------------------------------------------------------
# Load + downgrade
# ---------------------------------------------------------------------------


def test_load_reads_forked_from_and_fork_rows_last_wins(tmp_path):
    cut = cut_for_fork("\n".join(json.dumps(r) for r in [
        {"role": "meta", "event": "session_start", "session_id": "p"},
        _user("hi", "T0"), {"role": "assistant", "content": "yo"},
        _user("edit me", "T1"),
    ]), "T1")
    lines = fork_session_lines(cut, session_uuid="proj_f_1", name="n",
                               parent_session_id="p", at="T1", fork_rows=[
        {"role": "meta", "event": "sub_agent_fork", "handle": "claude-code",
         "from_session_id": "old", "at_uuid": None},
        {"role": "meta", "event": "sub_agent_fork", "handle": "claude-code",
         "from_session_id": "native-1", "at_uuid": "u-9"},
    ])
    path = write_new_session_file(str(tmp_path), "proj_f_1", lines)
    s = Session.load(path)
    assert s.forked_from == {"session_id": "p", "at": "T1"}
    assert s.sub_agent_forks == {"claude-code": {"from_session_id": "native-1",
                                                 "at_uuid": "u-9"}}
    assert s.sub_agent_threads == {}
    assert [m["content"] for m in s.get_messages()] == ["hi", "yo"]
    assert s.name == "n"


# ---------------------------------------------------------------------------
# Spec 128: copied transcript paths name the cut
# ---------------------------------------------------------------------------


def test_copied_transcript_paths_name_the_cut_in_a_fork():
    marker = {"role": "system", "content": "dispatched",
              "_meta": {"dispatch_id": "d1", "handle": "codex",
                        "transcript_path": "/ws/orbital/transcripts/codex.jsonl"}}
    kwargs = dict(fresh=True, thread_history={}, chat_log_path="/ws/chat.jsonl",
                  native_path_for=lambda h, r: None)
    forked = source_log_entries([marker], "claude-code",
                                fork_cut="2026-11-09T10:03:00+00:00", **kwargs)
    plain = source_log_entries([marker], "claude-code", **kwargs)
    assert forked[0] == (
        "Orbital worker transcript (codex, partial capture) (only entries before "
        "2026-11-09T10:03:00+00:00 belong to this conversation)",
        "/ws/orbital/transcripts/codex.jsonl")
    assert plain[0][0] == "Orbital worker transcript (codex, partial capture)"
    # The fork's own chat log never gets the suffix.
    assert forked[-1] == ("Orbital chat log", "/ws/chat.jsonl")


# ---------------------------------------------------------------------------
# Telemetry rollup
# ---------------------------------------------------------------------------


def test_rollup_counts_messages_edited():
    spool = MagicMock()
    spool.read_day.return_value = [{"event": "message_edited"},
                                   {"event": "message_edited"},
                                   {"event": "session_created"}]
    identity = MagicMock()
    identity.install_id = "inst_x"
    identity.first_seen = "2026-11-01"
    identity.milestones = {}
    ping = build_ping(identity, spool, "2026-11-09", "0.21.0", "darwin")
    assert ping["counters"]["messages_edited"] == 2
    assert ping["counters"]["sessions"] == 1


# ---------------------------------------------------------------------------
# The route
# ---------------------------------------------------------------------------


def _client(mgr):
    saved = {k: getattr(agents_v2, k) for k in
             ("_project_store", "_agent_manager", "_ws_manager", "_sub_agent_manager")}
    agents_v2.configure(project_store=mgr._project_store, agent_manager=mgr,
                        ws_manager=MagicMock(), sub_agent_manager=mgr._sub_agent_manager)
    app = FastAPI()
    app.include_router(agents_v2.router)
    return TestClient(app), saved


def _restore(saved):
    for k, v in saved.items():
        setattr(agents_v2, k, v)


def test_route_contract(tmp_path, emitted):
    ws = str(tmp_path)
    _s, _view, ts3 = _build_parent(ws)
    mgr = _manager(ws)
    client, saved = _client(mgr)
    url = f"/api/v2/agents/{PID}/sessions/{PARENT}/fork"
    try:
        ok = client.post(url, json={"at": ts3, "name": "Parent chat · 改自 10:03"})
        missing_msg = client.post(url, json={"at": "nope"})
        missing_session = client.post(
            f"/api/v2/agents/{PID}/sessions/proj156_zzzz9999/fork", json={"at": ts3})
        mgr._project_store.get_project.return_value = None
        missing_project = client.post(url, json={"at": ts3})
    finally:
        _restore(saved)
    assert ok.status_code == 200, ok.text
    body = ok.json()
    assert set(body) == {"session_id", "session_uuid", "target", "name"}
    assert body["name"] == "Parent chat · 改自 10:03" and body["target"] is None
    assert os.path.exists(os.path.join(_sessions_dir(ws), f"{body['session_id']}.jsonl"))
    assert (missing_msg.status_code, missing_msg.json()["detail"]) == (404, "message_not_found")
    assert missing_session.status_code == 404
    assert missing_project.status_code == 404


def test_route_409_while_running(tmp_path, emitted):
    ws = str(tmp_path)
    _s, _view, ts3 = _build_parent(ws)
    sub = MagicMock()
    sub.list_active = MagicMock(return_value=[{"handle": "codex", "status": "running"}])
    mgr = _manager(ws, sub)
    client, saved = _client(mgr)
    try:
        r = client.post(f"/api/v2/agents/{PID}/sessions/{PARENT}/fork", json={"at": ts3})
    finally:
        _restore(saved)
    assert (r.status_code, r.json()["detail"]) == (409, "session_running")


def test_route_400_for_a_trigger_row(tmp_path, emitted):
    ws = str(tmp_path)
    os.makedirs(_sessions_dir(ws))
    with open(_parent_path(ws), "w") as fh:
        fh.write(json.dumps({"role": "meta", "event": "session_start",
                             "session_id": PARENT}) + "\n")
        fh.write(json.dumps(_user("[Triggered by schedule 'x']\ngo", "T1")) + "\n")
    mgr = _manager(ws)
    client, saved = _client(mgr)
    try:
        r = client.post(f"/api/v2/agents/{PID}/sessions/{PARENT}/fork", json={"at": "T1"})
    finally:
        _restore(saved)
    assert (r.status_code, r.json()["detail"]) == (400, "message_not_editable")


def test_route_non_ascii_session_id_and_at_from_chat(tmp_path, emitted):
    """The client URL-encodes a non-ASCII session id and sends ``at`` exactly
    as /chat returned the row's timestamp; the fork's pin equals ``target``."""
    from urllib.parse import quote
    ws = str(tmp_path)
    sid = "合同审阅_08ebefdd"
    s = Session.new(sid, ws, session_id=sid)
    s.append(_user("第一条", "2026-11-09T10:45:30.000001+00:00", target="codex"))
    s.append({"role": "assistant", "content": "好", "timestamp": "2026-11-09T10:45:31.000001+00:00"})
    s.append(_user("第二条", "2026-11-09T10:45:34.277291+00:00", target="codex"))
    mgr = _manager(ws)
    client, saved = _client(mgr)
    try:
        chat = client.get(f"/api/v2/agents/{PID}/chat",
                          params={"session_id": sid, "limit": 50})
        assert chat.status_code == 200, chat.text
        body = chat.json()
        msgs = body.get("messages", body) if isinstance(body, dict) else body
        at = [m for m in msgs if m.get("role") == "user"][-1]["timestamp"]
        r = client.post(f"/api/v2/agents/{PID}/sessions/{quote(sid)}/fork",
                        json={"at": at, "name": "合同审阅 · 改自 18:45"})
    finally:
        _restore(saved)
    assert at == "2026-11-09T10:45:34.277291+00:00"
    assert r.status_code == 200, r.text
    out = r.json()
    assert out["target"] == "codex"
    rows = _rows(os.path.join(_sessions_dir(ws), f"{out['session_id']}.jsonl"))
    assert rows[0]["pinned_target"] == out["target"]
    assert [x["content"] for x in rows[1:]] == ["第一条", "好"]
