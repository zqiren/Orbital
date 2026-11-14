# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Spec 180: move a session to another project.

The owner's definition: the session's transcript moves, its sub-agent
transcripts move (and its images), the system prompt is rebuilt from the new
project, and no artifact or other file is touched. The session keeps its id.
Nothing is deleted: A's log is retired into ``A/orbital/sessions/.moved/``.
An interrupted move is harmless and finishes by moving again. Writes that
race the move land in B.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import os
import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from agent_os.agent import blob_store
from agent_os.agent.session import Session, move_session_lines
from agent_os.api.routes import agents_v2
from agent_os.daemon_v2 import session_move
from agent_os.daemon_v2.agent_manager import AgentManager
from agent_os.daemon_v2.recap import build_recap
from agent_os.daemon_v2.session_index import derive_disk_entry
from agent_os.daemon_v2.session_move import MoveError

A, B = "projA", "projB"
STEM = "proja_1a2b3c4d"
PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==")


def _sessions(ws):
    return os.path.join(ws, "orbital", "sessions")


def _rows(path):
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _sha(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def _jsonl(ws):
    d = _sessions(ws)
    return sorted(f for f in os.listdir(d) if f.endswith(".jsonl")) if os.path.isdir(d) else []


def _moved(ws):
    d = os.path.join(_sessions(ws), ".moved")
    return sorted(os.listdir(d)) if os.path.isdir(d) else []


@pytest.fixture
def two(tmp_path):
    a_ws = str(tmp_path / "ws-a")
    b_ws = str(tmp_path / "ws-b")
    os.makedirs(a_ws)
    os.makedirs(b_ws)
    return a_ws, b_ws


def _build(a_ws, *, routed_by=None, origin="chat", with_image=False,
           with_transcript=False, with_worker_log=False):
    s = Session.new(STEM, a_ws, session_id=STEM, provider="deepseek", model="v4",
                    origin=origin)
    if routed_by:
        s.set_routed_by(routed_by)
    s.set_pinned_target("codex")
    s.append({"role": "user", "content": "把 Q3 预测汇总进预算表", "source": "user",
              "nonce": "n1", "timestamp": "2026-11-14T01:00:00+00:00"})
    s.set_sub_agent_thread("codex", session_id="thr-A", model="m")
    s.append_meta("sub_agent_fork", handle="claude-code", from_session_id="x")
    s.append_meta("model_swap", old="a", new="b")
    s.append({"role": "assistant", "content": "好的", "source": "management",
              "timestamp": "2026-11-14T01:00:10+00:00", "_meta": {"k": 1}})
    if with_transcript:
        tdir = os.path.join(a_ws, "orbital", "sub_agents", "codex")
        os.makedirs(tdir, exist_ok=True)
        tpath = os.path.join(tdir, "abcd1234.jsonl")
        with open(tpath, "w") as fh:
            fh.write('{"type":"x"}\n')
        s.append({"role": "system", "source": "daemon", "content": "[Sub-agent] Message sent",
                  "timestamp": "2026-11-14T01:00:20+00:00",
                  "_meta": {"dispatch_id": f"{STEM}:d1", "handle": "codex",
                            "transcript_path": tpath}})
    if with_image:
        s.append({"role": "tool", "tool_call_id": "t1", "source": "management",
                  "timestamp": "2026-11-14T01:00:30+00:00",
                  "content": [{"type": "image_url", "image_url": {
                      "url": "data:image/png;base64," + base64.b64encode(PNG).decode()}}]})
    if with_worker_log:
        w = Session.new("proja_worker01", a_ws, session_id="proja_worker01")
        w.append_meta("session_kind", kind="worker", parent_session_id=STEM)
        w.append({"role": "user", "content": "sub task", "source": "user"})
    return s


def _store(a_ws, b_ws, *, b_exists=True):
    store = MagicMock()
    projects = {A: {"project_id": A, "name": "季度预算", "workspace": a_ws},
                B: {"project_id": B, "name": "Finance", "workspace": b_ws}}
    if not b_exists:
        projects.pop(B)
    store.get_project.side_effect = lambda pid: projects.get(pid)
    store.list_projects.return_value = list(projects.values())
    return store


def _manager(a_ws, b_ws, sub=None, **kw):
    if sub is None:
        sub = MagicMock()
        sub.list_active = MagicMock(return_value=[])
        sub.stop_all = AsyncMock()
        sub.stop_for_user = AsyncMock()
        sub.send = AsyncMock(return_value="ok")
        sub.running_session_ids = MagicMock(return_value=set())
    mgr = AgentManager(project_store=_store(a_ws, b_ws, **kw), ws_manager=MagicMock(),
                       sub_agent_manager=sub, activity_translator=MagicMock(),
                       process_manager=MagicMock())
    mgr._ws.broadcast = MagicMock()
    mgr._platform_provider = None
    return mgr


# ---------------------------------------------------------------------------
# 1. The destination lines
# ---------------------------------------------------------------------------


def test_move_lines_copy_start_whole_rows_verbatim_drop_resume_records(two):
    a_ws, _ = two
    s = _build(a_ws, routed_by="model")
    with open(s.filepath, encoding="utf-8") as fh:
        text = fh.read()
    src = [ln for ln in text.splitlines() if ln.strip()]
    move_row = {"role": "system", "content": "moved", "_meta": {"event": "session_moved"}}
    out = move_session_lines(text, moved_from={"project_id": A, "at": "T"},
                             move_row=move_row, session_uuid=STEM)
    start = json.loads(out[0])
    src_start = json.loads(src[0])
    assert "routed_by" not in start
    assert start["moved_from"] == {"project_id": A, "at": "T"}
    assert {k: v for k, v in start.items() if k != "moved_from"} == {
        k: v for k, v in src_start.items() if k != "routed_by"}
    events = [json.loads(ln).get("event") for ln in out]
    assert "sub_agent_thread" not in events and "sub_agent_fork" not in events
    assert "model_swap" in events
    # Conversation rows byte-identical, in order.
    convo_src = [ln for ln in src if json.loads(ln).get("role") != "meta"]
    convo_out = [ln.rstrip("\n") for ln in out[1:-1] if json.loads(ln).get("role") != "meta"]
    assert convo_out == convo_src
    assert json.loads(out[-1]) == move_row


def test_disk_entry_in_b_equals_a_except_mark_and_activity(two):
    a_ws, b_ws = two
    s = _build(a_ws, routed_by="fallback")
    before = derive_disk_entry(s.filepath, STEM)
    mgr = _manager(a_ws, b_ws)
    asyncio.run(mgr.move_session(A, STEM, to_project_id=B))
    after = derive_disk_entry(os.path.join(_sessions(b_ws), f"{STEM}.jsonl"), STEM)
    assert before["routed_by"] == "fallback" and after["routed_by"] is None
    strip = lambda e: {k: v for k, v in e.items() if k not in ("routed_by", "last_activity_at")}
    assert strip(before) == strip(after)
    # The move row is neither a user row nor a reply: the card keeps its place.
    assert after["last_user_at"] == before["last_user_at"]
    assert after["last_reply_at"] == before["last_reply_at"]


# ---------------------------------------------------------------------------
# 2. move_session: the transfer, both ways, refusals
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_idle_move_writes_b_retires_a_keeps_id(two):
    a_ws, b_ws = two
    s = _build(a_ws)
    a_sha = _sha(s.filepath)
    mgr = _manager(a_ws, b_ws)
    result = await mgr.move_session(A, STEM, to_project_id=B)
    assert result == {"project_id": B, "session_id": STEM, "woke": False}
    assert _jsonl(a_ws) == []
    retired = _moved(a_ws)
    assert len(retired) == 1 and retired[0].startswith(f"{STEM}.to-{B}.")
    # A's log is kept byte for byte.
    assert _sha(os.path.join(_sessions(a_ws), ".moved", retired[0])) == a_sha
    rows = _rows(os.path.join(_sessions(b_ws), f"{STEM}.jsonl"))
    last = rows[-1]
    assert last["role"] == "system" and last["_meta"]["event"] == "session_moved"
    assert last["_meta"]["from"] == {"project_id": A, "name": "季度预算", "workspace": a_ws}
    assert last["_meta"]["to"] == {"project_id": B, "name": "Finance"}
    assert a_ws in last["content"] and b_ws in last["content"]
    # Listing: gone from A, present in B under the same id.
    assert [e["session_id"] for e in mgr.list_sessions(A)] == []
    assert [e["session_id"] for e in mgr.list_sessions(B)] == [STEM]
    # Hydrates in B with no resume records.
    loaded = Session.load(os.path.join(_sessions(b_ws), f"{STEM}.jsonl"))
    assert loaded.sub_agent_threads == {} and loaded.sub_agent_forks == {}
    assert loaded.pinned_target == "codex"
    assert loaded.moved_from["project_id"] == A


@pytest.mark.asyncio
async def test_move_back_and_forth_keeps_every_copy(two):
    a_ws, b_ws = two
    _build(a_ws)
    mgr = _manager(a_ws, b_ws)
    await mgr.move_session(A, STEM, to_project_id=B)
    await mgr.move_session(B, STEM, to_project_id=A)
    assert _jsonl(a_ws) == [f"{STEM}.jsonl"] and _jsonl(b_ws) == []
    assert len(_moved(a_ws)) == 1 and len(_moved(b_ws)) == 1
    rows = _rows(os.path.join(_sessions(a_ws), f"{STEM}.jsonl"))
    moves = [r for r in rows if (r.get("_meta") or {}).get("event") == "session_moved"]
    assert [(m["_meta"]["from"]["project_id"], m["_meta"]["to"]["project_id"])
            for m in moves] == [(A, B), (B, A)]


@pytest.mark.asyncio
async def test_refusals(two):
    a_ws, b_ws = two
    _build(a_ws)
    mgr = _manager(a_ws, b_ws)
    with pytest.raises(MoveError) as e:
        await mgr.move_session(A, STEM, to_project_id=A)
    assert e.value.code == "same_project"
    with pytest.raises(MoveError) as e:
        await mgr.move_session(A, STEM, to_project_id="nope")
    assert e.value.code == "destination_missing"
    with pytest.raises(FileNotFoundError):
        await mgr.move_session(A, "proja_00000000", to_project_id=B)
    os.rmdir(b_ws)
    with pytest.raises(MoveError) as e:
        await mgr.move_session(A, STEM, to_project_id=B)
    assert e.value.code == "destination_missing"
    assert _jsonl(a_ws) == [f"{STEM}.jsonl"]


@pytest.mark.asyncio
async def test_memory_session_is_not_movable(two):
    a_ws, b_ws = two
    _build(a_ws, origin="memory")
    mgr = _manager(a_ws, b_ws)
    with pytest.raises(MoveError) as e:
        await mgr.move_session(A, STEM, to_project_id=B)
    assert e.value.code == "not_movable"


@pytest.mark.asyncio
async def test_second_simultaneous_move_is_409(two, monkeypatch):
    a_ws, b_ws = two
    _build(a_ws)
    mgr = _manager(a_ws, b_ws)
    release = threading.Event()
    reached = threading.Event()

    def hold(step):
        if step == "copied":
            reached.set()
            release.wait(5)
    monkeypatch.setattr(session_move, "checkpoint", hold)
    first = asyncio.create_task(mgr.move_session(A, STEM, to_project_id=B))
    while not reached.is_set():
        await asyncio.sleep(0.01)
    with pytest.raises(MoveError) as e:
        await mgr.move_session(A, STEM, to_project_id=B)
    assert e.value.code == "moving" and e.value.status == 409
    release.set()
    await first
    assert _jsonl(b_ws) == [f"{STEM}.jsonl"]


# ---------------------------------------------------------------------------
# 3. Interrupted moves: harmless at every write boundary, finished by moving again
# ---------------------------------------------------------------------------


class _Crash(Exception):
    pass


@pytest.mark.asyncio
@pytest.mark.parametrize("step", ["copied", "written", "workers_written", "retired"])
async def test_interrupted_move_is_harmless_and_finishable(two, monkeypatch, step):
    a_ws, b_ws = two
    s = _build(a_ws, with_image=True, with_transcript=True, with_worker_log=True)
    a_sha = _sha(s.filepath)
    mgr = _manager(a_ws, b_ws)

    def crash(at):
        if at == step:
            raise _Crash(at)
    monkeypatch.setattr(session_move, "checkpoint", crash)
    with pytest.raises(_Crash):
        await mgr.move_session(A, STEM, to_project_id=B)
    # Nothing lost at any boundary: A's log is either still live or retired
    # whole, and a B copy, when there, is complete.
    a_live = os.path.join(_sessions(a_ws), f"{STEM}.jsonl")
    retired = [f for f in _moved(a_ws) if f.startswith(f"{STEM}.to-{B}.")]
    if step == "retired":
        assert not os.path.exists(a_live) and len(retired) == 1
    else:
        assert _sha(a_live) == a_sha and retired == []
    if step in ("written", "workers_written", "retired"):
        assert _rows(os.path.join(_sessions(b_ws), f"{STEM}.jsonl"))[-1]["_meta"][
            "event"] == "session_moved"
    assert mgr._move_gates() == {}  # the door is open again

    monkeypatch.setattr(session_move, "checkpoint", lambda at: None)
    if step == "retired":
        # Already moved: moving again from A finds nothing to move.
        with pytest.raises(FileNotFoundError):
            await mgr.move_session(A, STEM, to_project_id=B)
    else:
        await mgr.move_session(A, STEM, to_project_id=B)
    assert _jsonl(a_ws) == []
    assert f"{STEM}.jsonl" in _jsonl(b_ws)
    b_rows = _rows(os.path.join(_sessions(b_ws), f"{STEM}.jsonl"))
    assert sum(1 for r in b_rows if (r.get("_meta") or {}).get("event") == "session_moved") == 1
    # The kept original is byte-identical to the log before the move.
    kept = [f for f in _moved(a_ws) if f.startswith(f"{STEM}.to-{B}.")]
    assert len(kept) == 1
    assert _sha(os.path.join(_sessions(a_ws), ".moved", kept[0])) == a_sha
    # The fanout worker log moved with it (or was already copied).
    assert "proja_worker01.jsonl" in _jsonl(b_ws)


@pytest.mark.asyncio
async def test_b_holding_a_different_log_of_this_id_is_409(two):
    a_ws, b_ws = two
    _build(a_ws)
    other = Session.new(STEM, b_ws, session_id=STEM)
    other.append({"role": "user", "content": "unrelated", "source": "user"})
    b_sha = _sha(other.filepath)
    mgr = _manager(a_ws, b_ws)
    with pytest.raises(MoveError) as e:
        await mgr.move_session(A, STEM, to_project_id=B)
    assert e.value.code == "destination_exists" and e.value.status == 409
    assert _sha(other.filepath) == b_sha
    assert _jsonl(a_ws) == [f"{STEM}.jsonl"]


# ---------------------------------------------------------------------------
# 3a. Writes that race the move land in B
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("step", ["copied", "written", "workers_written"])
async def test_system_row_during_the_copy_lands_in_b(two, monkeypatch, step):
    a_ws, b_ws = two
    _build(a_ws)
    mgr = _manager(a_ws, b_ws)
    release, reached = threading.Event(), threading.Event()

    def hold(at):
        if at == step:
            reached.set()
            release.wait(5)
    monkeypatch.setattr(session_move, "checkpoint", hold)
    move = asyncio.create_task(mgr.move_session(A, STEM, to_project_id=B))
    while not reached.is_set():
        await asyncio.sleep(0.01)
    late = asyncio.create_task(mgr.inject_system_message(
        A, "[Sub-agent] codex stopped", session_id=STEM,
        meta={"event": "sub_agent_terminal", "suppress_wake": True}))
    user = asyncio.create_task(mgr.move_redirect(A, STEM))
    await asyncio.sleep(0.05)
    assert not late.done() and not user.done()  # both wait for the move
    release.set()
    await move
    assert await late == "suppressed"
    assert await user == B
    rows = _rows(os.path.join(_sessions(b_ws), f"{STEM}.jsonl"))
    assert rows[-1]["content"] == "[Sub-agent] codex stopped"
    assert rows[-2]["_meta"]["event"] == "session_moved"
    assert _jsonl(a_ws) == []


@pytest.mark.asyncio
async def test_writes_after_the_move_are_redirected_by_the_retired_name(two):
    a_ws, b_ws = two
    _build(a_ws)
    mgr = _manager(a_ws, b_ws)
    await mgr.move_session(A, STEM, to_project_id=B)
    assert await mgr.move_redirect(A, STEM) == B
    assert await mgr.move_redirect(A, "proja_99999999") is None  # a new session
    assert await mgr.inject_system_message(
        A, "late row", session_id=STEM, meta={"suppress_wake": True}) == "suppressed"
    assert _jsonl(a_ws) == []
    assert _rows(os.path.join(_sessions(b_ws), f"{STEM}.jsonl"))[-1]["content"] == "late row"
    # A pinned send's persist (the inject route's funnel) goes to B as well.
    sid = await mgr.persist_mention_message(
        A, STEM, {"role": "user", "content": "继续", "target": "codex"})
    assert sid == STEM
    assert _rows(os.path.join(_sessions(b_ws), f"{STEM}.jsonl"))[-1]["content"] == "继续"
    assert _jsonl(a_ws) == []


@pytest.mark.asyncio
async def test_moved_on_from_b_to_c_follows_the_chain(two, tmp_path):
    a_ws, b_ws = two
    _build(a_ws)
    mgr = _manager(a_ws, b_ws)
    await mgr.move_session(A, STEM, to_project_id=B)
    await mgr.move_session(B, STEM, to_project_id=A)
    # Back in A: A has the live log again, nothing redirects from A.
    assert await mgr.move_redirect(A, STEM) is None
    assert await mgr.move_redirect(B, STEM) == A


# ---------------------------------------------------------------------------
# 4 / 8. What the transcript needs: images and sub-agent transcripts
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_images_and_transcripts_copied_rows_point_at_the_copies(two):
    a_ws, b_ws = two
    _build(a_ws, with_image=True, with_transcript=True, with_worker_log=True)
    a_tpath = os.path.join(a_ws, "orbital", "sub_agents", "codex", "abcd1234.jsonl")
    a_tsha = _sha(a_tpath)
    mgr = _manager(a_ws, b_ws)
    await mgr.move_session(A, STEM, to_project_id=B)
    b_tpath = os.path.join(b_ws, "orbital", "sub_agents", "codex", "abcd1234.jsonl")
    assert os.path.isfile(b_tpath) and _sha(b_tpath) == a_tsha
    assert _sha(a_tpath) == a_tsha  # A's per-handle file untouched
    assert not os.path.exists(os.path.join(b_ws, "orbital", "sub_agents", "codex", ".latest"))
    b_log = os.path.join(_sessions(b_ws), f"{STEM}.jsonl")
    markers = [r for r in _rows(b_log) if (r.get("_meta") or {}).get("dispatch_id")]
    assert markers[0]["_meta"]["transcript_path"] == b_tpath
    loaded = Session.load(b_log)
    tool = [m for m in loaded.get_messages() if m.get("role") == "tool"][0]
    url = tool["content"][0]["image_url"]["url"]
    assert url.startswith("data:image/png;base64,")
    assert base64.b64decode(url.split(",", 1)[1]) == PNG
    # The fanout worker log moved, A's is retired.
    assert "proja_worker01.jsonl" in _jsonl(b_ws)
    assert any(f.startswith("proja_worker01.to-") for f in _moved(a_ws))


# ---------------------------------------------------------------------------
# 5. Live work is stopped and restarted in B
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_running_pinned_worker_is_stopped_and_continues_in_b(two):
    a_ws, b_ws = two
    _build(a_ws)
    mgr = _manager(a_ws, b_ws)
    sub = mgr._sub_agent_manager
    sub.list_active = MagicMock(side_effect=lambda pid, session_id=None: (
        [{"handle": "codex", "status": "running"}] if pid == A else []))
    result = await mgr.move_session(A, STEM, to_project_id=B)
    assert result["woke"] is True
    sub.stop_for_user.assert_awaited()
    assert sub.stop_for_user.await_args.args[:2] == (A, "codex")
    send = sub.send.await_args
    assert send.args[0] == B and send.args[1] == "codex"
    assert b_ws in send.args[2]
    assert send.kwargs["session_id"] == STEM
    assert send.kwargs["initiator"] == "user_pinned"


@pytest.mark.asyncio
async def test_queued_manager_messages_are_resent_into_b_in_order(two, monkeypatch):
    a_ws, b_ws = two
    s = _build(a_ws)
    mgr = _manager(a_ws, b_ws)
    s.queue_message("first queued", nonce="q1")
    s.queue_message("second queued", nonce="q2")
    fake = SimpleNamespace(session=s, task=None, interceptor=None, loop=MagicMock())
    from agent_os.daemon_v2.models import make_session_key
    mgr._handles[make_session_key(A, STEM)] = fake

    async def fake_stop(pid, *, session_id=None, keep_browser=False):
        mgr._handles.pop(make_session_key(pid, session_id), None)
    monkeypatch.setattr(mgr, "stop_agent", fake_stop)
    sent = []

    async def fake_inject(pid, content, *, nonce=None, session_id=None, **kw):
        sent.append((pid, content, nonce, session_id))
        return "started"
    monkeypatch.setattr(mgr, "inject_message", fake_inject)
    result = await mgr.move_session(A, STEM, to_project_id=B)
    assert result["woke"] is True
    assert sent == [(B, "first queued", "q1", STEM), (B, "second queued", "q2", STEM)]


# ---------------------------------------------------------------------------
# 6. The worker learns of the move through the recap
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fresh_worker_recap_carries_the_move_line(two):
    a_ws, b_ws = two
    _build(a_ws)
    mgr = _manager(a_ws, b_ws)
    await mgr.move_session(A, STEM, to_project_id=B)
    loaded = Session.load(os.path.join(_sessions(b_ws), f"{STEM}.jsonl"))
    block = build_recap(loaded.get_messages(), "codex", fresh=True)
    assert "system: The user moved this conversation from the project" in block
    assert block.index("user: 把 Q3") < block.index("system: The user moved")


# ---------------------------------------------------------------------------
# 7. routed_by — "Orbital chose this project"
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_routed_by_stamped_only_when_the_send_creates_the_session(two):
    a_ws, b_ws = two
    mgr = _manager(a_ws, b_ws)
    sid = "proja_feedf00d"
    await mgr.persist_mention_message(
        A, sid, {"role": "user", "content": "hi", "target": "codex"}, routed_by="model")
    path = os.path.join(_sessions(a_ws), f"{sid}.jsonl")
    assert _rows(path)[0]["routed_by"] == "model"
    assert derive_disk_entry(path, sid)["routed_by"] == "model"
    # A later send into the session does not change it.
    await mgr.persist_mention_message(
        A, sid, {"role": "user", "content": "more", "target": "codex"}, routed_by="fallback")
    assert _rows(path)[0]["routed_by"] == "model"
    # Live list row too.
    assert mgr.list_sessions(A)[0]["routed_by"] == "model"


def _client(mgr):
    saved = {k: getattr(agents_v2, k) for k in
             ("_project_store", "_agent_manager", "_ws_manager", "_sub_agent_manager")}
    ws = MagicMock()
    agents_v2.configure(project_store=mgr._project_store, agent_manager=mgr,
                        ws_manager=ws, sub_agent_manager=mgr._sub_agent_manager)
    app = FastAPI()
    app.include_router(agents_v2.router)
    return TestClient(app), saved, ws


def _restore(saved):
    for k, v in saved.items():
        setattr(agents_v2, k, v)


def test_patch_clears_routed_by_and_broadcasts(two):
    a_ws, b_ws = two
    s = _build(a_ws, routed_by="model")
    mgr = _manager(a_ws, b_ws)
    client, saved, ws = _client(mgr)
    url = f"/api/v2/agents/{A}/sessions/{STEM}"
    try:
        bad = client.patch(url, json={"routed_by": "model"})
        ok = client.patch(url, json={"routed_by": None})
        again = client.patch(url, json={"routed_by": None})
    finally:
        _restore(saved)
    assert bad.status_code == 422
    assert ok.status_code == 200 and ok.json()["routed_by"] is None
    assert again.status_code == 200
    assert _rows(s.filepath)[0]["routed_by"] is None
    assert derive_disk_entry(s.filepath, STEM)["routed_by"] is None
    assert Session.load(s.filepath).routed_by is None
    assert any(c.args[1] == {"type": "workbench.changed", "project_id": A}
               for c in ws.broadcast.call_args_list)


def test_move_route_contract(two):
    a_ws, b_ws = two
    _build(a_ws, routed_by="model")
    mgr = _manager(a_ws, b_ws)
    client, saved, ws = _client(mgr)
    try:
        same = client.post(f"/api/v2/agents/{A}/sessions/{STEM}/move",
                           json={"to_project_id": A})
        missing = client.post(f"/api/v2/agents/{A}/sessions/proja_00000000/move",
                              json={"to_project_id": B})
        ok = client.post(f"/api/v2/agents/{A}/sessions/{STEM}/move",
                         json={"to_project_id": B})
        gone = client.post(f"/api/v2/agents/{A}/sessions/{STEM}/move",
                           json={"to_project_id": B})
        unknown = client.post(f"/api/v2/agents/nope/sessions/{STEM}/move",
                              json={"to_project_id": B})
    finally:
        _restore(saved)
    assert same.status_code == 400 and same.json()["detail"] == "same_project"
    assert missing.status_code == 404
    assert ok.status_code == 200, ok.text
    assert ok.json() == {"project_id": B, "session_id": STEM, "woke": False}
    assert gone.status_code == 404
    assert unknown.status_code == 404
    pids = {c.args[0] for c in ws.broadcast.call_args_list
            if c.args[1].get("type") == "workbench.changed"}
    assert {A, B} <= pids
    # The moved session carries no mark.
    assert mgr.list_sessions(B)[0]["routed_by"] is None


# ---------------------------------------------------------------------------
# 9. Read access to the folder the session came from (owner decision 11)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_moved_session_may_read_its_old_folder_and_only_read(two):
    from agent_os.agent.tools._path_utils import resolve_safe_read
    from agent_os.daemon_v2.models import make_session_key
    a_ws, b_ws = two
    _build(a_ws)
    draft = os.path.join(a_ws, "预算-草稿.xlsx")
    with open(draft, "wb") as fh:
        fh.write(b"PK")
    mgr = _manager(a_ws, b_ws)
    await mgr.move_session(A, STEM, to_project_id=B)
    moved = Session.load(os.path.join(_sessions(b_ws), f"{STEM}.jsonl"))
    other = Session.new("projb_other001", b_ws)
    other.append({"role": "user", "content": "unrelated", "source": "user"})
    mgr._handles[make_session_key(B, STEM)] = SimpleNamespace(session=moved)
    mgr._handles[make_session_key(B, "projb_other001")] = SimpleNamespace(session=other)

    roots = mgr._read_roots_with_attached(B, STEM, [b_ws])
    assert roots == [b_ws, a_ws]
    assert resolve_safe_read(roots, draft) is not None
    # Another session of B gets nothing from A.
    other_roots = mgr._read_roots_with_attached(B, "projb_other001", [b_ws])
    assert other_roots == [b_ws]
    assert resolve_safe_read(other_roots, draft) is None
    # Read only: the write tools are rooted at B's workspace alone.
    from agent_os.agent.tools.write import WriteTool
    result = WriteTool(workspace=b_ws, session_id=STEM).execute(
        path=draft, content="x")
    if asyncio.iscoroutine(result):
        result = await result
    with open(draft, "rb") as fh:
        assert fh.read() == b"PK"


def test_session_ids_in_any_script_redirect_but_paths_do_not():
    from agent_os.daemon_v2.session_move import safe_session_id
    assert safe_session_id("alpha_预算_e571b947")
    assert safe_session_id("proja_1a2b3c4d")
    for bad in ("", None, "../x", "a/b", "a\\b", "x*", "x?", "[x]", ".hidden", "a\nb"):
        assert not safe_session_id(bad)


@pytest.mark.asyncio
async def test_a_shared_transcript_that_grew_since_an_earlier_move_is_refreshed(two):
    """The per-handle transcript is shared by A's sessions and keeps growing:
    a second session moved later must point at a copy that has its turns."""
    a_ws, b_ws = two
    _build(a_ws, with_transcript=True)
    mgr = _manager(a_ws, b_ws)
    await mgr.move_session(A, STEM, to_project_id=B)
    a_t = os.path.join(a_ws, "orbital", "sub_agents", "codex", "abcd1234.jsonl")
    b_t = os.path.join(b_ws, "orbital", "sub_agents", "codex", "abcd1234.jsonl")
    with open(a_t, "a") as fh:
        fh.write('{"type":"second session turn"}\n')
    s2 = Session.new("proja_22222222", a_ws, session_id="proja_22222222")
    s2.append({"role": "user", "content": "two", "source": "user"})
    s2.append({"role": "system", "source": "daemon", "content": "[Sub-agent] Message sent",
               "_meta": {"dispatch_id": "proja_22222222:d2", "handle": "codex",
                         "transcript_path": a_t}})
    await mgr.move_session(A, "proja_22222222", to_project_id=B)
    assert _sha(b_t) == _sha(a_t)
    # A diverged file of the same name in B is never overwritten.
    with open(b_t, "w") as fh:
        fh.write('{"type":"unrelated"}\n')
    s3 = Session.new("proja_33333333", a_ws, session_id="proja_33333333")
    s3.append({"role": "user", "content": "three", "source": "user"})
    s3.append({"role": "system", "source": "daemon", "content": "[Sub-agent] Message sent",
               "_meta": {"dispatch_id": "proja_33333333:d3", "handle": "codex",
                         "transcript_path": a_t}})
    await mgr.move_session(A, "proja_33333333", to_project_id=B)
    with open(b_t) as fh:
        assert fh.read() == '{"type":"unrelated"}\n'
    rows = _rows(os.path.join(_sessions(b_ws), "proja_33333333.jsonl"))
    named = [r["_meta"]["transcript_path"] for r in rows
             if (r.get("_meta") or {}).get("dispatch_id")][0]
    assert named != b_t and os.path.dirname(named) == os.path.dirname(b_t)
    assert _sha(named) == _sha(a_t)


@pytest.mark.asyncio
async def test_stop_agent_survives_a_loop_task_that_ends_cancelled(two):
    """Moving a running manager task: stop_agent's shield-wait on a loop task
    that ends *cancelled* used to raise CancelledError into the caller (the
    move route answered 500 "No response returned")."""
    from agent_os.daemon_v2.models import make_session_key
    a_ws, b_ws = two
    s = _build(a_ws)
    mgr = _manager(a_ws, b_ws)

    async def loop_body():
        await asyncio.sleep(30)
    task = asyncio.create_task(loop_body())
    loop = MagicMock()

    async def terminate():
        task.cancel()
    loop.terminate = terminate
    handle = SimpleNamespace(session=s, task=task, interceptor=None, loop=loop,
                             last_activity=0)
    mgr._handles[make_session_key(A, STEM)] = handle
    await mgr.stop_agent(A, session_id=STEM)  # must not raise
    assert task.cancelled()
    assert make_session_key(A, STEM) not in mgr._handles

    # The caller's own cancellation still propagates.
    task2 = asyncio.create_task(asyncio.sleep(30))
    loop2 = MagicMock()

    async def terminate2():
        pass
    loop2.terminate = terminate2
    mgr._handles[make_session_key(A, STEM)] = SimpleNamespace(
        session=s, task=task2, interceptor=None, loop=loop2, last_activity=0)
    stopper = asyncio.create_task(mgr.stop_agent(A, session_id=STEM))
    await asyncio.sleep(0.05)
    stopper.cancel()
    with pytest.raises(asyncio.CancelledError):
        await stopper
    task2.cancel()


@pytest.mark.asyncio
async def test_move_row_marks_the_quick_tasks_side(two):
    """The divider shows Quick Tasks through the catalog (快速任务), so the
    row says which side was the scratch project; absent otherwise."""
    a_ws, b_ws = two
    _build(a_ws)
    mgr = _manager(a_ws, b_ws)
    projects = {A: {"project_id": A, "name": "Quick Tasks", "workspace": a_ws,
                    "is_scratch": True},
                B: {"project_id": B, "name": "Finance", "workspace": b_ws}}
    mgr._project_store.get_project.side_effect = lambda pid: projects.get(pid)
    await mgr.move_session(A, STEM, to_project_id=B)
    meta = _rows(os.path.join(_sessions(b_ws), f"{STEM}.jsonl"))[-1]["_meta"]
    assert meta["from"]["is_scratch"] is True
    assert "is_scratch" not in meta["to"]
    await mgr.move_session(B, STEM, to_project_id=A)
    meta = _rows(os.path.join(_sessions(a_ws), f"{STEM}.jsonl"))[-1]["_meta"]
    assert "is_scratch" not in meta["from"] and meta["to"]["is_scratch"] is True
