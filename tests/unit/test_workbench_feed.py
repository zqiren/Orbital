# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Spec 154 §5.1-5.2 — ``GET /api/v2/workbench/feed`` and
``POST /api/v2/workbench/checked``.

Three projects (one excluded from the Workbench), session rows served by a
fake agent manager, finished sessions as real JSONL files (the feed reads
their result line and stop through the left-off reader), asks in a real
``orbital/ASKS.md``, and a real ``CheckedStore`` in a tmp data dir.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone

import httpx
import pytest
from fastapi import FastAPI

from agent_os.agent import asks
from agent_os.api.routes import workbench as wb
from agent_os.daemon_v2.checked_store import CheckedStore

NOW = datetime(2026, 11, 4, 12, 0, 0, tzinfo=timezone.utc)
T_USER = "2026-11-04T08:00:00+00:00"
T_REPLY = "2026-11-04T08:05:00+00:00"
T_OLD_USER = "2026-10-20T08:00:00+00:00"
T_OLD_REPLY = "2026-10-20T08:05:00+00:00"  # 15 days old: outside the window


class FakeStore:
    def __init__(self, projects):
        self._p = {p["project_id"]: p for p in projects}

    def list_projects(self):
        return list(self._p.values())

    def get_project(self, pid):
        return self._p.get(pid)


class FakeManager:
    def __init__(self, sessions):
        self.sessions = sessions  # {pid: [row]}

    def list_sessions(self, pid, include_memory=False):
        return [dict(r) for r in self.sessions.get(pid, [])]


class FakeSubAgents:
    def __init__(self, running):
        self.running = running  # {pid: {sid}}

    def running_session_ids(self, pid):
        return set(self.running.get(pid, set()))


class FakeWS:
    def __init__(self):
        self.events = []

    def broadcast(self, pid, payload):
        self.events.append((pid, payload))


def _project(tmp_path, pid, **extra):
    ws = tmp_path / pid
    (ws / "orbital" / "sessions").mkdir(parents=True, exist_ok=True)
    return {"project_id": pid, "name": pid, "workspace": str(ws), "timezone": "UTC", **extra}


def _write_session(project, uuid, rows):
    path = os.path.join(project["workspace"], "orbital", "sessions", f"{uuid}.jsonl")
    with open(path, "w", encoding="utf-8") as f:
        f.write(json.dumps({"role": "meta", "event": "session_start", "session_id": uuid}) + "\n")
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _row(sid, *, status="idle", user=T_USER, reply=T_REPLY, **extra):
    return {"session_id": sid, "session_uuid": sid, "status": status, "name": f"task {sid}",
            "pinned_target": None, "last_user_at": user, "last_reply_at": reply,
            "last_activity_at": reply or user, "origin": "chat", **extra}


def _asks_file(project, lines):
    path = os.path.join(project["workspace"], "orbital", "ASKS.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join([asks.FORMAT_HEADER, *lines, ""]))


def _client(tmp_path, projects, sessions, running=None, *, store=None):
    ws = FakeWS()
    store = store or CheckedStore.in_dir(str(tmp_path / "data"), now_fn=lambda: NOW)
    wb.configure(FakeStore(projects), FakeManager(sessions), None, now_fn=lambda: NOW,
                 ws_manager=ws, checked_store=store,
                 sub_agent_manager=FakeSubAgents(running or {}))
    app = FastAPI()
    app.include_router(wb.router)
    client = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t")
    return client, store, ws


@pytest.fixture
def world(tmp_path):
    a = _project(tmp_path, "proj_a")
    b = _project(tmp_path, "proj_b")
    x = _project(tmp_path, "proj_x", workbench_exclude_global=True)
    # proj_a: a finished worker task, its reply in _meta.reply, then the record turn.
    _write_session(a, "s_done", [
        {"role": "user", "content": "fix it", "target": "codex", "timestamp": T_USER},
        {"role": "system", "source": "daemon", "timestamp": T_REPLY,
         "content": "[Sub-agent] codex completed. Summary: ...",
         "_meta": {"event": "sub_agent_terminal", "kind": "completed",
                   "reply": "## Fixed **safe-mode** cwd\n\nAdded a regression test."}},
        {"role": "user", "source": "daemon", "_kind": "memory_record",
         "content": "[MEMORY] record", "timestamp": "2026-11-04T08:06:00+00:00"},
        {"role": "assistant", "_kind": "memory_record", "content": "Recorded.",
         "timestamp": "2026-11-04T08:07:00+00:00"},
    ])
    # proj_a: a usage-limit stop.
    _write_session(a, "s_limit", [
        {"role": "user", "content": "run smoke", "target": "codex", "timestamp": T_USER},
        {"role": "system", "source": "daemon", "timestamp": T_REPLY,
         "content": "[Sub-agent] codex stopped with error: You've hit your usage limit. resets 21:00",
         "_meta": {"event": "sub_agent_terminal", "kind": "error"}},
    ])
    # proj_a: an ask opened in a session; plus a shell-written ask with no session.
    _asks_file(a, [
        "- open a1b2c3 2026-11-04 Should I sync the release notes too?",
        "- open e5e5e5 2026-11-01 due:2026-11-02 Renew the domain.",
    ])
    sessions = {
        "proj_a": [
            _row("s_done", pinned_target="codex"),
            _row("s_limit", pinned_target="codex"),
            _row("s_ask", ask_ids=["a1b2c3"]),
            _row("s_old", user=T_OLD_USER, reply=T_OLD_REPLY),          # 15 days: gone
            _row("s_user_last", user="2026-11-04T09:00:00+00:00"),      # user spoke last: gone
            _row("s_idle_none", reply=None),                            # never answered: gone
        ],
        "proj_b": [
            _row("s_run", status="running", reply=None),
            _row("s_worker", reply=None, pinned_target="claude-code"),  # worker_running
            _row("s_approval", status="pending_approval", reply=None),
            _row("s_queued", status="queued", reply=None),
        ],
        "proj_x": [_row("s_hidden")],
    }
    return [a, b, x], sessions


async def _feed(client):
    r = await client.get("/api/v2/workbench/feed")
    assert r.status_code == 200
    body = r.json()
    assert body["window_days"] == 14
    return {(row["project_id"], row["session_id"]): row for row in body["rows"]}


@pytest.mark.asyncio
async def test_feed_selection_rules(tmp_path, world):
    projects, sessions = world
    # Seed proj_a as already visited with nothing checked, so its replies count.
    store = CheckedStore.in_dir(str(tmp_path / "data"), now_fn=lambda: NOW)
    store.seed_if_absent("proj_a", [])
    store.seed_if_absent("proj_b", [])
    client, _, _ = _client(tmp_path, projects, sessions, {"proj_b": {"s_worker"}}, store=store)
    rows = await _feed(client)
    keys = set(rows)
    assert ("proj_a", "s_done") in keys
    assert ("proj_a", "s_limit") in keys
    assert ("proj_a", "s_ask") in keys
    assert ("proj_a", None) in keys                       # the session-less ask
    assert ("proj_a", "s_old") not in keys
    assert ("proj_a", "s_user_last") not in keys
    assert ("proj_a", "s_idle_none") not in keys
    assert {("proj_b", s) for s in ("s_run", "s_worker", "s_approval", "s_queued")} <= keys
    assert not any(pid == "proj_x" for pid, _ in keys)    # excluded: contributes nothing

    done = rows[("proj_a", "s_done")]
    assert done["agent"] == "codex"
    assert done["result_line"] == "Fixed safe-mode cwd"   # markdown stripped, record turn skipped
    assert done["stop"] is None
    limit = rows[("proj_a", "s_limit")]
    assert limit["stop"]["cause"] == "usage_limit"
    assert limit["result_line"] is None
    ask = rows[("proj_a", "s_ask")]["ask"]
    assert ask["id"] == "a1b2c3" and ask["text"].startswith("Should I sync")
    shell = rows[("proj_a", None)]["ask"]
    assert shell["id"] == "e5e5e5" and shell["overdue"] is True
    assert rows[("proj_b", "s_worker")]["worker_running"] is True
    run = rows[("proj_b", "s_run")]
    assert run["result_line"] is None and run["stop"] is None


@pytest.mark.asyncio
async def test_first_read_seeds_history_as_checked(tmp_path, world):
    projects, sessions = world
    client, store, _ = _client(tmp_path, projects, sessions)
    rows = await _feed(client)
    # Upgrade: every existing reply counts as checked; asks and live work stay.
    assert ("proj_a", "s_done") not in rows
    assert ("proj_a", "s_limit") not in rows
    assert ("proj_a", "s_ask") in rows and ("proj_a", None) in rows
    assert ("proj_b", "s_run") in rows
    assert store.project("proj_a") == {"s_done": T_REPLY, "s_limit": T_REPLY,
                                       "s_ask": T_REPLY, "s_user_last": T_REPLY}
    # The excluded project is never read, so never seeded.
    assert store.project("proj_x") is None


@pytest.mark.asyncio
async def test_checked_clears_a_card_and_broadcasts(tmp_path, world):
    projects, sessions = world
    store = CheckedStore.in_dir(str(tmp_path / "data"), now_fn=lambda: NOW)
    store.seed_if_absent("proj_a", [])
    client, _, ws = _client(tmp_path, projects, sessions, store=store)
    assert ("proj_a", "s_done") in await _feed(client)
    r = await client.post("/api/v2/workbench/checked", json={
        "project_id": "proj_a", "session_id": "s_done", "reply_at": T_REPLY})
    assert r.status_code == 200 and r.json()["checked_reply_at"] == T_REPLY
    assert ws.events == [("proj_a", {"type": "workbench.changed", "project_id": "proj_a"})]
    assert ("proj_a", "s_done") not in await _feed(client)
    # An older check never moves it back and does not broadcast again.
    r = await client.post("/api/v2/workbench/checked", json={
        "project_id": "proj_a", "session_id": "s_done", "reply_at": T_USER})
    assert r.json()["changed"] is False and len(ws.events) == 1


@pytest.mark.asyncio
async def test_checked_on_unseeded_project_seeds_first(tmp_path, world):
    projects, sessions = world
    client, store, _ = _client(tmp_path, projects, sessions)
    r = await client.post("/api/v2/workbench/checked", json={
        "project_id": "proj_a", "session_id": "s_done", "reply_at": T_REPLY})
    assert r.status_code == 200
    # The rest of the history was seeded as checked, not flooded.
    assert store.project("proj_a")["s_limit"] == T_REPLY


@pytest.mark.asyncio
async def test_checked_rejects_bad_input(tmp_path, world):
    projects, sessions = world
    client, _, _ = _client(tmp_path, projects, sessions)
    r = await client.post("/api/v2/workbench/checked", json={
        "project_id": "nope", "session_id": "s", "reply_at": T_REPLY})
    assert r.status_code == 404
    r = await client.post("/api/v2/workbench/checked", json={
        "project_id": "proj_a", "session_id": "s", "reply_at": "soon"})
    assert r.status_code == 400


def test_annotate_without_store_is_additive():
    wb.configure(FakeStore([]), FakeManager({}), None, now_fn=lambda: NOW)
    rows = wb.annotate_checked("p", [{"session_id": "a"}])
    assert rows == [{"session_id": "a", "checked_reply_at": None}]


def test_reply_line_rules():
    from agent_os.onboarding.left_off import reply_line
    assert reply_line("[STATUS: reading]\n\n- **Done**: see [docs](http://x)") == "Done: see docs"
    assert reply_line("---\n\n# Title") == "Title"
    assert reply_line("") is None and reply_line(None) is None
    long = "字" * 200
    out = reply_line(long)
    assert len(out) == 140 and out.endswith("…")


# ── Automation runs (2026-11-04) ───────────────────────────────────────

_TRIGGER_MSG = "[Triggered by file_watch '整理收件箱']\n\nChanged files: inbox/a.md\n\nReply PONG."


def _dispatch_row(handle, ts):
    """The row a worker-assigned trigger writes when it hands the run over."""
    return {"role": "system", "source": "daemon", "timestamp": ts,
            "content": f'[Sub-agent] Message sent to {handle}: "…"',
            "_meta": {"dispatch_id": "x:1", "handle": handle, "suppress_wake": True}}


@pytest.mark.asyncio
async def test_automation_runs_carry_the_agent_that_ran_them(tmp_path):
    a = _project(tmp_path, "proj_a")
    # A finished run a Codex-assigned file-watch automation started: the
    # trigger never pins the session, so only the dispatch names the worker.
    _write_session(a, "s_auto_done", [
        {"role": "user", "source": "user", "content": _TRIGGER_MSG, "timestamp": T_USER},
        _dispatch_row("codex", "2026-11-04T08:00:01+00:00"),
        {"role": "system", "source": "daemon", "timestamp": "2026-11-04T08:01:00+00:00",
         "content": "[Sub-agent] codex completed. Summary: PONG",
         "_meta": {"event": "sub_agent_terminal", "kind": "completed", "reply": "PONG"}},
        {"role": "assistant", "source": "management", "content": "PONG received.",
         "timestamp": T_REPLY},
    ])
    # A run of the same automation still going: the dispatch is already on disk.
    _write_session(a, "s_auto_run", [
        {"role": "user", "source": "user", "content": _TRIGGER_MSG, "timestamp": T_USER},
        _dispatch_row("codex", "2026-11-04T08:00:01+00:00"),
    ])
    # A manager-run automation: Orbital ran it, so no agent.
    _write_session(a, "s_auto_orbital", [
        {"role": "user", "source": "user", "content": _TRIGGER_MSG, "timestamp": T_USER},
        {"role": "assistant", "source": "management", "content": "Done.", "timestamp": T_REPLY},
    ])
    # A plain chat where the manager dispatched Codex: the card keeps the
    # session's own agent (its pin), as before.
    _write_session(a, "s_chat", [
        {"role": "user", "source": "user", "content": "ask codex", "timestamp": T_USER},
        _dispatch_row("codex", "2026-11-04T08:00:01+00:00"),
        {"role": "assistant", "source": "management", "content": "Asked.", "timestamp": T_REPLY},
    ])
    sessions = {"proj_a": [
        _row("s_auto_done", trigger_type="file_watch"),
        _row("s_auto_run", trigger_type="file_watch", reply=None),
        _row("s_auto_orbital", trigger_type="file_watch"),
        _row("s_chat"),
    ]}
    store = CheckedStore.in_dir(str(tmp_path / "data"), now_fn=lambda: NOW)
    store.seed_if_absent("proj_a", [])
    client, _, _ = _client(tmp_path, [a], sessions, {"proj_a": {"s_auto_run"}}, store=store)
    rows = await _feed(client)
    done = rows[("proj_a", "s_auto_done")]
    assert done["agent"] == "codex"
    assert done["trigger_type"] == "file_watch"
    assert done["result_line"] == "PONG received."
    running = rows[("proj_a", "s_auto_run")]
    assert running["worker_running"] is True and running["agent"] == "codex"
    assert rows[("proj_a", "s_auto_orbital")]["agent"] is None
    assert rows[("proj_a", "s_chat")]["agent"] is None


@pytest.mark.asyncio
async def test_a_checked_automation_run_leaves_the_feed(tmp_path):
    a = _project(tmp_path, "proj_a")
    _write_session(a, "s_auto", [
        {"role": "user", "source": "user", "content": _TRIGGER_MSG, "timestamp": T_USER},
        {"role": "assistant", "source": "management", "content": "Done.", "timestamp": T_REPLY},
    ])
    sessions = {"proj_a": [_row("s_auto", trigger_type="file_watch")]}
    store = CheckedStore.in_dir(str(tmp_path / "data"), now_fn=lambda: NOW)
    store.seed_if_absent("proj_a", [])
    client, _, ws = _client(tmp_path, [a], sessions, store=store)
    assert ("proj_a", "s_auto") in await _feed(client)
    r = await client.post("/api/v2/workbench/checked", json={
        "project_id": "proj_a", "session_id": "s_auto", "reply_at": T_REPLY})
    assert r.status_code == 200
    assert ("proj_a", "s_auto") not in await _feed(client)
    assert store.project("proj_a") == {"s_auto": T_REPLY}
    assert ws.events == [("proj_a", {"type": "workbench.changed", "project_id": "proj_a"})]
