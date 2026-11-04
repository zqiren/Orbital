# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Spec 138 left-off cards for one project: Orbital's own sessions plus the
native sessions of its folder and its stored group members; stop causes,
reset times, the noise filter and git facts."""

from __future__ import annotations

import json
import os
import subprocess
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from agent_os.api.routes import onboarding as onboarding_routes
from agent_os.onboarding import left_off as lo
from agent_os.onboarding.history import HistoryScanner, clear_git_cache
from agent_os.onboarding.history_paths import HistoryEnv
from agent_os.onboarding.session_import import imported_session_id
from agent_os.onboarding.sources import claude_code, codex

NOW = datetime.now(timezone.utc)


def iso(delta_hours: float) -> str:
    return (NOW - timedelta(hours=delta_hours)).isoformat()


def _write(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")


def claude_session(home, sid, cwd, *, user, at, ending="answer", entrypoint="cli"):
    rows = [{"type": "user", "sessionId": sid, "cwd": cwd, "timestamp": at,
             "entrypoint": entrypoint, "gitBranch": "feat/x",
             "message": {"role": "user", "content": user}}]
    if ending == "answer":
        rows.append({"type": "assistant", "sessionId": sid, "cwd": cwd, "timestamp": at,
                     "message": {"model": "m", "content": [
                         {"type": "text", "text": "Done — shall I also add tests?"}]}})
    elif ending == "limit":
        rows.append({"type": "assistant", "sessionId": sid, "cwd": cwd, "timestamp": at,
                     "isApiErrorMessage": True, "error": "rate_limit",
                     "message": {"model": "<synthetic>", "content": [
                         {"type": "text", "text": "You've hit your session limit · resets 11pm (UTC)"}]}})
    _write(os.path.join(home, ".claude", "projects", sid[:4], f"{sid}.jsonl"), rows)


def codex_session(home, tid, cwd, *, user, at, aborted=False):
    rows = [{"type": "session_meta", "timestamp": at,
             "payload": {"id": tid, "cwd": cwd, "originator": "codex_cli_rs", "timestamp": at}},
            {"type": "response_item", "timestamp": at,
             "payload": {"type": "message", "role": "user",
                         "content": [{"type": "input_text", "text": user}]}}]
    rows.append({"type": "event_msg", "timestamp": at,
                 "payload": {"type": "turn_aborted", "reason": "interrupted"} if aborted
                 else {"type": "task_complete"}})
    _write(os.path.join(home, ".codex", "sessions", "2026", f"rollout-{tid}.jsonl"), rows)


def orbital_session(ws, sid, rows):
    out = [{"role": "meta", "event": "session_start", "session_id": sid,
            "timestamp": rows[0]["timestamp"], "origin": None, "trigger_type": None}]
    out += [{**r, "session_id": sid} for r in rows]
    _write(os.path.join(ws, "orbital", "sessions", f"{sid}.jsonl"), out)


@pytest.fixture
def world(tmp_path, monkeypatch):
    home = str(tmp_path / "home")
    ws = os.path.join(home, "code", "app")
    desk = os.path.join(home, "Desktop")
    os.makedirs(ws)
    os.makedirs(desk)
    monkeypatch.delenv("ORBITAL_CODEX_HOME", raising=False)
    clear_git_cache()
    scanner = HistoryScanner(None, home=home, sources=[claude_code.SOURCE, codex.SOURCE],
                             env=HistoryEnv.for_home(home))
    return home, ws, desk, scanner


def test_empty_project(world):
    _home, ws, _desk, scanner = world
    assert lo.left_off_for_project({"workspace": ws}, scanner) == {
        "last_time": None, "unfinished": []}


def test_cards_from_native_and_orbital_sessions(world):
    home, ws, desk, scanner = world
    claude_session(home, "c-old", ws, user="build the settings page", at=iso(30))
    claude_session(home, "c-lim", ws, user="refactor the parser", at=iso(1), ending="limit")
    codex_session(home, "019a-ab", ws, user="wire the hide list", at=iso(2), aborted=True)
    codex_session(home, "019a-member", desk, user="the deck numbers please", at=iso(3),
                  aborted=True)
    codex_session(home, "019a-other", desk, user="unrelated desktop chat", at=iso(0.5),
                  aborted=True)
    claude_session(home, "c-probe", ws, user="PING", at=iso(0.2), ending="none")
    claude_session(home, "c-sdk", ws, user="worker task text", at=iso(0.1), ending="none",
                   entrypoint="sdk-py")
    orbital_session(ws, "orb_1", [
        {"role": "user", "source": "user", "content": "summarize yesterday", "timestamp": iso(5)},
        {"role": "assistant", "source": "management", "content": "Here it is.",
         "timestamp": iso(5)},
        {"role": "user", "source": "user", "content": "and the open PRs", "timestamp": iso(4)},
        {"role": "system", "source": "management",
         "content": "LLM error after 3 retries: Connection error.. Stopping.", "timestamp": iso(4)},
    ])
    project = {"workspace": ws, "imported_sessions": [{"agent": "codex", "native_id": "019a-member"}]}
    out = lo.left_off_for_project(project, scanner)

    last = out["last_time"]
    assert last["session_id"] == imported_session_id("claude-code", "c-old")
    assert last["agent"] == "claude-code"
    assert last["user_text"] == "build the settings page"
    assert last["git"] is None  # not a repo
    causes = [(u["session_id"], u["cause"]) for u in out["unfinished"]]
    assert causes == [
        (imported_session_id("claude-code", "c-lim"), "usage_limit"),
        (imported_session_id("codex", "019a-ab"), "interrupted"),
        (imported_session_id("codex", "019a-member"), "interrupted"),
    ]
    lim = out["unfinished"][0]
    assert lim["agent"] == "claude-code" and lim["user_text"] == "refactor the parser"
    reset = datetime.fromisoformat(lim["resets_at"]) if lim["resets_at"] else None
    assert reset is None or reset > NOW


def test_orbital_error_session_is_unfinished_and_last_time_skips_it(world):
    _home, ws, _desk, scanner = world
    orbital_session(ws, "orb_ok", [
        {"role": "user", "source": "user", "content": "plan the week", "timestamp": iso(10)},
        {"role": "assistant", "source": "management", "content": "Plan: …", "timestamp": iso(10)},
    ])
    orbital_session(ws, "orb_err", [
        {"role": "user", "source": "user", "content": "deploy it", "timestamp": iso(1)},
        {"role": "system", "source": "management",
         "content": "This project's API key was rejected (HTTP 401). Update it.", "timestamp": iso(1)},
    ])
    orbital_session(ws, "orb_pin", [
        {"role": "user", "content": "ask codex for a review", "target": "codex",
         "timestamp": iso(2)},
    ])
    out = lo.left_off_for_project({"workspace": ws}, scanner)
    assert out["last_time"]["session_id"] == "orb_ok" and out["last_time"]["agent"] is None
    assert [(u["session_id"], u["cause"], u["agent"]) for u in out["unfinished"]] == [
        ("orb_err", "network_auth", None), ("orb_pin", "no_reply", "codex")]


def test_memory_and_automation_sessions_are_never_cards(world):
    _home, ws, _desk, scanner = world
    rows = [{"role": "user", "source": "user", "content": "x y z", "timestamp": iso(1)}]
    orbital_session(ws, "orb_mem", rows)
    path = os.path.join(ws, "orbital", "sessions", "orb_mem.jsonl")
    lines = open(path).read().splitlines()
    first = json.loads(lines[0])
    first["origin"] = "memory"
    open(path, "w").write("\n".join([json.dumps(first), *lines[1:]]) + "\n")
    orbital_session(ws, "orb_trig", [
        {"role": "user", "source": "user", "timestamp": iso(1),
         "content": "[Triggered by schedule 'Daily' (every day)]\n\nSummarize"}])
    assert lo.left_off_for_project({"workspace": ws}, scanner)["unfinished"] == []


def test_opened_but_not_continued_import_keeps_its_native_stop(world):
    home, ws, _desk, scanner = world
    codex_session(home, "019a-cont", ws, user="finish the migration", at=iso(1), aborted=True)
    sid = imported_session_id("codex", "019a-cont")
    # Materialised: every reply rewritten as a completed terminal row.
    orbital_session(ws, sid, [
        {"role": "user", "source": "user", "content": "finish the migration", "target": "codex",
         "timestamp": iso(1), "_meta": {"imported": True}},
        {"role": "system", "source": "daemon", "content": "[Sub-agent] codex completed.",
         "timestamp": iso(1), "_meta": {"event": "sub_agent_terminal", "kind": "completed",
                                        "imported": True}},
    ])
    out = lo.left_off_for_project({"workspace": ws}, scanner)
    assert [(u["session_id"], u["cause"]) for u in out["unfinished"]] == [(sid, "interrupted")]
    # Once the user writes in it in Orbital, Orbital's rows decide.
    with open(os.path.join(ws, "orbital", "sessions", f"{sid}.jsonl"), "a") as fh:
        fh.write(json.dumps({"role": "user", "content": "continue please", "target": "claude-code",
                             "timestamp": iso(0.5), "session_id": sid}) + "\n")
        fh.write(json.dumps({"role": "system", "source": "daemon", "timestamp": iso(0.4),
                             "content": "[Sub-agent] claude-code completed.", "session_id": sid,
                             "_meta": {"event": "sub_agent_terminal", "kind": "completed"}}) + "\n")
    out = lo.left_off_for_project({"workspace": ws}, scanner)
    assert out["unfinished"] == []
    assert out["last_time"]["session_id"] == sid
    assert out["last_time"]["agent"] == "claude-code"
    assert out["last_time"]["user_text"] == "continue please"


def test_git_facts_on_last_time(world):
    home, ws, _desk, scanner = world
    subprocess.run(["git", "init", "-q", "-b", "main", ws], check=True)
    subprocess.run(["git", "-C", ws, "-c", "user.email=a@b", "-c", "user.name=a", "commit",
                    "-q", "--allow-empty", "-m", "after"], check=True)
    open(os.path.join(ws, "dirty.txt"), "w").close()
    clear_git_cache()
    claude_session(home, "c-git", ws, user="ship the fix", at=iso(48))
    last = lo.left_off_for_project({"workspace": ws}, scanner)["last_time"]
    assert last["git"] == {"branch": "main", "session_branch": "feat/x",
                           "uncommitted": 1, "commits_since": 1}


def test_route_404_and_shape(world, tmp_path):
    _home, ws, _desk, scanner = world
    store = MagicMock()
    store.get_project = lambda pid: {"workspace": ws} if pid == "p1" else None
    onboarding_routes.configure(project_store=store, scanner=scanner)
    app = FastAPI()
    app.include_router(onboarding_routes.projects_router)
    client = TestClient(app)
    try:
        assert client.get("/api/v2/projects/nope/left-off").status_code == 404
        r = client.get("/api/v2/projects/p1/left-off")
        assert r.status_code == 200
        assert r.json() == {"last_time": None, "unfinished": []}
    finally:
        onboarding_routes.configure()


def test_native_session_orbital_ran_as_a_worker_is_not_a_card(world):
    """Kimi Code / MiniMax Code / pi mark nothing as Orbital-started: a native
    session one of Orbital's own sessions drove (its sub_agent_thread record)
    is that Orbital session, never a second "your history" card."""
    home, ws, _desk, scanner = world
    codex_session(home, "019a-driven", ws, user="worker task text", at=iso(1), aborted=True)
    codex_session(home, "019a-users", ws, user="the user's own chat", at=iso(2), aborted=True)
    orbital_session(ws, "orb_w", [
        {"role": "user", "source": "user", "content": "fix the build", "timestamp": iso(1)},
        {"role": "meta", "event": "sub_agent_thread", "timestamp": iso(1),
         "handle": "codex", "thread": {"session_id": "019a-driven"}},
        {"role": "assistant", "source": "management", "content": "Fixed.", "timestamp": iso(1)},
    ])
    out = lo.left_off_for_project({"workspace": ws}, scanner)
    ids = [u["session_id"] for u in out["unfinished"]]
    assert imported_session_id("codex", "019a-driven") not in ids
    assert ids == [imported_session_id("codex", "019a-users")]
    assert out["last_time"]["session_id"] == "orb_w"


def test_orbital_run_native_threads_are_not_the_users(world, monkeypatch):
    """Orbital's own kimi / minimax ACP runs carry no marker in the native
    store; session_import's ``orbital_thread_ids`` names them (138-R)."""
    import agent_os.onboarding.session_import as si
    home, ws, _desk, scanner = world
    codex_session(home, "019a-mine", ws, user="a thing I asked", at=iso(2), aborted=True)
    codex_session(home, "019a-orb", ws, user="a worker task", at=iso(1), aborted=True)
    monkeypatch.setattr(si, "orbital_thread_ids", lambda workspace: {"019a-orb"}, raising=False)
    out = lo.left_off_for_project({"workspace": ws}, scanner)
    assert [u["session_id"] for u in out["unfinished"]] == [imported_session_id("codex", "019a-mine")]


def _complete_call(cid, summary):
    return {"id": cid, "type": "function",
            "function": {"name": "mark_task_complete",
                         "arguments": json.dumps({"summary": summary})}}


def test_text_and_mark_task_complete_in_one_row_is_finished(world):
    """Workbench walkthrough: a finished Bakery task read as 已中断. The
    agent's last row carries its answer AND mark_task_complete; the loop then
    writes the tool result and a "Task completed" system row. Finished."""
    _home, ws, _desk, scanner = world
    orbital_session(ws, "orb_done", [
        {"role": "user", "source": "user", "content": "add opening hours", "timestamp": iso(1)},
        {"role": "assistant", "source": "management", "content": "Added the hours to index.html.",
         "tool_calls": [_complete_call("c1", "Opening hours added")], "timestamp": iso(1)},
        {"role": "tool", "tool_call_id": "c1", "content": "Task marked complete.",
         "timestamp": iso(1)},
        {"role": "system", "content": "Task completed: Opening hours added",
         "source": "queue_signal", "signal": "complete",
         "payload": {"summary": "Opening hours added"}, "timestamp": iso(1)},
    ])
    parsed = lo.read_orbital_session(os.path.join(ws, "orbital", "sessions", "orb_done.jsonl"))
    assert parsed.entry.stop is None
    assert parsed.reply_line == "Added the hours to index.html."
    out = lo.left_off_for_project({"workspace": ws}, scanner)
    assert out["unfinished"] == []
    assert out["last_time"]["session_id"] == "orb_done"


def test_mark_task_complete_alone_is_finished_with_its_summary(world):
    _home, ws, _desk, scanner = world
    orbital_session(ws, "orb_sig", [
        {"role": "user", "source": "user", "content": "fix the menu grid", "timestamp": iso(1)},
        {"role": "assistant", "source": "management", "content": None,
         "tool_calls": [_complete_call("c2", "Menu grid fixed on mobile")], "timestamp": iso(1)},
        {"role": "tool", "tool_call_id": "c2", "content": "Task marked complete.",
         "timestamp": iso(1)},
    ])
    parsed = lo.read_orbital_session(os.path.join(ws, "orbital", "sessions", "orb_sig.jsonl"))
    assert parsed.entry.stop is None
    assert parsed.reply_line == "Menu grid fixed on mobile"


def test_tool_use_then_final_text_is_finished(world):
    _home, ws, _desk, scanner = world
    orbital_session(ws, "orb_tools", [
        {"role": "user", "source": "user", "content": "read the menu", "timestamp": iso(1)},
        {"role": "assistant", "source": "management", "content": "",
         "tool_calls": [{"id": "r1", "type": "function",
                         "function": {"name": "read", "arguments": "{\"path\": \"menu.html\"}"}}],
         "timestamp": iso(1)},
        {"role": "tool", "tool_call_id": "r1", "content": "<html>…", "timestamp": iso(1)},
        {"role": "assistant", "source": "management", "content": "The menu has 12 items.",
         "timestamp": iso(1)},
    ])
    parsed = lo.read_orbital_session(os.path.join(ws, "orbital", "sessions", "orb_tools.jsonl"))
    assert parsed.entry.stop is None and parsed.reply_line == "The menu has 12 items."


def test_a_tool_call_left_hanging_is_still_interrupted(world):
    _home, ws, _desk, scanner = world
    orbital_session(ws, "orb_hang", [
        {"role": "user", "source": "user", "content": "deploy", "timestamp": iso(1)},
        {"role": "assistant", "source": "management", "content": "Deploying now.",
         "tool_calls": [{"id": "s1", "type": "function",
                         "function": {"name": "shell", "arguments": "{\"command\": \"make\"}"}}],
         "timestamp": iso(1)},
        {"role": "tool", "tool_call_id": "s1", "content": "ok", "timestamp": iso(1)},
    ])
    parsed = lo.read_orbital_session(os.path.join(ws, "orbital", "sessions", "orb_hang.jsonl"))
    assert parsed.entry.stop is not None and parsed.entry.stop.cause == "interrupted"
