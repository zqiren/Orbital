# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Agent-held connectors: the per-agent registry of what each agent reported.

Names and a status only, Orbital's own mount never recorded, one atomic file
in the data dir that a missing or corrupt copy reads as empty.
"""

from __future__ import annotations

import json
import os

import pytest

from agent_os.daemon_v2 import agent_connectors as ac


@pytest.fixture(autouse=True)
def _fresh(tmp_path):
    ac.configure(str(tmp_path))
    yield
    ac.configure(None)


CLAUDE_INIT = [
    {"name": "plugin:chatcut:chatcut", "status": "needs-auth", "source": "plugin"},
    {"name": "figma", "status": "needs-auth", "source": "user"},
    {"name": "playwright", "status": "connected", "source": "user"},
    {"name": "claude.ai Gmail", "status": "pending", "source": "claudeai"},
    {"name": "claude.ai Google Calendar", "status": "pending", "source": "claudeai"},
    {"name": "orbital", "status": "connected", "source": "dynamic"},
]


def _file(tmp_path):
    with open(tmp_path / ac.FILE_NAME, encoding="utf-8") as f:
        return json.load(f)


def test_records_names_and_status_only_without_orbital(tmp_path):
    assert ac.record("claude-code", ac.claude_servers(CLAUDE_INIT)) is True
    data = _file(tmp_path)
    assert data["version"] == 1
    entry = data["agents"]["claude-code"]
    assert [s["name"] for s in entry["servers"]] == [
        "claude.ai Gmail", "claude.ai Google Calendar", "figma",
        "playwright", "plugin:chatcut:chatcut"]
    assert all(set(s) == {"name", "status"} for s in entry["servers"])
    assert entry["last_seen"].endswith("Z")
    assert "orbital" not in json.dumps(data)


def test_extra_fields_never_reach_the_file(tmp_path):
    secret = [{"name": "notes", "status": "connected",
               "config": {"url": "https://secret.example/mcp",
                          "headers": {"Authorization": "Bearer xyz"}},
               "env": {"TOKEN": "abc"}}]
    ac.record("claude-code", ac.claude_servers(secret))
    text = (tmp_path / ac.FILE_NAME).read_text(encoding="utf-8")
    assert "secret.example" not in text and "xyz" not in text and "abc" not in text


def test_display_names_strip_account_and_plugin_prefixes():
    ac.record("claude-code", ac.claude_servers(CLAUDE_INIT))
    held = ac.held("claude-code")
    assert [c["name"] for c in held] == [
        "chatcut", "figma", "Gmail", "Google Calendar", "playwright"]
    gmail = next(c for c in held if c["name"] == "Gmail")
    assert gmail == {"name": "Gmail", "raw_name": "claude.ai Gmail", "status": "pending"}


def test_unchanged_report_is_not_rewritten_but_the_first_one_is(tmp_path):
    assert ac.record("codex", [{"name": "node_repl", "status": "connected"}]) is True
    mtime = os.stat(tmp_path / ac.FILE_NAME).st_mtime_ns
    assert ac.record("codex", [{"name": "node_repl", "status": "connected"}]) is False
    assert os.stat(tmp_path / ac.FILE_NAME).st_mtime_ns == mtime
    assert ac.record("codex", [{"name": "node_repl", "status": "failed"}]) is True


def test_a_new_process_writes_the_first_report_again_for_last_seen(tmp_path):
    ac.record("codex", [{"name": "node_repl", "status": "connected"}])
    ac.configure(str(tmp_path))          # a daemon restart
    assert ac.record("codex", [{"name": "node_repl", "status": "connected"}]) is True


def test_survives_a_restart(tmp_path):
    ac.record("codex", ac.codex_servers({"data": [
        {"name": "node_repl", "tools": {"js": {}}},
        {"name": "computer-use", "tools": {}, "authStatus": "unsupported"}]}))
    ac.configure(str(tmp_path))
    assert [c["name"] for c in ac.held("codex")] == ["computer-use", "node_repl"]
    assert ac.last_seen("codex")


@pytest.mark.parametrize("content", ["", "{not json", "[]", '{"agents": []}',
                                     '{"agents": {"codex": "x"}}'])
def test_missing_or_corrupt_file_reads_as_empty(tmp_path, content):
    (tmp_path / ac.FILE_NAME).write_text(content, encoding="utf-8")
    ac.configure(str(tmp_path))
    assert ac.held("codex") == []
    assert ac.record("codex", [{"name": "node_repl", "status": "connected"}]) is True
    assert _file(tmp_path)["agents"]["codex"]["servers"][0]["name"] == "node_repl"


def test_unknown_keys_from_a_newer_build_are_tolerated(tmp_path):
    (tmp_path / ac.FILE_NAME).write_text(json.dumps({
        "version": 9, "future": True,
        "agents": {"codex": {"servers": [{"name": "x", "status": "connected", "more": 1}],
                             "last_seen": "2026-11-17T00:00:00Z", "new": "field"}}}),
        encoding="utf-8")
    ac.configure(str(tmp_path))
    assert [c["name"] for c in ac.held("codex")] == ["x"]


def test_unconfigured_records_and_knows_nothing(tmp_path):
    ac.configure(None)
    assert ac.record("codex", [{"name": "node_repl", "status": "connected"}]) is False
    assert ac.held("codex") == [] and ac.roster_names("codex") == ""
    assert not os.path.exists(tmp_path / ac.FILE_NAME)


def test_record_never_raises_on_a_write_failure(tmp_path, monkeypatch):
    def boom(*a, **kw):
        raise OSError("disk full")
    monkeypatch.setattr(ac, "atomic_write_text", boom)
    assert ac.record("codex", [{"name": "node_repl", "status": "connected"}]) is False


def test_codex_status_is_derived_from_tools_auth_and_errors():
    servers = ac.codex_servers({"data": [
        {"name": "node_repl", "tools": {"js": {}}, "authStatus": "unsupported"},
        {"name": "chatcut", "tools": {}, "authStatus": "notLoggedIn",
         "toolsError": "OAuth authorization required"},
        {"name": "broken", "tools": {}, "toolsError": "spawn failed"},
        {"name": "computer-use", "tools": {}, "authStatus": "unsupported"},
        {"name": "orbital", "tools": {"dispatch": {}}},
    ]})
    assert {s["name"]: s["status"] for s in servers} == {
        "node_repl": "connected", "chatcut": "needs-auth", "broken": "failed",
        "computer-use": "idle", "orbital": "connected"}


def test_codex_builtins_and_disabled_servers_are_not_held():
    ac.record("codex", [{"name": "codex_apps", "status": "connected"},
                        {"name": "codex_app", "status": "idle"},
                        {"name": "cua_repl", "status": "connected"},
                        {"name": "node_repl", "status": "connected"}])
    ac.record("claude-code", [{"name": "figma", "status": "disabled"},
                              {"name": "playwright", "status": "connected"}])
    assert [c["name"] for c in ac.held("codex")] == ["node_repl"]
    assert [c["name"] for c in ac.held("claude-code")] == ["playwright"]


@pytest.mark.parametrize("bad", [None, "x", 5, [{"status": "connected"}], {"data": "x"}])
def test_malformed_reports_are_dropped(bad):
    assert ac.claude_servers(bad if not isinstance(bad, dict) else None) in (None, [])
    assert ac.codex_servers(bad) in (None, [])


def test_roster_names_are_capped_and_ordered():
    ac.record("claude-code", [{"name": f"srv{i:02d}", "status": "connected"}
                              for i in range(11)])
    assert ac.roster_names("claude-code") == (
        "srv00, srv01, srv02, srv03, srv04, srv05, srv06, srv07 and 3 more")
    assert ac.roster_names("codex") == ""


def test_a_long_or_huge_report_is_bounded(tmp_path):
    ac.record("claude-code", [{"name": "x" * 500 + str(i), "status": "connected"}
                              for i in range(200)])
    entry = _file(tmp_path)["agents"]["claude-code"]
    assert len(entry["servers"]) <= 64
    assert all(len(s["name"]) <= 120 for s in entry["servers"])


async def test_the_sink_writes_off_the_event_loop(tmp_path):
    import asyncio
    sink = ac.make_sink("codex")
    sink([{"name": "node_repl", "status": "connected"}])
    for _ in range(50):
        if (tmp_path / ac.FILE_NAME).exists():
            break
        await asyncio.sleep(0.02)
    assert [c["name"] for c in ac.held("codex")] == ["node_repl"]
