# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Session import (keyless onboarding, 2026-10-21): a project folder's Claude
Code / Codex conversations listed as sessions (metadata only) and converted
into Orbital's own row shapes on first open.

Fixtures are SYNTHETIC rows in the real native formats: a fake Claude home
(``projects/<slug>/<id>.jsonl``) and a fake Codex home (``state_5.sqlite``
``threads`` table + ``sessions/YYYY/MM/DD/rollout-*.jsonl``).
"""

from __future__ import annotations

import json
import os
import sqlite3

import pytest

from agent_os.agent.transports.sdk_transport import SDKTransport
from agent_os.onboarding import session_import as si


CLAUDE_ID = "11111111-2222-3333-4444-555555555555"
CODEX_ID = "019f0000-aaaa-7bbb-8ccc-dddddddddddd"


def _symlink_or_skip(target, link):
    try:
        os.symlink(target, link)
    except (OSError, NotImplementedError):
        # Windows without Developer Mode / symlink privilege can't create one.
        pytest.skip("symlink creation not permitted on this platform")


@pytest.fixture(autouse=True)
def _fresh_cache(monkeypatch, tmp_path):
    """Empty native homes by default (never the real ~/.claude, ~/.codex —
    and never via HOME, which Windows' expanduser ignores); tests that need
    the env-derived homes set them explicitly."""
    si.clear_cache()
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path / "no-claude-home"))
    monkeypatch.setenv("ORBITAL_CODEX_HOME", str(tmp_path / "no-codex-home"))
    monkeypatch.delenv("CODEX_HOME", raising=False)
    yield
    si.clear_cache()


# --------------------------------------------------------------------------
# Fixture builders
# --------------------------------------------------------------------------


def _claude_user(text, ts, cwd, sid=CLAUDE_ID, **extra):
    return {"type": "user", "sessionId": sid, "cwd": cwd, "entrypoint": "cli",
            "isSidechain": False, "timestamp": ts,
            "message": {"role": "user", "content": text}, **extra}


def _claude_assistant(blocks, ts, cwd, sid=CLAUDE_ID, **extra):
    return {"type": "assistant", "sessionId": sid, "cwd": cwd, "entrypoint": "cli",
            "isSidechain": False, "timestamp": ts,
            "message": {"role": "assistant", "model": "claude-x", "content": blocks},
            **extra}


def _claude_tool_result(ts, cwd, sid=CLAUDE_ID):
    return {"type": "user", "sessionId": sid, "cwd": cwd, "isSidechain": False,
            "timestamp": ts, "message": {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "t1", "content": "ok"}]}}


def write_claude_session(claude_dir, workspace, rows, sid=CLAUDE_ID):
    d = os.path.join(claude_dir, "projects", SDKTransport.claude_project_slug(workspace))
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, f"{sid}.jsonl")
    with open(path, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    return path


def basic_claude_rows(ws):
    return [
        {"type": "permission-mode", "permissionMode": "default", "sessionId": CLAUDE_ID},
        _claude_user("Fix the flaky test", "2026-10-01T10:00:00.000Z", ws),
        _claude_assistant([{"type": "thinking", "thinking": "hmm"}],
                          "2026-10-01T10:00:01.000Z", ws),
        _claude_assistant([{"type": "text", "text": "Looking."}],
                          "2026-10-01T10:00:02.000Z", ws),
        _claude_assistant([{"type": "tool_use", "id": "t1", "name": "Read", "input": {}}],
                          "2026-10-01T10:00:03.000Z", ws),
        _claude_tool_result("2026-10-01T10:00:04.000Z", ws),
        _claude_assistant([{"type": "tool_use", "id": "t2", "name": "Read", "input": {}}],
                          "2026-10-01T10:00:05.000Z", ws),
        _claude_tool_result("2026-10-01T10:00:06.000Z", ws),
        _claude_assistant([{"type": "tool_use", "id": "t3", "name": "Edit", "input": {}}],
                          "2026-10-01T10:00:07.000Z", ws),
        _claude_tool_result("2026-10-01T10:00:08.000Z", ws),
        _claude_assistant([{"type": "text", "text": "Fixed it."}],
                          "2026-10-01T10:00:09.000Z", ws),
        {"type": "attachment", "sessionId": CLAUDE_ID, "attachment": {"x": 1}},
        {"type": "api-request-blob", "sessionId": CLAUDE_ID, "message": {"content": "blob"}},
        _claude_user("Thanks", "2026-10-01T11:00:00.000Z", ws),
        _claude_assistant([{"type": "text", "text": "You're welcome."}],
                          "2026-10-01T11:00:01.000Z", ws),
        {"type": "ai-title", "aiTitle": "Flaky test fix", "sessionId": CLAUDE_ID},
    ]


def make_codex_home(codex_dir, threads):
    """``threads``: list of dicts with id/cwd/rollout_path/… (missing columns
    default)."""
    os.makedirs(codex_dir, exist_ok=True)
    con = sqlite3.connect(os.path.join(codex_dir, "state_5.sqlite"))
    con.execute(
        "CREATE TABLE threads (id TEXT PRIMARY KEY, rollout_path TEXT NOT NULL,"
        " created_at INTEGER NOT NULL, updated_at INTEGER NOT NULL, source TEXT NOT NULL,"
        " cwd TEXT NOT NULL, title TEXT NOT NULL, archived INTEGER NOT NULL DEFAULT 0,"
        " first_user_message TEXT NOT NULL DEFAULT '', updated_at_ms INTEGER,"
        " thread_source TEXT, preview TEXT NOT NULL DEFAULT '', originator TEXT)")
    for t in threads:
        con.execute(
            "INSERT INTO threads (id, rollout_path, created_at, updated_at, source, cwd,"
            " title, archived, first_user_message, updated_at_ms, thread_source, originator)"
            " VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (t["id"], t["rollout_path"], 1, t.get("updated_at", 1790000000),
             t.get("source", "exec"), t["cwd"], t.get("title", ""), t.get("archived", 0),
             t.get("first_user_message", ""), t.get("updated_at_ms"),
             t.get("thread_source"), t.get("originator", "codex_exec")))
    con.commit()
    con.close()


def codex_rollout_rows(ws, tid=CODEX_ID):
    def ri(ts, payload):
        return {"timestamp": ts, "type": "response_item", "payload": payload}
    return [
        {"timestamp": "2026-10-02T09:00:00.000Z", "type": "session_meta",
         "payload": {"id": tid, "cwd": ws, "originator": "codex_exec", "source": "exec",
                     "base_instructions": {"text": "BASE"}}},
        ri("2026-10-02T09:00:00.100Z", {"type": "message", "role": "developer",
                                         "content": [{"type": "input_text", "text": "DEV"}]}),
        ri("2026-10-02T09:00:00.200Z", {"type": "message", "role": "user", "content": [
            {"type": "input_text", "text": "<environment_context>\n  <cwd>x</cwd>\n</environment_context>"}]}),
        ri("2026-10-02T09:00:01.000Z", {"type": "message", "role": "user", "content": [
            {"type": "input_text", "text": "Rename the helper"}]}),
        {"timestamp": "2026-10-02T09:00:01.100Z", "type": "event_msg",
         "payload": {"type": "token_count", "info": {}}},
        {"timestamp": "2026-10-02T09:00:01.200Z", "type": "turn_context", "payload": {"cwd": ws}},
        {"timestamp": "2026-10-02T09:00:01.300Z", "type": "world_state", "payload": {}},
        ri("2026-10-02T09:00:02.000Z", {"type": "reasoning", "summary": [], "encrypted_content": "x"}),
        ri("2026-10-02T09:00:03.000Z", {"type": "function_call", "name": "exec_command",
                                         "arguments": "{}", "call_id": "c1"}),
        ri("2026-10-02T09:00:04.000Z", {"type": "function_call_output", "call_id": "c1",
                                         "output": "ok"}),
        ri("2026-10-02T09:00:05.000Z", {"type": "custom_tool_call", "name": "apply_patch",
                                         "input": "*** Begin", "call_id": "c2"}),
        ri("2026-10-02T09:00:06.000Z", {"type": "custom_tool_call_output", "call_id": "c2",
                                         "output": "done"}),
        ri("2026-10-02T09:00:07.000Z", {"type": "message", "role": "assistant", "content": [
            {"type": "output_text", "text": "Renamed."}]}),
    ]


def write_codex_rollout(codex_dir, ws, rows, tid=CODEX_ID):
    d = os.path.join(codex_dir, "sessions", "2026", "10", "02")
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, f"rollout-2026-10-02T09-00-00-{tid}.jsonl")
    with open(path, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    return path


@pytest.fixture
def ws(tmp_path):
    w = tmp_path / "project"
    w.mkdir()
    return os.path.realpath(str(w))


@pytest.fixture
def homes(tmp_path):
    return str(tmp_path / "claude-home"), str(tmp_path / "codex-home")


# --------------------------------------------------------------------------
# I1 — listing
# --------------------------------------------------------------------------


def test_imported_session_id_is_stable_and_bounded():
    sid = si.imported_session_id("codex", CODEX_ID)
    assert sid == "imp_codex_019f0000aaaa7bbb"
    assert si.imported_session_id("codex", CODEX_ID) == sid
    assert si.imported_session_id("claude-code", CLAUDE_ID) == "imp_claude-code_1111111122223333"
    assert si.is_imported_session_id(sid)
    assert not si.is_imported_session_id("proj_abcd1234")
    # A project named "Imp" mints chat ids like this one.
    assert not si.is_imported_session_id("imp_85d5e77f")
    assert not si.is_imported_session_id("imp_codex_019f0000aaaa7bbbX")


def test_lists_a_claude_session_with_its_ai_title(ws, homes):
    claude_dir, codex_dir = homes
    path = write_claude_session(claude_dir, ws, basic_claude_rows(ws))
    found = si.list_native_sessions(ws, claude_dir=claude_dir, codex_dir=codex_dir)
    assert len(found) == 1
    s = found[0]
    assert (s.agent, s.native_id, s.path) == ("claude-code", CLAUDE_ID, path)
    assert s.title == "Flaky test fix"
    assert s.last_activity == "2026-10-01T11:00:01+00:00"
    assert s.size_bytes == os.path.getsize(path)
    assert s.session_id == "imp_claude-code_1111111122223333"


def test_claude_title_falls_back_to_the_first_user_message_truncated(ws, homes):
    claude_dir, codex_dir = homes
    long = "Please investigate why the nightly build keeps failing on the release branch"
    write_claude_session(claude_dir, ws, [
        _claude_user(long, "2026-10-01T10:00:00Z", ws),
        _claude_assistant([{"type": "text", "text": "ok"}], "2026-10-01T10:00:01Z", ws),
    ])
    (s,) = si.list_native_sessions(ws, claude_dir=claude_dir, codex_dir=codex_dir)
    from agent_os.agent.session import _derive_name
    assert s.title == _derive_name(long)
    assert s.title.endswith("…") and len(s.title) <= 51


def test_claude_sessions_without_a_user_message_are_skipped(ws, homes):
    claude_dir, codex_dir = homes
    # Sidechain-only (a sub-agent file) and a meta/tool-result-only file.
    write_claude_session(claude_dir, ws, [
        {**_claude_user("side task", "2026-10-01T10:00:00Z", ws), "isSidechain": True},
    ], sid="aaaaaaaa-0000-0000-0000-000000000000")
    write_claude_session(claude_dir, ws, [
        {**_claude_user("Caveat: generated", "2026-10-01T10:00:00Z", ws), "isMeta": True},
        _claude_user("<command-name>/clear</command-name>", "2026-10-01T10:00:01Z", ws),
        _claude_tool_result("2026-10-01T10:00:02Z", ws),
    ], sid="bbbbbbbb-0000-0000-0000-000000000000")
    assert si.list_native_sessions(ws, claude_dir=claude_dir, codex_dir=codex_dir) == []


def test_orbitals_own_sdk_sessions_are_not_listed(ws, homes):
    claude_dir, codex_dir = homes
    rows = [dict(r, entrypoint="sdk-py") if r.get("type") in ("user", "assistant") else r
            for r in basic_claude_rows(ws)]
    write_claude_session(claude_dir, ws, rows)
    assert si.list_native_sessions(ws, claude_dir=claude_dir, codex_dir=codex_dir) == []


def test_claude_slug_collision_is_resolved_by_the_recorded_cwd(ws, homes, tmp_path):
    """The slug is lossy: another folder can encode to the same dir name."""
    claude_dir, codex_dir = homes
    other = str(tmp_path / "elsewhere")
    write_claude_session(claude_dir, ws, [
        _claude_user("not here", "2026-10-01T10:00:00Z", other),
    ])
    assert si.list_native_sessions(ws, claude_dir=claude_dir, codex_dir=codex_dir) == []


def test_claude_cwd_probe_fallback_finds_a_drifted_slug_dir(ws, homes):
    claude_dir, codex_dir = homes
    d = os.path.join(claude_dir, "projects", "some-drifted-name")
    os.makedirs(d)
    with open(os.path.join(d, f"{CLAUDE_ID}.jsonl"), "w") as fh:
        for r in basic_claude_rows(ws):
            fh.write(json.dumps(r) + "\n")
    found = si.list_native_sessions(ws, claude_dir=claude_dir, codex_dir=codex_dir)
    assert [s.native_id for s in found] == [CLAUDE_ID]


def test_claude_config_dir_env_override(ws, homes, monkeypatch):
    claude_dir, codex_dir = homes
    write_claude_session(claude_dir, ws, basic_claude_rows(ws))
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", claude_dir)
    monkeypatch.setenv("ORBITAL_CODEX_HOME", codex_dir)
    found = si.list_native_sessions(ws)
    assert [s.agent for s in found] == ["claude-code"]


def test_title_read_is_bounded_head_and_tail(ws, homes, monkeypatch):
    """A big transcript is never read whole for its metadata: the ai-title
    in the tail is still found."""
    claude_dir, codex_dir = homes
    rows = [_claude_user("first ask", "2026-10-01T10:00:00Z", ws)]
    filler = "x" * 2000
    for i in range(400):  # ~0.9 MB of assistant rows
        rows.append(_claude_assistant([{"type": "text", "text": filler}],
                                      "2026-10-01T10:00:01Z", ws))
    rows.append({"type": "ai-title", "aiTitle": "Tail title", "sessionId": CLAUDE_ID})
    path = write_claude_session(claude_dir, ws, rows)
    reads = []
    real_open = open

    def spy_open(file, mode="r", *a, **k):
        fh = real_open(file, mode, *a, **k)
        if str(file) == path and "b" in mode:
            orig = fh.read

            def read(n=-1):
                data = orig(n)
                reads.append(len(data))
                return data
            fh.read = read
        return fh
    monkeypatch.setattr("builtins.open", spy_open)
    (s,) = si.list_native_sessions(ws, claude_dir=claude_dir, codex_dir=codex_dir)
    assert s.title == "Tail title"
    assert sum(reads) <= si._HEAD_BYTES + si._TAIL_BYTES
    assert os.path.getsize(path) > si._HEAD_BYTES + si._TAIL_BYTES


def test_listing_is_cached_by_path_mtime_and_size(ws, homes, monkeypatch):
    claude_dir, codex_dir = homes
    path = write_claude_session(claude_dir, ws, basic_claude_rows(ws))
    calls = []
    real = si._claude_metadata
    monkeypatch.setattr(si, "_claude_metadata",
                        lambda p, st: calls.append(p) or real(p, st))
    si.list_native_sessions(ws, claude_dir=claude_dir, codex_dir=codex_dir)
    si.list_native_sessions(ws, claude_dir=claude_dir, codex_dir=codex_dir)
    assert calls == [path]
    with open(path, "a") as fh:
        fh.write(json.dumps({"type": "ai-title", "aiTitle": "Renamed",
                             "sessionId": CLAUDE_ID}) + "\n")
    (s,) = si.list_native_sessions(ws, claude_dir=claude_dir, codex_dir=codex_dir)
    assert len(calls) == 2 and s.title == "Renamed"


def test_lists_codex_threads_from_the_index(ws, homes, tmp_path):
    claude_dir, codex_dir = homes
    rollout = write_codex_rollout(codex_dir, ws, codex_rollout_rows(ws))
    gone = str(tmp_path / "missing-rollout.jsonl")
    make_codex_home(codex_dir, [
        {"id": CODEX_ID, "cwd": ws, "rollout_path": rollout, "title": "Rename the helper",
         "first_user_message": "Rename the helper", "updated_at_ms": 1790000000123},
        # archived, another folder, no user message, Orbital's own, codex sub-agent,
        # and one whose rollout is gone — none are listed.
        {"id": "t-arch", "cwd": ws, "rollout_path": rollout, "first_user_message": "a",
         "archived": 1},
        {"id": "t-other", "cwd": str(tmp_path), "rollout_path": rollout,
         "first_user_message": "a"},
        {"id": "t-empty", "cwd": ws, "rollout_path": rollout},
        {"id": "t-orbital", "cwd": ws, "rollout_path": rollout, "first_user_message": "a",
         "originator": "orbital", "source": "vscode"},
        {"id": "t-sub", "cwd": ws, "rollout_path": rollout, "first_user_message": "a",
         "source": '{"subagent":{"other":"guardian"}}', "thread_source": "subagent"},
        {"id": "t-gone", "cwd": ws, "rollout_path": gone, "first_user_message": "a"},
    ])
    found = si.list_native_sessions(ws, claude_dir=claude_dir, codex_dir=codex_dir)
    assert [(s.agent, s.native_id) for s in found] == [("codex", CODEX_ID)]
    s = found[0]
    assert s.title == "Rename the helper"
    assert s.rollout_path == rollout
    assert s.last_activity == si._iso_epoch(1790000000.123)
    assert s.size_bytes == os.path.getsize(rollout)


def test_codex_cwd_matching_is_by_realpath(ws, homes, tmp_path):
    claude_dir, codex_dir = homes
    link = str(tmp_path / "link-to-project")
    _symlink_or_skip(ws, link)
    rollout = write_codex_rollout(codex_dir, ws, codex_rollout_rows(ws))
    make_codex_home(codex_dir, [
        {"id": CODEX_ID, "cwd": link, "rollout_path": rollout,
         "first_user_message": "Rename the helper"},
    ])
    found = si.list_native_sessions(ws, claude_dir=claude_dir, codex_dir=codex_dir)
    assert [s.native_id for s in found] == [CODEX_ID]
    # …and from the symlinked workspace too.
    si.clear_cache()
    found = si.list_native_sessions(link, claude_dir=claude_dir, codex_dir=codex_dir)
    assert [s.native_id for s in found] == [CODEX_ID]


def test_codex_falls_back_to_rollouts_without_an_index(ws, homes):
    claude_dir, codex_dir = homes
    write_codex_rollout(codex_dir, ws, codex_rollout_rows(ws))
    found = si.list_native_sessions(ws, claude_dir=claude_dir, codex_dir=codex_dir)
    assert [(s.agent, s.native_id, s.title) for s in found] == [
        ("codex", CODEX_ID, "Rename the helper")]


def test_codex_home_env_precedence(ws, homes, tmp_path, monkeypatch):
    claude_dir, codex_dir = homes
    write_codex_rollout(codex_dir, ws, codex_rollout_rows(ws))
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", claude_dir)
    monkeypatch.delenv("ORBITAL_CODEX_HOME", raising=False)
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "nothing-here"))
    assert si.list_native_sessions(ws) == []
    monkeypatch.setenv("ORBITAL_CODEX_HOME", codex_dir)  # wins over CODEX_HOME
    assert [s.agent for s in si.list_native_sessions(ws)] == ["codex"]


def test_sorted_newest_first_across_agents(ws, homes):
    claude_dir, codex_dir = homes
    write_claude_session(claude_dir, ws, basic_claude_rows(ws))  # 2026-10-01
    write_codex_rollout(codex_dir, ws, codex_rollout_rows(ws))   # 2026-10-02
    found = si.list_native_sessions(ws, claude_dir=claude_dir, codex_dir=codex_dir)
    assert [s.agent for s in found] == ["codex", "claude-code"]


def test_missing_sources_list_nothing(ws, tmp_path):
    assert si.list_native_sessions(ws, claude_dir=str(tmp_path / "a"),
                                   codex_dir=str(tmp_path / "b")) == []
    assert si.list_native_sessions(str(tmp_path / "no-such-dir")) == []


# --------------------------------------------------------------------------
# I2 — conversion
# --------------------------------------------------------------------------


def _terminals(rows):
    return [r for r in rows if r.get("_meta", {}).get("event") == "sub_agent_terminal"]


def test_claude_conversion_rows(ws, homes):
    claude_dir, _ = homes
    path = write_claude_session(claude_dir, ws, basic_claude_rows(ws))
    conv = si.convert_transcript("claude-code", path, handle="claude-code",
                                 now="2026-10-21T00:00:00+00:00")
    header, *rows = conv.rows
    assert header["role"] == "system" and header["source"] == "daemon"
    assert header["content"] == (
        "[Imported] Conversation imported from Claude Code (last active 2026-10-01)")
    assert header["_meta"]["event"] == "imported_transcript"
    assert header["_meta"]["agent"] == "claude-code"
    assert header["_meta"]["native_id"] == CLAUDE_ID
    assert header["_meta"]["truncated"] is False
    assert header["_meta"]["last_activity"] == "2026-10-01T11:00:01+00:00"

    assert [r["role"] for r in rows] == ["user", "system", "user", "system"]
    u1 = rows[0]
    assert u1 == {"role": "user", "source": "user", "content": "Fix the flaky test",
                  "target": "claude-code", "timestamp": "2026-10-01T10:00:00+00:00",
                  "_meta": {"imported": True}}
    t1 = rows[1]
    reply = "Looking.\n\nFixed it.\n\n(tools: Read ×2, Edit ×1)"
    assert t1["_meta"] == {
        "event": "sub_agent_terminal", "kind": "completed", "handle": "claude-code",
        "reply": reply, "display_content": t1["content"], "suppress_wake": True,
        "imported": True,
    }
    assert t1["content"] == f"[Sub-agent] claude-code completed. Summary: {reply[:200]}"
    assert t1["timestamp"] == "2026-10-01T10:00:09+00:00"
    assert rows[3]["_meta"]["reply"] == "You're welcome."
    # Thinking, attachments and api-request blobs never reach a row.
    blob = json.dumps(conv.all_rows())
    assert "hmm" not in blob and "blob" not in blob


def test_claude_thread_record_is_the_set_sub_agent_thread_shape(ws, homes):
    claude_dir, _ = homes
    path = write_claude_session(claude_dir, ws, basic_claude_rows(ws))
    conv = si.convert_transcript("claude-code", path, handle="claude-code",
                                 now="2026-10-21T00:00:00+00:00")
    rec = conv.all_rows()[-1]
    assert rec == {"role": "meta", "event": "sub_agent_thread",
                   "timestamp": "2026-10-21T00:00:00+00:00", "handle": "claude-code",
                   "thread": {"session_id": CLAUDE_ID, "model": None,
                              "last_used_at": "2026-10-21T00:00:00+00:00"}}


def test_codex_conversion_rows(ws, homes):
    _, codex_dir = homes
    path = write_codex_rollout(codex_dir, ws, codex_rollout_rows(ws))
    conv = si.convert_transcript("codex", path, handle="codex",
                                 now="2026-10-21T00:00:00+00:00")
    header, *rows = conv.rows
    assert header["content"].startswith("[Imported] Conversation imported from Codex ")
    assert [r["role"] for r in rows] == ["user", "system"]
    assert rows[0]["content"] == "Rename the helper"
    assert rows[1]["_meta"]["reply"] == "Renamed.\n\n(tools: exec_command ×1, apply_patch ×1)"
    thread = conv.thread_record["thread"]
    assert thread["session_id"] == CODEX_ID
    assert thread["rollout_path"] == path
    blob = json.dumps(conv.all_rows())
    for dropped in ("DEV", "BASE", "environment_context", "encrypted_content",
                    "token_count", "world_state"):
        assert dropped not in blob, dropped


def test_codex_injected_context_and_heartbeat_turns(ws, homes):
    """A turn whose user message is only injected context (an automation
    heartbeat) keeps its reply, with no user row."""
    _, codex_dir = homes
    rows = codex_rollout_rows(ws) + [
        {"timestamp": "2026-10-03T10:30:00Z", "type": "response_item", "payload": {
            "type": "message", "role": "user", "content": [
                {"type": "input_text", "text": "<heartbeat>\n<automation_id>a</automation_id>\n</heartbeat>"}]}},
        {"timestamp": "2026-10-03T10:30:05Z", "type": "response_item", "payload": {
            "type": "message", "role": "assistant", "content": [
                {"type": "output_text", "text": "Daily report."}]}},
    ]
    path = write_codex_rollout(codex_dir, ws, rows)
    conv = si.convert_transcript("codex", path)
    roles = [(r["role"], r.get("content", "")[:20]) for r in conv.rows[1:]]
    assert [r for r, _ in roles] == ["user", "system", "system"]
    assert _terminals(conv.rows)[-1]["_meta"]["reply"] == "Daily report."
    assert "heartbeat" not in json.dumps(conv.rows)


def test_conversion_keeps_the_last_400_turns(ws, homes):
    claude_dir, _ = homes
    rows = []
    for i in range(405):
        rows.append(_claude_user(f"ask {i}", f"2026-10-01T10:{i // 60:02d}:{i % 60:02d}Z", ws))
        rows.append(_claude_assistant([{"type": "text", "text": f"answer {i}"}],
                                      f"2026-10-01T10:{i // 60:02d}:{i % 60:02d}Z", ws))
    path = write_claude_session(claude_dir, ws, rows)
    conv = si.convert_transcript("claude-code", path)
    users = [r for r in conv.rows if r["role"] == "user"]
    assert len(users) == 400
    assert users[0]["content"] == "ask 5" and users[-1]["content"] == "ask 404"
    assert conv.truncated is True
    assert conv.rows[0]["_meta"]["truncated"] is True
    assert conv.rows[0]["content"].endswith(" — earlier messages not imported")


def test_conversion_caps_reply_text_at_2_mb(ws, homes, monkeypatch):
    claude_dir, _ = homes
    monkeypatch.setattr(si, "MAX_REPLY_BYTES", 1200)
    rows = []
    for i in range(5):
        rows.append(_claude_user(f"ask {i}", f"2026-10-01T10:00:0{i}Z", ws))
        rows.append(_claude_assistant([{"type": "text", "text": str(i) * 400}],
                                      f"2026-10-01T10:00:0{i}Z", ws))
    path = write_claude_session(claude_dir, ws, rows)
    conv = si.convert_transcript("claude-code", path)
    replies = [r["_meta"]["reply"] for r in _terminals(conv.rows)]
    assert replies == ["2" * 400, "3" * 400, "4" * 400]
    assert conv.truncated is True


def test_untruncated_header_has_no_suffix(ws, homes):
    claude_dir, _ = homes
    path = write_claude_session(claude_dir, ws, basic_claude_rows(ws))
    conv = si.convert_transcript("claude-code", path)
    assert "earlier messages" not in conv.rows[0]["content"]


def test_resume_pre_check_reads_the_same_claude_home_as_the_importer(ws, homes, monkeypatch):
    """A session the importer lists from $CLAUDE_CONFIG_DIR must also pass the
    resume pre-check, or an imported session could never resume."""
    claude_dir, _ = homes
    write_claude_session(claude_dir, ws, basic_claude_rows(ws))
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", claude_dir)
    assert [s.native_id for s in si.list_native_sessions(ws)] == [CLAUDE_ID]
    assert SDKTransport.resume_source_exists(ws, CLAUDE_ID) is True
    assert SDKTransport.resume_source_exists(ws, "00000000-0000-0000-0000-000000000000") is False


# --------------------------------------------------------------------------
# Spec 138 — Orbital's injected prompts are not the user's words
# --------------------------------------------------------------------------

RECAP = ("Conversation so far (earlier messages in this chat):\n\nuser: hi\n\n"
         "--- end of conversation so far ---\n\nNow add the docs")


def test_claude_conversion_reads_user_text_through_the_cleaner(ws, homes):
    claude_dir, _ = homes
    rows = [
        _claude_user("[Triggered by schedule 'nightly' (every day at 9)]\n\nRun the suite",
                     "2026-10-01T09:00:00Z", ws),
        _claude_assistant([{"type": "text", "text": "Suite green."}], "2026-10-01T09:00:01Z", ws),
        _claude_user("[Orbital memory] Your last turn is ending; record it.",
                     "2026-10-01T09:01:00Z", ws),
        _claude_assistant([{"type": "text", "text": "Recorded."}], "2026-10-01T09:01:01Z", ws),
        _claude_user("<system-reminder>\nctx\n</system-reminder>\n" + RECAP,
                     "2026-10-01T09:02:00Z", ws),
        _claude_assistant([{"type": "text", "text": "Docs added."}], "2026-10-01T09:02:01Z", ws),
    ]
    conv = si.convert_transcript("claude-code", write_claude_session(claude_dir, ws, rows))
    users = [r["content"] for r in conv.rows if r["role"] == "user"]
    assert users == ["Run the suite", "Now add the docs"]
    replies = [r["_meta"]["reply"] for r in _terminals(conv.rows)]
    assert replies == ["Suite green.", "Docs added."]  # the memory turn is gone with its reply
    assert "Orbital memory" not in json.dumps(conv.rows)


def test_codex_conversion_reads_user_text_through_the_cleaner(ws, homes):
    _, codex_dir = homes
    rows = codex_rollout_rows(ws) + [
        {"timestamp": "2026-10-03T10:00:00Z", "type": "response_item", "payload": {
            "type": "message", "role": "user", "content": [
                {"type": "input_text", "text": RECAP}]}},
        {"timestamp": "2026-10-03T10:00:05Z", "type": "response_item", "payload": {
            "type": "message", "role": "assistant", "content": [
                {"type": "output_text", "text": "Docs added."}]}},
    ]
    conv = si.convert_transcript("codex", write_codex_rollout(codex_dir, ws, rows))
    users = [r["content"] for r in conv.rows if r["role"] == "user"]
    assert users == ["Rename the helper", "Now add the docs"]


# --------------------------------------------------------------------------
# Spec 138 — pi / Kimi Code / MiniMax Code sessions, and group members
# --------------------------------------------------------------------------

from tests.unit import _history_138_builders as hb  # noqa: E402


@pytest.fixture
def hhome(tmp_path, monkeypatch):
    h = tmp_path / "history-home"
    h.mkdir()
    monkeypatch.setenv("ORBITAL_HISTORY_HOME", str(h))
    return str(h)


def _three(hhome, cwd):
    return {
        "pi": hb.install_pi(hhome, cwd),
        "kimi-code": hb.install_kimi(hhome, cwd),
        "minimax-code": hb.install_minimax(hhome, cwd),
    }


def test_lists_pi_kimi_and_minimax_sessions_of_the_folder(ws, hhome, tmp_path):
    paths = _three(hhome, ws)
    other = str(tmp_path / "elsewhere")
    os.makedirs(other)
    hb.install_pi(hhome, other, sid="0199bbbb-0000-7000-8000-000000000000")
    found = {s.agent: s for s in si.list_native_sessions(ws)}
    assert set(found) == {"pi", "kimi-code", "minimax-code"}
    p, k, m = found["pi"], found["kimi-code"], found["minimax-code"]
    assert (p.native_id, p.path, p.rollout_path) == (hb.PI_ID, paths["pi"], paths["pi"])
    assert p.title == "Rename helper" and p.last_activity == "2026-10-20T10:00:04+00:00"
    assert p.session_id == "imp_pi_0199aaaa11117222"
    assert (k.native_id, k.path, k.title) == (hb.KIMI_ID, paths["kimi-code"],
                                              "Add a test for app.py")
    assert k.rollout_path is None
    assert (m.native_id, m.title) == (hb.MINIMAX_ID, "Summarise README")
    assert m.size_bytes == os.path.getsize(paths["minimax-code"])
    for s in found.values():
        assert si.is_imported_session_id(s.session_id)
        assert si.find_native_session(ws, s.session_id).native_id == s.native_id


def test_members_list_sessions_that_ran_in_another_folder(ws, homes, hhome, tmp_path,
                                                          monkeypatch):
    claude_dir, codex_dir = homes
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", claude_dir)
    monkeypatch.setenv("ORBITAL_CODEX_HOME", codex_dir)
    desk = str(tmp_path / "Desktop")
    os.makedirs(desk)
    _three(hhome, desk)
    write_claude_session(claude_dir, desk, basic_claude_rows(desk))
    rollout = write_codex_rollout(codex_dir, desk, codex_rollout_rows(desk))
    make_codex_home(codex_dir, [{"id": CODEX_ID, "cwd": desk, "rollout_path": rollout,
                                 "first_user_message": "Rename the helper"}])
    assert si.list_native_sessions(ws) == []
    members = [{"agent": "pi", "native_id": hb.PI_ID},
               {"agent": "kimi-code", "native_id": hb.KIMI_ID},
               {"agent": "minimax-code", "native_id": hb.MINIMAX_ID},
               {"agent": "claude-code", "native_id": CLAUDE_ID},
               {"agent": "codex", "native_id": CODEX_ID},
               {"agent": "codex", "native_id": "019f0000-dead-7000-8000-000000000000"},
               {"agent": "nope", "native_id": "x"}, "garbage"]
    found = si.list_native_sessions(ws, members=members)
    assert sorted((s.agent, s.native_id) for s in found) == sorted([
        ("pi", hb.PI_ID), ("kimi-code", hb.KIMI_ID), ("minimax-code", hb.MINIMAX_ID),
        ("claude-code", CLAUDE_ID), ("codex", CODEX_ID)])
    codex = next(s for s in found if s.agent == "codex")
    assert codex.rollout_path == rollout
    sid = si.imported_session_id("kimi-code", hb.KIMI_ID)
    assert si.find_native_session(ws, sid) is None
    assert si.find_native_session(ws, sid, members=members).native_id == hb.KIMI_ID


def test_a_member_that_also_ran_here_is_listed_once(ws, hhome):
    hb.install_pi(hhome, ws)
    found = si.list_native_sessions(ws, members=[{"agent": "pi", "native_id": hb.PI_ID}])
    assert [(s.agent, s.native_id) for s in found] == [("pi", hb.PI_ID)]


def test_acp_sessions_orbital_itself_runs_are_not_listed(ws, hhome):
    """Orbital's kimi / mcode workers leave ordinary native sessions in the
    project folder; the thread records in the project's sessions say which."""
    _three(hhome, ws)
    sessions = os.path.join(ws, "orbital", "sessions")
    os.makedirs(sessions)
    with open(os.path.join(sessions, "s1.jsonl"), "w", encoding="utf-8") as fh:
        fh.write(json.dumps({"role": "meta", "event": "session_start"}) + "\n")
        for handle, nid in (("kimi-code", hb.KIMI_ID), ("minimax-code", hb.MINIMAX_ID)):
            fh.write(json.dumps({"role": "meta", "event": "sub_agent_thread", "handle": handle,
                                 "thread": {"session_id": nid}}) + "\n")
    assert [s.agent for s in si.list_native_sessions(ws)] == ["pi"]


def _write_orbital_thread_rows(ws, pairs):
    """A project session file holding one ``sub_agent_thread`` row per
    ``(handle, native_id)`` — what Orbital writes when a worker turn ends."""
    sessions = os.path.join(ws, "orbital", "sessions")
    os.makedirs(sessions, exist_ok=True)
    with open(os.path.join(sessions, "s1.jsonl"), "w", encoding="utf-8") as fh:
        fh.write(json.dumps({"role": "meta", "event": "session_start"}) + "\n")
        for handle, nid in pairs:
            fh.write(json.dumps({"role": "meta", "event": "sub_agent_thread", "handle": handle,
                                 "thread": {"session_id": nid, "model": None}}) + "\n")


@pytest.mark.parametrize("originator", ["codex_exec", None])
def test_codex_threads_orbital_itself_runs_are_not_listed(ws, homes, originator):
    """2026-11-24 Windows finding: Orbital's own Codex worker threads came
    back as "Imported from Codex" when the thread did not carry the "orbital"
    originator. The project's thread records name them."""
    claude_dir, codex_dir = homes
    own = "019f0000-0000-7000-8000-00000000000a"
    mine = write_codex_rollout(codex_dir, ws, codex_rollout_rows(ws, tid=own), tid=own)
    user = write_codex_rollout(codex_dir, ws, codex_rollout_rows(ws))
    make_codex_home(codex_dir, [
        {"id": own, "cwd": ws, "rollout_path": mine, "originator": originator,
         "first_user_message": "Other agents in this project: codex, dsh."},
        {"id": CODEX_ID, "cwd": ws, "rollout_path": user,
         "first_user_message": "Rename the helper"},
    ])
    _write_orbital_thread_rows(ws, [("codex", own)])
    found = si.list_native_sessions(ws, claude_dir=claude_dir, codex_dir=codex_dir)
    assert [(s.agent, s.native_id) for s in found] == [("codex", CODEX_ID)]


def test_codex_rollout_fallback_also_skips_threads_orbital_runs(ws, homes):
    claude_dir, codex_dir = homes
    own = "019f0000-0000-7000-8000-00000000000b"
    write_codex_rollout(codex_dir, ws, codex_rollout_rows(ws, tid=own), tid=own)
    write_codex_rollout(codex_dir, ws, codex_rollout_rows(ws))
    _write_orbital_thread_rows(ws, [("codex", own)])
    found = si.list_native_sessions(ws, claude_dir=claude_dir, codex_dir=codex_dir)
    assert [(s.agent, s.native_id) for s in found] == [("codex", CODEX_ID)]


def test_claude_sessions_orbital_itself_runs_are_not_listed(ws, homes):
    """A Claude transcript with a user entrypoint ("cli") that a project
    session names as a worker thread is Orbital's, not the user's. The SDK
    resume id Orbital records is the transcript's file stem."""
    claude_dir, codex_dir = homes
    own = "99999999-8888-7777-6666-555555555555"
    write_claude_session(claude_dir, ws, basic_claude_rows(ws))
    own_rows = [dict(r, sessionId=own) if "sessionId" in r else r for r in basic_claude_rows(ws)]
    write_claude_session(claude_dir, ws, own_rows, sid=own)
    _write_orbital_thread_rows(ws, [("claude-code", own)])
    found = si.list_native_sessions(ws, claude_dir=claude_dir, codex_dir=codex_dir)
    assert [(s.agent, s.native_id) for s in found] == [("claude-code", CLAUDE_ID)]


def test_thread_ids_read_each_file_once_and_take_one_cache_entry(ws, monkeypatch):
    """The Workbench lists every project's sessions on each load: an unchanged
    session file is never re-read, a changed one is, and a folder costs one
    entry in the shared cache however many session files it holds."""
    _write_orbital_thread_rows(ws, [("codex", "t-1")])
    sessions = os.path.join(ws, "orbital", "sessions")
    for n in range(5):
        with open(os.path.join(sessions, f"x{n}.jsonl"), "w", encoding="utf-8") as fh:
            fh.write(json.dumps({"role": "user", "content": "hi"}) + "\n")
    reads = []
    real = si._thread_ids_in
    monkeypatch.setattr(si, "_thread_ids_in", lambda p: reads.append(p) or real(p))
    assert si.orbital_thread_ids(ws) == {"t-1"}
    assert len(reads) == 6
    assert si.orbital_thread_ids(ws) == {"t-1"}
    assert len(reads) == 6
    with open(os.path.join(sessions, "x0.jsonl"), "a", encoding="utf-8") as fh:
        fh.write(json.dumps({"role": "meta", "event": "sub_agent_thread", "handle": "codex",
                             "thread": {"session_id": "t-2"}}) + "\n")
    assert si.orbital_thread_ids(ws) == {"t-1", "t-2"}
    assert [os.path.basename(p) for p in reads[6:]] == ["x0.jsonl"]
    os.remove(os.path.join(sessions, "s1.jsonl"))
    assert si.orbital_thread_ids(ws) == {"t-2"}
    assert len([k for k in si._cache if k[0] == "orbital-threads"]) == 1


# ---- thread ids ride the persistent session index (2026-11-24) -------------
#
# The Workbench feed lists every project after each daemon start, when the
# in-memory cache above is empty. Reading every session file in full then cost
# seconds; the session index already sits next to the files, keyed by each
# file's (mtime_ns, size), so it carries the ids too.

# The sessions table exactly as v0.24.1 (origin/main) creates it.
_OLD_SESSIONS_TABLE = (
    "CREATE TABLE sessions ( fname TEXT PRIMARY KEY, mtime_ns INTEGER NOT NULL,"
    " size INTEGER NOT NULL, session_uuid TEXT, origin TEXT, trigger_type TEXT,"
    " name TEXT, last_activity_at TEXT, entry TEXT)")
_OLD_INSERT = (
    "INSERT OR REPLACE INTO sessions (fname, mtime_ns, size, session_uuid,"
    " origin, trigger_type, name, last_activity_at, entry)"
    " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)")


def _thread_folder(ws):
    """s1 holds one worker thread; x0..x2 are plain chats."""
    _write_orbital_thread_rows(ws, [("codex", "t-1")])
    sessions = os.path.join(ws, "orbital", "sessions")
    for n in range(3):
        with open(os.path.join(sessions, f"x{n}.jsonl"), "w", encoding="utf-8") as fh:
            fh.write(json.dumps({"role": "user", "content": "hi"}) + "\n")
    return sessions


def _index_folder(sessions):
    """Index every file the way the daemon's index build does."""
    from agent_os.daemon_v2.session_index import SessionIndex, derive_disk_entry
    index = SessionIndex(sessions)
    index.load_rows()
    index.put_many([(f, os.stat(os.path.join(sessions, f)),
                     derive_disk_entry(os.path.join(sessions, f), f[:-6]))
                    for f in sorted(os.listdir(sessions)) if f.endswith(".jsonl")])
    return index


def _indexed_ids(sessions):
    from agent_os.daemon_v2.session_index import index_path
    conn = sqlite3.connect(index_path(sessions))
    try:
        return dict(conn.execute("SELECT fname, thread_ids FROM sessions"))
    finally:
        conn.close()


def _count_reads(monkeypatch):
    reads = []
    real = si._thread_ids_in
    monkeypatch.setattr(si, "_thread_ids_in", lambda p: reads.append(os.path.basename(p)) or real(p))
    return reads


def test_current_index_rows_give_the_thread_ids_without_opening_a_file(ws, monkeypatch):
    import builtins
    sessions = _thread_folder(ws)
    _index_folder(sessions)
    assert si.orbital_thread_ids(ws) == {"t-1"}  # reads the files, fills the index
    assert _indexed_ids(sessions) == {"s1.jsonl": '["t-1"]', "x0.jsonl": "[]",
                                      "x1.jsonl": "[]", "x2.jsonl": "[]"}
    si.clear_cache()  # a daemon restart
    reads = _count_reads(monkeypatch)
    opened = []
    real_open = builtins.open
    monkeypatch.setattr(builtins, "open", lambda f, *a, **k: (
        opened.append(str(f)) if str(f).endswith(".jsonl") else None) or real_open(f, *a, **k))
    assert si.orbital_thread_ids(ws) == {"t-1"}
    assert reads == [] and opened == []


def test_a_file_changed_after_its_index_row_is_read_again(ws, monkeypatch):
    sessions = _thread_folder(ws)
    index = _index_folder(sessions)
    si.orbital_thread_ids(ws)
    s1 = os.path.join(sessions, "s1.jsonl")
    with open(s1, "a", encoding="utf-8") as fh:
        fh.write(json.dumps({"role": "meta", "event": "sub_agent_thread", "handle": "codex",
                             "thread": {"session_id": "t-2"}}) + "\n")
    si.clear_cache()
    reads = _count_reads(monkeypatch)
    # The row's stamp no longer matches: the file is read, and the stale row
    # is not given ids it was not derived with.
    assert si.orbital_thread_ids(ws) == {"t-1", "t-2"}
    assert reads == ["s1.jsonl"]
    assert _indexed_ids(sessions)["s1.jsonl"] == '["t-1"]'
    # The index re-derives the row (its normal write): ids are unknown again
    # until the next read fills them.
    index.put("s1.jsonl", os.stat(s1), None)
    assert _indexed_ids(sessions)["s1.jsonl"] is None
    si.clear_cache()
    assert si.orbital_thread_ids(ws) == {"t-1", "t-2"}
    assert reads == ["s1.jsonl", "s1.jsonl"]
    assert _indexed_ids(sessions)["s1.jsonl"] == '["t-1", "t-2"]'
    si.clear_cache()
    assert si.orbital_thread_ids(ws) == {"t-1", "t-2"}
    assert len(reads) == 2


def test_an_index_from_the_previous_build_still_works(ws, monkeypatch):
    """A v0.24.1 index has no thread_ids column: every file is read, nothing
    raises, and the rows are kept (no rebuild) and filled in."""
    sessions = _thread_folder(ws)
    from agent_os.daemon_v2.session_index import INDEX_VERSION, index_path
    conn = sqlite3.connect(index_path(sessions))
    conn.execute("CREATE TABLE index_meta ( key TEXT PRIMARY KEY, value TEXT NOT NULL)")
    conn.execute("INSERT INTO index_meta VALUES ('version', ?)", (INDEX_VERSION,))
    conn.execute(_OLD_SESSIONS_TABLE)
    for f in os.listdir(sessions):
        if f.endswith(".jsonl"):
            st = os.stat(os.path.join(sessions, f))
            conn.execute(_OLD_INSERT, (f, st.st_mtime_ns, st.st_size, f[:-6], "chat",
                                       None, None, None, "{}"))
    conn.commit()
    conn.close()
    reads = _count_reads(monkeypatch)
    assert si.orbital_thread_ids(ws) == {"t-1"}
    assert sorted(reads) == ["s1.jsonl", "x0.jsonl", "x1.jsonl", "x2.jsonl"]
    assert _indexed_ids(sessions)["s1.jsonl"] == '["t-1"]'
    si.clear_cache()
    assert si.orbital_thread_ids(ws) == {"t-1"}
    assert len(reads) == 4


def test_the_previous_build_reads_and_writes_the_upgraded_index(ws, monkeypatch):
    """Downgrade: v0.24.1's own SELECT and INSERT still run on an index this
    build filled, and a row it rewrites reads as unknown here."""
    sessions = _thread_folder(ws)
    _index_folder(sessions)
    si.orbital_thread_ids(ws)
    from agent_os.daemon_v2.session_index import index_path
    conn = sqlite3.connect(index_path(sessions))
    rows = conn.execute("SELECT fname, mtime_ns, size, entry FROM sessions").fetchall()
    assert len(rows) == 4
    s1 = os.path.join(sessions, "s1.jsonl")
    with open(s1, "a", encoding="utf-8") as fh:
        fh.write(json.dumps({"role": "meta", "event": "sub_agent_thread", "handle": "codex",
                             "thread": {"session_id": "t-3"}}) + "\n")
    st = os.stat(s1)
    conn.execute(_OLD_INSERT, ("s1.jsonl", st.st_mtime_ns, st.st_size, "s1", "chat",
                               None, None, None, "{}"))
    conn.commit()
    conn.close()
    si.clear_cache()
    reads = _count_reads(monkeypatch)
    assert si.orbital_thread_ids(ws) == {"t-1", "t-3"}
    assert reads == ["s1.jsonl"]


def test_index_is_found_under_a_path_with_spaces_cjk_and_uri_marks(tmp_path, monkeypatch):
    """The index is opened through a ``file:`` URI (so a read never creates
    it): the folder name must survive the quoting. ``?`` cannot be in a
    Windows file name (WinError 123), so there the hardest legal name is
    used; the URI's Windows form is pinned by the pure tests below."""
    name = "我的 项目 #1 &a=b %20" if os.name == "nt" else "我的 项目 #1 ?a=b &c %20"
    ws = str(tmp_path / name)
    os.makedirs(ws)
    sessions = _thread_folder(ws)
    _index_folder(sessions)
    si.orbital_thread_ids(ws)
    si.clear_cache()
    reads = _count_reads(monkeypatch)
    assert si.orbital_thread_ids(ws) == {"t-1"}
    assert reads == []


def test_index_uri_for_a_windows_drive_path():
    """On Windows the URI is SQLite's documented ``file:///C:/…`` form: drive
    kept, backslashes turned, and ``%``, ``#``, ``&``, ``=``, spaces and CJK
    percent-encoded, so the appended ``?mode=`` is the only query."""
    import pathlib
    from agent_os.daemon_v2.session_index import _index_uri
    path = r"C:\Users\名字\我的 项目 #1 %20 &a=b\orbital\sessions\.session-index.sqlite"
    assert _index_uri(path, "ro", pathlib.PureWindowsPath) == (
        "file:///C:/Users/%E5%90%8D%E5%AD%97/"
        "%E6%88%91%E7%9A%84%20%E9%A1%B9%E7%9B%AE%20%231%20%2520%20%26a%3Db/"
        "orbital/sessions/.session-index.sqlite?mode=ro")


def test_index_uri_for_a_posix_path_quotes_the_query_mark():
    import pathlib
    from agent_os.daemon_v2.session_index import _index_uri
    assert _index_uri("/home/名/a ?b=c #d%20/.session-index.sqlite", "rw",
                      pathlib.PurePosixPath) == (
        "file:///home/%E5%90%8D/a%20%3Fb%3Dc%20%23d%2520/.session-index.sqlite?mode=rw")


@pytest.mark.parametrize("windows_path", [
    r"\\server\share\proj\orbital\sessions\.session-index.sqlite",  # UNC
    r"\\?\C:\proj\orbital\sessions\.session-index.sqlite",          # extended-length
])
def test_a_uri_sqlite_refuses_falls_back_to_reading_the_files(ws, monkeypatch, windows_path):
    """A UNC workspace's URI has an authority (``file://server/…``), which
    SQLite refuses on every OS: the read is empty, never an exception, the
    files are read instead."""
    import pathlib
    from agent_os.daemon_v2 import session_index
    uri = session_index._index_uri(windows_path, "ro", pathlib.PureWindowsPath)
    with pytest.raises(sqlite3.OperationalError):
        sqlite3.connect(uri, uri=True).close()
    sessions = _thread_folder(ws)
    _index_folder(sessions)
    monkeypatch.setattr(session_index, "_index_uri",
                        lambda _path, mode, *_a: uri.replace("?mode=ro", f"?mode={mode}"))
    assert session_index.read_thread_ids(sessions) == {}
    session_index.store_thread_ids(sessions, [("s1.jsonl", 1, 1, frozenset({"t-1"}))])
    reads = _count_reads(monkeypatch)
    assert si.orbital_thread_ids(ws) == {"t-1"}
    assert len(reads) == 4


def test_no_index_reads_the_files_and_creates_nothing(ws, monkeypatch):
    sessions = _thread_folder(ws)
    reads = _count_reads(monkeypatch)
    assert si.orbital_thread_ids(ws) == {"t-1"}
    assert len(reads) == 4
    assert sorted(os.listdir(sessions)) == ["s1.jsonl", "x0.jsonl", "x1.jsonl", "x2.jsonl"]


def test_pi_kimi_minimax_conversion_and_resume_records(ws, hhome):
    paths = _three(hhome, ws)
    pi_conv = si.convert_transcript("pi", paths["pi"], now="2026-10-21T00:00:00+00:00")
    header, *rows = pi_conv.rows
    assert header["content"] == "[Imported] Conversation imported from Pi (last active 2026-10-20)"
    assert header["_meta"]["native_id"] == hb.PI_ID
    assert [r["role"] for r in rows] == ["user", "system"]
    assert rows[0]["content"] == f"Rename the helper in {ws}/src/util.py"
    assert rows[0]["target"] == "pi"
    assert rows[1]["_meta"]["reply"] == "Renamed it to tool.\n\n(tools: read ×2, edit ×1)"
    assert pi_conv.thread_record["thread"] == {
        "session_id": hb.PI_ID, "model": None, "last_used_at": "2026-10-21T00:00:00+00:00",
        "rollout_path": paths["pi"]}
    kimi = si.convert_transcript("kimi-code", paths["kimi-code"])
    assert kimi.rows[0]["content"].startswith("[Imported] Conversation imported from Kimi Code ")
    assert [r["content"] for r in kimi.rows if r["role"] == "user"] == [
        f"Add a test for {ws}/app.py"]
    assert [r["_meta"]["reply"] for r in _terminals(kimi.rows)] == [
        "Writing it.\n\nAdded the test.\n\n(tools: Write ×1, Read ×1)"]
    assert set(kimi.thread_record["thread"]) == {"session_id", "model", "last_used_at"}
    assert kimi.thread_record["thread"]["session_id"] == hb.KIMI_ID
    mm = si.convert_transcript("minimax-code", paths["minimax-code"])
    assert [r["content"] for r in mm.rows if r["role"] == "user"] == [
        f"Summarise {ws}/README.md"]
    assert "agent-context" not in json.dumps(mm.all_rows())
    assert mm.thread_record["thread"]["session_id"] == hb.MINIMAX_ID


def test_session_list_reads_the_same_home_as_the_history_scan(monkeypatch, tmp_path):
    """Spec 138: the session list and the history scan must agree on where
    Claude Code and Codex history lives, or a project's cards name sessions
    its own list does not show."""
    from agent_os.onboarding import history_paths
    from agent_os.onboarding import session_import as si
    home = str(tmp_path / "staged")
    monkeypatch.setenv("ORBITAL_HISTORY_HOME", home)
    monkeypatch.delenv("ORBITAL_CODEX_HOME", raising=False)
    monkeypatch.delenv("CLAUDE_CONFIG_DIR", raising=False)
    monkeypatch.delenv("CODEX_HOME", raising=False)
    assert si.claude_home() == history_paths.claude_dir() == os.path.join(home, ".claude")
    assert si.codex_home() == history_paths.codex_dir() == os.path.join(home, ".codex")
    # The agents' own variables still win, for both readers alike.
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path / "elsewhere"))
    assert si.claude_home() == history_paths.claude_dir() == str(tmp_path / "elsewhere")
