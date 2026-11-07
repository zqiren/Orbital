# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Regression: /inject accepts a desktop-app attachment by local path.

Nothing is copied into ``uploads/``; the prefix carries the absolute path
marked as the user's local file, the JSONL row's ``attachments`` entry says
``local: true``, and the project's agents are granted a read of the file.
"""

import os

from fastapi.testclient import TestClient

from agent_os.daemon_v2 import attachment_grants
from tests.regression.test_inject_attachments_e2e import (
    _make_app,
    _read_session_jsonl,
)

MARK = "the user's local file, not copied"


def _outside_file(tmp_path, name="notes.txt", payload=b"FIRST LINE\nsecond\n"):
    d = tmp_path / "home"
    d.mkdir(exist_ok=True)
    p = d / name
    p.write_bytes(payload)
    return str(p), len(payload)


def test_manager_branch_gets_marked_absolute_path_and_no_copy(tmp_path):
    attachment_grants.reset()
    ws = tmp_path / "ws"
    ws.mkdir()
    path, size = _outside_file(tmp_path)
    app, mocks = _make_app(str(ws))
    resp = TestClient(app).post("/api/v2/agents/proj_test/inject", json={
        "content": "first line?",
        "attachments": [{"path": path, "mime": "text/plain", "size": size, "local": True}],
    })
    assert resp.status_code == 200, resp.text
    content = mocks["agent_manager"].inject_message.await_args.args[1]
    real = os.path.realpath(path)
    assert content == (f"<attached_files>\n- {real} (text/plain, {size} B, {MARK})\n"
                       "</attached_files>\n\nfirst line?")
    assert not (ws / "uploads").exists()
    assert attachment_grants.is_granted("proj_test", path)
    attachment_grants.reset()


def test_large_local_file_has_no_cap(tmp_path):
    attachment_grants.reset()
    ws = tmp_path / "ws"
    ws.mkdir()
    big = tmp_path / "big.bin"
    with open(big, "wb") as fh:
        fh.truncate(50 * 1024 * 1024)
    app, mocks = _make_app(str(ws))
    resp = TestClient(app).post("/api/v2/agents/proj_test/inject", json={
        "content": "size?",
        "attachments": [{"path": str(big), "mime": "application/octet-stream",
                         "size": 50 * 1024 * 1024, "local": True}],
    })
    assert resp.status_code == 200, resp.text
    content = mocks["agent_manager"].inject_message.await_args.args[1]
    assert f"(application/octet-stream, 50.0 MB, {MARK})" in content
    attachment_grants.reset()


def test_worker_branch_row_marks_local(tmp_path):
    attachment_grants.reset()
    ws = tmp_path / "ws"
    ws.mkdir()
    path, size = _outside_file(tmp_path)
    app, mocks = _make_app(str(ws), with_sub_agent=True)
    resp = TestClient(app).post("/api/v2/agents/proj_test/inject", json={
        "content": "read it", "target": "claude-code", "pinned": True,
        "attachments": [{"path": path, "mime": "text/plain", "size": size, "local": True}],
    })
    assert resp.status_code == 200, resp.text
    user = [m for m in _read_session_jsonl(str(ws)) if m.get("role") == "user"][0]
    assert user["attachments"] == [{"path": os.path.realpath(path), "mime": "text/plain",
                                    "size": size, "local": True}]
    sent = mocks["sub_agent_manager"].send.await_args.args[2]
    assert MARK in sent and os.path.realpath(path) in sent
    attachment_grants.reset()


def test_missing_local_file_is_400_and_nothing_written(tmp_path):
    attachment_grants.reset()
    ws = tmp_path / "ws"
    ws.mkdir()
    app, mocks = _make_app(str(ws), with_sub_agent=True)
    resp = TestClient(app).post("/api/v2/agents/proj_test/inject", json={
        "content": "x", "target": "claude-code", "pinned": True,
        "attachments": [{"path": str(tmp_path / "gone.txt"), "mime": "text/plain",
                         "size": 1, "local": True}],
    })
    assert resp.status_code == 400
    assert "file not found" in resp.text
    mocks["agent_manager"].persist_mention_message.assert_not_called()
    assert attachment_grants.granted("proj_test") == frozenset()


def test_directory_is_400(tmp_path):
    ws = tmp_path / "ws"
    ws.mkdir()
    d = tmp_path / "folder"
    d.mkdir()
    app, _ = _make_app(str(ws))
    resp = TestClient(app).post("/api/v2/agents/proj_test/inject", json={
        "content": "x",
        "attachments": [{"path": str(d), "mime": "text/plain", "size": 1, "local": True}],
    })
    assert resp.status_code == 400
    assert "not a regular file" in resp.text


def test_relative_local_path_is_422(tmp_path):
    ws = tmp_path / "ws"
    ws.mkdir()
    app, _ = _make_app(str(ws))
    resp = TestClient(app).post("/api/v2/agents/proj_test/inject", json={
        "content": "x",
        "attachments": [{"path": "notes.txt", "mime": "text/plain", "size": 1, "local": True}],
    })
    assert resp.status_code == 422
