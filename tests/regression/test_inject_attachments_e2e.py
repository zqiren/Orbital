# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Regression: /inject endpoint accepts an `attachments` array.

Behavior under test (Approach 1 — path-references only, no native multimodal):

1. When `attachments` is present and validates, the route prepends a
   ``<attached_files>...</attached_files>`` block to `content` before either
   branch (management agent, sub-agent) sees it.

2. On the sub-agent branch, the user message persisted to session JSONL
   carries the prefixed `content` AND a top-level `attachments` field with
   the validated metadata (audit trail).

3. On the management-agent branch, the prefixed `content` is forwarded to
   `_agent_manager.inject_message`. The structured `attachments` audit
   field is NOT persisted on this branch in v1 — see PR description for
   the deliberate v1 asymmetry.

4. Validation failures reject the request:
   - Pydantic-layer rejection (path with '..', size > 10MB, …) → 422
   - Filesystem rejection (file missing, size mismatch, symlink escape) → 400
"""

import glob
import json
import os
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from agent_os.api.routes import agents_v2


@pytest.fixture
def workspace(tmp_path):
    """Tmp workspace dir. Sub-directories created on demand by tests."""
    return str(tmp_path)


@pytest.fixture(autouse=True)
def reset_sub_agent_sessions():
    """The route caches sub-agent sessions on a module-level dict; reset it
    between tests so each one starts with a fresh session."""
    agents_v2._sub_agent_sessions.clear()
    yield
    agents_v2._sub_agent_sessions.clear()


def _make_app(workspace, *, with_sub_agent=False):
    """Wire the agents_v2 router with mocked deps and return (app, mocks)."""
    project_store = MagicMock()
    project_store.get_project.return_value = {
        "project_id": "proj_test",
        "name": "test",
        "workspace": workspace,
    }

    # Management agent: get_session returns None so the route falls back to
    # the sub-agent session cache (which creates a real Session writing
    # JSONL into the workspace).
    agent_manager = MagicMock()
    agent_manager.get_session = MagicMock(return_value=None)
    agent_manager.inject_message = AsyncMock(return_value="delivered")

    ws_manager = MagicMock()
    ws_manager.broadcast = MagicMock()

    if with_sub_agent:
        sub_agent_manager = MagicMock()
        sub_agent_manager.list_active = MagicMock(return_value=[])
        sub_agent_manager.send = AsyncMock(return_value="ok from sub-agent")
        sub_agent_manager.start = AsyncMock()
        sub_agent_manager.get_transcript = MagicMock(return_value=None)
    else:
        sub_agent_manager = None

    agents_v2.configure(
        project_store=project_store,
        agent_manager=agent_manager,
        ws_manager=ws_manager,
        sub_agent_manager=sub_agent_manager,
    )

    app = FastAPI()
    app.include_router(agents_v2.router)
    return app, {
        "project_store": project_store,
        "agent_manager": agent_manager,
        "ws_manager": ws_manager,
        "sub_agent_manager": sub_agent_manager,
    }


def _write_upload(workspace, rel_path, payload):
    """Write a file under workspace and return its size."""
    full = os.path.join(workspace, rel_path)
    os.makedirs(os.path.dirname(full), exist_ok=True)
    with open(full, "wb") as f:
        f.write(payload)
    return len(payload)


def _read_session_jsonl(workspace):
    """Return the list of messages in the (single) session JSONL file."""
    files = glob.glob(os.path.join(workspace, "orbital", "sessions", "*.jsonl"))
    assert len(files) == 1, f"expected one session file, got {files}"
    messages = []
    with open(files[0], "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                messages.append(json.loads(line))
    return messages


# ---------------------------------------------------------------------------
# Management-agent branch
# ---------------------------------------------------------------------------


class TestManagementAgentBranch:
    def test_single_attachment_prefixes_content_passed_to_inject_message(
        self, workspace
    ):
        size = _write_upload(workspace, "uploads/hello.txt", b"hello world")
        app, mocks = _make_app(workspace, with_sub_agent=False)
        client = TestClient(app)

        resp = client.post(
            "/api/v2/agents/proj_test/inject",
            json={
                "content": "what is this?",
                "attachments": [
                    {"path": "uploads/hello.txt", "mime": "text/plain", "size": size},
                ],
            },
        )
        assert resp.status_code == 200, resp.text

        # _agent_manager.inject_message was called with prefixed content.
        mocks["agent_manager"].inject_message.assert_awaited_once()
        call_args = mocks["agent_manager"].inject_message.await_args
        assert call_args.args[0] == "proj_test"
        forwarded_content = call_args.args[1]
        assert forwarded_content.startswith("<attached_files>\n"), forwarded_content
        assert "- uploads/hello.txt (text/plain, " in forwarded_content
        assert forwarded_content.endswith("</attached_files>\n\nwhat is this?")

    def test_three_attachments_use_bulleted_format(self, workspace):
        s1 = _write_upload(workspace, "uploads/a.txt", b"aaa")
        s2 = _write_upload(workspace, "uploads/b.txt", b"bbbb")
        s3 = _write_upload(workspace, "uploads/c.txt", b"ccccc")
        app, mocks = _make_app(workspace, with_sub_agent=False)
        client = TestClient(app)

        resp = client.post(
            "/api/v2/agents/proj_test/inject",
            json={
                "content": "summarise",
                "attachments": [
                    {"path": "uploads/a.txt", "mime": "text/plain", "size": s1},
                    {"path": "uploads/b.txt", "mime": "text/plain", "size": s2},
                    {"path": "uploads/c.txt", "mime": "text/plain", "size": s3},
                ],
            },
        )
        assert resp.status_code == 200, resp.text

        forwarded = mocks["agent_manager"].inject_message.await_args.args[1]
        assert forwarded.startswith("<attached_files>\n")
        assert "- uploads/a.txt" in forwarded
        assert "- uploads/b.txt" in forwarded
        assert "- uploads/c.txt" in forwarded
        assert forwarded.endswith("</attached_files>\n\nsummarise")

    def test_empty_content_with_attachment_passes_just_prefix(self, workspace):
        size = _write_upload(workspace, "uploads/x.txt", b"x")
        app, mocks = _make_app(workspace, with_sub_agent=False)
        client = TestClient(app)

        resp = client.post(
            "/api/v2/agents/proj_test/inject",
            json={
                "content": "",
                "attachments": [
                    {"path": "uploads/x.txt", "mime": "text/plain", "size": size},
                ],
            },
        )
        assert resp.status_code == 200, resp.text

        forwarded = mocks["agent_manager"].inject_message.await_args.args[1]
        # Prefix only, no synthesized user text appended.
        assert forwarded.startswith("<attached_files>\n")
        assert "- uploads/x.txt" in forwarded
        assert forwarded.endswith("</attached_files>\n\n")

    def test_attachments_none_behaves_identical_to_today(self, workspace):
        app, mocks = _make_app(workspace, with_sub_agent=False)
        client = TestClient(app)

        resp = client.post(
            "/api/v2/agents/proj_test/inject",
            json={"content": "plain message"},
        )
        assert resp.status_code == 200, resp.text

        forwarded = mocks["agent_manager"].inject_message.await_args.args[1]
        # No prefix when attachments is absent.
        assert forwarded == "plain message"

    def test_attachments_empty_list_behaves_identical_to_today(self, workspace):
        app, mocks = _make_app(workspace, with_sub_agent=False)
        client = TestClient(app)

        resp = client.post(
            "/api/v2/agents/proj_test/inject",
            json={"content": "plain", "attachments": []},
        )
        assert resp.status_code == 200, resp.text
        forwarded = mocks["agent_manager"].inject_message.await_args.args[1]
        assert forwarded == "plain"


# ---------------------------------------------------------------------------
# Sub-agent branch
# ---------------------------------------------------------------------------


class TestSubAgentBranch:
    def test_attachment_persisted_in_session_jsonl(self, workspace):
        size = _write_upload(workspace, "uploads/hello.txt", b"hello world")
        app, mocks = _make_app(workspace, with_sub_agent=True)
        client = TestClient(app)

        resp = client.post(
            "/api/v2/agents/proj_test/inject",
            json={
                "content": "check this",
                "target": "@worker",
                "attachments": [
                    {"path": "uploads/hello.txt", "mime": "text/plain", "size": size},
                ],
            },
        )
        assert resp.status_code == 200, resp.text

        messages = _read_session_jsonl(workspace)
        # Last user message has the prefix + the structured attachments
        # audit field.
        user_msgs = [m for m in messages if m.get("role") == "user"]
        assert len(user_msgs) == 1
        msg = user_msgs[0]
        assert msg["content"].startswith("<attached_files>\n")
        assert "- uploads/hello.txt (text/plain, " in msg["content"]
        assert msg["content"].endswith("</attached_files>\n\ncheck this")
        assert msg["target"] == "@worker"
        assert "attachments" in msg
        assert msg["attachments"] == [
            {"path": "uploads/hello.txt", "mime": "text/plain", "size": size},
        ]

        # The sub-agent process received the prefixed content too — the
        # agent needs the prefix to know to read the file.
        send_call = mocks["sub_agent_manager"].send.await_args
        assert send_call.args[0] == "proj_test"
        assert send_call.args[1] == "@worker"
        assert send_call.args[2].endswith("</attached_files>\n\ncheck this")
        assert "uploads/hello.txt" in send_call.args[2]

    def test_no_attachments_omits_attachments_field(self, workspace):
        app, mocks = _make_app(workspace, with_sub_agent=True)
        client = TestClient(app)

        resp = client.post(
            "/api/v2/agents/proj_test/inject",
            json={"content": "hi", "target": "@worker"},
        )
        assert resp.status_code == 200, resp.text

        messages = _read_session_jsonl(workspace)
        user_msgs = [m for m in messages if m.get("role") == "user"]
        assert len(user_msgs) == 1
        msg = user_msgs[0]
        assert msg["content"] == "hi"
        # Absent (not None) — None should not persist for an unused field.
        assert "attachments" not in msg


# ---------------------------------------------------------------------------
# Validation failures
# ---------------------------------------------------------------------------


class TestValidationFailures:
    def test_path_does_not_exist_returns_400_no_session_write(self, workspace):
        # Create a session so we can compare the line count before/after.
        # First request without attachments primes the session JSONL.
        app, _ = _make_app(workspace, with_sub_agent=True)
        client = TestClient(app)
        client.post(
            "/api/v2/agents/proj_test/inject",
            json={"content": "first", "target": "@worker"},
        )
        files = glob.glob(
            os.path.join(workspace, "orbital", "sessions", "*.jsonl"),
        )
        assert len(files) == 1
        before = sum(1 for _ in open(files[0], "r", encoding="utf-8"))

        # Now the bad request — file does not exist on disk.
        resp = client.post(
            "/api/v2/agents/proj_test/inject",
            json={
                "content": "second",
                "target": "@worker",
                "attachments": [
                    {"path": "uploads/missing.txt", "mime": "text/plain", "size": 11},
                ],
            },
        )
        assert resp.status_code == 400, resp.text
        # Detail mentions the path.
        assert "uploads/missing.txt" in resp.text

        # No new line should have been written to the session JSONL.
        after = sum(1 for _ in open(files[0], "r", encoding="utf-8"))
        assert after == before, (
            f"validation failure must not write to session JSONL "
            f"({before} -> {after})"
        )

    def test_path_with_dotdot_rejected_at_pydantic_layer(self, workspace):
        app, _ = _make_app(workspace, with_sub_agent=False)
        client = TestClient(app)
        resp = client.post(
            "/api/v2/agents/proj_test/inject",
            json={
                "content": "hi",
                "attachments": [
                    {"path": "../etc/passwd", "mime": "text/plain", "size": 1},
                ],
            },
        )
        # FastAPI returns 422 for Pydantic validation errors.
        assert resp.status_code >= 400 and resp.status_code < 500, resp.text

    def test_size_mismatch_returns_400(self, workspace):
        actual = _write_upload(workspace, "uploads/foo.txt", b"x" * 50)
        assert actual == 50
        app, _ = _make_app(workspace, with_sub_agent=False)
        client = TestClient(app)
        resp = client.post(
            "/api/v2/agents/proj_test/inject",
            json={
                "content": "hi",
                "attachments": [
                    # Declare 100, actual on disk is 50.
                    {"path": "uploads/foo.txt", "mime": "text/plain", "size": 100},
                ],
            },
        )
        assert resp.status_code == 400, resp.text

    def test_symlink_escape_returns_400(self, workspace):
        # outside_dir is a sibling of workspace.
        outside_file = os.path.join(workspace, "..", "secret.txt")
        # Write on outside path safely.
        outside_file_abs = os.path.realpath(outside_file)
        with open(outside_file_abs, "wb") as f:
            f.write(b"top-secret")
        try:
            uploads = os.path.join(workspace, "uploads")
            os.makedirs(uploads, exist_ok=True)
            link_path = os.path.join(uploads, "leak.txt")
            os.symlink(outside_file_abs, link_path)

            app, _ = _make_app(workspace, with_sub_agent=False)
            client = TestClient(app)
            resp = client.post(
                "/api/v2/agents/proj_test/inject",
                json={
                    "content": "hi",
                    "attachments": [
                        {"path": "uploads/leak.txt", "mime": "text/plain", "size": 10},
                    ],
                },
            )
            assert resp.status_code == 400, resp.text
        finally:
            try:
                os.remove(outside_file_abs)
            except OSError:
                pass
