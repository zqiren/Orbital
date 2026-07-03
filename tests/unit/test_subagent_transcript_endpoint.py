# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Regression tests for the sub-agent transcript drill-in endpoint (Task 4,
spec 009 Section 0.5):

    GET /api/v2/agents/{project_id}/sub-agents/{handle}/transcript?session_id=<sid>

This is the SOLE data source for the frontend's SubAgentDrillIn view (Task 5,
already committed and consuming the shape verbatim) — the response contract
(handle/display_name/kind/resumable/entries) must not drift. See
web/src/types.ts's ``SubAgentTranscriptResult``.

Covers:
- 404 for a handle with no transcript on disk.
- 200 + entries for a transcript seeded directly on disk (no live adapter) —
  proves the read path works for daemon-restarted / already-stopped handles,
  not just live ones.
- ``kind``/``resumable`` derivation: "worker:*" handles are workers
  (resumable=False); everything else is a CLI adapter (resumable=True).
- turn_complete boundary rows (empty-content instrumentation, never
  user-facing anywhere else in the codebase) are dropped from ``entries``.
- Invalid handle (path-traversal-shaped) is rejected with 400, not a path
  escape.
"""

from __future__ import annotations

import os

import pytest
from fastapi.testclient import TestClient

from agent_os.agent.project_paths import ProjectPaths
from agent_os.daemon_v2.sub_agent_transcript import SubAgentTranscript


@pytest.fixture
def app_state(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("USERPROFILE", str(tmp_path / "home"))
    os.makedirs(str(tmp_path / "home"), exist_ok=True)

    from agent_os.api.app import create_app
    app = create_app(data_dir=str(tmp_path / "data"))
    return app, tmp_path


def _make_project(client: TestClient, ws: str) -> str:
    payload = {
        "name": "transcripttest-" + os.path.basename(ws),
        "workspace": ws,
        "model": "gpt-4",
        "api_key": "test-key",
    }
    resp = client.post("/api/v2/projects", json=payload)
    assert resp.status_code == 201, resp.text
    return resp.json()["project_id"]


def _seed_transcript(ws: str, handle: str, entries: list[dict]) -> None:
    """Write a transcript directly to disk via the same on-disk convention
    the real daemon uses (.latest pointer + handle-scoped directory), with
    no live adapter involved — this is the "daemon restarted" / "sub-agent
    already stopped" shape the endpoint must still serve."""
    transcript = SubAgentTranscript.open_for_handle(ws, handle)
    for entry in entries:
        transcript.append(entry)


class TestSubAgentTranscriptEndpoint:

    def test_404_for_unknown_handle(self, app_state, tmp_path):
        app, _ = app_state
        client = TestClient(app)
        ws = str(tmp_path / "ws_unknown")
        os.makedirs(ws, exist_ok=True)
        pid = _make_project(client, ws)

        resp = client.get(
            f"/api/v2/agents/{pid}/sub-agents/claude-code/transcript",
        )
        assert resp.status_code == 404

    def test_200_with_entries_for_seeded_cli_transcript(self, app_state, tmp_path):
        app, _ = app_state
        client = TestClient(app)
        ws = str(tmp_path / "ws_seeded_cli")
        os.makedirs(ws, exist_ok=True)
        pid = _make_project(client, ws)

        _seed_transcript(ws, "claude-code", [
            {"source": "claude-code", "content": "on it",
             "timestamp": "2026-07-03T00:00:00+00:00", "chunk_type": "response"},
            {"source": "claude-code", "content": "[Using tool: Bash]",
             "timestamp": "2026-07-03T00:00:01+00:00", "chunk_type": "tool_activity"},
        ])

        resp = client.get(
            f"/api/v2/agents/{pid}/sub-agents/claude-code/transcript",
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["handle"] == "claude-code"
        assert body["kind"] == "cli"
        assert body["resumable"] is True
        assert body["display_name"] == "claude-code"  # no live adapter -> falls back to handle
        assert len(body["entries"]) == 2
        assert body["entries"][0]["content"] == "on it"
        assert body["entries"][0]["source"] == "claude-code"
        assert body["entries"][1]["chunk_type"] == "tool_activity"

    def test_worker_handle_is_not_resumable(self, app_state, tmp_path):
        app, _ = app_state
        client = TestClient(app)
        ws = str(tmp_path / "ws_worker")
        os.makedirs(ws, exist_ok=True)
        pid = _make_project(client, ws)

        handle = "worker:x-0"
        _seed_transcript(ws, handle, [
            {"source": handle, "content": "task result",
             "timestamp": "2026-07-03T00:00:00+00:00", "chunk_type": "response"},
        ])

        resp = client.get(
            f"/api/v2/agents/{pid}/sub-agents/{handle}/transcript",
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["handle"] == handle
        assert body["kind"] == "worker"
        assert body["resumable"] is False

    def test_turn_complete_boundary_rows_are_dropped(self, app_state, tmp_path):
        """turn_complete rows are empty-content instrumentation (see
        ProcessManager._append_turn_boundary) — never rendered anywhere else
        in the codebase, and must not leak into the drill-in entries."""
        app, _ = app_state
        client = TestClient(app)
        ws = str(tmp_path / "ws_boundary")
        os.makedirs(ws, exist_ok=True)
        pid = _make_project(client, ws)

        handle = "claude-code"
        _seed_transcript(ws, handle, [
            {"source": handle, "content": "hello",
             "timestamp": "2026-07-03T00:00:00+00:00", "chunk_type": "response"},
            {"source": handle, "content": "",
             "timestamp": "2026-07-03T00:00:01+00:00", "chunk_type": "turn_complete"},
        ])

        resp = client.get(
            f"/api/v2/agents/{pid}/sub-agents/{handle}/transcript",
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert len(body["entries"]) == 1
        assert body["entries"][0]["chunk_type"] == "response"

    def test_400_for_path_traversal_shaped_handle(self, app_state, tmp_path):
        app, _ = app_state
        client = TestClient(app)
        ws = str(tmp_path / "ws_traversal")
        os.makedirs(ws, exist_ok=True)
        pid = _make_project(client, ws)

        resp = client.get(
            f"/api/v2/agents/{pid}/sub-agents/..%2F..%2Fetc/transcript",
        )
        assert resp.status_code in (400, 404)  # never 200; may 404 via routing

        # An explicit, well-formed-but-traversal-bearing value fails 400 —
        # exercise the regex directly against a handle containing a dot,
        # which %2F-based URLs can't express through the path segment.
        resp2 = client.get(
            f"/api/v2/agents/{pid}/sub-agents/foo..bar/transcript",
        )
        assert resp2.status_code == 400

    def test_empty_transcript_file_returns_200_with_no_entries(self, app_state, tmp_path):
        """A handle that was started (file touched into existence) but has
        produced no chunks yet is a legitimate 200/empty, not a 404."""
        app, _ = app_state
        client = TestClient(app)
        ws = str(tmp_path / "ws_empty")
        os.makedirs(ws, exist_ok=True)
        pid = _make_project(client, ws)

        _seed_transcript(ws, "claude-code", [])  # touches the file, no entries

        resp = client.get(
            f"/api/v2/agents/{pid}/sub-agents/claude-code/transcript",
        )
        assert resp.status_code == 200, resp.text
        assert resp.json()["entries"] == []
