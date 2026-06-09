# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Integration test (Part 2) — file upload creates a queue item.

POST /files/upload must, after writing the file to disk, enqueue a queue
item (source="upload") so the running/queued agent is notified. Re-uploading
the same file must not create a duplicate (idempotency dedup).
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from agent_os.api.app import create_app


@pytest.fixture
def client(tmp_path, monkeypatch):
    # create_project reads the global API key via get_api_key(), which checks
    # AGENT_OS_API_KEY before the OS keychain. Setting it keeps the test off
    # the macOS Keychain (which blocks on an access prompt in headless/CI runs).
    monkeypatch.setenv("AGENT_OS_API_KEY", "sk-env-test")
    app = create_app(data_dir=str(tmp_path / "data"))
    return TestClient(app)


@pytest.fixture
def project(client, tmp_path):
    ws = tmp_path / "ws"
    ws.mkdir()
    resp = client.post("/api/v2/projects", json={
        "name": "UploadProj",
        "workspace": str(ws),
        "model": "gpt-4",
        "api_key": "sk-test",
    })
    assert resp.status_code == 201, resp.text
    return resp.json()["project_id"]


def _upload(client, pid, name="notes.txt", body=b"hello world"):
    return client.post(
        f"/api/v2/projects/{pid}/files/upload",
        files={"file": (name, body, "text/plain")},
    )


def test_upload_creates_queue_item(client, project):
    resp = _upload(client, project)
    assert resp.status_code == 200, resp.text

    q = client.get(f"/api/v2/projects/{project}/queue")
    assert q.status_code == 200, q.text
    items = q.json()["items"]
    upload_items = [it for it in items if it.get("source") == "upload"]
    assert len(upload_items) == 1
    item = upload_items[0]
    assert "notes.txt" in item["content"]
    assert any("notes.txt" in ref for ref in item["file_refs"])


def test_reupload_same_file_does_not_duplicate(client, project):
    _upload(client, project)
    _upload(client, project)  # network retry / same file again

    q = client.get(f"/api/v2/projects/{project}/queue")
    items = q.json()["items"]
    upload_items = [it for it in items if it.get("source") == "upload"]
    assert len(upload_items) == 1, "idempotency_key should dedup the re-upload"


def test_upload_with_notify_false_writes_file_but_no_queue_item(client, project):
    """Composer attachments pass notify=false: the file lands on disk but no
    standalone 'upload' queue item is created (the file is delivered inline in
    the chat message, so a separate queue task would double-process it)."""
    resp = client.post(
        f"/api/v2/projects/{project}/files/upload?notify=false",
        files={"file": ("attach.txt", b"inline attachment", "text/plain")},
    )
    assert resp.status_code == 200, resp.text
    assert "attach.txt" in resp.json()["path"]

    q = client.get(f"/api/v2/projects/{project}/queue")
    items = q.json()["items"]
    upload_items = [it for it in items if it.get("source") == "upload"]
    assert upload_items == [], "notify=false must not enqueue an upload item"


def test_upload_still_succeeds_when_file_written(client, project, tmp_path):
    """The file must land on disk regardless of queue wiring."""
    resp = _upload(client, project, name="report.md", body=b"# Report\nbody")
    assert resp.status_code == 200
    data = resp.json()
    assert data["size"] == len(b"# Report\nbody")
    assert "report.md" in data["path"]
