# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""The SPA route serves files from the SPA directory and nowhere else."""

from __future__ import annotations

from urllib.parse import quote

import pytest
from fastapi.testclient import TestClient

from agent_os.api.app import create_app


@pytest.fixture
def spa(tmp_path, monkeypatch):
    spa_dir = tmp_path / "spa"
    (spa_dir / "assets").mkdir(parents=True)
    (spa_dir / "index.html").write_text(
        "<html><head></head><body></body></html>", encoding="utf-8"
    )
    (spa_dir / "assets" / "app.js").write_text("console.log(1);", encoding="utf-8")
    (tmp_path / "outside.txt").write_text("outside", encoding="utf-8")
    monkeypatch.setenv("AGENT_OS_SPA_DIR", str(spa_dir))
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    return TestClient(create_app(data_dir=str(data_dir))), tmp_path


def test_relative_path_outside_the_spa_dir_is_refused(spa):
    client, _ = spa
    resp = client.get("/..%2Foutside.txt")
    assert resp.status_code == 404
    assert "outside" not in resp.text


def test_absolute_path_is_refused(spa):
    client, root = spa
    resp = client.get("/" + quote(str(root / "outside.txt"), safe=""))
    assert resp.status_code == 404
    assert "outside" not in resp.text


def test_sibling_folder_sharing_the_prefix_is_refused(spa):
    client, root = spa
    sibling = root / "spa-private"
    sibling.mkdir()
    (sibling / "secret.txt").write_text("outside", encoding="utf-8")
    resp = client.get("/..%2Fspa-private%2Fsecret.txt")
    assert resp.status_code == 404


def test_symlink_leaving_the_spa_dir_is_refused(spa):
    client, root = spa
    (root / "spa" / "link.txt").symlink_to(root / "outside.txt")
    assert client.get("/link.txt").status_code == 404


def test_files_inside_the_spa_dir_are_still_served(spa):
    client, _ = spa
    resp = client.get("/assets/app.js")
    assert resp.status_code == 200
    assert resp.text == "console.log(1);"


def test_unknown_routes_still_fall_back_to_the_app(spa):
    client, _ = spa
    resp = client.get("/projects/some-project")
    assert resp.status_code == 200
    assert "window.__AGENT_OS_LOCAL__=true" in resp.text
