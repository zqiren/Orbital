# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""The SPA fallback route must never serve a file outside the SPA folder.

The packaged app listens on every interface without auth, so a request path
that escapes the folder is an any-file read for anyone on the same network.

The route is driven through the real ``create_app()`` app with a raw ASGI
scope built the way uvicorn builds it (``path`` = percent-decoded,
``raw_path`` = bytes as sent, no dot-segment or slash normalisation), because
HTTP test clients normalise ``//`` and ``..`` before the app ever sees them.
The Windows-only forms (drive letters, backslashes, UNC) are checked against
the pure string guard, which must not depend on ``os.path`` semantics.
"""

from __future__ import annotations

import asyncio
import os
from urllib.parse import unquote

import pytest

from agent_os.api.app import create_app

SECRET = "TOP-SECRET-OUTSIDE-SPA"
INDEX = "<html><head></head><body>spa-index</body></html>"
ASSET = "console.log('asset');"


@pytest.fixture
def spa_app(tmp_path, monkeypatch):
    root = tmp_path / "web"
    (root / "assets").mkdir(parents=True)
    (root / "index.html").write_text(INDEX, encoding="utf-8")
    (root / "assets" / "app.js").write_text(ASSET, encoding="utf-8")

    # Outside the folder: a secret, and a sibling sharing the name prefix.
    (tmp_path / "secret.txt").write_text(SECRET, encoding="utf-8")
    evil = tmp_path / "web-evil"
    evil.mkdir()
    (evil / "loot.txt").write_text(SECRET, encoding="utf-8")

    # Symlinks inside the folder that point out of it.
    os.symlink(tmp_path / "secret.txt", root / "link.txt")
    os.symlink(evil, root / "evil")

    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("AGENT_OS_SPA_DIR", str(root))
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    return create_app(data_dir=str(data_dir)), tmp_path


def _raw_get(app, raw_target: str) -> tuple[int, bytes]:
    """Send one GET through the ASGI app exactly as uvicorn would."""
    raw_path = raw_target.encode("latin-1")
    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "GET",
        "scheme": "http",
        "path": unquote(raw_target),
        "raw_path": raw_path,
        "query_string": b"",
        "root_path": "",
        "headers": [(b"host", b"127.0.0.1:8000")],
        "client": ("127.0.0.1", 50000),
        "server": ("127.0.0.1", 8000),
    }
    status: list[int] = []
    body: list[bytes] = []
    sent = False

    async def receive():
        nonlocal sent
        if not sent:
            sent = True
            return {"type": "http.request", "body": b"", "more_body": False}
        await asyncio.sleep(3600)

    async def send(message):
        if message["type"] == "http.response.start":
            status.append(message["status"])
        elif message["type"] == "http.response.body":
            body.append(message.get("body", b""))

    asyncio.run(app(scope, receive, send))
    return status[0], b"".join(body)


@pytest.mark.parametrize("target_fmt", [
    "/{abs}",                      # //<tmp>/secret.txt: absolute once the route strips one slash
    "/..%2Fsecret.txt",
    "/%2e%2e/secret.txt",
    "/%2E%2E%2Fsecret.txt",
    "/../secret.txt",
    "/assets/../../secret.txt",
    "/..%2Fweb-evil/loot.txt",     # sibling that shares the folder's name prefix
    "/link.txt",                   # symlink inside the folder pointing out
    "/evil/loot.txt",              # symlinked dir into the prefix-sharing sibling
])
def test_escapes_are_refused_with_404(spa_app, target_fmt):
    app, tmp_path = spa_app
    target = target_fmt.format(abs=str(tmp_path / "secret.txt"))
    status, body = _raw_get(app, target)
    assert SECRET.encode() not in body
    assert status == 404, (target, status, body[:200])
    assert b"spa-index" not in body   # refused, not silently served index.html


def test_double_slash_etc_passwd_is_refused(spa_app):
    app, _ = spa_app
    status, body = _raw_get(app, "//etc/passwd")
    assert status == 404
    assert b"root:" not in body


def test_real_asset_is_served(spa_app):
    app, _ = spa_app
    status, body = _raw_get(app, "/assets/app.js")
    assert status == 200
    assert body.decode() == ASSET


def test_unknown_app_route_gets_index(spa_app):
    app, _ = spa_app
    for target in ("/", "/projects/abc/settings", "/assets/missing.js"):
        status, body = _raw_get(app, target)
        assert status == 200, target
        assert b"spa-index" in body
        assert b"window.__AGENT_OS_LOCAL__=true" in body


# ---------------------------------------------------------------------------
# The pure string guard (runs before any filesystem call; covers Windows forms
# on any OS).
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("full_path", [
    "/etc/passwd",                 # GET //etc/passwd
    "//host/share/x",              # GET ///host/share/x  (UNC)
    "\\\\host\\share\\x",          # UNC with backslashes
    "C:/Windows/win.ini",          # GET /C:/Windows/win.ini
    "c:/windows/win.ini",
    "C:Windows/win.ini",           # drive-relative
    "C:",
    "a:b/c",                       # colon in the first segment
    "..\\..\\Windows\\win.ini",    # ..%5C forms
    "assets\\..\\..\\secret.txt",
    "assets\\app.js",              # any backslash
    "..",
    "../secret.txt",
    "assets/../../secret.txt",
    "assets/..",
    "...",                         # Windows strips trailing dots: "..." -> ".."
    ".. /secret.txt",              # and trailing spaces
    "assets/index.html\x00.js",    # NUL
])
def test_plain_guard_rejects(full_path):
    from agent_os.api.app import _spa_request_path_is_plain
    assert _spa_request_path_is_plain(full_path) is False


@pytest.mark.parametrize("full_path", [
    "",
    "index.html",
    "assets/app.js",
    "assets/app-3f9a.js",
    "projects/abc/settings",
    "./assets/app.js",
    ".well-known/x",
    "favicon.ico",
])
def test_plain_guard_accepts_normal_paths(full_path):
    from agent_os.api.app import _spa_request_path_is_plain
    assert _spa_request_path_is_plain(full_path) is True
