# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Spec 103 P2 + P3 — conditional re-read and backend hygiene on ``/files``.

P2: ``GET /files/content`` takes an opt-in ``if_revision=<rev>``. When it
matches the file's current revision (``mtime_ns-size``, the same value the
``/files/preview`` ETag uses) the route answers a ~100-byte
``{"path", "revision", "unchanged": true}`` envelope WITHOUT opening the file.
The workspace panel sends it on its silent refresh after every tool result, so
an unchanged 2 MB PNG or 7 MB pptx no longer travels again per tick. Opt-in
because the relay serves its own older SPA build: a client that never sends it
keeps getting today's envelopes, byte for byte.

P3: the blocking bodies of ``list_files`` / ``get_file_content`` /
``resolve_file`` run in a worker thread, and the text cap is counted in bytes
(it used to count characters, so a CJK file sent up to three times the cap).
"""
import asyncio
import base64
import os
from unittest.mock import MagicMock
from urllib.parse import quote

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from agent_os.api.routes import files_v2

TINY_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
)
PDF_BYTES = b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n1 0 obj<<>>endobj\ntrailer<<>>\n%%EOF\n"
BLOB_BYTES = b"\x00\x80\x81\xff\xfe binary"
CAP = files_v2.MAX_PREVIEW_BYTES


@pytest.fixture
def workspace(tmp_path):
    ws = tmp_path / "proj"
    ws.mkdir()
    (ws / "notes.txt").write_text("plain", encoding="utf-8")
    (ws / "page.html").write_text("<p>hi</p>", encoding="utf-8")
    (ws / "icon.png").write_bytes(TINY_PNG)
    (ws / "blob.bin").write_bytes(BLOB_BYTES)
    (ws / "report.pdf").write_bytes(PDF_BYTES)
    (ws / "sub").mkdir()
    (ws / "sub" / "deep.txt").write_text("deep", encoding="utf-8")
    return ws


@pytest.fixture
def client(workspace):
    app = FastAPI()
    store = MagicMock()
    store.get_project.return_value = {"project_id": "p1", "workspace": str(workspace)}
    files_v2.configure(store)
    app.include_router(files_v2.router)
    return TestClient(app)


def _get(client, path, **params):
    query = "&".join(f"{k}={quote(str(v), safe='')}" for k, v in params.items())
    return client.get(f"/api/v2/projects/p1/files/content?path={quote(path)}" + (f"&{query}" if query else ""))


def _revision_of(workspace, path):
    stat = os.stat(workspace / path)
    return f"{stat.st_mtime_ns:x}-{stat.st_size:x}"


def _never_open(*_args, **_kwargs):
    raise AssertionError("the file must not be opened for an unchanged revision")


# ---------------------------------------------------------------------------
# P2 — if_revision
# ---------------------------------------------------------------------------


class TestConditionalRead:
    @pytest.mark.parametrize(
        "path,extra",
        [
            ("notes.txt", {}),
            ("page.html", {}),
            ("icon.png", {}),
            ("blob.bin", {}),
            ("report.pdf", {"document_preview": 1}),
            ("report.pdf", {}),
        ],
    )
    def test_same_revision_is_the_unchanged_envelope_without_opening_the_file(
        self, client, workspace, monkeypatch, path, extra
    ):
        rev = _get(client, path, **extra).json()["revision"]
        assert rev == _revision_of(workspace, path)

        # The module resolves `open` through its globals first, so shadowing it
        # there proves the short-circuit never touches the file's bytes.
        monkeypatch.setattr(files_v2, "open", _never_open, raising=False)
        resp = _get(client, path, if_revision=rev, **extra)
        assert resp.status_code == 200
        assert resp.json() == {"path": path, "revision": rev, "unchanged": True}
        # ~100 bytes on the wire instead of the whole file.
        assert len(resp.content) < 200

    def test_stale_revision_gets_the_full_envelope(self, client, workspace):
        full = _get(client, "icon.png").json()
        resp = _get(client, "icon.png", if_revision="0-0")
        assert resp.status_code == 200
        assert resp.json() == full
        assert "unchanged" not in resp.json()

    def test_a_rewrite_moves_the_revision_so_the_old_one_reads_in_full(self, client, workspace):
        rev = _get(client, "notes.txt").json()["revision"]
        assert _get(client, "notes.txt", if_revision=rev).json()["unchanged"] is True
        os.utime(workspace / "notes.txt", ns=(1_700_000_000_000_000_000, 1_700_000_000_000_000_000))
        (workspace / "notes.txt").write_text("plain, edited", encoding="utf-8")
        data = _get(client, "notes.txt", if_revision=rev).json()
        assert "unchanged" not in data
        assert data["content"] == "plain, edited"
        assert data["revision"] != rev

    def test_empty_param_is_never_a_match(self, client):
        data = _get(client, "notes.txt", if_revision="").json()
        assert "unchanged" not in data
        assert data["content"] == "plain"

    def test_missing_file_is_still_404_with_the_param(self, client):
        assert _get(client, "nope.txt", if_revision="1-1").status_code == 404

    def test_traversal_is_still_refused_with_the_param(self, client):
        assert _get(client, "../outside.txt", if_revision="1-1").status_code == 400


class TestAbsentParamIsTodaysEnvelope:
    """A client that never sends ``if_revision`` (the relay's SPA build, a cached
    bundle) gets exactly the pre-103 shapes: same keys, same values."""

    def test_text(self, client, workspace):
        assert _get(client, "notes.txt").json() == {
            "path": "notes.txt",
            "content": "plain",
            "type": "text",
            "size": 5,
            "revision": _revision_of(workspace, "notes.txt"),
            "truncated": False,
        }

    def test_html(self, client, workspace):
        assert _get(client, "page.html").json() == {
            "path": "page.html",
            "content": "<p>hi</p>",
            "type": "html",
            "mime": "text/html",
            "size": 9,
            "revision": _revision_of(workspace, "page.html"),
            "truncated": False,
        }

    def test_image(self, client, workspace):
        assert _get(client, "icon.png").json() == {
            "path": "icon.png",
            "content": base64.b64encode(TINY_PNG).decode("ascii"),
            "type": "image",
            "mime": "image/png",
            "size": len(TINY_PNG),
            "revision": _revision_of(workspace, "icon.png"),
        }

    def test_binary(self, client, workspace):
        assert _get(client, "blob.bin").json() == {
            "path": "blob.bin",
            "type": "binary",
            "mime": "application/octet-stream",
            "size": len(BLOB_BYTES),
            "revision": _revision_of(workspace, "blob.bin"),
            "content": base64.b64encode(BLOB_BYTES).decode("ascii"),
            "download_url": "/api/v2/projects/p1/files/download?path=blob.bin",
        }

    def test_document(self, client, workspace):
        rev = _revision_of(workspace, "report.pdf")
        assert _get(client, "report.pdf", document_preview=1).json() == {
            "path": "report.pdf",
            "type": "document",
            "format": "pdf",
            "mime": "application/pdf",
            "size": len(PDF_BYTES),
            "revision": rev,
            "content": "",
            "truncated": False,
            "preview_url": f"/api/v2/projects/p1/files/preview?path=report.pdf&v={rev}",
            "download_url": "/api/v2/projects/p1/files/download?path=report.pdf",
        }

    def test_crlf_text_keeps_universal_newlines(self, client, workspace):
        # The pre-103 text branch read in text mode, which folds \r\n and \r
        # into \n. The byte-mode read must produce the same string.
        (workspace / "crlf.txt").write_bytes(b"a\r\nb\rc\n")
        data = _get(client, "crlf.txt").json()
        assert data["content"] == "a\nb\nc\n"
        assert data["size"] == 7

    def test_bom_is_kept(self, client, workspace):
        (workspace / "bom.txt").write_bytes(b"\xef\xbb\xbfhi")
        assert _get(client, "bom.txt").json()["content"] == "﻿hi"


# ---------------------------------------------------------------------------
# P3 — byte-accurate text cap
# ---------------------------------------------------------------------------


class TestByteAccurateCap:
    def test_cjk_file_over_the_cap_sends_at_most_the_cap_in_bytes(self, client, workspace):
        # 200,000 three-byte characters = 600,000 bytes. The old cap counted
        # characters and sent all 600,000 bytes.
        (workspace / "cjk.md").write_text("中" * 200_000, encoding="utf-8")
        data = _get(client, "cjk.md").json()
        assert data["type"] == "text"
        assert data["truncated"] is True
        assert data["size"] == 600_000
        assert len(data["content"].encode("utf-8")) <= CAP
        # The cut lands between characters: 512,000 // 3 whole characters.
        assert data["content"] == "中" * (CAP // 3)

    def test_a_cut_four_byte_sequence_is_trimmed_not_binary(self, client, workspace):
        # The cap falls two bytes into the emoji; trim the partial sequence.
        body = ("a" * (CAP - 2)).encode("utf-8") + "😀".encode("utf-8") + b"tail"
        (workspace / "emoji.txt").write_bytes(body)
        data = _get(client, "emoji.txt").json()
        assert data["type"] == "text"
        assert data["truncated"] is True
        assert data["content"] == "a" * (CAP - 2)

    def test_html_branch_shares_the_byte_cap(self, client, workspace):
        (workspace / "big.html").write_text("中" * 200_000, encoding="utf-8")
        data = _get(client, "big.html").json()
        assert data["type"] == "html"
        assert data["truncated"] is True
        assert len(data["content"].encode("utf-8")) <= CAP

    def test_a_file_under_the_cap_is_complete(self, client, workspace):
        (workspace / "small.md").write_text("中" * 1000 + "😀", encoding="utf-8")
        data = _get(client, "small.md").json()
        assert data["content"] == "中" * 1000 + "😀"
        assert data["truncated"] is False

    def test_a_genuine_decode_error_elsewhere_is_still_binary(self, client, workspace):
        body = b"text " * 1000 + b"\xff\xfe" + b"text " * 1000
        (workspace / "mixed.dat").write_bytes(body)
        data = _get(client, "mixed.dat").json()
        assert data["type"] == "binary"
        assert base64.b64decode(data["content"]) == body

    def test_an_invalid_byte_inside_the_last_three_of_the_cap_is_still_binary(self, client, workspace):
        # Not an incomplete sequence — a stray 0xFF right before the cut. The
        # old text-mode read raised on it; so must the byte-mode read.
        body = b"a" * (CAP - 2) + b"\xff" + b"a" * 100
        (workspace / "stray.dat").write_bytes(body)
        data = _get(client, "stray.dat").json()
        assert data["type"] == "binary"

    def test_an_incomplete_sequence_at_the_true_end_of_a_small_file_is_binary(self, client, workspace):
        # Nothing was cut by the cap; the file itself is malformed.
        (workspace / "broken.txt").write_bytes(b"abc\xe4\xb8")
        data = _get(client, "broken.txt").json()
        assert data["type"] == "binary"


# ---------------------------------------------------------------------------
# P3 — the blocking bodies run off the event loop
# ---------------------------------------------------------------------------


class TestOffLoop:
    @pytest.fixture
    def threaded(self, monkeypatch):
        calls: list[str] = []
        real = asyncio.to_thread

        async def spy(fn, /, *args, **kwargs):
            calls.append(getattr(fn, "__name__", repr(fn)))
            return await real(fn, *args, **kwargs)

        monkeypatch.setattr(files_v2.asyncio, "to_thread", spy)
        return calls

    def test_list_files_runs_in_a_worker_thread(self, client, threaded):
        resp = client.get("/api/v2/projects/p1/files?path=sub")
        assert resp.status_code == 200
        assert [e["name"] for e in resp.json()["entries"]] == ["deep.txt"]
        assert len(threaded) == 1

    def test_get_file_content_runs_in_a_worker_thread(self, client, threaded):
        for path in ("notes.txt", "icon.png", "blob.bin"):
            assert _get(client, path).status_code == 200
        assert len(threaded) == 3

    def test_resolve_file_runs_in_a_worker_thread(self, client, threaded):
        resp = client.get("/api/v2/projects/p1/files/resolve?path=deep.txt")
        assert resp.status_code == 200
        assert resp.json()["matches"] == ["sub/deep.txt"]
        assert len(threaded) == 1

    def test_http_errors_raised_in_the_worker_still_map_to_status_codes(self, client, threaded):
        assert client.get("/api/v2/projects/p1/files?path=nope").status_code == 404
        assert _get(client, "nope.txt").status_code == 404
        assert client.get("/api/v2/projects/p1/files/resolve?path=../x").status_code == 400
        assert len(threaded) == 3
