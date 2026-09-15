# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Spec 090 — document preview.

``/files/content`` classifies PDF / DOC / DOCX / XLSX / XLS / CSV BEFORE the
UTF-8 attempt and answers with a metadata envelope (never base64-in-JSON), but
only for a client that opts in with ``document_preview=1``. Older frontends
(the relay's own SPA build, a cached bundle) never send it and keep getting the
pre-090 shapes. The bytes stream from ``/files/preview``: realpath containment,
a fixed content-type map, inline disposition, ETag/Last-Modified, and the 50 MB
download ceiling.
"""
import base64
import os
from unittest.mock import MagicMock
from urllib.parse import quote

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from agent_os.api.routes import files_v2

PDF_BYTES = b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n1 0 obj<<>>endobj\ntrailer<<>>\n%%EOF\n"
# ZIP / Compound File headers followed by bytes that are not valid UTF-8, as in
# real Office files (so the pre-090 path really does classify them as binary).
DOCX_BYTES = b"PK\x03\x04\x14\x00\x06\x00\x08\x00\xa0\xff docx-ish"
XLSX_BYTES = b"PK\x03\x04\x14\x00\x06\x00\x08\x00\xa0\xff xlsx-ish"
DOC_BYTES = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1doc-ish"
XLS_BYTES = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1xls-ish"
CSV_TEXT = "名称,数量\n苹果,3\n"


@pytest.fixture
def workspace(tmp_path):
    ws = tmp_path / "proj"
    ws.mkdir()
    (ws / "report.pdf").write_bytes(PDF_BYTES)
    (ws / "REPORT-UPPER.PDF").write_bytes(PDF_BYTES)
    (ws / "memo.docx").write_bytes(DOCX_BYTES)
    (ws / "form.doc").write_bytes(DOC_BYTES)
    (ws / "book.xlsx").write_bytes(XLSX_BYTES)
    (ws / "legacy.xls").write_bytes(XLS_BYTES)
    (ws / "data.csv").write_text(CSV_TEXT, encoding="utf-8")
    (ws / "gbk.csv").write_bytes(CSV_TEXT.encode("gb18030"))
    (ws / "notes.txt").write_text("plain", encoding="utf-8")
    (ws / "blob.bin").write_bytes(b"\x00\x80\x81\xff")
    (ws / "Q3 report & notes#1.pdf").write_bytes(PDF_BYTES)
    (ws / "报告.pdf").write_bytes(PDF_BYTES)
    (ws / "folder.pdf").mkdir()
    return ws


@pytest.fixture
def client(workspace):
    app = FastAPI()
    store = MagicMock()
    store.get_project.return_value = {"project_id": "p1", "workspace": str(workspace)}
    files_v2.configure(store)
    app.include_router(files_v2.router)
    return TestClient(app)


def _content(client, path):
    """A document-preview client (spec 090 frontends send the opt-in flag)."""
    return client.get(f"/api/v2/projects/p1/files/content?path={quote(path)}&document_preview=1")


def _content_legacy(client, path, extra=""):
    """An older frontend: no flag."""
    return client.get(f"/api/v2/projects/p1/files/content?path={quote(path)}{extra}")


def _preview(client, path, **kwargs):
    return client.get(f"/api/v2/projects/p1/files/preview?path={quote(path)}", **kwargs)


class TestContentEnvelope:
    @pytest.mark.parametrize(
        "path,fmt,mime",
        [
            ("report.pdf", "pdf", "application/pdf"),
            ("REPORT-UPPER.PDF", "pdf", "application/pdf"),
            (
                "memo.docx",
                "docx",
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            ),
            ("form.doc", "doc", "application/msword"),
            (
                "book.xlsx",
                "xlsx",
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            ),
            ("legacy.xls", "xls", "application/vnd.ms-excel"),
            ("data.csv", "csv", "text/csv"),
        ],
    )
    def test_document_types_return_metadata_not_bytes(self, client, path, fmt, mime):
        resp = _content(client, path)
        assert resp.status_code == 200
        data = resp.json()
        assert data["type"] == "document"
        assert data["format"] == fmt
        assert data["mime"] == mime
        assert data["path"] == path
        assert data["size"] > 0
        # Never base64-in-JSON for a document type.
        assert data["content"] == ""
        assert data["truncated"] is False
        assert data["preview_url"] == f"/api/v2/projects/p1/files/preview?path={quote(path)}"
        assert data["download_url"] == f"/api/v2/projects/p1/files/download?path={quote(path)}"
        assert "preview_unavailable" not in data

    def test_gbk_csv_is_classified_before_the_utf8_attempt(self, client):
        # A GBK CSV fails UTF-8 decoding; it must still be a document (the client
        # decodes it), not fall through to the base64 binary branch.
        data = _content(client, "gbk.csv").json()
        assert data["type"] == "document"
        assert data["format"] == "csv"
        assert data["content"] == ""

    def test_preview_url_round_trips_awkward_names(self, client):
        for name in ("Q3 report & notes#1.pdf", "报告.pdf"):
            data = _content(client, name).json()
            resp = client.get(data["preview_url"])
            assert resp.status_code == 200, name
            assert resp.content == PDF_BYTES

    @pytest.mark.parametrize("path", ["report.pdf", "form.doc"])
    def test_over_ceiling_document_is_flagged(self, client, monkeypatch, path):
        monkeypatch.setattr(files_v2, "MAX_DOWNLOAD_BYTES", 10)
        data = _content(client, path).json()
        assert data["type"] == "document"
        assert data["preview_unavailable"] == "too_large"
        assert data["content"] == ""

    def test_other_binaries_keep_todays_base64_behaviour(self, client):
        data = _content(client, "blob.bin").json()
        assert data["type"] == "binary"
        assert base64.b64decode(data["content"]) == b"\x00\x80\x81\xff"

    def test_text_files_unchanged(self, client):
        data = _content(client, "notes.txt").json()
        assert data["type"] == "text"
        assert data["content"] == "plain"


class TestLegacyShapesWithoutFlag:
    """Older frontends never send ``document_preview=1``; they must get exactly
    the pre-090 responses (a download card for Office/PDF, text for CSV)."""

    @pytest.mark.parametrize(
        "path,raw,mime",
        [
            ("report.pdf", PDF_BYTES, "application/pdf"),
            (
                "memo.docx",
                DOCX_BYTES,
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            ),
            ("form.doc", DOC_BYTES, "application/msword"),
            (
                "book.xlsx",
                XLSX_BYTES,
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            ),
            ("legacy.xls", XLS_BYTES, "application/vnd.ms-excel"),
        ],
    )
    def test_office_and_pdf_are_base64_binary(self, client, path, raw, mime):
        resp = _content_legacy(client, path)
        assert resp.status_code == 200
        data = resp.json()
        assert data["type"] == "binary"
        assert data["mime"] == mime
        assert data["size"] == len(raw)
        assert base64.b64decode(data["content"]) == raw
        assert data["download_url"] == f"/api/v2/projects/p1/files/download?path={path}"
        assert "preview_url" not in data
        assert "format" not in data

    def test_utf8_csv_is_text(self, client):
        data = _content_legacy(client, "data.csv").json()
        assert data["type"] == "text"
        assert data["content"] == CSV_TEXT
        assert data["truncated"] is False

    def test_non_utf8_csv_falls_back_to_binary(self, client):
        data = _content_legacy(client, "gbk.csv").json()
        assert data["type"] == "binary"
        assert base64.b64decode(data["content"]) == CSV_TEXT.encode("gb18030")

    def test_flag_off_is_the_legacy_shape(self, client):
        data = _content_legacy(client, "report.pdf", "&document_preview=0").json()
        assert data["type"] == "binary"

    def test_over_ceiling_binary_keeps_the_empty_content_rule(self, client, monkeypatch):
        monkeypatch.setattr(files_v2, "MAX_DOWNLOAD_BYTES", 10)
        data = _content_legacy(client, "report.pdf").json()
        assert data["type"] == "binary"
        assert data["content"] == ""


class TestPreviewRoute:
    def test_streams_raw_bytes_inline(self, client):
        resp = _preview(client, "report.pdf")
        assert resp.status_code == 200
        assert resp.content == PDF_BYTES
        assert resp.headers["content-type"] == "application/pdf"
        disposition = resp.headers["content-disposition"]
        assert disposition.startswith("inline")
        assert "attachment" not in disposition
        assert resp.headers["x-content-type-options"] == "nosniff"
        assert resp.headers["etag"]
        assert resp.headers["last-modified"]

    @pytest.mark.parametrize(
        "path,ctype",
        [
            ("memo.docx", "application/vnd.openxmlformats-officedocument.wordprocessingml.document"),
            ("form.doc", "application/msword"),
            ("book.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"),
            ("legacy.xls", "application/vnd.ms-excel"),
            ("gbk.csv", "text/csv"),
        ],
    )
    def test_content_type_comes_from_the_fixed_map(self, client, path, ctype):
        resp = _preview(client, path)
        assert resp.status_code == 200
        assert resp.headers["content-type"].split(";")[0] == ctype
        assert resp.headers["content-disposition"].startswith("inline")

    def test_non_ascii_filename_disposition(self, client):
        resp = _preview(client, "报告.pdf")
        assert resp.status_code == 200
        assert resp.headers["content-disposition"].startswith("inline")

    def test_etag_revalidation_returns_304(self, client):
        first = _preview(client, "report.pdf")
        etag = first.headers["etag"]
        again = _preview(client, "report.pdf", headers={"If-None-Match": etag})
        assert again.status_code == 304
        assert again.content == b""
        assert again.headers["etag"] == etag

    def test_etag_changes_with_the_file_revision(self, client, workspace):
        etag = _preview(client, "report.pdf").headers["etag"]
        (workspace / "report.pdf").write_bytes(PDF_BYTES + b"% appended\n")
        resp = _preview(client, "report.pdf", headers={"If-None-Match": etag})
        assert resp.status_code == 200
        assert resp.headers["etag"] != etag

    def test_unsupported_type_is_refused(self, client):
        for path in ("notes.txt", "blob.bin"):
            resp = _preview(client, path)
            assert resp.status_code == 415, path
            assert "preview_unavailable" in resp.json()["detail"]

    @pytest.mark.parametrize("path", ["report.pdf", "form.doc"])
    def test_over_ceiling_is_413(self, client, monkeypatch, path):
        monkeypatch.setattr(files_v2, "MAX_DOWNLOAD_BYTES", 10)
        resp = _preview(client, path)
        assert resp.status_code == 413
        assert resp.json()["detail"] == "preview_unavailable: too_large"

    def test_missing_file_is_404(self, client):
        assert _preview(client, "missing.pdf").status_code == 404

    def test_directory_is_404(self, client):
        assert _preview(client, "folder.pdf").status_code == 404

    def test_project_not_found_is_404(self, workspace):
        app = FastAPI()
        store = MagicMock()
        store.get_project.return_value = None
        files_v2.configure(store)
        app.include_router(files_v2.router)
        resp = TestClient(app).get("/api/v2/projects/nope/files/preview?path=report.pdf")
        assert resp.status_code == 404

    def test_traversal_is_refused(self, client):
        resp = _preview(client, "../outside.pdf")
        assert resp.status_code == 400

    def test_symlink_escape_is_refused(self, client, workspace, tmp_path):
        secret = tmp_path / "secret.pdf"
        secret.write_bytes(PDF_BYTES)
        os.symlink(secret, workspace / "innocent.pdf")
        resp = _preview(client, "innocent.pdf")
        assert resp.status_code == 400
        assert resp.content != PDF_BYTES

    def test_sibling_prefix_directory_is_refused(self, client, tmp_path):
        evil = tmp_path / "proj-evil"
        evil.mkdir()
        (evil / "secret.pdf").write_bytes(PDF_BYTES)
        resp = _preview(client, "../proj-evil/secret.pdf")
        assert resp.status_code == 400


class TestRelayTransport:
    """The relay tunnel re-serialises every body as text (``resp.text``), which
    replaces non-UTF-8 bytes with U+FFFD. Relayed preview requests therefore get
    the bytes base64-wrapped in JSON — the only encoding that survives it."""

    def test_relayed_request_gets_a_base64_envelope(self, client):
        resp = _preview(client, "report.pdf", headers={"X-Via-Relay": "true"})
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("application/json")
        data = resp.json()
        assert data["encoding"] == "base64"
        assert data["size"] == len(PDF_BYTES)
        assert base64.b64decode(data["content"]) == PDF_BYTES

    def test_relayed_over_ceiling_is_413(self, client, monkeypatch):
        monkeypatch.setattr(files_v2, "MAX_DOWNLOAD_BYTES", 10)
        resp = _preview(client, "report.pdf", headers={"X-Via-Relay": "true"})
        assert resp.status_code == 413

    def test_local_request_is_never_base64(self, client):
        resp = _preview(client, "report.pdf")
        assert resp.headers["content-type"] == "application/pdf"
