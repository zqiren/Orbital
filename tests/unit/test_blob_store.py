# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Spec 066 phase 1 — the content-addressed blob store.

Identical bytes are stored once and referenced by hash. Session rows stop
carrying base64 images: the image goes to the store and the row keeps a
reference that rehydrates to the exact original row, so every reader that
rehydrates sees what it saw before.
"""

from __future__ import annotations

import base64
import copy
import json
import os

import pytest

from agent_os.agent import blob_store


PNG_A = b"\x89PNG\r\n\x1a\n" + b"A" * 2048
PNG_B = b"\x89PNG\r\n\x1a\n" + b"B" * 2048


def _data_url(raw: bytes, mime: str = "image/png") -> str:
    return f"data:{mime};base64,{base64.b64encode(raw).decode('ascii')}"


def _image_row(raw: bytes = PNG_A, *, meta: dict | None = None, detail: bool = True) -> dict:
    image_url = {"url": _data_url(raw)}
    if detail:
        image_url["detail"] = "low"
    row = {
        "role": "tool",
        "content": [
            {"type": "text", "text": "Screenshot of https://example.com. Title: Example"},
            {"type": "image_url", "image_url": image_url},
        ],
        "tool_call_id": "call_1",
        "source": "management",
        "timestamp": "2026-09-19T00:00:00+00:00",
    }
    if meta is not None:
        row["_meta"] = meta
    return row


@pytest.fixture
def orbital_dir(tmp_path):
    d = tmp_path / "ws" / "orbital"
    d.mkdir(parents=True)
    return str(d)


def _blob_files(orbital_dir):
    root = blob_store.blobs_dir(orbital_dir)
    out = []
    for dirpath, _dirs, files in os.walk(root):
        out.extend(os.path.join(dirpath, f) for f in files)
    return sorted(out)


# ---------------------------------------------------------------------------
# put / read
# ---------------------------------------------------------------------------

def test_identical_bytes_are_stored_once(orbital_dir):
    sha1, path1 = blob_store.put_bytes(orbital_dir, PNG_A, "png")
    sha2, path2 = blob_store.put_bytes(orbital_dir, PNG_A, "png")
    assert sha1 == sha2
    assert path1 == path2
    assert _blob_files(orbital_dir) == [path1]
    with open(path1, "rb") as f:
        assert f.read() == PNG_A


def test_different_bytes_get_different_blobs(orbital_dir):
    _, path_a = blob_store.put_bytes(orbital_dir, PNG_A, "png")
    _, path_b = blob_store.put_bytes(orbital_dir, PNG_B, "png")
    assert path_a != path_b
    assert len(_blob_files(orbital_dir)) == 2


def test_blob_is_addressed_by_sha256(orbital_dir):
    import hashlib
    sha, path = blob_store.put_bytes(orbital_dir, PNG_A, "png")
    assert sha == hashlib.sha256(PNG_A).hexdigest()
    assert path.endswith(f"{sha}.png")
    assert path.startswith(blob_store.blobs_dir(orbital_dir))


def test_read_verified_returns_bytes_or_none(orbital_dir):
    sha, _ = blob_store.put_bytes(orbital_dir, PNG_A, "png")
    assert blob_store.read_verified(orbital_dir, sha, "png") == PNG_A
    assert blob_store.read_verified(orbital_dir, "0" * 64, "png") is None


def test_read_verified_rejects_a_blob_whose_bytes_changed(orbital_dir):
    sha, path = blob_store.put_bytes(orbital_dir, PNG_A, "png")
    with open(path, "wb") as f:
        f.write(PNG_B)
    assert blob_store.read_verified(orbital_dir, sha, "png") is None


def test_put_heals_a_corrupted_blob(orbital_dir):
    sha, path = blob_store.put_bytes(orbital_dir, PNG_A, "png")
    with open(path, "wb") as f:
        f.write(b"garbage")
    blob_store.put_bytes(orbital_dir, PNG_A, "png")
    assert blob_store.read_verified(orbital_dir, sha, "png") == PNG_A


# ---------------------------------------------------------------------------
# de-inline / rehydrate rows
# ---------------------------------------------------------------------------

def test_deinline_replaces_base64_with_a_reference(orbital_dir):
    row = _image_row(meta={"screenshot_path": "/x/step_0001.png"})
    before = copy.deepcopy(row)
    out = blob_store.deinline_row_images(row, orbital_dir)

    assert row == before, "the caller's row must never be mutated"
    line = json.dumps(out)
    assert "base64," not in line
    assert len(line) < 1000
    # The image block becomes a plain text block: a reader that knows nothing
    # about blobs (an older Orbital) sees a harmless placeholder.
    assert out["content"][1]["type"] == "text"
    assert "tool-results/blobs/" in out["content"][1]["text"]
    refs = out["_meta"]["image_blobs"]
    assert refs[0]["index"] == 1
    assert refs[0]["mime"] == "image/png"
    assert _blob_files(orbital_dir) and len(_blob_files(orbital_dir)) == 1


def test_rehydrate_restores_the_exact_original_row(orbital_dir):
    row = _image_row(meta={"screenshot_path": "/x/step_0001.png", "url": "u"})
    stored = json.loads(json.dumps(blob_store.deinline_row_images(row, orbital_dir)))
    back = blob_store.rehydrate_row_images(stored, orbital_dir)
    assert json.dumps(back) == json.dumps(row), "byte-identical JSON, key order included"


def test_rehydrate_round_trip_without_meta_or_detail(orbital_dir):
    row = _image_row(detail=False)
    assert "_meta" not in row
    stored = json.loads(json.dumps(blob_store.deinline_row_images(row, orbital_dir)))
    back = blob_store.rehydrate_row_images(stored, orbital_dir)
    assert json.dumps(back) == json.dumps(row)


def test_deinline_twice_the_same_image_keeps_one_blob(orbital_dir):
    blob_store.deinline_row_images(_image_row(), orbital_dir)
    blob_store.deinline_row_images(_image_row(), orbital_dir)
    assert len(_blob_files(orbital_dir)) == 1


def test_rows_without_inline_images_pass_through_untouched(orbital_dir):
    plain = {"role": "tool", "content": "text result", "tool_call_id": "c"}
    remote = {"role": "tool", "content": [
        {"type": "image_url", "image_url": {"url": "https://example.com/a.png"}},
    ]}
    assert blob_store.deinline_row_images(plain, orbital_dir) is plain
    assert blob_store.deinline_row_images(remote, orbital_dir) is remote
    assert blob_store.rehydrate_row_images(plain, orbital_dir) is plain
    assert not os.path.exists(blob_store.blobs_dir(orbital_dir))


def test_non_canonical_base64_stays_inline(orbital_dir):
    """Anything that would not rehydrate byte-for-byte is left alone."""
    b64 = base64.b64encode(PNG_A).decode("ascii")
    wrapped = "\n".join(b64[i:i + 76] for i in range(0, len(b64), 76))
    row = {"role": "tool", "content": [
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{wrapped}"}},
    ]}
    assert blob_store.deinline_row_images(row, orbital_dir) is row


def test_rehydrate_with_missing_blob_keeps_the_placeholder(orbital_dir):
    stored = blob_store.deinline_row_images(_image_row(meta={"k": 1}), orbital_dir)
    for path in _blob_files(orbital_dir):
        os.remove(path)
    back = blob_store.rehydrate_row_images(stored, orbital_dir)
    # No crash, the placeholder and the reference both survive — a later
    # restore of the blob store brings the image back.
    assert back["content"][1]["type"] == "text"
    assert back["_meta"]["image_blobs"]


def test_deinline_is_idempotent_on_an_already_deinlined_row(orbital_dir):
    stored = blob_store.deinline_row_images(_image_row(meta={}), orbital_dir)
    again = blob_store.deinline_row_images(stored, orbital_dir)
    assert json.dumps(again) == json.dumps(stored)


def test_has_inline_images(orbital_dir):
    assert blob_store.has_inline_images(_image_row())
    assert not blob_store.has_inline_images({"role": "user", "content": "hi"})
    stored = blob_store.deinline_row_images(_image_row(), orbital_dir)
    assert not blob_store.has_inline_images(stored)


def test_blob_write_failure_leaves_the_row_inline(orbital_dir, monkeypatch):
    def boom(*a, **k):
        raise OSError("disk full")
    monkeypatch.setattr(blob_store, "put_bytes", boom)
    row = _image_row()
    assert blob_store.deinline_row_images(row, orbital_dir) is row
