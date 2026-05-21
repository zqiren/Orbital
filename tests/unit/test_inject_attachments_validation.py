# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Pydantic validation tests for InjectAttachment and the extended InjectRequest.

These tests exercise the model layer only — no FastAPI app, no TestClient.
"""

import pytest
from pydantic import ValidationError

from agent_os.api.routes.agents_v2 import InjectAttachment, InjectRequest


# ---------------------------------------------------------------------------
# InjectAttachment
# ---------------------------------------------------------------------------


class TestInjectAttachment:
    def test_valid_input_accepted(self):
        a = InjectAttachment(path="uploads/foo.png", mime="image/png", size=1024)
        assert a.path == "uploads/foo.png"
        assert a.mime == "image/png"
        assert a.size == 1024

    def test_path_empty_rejected(self):
        with pytest.raises(ValidationError):
            InjectAttachment(path="", mime="image/png", size=1)

    def test_path_leading_slash_rejected(self):
        with pytest.raises(ValidationError):
            InjectAttachment(path="/etc/passwd", mime="text/plain", size=1)

    def test_path_leading_backslash_rejected(self):
        with pytest.raises(ValidationError):
            InjectAttachment(path="\\Windows\\System32", mime="text/plain", size=1)

    @pytest.mark.parametrize("bad_path", [
        "../foo",
        "foo/../bar",
        "foo/..",
        "..",
    ])
    def test_path_with_dotdot_rejected(self, bad_path):
        with pytest.raises(ValidationError):
            InjectAttachment(path=bad_path, mime="text/plain", size=1)

    def test_path_too_long_rejected(self):
        long_path = "a" * 1025
        with pytest.raises(ValidationError):
            InjectAttachment(path=long_path, mime="text/plain", size=1)

    def test_path_with_nul_byte_rejected(self):
        with pytest.raises(ValidationError):
            InjectAttachment(path="foo\x00bar", mime="text/plain", size=1)

    def test_path_with_backslash_separators_normalizes(self):
        # Backslashes are normalized to '/' for the purposes of segment
        # checks, so a Windows-style relative path with no '..' segments is
        # accepted.
        a = InjectAttachment(
            path="uploads\\foo.png", mime="image/png", size=1,
        )
        assert "foo.png" in a.path

    def test_mime_missing_slash_rejected(self):
        with pytest.raises(ValidationError):
            InjectAttachment(path="uploads/foo.png", mime="imagepng", size=1)

    @pytest.mark.parametrize("good_mime", [
        "application/vnd.api+json",
        "image/svg+xml",
        "application/x-www-form-urlencoded",
    ])
    def test_mime_with_valid_special_chars_accepted(self, good_mime):
        a = InjectAttachment(path="uploads/foo", mime=good_mime, size=1)
        assert a.mime == good_mime

    def test_size_negative_rejected(self):
        with pytest.raises(ValidationError):
            InjectAttachment(path="uploads/foo.png", mime="image/png", size=-1)

    def test_size_over_10mb_rejected(self):
        with pytest.raises(ValidationError):
            InjectAttachment(
                path="uploads/foo.png", mime="image/png",
                size=10 * 1024 * 1024 + 1,
            )

    def test_size_exactly_10mb_accepted(self):
        a = InjectAttachment(
            path="uploads/foo.png", mime="image/png",
            size=10 * 1024 * 1024,
        )
        assert a.size == 10 * 1024 * 1024


# ---------------------------------------------------------------------------
# InjectRequest extension
# ---------------------------------------------------------------------------


class TestInjectRequestAttachments:
    def test_attachments_defaults_to_none(self):
        req = InjectRequest(content="hi")
        assert req.attachments is None

    def test_attachments_none_accepted(self):
        req = InjectRequest(content="hi", attachments=None)
        assert req.attachments is None

    def test_attachments_empty_list_accepted(self):
        req = InjectRequest(content="hi", attachments=[])
        assert req.attachments == []

    def test_attachments_with_10_items_accepted(self):
        items = [
            {"path": f"uploads/f{i}.txt", "mime": "text/plain", "size": 1}
            for i in range(10)
        ]
        req = InjectRequest(content="hi", attachments=items)
        assert len(req.attachments) == 10

    def test_attachments_with_11_items_rejected(self):
        items = [
            {"path": f"uploads/f{i}.txt", "mime": "text/plain", "size": 1}
            for i in range(11)
        ]
        with pytest.raises(ValidationError):
            InjectRequest(content="hi", attachments=items)

    def test_existing_fields_still_work(self):
        req = InjectRequest(
            content="hi", target="@sub", nonce="abc",
            attachments=[
                {"path": "uploads/f.txt", "mime": "text/plain", "size": 1},
            ],
        )
        assert req.content == "hi"
        assert req.target == "@sub"
        assert req.nonce == "abc"
        assert len(req.attachments) == 1
        assert req.attachments[0].path == "uploads/f.txt"

    def test_existing_ufffd_validator_still_fires(self):
        # The pre-existing validator on `content` must continue to fire
        # regardless of the new attachments field.
        with pytest.raises(ValidationError):
            InjectRequest(content="hello � world")
        with pytest.raises(ValidationError):
            InjectRequest(
                content="hello � world",
                attachments=[
                    {"path": "uploads/f.txt", "mime": "text/plain", "size": 1},
                ],
            )

    def test_attachments_field_holds_inject_attachment_instances(self):
        req = InjectRequest(
            content="hi",
            attachments=[
                {"path": "uploads/f.txt", "mime": "text/plain", "size": 1},
            ],
        )
        assert isinstance(req.attachments[0], InjectAttachment)
