# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Unit tests for the attachment formatter helpers.

Pure helpers — no FastAPI / no app setup. The formatter module is consumed
by the inject route to:

1. validate_attachments(): confirm each declared path resolves inside the
   project workspace, exists, and matches the declared size.
2. format_prefix(): build the "[Attached file: ...]" / "[Attached files: ..."
   block prepended to the user's chat content.
3. _human_size(): render a byte count using binary units (1024-based) since
   the upload cap is 10 * 1024 * 1024.
"""

import os
import pytest

from agent_os.api.routes._attachment_formatter import (
    validate_attachments,
    format_prefix,
    _human_size,
)
from agent_os.api.routes.agents_v2 import InjectAttachment


def _att(path, mime="text/plain", size=None):
    """Build an InjectAttachment, defaulting size to len('hello world')."""
    return InjectAttachment(path=path, mime=mime, size=size if size is not None else 11)


def _write(workspace, rel_path, payload=b"hello world"):
    """Write a file under workspace and return its declared size."""
    full = os.path.join(workspace, rel_path)
    os.makedirs(os.path.dirname(full), exist_ok=True)
    with open(full, "wb") as f:
        f.write(payload)
    return len(payload)


# ---------------------------------------------------------------------------
# validate_attachments
# ---------------------------------------------------------------------------


class TestValidateAttachments:
    def test_valid_single_attachment_passes(self, tmp_path):
        size = _write(str(tmp_path), "uploads/foo.txt")
        validate_attachments(
            str(tmp_path),
            [InjectAttachment(path="uploads/foo.txt", mime="text/plain", size=size)],
        )

    def test_valid_multi_attachment_list_passes(self, tmp_path):
        s1 = _write(str(tmp_path), "uploads/a.txt", b"aaa")
        s2 = _write(str(tmp_path), "uploads/b.txt", b"bbbb")
        s3 = _write(str(tmp_path), "uploads/c.txt", b"ccccc")
        validate_attachments(
            str(tmp_path),
            [
                InjectAttachment(path="uploads/a.txt", mime="text/plain", size=s1),
                InjectAttachment(path="uploads/b.txt", mime="text/plain", size=s2),
                InjectAttachment(path="uploads/c.txt", mime="text/plain", size=s3),
            ],
        )

    def test_symlink_pointing_outside_workspace_rejected(self, tmp_path):
        # Create an outside file and symlink to it from inside the workspace.
        outside = tmp_path.parent / "secret.txt"
        outside.write_bytes(b"top-secret")
        link = tmp_path / "uploads"
        link.mkdir()
        symlink_path = link / "leak.txt"
        os.symlink(str(outside), str(symlink_path))

        with pytest.raises(ValueError) as exc_info:
            validate_attachments(
                str(tmp_path),
                [InjectAttachment(
                    path="uploads/leak.txt", mime="text/plain", size=10,
                )],
            )
        # The realpath check should reject this since the symlink resolves
        # outside the workspace.
        assert "uploads/leak.txt" in str(exc_info.value)

    def test_nonexistent_file_rejected(self, tmp_path):
        with pytest.raises(ValueError) as exc_info:
            validate_attachments(
                str(tmp_path),
                [InjectAttachment(
                    path="uploads/missing.txt", mime="text/plain", size=11,
                )],
            )
        assert "uploads/missing.txt" in str(exc_info.value)
        assert "not found" in str(exc_info.value).lower()

    def test_size_mismatch_rejected(self, tmp_path):
        actual_size = _write(str(tmp_path), "uploads/foo.txt", b"x" * 50)
        # Declared size 100, actual 50 → mismatch.
        with pytest.raises(ValueError) as exc_info:
            validate_attachments(
                str(tmp_path),
                [InjectAttachment(
                    path="uploads/foo.txt", mime="text/plain", size=100,
                )],
            )
        msg = str(exc_info.value)
        assert "100" in msg
        assert str(actual_size) in msg

    def test_empty_list_returns_without_error(self, tmp_path):
        validate_attachments(str(tmp_path), [])

    def test_stops_at_first_failure(self, tmp_path):
        # First attachment is bad, second is good; we don't care about the
        # second since validation should have already raised.
        good_size = _write(str(tmp_path), "uploads/good.txt")
        with pytest.raises(ValueError) as exc_info:
            validate_attachments(
                str(tmp_path),
                [
                    InjectAttachment(
                        path="uploads/missing.txt", mime="text/plain", size=11,
                    ),
                    InjectAttachment(
                        path="uploads/good.txt", mime="text/plain", size=good_size,
                    ),
                ],
            )
        # Error message names the offending path (not the good one).
        assert "missing.txt" in str(exc_info.value)
        assert "good.txt" not in str(exc_info.value)


# ---------------------------------------------------------------------------
# format_prefix
# ---------------------------------------------------------------------------


class TestFormatPrefix:
    def test_single_attachment_one_line(self):
        out = format_prefix([
            InjectAttachment(path="uploads/foo.png", mime="image/png", size=2048),
        ])
        assert out == "[Attached file: uploads/foo.png (image/png, 2.0 KB)]\n\n"

    def test_two_attachments_bulleted(self):
        out = format_prefix([
            InjectAttachment(path="uploads/a.png", mime="image/png", size=1024),
            InjectAttachment(path="uploads/b.txt", mime="text/plain", size=512),
        ])
        # Bulleted block: opens with "[Attached files:", each entry on its own
        # line, closing "]" on the last entry's line, followed by exactly one
        # blank line.
        assert out.startswith("[Attached files:\n")
        assert "- uploads/a.png (image/png, 1.0 KB)" in out
        assert "- uploads/b.txt (text/plain, 512 B)]" in out
        assert out.endswith("\n\n")

    def test_three_attachments_bulleted(self):
        out = format_prefix([
            InjectAttachment(path="a.txt", mime="text/plain", size=100),
            InjectAttachment(path="b.txt", mime="text/plain", size=200),
            InjectAttachment(path="c.txt", mime="text/plain", size=300),
        ])
        assert out.startswith("[Attached files:\n")
        assert "- a.txt (text/plain, 100 B)" in out
        assert "- b.txt (text/plain, 200 B)" in out
        # Last entry carries the closing "]"
        assert "- c.txt (text/plain, 300 B)]" in out
        assert out.endswith("\n\n")

    def test_empty_list_returns_empty_string(self):
        assert format_prefix([]) == ""

    def test_non_empty_output_ends_with_double_newline(self):
        out = format_prefix([
            InjectAttachment(path="x.txt", mime="text/plain", size=1),
        ])
        assert out.endswith("\n\n")
        out_multi = format_prefix([
            InjectAttachment(path="x.txt", mime="text/plain", size=1),
            InjectAttachment(path="y.txt", mime="text/plain", size=2),
        ])
        assert out_multi.endswith("\n\n")


# ---------------------------------------------------------------------------
# _human_size
# ---------------------------------------------------------------------------


class TestHumanSize:
    @pytest.mark.parametrize("value,expected", [
        (0, "0 B"),
        (1, "1 B"),
        (1023, "1023 B"),
        (1024, "1.0 KB"),
        (1536, "1.5 KB"),
        (1048576, "1.0 MB"),
        (10485760, "10.0 MB"),
        (1073741824, "1.0 GB"),
    ])
    def test_human_size(self, value, expected):
        assert _human_size(value) == expected
