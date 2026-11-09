# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Unit tests for ReadTool image handling."""

import os
import struct
import zlib

import pytest

from agent_os.agent.tools.read import ReadTool, _parse_image_dimensions, _human_size


# ---------------------------------------------------------------------------
# Test PNG/JPEG generators (no PIL)
# ---------------------------------------------------------------------------

def _make_test_png(path, width=100, height=100, r=255, g=0, b=0):
    """Write a solid-color PNG. No dependencies."""
    raw = b""
    for _ in range(height):
        raw += b"\x00" + bytes([r, g, b]) * width

    def chunk(ctype, data):
        c = ctype + data
        return struct.pack(">I", len(data)) + c + struct.pack(">I", zlib.crc32(c) & 0xFFFFFFFF)

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    with open(path, "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n")
        f.write(chunk(b"IHDR", ihdr))
        f.write(chunk(b"IDAT", zlib.compress(raw)))
        f.write(chunk(b"IEND", b""))


def _make_minimal_jpeg(path, width=8, height=8):
    """Write a minimal valid JPEG with SOF0 marker encoding dimensions."""
    # Minimal JPEG: SOI + SOF0 (with dimensions) + EOI
    sof0_data = struct.pack(
        ">BHHIBB",
        8,       # precision
        height,  # Y
        width,   # X
        1,       # number of components
        0x11,    # component 1: h/v sampling = 1x1
        0,       # quant table id = 0
    )
    sof0_marker = b"\xFF\xC0" + struct.pack(">H", len(sof0_data) + 2) + sof0_data
    with open(path, "wb") as f:
        f.write(b"\xFF\xD8")  # SOI
        f.write(sof0_marker)
        f.write(b"\xFF\xD9")  # EOI


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def workspace(tmp_path):
    return str(tmp_path)


@pytest.fixture
def tool(workspace):
    return ReadTool(workspace)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestReadPNG:
    def test_read_png_returns_multimodal_content(self, workspace, tool):
        path = os.path.join(workspace, "red.png")
        _make_test_png(path, 100, 100, 255, 0, 0)

        result = tool.execute(path="red.png")

        assert isinstance(result.content, list), "PNG should return list content"
        assert len(result.content) == 2
        assert result.content[0]["type"] == "text"
        assert "red.png" in result.content[0]["text"]
        assert result.content[1]["type"] == "image_url"
        assert result.content[1]["image_url"]["url"].startswith("data:image/png;base64,")
        assert result.content[1]["image_url"]["detail"] == "low"

    def test_png_dimensions_parsed(self, workspace, tool):
        path = os.path.join(workspace, "sized.png")
        _make_test_png(path, 200, 150)

        result = tool.execute(path="sized.png")

        assert result.meta is not None
        assert result.meta["dimensions"] == "200x150"
        assert "200x150" in result.content[0]["text"]


class TestReadJPEG:
    def test_read_jpg_returns_multimodal_content(self, workspace, tool):
        path = os.path.join(workspace, "test.jpg")
        _make_minimal_jpeg(path, 16, 12)

        result = tool.execute(path="test.jpg")

        assert isinstance(result.content, list), "JPEG should return list content"
        assert result.content[0]["type"] == "text"
        assert "test.jpg" in result.content[0]["text"]
        assert result.content[1]["type"] == "image_url"
        assert result.content[1]["image_url"]["url"].startswith("data:image/jpeg;base64,")

    def test_jpeg_dimensions_parsed(self, workspace, tool):
        path = os.path.join(workspace, "sized.jpeg")
        _make_minimal_jpeg(path, 320, 240)

        result = tool.execute(path="sized.jpeg")

        assert result.meta is not None
        assert result.meta["dimensions"] == "320x240"


class TestImageMeta:
    def test_image_meta_includes_path_and_mime(self, workspace, tool):
        path = os.path.join(workspace, "photo.png")
        _make_test_png(path, 10, 10)

        result = tool.execute(path="photo.png")

        assert result.meta is not None
        assert result.meta["image_path"].endswith("photo.png")
        assert result.meta["mime"] == "image/png"
        assert result.meta["size"] > 0


class TestImageSizeLimit:
    def test_read_large_image_returns_error(self, workspace, tool):
        path = os.path.join(workspace, "huge.png")
        # Write >5MB of zeros (not a valid PNG but extension triggers image path)
        with open(path, "wb") as f:
            f.write(b"\x89PNG\r\n\x1a\n")  # PNG header
            f.write(b"\x00" * (6 * 1024 * 1024))  # >5MB

        result = tool.execute(path="huge.png")

        assert isinstance(result.content, str), "Oversized image should return error string"
        assert "too large" in result.content.lower()
        assert "5MB" in result.content


class TestTextFileUnchanged:
    def test_read_text_file_returns_string(self, workspace, tool):
        path = os.path.join(workspace, "hello.txt")
        with open(path, "w") as f:
            f.write("Hello, world!")

        result = tool.execute(path="hello.txt")

        assert isinstance(result.content, str)
        assert result.content == "Hello, world!"


class TestSVG:
    """Spec 165: an SVG is XML. The Messages API (and OpenAI vision) reject
    image/svg+xml, so it is read as text, which is also what an agent editing
    a logo or icon needs."""

    def test_svg_returns_its_markup_as_text(self, workspace, tool):
        path = os.path.join(workspace, "icon.svg")
        svg = '<svg xmlns="http://www.w3.org/2000/svg"><rect width="10" height="10"/></svg>'
        with open(path, "w") as f:
            f.write(svg)

        result = tool.execute(path="icon.svg")

        assert result.content == svg
        assert not (result.meta or {}).get("mime")

    def test_svg_pages_by_line_like_any_text_file(self, workspace, tool):
        path = os.path.join(workspace, "logo.svg")
        lines = ['<svg xmlns="http://www.w3.org/2000/svg">\n']
        lines += [f'  <rect x="{i}" width="1" height="1"/>\n' for i in range(5)]
        lines += ["</svg>\n"]
        with open(path, "w") as f:
            f.writelines(lines)

        result = tool.execute(path="logo.svg", offset=1, limit=2)

        assert isinstance(result.content, str)
        assert result.content.startswith(lines[1] + lines[2])
        assert "offset=3" in result.content

    def test_an_oversized_svg_is_truncated_with_a_continue_hint(self, workspace, tool):
        path = os.path.join(workspace, "big.svg")
        row = '  <path d="M0 0 L10 10 L20 0 Z" fill="#123456"/>\n'
        with open(path, "w") as f:
            f.write("<svg>\n" + row * 5000 + "</svg>\n")

        result = tool.execute(path="big.svg")

        assert isinstance(result.content, str)
        assert "TRUNCATED" in result.content
        assert "offset=" in result.content


class TestUnviewableImageFormats:
    """Spec 165: BMP/ICO/TIFF can't be sent to the model as images (only
    JPEG/PNG/GIF/WebP are accepted). They stay recognised, so they never fall
    to the text path as binary noise, and return a plain hint without the
    file being read at all."""

    @pytest.mark.parametrize("name,fmt", [
        ("favicon.ico", "ICO"), ("scan.bmp", "BMP"), ("photo.tiff", "TIFF"),
        ("UPPER.ICO", "ICO"),
    ])
    def test_returns_a_convert_hint_without_opening_the_file(
            self, workspace, tool, monkeypatch, name, fmt):
        path = os.path.join(workspace, name)
        with open(path, "wb") as f:
            f.write(b"\x00\x01\x02 not really an image")

        def _no_open(*a, **k):
            raise AssertionError("the file must not be opened")
        monkeypatch.setattr("agent_os.agent.tools.read.open", _no_open, raising=False)

        result = tool.execute(path=name)

        assert isinstance(result.content, str)
        assert name in result.content
        assert fmt in result.content
        assert "PNG" in result.content
        assert "image_url" not in result.content
        assert not result.meta


# ---------------------------------------------------------------------------
# Helper function tests
# ---------------------------------------------------------------------------

class TestParseImageDimensions:
    def test_png_dimensions(self, tmp_path):
        path = str(tmp_path / "test.png")
        _make_test_png(path, 42, 17)
        with open(path, "rb") as f:
            data = f.read()
        assert _parse_image_dimensions(data, ".png") == (42, 17)

    def test_jpeg_dimensions(self, tmp_path):
        path = str(tmp_path / "test.jpg")
        _make_minimal_jpeg(path, 64, 48)
        with open(path, "rb") as f:
            data = f.read()
        assert _parse_image_dimensions(data, ".jpg") == (64, 48)

    def test_unknown_extension_returns_none(self):
        assert _parse_image_dimensions(b"garbage", ".bmp") is None

    def test_truncated_png_returns_none(self):
        assert _parse_image_dimensions(b"\x89PNG\r\n\x1a\n" + b"\x00" * 5, ".png") is None


class TestHumanSize:
    def test_bytes(self):
        assert _human_size(500) == "500B"

    def test_kilobytes(self):
        assert _human_size(2048) == "2KB"

    def test_megabytes(self):
        assert _human_size(3 * 1024 * 1024) == "3.0MB"
