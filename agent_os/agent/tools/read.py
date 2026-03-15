# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""ReadTool — read file or directory listing within workspace."""

import base64
import os
import struct

from .base import Tool, ToolResult

_MAX_CHARS = 100_000
_MAX_IMAGE_BYTES = 5 * 1024 * 1024  # 5MB

_IMAGE_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".gif", ".webp",
    ".bmp", ".ico", ".tiff", ".svg",
}

_MIME_MAP = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
    ".bmp": "image/bmp",
    ".ico": "image/x-icon",
    ".tiff": "image/tiff",
    ".svg": "image/svg+xml",
}


class ReadTool(Tool):
    """Read a file or list a directory within the workspace."""

    def __init__(self, workspace: str):
        self._workspace = os.path.realpath(workspace)
        self.name = "read"
        self.description = "Read a file or list a directory within the workspace."
        self.parameters = {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path relative to workspace"},
            },
            "required": ["path"],
        }

    def _resolve_safe(self, path: str) -> str | None:
        """Resolve path relative to workspace. Returns None if outside workspace."""
        # Strip leading '/' so os.path.join treats it as relative, not absolute.
        # On POSIX, os.path.join("/workspace", "/sub/path") ignores the first arg.
        path = path.lstrip("/")
        resolved = os.path.realpath(os.path.join(self._workspace, path))
        # Ensure the resolved path is within the workspace
        if not resolved.startswith(self._workspace):
            return None
        return resolved

    def execute(self, **arguments) -> ToolResult:
        try:
            path = arguments.get("path", ".")
            resolved = self._resolve_safe(path)
            if resolved is None:
                return ToolResult(content=f"Error: path outside workspace: {path}")

            if os.path.isdir(resolved):
                return self._list_directory(resolved)
            elif os.path.isfile(resolved):
                return self._read_file(resolved)
            else:
                return ToolResult(content=f"Error: file not found: {path}")
        except Exception as e:
            return ToolResult(content=f"Error: {str(e)}")

    def _list_directory(self, dirpath: str) -> ToolResult:
        """List directory contents."""
        try:
            entries = []
            for entry in sorted(os.listdir(dirpath)):
                full = os.path.join(dirpath, entry)
                if os.path.isdir(full):
                    entries.append(f"  {entry}/")
                else:
                    try:
                        size = os.path.getsize(full)
                        entries.append(f"  {entry}  ({size} bytes)")
                    except OSError:
                        entries.append(f"  {entry}")
            return ToolResult(content="\n".join(entries) if entries else "(empty directory)")
        except Exception as e:
            return ToolResult(content=f"Error: {str(e)}")

    def _read_file(self, filepath: str) -> ToolResult:
        """Read file contents. Images are returned as multimodal content blocks."""
        ext = os.path.splitext(filepath)[1].lower()

        if ext in _IMAGE_EXTENSIONS:
            return self._read_image(filepath, ext)

        try:
            with open(filepath, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()
            if len(content) > _MAX_CHARS:
                total = len(content)
                content = content[:_MAX_CHARS] + (
                    f"\n[TRUNCATED — file is {total} chars, showing first 100,000]"
                )
            return ToolResult(content=content)
        except PermissionError:
            return ToolResult(content="Error: permission denied")
        except Exception as e:
            return ToolResult(content=f"Error: {str(e)}")

    def _read_image(self, filepath: str, ext: str) -> ToolResult:
        """Read an image file and return multimodal content blocks."""
        try:
            size = os.path.getsize(filepath)
            if size > _MAX_IMAGE_BYTES:
                mb = size / (1024 * 1024)
                return ToolResult(
                    content=f"Image too large ({mb:.1f} MB). Maximum supported size is 5MB."
                )

            with open(filepath, "rb") as f:
                raw = f.read()

            mime = _MIME_MAP.get(ext, "application/octet-stream")
            b64 = base64.b64encode(raw).decode("ascii")
            dims = _parse_image_dimensions(raw, ext)
            filename = os.path.basename(filepath)

            # Build text metadata
            parts = [filename]
            if dims:
                parts.append(f"{dims[0]}x{dims[1]}")
            parts.append(f"{_human_size(size)}")
            parts.append(mime)
            text_desc = f"Image file: {', '.join(parts)}"

            content = [
                {"type": "text", "text": text_desc},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{mime};base64,{b64}",
                        "detail": "low",
                    },
                },
            ]

            meta = {
                "image_path": filepath,
                "mime": mime,
                "size": size,
            }
            if dims:
                meta["dimensions"] = f"{dims[0]}x{dims[1]}"

            return ToolResult(content=content, meta=meta)

        except PermissionError:
            return ToolResult(content="Error: permission denied")
        except Exception as e:
            return ToolResult(content=f"Error reading image: {str(e)}")


def _parse_image_dimensions(data: bytes, ext: str) -> tuple[int, int] | None:
    """Parse image dimensions from raw bytes. No PIL dependency."""
    try:
        if ext == ".png" and len(data) >= 24:
            # PNG IHDR: bytes 16-23 are width (4 bytes) and height (4 bytes)
            if data[:8] == b"\x89PNG\r\n\x1a\n":
                w, h = struct.unpack(">II", data[16:24])
                return (w, h)

        if ext in (".jpg", ".jpeg") and len(data) >= 2:
            # Scan for SOF0/SOF2 markers (0xFFC0, 0xFFC2)
            i = 0
            while i < len(data) - 1:
                if data[i] == 0xFF:
                    marker = data[i + 1]
                    if marker in (0xC0, 0xC2) and i + 9 < len(data):
                        h, w = struct.unpack(">HH", data[i + 5 : i + 9])
                        return (w, h)
                    if marker == 0xD8 or marker == 0xD9:
                        # SOI or EOI — skip
                        i += 2
                    elif marker == 0x00:
                        # Stuffed byte
                        i += 2
                    elif 0xD0 <= marker <= 0xD7:
                        # RST markers
                        i += 2
                    else:
                        # Variable-length segment
                        if i + 3 < len(data):
                            seg_len = struct.unpack(">H", data[i + 2 : i + 4])[0]
                            i += 2 + seg_len
                        else:
                            break
                else:
                    i += 1
    except Exception:
        pass
    return None


def _human_size(nbytes: int) -> str:
    """Format byte count as human-readable string."""
    if nbytes < 1024:
        return f"{nbytes}B"
    if nbytes < 1024 * 1024:
        return f"{nbytes / 1024:.0f}KB"
    return f"{nbytes / (1024 * 1024):.1f}MB"
