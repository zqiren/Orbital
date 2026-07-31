# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""ReadTool — read file or directory listing within workspace."""

import base64
import logging
import os
import struct
from typing import Callable

from ._path_utils import resolve_safe_read
from .base import Tool, ToolResult

logger = logging.getLogger(__name__)

# Size cap is derived from the ACTIVE MODEL's window, not a constant: 100_000
# chars is 2.5% of a 1M-token window but 19.5% of a 128k one — the same number
# means wildly different things. Mirrors ``memory_entries.budgets_for_window``:
# floor first, window only ever scales *down*, ``int()`` truncation, warn when
# the window is unknown.
#
# This must sit BELOW ``Session._cap_tool_result`` (30% of window, 400_000
# hard max), which caps every tool result again at append time. 10% vs 30%
# keeps them from fighting at every window size, and the notice is counted
# INSIDE the cap (see ``_NOTICE_RESERVE``) so the session cap never lops the
# "use offset=N to continue" hint off the end at the 400_000 ceiling.
_WINDOW_SHARE = 0.10
_CHARS_PER_TOKEN = 4
_MIN_CAP_CHARS = 20_000
_MAX_CAP_CHARS = 400_000
# Fallback for callers that don't know the window — today's fixed behavior.
_DEFAULT_MAX_CHARS = 100_000
# Room inside the cap for the truncation notice, so body + notice <= cap.
_NOTICE_RESERVE = 240

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

    def __init__(self, workspace: str,
                 read_roots: Callable[[], list[str]] | None = None,
                 context_window: int | None = None):
        self._workspace = os.path.realpath(workspace)
        self._max_chars = _cap_for_window(context_window)
        # ``read_roots`` is a CALLABLE (not a snapshot) so a per-session scope
        # change applies on the next tool call without rebuilding the tool.
        # ``None`` → single-root [workspace], byte-identical to the old path.
        self._read_roots = read_roots
        self.name = "read"
        if read_roots is None:
            self.description = "Read a file or list a directory within the workspace."
            path_desc = "Path within your workspace. Use a relative path like 'src/main.py' or 'docs/notes.md'. Do NOT start with '/' and do NOT pass an absolute path."
        else:
            self.description = (
                "Read a file or list a directory. Defaults to your own workspace; "
                "this session can ALSO read files in other in-scope project "
                "workspaces (read-only) by absolute path."
            )
            path_desc = (
                "Relative path (e.g. 'src/main.py') reads from YOUR workspace. To "
                "read a file in another in-scope project, pass its ABSOLUTE path. "
                "Other projects' orbital/ and .git/ internals are not readable."
            )
        # NOTE: the pagination contract lives in the offset/limit parameter
        # descriptions rather than in ``self.description``. Both go to the
        # model in the same function schema (see ``Tool.schema``), and the
        # single-root ``self.description`` string is asserted byte-exactly by
        # tests/unit/test_scope_aware_tool_descriptions.py.
        self.parameters = {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": path_desc},
                "offset": {
                    "type": "integer",
                    "description": (
                        "Optional. 0-indexed line number to start reading at. "
                        "Large files are cut off at a size cap; when that "
                        "happens the result ends with a '[TRUNCATED ...]' "
                        "notice naming the offset to resume from — call read "
                        "again with that offset to page through the rest. "
                        "Ignored for directories and images."
                    ),
                },
                "limit": {
                    "type": "integer",
                    "description": (
                        "Optional. Maximum number of lines to return, starting "
                        "at 'offset'. Omit to read to the end of the file (or "
                        "to the size cap, whichever comes first). Ignored for "
                        "directories and images."
                    ),
                },
            },
            "required": ["path"],
        }

    def execute(self, **arguments) -> ToolResult:
        try:
            path = arguments.get("path", ".")
            offset = _coerce_int(arguments.get("offset"))
            limit = _coerce_int(arguments.get("limit"))
            roots = self._read_roots() if self._read_roots else [self._workspace]
            if not roots:
                roots = [self._workspace]
            resolved = resolve_safe_read(roots, path)
            if resolved is None:
                return ToolResult(content=f"Error: path outside workspace: {path}")

            if os.path.isdir(resolved):
                return self._list_directory(resolved)
            elif os.path.isfile(resolved):
                return self._read_file(resolved, offset=offset, limit=limit)
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

    def _read_file(self, filepath: str, offset: int | None = None,
                   limit: int | None = None) -> ToolResult:
        """Read file contents. Images are returned as multimodal content blocks."""
        ext = os.path.splitext(filepath)[1].lower()

        if ext in _IMAGE_EXTENSIONS:
            # Images are whole-or-nothing — offset/limit are line concepts.
            return self._read_image(filepath, ext)

        try:
            with open(filepath, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()
            return ToolResult(content=self._window(content, offset, limit))
        except PermissionError:
            return ToolResult(content="Error: permission denied")
        except Exception as e:
            return ToolResult(content=f"Error: {str(e)}")

    def _window(self, content: str, offset: int | None,
                limit: int | None) -> str:
        """Slice ``content`` to the requested lines, bounded by the size cap.

        Whole lines only — there is deliberately no per-line character cap
        (the agent's own memory files carry single lines over 2,000 chars).
        The one exception is a first line that alone exceeds the budget: cut
        it rather than return nothing.
        """
        cap = self._max_chars
        lines = content.splitlines(keepends=True)
        total_lines = len(lines)

        start = max(0, offset or 0)
        if start >= total_lines and (start > 0 or total_lines == 0):
            if start > 0:
                return (
                    f"[offset {start} is past end of file — "
                    f"file has {total_lines} lines]"
                )
            return content  # empty file, offset 0

        end_limit = total_lines
        if limit is not None and limit > 0:
            end_limit = min(total_lines, start + limit)

        # Fast path: the whole file was asked for and it fits — byte-identical
        # to the pre-pagination behavior.
        if start == 0 and end_limit >= total_lines and len(content) <= cap:
            return content

        budget = max(_NOTICE_RESERVE, cap - _NOTICE_RESERVE)
        parts: list[str] = []
        used = 0
        end = start
        cut_line_len = 0
        for i in range(start, end_limit):
            line = lines[i]
            if used + len(line) <= budget:
                parts.append(line)
                used += len(line)
                end = i + 1
                continue
            if end == start:
                # A single line bigger than the whole budget. Returning an
                # empty body would make the file permanently unreadable, so
                # cut this one line and say so.
                parts.append(line[:budget])
                cut_line_len = len(line)
                used = budget
                end = i + 1
            break

        body = "".join(parts)
        if end >= total_lines and not cut_line_len:
            return body  # reached EOF cleanly — nothing left to page to

        if cut_line_len:
            notice = (
                f"\n[TRUNCATED — line {start} is {cut_line_len} chars, over the "
                f"{budget}-char budget; showing its first {used} chars. "
                f"Use offset={end} to continue at the next line.]"
            )
        elif end >= end_limit:
            # The caller's own `limit` ended this read, not the size cap.
            # Calling that "TRUNCATED" tells the agent something went wrong
            # when it got exactly what it asked for — the same class of
            # misleading signal this whole change exists to remove.
            notice = (
                f"\n[Showing lines {start}-{end} of {total_lines} as requested "
                f"({used} chars). Use offset={end} to continue.]"
            )
        else:
            notice = (
                f"\n[TRUNCATED — showing lines {start}-{end} of {total_lines} "
                f"({used} chars), cut off at the {budget}-char size cap. "
                f"Use offset={end} to continue.]"
            )
        return body + notice

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


def _cap_for_window(context_window: int | None) -> int:
    """Max chars a single read may return, derived from the model's window.

    ``10% of window * ~4 chars/token``, floored at 20_000 and ceilinged at
    400_000. A missing/zero window falls back to the historical fixed
    100_000 with a warning (same shape as ``budgets_for_window``).
    """
    if not context_window or context_window <= 0:
        logger.warning(
            "read tool: no context_window for active model; using %d-char cap",
            _DEFAULT_MAX_CHARS,
        )
        return _DEFAULT_MAX_CHARS
    cap = int(context_window * _WINDOW_SHARE * _CHARS_PER_TOKEN)
    cap = min(cap, _MAX_CAP_CHARS)
    return max(cap, _MIN_CAP_CHARS)


def _coerce_int(value) -> int | None:
    """Best-effort int from a model-supplied argument. None on anything else."""
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


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
