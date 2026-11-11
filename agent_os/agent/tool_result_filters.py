# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Pre-filter tool results before storage in session history.

Deterministic, fast, no LLM calls. Reduces token bloat from raw tool output
by applying tool-type-specific extractors.
"""

from __future__ import annotations

import json
import logging

logger = logging.getLogger(__name__)

# Tools whose results are structured API responses, safe to re-serialize.
# Everything else (``read``, ``grep``, ``glob``, worker replies…) can carry
# file bytes the model later quotes to ``edit``, which matches exactly, so it
# must reach the model as-is. An allowlist on purpose: a tool missing here
# costs a few tokens; a file tool wrongly compacted breaks edits.
_JSON_API_TOOLS = frozenset({
    "browser",
    "calendar_read",
    "list_sessions",
    "read_session",
    "create_trigger",
    "list_triggers",
    "update_trigger",
    "delete_trigger",
})


def _is_json_api_tool(tool_name: str) -> bool:
    # Connector shims are named "<connector_id>.<tool>" (connectors/reflection.py);
    # no built-in tool name has a dot.
    return tool_name in _JSON_API_TOOLS or "." in tool_name


def dispatch_prefilter(tool_name: str, arguments: dict, content: str | list) -> str | list:
    """Apply the appropriate pre-filter based on tool type.

    Returns the filtered content, or the original content unchanged if
    no filter applies or the filter fails. Multimodal (list) content
    passes through unfiltered.
    """
    if not content:
        return content
    if isinstance(content, list):
        return content  # multimodal blocks pass through unfiltered

    # Browser fetch action → HTML extraction
    if tool_name == "browser" and arguments.get("action") == "fetch":
        return _prefilter_html(content)

    # Shell output → tail-truncate
    if tool_name == "shell":
        return _prefilter_shell(content)

    # JSON API responses → compact whitespace (API-style tools only)
    if _is_json_api_tool(tool_name):
        stripped = content.lstrip()
        if stripped and stripped[0] in ('{', '['):
            return _prefilter_json(content)

    return content


def _prefilter_html(content: str) -> str:
    """Extract article text from HTML content using trafilatura.

    If content is not HTML or extraction fails, returns content unchanged.
    """
    try:
        import trafilatura
        extracted = trafilatura.extract(content)
        if extracted:
            return extracted
    except Exception:
        logger.debug("trafilatura extraction failed, returning content as-is")
    return content


def _prefilter_json(content: str) -> str:
    """Compact pretty-printed JSON by removing whitespace."""
    try:
        parsed = json.loads(content)
        return json.dumps(parsed, separators=(',', ':'), ensure_ascii=False)
    except (json.JSONDecodeError, ValueError, TypeError):
        return content


def _prefilter_shell(content: str, max_lines: int = 200) -> str:
    """Tail-truncate shell output if it exceeds max_lines.

    Note: shell.py already caps at 200 lines / 50K chars. This is a safety net.
    """
    lines = content.split('\n')
    if len(lines) <= max_lines:
        return content
    omitted = len(lines) - max_lines
    return f"[{omitted} earlier lines omitted]\n" + '\n'.join(lines[-max_lines:])
