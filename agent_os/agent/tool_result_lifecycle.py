# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Tool result lifecycle management: truncation to stubs + disk backup.

After the LLM consumes tool results and responds, this module replaces
the full results in session history with compact stubs and saves the
full content to disk for later recall via read_file.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


def truncate_consumed_tool_results(
    session,
    assistant_response_text: str | None,
    iteration: int,
) -> None:
    """Replace consumed (non-stubbed) tool results with compact stubs.

    Called after the LLM responds, meaning it has consumed all tool results
    currently in the session. Replaces them with metadata stubs and saves
    full content to disk.

    Args:
        session: The Session instance.
        assistant_response_text: The LLM's response text (used as summary).
        iteration: Current iteration number (used in disk path).
    """
    messages = session.get_messages()
    stubs: dict[str, str] = {}

    for msg in messages:
        if msg.get("role") != "tool":
            continue
        if msg.get("_stubbed"):
            continue

        tool_call_id = msg.get("tool_call_id", "")
        content = msg.get("content", "")

        # Skip multimodal content (list) and very small results
        if isinstance(content, list):
            continue
        if len(content) < 500:
            continue

        # Find tool name and arguments from the preceding assistant message
        tool_name, arguments = _find_tool_info(messages, tool_call_id)
        key_param = _extract_key_param(tool_name, arguments)
        token_count = int(len(content) / 4)

        # Save full content to disk
        disk_path = _export_to_disk(session, msg, tool_name, key_param, iteration)

        # Build compact stub
        summary = (assistant_response_text or "")[:350]
        stub = (
            f"[Tool: {tool_name} | Target: {key_param} | "
            f"Original: {token_count} tokens | Full result: {disk_path}]\n\n"
            f"Agent summary: {summary}"
        )

        stubs[tool_call_id] = stub

    if stubs:
        session.replace_tool_results_with_stubs(stubs)
        logger.info(
            "Truncated %d tool results to stubs (iteration %d)",
            len(stubs), iteration,
        )


def _find_tool_info(
    messages: list[dict], tool_call_id: str
) -> tuple[str, dict]:
    """Scan backward to find the tool name and arguments for a tool_call_id."""
    for msg in reversed(messages):
        if msg.get("role") != "assistant":
            continue
        for tc in msg.get("tool_calls", []):
            tc_id = tc.get("id", "")
            if not tc_id:
                # Nested format
                tc_id = tc.get("function", {}).get("id", "")
            if tc_id == tool_call_id:
                # Handle both flat and nested tool call formats
                if "function" in tc:
                    func = tc["function"]
                    name = func.get("name", "unknown")
                    args = func.get("arguments", {})
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except (json.JSONDecodeError, ValueError):
                            args = {}
                    return name, args
                return tc.get("name", "unknown"), tc.get("arguments", {})
    return "unknown", {}


def _extract_key_param(tool_name: str, arguments: dict) -> str:
    """Extract the most informative parameter for the stub metadata."""
    if tool_name == "browser":
        return str(arguments.get("url", arguments.get("ref", "page")))[:80]
    if tool_name == "shell":
        return str(arguments.get("command", ""))[:50]
    if tool_name in ("read", "write", "edit", "read_file", "write_file"):
        return str(
            arguments.get("path", arguments.get("file_path", ""))
        )[:80]
    # Default: first value
    for v in arguments.values():
        return str(v)[:50]
    return "n/a"


def _export_to_disk(
    session, msg: dict, tool_name: str, key_param: str, iteration: int
) -> str:
    """Save the full tool result content to disk.

    Returns the absolute path to the written file.
    """
    tool_call_id = msg.get("tool_call_id", "unknown")
    content = msg.get("content", "")

    # Derive tool-results directory (sibling to sessions/ under .agent-os/)
    sessions_dir = os.path.dirname(session._filepath)
    parent = os.path.dirname(sessions_dir)
    tool_results_dir = os.path.join(parent, "tool-results", session.session_id)
    os.makedirs(tool_results_dir, exist_ok=True)

    filename = f"turn_{iteration}_call_{tool_call_id}.json"
    disk_path = os.path.join(tool_results_dir, filename)

    record = {
        "turn": iteration,
        "call_id": tool_call_id,
        "tool_name": tool_name,
        "key_param": key_param,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "pre_filter_tokens": int(len(content) / 4),
        "content": content,
    }

    with open(disk_path, "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False)

    return disk_path
