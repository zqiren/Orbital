# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Tool result lifecycle: unconditional disk archive + supersession.

Two responsibilities, deliberately decoupled:

1. **Archive.** Every tool result above :data:`SIZE_THRESHOLD` is written to
   ``{workspace}/orbital/tool-results/{session_uuid}/turn_{n}_call_{id}.json``.
   This is observability, not context management: the archive is written
   whether or not session history is ever rewritten, and each ``tool_call_id``
   is archived exactly once.

2. **Supersession.** History is rewritten in exactly one case — a tool result
   whose target was fetched *again* later in the same session. The older copy
   is replaced with a stub; the newest copy is kept intact.

What this module no longer does is blanket stubbing of every consumed tool
result. That fired after every LLM response and destroyed documents the agent
was still reasoning about mid-turn, and because the stub was shaped like a
successful retrieval the model's own history told it "I already looked at
this" — which suppressed the re-read. Context pressure is compaction's job
(``compaction.py``, the token-pressure trigger in ``loop.py``, and the sliding
window in ``context.py``); it is not a reason to mutate tool results in place.

Supersession is not a general eviction policy: no budget, no water marks, no
ordering policy. It fires on a factual condition — this target was re-fetched —
and the prior copy is superseded by definition. For ``browser`` this is
measured: only ~9% of same-URL re-fetches return byte-identical content, so a
prior snapshot describes a page state that no longer exists. Keeping the
*newest* copy (rather than the oldest) also sidesteps any question of whether
the original is still inside the context window: the newest was just appended.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

# Results below this are left completely alone — not archived, not superseded.
SIZE_THRESHOLD = 500

# Tools whose key parameter names a real target, so a later result for the same
# key genuinely supersedes the earlier one. The generic "first argument value"
# fallback used by _extract_key_param is NOT a target identity (a grep pattern,
# for instance, says nothing about which directory was searched), so tools that
# fall through to it never supersede.
#
# Supersession is only justified when the prior copy is provably WRONG, not
# merely older. A browser snapshot is a view of current page state, so the old
# one describes a state that no longer exists and keeping it is a hazard. A
# re-read is the same: 61.6% of same-path re-reads returned byte-identical
# content (superseding an exact duplicate loses nothing), and when the bytes
# differ the file genuinely changed, so the newest is authoritative.
#
# `shell` is deliberately EXCLUDED. Shell output is a measurement at a point in
# time, and comparing two measurements is a legitimate, common workload: `git
# diff` before and after an edit, a test run before and after a fix, `ls`
# before and after a build. Superseding the earlier one deletes exactly what
# the agent was comparing against — structurally the same failure as the
# blanket stubbing this module exists to remove, just narrower. The economics
# confirm it: shell was 531,935 tokens in the production corpus (5.6% of all
# archived tool output, median 547 tokens per result) — negligible upside
# against a real correctness downside. The before/after risk is much weaker for
# `read`, because the `edit` result already records what changed.
_SUPERSEDABLE_TOOLS = frozenset({"browser", "read", "read_file"})


def archive_and_supersede_tool_results(session, iteration: int) -> None:
    """Archive large tool results to disk; stub only superseded ones.

    Every tool result over :data:`SIZE_THRESHOLD` is written to disk (once per
    ``tool_call_id``). Session history is left untouched *except* where a later
    result in the same session targets the same thing — then every earlier copy
    of that target is replaced with a stub and the newest is kept.

    Args:
        session: The Session instance.
        iteration: Current loop iteration (used in the disk path).
    """
    messages = session.get_messages()
    archived = _archived_call_ids(session)

    # (tool, target) -> live copies in message order (oldest first).
    by_target: dict[tuple[str, str], list[dict]] = {}

    for msg in messages:
        if msg.get("role") != "tool":
            continue
        if msg.get("_stubbed"):
            continue

        content = msg.get("content", "")

        # Skip multimodal content (list) and very small results
        if isinstance(content, list):
            continue
        if len(content) < SIZE_THRESHOLD:
            continue

        tool_call_id = msg.get("tool_call_id", "")

        # Find tool name and arguments from the preceding assistant message
        tool_name, arguments = _find_tool_info(messages, tool_call_id)
        key_param = _extract_key_param(tool_name, arguments)

        # Archive unconditionally, but exactly once per call id — the loop
        # calls this after every LLM response, and the old _stubbed flag is no
        # longer a "seen it" marker now that most results are never stubbed.
        disk_path = archived.get(tool_call_id)
        if disk_path is None:
            disk_path = _export_to_disk(
                session, msg, tool_name, key_param, iteration,
            )
            archived[tool_call_id] = disk_path

        target = _supersession_target(tool_name, arguments)
        if target is None:
            continue
        by_target.setdefault(target, []).append({
            "tool_call_id": tool_call_id,
            "tool_name": tool_name,
            "key_param": key_param,
            "tokens": int(len(content) / 4),
            "disk_path": disk_path,
        })

    stubs: dict[str, str] = {}
    for entries in by_target.values():
        if len(entries) < 2:
            continue
        # Keep the newest copy; every earlier copy of this target is superseded.
        for entry in entries[:-1]:
            stubs[entry["tool_call_id"]] = _superseded_stub(
                entry["tool_name"],
                entry["key_param"],
                entry["tokens"],
                entry["disk_path"],
            )

    if stubs:
        session.replace_tool_results_with_stubs(stubs)
        logger.info(
            "Superseded %d tool results by a later fetch of the same target "
            "(iteration %d)",
            len(stubs), iteration,
        )


def _superseded_stub(
    tool_name: str, key_param: str, tokens: int, disk_path: str,
) -> str:
    """Build the stub for a result a later fetch of the same target replaced.

    The body must lead with the ABSENCE and must never be the model's own
    narration dressed up as a summary of the content. A narration-as-summary
    stub caused the model to reconstruct exact ``old_text`` for an ``edit``
    from a non-authoritative paraphrase, so exact-match edits failed.
    """
    return (
        "[SUPERSEDED — the content of this tool result is GONE from the "
        "conversation. This is a placeholder, NOT the content.]\n"
        "Do not quote it, summarize it, or treat it as evidence that you have "
        "already seen this target.\n"
        f"A newer {tool_name} result for the same target ({key_param}) appears "
        "later in this conversation — use that one. It replaced this copy, "
        "which may describe a state that no longer exists.\n"
        "If you need this exact older copy, re-read the target or open the "
        "disk path below.\n"
        f"[Tool: {tool_name} | Target: {key_param} | "
        f"Original: {tokens} tokens | Full result: {disk_path}]"
    )


def _supersession_target(tool_name: str, arguments: dict) -> tuple[str, str] | None:
    """Identity of the thing a tool result describes, or None if it has none.

    Deliberately uses the UNTRUNCATED argument value: _extract_key_param clips
    to 50/80 chars for display, and two long URLs can agree on their first 80
    chars while pointing at different pages. Supersession must never fire on a
    display-string collision.
    """
    if tool_name not in _SUPERSEDABLE_TOOLS:
        return None
    if tool_name == "browser":
        target = arguments.get("url") or arguments.get("ref")
    elif tool_name == "shell":
        target = arguments.get("command")
    else:  # read / read_file
        target = arguments.get("path") or arguments.get("file_path")
        if target:
            # A paged read targets a RANGE, not a whole file. Keying on the
            # path alone would make page 2 supersede page 1, so an agent
            # walking a large file with offset/limit would watch each page it
            # had already collected get stubbed out from under it — which is
            # precisely what pagination exists to make possible.
            #
            # Normalization must match ReadTool._window (read.py): a junk or
            # negative offset behaves as 0, and a junk or non-positive limit
            # means "to EOF". Otherwise offset=0 and offset="0" would look
            # like different ranges and never supersede each other.
            target = f"{target}#L{_norm_offset(arguments)}+{_norm_limit(arguments)}"
    if not target:
        return None
    return (tool_name, str(target))


def _coerce_int(value) -> int | None:
    """Mirror of ReadTool._coerce_int — models send "3" as often as 3."""
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _norm_offset(arguments: dict) -> int:
    """Start line a read actually used (see ReadTool._window)."""
    return max(0, _coerce_int(arguments.get("offset")) or 0)


def _norm_limit(arguments: dict) -> str:
    """Line budget a read actually used, or ``eof`` when unbounded."""
    limit = _coerce_int(arguments.get("limit"))
    return str(limit) if limit is not None and limit > 0 else "eof"


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


def _tool_results_dir(session) -> str:
    """Absolute path of this session's tool-results directory (not created)."""
    # tool-results lives as a sibling of sessions under orbital/. Derived from
    # session._filepath rather than ProjectPaths to keep this module decoupled,
    # but it relies on ProjectPaths.tool_results_dir == dirname(sessions_dir)/tool-results.
    sessions_dir = os.path.dirname(session._filepath)
    parent = os.path.dirname(sessions_dir)
    # tool-results is keyed by the Format-2 stem (filename), not the F1 chat id.
    return os.path.join(parent, "tool-results", session.session_uuid)


def _archived_call_ids(session) -> dict[str, str]:
    """Map already-archived tool_call_id -> its file path (one listdir).

    Archiving is idempotent per call id: the loop calls the entry point after
    every LLM response, and without this the same result would be rewritten
    once per iteration under a new turn_N name.
    """
    directory = _tool_results_dir(session)
    out: dict[str, str] = {}
    try:
        names = os.listdir(directory)
    except OSError:
        return out
    marker = "_call_"
    for name in names:
        if not name.endswith(".json"):
            continue
        idx = name.find(marker)
        if idx == -1:
            continue
        call_id = name[idx + len(marker):-len(".json")]
        out.setdefault(call_id, os.path.join(directory, name))
    return out


def _export_to_disk(
    session, msg: dict, tool_name: str, key_param: str, iteration: int
) -> str:
    """Save the full tool result content to disk.

    Returns the absolute path to the written file.
    """
    tool_call_id = msg.get("tool_call_id", "unknown")
    content = msg.get("content", "")

    tool_results_dir = _tool_results_dir(session)
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
