# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Live tool-result events carry a capped preview of the result.

Before this, the live ``tool_result`` activity carried only the placeholder
"Tool result received", so every tool row expanded mid-turn showed
"no result content" until the session was reloaded. The event now carries
``result_preview`` — exactly the part the chat capsule displays (the
frontend's ``truncateResult`` bounds: 500 chars / 12 lines) — plus the full
result's totals when it was cut, so the capsule footer stays accurate. The
fields are additive: older phone UIs served by the relay ignore them.

The expected previews live in ``web/src/utils/toolResultPreviewFixtures.json``,
shared with ``chatTransform.test.ts``, which asserts that rendering a preview
with its totals is identical to rendering the full result after a reload.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from agent_os.daemon_v2.activity_translator import ActivityTranslator

FIXTURES = json.loads(
    (
        Path(__file__).resolve().parents[2]
        / "web" / "src" / "utils" / "toolResultPreviewFixtures.json"
    ).read_text(encoding="utf-8")
)


def _emit(message: dict) -> dict:
    ws = MagicMock()
    ActivityTranslator(ws).on_message(message, "p1", session_id="s1")
    assert ws.broadcast.call_count == 1
    project_id, event = ws.broadcast.call_args.args
    assert project_id == "p1"
    return event


def _tool_row(content, tool_call_id="call_1") -> dict:
    return {
        "role": "tool",
        "content": content,
        "tool_call_id": tool_call_id,
        "source": "management",
    }


@pytest.mark.parametrize("row", FIXTURES, ids=[r["name"] for r in FIXTURES])
def test_preview_matches_shared_fixture(row):
    event = _emit(_tool_row(row["full"]))
    assert event["result_preview"] == row["preview"]
    if "total_chars" in row:
        assert event["result_total_chars"] == row["total_chars"]
        assert event["result_total_lines"] == row["total_lines"]
    else:
        assert "result_total_chars" not in event
        assert "result_total_lines" not in event


def test_tool_result_event_keeps_old_fields_and_adds_tool_call_id():
    event = _emit(_tool_row("ok", tool_call_id="call_42"))
    # Old frontends key off these; they must not change.
    assert event["type"] == "agent.activity"
    assert event["category"] == "tool_result"
    assert event["description"] == "Tool result received"
    assert event["tool_name"] == "call_42"
    assert event["session_id"] == "s1"
    # New, additive.
    assert event["tool_call_id"] == "call_42"
    assert event["result_preview"] == "ok"


def test_large_result_is_capped_on_the_wire():
    big = "\n".join(f"{i:06d} " + "z" * 90 for i in range(5000))  # ~485 KB
    event = _emit(_tool_row(big))
    assert event["result_preview"] == big[:500]
    assert event["result_total_chars"] == len(big)
    assert event["result_total_lines"] == 5000
    assert len(json.dumps(event).encode("utf-8")) < 2048


def test_non_text_result_sends_empty_preview():
    # Multimodal tool results (lists of content blocks) render as "no result
    # content" after a reload; the live event mirrors that.
    content = [
        {"type": "text", "text": "screenshot taken"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
    ]
    event = _emit(_tool_row(content))
    assert event["result_preview"] == ""
    assert "result_total_chars" not in event
    assert "base64" not in json.dumps(event)


def test_missing_content_sends_empty_preview():
    event = _emit({"role": "tool", "tool_call_id": "call_1", "source": "management"})
    assert event["result_preview"] == ""


def test_tool_use_event_carries_the_tool_call_id():
    ws = MagicMock()
    ActivityTranslator(ws).on_message({
        "role": "assistant",
        "content": None,
        "source": "management",
        "tool_calls": [
            {"id": "call_a", "type": "function",
             "function": {"name": "read", "arguments": json.dumps({"path": "a.md"})}},
            {"id": "call_b", "type": "function",
             "function": {"name": "agent_message", "arguments": json.dumps(
                 {"action": "send", "agent": "claude-code", "message": "full brief"})}},
        ],
    }, "p1", session_id="s1")
    events = [c.args[1] for c in ws.broadcast.call_args_list]
    assert [e["tool_call_id"] for e in events] == ["call_a", "call_b"]
    # The dispatch message already rides on the tool-use event's arguments.
    assert events[1]["arguments"]["message"] == "full brief"
    assert events[1]["arguments"]["agent"] == "claude-code"
