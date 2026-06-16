# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Regression: truncation stub must be honest, not a faux content summary.

Bug: when a tool result was evicted from context, truncate_consumed_tool_results
replaced it with a stub whose body was the first ~350 chars of the MODEL'S OWN
narration — which reads like a summary of the file. The model then built exact
`old_text` for an `edit` from that "summary" and the exact-match edit failed,
because the stub is not the file.

Fix: the stub states plainly that the content was moved to disk, is NOT the
content, and must be re-read before being relied on. Truncation threshold and
timing are unchanged (this is an honesty change, not a token-budget change).
"""

import json

import pytest

from agent_os.agent.session import Session
from agent_os.agent.tool_result_lifecycle import truncate_consumed_tool_results


@pytest.fixture
def session(tmp_path):
    return Session.new("honest-stub-test", str(tmp_path))


def _add_tool_call_and_result(session, call_id, tool_name, arguments, content):
    session.append({
        "role": "assistant",
        "tool_calls": [{
            "id": call_id,
            "type": "function",
            "function": {"name": tool_name, "arguments": json.dumps(arguments)},
        }],
        "source": "management",
    })
    session.append_tool_result(call_id, content)


def _only_stub(session):
    return [m for m in session.get_messages() if m.get("role") == "tool"][0]["content"]


def test_stub_does_not_embed_model_narration(session):
    """The model's narration must NOT appear in the stub as a faux summary."""
    narration = "I read the file; the header is exactly '## Section 4: Demo (wiki)'."
    _add_tool_call_and_result(
        session, "tc_n", "read", {"path": "shot-list.md"}, "Z" * 4000,
    )
    truncate_consumed_tool_results(session, narration, iteration=1)
    stub = _only_stub(session)

    assert "Agent summary:" not in stub
    assert narration not in stub
    assert narration[:40] not in stub


def test_stub_states_not_content_and_instructs_reread(session):
    """The stub must say it is not the content and must instruct a re-read."""
    _add_tool_call_and_result(
        session, "tc_h", "read", {"path": "shot-list.md"}, "Y" * 4000,
    )
    truncate_consumed_tool_results(session, "Some narration.", iteration=1)
    stub = _only_stub(session)

    low = stub.lower()
    assert "not the content" in low
    assert "re-read" in low


def test_stub_preserves_metadata_header(session):
    """Useful metadata (tool, target, size, disk path) is still present."""
    _add_tool_call_and_result(
        session, "tc_m", "read", {"path": "content-bank/index.md"}, "X" * 4000,
    )
    truncate_consumed_tool_results(session, "narration", iteration=3)
    stub = _only_stub(session)

    assert stub.startswith("[Tool: read")
    assert "Target: content-bank/index.md" in stub
    assert "Original: 1000 tokens" in stub
    assert "Full result:" in stub
    assert ".json" in stub


def test_threshold_and_timing_unchanged(session):
    """Small (<500 char) results are still NOT stubbed; large ones are."""
    _add_tool_call_and_result(
        session, "tc_small", "shell", {"command": "echo hi"}, "small output",
    )
    _add_tool_call_and_result(
        session, "tc_big", "read", {"path": "big.md"}, "B" * 600,
    )
    truncate_consumed_tool_results(session, "done", iteration=1)

    tool_msgs = {
        m["tool_call_id"]: m
        for m in session.get_messages() if m.get("role") == "tool"
    }
    assert not tool_msgs["tc_small"].get("_stubbed")
    assert tool_msgs["tc_small"]["content"] == "small output"
    assert tool_msgs["tc_big"].get("_stubbed") is True
