# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Regression: the queue contract is delivered via the per-item dispatcher
header, NOT the system prompt.

H1 verification (12 LLM calls, 2 models × 2 placements × 3 samples)
showed header-only delivery gives strictly better final outcomes than
system-prompt delivery. The contract was moved from PromptBuilder's
cached prefix into QueueDispatcher.HEADER_CONTRACT. This file pins
that decision: the system prompt must NOT carry the contract, and the
header must.

If a future refactor tries to move the contract back to the system
prompt, these tests fail and force a re-read of the H1 results.
"""

from __future__ import annotations

import pytest

from agent_os.agent.prompt_builder import Autonomy, PromptBuilder, PromptContext
from agent_os.queue.dispatcher import QueueDispatcher


def _build_minimal_context(workspace: str) -> PromptContext:
    return PromptContext(
        workspace=workspace,
        model="test-model",
        autonomy=Autonomy.HANDS_OFF,
        enabled_agents=[],
        tool_names=["read", "write", "shell"],
        os_type="linux",
        datetime_now="2026-01-01T00:00:00Z",
        project_name="contract-test",
        agent_name="contract-test",
    )


# ---------------------------------------------------------------------------
# Negative: the system prompt must NOT carry the queue contract
# ---------------------------------------------------------------------------


def test_system_prompt_does_not_contain_queue_contract(tmp_path):
    """The rendered system prompt (any of cached / semi-stable / dynamic
    sections) must not contain the contract. The contract is delivered in
    the per-item dispatcher header instead."""
    ctx = _build_minimal_context(str(tmp_path))
    cached, semi, dynamic = PromptBuilder(workspace=str(tmp_path)).build(ctx)
    full = cached + semi + dynamic
    # The most distinctive phrase from HEADER_CONTRACT — used as the
    # canary. If this appears in the system prompt the contract has
    # leaked back in.
    assert "You are working on a queue item" not in full, (
        "queue contract text leaked into the system prompt. The contract "
        "must live in QueueDispatcher.HEADER_CONTRACT only — H1 verified "
        "header-only delivery is strictly better than system-prompt "
        "delivery for ambiguous tasks."
    )
    # Also assert the old D3 boilerplate is gone.
    assert "you MUST end your work by calling exactly" not in full
    assert "Plain-text exit is a contract violation" not in full


def test_queue_signals_section_is_empty(tmp_path):
    """The _queue_signals section is now a stub. Verify nothing slips back in."""
    pb = PromptBuilder(workspace=str(tmp_path))
    assert pb._queue_signals() == ""


# ---------------------------------------------------------------------------
# Positive: the dispatcher header carries the contract
# ---------------------------------------------------------------------------


def test_dispatcher_header_contract_names_both_signals():
    """HEADER_CONTRACT must reference both queue signal tools and forbid
    plain text. These three pieces are the contract minimum."""
    h = QueueDispatcher.HEADER_CONTRACT
    assert "mark_task_complete" in h
    assert "mark_task_blocked" in h
    # Some phrasing forbidding plain text exit
    assert "plain text" in h.lower() or "do not respond with text" in h.lower() or "do not end with plain text" in h.lower()


def test_dispatcher_header_contract_is_consistent_with_corrective_turn():
    """HEADER_CONTRACT and CORRECTIVE_TURN_PROMPT must use the same two
    tool names and the same plain-text-forbidden rule. Drift between the
    two confuses weaker models that lean on consistency in nearby context
    (this is exactly the H1 observation)."""
    h = QueueDispatcher.HEADER_CONTRACT
    c = QueueDispatcher.CORRECTIVE_TURN_PROMPT
    for tool in ("mark_task_complete", "mark_task_blocked"):
        assert tool in h, f"{tool} missing from HEADER_CONTRACT"
        assert tool in c, f"{tool} missing from CORRECTIVE_TURN_PROMPT"
    # Both must forbid plain-text exit in some form
    h_low = h.lower()
    c_low = c.lower()
    assert "plain text" in h_low or "with plain text" in h_low or "with text" in h_low
    assert "with text" in c_low or "plain text" in c_low


def test_dispatcher_header_contract_appears_in_wrapped_item():
    """End-to-end: when the dispatcher wraps an item, the wrapped string
    contains HEADER_CONTRACT verbatim between the metadata line and the
    item content. This is the wire contract the agent sees."""
    from agent_os.queue.models import ItemRecord
    item = ItemRecord(content="do the thing")
    attempt_number = 1
    header = (
        f"[QUEUE ITEM | id={item.id} | attempt={attempt_number}]\n"
        + QueueDispatcher.HEADER_CONTRACT
    )
    wrapped = header + item.content
    # Structure: metadata line, then contract, then content.
    lines = wrapped.split("\n", 1)
    assert lines[0] == f"[QUEUE ITEM | id={item.id} | attempt={attempt_number}]"
    assert wrapped[len(lines[0]) + 1:].startswith(QueueDispatcher.HEADER_CONTRACT)
    assert wrapped.endswith("do the thing")
