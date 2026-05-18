# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Regression: the rendered management-agent system prompt contains the D3
queue-contract section verbatim (Concern 4 of
TASK-queue-architecture-amendments.md).

If this test fails after a prompt refactor, do NOT relax the assertion —
the agent's behavioral guarantees depend on the exact wording (queue-item
header recognition, mandatory signal-or-violation rule, tool-discard
short-circuit, chat-message carve-out)."""

from __future__ import annotations

import pytest

from agent_os.agent.prompt_builder import Autonomy, PromptBuilder, PromptContext


# Verbatim D3 contract text. Any character drift here (em-dash → hyphen,
# casing changes, paragraph collapse, etc.) breaks the contract test.
_D3_CONTRACT = (
    "You operate inside an Orbital project. When a queue item is dispatched to you, "
    "you will receive it as a user message with a `[QUEUE ITEM | id=... | attempt=...]` "
    "header. When operating on a queue item, you MUST end your work by calling exactly "
    "one of:\n"
    "- `mark_task_complete(summary)` if you finished the task.\n"
    "- `mark_task_blocked(reason)` if you cannot proceed — stuck, missing credentials, "
    "ambiguous requirements, or you need to ask the user a clarifying question. Put any "
    "question for the user directly in the `reason` field; the user will see it and respond.\n"
    "\n"
    "Do not exit with plain text on a queue item. Plain-text exit is a contract violation "
    "and the queue cannot advance.\n"
    "\n"
    "When you call a queue signal, any other tools you call in the same response are "
    "discarded. Finish all other work first, then call the signal alone.\n"
    "\n"
    "When operating on a chat message (no queue header), respond conversationally as "
    "normal — no signal required."
)


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


def test_cached_prefix_contains_verbatim_d3_contract(tmp_path):
    """The cached prefix is where _queue_signals() lives. Asserting on the
    cached portion ensures the contract text is part of the prefix-cached
    block (so the model always sees it, and we don't pay tokens for it on
    every turn)."""
    ctx = _build_minimal_context(str(tmp_path))
    cached, _semi, _dynamic = PromptBuilder(workspace=str(tmp_path)).build(ctx)
    assert _D3_CONTRACT in cached, (
        "verbatim D3 contract text not found in cached system-prompt prefix.\n"
        "Expected the D3 block to appear inside the cached section as a "
        "contiguous substring. If the prompt was refactored, restore the "
        "exact wording — the agent's contract depends on it."
    )


def test_d3_contract_mentions_each_required_signal(tmp_path):
    """Belt-and-suspenders: even if the verbatim assertion above is
    accidentally weakened, this test catches partial regressions where
    one of the required pieces of the contract is dropped."""
    ctx = _build_minimal_context(str(tmp_path))
    cached, _semi, _dynamic = PromptBuilder(workspace=str(tmp_path)).build(ctx)
    # Queue item header signal
    assert "[QUEUE ITEM | id=... | attempt=...]" in cached
    # Mandatory signal-or-violation rule
    assert "contract violation" in cached
    # Both signals named
    assert "mark_task_complete" in cached
    assert "mark_task_blocked" in cached
    # Tool-discard short-circuit
    assert "discarded" in cached
