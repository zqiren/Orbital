# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Step 5 — formalize the loop completion accessor.

The dispatcher used to read AgentLoop._exit_reason / _exit_summary /
_exit_block_reason as raw attributes. This formalizes that into
AgentLoop.get_completion_state() and has the dispatcher read through a helper
that prefers the accessor (falling back to attribute reads for lightweight test
doubles, so behavior is unchanged).
"""

from __future__ import annotations

from agent_os.agent.loop import AgentLoop
from agent_os.queue.dispatcher import QueueDispatcher


def test_agent_loop_get_completion_state_returns_tuple():
    # Build without __init__ (which needs a full provider/session/registry):
    # we only exercise the accessor over the three exit attributes.
    loop = AgentLoop.__new__(AgentLoop)
    loop._exit_reason = "blocked"
    loop._exit_summary = None
    loop._exit_block_reason = "need creds"
    assert loop.get_completion_state() == ("blocked", None, "need creds")


def test_read_completion_prefers_accessor():
    class _Loop:
        def get_completion_state(self):
            return ("complete", "did it", None)

    assert QueueDispatcher._read_completion(_Loop()) == ("complete", "did it", None)


def test_read_completion_falls_back_to_attributes():
    class _LegacyDouble:
        _exit_reason = "blocked"
        _exit_summary = "partial"
        _exit_block_reason = "stuck"

    assert QueueDispatcher._read_completion(_LegacyDouble()) == (
        "blocked", "partial", "stuck",
    )


def test_read_completion_none_loop_defaults_text():
    assert QueueDispatcher._read_completion(None) == ("text", None, None)
