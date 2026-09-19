# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Spec 086: text tool results are never rewritten in the prepared prompt.

``_prune_old_tool_results`` used to replace every text tool result over 500
chars with a ``[Truncated]`` stub once it was more than 5 assistant turns old.
The horizon moved one turn per call, so each call rewrote exactly the result
that had just crossed it, and everything after it missed the provider's prompt
cache (replay-measured ceiling 38-62% on a tool-heavy session). These tests pin
the invariant the cache depends on: between two calls, the prepared prompt only
changes at its tail.
"""

import json

from agent_os.agent.context import ContextManager
from agent_os.agent.prompt_builder import Autonomy, PromptContext
from agent_os.agent.session import Session


class _Builder:
    def build(self, context: PromptContext) -> tuple[str, str, str]:
        return ("cached-system-prefix", "semi-stable-suffix", "dynamic-runtime")


def _ctx(workspace: str) -> PromptContext:
    return PromptContext(
        workspace=workspace,
        model="test-model",
        autonomy=Autonomy.HANDS_OFF,
        enabled_agents=[],
        tool_names=["read"],
        os_type="linux",
        datetime_now="2026-01-01T00:00:00",
        context_usage_pct=0.0,
    )


def _append_read_turn(session: Session, n: int, content: str) -> None:
    tc_id = f"tc_read_{n}"
    session.append({
        "role": "assistant",
        "content": None,
        "source": "management",
        "tool_calls": [{"id": tc_id, "type": "function",
                        "function": {"name": "read",
                                     "arguments": json.dumps({"path": f"f{n}.md"})}}],
    })
    session.append_tool_result(tc_id, content)


def _first_difference(a: list[dict], b: list[dict]) -> int:
    for i, (x, y) in enumerate(zip(a, b)):
        if json.dumps(x, sort_keys=True) != json.dumps(y, sort_keys=True):
            return i
    return min(len(a), len(b))


def _tool_contents(messages: list[dict]) -> dict[str, object]:
    return {m["tool_call_id"]: m["content"] for m in messages if m.get("role") == "tool"}


def test_text_tool_results_never_rewritten_in_context(tmp_path):
    session = Session.new("cache_stability", str(tmp_path))
    session.append({"role": "user", "content": "Read the docs.", "source": "user"})
    old_result = "\n".join(f"line {i}: " + "A" * 60 for i in range(30))
    assert len(old_result) >= 2_000
    _append_read_turn(session, 0, old_result)
    for n in range(1, 13):
        _append_read_turn(session, n, f"file {n}\n" + "B" * 800)

    mgr = ContextManager(session, _Builder(), _ctx(str(tmp_path)),
                         model_context_limit=200_000)
    first = mgr.prepare()

    _append_read_turn(session, 13, "file 13\n" + "C" * 800)
    second = mgr.prepare()

    # (a) Every tool result, including the 13-turn-old one, reaches the model
    # exactly as the session holds it, on both calls.
    stored = _tool_contents(session.get_messages())
    for prepared in (first, second):
        sent = _tool_contents(prepared)
        assert sent["tc_read_0"] == old_result
        assert sent == {k: stored[k] for k in sent}

    # (b) The second prompt extends the first: nothing before the appended
    # tail differs (the first call's trailing runtime row is where the new
    # turn lands).
    diff_at = _first_difference(first, second)
    assert diff_at >= len(first) - 2, (
        f"prompt rewritten mid-history at message {diff_at} of {len(first)}: "
        f"{json.dumps(first[diff_at])[:200]}"
    )


def test_old_multimodal_text_result_is_not_summarised(tmp_path):
    """A list-shaped result with no image is no longer summarised as [Pruned]."""
    blocks = [{"type": "text", "text": "page one " * 80},
              {"type": "text", "text": "page two " * 80}]
    msgs = [{"role": "assistant", "content": None,
             "tool_calls": [{"id": "tc_doc", "function": {"name": "read"}}]},
            {"role": "tool", "tool_call_id": "tc_doc", "content": blocks}]
    for n in range(8):
        msgs.append({"role": "assistant", "content": f"reply {n}"})

    out = ContextManager._prune_old_tool_results(None, msgs)

    assert out[1]["content"] == blocks
