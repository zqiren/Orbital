# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Backlog #37 + #38 — daemon event rows and the per-call runtime block ride
the wire as POSITIONAL turns, not as `role:"system"` messages.

Both bugs share one mechanism: a chat API's system-message *position* is not
portable. Our own Anthropic adapter hoists every system row into the top-level
`system` param (position discarded by design) and DeepSeek's serving stack does
the equivalent server-side. Consequences:

  #37  a sub-agent/fanout terminal wake event lands in the preamble, so the
       manager's view of the *conversation* ends at "Dispatched… awaiting
       completion" — it goes back to waiting forever.
  #38  the per-call `truly_dynamic` runtime block (timestamp / context usage /
       checkpoint status) lands right after the static system region, so every
       byte after it — i.e. the entire conversation — misses the prefix cache
       on every single call (23.5% hit rate, frozen at the static prefix).

The fix for both is provider-INDEPENDENT and lives at the ContextManager wire
boundary: tagged event rows become `role:"user"` turns in place, and the
runtime block is folded into (or emitted as) the final user turn. Session
JSONL rows, daemon event flow, wake triggers and UI rendering are untouched.

There is deliberately NO provider-conditional logic anywhere in this seam —
one wire shape, both adapters inherit it.
"""

import re

import pytest

from agent_os.agent.context import (
    _POSITIONAL_EVENT_TAGS,
    _RUNTIME_MARKER,
    _RUNTIME_SEP,
    ContextManager,
)
from agent_os.agent.prompt_builder import Autonomy, PromptBuilder, PromptContext
from agent_os.agent.providers.anthropic_adapter import translate_messages_to_anthropic
from agent_os.agent.providers.openai_compat import LLMProvider
from agent_os.agent.session import Session


# ── helpers ──────────────────────────────────────────────────────────────

_DYNAMIC = (
    "Runtime: macos | Model: deepseek-v4-flash | Workspace: /ws\n"
    "Current time: 2026-08-03T09:29\n\n"
    "Context usage: ~25%.\n\n"
    "State checkpoint: no consolidation yet this session."
)

# The exact wake row the fanout resolver / lifecycle observer injects.
_TERMINAL_CONTENT = (
    "[Sub-agent] claude-code completed. Summary: login state verified. "
    "Transcript: /ws/orbital/sub_agents/cc.jsonl. The user can already see "
    "this summary in chat as the sub-agent's own message — do NOT repeat or "
    "re-summarize it."
)


class _StubBuilder:
    """Deterministic three-part prompt split (no clock, no disk)."""

    def __init__(self, cached="CACHED", semi_stable="SEMI", dynamic=_DYNAMIC):
        self._cached = cached
        self._semi_stable = semi_stable
        self._dynamic = dynamic
        self.contexts = []

    def build(self, context):
        self.contexts.append(context)
        return (self._cached, self._semi_stable, self._dynamic)


def _base_ctx(workspace: str) -> PromptContext:
    return PromptContext(
        workspace=workspace,
        model="deepseek-v4-flash",
        autonomy=Autonomy.HANDS_OFF,
        enabled_agents=[],
        tool_names=["read"],
        os_type="macos",
        datetime_now="2026-01-01T00:00:00",
        context_usage_pct=0.0,
    )


def _mgr(tmp_path, rows, builder=None):
    """ContextManager over a session seeded with `rows` (appended verbatim)."""
    workspace = str(tmp_path)
    session = Session.new("wire-test", workspace)
    for row in rows:
        session.append(dict(row))
    return ContextManager(session, builder or _StubBuilder(), _base_ctx(workspace)), session


def _remap_mgr(tmp_path, rows):
    """A manager with an empty runtime block, so #37's remap can be asserted on
    exact bytes without #38's fold landing on the same tail row."""
    return _mgr(tmp_path, rows, builder=_StubBuilder(dynamic=""))


def _terminal_row(content=_TERMINAL_CONTENT, kind="completed"):
    return {
        "role": "system",
        "content": content,
        "source": "daemon",
        "_meta": {"event": "sub_agent_terminal", "kind": kind},
    }


def _shape(messages):
    return [(m.get("role"), m.get("content")) for m in messages]


def _framed(block: str = _DYNAMIC) -> str:
    return f"{_RUNTIME_MARKER}\n{block}"


# ── #37: tagged daemon event rows become positional user turns ───────────


class TestEventRowRemap:

    def test_sub_agent_terminal_row_is_emitted_as_user_turn(self, tmp_path):
        mgr, _ = _remap_mgr(tmp_path, [
            {"role": "user", "content": "check the login state", "source": "user"},
            {"role": "assistant", "content": "dispatching claude-code"},
            _terminal_row(),
        ])
        prepared = mgr.prepare()

        wake = [m for m in prepared if _TERMINAL_CONTENT in str(m.get("content"))]
        assert len(wake) == 1, "the wake row must appear exactly once"
        assert wake[0]["role"] == "user", (
            "a terminal wake row emitted as role:'system' is hoisted out of the "
            "conversation by DeepSeek (and by our own Anthropic adapter), which "
            "is exactly why managers went blind to worker completion"
        )
        assert wake[0]["content"] == _TERMINAL_CONTENT, "content must be untouched"

    def test_interaction_required_row_is_emitted_as_user_turn(self, tmp_path):
        content = (
            "[Sub-agent] claude-code requires input (question): pick a branch. "
            'Respond with agent_message(action="respond", ...).'
        )
        mgr, _ = _remap_mgr(tmp_path, [
            {"role": "user", "content": "go", "source": "user"},
            {
                "role": "system",
                "content": content,
                "source": "daemon",
                "_meta": {"event": "interaction_required", "handle": "claude-code",
                          "interaction_id": "i1", "kind": "question"},
            },
        ])
        prepared = mgr.prepare()

        row = [m for m in prepared if m.get("content") == content]
        assert len(row) == 1
        assert row[0]["role"] == "user", (
            "a blocked worker waiting on an answer is the worse half of #37: "
            "the manager never sees the question"
        )

    @pytest.mark.parametrize("kind", ["completed", "error", "failed", "stopped",
                                      "interrupted", "fanout_join"])
    def test_every_terminal_kind_is_remapped(self, tmp_path, kind):
        mgr, _ = _remap_mgr(tmp_path, [
            {"role": "user", "content": "go", "source": "user"},
            _terminal_row(content=f"[Sub-agent] worker {kind}.", kind=kind),
        ])
        prepared = mgr.prepare()
        row = [m for m in prepared if m.get("content") == f"[Sub-agent] worker {kind}."]
        assert row and row[0]["role"] == "user"

    def test_untagged_system_row_is_left_alone(self, tmp_path):
        """Narrow scope (#37 Q1a): loop nudges and other untagged system rows
        keep their role. Only `_meta.event`-tagged rows are remapped."""
        nudge = "Repetitive action detected. Try a different approach."
        mgr, _ = _mgr(tmp_path, [
            {"role": "user", "content": "go", "source": "user"},
            {"role": "system", "content": nudge, "source": "management"},
        ])
        prepared = mgr.prepare()
        row = [m for m in prepared if m.get("content") == nudge]
        assert len(row) == 1
        assert row[0]["role"] == "system"

    def test_meta_without_event_tag_is_left_alone(self, tmp_path):
        """A `_meta` that carries no `event` key (e.g. a tool-result stub or a
        display split alone) must not be swept up by the remap."""
        content = "[Sub-agent] some untagged daemon note."
        mgr, _ = _mgr(tmp_path, [
            {"role": "user", "content": "go", "source": "user"},
            {"role": "system", "content": content, "source": "daemon",
             "_meta": {"display_content": "trimmed"}},
        ])
        prepared = mgr.prepare()
        row = [m for m in prepared if m.get("content") == content]
        assert row[0]["role"] == "system"

    def test_unknown_event_tag_is_left_alone(self, tmp_path):
        content = "[Queue] some future event kind."
        mgr, _ = _mgr(tmp_path, [
            {"role": "user", "content": "go", "source": "user"},
            {"role": "system", "content": content, "source": "daemon",
             "_meta": {"event": "queue_item_started"}},
        ])
        prepared = mgr.prepare()
        row = [m for m in prepared if m.get("content") == content]
        assert row[0]["role"] == "system"
        assert "queue_item_started" not in _POSITIONAL_EVENT_TAGS

    def test_window_ordering_is_preserved(self, tmp_path):
        rows = [
            {"role": "user", "content": "u1", "source": "user"},
            {"role": "assistant", "content": "a1"},
            _terminal_row(content="[Sub-agent] one done."),
            {"role": "assistant", "content": "a2"},
            _terminal_row(content="[Sub-agent] two done."),
        ]
        mgr, _ = _remap_mgr(tmp_path, rows)
        prepared = mgr.prepare()

        history = [m for m in prepared if m.get("role") in ("user", "assistant", "tool")]
        # Drop the trailing runtime turn (#38) before comparing history order.
        history = [m for m in history if not str(m.get("content", "")).startswith(_RUNTIME_MARKER)]
        assert [m["content"] for m in history] == [
            "u1", "a1", "[Sub-agent] one done.", "a2", "[Sub-agent] two done.",
        ]

    def test_session_rows_are_not_mutated(self, tmp_path):
        """Spec §5: the wire encoding changes, the session JSONL does not.
        The UI keys off `role:"system"` + `source:"daemon"` + `_meta`."""
        mgr, session = _mgr(tmp_path, [
            {"role": "user", "content": "go", "source": "user"},
            _terminal_row(),
        ])
        mgr.prepare()

        stored = [m for m in session.get_messages() if m.get("content") == _TERMINAL_CONTENT]
        assert len(stored) == 1
        assert stored[0]["role"] == "system", "session row must stay role:'system'"
        assert stored[0]["source"] == "daemon"
        assert stored[0]["_meta"]["event"] == "sub_agent_terminal"

    def test_meta_survives_to_the_wire_boundary(self, tmp_path):
        """The remapped row keeps `_meta` (the adapter strips it) so downstream
        auditing can still tell a daemon event from a typed user message."""
        mgr, _ = _remap_mgr(tmp_path, [
            {"role": "user", "content": "go", "source": "user"},
            _terminal_row(),
        ])
        row = [m for m in mgr.prepare() if m.get("content") == _TERMINAL_CONTENT][0]
        assert row["_meta"]["event"] == "sub_agent_terminal"


# ── #38: the runtime block rides the final user turn, never a system row ─


class TestRuntimeBlockPosition:

    def test_runtime_block_is_never_a_system_row(self, tmp_path):
        mgr, _ = _mgr(tmp_path, [
            {"role": "user", "content": "hello", "source": "user"},
        ])
        prepared = mgr.prepare()

        for msg in prepared:
            if msg.get("role") == "system":
                assert "Current time:" not in str(msg.get("content")), (
                    "a trailing dynamic system row is hoisted to the FRONT by "
                    "DeepSeek, invalidating the whole conversation's prefix cache"
                )
                assert _RUNTIME_MARKER not in str(msg.get("content"))

    def test_folded_into_the_final_user_message(self, tmp_path):
        """#38 Q1(a): fold into the final user turn when there is one — no new
        message when it can be avoided."""
        mgr, _ = _mgr(tmp_path, [
            {"role": "assistant", "content": "anything else?"},
            {"role": "user", "content": "yes, keep going", "source": "user"},
        ])
        prepared = mgr.prepare()

        assert prepared[-1]["role"] == "user"
        assert prepared[-1]["content"] == f"yes, keep going{_RUNTIME_SEP}{_framed()}"
        # No extra row was created.
        assert sum(1 for m in prepared if _RUNTIME_MARKER in str(m.get("content"))) == 1

    def test_standalone_user_row_after_a_tool_result(self, tmp_path):
        """The common mid-agentic-turn case: the window tail is a tool result,
        so the runtime block gets its own positional user row."""
        mgr, _ = _mgr(tmp_path, [
            {"role": "user", "content": "read it", "source": "user"},
            {"role": "assistant", "content": None,
             "tool_calls": [{"id": "c1", "type": "function",
                             "function": {"name": "read", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "c1", "content": "file body"},
        ])
        prepared = mgr.prepare()

        assert prepared[-1] == {"role": "user", "content": _framed()}
        assert prepared[-2]["role"] == "tool", "the tool result must stay in place"

    def test_folds_into_a_remapped_event_row(self, tmp_path):
        """A wake event is now the final *user* turn, so the runtime block folds
        into it rather than adding a row — the engaged-wake shape."""
        mgr, _ = _mgr(tmp_path, [
            {"role": "user", "content": "go", "source": "user"},
            _terminal_row(),
        ])
        prepared = mgr.prepare()

        assert prepared[-1]["role"] == "user"
        assert prepared[-1]["content"] == f"{_TERMINAL_CONTENT}{_RUNTIME_SEP}{_framed()}"

    def test_empty_runtime_block_appends_nothing(self, tmp_path):
        mgr, _ = _mgr(tmp_path, [
            {"role": "user", "content": "hello", "source": "user"},
        ], builder=_StubBuilder(semi_stable="", dynamic=""))
        prepared = mgr.prepare()

        assert prepared[-1]["content"] == "hello", "no empty runtime turn"
        assert not any(_RUNTIME_MARKER in str(m.get("content")) for m in prepared)
        assert [m["role"] for m in prepared] == ["system", "user"]

    def test_multimodal_final_user_turn_gets_a_text_block(self, tmp_path):
        mgr, _ = _mgr(tmp_path, [
            {"role": "user", "source": "user", "content": [
                {"type": "text", "text": "what is this?"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
            ]},
        ])
        prepared = mgr.prepare()

        content = prepared[-1]["content"]
        assert prepared[-1]["role"] == "user"
        assert isinstance(content, list)
        assert content[-1] == {"type": "text", "text": _framed()}
        assert len(content) == 3, "the original blocks must be preserved"

    def test_memory_hygiene_flags_ride_in_the_runtime_turn(self, tmp_path, monkeypatch):
        """`[MEMORY HYGIENE]` flags append into truly_dynamic (context.py) — they
        must travel with it into the user turn, not back into a system row."""
        from agent_os.agent import memory_entries as _mem

        monkeypatch.setattr(_mem, "soft_flag",
                            lambda *a, **k: "PROJECT_STATE.md is over budget.")
        orbital = tmp_path / "orbital"
        orbital.mkdir(parents=True, exist_ok=True)
        (orbital / "PROJECT_STATE.md").write_text("# State\nstuff", encoding="utf-8")

        mgr, _ = _mgr(tmp_path, [
            {"role": "user", "content": "hello", "source": "user"},
        ])
        prepared = mgr.prepare()

        assert "[MEMORY HYGIENE] PROJECT_STATE.md is over budget." in prepared[-1]["content"]
        assert prepared[-1]["role"] == "user"
        for msg in prepared:
            if msg.get("role") == "system":
                assert "[MEMORY HYGIENE]" not in str(msg.get("content"))

    def test_session_is_not_mutated_by_the_fold(self, tmp_path):
        mgr, session = _mgr(tmp_path, [
            {"role": "user", "content": "yes, keep going", "source": "user"},
        ])
        mgr.prepare()
        stored = [m for m in session.get_messages() if m.get("role") == "user"]
        assert stored[-1]["content"] == "yes, keep going", (
            "prepare() must copy, never edit the persisted row"
        )

    def test_prepare_is_repeatable(self, tmp_path):
        """Two prepare() calls must not accumulate runtime blocks."""
        mgr, _ = _mgr(tmp_path, [
            {"role": "user", "content": "hello", "source": "user"},
        ])
        mgr.prepare()
        second = mgr.prepare()
        assert second[-1]["content"].count(_RUNTIME_MARKER) == 1


# ── Wire shape: both adapters inherit the positional encoding ────────────


def _provider() -> LLMProvider:
    return LLMProvider(model="test-model", api_key="fake-key",
                       base_url="http://localhost:1234")


class TestOpenAICompatWireShape:

    def test_event_and_runtime_stay_positional(self, tmp_path):
        mgr, _ = _mgr(tmp_path, [
            {"role": "user", "content": "go", "source": "user"},
            {"role": "assistant", "content": None,
             "tool_calls": [{"id": "c1", "type": "function",
                             "function": {"name": "agent_message", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "c1",
             "content": "Dispatched. Awaiting completion."},
            _terminal_row(),
        ])
        wire = _provider()._prepare_messages_openai(mgr.prepare())

        roles = [m["role"] for m in wire]
        assert roles == ["system", "system", "user", "assistant", "tool", "user"]
        assert wire[-1]["content"].startswith(_TERMINAL_CONTENT)
        assert wire[-1]["content"].endswith(_framed())
        # Nothing Orbital-internal leaks.
        assert "_meta" not in wire[-1]

    def test_runtime_row_does_not_split_a_tool_call_pair(self, tmp_path):
        """MiniMax 2013-class contiguity: a tool result must immediately follow
        the assistant message holding its tool_calls."""
        mgr, _ = _mgr(tmp_path, [
            {"role": "user", "content": "read both", "source": "user"},
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": "c1", "type": "function",
                 "function": {"name": "read", "arguments": "{}"}},
                {"id": "c2", "type": "function",
                 "function": {"name": "read", "arguments": "{}"}},
            ]},
            {"role": "tool", "tool_call_id": "c1", "content": "one"},
            {"role": "tool", "tool_call_id": "c2", "content": "two"},
        ])
        wire = _provider()._prepare_messages_openai(mgr.prepare())

        assert [m["role"] for m in wire] == [
            "system", "system", "user", "assistant", "tool", "tool", "user",
        ]
        assert [m.get("tool_call_id") for m in wire if m["role"] == "tool"] == ["c1", "c2"]
        assert wire[-1]["content"] == _framed()

    def test_consecutive_user_turns_after_a_tool_result(self, tmp_path):
        """tool result → wake event → queued real user message. Consecutive
        user turns are legal on OpenAI-compat; the sequence must survive
        untouched (the contiguity pass only moves rows *inside* a tool block)."""
        mgr, _ = _mgr(tmp_path, [
            {"role": "assistant", "content": None,
             "tool_calls": [{"id": "c1", "type": "function",
                             "function": {"name": "agent_message", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "c1", "content": "Dispatched."},
            _terminal_row(),
            {"role": "user", "content": "结果已经出来了呀", "source": "user"},
        ])
        wire = _provider()._prepare_messages_openai(mgr.prepare())

        assert [m["role"] for m in wire] == [
            "system", "system", "assistant", "tool", "user", "user",
        ]
        assert wire[-2]["content"] == _TERMINAL_CONTENT
        assert wire[-1]["content"] == f"结果已经出来了呀{_RUNTIME_SEP}{_framed()}"

    def test_no_blank_content_repair_is_triggered(self, tmp_path):
        """A standalone runtime row must carry real text — `_ensure_chat_content`
        rewriting it to "(no content)" would mean we emitted an empty turn."""
        mgr, _ = _mgr(tmp_path, [
            {"role": "user", "content": "read it", "source": "user"},
            {"role": "assistant", "content": None,
             "tool_calls": [{"id": "c1", "type": "function",
                             "function": {"name": "read", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "c1", "content": "body"},
        ])
        wire = _provider()._prepare_messages_openai(mgr.prepare())
        assert "(no content)" not in [m.get("content") for m in wire]


class TestAnthropicWireShape:

    def test_event_and_runtime_are_not_hoisted_into_the_system_param(self, tmp_path):
        """`translate_messages_to_anthropic` hoists EVERY system row into the
        top-level `system` param — position discarded by design. That is the
        in-tree proof that a system row is not a positional turn, and why the
        remap has to happen before the adapter."""
        mgr, _ = _mgr(tmp_path, [
            {"role": "user", "content": "go", "source": "user"},
            {"role": "assistant", "content": None,
             "tool_calls": [{"id": "c1", "type": "function",
                             "function": {"name": "agent_message", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "c1", "content": "Dispatched."},
            _terminal_row(),
        ])
        out = translate_messages_to_anthropic(mgr.prepare())

        assert _TERMINAL_CONTENT not in (out["system"] or "")
        assert _RUNTIME_MARKER not in (out["system"] or "")
        assert out["system"] == "CACHED\nSEMI", (
            "only the two static prompt parts belong in the hoisted region"
        )

        assert [m["role"] for m in out["messages"]] == [
            "user", "assistant", "user", "user",
        ]
        # tool_result block, then the wake turn carrying the runtime block.
        assert out["messages"][2]["content"][0]["type"] == "tool_result"
        assert out["messages"][3]["content"].startswith(_TERMINAL_CONTENT)
        assert out["messages"][3]["content"].endswith(_framed())

    def test_untagged_system_rows_still_hoist(self, tmp_path):
        """Narrow scope: everything else keeps today's Anthropic behavior."""
        mgr, _ = _mgr(tmp_path, [
            {"role": "user", "content": "go", "source": "user"},
            {"role": "system", "content": "Max iterations reached.", "source": "management"},
        ])
        out = translate_messages_to_anthropic(mgr.prepare())
        assert "Max iterations reached." in out["system"]


# ── Golden regression: same bytes, different role/position ───────────────


def _to_legacy_wire(prepared: list[dict]) -> list[tuple[str, str]]:
    """Reverse the #37/#38 remap — what v0.8.2 put on the wire.

    Any surviving difference against the golden means the change altered
    prompt CONTENT, not just role/position.
    """
    framed = _framed()
    unframed = framed[len(_RUNTIME_MARKER) + 1:]  # drop the "[runtime]\n" line
    legacy: list[tuple[str, str]] = []
    runtime_block: str | None = None

    for msg in prepared:
        role, content = msg.get("role"), msg.get("content")
        if isinstance(content, str) and content.endswith(framed):
            runtime_block = unframed
            content = content[: -len(framed)].removesuffix(_RUNTIME_SEP)
            if not content:
                continue  # standalone runtime row: did not exist before
        meta = msg.get("_meta") or {}
        if role == "user" and meta.get("event") in _POSITIONAL_EVENT_TAGS:
            role = "system"
        legacy.append((role, content))

    if runtime_block is not None:
        legacy.append(("system", runtime_block))
    return legacy


# An M3-shaped engaged wake: dispatch → tool result → terminal event.
# M3 honors positional system turns, which is why it never went blind — so its
# prompt is the reference for "did the content change for everyone else?".
_M3_WAKE_ROWS = [
    {"role": "user", "content": "帮我查一下两个平台的登录状态", "source": "user"},
    {"role": "assistant", "content": "派给 claude-code。", "tool_calls": [
        {"id": "call_1", "type": "function",
         "function": {"name": "agent_message", "arguments": '{"action":"send"}'}},
    ]},
    {"role": "tool", "tool_call_id": "call_1",
     "content": "Dispatched to claude-code. Awaiting completion."},
    _terminal_row(),
]

_LEGACY_GOLDEN = [
    ("system", "CACHED"),
    ("system", "SEMI"),
    ("user", "帮我查一下两个平台的登录状态"),
    ("assistant", "派给 claude-code。"),
    ("tool", "Dispatched to claude-code. Awaiting completion."),
    ("system", _TERMINAL_CONTENT),
    ("system", _DYNAMIC),
]


class TestGoldenRegression:

    def test_m3_wake_prompt_is_byte_identical_except_role_and_position(self, tmp_path):
        mgr, _ = _mgr(tmp_path, _M3_WAKE_ROWS)
        prepared = mgr.prepare()

        assert _to_legacy_wire(prepared) == _LEGACY_GOLDEN, (
            "every byte other than the remapped rows' role/position must be "
            "unchanged — other providers see exactly the prompt they saw before"
        )

    def test_new_shape_is_the_only_difference(self, tmp_path):
        """Spell out the delta so a future reader sees precisely what moved."""
        mgr, _ = _mgr(tmp_path, _M3_WAKE_ROWS)
        assert _shape(mgr.prepare()) == [
            ("system", "CACHED"),
            ("system", "SEMI"),
            ("user", "帮我查一下两个平台的登录状态"),
            ("assistant", "派给 claude-code。"),
            ("tool", "Dispatched to claude-code. Awaiting completion."),
            # role system → user; content byte-identical
            ("user", f"{_TERMINAL_CONTENT}{_RUNTIME_SEP}{_RUNTIME_MARKER}\n{_DYNAMIC}"),
            # ...and the runtime block folded into it instead of trailing as a
            # system row. Its text is byte-identical after the marker line.
        ]

    def test_runtime_frame_is_exactly_one_marker_line(self, tmp_path):
        """The golden's `_to_legacy_wire` relies on this framing; pin it."""
        mgr, _ = _mgr(tmp_path, [{"role": "user", "content": "hi", "source": "user"}])
        tail = mgr.prepare()[-1]["content"]
        assert tail == f"hi{_RUNTIME_SEP}{_RUNTIME_MARKER}\n{_DYNAMIC}"


# ── #38 fix 2: churn trim ────────────────────────────────────────────────


class TestChurnTrim:

    def test_timestamp_is_minute_granular(self, tmp_path):
        builder = _StubBuilder()
        mgr, _ = _mgr(tmp_path, [{"role": "user", "content": "hi", "source": "user"}],
                      builder=builder)
        mgr.prepare()
        mgr.prepare()

        stamps = [c.datetime_now for c in builder.contexts]
        assert all(re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}", s) for s in stamps), (
            f"microsecond precision serves nothing and churns every call: {stamps}"
        )
        assert stamps[0] == stamps[1], "two calls in the same minute must match"

    def test_context_usage_rounds_to_five_percent_steps(self):
        def pct_line(value):
            return PromptBuilder()._context_budget(PromptContext(
                workspace="/ws", model="m", autonomy=Autonomy.HANDS_OFF,
                enabled_agents=[], tool_names=[], os_type="macos",
                datetime_now="2026-08-03T09:29", context_usage_pct=value,
            )).splitlines()[0]

        assert pct_line(0.234) == "Context usage: ~20%."
        assert pct_line(0.2) == "Context usage: ~20%."
        assert pct_line(0.29) == "Context usage: ~25%."
        assert pct_line(0.0) == "Context usage: ~0%."

    def test_usage_thresholds_still_use_the_raw_value(self):
        """Rounding is display-only — the 70% / 85% nudges must not shift."""
        def budget(value):
            return PromptBuilder()._context_budget(PromptContext(
                workspace="/ws", model="m", autonomy=Autonomy.HANDS_OFF,
                enabled_agents=[], tool_names=[], os_type="macos",
                datetime_now="2026-08-03T09:29", context_usage_pct=value,
            ))

        assert "Consider updating PROJECT_STATE.md" in budget(0.71)
        assert "Consider updating PROJECT_STATE.md" not in budget(0.70)
        assert "URGENT" in budget(0.86)
        assert "URGENT" not in budget(0.85)

    def test_checkpoint_status_is_stable_while_state_is_unchanged(self):
        def status(turns_since):
            return PromptBuilder()._state_checkpoint_status(PromptContext(
                workspace="/ws", model="m", autonomy=Autonomy.HANDS_OFF,
                enabled_agents=[], tool_names=[], os_type="macos",
                datetime_now="2026-08-03T09:29",
                last_state_update_turn=14,
                last_state_update_ts="2026-08-03T09:21:01+00:00",
                turns_since_last_update=turns_since,
                last_state_update_outcome="llm_merged",
            ))

        assert status(2) == status(9), (
            "a per-turn counter makes the block churn on every call even when "
            "nothing about the checkpoint changed"
        )
        assert "turn 14" in status(2), "the checkpoint state itself must survive"

    def test_checkpoint_status_still_changes_when_state_changes(self):
        def status(**over):
            base = dict(
                workspace="/ws", model="m", autonomy=Autonomy.HANDS_OFF,
                enabled_agents=[], tool_names=[], os_type="macos",
                datetime_now="2026-08-03T09:29",
                last_state_update_turn=14,
                last_state_update_ts="2026-08-03T09:21:01+00:00",
                turns_since_last_update=2,
                last_state_update_outcome="llm_merged",
            )
            base.update(over)
            return PromptBuilder()._state_checkpoint_status(PromptContext(**base))

        assert status() != status(last_state_update_turn=21)
        assert status() != status(last_state_update_outcome="failed")
        assert status() != status(refresh_in_flight=True, refresh_in_flight_since_turn=22)


# ── Prompt wording (#37 Q3a) ─────────────────────────────────────────────


class TestPromptWording:

    def test_wake_is_not_described_as_a_system_message(self):
        ctx = PromptContext(
            workspace="/ws", model="m", autonomy=Autonomy.HANDS_OFF,
            enabled_agents=[], tool_names=[], os_type="macos",
            datetime_now="2026-08-03T09:29",
            active_sub_agents=[{"handle": "claude-code", "status": "running"}],
        )
        section = PromptBuilder()._sub_agent_awareness(ctx)
        assert "[Sub-agent] system message" not in section, (
            "the wake row is no longer a system turn; mislabeling it teaches "
            "the model to look in the wrong place"
        )
        assert "[Sub-agent] message" in section
