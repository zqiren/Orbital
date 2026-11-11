# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Spec 171 B1 — every mid-conversation system row rides the wire as a
positional, framed ``user`` turn.

Backlog #37 moved three tagged daemon events off ``role:"system"``; every other
system row in the conversation (``[cancelled by user]``, the unsandboxed-run
output, approval-dismissed, the loop's own notices) still went out as a system
message. The Anthropic adapter hoists those into the top-level ``system``
param and DeepSeek does the same server-side, so each one changed the preamble
(the whole conversation missed the prompt cache once) and lost its place in the
conversation.

Now: the leading prompt (static prompt, semi-stable block, Layer 1/3 rows, and
a compaction summary that leads the window) is the only system content. Every
later system row is sent as a ``user`` turn where it sits, framed with the
runtime marker so the model reads it as an injected note, not the user's words.
Only the copy bound for the provider changes; the session file does not.
"""

import json

from agent_os.agent.context import (
    _RUNTIME_MARKER,
    _RUNTIME_SEP,
    ContextManager,
)
from agent_os.agent.prompt_builder import Autonomy, PromptContext
from agent_os.agent.providers.anthropic_adapter import translate_messages_to_anthropic
from agent_os.agent.providers.openai_compat import LLMProvider
from agent_os.agent.session import Session


_DYNAMIC = "Current time: 2026-11-11T10:00\n\nContext usage: ~10%."
_CANCEL = "[cancelled by user]"


class _StubBuilder:
    def __init__(self, dynamic=_DYNAMIC):
        self._dynamic = dynamic

    def build(self, context):
        return ("CACHED", "SEMI", self._dynamic)


def _ctx(workspace: str) -> PromptContext:
    return PromptContext(
        workspace=workspace, model="m", autonomy=Autonomy.HANDS_OFF,
        enabled_agents=[], tool_names=["read"], os_type="macos",
        datetime_now="2026-01-01T00:00:00", context_usage_pct=0.0,
    )


def _mgr(tmp_path, rows, dynamic=_DYNAMIC):
    session = Session.new("b1", str(tmp_path))
    for row in rows:
        session.append(dict(row))
    return ContextManager(session, _StubBuilder(dynamic), _ctx(str(tmp_path))), session


def _provider() -> LLMProvider:
    return LLMProvider(model="test-model", api_key="fake-key",
                       base_url="http://localhost:1234")


def _notice(text: str) -> str:
    return f"{_RUNTIME_MARKER}\n{text}"


def _runtime() -> str:
    return f"{_RUNTIME_MARKER}\n{_DYNAMIC}"


def _cancel_row():
    return {"role": "system", "content": _CANCEL, "source": "management",
            "cancelled_by_user": True}


def _sys(text, **extra):
    return {"role": "system", "content": text, "source": "management", **extra}


def _call(cid, name="read"):
    return {"id": cid, "type": "function", "function": {"name": name, "arguments": "{}"}}


def _leading_system(messages):
    out = []
    for m in messages:
        if m.get("role") != "system":
            break
        out.append(m)
    return out


def _assert_openai_valid(wire):
    """System rows only lead; every tool_calls turn is followed by exactly its
    results, contiguously."""
    lead = len(_leading_system(wire))
    assert all(m["role"] != "system" for m in wire[lead:]), [m["role"] for m in wire]
    for i, m in enumerate(wire):
        if m["role"] == "assistant" and m.get("tool_calls"):
            ids = [tc["id"] for tc in m["tool_calls"]]
            following = wire[i + 1:i + 1 + len(ids)]
            assert [f["role"] for f in following] == ["tool"] * len(ids)
            assert sorted(f["tool_call_id"] for f in following) == sorted(ids)


def _assert_anthropic_valid(out):
    """Messages start with user; every tool_use turn is immediately followed by
    a user turn whose content opens with all its tool_result blocks."""
    msgs = out["messages"]
    assert msgs[0]["role"] == "user"
    for i, m in enumerate(msgs):
        if m["role"] != "assistant" or not isinstance(m["content"], list):
            continue
        ids = [b["id"] for b in m["content"] if b.get("type") == "tool_use"]
        if not ids:
            continue
        nxt = msgs[i + 1]
        assert nxt["role"] == "user"
        assert isinstance(nxt["content"], list)
        head = nxt["content"][:len(ids)]
        assert [b.get("type") for b in head] == ["tool_result"] * len(ids)
        assert sorted(b["tool_use_id"] for b in head) == sorted(ids)


# ── the leading prompt never moves ───────────────────────────────────────


class TestLeadingPromptIsStable:

    _BASE = [
        {"role": "user", "content": "u1", "source": "user"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "u2", "source": "user"},
    ]

    def _with_rows(self):
        return [
            self._BASE[0], self._BASE[1],
            _cancel_row(),
            _sys("Primary model unavailable (503). Switching to x."),
            self._BASE[2],
        ]

    def test_prepared_leading_system_rows_are_byte_identical(self, tmp_path):
        plain, _ = _mgr(tmp_path / "a", self._BASE)
        busy, _ = _mgr(tmp_path / "b", self._with_rows())
        assert _leading_system(plain.prepare()) == _leading_system(busy.prepare())

    def test_no_system_row_after_the_leading_prompt(self, tmp_path):
        mgr, _ = _mgr(tmp_path, self._with_rows())
        prepared = mgr.prepare()
        lead = len(_leading_system(prepared))
        assert [m["role"] for m in prepared[lead:]].count("system") == 0

    def test_anthropic_system_param_is_byte_identical(self, tmp_path):
        plain, _ = _mgr(tmp_path / "a", self._BASE)
        busy, _ = _mgr(tmp_path / "b", self._with_rows())
        for cache in (False, True):
            a = translate_messages_to_anthropic(plain.prepare(), cache=cache)
            b = translate_messages_to_anthropic(busy.prepare(), cache=cache)
            assert a["system"] == b["system"]
        assert translate_messages_to_anthropic(busy.prepare())["system"] == "CACHED\nSEMI"

    def test_openai_system_rows_are_byte_identical(self, tmp_path):
        plain, _ = _mgr(tmp_path / "a", self._BASE)
        busy, _ = _mgr(tmp_path / "b", self._with_rows())
        a = _provider()._prepare_messages_openai(plain.prepare())
        b = _provider()._prepare_messages_openai(busy.prepare())
        assert [m for m in a if m["role"] == "system"] == [m for m in b if m["role"] == "system"]


# ── each row keeps its position, framed ──────────────────────────────────


class TestPositionAndFraming:

    def test_cancel_row_sits_where_it_happened(self, tmp_path):
        mgr, _ = _mgr(tmp_path, [
            {"role": "user", "content": "u1", "source": "user"},
            {"role": "assistant", "content": "a1"},
            _cancel_row(),
            {"role": "user", "content": "u2", "source": "user"},
        ])
        prepared = mgr.prepare()
        assert [(m["role"], m["content"]) for m in prepared[2:]] == [
            ("user", "u1"),
            ("assistant", "a1"),
            ("user", _notice(_CANCEL)),
            ("user", f"u2{_RUNTIME_SEP}{_runtime()}"),
        ]

    def test_row_directly_after_a_user_turn(self, tmp_path):
        """The notice is its own turn: the user's row keeps its bytes and its
        one-turn note (the note belongs to the turn being answered)."""
        typed = "do it <orbital_turn_note>note for this turn</orbital_turn_note>"
        mgr, _ = _mgr(tmp_path, [
            {"role": "user", "content": typed, "source": "user"},
            _sys("This project's API key was rejected (HTTP 401). Using the global default."),
        ])
        prepared = mgr.prepare()
        assert prepared[2] == {**prepared[2], "role": "user", "content": typed}
        assert prepared[3]["role"] == "user"
        assert prepared[3]["content"].startswith(_notice("This project's API key"))

        wire = _provider()._prepare_messages_openai(prepared)
        assert [m["role"] for m in wire] == ["system", "system", "user", "user"]
        out = translate_messages_to_anthropic(prepared)
        assert [m["role"] for m in out["messages"]] == ["user", "user"]
        _assert_anthropic_valid(out)

    def test_two_rows_in_a_row(self, tmp_path):
        mgr, _ = _mgr(tmp_path, [
            {"role": "user", "content": "u1", "source": "user"},
            {"role": "assistant", "content": "a1"},
            _sys("[autonomy changed to hands_off]"),
            _cancel_row(),
            {"role": "user", "content": "u2", "source": "user"},
        ])
        prepared = mgr.prepare()
        assert [(m["role"], m["content"]) for m in prepared[4:6]] == [
            ("user", _notice("[autonomy changed to hands_off]")),
            ("user", _notice(_CANCEL)),
        ]
        _assert_openai_valid(_provider()._prepare_messages_openai(prepared))
        out = translate_messages_to_anthropic(prepared)
        assert out["system"] == "CACHED\nSEMI"
        assert [m["role"] for m in out["messages"]] == ["user", "assistant", "user", "user", "user"]

    def test_row_as_the_last_message_carries_the_runtime_block(self, tmp_path):
        mgr, _ = _mgr(tmp_path, [
            {"role": "user", "content": "u1", "source": "user"},
            {"role": "assistant", "content": "a1"},
            _sys("Denied for this run: the user was unavailable."),
        ])
        tail = mgr.prepare()[-1]
        assert tail["role"] == "user"
        assert tail["content"] == (
            _notice("Denied for this run: the user was unavailable.")
            + _RUNTIME_SEP + _runtime()
        )

    def test_row_between_tool_use_and_tool_result(self, tmp_path):
        """A cancel lands mid-batch: [cancelled] is written before the
        skipped calls' CANCELLED results. Both wires get the results
        contiguous after the tool call, then the notice."""
        mgr, _ = _mgr(tmp_path, [
            {"role": "user", "content": "read both", "source": "user"},
            {"role": "assistant", "content": None, "tool_calls": [_call("c1"), _call("c2")]},
            {"role": "tool", "tool_call_id": "c1", "content": "one"},
            _cancel_row(),
            {"role": "tool", "tool_call_id": "c2", "content": "CANCELLED"},
            {"role": "user", "content": "next", "source": "user"},
        ])
        prepared = mgr.prepare()

        wire = _provider()._prepare_messages_openai(prepared)
        _assert_openai_valid(wire)
        assert [m["role"] for m in wire] == [
            "system", "system", "user", "assistant", "tool", "tool", "user", "user"]
        assert wire[6]["content"] == _notice(_CANCEL)

        out = translate_messages_to_anthropic(prepared)
        _assert_anthropic_valid(out)
        assert out["system"] == "CACHED\nSEMI"
        assert out["messages"][3]["content"] == _notice(_CANCEL)

    def test_row_between_tool_use_and_tool_result_anthropic_cached(self, tmp_path):
        mgr, _ = _mgr(tmp_path, [
            {"role": "user", "content": "go", "source": "user"},
            {"role": "assistant", "content": None, "tool_calls": [_call("c1")]},
            _sys("Task blocked: needs a key", source="queue_signal", signal="blocked"),
            {"role": "tool", "tool_call_id": "c1", "content": "CANCELLED"},
        ])
        out = translate_messages_to_anthropic(mgr.prepare(), cache=True)
        _assert_anthropic_valid(out)
        assert [b["text"] for b in out["system"]] == ["CACHED", "SEMI"]

    def test_row_inside_a_tool_block_with_a_missing_result(self, tmp_path):
        """The synthetic 'result was lost' row is injected; the wire stays valid."""
        mgr, _ = _mgr(tmp_path, [
            {"role": "user", "content": "go", "source": "user"},
            {"role": "assistant", "content": None, "tool_calls": [_call("c1"), _call("c2")]},
            _cancel_row(),
            {"role": "tool", "tool_call_id": "c2", "content": "CANCELLED"},
        ])
        prepared = mgr.prepare()
        _assert_openai_valid(_provider()._prepare_messages_openai(prepared))
        _assert_anthropic_valid(translate_messages_to_anthropic(prepared))


# ── what stays as it was ─────────────────────────────────────────────────


class TestUnchanged:

    def test_session_file_is_untouched(self, tmp_path):
        mgr, session = _mgr(tmp_path, [
            {"role": "user", "content": "u1", "source": "user"},
            {"role": "assistant", "content": "a1"},
            _cancel_row(),
            _sys("Repetitive action detected."),
        ])
        path = session._filepath
        before = open(path, "rb").read()
        mgr.prepare()
        mgr.prepare()
        assert open(path, "rb").read() == before
        stored = [m for m in session.get_messages() if m.get("content") == _CANCEL]
        assert stored[0]["role"] == "system"

    def test_three_event_path_output_is_unchanged(self, tmp_path):
        """Tagged events keep #37's encoding: role user, content unframed."""
        rows = [
            {"role": "user", "content": "u1", "source": "user"},
            {"role": "assistant", "content": "a1"},
            {"role": "system", "content": "[Sub-agent] cc completed.", "source": "daemon",
             "_meta": {"event": "sub_agent_terminal", "kind": "completed"}},
            {"role": "system", "content": "[Sub-agent] cc needs input.", "source": "daemon",
             "_meta": {"event": "interaction_required"}},
            {"role": "system", "content": "Record what is worth keeping.", "source": "daemon",
             "_meta": {"event": "memory_record"}},
        ]
        mgr, _ = _mgr(tmp_path, rows, dynamic="")
        assert [(m["role"], m["content"]) for m in mgr.prepare()[4:]] == [
            ("user", "[Sub-agent] cc completed."),
            ("user", "[Sub-agent] cc needs input."),
            ("user", "Record what is worth keeping."),
        ]

    def test_compaction_summary_leads_as_system(self, tmp_path):
        """The summary is the start of the conversation the model sees; it
        stays with the leading prompt."""
        session = Session.new("b1", str(tmp_path))
        for row in [
            {"role": "user", "content": "old", "source": "user"},
            {"role": "assistant", "content": "old reply"},
            {"role": "user", "content": "recent", "source": "user"},
        ]:
            session.append(row)
        session._compact({"role": "system", "content": "SUMMARY", "_compaction": True,
                          "source": "management"}, 2)
        mgr = ContextManager(session, _StubBuilder(), _ctx(str(tmp_path)))
        prepared = mgr.prepare()
        assert [(m["role"], m["content"]) for m in _leading_system(prepared)] == [
            ("system", "CACHED"), ("system", "SEMI"), ("system", "SUMMARY"),
        ]

    def test_rows_with_no_text_are_not_sent(self, tmp_path):
        """The shutdown marker and the budget-blocked event carry no content;
        they used to reach strict providers as a '(no content)' system row."""
        mgr, _ = _mgr(tmp_path, [
            {"role": "user", "content": "u1", "source": "user"},
            {"role": "assistant", "content": "a1"},
            {"role": "system", "type": "daemon_shutdown", "timestamp": "t"},
            {"role": "system", "source": "budget", "event": "budget_blocked",
             "payload": {"action": "pause"}},
            {"role": "user", "content": "u2", "source": "user"},
        ])
        prepared = mgr.prepare()
        assert [m["role"] for m in prepared] == ["system", "system", "user", "assistant", "user"]
        wire = _provider()._prepare_messages_openai(prepared)
        assert "(no content)" not in json.dumps(wire)
