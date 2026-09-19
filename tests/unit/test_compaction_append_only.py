# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Spec 086: compaction appends a marker, never rewrites the session file.

Compaction used to rewrite the JSONL as [summary] + the newest 30% of rows,
deleting the rest of the user's chat history from disk (the chat route reads
that file). Now every row stays on disk; the summary is appended as a marker
row that says where the model's history resumes, and the model-facing view is
derived from the latest marker. The summary is validated, the current turn's
user message is carried verbatim, and the split is chosen by size.
"""

from __future__ import annotations

import json
import os

import pytest

from agent_os.agent import compaction
from agent_os.agent.providers.types import LLMResponse, TokenUsage
from agent_os.agent.session import Session


def _resp(text: str, tool_calls=None) -> LLMResponse:
    return LLMResponse(
        raw_message={"role": "assistant", "content": text},
        text=text,
        tool_calls=tool_calls or [],
        has_tool_calls=bool(tool_calls),
        finish_reason="stop",
        status_text=None,
        usage=TokenUsage(input_tokens=10, output_tokens=5),
    )


class _Summarizer:
    """complete() returns the scripted replies in order."""

    def __init__(self, *replies: str):
        self._replies = list(replies)
        self.calls: list[list[dict]] = []

    async def complete(self, messages, tools=None):
        self.calls.append(messages)
        return _resp(self._replies.pop(0))


def _rows(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _conversation(session: Session, reads: int, size: int = 4_000) -> None:
    """One long turn: the user's request, then `reads` read exchanges."""
    session.append({"role": "user", "content": "Read every file and report the code words.",
                    "source": "user"})
    for n in range(reads):
        tc = f"tc_{n}"
        session.append({
            "role": "assistant", "content": None, "source": "management",
            "tool_calls": [{"id": tc, "type": "function",
                            "function": {"name": "read",
                                         "arguments": json.dumps({"path": f"f{n}.md"})}}],
        })
        session.append_tool_result(tc, f"file {n}\n" + "x" * size)


def _messages_only(rows: list[dict]) -> list[dict]:
    return [r for r in rows if r.get("role") != "meta"]


# ---------------------------------------------------------------------------
# Append-only on disk
# ---------------------------------------------------------------------------

class TestAppendOnly:

    @pytest.mark.asyncio
    async def test_every_original_row_stays_on_disk(self, tmp_path, monkeypatch):
        session = Session.new("keep_rows", str(tmp_path))
        _conversation(session, reads=10)
        before = _messages_only(_rows(session._filepath))

        def _no_replace(*a, **k):
            raise AssertionError("compaction must not replace the session file")

        monkeypatch.setattr(os, "replace", _no_replace)
        await compaction.run(session, _Summarizer("Read files 0-7; code words noted."),
                             keep_tokens=3_000)

        after = _messages_only(_rows(session._filepath))
        assert after[:len(before)] == before
        assert len(after) == len(before) + 1
        assert after[-1]["_compaction"] is True
        assert after[-1]["content"] == "Read files 0-7; code words noted."
        assert session.get_messages() == after

    @pytest.mark.asyncio
    async def test_model_view_after_restart(self, tmp_path):
        session = Session.new("view_restart", str(tmp_path))
        _conversation(session, reads=10)
        await compaction.run(session, _Summarizer("Summary of reads."), keep_tokens=3_000)
        session.append({"role": "assistant", "content": "after compaction",
                        "source": "management"})
        live_view = session.get_model_messages()

        loaded = Session.load(session._filepath)
        view = loaded.get_model_messages()

        assert view == live_view
        assert view[0]["_compaction"] is True
        assert view[0]["content"] == "Summary of reads."
        # The user's request rides verbatim right after the summary.
        assert view[1]["role"] == "user"
        assert view[1]["content"] == "Read every file and report the code words."
        # The kept tail opens on an assistant turn (never an orphan tool row)
        # and runs through everything appended after the marker.
        assert view[2]["role"] == "assistant"
        assert view[-1]["content"] == "after compaction"
        assert sum(1 for m in view if m.get("role") == "tool") < 10
        # The prompt's sliding window reads the same view.
        from agent_os.agent.context import ContextManager
        from agent_os.agent.prompt_builder import Autonomy, PromptContext

        class _Builder:
            def build(self, context):
                return ("cached-system-prefix", "", "")

        ctx = PromptContext(workspace=str(tmp_path), model="m", autonomy=Autonomy.HANDS_OFF,
                            enabled_agents=[], tool_names=[], os_type="linux",
                            datetime_now="2026-01-01T00:00")
        prompt = ContextManager(loaded, _Builder(), ctx, model_context_limit=1_000_000).prepare()
        # The per-call runtime block (e.g. the asks cue, spec 089) rides its
        # own positional user row when the history ends on a non-user turn;
        # it is not part of the session's history.
        history = [m for m in prompt[1:]
                   if not (m.get("role") == "user"
                           and str(m.get("content", "")).startswith("[runtime]"))]
        assert history == view
        assert loaded.get_recent(10_000_000) == view

    @pytest.mark.asyncio
    async def test_two_compactions_chain(self, tmp_path):
        session = Session.new("chain", str(tmp_path))
        _conversation(session, reads=8)
        await compaction.run(session, _Summarizer("First summary."), keep_tokens=3_000)
        for n in range(8, 16):
            tc = f"tc_{n}"
            session.append({"role": "assistant", "content": None, "source": "management",
                            "tool_calls": [{"id": tc, "type": "function",
                                            "function": {"name": "read", "arguments": "{}"}}]})
            session.append_tool_result(tc, f"file {n}\n" + "y" * 4_000)
        summarizer = _Summarizer("Second summary.")
        await compaction.run(session, summarizer, keep_tokens=3_000)

        # The second summarizer saw the first summary, so it covers it.
        assert "First summary." in summarizer.calls[0][-1]["content"]
        view = Session.load(session._filepath).get_model_messages()
        markers = [m for m in view if m.get("_compaction")]
        assert [m["content"] for m in markers] == ["Second summary."]
        assert view[0]["content"] == "Second summary."
        assert view[1]["content"] == "Read every file and report the code words."
        # Disk still holds everything: both markers and all 16 reads.
        rows = _messages_only(_rows(session._filepath))
        assert sum(1 for r in rows if r.get("_compaction")) == 2
        assert sum(1 for r in rows if r.get("role") == "tool") == 16

    @pytest.mark.asyncio
    async def test_rewrite_paths_keep_the_history_and_the_view(self, tmp_path):
        """Supersession stubbing and cancellation splicing rebuild the file
        from the session's rows; neither may drop pre-compaction history."""
        session = Session.new("rewrites", str(tmp_path))
        _conversation(session, reads=8)
        await compaction.run(session, _Summarizer("Summary."), keep_tokens=3_000)
        view_before = [m.get("content") for m in session.get_model_messages()]

        session.replace_tool_results_with_stubs({"tc_0": "[superseded]"})
        session.append({"role": "assistant", "content": None, "source": "management",
                        "tool_calls": [{"id": "tc_pending", "type": "function",
                                        "function": {"name": "read", "arguments": "{}"}}]})
        session.resolve_pending_tool_calls()

        rows = _messages_only(_rows(session._filepath))
        assert sum(1 for r in rows if r.get("role") == "tool") == 9  # 8 reads + CANCELLED
        assert rows[0]["content"] == "Read every file and report the code words."
        view = Session.load(session._filepath).get_model_messages()
        assert [m.get("content") for m in view][:len(view_before)] == view_before

    def test_legacy_destructive_file_loads_unchanged(self, tmp_path):
        """A file compacted by an older version: summary first, history gone."""
        path = tmp_path / "legacy_1234abcd.jsonl"
        rows = [
            {"role": "meta", "event": "session_start", "session_id": "legacy_1234abcd"},
            {"role": "system", "content": "Old summary.", "_compaction": True,
             "source": "management"},
            {"role": "assistant", "content": "kept reply", "source": "management"},
            {"role": "user", "content": "next question", "source": "user"},
        ]
        path.write_text("".join(json.dumps(r) + "\n" for r in rows))

        loaded = Session.load(str(path))

        assert loaded.get_model_messages() == rows[1:]
        assert loaded.get_messages() == rows[1:]
        assert loaded.get_recent(1_000_000) == rows[1:]

    def test_file_compacted_again_by_an_older_version_still_loads(self, tmp_path):
        """Downgrade then upgrade: v0.13.0 rewrote the file (its own summary
        first, the newest 30% of rows after it), keeping a new-style marker
        whose offsets now point past the start of the file. It still loads;
        offsets are clamped and orphan pins dropped."""
        path = tmp_path / "mixed_1234abcd.jsonl"
        rows = [
            {"role": "meta", "event": "session_start", "session_id": "mixed_1234abcd"},
            {"role": "system", "content": "OLD-VERSION SUMMARY", "_compaction": True},
            {"role": "tool", "tool_call_id": "tc_7", "content": "file 7"},
            {"role": "system", "content": "New summary.", "_compaction": True,
             "_compaction_keep": 5, "_compaction_pinned": [17]},
            {"role": "assistant", "content": "done", "source": "management"},
        ]
        path.write_text("".join(json.dumps(r) + "\n" for r in rows))

        view = Session.load(str(path)).get_model_messages()

        assert [m.get("content") for m in view] == ["New summary.", "file 7", "done"]

    @pytest.mark.asyncio
    async def test_chat_reader_returns_all_rows_and_the_marker(self, tmp_path):
        from agent_os.api.routes.agents_v2 import _read_chat_messages_single

        session = Session.new("chat_rows", str(tmp_path))
        _conversation(session, reads=8)
        before = _messages_only(_rows(session._filepath))
        await compaction.run(session, _Summarizer("Summary."), keep_tokens=3_000)

        messages, total = _read_chat_messages_single(session._filepath, 0, 0)

        chat = _messages_only(messages)
        assert chat[:len(before)] == before
        assert chat[-1]["_compaction"] is True
        assert total == len(messages)


# ---------------------------------------------------------------------------
# What the model keeps
# ---------------------------------------------------------------------------

class TestSplit:

    @pytest.mark.asyncio
    async def test_kept_tail_is_bounded_by_tokens(self, tmp_path):
        from agent_os.agent.token_utils import estimate_message_tokens

        session = Session.new("by_tokens", str(tmp_path))
        _conversation(session, reads=12, size=6_000)
        await compaction.run(session, _Summarizer("Summary."), keep_tokens=5_000)

        view = session.get_model_messages()
        tail = view[2:]  # after the summary and the pinned request
        assert tail[0]["role"] == "assistant"
        assert sum(estimate_message_tokens(m) for m in tail) <= 5_000

    @pytest.mark.asyncio
    async def test_latest_exchange_is_kept_even_over_budget(self, tmp_path):
        session = Session.new("min_tail", str(tmp_path))
        _conversation(session, reads=6, size=20_000)
        await compaction.run(session, _Summarizer("Summary."), keep_tokens=100)

        view = session.get_model_messages()
        assert [m.get("role") for m in view[2:]] == ["assistant", "tool"]
        assert view[-1]["tool_call_id"] == "tc_5"

    @pytest.mark.asyncio
    async def test_no_new_rows_means_no_new_marker(self, tmp_path):
        session = Session.new("no_progress", str(tmp_path))
        _conversation(session, reads=6, size=20_000)
        await compaction.run(session, _Summarizer("Summary."), keep_tokens=100)
        rows_before = len(session.get_messages())

        await compaction.run(session, _Summarizer("Again."), keep_tokens=100)

        assert len(session.get_messages()) == rows_before

    @pytest.mark.asyncio
    async def test_current_turn_user_messages_survive_verbatim(self, tmp_path):
        session = Session.new("task", str(tmp_path))
        session.append({"role": "user", "content": "old question", "source": "user"})
        session.append({"role": "assistant", "content": "old answer", "source": "management"})
        _conversation(session, reads=4)
        session.append({"role": "user", "content": "also check f9.md", "source": "user"})
        for n in range(4, 9):
            tc = f"tc_{n}"
            session.append({"role": "assistant", "content": None, "source": "management",
                            "tool_calls": [{"id": tc, "type": "function",
                                            "function": {"name": "read", "arguments": "{}"}}]})
            session.append_tool_result(tc, "z" * 4_000)

        await compaction.run(session, _Summarizer("Summary."), keep_tokens=3_000)

        view = session.get_model_messages()
        users = [m["content"] for m in view if m.get("role") == "user"]
        assert users == ["Read every file and report the code words.", "also check f9.md"]
        assert view[1]["content"] == "Read every file and report the code words."


# ---------------------------------------------------------------------------
# Summary quality
# ---------------------------------------------------------------------------

class TestSummaryValidation:

    @pytest.mark.asyncio
    @pytest.mark.parametrize("garbage", [
        '{"type": "tool_use", "id": "read_04", "name": "read_file", '
        '"input": {"file_path": "docs/ledger04.md"}}',
        "<tool_call> <function=read> <parameter=path>docs/x.md</parameter>",
        "",
        "   ",
        "<silent>",
        '[{"name": "read"}]',
    ], ids=["tool_use-json", "tool_call-xml", "empty", "blank", "silent", "json-array"])
    async def test_garbage_is_retried_once(self, tmp_path, garbage):
        session = Session.new("retry", str(tmp_path))
        _conversation(session, reads=8)
        summarizer = _Summarizer(garbage, "Plain prose summary.")

        await compaction.run(session, summarizer, keep_tokens=3_000)

        assert len(summarizer.calls) == 2
        assert session.get_model_messages()[0]["content"] == "Plain prose summary."

    @pytest.mark.asyncio
    async def test_two_failures_fall_back_to_a_transcript_summary(self, tmp_path):
        session = Session.new("fallback", str(tmp_path))
        _conversation(session, reads=8)
        summarizer = _Summarizer('{"type": "tool_use"}', "<tool_call>read</tool_call>")

        await compaction.run(session, summarizer, keep_tokens=3_000)

        summary = session.get_model_messages()[0]
        assert summary["_compaction"] is True
        text = summary["content"]
        assert "tool_use" not in text and "<tool_call>" not in text
        assert "Read every file and report the code words." in text
        assert 'read {"path": "f0.md"}' in text

    @pytest.mark.asyncio
    async def test_summarizer_error_falls_back(self, tmp_path):
        class _Broken:
            async def complete(self, messages, tools=None):
                raise RuntimeError("gateway 502")

        session = Session.new("broken", str(tmp_path))
        _conversation(session, reads=8)

        await compaction.run(session, _Broken(), keep_tokens=3_000)

        assert "Read every file" in session.get_model_messages()[0]["content"]


class TestSummarizerRequest:

    @pytest.mark.asyncio
    async def test_transcript_is_quoted_not_replayed(self, tmp_path):
        """mimo answered the transcript's user message ("I cannot read your
        files") instead of summarizing it: the rows must arrive as a quoted
        record with the instruction next to them, not as live turns."""
        session = Session.new("framing", str(tmp_path))
        _conversation(session, reads=8)
        summarizer = _Summarizer("Summary.")

        await compaction.run(session, summarizer, keep_tokens=3_000)

        request = summarizer.calls[0]
        assert [m["role"] for m in request] == ["system", "user"]
        body = request[1]["content"]
        assert body.index("<transcript>") < body.index("[user]: Read every file")
        assert body.rstrip().endswith("</transcript>\n\nWrite the summary now.".rstrip())
        assert "not a request to you" in body

    @pytest.mark.asyncio
    async def test_slow_summarizer_falls_back_without_stalling(self, tmp_path, monkeypatch):
        import asyncio

        class _Slow:
            calls = 0

            async def complete(self, messages, tools=None):
                _Slow.calls += 1
                await asyncio.sleep(5)
                return _resp("late")

        monkeypatch.setattr(compaction, "_SUMMARY_TIMEOUT_S", 0.05)
        session = Session.new("slow", str(tmp_path))
        _conversation(session, reads=8)

        await compaction.run(session, _Slow(), keep_tokens=3_000)

        assert _Slow.calls == 1  # a timeout is not retried
        assert "Read every file" in session.get_model_messages()[0]["content"]

    @pytest.mark.asyncio
    async def test_chained_summary_carries_the_earlier_one_forward(self, tmp_path):
        """Live mimo run: the 2nd summary restated the 1st word for word and
        dropped what happened after it (a code word the agent had found). The
        earlier summary goes in its own section, the new rows in another, and
        the instruction asks for one updated summary covering both."""
        session = Session.new("chain_frame", str(tmp_path))
        _conversation(session, reads=8)
        await compaction.run(session, _Summarizer("Found WINDOW-61 in f0."), keep_tokens=3_000)
        session.append({"role": "assistant", "content": "Found MEADOW-183 in f8.",
                        "source": "management"})
        for n in range(8, 14):
            tc = f"tc_{n}"
            session.append({"role": "assistant", "content": None, "source": "management",
                            "tool_calls": [{"id": tc, "type": "function",
                                            "function": {"name": "read", "arguments": "{}"}}]})
            session.append_tool_result(tc, "y" * 4_000)
        summarizer = _Summarizer("Updated.")

        await compaction.run(session, summarizer, keep_tokens=3_000)

        body = summarizer.calls[0][1]["content"]
        earlier = body[body.index("<earlier_summary>"):body.index("</earlier_summary>")]
        later = body[body.index("<transcript>"):body.index("</transcript>")]
        assert "Found WINDOW-61 in f0." in earlier
        assert "Found WINDOW-61" not in later
        assert "Found MEADOW-183 in f8." in later
        assert "keeps every fact" in body
