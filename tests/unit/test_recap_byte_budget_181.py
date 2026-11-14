# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Spec 181: the hand-off a taking-over agent gets is bounded in UTF-8 bytes.

The "Conversation so far" block used to be capped at 60,000 *characters*:
~15K tokens of English but up to ~60K tokens of Chinese, cut silently (the
worker could not know older messages existed), and emptied to "" when the
newest line alone was over the cap. Now:

* the WHOLE rendered block (header, note, lines, joins, footer) is at most
  ``RECAP_CAP_BYTES`` UTF-8 bytes;
* when lines are left out, a note says how many and names the chat log;
* the conversation's first user message is kept (first in the block) unless
  it alone is over a quarter of the budget;
* the budget can only shrink, by the room the receiving thread has left
  (T3 Code's shape) when the agent itself reported its window and usage;
* the block is never "" when there was something to hand off.
"""

from __future__ import annotations

import pytest

from agent_os.daemon_v2.recap import (
    RECAP_CAP_BYTES,
    build_recap,
    handoff_budget_bytes,
)

LOG = "/ws/orbital/sessions/proj_1a2b3c4d.jsonl"
HEADER = "Conversation so far"
FOOTER = "--- end of conversation so far ---\n\n"


def _user(content, target=None):
    m = {"role": "user", "content": content, "source": "user"}
    if target:
        m["target"] = target
    return m


def _assistant(content):
    return {"role": "assistant", "content": content, "source": "management"}


def _size(block: str) -> int:
    return len(block.encode("utf-8"))


def _lines_kept(block: str, marker: str) -> int:
    return block.count(marker)


class TestWholeBlockInBytes:

    def test_ascii_over_budget_whole_block_within_cap(self):
        rows = [_user(f"m{i} " + "x" * 400) for i in range(400)]
        block = build_recap(rows, "codex", fresh=True, chat_log_path=LOG)
        assert _size(block) <= RECAP_CAP_BYTES
        assert block.startswith(HEADER) and block.endswith(FOOTER)
        assert "m399 " in block

    def test_chinese_over_budget_whole_block_within_cap_and_fewer_lines(self):
        ascii_rows = [_user(f"<{i}> " + "x" * 300) for i in range(600)]
        cjk_rows = [_user(f"<{i}> " + "字" * 300) for i in range(600)]
        a = build_recap(ascii_rows, "codex", fresh=True, chat_log_path=LOG)
        c = build_recap(cjk_rows, "codex", fresh=True, chat_log_path=LOG)
        assert _size(a) <= RECAP_CAP_BYTES
        assert _size(c) <= RECAP_CAP_BYTES
        kept_a = _lines_kept(a, "user: <")
        kept_c = _lines_kept(c, "user: <")
        # A CJK char costs 3 bytes: about a third as many lines fit.
        assert kept_c * 2.5 < kept_a < kept_c * 3.5

    def test_mixed_with_assistant_and_long_lines(self):
        rows = []
        for i in range(300):
            rows.append(_user(f"第{i}轮：" + "请帮我整理季度预算" * 15))
            rows.append(_assistant("好的，" + "这是整理后的结果。" * 160))
        block = build_recap(rows, "codex", fresh=True, chat_log_path=LOG)
        assert _size(block) <= RECAP_CAP_BYTES


class TestNoteAndFirstUserMessage:

    def test_note_counts_left_out_lines_and_names_the_log(self):
        rows = [_user("the original request: build the budget sheet")]
        rows += [_user(f"m{i} " + "x" * 1000) for i in range(100)]
        block = build_recap(rows, "codex", fresh=True, chat_log_path=LOG)
        parts = block[len(block.split("\n\n", 1)[0]) + 2:].split("\n\n")
        # First the conversation's first user message, then the note.
        assert parts[0] == "user: the original request: build the budget sheet"
        kept = _lines_kept(block, "user: m")
        n = 100 - kept
        assert parts[1] == (
            f"(The {n} earlier messages of this conversation are not shown "
            f"here. The whole conversation is in the Orbital chat log: "
            f"\"{LOG}\")")
        assert "user: m99 " in block

    def test_no_eviction_no_note_and_byte_identical_to_old_shape(self):
        rows = [_user("first"), _assistant("second"), _user("third")]
        block = build_recap(rows, "codex", fresh=True, chat_log_path=LOG)
        assert block == (
            "Conversation so far (earlier messages in this chat, provided as "
            "context; prior replies are labeled \"assistant\"):\n\n"
            "user: first\n\nassistant: second\n\nuser: third"
            "\n\n--- end of conversation so far ---\n\n")
        assert "not shown here" not in block

    def test_no_chat_log_path_drops_the_path_sentence(self):
        rows = [_user(f"m{i} " + "x" * 1000) for i in range(100)]
        block = build_recap(rows, "codex", fresh=True, chat_log_path=None)
        assert "not shown here.)" in block
        assert "Orbital chat log" not in block

    def test_singular_note(self):
        big = "y" * 59_000
        rows = [_assistant("a" * 2000), _user(big)]
        block = build_recap(rows, "codex", fresh=True, chat_log_path=LOG)
        assert "(The 1 earlier message of this conversation is not shown" in block

    def test_first_user_message_skipped_when_over_a_quarter_of_budget(self):
        first = "z" * (RECAP_CAP_BYTES // 4 + 10)
        rows = [_user(first)] + [_user(f"m{i} " + "x" * 1000) for i in range(100)]
        block = build_recap(rows, "codex", fresh=True, chat_log_path=LOG)
        assert first not in block
        assert _size(block) <= RECAP_CAP_BYTES
        # Note directly after the header when the first message is skipped.
        after_header = block.split("\n\n", 1)[1]
        assert after_header.startswith("(The ")

    def test_newest_line_over_budget_is_header_note_footer_not_empty(self):
        rows = [_user("hello"), _assistant("x" * 70_000)]
        block = build_recap(rows, "codex", fresh=True, chat_log_path=LOG)
        assert block != ""
        assert block.startswith(HEADER) and block.endswith(FOOTER)
        assert "x" * 1000 not in block
        # The first user message still fits and is kept; the big reply is
        # the one left out.
        assert "user: hello" in block
        assert "(The 1 earlier message" in block

    def test_only_line_over_budget_counts_it(self):
        rows = [_assistant("x" * 70_000)]
        block = build_recap(rows, "codex", fresh=True, chat_log_path=LOG)
        assert "(The 1 earlier message" in block
        assert _size(block) <= RECAP_CAP_BYTES

    def test_empty_scope_still_empty(self):
        assert build_recap([], "codex", fresh=True, chat_log_path=LOG) == ""


class TestTextCleanStillStrips:

    def test_text_clean_strips_a_block_with_the_note(self):
        from agent_os.onboarding.text_clean import KIND_USER, clean_user_text
        rows = [_user(f"m{i} " + "x" * 1000) for i in range(100)]
        block = build_recap(rows, "codex", fresh=True, chat_log_path=LOG)
        assert "not shown here" in block
        assert clean_user_text(block + "the real ask") == (KIND_USER, "the real ask")


class TestBudgetShape:

    def test_unknown_window_leaves_fixed_cap(self):
        assert handoff_budget_bytes(message_bytes=100) == RECAP_CAP_BYTES

    def test_budget_only_shrinks(self):
        assert handoff_budget_bytes(
            message_bytes=0, reported=(10_000_000, 0)) == RECAP_CAP_BYTES

    def test_room_from_reported_window_and_usage(self):
        # window 200K, used 120K, message 5K, reserve max(16K, 50K) = 50K
        assert handoff_budget_bytes(
            message_bytes=5_000, reported=(200_000, 120_000)) == 25_000

    def test_room_never_negative(self):
        assert handoff_budget_bytes(
            message_bytes=0, reported=(200_000, 199_000)) == 0

    def test_huge_user_message_shrinks_default_budget(self):
        # 256K default window: room = 256000 - 0 - 150000 - 64000 = 42000
        assert handoff_budget_bytes(message_bytes=150_000) == 42_000

    def test_malformed_report_ignored(self):
        assert handoff_budget_bytes(message_bytes=0, reported=(None, 5)) == RECAP_CAP_BYTES
        assert handoff_budget_bytes(message_bytes=0, reported=("x",)) == RECAP_CAP_BYTES

    def test_zero_room_is_header_and_note_only(self):
        rows = [_user("first ask")] + [_user(f"m{i}") for i in range(5)]
        block = build_recap(rows, "codex", fresh=True, chat_log_path=LOG,
                            budget_bytes=0)
        assert block.startswith(HEADER)
        assert "(The 6 earlier messages" in block
        assert "user: m4" not in block and "first ask" not in block

    def test_shrunk_budget_is_respected(self):
        rows = [_user(f"m{i} " + "x" * 500) for i in range(100)]
        block = build_recap(rows, "codex", fresh=True, chat_log_path=LOG,
                            budget_bytes=10_000)
        assert _size(block) <= 10_000
        assert "user: m99 " in block


class TestDispatchUsesTheBudget:

    @pytest.mark.asyncio
    async def test_build_recap_passes_chat_log_and_reported_context(self):
        from types import SimpleNamespace
        from unittest.mock import MagicMock
        from agent_os.daemon_v2.sub_agent_manager import SubAgentManager

        rows = [_user("the original ask")] + [
            _user(f"第{i}轮 " + "字" * 600) for i in range(300)]

        class _S:
            filepath = LOG
            sub_agent_thread_history: dict = {}
            forked_from = None

            def get_messages(self):
                return list(rows)

        mgr = SubAgentManager(process_manager=MagicMock())
        mgr._session_resolver = lambda pid, sid: _S()
        mgr._native_log_path = lambda *a, **kw: None
        adapter = SimpleNamespace(_resume_status=("fresh", None),
                                  _transport=SimpleNamespace(reported_context=None))
        block = await mgr._build_recap(adapter, "proj", "proj_1a2b3c4d", "codex",
                                       message="继续")
        recap = block.split("--- end of conversation so far ---\n\n", 1)[0] \
            + "--- end of conversation so far ---\n\n"
        assert _size(recap) <= RECAP_CAP_BYTES
        assert recap.split("\n\n")[1] == "user: the original ask"
        assert "earlier messages of this conversation are not shown" in recap
        assert f'"{LOG}"' in recap

        # A thread that reported a nearly full window gets a smaller block.
        adapter2 = SimpleNamespace(
            _resume_status=("fresh", None),
            _transport=SimpleNamespace(reported_context=(256_000, 180_000)))
        block2 = await mgr._build_recap(adapter2, "proj", "proj_1a2b3c4d",
                                        "codex", message="继续")
        recap2 = block2.split("--- end of conversation so far ---\n\n", 1)[0]
        assert _size(recap2) <= 256_000 - 180_000 - len("继续".encode()) - 64_000


class TestAgentsReportTheirWindow:
    """The one optional pair of numbers on the transport: set only from what
    the agent itself reported, read only to shrink the hand-off."""

    @pytest.mark.asyncio
    async def test_codex_token_usage_sets_reported_context(self):
        from agent_os.agent.transports.codex_transport import CodexTransport
        t = CodexTransport()
        assert t.reported_context is None
        await t._route_server_message({
            "jsonrpc": "2.0", "method": "thread/tokenUsage/updated",
            "params": {"threadId": "T1", "turnId": "U1", "tokenUsage": {
                "last": {"totalTokens": 41_000, "inputTokens": 40_000},
                "total": {"totalTokens": 90_000},
                "modelContextWindow": 258_400}}})
        assert t.reported_context == (258_400, 41_000)

    @pytest.mark.asyncio
    async def test_codex_without_window_reports_nothing(self):
        from agent_os.agent.transports.codex_transport import CodexTransport
        t = CodexTransport()
        await t._route_server_message({
            "jsonrpc": "2.0", "method": "thread/tokenUsage/updated",
            "params": {"threadId": "T1", "turnId": "U1", "tokenUsage": {
                "last": {"totalTokens": 41_000}, "modelContextWindow": None}}})
        assert t.reported_context is None

    def test_claude_result_sets_reported_context_from_last_request(self):
        from claude_agent_sdk.types import AssistantMessage, ResultMessage, TextBlock
        from agent_os.agent.transports.sdk_transport import SDKTransport
        t = SDKTransport()
        t._message_to_events(AssistantMessage(
            content=[TextBlock(text="hi")], model="claude-x",
            usage={"input_tokens": 10, "cache_read_input_tokens": 30_000,
                   "cache_creation_input_tokens": 1_000, "output_tokens": 500}))
        t._message_to_events(ResultMessage(
            subtype="success", duration_ms=1, duration_api_ms=1, is_error=False,
            num_turns=3, session_id="s1",
            # The result's usage sums every request of the turn: not used.
            usage={"input_tokens": 90_000, "output_tokens": 2_000},
            model_usage={"claude-x": {"contextWindow": 200_000}}))
        assert t.reported_context == (200_000, 31_510)

    def test_claude_result_without_model_usage_reports_nothing(self):
        from claude_agent_sdk.types import ResultMessage
        from agent_os.agent.transports.sdk_transport import SDKTransport
        t = SDKTransport()
        t._message_to_events(ResultMessage(
            subtype="success", duration_ms=1, duration_api_ms=1, is_error=False,
            num_turns=1, session_id="s1", usage={"input_tokens": 5}))
        assert t.reported_context is None
