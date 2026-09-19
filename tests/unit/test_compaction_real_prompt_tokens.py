# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Spec 086 §8.1: compaction triggers on the prompt size the provider reported.

``should_compact()`` used to compare the ``len(json.dumps(msg)) / 4`` estimate
of the prepared prompt with the threshold. That estimate escapes CJK to
``\\uXXXX`` and counts Chinese about 2x its real size, so once text tool
results stop being pruned a Chinese-heavy session on a 128k card would compact
at roughly 40% of real usage. The loop now hands the provider's reported
prompt tokens for the call it just made to the context manager, and the
estimate is only the fallback for a call that reported no usage.

Compaction also no longer runs a blocking memory consolidation first (the old
``token_pressure`` refresh). Consolidation is scheduled by memory-file budget;
compaction keeps only its own flush turn.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from agent_os.agent.context import ContextManager
from agent_os.agent.loop import AgentLoop
from agent_os.agent.prompt_builder import Autonomy, PromptContext
from agent_os.agent.providers.types import LLMResponse, TokenUsage
from agent_os.agent.session import Session, persist_user_row
from agent_os.agent.tools.base import ToolResult


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


def _response(text: str, usage: TokenUsage | None, tool_calls=None) -> LLMResponse:
    tcs = tool_calls or []
    raw = {"role": "assistant", "content": text}
    if tcs:
        raw["tool_calls"] = tcs
    return LLMResponse(
        raw_message=raw,
        text=text,
        tool_calls=tcs,
        has_tool_calls=bool(tcs),
        finish_reason="tool_calls" if tcs else "stop",
        status_text=None,
        usage=usage,
    )


class _Provider:
    """Primary provider; serves the pre-compaction flush via complete()."""

    def __init__(self, sdk: str = "openai"):
        self.provider = "opencode-go"
        self.model = "deepseek-v4-flash"
        self.sdk = sdk
        self.flushes = 0

    async def complete(self, messages, tools=None):
        self.flushes += 1
        return _response("<silent>", TokenUsage(input_tokens=10, output_tokens=1))


def _chinese_session(tmp_path) -> Session:
    """A session whose estimate is past a 128k card's threshold while the real
    prompt (~0.6 tokens per Chinese char) is well under it."""
    session = Session.new("zh_heavy", str(tmp_path))
    session.append({"role": "user", "content": "请阅读" + "中文内容" * 17_500,
                    "source": "user"})
    return session


async def _run_one_tool_turn(tmp_path, session, *, window, usage, sdk="openai",
                             on_refresh=None):
    """Drive one tool-call iteration (the compaction check runs after it),
    then a text reply. Returns (provider, call_order)."""
    mgr = ContextManager(session, _Builder(), _ctx(str(tmp_path)),
                         model_context_limit=window)
    provider = _Provider(sdk)
    registry = MagicMock()
    registry.schemas.return_value = []
    registry.is_async.return_value = False
    registry.execute.return_value = ToolResult(content="file contents")
    loop = AgentLoop(session, provider, registry, mgr,
                     on_session_end_refresh=on_refresh)

    tc = [{"id": "tc_1", "function": {"name": "read", "arguments": "{}"}}]
    responses = iter([
        _response("", usage, tool_calls=tc),
        _response("Done.", TokenUsage(input_tokens=10, output_tokens=1)),
    ])

    async def stream(context, tool_schemas):
        return next(responses)

    loop._stream_response = stream

    call_order: list[str] = []

    async def compact(sess, prov, utility_provider=None, **kwargs):
        call_order.append("compaction")

    with patch("agent_os.agent.compaction.run", new=compact), \
            patch("agent_os.agent.compaction.inject_reorientation"):
        persist_user_row(session, "go")
        await loop.run()
    return provider, call_order


# ---------------------------------------------------------------------------
# ContextManager: the reported count replaces the estimate
# ---------------------------------------------------------------------------

class TestReportedPromptTokens:

    def _mgr(self, tmp_path, window=200_000):
        session = Session.new("reported", str(tmp_path))
        session.append({"role": "user", "content": "hi", "source": "user"})
        return ContextManager(session, _Builder(), _ctx(str(tmp_path)),
                              model_context_limit=window)

    def test_reported_tokens_drive_should_compact(self, tmp_path):
        mgr = self._mgr(tmp_path)
        mgr.prepare()
        assert mgr.should_compact() is False

        mgr.record_prompt_tokens(159_999)
        assert mgr.should_compact() is False
        mgr.record_prompt_tokens(160_000)
        assert mgr.should_compact() is True

    def test_context_usage_line_uses_the_reported_tokens(self, tmp_path):
        """The runtime block's "Context usage ~X%" (and its "compacted soon"
        nudges) reads the same measure compaction triggers on. On estimate
        alone, a Chinese-heavy 128k session was told URGENT at ~46k real."""
        seen: list[float] = []

        class _Recording(_Builder):
            def build(self, context):
                seen.append(context.context_usage_pct)
                return super().build(context)

        session = _chinese_session(tmp_path)
        mgr = ContextManager(session, _Recording(), _ctx(str(tmp_path)),
                             model_context_limit=128_000)
        mgr.prepare()
        mgr.prepare()
        assert seen[-1] > 0.85  # the estimate alone: past the URGENT line

        mgr.record_prompt_tokens(54_000)
        mgr.prepare()

        assert seen[-1] == pytest.approx(54_000 / (128_000 - 20_000))

    def test_next_prepare_falls_back_to_the_estimate(self, tmp_path):
        """A call whose provider reports nothing is measured by its estimate,
        never by an older call's report."""
        mgr = self._mgr(tmp_path)
        mgr.prepare()
        mgr.record_prompt_tokens(170_000)

        mgr.prepare()

        assert 0 < mgr._last_used_tokens < 1_000
        assert mgr.should_compact() is False


# ---------------------------------------------------------------------------
# AgentLoop: hands the provider's prompt tokens to the context manager
# ---------------------------------------------------------------------------

class TestLoopCompactsOnReportedTokens:

    @pytest.mark.asyncio
    async def test_compacts_when_reported_prompt_crosses_threshold(self, tmp_path):
        """The estimate of this tiny session is far below 160k; the provider
        says the prompt was 165k, so the loop compacts."""
        session = Session.new("small", str(tmp_path))
        provider, order = await _run_one_tool_turn(
            tmp_path, session, window=200_000,
            usage=TokenUsage(input_tokens=165_000, output_tokens=50,
                             cache_read_tokens=150_000),
        )
        assert order == ["compaction"]
        assert provider.flushes == 1

    @pytest.mark.asyncio
    async def test_chinese_estimate_alone_does_not_compact(self, tmp_path):
        """128k card: the estimate is past the 102.4k threshold, the real
        prompt is ~45k. No compaction."""
        session = _chinese_session(tmp_path)
        probe = ContextManager(session, _Builder(), _ctx(str(tmp_path)),
                               model_context_limit=128_000)
        probe.prepare()
        assert probe.should_compact() is True  # what the estimate alone says

        provider, order = await _run_one_tool_turn(
            tmp_path, session, window=128_000,
            usage=TokenUsage(input_tokens=45_000, output_tokens=50),
        )
        assert order == []
        assert provider.flushes == 0

    @pytest.mark.asyncio
    @pytest.mark.parametrize("usage", [
        None,
        TokenUsage(input_tokens=0, output_tokens=0),
    ], ids=["no-usage", "zero-usage"])
    async def test_falls_back_to_estimate_without_reported_usage(self, tmp_path, usage):
        session = _chinese_session(tmp_path)
        _, order = await _run_one_tool_turn(
            tmp_path, session, window=128_000, usage=usage,
        )
        assert order == ["compaction"]

    @pytest.mark.asyncio
    @pytest.mark.parametrize("cache_write,compacts", [(10_000, True), (0, False)])
    async def test_anthropic_prompt_counts_cache_reads_and_writes(
        self, tmp_path, cache_write, compacts,
    ):
        """Anthropic's input_tokens excludes cached tokens; the prompt is
        input + cache_read + cache_write (162k or 152k against 160k)."""
        session = Session.new("anthropic", str(tmp_path))
        _, order = await _run_one_tool_turn(
            tmp_path, session, window=200_000, sdk="anthropic",
            usage=TokenUsage(input_tokens=2_000, output_tokens=50,
                             cache_read_tokens=150_000,
                             cache_write_tokens=cache_write),
        )
        assert (order == ["compaction"]) is compacts


class TestNoConsolidationBeforeCompaction:

    @pytest.mark.asyncio
    async def test_compaction_does_not_spawn_a_refresh(self, tmp_path):
        """The token-pressure consolidation is gone: compaction runs its own
        flush turn and nothing else."""
        session = Session.new("no_tp", str(tmp_path))
        refreshes: list[str] = []

        async def on_refresh(trigger_name: str):
            refreshes.append(trigger_name)

        provider, order = await _run_one_tool_turn(
            tmp_path, session, window=200_000,
            usage=TokenUsage(input_tokens=170_000, output_tokens=50),
            on_refresh=on_refresh,
        )
        assert order == ["compaction"]
        assert provider.flushes == 1
        assert refreshes == []


# ---------------------------------------------------------------------------
# The "Context usage" nudge follows the real count
# ---------------------------------------------------------------------------

def _zh_turns(session: Session, rows: int, chars: int = 2_000) -> None:
    session.append({"role": "user", "content": "任务：整理所有文档的要点。", "source": "user"})
    for n in range(rows):
        session.append({"role": "assistant", "source": "management",
                        "content": f"第{n}部分：" + "中文内容" * (chars // 4)})


def _all_text(messages: list[dict]) -> str:
    return "\n".join(str(m.get("content")) for m in messages)


def test_no_urgent_nudge_when_only_the_estimate_is_high(tmp_path):
    """Chinese history: the estimate says >85% of a 128k card, the provider
    says ~45%. The URGENT "save state, compaction soon" line must not show."""
    from agent_os.agent.prompt_builder import PromptBuilder

    session = Session.new("zh_nudge", str(tmp_path))
    _zh_turns(session, rows=40)
    mgr = ContextManager(session, PromptBuilder(str(tmp_path)), _ctx(str(tmp_path)),
                         model_context_limit=128_000)
    mgr.prepare()
    on_estimate = mgr.prepare()
    assert "URGENT" in _all_text(on_estimate)  # the estimate alone trips it

    mgr.record_prompt_tokens(48_000)
    on_real = mgr.prepare()

    assert "URGENT" not in _all_text(on_real)


# ---------------------------------------------------------------------------
# Sliding window calibrated to real tokens (spec 086 §8.3)
# ---------------------------------------------------------------------------

class TestCalibratedWindow:

    def test_window_keeps_the_task_once_calibrated(self, tmp_path):
        """128k card, ~120k estimated / ~48k real Chinese history. On the raw
        estimate get_recent drops the oldest rows, the user's task first;
        calibrated by the provider's count, everything fits."""
        session = Session.new("zh_window", str(tmp_path))
        _zh_turns(session, rows=40)
        mgr = ContextManager(session, _Builder(), _ctx(str(tmp_path)),
                             model_context_limit=128_000)
        first = mgr.prepare()
        assert "任务：整理所有文档的要点。" not in _all_text(first)

        mgr.record_prompt_tokens(int(mgr._last_estimated_tokens / 2.5))
        second = mgr.prepare()

        assert "任务：整理所有文档的要点。" in _all_text(second)
        assert sum(1 for m in second if m.get("role") == "assistant") == 40

    def test_ratio_counts_the_tool_schema_overhead(self, tmp_path):
        """Tool schemas ride every request but are not in the prepared
        messages. Left out, a short prompt reads as ~0.3 estimate/real and
        would shrink the window to a third."""
        session = Session.new("overhead", str(tmp_path))
        session.append({"role": "user", "content": "x" * 4_000, "source": "user"})
        mgr = ContextManager(session, _Builder(), _ctx(str(tmp_path)),
                             model_context_limit=128_000)
        mgr.prepare()
        est = mgr._last_estimated_tokens

        mgr.record_prompt_tokens(int(est + 7_000), overhead_estimate=7_000)

        assert mgr._estimate_ratio == pytest.approx(1.0, abs=0.01)

    @pytest.mark.parametrize("real,ratio", [(1, 4.0), (10_000_000, 0.25)])
    def test_ratio_is_clamped(self, tmp_path, real, ratio):
        session = Session.new("clamp", str(tmp_path))
        session.append({"role": "user", "content": "x" * 4_000, "source": "user"})
        mgr = ContextManager(session, _Builder(), _ctx(str(tmp_path)),
                             model_context_limit=128_000)
        mgr.prepare()
        mgr.record_prompt_tokens(real)
        assert mgr._estimate_ratio == ratio

    def test_keep_budget_is_a_share_of_the_window_in_estimate_units(self, tmp_path):
        session = Session.new("keep", str(tmp_path))
        session.append({"role": "user", "content": "x" * 4_000, "source": "user"})
        mgr = ContextManager(session, _Builder(), _ctx(str(tmp_path)),
                             model_context_limit=128_000)
        assert mgr.compaction_keep_tokens() == int(0.30 * 128_000)
        mgr.prepare()
        mgr.record_prompt_tokens(int(mgr._last_estimated_tokens / 2))
        assert mgr.compaction_keep_tokens() == pytest.approx(0.30 * 128_000 * 2, rel=0.01)


class TestShouldCompactProjection:

    def _loaded(self, tmp_path, reported):
        session = Session.new("proj", str(tmp_path))
        session.append({"role": "user", "content": "go", "source": "user"})
        mgr = ContextManager(session, _Builder(), _ctx(str(tmp_path)),
                             model_context_limit=128_000)
        mgr.prepare()
        mgr.record_prompt_tokens(reported)
        return session, mgr

    def test_counts_what_was_appended_since_the_call(self, tmp_path):
        """100k reported, then a ~12k-token read lands: the next prompt is
        ~112k, past the 102.4k trigger. Compact now, before the sliding
        window has to cut the oldest rows (the user's task) to fit."""
        session, mgr = self._loaded(tmp_path, 100_000)
        assert mgr.should_compact() is False
        mgr._estimate_ratio = 1.0
        session.append({"role": "tool", "tool_call_id": "t1", "content": "y" * 48_000})
        assert mgr.should_compact() is True

    def test_no_second_compaction_on_the_very_next_check(self, tmp_path, caplog):
        session, mgr = self._loaded(tmp_path, 110_000)
        assert mgr.should_compact() is True
        mgr.note_compacted()

        mgr.prepare()
        mgr.record_prompt_tokens(105_000)  # kept tail alone still over 102.4k
        with caplog.at_level("WARNING"):
            assert mgr.should_compact() is False
        assert "still over the compaction threshold" in caplog.text

        mgr.prepare()
        mgr.record_prompt_tokens(106_000)
        assert mgr.should_compact() is True


# ---------------------------------------------------------------------------
# End to end through the loop: append-only compaction, then the next prompt
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_loop_compacts_append_only_and_resumes_from_the_summary(tmp_path):
    import json as _json

    session = Session.new("e2e", str(tmp_path))
    session.append({"role": "user", "content": "earlier question", "source": "user"})
    session.append({"role": "assistant", "content": "earlier answer", "source": "management"})
    persist_user_row(session, "Read the docs and list the code words.")
    for n in range(6):
        tc = f"tc_{n}"
        session.append({"role": "assistant", "content": None, "source": "management",
                        "tool_calls": [{"id": tc, "type": "function",
                                        "function": {"name": "read", "arguments": "{}"}}]})
        session.append_tool_result(tc, f"doc {n}\n" + "x" * 20_000)
    rows_before = len(session.get_messages())

    mgr = ContextManager(session, _Builder(), _ctx(str(tmp_path)),
                         model_context_limit=128_000)
    replies = iter(["<silent>", "Read docs 0-4. Code words: A, B, C, D, E."])

    class _P(_Provider):
        async def complete(self, messages, tools=None):
            self.flushes += 1
            return _response(next(replies), TokenUsage(input_tokens=10, output_tokens=1))

    provider = _P()
    registry = MagicMock()
    registry.schemas.return_value = []
    registry.is_async.return_value = False
    registry.execute.return_value = ToolResult(content="doc 6\n" + "x" * 20_000)
    loop = AgentLoop(session, provider, registry, mgr)

    seen_contexts: list[list[dict]] = []
    tc = [{"id": "tc_6", "function": {"name": "read", "arguments": "{}"}}]
    responses = iter([
        _response("", TokenUsage(input_tokens=110_000, output_tokens=5), tool_calls=tc),
        _response("Done.", TokenUsage(input_tokens=30_000, output_tokens=5)),
    ])

    async def stream(context, tool_schemas):
        seen_contexts.append(context)
        return next(responses)

    loop._stream_response = stream
    with patch("agent_os.agent.compaction.inject_reorientation"):
        await loop.run()

    with open(session._filepath, encoding="utf-8") as f:
        on_disk = [_json.loads(line) for line in f if line.strip()]
    messages = [r for r in on_disk if r.get("role") != "meta"]
    assert messages[0]["content"] == "earlier question"
    assert sum(1 for r in messages if r.get("role") == "tool") == 7
    assert len(messages) > rows_before
    assert sum(1 for r in messages if r.get("_compaction")) == 1

    history = [m for m in seen_contexts[1]
               if m.get("content") not in ("cached-system-prefix", "semi-stable-suffix")]
    assert history[0]["content"] == "Read docs 0-4. Code words: A, B, C, D, E."
    assert history[1]["content"] == "Read the docs and list the code words."
    assert "earlier question" not in _all_text(seen_contexts[1])


@pytest.mark.asyncio
@pytest.mark.parametrize("reported,new_result_chars,flushed", [
    (104_000, 400, True),       # next prompt ~104k: fits 108k, flush sees it all
    (104_000, 40_000, False),   # a 10k-token read lands: ~114k would be trimmed
])
async def test_flush_only_when_its_prompt_fits_untrimmed(
    tmp_path, reported, new_result_chars, flushed,
):
    """The pre-compaction flush asks the agent to save its state. When the
    step that crossed the threshold also overflows the window, the flush
    prompt is cut by the sliding window, oldest rows (the task) first; in a
    live run the agent then saved "awaiting user instruction". Skip the flush
    there: compaction keeps the task verbatim anyway."""
    session = Session.new("flush_fit", str(tmp_path))
    persist_user_row(session, "Read the docs.")
    mgr = ContextManager(session, _Builder(), _ctx(str(tmp_path)),
                         model_context_limit=128_000)
    requests: list[str] = []

    class _P(_Provider):
        async def complete(self, messages, tools=None):
            requests.append(messages[0]["content"])
            return _response("<silent>" if len(requests) == 1 and flushed
                             else "Summary of the reads.",
                             TokenUsage(input_tokens=10, output_tokens=1))

    registry = MagicMock()
    registry.schemas.return_value = []
    registry.is_async.return_value = False
    registry.execute.return_value = ToolResult(content="z" * new_result_chars)
    loop = AgentLoop(session, _P(), registry, mgr)
    tc = [{"id": "tc_1", "function": {"name": "read", "arguments": "{}"}}]
    responses = iter([
        _response("", TokenUsage(input_tokens=reported, output_tokens=5), tool_calls=tc),
        _response("Done.", TokenUsage(input_tokens=30_000, output_tokens=5)),
    ])

    async def stream(context, tool_schemas):
        return next(responses)

    loop._stream_response = stream
    compactions: list[str] = []

    async def compact(sess, prov, utility_provider=None, **kwargs):
        compactions.append("compaction")

    with patch("agent_os.agent.compaction.run", new=compact), \
            patch("agent_os.agent.compaction.inject_reorientation"):
        await loop.run()

    from agent_os.agent.compaction import MEMORY_FLUSH_PROMPT
    flush_rows = [m for m in session.get_messages() if m.get("content") == MEMORY_FLUSH_PROMPT]
    assert compactions == ["compaction"]
    assert bool(flush_rows) is flushed
    assert (len(requests) == 1) is flushed
