# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Pinned regression for Budget Piece 3 — Task A (guard isolation).

THE invariant this piece exists to protect, BEFORE sub-agent usage capture
(P3-B) plants real ``subagent:*`` ledger lines:

    Sub-agent token usage is captured to the ledger (for the display view) but
    must NEVER count toward budget ENFORCEMENT — it must not trip the pre-flight
    guard, must not fire a threshold/trip relay push, and must not appear in the
    guard's converted spend.

Mechanism: ``spend()`` gained a ``sources`` filter; the guard
(``budget/guard.py::evaluate_budget``) and the threshold detector
(``budget/notify.py::maybe_emit_budget_event``) pass ``["management"]``. The
trip-emission site (``maybe_emit_trip`` from ``AgentLoop._trip_budget``) carries
the guard's already-filtered numbers, so it inherits the isolation transitively.

The ledger schema accepts ANY ``source`` string on read, so we plant
``subagent:claude-code`` / ``subagent:codex`` lines by hand (the writer only ever
emits ``"management"`` today; subagent values are reserved for P3-B).

Rates here are the real moonshot defaults (kimi: input 1.0 CNY/1M, output
3.0 CNY/1M, currency CNY), so a CNY-denominated limit makes spend deterministic
without an FX rate. Limit = 1.0 CNY; planted subagent spend = 10× the limit.
"""

from __future__ import annotations

import json
import os

import pytest

from agent_os.agent.loop import AgentLoop
from agent_os.agent.providers.types import StreamChunk, TokenUsage
from agent_os.agent.session import Session, persist_user_row
from agent_os.budget import ledger as ledger_mod
from agent_os.budget.guard import evaluate_budget
from agent_os.budget.ledger import (
    LedgerEvent,
    SOURCE_MANAGEMENT,
    ledger_path,
    spend,
)
from agent_os.budget.normalize import NormalizedUsage
from agent_os.budget.notify import (
    EVENT_THRESHOLD,
    EVENT_TRIP,
    maybe_emit_budget_event,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _plant(workspace, *, source, uncached_input=0, output=0,
           provider="moonshot", model="kimi"):
    """Append ONE real ledger line with an arbitrary source (hand-written JSONL).

    Uses ``append_event`` so the on-disk shape is byte-identical to the real
    writer — only ``source`` is overridden (the schema allows any source string
    on read; subagent values are reserved for P3-B and not emitted yet)."""
    ledger_mod.append_event(
        workspace,
        LedgerEvent(
            session_id="sess",
            source=source,
            provider=provider,
            model=model,
            usage=NormalizedUsage(
                uncached_input=uncached_input,
                cache_read=0,
                cache_write=0,
                output=output,
            ),
        ),
    )


def _ledger_lines(workspace):
    path = ledger_path(workspace)
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


# Limit owns its currency (legacy field name budget_limit_usd carries the VALUE;
# the currency is budget_currency). moonshot prices in CNY, so a CNY limit makes
# spend convertible with no FX rate.
def _cfg(limit=1.0):
    return {
        "budget_limit_usd": limit,   # field name is legacy; VALUE is in CNY
        "budget_action": "pause",
        "budget_period": "total",
        "budget_currency": "CNY",
        "fx_rates": {},
    }


def _plant_subagent_10x(workspace):
    """Plant subagent ledger spend worth 10× the 1.0 CNY limit.

    5 CNY (5M uncached input @ 1.0 CNY/1M) on subagent:claude-code +
    5 CNY (5M uncached input)            on subagent:codex  = 10 CNY total."""
    _plant(workspace, source="subagent:claude-code", uncached_input=5_000_000)
    _plant(workspace, source="subagent:codex", uncached_input=5_000_000)


# ---------------------------------------------------------------------------
# THE pinned invariant: 10× subagent spend does NOT trip / fire / show
# ---------------------------------------------------------------------------

class TestSubagentSpendDoesNotEnforce:
    def test_guard_not_tripped_by_subagent_spend(self, tmp_path):
        ws = str(tmp_path)
        _plant_subagent_10x(ws)  # 10× the limit, all subagent
        decision = evaluate_budget(ws, _cfg(limit=1.0))
        assert decision.tripped is False, (
            "subagent spend at 10× the limit must NOT trip the guard"
        )
        # And the guard's converted spend EXCLUDES the subagent rows entirely.
        assert decision.spend == 0.0, (
            "guard spend must be 0 — no management lines, subagent excluded"
        )

    def test_notify_does_not_fire_on_subagent_spend(self, tmp_path):
        """Drive the REAL threshold/trip detector (real spend, real marker).

        10× the limit in subagent rows must fire NEITHER threshold (80%) NOR
        trip (100%) — the detector filters to management-only spend."""
        ws = str(tmp_path)
        _plant_subagent_10x(ws)
        events = []
        marker = str(tmp_path / "notify_state.json")
        maybe_emit_budget_event(
            ws, _cfg(limit=1.0), emit=events.append, marker_path=marker,
        )
        assert events == [], (
            "no threshold/trip event may fire on subagent-only spend"
        )
        # No marker written either (nothing crossed) — re-eval stays silent.
        assert not os.path.exists(marker)

    def test_subagent_rows_absent_from_guard_spend_but_present_in_display(
        self, tmp_path,
    ):
        """The display view (sources=None) still aggregates everything; the
        guard/notify view (sources=["management"]) sees none of it."""
        ws = str(tmp_path)
        _plant_subagent_10x(ws)
        # Display: GET /cost behavior — everything is present.
        display = spend(ws, "total", target_currency="CNY")
        display_sources = sorted(b["source"] for b in display["breakdown"])
        assert display_sources == ["subagent:claude-code", "subagent:codex"]
        assert display["by_currency"]["CNY"] == pytest.approx(10.0)
        # Guard/notify: management-only — empty.
        guard_view = spend(ws, "total", target_currency="CNY",
                           sources=["management"])
        assert guard_view["breakdown"] == []
        assert guard_view["converted_total"]["amount"] == 0.0

    @pytest.mark.asyncio
    async def test_real_loop_does_not_block_on_subagent_spend(self, tmp_path):
        """End-to-end: a real loop on a project with 10× subagent spend runs the
        management LLM (the guard does NOT pre-empt it) and exits cleanly — no
        budget_blocked, and the on_budget_event sink stays empty."""
        ws = str(tmp_path)
        _plant_subagent_10x(ws)
        events = []
        loop = _build_loop(ws, _cfg(limit=1.0), on_budget_event=events.append)
        persist_user_row(loop._session, "hi")
        await loop.run()
        # The loop completed a normal response — NOT a budget trip.
        assert loop.get_completion_state()[0] != "budget_blocked"
        roles = [m.get("role") for m in loop._session.get_messages()]
        assert "assistant" in roles, roles
        # No threshold/trip push fired from the subagent spend. (P3-C's
        # budget.spend_updated broadcast MAY fire — it is display freshness,
        # not a push — and its spend must be management-only, i.e. the 10x
        # subagent total must not leak into it.)
        kinds = [e.get("type") for e in events]
        assert "budget.threshold" not in kinds
        assert "budget.exhausted" not in kinds
        for e in events:
            if e.get("type") == "budget.spend_updated":
                assert e.get("spend", 0.0) < 1.0, e


# ---------------------------------------------------------------------------
# The filter does NOT break enforcement: management spend over limit trips.
# ---------------------------------------------------------------------------

class TestManagementSpendStillEnforces:
    def test_management_over_limit_still_trips(self, tmp_path):
        ws = str(tmp_path)
        _plant_subagent_10x(ws)  # noise: 10× subagent (must be ignored)
        # 2M uncached input @ 1.0 CNY/1M = 2.0 CNY > 1.0 CNY limit.
        _plant(ws, source=SOURCE_MANAGEMENT, uncached_input=2_000_000)
        decision = evaluate_budget(ws, _cfg(limit=1.0))
        assert decision.tripped is True, (
            "management spend over the limit must STILL trip — filter must not "
            "break enforcement"
        )
        # The spend is the MANAGEMENT 2.0 CNY only — subagent's 10 CNY excluded.
        assert decision.spend == pytest.approx(2.0)
        assert decision.limit == pytest.approx(1.0)

    def test_notify_fires_on_management_over_limit(self, tmp_path):
        ws = str(tmp_path)
        _plant_subagent_10x(ws)  # noise
        _plant(ws, source=SOURCE_MANAGEMENT, uncached_input=2_000_000)  # 2.0 CNY
        events = []
        marker = str(tmp_path / "notify_state.json")
        maybe_emit_budget_event(
            ws, _cfg(limit=1.0), emit=events.append, marker_path=marker,
        )
        types = [e["type"] for e in events]
        # 200% of limit → both the 80% threshold AND the 100% trip fire.
        assert EVENT_THRESHOLD in types
        assert EVENT_TRIP in types
        # Every emitted spend reflects MANAGEMENT only (2.0), never +10 subagent.
        for e in events:
            assert e["spend"] == pytest.approx(2.0)
            assert e["limit"] == pytest.approx(1.0)

    @pytest.mark.asyncio
    async def test_real_loop_blocks_on_management_over_limit(self, tmp_path):
        """End-to-end: management spend over limit (with subagent noise present)
        still trips the real loop's pre-flight guard → no LLM call, exit
        budget_blocked."""
        ws = str(tmp_path)
        _plant_subagent_10x(ws)  # noise
        _plant(ws, source=SOURCE_MANAGEMENT, uncached_input=2_000_000)  # 2.0 CNY
        events = []
        calls = {"n": 0}

        class _RecordingProvider:
            sdk = "openai"
            provider = "moonshot"
            model = "kimi"

            async def stream(self, context, tools=None):
                calls["n"] += 1
                if False:
                    yield  # pragma: no cover — make this an async generator

        loop = _build_loop(
            ws, _cfg(limit=1.0), on_budget_event=events.append,
            provider=_RecordingProvider(),
        )
        persist_user_row(loop._session, "do expensive work")
        await loop.run()
        assert calls["n"] == 0, "no LLM call when management spend is over limit"
        assert loop.get_completion_state()[0] == "budget_blocked"


# ---------------------------------------------------------------------------
# Loop wiring (mirrors test_budget_notify_p2e / test_budget_enforcement_p2c)
# ---------------------------------------------------------------------------

class _Builder:
    def build(self, context):
        return ("sys", "suffix", "dynamic")


class _Registry:
    def schemas(self):
        return []

    def is_async(self, name):
        return False

    def execute(self, name, arguments):
        from agent_os.agent.tools.base import ToolResult
        return ToolResult(content="ok")

    async def execute_async(self, name, arguments):
        from agent_os.agent.tools.base import ToolResult
        return ToolResult(content="ok")

    def tool_names(self):
        return []

    def reset_run_state(self):
        pass


class _TextProvider:
    """Single-turn text provider with a tiny usage (one management ledger line).

    Its usage is small enough that even after appending, management spend stays
    well UNDER the 1.0 CNY limit — so the isolation test exercises a project that
    is under-limit on management but 10× over on subagent."""

    provider = "moonshot"
    model = "kimi"
    sdk = "openai"

    async def stream(self, messages, tools=None):
        yield StreamChunk(text="hi")
        # 100 in / 50 out tokens → ~0.00025 CNY: negligible vs the 1.0 limit.
        yield StreamChunk(
            is_final=True,
            usage=TokenUsage(input_tokens=100, output_tokens=50),
        )


def _build_loop(workspace, budget_config, *, on_budget_event, provider=None):
    from agent_os.agent.context import ContextManager
    from agent_os.agent.prompt_builder import Autonomy, PromptContext

    session = Session.new(
        "p3a_iso", workspace, session_id="p3a_iso",
        provider="moonshot", model="kimi", sdk="openai",
    )
    if provider is None:
        provider = _TextProvider()
    ctx = PromptContext(
        workspace=workspace,
        model="kimi",
        autonomy=Autonomy.HANDS_OFF,
        enabled_agents=[],
        tool_names=[],
        os_type="linux",
        datetime_now="2026-01-01T00:00:00",
        context_usage_pct=0.0,
    )
    context_mgr = ContextManager(session, _Builder(), ctx)

    def get_budget_config():
        return budget_config

    loop = AgentLoop(
        session, provider, _Registry(), context_mgr,
        project_dir=workspace,
        get_budget_config=get_budget_config,
        on_budget_event=on_budget_event,
        max_iterations=5,
    )
    return loop
