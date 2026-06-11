# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Regression: the budget threshold/trip emitter is wired into the REAL loop
ledger-append path and routes codes-only events through the emit sink
(Budget Piece 2 — Task E, the E17 fix).

The unit tests in ``tests/unit/test_budget_notify.py`` pin the emitter's
once-per-window / re-arm / corruption / payload-shape semantics in isolation.
These exercise the LIVE wiring: a real ``AgentLoop.run`` writes a real ledger,
the loop calls the emitter after each append, and a real ``on_budget_event``
sink collects the forwarded events. They also pin the no-relay no-op and the
relay-client Tier-1 push routing for both event types.
"""

import pytest

from agent_os.agent.session import Session
from agent_os.agent.loop import AgentLoop
from agent_os.agent.context import ContextManager
from agent_os.agent.providers.types import StreamChunk, TokenUsage
from agent_os.agent.tools.base import ToolResult
from agent_os.agent.prompt_builder import PromptContext, Autonomy
from agent_os.budget.notify import EVENT_THRESHOLD, EVENT_TRIP


def _ctx(workspace: str) -> PromptContext:
    return PromptContext(
        workspace=workspace,
        model="test-model",
        autonomy=Autonomy.HANDS_OFF,
        enabled_agents=[],
        tool_names=[],
        os_type="linux",
        datetime_now="2026-01-01T00:00:00",
        context_usage_pct=0.0,
    )


class _Builder:
    def build(self, context):
        return ("sys", "suffix", "dynamic")


class _Registry:
    def schemas(self):
        return []

    def is_async(self, name):
        return False

    def execute(self, name, arguments):
        return ToolResult(content="ok")

    async def execute_async(self, name, arguments):
        return ToolResult(content="ok")

    def tool_names(self):
        return []

    def reset_run_state(self):
        pass


class _TextProvider:
    """Single-turn text provider with a large output so spend is non-trivial."""

    def __init__(self, output_tokens=50):
        self.provider = "moonshot"
        self.model = "kimi-k2.5"
        self.sdk = "openai"
        self._usage = TokenUsage(input_tokens=100, output_tokens=output_tokens)

    async def stream(self, messages, tools=None):
        yield StreamChunk(text="hi")
        yield StreamChunk(is_final=True, usage=self._usage)


def _build_loop(tmp_path, *, budget_config, on_budget_event, output_tokens=50):
    session = Session.new("notify_wire", str(tmp_path))
    provider = _TextProvider(output_tokens=output_tokens)
    context_mgr = ContextManager(session, _Builder(), _ctx(str(tmp_path)))

    def get_budget_config():
        return budget_config

    loop = AgentLoop(
        session, provider, _Registry(), context_mgr,
        project_dir=str(tmp_path),
        get_budget_config=get_budget_config,
        on_budget_event=on_budget_event,
        max_iterations=5,
    )
    return loop


def _stub_spend(amount):
    """Patch the ledger.spend symbol notify imports so spend is deterministic
    (decoupled from the real rate table)."""
    def _spend(project_dir, window, *, target_currency=None, fx_rates=None,
               now=None, anchor_ts=None, sources=None):
        return {
            "window": window,
            "by_currency": {},
            "converted_total": {
                "currency": target_currency or "USD",
                "amount": amount,
                "estimated": True,
            },
            "breakdown": [],
        }
    return _spend


@pytest.mark.asyncio
async def test_loop_emits_threshold_after_crossing(tmp_path, monkeypatch):
    """A spend that crosses 80% of the limit fires a codes-only threshold event
    through the loop's on_budget_event sink — exactly once. Spend is stubbed so
    the boundary is deterministic (not coupled to the live rate table)."""
    events = []
    cfg = {
        "budget_limit_usd": 10.0,
        "budget_period": "daily",
        "budget_currency": "USD",
        "budget_action": "pause",
        "fx_rates": {},
    }
    # 8.5 / 10 = 85% → crosses 80% but under 100% → threshold ONLY.
    monkeypatch.setattr("agent_os.budget.ledger.spend", _stub_spend(8.5))
    loop = _build_loop(tmp_path, budget_config=cfg, on_budget_event=events.append)
    await loop.run(initial_message="hi")

    types = [e["type"] for e in events]
    assert EVENT_THRESHOLD in types
    assert EVENT_TRIP not in types  # under 100%
    assert types.count(EVENT_THRESHOLD) == 1
    ev = events[0]
    assert ev["pct"] == 80
    assert ev["spend"] == 8.5
    assert ev["limit"] == 10.0
    assert ev["currency"] == "USD"
    assert ev["window"] == "daily"
    # Codes/numbers only — no display strings (no spaces in any string field).
    for e in events:
        for k, v in e.items():
            if isinstance(v, str):
                assert " " not in v, f"{k}={v!r} looks like a display string"


@pytest.mark.asyncio
async def test_loop_no_limit_emits_nothing(tmp_path):
    """No limit configured → the emitter never fires even though spend is recorded."""
    events = []
    cfg = {
        "budget_limit_usd": None,
        "budget_period": "daily",
        "budget_currency": "USD",
        "budget_action": "pause",
        "fx_rates": {},
    }
    loop = _build_loop(tmp_path, budget_config=cfg, on_budget_event=events.append)
    await loop.run(initial_message="hi")
    assert events == []


@pytest.mark.asyncio
async def test_loop_no_sink_is_silent_noop(tmp_path, monkeypatch):
    """on_budget_event=None (relay absent) → loop runs, ledger written, no crash."""
    cfg = {
        "budget_limit_usd": 10.0,
        "budget_period": "daily",
        "budget_currency": "USD",
        "budget_action": "pause",
        "fx_rates": {},
    }
    # Spend at 85% (under the limit) so the pre-flight guard does NOT trip and
    # the loop completes a normal response (writing a ledger line). The 80%
    # threshold WOULD cross, but the None sink means nothing is emitted.
    monkeypatch.setattr("agent_os.budget.ledger.spend", _stub_spend(8.5))
    loop = _build_loop(tmp_path, budget_config=cfg, on_budget_event=None)
    # Must not raise.
    await loop.run(initial_message="hi")
    # Ledger still written (the append path is independent of the emitter).
    from agent_os.budget.ledger import ledger_path
    import os
    roles = [m.get("role") for m in loop._session.get_messages()]
    assert "assistant" in roles, roles
    assert os.path.isfile(ledger_path(str(tmp_path)))


@pytest.mark.asyncio
async def test_loop_threshold_fires_once_across_two_responses(tmp_path, monkeypatch):
    """Two management responses in one window cross 80% on the first; the second
    append must NOT re-fire the threshold (once-per-window, live wiring)."""
    events = []
    cfg = {
        "budget_limit_usd": 10.0,
        "budget_period": "daily",
        "budget_currency": "USD",
        "budget_action": "pause",
        "fx_rates": {},
    }
    # Both appends see the same 85% spend → threshold fires once, never trips.
    monkeypatch.setattr("agent_os.budget.ledger.spend", _stub_spend(8.5))
    session = Session.new("notify_twice", str(tmp_path))

    class _TwoTurn:
        provider = "moonshot"
        model = "kimi-k2.5"
        sdk = "openai"

        def __init__(self):
            self._calls = 0

        async def stream(self, messages, tools=None):
            self._calls += 1
            if self._calls == 1:
                # First turn: a tool call so the loop continues to a 2nd LLM call.
                yield StreamChunk(tool_calls_delta=[{
                    "index": 0, "id": "t1", "type": "function",
                    "function": {"name": "noop", "arguments": "{}"},
                }])
                yield StreamChunk(is_final=True,
                                  usage=TokenUsage(input_tokens=100, output_tokens=50))
            else:
                yield StreamChunk(text="done")
                yield StreamChunk(is_final=True,
                                  usage=TokenUsage(input_tokens=80, output_tokens=30))

    provider = _TwoTurn()
    context_mgr = ContextManager(session, _Builder(), _ctx(str(tmp_path)))

    class _NoopRegistry(_Registry):
        def schemas(self):
            return [{"name": "noop"}]

    loop = AgentLoop(
        session, provider, _NoopRegistry(), context_mgr,
        project_dir=str(tmp_path),
        get_budget_config=lambda: cfg,
        on_budget_event=events.append,
        max_iterations=5,
    )
    await loop.run(initial_message="hi")

    # Two ledger appends happened (two responses with usage), but the threshold
    # fired only ONCE across the window, and never tripped (85% < 100%).
    types = [e["type"] for e in events]
    assert types.count(EVENT_THRESHOLD) == 1, types
    assert EVENT_TRIP not in types, types


@pytest.mark.asyncio
async def test_loop_trip_emitted_via_ledger_append(tmp_path, monkeypatch):
    """The trip event fires on the ledger append of the response that crosses
    100% — BEFORE the next iteration's pre-flight guard trips the loop. Pins the
    ledger-append placement for the trip (not just the threshold).

    Turn 1 response appends at 85% (threshold fires, no trip, guard stays open).
    Turn 2 response appends at 110% (trip fires). The NEXT pre-flight guard then
    trips and ends the run — but the trip event was already emitted at append.
    """
    events = []
    cfg = {
        "budget_limit_usd": 10.0,
        "budget_period": "daily",
        "budget_currency": "USD",
        "budget_action": "pause",
        "fx_rates": {},
    }

    # Spend climbs across the run: the pre-flight guard and the notifier share
    # this same stub. Sequence of returned amounts as spend() is called:
    #   guard(pre-turn1)=0 → turn1 append notify=8.5 → guard(pre-turn2)=8.5
    #   → turn2 append notify=11.0 → guard(pre-turn3)=11.0 TRIPS.
    amounts = iter([0.0, 8.5, 8.5, 11.0, 11.0, 11.0])

    def _climbing_spend(project_dir, window, *, target_currency=None,
                        fx_rates=None, now=None, anchor_ts=None, sources=None):
        try:
            amt = next(amounts)
        except StopIteration:
            amt = 11.0
        return {
            "window": window,
            "by_currency": {},
            "converted_total": {"currency": target_currency or "USD",
                                "amount": amt, "estimated": True},
            "breakdown": [],
        }

    monkeypatch.setattr("agent_os.budget.ledger.spend", _climbing_spend)
    session = Session.new("notify_trip", str(tmp_path))

    class _TwoTurn:
        provider = "moonshot"
        model = "kimi-k2.5"
        sdk = "openai"

        def __init__(self):
            self._calls = 0

        async def stream(self, messages, tools=None):
            self._calls += 1
            if self._calls == 1:
                yield StreamChunk(tool_calls_delta=[{
                    "index": 0, "id": "t1", "type": "function",
                    "function": {"name": "noop", "arguments": "{}"},
                }])
                yield StreamChunk(is_final=True,
                                  usage=TokenUsage(input_tokens=100, output_tokens=50))
            else:
                yield StreamChunk(text="done")
                yield StreamChunk(is_final=True,
                                  usage=TokenUsage(input_tokens=80, output_tokens=30))

    context_mgr = ContextManager(session, _Builder(), _ctx(str(tmp_path)))

    class _NoopRegistry(_Registry):
        def schemas(self):
            return [{"name": "noop"}]

    loop = AgentLoop(
        session, _TwoTurn(), _NoopRegistry(), context_mgr,
        project_dir=str(tmp_path),
        get_budget_config=lambda: cfg,
        on_budget_event=events.append,
        max_iterations=10,
    )
    await loop.run(initial_message="hi")

    types = [e["type"] for e in events]
    assert types.count(EVENT_THRESHOLD) == 1, types
    assert types.count(EVENT_TRIP) == 1, types
    # The trip event carries the over-limit spend recorded at append time.
    trip = [e for e in events if e["type"] == EVENT_TRIP][0]
    assert trip["spend"] == 11.0
    assert trip["limit"] == 10.0


# ---------------------------------------------------------------------------
# Relay-client push routing: both event types are Tier-1.
# ---------------------------------------------------------------------------

def _relay_client():
    from agent_os.relay.client import RelayClient
    return RelayClient(
        relay_url="https://relay.example",
        device_id="dev",
        device_secret="secret",
    )


def test_relay_threshold_is_tier1_push():
    client = _relay_client()
    push = client._should_push(
        {"type": EVENT_THRESHOLD, "pct": 80, "spend": 8.5, "limit": 10.0,
         "currency": "USD", "window": "daily"},
        "proj1",
    )
    assert push is not None
    assert "title" in push and "body" in push


def test_relay_trip_is_tier1_push():
    client = _relay_client()
    push = client._should_push(
        {"type": EVENT_TRIP, "spend": 12.0, "limit": 10.0, "currency": "USD",
         "window": "daily"},
        "proj1",
    )
    assert push is not None
    assert "title" in push and "body" in push


def test_relay_threshold_push_not_rate_limited():
    """Tier-1 budget pushes bypass the per-project sliding-window rate limit
    (mirrors approval.request / budget.exhausted)."""
    client = _relay_client()
    # Even after many calls, a Tier-1 budget event still pushes.
    for _ in range(10):
        push = client._should_push(
            {"type": EVENT_THRESHOLD, "pct": 80}, "proj1",
        )
        assert push is not None


# ---------------------------------------------------------------------------
# Guard-trip emission site (coordinator ruling): the trip fires at whichever
# happens first — crossing append OR guard trip — exactly once per window.
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_guard_trip_no_append_emits_trip_once(tmp_path, monkeypatch):
    """Fresh run on an already-over-limit project, NO marker file and NO new
    append: the pre-flight guard trips before any LLM call, and the trip event
    fires from the guard-trip site — exactly once across repeated runs."""
    events = []
    cfg = {
        "budget_limit_usd": 10.0,
        "budget_period": "daily",
        "budget_currency": "USD",
        "budget_action": "pause",
        "fx_rates": {},
    }
    # Already over: 12 >= 10 from the very first guard evaluation.
    monkeypatch.setattr("agent_os.budget.ledger.spend", _stub_spend(12.0))

    loop = _build_loop(tmp_path, budget_config=cfg, on_budget_event=events.append)
    await loop.run(initial_message="hi")

    # Guard tripped pre-flight: budget_blocked exit, no LLM call, no append.
    assert loop.get_completion_state()[0] == "budget_blocked"
    from agent_os.budget.ledger import ledger_path
    import os
    assert not os.path.isfile(ledger_path(str(tmp_path)))  # no append happened

    types = [e["type"] for e in events]
    assert types == [EVENT_TRIP], types  # trip once; threshold is append-only
    assert events[0]["spend"] == 12.0
    assert events[0]["limit"] == 10.0

    # Second run, FRESH loop objects, same project (same marker on disk):
    # the guard trips again but the trip must NOT re-fire this window.
    loop2 = _build_loop(tmp_path, budget_config=cfg, on_budget_event=events.append)
    await loop2.run(initial_message="hi again")
    assert loop2.get_completion_state()[0] == "budget_blocked"
    assert [e["type"] for e in events] == [EVENT_TRIP]  # still exactly one


@pytest.mark.asyncio
async def test_append_trip_then_guard_trip_no_refire(tmp_path, monkeypatch):
    """Crossing append fired the trip earlier in the window → the subsequent
    guard trip (next pre-flight evaluation) does NOT re-fire (marker dedup).

    This is the loop-level pin of the shared-marker dedup; spend climbs across
    the run so the append observes the crossing first, then the guard trips.
    (test_loop_trip_emitted_via_ledger_append already runs this sequence; this
    test additionally asserts the guard-trip site was REACHED and dedup'd by
    checking the loop exited budget_blocked after the trip event count froze.)
    """
    events = []
    cfg = {
        "budget_limit_usd": 10.0,
        "budget_period": "daily",
        "budget_currency": "USD",
        "budget_action": "pause",
        "fx_rates": {},
    }
    # guard(pre-turn1)=0 → turn1 append=11.0 (trip fires at append)
    # → guard(pre-turn2)=11.0 TRIPS (guard-trip site runs → must dedup).
    amounts = iter([0.0, 11.0])

    def _climbing_spend(project_dir, window, *, target_currency=None,
                        fx_rates=None, now=None, anchor_ts=None, sources=None):
        try:
            amt = next(amounts)
        except StopIteration:
            amt = 11.0
        return {
            "window": window,
            "by_currency": {},
            "converted_total": {"currency": target_currency or "USD",
                                "amount": amt, "estimated": True},
            "breakdown": [],
        }

    monkeypatch.setattr("agent_os.budget.ledger.spend", _climbing_spend)
    session = Session.new("notify_dedup", str(tmp_path))

    class _TwoTurn:
        provider = "moonshot"
        model = "kimi-k2.5"
        sdk = "openai"

        def __init__(self):
            self._calls = 0

        async def stream(self, messages, tools=None):
            self._calls += 1
            if self._calls == 1:
                yield StreamChunk(tool_calls_delta=[{
                    "index": 0, "id": "t1", "type": "function",
                    "function": {"name": "noop", "arguments": "{}"},
                }])
                yield StreamChunk(is_final=True,
                                  usage=TokenUsage(input_tokens=100, output_tokens=50))
            else:
                yield StreamChunk(text="done")
                yield StreamChunk(is_final=True,
                                  usage=TokenUsage(input_tokens=80, output_tokens=30))

    context_mgr = ContextManager(session, _Builder(), _ctx(str(tmp_path)))

    class _NoopRegistry(_Registry):
        def schemas(self):
            return [{"name": "noop"}]

    loop = AgentLoop(
        session, _TwoTurn(), _NoopRegistry(), context_mgr,
        project_dir=str(tmp_path),
        get_budget_config=lambda: cfg,
        on_budget_event=events.append,
        max_iterations=10,
    )
    await loop.run(initial_message="hi")

    # The guard-trip site WAS reached (loop exited budget_blocked) ...
    assert loop.get_completion_state()[0] == "budget_blocked"
    # ... but the trip event fired exactly once (at the append), not twice.
    types = [e["type"] for e in events]
    assert types.count(EVENT_TRIP) == 1, types
    # Threshold also fired once (the append jumped straight past 80%).
    assert types.count(EVENT_THRESHOLD) == 1, types


# ---------------------------------------------------------------------------
# The dispatcher's paused-cadence guard usage must stay pure: NOT wired to
# notify, never emits — repeated 5s-cadence evaluations produce zero events.
# ---------------------------------------------------------------------------

def test_dispatcher_source_not_wired_to_notify():
    """Static pin: dispatcher.py never imports/calls the notify module."""
    import inspect
    import agent_os.queue.dispatcher as disp
    src = inspect.getsource(disp)
    assert "budget.notify" not in src
    assert "maybe_emit" not in src
    assert "on_budget_event" not in src


@pytest.mark.asyncio
async def test_dispatcher_paused_cadence_never_emits(tmp_path, monkeypatch):
    """Behavioral pin: repeated _budget_auto_resume_ready evaluations (the 5s
    paused cadence) call evaluate_budget but never the notify emitters."""
    from unittest.mock import MagicMock
    from agent_os.queue.dispatcher import QueueDispatcher
    import agent_os.budget.notify as notify_mod

    calls = []
    monkeypatch.setattr(
        notify_mod, "maybe_emit_budget_event",
        lambda *a, **k: calls.append("event"),
    )
    monkeypatch.setattr(
        notify_mod, "maybe_emit_trip",
        lambda *a, **k: calls.append("trip"),
    )
    # Project sits at 85% — would fire the threshold IF the dispatcher were
    # (wrongly) wired to notify.
    monkeypatch.setattr("agent_os.budget.ledger.spend", _stub_spend(8.5))

    manager = MagicMock()
    manager.build_budget_config.return_value = {
        "budget_limit_usd": 10.0,
        "budget_period": "daily",
        "budget_currency": "USD",
        "budget_action": "pause",
        "fx_rates": {},
    }
    dispatcher = QueueDispatcher(
        project_id="proj1",
        store=MagicMock(),
        agent_manager=manager,
        ws_manager=MagicMock(),
        workspace=str(tmp_path),
    )

    qstate = MagicMock()
    qstate.pause_reason = "budget"

    # Simulate many 5s-cadence ticks.
    for _ in range(10):
        ready = dispatcher._budget_auto_resume_ready(qstate)
        assert ready is True  # 8.5 < 10 → under limit → may resume

    assert calls == []  # the dispatcher path NEVER touched notify
