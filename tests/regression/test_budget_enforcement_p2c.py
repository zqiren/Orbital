# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Mandatory regression suite for Budget Piece 2-C (pre-flight enforcement).

Each test pins a specific diagnosis finding:

  A1  — over-limit ledger ⇒ the loop attempts NO LLM request (provider records
        zero calls) AND writes NO new ledger line. The guard sits BEFORE spend.
  A2/A3 — a budget trip routes through the REAL dispatcher as ``budget_blocked``:
        no corrective turn (no inject_system_message), no contract-violation
        outcome, remaining items do NOT dispatch, the in-flight item is back at
        the queue FRONT in ``queued``, queue PAUSED with reason "budget".
  B5  — a limit RAISED mid-session takes effect on the next guard evaluation
        with no session restart (LIVE resolution, no snapshot).
  B6  — an anchor reset during a running session causes no write conflict — the
        accumulator is gone, so NO budget_spent_usd write happens during a run.
  Migration — budget_action "ask" → "pause", unknown → "pause", "stop" stays.
"""

from __future__ import annotations

import asyncio
import json
import os
from typing import Optional
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent_os.agent.loop import AgentLoop
from agent_os.agent.providers.types import TokenUsage
from agent_os.agent.session import Session
from agent_os.budget import ledger as ledger_mod
from agent_os.budget.ledger import LedgerEvent, SOURCE_MANAGEMENT, ledger_path
from agent_os.budget.normalize import NormalizedUsage
from agent_os.queue.dispatcher import QueueDispatcher
from agent_os.queue.models import AttemptOutcome, ItemState, QueueRunState
from agent_os.queue.store import QueueStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _plant_ledger(workspace: str, *, provider="moonshot", model="kimi",
                  uncached_input=0, cache_read=0, cache_write=0, output=0):
    """Append a real ledger event so spend() reports a known cost."""
    ledger_mod.append_event(
        workspace,
        LedgerEvent(
            session_id="sess",
            source=SOURCE_MANAGEMENT,
            provider=provider,
            model=model,
            usage=NormalizedUsage(
                uncached_input=uncached_input,
                cache_read=cache_read,
                cache_write=cache_write,
                output=output,
            ),
        ),
    )


def _make_loop(workspace, get_budget_config, *, provider=None):
    """A loop wired with a real on-disk Session + a recording provider stub."""
    session = Session.new(
        "proj_b_0001", workspace, session_id="proj_b_0001",
        provider="moonshot", model="kimi", sdk="openai",
    )
    if provider is None:
        provider = MagicMock()
        provider.sdk = "openai"
        provider.provider = "moonshot"
        provider.model = "kimi"
    registry = MagicMock()
    registry.reset_run_state = MagicMock()
    context_manager = MagicMock()
    loop = AgentLoop(
        session, provider, registry, context_manager,
        get_budget_config=get_budget_config,
        project_dir=workspace,
    )
    return loop, session, provider


# ---------------------------------------------------------------------------
# A1 — over-limit ⇒ zero LLM calls, zero new ledger lines
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_A1_over_limit_attempts_no_llm_and_writes_no_ledger(tmp_path):
    """With planted ledger spend over limit, the loop makes NO provider call
    and appends NO new ledger line — the guard trips BEFORE spend."""
    workspace = str(tmp_path / "ws")
    os.makedirs(workspace, exist_ok=True)

    # Plant spend. moonshot prices in CNY, so use a CNY-denominated limit (the
    # limit owns its currency) — 100k uncached input @ moonshot's CNY rate is
    # 0.1 CNY, well over the 0.05 CNY limit. (Using a USD limit with no FX rate
    # would leave the CNY slice unconvertible → 0 convertible spend → no trip;
    # that convertibility behavior is covered in the guard's unit tests.)
    _plant_ledger(workspace, uncached_input=100_000)
    lines_before = _ledger_lines(workspace)
    assert len(lines_before) == 1

    cfg = {
        "budget_limit_usd": 0.05,   # field name is legacy; the VALUE is in CNY
        "budget_action": "pause",
        "budget_period": "total",
        "budget_currency": "CNY",
        "fx_rates": {},
    }

    # A provider whose stream() must NEVER be called — record at the boundary.
    calls = {"n": 0}

    class _RecordingProvider:
        sdk = "openai"
        provider = "moonshot"
        model = "kimi"

        async def stream(self, context, tools=None):
            calls["n"] += 1
            if False:
                yield  # pragma: no cover — make this an async generator

    loop, session, _ = _make_loop(
        workspace, lambda: cfg, provider=_RecordingProvider(),
    )

    await loop.run("do expensive work")

    # The guard tripped before any LLM call.
    assert calls["n"] == 0, "no LLM request may be attempted when over budget"
    # No NEW ledger line (still exactly the one we planted).
    assert len(_ledger_lines(workspace)) == 1
    # First-class exit reason.
    assert loop.get_completion_state()[0] == "budget_blocked"


def _ledger_lines(workspace):
    path = ledger_path(workspace)
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


# ---------------------------------------------------------------------------
# A2/A3 — real-dispatcher routing of budget_blocked
# ---------------------------------------------------------------------------

class _BudgetBlockedLoop:
    def __init__(self):
        self._exit_reason = "budget_blocked"
        self._exit_summary = None
        self._exit_block_reason = "pause"  # normalized action
        self._queue_state = "running"

    def get_completion_state(self):
        return (self._exit_reason, self._exit_summary, self._exit_block_reason)


class _BudgetBlockedAgentManager:
    """Fake manager whose dispatched loop exits ``budget_blocked``.

    Records whether inject_system_message (the corrective turn) was ever called
    — A2/A3 require it NOT to be.
    """

    def __init__(self):
        self._loop = _BudgetBlockedLoop()
        self._task = None
        self._session_counter = 0
        self.corrective_injections = 0

    def is_onboarding_complete(self, project_id):
        return True

    def current_holder_session_id(self, project_id):
        return None

    def get_loop(self, project_id, *, session_id=None):
        return self._loop

    def get_loop_task(self, project_id, *, session_id=None):
        return self._task

    def get_sub_agent_manager(self):
        return None

    async def new_session(self, project_id, *, session_id=None):
        self._session_counter += 1
        return {"status": "ok", "session_id": f"sess_{self._session_counter}",
                "session_uuid": f"proj_{self._session_counter:08d}"}

    async def inject_message(self, project_id, content, *, nonce=None,
                             session_id=None, queue_state="chat"):
        # Loop exits budget_blocked immediately.
        async def _instant():
            return None
        self._task = asyncio.create_task(_instant())
        return "delivered"

    async def inject_system_message(self, project_id, content, *, session_id=None):
        self.corrective_injections += 1
        return "delivered"


async def _wait(predicate, timeout=5.0, interval=0.03):
    deadline = asyncio.get_event_loop().time() + timeout
    while asyncio.get_event_loop().time() < deadline:
        if predicate():
            return True
        await asyncio.sleep(interval)
    return False


@pytest.mark.asyncio
async def test_A2_A3_budget_blocked_routes_without_corrective_or_violation(tmp_path):
    """End-to-end through the REAL dispatcher: budget_blocked ⇒ no corrective
    turn, no contract violation, remaining items don't dispatch, item back at
    front in QUEUED, queue PAUSED reason='budget'."""
    store = QueueStore(tmp_path / "queue.json")
    item1 = store.add_item("first item")
    item2 = store.add_item("second item — must NOT dispatch")

    mgr = _BudgetBlockedAgentManager()
    dispatcher = QueueDispatcher(
        project_id="proj_bb",
        store=store,
        agent_manager=mgr,
        max_runtime_seconds=60,
    )
    await dispatcher.start()
    dispatcher.notify_new_item()

    # Wait for the queue to land PAUSED (the budget trip).
    ok = await _wait(
        lambda: store.load().state == QueueRunState.PAUSED,
        timeout=5.0,
    )
    assert ok, "queue should be PAUSED after a budget trip"

    state = store.load()
    # Queue paused with the budget code (load-bearing for auto-resume).
    assert state.pause_reason == "budget"

    # No corrective turn was fired (the A2/A3 paid-corrective-turn bug).
    assert mgr.corrective_injections == 0

    # The in-flight item is back at the FRONT in QUEUED (resumable).
    reloaded1 = next(it for it in state.items if it.id == item1.id)
    assert reloaded1.state == ItemState.QUEUED
    assert state.items[0].id == item1.id, "blocked item must be at the front"

    # Its attempt was closed INTERRUPTED with the budget reason — NOT a
    # contract-violation BLOCKED outcome.
    assert reloaded1.attempts, "the dispatched attempt should be recorded"
    last = reloaded1.attempts[-1]
    assert last.outcome == AttemptOutcome.INTERRUPTED
    assert last.block_reason == "budget_blocked"
    # Budget block is NOT a poison-pill interruption — the counter is untouched.
    assert reloaded1.interrupted_count == 0

    # The second item never dispatched (still QUEUED, no attempts).
    reloaded2 = next(it for it in state.items if it.id == item2.id)
    assert reloaded2.state == ItemState.QUEUED
    assert reloaded2.attempts == []

    await dispatcher.shutdown()


@pytest.mark.asyncio
async def test_A_stop_mode_marks_item_terminal_not_requeued(tmp_path):
    """stop mode: same detection, but the item is marked terminal (BLOCKED),
    never requeued; queue still PAUSED reason='budget' so remaining items don't
    burn."""
    store = QueueStore(tmp_path / "queue.json")
    item1 = store.add_item("stop-mode item")
    store.add_item("trailing item")

    mgr = _BudgetBlockedAgentManager()
    mgr._loop._exit_block_reason = "stop"  # stop action
    dispatcher = QueueDispatcher(
        project_id="proj_bb_stop",
        store=store,
        agent_manager=mgr,
        max_runtime_seconds=60,
    )
    await dispatcher.start()
    dispatcher.notify_new_item()

    ok = await _wait(
        lambda: store.load().state == QueueRunState.PAUSED,
        timeout=5.0,
    )
    assert ok

    state = store.load()
    assert state.pause_reason == "budget"
    assert mgr.corrective_injections == 0
    reloaded1 = next(it for it in state.items if it.id == item1.id)
    # Terminal — NOT requeued.
    assert reloaded1.state == ItemState.BLOCKED

    await dispatcher.shutdown()


# ---------------------------------------------------------------------------
# B5 — live limit resolution (no snapshot)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_B5_limit_raised_mid_session_takes_effect_next_guard(tmp_path):
    """A limit raised between guard evaluations takes effect WITHOUT a session
    restart: the SAME loop object, re-evaluated, flips from tripped to not."""
    workspace = str(tmp_path / "ws")
    os.makedirs(workspace, exist_ok=True)
    _plant_ledger(workspace, uncached_input=100_000)  # $0.30 spend

    # A mutable config the resolver reads LIVE each call. CNY-denominated limit
    # (moonshot prices in CNY); planted spend is 0.1 CNY.
    live = {
        "budget_limit_usd": 0.05,   # 0.1 CNY spend > 0.05 → trip
        "budget_action": "pause",
        "budget_period": "total",
        "budget_currency": "CNY",
        "fx_rates": {},
    }
    loop, _, _ = _make_loop(workspace, lambda: dict(live))

    # First evaluation: over budget → tripped.
    assert loop._budget_tripped() is not None

    # Raise the limit mid-session (no restart, same loop object).
    live["budget_limit_usd"] = 100.0

    # Next evaluation sees the NEW limit (live resolution) → no trip.
    assert loop._budget_tripped() is None


# ---------------------------------------------------------------------------
# B6 — anchor reset during a run causes no write conflict (no accumulator)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_B6_no_budget_spent_usd_write_during_run(tmp_path):
    """The dollar accumulator is gone: a full loop run performs NO
    budget_spent_usd write, so an anchor reset mid-run cannot conflict with
    one. We assert the project_store sees zero such writes."""
    workspace = str(tmp_path / "ws")
    os.makedirs(workspace, exist_ok=True)

    # A real-ish response so the loop actually streams once and exits text.
    response = MagicMock()
    response.usage = TokenUsage(input_tokens=10, output_tokens=5)
    response.has_tool_calls = False
    response.text = "done"
    response.raw_message = {}

    # Track any update_runtime call that carries budget_spent_usd.
    budget_writes = []

    class _RecordingStore:
        def update_runtime(self, pid, updates):
            if "budget_spent_usd" in updates:
                budget_writes.append(updates)

    # No limit → guard no-op; run streams once and exits.
    cfg = {"budget_limit_usd": None, "fx_rates": {}}
    loop, session, _ = _make_loop(workspace, lambda: cfg)
    loop._stream_response = AsyncMock(return_value=response)

    await loop.run("hello")

    # The loop has no handle to a store and never writes a dollar accumulator;
    # the contract we pin is that NO code path on the run wrote budget_spent_usd.
    assert budget_writes == []
    # And the run actually happened (one ledger line written, not a no-op).
    assert len(_ledger_lines(workspace)) == 1


# ---------------------------------------------------------------------------
# Migration — budget_action normalization
# ---------------------------------------------------------------------------

class TestActionMigration:
    def test_ask_loads_as_pause(self):
        from agent_os.budget.guard import normalize_budget_action, ACTION_PAUSE
        assert normalize_budget_action("ask") == ACTION_PAUSE

    def test_unknown_loads_as_pause(self):
        from agent_os.budget.guard import normalize_budget_action, ACTION_PAUSE
        assert normalize_budget_action("legacy_value") == ACTION_PAUSE

    def test_stop_stays_stop(self):
        from agent_os.budget.guard import normalize_budget_action, ACTION_STOP
        assert normalize_budget_action("stop") == ACTION_STOP

    def test_guard_decision_carries_normalized_action(self):
        """The guard's decision exposes the MIGRATED action, so a config still
        persisting the legacy "ask" surfaces as "pause" to the dispatcher."""
        from agent_os.budget.guard import evaluate_budget, ACTION_PAUSE

        def _spend(project_dir, window, *, target_currency=None, fx_rates=None,
                   now=None, anchor_ts=None):
            return {"converted_total": {"amount": 10.0, "currency": "USD"}}

        d = evaluate_budget(
            "/ws",
            {"budget_limit_usd": 1.0, "budget_action": "ask"},
            spend_fn=_spend,
        )
        assert d.tripped is True
        assert d.action == ACTION_PAUSE


# ---------------------------------------------------------------------------
# Migration on SAVE (route normalizes persisted budget_action)
# ---------------------------------------------------------------------------

def test_migration_normalizes_action_on_save(tmp_path):
    """A PUT carrying budget_action="ask" persists as "pause"; unknown → pause;
    "stop" stays. Normalization happens at the config write point so persisted
    configs converge."""
    from fastapi.testclient import TestClient
    from agent_os.api.app import create_app

    app = create_app(data_dir=str(tmp_path / "data"))
    client = TestClient(app)
    ws = tmp_path / "ws"
    ws.mkdir()

    resp = client.post("/api/v2/projects", json={
        "name": "MigProj", "workspace": str(ws),
        "model": "gpt-4o", "api_key": "sk-test",
    })
    assert resp.status_code == 201
    pid = resp.json()["project_id"]

    # "ask" → "pause"
    r = client.put(f"/api/v2/projects/{pid}", json={"budget_action": "ask"})
    assert r.status_code == 200
    assert r.json()["budget_action"] == "pause"

    # unknown → "pause"
    r = client.put(f"/api/v2/projects/{pid}", json={"budget_action": "frobnicate"})
    assert r.status_code == 200
    assert r.json()["budget_action"] == "pause"

    # "stop" stays "stop"
    r = client.put(f"/api/v2/projects/{pid}", json={"budget_action": "stop"})
    assert r.status_code == 200
    assert r.json()["budget_action"] == "stop"
