# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Budget Piece 2-D — lazy auto-resume + reset-endpoint repurpose.

Pins, through the REAL dispatcher and the REAL PUT route:

  D1  — a budget-paused queue whose guard query now reports UNDER limit
        auto-resumes on the dispatcher's existing cadence and dispatches the
        requeued-to-front item FIRST.
  D2  — a budget-paused queue STILL over limit stays paused: no resume/re-pause
        flap, reason stays "budget".
  D3  — a USER-paused queue under limit NEVER auto-resumes (only reason=="budget"
        is ever touched).
  D4  — coordinator ruling: a budget-paused queue whose project is now in
        ``budget_action == "stop"`` does NOT auto-resume even when under limit.
  D5  — a timed USER pause still resumes on its deadline (paused_until), not via
        budget logic — the two paths coexist.
  D6  — reset endpoint: total mode sets the anchor to now (a planted old event
        then falls outside the window); non-total mode is a no-op returning
        ``{code: "not_total_mode"}``; anchor reset while a session runs is fine.
  D7  — manual resume of an over-limit budget-paused queue is allowed; the next
        dispatch trips the loop's pre-flight guard BEFORE any provider call
        (end-to-end self-healing).
"""

from __future__ import annotations

import asyncio
import os
from datetime import datetime, timedelta, timezone

import pytest

from agent_os.budget import ledger as ledger_mod
from agent_os.budget.ledger import LedgerEvent, SOURCE_MANAGEMENT
from agent_os.budget.normalize import NormalizedUsage
from agent_os.queue.dispatcher import QueueDispatcher
from agent_os.queue.models import AttemptOutcome, ItemState, QueueRunState
from agent_os.queue.store import QueueStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def _wait(predicate, timeout=5.0, interval=0.02):
    deadline = asyncio.get_event_loop().time() + timeout
    while asyncio.get_event_loop().time() < deadline:
        if predicate():
            return True
        await asyncio.sleep(interval)
    return False


def _plant_ledger(workspace: str, *, uncached_input=0):
    ledger_mod.append_event(
        workspace,
        LedgerEvent(
            session_id="sess",
            source=SOURCE_MANAGEMENT,
            provider="moonshot",
            model="kimi",
            usage=NormalizedUsage(
                uncached_input=uncached_input, cache_read=0,
                cache_write=0, output=0,
            ),
        ),
    )


class _CompletingLoop:
    """A dispatched loop that exits ``complete`` immediately (clean advance)."""

    def __init__(self):
        self._exit_reason = "complete"
        self._exit_summary = "done"
        self._exit_block_reason = None
        self._queue_state = "running"

    def get_completion_state(self):
        return (self._exit_reason, self._exit_summary, self._exit_block_reason)


class _AutoResumeManager:
    """Fake AgentManager driving the dispatcher through the auto-resume path.

    ``budget_config`` is mutable so a test can flip over→under limit and confirm
    the dispatcher's lazy guard query picks it up. A dispatched item runs a
    ``_CompletingLoop`` and completes, so we can assert dispatch ORDER.
    """

    def __init__(self, *, budget_config: dict):
        self.budget_config = budget_config
        self._loop = _CompletingLoop()
        self._task = None
        self._session_counter = 0
        self.dispatched_contents: list[str] = []
        self.corrective_injections = 0

    # --- budget config (the shared reader the dispatcher consults) ---
    def build_budget_config(self, project_id):
        return dict(self.budget_config)

    # --- dispatch wiring ---
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
        # Record the queue-item body (strip the contract header for legibility).
        self.dispatched_contents.append(content)

        async def _instant():
            return None
        self._task = asyncio.create_task(_instant())
        return "delivered"

    async def inject_system_message(self, project_id, content, *, session_id=None):
        self.corrective_injections += 1
        return "delivered"


# ---------------------------------------------------------------------------
# D1 — under-limit budget pause auto-resumes and dispatches head-first
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_D1_under_limit_budget_pause_auto_resumes_and_dispatches_head(tmp_path):
    store = QueueStore(tmp_path / "queue.json")
    item1 = store.add_item("HEAD item — resumes first")
    item2 = store.add_item("trailing item")
    # Simulate the budget_blocked disposition: head item back at front QUEUED,
    # queue PAUSED reason=budget.
    store.set_queue_state(QueueRunState.PAUSED, reason="budget")

    # UNDER limit: huge limit, tiny (or no) spend → guard reports not tripped.
    mgr = _AutoResumeManager(budget_config={
        "budget_limit_usd": 1000.0, "budget_action": "pause",
        "budget_period": "total", "budget_currency": "CNY",
        "budget_anchor_ts": None, "fx_rates": {},
    })
    dispatcher = QueueDispatcher(
        project_id="proj_d1", store=store, agent_manager=mgr,
        max_runtime_seconds=60, workspace=str(tmp_path),
    )
    await dispatcher.start()

    # Auto-resume should fire on the cadence and dispatch the HEAD item first,
    # completing it.
    ok = await _wait(
        lambda: any(it.id == item1.id and it.state == ItemState.DONE
                    for it in store.load().items),
        timeout=5.0,
    )
    assert ok, "head item should auto-resume and complete"

    # Order: item1 dispatched before item2.
    assert mgr.dispatched_contents, "something must have dispatched"
    assert "HEAD item" in mgr.dispatched_contents[0]
    # Queue left RUNNING/IDLE, reason cleared (no longer budget-paused).
    state = store.load()
    assert state.state in (QueueRunState.RUNNING, QueueRunState.IDLE)
    assert state.pause_reason is None
    assert mgr.corrective_injections == 0

    await dispatcher.shutdown()


# ---------------------------------------------------------------------------
# D2 — still over limit ⇒ stays paused, no flap
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_D2_still_over_limit_stays_paused_no_flap(tmp_path):
    workspace = str(tmp_path / "ws")
    os.makedirs(workspace, exist_ok=True)
    # Plant real over-limit spend (0.1 CNY > 0.05 CNY limit).
    _plant_ledger(workspace, uncached_input=100_000)

    store = QueueStore(tmp_path / "queue.json")
    item1 = store.add_item("blocked item")
    store.set_queue_state(QueueRunState.PAUSED, reason="budget")

    mgr = _AutoResumeManager(budget_config={
        "budget_limit_usd": 0.05, "budget_action": "pause",
        "budget_period": "total", "budget_currency": "CNY",
        "budget_anchor_ts": None, "fx_rates": {},
    })
    dispatcher = QueueDispatcher(
        project_id="proj_d2", store=store, agent_manager=mgr,
        max_runtime_seconds=60, workspace=workspace,
    )

    # Wrap set_queue_state to detect any resume/re-pause churn.
    transitions: list = []
    orig_set = store.set_queue_state

    def _spy(new_state, **kw):
        transitions.append((new_state, kw.get("reason")))
        return orig_set(new_state, **kw)

    store.set_queue_state = _spy  # type: ignore[assignment]

    await dispatcher.start()
    # Give the cadence a few ticks to (not) act.
    await asyncio.sleep(0.3)

    state = store.load()
    assert state.state == QueueRunState.PAUSED, "must stay paused while over limit"
    assert state.pause_reason == "budget", "reason must remain budget (no flip)"
    # The item never dispatched.
    reloaded1 = next(it for it in state.items if it.id == item1.id)
    assert reloaded1.state == ItemState.QUEUED
    assert reloaded1.attempts == []
    assert mgr.dispatched_contents == []
    # No churn: the dispatcher did not write any new queue-state transition.
    assert transitions == [], f"no resume/re-pause churn expected, saw {transitions}"

    await dispatcher.shutdown()


# ---------------------------------------------------------------------------
# D3 — user pause under limit NEVER auto-resumes
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_D3_user_pause_under_limit_never_auto_resumes(tmp_path):
    store = QueueStore(tmp_path / "queue.json")
    item1 = store.add_item("user-paused item")
    # USER pause (reason="user"), under limit.
    store.set_queue_state(QueueRunState.PAUSED, reason="user")

    mgr = _AutoResumeManager(budget_config={
        "budget_limit_usd": 1000.0, "budget_action": "pause",
        "budget_period": "total", "budget_currency": "CNY",
        "budget_anchor_ts": None, "fx_rates": {},
    })
    dispatcher = QueueDispatcher(
        project_id="proj_d3", store=store, agent_manager=mgr,
        max_runtime_seconds=60, workspace=str(tmp_path),
    )
    await dispatcher.start()
    await asyncio.sleep(0.3)

    state = store.load()
    assert state.state == QueueRunState.PAUSED, "user pause must NOT auto-resume"
    assert state.pause_reason == "user"
    reloaded1 = next(it for it in state.items if it.id == item1.id)
    assert reloaded1.attempts == []
    assert mgr.dispatched_contents == []

    await dispatcher.shutdown()


# ---------------------------------------------------------------------------
# D4 — stop-mode project NEVER auto-resumes (coordinator ruling)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_D4_stop_mode_never_auto_resumes_even_under_limit(tmp_path):
    store = QueueStore(tmp_path / "queue.json")
    item1 = store.add_item("stop-mode item")
    store.set_queue_state(QueueRunState.PAUSED, reason="budget")

    # UNDER limit, but action is "stop" — must not resume.
    mgr = _AutoResumeManager(budget_config={
        "budget_limit_usd": 1000.0, "budget_action": "stop",
        "budget_period": "total", "budget_currency": "CNY",
        "budget_anchor_ts": None, "fx_rates": {},
    })
    dispatcher = QueueDispatcher(
        project_id="proj_d4", store=store, agent_manager=mgr,
        max_runtime_seconds=60, workspace=str(tmp_path),
    )
    await dispatcher.start()
    await asyncio.sleep(0.3)

    state = store.load()
    assert state.state == QueueRunState.PAUSED, "stop-mode must never auto-resume"
    assert state.pause_reason == "budget"
    reloaded1 = next(it for it in state.items if it.id == item1.id)
    assert reloaded1.attempts == []
    assert mgr.dispatched_contents == []

    await dispatcher.shutdown()


# ---------------------------------------------------------------------------
# D5 — timed USER pause still resumes on its deadline, not via budget logic
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_D5_timed_user_pause_resumes_on_deadline(tmp_path):
    store = QueueStore(tmp_path / "queue.json")
    store.add_item("timed-pause item")
    # A timed USER pause whose deadline has ALREADY passed.
    past = (datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat()
    store.set_queue_state(QueueRunState.PAUSED, paused_until=past, reason="user")

    # OVER limit — proving the resume is the timed path, not budget logic
    # (budget logic would refuse a user pause AND refuse an over-limit queue).
    mgr = _AutoResumeManager(budget_config={
        "budget_limit_usd": 0.0, "budget_action": "pause",
        "budget_period": "total", "budget_currency": "CNY",
        "budget_anchor_ts": None, "fx_rates": {},
    })
    dispatcher = QueueDispatcher(
        project_id="proj_d5", store=store, agent_manager=mgr,
        max_runtime_seconds=60, workspace=str(tmp_path),
    )
    await dispatcher.start()

    ok = await _wait(
        lambda: store.load().state != QueueRunState.PAUSED, timeout=5.0,
    )
    assert ok, "timed user pause should auto-resume on its deadline"
    state = store.load()
    assert state.pause_reason is None  # cleared on resume

    await dispatcher.shutdown()


# ---------------------------------------------------------------------------
# D6 — reset endpoint: total-mode sets anchor; non-total no-op + code
# ---------------------------------------------------------------------------

def _make_app(tmp_path):
    from fastapi.testclient import TestClient
    from agent_os.api.app import create_app

    app = create_app(data_dir=str(tmp_path / "data"))
    return TestClient(app)


def _create_project(client, ws, **extra):
    body = {"name": "P", "workspace": str(ws), "model": "gpt-4o",
            "api_key": "sk-test"}
    body.update(extra)
    r = client.post("/api/v2/projects", json=body)
    assert r.status_code == 201, r.text
    return r.json()["project_id"]


def test_D6_reset_total_mode_sets_anchor_and_honors_window(tmp_path):
    """Total mode: reset sets budget_anchor_ts to now. A planted OLD ledger
    event then falls OUTSIDE the spend window (its spend stops counting)."""
    client = _make_app(tmp_path)
    ws = tmp_path / "ws"
    ws.mkdir()
    pid = _create_project(client, ws, budget_period="total",
                          budget_currency="CNY", budget_limit_usd=100.0)

    # Plant an OLD event (timestamp 2 days in the past) by writing the ledger
    # line directly — append_event stamps "now", but we need a controlled past
    # ts to prove the anchor excludes it after reset.
    old_ts = (datetime.now(timezone.utc) - timedelta(days=2)).isoformat()
    ledger_file = ledger_mod.ledger_path(str(ws))
    os.makedirs(os.path.dirname(ledger_file), exist_ok=True)
    record = LedgerEvent(
        session_id="sess", source=SOURCE_MANAGEMENT,
        provider="moonshot", model="kimi",
        usage=NormalizedUsage(uncached_input=100_000, cache_read=0,
                              cache_write=0, output=0),
    ).to_record(old_ts)
    with open(ledger_file, "a", encoding="utf-8") as f:
        f.write(__import__("json").dumps(record) + "\n")

    # Before reset (no anchor): the old event counts → spend > 0.
    before = client.get(f"/api/v2/projects/{pid}/cost?window=total").json()
    assert before["converted_total"]["amount"] > 0

    # Legacy reset surface: the existing UI sends {budget_spent_usd: 0}.
    r = client.put(f"/api/v2/projects/{pid}", json={"budget_spent_usd": 0})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["budget_reset"]["applied"] is True
    assert body["budget_reset"]["code"] is None
    anchor = body["budget_reset"]["budget_anchor_ts"]
    assert anchor and body["budget_anchor_ts"] == anchor

    # After reset: the anchor is "now", so the 2-day-old event is excluded.
    after = client.get(f"/api/v2/projects/{pid}/cost?window=total").json()
    assert after["converted_total"]["amount"] == 0.0


def test_D6_reset_non_total_mode_is_noop_with_code(tmp_path):
    """Non-total mode (daily): reset is a no-op returning code not_total_mode,
    and budget_anchor_ts is unchanged."""
    client = _make_app(tmp_path)
    ws = tmp_path / "ws"
    ws.mkdir()
    pid = _create_project(client, ws, budget_period="daily",
                          budget_limit_usd=100.0)

    r = client.put(f"/api/v2/projects/{pid}", json={"budget_spent_usd": 0})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["budget_reset"]["applied"] is False
    assert body["budget_reset"]["code"] == "not_total_mode"
    # No anchor was set.
    assert not body.get("budget_anchor_ts")


def test_D6_reset_via_reset_budget_anchor_flag_total_mode(tmp_path):
    """The new reset_budget_anchor:true surface unifies with the legacy one."""
    client = _make_app(tmp_path)
    ws = tmp_path / "ws"
    ws.mkdir()
    pid = _create_project(client, ws, budget_period="total",
                          budget_limit_usd=100.0)

    r = client.put(f"/api/v2/projects/{pid}", json={"reset_budget_anchor": True})
    assert r.status_code == 200, r.text
    assert r.json()["budget_reset"]["applied"] is True
    assert r.json()["budget_anchor_ts"]


def test_D6_no_reset_field_means_no_budget_reset_key(tmp_path):
    """A plain config PUT (no reset requested) carries no budget_reset key."""
    client = _make_app(tmp_path)
    ws = tmp_path / "ws"
    ws.mkdir()
    pid = _create_project(client, ws)

    r = client.put(f"/api/v2/projects/{pid}", json={"name": "Renamed"})
    assert r.status_code == 200, r.text
    assert "budget_reset" not in r.json()
    # The legacy budget_spent_usd flatten is PRESERVED (frontend reads it).
    assert r.json()["budget_spent_usd"] == 0.0


def test_D6_reset_while_running_no_conflict(tmp_path):
    """An anchor reset issued while a session is 'running' raises no conflict —
    there is no dollar accumulator to race; the PUT just sets the anchor."""
    client = _make_app(tmp_path)
    ws = tmp_path / "ws"
    ws.mkdir()
    pid = _create_project(client, ws, budget_period="total",
                          budget_limit_usd=100.0)
    # No live agent needed: the reset path is pure config; this asserts the
    # route returns cleanly and applies the anchor regardless of run state.
    r = client.put(f"/api/v2/projects/{pid}", json={"budget_spent_usd": 0})
    assert r.status_code == 200
    assert r.json()["budget_reset"]["applied"] is True


# ---------------------------------------------------------------------------
# D7 — manual resume of an over-limit budget pause: next dispatch trips the
# loop's pre-flight guard BEFORE any provider call (self-healing).
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_D7_manual_resume_over_limit_trips_guard_pre_spend(tmp_path):
    """Manual resume of an over-limit budget-paused queue is allowed; the next
    dispatch runs the loop whose pre-flight guard trips BEFORE any provider
    call — no provider.stream(), and the queue lands back PAUSED reason=budget.
    """
    from agent_os.agent.loop import AgentLoop
    from agent_os.agent.session import Session

    workspace = str(tmp_path / "ws")
    os.makedirs(workspace, exist_ok=True)
    _plant_ledger(workspace, uncached_input=100_000)  # 0.1 CNY > 0.05 limit

    over_cfg = {
        "budget_limit_usd": 0.05, "budget_action": "pause",
        "budget_period": "total", "budget_currency": "CNY",
        "budget_anchor_ts": None, "fx_rates": {},
    }

    calls = {"n": 0}

    class _RecordingProvider:
        sdk = "openai"
        provider = "moonshot"
        model = "kimi"

        async def stream(self, context, tools=None):
            calls["n"] += 1
            if False:
                yield  # pragma: no cover

    # A real loop + real session driven by a dispatch-shaped fake manager. The
    # manager hands the dispatcher this loop's task and reads its completion.
    from unittest.mock import MagicMock

    session = Session.new("proj_d7_0001", workspace, session_id="proj_d7_0001",
                          provider="moonshot", model="kimi", sdk="openai")
    registry = MagicMock()
    registry.reset_run_state = MagicMock()
    context_manager = MagicMock()
    loop = AgentLoop(
        session, _RecordingProvider(), registry, context_manager,
        get_budget_config=lambda: dict(over_cfg),
        project_dir=workspace,
    )

    class _RealLoopManager:
        def __init__(self):
            self._task = None
            self._session_counter = 0
            self.corrective_injections = 0

        def build_budget_config(self, project_id):
            return dict(over_cfg)

        def is_onboarding_complete(self, project_id):
            return True

        def current_holder_session_id(self, project_id):
            return None

        def get_loop(self, project_id, *, session_id=None):
            return loop

        def get_loop_task(self, project_id, *, session_id=None):
            return self._task

        def get_sub_agent_manager(self):
            return None

        async def new_session(self, project_id, *, session_id=None):
            self._session_counter += 1
            return {"status": "ok", "session_id": "proj_d7_0001",
                    "session_uuid": "proj_d7_0001"}

        async def inject_message(self, project_id, content, *, nonce=None,
                                 session_id=None, queue_state="chat"):
            self._task = asyncio.create_task(loop.run(content))
            return "delivered"

        async def inject_system_message(self, project_id, content, *, session_id=None):
            self.corrective_injections += 1
            return "delivered"

    store = QueueStore(tmp_path / "queue.json")
    item1 = store.add_item("over-limit work")
    store.set_queue_state(QueueRunState.PAUSED, reason="budget")

    mgr = _RealLoopManager()
    dispatcher = QueueDispatcher(
        project_id="proj_d7", store=store, agent_manager=mgr,
        max_runtime_seconds=60, workspace=workspace,
    )
    await dispatcher.start()

    # MANUAL resume (the existing resume route) — allowed anytime.
    await dispatcher.resume()

    # The dispatch runs the loop; its pre-flight guard trips → budget_blocked,
    # so the queue lands PAUSED reason=budget again, and NO provider call fired.
    ok = await _wait(
        lambda: store.load().state == QueueRunState.PAUSED
        and store.load().pause_reason == "budget"
        and any(it.id == item1.id and it.attempts
                and it.attempts[-1].outcome == AttemptOutcome.INTERRUPTED
                for it in store.load().items),
        timeout=8.0,
    )
    assert ok, "self-heal: over-limit resume should re-trip the guard and re-pause"
    assert calls["n"] == 0, "guard must trip BEFORE any provider.stream() call"
    assert mgr.corrective_injections == 0, "no paid corrective turn on a budget trip"

    await dispatcher.shutdown()
