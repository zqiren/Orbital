# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Integration: the full budget-ENFORCEMENT journey over the REAL app surface.

Where the unit + regression suites pin each enforcement seam in isolation
(``tests/unit/test_budget_notify.py``, ``tests/regression/test_budget_*_p2*.py``),
THIS suite drives the seams together through the genuine FastAPI surface:

  * Projects are created and reconfigured through the REAL HTTP routes
    (``POST /api/v2/projects``, ``PUT /api/v2/projects/{pid}`` for a limit raise,
    ``POST .../queue/stop`` + ``.../queue/resume`` for manual pause/resume),
    backed by a real ``ProjectStore`` / ``SettingsStore`` on a temp ``data_dir``.
  * The queue runs on the REAL ``QueueDispatcher`` + ``QueueStore`` + (for the
    pre-spend block) the REAL ``AgentLoop``, so the PROVIDER-BOUNDARY zero-call
    assertion is made against a real loop's real pre-flight guard.
  * Spend is planted in the REAL on-disk ledger; the guard reads it LIVE.

Harness choices (documented per the task spec):

  - **App surface vs. real AgentManager.** The real daemon dispatch path
    (queue → dispatcher → ``AgentManager`` → provider client keyed on an API key)
    makes real LLM network calls — this suite has no credentials, exactly the
    constraint ``tests/integration/test_budget_cost_journey.py`` documents for
    its session-run scenario. So we take the SAME documented fallback: the REAL
    app/routes own every CONFIG mutation and the real stores, while the queue is
    driven by a REAL ``QueueDispatcher`` wired to a dispatch-shaped fake manager
    (the P2-C/P2-D regression pattern). Critically, the fake manager's
    ``build_budget_config`` reads the REAL ``ProjectStore`` + ``SettingsStore``
    (via the app's wired ``AgentManager.build_budget_config``), so a limit raised
    through the real PUT route is seen LIVE by the dispatcher's lazy guard and by
    the loop's pre-flight guard — there is no second source of truth.

  - **WS broadcast capture.** The dispatcher's ``queue.state_changed`` broadcast
    goes through its injected ``ws_manager.broadcast(project_id, payload)`` seam
    (``dispatcher._broadcast_state_changed``). We capture it with a tiny in-proc
    hub stub recording ``(project_id, payload)`` — the acceptable integration-seam
    choice the task allows when the live WS surface can't be captured cleanly in
    a TestClient (the dispatcher here is constructed directly, not via a live
    websocket). The payload asserted is the REAL one the dispatcher emits.

  - **Trip/threshold sink.** The loop's ``on_budget_event`` sink (wired in
    production by ``AgentManager`` to ``_broadcast``) is wired here the way the
    P2-E regression tests wire it — a plain ``events.append`` sink — because
    running the real ``AgentManager._broadcast`` chain needs the relay/WS hub.
    The once-only assertion is the point and is unaffected by the sink identity.

  - **Cadence.** ``IDLE_WAIT_TIMEOUT_SEC`` is 5.0s; no test sleeps that long. We
    drive the dispatcher exactly as production does: ``notify_new_item()`` (the
    same nudge the add-item and PUT-budget-change routes fire) and direct
    ``resume()``, then poll a predicate with a short interval (the P2-C/P2-D
    ``_wait`` helper). The whole suite runs well under the 30s budget.

Timezone safety: no test asserts a wall-clock value; planted ledger spend is
window-independent (``total`` window, no anchor) so it counts in any zone.
"""

from __future__ import annotations

import asyncio
import json
import os

import pytest
from fastapi.testclient import TestClient

from agent_os.api.app import create_app

from agent_os.agent.loop import AgentLoop
from agent_os.agent.session import Session
from agent_os.agent.providers.types import StreamChunk, TokenUsage
from agent_os.agent.tools.base import ToolResult
from agent_os.agent.context import ContextManager
from agent_os.agent.prompt_builder import PromptContext, Autonomy

from agent_os.budget import ledger as ledger_mod
from agent_os.budget.ledger import LedgerEvent, SOURCE_MANAGEMENT, ledger_path
from agent_os.budget.normalize import NormalizedUsage
from agent_os.budget.notify import EVENT_THRESHOLD, EVENT_TRIP

from agent_os.queue.dispatcher import QueueDispatcher
from agent_os.queue.models import AttemptOutcome, ItemState, QueueRunState
from agent_os.queue.store import QueueStore


# ===========================================================================
# Fixtures + harness scaffolding
# ===========================================================================

@pytest.fixture
def isolated_home(tmp_path, monkeypatch):
    """Point HOME at a throwaway dir so create_app's runtime expanduser("~")
    lookups (notably ~/.orbital/sub_agent_config.json) never read the
    developer's real home state — same rationale as test_budget_cost_journey."""
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    return home


@pytest.fixture
def data_dir(tmp_path):
    d = tmp_path / "data"
    d.mkdir()
    return str(d)


@pytest.fixture
def app(isolated_home, data_dir):
    """The REAL FastAPI app on an isolated temp data_dir. ``create_app`` runs
    ``agents_v2.configure(...)``, which stows the wired ``AgentManager`` (real
    ProjectStore/SettingsStore) on the route module; ``_real_manager`` reads it
    so tests can call the REAL ``build_budget_config`` reader."""
    return create_app(data_dir=data_dir)


@pytest.fixture
def client(app):
    return TestClient(app)


def _real_manager(app):
    """The app's wired AgentManager (real project/settings stores), as held by
    the route module after ``agents_v2.configure`` ran inside ``create_app``."""
    from agent_os.api.routes import agents_v2
    assert agents_v2._agent_manager is not None, "app did not wire the manager"
    return agents_v2._agent_manager


def _create_project(client: TestClient, workspace: str, **budget) -> str:
    """Create a project through the REAL route; return project_id. The budget
    fields (limit/action/period/currency) flow through the real config writer."""
    body = {
        "name": "budget-enf",
        "workspace": workspace,
        "model": "kimi-k2.5",
        "api_key": "test-key",
        "provider": "moonshot",
        "sdk": "openai",
    }
    body.update(budget)
    resp = client.post("/api/v2/projects", json=body)
    assert resp.status_code == 201, resp.text
    return resp.json()["project_id"]


def _plant_ledger(workspace: str, *, uncached_input=0, provider="moonshot",
                  model="kimi") -> None:
    """Append a real ledger event (moonshot prices in CNY) so spend() reports a
    known cost the guard reads live."""
    ledger_mod.append_event(
        workspace,
        LedgerEvent(
            session_id="planted",
            source=SOURCE_MANAGEMENT,
            provider=provider,
            model=model,
            usage=NormalizedUsage(
                uncached_input=uncached_input, cache_read=0,
                cache_write=0, output=0,
            ),
        ),
    )


async def _wait(predicate, timeout=8.0, interval=0.02):
    deadline = asyncio.get_event_loop().time() + timeout
    while asyncio.get_event_loop().time() < deadline:
        if predicate():
            return True
        await asyncio.sleep(interval)
    return False


class _RecordingHub:
    """In-proc WS hub stub: records (project_id, payload) for every broadcast.
    The integration-seam capture point for queue.state_changed (see module
    docstring) — it records the REAL payload the dispatcher emits."""

    def __init__(self):
        self.events: list[tuple[str, dict]] = []

    def broadcast(self, project_id: str, payload: dict) -> None:
        self.events.append((project_id, dict(payload)))

    def state_changed(self) -> list[dict]:
        return [p for _pid, p in self.events
                if p.get("type") == "queue.state_changed"]


# --- The REAL pre-spend-blocking loop (provider boundary records zero calls). --

class _ProviderBoundary:
    """A provider whose stream() MUST NEVER be called when over budget. Records
    calls at the boundary so the pre-spend assertion is made where spend would
    actually happen."""

    sdk = "openai"
    provider = "moonshot"
    model = "kimi"

    def __init__(self):
        self.calls = 0

    async def stream(self, context, tools=None):
        self.calls += 1
        if False:  # pragma: no cover — make this an async generator
            yield


def _make_real_loop(workspace: str, build_cfg, *, session_id: str,
                    provider=None):
    """A REAL AgentLoop + REAL Session whose pre-flight guard reads LIVE config
    via ``build_cfg`` (a 0-arg callable). Used where the spec demands a real
    provider-boundary zero-call assertion."""
    from unittest.mock import MagicMock

    session = Session.new(session_id, workspace, session_id=session_id,
                          provider="moonshot", model="kimi", sdk="openai")
    registry = MagicMock()
    registry.reset_run_state = MagicMock()
    context_manager = MagicMock()
    loop = AgentLoop(
        session, provider or _ProviderBoundary(), registry, context_manager,
        get_budget_config=build_cfg,
        project_dir=workspace,
    )
    return loop, session


class _RealLoopManager:
    """Dispatch-shaped fake manager that runs a REAL AgentLoop per dispatched
    item (so the over-budget pre-flight guard trips at the real provider
    boundary). ``build_budget_config`` delegates to the REAL app manager, so a
    limit raised via the real PUT route is seen LIVE by both the dispatcher's
    lazy guard and the loop's pre-flight guard — one source of truth.
    """

    def __init__(self, project_id: str, workspace: str, real_manager,
                 provider: "_ProviderBoundary"):
        self._project_id = project_id
        self._workspace = workspace
        self._real_manager = real_manager
        self._provider = provider
        self._task = None
        self._loop = None
        self._session_counter = 0
        self.corrective_injections = 0

    # The shared LIVE config reader — delegates to the REAL stores.
    def build_budget_config(self, project_id):
        return self._real_manager.build_budget_config(project_id)

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
        sid = f"sess_{self._session_counter}"
        return {"status": "ok", "session_id": sid, "session_uuid": sid}

    async def inject_message(self, project_id, content, *, nonce=None,
                             session_id=None, queue_state="chat"):
        # Build a fresh REAL loop for this dispatch and run it. Its pre-flight
        # guard reads LIVE config; over budget → budget_blocked with NO
        # provider.stream() call.
        self._loop, _session = _make_real_loop(
            self._workspace, lambda: self.build_budget_config(project_id),
            session_id=session_id or f"sess_{self._session_counter}",
            provider=self._provider,
        )
        self._task = asyncio.create_task(self._loop.run(content))
        return "delivered"

    async def inject_system_message(self, project_id, content, *, session_id=None):
        self.corrective_injections += 1
        return "delivered"


# --- A lightweight completing manager (for ORDER-of-dispatch assertions). -----

class _CompletingLoop:
    def __init__(self):
        self._exit_reason = "complete"
        self._exit_summary = "done"
        self._exit_block_reason = None

    def get_completion_state(self):
        return (self._exit_reason, self._exit_summary, self._exit_block_reason)


class _CompletingManager:
    """Runs a trivially-completing loop per dispatch, recording dispatch order.
    ``build_budget_config`` delegates to the REAL manager so a real PUT limit
    raise is what clears the dispatcher's lazy guard."""

    def __init__(self, project_id, real_manager):
        self._project_id = project_id
        self._real_manager = real_manager
        self._loop = _CompletingLoop()
        self._task = None
        self._session_counter = 0
        self.dispatched_contents: list[str] = []
        self.corrective_injections = 0

    def build_budget_config(self, project_id):
        return self._real_manager.build_budget_config(project_id)

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
        sid = f"sess_{self._session_counter}"
        return {"status": "ok", "session_id": sid, "session_uuid": sid}

    async def inject_message(self, project_id, content, *, nonce=None,
                             session_id=None, queue_state="chat"):
        self.dispatched_contents.append(content)

        async def _instant():
            return None
        self._task = asyncio.create_task(_instant())
        return "delivered"

    async def inject_system_message(self, project_id, content, *, session_id=None):
        self.corrective_injections += 1
        return "delivered"


def _ledger_lines(workspace):
    path = ledger_path(workspace)
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(ln) for ln in f if ln.strip()]


# ===========================================================================
# 1 — BLOCK journey (pre-spend block; provider boundary records zero calls)
# ===========================================================================

@pytest.mark.asyncio
async def test_block_journey_pre_spend_block_requeue_pause_and_timeline(
        client, app, tmp_path):
    """Tiny limit + 3 queued items + planted over-limit ledger → dispatch →
    item 1 blocks PRE-SPEND (zero provider calls at the boundary; zero new
    ledger lines), returns to queue FRONT in QUEUED, queue PAUSED reason=budget,
    the queue.state_changed WS broadcast carries the reason, the session timeline
    holds the codes-only budget_blocked event, and the trip fires EXACTLY once
    (not again on subsequent dispatcher cadence ticks)."""
    workspace = str(tmp_path / "ws")
    os.makedirs(workspace, exist_ok=True)

    # REAL route: tiny CNY limit (moonshot prices in CNY); total window.
    pid = _create_project(
        client, workspace,
        budget_limit_usd=0.05, budget_action="pause",
        budget_period="total", budget_currency="CNY",
    )
    # Planted spend 0.1 CNY > 0.05 limit — over budget before any dispatch.
    _plant_ledger(workspace, uncached_input=100_000)
    assert len(_ledger_lines(workspace)) == 1

    # Three queued items via the REAL queue store the route resolves.
    store = QueueStore(tmp_path / "queue.json")
    item1 = store.add_item("first item")
    item2 = store.add_item("second item — must NOT dispatch")
    item3 = store.add_item("third item — must NOT dispatch")

    hub = _RecordingHub()
    provider = _ProviderBoundary()
    mgr = _RealLoopManager(pid, workspace, _real_manager(app), provider)
    dispatcher = QueueDispatcher(
        project_id=pid, store=store, agent_manager=mgr,
        ws_manager=hub, max_runtime_seconds=60, workspace=workspace,
    )
    await dispatcher.start()
    dispatcher.notify_new_item()

    ok = await _wait(
        lambda: store.load().state == QueueRunState.PAUSED
        and store.load().pause_reason == "budget",
    )
    assert ok, "queue should land PAUSED reason=budget on a pre-spend block"

    # PROVIDER BOUNDARY: zero calls — the guard tripped before any LLM request.
    assert provider.calls == 0, "no provider.stream() may run when over budget"
    # Zero NEW ledger lines (still exactly the one planted).
    assert len(_ledger_lines(workspace)) == 1

    state = store.load()
    # Item 1 back at the FRONT in QUEUED (resumable).
    reloaded1 = next(it for it in state.items if it.id == item1.id)
    assert reloaded1.state == ItemState.QUEUED
    assert state.items[0].id == item1.id
    # Its attempt closed INTERRUPTED with the budget reason; not a poison pill.
    assert reloaded1.attempts
    assert reloaded1.attempts[-1].outcome == AttemptOutcome.INTERRUPTED
    assert reloaded1.attempts[-1].block_reason == "budget_blocked"
    assert reloaded1.interrupted_count == 0
    # Items 2 + 3 never dispatched.
    for it in (item2, item3):
        r = next(x for x in state.items if x.id == it.id)
        assert r.state == ItemState.QUEUED
        assert r.attempts == []
    assert mgr.corrective_injections == 0

    # WS broadcast carried the reason.
    sc = hub.state_changed()
    assert sc, "a queue.state_changed broadcast should have fired"
    paused = [p for p in sc if p["state"] == QueueRunState.PAUSED.value]
    assert paused and paused[-1]["pause_reason"] == "budget"

    # SESSION TIMELINE: the structured codes-only budget_blocked event.
    msgs = mgr._loop._session.get_messages()
    blocked = [m for m in msgs if m.get("event") == "budget_blocked"]
    assert len(blocked) == 1, blocked
    ev = blocked[0]
    assert ev["source"] == "budget"
    payload = ev["payload"]
    assert payload["action"] == "pause"
    assert payload["window"] == "total"
    assert payload["currency"] == "CNY"
    assert payload["limit"] == 0.05
    assert payload["spend"] >= 0.05
    # Codes/numbers only — no display sentences in any string field.
    for k, v in payload.items():
        if isinstance(v, str):
            assert " " not in v, f"{k}={v!r} looks like a display string"

    # ONCE-only: let the paused dispatcher run several more cadence ticks
    # (nudge it repeatedly) — still over limit → it stays paused and NEVER
    # re-runs the in-flight item, so no second budget_blocked timeline event
    # and no second attempt is opened on item 1.
    for _ in range(5):
        dispatcher.notify_new_item()
    await asyncio.sleep(0.2)
    state2 = store.load()
    assert state2.state == QueueRunState.PAUSED
    assert state2.pause_reason == "budget"
    reloaded1b = next(it for it in state2.items if it.id == item1.id)
    assert len(reloaded1b.attempts) == 1, "no re-dispatch of the blocked item"
    # The session object the trip was recorded on is unchanged — exactly one event.
    assert len([m for m in mgr._loop._session.get_messages()
                if m.get("event") == "budget_blocked"]) == 1

    await dispatcher.shutdown()


# ===========================================================================
# 2 — LAZY RESUME (raise the limit via the REAL PUT; head item dispatches first)
# ===========================================================================

@pytest.mark.asyncio
async def test_lazy_resume_after_real_put_dispatches_blocked_item_first(
        client, app, tmp_path):
    """A budget-paused queue (head item requeued to front) auto-resumes after a
    REAL PUT raises the limit, and dispatches the previously-blocked item FIRST;
    items proceed."""
    workspace = str(tmp_path / "ws")
    os.makedirs(workspace, exist_ok=True)
    # Over limit at first (0.1 CNY > 0.05).
    pid = _create_project(
        client, workspace,
        budget_limit_usd=0.05, budget_action="pause",
        budget_period="total", budget_currency="CNY",
    )
    _plant_ledger(workspace, uncached_input=100_000)

    # Simulate the post-block disposition: head item at front QUEUED, a trailing
    # item, queue PAUSED reason=budget.
    store = QueueStore(tmp_path / "queue.json")
    item1 = store.add_item("HEAD item — resumes first")
    item2 = store.add_item("trailing item")
    store.set_queue_state(QueueRunState.PAUSED, reason="budget")

    hub = _RecordingHub()
    mgr = _CompletingManager(pid, _real_manager(app))
    dispatcher = QueueDispatcher(
        project_id=pid, store=store, agent_manager=mgr,
        ws_manager=hub, max_runtime_seconds=60, workspace=workspace,
    )
    await dispatcher.start()

    # Still over limit → stays paused, nothing dispatches.
    await asyncio.sleep(0.15)
    assert store.load().state == QueueRunState.PAUSED
    assert mgr.dispatched_contents == []

    # REAL PUT raises the limit far above spend. The route fires the same
    # notify_new_item nudge the dispatcher needs to re-evaluate promptly.
    r = client.put(f"/api/v2/projects/{pid}", json={"budget_limit_usd": 1000.0})
    assert r.status_code == 200, r.text
    assert r.json()["budget_limit_usd"] == 1000.0
    # The app's dispatcher (created on project-create) got the nudge, but THIS
    # test drives its own dispatcher instance against the same store/config; nudge
    # it the same way the route does so the lazy guard re-runs immediately.
    dispatcher.notify_new_item()

    ok = await _wait(
        lambda: any(it.id == item1.id and it.state == ItemState.DONE
                    for it in store.load().items),
    )
    assert ok, "head item should auto-resume and complete after the limit raise"
    # ORDER: the previously-blocked HEAD item dispatched first.
    assert mgr.dispatched_contents
    assert "HEAD item" in mgr.dispatched_contents[0]
    # Items proceed: the trailing item also completes.
    ok2 = await _wait(
        lambda: any(it.id == item2.id and it.state == ItemState.DONE
                    for it in store.load().items),
    )
    assert ok2, "trailing item should proceed after the head item"
    final = store.load()
    assert final.state in (QueueRunState.RUNNING, QueueRunState.IDLE)
    assert final.pause_reason is None

    await dispatcher.shutdown()


# ===========================================================================
# 3 — REASON PRECEDENCE (budget pause + manual pause via real route → "user")
# ===========================================================================

@pytest.mark.asyncio
async def test_reason_precedence_user_pause_over_budget_never_auto_resumes(
        client, app, tmp_path):
    """A budget-paused queue that the user then manually pauses (REAL stop route)
    becomes reason="user"; raising the limit afterwards must NOT auto-resume a
    user pause."""
    workspace = str(tmp_path / "ws")
    os.makedirs(workspace, exist_ok=True)
    pid = _create_project(
        client, workspace,
        budget_limit_usd=0.05, budget_action="pause",
        budget_period="total", budget_currency="CNY",
    )
    _plant_ledger(workspace, uncached_input=100_000)

    store = QueueStore(tmp_path / "queue.json")
    item1 = store.add_item("blocked then user-paused")
    # Budget pause first (the post-block disposition).
    store.set_queue_state(QueueRunState.PAUSED, reason="budget")

    hub = _RecordingHub()
    mgr = _CompletingManager(pid, _real_manager(app))
    dispatcher = QueueDispatcher(
        project_id=pid, store=store, agent_manager=mgr,
        ws_manager=hub, max_runtime_seconds=60, workspace=workspace,
    )
    await dispatcher.start()
    assert store.load().pause_reason == "budget"

    # MANUAL pause over the budget pause → store precedence sets reason "user".
    await dispatcher.stop()
    assert store.load().pause_reason == "user"

    # Raise the limit via the REAL PUT (would clear a budget pause) and nudge.
    r = client.put(f"/api/v2/projects/{pid}", json={"budget_limit_usd": 1000.0})
    assert r.status_code == 200, r.text
    dispatcher.notify_new_item()

    # The user pause must NOT auto-resume — give the cadence room to (not) act.
    await asyncio.sleep(0.3)
    state = store.load()
    assert state.state == QueueRunState.PAUSED, "user pause must stay paused"
    assert state.pause_reason == "user"
    reloaded1 = next(it for it in state.items if it.id == item1.id)
    assert reloaded1.attempts == []
    assert mgr.dispatched_contents == []

    await dispatcher.shutdown()


# ===========================================================================
# 4 — STOP MODE (trip → terminal, not requeued; raising limit does NOT resume)
# ===========================================================================

@pytest.mark.asyncio
async def test_stop_mode_terminal_item_and_no_auto_resume_on_limit_raise(
        client, app, tmp_path):
    """budget_action "stop": on trip the in-flight item is terminal (BLOCKED, not
    requeued), queue paused reason=budget; raising the limit does NOT auto-resume
    (the binding coordinator ruling: nothing ever auto-resumes a stop)."""
    workspace = str(tmp_path / "ws")
    os.makedirs(workspace, exist_ok=True)
    pid = _create_project(
        client, workspace,
        budget_limit_usd=0.05, budget_action="stop",
        budget_period="total", budget_currency="CNY",
    )
    _plant_ledger(workspace, uncached_input=100_000)

    store = QueueStore(tmp_path / "queue.json")
    item1 = store.add_item("stop-mode item")
    item2 = store.add_item("trailing item")

    hub = _RecordingHub()
    provider = _ProviderBoundary()
    mgr = _RealLoopManager(pid, workspace, _real_manager(app), provider)
    dispatcher = QueueDispatcher(
        project_id=pid, store=store, agent_manager=mgr,
        ws_manager=hub, max_runtime_seconds=60, workspace=workspace,
    )
    await dispatcher.start()
    dispatcher.notify_new_item()

    ok = await _wait(
        lambda: store.load().state == QueueRunState.PAUSED
        and store.load().pause_reason == "budget"
        and any(it.id == item1.id and it.state == ItemState.BLOCKED
                for it in store.load().items),
    )
    assert ok, "stop-mode trip → item BLOCKED, queue PAUSED reason=budget"
    # Pre-spend: zero provider calls.
    assert provider.calls == 0

    state = store.load()
    reloaded1 = next(it for it in state.items if it.id == item1.id)
    assert reloaded1.state == ItemState.BLOCKED, "stop item must be terminal"
    # The loop exit carried the normalized "stop" action.
    assert mgr._loop.get_completion_state()[2] == "stop"
    assert mgr.corrective_injections == 0
    # Trailing item never dispatched.
    reloaded2 = next(it for it in state.items if it.id == item2.id)
    assert reloaded2.state == ItemState.QUEUED
    assert reloaded2.attempts == []

    # Raise the limit via the REAL PUT and nudge — stop-mode must NOT resume.
    r = client.put(f"/api/v2/projects/{pid}", json={"budget_limit_usd": 1000.0})
    assert r.status_code == 200, r.text
    dispatcher.notify_new_item()
    await asyncio.sleep(0.3)
    final = store.load()
    assert final.state == QueueRunState.PAUSED, "stop-mode never auto-resumes"
    assert final.pause_reason == "budget"
    # The trailing item still never dispatched.
    assert next(it for it in final.items
                if it.id == item2.id).attempts == []

    await dispatcher.shutdown()


# ===========================================================================
# 5 — THRESHOLD (85% of limit → threshold fires once, codes-only; second
# response in the same window does not re-fire)
# ===========================================================================

@pytest.mark.asyncio
async def test_threshold_fires_once_with_codes_only_across_two_responses(
        tmp_path, monkeypatch):
    """Spend planted at 85% of the limit + one stubbed-provider response → the
    threshold event fires EXACTLY once with a codes-only payload; a SECOND
    response in the same window does NOT re-fire (once-per-window, live wiring
    through the loop's on_budget_event sink)."""
    workspace = str(tmp_path / "ws")
    os.makedirs(workspace, exist_ok=True)

    events: list[dict] = []
    cfg = {
        "budget_limit_usd": 10.0,
        "budget_period": "daily",
        "budget_currency": "USD",
        "budget_action": "pause",
        "fx_rates": {},
    }
    # 8.5 / 10 = 85% → crosses 80%, under 100% → threshold ONLY, both appends.
    def _spend(project_dir, window, *, target_currency=None, fx_rates=None,
               now=None, anchor_ts=None):
        return {
            "window": window, "by_currency": {},
            "converted_total": {"currency": target_currency or "USD",
                                "amount": 8.5, "estimated": True},
            "breakdown": [],
        }
    monkeypatch.setattr("agent_os.budget.ledger.spend", _spend)

    # Two management responses in one window — both append, threshold once.
    session = Session.new("threshold_int", workspace)

    class _Builder:
        def build(self, context):
            return ("sys", "suffix", "dynamic")

    class _Registry:
        def schemas(self):
            return [{"name": "noop"}]

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

    class _TwoTurn:
        provider = "moonshot"
        model = "kimi-k2.5"
        sdk = "openai"

        def __init__(self):
            self._calls = 0

        async def stream(self, messages, tools=None):
            self._calls += 1
            if self._calls == 1:
                # First turn: a tool call so the loop continues to a 2nd LLM call
                # (a second ledger append in the SAME window).
                yield StreamChunk(tool_calls_delta=[{
                    "index": 0, "id": "t1", "type": "function",
                    "function": {"name": "noop", "arguments": "{}"},
                }])
                yield StreamChunk(is_final=True,
                                  usage=TokenUsage(input_tokens=100,
                                                   output_tokens=50))
            else:
                yield StreamChunk(text="done")
                yield StreamChunk(is_final=True,
                                  usage=TokenUsage(input_tokens=80,
                                                   output_tokens=30))

    ctx = PromptContext(
        workspace=workspace, model="kimi-k2.5", autonomy=Autonomy.HANDS_OFF,
        enabled_agents=[], tool_names=[], os_type="linux",
        datetime_now="2026-01-01T00:00:00", context_usage_pct=0.0,
    )
    context_mgr = ContextManager(session, _Builder(), ctx)
    loop = AgentLoop(
        session, _TwoTurn(), _Registry(), context_mgr,
        project_dir=workspace,
        get_budget_config=lambda: cfg,
        on_budget_event=events.append,
        max_iterations=5,
    )
    await loop.run(initial_message="hi")

    types = [e["type"] for e in events]
    # Threshold fired EXACTLY once across the two responses; never tripped.
    assert types.count(EVENT_THRESHOLD) == 1, types
    assert EVENT_TRIP not in types, types
    # Two ledger appends actually happened in the window (real wiring, not a
    # single-turn shortcut).
    assert len(_ledger_lines(workspace)) == 2

    ev = events[0]
    assert ev["type"] == EVENT_THRESHOLD
    assert ev["pct"] == 80
    assert ev["spend"] == 8.5
    assert ev["limit"] == 10.0
    assert ev["currency"] == "USD"
    assert ev["window"] == "daily"
    # Codes/numbers only — no display sentences in any string field.
    for e in events:
        for k, v in e.items():
            if isinstance(v, str):
                assert " " not in v, f"{k}={v!r} looks like a display string"
