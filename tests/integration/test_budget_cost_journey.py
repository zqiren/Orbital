# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Integration: token ledger → GET /api/v2/projects/{pid}/cost over the REAL app.

These tests stand up the real FastAPI application via ``create_app`` and a real
``ProjectStore`` / ``SettingsStore`` on a temp ``data_dir``, then exercise the
REAL ``GET /cost`` route (real spend() aggregation, real providers.json default
rates, real ledger files on disk). Only the LLM provider is stubbed — we are
testing budget plumbing, not the model.

Harness choices (documented per the task spec):

  - **App surface:** the genuine ``create_app(data_dir=...)`` factory behind a
    ``fastapi.testclient.TestClient`` — same router, stores, and wiring the
    daemon uses. Projects are created through the real ``POST /api/v2/projects``
    route; cost is read through the real ``GET /cost`` route.

  - **HOME isolation:** ``create_app`` reads ``os.path.expanduser("~")`` at
    runtime for the ``SubAgentConfigStore`` path
    (``~/.orbital/sub_agent_config.json``, ``app.py``); the ``isolated_home``
    fixture points ``HOME``/``USERPROFILE`` at a throwaway tmp dir so these
    tests never read a developer's real ``~/.orbital`` state. (The daemon
    PID-file singleton is NOT the reason: its default path is frozen at import
    time, and the root ``tests/conftest.py`` already no-ops ``acquire_pid_file``
    suite-wide.) Pure test-env manipulation — no production code is modified.

  - **Scenario B1 — session run:** the task allows a documented fallback when no
    existing harness can drive a full-app session with a stubbed provider. We
    take that fallback: run the REAL ``AgentLoop`` directly against the created
    project's workspace with a stub provider (mirroring
    ``tests/unit/test_loop_ledger_emission.py``), which appends to the REAL
    ledger file via the loop's real emission path, then read it back through the
    REAL ``GET /cost`` route. The full daemon dispatch path (queue → dispatcher
    → provider client keyed on an API key) requires real LLM credentials, which
    this suite does not have; driving the loop directly exercises every budget
    seam (normalize → ledger append → spend → route) end-to-end while keeping
    the provider a stub.

  - **Rates:** resolved from the checked-in ``providers.json`` defaults (no
    override file), so the numbers below are the production default rates:
    ``anthropic/claude-sonnet-4-6`` = 3.0 / 15.0 USD per 1M (in/out);
    ``moonshot/kimi-k2.5`` = 0.6 / 3.0 CNY per 1M. Using defaults keeps the
    test independent of the override-injection plumbing covered by the unit
    suite.

Timezone safety: no test asserts a wall-clock value. The window test uses
``datetime.now(timezone.utc)`` for the "in-window" event (always inside today's
``daily`` window in any zone) and a fixed 2020 timestamp for the excluded one
(always before any 2026+ daily/weekly/monthly window start).
"""

from __future__ import annotations

import asyncio
import json
import os
from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient

from agent_os.api.app import create_app
from agent_os.daemon_v2.settings_store import SettingsStore
from agent_os.budget.ledger import ledger_path

from agent_os.agent.session import Session
from agent_os.agent.loop import AgentLoop
from agent_os.agent.context import ContextManager
from agent_os.agent.providers.types import StreamChunk, TokenUsage
from agent_os.agent.tools.base import ToolResult
from agent_os.agent.prompt_builder import PromptContext, Autonomy


# ===========================================================================
# Fixtures
# ===========================================================================

@pytest.fixture
def isolated_home(tmp_path, monkeypatch):
    """Point HOME at a throwaway dir so create_app's runtime expanduser("~")
    lookups (notably ~/.orbital/sub_agent_config.json) never read the
    developer's real home state. See the module docstring for why the PID
    file is NOT the concern here."""
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
def app_client(isolated_home, data_dir):
    """Real FastAPI app behind a TestClient, on an isolated temp data_dir."""
    app = create_app(data_dir=data_dir)
    return TestClient(app)


def _create_project(client: TestClient, workspace: str, *, provider: str,
                    model: str, sdk: str, **extra) -> tuple[str, str]:
    """Create a project through the REAL route; return (project_id, workspace)."""
    body = {
        "name": "budget-int",
        "workspace": workspace,
        "model": model,
        "api_key": "test-key",
        "provider": provider,
        "sdk": sdk,
    }
    body.update(extra)
    resp = client.post("/api/v2/projects", json=body)
    assert resp.status_code == 201, resp.text
    j = resp.json()
    return j["project_id"], j["workspace"]


def _write_ledger_line(workspace: str, record: dict) -> None:
    """Hand-append one ledger JSONL line (used to plant out-of-window / mixed
    -currency events directly, exactly as the on-disk schema stores them)."""
    p = ledger_path(workspace)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def _ledger_event(*, ts: str, provider: str, model: str, source="management",
                  uncached_input=0, cache_read=0, cache_write=0, output=0) -> dict:
    return {
        "ts": ts, "session_id": "s", "source": source,
        "provider": provider, "model": model,
        "uncached_input": uncached_input, "cache_read": cache_read,
        "cache_write": cache_write, "output": output,
    }


# --- Stub provider + minimal loop scaffolding (mirrors the unit emission test).

class _Builder:
    def build(self, context):
        return ("cached-prefix", "stable-suffix", "dynamic-runtime")


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


class _StubProvider:
    """Single-turn text provider carrying real provider/model/sdk identity."""

    def __init__(self, provider, model, sdk, usage: TokenUsage):
        self.provider = provider
        self.model = model
        self.sdk = sdk
        self._usage = usage

    async def stream(self, messages, tools=None):
        yield StreamChunk(text="hello")
        yield StreamChunk(is_final=True, usage=self._usage)


def _run_real_loop(workspace: str, provider: _StubProvider, model: str) -> None:
    """Drive the REAL AgentLoop against *workspace* so it appends to the real
    ledger via its real emission path."""
    session = Session.new("budget_int", workspace)
    ctx = PromptContext(
        workspace=workspace,
        model=model,
        autonomy=Autonomy.HANDS_OFF,
        enabled_agents=[],
        tool_names=[],
        os_type="linux",
        datetime_now="2026-01-01T00:00:00",
        context_usage_pct=0.0,
    )
    context_mgr = ContextManager(session, _Builder(), ctx)
    loop = AgentLoop(
        session, provider, _Registry(), context_mgr,
        project_dir=workspace, max_iterations=5,
    )
    asyncio.run(loop.run(initial_message="go"))


# ===========================================================================
# B1 — stubbed-provider session run → ledger → /cost
# ===========================================================================

class TestSessionRunToCost:
    def test_real_loop_writes_ledger_then_cost_route_reads_it(self, app_client,
                                                              tmp_path):
        workspace = str(tmp_path / "ws")
        os.makedirs(workspace, exist_ok=True)
        pid, ws = _create_project(
            app_client, workspace, provider="anthropic",
            model="claude-sonnet-4-6", sdk="anthropic",
            budget_period="total", budget_currency="USD",
        )
        assert ws == workspace  # the route reads cost from this workspace

        # Run the real loop with a stub provider: 1M uncached input + 0.5M output
        # under anthropic additive semantics → disjoint fields land verbatim.
        provider = _StubProvider(
            "anthropic", "claude-sonnet-4-6", "anthropic",
            TokenUsage(input_tokens=1_000_000, output_tokens=500_000,
                       cache_read_tokens=0, cache_write_tokens=0),
        )
        _run_real_loop(workspace, provider, "claude-sonnet-4-6")

        # The loop's real emission path wrote exactly one DISJOINT-field line.
        with open(ledger_path(workspace), "r", encoding="utf-8") as f:
            lines = [json.loads(ln) for ln in f if ln.strip()]
        assert len(lines) == 1, lines
        ev = lines[0]
        assert ev["provider"] == "anthropic"
        assert ev["model"] == "claude-sonnet-4-6"
        assert ev["source"] == "management"
        # anthropic additive: uncached_input copied through; no cache fields.
        assert ev["uncached_input"] == 1_000_000
        assert ev["cache_read"] == 0
        assert ev["cache_write"] == 0
        assert ev["output"] == 500_000
        # Disjoint: the four fields are mutually exclusive (no double counting).
        assert (ev["uncached_input"] + ev["cache_read"]
                + ev["cache_write"] + ev["output"]) == 1_500_000

        # The REAL route reflects it. claude-sonnet-4-6 defaults: 3.0/15.0 USD.
        # cost = 1M*3.0/1M + 0.5M*15.0/1M = 3.0 + 7.5 = 10.5 USD.
        resp = app_client.get(f"/api/v2/projects/{pid}/cost?window=total")
        assert resp.status_code == 200
        body = resp.json()
        assert body["window"] == "total"
        assert body["by_currency"]["USD"] == pytest.approx(10.5)
        assert body["converted_total"]["currency"] == "USD"
        assert body["converted_total"]["amount"] == pytest.approx(10.5)
        assert body["converted_total"]["estimated"] is True
        assert len(body["breakdown"]) == 1
        br = body["breakdown"][0]
        assert br["provider"] == "anthropic"
        assert br["model"] == "claude-sonnet-4-6"
        assert br["currency"] == "USD"
        assert br["tokens"]["uncached_input"] == 1_000_000
        assert br["cost"]["uncached_input"] == pytest.approx(3.0)
        assert br["cost"]["output"] == pytest.approx(7.5)
        assert br["cost"]["total"] == pytest.approx(10.5)


# ===========================================================================
# B2 — window exclusion (out-of-window event excluded from daily, in total)
# ===========================================================================

class TestWindowExclusion:
    def test_old_event_excluded_from_daily_included_in_total(self, app_client,
                                                            tmp_path):
        workspace = str(tmp_path / "ws")
        os.makedirs(workspace, exist_ok=True)
        pid, ws = _create_project(
            app_client, workspace, provider="anthropic",
            model="claude-sonnet-4-6", sdk="anthropic",
            budget_currency="USD",
        )

        # An OLD event (2020) — always before any 2026+ window start, in any tz.
        _write_ledger_line(ws, _ledger_event(
            ts="2020-01-01T00:00:00+00:00",
            provider="anthropic", model="claude-sonnet-4-6",
            output=1_000_000,  # would be $15 if counted
        ))
        # A FRESH event (now, UTC) — always inside today's daily window, any tz.
        now_iso = datetime.now(timezone.utc).isoformat()
        _write_ledger_line(ws, _ledger_event(
            ts=now_iso,
            provider="anthropic", model="claude-sonnet-4-6",
            output=2_000_000,  # $30
        ))

        # daily: only the fresh event counts → $30, one breakdown row.
        daily = app_client.get(
            f"/api/v2/projects/{pid}/cost?window=daily").json()
        assert daily["window"] == "daily"
        assert daily["by_currency"].get("USD") == pytest.approx(30.0)
        assert len(daily["breakdown"]) == 1
        assert daily["breakdown"][0]["tokens"]["output"] == 2_000_000

        # total: both events count → $15 + $30 = $45, summed into one group
        # (same provider/model/source) → still one breakdown row, 3M output.
        total = app_client.get(
            f"/api/v2/projects/{pid}/cost?window=total").json()
        assert total["window"] == "total"
        assert total["by_currency"].get("USD") == pytest.approx(45.0)
        assert len(total["breakdown"]) == 1
        assert total["breakdown"][0]["tokens"]["output"] == 3_000_000


# ===========================================================================
# B3 — mixed currency (USD + CNY events → per-currency totals + converted)
# ===========================================================================

class TestMixedCurrency:
    def test_per_currency_totals_and_converted_estimate(self, app_client,
                                                        data_dir, tmp_path):
        # Pin an explicit FX rate in the REAL settings store; the route reads it
        # at request time. 6.0 (not the 7.2 default) proves the value flows from
        # the settings store, not a hardcoded constant.
        ss = SettingsStore(data_dir=data_dir)
        settings = ss.get()
        settings.fx_rates = {"CNY_per_USD": 6.0}
        ss.update(settings)

        workspace = str(tmp_path / "ws")
        os.makedirs(workspace, exist_ok=True)
        pid, ws = _create_project(
            app_client, workspace, provider="anthropic",
            model="claude-sonnet-4-6", sdk="anthropic",
            budget_period="total", budget_currency="USD",
        )

        now_iso = datetime.now(timezone.utc).isoformat()
        # USD line: anthropic claude-sonnet-4-6, 1M output → $15.0 USD.
        _write_ledger_line(ws, _ledger_event(
            ts=now_iso, provider="anthropic", model="claude-sonnet-4-6",
            output=1_000_000,
        ))
        # CNY line: moonshot kimi-k2.5, 1M output → 3.0 CNY.
        _write_ledger_line(ws, _ledger_event(
            ts=now_iso, provider="moonshot", model="kimi-k2.5",
            output=1_000_000,
        ))

        body = app_client.get(
            f"/api/v2/projects/{pid}/cost?window=total").json()

        # Per-currency subtotals are reported separately, never mixed.
        assert body["by_currency"]["USD"] == pytest.approx(15.0)
        assert body["by_currency"]["CNY"] == pytest.approx(3.0)

        # Converted total honors the settings-store FX rate (6.0), is flagged
        # estimated, and converts CNY→USD: 15 + 3/6.0 = 15.5 USD.
        conv = body["converted_total"]
        assert conv["currency"] == "USD"
        assert conv["estimated"] is True
        assert conv["amount"] == pytest.approx(15.5)
        assert "unconverted" not in conv  # CNY_per_USD is known → nothing skipped

        # Two distinct breakdown groups, each in its native currency.
        currencies = {b["currency"] for b in body["breakdown"]}
        assert currencies == {"USD", "CNY"}

    def test_unknown_fx_pair_marks_currency_unconverted(self, app_client,
                                                       data_dir, tmp_path):
        # Empty FX table → CNY cannot be converted to USD; the route must surface
        # the slice as 'unconverted' rather than inventing a rate or 500ing.
        ss = SettingsStore(data_dir=data_dir)
        settings = ss.get()
        settings.fx_rates = {}  # no CNY_per_USD
        ss.update(settings)

        workspace = str(tmp_path / "ws")
        os.makedirs(workspace, exist_ok=True)
        pid, ws = _create_project(
            app_client, workspace, provider="moonshot",
            model="kimi-k2.5", sdk="openai",
            budget_period="total", budget_currency="USD",
        )
        now_iso = datetime.now(timezone.utc).isoformat()
        _write_ledger_line(ws, _ledger_event(
            ts=now_iso, provider="moonshot", model="kimi-k2.5",
            output=1_000_000,  # 3.0 CNY
        ))

        body = app_client.get(
            f"/api/v2/projects/{pid}/cost?window=total").json()
        assert body["by_currency"]["CNY"] == pytest.approx(3.0)
        conv = body["converted_total"]
        # CNY omitted from the converted amount; surfaced as unconverted.
        assert conv["amount"] == pytest.approx(0.0)
        assert conv["unconverted"] == ["CNY"]


# ===========================================================================
# B4 — i18n shape: no English sentence strings in data fields
# ===========================================================================

# Fields whose string VALUES are legitimately code/enum/ISO/identifier tokens —
# these are allowed to be non-translatable machine strings. Anything else that
# is a string and looks like a human sentence fails the checker.
_ALLOWED_CODE_FIELDS = frozenset({
    "window",      # enum: daily|weekly|monthly|total
    "currency",    # ISO 4217: USD, CNY
    "provider",    # provider key: anthropic, moonshot
    "model",       # model id: claude-sonnet-4-6, kimi-k2.5
    "source",      # enum: management, ...
    "code",        # machine error code: invalid_window
})
# Field whose value is a LIST of code tokens (enum members / ISO codes).
_ALLOWED_CODE_LIST_FIELDS = frozenset({"allowed", "unconverted"})


def _looks_like_english_sentence(s: str) -> bool:
    """Heuristic: a string is a 'display sentence' if it contains whitespace
    separating word-like tokens that include lowercase letters.

    This catches things like "Budget exceeded", "Project not found", or
    "service unavailable" — strings that SHOULD have been i18n keys — while NOT
    flagging hyphenated/underscored identifiers ("claude-sonnet-4-6",
    "invalid_window", "kimi-k2.5") which contain no whitespace, nor single
    tokens ("USD", "anthropic", "total").
    """
    if " " not in s:
        return False
    # At least two whitespace-separated tokens, and some lowercase letter
    # present (filters out e.g. acronym-only or numeric strings).
    parts = [p for p in s.split() if p]
    if len(parts) < 2:
        return False
    return any(c.islower() for c in s)


def _assert_no_display_strings(node, *, path="$", field_name=None,
                               violations=None) -> list[str]:
    """Recursively walk a JSON-decoded API body; collect any string value that
    looks like an English sentence and is NOT in an allow-listed code field.

    Returns the list of violations (each a "<path> = <value>" string). The
    checker is deliberately explicit about WHICH fields are exempt so a future
    field that adds a display string (e.g. ``"reason": "Budget exceeded"``)
    trips this and fails loudly.
    """
    if violations is None:
        violations = []

    if isinstance(node, dict):
        for k, v in node.items():
            _assert_no_display_strings(
                v, path=f"{path}.{k}", field_name=k, violations=violations)
    elif isinstance(node, list):
        # A list under an allow-listed code-list field carries only code tokens;
        # still descend, but the per-string check below will exempt them via the
        # parent field_name passed through.
        for i, v in enumerate(node):
            _assert_no_display_strings(
                v, path=f"{path}[{i}]", field_name=field_name,
                violations=violations)
    elif isinstance(node, str):
        exempt = (field_name in _ALLOWED_CODE_FIELDS
                  or field_name in _ALLOWED_CODE_LIST_FIELDS)
        if not exempt and _looks_like_english_sentence(node):
            violations.append(f"{path} = {node!r}")

    return violations


class TestI18nResponseShape:
    def test_checker_itself_flags_a_sentence_but_not_a_model_name(self):
        """Guard the guard: the heuristic must catch a real display sentence and
        must NOT false-positive on identifiers/codes."""
        # Positive: a leaked display string under a non-code field is caught.
        bad = {"reason": "Budget exceeded for this project"}
        assert _assert_no_display_strings(bad) == [
            "$.reason = 'Budget exceeded for this project'"
        ]
        # Negative: model names, codes, providers, ISO codes are NOT flagged.
        good = {
            "model": "claude-sonnet-4-6",
            "provider": "anthropic",
            "currency": "USD",
            "code": "invalid_window",
            "window": "total",
            "allowed": ["daily", "weekly", "monthly", "total"],
            "unconverted": ["CNY"],
            "amount": 15.5,
            "estimated": True,
        }
        assert _assert_no_display_strings(good) == []
        # A model name even if it were placed under an unknown field has no
        # whitespace, so it is still not flagged.
        assert _assert_no_display_strings({"x": "claude-sonnet-4-6"}) == []

    def test_cost_200_body_has_no_display_strings(self, app_client, tmp_path):
        workspace = str(tmp_path / "ws")
        os.makedirs(workspace, exist_ok=True)
        pid, ws = _create_project(
            app_client, workspace, provider="anthropic",
            model="claude-sonnet-4-6", sdk="anthropic",
            budget_period="total", budget_currency="USD",
        )
        now_iso = datetime.now(timezone.utc).isoformat()
        _write_ledger_line(ws, _ledger_event(
            ts=now_iso, provider="anthropic", model="claude-sonnet-4-6",
            uncached_input=1000, cache_read=500, cache_write=200, output=1000,
        ))
        resp = app_client.get(f"/api/v2/projects/{pid}/cost?window=total")
        assert resp.status_code == 200
        violations = _assert_no_display_strings(resp.json())
        assert violations == [], violations

    def test_cost_400_invalid_window_body_has_no_display_strings(self,
                                                                app_client,
                                                                tmp_path):
        workspace = str(tmp_path / "ws")
        os.makedirs(workspace, exist_ok=True)
        pid, ws = _create_project(
            app_client, workspace, provider="anthropic",
            model="claude-sonnet-4-6", sdk="anthropic",
        )
        resp = app_client.get(f"/api/v2/projects/{pid}/cost?window=yearly")
        assert resp.status_code == 400
        body = resp.json()
        # Sanity: the route returns a machine code + allowed list (no sentence).
        assert body["detail"]["code"] == "invalid_window"
        assert set(body["detail"]["allowed"]) == {
            "daily", "weekly", "monthly", "total"}
        violations = _assert_no_display_strings(body)
        assert violations == [], violations
