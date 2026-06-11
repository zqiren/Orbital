# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Integration: sub-agent transport capture → ledger → GET /cost (P3-B).

Stands up the REAL FastAPI app via ``create_app`` on a temp ``data_dir`` and
exercises the REAL ``GET /cost`` route over a project workspace, then drives the
REAL transport capture paths so the ledger line that lands on disk is written by
production code — not hand-planted JSONL.

Harness choices (documented per the task spec):

  - **App surface:** the genuine ``create_app(data_dir=...)`` factory behind a
    ``fastapi.testclient.TestClient`` — same router, stores, and wiring the
    daemon uses. Projects are created through the real ``POST /api/v2/projects``
    route; cost is read through the real ``GET /cost`` route.

  - **Transport capture surface:** rather than spawn a real ``claude`` / ``codex``
    subprocess (which needs credentials and a live model), we feed each
    transport its protocol's turn-final message DIRECTLY through the real
    capture code path:
      * SDKTransport: a real ``ResultMessage`` (verbatim Anthropic usage shape +
        ``total_cost_usd``) through ``_message_to_events`` — the same call the
        background consumer makes per message.
      * CodexTransport: real ``thread/tokenUsage/updated`` notifications +
        ``turn/completed`` through ``_route_server_message`` — the same wire
        entry point the read loop drives. Payloads are verbatim-shaped from the
        probe traces in artifacts-2026-06-06-codex-lifecycle/.../traces.
    The transport's own ``_workspace`` points at the project workspace, so its
    ``append_event`` lands on the SAME ledger file ``GET /cost`` reads back.

  - **The pinned invariant (ties to P3-A):** a management ledger line is planted
    alongside the captured sub-agent lines; the cost route's MANAGEMENT block
    (by_currency / converted_total / breakdown) must reflect ONLY the management
    spend, with the sub-agent rows moved into the separate ``subagents`` block.

  - **HOME isolation:** ``create_app`` reads ``expanduser("~")`` at runtime for
    the ``SubAgentConfigStore`` path; ``isolated_home`` points HOME at a
    throwaway dir so these tests never read a developer's real ``~/.orbital``.
"""

from __future__ import annotations

import json
import os

import pytest
from fastapi.testclient import TestClient

from agent_os.api.app import create_app
from agent_os.budget.ledger import ledger_path


@pytest.fixture
def isolated_home(tmp_path, monkeypatch):
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
    app = create_app(data_dir=data_dir)
    return TestClient(app)


def _create_project(client, workspace, *, provider, model, sdk, **extra):
    body = {
        "name": "subagent-capture-int",
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


def _plant_management_line(workspace, *, provider, model, output):
    """One real management ledger line via the real writer (so the on-disk
    shape is byte-identical to production)."""
    from agent_os.budget.ledger import LedgerEvent, append_event
    from agent_os.budget.normalize import NormalizedUsage

    append_event(
        workspace,
        LedgerEvent(
            session_id="mgmt-sess",
            source="management",
            provider=provider,
            model=model,
            usage=NormalizedUsage(
                uncached_input=0, cache_read=0, cache_write=0, output=output),
        ),
    )


def _ledger_lines(workspace):
    path = ledger_path(workspace)
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


# ===========================================================================
# SDK transport (claude-code) capture → ledger → /cost
# ===========================================================================

class TestSDKCaptureToCost:
    def test_sdk_result_capture_lands_in_subagents_block(self, app_client, tmp_path):
        from agent_os.agent.transports.sdk_transport import SDKTransport
        from claude_agent_sdk.types import ResultMessage

        workspace = str(tmp_path / "ws")
        os.makedirs(workspace, exist_ok=True)
        pid, ws = _create_project(
            app_client, workspace, provider="anthropic",
            model="claude-sonnet-4-6", sdk="anthropic",
            budget_period="total", budget_currency="USD",
        )
        assert ws == workspace

        # Management spend the cost route's management block must reflect: 1M
        # output on claude-sonnet-4-6 (defaults 3.0/15.0 USD) → $15.
        _plant_management_line(workspace, provider="anthropic",
                               model="claude-sonnet-4-6", output=1_000_000)

        # Drive the REAL SDK capture path: a turn-final ResultMessage carrying
        # the Anthropic usage shape + a provider-reported total_cost_usd.
        transport = SDKTransport()
        transport._workspace = workspace
        result = ResultMessage(
            subtype="success", duration_ms=100, duration_api_ms=80,
            is_error=False, num_turns=1, session_id="claude-sess-1",
            total_cost_usd=0.1734,
            usage={
                "input_tokens": 2000,
                "output_tokens": 400,
                "cache_read_input_tokens": 1500,
                "cache_creation_input_tokens": 100,
            },
            result=None, structured_output=None,
        )
        transport._message_to_events(result)

        # Two ledger lines on disk: one management, one subagent:claude-code.
        lines = _ledger_lines(workspace)
        sources = sorted(l["source"] for l in lines)
        assert sources == ["management", "subagent:claude-code"]

        # GET /cost: management block carries the $15 ONLY; the sub-agent row is
        # in the separate subagents block with the reported cost VERBATIM.
        body = app_client.get(f"/api/v2/projects/{pid}/cost?window=total").json()
        assert body["by_currency"] == {"USD": pytest.approx(15.0)}
        assert body["converted_total"]["amount"] == pytest.approx(15.0)
        assert [b["source"] for b in body["breakdown"]] == ["management"]

        assert len(body["subagents"]) == 1
        entry = body["subagents"][0]
        assert entry["source"] == "subagent:claude-code"
        assert entry["provider"] == "anthropic"
        # Additive normalization: uncached copied through, cache fields split.
        assert entry["tokens"]["uncached_input"] == 2000
        assert entry["tokens"]["cache_read"] == 1500
        assert entry["tokens"]["cache_write"] == 100
        assert entry["tokens"]["output"] == 400
        # Reported cost verbatim (0.1734 USD), NOT recomputed from our rates.
        assert entry["reported_cost"] == {"amount": pytest.approx(0.1734),
                                          "currency": "USD"}


# ===========================================================================
# Codex transport capture → ledger → /cost
# ===========================================================================

class TestCodexCaptureToCost:
    @pytest.mark.asyncio
    async def test_codex_turn_capture_lands_in_subagents_block(self, app_client,
                                                              tmp_path):
        from agent_os.agent.transports.codex_transport import CodexTransport

        workspace = str(tmp_path / "ws")
        os.makedirs(workspace, exist_ok=True)
        pid, ws = _create_project(
            app_client, workspace, provider="anthropic",
            model="claude-sonnet-4-6", sdk="anthropic",
            budget_period="total", budget_currency="USD",
        )

        # Management spend → $15 (same as above).
        _plant_management_line(workspace, provider="anthropic",
                               model="claude-sonnet-4-6", output=1_000_000)

        # Drive the REAL Codex capture path: repeated tokenUsage notifications
        # for one turn, then turn/completed. One ledger event from the LAST
        # `last` breakdown. Payload verbatim-shaped from the probe traces.
        t = CodexTransport()
        t._workspace = workspace
        t._thread_id = "T1"
        t._effective_model = "gpt-5.4-mini"
        t._begin_turn()
        t._turn_id = "U1"
        await t._route_server_message({
            "jsonrpc": "2.0", "method": "thread/tokenUsage/updated",
            "params": {"threadId": "T1", "turnId": "U1", "tokenUsage": {
                "last": {"totalTokens": 12104, "inputTokens": 11989,
                         "cachedInputTokens": 9088, "outputTokens": 115,
                         "reasoningOutputTokens": 0},
                "total": {"totalTokens": 12104, "inputTokens": 11989,
                          "cachedInputTokens": 9088, "outputTokens": 115}}}})
        await t._route_server_message({
            "jsonrpc": "2.0", "method": "thread/tokenUsage/updated",
            "params": {"threadId": "T1", "turnId": "U1", "tokenUsage": {
                "last": {"totalTokens": 12214, "inputTokens": 12151,
                         "cachedInputTokens": 11648, "outputTokens": 63,
                         "reasoningOutputTokens": 0},
                "total": {"totalTokens": 24318, "inputTokens": 24140,
                          "cachedInputTokens": 20736, "outputTokens": 178}}}})
        await t._route_server_message({
            "jsonrpc": "2.0", "method": "turn/completed",
            "params": {"threadId": "T1",
                       "turn": {"id": "U1", "status": "completed"}}})

        lines = _ledger_lines(workspace)
        sources = sorted(l["source"] for l in lines)
        assert sources == ["management", "subagent:codex"]

        body = app_client.get(f"/api/v2/projects/{pid}/cost?window=total").json()
        # Management block unchanged by the codex line: $15 only.
        assert body["by_currency"] == {"USD": pytest.approx(15.0)}
        assert [b["source"] for b in body["breakdown"]] == ["management"]

        assert len(body["subagents"]) == 1
        entry = body["subagents"][0]
        assert entry["source"] == "subagent:codex"
        assert entry["provider"] == "openai"
        # Subset normalization on the FINAL `last`: uncached = 12151 - 11648 = 503.
        assert entry["tokens"]["uncached_input"] == 503
        assert entry["tokens"]["cache_read"] == 11648
        assert entry["tokens"]["cache_write"] == 0
        assert entry["tokens"]["output"] == 63
        # Codex emits no cost → null in the display block.
        assert entry["reported_cost"] is None
