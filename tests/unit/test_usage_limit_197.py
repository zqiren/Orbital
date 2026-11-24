# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Spec 197 — an outside agent that hit its usage limit is known as limited.

ONE limited fact per agent in the existing quota store (``limited`` plus a
top-level ``resets_at``, newest observation wins). It is set from the turn's
own failure at the one terminal route every transport ends in
(``LifecycleObserver.on_error``), from structured transport signals first and
the shared text classifier otherwise. It clears when the reset passes (read
time), on the agent's next successful turn, on a fresh not-limited reading,
and on an account switch. Only a KNOWN future reset reorders Orbital's own
picks.

Fixtures: the Claude shapes are T3 Code's recorded ``usage_limit`` transcript
(CLI 2.1.326, MIT) and Orbital's own test_left_off_138 rows; the Codex shapes
are T3's CodexAdapterV2 test scenarios (error notification with
``codexErrorInfo``, ``turn/completed`` with ``turn.error``).
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timedelta, timezone

import pytest

from agent_os.daemon_v2 import quota_store as qs
from agent_os.daemon_v2.quota_store import (
    QuotaStore,
    normalize_claude_rate_limit,
    normalize_codex_snapshot,
    note_turn_failure,
    prefer_not_limited,
)
from agent_os.onboarding import history_types as ht

NOW = time.time()
SOON = int(NOW + 3 * 3600)          # a reset three hours from now
LATER = int(NOW + 5 * 3600)


def _iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()


@pytest.fixture(autouse=True)
def _no_default_store():
    prev = qs.get_quota_store()
    qs.install(None)
    yield
    qs.install(prev)


@pytest.fixture
def store(tmp_path):
    s = QuotaStore(str(tmp_path / "quota.json"))
    qs.install(s)
    return s


def _set_tz(monkeypatch, name: str) -> None:
    if not hasattr(time, "tzset"):
        pytest.skip("time.tzset is POSIX-only")
    monkeypatch.setenv("TZ", name)
    time.tzset()


@pytest.fixture
def shanghai(monkeypatch):
    _set_tz(monkeypatch, "Asia/Shanghai")
    yield
    monkeypatch.delenv("TZ", raising=False)
    time.tzset()


# ---------------------------------------------------------------------------
# Readings: one fact (limited + resets_at)
# ---------------------------------------------------------------------------

class TestReadings:
    def test_claude_rejected_carries_its_reset(self):
        snap = normalize_claude_rate_limit(
            {"status": "rejected", "resetsAt": SOON, "rateLimitType": "five_hour",
             "overageStatus": "rejected"})
        assert snap["limited"] is True
        assert snap["resets_at"] == _iso(SOON)

    @pytest.mark.parametrize("overage", ["allowed", "allowed_warning"])
    def test_claude_with_paid_extra_usage_is_not_limited(self, overage):
        snap = normalize_claude_rate_limit(
            {"status": "rejected", "resetsAt": SOON, "rateLimitType": "five_hour",
             "overageStatus": overage,
             "unifiedWindows": {"five_hour": {"utilization": 1.0, "resetsAt": SOON}}})
        assert snap["limited"] is False
        assert snap["resets_at"] is None

    def test_claude_allowed_reading_is_not_limited(self):
        snap = normalize_claude_rate_limit(
            {"status": "allowed", "resetsAt": SOON, "rateLimitType": "five_hour",
             "utilization": 0.4})
        assert snap["limited"] is False and snap["resets_at"] is None

    def test_codex_reached_takes_the_latest_full_window_reset(self):
        snap = normalize_codex_snapshot({
            "primary": {"usedPercent": 100, "windowDurationMins": 300, "resetsAt": SOON},
            "secondary": {"usedPercent": 100, "windowDurationMins": 10080, "resetsAt": LATER},
            "rateLimitReachedType": "rate_limit_reached"})
        assert snap["limited"] is True
        assert snap["resets_at"] == _iso(LATER)

    def test_codex_reached_without_a_full_window_has_unknown_reset(self):
        snap = normalize_codex_snapshot({
            "primary": {"usedPercent": 40, "windowDurationMins": 300, "resetsAt": SOON},
            "rateLimitReachedType": "workspace_owner_credits_depleted"})
        assert snap["limited"] is True and snap["resets_at"] is None


# ---------------------------------------------------------------------------
# The store: any agent, expiry at read, clears
# ---------------------------------------------------------------------------

class TestStore:
    def test_any_agent_can_hold_a_limit_and_it_persists(self, tmp_path, store):
        store.mark_limited("kimi-code", _iso(SOON))
        again = QuotaStore(str(tmp_path / "quota.json"))
        snap = again.get("kimi-code")
        assert snap["limited"] is True and snap["resets_at"] == _iso(SOON)
        assert snap["windows"] == []   # an older build reads "Limit reached"

    def test_mark_keeps_known_windows_and_plan(self, store):
        store.record("codex", normalize_codex_snapshot({
            "primary": {"usedPercent": 30, "windowDurationMins": 300, "resetsAt": SOON},
            "planType": "plus"}))
        store.mark_limited("codex", _iso(SOON))
        snap = store.get("codex")
        assert snap["limited"] is True and snap["plan"] == "plus"
        assert [w["kind"] for w in snap["windows"]] == ["five_hour"]

    def test_a_passed_reset_is_void_when_read(self, store):
        store.mark_limited("kimi-code", _iso(NOW - 60))
        assert store.get("kimi-code") is None
        assert "kimi-code" not in store.snapshots()
        store.mark_limited("codex", _iso(NOW - 60))
        assert store.is_limited("codex") is False

    def test_next_success_clears_and_is_silent_otherwise(self, store):
        seen = []
        store.set_on_change(lambda a, s: seen.append((a, s)))
        store.clear_limit("codex")                     # nothing stored
        store.record("codex", normalize_codex_snapshot({
            "primary": {"usedPercent": 30, "windowDurationMins": 300, "resetsAt": SOON}}))
        seen.clear()
        store.clear_limit("codex")                     # stored, not limited
        assert seen == []
        store.mark_limited("codex", _iso(SOON))
        store.clear_limit("codex")
        assert store.get("codex")["limited"] is False
        assert store.get("codex")["resets_at"] is None
        store.mark_limited("kimi-code", None)
        store.clear_limit("kimi-code")                 # limit-only entry drops
        assert store.get("kimi-code") is None

    def test_a_fresh_not_limited_reading_wins(self, store):
        store.mark_limited("claude-code", _iso(SOON))
        store.record("claude-code", normalize_claude_rate_limit(
            {"status": "allowed", "rateLimitType": "five_hour", "utilization": 0.1,
             "resetsAt": SOON}))
        assert store.is_limited("claude-code") is False

    def test_account_switch_drops_it(self, store):
        store.mark_limited("codex", _iso(SOON))
        store.clear("codex")
        assert store.get("codex") is None

    def test_only_a_known_future_reset_is_avoided(self, store):
        store.mark_limited("codex", _iso(SOON))
        store.mark_limited("kimi-code", None)
        assert store.avoids("codex") is True
        assert store.avoids("kimi-code") is False        # unknown reset
        assert store.is_limited("kimi-code") is True     # still shown
        assert store.avoids("claude-code") is False

    def test_prefer_not_limited_is_a_stable_partition(self, store):
        store.mark_limited("claude-code", _iso(SOON))
        assert prefer_not_limited(["claude-code", "codex", "pi"]) == ["codex", "pi", "claude-code"]
        store.mark_limited("codex", _iso(SOON))
        store.mark_limited("pi", _iso(SOON))
        assert prefer_not_limited(["claude-code", "codex", "pi"]) == ["claude-code", "codex", "pi"]

    def test_prefer_not_limited_without_a_store_is_identity(self):
        assert prefer_not_limited(["a", "b"]) == ["a", "b"]

    def test_an_unreadable_row_is_ignored_on_load(self, tmp_path):
        path = tmp_path / "quota.json"
        path.write_text(json.dumps({"version": 1, "snapshots": {
            "kimi-code": {"agent": "kimi-code", "observed_at": _iso(NOW),
                          "plan": None, "limited": True, "windows": [],
                          "resets_at": _iso(SOON)},
            "cursor": {"agent": "cursor", "windows": []},     # no observed_at
            "codex": "garbage",
        }}), encoding="utf-8")
        assert set(QuotaStore(str(path)).snapshots()) == {"kimi-code"}


# ---------------------------------------------------------------------------
# The shared text classifier
# ---------------------------------------------------------------------------

class TestClassifierText:
    def test_codex_try_again_today_is_local_time(self, shanghai):
        at = "2026-11-24T06:00:00+00:00"       # 14:00 in Shanghai
        stop = ht.classify_error_text(
            "You've hit your usage limit. Upgrade to Pro (https://chatgpt.com/explore/pro), "
            "visit https://chatgpt.com/codex/settings/usage to purchase more credits "
            "or try again at 5:23 PM.", at=at)
        assert stop.cause == ht.CAUSE_USAGE_LIMIT
        assert stop.resets_at == "2026-11-24T09:23:00+00:00"   # 17:23 Shanghai

    def test_codex_try_again_with_a_date(self, shanghai):
        stop = ht.classify_error_text(
            "You’ve hit your usage limit. Visit https://chatgpt.com/codex/settings/usage "
            "to purchase more credits or try again at Jan 3rd, 2027 5:23 PM.",
            at="2026-11-24T06:00:00+00:00")
        assert stop.resets_at == "2027-01-03T09:23:00+00:00"

    def test_a_reset_with_no_zone_is_local(self, shanghai):
        stop = ht.classify_error_text(
            "You've hit your usage limit · resets 9:29pm", at="2026-11-24T06:00:00+00:00")
        assert stop.resets_at == "2026-11-24T13:29:00+00:00"

    def test_a_named_zone_still_wins(self, shanghai):
        stop = ht.classify_error_text(
            "You've hit your session limit · resets 10pm (UTC)", at="2026-11-24T06:00:00+00:00")
        assert stop.resets_at == "2026-11-24T22:00:00+00:00"


# ---------------------------------------------------------------------------
# note_turn_failure: the one helper at the terminal route
# ---------------------------------------------------------------------------

class TestNoteTurnFailure:
    def test_structured_reason_takes_the_reset_from_the_snapshot(self, store):
        qs.publish_claude_rate_limit({"status": "rejected", "resetsAt": SOON,
                                      "rateLimitType": "five_hour"})
        got = note_turn_failure("claude-code", "API Error: Request rejected (429)",
                                reason="usage_limit")
        assert got == {"reason": "usage_limit", "resets_at": _iso(SOON)}
        assert store.avoids("claude-code") is True

    def test_codex_full_window_supplies_the_reset(self, store):
        store.record("codex", normalize_codex_snapshot({
            "primary": {"usedPercent": 100, "windowDurationMins": 300, "resetsAt": SOON}}))
        got = note_turn_failure("codex", "Error: something", reason="usage_limit")
        assert got["resets_at"] == _iso(SOON)

    def test_a_stale_full_window_is_not_a_reset(self, store):
        store.record("codex", normalize_codex_snapshot({
            "primary": {"usedPercent": 100, "windowDurationMins": 300,
                        "resetsAt": int(NOW - 600)}}))
        got = note_turn_failure("codex", "Error: limit", reason="usage_limit")
        assert got["resets_at"] is None

    def test_text_fallback_for_the_reset(self, store, shanghai):
        got = note_turn_failure(
            "codex", "Error: You've hit your usage limit. … or try again at 11:58 PM.",
            reason="usage_limit", at="2026-11-24T06:00:00+00:00")
        assert got["resets_at"] == "2026-11-24T15:58:00+00:00"

    def test_text_only_classification_for_any_agent(self, store):
        got = note_turn_failure("pi", "You've hit your usage limit · resets 10pm (UTC)",
                                at=_iso(NOW))
        assert got["reason"] == "usage_limit" and got["resets_at"] is not None
        assert store.is_limited("pi") is True

    def test_unknown_reset_is_noted_but_not_avoided(self, store):
        got = note_turn_failure("kimi-code", "Error: rate limit reached for your plan")
        assert got == {"reason": "usage_limit", "resets_at": None}
        assert store.is_limited("kimi-code") and not store.avoids("kimi-code")

    def test_other_failures_are_not_limits(self, store):
        assert note_turn_failure("codex", "Error: model is not supported") is None
        assert note_turn_failure("antigravity", "You've hit your usage limit",
                                 reason="region_unavailable") is None
        assert note_turn_failure("worker:abc-1", "You've hit your usage limit") is None
        assert store.snapshots() == {}

    def test_without_a_store_it_still_classifies(self):
        got = note_turn_failure("codex", "x", reason="usage_limit")
        assert got == {"reason": "usage_limit", "resets_at": None}


# ---------------------------------------------------------------------------
# LifecycleObserver: the error row carries the fact; success clears it
# ---------------------------------------------------------------------------

class _AgentManager:
    def __init__(self):
        self.injections = []

    async def inject_system_message(self, project_id, content, **kwargs):
        self.injections.append((project_id, content, kwargs))


class _WS:
    def __init__(self):
        self.events = []

    def broadcast(self, project_id, payload):
        self.events.append(payload)


def _observer():
    from agent_os.daemon_v2.lifecycle_observer import LifecycleObserver
    am, ws = _AgentManager(), _WS()
    return LifecycleObserver(am, ws), am, ws


class TestLifecycle:
    @pytest.mark.asyncio
    async def test_limit_row_meta_ws_and_manager_clause(self, store, shanghai):
        store.mark_limited("codex", None)
        store.record("codex", normalize_codex_snapshot({
            "primary": {"usedPercent": 100, "windowDurationMins": 300, "resetsAt": SOON},
            "rateLimitReachedType": "rate_limit_reached"}))
        obs, am, ws = _observer()
        await obs.on_error("p1", "codex",
                           "Error: You've hit your usage limit. … or try again at 5:23 PM.",
                           "/t/codex.log", session_id="s1",
                           detail={"reason": "usage_limit"})
        (_pid, content, kw), = am.injections
        meta = kw["meta"]
        assert meta["reason"] == "usage_limit"
        assert meta["resets_at"] == _iso(SOON)
        local = datetime.fromtimestamp(SOON).strftime("%H:%M")
        # one short clause, ahead of the vendor's own words, at the tail row
        assert content.startswith(
            f"[Sub-agent] codex stopped with error: Usage limit reached; back at {local} local time")
        assert "try again at 5:23 PM" in content
        err = [e for e in ws.events if e["type"] == "sub_agent.error"][0]
        assert err["reason"] == "usage_limit" and err["resets_at"] == _iso(SOON)

    @pytest.mark.asyncio
    async def test_unknown_reset_clause(self, store):
        obs, am, _ws = _observer()
        await obs.on_error("p1", "pi", "Error: insufficient credits", "/t", session_id="s1")
        content, meta = am.injections[0][1], am.injections[0][2]["meta"]
        assert "Usage limit reached; reset time unknown." in content
        assert meta["reason"] == "usage_limit" and meta["resets_at"] is None

    @pytest.mark.asyncio
    async def test_a_plain_error_row_is_unchanged(self, store):
        obs, am, ws = _observer()
        await obs.on_error("p1", "codex", "Error: boom", "/t", session_id="s1")
        content, meta = am.injections[0][1], am.injections[0][2]["meta"]
        assert content == "[Sub-agent] codex stopped with error: Error: boom. Transcript: /t"
        assert "reason" not in meta and "resets_at" not in meta
        assert store.snapshots() == {}

    @pytest.mark.asyncio
    async def test_region_reason_is_kept(self, store):
        obs, am, _ws = _observer()
        await obs.on_error("p1", "antigravity", "no licence", "/t", session_id="s1",
                           detail={"reason": "region_unavailable", "name": "Antigravity"})
        meta = am.injections[0][2]["meta"]
        assert meta["reason"] == "region_unavailable" and meta["name"] == "Antigravity"

    @pytest.mark.asyncio
    async def test_successful_turn_clears(self, store):
        store.mark_limited("codex", _iso(SOON))
        obs, _am, _ws = _observer()
        await obs.on_completed("p1", "codex", "done", "/t", session_id="s1")
        assert store.is_limited("codex") is False


# ---------------------------------------------------------------------------
# Claude Code (SDK transport)
# ---------------------------------------------------------------------------

def _sdk():
    from agent_os.agent.transports.sdk_transport import SDKTransport
    return SDKTransport()


def _synthetic(text, error="rate_limit"):
    from claude_agent_sdk.types import AssistantMessage, TextBlock
    return AssistantMessage(content=[TextBlock(text=text)], model="<synthetic>",
                            error=error)


def _result(text, *, is_error=True, status=429):
    from claude_agent_sdk.types import ResultMessage
    return ResultMessage(subtype="success", duration_ms=1, duration_api_ms=0,
                         is_error=is_error, num_turns=1, session_id="s",
                         result=text, api_error_status=status,
                         terminal_reason="api_error")


T3_TEXT = ("API Error: Request rejected (429) · All credentials for model "
           "claude-sonnet-4-6 are cooling down via provider claude")


class TestClaude:
    def test_t3_fixture_one_error_no_reply_bubble(self, store):
        t = _sdk()
        events = t._message_to_events(_synthetic(T3_TEXT))
        events += t._message_to_events(_result(T3_TEXT))
        assert [e.event_type for e in events] == ["error"]
        assert events[0].data["reason"] == "usage_limit"
        assert events[0].data["error"] == T3_TEXT

    def test_rejected_event_in_the_turn_marks_the_failure(self, store):
        from claude_agent_sdk.types import RateLimitEvent, RateLimitInfo
        t = _sdk()
        raw = {"status": "rejected", "resetsAt": SOON, "rateLimitType": "five_hour",
               "overageStatus": "rejected"}
        t._message_to_events(RateLimitEvent(
            rate_limit_info=RateLimitInfo(status="rejected", resets_at=SOON,
                                          rate_limit_type="five_hour",
                                          overage_status="rejected", raw=raw),
            uuid="u", session_id="s"))
        text = "You've hit your session limit · resets 10pm (Asia/Shanghai)"
        events = t._message_to_events(_synthetic(text))
        events += t._message_to_events(_result(text, status=None))
        assert [e.event_type for e in events] == ["error"]
        assert events[0].data["reason"] == "usage_limit"
        assert store.get("claude-code")["resets_at"] == _iso(SOON)

    def test_overage_allowed_rejection_does_not_mark(self, store):
        from claude_agent_sdk.types import RateLimitEvent, RateLimitInfo
        t = _sdk()
        raw = {"status": "rejected", "resetsAt": SOON, "rateLimitType": "five_hour",
               "overageStatus": "allowed"}
        t._message_to_events(RateLimitEvent(
            rate_limit_info=RateLimitInfo(status="rejected", resets_at=SOON,
                                          overage_status="allowed", raw=raw),
            uuid="u", session_id="s"))
        events = t._message_to_events(_result("Error: boom", status=500))
        assert "reason" not in events[0].data

    def test_the_turn_flag_does_not_leak_into_the_next_turn(self, store):
        t = _sdk()
        t._message_to_events(_synthetic(T3_TEXT))
        t._message_to_events(_result(T3_TEXT))
        events = t._message_to_events(_result("Error: other", status=500))
        assert "reason" not in events[0].data

    def test_other_errors_keep_no_reason_and_still_drop_the_synthetic_bubble(self):
        t = _sdk()
        events = t._message_to_events(_synthetic("Invalid API key", error="authentication_failed"))
        events += t._message_to_events(_result("Invalid API key", status=401))
        assert [e.event_type for e in events] == ["error"]
        assert "reason" not in events[0].data

    def test_a_synthetic_message_without_an_error_result_is_still_shown(self):
        t = _sdk()
        events = t._message_to_events(_synthetic("odd text"))
        events += t._message_to_events(_result("fine", is_error=False, status=None))
        assert [(e.event_type, e.raw_text) for e in events] == [("message", "odd text")]

    def test_a_real_reply_is_untouched(self):
        from claude_agent_sdk.types import AssistantMessage, TextBlock
        t = _sdk()
        events = t._message_to_events(AssistantMessage(
            content=[TextBlock(text="hello")], model="claude-sonnet-4-6"))
        assert [(e.event_type, e.raw_text) for e in events] == [("message", "hello")]


# ---------------------------------------------------------------------------
# Codex (app-server transport)
# ---------------------------------------------------------------------------

def _codex():
    from agent_os.agent.transports.codex_transport import CodexTransport
    t = CodexTransport()
    t._thread_id = "T1"
    t._begin_turn()
    return t


def _drain(t) -> list:
    out = []
    while not t._event_queue.empty():
        out.append(t._event_queue.get_nowait())
    return out


CODEX_TEXT = ("You've hit your usage limit. Upgrade to Pro (https://chatgpt.com/explore/pro), "
              "visit https://chatgpt.com/codex/settings/usage to purchase more credits or "
              "try again at 5:23 PM.")


def _err(info, *, message=CODEX_TEXT, will_retry=False):
    return {"jsonrpc": "2.0", "method": "error", "params": {
        "threadId": "T1", "turnId": "u1", "willRetry": will_retry,
        "error": {"message": message, "codexErrorInfo": info}}}


def _completed(error=None):
    turn = {"id": "u1", "status": "failed", "items": []}
    if error is not None:
        turn["error"] = error
    return {"jsonrpc": "2.0", "method": "turn/completed",
            "params": {"threadId": "T1", "turn": turn}}


class TestCodex:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("info", [
        "usageLimitExceeded", "rateLimitExceeded",
        {"responseTooManyFailedAttempts": {"httpStatusCode": 429}},
    ])
    async def test_usage_limit_codes(self, info):
        t = _codex()
        t._alive = False                          # no live read
        await t._route_server_message(_err(info))
        (ev,) = _drain(t)
        assert ev.event_type == "error" and ev.data["reason"] == "usage_limit"

    @pytest.mark.asyncio
    @pytest.mark.parametrize("info", [
        "other", {"responseTooManyFailedAttempts": {"httpStatusCode": 500}},
        {"responseStreamDisconnected": {"httpStatusCode": None}}, None,
    ])
    async def test_other_codes_have_no_reason(self, info):
        t = _codex()
        await t._route_server_message(_err(info, message="Connection lost."))
        (ev,) = _drain(t)
        assert "reason" not in ev.data

    @pytest.mark.asyncio
    async def test_a_later_error_in_the_same_turn_keeps_the_reason(self):
        t = _codex()
        t._alive = False
        await t._route_server_message(_err("usageLimitExceeded", will_retry=True))
        await t._route_server_message(_err("other", message="stream closed"))
        assert [e.data.get("reason") for e in _drain(t)] == ["usage_limit", "usage_limit"]

    @pytest.mark.asyncio
    async def test_failure_only_on_turn_completed_still_surfaces(self):
        t = _codex()
        await t._route_server_message(_completed(
            {"message": CODEX_TEXT, "codexErrorInfo": "usageLimitExceeded"}))
        evs = _drain(t)
        assert [e.event_type for e in evs] == ["error", "turn_complete"]
        assert evs[0].data["reason"] == "usage_limit"
        assert evs[0].raw_text == f"Error: {CODEX_TEXT}"
        assert evs[1].data["cause"] == "error"

    @pytest.mark.asyncio
    async def test_no_second_error_when_the_notification_came(self):
        t = _codex()
        t._alive = False
        await t._route_server_message(_err("usageLimitExceeded"))
        await t._route_server_message(_completed({"message": CODEX_TEXT}))
        assert [e.event_type for e in _drain(t)] == ["error", "turn_complete"]

    @pytest.mark.asyncio
    async def test_the_next_turn_starts_clean(self):
        t = _codex()
        t._alive = False
        await t._route_server_message(_err("usageLimitExceeded"))
        await t._route_server_message(_completed())
        _drain(t)
        t._begin_turn()
        await t._route_server_message(_err("other", message="boom"))
        assert "reason" not in _drain(t)[0].data

    @pytest.mark.asyncio
    async def test_a_limit_reads_the_rate_limits_once(self, store):
        import asyncio
        t = _codex()
        t._alive = True
        calls = []

        async def fake_request(method, params=None, timeout=30.0):
            calls.append(method)
            return {"rateLimitsByLimitId": {"codex": {
                "limitId": "codex",
                "primary": {"usedPercent": 100, "windowDurationMins": 300, "resetsAt": SOON},
                "rateLimitReachedType": "rate_limit_reached"}}}

        t._request = fake_request
        await t._route_server_message(_err("usageLimitExceeded", will_retry=True))
        await t._route_server_message(_err("usageLimitExceeded"))
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        assert calls == ["account/rateLimits/read"]
        assert store.get("codex")["resets_at"] == _iso(SOON)


# ---------------------------------------------------------------------------
# ACP: vendor wording lives in the manifest's refusal_patterns
# ---------------------------------------------------------------------------

class TestAcpManifestReason:
    def test_a_usage_limit_pattern_rides_the_existing_route(self):
        from agent_os.agent.transports.acp_sdk_transport import ACPSDKTransport
        t = ACPSDKTransport.__new__(ACPSDKTransport)
        from agent_os.agent.transports.acp_sdk_transport import _compile_refusal_patterns
        t._refusal_patterns = _compile_refusal_patterns(
            [{"regex": r"quota exhausted", "reason": "usage_limit"}])
        t._agent_name = "Vendor"
        assert t._match_refusal("Error: quota exhausted (429)", whole=False) == "usage_limit"
        ev = t._refusal_event("usage_limit", "quota exhausted")
        assert ev.data["reason"] == "usage_limit"

    def test_no_shipped_manifest_carries_an_unobserved_limit_pattern(self):
        import glob
        import yaml
        root = os.path.join(os.path.dirname(qs.__file__), "..", "agents", "manifests")
        for path in glob.glob(os.path.join(root, "*.yaml")):
            with open(path, encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            pats = (data.get("runtime") or {}).get("refusal_patterns") or []
            assert all(p.get("reason") != "usage_limit" for p in pats), path


# ---------------------------------------------------------------------------
# Orbital's own picks
# ---------------------------------------------------------------------------

from tests.unit.keyless_harness import agent_status, build_client, create_project  # noqa: E402


def _project(tmp_path, *, keyed, default=None, statuses=None):
    tc, deps = build_client(tmp_path, keyed=keyed, statuses=statuses)
    ws = tmp_path / "ws-limit"
    ws.mkdir(parents=True)
    row = create_project(tc, ws, name="limit")
    if default:
        deps["project_store"].update_project(row["project_id"],
                                             {"default_pinned_target": default})
    return tc, deps, row["project_id"]


class TestPicks:
    @pytest.fixture(autouse=True)
    def _no_env_key(self, monkeypatch):
        monkeypatch.delenv("AGENT_OS_API_KEY", raising=False)

    def test_keyless_first_worker_skips_a_limited_one(self, tmp_path, store):
        store.mark_limited("claude-code", _iso(SOON))
        tc, deps, pid = _project(tmp_path, keyed=False)
        assert tc.get(f"/api/v2/projects/{pid}").json()["effective_pinned_target"] == "codex"
        body = tc.post(f"/api/v2/agents/{pid}/new-session").json()
        assert body["pinned_target"] == "codex"

    def test_the_remembered_agent_yields_but_is_never_rewritten(self, tmp_path, store):
        store.mark_limited("codex", _iso(SOON))
        for keyed, expected in ((False, "claude-code"), (True, None)):
            tc, deps, pid = _project(tmp_path / str(keyed), keyed=keyed, default="codex")
            got = tc.get(f"/api/v2/projects/{pid}").json()
            assert got["default_pinned_target"] == "codex"
            assert got["effective_pinned_target"] == expected
            assert deps["project_store"].get_project(pid)["default_pinned_target"] == "codex"

    def test_an_unknown_reset_never_reorders(self, tmp_path, store):
        store.mark_limited("codex", None)
        store.mark_limited("claude-code", None)
        tc, _deps, pid = _project(tmp_path, keyed=False, default="codex")
        assert tc.get(f"/api/v2/projects/{pid}").json()["effective_pinned_target"] == "codex"

    def test_all_limited_keeps_todays_choice(self, tmp_path, store):
        store.mark_limited("codex", _iso(SOON))
        store.mark_limited("claude-code", _iso(SOON))
        tc, _deps, pid = _project(tmp_path, keyed=False, default="codex")
        assert tc.get(f"/api/v2/projects/{pid}").json()["effective_pinned_target"] == "codex"

    def test_a_passed_reset_restores_the_remembered_agent(self, tmp_path, store):
        store.mark_limited("codex", _iso(NOW - 5))
        tc, _deps, pid = _project(tmp_path, keyed=False, default="codex")
        assert tc.get(f"/api/v2/projects/{pid}").json()["effective_pinned_target"] == "codex"

    def test_available_workers_still_lists_a_limited_agent(self, tmp_path, store):
        store.mark_limited("codex", _iso(SOON))
        _tc, deps, pid = _project(tmp_path, keyed=False)
        am = deps["agent_manager"]
        assert am.available_worker_slugs(pid) == ["claude-code", "codex"]

    @pytest.mark.asyncio
    async def test_router_prefers_a_not_limited_agent(self, store, monkeypatch):
        from agent_os.api.routes import workbench

        class _Subs:
            def can_ask_once(self, slug):
                return True

        class _AM:
            def available_worker_slugs(self, pid):
                return ["claude-code", "codex"]

        monkeypatch.setattr(workbench, "_sub_agent_manager", _Subs())
        monkeypatch.setattr(workbench, "_agent_manager", _AM())
        assert await workbench._router_agent("q") == "claude-code"
        store.mark_limited("claude-code", _iso(SOON))
        assert await workbench._router_agent("q") == "codex"


# ---------------------------------------------------------------------------
# Finished-session reader agrees with the live fact
# ---------------------------------------------------------------------------

def test_left_off_takes_the_rows_reset(tmp_path):
    from agent_os.onboarding import left_off as lo
    ws = tmp_path / "ws"
    sessions = ws / "orbital" / "sessions"
    sessions.mkdir(parents=True)
    rows = [
        {"role": "user", "content": "go", "timestamp": _iso(NOW - 60), "target": "codex"},
        {"role": "system", "source": "daemon", "timestamp": _iso(NOW),
         "content": ("[Sub-agent] codex stopped with error: Usage limit reached; back at "
                     "17:23 local time. Error: limit hit. Transcript: /t"),
         "_meta": {"event": "sub_agent_terminal", "kind": "error",
                   "reason": "usage_limit", "resets_at": _iso(SOON)}},
    ]
    path = sessions / "s1.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    parsed = lo.read_orbital_session(str(path))
    assert parsed.entry.stop.cause == "usage_limit"
    assert parsed.entry.stop.resets_at == _iso(SOON)
