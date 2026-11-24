# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Spec 099 §3.2 — subscription quota capture (QuotaStore).

Fixture payloads are the shapes live-probed 2026-09-27 (spec §2.1):
claude-code's ``rate_limit_event`` (per-window split only in
``RateLimitInfo.raw["unifiedWindows"]``) and codex's
``account/rateLimits/read`` result (windows identified by
``windowDurationMins``, never by primary/secondary position).
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time

import pytest

from agent_os.daemon_v2 import quota_store as qs
from agent_os.daemon_v2.quota_store import (
    QuotaStore,
    normalize_claude_rate_limit,
    normalize_codex_snapshot,
    pick_codex_snapshot,
)

FIVE_H_RESET = 1790517600
WEEK_RESET = 1790672400

CLAUDE_RAW = {
    "status": "allowed",
    "resetsAt": FIVE_H_RESET,
    "rateLimitType": "five_hour",
    "overageStatus": "rejected",
    "overageDisabledReason": "out_of_credits",
    "isUsingOverage": False,
    "unifiedWindows": {
        "five_hour": {"utilization": 0.79, "resetsAt": FIVE_H_RESET},
        "seven_day": {"utilization": 0.08, "resetsAt": WEEK_RESET},
    },
}

CODEX_SNAPSHOT = {
    "limitId": "codex",
    "primary": {"usedPercent": 18, "windowDurationMins": 300,
                "resetsAt": 1790517805},
    "secondary": {"usedPercent": 3, "windowDurationMins": 10080,
                  "resetsAt": 1791104605},
    "planType": "plus",
    "rateLimitReachedType": None,
    "credits": {"hasCredits": False, "unlimited": False, "balance": None},
}

CODEX_READ_RESULT = {
    "rateLimits": dict(CODEX_SNAPSHOT),
    "rateLimitsByLimitId": {"codex": dict(CODEX_SNAPSHOT)},
    "rateLimitResetCredits": None,
}


def _iso(ts: int) -> str:
    from datetime import datetime, timezone
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()


def _windows(snap: dict) -> dict:
    return {w["kind"]: w for w in snap["windows"]}


@pytest.fixture(autouse=True)
def _no_default_store():
    """Never leak a process-wide store between tests."""
    prev = qs.get_quota_store()
    qs.install(None)
    yield
    qs.install(prev)


# ---------------------------------------------------------------------------
# Normalization — claude-code
# ---------------------------------------------------------------------------

class TestNormalizeClaude:
    def test_unified_windows_become_five_hour_and_weekly(self):
        snap = normalize_claude_rate_limit(CLAUDE_RAW, status="allowed")
        assert snap["agent"] == "claude-code"
        assert snap["plan"] is None
        assert snap["limited"] is False
        w = _windows(snap)
        assert set(w) == {"five_hour", "weekly"}
        # utilization 0..1 -> used_pct x100
        assert w["five_hour"]["used_pct"] == pytest.approx(79.0)
        assert w["weekly"]["used_pct"] == pytest.approx(8.0)
        assert w["five_hour"]["resets_at"] == _iso(FIVE_H_RESET)
        assert w["weekly"]["resets_at"] == _iso(WEEK_RESET)
        assert snap["observed_at"]

    def test_accepts_sdk_rate_limit_info(self):
        from claude_agent_sdk.types import RateLimitInfo
        info = RateLimitInfo(status="allowed", resets_at=FIVE_H_RESET,
                             rate_limit_type="five_hour", utilization=0.79,
                             raw=CLAUDE_RAW)
        snap = normalize_claude_rate_limit(info)
        assert _windows(snap)["weekly"]["used_pct"] == pytest.approx(8.0)

    def test_rejected_means_limited(self):
        raw = dict(CLAUDE_RAW, status="rejected")
        snap = normalize_claude_rate_limit(raw)
        assert snap["limited"] is True

    def test_per_model_weekly_windows_map(self):
        raw = dict(CLAUDE_RAW, unifiedWindows={
            "seven_day_opus": {"utilization": 0.6, "resetsAt": WEEK_RESET},
            "seven_day_sonnet": {"utilization": 0.1, "resetsAt": WEEK_RESET},
        })
        w = _windows(normalize_claude_rate_limit(raw))
        assert set(w) == {"weekly_opus", "weekly_sonnet"}

    def test_malformed_window_is_dropped_not_raised(self):
        raw = dict(CLAUDE_RAW, unifiedWindows={
            "five_hour": {"utilization": "lots", "resetsAt": FIVE_H_RESET},
            "seven_day": {"utilization": 0.08, "resetsAt": "soon"},
            "overage": {"utilization": 0.5},       # not a quota window
            "made_up": {"utilization": 0.5},       # unknown kind
            "bad": "not-a-dict",
        })
        w = _windows(normalize_claude_rate_limit(raw))
        assert set(w) == {"weekly"}
        assert w["weekly"]["resets_at"] is None

    def test_headline_fallback_when_unified_windows_absent(self):
        raw = {"status": "allowed_warning", "rateLimitType": "seven_day",
               "utilization": 0.91, "resetsAt": WEEK_RESET}
        w = _windows(normalize_claude_rate_limit(raw))
        assert w["weekly"]["used_pct"] == pytest.approx(91.0)

    def test_nothing_usable_returns_none(self):
        assert normalize_claude_rate_limit({"status": "allowed"}) is None
        assert normalize_claude_rate_limit(None) is None
        assert normalize_claude_rate_limit("garbage") is None

    def test_rejected_without_windows_still_records_the_limit(self):
        snap = normalize_claude_rate_limit({"status": "rejected"})
        assert snap["limited"] is True
        assert snap["windows"] == []


# ---------------------------------------------------------------------------
# Normalization — codex
# ---------------------------------------------------------------------------

class TestNormalizeCodex:
    def test_windows_identified_by_duration(self):
        snap = normalize_codex_snapshot(CODEX_SNAPSHOT)
        assert snap["agent"] == "codex"
        assert snap["plan"] == "plus"
        assert snap["limited"] is False
        w = _windows(snap)
        assert w["five_hour"]["used_pct"] == 18
        assert w["weekly"]["used_pct"] == 3
        assert w["five_hour"]["resets_at"] == _iso(1790517805)

    def test_swapped_positions_still_map_by_duration(self):
        swapped = dict(CODEX_SNAPSHOT, primary=CODEX_SNAPSHOT["secondary"],
                       secondary=CODEX_SNAPSHOT["primary"])
        w = _windows(normalize_codex_snapshot(swapped))
        assert w["five_hour"]["used_pct"] == 18
        assert w["weekly"]["used_pct"] == 3

    def test_unknown_duration_dropped(self):
        odd = dict(CODEX_SNAPSHOT, primary={"usedPercent": 50,
                                            "windowDurationMins": 60,
                                            "resetsAt": 1})
        w = _windows(normalize_codex_snapshot(odd))
        assert set(w) == {"weekly"}

    def test_reached_type_means_limited(self):
        snap = normalize_codex_snapshot(
            dict(CODEX_SNAPSHOT, rateLimitReachedType="rate_limit_reached"))
        assert snap["limited"] is True

    def test_malformed_is_tolerated(self):
        assert normalize_codex_snapshot(None) is None
        assert normalize_codex_snapshot({"primary": "x", "secondary": None}) is None
        snap = normalize_codex_snapshot(dict(CODEX_SNAPSHOT, primary={
            "usedPercent": "high", "windowDurationMins": 300}))
        assert set(_windows(snap)) == {"weekly"}

    def test_pick_prefers_by_limit_id_codex(self):
        other = dict(CODEX_SNAPSHOT, planType="pro")
        result = {"rateLimits": other,
                  "rateLimitsByLimitId": {"codex": CODEX_SNAPSHOT}}
        assert pick_codex_snapshot(result)["planType"] == "plus"

    def test_pick_falls_back_to_rate_limits(self):
        assert pick_codex_snapshot({"rateLimits": CODEX_SNAPSHOT,
                                    "rateLimitsByLimitId": None}) is CODEX_SNAPSHOT
        assert pick_codex_snapshot({}) is None
        assert pick_codex_snapshot("nope") is None


# ---------------------------------------------------------------------------
# Store: persistence, notification, merge, clear
# ---------------------------------------------------------------------------

class TestStore:
    def test_record_get_and_notify(self, tmp_path):
        seen = []
        store = QuotaStore(str(tmp_path / "quota.json"),
                           on_change=lambda a, s: seen.append((a, s)))
        snap = normalize_codex_snapshot(CODEX_SNAPSHOT)
        store.record("codex", snap)
        assert store.get("codex") == snap
        assert store.get("claude-code") is None
        assert seen == [("codex", snap)]

    def test_persistence_round_trip(self, tmp_path):
        path = str(tmp_path / "quota.json")
        store = QuotaStore(path)
        claude = normalize_claude_rate_limit(CLAUDE_RAW)
        store.record("claude-code", claude)
        # Atomic: no tmp file left behind; the file is valid JSON.
        assert os.path.isfile(path)
        assert not [p for p in os.listdir(tmp_path) if p != "quota.json"]
        with open(path, encoding="utf-8") as f:
            json.load(f)
        again = QuotaStore(path)
        assert again.get("claude-code") == claude

    @pytest.mark.asyncio
    async def test_write_happens_off_the_event_loop(self, tmp_path, monkeypatch):
        path = str(tmp_path / "quota.json")
        store = QuotaStore(path)
        loop_thread = __import__("threading").get_ident()
        writer_threads = []
        real_write = store._write_file

        def spy():
            writer_threads.append(__import__("threading").get_ident())
            real_write()

        monkeypatch.setattr(store, "_write_file", spy)
        store.record("codex", normalize_codex_snapshot(CODEX_SNAPSHOT))
        await store.drain_writes()
        assert writer_threads and writer_threads[0] != loop_thread
        assert QuotaStore(path).get("codex") is not None

    def test_corrupt_file_loads_empty(self, tmp_path):
        path = tmp_path / "quota.json"
        path.write_text("{not json", encoding="utf-8")
        store = QuotaStore(str(path))
        assert store.snapshots() == {}

    def test_bad_rows_ignored_on_load(self, tmp_path):
        # Spec 197: any agent may hold a limit now; a row without
        # observed_at or not a dict is still dropped.
        path = tmp_path / "quota.json"
        path.write_text(json.dumps({"version": 1, "snapshots": {
            "cursor": {"agent": "cursor", "windows": []},
            "codex": "garbage",
        }}), encoding="utf-8")
        assert QuotaStore(str(path)).snapshots() == {}

    def test_clear_drops_and_persists_and_notifies(self, tmp_path):
        path = str(tmp_path / "quota.json")
        seen = []
        store = QuotaStore(path, on_change=lambda a, s: seen.append((a, s)))
        store.record("codex", normalize_codex_snapshot(CODEX_SNAPSHOT))
        store.clear("codex")
        assert store.get("codex") is None
        assert seen[-1] == ("codex", None)
        assert QuotaStore(path).get("codex") is None
        # Clearing an agent with nothing stored is silent.
        n = len(seen)
        store.clear("claude-code")
        assert len(seen) == n

    def test_notify_failure_never_raises(self, tmp_path):
        def boom(agent, snap):
            raise RuntimeError("ws down")
        store = QuotaStore(str(tmp_path / "q.json"), on_change=boom)
        store.record("codex", normalize_codex_snapshot(CODEX_SNAPSHOT))
        assert store.get("codex") is not None

    def test_codex_push_merges_sparse_update(self, tmp_path):
        store = QuotaStore(str(tmp_path / "q.json"))
        store.record("codex", normalize_codex_snapshot(CODEX_SNAPSHOT))
        # Sparse rolling update: only the 5h window, no plan.
        store.merge_codex_push({"limitId": "codex", "primary": {
            "usedPercent": 40, "windowDurationMins": 300, "resetsAt": 1790517805},
            "planType": None, "rateLimitReachedType": None})
        w = _windows(store.get("codex"))
        assert w["five_hour"]["used_pct"] == 40
        assert w["weekly"]["used_pct"] == 3          # kept
        assert store.get("codex")["plan"] == "plus"  # null does not clear

    def test_codex_push_for_other_limit_id_ignored(self, tmp_path):
        store = QuotaStore(str(tmp_path / "q.json"))
        store.record("codex", normalize_codex_snapshot(CODEX_SNAPSHOT))
        store.merge_codex_push(dict(CODEX_SNAPSHOT, limitId="codex_other",
                                    primary={"usedPercent": 99,
                                             "windowDurationMins": 300}))
        assert _windows(store.get("codex"))["five_hour"]["used_pct"] == 18

    def test_codex_push_with_nothing_prior_records(self, tmp_path):
        store = QuotaStore(str(tmp_path / "q.json"))
        store.merge_codex_push(CODEX_SNAPSHOT)
        assert _windows(store.get("codex"))["weekly"]["used_pct"] == 3


# ---------------------------------------------------------------------------
# Codex on-demand refresh: single-flight, throttled, bounded
# ---------------------------------------------------------------------------

class TestCodexRefresh:
    @pytest.mark.asyncio
    async def test_single_flight(self, tmp_path):
        store = QuotaStore(str(tmp_path / "q.json"))
        calls = 0
        gate = asyncio.Event()

        async def reader():
            nonlocal calls
            calls += 1
            await gate.wait()
            return CODEX_READ_RESULT

        t1 = store.ensure_codex_fresh(reader)
        t2 = store.ensure_codex_fresh(reader)
        assert t1 is not None and t1 is t2
        gate.set()
        await t1
        assert calls == 1
        assert _windows(store.get("codex"))["five_hour"]["used_pct"] == 18

    @pytest.mark.asyncio
    async def test_fresh_snapshot_skips_read(self, tmp_path):
        store = QuotaStore(str(tmp_path / "q.json"))
        store.record("codex", normalize_codex_snapshot(CODEX_SNAPSHOT))

        async def reader():
            raise AssertionError("must not read while fresh")

        assert store.ensure_codex_fresh(reader) is None

    @pytest.mark.asyncio
    async def test_at_most_once_per_interval_even_on_failure(self, tmp_path):
        store = QuotaStore(str(tmp_path / "q.json"))
        calls = 0

        async def reader():
            nonlocal calls
            calls += 1
            return None  # codex not logged in / probe failed

        task = store.ensure_codex_fresh(reader)
        await task
        assert store.get("codex") is None
        assert store.ensure_codex_fresh(reader) is None
        assert calls == 1

    @pytest.mark.asyncio
    async def test_stale_snapshot_rereads(self, tmp_path):
        clock = [1000.0]
        store = QuotaStore(str(tmp_path / "q.json"), clock=lambda: clock[0])
        store.record("codex", normalize_codex_snapshot(
            CODEX_SNAPSHOT, clock=lambda: clock[0]))
        clock[0] += 61

        async def reader():
            return {"rateLimits": dict(CODEX_SNAPSHOT, primary={
                "usedPercent": 70, "windowDurationMins": 300,
                "resetsAt": 1790517805})}

        await store.ensure_codex_fresh(reader)
        assert _windows(store.get("codex"))["five_hour"]["used_pct"] == 70

    @pytest.mark.asyncio
    async def test_timeout_keeps_old_value_and_never_raises(self, tmp_path):
        store = QuotaStore(str(tmp_path / "q.json"), codex_timeout=0.05)

        async def reader():
            await asyncio.sleep(5)

        task = store.ensure_codex_fresh(reader)
        assert await task is None
        assert store.get("codex") is None

    @pytest.mark.asyncio
    async def test_reader_exception_never_raises(self, tmp_path):
        store = QuotaStore(str(tmp_path / "q.json"))

        async def reader():
            raise RuntimeError("codex died")

        assert await store.ensure_codex_fresh(reader) is None


# ---------------------------------------------------------------------------
# Process-wide publish helpers (the transports' entry point)
# ---------------------------------------------------------------------------

class TestPublish:
    def test_publish_without_store_is_a_noop(self):
        qs.publish_claude_rate_limit(CLAUDE_RAW)
        qs.publish_codex_snapshot(CODEX_SNAPSHOT)

    def test_publish_routes_to_installed_store(self, tmp_path):
        store = QuotaStore(str(tmp_path / "q.json"))
        qs.install(store)
        qs.publish_claude_rate_limit(CLAUDE_RAW)
        qs.publish_codex_snapshot(CODEX_SNAPSHOT, push=True)
        assert store.get("claude-code") is not None
        assert store.get("codex") is not None

    def test_publish_garbage_never_raises(self, tmp_path):
        store = QuotaStore(str(tmp_path / "q.json"))
        qs.install(store)
        qs.publish_claude_rate_limit(object())
        qs.publish_codex_snapshot(["x"], push=True)
        assert store.snapshots() == {}


# ---------------------------------------------------------------------------
# Transports: quota is captured, and never becomes a chat/transcript event
# ---------------------------------------------------------------------------

class TestSDKTransportCapture:
    def test_rate_limit_event_feeds_store_and_emits_no_event(self, tmp_path):
        from claude_agent_sdk.types import RateLimitEvent, RateLimitInfo
        from agent_os.agent.transports.sdk_transport import SDKTransport

        store = QuotaStore(str(tmp_path / "q.json"))
        qs.install(store)
        t = SDKTransport()
        msg = RateLimitEvent(
            rate_limit_info=RateLimitInfo(status="allowed", raw=CLAUDE_RAW),
            uuid="u1", session_id="s1")
        assert t._message_to_events(msg) == []
        assert _windows(store.get("claude-code"))["five_hour"]["used_pct"] == \
            pytest.approx(79.0)

    def test_malformed_rate_limit_event_never_breaks_the_turn(self, tmp_path):
        from claude_agent_sdk.types import RateLimitEvent, RateLimitInfo
        from agent_os.agent.transports.sdk_transport import SDKTransport

        qs.install(QuotaStore(str(tmp_path / "q.json")))
        t = SDKTransport()
        msg = RateLimitEvent(
            rate_limit_info=RateLimitInfo(status="allowed", raw="bogus"),  # type: ignore[arg-type]
            uuid="u1", session_id="s1")
        assert t._message_to_events(msg) == []


def _codex_transport():
    from agent_os.agent.transports.codex_transport import CodexTransport
    t = CodexTransport()
    t._thread_id = "T1"
    return t


def _drain(transport) -> list:
    out = []
    while not transport._event_queue.empty():
        out.append(transport._event_queue.get_nowait())
    return out


class TestCodexTransportCapture:
    @pytest.mark.asyncio
    async def test_rate_limits_updated_feeds_store_and_emits_nothing(self, tmp_path):
        store = QuotaStore(str(tmp_path / "q.json"))
        qs.install(store)
        t = _codex_transport()
        await t._route_server_message({
            "jsonrpc": "2.0", "method": "account/rateLimits/updated",
            "params": {"rateLimits": CODEX_SNAPSHOT}})
        assert _drain(t) == []
        assert _windows(store.get("codex"))["weekly"]["used_pct"] == 3

    @pytest.mark.asyncio
    async def test_read_rate_limits_uses_the_live_connection(self):
        t = _codex_transport()
        sent = []

        async def fake_request(method, params=None, timeout=30.0):
            sent.append((method, params))
            return CODEX_READ_RESULT

        t._request = fake_request
        t._alive = True
        assert await t.read_rate_limits() == CODEX_READ_RESULT
        assert sent == [("account/rateLimits/read", None)]

    @pytest.mark.asyncio
    async def test_read_rate_limits_on_dead_transport_returns_none(self):
        t = _codex_transport()
        t._alive = False
        assert await t.read_rate_limits() is None


FAKE_APP_SERVER = r'''
import json, sys
for line in sys.stdin:
    msg = json.loads(line)
    if msg.get("method") == "initialize":
        print(json.dumps({"jsonrpc": "2.0", "id": msg["id"],
                          "result": {"userAgent": "codex/0.144.5"}}), flush=True)
    elif msg.get("method") == "account/rateLimits/read":
        print(json.dumps({"jsonrpc": "2.0", "method": "thread/started",
                          "params": {}}), flush=True)
        print(json.dumps({"jsonrpc": "2.0", "id": msg["id"],
                          "result": RESULT}), flush=True)
'''


@pytest.mark.skipif(sys.platform == "win32", reason="posix shebang fake binary")
class TestShortLivedProbe:
    def _fake_codex(self, tmp_path, body: str) -> str:
        path = tmp_path / "codex"
        path.write_text(
            f"#!{sys.executable}\nimport sys\n"
            f"if sys.argv[1:] != ['app-server']: sys.exit(2)\n{body}",
            encoding="utf-8")
        path.chmod(0o755)
        return str(path)

    @pytest.mark.asyncio
    async def test_probe_speaks_handshake_and_returns_result(self, tmp_path):
        from agent_os.agent.transports.codex_transport import (
            fetch_codex_rate_limits,
        )
        # json.dumps of the JSON text is a valid Python string literal.
        body = ("import json\nRESULT = json.loads("
                f"{json.dumps(json.dumps(CODEX_READ_RESULT))})\n")
        binary = self._fake_codex(tmp_path, body + FAKE_APP_SERVER)
        result = await fetch_codex_rate_limits(binary, timeout=10.0)
        assert pick_codex_snapshot(result)["planType"] == "plus"

    @pytest.mark.asyncio
    async def test_probe_timeout_returns_none(self, tmp_path):
        from agent_os.agent.transports.codex_transport import (
            fetch_codex_rate_limits,
        )
        binary = self._fake_codex(tmp_path, "import time\ntime.sleep(30)\n")
        start = time.monotonic()
        assert await fetch_codex_rate_limits(binary, timeout=0.5) is None
        assert time.monotonic() - start < 5

    @pytest.mark.asyncio
    async def test_probe_missing_binary_returns_none(self, tmp_path):
        from agent_os.agent.transports.codex_transport import (
            fetch_codex_rate_limits,
        )
        assert await fetch_codex_rate_limits(str(tmp_path / "nope")) is None
