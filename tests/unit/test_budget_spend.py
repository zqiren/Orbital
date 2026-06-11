# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Unit tests for derived cost views (Budget Piece 1, Task 4).

Covers ``agent_os.budget.ledger.spend()`` + ``window_start()``:

  - window boundary math (daily / weekly / monthly / total) incl. the
    local-midnight edge, exercised with timezone-controlled timestamps
  - anchor reset: events before ``budget_anchor_ts`` excluded from ``total``
  - per-model × per-tier × per-source aggregation with cost math vs known rates
  - mixed-currency ``by_currency`` + ``converted_total`` (estimated flag,
    unknown-pair degradation)
  - malformed ledger lines skipped, never crash
  - query-time rates: editing the override between two spend() calls changes
    cost while token counts stay identical
"""

import json
import os
from datetime import datetime, timedelta, timezone

import pytest

import agent_os.agent.pricing as pricing_mod
from agent_os.budget.ledger import (
    LedgerEvent,
    append_event,
    ledger_path,
    spend,
    window_start,
    WINDOWS,
)
from agent_os.budget.normalize import NormalizedUsage


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _reset_pricing_caches():
    """Reset pricing caches around each test so override edits take effect."""
    pricing_mod._pricing_cache = None
    pricing_mod._full_providers_cache = None
    pricing_mod._override_cache = None
    pricing_mod._override_mtime = None
    yield
    pricing_mod._pricing_cache = None
    pricing_mod._full_providers_cache = None
    pricing_mod._override_cache = None
    pricing_mod._override_mtime = None


@pytest.fixture
def override_file(tmp_path):
    """A temp pricing-override file with known anthropic + moonshot rates."""
    path = str(tmp_path / "ov.json")
    _write_override(path, {
        "anthropic": {
            "_currency": "USD",
            "claude-x": {
                "input_per_1m": 3.0,
                "cached_input_per_1m": 0.3,
                "cache_write_per_1m": 3.75,
                "output_per_1m": 15.0,
            },
        },
        "moonshot": {
            "_currency": "CNY",
            "kimi-x": {"input_per_1m": 12.0, "output_per_1m": 12.0},
        },
    })
    return path


def _write_override(path: str, data: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f)


def _append_raw(project_dir: str, record: dict) -> None:
    """Write a raw ledger line bypassing append_event (controls ts/shape)."""
    path = ledger_path(project_dir)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def _event_record(ts: str, provider="anthropic", model="claude-x",
                  source="management", uncached=0, cache_read=0,
                  cache_write=0, output=0) -> dict:
    return {
        "ts": ts,
        "session_id": "s",
        "source": source,
        "provider": provider,
        "model": model,
        "uncached_input": uncached,
        "cache_read": cache_read,
        "cache_write": cache_write,
        "output": output,
    }


# ---------------------------------------------------------------------------
# window_start: boundary math
# ---------------------------------------------------------------------------

class TestWindowStart:
    def test_daily_is_local_midnight(self):
        # A fixed local time; daily start must be that day's 00:00 local.
        tz = timezone(timedelta(hours=8))  # e.g. Asia/Shanghai
        now = datetime(2026, 6, 11, 15, 30, 0, tzinfo=tz)
        start = window_start("daily", now=now)
        # 2026-06-11 00:00 +08:00 == 2026-06-10 16:00 UTC
        assert start == datetime(2026, 6, 10, 16, 0, tzinfo=timezone.utc)

    def test_weekly_is_most_recent_monday(self):
        tz = timezone(timedelta(hours=0))
        # 2026-06-11 is a Thursday → Monday is 2026-06-08.
        now = datetime(2026, 6, 11, 9, 0, 0, tzinfo=tz)
        start = window_start("weekly", now=now)
        assert start == datetime(2026, 6, 8, 0, 0, tzinfo=timezone.utc)
        assert start.weekday() == 0  # Monday

    def test_weekly_on_monday_returns_today(self):
        tz = timezone.utc
        now = datetime(2026, 6, 8, 10, 0, 0, tzinfo=tz)  # Monday
        start = window_start("weekly", now=now)
        assert start == datetime(2026, 6, 8, 0, 0, tzinfo=timezone.utc)

    def test_monthly_is_first_of_month(self):
        tz = timezone(timedelta(hours=-5))
        now = datetime(2026, 6, 11, 23, 0, 0, tzinfo=tz)
        start = window_start("monthly", now=now)
        # 2026-06-01 00:00 -05:00 == 2026-06-01 05:00 UTC
        assert start == datetime(2026, 6, 1, 5, 0, tzinfo=timezone.utc)

    def test_total_with_anchor(self):
        start = window_start("total", anchor_ts="2026-03-01T00:00:00+00:00")
        assert start == datetime(2026, 3, 1, 0, 0, tzinfo=timezone.utc)

    def test_total_without_anchor_is_epoch(self):
        start = window_start("total", anchor_ts=None)
        assert start == datetime(1970, 1, 1, tzinfo=timezone.utc)

    def test_total_with_malformed_anchor_is_epoch(self):
        start = window_start("total", anchor_ts="not-a-timestamp")
        assert start == datetime(1970, 1, 1, tzinfo=timezone.utc)

    def test_bad_window_raises(self):
        with pytest.raises(ValueError):
            window_start("yearly")


# ---------------------------------------------------------------------------
# spend: local-midnight boundary inclusion/exclusion
# ---------------------------------------------------------------------------

class TestLocalMidnightBoundary:
    def test_event_just_before_local_midnight_excluded(self, tmp_path, override_file):
        # Local tz UTC+8. "Now" = 2026-06-11 10:00 local → daily window start =
        # 2026-06-11 00:00 +08:00 = 2026-06-10 16:00 UTC.
        tz = timezone(timedelta(hours=8))
        now = datetime(2026, 6, 11, 10, 0, 0, tzinfo=tz)
        project = str(tmp_path)
        # Event at 2026-06-10 23:59 LOCAL (just before midnight) → 15:59 UTC,
        # which is BEFORE the 16:00 UTC window start → excluded.
        before = datetime(2026, 6, 10, 23, 59, 0, tzinfo=tz).astimezone(timezone.utc)
        _append_raw(project, _event_record(before.isoformat(), output=1_000_000))
        # Event at 2026-06-11 00:01 LOCAL → 16:01 UTC → after start → included.
        after = datetime(2026, 6, 11, 0, 1, 0, tzinfo=tz).astimezone(timezone.utc)
        _append_raw(project, _event_record(after.isoformat(), output=1_000_000))

        r = spend(project, "daily", now=now, override_path=override_file)
        # Only the "after" event (1M output × $15/1M = $15) counts.
        assert len(r["breakdown"]) == 1
        assert r["breakdown"][0]["tokens"]["output"] == 1_000_000
        assert r["by_currency"]["USD"] == pytest.approx(15.0)


# ---------------------------------------------------------------------------
# spend: anchor reset excludes pre-anchor events from total
# ---------------------------------------------------------------------------

class TestAnchorReset:
    def test_event_before_anchor_excluded_from_total(self, tmp_path, override_file):
        project = str(tmp_path)
        anchor = "2026-06-01T00:00:00+00:00"
        # Pre-anchor event — excluded.
        _append_raw(project, _event_record("2026-05-31T12:00:00+00:00", output=1_000_000))
        # Post-anchor event — included.
        _append_raw(project, _event_record("2026-06-02T12:00:00+00:00", output=2_000_000))

        r = spend(project, "total", anchor_ts=anchor, override_path=override_file)
        assert len(r["breakdown"]) == 1
        assert r["breakdown"][0]["tokens"]["output"] == 2_000_000

    def test_total_without_anchor_includes_everything(self, tmp_path, override_file):
        project = str(tmp_path)
        _append_raw(project, _event_record("2020-01-01T00:00:00+00:00", output=1_000_000))
        _append_raw(project, _event_record("2026-06-02T12:00:00+00:00", output=1_000_000))
        r = spend(project, "total", anchor_ts=None, override_path=override_file)
        # Both events aggregate into the single (anthropic, claude-x, management) group.
        assert len(r["breakdown"]) == 1
        assert r["breakdown"][0]["tokens"]["output"] == 2_000_000


# ---------------------------------------------------------------------------
# spend: ``sources`` filter (P3-A guard isolation)
#
# ``None`` (default) aggregates ALL sources — the GET /cost display view keeps
# returning everything. A non-None list filters by EXACT source match so the
# pre-flight guard / threshold detector can count management spend ONLY.
# Sub-agent ledger lines (``source="subagent:*"``) are valid on read (the schema
# allows any source string) and reserved for a later display piece.
# ---------------------------------------------------------------------------

class TestSourcesFilter:
    def _plant_mixed(self, project):
        """One management + two subagent rows, all in the total window."""
        _append_raw(project, _event_record(
            "2026-06-02T01:00:00+00:00", source="management", output=1_000_000))
        _append_raw(project, _event_record(
            "2026-06-02T02:00:00+00:00", source="subagent:claude-code",
            output=2_000_000))
        _append_raw(project, _event_record(
            "2026-06-02T03:00:00+00:00", source="subagent:codex",
            output=3_000_000))

    def test_none_aggregates_all_sources(self, tmp_path, override_file):
        project = str(tmp_path)
        self._plant_mixed(project)
        # Default (no sources kwarg) → every row counts. spend() groups by
        # source, so three distinct source values → three breakdown rows.
        r = spend(project, "total", override_path=override_file)
        sources = sorted(b["source"] for b in r["breakdown"])
        assert sources == ["management", "subagent:claude-code", "subagent:codex"]
        total_output = sum(b["tokens"]["output"] for b in r["breakdown"])
        assert total_output == 6_000_000
        # 6M output × $15/1M = $90.
        assert r["by_currency"]["USD"] == pytest.approx(90.0)

    def test_management_only_excludes_subagent_rows(self, tmp_path, override_file):
        project = str(tmp_path)
        self._plant_mixed(project)
        r = spend(project, "total", override_path=override_file,
                  sources=["management"])
        assert [b["source"] for b in r["breakdown"]] == ["management"]
        assert r["breakdown"][0]["tokens"]["output"] == 1_000_000
        # Only the 1M management row → $15. Subagent's 5M ($75) excluded.
        assert r["by_currency"]["USD"] == pytest.approx(15.0)

    def test_explicit_none_matches_default(self, tmp_path, override_file):
        project = str(tmp_path)
        self._plant_mixed(project)
        r_default = spend(project, "total", override_path=override_file)
        r_none = spend(project, "total", override_path=override_file, sources=None)
        assert r_default == r_none

    def test_multiple_sources_keeps_each_listed(self, tmp_path, override_file):
        project = str(tmp_path)
        self._plant_mixed(project)
        r = spend(project, "total", override_path=override_file,
                  sources=["management", "subagent:codex"])
        sources = sorted(b["source"] for b in r["breakdown"])
        assert sources == ["management", "subagent:codex"]
        total_output = sum(b["tokens"]["output"] for b in r["breakdown"])
        # 1M management + 3M codex = 4M; the 2M claude-code row excluded.
        assert total_output == 4_000_000
        assert r["by_currency"]["USD"] == pytest.approx(60.0)

    def test_unmatched_source_yields_empty(self, tmp_path, override_file):
        project = str(tmp_path)
        self._plant_mixed(project)
        r = spend(project, "total", override_path=override_file,
                  sources=["nonexistent"])
        assert r["breakdown"] == []
        assert r["by_currency"] == {}
        assert r["converted_total"]["amount"] == 0.0

    def test_empty_list_excludes_everything(self, tmp_path, override_file):
        project = str(tmp_path)
        self._plant_mixed(project)
        # Empty list is a list (not None) → matches nothing.
        r = spend(project, "total", override_path=override_file, sources=[])
        assert r["breakdown"] == []
        assert r["by_currency"] == {}


# ---------------------------------------------------------------------------
# spend: per-model × per-tier × per-source aggregation + cost math
# ---------------------------------------------------------------------------

class TestAggregationAndCost:
    def test_per_tier_cost_against_known_rates(self, tmp_path, override_file):
        project = str(tmp_path)
        # 1M of each tier for anthropic/claude-x (USD rates from override).
        _append_raw(project, _event_record(
            "2026-06-10T12:00:00+00:00",
            uncached=1_000_000, cache_read=1_000_000,
            cache_write=1_000_000, output=1_000_000,
        ))
        r = spend(project, "total", override_path=override_file)
        b = r["breakdown"][0]
        # input 3.0, cached 0.3, write 3.75, output 15.0
        assert b["cost"]["uncached_input"] == pytest.approx(3.0)
        assert b["cost"]["cache_read"] == pytest.approx(0.3)
        assert b["cost"]["cache_write"] == pytest.approx(3.75)
        assert b["cost"]["output"] == pytest.approx(15.0)
        assert b["cost"]["total"] == pytest.approx(22.05)
        assert r["by_currency"]["USD"] == pytest.approx(22.05)

    def test_splits_by_provider_model_source(self, tmp_path, override_file):
        project = str(tmp_path)
        ts = "2026-06-10T12:00:00+00:00"
        _append_raw(project, _event_record(ts, provider="anthropic",
                                           model="claude-x", source="management",
                                           output=1_000_000))
        _append_raw(project, _event_record(ts, provider="moonshot",
                                           model="kimi-x", source="management",
                                           output=1_000_000))
        # Same provider/model, different source → separate group.
        _append_raw(project, _event_record(ts, provider="anthropic",
                                           model="claude-x", source="subagent:foo",
                                           output=1_000_000))
        r = spend(project, "total", override_path=override_file)
        keys = {(b["provider"], b["model"], b["source"]) for b in r["breakdown"]}
        assert keys == {
            ("anthropic", "claude-x", "management"),
            ("moonshot", "kimi-x", "management"),
            ("anthropic", "claude-x", "subagent:foo"),
        }

    def test_same_group_lines_accumulate(self, tmp_path, override_file):
        project = str(tmp_path)
        ts = "2026-06-10T12:00:00+00:00"
        _append_raw(project, _event_record(ts, output=500_000))
        _append_raw(project, _event_record(ts, output=500_000))
        r = spend(project, "total", override_path=override_file)
        assert len(r["breakdown"]) == 1
        assert r["breakdown"][0]["tokens"]["output"] == 1_000_000
        assert r["breakdown"][0]["cost"]["output"] == pytest.approx(15.0)


# ---------------------------------------------------------------------------
# spend: mixed currency + converted_total
# ---------------------------------------------------------------------------

class TestMixedCurrency:
    def test_by_currency_and_converted_total(self, tmp_path, override_file):
        project = str(tmp_path)
        ts = "2026-06-10T12:00:00+00:00"
        # anthropic USD: 1M output → $15
        _append_raw(project, _event_record(ts, provider="anthropic",
                                           model="claude-x", output=1_000_000))
        # moonshot CNY: 2M input + 1M output @ 12/1M → 24 + 12 = 36 CNY
        _append_raw(project, _event_record(ts, provider="moonshot",
                                           model="kimi-x",
                                           uncached=2_000_000, output=1_000_000))
        r = spend(project, "total", target_currency="USD",
                  fx_rates={"CNY_per_USD": 7.2}, override_path=override_file)
        assert r["by_currency"]["USD"] == pytest.approx(15.0)
        assert r["by_currency"]["CNY"] == pytest.approx(36.0)
        # 36 CNY / 7.2 = 5 USD; + 15 USD = 20 USD
        assert r["converted_total"]["currency"] == "USD"
        assert r["converted_total"]["amount"] == pytest.approx(20.0)
        assert r["converted_total"]["estimated"] is True
        assert "unconverted" not in r["converted_total"]

    def test_unknown_pair_degrades_gracefully(self, tmp_path, override_file):
        project = str(tmp_path)
        ts = "2026-06-10T12:00:00+00:00"
        # moonshot CNY only; request conversion to USD but supply NO fx rate.
        _append_raw(project, _event_record(ts, provider="moonshot",
                                           model="kimi-x", output=1_000_000))
        r = spend(project, "total", target_currency="USD",
                  fx_rates={}, override_path=override_file)
        # CNY can't convert to USD → omitted from amount, flagged in unconverted.
        assert r["by_currency"]["CNY"] == pytest.approx(12.0)
        assert r["converted_total"]["amount"] == pytest.approx(0.0)
        assert r["converted_total"]["unconverted"] == ["CNY"]
        assert r["converted_total"]["estimated"] is True

    def test_target_cny_converts_usd_in(self, tmp_path, override_file):
        project = str(tmp_path)
        ts = "2026-06-10T12:00:00+00:00"
        _append_raw(project, _event_record(ts, provider="anthropic",
                                           model="claude-x", output=1_000_000))
        r = spend(project, "total", target_currency="CNY",
                  fx_rates={"CNY_per_USD": 7.2}, override_path=override_file)
        # 15 USD × 7.2 = 108 CNY
        assert r["converted_total"]["currency"] == "CNY"
        assert r["converted_total"]["amount"] == pytest.approx(108.0)


# ---------------------------------------------------------------------------
# spend: malformed lines skipped
# ---------------------------------------------------------------------------

class TestMalformedLines:
    def test_malformed_line_skipped(self, tmp_path, override_file):
        project = str(tmp_path)
        path = ledger_path(project)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write("this is not json\n")
            f.write(json.dumps({"ts": "bad-ts", "provider": "anthropic",
                                "model": "claude-x", "source": "management",
                                "output": 1}) + "\n")
            f.write(json.dumps({"provider": "anthropic"}) + "\n")  # missing ts
            f.write(json.dumps(_event_record("2026-06-10T12:00:00+00:00",
                                             output=1_000_000)) + "\n")
            f.write("\n")  # blank line
        r = spend(project, "total", override_path=override_file)
        # Only the single well-formed line survives.
        assert len(r["breakdown"]) == 1
        assert r["breakdown"][0]["tokens"]["output"] == 1_000_000

    def test_missing_ledger_file_returns_empty(self, tmp_path, override_file):
        r = spend(str(tmp_path), "total", override_path=override_file)
        assert r["breakdown"] == []
        assert r["by_currency"] == {}
        assert r["converted_total"]["amount"] == 0.0


# ---------------------------------------------------------------------------
# spend: query-time rates (no snapshot)
# ---------------------------------------------------------------------------

class TestQueryTimeRates:
    def test_editing_override_changes_cost_not_tokens(self, tmp_path):
        project = str(tmp_path)
        ov = str(tmp_path / "ov.json")
        _write_override(ov, {"anthropic": {"_currency": "USD",
                             "claude-x": {"input_per_1m": 3.0, "output_per_1m": 15.0}}})
        _append_raw(project, _event_record("2026-06-10T12:00:00+00:00",
                                           output=1_000_000))

        r1 = spend(project, "total", override_path=ov)
        assert r1["breakdown"][0]["cost"]["output"] == pytest.approx(15.0)
        tokens_before = r1["breakdown"][0]["tokens"]

        # Bump output rate; mtime must change so the override cache invalidates.
        import time
        time.sleep(0.01)
        _write_override(ov, {"anthropic": {"_currency": "USD",
                             "claude-x": {"input_per_1m": 3.0, "output_per_1m": 30.0}}})
        os.utime(ov, (os.stat(ov).st_atime, os.stat(ov).st_mtime + 5))

        r2 = spend(project, "total", override_path=ov)
        assert r2["breakdown"][0]["cost"]["output"] == pytest.approx(30.0)
        # Tokens identical — only the rate (and thus cost) changed.
        assert r2["breakdown"][0]["tokens"] == tokens_before


# ---------------------------------------------------------------------------
# spend: validation
# ---------------------------------------------------------------------------

class TestSpendValidation:
    def test_bad_window_raises(self, tmp_path):
        with pytest.raises(ValueError):
            spend(str(tmp_path), "yearly")

    def test_windows_constant_shape(self):
        assert WINDOWS == ("daily", "weekly", "monthly", "total")
