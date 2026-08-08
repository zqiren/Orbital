"""Telemetry runtime singleton + seam behavior (spec 046 §4, §10 step 4)."""

import json

import pytest

from agent_os import telemetry


@pytest.fixture(autouse=True)
def _clean_runtime():
    telemetry.reset_for_tests()
    yield
    telemetry.reset_for_tests()


def spool_rows(tmp_path):
    path = tmp_path / "telemetry" / "events.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_unconfigured_emit_and_latch_are_noops(tmp_path):
    telemetry.emit("app_start", {"version": "x"})
    telemetry.latch("key_set")  # must not raise
    assert spool_rows(tmp_path) == []


def test_emit_spools_when_enabled(tmp_path):
    telemetry.configure(tmp_path, is_enabled=lambda: True)
    telemetry.emit("project_created")
    telemetry.latch("first_project")
    rows = spool_rows(tmp_path)
    assert [r["event"] for r in rows] == ["project_created"]
    ident = json.loads((tmp_path / "telemetry" / "install.json").read_text())
    assert ident["milestones"] == {"first_project": True}


def test_toggle_off_blocks_counters_but_not_llm_error(tmp_path):
    telemetry.configure(tmp_path, is_enabled=lambda: False)
    telemetry.emit("project_created")
    telemetry.latch("first_project")
    telemetry.emit("llm_error", {"error_code": "invalid_api_key"}, always_spool=True)
    rows = spool_rows(tmp_path)
    # Q2 decision: error persistence is local-only debugging value, kept on.
    assert [r["event"] for r in rows] == ["llm_error"]
    ident = json.loads((tmp_path / "telemetry" / "install.json").read_text())
    assert ident["milestones"] == {}


def test_fresh_install_detection(tmp_path):
    telemetry.configure(tmp_path, is_enabled=lambda: True)
    assert telemetry.was_fresh_install() is True
    telemetry.reset_for_tests()
    telemetry.configure(tmp_path, is_enabled=lambda: True)
    assert telemetry.was_fresh_install() is False


def test_ledger_append_emits_turn_completed(tmp_path):
    from agent_os.budget.ledger import (
        SOURCE_MANAGEMENT,
        LedgerEvent,
        append_event,
    )
    from agent_os.budget.normalize import NormalizedUsage

    telemetry.configure(tmp_path, is_enabled=lambda: True)
    project_dir = tmp_path / "proj"
    project_dir.mkdir()
    append_event(
        str(project_dir),
        LedgerEvent(
            session_id="s1",
            source=SOURCE_MANAGEMENT,
            provider="deepseek",
            model="deepseek-v4-flash",
            usage=NormalizedUsage(
                uncached_input=10, cache_read=90, cache_write=0, output=5
            ),
        ),
    )
    rows = spool_rows(tmp_path)
    assert len(rows) == 1
    row = rows[0]
    assert row["event"] == "turn_completed"
    assert row["provider"] == "deepseek"
    assert row["source"] == SOURCE_MANAGEMENT
    assert (row["uncached_input"], row["cache_read"], row["output"]) == (10, 90, 5)
    # No ids leak into the spool row.
    assert "session_id" not in row and "model" not in row
