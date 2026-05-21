# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Tests for batched budget writes in ``ProjectStore``.

Track C3 (perf): ``update_runtime`` is on the per-LLM-call hot path. Writing
``projects.json`` synchronously every turn floods the disk with redundant
writes (the entire file is rewritten each time, not just the budget field).

The batching contract this module pins down:

1. **Reads after a write return the new value** even when the value has not
   yet been flushed to disk. (Regression: protect against the obvious bug
   where in-memory dirty state is invisible to ``get_project``.)

2. **Background flushes drain dirty state on a fixed interval.** Used here:
   10 seconds (default). The flush is an async task started at daemon
   startup and cancelled (after a final drain) at shutdown.

3. **Threshold-triggered flush.** Once accumulated unflushed spend for a
   single project crosses a configured dollar threshold, ``update_runtime``
   flushes synchronously *for that project* before returning. This bounds
   the data-loss window in dollar terms in addition to time terms.

4. **Graceful shutdown drains.** ``stop_flush_task`` awaits a final flush
   so no in-memory updates are lost on clean exit.

5. **Crash window is bounded by the flush interval.** Simulated by
   constructing a fresh ``ProjectStore`` against the same data dir without
   calling ``flush_runtime``: on-disk state reflects whatever was last
   flushed, never partial garbage.

Telemetry counters (``_runtime_updates_since_flush``, ``_skipped_flushes``)
are also asserted so the operator can see batching health at a glance.
"""
from __future__ import annotations

import asyncio
import json
import os

import pytest

from agent_os.daemon_v2.project_store import ProjectStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _read_disk(data_dir: str) -> dict:
    path = os.path.join(data_dir, "projects.json")
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _new_store(tmp_path, **kwargs) -> ProjectStore:
    data_dir = str(tmp_path / "data")
    os.makedirs(data_dir, exist_ok=True)
    return ProjectStore(data_dir=data_dir, **kwargs)


def _make_project(store: ProjectStore, name: str = "P") -> str:
    return store.create_project({
        "name": name, "workspace": "/tmp", "model": "m", "api_key": "k",
    })


# ---------------------------------------------------------------------------
# Read-after-write: in-memory wins over stale disk
# ---------------------------------------------------------------------------


class TestReadAfterWrite:
    def test_get_project_reflects_pending_update(self, tmp_path):
        """After update_runtime, get_project must return the new value
        even though it has not yet been flushed to disk."""
        store = _new_store(tmp_path, flush_interval_seconds=999, budget_flush_threshold_usd=999)
        pid = _make_project(store)
        store.update_runtime(pid, {"budget_spent_usd": 1.23})

        project = store.get_project(pid)
        assert project["runtime"]["budget_spent_usd"] == 1.23

    def test_disk_still_stale_until_flush(self, tmp_path):
        """The whole point of batching: disk should NOT see the new
        value until flush_runtime is called (or threshold hit)."""
        store = _new_store(tmp_path, flush_interval_seconds=999, budget_flush_threshold_usd=999)
        pid = _make_project(store)
        store.update_runtime(pid, {"budget_spent_usd": 4.56})

        disk = _read_disk(str(tmp_path / "data"))
        # Project exists on disk (it was saved on create), but the runtime
        # update has not yet been persisted.
        assert disk[pid].get("runtime", {}).get("budget_spent_usd", 0.0) == 0.0

    def test_flush_propagates_to_disk(self, tmp_path):
        """flush_runtime writes the in-memory pending state to disk."""
        store = _new_store(tmp_path, flush_interval_seconds=999, budget_flush_threshold_usd=999)
        pid = _make_project(store)
        store.update_runtime(pid, {"budget_spent_usd": 7.89})
        store.flush_runtime()

        disk = _read_disk(str(tmp_path / "data"))
        assert disk[pid]["runtime"]["budget_spent_usd"] == 7.89

    def test_unknown_project_is_silent(self, tmp_path):
        """update_runtime for an unknown project does not crash, does not
        create a phantom in-memory entry, does not pollute flush state."""
        store = _new_store(tmp_path)
        store.update_runtime("proj_nonexistent", {"budget_spent_usd": 1.0})
        store.flush_runtime()  # should be a no-op
        assert "proj_nonexistent" not in store._projects


# ---------------------------------------------------------------------------
# Threshold-triggered flush: bound dollar-window of loss
# ---------------------------------------------------------------------------


class TestThresholdFlush:
    def test_below_threshold_does_not_flush(self, tmp_path):
        """Tiny accumulation stays buffered."""
        store = _new_store(
            tmp_path, flush_interval_seconds=999, budget_flush_threshold_usd=1.00,
        )
        pid = _make_project(store)
        store.update_runtime(pid, {"budget_spent_usd": 0.10})
        store.update_runtime(pid, {"budget_spent_usd": 0.20})
        # Still under $1 threshold from baseline 0
        disk = _read_disk(str(tmp_path / "data"))
        assert disk[pid].get("runtime", {}).get("budget_spent_usd", 0.0) == 0.0

    def test_crossing_threshold_flushes(self, tmp_path):
        """When unflushed delta crosses the threshold, ``update_runtime``
        flushes synchronously (the spec calls this a 'configured threshold'
        crossing)."""
        store = _new_store(
            tmp_path, flush_interval_seconds=999, budget_flush_threshold_usd=0.50,
        )
        pid = _make_project(store)
        store.update_runtime(pid, {"budget_spent_usd": 0.30})
        # Delta from disk-baseline (0.0) is 0.30 → below 0.50, no flush
        disk1 = _read_disk(str(tmp_path / "data"))
        assert disk1[pid].get("runtime", {}).get("budget_spent_usd", 0.0) == 0.0

        store.update_runtime(pid, {"budget_spent_usd": 0.60})
        # Delta from disk-baseline (0.0) is now 0.60 → ≥ 0.50, flush triggered
        disk2 = _read_disk(str(tmp_path / "data"))
        assert disk2[pid]["runtime"]["budget_spent_usd"] == 0.60

    def test_threshold_resets_after_flush(self, tmp_path):
        """After a threshold-driven flush, the baseline moves forward; the
        NEXT threshold check is against the new baseline."""
        store = _new_store(
            tmp_path, flush_interval_seconds=999, budget_flush_threshold_usd=0.50,
        )
        pid = _make_project(store)
        store.update_runtime(pid, {"budget_spent_usd": 0.60})  # flushes
        # New baseline is 0.60. Add 0.20 → unflushed delta 0.20, should NOT flush.
        store.update_runtime(pid, {"budget_spent_usd": 0.80})
        disk = _read_disk(str(tmp_path / "data"))
        assert disk[pid]["runtime"]["budget_spent_usd"] == 0.60  # still pre-update value


# ---------------------------------------------------------------------------
# Telemetry counters
# ---------------------------------------------------------------------------


class TestTelemetry:
    def test_updates_counter_increments(self, tmp_path):
        store = _new_store(tmp_path, flush_interval_seconds=999, budget_flush_threshold_usd=999)
        pid = _make_project(store)
        store.update_runtime(pid, {"budget_spent_usd": 0.10})
        store.update_runtime(pid, {"budget_spent_usd": 0.20})
        store.update_runtime(pid, {"budget_spent_usd": 0.30})
        assert store._runtime_updates_since_flush[pid] == 3

    def test_updates_counter_resets_on_flush(self, tmp_path):
        store = _new_store(tmp_path, flush_interval_seconds=999, budget_flush_threshold_usd=999)
        pid = _make_project(store)
        store.update_runtime(pid, {"budget_spent_usd": 0.10})
        store.update_runtime(pid, {"budget_spent_usd": 0.20})
        store.flush_runtime()
        assert store._runtime_updates_since_flush.get(pid, 0) == 0

    def test_skipped_flushes_counter_increments_on_empty_flush(self, tmp_path):
        """If the periodic timer fires with no dirty state, that's a
        'skipped flush' — useful signal for tuning the interval."""
        store = _new_store(tmp_path, flush_interval_seconds=999, budget_flush_threshold_usd=999)
        before = store._skipped_flushes
        store.flush_runtime()  # nothing dirty
        store.flush_runtime()
        assert store._skipped_flushes == before + 2

    def test_skipped_flushes_does_not_count_real_flush(self, tmp_path):
        store = _new_store(tmp_path, flush_interval_seconds=999, budget_flush_threshold_usd=999)
        pid = _make_project(store)
        store.update_runtime(pid, {"budget_spent_usd": 0.10})
        before = store._skipped_flushes
        store.flush_runtime()
        assert store._skipped_flushes == before  # didn't skip; it did real work


# ---------------------------------------------------------------------------
# Periodic background flush task
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestBackgroundFlush:
    async def test_periodic_task_drains_dirty_state(self, tmp_path):
        """start_flush_task spawns an asyncio task that flushes on every
        tick of the interval. We use a very small interval to keep the
        test fast."""
        store = _new_store(
            tmp_path,
            flush_interval_seconds=0.05,
            budget_flush_threshold_usd=999,
        )
        pid = _make_project(store)
        store.update_runtime(pid, {"budget_spent_usd": 9.99})

        await store.start_flush_task()
        try:
            # Wait for at least one tick
            for _ in range(50):
                disk = _read_disk(str(tmp_path / "data"))
                if disk[pid].get("runtime", {}).get("budget_spent_usd", 0.0) == 9.99:
                    break
                await asyncio.sleep(0.02)
            else:
                pytest.fail("background flush never persisted update")
        finally:
            await store.stop_flush_task()

    async def test_stop_flush_task_drains_final_state(self, tmp_path):
        """Graceful-shutdown contract: stop_flush_task does a final flush
        so no in-memory updates are lost."""
        store = _new_store(
            tmp_path,
            flush_interval_seconds=999,  # background task won't fire during test
            budget_flush_threshold_usd=999,
        )
        pid = _make_project(store)
        await store.start_flush_task()
        store.update_runtime(pid, {"budget_spent_usd": 12.34})

        # Disk is still stale right now
        disk1 = _read_disk(str(tmp_path / "data"))
        assert disk1[pid].get("runtime", {}).get("budget_spent_usd", 0.0) == 0.0

        # Graceful stop should drain
        await store.stop_flush_task()
        disk2 = _read_disk(str(tmp_path / "data"))
        assert disk2[pid]["runtime"]["budget_spent_usd"] == 12.34

    async def test_stop_flush_task_is_idempotent(self, tmp_path):
        """Calling stop twice is safe (mirrors how shutdown handlers may
        run concurrently with errors)."""
        store = _new_store(tmp_path, flush_interval_seconds=0.05)
        await store.start_flush_task()
        await store.stop_flush_task()
        await store.stop_flush_task()  # no exception


# ---------------------------------------------------------------------------
# Crash simulation: bounded loss window
# ---------------------------------------------------------------------------


class TestCrashLossBound:
    def test_crash_between_flushes_loses_only_unflushed_writes(self, tmp_path):
        """Simulate a hard daemon crash by abandoning the store instance
        without calling flush. A fresh ProjectStore against the same data
        dir should see only what was last flushed — never partial data,
        never garbage, never a crash on load.
        """
        data_dir = str(tmp_path / "data")
        os.makedirs(data_dir, exist_ok=True)

        # First "daemon lifetime" — writes 1.00 (flushed), then 1.50 (unflushed)
        store1 = ProjectStore(
            data_dir=data_dir,
            flush_interval_seconds=999,
            budget_flush_threshold_usd=999,
        )
        pid = store1.create_project({
            "name": "P", "workspace": "/tmp", "model": "m", "api_key": "k",
        })
        store1.update_runtime(pid, {"budget_spent_usd": 1.00})
        store1.flush_runtime()  # committed
        store1.update_runtime(pid, {"budget_spent_usd": 1.50})  # NOT flushed
        # Hard crash: drop the reference. The unflushed write is lost.
        del store1

        # Second "daemon lifetime" — fresh instance reads only the flushed value
        store2 = ProjectStore(data_dir=data_dir)
        project = store2.get_project(pid)
        # The unflushed update is gone — but the flushed value survives
        # and the file is well-formed JSON (no crash on load).
        assert project["runtime"]["budget_spent_usd"] == 1.00

    def test_loss_bounded_in_time_via_interval(self, tmp_path):
        """If the background flush task is running with interval N, then
        a crash loses at most N seconds of accumulated writes. We assert
        this contract by checking that after one interval, the most
        recent write IS on disk."""
        async def _run():
            data_dir = str(tmp_path / "data")
            os.makedirs(data_dir, exist_ok=True)
            store = ProjectStore(
                data_dir=data_dir,
                flush_interval_seconds=0.05,
                budget_flush_threshold_usd=999,
            )
            pid = store.create_project({
                "name": "P", "workspace": "/tmp", "model": "m", "api_key": "k",
            })
            await store.start_flush_task()
            store.update_runtime(pid, {"budget_spent_usd": 5.55})
            # Wait > one interval; flush should have happened
            await asyncio.sleep(0.20)
            try:
                disk = _read_disk(data_dir)
                # "Crash" right after the interval — last write must be persisted
                assert disk[pid]["runtime"]["budget_spent_usd"] == 5.55
            finally:
                await store.stop_flush_task()

        asyncio.run(_run())


# ---------------------------------------------------------------------------
# Existing semantics preserved
# ---------------------------------------------------------------------------


class TestSemanticsUnchanged:
    """These mirror tests/regression/test_budget_persistence.py to confirm
    the batching change does NOT alter the user-visible contract."""

    def test_overwrite_not_accumulate(self, tmp_path):
        """update_runtime overwrites, does not accumulate."""
        store = _new_store(tmp_path)
        pid = _make_project(store)
        store.update_runtime(pid, {"budget_spent_usd": 1.0})
        store.update_runtime(pid, {"budget_spent_usd": 2.5})
        store.flush_runtime()
        project = store.get_project(pid)
        assert project["runtime"]["budget_spent_usd"] == 2.5

    def test_other_runtime_fields_still_work(self, tmp_path):
        """update_runtime supports arbitrary keys — batching is keyed by
        budget but other fields must still flow through."""
        store = _new_store(
            tmp_path, flush_interval_seconds=999, budget_flush_threshold_usd=999,
        )
        pid = _make_project(store)
        store.update_runtime(pid, {"some_other_field": "hello"})
        # Non-budget field should be visible via read-after-write
        assert store.get_project(pid)["runtime"]["some_other_field"] == "hello"
        store.flush_runtime()
        disk = _read_disk(str(tmp_path / "data"))
        assert disk[pid]["runtime"]["some_other_field"] == "hello"
