# Queue: Upload Wake + Timed Pause + Idle Start Button — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make upload-sourced queue items processable (wake an idle+empty queue), add timed pause ("pause until I resume / for 1 hour / until tomorrow") with auto-resume, and add the missing Start button for an idle queue with staged items.

**Architecture:** The queue run-state encodes explicit user intent. PAUSED gates everything (uploads stage silently). IDLE is wakeable by an upload **only when the queue holds no other queueable items** (staged user batches keep their explicit-start semantics). Timed pause persists an absolute `paused_until` ISO timestamp on `QueueState`; the dispatcher's existing 5-second `_run` tick checks expiry and calls `resume()`. No new background timers, no new processes.

**Tech Stack:** Python/FastAPI/pydantic (backend), React/TypeScript/Vite + Vitest (frontend), pytest + pytest-asyncio.

**Product decisions locked (conversation 2026-06-11):**
1. Uploads follow queue state; exception: an upload wakes an IDLE queue that has no other queueable items. PAUSED always gates.
2. No implicit auto-resume. Auto-resume happens only when the user chose a duration at pause time (DND-style snooze). Default pause = until resumed.
3. `QueueHeader` must show Start when idle with queued items (today it wrongly shows Stop and there is no way to start an idle queue from the UI).

---

## File map

| File | Change |
|---|---|
| `agent_os/queue/models.py` | `QueueState.paused_until: Optional[str]` (tail, additive) |
| `agent_os/queue/store.py` | `set_queue_state(..., paused_until=...)`; clear on leaving PAUSED and in `auto_idle_if_empty` |
| `agent_os/queue/dispatcher.py` | `stop(duration_seconds=None)`; `_run` PAUSED branch checks expiry → `resume()` |
| `agent_os/api/routes/agents_v2.py` | `/queue/stop` accepts optional `{duration_seconds}` body |
| `agent_os/api/routes/files_v2.py` | `_notify_upload` → async; wake idle+empty queue via `dispatcher.resume()` |
| `tests/unit/test_queue_store.py` | paused_until persistence/clearing tests |
| `tests/unit/test_timed_pause.py` | NEW — dispatcher stop-with-duration + auto-resume |
| `tests/unit/test_upload_wake.py` | NEW — `_notify_upload` wake matrix |
| `web/src/types.ts` | `QueueSnapshot.paused_until?: string \| null` |
| `web/src/hooks/useQueue.ts` | `stopQueue(durationSeconds?)` sends body |
| `web/src/components/QueueHeader.tsx` | Start button (idle+queued), pause menu (running), auto-resume hint (paused) |
| `web/src/components/QueueTab.tsx` | thread new `onStop` signature |
| `web/src/i18n/strings.ts` | 5 new keys |
| `web/src/components/QueueHeader.test.tsx` | NEW |
| `web/src/components/QueueTab.test.tsx` | update if header assertions break |

Tasks 1, 2, 3 touch disjoint files and may run in parallel. Sub-agents MUST NOT commit (coordinator commits) and MUST NOT touch files outside their scope.

---

### Task 1: Backend — timed pause (`paused_until`)

**Scope (allowed files):** `agent_os/queue/models.py`, `agent_os/queue/store.py`, `agent_os/queue/dispatcher.py`, `agent_os/api/routes/agents_v2.py`, `tests/unit/test_queue_store.py`, `tests/unit/test_timed_pause.py`.
**Forbidden:** `agent_os/api/routes/files_v2.py`, `web/**`, everything else.

- [ ] **Step 1.1: Write failing store tests** — append to `tests/unit/test_queue_store.py`:

```python
def test_paused_until_round_trip(tmp_path):
    store = _store(tmp_path)
    store.set_queue_state(QueueRunState.PAUSED, paused_until="2026-06-11T09:00:00+00:00")
    store2 = QueueStore(tmp_path / "queue.json")
    state = store2.load()
    assert state.state == QueueRunState.PAUSED
    assert state.paused_until == "2026-06-11T09:00:00+00:00"


def test_paused_until_cleared_on_leaving_paused(tmp_path):
    store = _store(tmp_path)
    store.set_queue_state(QueueRunState.PAUSED, paused_until="2026-06-11T09:00:00+00:00")
    store.set_queue_state(QueueRunState.RUNNING)
    assert store.load().paused_until is None


def test_pause_without_duration_has_no_deadline(tmp_path):
    store = _store(tmp_path)
    store.set_queue_state(QueueRunState.PAUSED)
    assert store.load().paused_until is None


def test_auto_idle_clears_paused_until(tmp_path):
    store = _store(tmp_path)
    store.set_queue_state(QueueRunState.PAUSED, paused_until="2026-06-11T09:00:00+00:00")
    changed = store.auto_idle_if_empty()
    assert changed is True
    assert store.load().paused_until is None
```

- [ ] **Step 1.2: Run them, verify they fail** — `PYTHON_KEYRING_BACKEND=in-memory AGENT_OS_API_KEY=test python -m pytest tests/unit/test_queue_store.py -q`. Expected: 4 failures (`set_queue_state() got an unexpected keyword argument` / missing attribute).

- [ ] **Step 1.3: Model + store implementation.**

`agent_os/queue/models.py` — add at the tail of `QueueState` (after `items`):

```python
    # When state == PAUSED and the user chose a snooze duration, the absolute
    # UTC ISO deadline after which the dispatcher auto-resumes. None means
    # "paused until explicitly resumed" (the default). Cleared on any
    # transition out of PAUSED.
    paused_until: Optional[str] = None
```

`agent_os/queue/store.py` — replace `set_queue_state` and patch `auto_idle_if_empty`:

```python
    def set_queue_state(
        self,
        new_state: QueueRunState,
        *,
        paused_until: Optional[str] = None,
    ) -> None:
        """Set the queue run state. ``paused_until`` only applies when
        entering PAUSED (timed pause); any other state clears it — the
        deadline is meaningless outside PAUSED."""
        state = self.load()
        with self._lock:
            state.state = new_state
            state.paused_until = (
                paused_until if new_state == QueueRunState.PAUSED else None
            )
            self._save_locked()
```

In `auto_idle_if_empty`, add `state.paused_until = None` on the line before `state.state = QueueRunState.IDLE`.

NOTE: `set_queue_state(PAUSED)` without the kwarg clears a prior deadline — that is intended (a fresh pause supersedes an old snooze).

- [ ] **Step 1.4: Run store tests, verify pass** — same command as 1.2. Expected: all pass.

- [ ] **Step 1.5: Write failing dispatcher tests** — create `tests/unit/test_timed_pause.py`:

```python
# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Timed pause: stop(duration_seconds=N) records paused_until; the dispatcher's
_run tick auto-resumes once the deadline passes. A pause without a duration
never auto-resumes (product decision 2026-06-11: pause is a consent boundary;
auto-resume only when the user chose a snooze)."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

import pytest

from agent_os.queue.dispatcher import QueueDispatcher
from agent_os.queue.models import ItemState, QueueRunState
from agent_os.queue.store import QueueStore
from tests.integration.test_queue_phase2 import _ScriptedAgentManager


async def _wait(predicate, timeout=10.0, interval=0.05):
    deadline = asyncio.get_event_loop().time() + timeout
    while asyncio.get_event_loop().time() < deadline:
        if predicate():
            return True
        await asyncio.sleep(interval)
    return False


@pytest.mark.asyncio
async def test_stop_with_duration_records_paused_until(tmp_path):
    store = QueueStore(tmp_path / "queue.json")
    mgr = _ScriptedAgentManager([])
    dispatcher = QueueDispatcher(
        project_id="proj_timed", store=store, agent_manager=mgr,
    )
    result = await dispatcher.stop(duration_seconds=3600)
    state = store.load()
    assert state.state == QueueRunState.PAUSED
    assert state.paused_until is not None
    deadline = datetime.fromisoformat(state.paused_until)
    delta = deadline - datetime.now(timezone.utc)
    assert timedelta(minutes=59) < delta < timedelta(minutes=61)
    assert result["paused_until"] == state.paused_until


@pytest.mark.asyncio
async def test_stop_without_duration_has_no_deadline(tmp_path):
    store = QueueStore(tmp_path / "queue.json")
    mgr = _ScriptedAgentManager([])
    dispatcher = QueueDispatcher(
        project_id="proj_timed", store=store, agent_manager=mgr,
    )
    result = await dispatcher.stop()
    assert store.load().paused_until is None
    assert result.get("paused_until") is None


@pytest.mark.asyncio
async def test_expired_timed_pause_auto_resumes_and_drains(tmp_path):
    """PAUSED with a deadline in the past + one queued item: the running
    dispatcher must flip to RUNNING on its next tick and drain the item."""
    store = QueueStore(tmp_path / "queue.json")
    store.add_item("snoozed work")
    past = (datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat()
    store.set_queue_state(QueueRunState.PAUSED, paused_until=past)

    mgr = _ScriptedAgentManager([
        {"reason": "complete", "summary": "done"},
    ])
    dispatcher = QueueDispatcher(
        project_id="proj_timed", store=store, agent_manager=mgr,
    )
    await dispatcher.start()
    dispatcher.notify_new_item()

    ok = await _wait(lambda: (
        store.load().items
        and store.load().items[0].state == ItemState.DONE
    ), timeout=10.0)
    s = store.load()
    await dispatcher.shutdown()
    assert ok, (
        f"timed pause did not auto-resume: queue={s.state.value}, "
        f"paused_until={s.paused_until}, items="
        + ", ".join(f"{it.id}={it.state.value}" for it in s.items)
    )
    assert s.paused_until is None


@pytest.mark.asyncio
async def test_untimed_pause_never_auto_resumes(tmp_path):
    """PAUSED without a deadline stays paused across many ticks."""
    store = QueueStore(tmp_path / "queue.json")
    store.add_item("parked work")
    store.set_queue_state(QueueRunState.PAUSED)

    mgr = _ScriptedAgentManager([])
    dispatcher = QueueDispatcher(
        project_id="proj_timed", store=store, agent_manager=mgr,
    )
    await dispatcher.start()
    dispatcher.notify_new_item()
    await asyncio.sleep(1.0)  # several wake/check cycles via notify + tick
    s = store.load()
    await dispatcher.shutdown()
    assert s.state == QueueRunState.PAUSED
    assert s.items[0].state == ItemState.QUEUED


@pytest.mark.asyncio
async def test_malformed_paused_until_is_cleared_not_looping(tmp_path):
    store = QueueStore(tmp_path / "queue.json")
    store.set_queue_state(QueueRunState.PAUSED, paused_until="not-a-date")

    mgr = _ScriptedAgentManager([])
    dispatcher = QueueDispatcher(
        project_id="proj_timed", store=store, agent_manager=mgr,
    )
    await dispatcher.start()
    ok = await _wait(lambda: store.load().paused_until is None, timeout=5.0)
    s = store.load()
    await dispatcher.shutdown()
    assert ok
    assert s.state == QueueRunState.PAUSED  # still paused, just deadline dropped
```

Check `_ScriptedAgentManager([])` constructs cleanly with an empty script before relying on it (read `tests/integration/test_queue_phase2.py`); if it needs a non-empty list, adapt (e.g. pass a one-entry script that is never consumed).

- [ ] **Step 1.6: Run, verify fail** — `PYTHON_KEYRING_BACKEND=in-memory AGENT_OS_API_KEY=test python -m pytest tests/unit/test_timed_pause.py -q`. Expected: failures (`stop() got an unexpected keyword argument 'duration_seconds'`, paused state persists, etc.).

- [ ] **Step 1.7: Dispatcher implementation** (`agent_os/queue/dispatcher.py`).

(a) `stop()` — new signature and pause write (only the first lines change; the sub-agent stop / loop terminate body is untouched):

```python
    async def stop(self, duration_seconds: Optional[float] = None) -> dict:
```

Replace `self._store.set_queue_state(QueueRunState.PAUSED)` with:

```python
        paused_until: Optional[str] = None
        if duration_seconds is not None and duration_seconds > 0:
            paused_until = (
                datetime.now(timezone.utc) + timedelta(seconds=duration_seconds)
            ).isoformat()
        # Set queue state first so the main loop sees PAUSED on its next tick.
        self._store.set_queue_state(
            QueueRunState.PAUSED, paused_until=paused_until,
        )
```

and the return becomes `return {"status": "paused", "paused_until": paused_until}`. Add `timedelta` to the existing `from datetime import datetime, timezone` import.

(b) `_run()` — split the PAUSED/IDLE branch so PAUSED checks the deadline:

```python
                    qstate = self._store.load()
                    if qstate.state == QueueRunState.PAUSED:
                        if self._timed_pause_expired(qstate):
                            logger.info(
                                "dispatcher(%s): timed pause expired; auto-resuming",
                                self._project_id,
                            )
                            await self.resume()
                        else:
                            await self._wait_idle()
                        continue
                    if qstate.state == QueueRunState.IDLE:
                        await self._wait_idle()
                        continue
```

(c) New helper next to `_wait_idle`:

```python
    def _timed_pause_expired(self, qstate) -> bool:
        """True when a snooze deadline exists and has passed. Malformed
        deadlines are cleared (stay paused, drop the timer) so a corrupt
        value can't trip a resume loop or spam the log every tick."""
        if qstate.paused_until is None:
            return False
        try:
            deadline = datetime.fromisoformat(qstate.paused_until)
        except ValueError:
            logger.warning(
                "dispatcher(%s): malformed paused_until %r; clearing",
                self._project_id, qstate.paused_until,
            )
            self._store.set_queue_state(QueueRunState.PAUSED, paused_until=None)
            return False
        return datetime.now(timezone.utc) >= deadline
```

Note: `resume()` already calls `set_queue_state(QueueRunState.RUNNING)` which now clears `paused_until` — no change needed there. `reclaim_on_startup` leaves PAUSED untouched, so a deadline that expired while the daemon was down fires on the first tick after restart (intended).

- [ ] **Step 1.8: Run, verify pass** — command from 1.6. Expected: all pass. Also re-run `tests/unit/test_queue_store.py` and the dispatcher regression suite: `PYTHON_KEYRING_BACKEND=in-memory AGENT_OS_API_KEY=test python -m pytest tests/unit/ tests/regression/ -q` — no new failures.

- [ ] **Step 1.9: Route** (`agent_os/api/routes/agents_v2.py`). Next to the other queue request models add:

```python
class QueueStopRequest(BaseModel):
    # Snooze duration in seconds; omitted/None = pause until explicitly
    # resumed. gt=0 → 422 on zero/negative.
    duration_seconds: Optional[float] = Field(default=None, gt=0)
```

(match the file's existing pydantic import style — it already imports `BaseModel`; add `Field` if missing). Change the endpoint:

```python
@router.post("/projects/{project_id}/queue/stop")
async def stop_queue(
    project_id: str, req: Optional[QueueStopRequest] = None,
) -> dict:
```

and the final line to:

```python
    return await dispatcher.stop(
        duration_seconds=req.duration_seconds if req else None,
    )
```

Keep the docstring, adding: `Body {"duration_seconds": N} = timed pause (auto-resumes after N seconds); no body = pause until resumed.`

- [ ] **Step 1.10: Full unit sweep** — `PYTHON_KEYRING_BACKEND=in-memory AGENT_OS_API_KEY=test python -m pytest tests/unit/ -q`. Expected: green. Report results; DO NOT commit.

---

### Task 2: Backend — upload wakes an idle+empty queue

**Scope (allowed files):** `agent_os/api/routes/files_v2.py`, `tests/unit/test_upload_wake.py`.
**Forbidden:** `agent_os/queue/**`, `agent_os/api/routes/agents_v2.py`, `web/**`.

- [ ] **Step 2.1: Write failing tests** — create `tests/unit/test_upload_wake.py`:

```python
# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Upload wake semantics (product decision 2026-06-11):

An upload-sourced queue item may wake an IDLE queue that holds no other
queueable item — the drag-drop itself is the user's intent. PAUSED always
gates (pause is a consent boundary). Staged user items keep explicit-start
semantics: idle-with-staged-items stages the upload alongside them.
"""

from __future__ import annotations

import asyncio

import pytest

from agent_os.api.routes import files_v2
from agent_os.queue.models import ItemState, QueueRunState
from agent_os.queue.store import QueueStore


class _FakeDispatcher:
    def __init__(self):
        self.resume_calls = 0
        self.notify_calls = 0

    async def resume(self):
        self.resume_calls += 1
        return {"status": "running"}

    def notify_new_item(self):
        self.notify_calls += 1


class _FakeAgentManager:
    def __init__(self, store, dispatcher, onboarded=True):
        self._store = store
        self._dispatcher = dispatcher
        self._onboarded = onboarded

    def get_queue_store(self, project_id, workspace=""):
        return self._store

    def get_dispatcher(self, project_id):
        return self._dispatcher

    def is_onboarding_complete(self, project_id):
        return self._onboarded


class _FakeProjectStore:
    def __init__(self, workspace):
        self._workspace = workspace

    def get_project(self, project_id):
        return {"project_id": project_id, "workspace": self._workspace}


def _wire(tmp_path, *, onboarded=True):
    store = QueueStore(tmp_path / "queue.json")
    dispatcher = _FakeDispatcher()
    mgr = _FakeAgentManager(store, dispatcher, onboarded=onboarded)
    files_v2.configure(_FakeProjectStore(str(tmp_path)), agent_manager=mgr)
    return store, dispatcher


def _notify(project_id="proj_x"):
    asyncio.get_event_loop().run_until_complete(
        files_v2._notify_upload(project_id, "uploads/f.txt", "f.txt", 10)
    )


@pytest.mark.asyncio
async def test_upload_wakes_idle_empty_queue(tmp_path):
    store, dispatcher = _wire(tmp_path)
    store.set_queue_state(QueueRunState.IDLE)
    await files_v2._notify_upload("proj_x", "uploads/f.txt", "f.txt", 10)
    assert dispatcher.resume_calls == 1
    items = store.load().items
    assert len(items) == 1 and items[0].state == ItemState.QUEUED


@pytest.mark.asyncio
async def test_upload_does_not_wake_paused_queue(tmp_path):
    store, dispatcher = _wire(tmp_path)
    store.set_queue_state(QueueRunState.PAUSED)
    await files_v2._notify_upload("proj_x", "uploads/f.txt", "f.txt", 10)
    assert dispatcher.resume_calls == 0
    assert dispatcher.notify_calls == 1  # staged + dispatcher poked, not started
    assert store.load().state == QueueRunState.PAUSED


@pytest.mark.asyncio
async def test_upload_does_not_wake_idle_queue_with_staged_items(tmp_path):
    store, dispatcher = _wire(tmp_path)
    store.add_item("user staged batch item")
    store.set_queue_state(QueueRunState.IDLE)
    await files_v2._notify_upload("proj_x", "uploads/f.txt", "f.txt", 10)
    assert dispatcher.resume_calls == 0
    assert dispatcher.notify_calls == 1
    assert store.load().state == QueueRunState.IDLE
    assert len(store.load().items) == 2


@pytest.mark.asyncio
async def test_upload_does_not_wake_before_onboarding(tmp_path):
    store, dispatcher = _wire(tmp_path, onboarded=False)
    store.set_queue_state(QueueRunState.IDLE)
    await files_v2._notify_upload("proj_x", "uploads/f.txt", "f.txt", 10)
    assert dispatcher.resume_calls == 0


@pytest.mark.asyncio
async def test_running_queue_just_notifies(tmp_path):
    store, dispatcher = _wire(tmp_path)
    # fresh store defaults to RUNNING
    await files_v2._notify_upload("proj_x", "uploads/f.txt", "f.txt", 10)
    assert dispatcher.resume_calls == 0
    assert dispatcher.notify_calls == 1


@pytest.mark.asyncio
async def test_dedup_reupload_of_done_item_does_not_wake(tmp_path):
    """Idempotency hit on an already-DONE item must not restart the queue."""
    store, dispatcher = _wire(tmp_path)
    item = store.add_item(
        "x", source="upload", idempotency_key="upload:uploads/f.txt:10",
    )
    store.set_item_state(item.id, ItemState.DONE)
    store.set_queue_state(QueueRunState.IDLE)
    await files_v2._notify_upload("proj_x", "uploads/f.txt", "f.txt", 10)
    assert dispatcher.resume_calls == 0
```

Remove the unused `_notify` helper if flagged by lint; tests call the coroutine directly.

- [ ] **Step 2.2: Run, verify fail** — `PYTHON_KEYRING_BACKEND=in-memory AGENT_OS_API_KEY=test python -m pytest tests/unit/test_upload_wake.py -q`. Expected: TypeError (`_notify_upload` is sync today) / wake assertions fail.

- [ ] **Step 2.3: Implement** in `agent_os/api/routes/files_v2.py`.

(a) Add imports near the top: `from agent_os.queue.models import ItemState, QueueRunState`.

(b) Make `_notify_upload` async and replace its dispatcher block. Full new function body (docstring kept, content shown in full so this task is self-contained):

```python
async def _notify_upload(project_id: str, rel_path: str, filename: str, size: int) -> None:
    """Enqueue an 'upload' queue item so the agent is notified.

    Idempotency key is ``upload:{rel_path}:{size}`` (no wall-clock component):
    re-uploading the same file — e.g. a network retry — must dedup to a single
    queue item rather than spamming the agent.

    Wake semantics (product decision 2026-06-11): the upload itself is user
    intent, so it may WAKE an IDLE queue that holds no other queueable item
    (via dispatcher.resume(), same as POST /queue/start). It must NOT wake a
    PAUSED queue (pause is a consent boundary) and must NOT start an idle
    queue holding staged user items (those keep explicit-start semantics) —
    in both cases the item stages and the dispatcher is merely poked.
    Onboarding-incomplete projects never wake (mirrors /queue/start's gate).
    """
    if _agent_manager is None:
        # Minimal/unit configuration without an agent manager — nothing to
        # notify. File is on disk; that's the contract.
        return
    try:
        project = _project_store.get_project(project_id) if _project_store else None
        workspace = (project or {}).get("workspace") if project else None
        if not workspace:
            return
        store = _agent_manager.get_queue_store(project_id, workspace=workspace)
        item = store.add_item(
            content=(
                f"User uploaded `{filename}` to `{rel_path}`. Read the file and "
                "determine if it's relevant to your current work. If it changes "
                "your understanding of the project, update CONTEXT.md."
            ),
            file_refs=[rel_path],
            source="upload",
            priority=0,
            review_before_advance=False,
            idempotency_key=f"upload:{rel_path}:{size}",
        )

        qstate = store.load()
        queueable = (ItemState.QUEUED, ItemState.RUNNING, ItemState.BLOCKED)
        others = [
            it for it in qstate.items
            if it.id != item.id and it.state in queueable
        ]
        should_wake = (
            qstate.state == QueueRunState.IDLE
            and not others
            and item.state == ItemState.QUEUED  # dedup hit on a DONE item must not wake
            and _agent_manager.is_onboarding_complete(project_id)
        )

        dispatcher = _agent_manager.get_dispatcher(project_id)
        if dispatcher is None and should_wake:
            await _agent_manager._ensure_dispatcher(project_id, workspace)
            dispatcher = _agent_manager.get_dispatcher(project_id)
        if dispatcher is not None:
            if should_wake:
                await dispatcher.resume()
            else:
                dispatcher.notify_new_item()
        if _ws_manager is not None:
            _ws_manager.broadcast(project_id, {
                "type": "queue.item_added",
                "project_id": project_id,
                "item_id": item.id,
            })
    except Exception:
        logger.warning(
            "upload queue notification failed for %s (%s)",
            project_id, rel_path, exc_info=True,
        )
```

(c) Update the call site in `upload_file` (it is an async route): `_notify_upload(project_id, rel_path, safe_name, len(data))` → `await _notify_upload(project_id, rel_path, safe_name, len(data))`. Verify with `grep -n "_notify_upload" agent_os/api/routes/files_v2.py` that there are no other callers; also `grep -rn "_notify_upload" tests/` and fix any existing test that calls it synchronously.

- [ ] **Step 2.4: Run, verify pass** — command from 2.2, then `PYTHON_KEYRING_BACKEND=in-memory AGENT_OS_API_KEY=test python -m pytest tests/unit/ -q`. Expected: green. Report results; DO NOT commit.

---

### Task 3: Frontend — Start button, pause menu, auto-resume hint

**Scope (allowed files):** `web/src/types.ts`, `web/src/hooks/useQueue.ts`, `web/src/components/QueueHeader.tsx`, `web/src/components/QueueTab.tsx`, `web/src/i18n/strings.ts`, `web/src/components/QueueHeader.test.tsx`, `web/src/components/QueueTab.test.tsx`.
**Forbidden:** `agent_os/**`, `tests/**`, `docs/i18n/ui-terms.zh-Hans.csv` (never hand-edit the CSV; `strings.ts` is the runtime source of truth).

Header behavior matrix to implement:

| Queue state | Right-side control |
|---|---|
| `running` | **Stop** button → opens 3-option pause menu |
| `paused` | **Resume** button; if `paused_until` set, hint "Auto-resumes {time}" |
| `idle`, queued > 0 | **Start** button (Play, success styling) → `onResume` |
| `idle`, queued = 0 | no button |

- [ ] **Step 3.1: i18n keys** — in `web/src/i18n/strings.ts`, insert alphabetically among the existing `queue.*` keys:

```ts
  "queue.header.autoResume": { en: "Auto-resumes {time}", zh: "{time} 自动恢复" },
  "queue.pause.menu.oneHour": { en: "Pause for 1 hour", zh: "暂停 1 小时" },
  "queue.pause.menu.untilResume": { en: "Pause until I resume", zh: "暂停，直到我恢复" },
  "queue.pause.menu.untilTomorrow": { en: "Pause until tomorrow 9:00", zh: "暂停至明天 9:00" },
  "queue.start": { en: "Start", zh: "开始" },
```

- [ ] **Step 3.2: types.ts** — add to `QueueSnapshot` (optional, so old daemons and existing test mocks stay valid):

```ts
  /** UTC ISO deadline for a timed pause; null/absent = paused until resumed. */
  paused_until?: string | null;
```

- [ ] **Step 3.3: useQueue.ts** — replace `stopQueue`:

```ts
  const stopQueue = useCallback(
    async (durationSeconds?: number) => {
      if (!projectId) return;
      await api(`/api/v2/projects/${projectId}/queue/stop`, {
        method: 'POST',
        ...(durationSeconds != null
          ? { body: JSON.stringify({ duration_seconds: durationSeconds }) }
          : {}),
      });
    },
    [projectId],
  );
```

Check `api()` in `web/src/config.ts` first: if it always sets a JSON content-type header, sending no body for the untimed case must still work against the backend's `Optional[QueueStopRequest]` (FastAPI accepts an absent body). If `api()` requires a body when method is POST, send `JSON.stringify({})` instead of omitting.

- [ ] **Step 3.4: Write failing QueueHeader tests** — create `web/src/components/QueueHeader.test.tsx`:

```tsx
// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

// @vitest-environment jsdom

import { describe, it, expect, vi, afterEach } from 'vitest';
import { render, screen, cleanup, fireEvent } from '@testing-library/react';
import QueueHeader from './QueueHeader';
import type { QueueSnapshot, QueueItem } from '../types';

afterEach(() => cleanup());

function snap(state: QueueSnapshot['state'], opts?: {
  queued?: number;
  paused_until?: string | null;
}): QueueSnapshot {
  const items: QueueItem[] = Array.from({ length: opts?.queued ?? 0 }, (_, i) => ({
    id: `item_${i}`,
    content: `task ${i}`,
    file_refs: [],
    priority: 0,
    review_before_advance: false,
    state: 'queued',
    source: 'user',
    attempts: [],
    idempotency_key: null,
    interrupted_count: 0,
    created_at: '2026-06-11T00:00:00+00:00',
  }));
  return {
    version: 1,
    state,
    items,
    chat_session_id: null,
    paused_until: opts?.paused_until ?? null,
  };
}

describe('QueueHeader', () => {
  it('idle with queued items shows Start wired to onResume', () => {
    const onResume = vi.fn();
    render(
      <QueueHeader snapshot={snap('idle', { queued: 2 })} onStop={vi.fn()} onResume={onResume} />,
    );
    const start = screen.getByTestId('queue-start-btn');
    fireEvent.click(start);
    expect(onResume).toHaveBeenCalledTimes(1);
    expect(screen.queryByTestId('queue-stop-btn')).toBeNull();
  });

  it('idle with no queued items shows no action button', () => {
    render(
      <QueueHeader snapshot={snap('idle')} onStop={vi.fn()} onResume={vi.fn()} />,
    );
    expect(screen.queryByTestId('queue-start-btn')).toBeNull();
    expect(screen.queryByTestId('queue-stop-btn')).toBeNull();
    expect(screen.queryByTestId('queue-resume-btn')).toBeNull();
  });

  it('running shows Stop which opens the pause menu', () => {
    const onStop = vi.fn();
    render(
      <QueueHeader snapshot={snap('running')} onStop={onStop} onResume={vi.fn()} />,
    );
    fireEvent.click(screen.getByTestId('queue-stop-btn'));
    expect(screen.getByTestId('queue-pause-menu')).toBeTruthy();
    fireEvent.click(screen.getByTestId('queue-pause-until-resume'));
    expect(onStop).toHaveBeenCalledWith(undefined);
  });

  it('pause menu 1-hour option passes 3600 seconds', () => {
    const onStop = vi.fn();
    render(
      <QueueHeader snapshot={snap('running')} onStop={onStop} onResume={vi.fn()} />,
    );
    fireEvent.click(screen.getByTestId('queue-stop-btn'));
    fireEvent.click(screen.getByTestId('queue-pause-1h'));
    expect(onStop).toHaveBeenCalledWith(3600);
  });

  it('pause menu until-tomorrow passes a positive duration', () => {
    const onStop = vi.fn();
    render(
      <QueueHeader snapshot={snap('running')} onStop={onStop} onResume={vi.fn()} />,
    );
    fireEvent.click(screen.getByTestId('queue-stop-btn'));
    fireEvent.click(screen.getByTestId('queue-pause-tomorrow'));
    expect(onStop).toHaveBeenCalledTimes(1);
    const arg = onStop.mock.calls[0][0] as number;
    expect(arg).toBeGreaterThan(0);
    expect(arg).toBeLessThanOrEqual(33 * 3600); // ≤ 33h covers any clock time
  });

  it('paused shows Resume; timed pause shows auto-resume hint', () => {
    const future = new Date(Date.now() + 30 * 60 * 1000).toISOString();
    render(
      <QueueHeader
        snapshot={snap('paused', { queued: 1, paused_until: future })}
        onStop={vi.fn()}
        onResume={vi.fn()}
      />,
    );
    expect(screen.getByTestId('queue-resume-btn')).toBeTruthy();
    expect(screen.getByTestId('queue-autoresume-hint')).toBeTruthy();
  });

  it('untimed pause shows no auto-resume hint', () => {
    render(
      <QueueHeader snapshot={snap('paused', { queued: 1 })} onStop={vi.fn()} onResume={vi.fn()} />,
    );
    expect(screen.queryByTestId('queue-autoresume-hint')).toBeNull();
  });
});
```

If `useT` needs a provider, check how `QueueTab.test.tsx` handles i18n (it renders components that call `useT()` — mirror whatever it does; if it renders bare, `useT` falls back fine).

- [ ] **Step 3.5: Run, verify fail** — `cd web && npx vitest run src/components/QueueHeader.test.tsx`. Expected: failures (no `queue-start-btn`, no menu).

- [ ] **Step 3.6: Implement QueueHeader.tsx.** Replace the single bottom button block with the state-matrix control. Full replacement component:

```tsx
// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

import { useState } from 'react';
import { Pause, Play } from 'lucide-react';
import type { QueueRunState, QueueSnapshot } from '../types';
import { useT } from '../i18n/useT';
import type { StringKey } from '../i18n/strings';

const QUEUE_STATE_LABEL_KEY: Record<QueueRunState, StringKey> = {
  running: 'queue.header.running',
  paused: 'queue.header.paused',
  idle: 'queue.header.idle',
};

/** Seconds until the next 9:00 AM local time ("until tomorrow" snooze). */
function secondsUntilNextMorning(): number {
  const now = new Date();
  const next = new Date(now);
  next.setHours(9, 0, 0, 0);
  if (next <= now) next.setDate(next.getDate() + 1);
  return Math.round((next.getTime() - now.getTime()) / 1000);
}

/** "14:30" for same-day deadlines, "Jun 12, 09:00" otherwise. */
function formatResumeTime(iso: string): string {
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return '';
  const sameDay = d.toDateString() === new Date().toDateString();
  return sameDay
    ? d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
    : d.toLocaleString([], { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' });
}

interface QueueHeaderProps {
  snapshot: QueueSnapshot | null;
  onStop: (durationSeconds?: number) => void | Promise<void>;
  onResume: () => void | Promise<void>;
  disabled?: boolean;
}

export default function QueueHeader({
  snapshot,
  onStop,
  onResume,
  disabled,
}: QueueHeaderProps) {
  const t = useT();
  const [menuOpen, setMenuOpen] = useState(false);
  if (!snapshot) {
    return null;
  }

  const counts = {
    running: snapshot.items.filter((it) => it.state === 'running').length,
    queued: snapshot.items.filter((it) => it.state === 'queued').length,
    blocked: snapshot.items.filter((it) => it.state === 'blocked').length,
    done: snapshot.items.filter((it) => it.state === 'done').length,
  };

  const isPaused = snapshot.state === 'paused';
  const isIdle = snapshot.state === 'idle';
  const resumeHint =
    isPaused && snapshot.paused_until ? formatResumeTime(snapshot.paused_until) : '';

  const pick = (duration?: number) => {
    setMenuOpen(false);
    void onStop(duration);
  };

  return (
    <div className="flex items-center justify-between gap-3 px-6 py-3 border-b border-border max-md:px-4 max-md:flex-wrap">
      <div className="flex items-center gap-4 text-xs text-secondary flex-wrap">
        <span
          className={`px-2 py-0.5 rounded-full text-[11px] font-medium ${
            isPaused
              ? 'bg-warning/15 text-warning'
              : snapshot.state === 'running'
                ? 'bg-accent/15 text-accent'
                : 'bg-secondary/15 text-secondary'
          }`}
          data-testid="queue-state-pill"
        >
          {QUEUE_STATE_LABEL_KEY[snapshot.state] ? t(QUEUE_STATE_LABEL_KEY[snapshot.state]) : snapshot.state}
        </span>
        <span data-testid="queue-count-running">{t('queue.count.running', { n: counts.running })}</span>
        <span data-testid="queue-count-queued">{t('queue.count.queued', { n: counts.queued })}</span>
        {counts.blocked > 0 && (
          <span className="text-warning" data-testid="queue-count-blocked">
            {t('queue.count.blocked', { n: counts.blocked })}
          </span>
        )}
        {counts.done > 0 && (
          <span data-testid="queue-count-done">{t('queue.count.done', { n: counts.done })}</span>
        )}
      </div>
      <div className="flex items-center gap-3">
        {resumeHint && (
          <span className="text-xs text-secondary" data-testid="queue-autoresume-hint">
            {t('queue.header.autoResume', { time: resumeHint })}
          </span>
        )}
        {isPaused ? (
          <button
            onClick={() => void onResume()}
            disabled={disabled}
            data-testid="queue-resume-btn"
            className="flex items-center gap-1.5 text-sm font-medium rounded-lg px-3 py-1.5 transition-all duration-150 disabled:opacity-40 disabled:cursor-not-allowed max-md:min-h-[44px] bg-success text-on-accent hover:bg-success/90"
          >
            <Play className="w-3.5 h-3.5" /> {t('queue.resume')}
          </button>
        ) : isIdle ? (
          counts.queued > 0 ? (
            <button
              onClick={() => void onResume()}
              disabled={disabled}
              data-testid="queue-start-btn"
              className="flex items-center gap-1.5 text-sm font-medium rounded-lg px-3 py-1.5 transition-all duration-150 disabled:opacity-40 disabled:cursor-not-allowed max-md:min-h-[44px] bg-success text-on-accent hover:bg-success/90"
            >
              <Play className="w-3.5 h-3.5" /> {t('queue.start')}
            </button>
          ) : null
        ) : (
          <div className="relative">
            <button
              onClick={() => setMenuOpen((v) => !v)}
              disabled={disabled}
              data-testid="queue-stop-btn"
              className="flex items-center gap-1.5 text-sm font-medium rounded-lg px-3 py-1.5 transition-all duration-150 disabled:opacity-40 disabled:cursor-not-allowed max-md:min-h-[44px] border border-border text-secondary hover:text-warning hover:border-warning/40"
            >
              <Pause className="w-3.5 h-3.5" /> {t('queue.stop')}
            </button>
            {menuOpen && (
              <>
                <div className="fixed inset-0 z-10" onClick={() => setMenuOpen(false)} />
                <div
                  className="absolute right-0 top-full mt-1 z-20 min-w-[200px] rounded-lg border border-border bg-surface shadow-lg py-1"
                  data-testid="queue-pause-menu"
                >
                  <button
                    onClick={() => pick(undefined)}
                    data-testid="queue-pause-until-resume"
                    className="w-full text-left text-sm px-3 py-2 hover:bg-secondary/10"
                  >
                    {t('queue.pause.menu.untilResume')}
                  </button>
                  <button
                    onClick={() => pick(3600)}
                    data-testid="queue-pause-1h"
                    className="w-full text-left text-sm px-3 py-2 hover:bg-secondary/10"
                  >
                    {t('queue.pause.menu.oneHour')}
                  </button>
                  <button
                    onClick={() => pick(secondsUntilNextMorning())}
                    data-testid="queue-pause-tomorrow"
                    className="w-full text-left text-sm px-3 py-2 hover:bg-secondary/10"
                  >
                    {t('queue.pause.menu.untilTomorrow')}
                  </button>
                </div>
              </>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
```

Before finalizing styling, check the menu surface class: look at `SessionThreeDotMenu.tsx` for the project's established dropdown idiom (z-index, bg token — e.g. `bg-surface` vs `bg-panel`) and match it.

- [ ] **Step 3.7: QueueTab.tsx** — no signature change needed if `stopQueue` already flows through as `onStop={stopQueue}` (the new optional param is compatible). Verify only; update `QueueTab.test.tsx` if its mocks/assertions reference the old header behavior (e.g. a stop button asserted present at idle).

- [ ] **Step 3.8: Run, verify pass** — `cd web && npx vitest run src/components/QueueHeader.test.tsx src/components/QueueTab.test.tsx`. Expected: green.

- [ ] **Step 3.9: Full frontend gate** — `cd web && npx tsc --noEmit && npm run test:run && node scripts/check-i18n.mjs`. Expected: zero tsc errors, all tests green, check-i18n no errors (warnings on missing zh acceptable — but all 5 new keys ship with zh). Report results; DO NOT commit.

---

### Task 4: Coordinator — verification, integration, commits

(Performed by the coordinator after merging sub-agent work; not delegated.)

- [ ] **Step 4.1:** Full backend sweep (excluding the two sandbox-exec suites that can revoke ~/Desktop access — see CLAUDE.md Known Issues):

```bash
PYTHON_KEYRING_BACKEND=in-memory AGENT_OS_API_KEY=test python -m pytest tests/unit/ tests/platform/ -q \
  --ignore=tests/platform/test_e2e_agent_isolation.py \
  --ignore=tests/platform/test_macos_provider_integration.py
```

Expected: green except documented pre-existing env-fails. Also run `tests/regression/ tests/integration/` queue-related modules.

- [ ] **Step 4.2:** Frontend gate (tsc, vitest, check-i18n) — as Task 3 step 3.9.

- [ ] **Step 4.3:** Daemon integration test on an isolated port (the packaged Orbital.app owns :8000 — do NOT kill it):
  - `PYTHON_KEYRING_BACKEND=in-memory AGENT_OS_API_KEY=test python -m uvicorn agent_os.api.app:create_app --factory --port 8321` (dev daemon uses isolated `./orbital-data`)
  - Create a temp project; touch its `PROJECT_STATE.md` (onboarding gate); verify timed pause: `POST /queue/stop {"duration_seconds": 5}` → snapshot shows `paused` + `paused_until`; ~10s later snapshot shows the queue left `paused`.
  - Verify upload wake: `POST /files/upload` (idle, empty queue) → queue state flips to `running` and the item dispatches; upload while `paused` → stays `paused`, item staged.

- [ ] **Step 4.4:** Commits (coordinator only), three logical commits:
  1. `feat(queue): timed pause — paused_until on QueueState, dispatcher auto-resume, stop route duration` (Task 1 files)
  2. `feat(queue): uploads wake an idle+empty queue; paused queues stage uploads` (Task 2 files)
  3. `feat(queue): idle Start button, pause-duration menu, auto-resume hint` (Task 3 files + this plan doc)

- [ ] **Step 4.5:** Mobile QR verification per CLAUDE.md (Vite with `--host`, QR code printed). Note: the live packaged daemon on :8000 predates `paused_until`/upload-wake; the Start button works against it (uses pre-existing `/queue/resume`) — useful to unstick the real qclaw queue.
