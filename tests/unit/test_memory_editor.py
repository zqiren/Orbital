# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""The memory pass shell (spec 124 §3.7) — what stays deterministic around
the Memory session's turn.

What it must guarantee, because it runs on real users' memory
automatically: the four files are backed up first (rolling, last 5); the
pass starts from normalised, stamped files; an id that left a live file
without reaching the archive is restored from the pass snapshot; INDEX
keeps its archive lines; the watermark advances only when the turn ran;
one pass per project at a time. The turn itself (``run_turn``) is the agent
manager's; here it is a fake that edits the files the way the agent would,
through ``WorkspaceFileManager`` (the same chokepoint the write tool uses).
"""

import asyncio
import json
import os
from datetime import datetime, timedelta, timezone

import pytest

from agent_os.agent import memory_editor as E
from agent_os.agent import state_blocks
from agent_os.agent import workspace_files as WF
from agent_os.agent.workspace_files import WorkspaceFileManager, run_session_end_routine

TODAY = "2026-09-19"


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------

def _bullet(text, eid, created="2026-09-01"):
    return f"- {text}\n  <!--mem id:{eid} created:{created} touched:{created}-->"


def _decision(title, eid, body="**Chose:** A.\n**Reason:** B."):
    detail = "".join("  " + ln + "\n" for ln in body.split("\n"))
    return f"- {title}\n  <!--mem id:{eid} created:2026-08-01 touched:2026-08-01-->\n{detail}\n"


STATE = "## Now\n" + "\n".join([
    _bullet("GOAI booth prep — posters printed, flyers ordered", "goai01", "2026-09-14"),
    _bullet("Newsjack radar 9/18: no new target", "news18", "2026-09-18"),
    _bullet("Newsjack radar 9/17: no new target", "news17", "2026-09-17"),
    _bullet("Cold-DM automation paused since 08-15", "cdm001", "2026-08-10"),
]) + "\n"

DECISIONS = (
    "# DECISIONS\n\n"
    + _decision("TabTin → watchlist", "tabtin")
    + _decision("PenguinHarness → watchlist", "penguin")
    + _decision("Pinned rule", "pinrule").replace("touched:2026-08-01", "touched:2026-08-01 tag:pinned")
)

LESSONS = "# LESSONS\n\n- Curl beats the browser for GitHub.\n  <!--mem id:curl created:2026-08-01 touched:2026-08-01-->\n\n"
INDEX = "# INDEX\n\n- agent_output/ — reports\n- DECISIONS_ARCHIVE.md — older decisions (read on demand)\n"


@pytest.fixture
def wf(tmp_path):
    w = WorkspaceFileManager(str(tmp_path))
    w.write("state", STATE)
    w.write("decisions", DECISIONS)
    w.write("lessons", LESSONS)
    w.write("index", INDEX)
    return w


@pytest.fixture(autouse=True)
def _clear_single_flight():
    E._RUNNING.clear()
    WF._completed_session_ends.clear()
    yield
    E._RUNNING.clear()


def _turn(**fields) -> E.TurnResult:
    base = dict(mode="disk", llm_calls=1, tool_calls=0, input_tokens=10,
                cached_tokens=0, output_tokens=1, ended="text_complete")
    base.update(fields)
    return E.TurnResult(**base)


def _archive_block(wf, key, block, heading=None):
    """What the Memory agent does to archive one entry: archive first (the
    daemon appends the section and regenerates the contents table), then
    the live file without it. ``heading`` defaults to ``## [archived DATE]``;
    an entry replaced by a live one goes under ``## [replaced DATE by id:X]``
    (spec 182)."""
    from agent_os.agent import memory_entries as M
    WF._append_archive_section(wf, M.ARCHIVE_OF[key], heading or f"## [archived {TODAY}]", block)
    wf.write(key, wf.read(key).replace(block + "\n", ""))


def _live_lines(wf):
    return {k: (wf.read(k) or "") for k in E.LIVE_KEYS}


# ---------------------------------------------------------------------------
# backup + watermark + ids
# ---------------------------------------------------------------------------

def test_backup_copies_the_four_files_and_keeps_the_last_five(wf):
    base = datetime(2026, 9, 19, tzinfo=timezone.utc)
    dirs = [E.backup_live_files(wf.dir, now=base + timedelta(seconds=i)) for i in range(7)]
    root = os.path.join(wf.dir, "backups")
    kept = sorted(d for d in os.listdir(root) if d.startswith("memory-editor-"))
    assert len(kept) == 5
    assert os.path.basename(dirs[-1]) in kept and os.path.basename(dirs[0]) not in kept
    assert sorted(os.listdir(dirs[-1])) == ["DECISIONS.md", "INDEX.md", "LESSONS.md", "PROJECT_STATE.md"]
    with open(os.path.join(dirs[-1], "PROJECT_STATE.md"), encoding="utf-8") as f:
        assert f.read() == wf.read("state")


def test_backup_never_prunes_other_backups(wf):
    other = os.path.join(wf.dir, "backups", "asks-migration-20260919")
    os.makedirs(other)
    for i in range(7):
        E.backup_live_files(wf.dir, now=datetime(2026, 9, 19, tzinfo=timezone.utc) + timedelta(seconds=i))
    assert os.path.isdir(other)


def test_watermark_falls_back_to_the_old_marker_mtime(wf):
    with open(os.path.join(wf.dir, ".memory_cleanup.json"), "w") as f:
        json.dump({"state": 1, "decisions": 2, "lessons": 3, "index": 4}, f)
    mark = E.watermark(wf.dir)
    assert mark is not None and abs((datetime.now(timezone.utc) - mark).total_seconds()) < 60


def test_write_marker_records_watermark_and_preserves_it(wf):
    ran = datetime(2026, 9, 19, 8, 0, tzinfo=timezone.utc)
    E.write_marker(wf.dir, editor_ran_at=ran)
    assert E.watermark(wf.dir) == ran
    WF._write_cleanup_marker(wf)            # a later floor-only marker write
    assert E.watermark(wf.dir) == ran
    data = E.read_marker(wf.dir)
    # The four mtimes, their byte sizes (same-tick rewrite guard) and the watermark.
    assert set(data) == {
        "state", "decisions", "lessons", "index",
        "state_size", "decisions_size", "lessons_size", "index_size",
        "editor_last_run",
    }


def test_auto_run_due_needs_a_change_and_an_old_enough_watermark(wf):
    now = datetime(2026, 9, 19, 12, 0, tzinfo=timezone.utc)
    assert E.auto_run_due(wf.dir, now=now)                       # never ran
    E.write_marker(wf.dir, editor_ran_at=now - timedelta(minutes=5))
    assert not E.auto_run_due(wf.dir, now=now)                   # no change
    wf.write("state", STATE + "- one more\n")
    assert not E.auto_run_due(wf.dir, now=now)                   # changed, but too soon
    assert E.auto_run_due(wf.dir, now=now + timedelta(hours=1))  # changed, and old enough


def test_prepass_normalises_and_stamps_unstamped_entries_only(tmp_path):
    w = WorkspaceFileManager(str(tmp_path))
    os.makedirs(w.dir)
    with open(w._file_path("state"), "w") as f:     # an external writer
        f.write("## Now\n- written by an external agent\n")
    with open(w._file_path("decisions"), "w") as f:  # legacy heading shape, one unstamped
        f.write("# D\n\n## New decision\nbody\n\n"
                "## Old <!--mem id:old01 created:2026-08-01 touched:2026-08-01-->\n**Chose:** A.\n")
    result = WF.prepass(w, today=TODAY)
    assert set(result.touched) == {"state", "decisions"}
    assert state_blocks.parse(w.read("state"))[1][0].id
    d = w.read("decisions")
    assert "- New decision\n  <!--mem id:new-decision created:2026-09-19 touched:2026-09-19-->\n  body" in d
    assert "- Old\n  <!--mem id:old01 created:2026-08-01 touched:2026-08-01-->\n  **Chose:** A." in d
    # a second prepass is a no-op
    assert WF.prepass(w, today=TODAY).touched == []


# ---------------------------------------------------------------------------
# a whole pass: the agent moves entries through the write chokepoint
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_pass_sees_prepassed_files_and_keeps_every_moved_line_in_an_archive(wf):
    before = _live_lines(wf)
    seen = {}

    async def run_turn(task_text):
        seen["task"] = task_text
        _archive_block(wf, "state", _bullet("Cold-DM automation paused since 08-15", "cdm001", "2026-08-10"))
        _archive_block(wf, "decisions", _decision("PenguinHarness → watchlist", "penguin").rstrip("\n"))
        return _turn(tool_calls=4)

    result = await E.run_pass(wf, run_turn, today=TODAY)
    assert result.ok and result.outcome == "edited"
    assert "PROJECT_STATE.md" in seen["task"] and "DECISIONS.md" in seen["task"]
    archives = (wf.read("state_archive") or "") + (wf.read("decisions_archive") or "")
    for key in ("state", "decisions"):
        now_lines = set(wf.read(key).split("\n"))
        for line in before[key].split("\n"):
            if line.strip() and line not in now_lines:
                assert line in archives, line
    live = wf.read("state") + wf.read("decisions")
    assert "[archived" not in live
    for eid in ("cdm001", "penguin"):
        assert f"id:{eid}" not in live
        assert f"id:{eid}" in archives
        assert f"- id:{eid} · " in archives            # contents-table row
    assert result.restored == []
    assert "DECISIONS_ARCHIVE.md — 1 archived entry" in wf.read("index")
    assert os.path.isfile(os.path.join(result.backup_dir, "PROJECT_STATE.md"))
    with open(os.path.join(result.backup_dir, "editor-run.json")) as f:
        record = json.load(f)
    assert record["outcome"] == "edited" and record["turn"]["tool_calls"] == 4


@pytest.mark.asyncio
async def test_pass_archives_a_replaced_decision_under_the_replaced_heading(wf):
    """Spec 182 §5.4: the Memory session links a replaced entry to the live
    entry that replaced it; the contents row carries the arrow and the
    post-pass leaves the entry in the archive."""
    from agent_os.agent import memory_entries as M
    old = _decision("PenguinHarness → watchlist", "penguin").rstrip("\n")
    live_ids = [b.id for b in state_blocks.parse(wf.read("decisions"))[1]]
    keeper = [i for i in live_ids if i != "penguin"][0]

    async def run_turn(task_text):
        _archive_block(wf, "decisions", old, heading=f"## [replaced {TODAY} by id:{keeper}]")
        return _turn(tool_calls=2)

    result = await E.run_pass(wf, run_turn, today=TODAY)
    assert result.ok and result.restored == []
    archive = wf.read("decisions_archive")
    assert f"## [replaced {TODAY} by id:{keeper}]" in archive
    assert f"- id:penguin · replaced {TODAY} → id:{keeper} · " in archive
    assert "id:penguin" not in wf.read("decisions")
    assert M.archive_rows(archive)[0].kind == "replaced"


@pytest.mark.asyncio
async def test_a_bare_deletion_during_the_pass_is_restored_from_the_snapshot(wf, caplog):
    async def run_turn(task_text):
        text = wf.read("decisions").replace(_decision("TabTin → watchlist", "tabtin"), "")
        with open(wf._file_path("decisions"), "w", encoding="utf-8") as f:   # not through the write guard
            f.write(text)
        return _turn(tool_calls=1)

    with caplog.at_level("WARNING"):
        result = await E.run_pass(wf, run_turn, today=TODAY)
    assert result.outcome == "restored" and result.restored == ["tabtin"]
    assert "- TabTin → watchlist\n  <!--mem id:tabtin" in wf.read("decisions")
    assert "- Pinned rule" in wf.read("decisions")


@pytest.mark.asyncio
async def test_turn_failure_is_reported_not_raised_and_the_files_are_untouched(wf):
    async def boom(task_text):
        raise RuntimeError("provider 500")
    result = await E.run_pass(wf, boom, today=TODAY)
    assert not result.ok and result.outcome == "failed"
    assert wf.read("state").count("<!--mem") == 4


@pytest.mark.asyncio
async def test_an_index_rewrite_keeps_archive_lines(wf):
    async def run_turn(task_text):
        wf.write("index", "# INDEX\n\n- src/ — code\n")
        return _turn(tool_calls=1)
    result = await E.run_pass(wf, run_turn, today=TODAY)
    assert result.outcome == "index_only"
    idx = wf.read("index")
    assert "- src/ — code" in idx and "DECISIONS_ARCHIVE.md" in idx


# ---------------------------------------------------------------------------
# run_session_end_routine: gates, single flight, watermark, floor
# ---------------------------------------------------------------------------

def _over_budget(wf):
    big = "\n".join(_bullet(f"recent item {i} " + "r" * 400, f"big{i:03d}", "2026-09-18") for i in range(25))
    wf.write("state", STATE + big + "\n")


async def _quiet(task_text):
    return _turn()


@pytest.mark.asyncio
async def test_routine_not_needed_when_under_budget_unless_forced(wf):
    calls = []

    async def run_turn(task_text):
        calls.append(task_text)
        return _turn()
    assert await run_session_end_routine(None, None, wf, session_uuid="s", bypass_idempotency=True,
                                         run_turn=run_turn) == "not_needed"
    assert calls == []
    assert await run_session_end_routine(
        None, None, wf, session_uuid="s", bypass_idempotency=True, force=True, run_turn=run_turn) == "no_change"
    assert len(calls) == 1 and "requested explicitly (checkpoint_state)" in calls[0]


@pytest.mark.asyncio
async def test_routine_success_advances_the_watermark_then_no_delta(wf):
    _over_budget(wf)
    assert E.read_marker(wf.dir).get("editor_last_run") is None
    out = await run_session_end_routine(None, None, wf, session_uuid="s", bypass_idempotency=True, run_turn=_quiet)
    assert out == "no_change"
    assert E.read_marker(wf.dir).get("editor_last_run")
    again = await run_session_end_routine(None, None, wf, session_uuid="s", bypass_idempotency=True, run_turn=_quiet)
    assert again == "no_delta"


@pytest.mark.asyncio
async def test_routine_failure_runs_floor_and_leaves_watermark_alone(wf):
    _over_budget(wf)

    async def boom(task_text):
        raise RuntimeError("x")
    out = await run_session_end_routine(None, None, wf, session_uuid="s", bypass_idempotency=True, run_turn=boom)
    assert out == "backstop_only"
    assert "editor_last_run" not in E.read_marker(wf.dir)


@pytest.mark.asyncio
async def test_single_flight_across_two_callers(wf):
    _over_budget(wf)
    gate = asyncio.Event()
    calls = []

    async def slow(task_text):
        calls.append(1)
        await gate.wait()
        return _turn()

    first = asyncio.create_task(run_session_end_routine(
        None, None, wf, session_uuid="a", bypass_idempotency=True, run_turn=slow))
    await asyncio.sleep(0.05)
    second = await run_session_end_routine(None, None, wf, session_uuid="b", bypass_idempotency=True, run_turn=slow)
    assert second == "in_flight"
    gate.set()
    assert await first == "no_change"
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_a_declined_turn_leaves_the_pass_unfinished_so_it_retries(wf):
    """The Memory session was busy (the user mid-turn there): nothing ran,
    so the watermark must not advance and the delta stays."""
    _over_budget(wf)

    async def declined(task_text):
        return None
    out = await run_session_end_routine(None, None, wf, session_uuid="s", bypass_idempotency=True, run_turn=declined)
    assert out == "skipped"
    assert "editor_last_run" not in E.read_marker(wf.dir)
    assert WF._has_cleanup_delta(wf)


def test_same_tick_rewrite_with_a_different_size_is_a_delta(tmp_path):
    """Windows NTFS rounds LastWriteTime to the system timer, so two writes
    in one tick share an mtime; the size must still count as a change."""
    from datetime import datetime, timezone
    import os
    from agent_os.agent import memory_editor as E

    orbital = tmp_path / "orbital"
    orbital.mkdir()
    state = orbital / E.FILE_BY_KEY["state"]
    state.write_text("one\n", encoding="utf-8")
    E.write_marker(str(orbital), editor_ran_at=datetime.now(timezone.utc))
    assert E.has_delta(str(orbital)) is False

    st = os.stat(state)
    state.write_text("one\ntwo\n", encoding="utf-8")
    os.utime(state, ns=(st.st_atime_ns, st.st_mtime_ns))   # same tick, same mtime
    assert E.has_delta(str(orbital)) is True


def test_marker_without_size_keys_still_answers_from_mtimes(tmp_path):
    """A marker written by an older daemon carries only the mtime keys."""
    import json
    from agent_os.agent import memory_editor as E

    orbital = tmp_path / "orbital"
    orbital.mkdir()
    state = orbital / E.FILE_BY_KEY["state"]
    state.write_text("one\n", encoding="utf-8")
    E.write_marker(str(orbital), editor_ran_at=None)
    marker = E._marker_path(str(orbital))
    data = json.loads(open(marker, encoding="utf-8").read())
    for k in list(data):
        if k.endswith("_size"):
            del data[k]
    open(marker, "w", encoding="utf-8").write(json.dumps(data))
    assert E.has_delta(str(orbital)) is False
