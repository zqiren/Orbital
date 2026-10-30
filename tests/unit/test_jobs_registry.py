# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Spec 134 Part B: the background-job registry (fanout with process workers).

Real processes, driven by explicit ``tick()`` calls instead of the 2 s poller.
"""

import asyncio
import json
import os
import subprocess
import sys
import time

import psutil
import pytest

from agent_os.agent.project_paths import ProjectPaths
from agent_os.daemon_v2 import job_launcher, jobs
from agent_os.daemon_v2.jobs import JobRegistry

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="POSIX shells")

PROJECT = "proj_jobs"
SID = "sess-jobs-0001"


class _Recorder:
    def __init__(self):
        self.injected: list[dict] = []
        self.broadcasts: list[dict] = []
        self.dispatched: list[dict] = []

    async def inject(self, project_id, content, *, session_id=None, meta=None):
        self.injected.append({"project_id": project_id, "content": content,
                              "session_id": session_id, "meta": meta or {}})
        return "delivered"

    def broadcast(self, project_id, payload):
        self.broadcasts.append(payload)

    async def dispatch(self, project_id, session_id, handle, content):
        self.dispatched.append({"project_id": project_id, "session_id": session_id,
                                "handle": handle, "content": content})

    def wakes(self, event=jobs.EVENT_TERMINAL):
        return [r for r in self.injected if r["meta"].get("event") == event]


class _Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


@pytest.fixture
def rec():
    return _Recorder()


@pytest.fixture
def clock():
    return _Clock()


@pytest.fixture
def registry(rec, clock, tmp_path):
    reg = JobRegistry(inject=rec.inject, broadcast=rec.broadcast,
                      dispatch_worker=rec.dispatch,
                      workspaces=lambda: [(PROJECT, str(tmp_path))],
                      clock=clock, poll_interval_s=3600)
    yield reg
    reg.stop_all()


def _spawn(registry, ws, command, *, batch="b1", session=SID):
    group = registry.open_group(PROJECT, session, str(ws), batch_key=batch)
    return registry.spawn_job(group, command, ["/bin/sh", "-c", command])


async def _tick_until(registry, cond, timeout=10.0):
    start = time.monotonic()
    while not cond():
        if time.monotonic() - start > timeout:
            raise AssertionError("condition not met")
        await registry.tick()
        await asyncio.sleep(0.05)


async def _started(registry):
    await registry.start()


# ---------------------------------------------------------------------------
# Groups and the join rule
# ---------------------------------------------------------------------------


async def test_two_background_calls_in_one_batch_form_one_group_and_one_join(
        registry, rec, tmp_path):
    await _started(registry)
    a = _spawn(registry, tmp_path, "echo alpha; sleep 0.3")
    b = _spawn(registry, tmp_path, "echo beta; sleep 1")
    assert a.group_id == b.group_id

    await _tick_until(registry, lambda: a.state != "running")
    assert rec.wakes() == []          # a succeeded, b still running: no wake yet

    await _tick_until(registry, lambda: rec.wakes())
    [wake] = rec.wakes()
    assert wake["session_id"] == SID
    assert wake["meta"]["kind"] == "job_join"
    assert "2 background jobs finished" in wake["content"]
    assert "alpha" in wake["content"] and "beta" in wake["content"]
    assert f"{a.id} `echo alpha; sleep 0.3`: exited 0" in wake["meta"]["display_content"]
    states = {j["job_id"]: j["status"] for j in wake["meta"]["job_group"]["jobs"]}
    assert states == {a.id: "completed", b.id: "completed"}
    await registry.tick()
    assert len(rec.wakes()) == 1      # idempotent


async def test_a_failing_job_wakes_at_once_and_the_rest_wake_again_when_they_end(
        registry, rec, tmp_path):
    await _started(registry)
    bad = _spawn(registry, tmp_path, "echo boom >&2; exit 3")
    slow = _spawn(registry, tmp_path, "sleep 1.2; echo done")

    await _tick_until(registry, lambda: rec.wakes())
    [first] = rec.wakes()
    assert slow.state == "running"
    assert "1 of 2 background jobs did not succeed; 1 still running" in first["content"]
    assert "failed: exit 3" in first["content"]
    assert "boom" in first["content"]
    assert "still running" in first["content"]

    await _tick_until(registry, lambda: len(rec.wakes()) == 2)
    second = rec.wakes()[1]
    assert f"{slow.id} `sleep 1.2; echo done`: exited 0" in second["content"]
    assert "1 of 2 background jobs did not succeed." in second["content"]
    assert bad.reported and slow.reported


async def test_a_separate_batch_is_a_separate_group(registry, tmp_path):
    a = _spawn(registry, tmp_path, "true", batch="b1")
    b = _spawn(registry, tmp_path, "true", batch="b2")
    c = _spawn(registry, tmp_path, "true", batch=None)
    assert len({a.group_id, b.group_id, c.group_id}) == 3


async def test_the_join_carries_the_last_20_lines(registry, rec, tmp_path):
    await _started(registry)
    job = _spawn(registry, tmp_path, "for i in $(seq 1 50); do echo line$i; done")
    await _tick_until(registry, lambda: rec.wakes())
    content = rec.wakes()[0]["content"]
    assert "line50" in content and "line31" in content
    assert "line30\n" not in content
    assert job.rel_log_path in content


async def test_the_start_text_names_the_job_the_output_and_the_kill_command(
        registry, tmp_path):
    job = _spawn(registry, tmp_path, "sleep 5")
    text = registry.start_text(job)
    assert f"Started background job {job.id} (pid {job.pid}) in group {job.group_id}." in text
    assert f"orbital/jobs/{job.id}/output.log" in text
    assert f"kill -TERM -- -{job.pid}" in text
    assert "Your turn ends after this message's tool calls" in text
    assert "every 15 minutes" in text


async def test_job_json_is_minimal_and_atomic(registry, tmp_path):
    job = _spawn(registry, tmp_path, "sleep 5")
    data = json.loads(open(os.path.join(job.dir, "job.json")).read())
    assert data["state"] == "running" and data["pid"] == job.pid
    assert data["session_id"] == SID and data["create_time"]
    assert not [n for n in os.listdir(job.dir) if n.endswith(".tmp")]


# ---------------------------------------------------------------------------
# Check-ins
# ---------------------------------------------------------------------------


async def test_a_check_in_wakes_with_runtime_growth_and_tail(registry, rec, clock, tmp_path):
    await _started(registry)
    _spawn(registry, tmp_path, "echo progress-1; sleep 30")
    await asyncio.sleep(0.3)
    clock.now += jobs.CHECKIN_INTERVAL_S - 1
    await registry.tick()
    assert rec.wakes(jobs.EVENT_CHECKIN) == []
    clock.now += 2
    await registry.tick()
    [checkin] = rec.wakes(jobs.EVENT_CHECKIN)
    assert "Check-in: 1 job still running" in checkin["content"]
    assert "output grew since the last check-in" in checkin["content"]
    assert "progress-1" in checkin["content"]
    assert "To keep waiting, just end your turn" in checkin["content"]


async def test_three_silent_check_ins_stop_check_ins_but_the_exit_still_wakes(
        registry, rec, clock, tmp_path):
    await _started(registry)
    job = _spawn(registry, tmp_path, "sleep 1.5; exit 0")
    for _ in range(5):
        clock.now += jobs.CHECKIN_INTERVAL_S + 1
        await registry.tick()
    checkins = rec.wakes(jobs.EVENT_CHECKIN)
    assert len(checkins) == 3
    assert all("no new output" in c["content"] for c in checkins)
    [note] = rec.wakes(jobs.EVENT_NOTE)
    assert note["meta"]["suppress_wake"] is True
    assert "stopped checking in" in note["content"]
    await _tick_until(registry, lambda: rec.wakes())
    assert job.state == "exited"


async def test_a_user_message_between_check_ins_resets_the_silent_count(
        registry, rec, clock, tmp_path):
    await _started(registry)
    _spawn(registry, tmp_path, "sleep 30")
    for i in range(5):
        if i == 2:
            registry.note_user_message(PROJECT, SID)
        clock.now += jobs.CHECKIN_INTERVAL_S + 1
        await registry.tick()
    assert len(rec.wakes(jobs.EVENT_CHECKIN)) == 5
    assert rec.wakes(jobs.EVENT_NOTE) == []


async def test_the_check_in_interval_env_override(monkeypatch):
    monkeypatch.setenv(jobs.CHECKIN_ENV, "45")
    assert jobs.checkin_interval_s() == 45.0
    monkeypatch.setenv(jobs.CHECKIN_ENV, "nonsense")
    assert jobs.checkin_interval_s() == jobs.CHECKIN_INTERVAL_S


# ---------------------------------------------------------------------------
# Guards
# ---------------------------------------------------------------------------


async def test_the_same_command_failing_three_times_stops_auto_waking(
        registry, rec, tmp_path):
    await _started(registry)
    for i in range(4):
        _spawn(registry, tmp_path, "exit 1", batch=f"b{i}")
        await _tick_until(registry, lambda: len(rec.wakes()) == i + 1)
    metas = [w["meta"] for w in rec.wakes()]
    assert [bool(m.get("suppress_wake")) for m in metas] == [False, False, False, True]
    assert "failed 3 times in a row. Ask the user." in rec.wakes()[3]["content"]
    # the next user message resets it
    registry.note_user_message(PROJECT, SID)
    _spawn(registry, tmp_path, "exit 1", batch="b9")
    await _tick_until(registry, lambda: len(rec.wakes()) == 5)
    assert not rec.wakes()[4]["meta"].get("suppress_wake")


async def test_a_success_between_failures_resets_the_failure_guard(registry, rec, tmp_path):
    await _started(registry)
    for i, cmd in enumerate(["exit 1", "exit 1", "true", "exit 1", "exit 1"]):
        _spawn(registry, tmp_path, cmd, batch=f"b{i}")
        await _tick_until(registry, lambda: len(rec.wakes()) == i + 1)
    assert not any(w["meta"].get("suppress_wake") for w in rec.wakes())


async def test_at_most_20_job_wakes_per_session_per_hour(registry, rec, clock):
    group = registry.open_group(PROJECT, SID, "/nonexistent")
    for i in range(21):
        await registry._deliver(group, f"msg {i}", {"event": jobs.EVENT_CHECKIN})
    assert [bool(r["meta"].get("suppress_wake")) for r in rec.injected][-2:] == [False, True]
    assert "20 background-job wakes in the last hour" in rec.injected[-1]["content"]
    clock.now += 3601
    await registry._deliver(group, "later", {"event": jobs.EVENT_CHECKIN})
    assert not rec.injected[-1]["meta"].get("suppress_wake")


# ---------------------------------------------------------------------------
# Limits
# ---------------------------------------------------------------------------


async def test_the_ninth_running_job_is_refused(registry, tmp_path):
    for i in range(8):
        _spawn(registry, tmp_path, "sleep 30", batch=f"b{i}")
    refusal = registry.admission(PROJECT, str(tmp_path))
    assert refusal and "8 background jobs running" in refusal
    assert registry.admission("other_project", str(tmp_path)) is None


async def test_new_jobs_are_refused_past_1_gb_of_job_output(registry, tmp_path, monkeypatch):
    monkeypatch.setattr(jobs, "JOBS_DIR_CEILING_BYTES", 10)
    job = _spawn(registry, tmp_path, "echo 0123456789abcdef")
    job.proc.wait(5)
    registry._size_cache.clear()
    assert "more than 1 GB" in registry.admission(PROJECT, str(tmp_path))


async def test_output_log_rotates_keeping_the_tail(registry, tmp_path, monkeypatch):
    monkeypatch.setattr(jobs, "ROTATE_AT_BYTES", 2000)
    monkeypatch.setattr(jobs, "ROTATE_KEEP_BYTES", 500)
    job = _spawn(registry, tmp_path,
                 "i=0; while [ $i -lt 400 ]; do echo row$i; i=$((i+1)); done; "
                 "sleep 0.5; echo after-rotation; sleep 30")
    deadline = time.monotonic() + 5
    while job.log_size() < 2500 and time.monotonic() < deadline:
        await asyncio.sleep(0.05)
    registry._maybe_rotate(job)
    kept = open(job.log_path + ".1", "rb").read()
    assert len(kept) == 500 and b"row399" in kept
    await asyncio.sleep(0.8)
    log = open(job.log_path, "rb").read()
    assert log.startswith(b"after-rotation"), log[:40]   # O_APPEND: no hole
    assert job.bytes_written() > 2500


# ---------------------------------------------------------------------------
# Cancel, quit, delete
# ---------------------------------------------------------------------------


def _children(pid):
    try:
        return psutil.Process(pid).children(recursive=True)
    except psutil.Error:
        return []


async def test_cancel_ends_the_whole_tree_and_reports_cancelled(registry, rec, tmp_path):
    await _started(registry)
    job = _spawn(registry, tmp_path, "sleep 60 & sleep 60; wait")
    await asyncio.sleep(0.3)
    kids = _children(job.pid)
    assert kids
    result = await asyncio.to_thread(registry.cancel, PROJECT, job.id)
    assert result.state == "cancelled"
    await asyncio.sleep(0.2)
    assert not any(k.is_running() and k.status() != psutil.STATUS_ZOMBIE for k in kids)
    await _tick_until(registry, lambda: rec.wakes())
    assert "cancelled after" in rec.wakes()[0]["content"]
    assert json.load(open(os.path.join(job.dir, "job.json")))["state"] == "cancelled"


async def test_cancel_scope_cancels_without_waking(registry, rec, tmp_path):
    await _started(registry)
    a = _spawn(registry, tmp_path, "sleep 60", session="s-a")
    b = _spawn(registry, tmp_path, "sleep 60", session="s-b")
    assert registry.cancel_scope(PROJECT, "s-a") == [a.id]
    assert b.state == "running"
    await registry.tick()
    await asyncio.sleep(0.05)
    assert rec.wakes() == []
    assert registry.cancel_scope(PROJECT) == [b.id]


async def test_cancel_of_another_projects_job_is_refused(registry, tmp_path):
    job = _spawn(registry, tmp_path, "sleep 30")
    assert registry.cancel("proj_other", job.id) is None
    assert job.state == "running"


async def test_stop_all_ends_every_tree_and_owes_a_note(registry, tmp_path):
    job = _spawn(registry, tmp_path, "sleep 60 & sleep 60; wait")
    await asyncio.sleep(0.2)
    kids = _children(job.pid)
    assert registry.stop_all() == 1
    await asyncio.sleep(0.2)
    assert not psutil.pid_exists(job.pid) or psutil.Process(job.pid).status() == psutil.STATUS_ZOMBIE
    assert not any(k.is_running() and k.status() != psutil.STATUS_ZOMBIE for k in kids)
    data = json.load(open(os.path.join(job.dir, "job.json")))
    assert data["state"] == "stopped" and data["noted"] is False


async def test_module_level_quit_helpers(registry, tmp_path):
    await registry.start()
    _spawn(registry, tmp_path, "sleep 30")
    assert jobs.running_job_count() == 1
    assert jobs.stop_all_jobs() == 1
    assert jobs.running_job_count() == 0


async def test_is_running_dir_protects_a_running_jobs_folder(registry, tmp_path):
    job = _spawn(registry, tmp_path, "sleep 30")
    assert registry.is_running_dir(job.dir)
    assert registry.is_running_dir(os.path.join(str(tmp_path), "orbital"))
    assert not registry.is_running_dir(str(tmp_path / "elsewhere"))


# ---------------------------------------------------------------------------
# After a crash: the next launch ends leftovers and notes them
# ---------------------------------------------------------------------------


def _orphan(tmp_path, job_id, *, create_time=True, cmd="sleep 60"):
    d = ProjectPaths(str(tmp_path)).job_dir(job_id)
    os.makedirs(d)
    proc = subprocess.Popen(["/bin/sh", "-c", f"{cmd} & {cmd}; wait"], start_new_session=True)
    time.sleep(0.2)
    data = {"id": job_id, "group_id": "grp_old", "project_id": PROJECT,
            "session_id": SID, "command": cmd, "pid": proc.pid,
            "create_time": psutil.Process(proc.pid).create_time() if create_time else None,
            "started_at": time.time() - 100, "state": "running"}
    if create_time == "mismatch":
        data["create_time"] = 12345.0
    json.dump(data, open(os.path.join(d, "job.json"), "w"))
    return proc, d


async def test_next_launch_kills_a_leftover_job_and_appends_a_non_waking_note(
        registry, rec, tmp_path):
    proc, d = _orphan(tmp_path, "job_dead0001")
    kids = _children(proc.pid)
    await registry.start()
    proc.wait(5)
    await asyncio.sleep(0.2)
    assert not any(k.is_running() and k.status() != psutil.STATUS_ZOMBIE for k in kids)
    [note] = rec.injected
    assert note["session_id"] == SID
    assert note["meta"]["suppress_wake"] is True
    assert note["meta"]["event"] == jobs.EVENT_NOTE
    assert note["content"] == "Background job job_dead0001 (`sleep 60`) was stopped because Orbital closed."
    data = json.load(open(os.path.join(d, "job.json")))
    assert data["state"] == "stopped" and data["noted"] is True
    # a second launch does not note it again
    rec.injected.clear()
    registry2 = JobRegistry(inject=rec.inject, broadcast=rec.broadcast,
                            workspaces=lambda: [(PROJECT, str(tmp_path))])
    await registry2.start()
    await registry2.shutdown()
    assert rec.injected == []


async def test_a_reused_pid_is_never_killed(registry, rec, tmp_path):
    proc, _d = _orphan(tmp_path, "job_reuse001", create_time="mismatch")
    try:
        await registry.start()
        assert proc.poll() is None            # not ours: left alone
        assert "was stopped because Orbital closed" in rec.injected[0]["content"]
    finally:
        os.killpg(proc.pid, 15)
        proc.wait(5)


async def test_a_clean_quits_stopped_jobs_get_their_note_on_the_next_launch(
        registry, rec, tmp_path):
    job = _spawn(registry, tmp_path, "sleep 30")
    registry.stop_all()
    registry2 = JobRegistry(inject=rec.inject, broadcast=rec.broadcast,
                            workspaces=lambda: [(PROJECT, str(tmp_path))])
    await registry2.start()
    await registry2.shutdown()
    [note] = rec.injected
    assert note["content"].startswith(f"Background job {job.id} (`sleep 30`) was stopped")


# ---------------------------------------------------------------------------
# Launcher jobs (codex / pi workers)
# ---------------------------------------------------------------------------


def _run_launcher(ws, command, **env):
    job_launcher.ensure_launcher(str(ws))
    out = subprocess.run(
        [os.path.join(str(ws), "orbital", "bin", "bg"), command], cwd=str(ws),
        capture_output=True, text=True, timeout=10,
        env={**os.environ, **env})
    assert out.returncode == 0, out.stderr
    return out.stdout


async def test_a_launcher_job_is_attributed_and_its_join_is_dispatched_to_the_worker(
        registry, rec, tmp_path):
    await registry.start()
    registry.watch(str(tmp_path), PROJECT)
    out = _run_launcher(tmp_path, "echo from-codex; sleep 0.5; exit 4",
                        ORBITAL_SESSION_ID=SID, ORBITAL_PROJECT_ID=PROJECT,
                        ORBITAL_AGENT="codex")
    assert "Started background job job_" in out
    assert "Orbital will send you a message when it finishes" in out
    await _tick_until(registry, lambda: rec.dispatched)
    [d] = rec.dispatched
    assert (d["project_id"], d["session_id"], d["handle"]) == (PROJECT, SID, "codex")
    assert "failed: exit 4" in d["content"] and "from-codex" in d["content"]
    [job] = registry.list_jobs(PROJECT)
    assert job["agent"] == "codex" and job["exit_code"] == 4
    started = [r for r in rec.injected if r["meta"].get("event") == jobs.EVENT_STARTED]
    assert started and started[0]["meta"]["suppress_wake"] is True


async def test_the_launchers_kill_command_ends_the_job_and_records_143(registry, tmp_path):
    out = _run_launcher(tmp_path, "sleep 60", ORBITAL_SESSION_ID=SID)
    pid = int(out.split("(pid ")[1].split(")")[0])
    job_id = out.split("Started background job ")[1].split(" ")[0]
    time.sleep(0.2)
    os.killpg(pid, 15)
    exit_path = os.path.join(str(tmp_path), "orbital", "jobs", job_id, "exit")
    deadline = time.monotonic() + 5
    while not os.path.exists(exit_path) and time.monotonic() < deadline:
        time.sleep(0.05)
    assert open(exit_path).read().split()[0] == "143"


async def test_a_launcher_without_a_session_says_it_is_not_tracked(tmp_path):
    env = {k: v for k, v in os.environ.items() if not k.startswith("ORBITAL_")}
    job_launcher.ensure_launcher(str(tmp_path))
    out = subprocess.run([os.path.join(str(tmp_path), "orbital", "bin", "bg"), "true"],
                         cwd=str(tmp_path), capture_output=True, text=True, env=env,
                         timeout=10).stdout
    assert "not tracking this job" in out


async def test_a_launcher_job_killed_without_an_exit_file_is_never_success(
        registry, rec, tmp_path):
    await registry.start()
    registry.watch(str(tmp_path), PROJECT)
    out = _run_launcher(tmp_path, "sleep 60", ORBITAL_SESSION_ID=SID,
                        ORBITAL_PROJECT_ID=PROJECT)
    pid = int(out.split("(pid ")[1].split(")")[0])
    await _tick_until(registry, lambda: registry.list_jobs(PROJECT))
    os.killpg(pid, 9)
    await _tick_until(registry, lambda: rec.wakes())
    assert "exit code unknown" in rec.wakes()[0]["content"]
    assert "did not succeed" in rec.wakes()[0]["content"]


def test_ensure_launcher_writes_executable_files_once(tmp_path):
    d = job_launcher.ensure_launcher(str(tmp_path))
    bg = os.path.join(d, "bg")
    assert os.access(bg, os.X_OK)
    assert sorted(os.listdir(d)) == ["bg", "bg-run.cmd", "bg.cmd", "bg.ps1"]
    mtime = os.stat(bg).st_mtime_ns
    time.sleep(0.01)
    job_launcher.ensure_launcher(str(tmp_path))
    assert os.stat(bg).st_mtime_ns == mtime
    assert job_launcher.ensure_launcher("") is None


# ---------------------------------------------------------------------------
# Retention
# ---------------------------------------------------------------------------


def _ended_job(ws, job_id, group_id, *, age_s=0.0, state="exited"):
    d = ProjectPaths(str(ws)).job_dir(job_id)
    os.makedirs(d)
    t = time.time() - age_s
    json.dump({"id": job_id, "group_id": group_id, "state": state, "started_at": t,
               "ended_at": t}, open(os.path.join(d, "job.json"), "w"))
    return d


def test_retention_keeps_the_last_50_groups_and_7_days(registry, tmp_path):
    for i in range(60):
        _ended_job(tmp_path, f"job_{i:08d}", f"grp_{i:04d}", age_s=1000 - i)
    _ended_job(tmp_path, "job_old00000", "grp_old", age_s=8 * 24 * 3600)
    _ended_job(tmp_path, "job_live0000", "grp_live", age_s=9 * 24 * 3600, state="running")
    removed = registry.apply_retention(str(tmp_path))
    left = sorted(os.listdir(ProjectPaths(str(tmp_path)).jobs_dir))
    assert "job_live0000" in left            # never a running job's folder
    assert "job_old00000" in removed
    assert len(left) == 51
    assert "job_00000059" in left and "job_00000009" not in left


async def test_a_worker_that_refuses_the_dispatch_gets_it_in_the_session_instead(rec, tmp_path):
    async def refuse(*a):
        return "Error: agent 'codex' transport is broken"
    reg = JobRegistry(inject=rec.inject, broadcast=rec.broadcast, dispatch_worker=refuse)
    group = reg.open_group(PROJECT, SID, str(tmp_path), agent="codex")
    await reg._deliver(group, "job ended", {"event": jobs.EVENT_TERMINAL})
    assert [r["content"] for r in rec.injected] == ["job ended"]


# ---- spec 148: what the agent is told it can rely on ----

@pytest.mark.asyncio
async def test_start_text_says_detached_work_gets_no_message(registry, tmp_path):
    job = _spawn(registry, tmp_path, "true")
    assert "detaches gets no message" in registry.start_text(job)


@pytest.mark.asyncio
async def test_the_ended_line_is_only_on_a_join_with_nothing_running(registry, tmp_path):
    a = _spawn(registry, tmp_path, "true", batch="ended-line")
    group = registry._groups[a.group_id]
    b = registry.spawn_job(group, "sleep 60", ["/bin/sh", "-c", "sleep 60"])
    await _tick_until(registry, lambda: a.state != "running")
    early, _ = registry._compose_join(group, [])
    assert "isn't tracking anything they left running" not in early
    assert "You will get another message" in early

    registry.cancel(PROJECT, b.id, notify=False)
    final, _ = registry._compose_join(group, [])
    assert "isn't tracking anything they left running" in final
