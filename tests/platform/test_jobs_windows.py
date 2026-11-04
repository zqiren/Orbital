# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Spec 134 Part B on a real Windows host (CI windows-latest): background jobs
in a kill-on-close Job Object, tree cancel, quit, and the bg.cmd launcher."""

import asyncio
import json
import os
import subprocess
import sys
import time

import psutil
import pytest

pytestmark = pytest.mark.skipif(sys.platform != "win32", reason="Windows only")

from agent_os.daemon_v2 import job_launcher  # noqa: E402
from agent_os.daemon_v2.jobs import JobRegistry  # noqa: E402

SID = "sess-win-1"


async def _inject(*a, **k):
    return "delivered"


def _registry():
    return JobRegistry(inject=_inject, broadcast=lambda *a: None, poll_interval_s=3600)


def _ps(command):
    return ["powershell", "-NoProfile", "-NonInteractive", "-Command", command]


def _tree(pid):
    try:
        p = psutil.Process(pid)
        return [p, *p.children(recursive=True)]
    except psutil.Error:
        return []


def _wait_gone(procs, timeout=10):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not any(p.is_running() for p in procs):
            return True
        time.sleep(0.2)
    return False


def test_cancel_ends_the_job_and_its_grandchild(tmp_path):
    reg = _registry()
    group = reg.open_group("p", SID, str(tmp_path))
    job = reg.spawn_job(group, "sleep with a child",
                        _ps("Start-Process -NoNewWindow ping -ArgumentList '-n','60','127.0.0.1'; "
                            "Start-Sleep 60"))
    time.sleep(3)
    procs = _tree(job.pid)
    assert len(procs) >= 2
    reg.cancel("p", job.id, notify=False)
    assert _wait_gone(procs)
    assert job.state == "cancelled"


def test_stop_all_closes_the_kill_on_close_job(tmp_path):
    reg = _registry()
    group = reg.open_group("p", SID, str(tmp_path))
    job = reg.spawn_job(group, "long", _ps("Start-Sleep 60"))
    time.sleep(2)
    procs = _tree(job.pid)
    assert reg.stop_all() == 1
    assert _wait_gone(procs)


def test_the_bg_cmd_launcher_records_output_and_exit_code(tmp_path):
    job_launcher.ensure_launcher(str(tmp_path))
    env = {**os.environ, "ORBITAL_SESSION_ID": SID, "ORBITAL_PROJECT_ID": "p"}
    out = subprocess.run(
        [os.path.join(str(tmp_path), "orbital", "bin", "bg.cmd"), "echo hello-win & exit /b 3"],
        cwd=str(tmp_path), capture_output=True, text=True, env=env, timeout=60)
    assert out.returncode == 0, out.stderr
    assert "Started background job job_" in out.stdout
    assert "taskkill /T /F /PID" in out.stdout
    job_id = out.stdout.split("Started background job ")[1].split(" ")[0]
    d = os.path.join(str(tmp_path), "orbital", "jobs", job_id)
    exit_path = os.path.join(d, "exit")
    deadline = time.monotonic() + 30
    while not os.path.exists(exit_path) and time.monotonic() < deadline:
        time.sleep(0.2)
    assert open(exit_path).read().split()[0] == "3"
    assert "hello-win" in open(os.path.join(d, "output.log"), errors="replace").read()

    reg = _registry()
    reg.watch(str(tmp_path), "p")
    asyncio.run(reg.tick())
    [row] = reg.list_jobs("p")
    assert row["session_id"] == SID


def test_the_bg_cmd_launcher_keeps_a_chinese_session_id(tmp_path):
    # Spec 159: bg.ps1's Clean keeps Unicode word characters, so the job-end
    # message reaches a session of a Chinese-named project; quotes still go.
    job_launcher.ensure_launcher(str(tmp_path))
    sid = "记忆测试_v1_0_13100333"
    env = {**os.environ, "ORBITAL_SESSION_ID": sid + '"', "ORBITAL_AGENT": "codex",
           "ORBITAL_PROJECT_ID": "p"}
    out = subprocess.run(
        [os.path.join(str(tmp_path), "orbital", "bin", "bg.cmd"), "echo hi"],
        cwd=str(tmp_path), capture_output=True, text=True, env=env, timeout=60)
    assert out.returncode == 0, out.stderr
    assert "Orbital will send you a message when it finishes" in out.stdout
    job_id = out.stdout.split("Started background job ")[1].split(" ")[0]
    with open(os.path.join(str(tmp_path), "orbital", "jobs", job_id, "job.json"),
              encoding="utf-8") as f:
        record = json.load(f)
    assert record["session_id"] == sid
    assert record["agent"] == "codex"


def test_the_bg_cmd_job_sees_the_callers_path(tmp_path):
    # Batch 1026 Windows pass (D6): the WMI-created job got the account's
    # default environment, so in safe mode the worker's job could not find
    # the per-user python the sandboxed shell had on its PATH.
    job_launcher.ensure_launcher(str(tmp_path))
    tools = tmp_path / "only (on) caller path"
    tools.mkdir()
    (tools / "orbtool.cmd").write_text("@echo orbtool-ran\r\n")
    env = {**os.environ, "ORBITAL_SESSION_ID": SID, "ORBITAL_PROJECT_ID": "p",
           "PATH": str(tools) + os.pathsep + os.environ["PATH"]}
    out = subprocess.run(
        [os.path.join(str(tmp_path), "orbital", "bin", "bg.cmd"), "orbtool"],
        cwd=str(tmp_path), capture_output=True, text=True, env=env, timeout=60)
    assert out.returncode == 0, out.stderr
    job_id = out.stdout.split("Started background job ")[1].split(" ")[0]
    d = os.path.join(str(tmp_path), "orbital", "jobs", job_id)
    exit_path = os.path.join(d, "exit")
    deadline = time.monotonic() + 30
    while not os.path.exists(exit_path) and time.monotonic() < deadline:
        time.sleep(0.2)
    assert "orbtool-ran" in open(os.path.join(d, "output.log"), errors="replace").read()
    assert open(exit_path).read().split()[0] == "0"


def test_the_bg_cmd_launcher_survives_a_bare_exit_and_keeps_bangs(tmp_path):
    # Windows pass 2026-10-27: the command's own `exit` ended the cmd.exe that
    # was meant to record the exit code, its /v:on ate every `!`, and the
    # WMI-created cmd.exe could not start a nested cmd.exe ("Access is denied").
    job_launcher.ensure_launcher(str(tmp_path))
    env = {**os.environ, "ORBITAL_SESSION_ID": SID, "ORBITAL_PROJECT_ID": "p"}
    out = subprocess.run(
        [os.path.join(str(tmp_path), "orbital", "bin", "bg.cmd"),
         'echo bang! & echo a^&b & cmd /d /c echo nested-ok & '
         'python -c "print(\'x y\')" & exit 4'],
        cwd=str(tmp_path), capture_output=True, text=True, env=env, timeout=60)
    assert out.returncode == 0, out.stderr
    job_id = out.stdout.split("Started background job ")[1].split(" ")[0]
    d = os.path.join(str(tmp_path), "orbital", "jobs", job_id)
    exit_path = os.path.join(d, "exit")
    deadline = time.monotonic() + 30
    while not os.path.exists(exit_path) and time.monotonic() < deadline:
        time.sleep(0.2)
    assert open(exit_path).read().split()[0] == "4"
    log = open(os.path.join(d, "output.log"), errors="replace").read()
    assert "bang!" in log
    assert "x y" in log
    assert "a&b" in log                                    # parsed once, by the child
    assert "nested-ok" in log


class _RunningProvider:
    """A provider whose sandbox is set up and which just runs the command."""

    def get_capabilities(self):
        from types import SimpleNamespace
        return SimpleNamespace(setup_complete=True)

    async def run_command(self, *, project_id, command, args, working_dir, timeout_sec):
        from types import SimpleNamespace
        out = subprocess.run([command, *args], cwd=working_dir, capture_output=True,
                             text=True, timeout=timeout_sec)
        return SimpleNamespace(stdout=out.stdout, stderr=out.stderr,
                               exit_code=out.returncode, timed_out=False)


def test_a_sandboxed_background_call_runs_powershell_like_every_other_call(tmp_path):
    # Batch 1026 Windows pass (D6): in a safe-mode project the manager's
    # background command went to the launcher raw and ran under cmd.exe, so
    # `Write-Output hi` failed with "'Write-Output' is not recognized" while
    # the same call worked in an open project and in the foreground.
    import contextvars
    from agent_os.agent import loop as loop_mod
    from agent_os.agent.tools.shell import ShellTool

    reg = _registry()
    tool = ShellTool(workspace=str(tmp_path), os_type="windows", project_id="p",
                     platform_provider=_RunningProvider(), safe_mode_cb=lambda: True,
                     jobs=reg, session_id=SID)
    command = "Write-Output hi; Start-Sleep 1; Write-Output bye"
    ctx = contextvars.copy_context()

    def run():
        loop_mod._tool_batch.set("batch-w")
        return tool.execute(command=command, background=True)

    try:
        result = ctx.run(run)
        job = reg.get(result.meta["background_job"]["job_id"])
        assert job is not None and job.source == "launcher", result.content
        assert job.command == command
        exit_path = os.path.join(job.dir, "exit")
        deadline = time.monotonic() + 30
        while not os.path.exists(exit_path) and time.monotonic() < deadline:
            time.sleep(0.2)
        assert open(exit_path).read().split()[0] == "0"
        log = open(job.log_path, errors="replace").read()
        assert "hi" in log and "bye" in log
    finally:
        reg.stop_all()
