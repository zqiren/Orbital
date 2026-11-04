# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""A launcher job that arrives without a session (the worker's command ran
without the ORBITAL_* env, e.g. a Codex whose shell_environment_policy kept
them out) goes to the project's one worker that is mid-turn and uses the
launcher. With none or several, it stays untracked rather than guessed.
Platform-neutral: no process is started."""

import json
import os

from agent_os.daemon_v2.jobs import JobRegistry


def _job_dir(workspace, job_id="job_0001", **fields):
    d = os.path.join(str(workspace), "orbital", "jobs", job_id)
    os.makedirs(d)
    record = {"id": job_id, "project_id": "p", "session_id": "", "agent": "",
              "pid": None, "started_at": 1.0, "source": "launcher", "state": "running"}
    record.update(fields)
    with open(os.path.join(d, "job.json"), "w", encoding="utf-8") as f:
        json.dump(record, f)
    with open(os.path.join(d, "cmd"), "w", encoding="utf-8") as f:
        f.write("echo hi\n")
    return d


def _registry(sole):
    async def inject(*a, **k):
        return "delivered"
    return JobRegistry(inject=inject, broadcast=lambda *a: None,
                       sole_launcher_worker=sole)


def test_an_unattributed_job_goes_to_the_sole_running_launcher_worker(tmp_path):
    reg = _registry(lambda pid: ("后台任务_b7aacb56", "codex") if pid == "p" else None)
    job = reg.adopt(_job_dir(tmp_path), project_id="p")
    assert job.session_id == "后台任务_b7aacb56"
    assert job.agent == "codex"


def test_no_sole_worker_leaves_the_job_untracked(tmp_path):
    reg = _registry(lambda pid: None)
    job = reg.adopt(_job_dir(tmp_path), project_id="p")
    assert job.session_id == ""


def test_an_attributed_job_is_never_reassigned(tmp_path):
    reg = _registry(lambda pid: ("other", "pi"))
    job = reg.adopt(_job_dir(tmp_path, session_id="s1", agent="codex"), project_id="p")
    assert (job.session_id, job.agent) == ("s1", "codex")
