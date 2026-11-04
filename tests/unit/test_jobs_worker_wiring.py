# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Spec 134 Part B (d): Codex and Pi workers get background jobs through the
workspace launcher. Orbital sets ``ORBITAL_SESSION_ID`` / project / agent in
the worker's env at spawn, writes ``orbital/bin/bg``, watches the workspace,
and tells the worker about the launcher once (Pi: its system prompt; Codex,
which takes no system prompt: its first dispatch after a spawn)."""

import os
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent_os.agent.sub_agent_prompt import render_sub_agent_prompt
from agent_os.agent.transports.pi_rpc_transport import _secrets_from_env
from agent_os.daemon_v2 import job_launcher
from agent_os.daemon_v2.sub_agent_manager import SubAgentManager


def _manifest(transport):
    return SimpleNamespace(runtime=SimpleNamespace(transport=transport))


@pytest.fixture
def manager():
    mgr = SubAgentManager.__new__(SubAgentManager)
    mgr._job_registry = MagicMock()
    return mgr


@pytest.mark.parametrize("transport", ["codex-appserver", "pi-rpc"])
def test_codex_and_pi_spawns_get_the_env_the_launcher_and_a_watch(manager, tmp_path, transport):
    env = {"OPENAI_API_KEY": "sk-x"}
    assert manager._prepare_worker_jobs(_manifest(transport), "proj_1", "sess-1",
                                        "codex", str(tmp_path), env) is True
    assert env["ORBITAL_PROJECT_ID"] == "proj_1"
    assert env["ORBITAL_SESSION_ID"] == "sess-1"
    assert env["ORBITAL_AGENT"] == "codex"
    assert env["OPENAI_API_KEY"] == "sk-x"
    assert os.access(os.path.join(str(tmp_path), "orbital", "bin", "bg"), os.X_OK)
    manager._job_registry.watch.assert_called_once_with(str(tmp_path), "proj_1")


@pytest.mark.parametrize("transport", ["sdk", "acp-sdk", "pty", "auto"])
def test_claude_code_and_others_are_left_alone(manager, tmp_path, transport):
    env = {}
    assert manager._prepare_worker_jobs(_manifest(transport), "p", "s", "claude-code",
                                        str(tmp_path), env) is False
    assert env == {}
    assert not os.path.exists(os.path.join(str(tmp_path), "orbital", "bin"))


def test_the_worker_instructions_name_the_launcher_per_os():
    posix = job_launcher.worker_instructions("macos")
    assert 'orbital/bin/bg "<command>"' in posix and "end your turn" in posix.lower()
    assert 'orbital\\bin\\bg.cmd "<command>"' in job_launcher.worker_instructions("windows")


def test_windows_codex_is_told_to_run_the_launcher_escalated():
    # Codex's Windows sandbox denies the WMI process create bg.ps1 needs
    # (Invoke-CimMethod: access denied, 0x80041003); a child it starts is
    # killed with the sandboxed command. Workers run hands-off, so the
    # escalation is approved without a prompt.
    hint = job_launcher.worker_instructions("windows", codex=True)
    assert "escalat" in hint.lower() and "sandbox" in hint.lower()
    for quiet in (job_launcher.worker_instructions("windows"),
                  job_launcher.worker_instructions("macos", codex=True)):
        assert "escalat" not in quiet.lower()


def test_sole_launcher_worker_is_the_one_codex_or_pi_mid_turn(manager, tmp_path):
    def adapter(idle):
        return SimpleNamespace(is_alive=lambda: True, is_idle=lambda: idle)
    manager._adapters = {}
    manager._launcher_workers = set()
    manager._resolve_session_id = lambda sid: sid
    manager._adapter_status = lambda a, *rest: "idle" if a.is_idle() else "running"
    env = {}
    manager._prepare_worker_jobs(_manifest("codex-appserver"), "p", "s1", "codex",
                                 str(tmp_path), env)
    manager._adapters[("p", "s1")] = {"codex": adapter(idle=False)}
    manager._adapters[("p", "s2")] = {"claude-code": adapter(idle=False)}
    assert manager.sole_launcher_worker("p") == ("s1", "codex")
    manager._adapters[("p", "s1")] = {"codex": adapter(idle=True)}
    assert manager.sole_launcher_worker("p") is None
    manager._prepare_worker_jobs(_manifest("pi-rpc"), "p", "s3", "pi", str(tmp_path), env)
    manager._adapters[("p", "s1")] = {"codex": adapter(idle=False)}
    manager._adapters[("p", "s3")] = {"pi": adapter(idle=False)}
    assert manager.sole_launcher_worker("p") is None  # two candidates: no guess


def test_pi_gets_the_paragraph_in_its_system_prompt(tmp_path):
    hint = job_launcher.worker_instructions("macos")
    with_hint = render_sub_agent_prompt(str(tmp_path), None, "pi", ["pi"], background_jobs=hint)
    without = render_sub_agent_prompt(str(tmp_path), None, "pi", ["pi"])
    assert "orbital/bin/bg" in with_hint
    assert "orbital/bin/bg" not in without


async def test_codex_gets_the_paragraph_on_its_first_dispatch_only(tmp_path):
    mgr = SubAgentManager.__new__(SubAgentManager)
    mgr._lifecycle_observer = None
    mgr._process_manager = MagicMock()
    mgr._build_recap = AsyncMock(return_value="RECAP\n")
    mgr._dispatch_async = AsyncMock()
    adapter = SimpleNamespace(_orbital_jobs_hint="JOBS HINT")
    prompt = SimpleNamespace(message="do the thing", dispatch_id="d1",
                             transcript_path="t", initiator="management_agent")
    await mgr._dispatch_prompt_locked(adapter, prompt, "p", "s", "codex")
    await mgr._dispatch_prompt_locked(adapter, prompt, "p", "s", "codex")
    first = mgr._dispatch_async.await_args_list[0].args[3]
    second = mgr._dispatch_async.await_args_list[1].args[3]
    assert first == "JOBS HINT\n\nRECAP\ndo the thing"
    assert second == "RECAP\ndo the thing"


def test_pi_never_redacts_orbitals_own_env_values():
    secrets = _secrets_from_env({"ANTHROPIC_API_KEY": "sk-ant-1234567890",
                                 "ORBITAL_SESSION_ID": "sess-0123456789",
                                 "ORBITAL_AGENT": "pi-agent-x",
                                 "SHORT": "abc"})
    assert secrets == ["sk-ant-1234567890"]
