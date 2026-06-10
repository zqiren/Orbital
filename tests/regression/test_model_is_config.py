# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Model-is-config amendment (AMENDS piece 2 / TASK-resume-persistence).

Wire-verified verdict (INVESTIGATION-resume-semantics): both providers apply
a passed model on resume and serve the session's last-used model when the
param is omitted. The persisted record's model is therefore display/debug
metadata — NEVER consulted to drive resume. The rejected-model trap (an
errored thread frozen to a 400'd model that overrides the user's fix) dies
by construction.

Also locked here: the caveat-4 landmine — thread/resume applies ITS OWN
DEFAULTS for omitted params (observed: approvalPolicy drifted never ->
on-request). Governance params must stay explicitly re-passed on resume;
only the MODEL param becomes conditional.
"""

import pytest

from agent_os.agent.prompt_builder import Autonomy
from agent_os.agent.transports.codex_transport import CodexTransport


class TestCodexResumeModelSource:
    def test_record_model_is_never_consulted(self):
        t = CodexTransport(resume_record={"session_id": "T9",
                                          "model": "gpt-5.3-codex-REJECTED"})
        assert t._model is None, \
            "the record's model must not drive resume (model-is-config)"
        assert t._resume_session_id == "T9"  # identity untouched

    def test_argv_override_wins_on_resume(self):
        t = CodexTransport(autonomy=Autonomy.HANDS_OFF,
                           resume_record={"session_id": "T9",
                                          "model": "gpt-5.3-codex-REJECTED"})
        t._model = t._argv_model(["-m", "gpt-5.4-mini"])
        method, params = t._thread_open_request("/tmp/ws")
        assert method == "thread/resume"
        assert params["model"] == "gpt-5.4-mini", \
            "current override heals the existing thread"

    def test_no_override_omits_model_param(self):
        # Omitted -> the provider serves the thread's last-used model
        # (observed on both transports). Forcing a frozen value here was
        # the entire mechanism of the rejected-model trap.
        t = CodexTransport(autonomy=Autonomy.HANDS_OFF,
                           resume_record={"session_id": "T9",
                                          "model": "gpt-5.3-codex-REJECTED"})
        t._model = t._argv_model([])
        method, params = t._thread_open_request("/tmp/ws")
        assert method == "thread/resume"
        assert "model" not in params

    def test_fresh_start_unchanged_argv_applies(self):
        t = CodexTransport(autonomy=Autonomy.HANDS_OFF)
        t._model = t._argv_model(["-m", "gpt-5.4-mini"])
        method, params = t._thread_open_request("/tmp/ws")
        assert method == "thread/start"
        assert params["model"] == "gpt-5.4-mini"


class TestGovernanceParamsAlwaysRepassed:
    """Caveat-4 landmine guard: omitting governance params on resume lets
    the provider silently LOOSEN the approval policy (observed drift:
    never -> on-request). This test fails if a future change omits them."""

    @pytest.mark.parametrize("resume", [False, True])
    def test_cwd_approval_sandbox_present_in_both_branches(self, resume):
        record = {"session_id": "T9"} if resume else None
        t = CodexTransport(autonomy=Autonomy.HANDS_OFF, resume_record=record)
        t._model = None
        method, params = t._thread_open_request("/tmp/ws")
        assert method == ("thread/resume" if resume else "thread/start")
        assert params["cwd"] == "/tmp/ws"
        assert params["approvalPolicy"] == "never"
        assert params["sandbox"] == "workspace-write"
        if resume:
            assert params["threadId"] == "T9"


class TestSdkResumeModelSource:
    def _manager(self):
        from unittest.mock import MagicMock
        from agent_os.daemon_v2.sub_agent_manager import SubAgentManager
        return SubAgentManager(
            process_manager=MagicMock(), registry=MagicMock(),
            setup_engine=MagicMock(), project_store=MagicMock(),
        )

    def _sdk_manifest(self):
        from unittest.mock import MagicMock
        manifest = MagicMock()
        manifest.runtime.transport = "sdk"
        manifest.runtime.mode = "pipe"
        manifest.runtime.command = "claude"
        return manifest

    def test_resume_model_comes_from_current_config_not_record(self):
        mgr = self._manager()
        transport = mgr._resolve_transport(
            self._sdk_manifest(),
            {"args": ["--output-format", "stream-json", "--model", "sonnet"]},
            autonomy=Autonomy.HANDS_OFF,
            resume_record={"session_id": "S1", "model": "claude-haiku-4-5-20251001"},
        )
        assert transport._resume_session_id == "S1"
        assert transport._model == "sonnet", \
            "current config override must win over the record's model"

    def test_resume_without_override_omits_model(self):
        mgr = self._manager()
        transport = mgr._resolve_transport(
            self._sdk_manifest(), {"args": []},
            autonomy=Autonomy.HANDS_OFF,
            resume_record={"session_id": "S1", "model": "claude-haiku-4-5-20251001"},
        )
        assert transport._model is None, \
            "no override -> omit; the CLI serves the session's last-used model"


class TestStartupResolutionScope:
    """Resolution fills ONLY the unset fresh-start case. Override always
    wins (resolver must not run); resume keeps omit-when-no-override (the
    amendment — provider serves the thread's own last-used model)."""

    @pytest.mark.asyncio
    async def test_override_wins_resolver_never_runs(self, monkeypatch):
        t = CodexTransport(autonomy=Autonomy.HANDS_OFF)
        async def sentinel():
            raise AssertionError("resolver must not run when override is set")
        monkeypatch.setattr(t, "_resolve_startup_model", sentinel)
        t._model = t._argv_model(["-m", "gpt-5.4-mini"])
        method, params = t._thread_open_request("/tmp/ws")
        assert params["model"] == "gpt-5.4-mini"

    @pytest.mark.asyncio
    async def test_resume_without_override_still_omits_model(self, monkeypatch):
        t = CodexTransport(autonomy=Autonomy.HANDS_OFF,
                           resume_record={"session_id": "T9"})
        async def sentinel():
            raise AssertionError("resolver must not run on resume")
        monkeypatch.setattr(t, "_resolve_startup_model", sentinel)
        t._model = t._argv_model([])
        method, params = t._thread_open_request("/tmp/ws")
        assert method == "thread/resume"
        assert "model" not in params
