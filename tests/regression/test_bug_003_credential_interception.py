# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Regression tests for BUG-003a: request_credential must be intercepted.

Verifies that request_credential is always intercepted regardless of
autonomy preset, and is not bypassed by approve-all.
"""

import time
from unittest.mock import MagicMock

import pytest

from agent_os.agent.prompt_builder import Autonomy
from agent_os.daemon_v2.autonomy import AutonomyInterceptor


def _make_interceptor(preset: Autonomy) -> AutonomyInterceptor:
    ws = MagicMock()
    return AutonomyInterceptor(
        preset=preset,
        ws_manager=ws,
        project_id="test-project",
    )


def _credential_tool_call():
    return {
        "id": "tc-1",
        "name": "request_credential",
        "arguments": {
            "name": "github",
            "domain": "github.com",
            "fields": ["username", "password"],
            "reason": "Need to access repository",
        },
    }


class TestCredentialInterception:

    def test_request_credential_intercepted_in_hands_off(self):
        """Credentials must be intercepted even in hands-off mode."""
        interceptor = _make_interceptor(Autonomy.HANDS_OFF)
        assert interceptor.should_intercept(_credential_tool_call()) is True

    def test_request_credential_intercepted_in_check_in(self):
        """Credentials must be intercepted in check-in mode."""
        interceptor = _make_interceptor(Autonomy.CHECK_IN)
        assert interceptor.should_intercept(_credential_tool_call()) is True

    def test_request_credential_intercepted_in_supervised(self):
        """Credentials must be intercepted in supervised mode."""
        interceptor = _make_interceptor(Autonomy.SUPERVISED)
        assert interceptor.should_intercept(_credential_tool_call()) is True

    def test_request_credential_not_bypassed_by_approve_all(self):
        """approve-all bypass must NOT skip credential requests."""
        interceptor = _make_interceptor(Autonomy.HANDS_OFF)
        interceptor.activate_bypass_all(duration=600)
        assert interceptor.should_intercept(_credential_tool_call()) is True

    def test_request_credential_not_bypassed_by_per_action(self):
        """Per-action bypass window must NOT skip credential requests."""
        interceptor = _make_interceptor(Autonomy.HANDS_OFF)
        tc = _credential_tool_call()
        # Record a recent approval for the same tool+args
        interceptor.record_approval(tc["name"], tc["arguments"])
        assert interceptor.should_intercept(tc) is True

    def test_other_tools_still_work_in_hands_off(self):
        """Hands-off should still only intercept request_access (not shell etc)."""
        interceptor = _make_interceptor(Autonomy.HANDS_OFF)
        shell_call = {"name": "shell", "arguments": {"command": "ls"}}
        assert interceptor.should_intercept(shell_call) is False

        access_call = {"name": "request_access", "arguments": {}}
        assert interceptor.should_intercept(access_call) is True
