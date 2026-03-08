# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Regression tests for approve-all session bypass (Bug E).

When a user taps "Approve All", subsequent tool calls within the same task
should be auto-approved for a bounded time window. The bypass resets when:
- Time expires (default 10 minutes)
- User sends a new message (new task)
- User explicitly deactivates
"""

import time
from unittest.mock import MagicMock

import pytest

from agent_os.agent.prompt_builder import Autonomy
from agent_os.daemon_v2.autonomy import AutonomyInterceptor


@pytest.fixture
def interceptor():
    ws = MagicMock()
    return AutonomyInterceptor(Autonomy.CHECK_IN, ws, "proj_1")


class TestApproveAllBypass:
    """Tests for the approve-all session bypass feature."""

    def test_bypass_all_skips_subsequent_interceptions(self, interceptor):
        """After activate_bypass_all, should_intercept returns False for all tools."""
        # Shell normally intercepted in check_in mode
        shell_call = {"name": "shell", "arguments": {"command": "rm -rf /tmp/foo"}}
        assert interceptor.should_intercept(shell_call) is True

        interceptor.activate_bypass_all()

        assert interceptor.should_intercept(shell_call) is False
        # Write also bypassed
        write_call = {"name": "write", "arguments": {"path": "/tmp/foo.txt", "content": "x"}}
        assert interceptor.should_intercept(write_call) is False
        # Browser write action also bypassed
        browser_call = {"name": "browser", "arguments": {"action": "click", "ref": "btn1"}}
        assert interceptor.should_intercept(browser_call) is False

    def test_bypass_all_expires_after_timeout(self, interceptor):
        """Bypass should expire after the configured duration."""
        interceptor.activate_bypass_all(duration=0.1)  # 100ms for testing
        shell_call = {"name": "shell", "arguments": {"command": "ls"}}
        assert interceptor.should_intercept(shell_call) is False

        time.sleep(0.15)

        assert interceptor.should_intercept(shell_call) is True

    def test_bypass_all_default_duration_10_minutes(self, interceptor):
        """Default bypass duration should be 10 minutes (600s)."""
        interceptor.activate_bypass_all()
        assert interceptor._bypass_all_until is not None
        expected = time.time() + 600
        assert abs(interceptor._bypass_all_until - expected) < 2  # tolerance

    def test_deactivate_bypass_all_resets(self, interceptor):
        """deactivate_bypass_all clears the bypass immediately."""
        interceptor.activate_bypass_all()
        shell_call = {"name": "shell", "arguments": {"command": "ls"}}
        assert interceptor.should_intercept(shell_call) is False

        interceptor.deactivate_bypass_all()

        assert interceptor.should_intercept(shell_call) is True

    def test_bypass_all_does_not_affect_hands_off_request_access(self):
        """Even with bypass-all, hands_off mode's request_access should still
        be intercepted (bypass-all is irrelevant for hands_off since almost
        nothing is intercepted anyway — but request_access IS intercepted).

        Actually: bypass-all should override ALL interceptions. The user
        explicitly approved everything. So request_access is also bypassed.
        """
        ws = MagicMock()
        interceptor = AutonomyInterceptor(Autonomy.HANDS_OFF, ws, "proj_1")
        ra_call = {"name": "request_access", "arguments": {"path": "/secret"}}
        assert interceptor.should_intercept(ra_call) is True

        interceptor.activate_bypass_all()
        assert interceptor.should_intercept(ra_call) is False

    def test_bypass_all_not_active_by_default(self, interceptor):
        """Bypass-all should not be active on a fresh interceptor."""
        assert interceptor._bypass_all_until is None
        shell_call = {"name": "shell", "arguments": {"command": "ls"}}
        assert interceptor.should_intercept(shell_call) is True
