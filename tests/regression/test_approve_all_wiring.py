# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Regression: approve_all field wiring.

Root cause: Backend accepted approve_all field on POST /approve endpoint
to activate a 10-minute bypass window, but frontend never sent it.

Fix: Added approve_all parameter to useAgent.approveToolCall() and
"Approve All" button to ApprovalCard component.
"""

from unittest.mock import MagicMock
import time

from agent_os.daemon_v2.autonomy import AutonomyInterceptor
from agent_os.agent.prompt_builder import Autonomy


class TestApproveAllBypass:
    """Verify approve_all activates the bypass window."""

    def test_activate_bypass_all(self):
        """activate_bypass_all() sets a future timestamp."""
        ws = MagicMock()
        interceptor = AutonomyInterceptor(Autonomy.CHECK_IN, ws, "proj_1")
        assert interceptor._bypass_all_until is None

        interceptor.activate_bypass_all(duration=600)
        assert interceptor._bypass_all_until is not None
        assert interceptor._bypass_all_until > time.time()

    def test_bypass_all_skips_interception(self):
        """During bypass window, should_intercept returns False."""
        ws = MagicMock()
        interceptor = AutonomyInterceptor(Autonomy.CHECK_IN, ws, "proj_1")
        interceptor.activate_bypass_all(duration=600)

        # shell would normally be intercepted in CHECK_IN
        result = interceptor.should_intercept({
            "name": "shell", "arguments": {"command": "ls"},
        })
        assert result is False

    def test_bypass_all_expires(self):
        """After bypass duration, interception resumes."""
        ws = MagicMock()
        interceptor = AutonomyInterceptor(Autonomy.CHECK_IN, ws, "proj_1")
        # Set bypass to already-expired time
        interceptor._bypass_all_until = time.time() - 1

        result = interceptor.should_intercept({
            "name": "shell", "arguments": {"command": "ls"},
        })
        assert result is True

    def test_deactivate_bypass_all(self):
        """deactivate_bypass_all() clears the window."""
        ws = MagicMock()
        interceptor = AutonomyInterceptor(Autonomy.CHECK_IN, ws, "proj_1")
        interceptor.activate_bypass_all(duration=600)
        interceptor.deactivate_bypass_all()
        assert interceptor._bypass_all_until is None

        result = interceptor.should_intercept({
            "name": "shell", "arguments": {"command": "ls"},
        })
        assert result is True
