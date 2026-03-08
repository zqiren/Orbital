# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Regression: AutonomyInterceptor must accept user_credential_store kwarg.

Commit 45b6e92 updated AgentManager.start_agent() to pass
user_credential_store= to AutonomyInterceptor(), but never updated
AutonomyInterceptor.__init__() to accept it.  This caused a TypeError
on every message inject that auto-starts an agent — 100% failure rate.
"""

from agent_os.agent.prompt_builder import Autonomy
from agent_os.daemon_v2.autonomy import AutonomyInterceptor


class FakeWsManager:
    def broadcast(self, project_id, payload):
        pass


def test_autonomy_interceptor_accepts_user_credential_store_kwarg():
    """AutonomyInterceptor(..., user_credential_store=X) must not raise TypeError."""
    interceptor = AutonomyInterceptor(
        Autonomy.CHECK_IN,
        FakeWsManager(),
        "proj_test",
        user_credential_store=None,
    )
    assert interceptor is not None


def test_autonomy_interceptor_works_without_credential_store():
    """Existing callers that omit user_credential_store must still work."""
    interceptor = AutonomyInterceptor(
        Autonomy.CHECK_IN,
        FakeWsManager(),
        "proj_test",
    )
    assert interceptor is not None
