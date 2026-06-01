# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Phase 4 / decision D1: the resolver rule + retirement of DEFAULT_SESSION_ID.

- ``DEFAULT_SESSION_ID`` / the literal "default" is no longer a routing value;
  the sentinel is ``None``.
- One shared helper ``resolve_session_id(session_id, *, on_none)``: the common
  rule is non-None passthrough; the ONLY per-caller variation is the None-policy.
- read/affect None → holder (idle/no-op if no holder, never mints).
- inject None → holder if queue draining, else the persistent chat-origin
  session (lazy-mint via the single funnel).
- SubAgentManager None → hard raise (a sub-agent always has a parent).
- make_session_key(pid, None) → (pid, None), NOT (pid, "default").
"""

from __future__ import annotations

import pytest


def test_default_session_id_constant_is_gone():
    import agent_os.daemon_v2.models as models
    assert not hasattr(models, "DEFAULT_SESSION_ID"), \
        "DEFAULT_SESSION_ID must be deleted (sentinel is None)"


def test_make_session_key_passes_none_through_not_default():
    from agent_os.daemon_v2.models import make_session_key
    assert make_session_key("p1", None) == ("p1", None)
    assert make_session_key("p1", "uuid_x") == ("p1", "uuid_x")


def test_resolve_session_id_helper_passthrough_and_on_none():
    from agent_os.daemon_v2.models import resolve_session_id
    # Common rule: a provided id passes through, on_none is NOT consulted.
    sentinel = {"called": False}
    def on_none():
        sentinel["called"] = True
        return "should_not_be_used"
    assert resolve_session_id("uuid_x", on_none=on_none) == "uuid_x"
    assert sentinel["called"] is False
    # None → the caller's policy decides.
    assert resolve_session_id(None, on_none=lambda: "holder_uuid") == "holder_uuid"


def test_subagent_resolver_raises_on_none():
    from agent_os.daemon_v2.sub_agent_manager import SubAgentManager
    with pytest.raises(ValueError):
        SubAgentManager._resolve_session_id(None)
    # A provided id still passes through.
    assert SubAgentManager._resolve_session_id("uuid_x") == "uuid_x"
