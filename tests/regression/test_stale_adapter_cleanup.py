# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""`list_active` lazily evicts dead adapters from `_adapters`.

Per REPORT-is-idle-and-adapter-lifecycle.md Q6: an adapter is removed only by
`stop()`, so a sub-agent process that exits on its own leaves a stale entry in
`_adapters` forever (filtered from results by `is_alive()` but never popped).
`list_active` now pops `is_alive()==False` adapters as it scans.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from agent_os.daemon_v2.models import make_session_key
from agent_os.daemon_v2.sub_agent_manager import SubAgentManager


def _manager() -> SubAgentManager:
    return SubAgentManager(process_manager=MagicMock())


def _adapter(alive: bool, idle: bool = True) -> MagicMock:
    a = MagicMock()
    a.is_alive.return_value = alive
    a.is_idle.return_value = idle
    a.display_name = "codex"
    return a


def test_dead_adapter_cleaned_from_adapters():
    mgr = _manager()
    sk = make_session_key("proj", "default")
    mgr._adapters[sk] = {"codex": _adapter(alive=False)}

    result = mgr.list_active("proj", session_id="default")

    assert result == []  # dead adapter not reported
    # The stale entry is popped; the now-empty SessionKey bucket is dropped too.
    assert "codex" not in mgr._adapters.get(sk, {})
    assert sk not in mgr._adapters


def test_alive_adapter_preserved():
    mgr = _manager()
    sk = make_session_key("proj", "default")
    mgr._adapters[sk] = {"codex": _adapter(alive=True, idle=True)}

    result = mgr.list_active("proj", session_id="default")

    assert len(result) == 1
    assert result[0]["status"] == "idle"
    assert mgr._adapters[sk]["codex"] is not None  # still registered
