# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Phase 4 / decision D2: new_session mints the UUID only.

The F1 (`sess_…`) mint is retired — `session_id` returned IS the `session_uuid`
(the canonical routing identity). No vestigial F1 field. Both callers (the
queue dispatcher and the /new-session route) adopt the uuid transparently
because they read ``result["session_id"]``.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from agent_os.daemon_v2.agent_manager import AgentManager


def _mgr():
    mgr = AgentManager(
        project_store=MagicMock(), ws_manager=MagicMock(),
        sub_agent_manager=MagicMock(), activity_translator=MagicMock(),
        process_manager=MagicMock(), platform_provider=None,
        registry=MagicMock(), setup_engine=MagicMock(),
        settings_store=None, credential_store=None,
    )
    mgr._project_store.get_project.return_value = {"workspace": "/tmp/x", "name": "My Project"}
    return mgr


@pytest.mark.asyncio
async def test_new_session_returns_uuid_only():
    mgr = _mgr()
    result = await mgr.new_session("proj_1")
    assert result["status"] == "ok"
    # session_id IS the uuid — no separate F1.
    assert result["session_id"] == result["session_uuid"], (
        f"D2: session_id must equal session_uuid (uuid-only), got "
        f"session_id={result['session_id']!r} session_uuid={result['session_uuid']!r}"
    )
    # The retired F1 scheme used a 'sess_' prefix — it must be gone.
    assert not result["session_id"].startswith("sess_"), (
        "D2: the 'sess_' F1 mint is retired"
    )
    # uuid stem is the sanitized project name + hex (the JSONL filename form).
    assert result["session_uuid"].startswith("my_project_")
