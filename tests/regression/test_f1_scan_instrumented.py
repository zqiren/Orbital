# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Phase 4 / decision D4: the legacy F1-scan fallback in
``_resolve_session_uuid_on_disk`` is KEPT (back-compat for any persisted F1 id)
but INSTRUMENTED — it logs when it fires, so we can confirm the path goes cold
before deleting it in a later pass. A direct uuid (filename) match must NOT log.
"""

from __future__ import annotations

import json
import logging
import os
from unittest.mock import MagicMock

from agent_os.agent.project_paths import ProjectPaths
from agent_os.daemon_v2.agent_manager import AgentManager


def _mgr(workspace):
    mgr = AgentManager(
        project_store=MagicMock(), ws_manager=MagicMock(),
        sub_agent_manager=MagicMock(), activity_translator=MagicMock(),
        process_manager=MagicMock(), platform_provider=None,
        registry=MagicMock(), setup_engine=MagicMock(),
        settings_store=None, credential_store=None,
    )
    mgr._project_store.get_project.return_value = {"workspace": str(workspace), "name": "P"}
    return mgr


def _seed(workspace, uuid, f1):
    sdir = ProjectPaths(str(workspace)).sessions_dir
    os.makedirs(sdir, exist_ok=True)
    row = {"role": "meta", "event": "session_start", "session_id": f1,
           "session_uuid": uuid, "timestamp": "2026-05-26T00:00:00+00:00"}
    with open(os.path.join(sdir, f"{uuid}.jsonl"), "w", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")


def test_f1_scan_logs_when_it_fires(tmp_path, caplog):
    ws = tmp_path / "ws"
    _seed(ws, uuid="proj_aaaa1111", f1="sess_oldf1")
    mgr = _mgr(ws)
    with caplog.at_level(logging.INFO, logger="agent_os.daemon_v2.agent_manager"):
        resolved = mgr._resolve_session_uuid_on_disk("p1", "sess_oldf1")  # addressed by F1 -> scan
    assert resolved is not None and resolved[0] == "proj_aaaa1111"
    assert any("f1" in r.getMessage().lower() and "scan" in r.getMessage().lower()
               for r in caplog.records), \
        "F1-scan fallback must emit a log line when it resolves via the scan branch"


def test_direct_uuid_match_does_not_log_scan(tmp_path, caplog):
    ws = tmp_path / "ws"
    _seed(ws, uuid="proj_bbbb2222", f1="sess_oldf1")
    mgr = _mgr(ws)
    with caplog.at_level(logging.INFO, logger="agent_os.daemon_v2.agent_manager"):
        resolved = mgr._resolve_session_uuid_on_disk("p1", "proj_bbbb2222")  # direct filename match
    assert resolved is not None and resolved[0] == "proj_bbbb2222"
    assert not any("scan" in r.getMessage().lower() for r in caplog.records), \
        "Direct uuid match must NOT trigger the F1-scan log"
