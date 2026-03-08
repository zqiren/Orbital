# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Elevated setup helper for Agent OS.

This script is launched via UAC (ShellExecuteW with 'runas').
It creates the sandbox user, sets up the workspace, and writes status to a file.
"""

import json
import logging
import os
import sys

logging.basicConfig(level=logging.INFO, format="%(name)s %(levelname)s: %(message)s")
logger = logging.getLogger("agent_os.platform.windows.setup_helper")


def _write_status(path: str, data: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f)


def main() -> None:
    status_path = os.path.join(
        os.environ.get("LOCALAPPDATA", ""), "AgentOS", "setup_status.json"
    )
    try:
        from agent_os.platform.windows.credentials import CredentialStore
        from agent_os.platform.windows.permissions import PermissionManager
        from agent_os.platform.windows.sandbox import SandboxAccountManager
        from agent_os.platform.windows.setup import SetupOrchestrator

        cs = CredentialStore()
        am = SandboxAccountManager(cs)
        pm = PermissionManager()
        orchestrator = SetupOrchestrator(am, pm)

        result = orchestrator.run_setup()

        _write_status(status_path, {
            "status": "complete",
            "success": result.success,
            "error": result.error,
        })

        if not result.success:
            logger.error("Setup failed: %s", result.error)
            sys.exit(1)
        else:
            logger.info("Setup completed successfully")

    except Exception as exc:
        logger.error("Setup raised exception: %s", exc)
        _write_status(status_path, {
            "status": "complete",
            "success": False,
            "error": str(exc),
        })
        sys.exit(1)


if __name__ == "__main__":
    main()
