# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""REST endpoints for first-run onboarding import discovery (backlog #34).

Surfaces the read-only :mod:`agent_os.onboarding.import_scanner` to the setup
wizard: a single ``GET`` that returns ranked, deduplicated, path-verified
candidate projects discovered from other CLI agents (Claude Code, Codex) and
Obsidian vaults. It reads only metadata (paths, cwd, names, timestamps, counts)
— never transcript bodies — and creates nothing. Confirmed candidates are
created by the caller through the existing ``POST /api/v2/projects`` flow.
"""

import logging

from fastapi import APIRouter

from agent_os.onboarding import scan_importable_projects

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v2/onboarding")


def configure() -> None:
    """No dependencies to inject — the scanner is stateless and read-only.

    Present for parity with the other route modules' app-factory wiring.
    """
    return None


@router.get("/importable-projects")
async def importable_projects():
    """Ranked, deduped, path-verified candidates for link-only import.

    Each candidate is ``{source, name, path, session_count, last_activity}``.
    The scan is best-effort and never raises: an unreadable or missing source
    contributes nothing rather than failing the whole response.
    """
    try:
        candidates = scan_importable_projects()
    except Exception:
        logger.exception("importable-projects scan failed")
        candidates = []
    return {"candidates": [c.to_dict() for c in candidates]}
