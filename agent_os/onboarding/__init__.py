# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""First-run onboarding: read-only discovery of importable projects.

The only public surface is :func:`scan_importable_projects` and the
:class:`ImportCandidate` dataclass from :mod:`agent_os.onboarding.import_scanner`.
"""

from agent_os.onboarding.import_scanner import (
    ImportCandidate,
    scan_importable_projects,
)

__all__ = ["ImportCandidate", "scan_importable_projects"]
