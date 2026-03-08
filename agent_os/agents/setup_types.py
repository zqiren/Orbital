# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Setup status types for agent installation and configuration checks."""

from dataclasses import dataclass, field


@dataclass
class SetupAction:
    """A single action needed to get an agent ready."""
    action: str         # "install_dependency" | "install_agent" | "configure_credential" | "none"
    label: str          # human-readable: "Install Node.js 18+"
    command: str | None  # suggested command, or None for credential prompts
    blocking: bool      # can't proceed without this


@dataclass
class AgentSetupStatus:
    """Full setup status for a single agent."""
    slug: str
    name: str
    installed: bool
    binary_path: str | None        # resolved absolute path to binary
    version: str | None            # detected version string
    dependencies_met: bool
    missing_dependencies: list[str]
    credentials_configured: bool
    missing_credentials: list[str]
    setup_actions: list[SetupAction] = field(default_factory=list)
