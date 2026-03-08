# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Agent registry — loads, stores, and queries agent manifests."""

import logging
import os

from agent_os.agents.manifest import AgentManifest, ManifestError, ManifestLoader

logger = logging.getLogger(__name__)


class AgentRegistry:
    """In-memory registry of agent manifests, keyed by slug."""

    def __init__(self) -> None:
        self._manifests: dict[str, AgentManifest] = {}

    def load_directory(self, path: str) -> None:
        """Load all .yaml files from *path*. Skip invalid with a warning."""
        if not os.path.isdir(path):
            logger.warning("Manifest directory does not exist: %s", path)
            return

        for filename in sorted(os.listdir(path)):
            if not filename.endswith(".yaml"):
                continue
            filepath = os.path.join(path, filename)
            try:
                manifest = ManifestLoader.load(filepath)
                self.register(manifest)
            except ManifestError as exc:
                logger.warning("Skipping invalid manifest %s: %s", filepath, exc)

    def register(self, manifest: AgentManifest) -> None:
        """Register a manifest. Overwrites if same slug already exists."""
        self._manifests[manifest.slug] = manifest

    def get(self, slug: str) -> AgentManifest | None:
        """Get manifest by slug, or None if not found."""
        return self._manifests.get(slug)

    def list_all(self) -> list[AgentManifest]:
        """Return all registered manifests."""
        return list(self._manifests.values())

    def list_by_adapter(self, adapter: str) -> list[AgentManifest]:
        """Return manifests whose runtime adapter matches *adapter*."""
        return [m for m in self._manifests.values() if m.runtime.adapter == adapter]

    def get_for_routing(self, skill: str) -> list[AgentManifest]:
        """Find agents whose capabilities.skills include *skill*."""
        return [
            m for m in self._manifests.values()
            if skill in m.capabilities.skills
        ]
