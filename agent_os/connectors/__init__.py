# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Connector subsystem (Spec 011): MCP-client catalog, OAuth, tool reflection.

Orbital speaks MCP as a *client*; every service with a remote MCP server
becomes catalog configuration (a manifest), not integration code. This package
owns the manifest schema + bundled catalog (``manifest``), the connector
lifecycle + OAuth + cached MCP sessions (``manager``, ``oauth``, ``mcp_client``),
and the reflection of remote tools into the agent's ``ToolRegistry``
(``reflection``).
"""

from .manifest import ConnectorManifest, custom_manifest, load_catalog, manifest_from_dict
from .manager import AuthStart, Connected, ConnectorError, ConnectorManager
from .reflection import build_connector_tools, classify_is_write, register_connector_tools

__all__ = [
    "ConnectorManifest",
    "load_catalog",
    "manifest_from_dict",
    "custom_manifest",
    "ConnectorManager",
    "ConnectorError",
    "Connected",
    "AuthStart",
    "register_connector_tools",
    "build_connector_tools",
    "classify_is_write",
]
