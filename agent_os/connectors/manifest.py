# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Connector manifests + the bundled catalog (Spec 011 §2, §3; Task B1).

A connector is *data*, not code: a manifest describes a remote MCP server, its
auth shape, and how its tools should be classified for the autonomy layer.
``load_catalog()`` reads the JSON files shipped under ``catalog/``; user-added
Tier-0 servers become ``custom-<slug>`` manifests built by ``custom_manifest``.
Validation is plain-python (no jsonschema dependency).
"""

import json
import os
import re
from dataclasses import dataclass, field, fields

# The locked auth-spec enum (Spec 011 §0.5). ``local_native`` (EventKit) and
# ``app_password`` (CalDAV) are reserved for later calendar sources; the launch
# catalog only uses ``oauth2``. ``none`` covers unauthenticated custom servers.
AUTH_TYPES = {"oauth2", "app_password", "local_native", "none"}

# Autonomy classification a manifest may assert per tool. Anything not listed
# here (or absent) falls back to the MCP ``readOnlyHint`` and then fail-closed
# to write — see ``reflection.is_write``.
TOOL_CLASSES = {"read", "write"}

STATUSES = {"available", "pending_verification"}

_DEFAULT_ICON = "🔌"

_CATALOG_DIR = os.path.join(os.path.dirname(__file__), "catalog")


@dataclass(frozen=True)
class ConnectorManifest:
    """A single catalog entry (Spec 011 §2). Fields are locked by the plan."""

    id: str                          # "google-calendar"
    name: str                        # "Google Calendar"
    icon: str                        # emoji or asset key
    auth_provider: str               # "google" — connectors sharing tokens share this
    auth_type: str                   # one of AUTH_TYPES
    server_url: str | None           # remote MCP server URL (None for local_native)
    oauth_scopes: list[str]          # incremental-auth scopes this connector needs
    tool_overrides: dict[str, str]   # tool name -> "read" | "write"
    featured: bool                   # drives the onboarding card
    status: str                      # one of STATUSES

    def to_dict(self) -> dict:
        return {f.name: getattr(self, f.name) for f in fields(self)}


def slugify(name: str) -> str:
    """Lowercase, collapse non-alphanumerics to single hyphens, trim hyphens."""
    s = re.sub(r"[^a-z0-9]+", "-", name.strip().lower())
    return s.strip("-")


def manifest_from_dict(data: dict) -> ConnectorManifest:
    """Build + validate a manifest from a plain dict (bundled JSON or custom).

    Optional fields default; required fields (id, name, auth_provider,
    auth_type, server_url) must be present. Raises ``ValueError`` on any
    unknown auth_type / status / tool-class so a malformed catalog entry fails
    loudly at load time rather than silently mis-classifying a write tool.
    """
    for required in ("id", "name", "auth_provider", "auth_type"):
        if not data.get(required):
            raise ValueError(f"connector manifest missing required field: {required}")

    auth_type = data["auth_type"]
    if auth_type not in AUTH_TYPES:
        raise ValueError(f"unknown auth_type {auth_type!r} (allowed: {sorted(AUTH_TYPES)})")

    status = data.get("status", "available")
    if status not in STATUSES:
        raise ValueError(f"unknown status {status!r} (allowed: {sorted(STATUSES)})")

    server_url = data.get("server_url")
    # local_native connectors legitimately have no remote server; every other
    # type must point at one.
    if server_url is None and auth_type != "local_native":
        raise ValueError(f"connector {data['id']!r} requires a server_url for auth_type {auth_type!r}")

    tool_overrides = data.get("tool_overrides") or {}
    for tool_name, klass in tool_overrides.items():
        if klass not in TOOL_CLASSES:
            raise ValueError(
                f"connector {data['id']!r} tool_overrides[{tool_name!r}] = {klass!r} "
                f"(allowed: {sorted(TOOL_CLASSES)})"
            )

    return ConnectorManifest(
        id=data["id"],
        name=data["name"],
        icon=data.get("icon") or _DEFAULT_ICON,
        auth_provider=data["auth_provider"],
        auth_type=auth_type,
        server_url=server_url,
        oauth_scopes=list(data.get("oauth_scopes") or []),
        tool_overrides=dict(tool_overrides),
        featured=bool(data.get("featured", False)),
        status=status,
    )


def custom_manifest(name: str, url: str, auth_type: str) -> ConnectorManifest:
    """Build a Tier-0 user-added manifest (Spec 011 §3, Tier 0).

    Custom servers are ``custom-<slug>`` with their own per-server auth
    provider (so their tokens never collide with a first-party provider's).
    Only ``none`` and ``oauth2`` are offered in the add-server form.
    """
    if auth_type not in {"none", "oauth2"}:
        raise ValueError(
            f"custom connectors support auth_type 'none' or 'oauth2', not {auth_type!r}"
        )
    slug = slugify(name)
    if not slug:
        raise ValueError("custom connector name must contain at least one alphanumeric character")
    cid = f"custom-{slug}"
    return manifest_from_dict({
        "id": cid,
        "name": name,
        "icon": _DEFAULT_ICON,
        "auth_provider": cid,  # per-server provider; tokens keyed on this
        "auth_type": auth_type,
        "server_url": url,
        "oauth_scopes": [],
        "tool_overrides": {},
        "featured": False,
        "status": "available",
    })


def load_catalog(catalog_dir: str | None = None) -> list[ConnectorManifest]:
    """Load + validate every ``*.json`` under the bundled catalog directory.

    Sorted by id for stable ordering. Raises ``ValueError`` (via
    ``manifest_from_dict``) if any bundled entry is malformed.
    """
    directory = catalog_dir or _CATALOG_DIR
    manifests: list[ConnectorManifest] = []
    if not os.path.isdir(directory):
        return manifests
    for fname in sorted(os.listdir(directory)):
        if not fname.endswith(".json"):
            continue
        with open(os.path.join(directory, fname), "r", encoding="utf-8") as fh:
            manifests.append(manifest_from_dict(json.load(fh)))
    manifests.sort(key=lambda m: m.id)
    return manifests
