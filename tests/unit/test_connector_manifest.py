# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Tests for the connector manifest schema + bundled catalog (Task B1)."""

import dataclasses

import pytest

from agent_os.connectors import (
    ConnectorManifest,
    load_catalog,
)
from agent_os.connectors.manifest import (
    AUTH_TYPES,
    TOOL_CLASSES,
    custom_manifest,
    manifest_from_dict,
    slugify,
)


def _by_id(catalog):
    return {m.id: m for m in catalog}


def test_manifest_is_frozen_dataclass():
    m = ConnectorManifest(
        id="x",
        name="X",
        icon="🔌",
        auth_provider="none",
        auth_type="none",
        server_url="https://example.com/mcp",
        oauth_scopes=[],
        tool_overrides={},
        featured=False,
        status="available",
    )
    assert dataclasses.is_dataclass(m)
    with pytest.raises(dataclasses.FrozenInstanceError):
        m.id = "y"  # type: ignore[misc]


def test_load_catalog_has_the_three_launch_connectors():
    catalog = load_catalog()
    ids = {m.id for m in catalog}
    assert {"google-calendar", "google-drive", "gmail"} <= ids
    # Every entry is a ConnectorManifest with the exact field set.
    expected_fields = {
        "id", "name", "icon", "auth_provider", "auth_type", "server_url",
        "oauth_scopes", "tool_overrides", "featured", "status",
    }
    for m in catalog:
        assert isinstance(m, ConnectorManifest)
        assert {f.name for f in dataclasses.fields(m)} == expected_fields


def test_gmail_is_pending_verification_the_others_available():
    cat = _by_id(load_catalog())
    assert cat["gmail"].status == "pending_verification"
    assert cat["google-calendar"].status == "available"
    assert cat["google-drive"].status == "available"


def test_google_connectors_share_one_auth_provider():
    cat = _by_id(load_catalog())
    assert cat["google-calendar"].auth_provider == "google"
    assert cat["google-drive"].auth_provider == "google"
    assert cat["gmail"].auth_provider == "google"
    # All oauth2, all remote (server_url set).
    for cid in ("google-calendar", "google-drive", "gmail"):
        assert cat[cid].auth_type == "oauth2"
        assert cat[cid].server_url
        assert cat[cid].oauth_scopes  # incremental-auth scopes declared


def test_calendar_and_drive_are_featured():
    cat = _by_id(load_catalog())
    assert cat["google-calendar"].featured is True
    assert cat["google-drive"].featured is True


def test_tool_overrides_are_read_or_write_only():
    for m in load_catalog():
        for tool_name, klass in m.tool_overrides.items():
            assert klass in TOOL_CLASSES, f"{m.id}.{tool_name} -> {klass!r}"


def test_manifest_from_dict_rejects_bad_auth_type():
    good = {
        "id": "x", "name": "X", "icon": "🔌", "auth_provider": "none",
        "auth_type": "none", "server_url": "https://e/mcp", "oauth_scopes": [],
        "tool_overrides": {}, "featured": False, "status": "available",
    }
    assert manifest_from_dict(good).id == "x"
    with pytest.raises(ValueError):
        manifest_from_dict({**good, "auth_type": "totally-bogus"})
    with pytest.raises(ValueError):
        manifest_from_dict({**good, "status": "nope"})
    with pytest.raises(ValueError):
        manifest_from_dict({**good, "tool_overrides": {"t": "sideways"}})


def test_manifest_from_dict_defaults_optional_fields():
    minimal = {
        "id": "y", "name": "Y", "auth_provider": "none", "auth_type": "none",
        "server_url": "https://e/mcp",
    }
    m = manifest_from_dict(minimal)
    assert m.oauth_scopes == []
    assert m.tool_overrides == {}
    assert m.featured is False
    assert m.status == "available"
    assert m.icon  # some default icon


def test_auth_types_enum_is_the_locked_set():
    assert AUTH_TYPES == {"oauth2", "app_password", "local_native", "none"}


def test_slugify_makes_url_safe_ids():
    assert slugify("My Notion Server!") == "my-notion-server"
    assert slugify("  Spaces  ") == "spaces"
    assert slugify("weird__chars//") == "weird-chars"


def test_custom_manifest_is_tier0_shaped():
    m = custom_manifest("My Notion Server", "https://notion.mcp/x", "none")
    assert m.id == "custom-my-notion-server"
    assert m.auth_type == "none"
    assert m.auth_provider.startswith("custom-")
    assert m.server_url == "https://notion.mcp/x"
    assert m.featured is False
    assert m.status == "available"
    # oauth2 custom servers are allowed too.
    m2 = custom_manifest("Thing", "https://thing/mcp", "oauth2")
    assert m2.auth_type == "oauth2"
    # bogus auth types rejected.
    with pytest.raises(ValueError):
        custom_manifest("Bad", "https://bad/mcp", "local_native")
