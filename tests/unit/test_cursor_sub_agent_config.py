# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Cursor manifest and daemon-level configuration contract."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from agent_os.agents.manifest import ManifestLoader
from agent_os.daemon_v2.sub_agent_config_store import (
    SubAgentConfigError,
    SubAgentConfigStore,
)


MANIFEST = (
    Path(__file__).parents[2]
    / "agent_os"
    / "agents"
    / "manifests"
    / "cursor.yaml"
)


def test_cursor_manifest_uses_official_acp_command_layout():
    manifest = ManifestLoader.load(str(MANIFEST))

    assert manifest.slug == "cursor"
    assert manifest.runtime.command == "agent"
    assert manifest.runtime.transport == "acp-sdk"
    assert manifest.runtime.args == ["acp"]
    assert manifest.runtime.interactive is False


def test_cursor_manifest_detects_both_official_binary_names():
    manifest = ManifestLoader.load(str(MANIFEST))
    paths = [
        path
        for platform_paths in manifest.setup.auto_detect.values()
        for path in platform_paths
    ]

    assert any(path.endswith("/agent") or path.endswith("\\agent.cmd") for path in paths)
    assert any(
        path.endswith("/cursor-agent") or path.endswith("\\cursor-agent.cmd")
        for path in paths
    )


def test_cursor_config_defaults_to_auto_without_persisting_override(tmp_path):
    store = SubAgentConfigStore(str(tmp_path / "config.json"))

    assert store.get("cursor") == {}
    assert store.schema_for("cursor")["permission-mode"] == {
        "allowed": ["auto", "ask"],
        "default": "auto",
    }
    assert store.build_extra_args("cursor") == [
        "--orbital-permission-mode",
        "auto",
    ]


def test_cursor_model_and_ask_policy_are_forwarded_as_transport_config(tmp_path):
    store = SubAgentConfigStore(str(tmp_path / "config.json"))
    store.set(
        "cursor",
        {
            "model": "cursor-grok-4.5-low",
            "permission-mode": "ask",
        },
    )

    assert store.build_extra_args("cursor") == [
        "--orbital-permission-mode",
        "ask",
        "--model",
        "cursor-grok-4.5-low",
    ]


def test_cursor_rejects_unknown_permission_policy(tmp_path):
    store = SubAgentConfigStore(str(tmp_path / "config.json"))

    with pytest.raises(SubAgentConfigError):
        store.set("cursor", {"permission-mode": "allow-always"})


def test_cursor_logout_uses_resolved_official_binary(monkeypatch):
    from agent_os.api.routes import settings

    manifest = ManifestLoader.load(str(MANIFEST))
    registry = MagicMock()
    registry.get.return_value = manifest
    setup_engine = MagicMock()
    setup_engine._registry = registry
    # The resolver may find either official installer name. The logout
    # subcommand is identical for both.
    setup_engine.resolve_binary.return_value = "/opt/cursor/cursor-agent"
    monkeypatch.setattr(settings, "_setup_engine", setup_engine)

    assert settings._resolve_setup_command("cursor", "logout") == (
        "/opt/cursor/cursor-agent logout"
    )
