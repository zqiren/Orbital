# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Unit tests for oauth_cli credential type.

Tests cover: ManifestCredential new fields, SetupEngine.check_credentials()
with oauth_cli type, _check_cli_auth() success/failure/timeout, and
_build_actions() producing run_cli_auth action.
"""

import json
import os
import subprocess
from unittest.mock import MagicMock, patch

import pytest

from agent_os.agents.manifest import (
    ManifestCredential,
    ManifestLoader,
    ManifestRuntime,
    ManifestSetup,
    AgentManifest,
)
from agent_os.agents.registry import AgentRegistry
from agent_os.agents.setup_engine import SetupEngine


class TestManifestOAuthCLIFields:
    """Test ManifestCredential has oauth_cli fields."""

    def test_default_credential_fields(self):
        cred = ManifestCredential(key="test", label="Test")
        assert cred.type == "secret"
        assert cred.check_command == ""
        assert cred.check_field == ""
        assert cred.check_value == ""
        assert cred.setup_command == ""
        assert cred.setup_label == ""

    def test_oauth_cli_credential_fields(self):
        cred = ManifestCredential(
            key="claude_auth",
            label="Claude Account",
            type="oauth_cli",
            check_command="claude auth status --json",
            check_field="loggedIn",
            check_value="true",
            setup_command="claude login",
            setup_label="Log in to Claude",
        )
        assert cred.type == "oauth_cli"
        assert cred.check_command == "claude auth status --json"
        assert cred.check_field == "loggedIn"
        assert cred.check_value == "true"
        assert cred.setup_command == "claude login"
        assert cred.setup_label == "Log in to Claude"

    def test_manifest_loads_oauth_cli_credential_fields(self):
        path = os.path.join(
            os.path.dirname(__file__), os.pardir, os.pardir,
            "agent_os", "agents", "manifests", "claude_code.yaml",
        )
        path = os.path.normpath(path)
        m = ManifestLoader.load(path)
        assert len(m.setup.credentials) >= 1
        cred = m.setup.credentials[0]
        assert cred.key == "claude_auth"
        assert cred.type == "oauth_cli"
        assert cred.check_command == "claude auth status --json"
        assert cred.check_field == "loggedIn"
        assert cred.check_value == "true"
        assert cred.setup_command == "claude login"
        assert cred.setup_label == "Log in to Claude"


def _make_manifest_with_oauth_cli():
    """Helper to create a manifest with an oauth_cli credential."""
    return AgentManifest(
        manifest_version="1",
        name="Test Agent",
        slug="test-agent",
        description="Test",
        author="test",
        version="1.0.0",
        runtime=ManifestRuntime(adapter="cli", command="test-cmd"),
        setup=ManifestSetup(
            credentials=[
                ManifestCredential(
                    key="test_auth",
                    label="Test Auth",
                    type="oauth_cli",
                    required=True,
                    check_command="test-cmd auth status --json",
                    check_field="loggedIn",
                    check_value="true",
                    setup_command="test-cmd login",
                    setup_label="Log in to Test",
                ),
            ],
        ),
    )


class TestCheckCredentialsOAuthCLI:
    """Test SetupEngine.check_credentials() with oauth_cli type."""

    def test_check_credentials_oauth_cli_success(self):
        registry = AgentRegistry()
        manifest = _make_manifest_with_oauth_cli()
        registry.register(manifest)
        engine = SetupEngine(registry=registry)

        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = json.dumps({"loggedIn": True})

        with patch("agent_os.agents.setup_engine.subprocess.run", return_value=mock_result):
            ok, missing = engine.check_credentials(manifest)

        assert ok is True
        assert missing == []

    def test_check_credentials_oauth_cli_not_logged_in(self):
        registry = AgentRegistry()
        manifest = _make_manifest_with_oauth_cli()
        registry.register(manifest)
        engine = SetupEngine(registry=registry)

        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = json.dumps({"loggedIn": False})

        with patch("agent_os.agents.setup_engine.subprocess.run", return_value=mock_result):
            ok, missing = engine.check_credentials(manifest)

        assert ok is False
        assert "test_auth" in missing

    def test_check_credentials_oauth_cli_command_fails(self):
        registry = AgentRegistry()
        manifest = _make_manifest_with_oauth_cli()
        registry.register(manifest)
        engine = SetupEngine(registry=registry)

        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stdout = ""

        with patch("agent_os.agents.setup_engine.subprocess.run", return_value=mock_result):
            ok, missing = engine.check_credentials(manifest)

        assert ok is False
        assert "test_auth" in missing

    def test_check_credentials_oauth_cli_timeout(self):
        registry = AgentRegistry()
        manifest = _make_manifest_with_oauth_cli()
        registry.register(manifest)
        engine = SetupEngine(registry=registry)

        with patch("agent_os.agents.setup_engine.subprocess.run",
                    side_effect=subprocess.TimeoutExpired("test-cmd", 10)):
            ok, missing = engine.check_credentials(manifest)

        assert ok is False
        assert "test_auth" in missing

    def test_check_credentials_oauth_cli_invalid_json(self):
        registry = AgentRegistry()
        manifest = _make_manifest_with_oauth_cli()
        registry.register(manifest)
        engine = SetupEngine(registry=registry)

        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "not json"

        with patch("agent_os.agents.setup_engine.subprocess.run", return_value=mock_result):
            ok, missing = engine.check_credentials(manifest)

        assert ok is False
        assert "test_auth" in missing

    def test_check_credentials_oauth_cli_missing_field(self):
        registry = AgentRegistry()
        manifest = _make_manifest_with_oauth_cli()
        registry.register(manifest)
        engine = SetupEngine(registry=registry)

        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = json.dumps({"someOtherField": True})

        with patch("agent_os.agents.setup_engine.subprocess.run", return_value=mock_result):
            ok, missing = engine.check_credentials(manifest)

        assert ok is False
        assert "test_auth" in missing

    def test_check_credentials_oauth_cli_not_required_skipped(self):
        """Non-required oauth_cli credentials should be skipped."""
        manifest = AgentManifest(
            manifest_version="1", name="T", slug="t", description="",
            author="", version="1.0.0",
            runtime=ManifestRuntime(adapter="cli", command="t"),
            setup=ManifestSetup(
                credentials=[
                    ManifestCredential(
                        key="opt_auth", label="Optional",
                        type="oauth_cli", required=False,
                        check_command="t auth --json",
                        check_field="loggedIn", check_value="true",
                    ),
                ],
            ),
        )
        registry = AgentRegistry()
        registry.register(manifest)
        engine = SetupEngine(registry=registry)

        # No subprocess mock needed — should skip entirely
        ok, missing = engine.check_credentials(manifest)
        assert ok is True
        assert missing == []


class TestBuildActionsOAuthCLI:
    """Test _build_actions() produces run_cli_auth for oauth_cli credentials."""

    def test_build_actions_oauth_cli_produces_run_cli_auth_action(self):
        registry = AgentRegistry()
        manifest = _make_manifest_with_oauth_cli()
        registry.register(manifest)
        engine = SetupEngine(registry=registry)

        actions = engine._build_actions(
            manifest, installed=True, missing_deps=[], missing_creds=["test_auth"],
        )

        assert len(actions) == 1
        action = actions[0]
        assert action.action == "run_cli_auth"
        assert action.label == "Log in to Test"
        assert action.command == "test-cmd login"
        assert action.blocking is True

    def test_build_actions_regular_credential_still_works(self):
        manifest = AgentManifest(
            manifest_version="1", name="T", slug="t", description="",
            author="", version="1.0.0",
            runtime=ManifestRuntime(adapter="cli", command="t"),
            setup=ManifestSetup(
                credentials=[
                    ManifestCredential(
                        key="API_KEY", label="API Key",
                        type="secret", required=True,
                    ),
                ],
            ),
        )
        registry = AgentRegistry()
        registry.register(manifest)
        engine = SetupEngine(registry=registry)

        actions = engine._build_actions(
            manifest, installed=True, missing_deps=[], missing_creds=["API_KEY"],
        )

        assert len(actions) == 1
        assert actions[0].action == "configure_credential"
        assert actions[0].label == "Set API Key"
