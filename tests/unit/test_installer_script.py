# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Validate the Inno Setup script has required sections."""

import os

import pytest

INSTALLER_SCRIPT = os.path.join(
    os.path.dirname(__file__), "..", "..", "installer", "agentos-setup.iss"
)


@pytest.fixture
def iss_content():
    if not os.path.exists(INSTALLER_SCRIPT):
        pytest.skip("Installer script not found")
    with open(INSTALLER_SCRIPT, "r") as f:
        return f.read()


class TestInnoSetupScript:
    """Validate installer script has correct configuration."""

    def test_requires_admin(self, iss_content):
        """Installer must request admin privileges for sandbox setup."""
        assert "PrivilegesRequired=admin" in iss_content

    def test_has_sandbox_setup_run(self, iss_content):
        """Installer must run sandbox setup post-install."""
        assert "--setup-sandbox" in iss_content

    def test_has_sandbox_teardown_uninstall(self, iss_content):
        """Uninstaller should tear down sandbox."""
        assert "--teardown-sandbox" in iss_content

    def test_setup_runs_hidden(self, iss_content):
        """Sandbox setup should run hidden (no console window flash)."""
        # Inno Setup uses \ for line continuation — join them
        joined = iss_content.replace("\\\n", " ")
        lines = joined.split("\n")
        setup_lines = [l for l in lines if "--setup-sandbox" in l]
        assert len(setup_lines) >= 1
        assert "runhidden" in setup_lines[0].lower()

    def test_setup_waits_for_completion(self, iss_content):
        """Installer must wait for sandbox setup to finish before proceeding."""
        joined = iss_content.replace("\\\n", " ")
        lines = joined.split("\n")
        setup_lines = [l for l in lines if "--setup-sandbox" in l]
        assert len(setup_lines) >= 1
        assert "waituntilterminated" in setup_lines[0].lower()

    def test_app_launch_is_after_setup(self, iss_content):
        """App launch must come AFTER sandbox setup in [Run] section."""
        setup_pos = iss_content.find("--setup-sandbox")
        launch_pos = iss_content.find("postinstall")
        assert setup_pos > 0, "Sandbox setup not found"
        assert launch_pos > 0, "App launch not found"
        assert setup_pos < launch_pos, "Sandbox setup must run before app launch"
