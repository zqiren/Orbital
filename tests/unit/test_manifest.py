# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Unit tests for agent manifest schema, loader, and registry.

Tests cover: ManifestLoader (load, validate), AgentRegistry (load_directory,
get, list_all, list_by_adapter, get_for_routing), and seed manifests.
"""

import os

import pytest

# Path to the seed manifests directory
_MANIFESTS_DIR = os.path.join(
    os.path.dirname(__file__), os.pardir, os.pardir,
    "agent_os", "agents", "manifests",
)
_MANIFESTS_DIR = os.path.normpath(_MANIFESTS_DIR)


# ---------------------------------------------------------------------------
# ManifestLoader tests
# ---------------------------------------------------------------------------


class TestManifestLoader:

    def test_load_valid_manifest(self):
        from agent_os.agents.manifest import ManifestLoader

        path = os.path.join(_MANIFESTS_DIR, "claude_code.yaml")
        m = ManifestLoader.load(path)

        assert m.name == "Claude Code"
        assert m.slug == "claude-code"
        assert m.manifest_version == "1"
        assert m.author == "anthropic"
        assert m.version == "1.0.0"
        assert m.description == "Anthropic's autonomous coding agent"

        # Runtime
        assert m.runtime.adapter == "cli"
        assert m.runtime.command == "claude"
        assert m.runtime.args == ["--output-format", "stream-json", "--verbose"]
        assert m.runtime.transport == "sdk"
        assert m.runtime.mode == "pipe"
        assert m.runtime.prompt_flag == "-p"
        assert m.runtime.resume_flag == "--resume"
        assert len(m.runtime.approval_patterns) == 2
        assert len(m.runtime.activity_patterns) == 3

        # Setup
        assert len(m.setup.dependencies) == 1
        assert m.setup.dependencies[0].name == "Node.js"
        assert m.setup.dependencies[0].min_version == "18.0.0"
        assert "windows" in m.setup.auto_detect
        assert "macos" in m.setup.auto_detect
        assert "linux" in m.setup.auto_detect
        assert m.setup.install_command == "npm install -g @anthropic-ai/claude-code"
        assert m.setup.check_command == "claude --version"

        # Capabilities
        assert "debugging" in m.capabilities.skills
        assert "code_generation" in m.capabilities.skills
        assert ".py" in m.capabilities.file_extensions
        assert m.capabilities.routing_hint != ""
        assert m.capabilities.needs_shell is True
        assert m.capabilities.needs_network is True

        # Permissions
        assert "api.anthropic.com" in m.permissions.network_domains
        assert m.permissions.shell is True
        assert m.permissions.workspace_access == "read_write"

    def test_load_built_in_manifest(self):
        from agent_os.agents.manifest import ManifestLoader

        path = os.path.join(_MANIFESTS_DIR, "built_in.yaml")
        m = ManifestLoader.load(path)

        assert m.slug == "built-in"
        assert m.name == "Orbital Assistant"
        assert m.runtime.adapter == "built_in"
        assert m.runtime.command is None
        assert "research" in m.capabilities.skills
        assert m.permissions.shell is True

    def test_validate_missing_required(self):
        from agent_os.agents.manifest import ManifestLoader

        data = {
            "manifest_version": "1",
            "name": "Test",
            # slug missing
            "description": "test",
            "author": "test",
            "version": "1.0.0",
            "runtime": {"adapter": "cli"},
        }
        errors = ManifestLoader.validate(data)
        assert any("slug" in e for e in errors)

    def test_validate_bad_slug(self):
        from agent_os.agents.manifest import ManifestLoader

        data = {
            "manifest_version": "1",
            "name": "Test",
            "slug": "claude code",  # spaces not allowed
            "description": "test",
            "author": "test",
            "version": "1.0.0",
            "runtime": {"adapter": "cli"},
        }
        errors = ManifestLoader.validate(data)
        assert any("slug" in e.lower() for e in errors)

    def test_validate_unknown_adapter(self):
        from agent_os.agents.manifest import ManifestLoader

        data = {
            "manifest_version": "1",
            "name": "Test",
            "slug": "test-agent",
            "description": "test",
            "author": "test",
            "version": "1.0.0",
            "runtime": {"adapter": "foobar"},
        }
        errors = ManifestLoader.validate(data)
        assert any("adapter" in e.lower() for e in errors)

    def test_load_nonexistent_file(self):
        from agent_os.agents.manifest import ManifestError, ManifestLoader

        with pytest.raises(ManifestError, match="not found"):
            ManifestLoader.load("/nonexistent/path/manifest.yaml")

    def test_validate_bad_version(self):
        from agent_os.agents.manifest import ManifestLoader

        data = {
            "manifest_version": "1",
            "name": "Test",
            "slug": "test-agent",
            "description": "test",
            "author": "test",
            "version": "not-a-version",
            "runtime": {"adapter": "cli"},
        }
        errors = ManifestLoader.validate(data)
        assert any("version" in e.lower() for e in errors)


# ---------------------------------------------------------------------------
# AgentRegistry tests
# ---------------------------------------------------------------------------


class TestAgentRegistry:

    def test_registry_load_directory(self):
        from agent_os.agents.registry import AgentRegistry

        registry = AgentRegistry()
        registry.load_directory(_MANIFESTS_DIR)

        all_manifests = registry.list_all()
        assert len(all_manifests) == 2

    def test_registry_get_by_slug(self):
        from agent_os.agents.registry import AgentRegistry

        registry = AgentRegistry()
        registry.load_directory(_MANIFESTS_DIR)

        m = registry.get("claude-code")
        assert m is not None
        assert m.name == "Claude Code"
        assert m.runtime.adapter == "cli"

    def test_registry_get_nonexistent(self):
        from agent_os.agents.registry import AgentRegistry

        registry = AgentRegistry()
        registry.load_directory(_MANIFESTS_DIR)

        assert registry.get("foobar") is None

    def test_registry_list_by_adapter(self):
        from agent_os.agents.registry import AgentRegistry

        registry = AgentRegistry()
        registry.load_directory(_MANIFESTS_DIR)

        cli_agents = registry.list_by_adapter("cli")
        assert len(cli_agents) == 1
        assert cli_agents[0].slug == "claude-code"

        builtin_agents = registry.list_by_adapter("built_in")
        assert len(builtin_agents) == 1
        assert builtin_agents[0].slug == "built-in"

        api_agents = registry.list_by_adapter("api")
        assert len(api_agents) == 0

    def test_registry_get_for_routing(self):
        from agent_os.agents.registry import AgentRegistry

        registry = AgentRegistry()
        registry.load_directory(_MANIFESTS_DIR)

        debugging = registry.get_for_routing("debugging")
        assert len(debugging) == 1
        assert debugging[0].slug == "claude-code"

        research = registry.get_for_routing("research")
        assert len(research) == 1
        assert research[0].slug == "built-in"

        # Nonexistent skill
        assert len(registry.get_for_routing("teleportation")) == 0

    def test_registry_register_overwrites(self):
        from agent_os.agents.manifest import AgentManifest, ManifestRuntime
        from agent_os.agents.registry import AgentRegistry

        registry = AgentRegistry()
        m1 = AgentManifest(
            manifest_version="1", name="Agent V1", slug="test",
            description="v1", author="test", version="1.0.0",
            runtime=ManifestRuntime(adapter="cli"),
        )
        m2 = AgentManifest(
            manifest_version="1", name="Agent V2", slug="test",
            description="v2", author="test", version="2.0.0",
            runtime=ManifestRuntime(adapter="cli"),
        )
        registry.register(m1)
        assert registry.get("test").name == "Agent V1"
        registry.register(m2)
        assert registry.get("test").name == "Agent V2"
        assert len(registry.list_all()) == 1

    def test_registry_load_nonexistent_directory(self):
        from agent_os.agents.registry import AgentRegistry

        registry = AgentRegistry()
        # Should not raise, just warn
        registry.load_directory("/nonexistent/directory")
        assert len(registry.list_all()) == 0
