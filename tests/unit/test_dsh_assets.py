# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Unit tests for the shipped dsh composition assets.

These assets (``package.json``, ``package-lock.json``, ``cordis.template.yml``)
are what Task 3's installer copies into ``<data_dir>/agents/dsh/`` and installs
with ``npm ci``, and what Task 4's renderer loads per spawn. dsh is pre-1.0
(rc.6, "THERE WILL BE COMPATIBILITY-BREAKING CHANGES"), so the freeze has to be
airtight: exact direct pins alone do not freeze the intra-tree graph, because
every dsh package depends on its siblings through ``^`` ranges. The lockfile is
the real freeze; these tests guard it.
"""

import json
import re
from pathlib import Path

import pytest
import yaml

ASSETS_DIR = Path(__file__).resolve().parents[2] / "agent_os" / "agents" / "assets" / "dsh"
PACKAGE_JSON = ASSETS_DIR / "package.json"
PACKAGE_LOCK = ASSETS_DIR / "package-lock.json"
CORDIS_TEMPLATE = ASSETS_DIR / "cordis.template.yml"

# The nine plugins the verified composition names. Two of them
# (dsh-sandbox-policy, dsh-user-approval) resolved only transitively during the
# evaluation probe — they must be direct pins here or a future rc floats them.
EXPECTED_DEPENDENCIES = {
    "@deepseek-ai/dsh-acp-demo",
    "@deepseek-ai/dsh-llm-deepseek",
    "@deepseek-ai/dsh-sandbox-local",
    "@deepseek-ai/dsh-sandbox-policy",
    "@deepseek-ai/dsh-subprocess-local",
    "@deepseek-ai/dsh-bash-sandbox",
    "@deepseek-ai/dsh-user-approval",
    "@deepseek-ai/dsh-fs-sandbox",
    "@deepseek-ai/dsh-tool-fs",
}

EXACT_VERSION_RE = re.compile(r"^\d+\.\d+\.\d+(?:-[0-9A-Za-z.-]+)?$")

# A key-shaped literal: an sk- prefix followed by enough payload to be real.
KEY_MATERIAL_RE = re.compile(r"sk-[A-Za-z0-9_-]{16,}")
# An api key field with a non-empty value assigned to it.
API_KEY_ASSIGNMENT_RE = re.compile(
    r"""["']?(?:api_?key|apiKey|secret|token)["']?\s*[:=]\s*["']?(?!["'\s]|null|\$)\S""",
    re.IGNORECASE,
)
DEVELOPER_PATH_RE = re.compile(r"/Users/|/home/[a-z]|[A-Z]:\\\\Users")


@pytest.fixture(scope="module")
def package_json() -> dict:
    return json.loads(PACKAGE_JSON.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def package_lock() -> dict:
    return json.loads(PACKAGE_LOCK.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def cordis_template() -> list:
    return yaml.safe_load(CORDIS_TEMPLATE.read_text(encoding="utf-8"))


class TestPackageJson:
    def test_package_json_exists(self):
        assert PACKAGE_JSON.is_file(), f"missing shipped asset: {PACKAGE_JSON}"

    def test_declares_all_nine_plugins(self, package_json):
        deps = package_json.get("dependencies", {})
        assert set(deps) == EXPECTED_DEPENDENCIES, (
            "dependencies must name all nine composition plugins exactly; "
            f"missing={sorted(EXPECTED_DEPENDENCIES - set(deps))} "
            f"unexpected={sorted(set(deps) - EXPECTED_DEPENDENCIES)}"
        )

    def test_every_dependency_is_pinned_exactly(self, package_json):
        """No ^ or ~ anywhere — every dsh package depends on siblings via ^ ranges."""
        deps = package_json.get("dependencies", {})
        floating = {
            name: spec
            for name, spec in deps.items()
            if not EXACT_VERSION_RE.match(spec)
        }
        assert not floating, f"dependencies must be exact-pinned, found ranges: {floating}"

    def test_is_private_and_not_publishable(self, package_json):
        assert package_json.get("private") is True, (
            "the composition manifest is an internal install target, never published"
        )

    def test_has_no_lifecycle_scripts(self, package_json):
        """`npm ci` runs scripts; the composition must not add its own."""
        assert not package_json.get("scripts"), (
            "the composition manifest must not declare lifecycle scripts"
        )


class TestPackageLock:
    def test_lockfile_exists(self):
        assert PACKAGE_LOCK.is_file(), (
            f"missing {PACKAGE_LOCK} — the committed lockfile IS the version freeze "
            "(exact direct pins do not freeze the ^-ranged intra-tree graph)"
        )

    def test_locked_versions_match_the_pins(self, package_json, package_lock):
        packages = package_lock.get("packages", {})
        for name, pinned in sorted(package_json.get("dependencies", {}).items()):
            entry = packages.get(f"node_modules/{name}")
            assert entry is not None, f"{name} is pinned but absent from the lockfile tree"
            assert entry.get("version") == pinned, (
                f"{name}: package.json pins {pinned} but the lockfile resolves "
                f"{entry.get('version')}"
            )

    def test_lockfile_root_mirrors_the_manifest(self, package_json, package_lock):
        root = package_lock.get("packages", {}).get("")
        assert root is not None, "lockfile has no root package entry"
        assert root.get("dependencies") == package_json.get("dependencies"), (
            "the lockfile root entry has drifted from package.json — regenerate it"
        )

    def test_every_dsh_package_in_the_tree_is_fully_resolved(self, package_lock):
        """Transitive dsh packages must carry a concrete version + integrity hash."""
        for path, entry in package_lock.get("packages", {}).items():
            if "@deepseek-ai/dsh" not in path:
                continue
            assert entry.get("version"), f"{path} has no locked version"
            assert entry.get("resolved"), f"{path} has no resolved URL"
            assert entry.get("integrity"), f"{path} has no integrity hash"


class TestCordisTemplate:
    def test_template_exists(self):
        assert CORDIS_TEMPLATE.is_file(), f"missing shipped asset: {CORDIS_TEMPLATE}"

    def test_template_parses_as_yaml(self, cordis_template):
        assert isinstance(cordis_template, list) and cordis_template, (
            "the composition is a non-empty YAML list of plugin entries"
        )

    def test_template_has_no_placeholder_syntax(self, cordis_template):
        """Task 4 renders via safe_load/safe_dump, never string templating.

        Scans the parsed values rather than the raw bytes: ``safe_load`` drops
        comments, so a comment mentioning the syntax is harmless, while a
        placeholder that survives into a config value is the actual bug.
        """
        offenders = []

        def walk(node, path):
            if isinstance(node, dict):
                for key, value in node.items():
                    walk(value, f"{path}.{key}")
            elif isinstance(node, list):
                for index, value in enumerate(node):
                    walk(value, f"{path}[{index}]")
            elif isinstance(node, str) and ("{{" in node or "}}" in node):
                offenders.append((path, node))

        walk(cordis_template, "")
        assert not offenders, (
            "template values must not carry placeholder syntax — the renderer "
            "overwrites fields programmatically, and dsh itself resolves "
            "double-brace names inside persona text: "
            f"{offenders}"
        )

    def test_every_plugin_name_is_a_declared_dependency(self, cordis_template, package_json):
        """The assertion that catches a composition naming an uninstalled plugin."""
        deps = set(package_json.get("dependencies", {}))
        names = {entry["name"] for entry in cordis_template if "name" in entry}
        assert names, "the template declares no plugin names"
        missing = names - deps
        assert not missing, (
            f"composition names plugins that package.json does not install: {sorted(missing)}"
        )

    def test_acp_agent_entry_declares_provider_and_workspace_context(self, cordis_template):
        agent = next(
            (e for e in cordis_template if e.get("name") == "@deepseek-ai/dsh-acp-demo"),
            None,
        )
        assert agent is not None, "the composition must include the ACP agent plugin"
        config = agent.get("config") or {}
        assert config.get("provider"), "acp-agent must declare a provider"
        assert isinstance(config.get("workspaceContext"), dict), (
            "acp-agent must declare a workspaceContext block"
        )

    def test_renderer_target_fields_are_present(self, cordis_template):
        """The renderer overwrites these in place; absent keys mean a silent no-op."""
        by_name = {e.get("name"): (e.get("config") or {}) for e in cordis_template}
        agent = by_name["@deepseek-ai/dsh-acp-demo"]
        for field in ("model", "persistenceRoot", "persona"):
            assert field in agent, f"acp-agent config must ship a default {field!r}"
        policy = by_name["@deepseek-ai/dsh-sandbox-policy"]
        assert "mode" in policy, "sandbox-policy must ship a default mode"

    def test_approval_policy_is_non_interactive(self, cordis_template):
        """Product Decision 9 — dsh-acp puts no permission traffic on the wire."""
        approval = next(
            e for e in cordis_template if e.get("name") == "@deepseek-ai/dsh-user-approval"
        )
        assert (approval.get("config") or {}).get("policy") == "never", (
            "an interactive approval policy risks an invisible internal block "
            "indistinguishable from a slow turn"
        )


class TestNoLeakedSecretsOrPaths:
    @pytest.mark.parametrize(
        "path", [PACKAGE_JSON, PACKAGE_LOCK, CORDIS_TEMPLATE], ids=lambda p: p.name
    )
    def test_no_key_material(self, path):
        raw = path.read_text(encoding="utf-8")
        assert not KEY_MATERIAL_RE.search(raw), f"{path.name} contains key-shaped material"
        found = API_KEY_ASSIGNMENT_RE.search(raw)
        assert found is None, (
            f"{path.name} assigns a credential field: {found.group(0)!r}"
        )

    @pytest.mark.parametrize(
        "path", [PACKAGE_JSON, PACKAGE_LOCK, CORDIS_TEMPLATE], ids=lambda p: p.name
    )
    def test_no_absolute_developer_paths(self, path):
        raw = path.read_text(encoding="utf-8")
        found = DEVELOPER_PATH_RE.search(raw)
        assert found is None, (
            f"{path.name} embeds an absolute developer path: {found.group(0)!r}"
        )
