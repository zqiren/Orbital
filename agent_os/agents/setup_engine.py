# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Setup engine — auto-detect, dependency check, and credential resolution.

Reads agent manifests from the registry, probes the system for installed
binaries and satisfied dependencies, and reports what is available.
"""

import logging
import os
import shutil
import subprocess

from agent_os.agents.manifest import AgentManifest
from agent_os.agents.registry import AgentRegistry
from agent_os.agents.setup_types import AgentSetupStatus, SetupAction
from agent_os.daemon_v2.models import detect_os

logger = logging.getLogger(__name__)


class SetupEngine:
    """Probes the system for agent readiness based on manifest metadata."""

    def __init__(self, registry: AgentRegistry, credential_store=None) -> None:
        self._registry = registry
        self._credential_store = credential_store
        self._resolved_paths: dict[str, str] = {}  # slug -> resolved binary path

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def check_agent(self, slug: str) -> AgentSetupStatus:
        """Full read-only status check for one agent."""
        manifest = self._registry.get(slug)
        if manifest is None:
            raise ValueError(f"Unknown agent: {slug}")

        # Built-in agents are always installed
        if manifest.runtime.adapter == "built_in":
            return AgentSetupStatus(
                slug=manifest.slug,
                name=manifest.name,
                installed=True,
                binary_path=None,
                version=manifest.version,
                dependencies_met=True,
                missing_dependencies=[],
                credentials_configured=True,
                missing_credentials=[],
                setup_actions=[],
            )

        # Resolve binary
        binary = self.resolve_binary(manifest)
        installed = binary is not None

        # Check version via check_command
        version = None
        if installed and manifest.setup.check_command:
            version = self._run_check_command(manifest.setup.check_command)

        # Dependencies
        deps_met, missing_deps = self.check_dependencies(manifest)

        # Credentials
        creds_ok, missing_creds = self.check_credentials(manifest)

        # Build setup actions
        actions = self._build_actions(manifest, installed, missing_deps, missing_creds)

        return AgentSetupStatus(
            slug=manifest.slug,
            name=manifest.name,
            installed=installed,
            binary_path=binary,
            version=version,
            dependencies_met=deps_met,
            missing_dependencies=missing_deps,
            credentials_configured=creds_ok,
            missing_credentials=missing_creds,
            setup_actions=actions,
        )

    def check_all(self) -> list[AgentSetupStatus]:
        """Status check for all registered agents."""
        results = []
        for manifest in self._registry.list_all():
            results.append(self.check_agent(manifest.slug))
        return results

    def resolve_binary(self, manifest: AgentManifest) -> str | None:
        """Find the agent binary.

        Strategy:
        1. shutil.which(command) — works if PATH is correct
        2. Expand and probe auto_detect paths for current OS
        3. Try check_command via subprocess

        Returns absolute path or None. Caches result.
        """
        # Check cache first
        if manifest.slug in self._resolved_paths:
            return self._resolved_paths[manifest.slug]

        command = manifest.runtime.command
        if not command:
            return None

        # Strategy 1: shutil.which
        path = shutil.which(command)
        if path:
            self._resolved_paths[manifest.slug] = path
            return path

        # Strategy 2: auto_detect paths for current OS
        current_os = detect_os()
        auto_paths = manifest.setup.auto_detect.get(current_os, [])
        for raw_path in auto_paths:
            expanded = os.path.expandvars(os.path.expanduser(raw_path))
            if os.path.isfile(expanded):
                self._resolved_paths[manifest.slug] = expanded
                return expanded

        # Strategy 3: try check_command via subprocess
        if manifest.setup.check_command:
            result = self._run_check_command(manifest.setup.check_command)
            if result is not None:
                # check_command succeeded, so the command is available
                # Use shutil.which one more time or fall back to bare command
                path = shutil.which(command)
                if path:
                    self._resolved_paths[manifest.slug] = path
                    return path

        return None

    def check_dependencies(self, manifest: AgentManifest) -> tuple[bool, list[str]]:
        """Run each dependency check_command via subprocess.

        Returns (all_met, missing_names).
        """
        missing: list[str] = []
        for dep in manifest.setup.dependencies:
            if not self._check_single_dependency(dep.check_command, dep.min_version):
                missing.append(dep.name)
        return (len(missing) == 0, missing)

    def check_credentials(self, manifest: AgentManifest) -> tuple[bool, list[str]]:
        """Check required credentials in credential_store or env vars.

        Returns (all_configured, missing_keys).
        """
        missing: list[str] = []
        for cred in manifest.setup.credentials:
            if not cred.required:
                continue
            # oauth_cli: check via CLI command
            if cred.type == "oauth_cli":
                if not self._check_cli_auth(cred):
                    missing.append(cred.key)
                continue
            # Check credential store
            if self._credential_store is not None:
                value = self._credential_store.get(cred.key)
                if value:
                    continue
            # Check environment variable
            env_key = cred.env_var or cred.key
            if os.environ.get(env_key):
                continue
            missing.append(cred.key)
        return (len(missing) == 0, missing)

    def _check_cli_auth(self, cred) -> bool:
        """Run a CLI command to check auth status (e.g. claude auth status --json).

        Parses JSON output and checks check_field == check_value.
        """
        import json as _json
        if not cred.check_command:
            return False
        try:
            result = subprocess.run(
                cred.check_command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode != 0:
                return False
            data = _json.loads(result.stdout)
            actual = str(data.get(cred.check_field, ""))
            return actual.lower() == cred.check_value.lower()
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError,
                _json.JSONDecodeError, ValueError):
            return False

    def get_resolved_path(self, slug: str) -> str | None:
        """Return cached resolved binary path. Call check_agent first."""
        return self._resolved_paths.get(slug)

    def get_adapter_config(
        self,
        slug: str,
        project_workspace: str,
        credential_overrides: dict | None = None,
    ) -> dict:
        """Build adapter-config-compatible dict from manifest + resolved path.

        This is the bridge between manifest world and existing CLIAdapter world.

        Raises ValueError if agent not found or not installed.
        """
        manifest = self._registry.get(slug)
        if manifest is None:
            raise ValueError(f"Unknown agent: {slug}")

        binary = self._resolved_paths.get(slug)
        if binary is None:
            binary = self.resolve_binary(manifest)
        if binary is None:
            raise ValueError(f"Agent '{slug}' not installed or binary not found")

        # Build env from manifest credentials + overrides
        env: dict[str, str] = {}
        for cred in manifest.setup.credentials:
            value = (credential_overrides or {}).get(cred.key)
            if value is None and self._credential_store is not None:
                value = self._credential_store.get(cred.key)
            if value is None:
                value = os.environ.get(cred.env_var or cred.key, "")
            if value and cred.env_var:
                env[cred.env_var] = value

        return {
            "command": binary,
            "args": list(manifest.runtime.args),
            "workspace": project_workspace,
            "approval_patterns": [
                p.get("regex", "") for p in manifest.runtime.approval_patterns
            ],
            "env": env,
            "network_domains": list(manifest.permissions.network_domains),
            "interactive": manifest.runtime.interactive,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _run_check_command(self, command: str) -> str | None:
        """Run a check command and return its stripped stdout, or None on failure."""
        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0 and result.stdout.strip():
                return result.stdout.strip()
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
            pass
        return None

    def _check_single_dependency(
        self, check_command: str, min_version: str | None
    ) -> bool:
        """Run a single dependency check_command.

        Returns True if the command succeeds (and version >= min_version if specified).
        """
        try:
            result = subprocess.run(
                check_command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode != 0:
                return False
            if min_version and result.stdout.strip():
                detected = self._extract_version(result.stdout.strip())
                if detected and not self._version_gte(detected, min_version):
                    return False
            return True
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
            return False

    @staticmethod
    def _extract_version(raw: str) -> str | None:
        """Extract a version number from command output like 'v20.11.0'."""
        import re
        match = re.search(r"(\d+\.\d+\.\d+)", raw)
        return match.group(1) if match else None

    @staticmethod
    def _version_gte(detected: str, minimum: str) -> bool:
        """Simple version comparison: is detected >= minimum?"""
        def parts(v: str) -> list[int]:
            try:
                return [int(x) for x in v.split(".")]
            except ValueError:
                return [0]
        return parts(detected) >= parts(minimum)

    def _build_actions(
        self,
        manifest: AgentManifest,
        installed: bool,
        missing_deps: list[str],
        missing_creds: list[str],
    ) -> list[SetupAction]:
        """Build the list of setup actions needed."""
        actions: list[SetupAction] = []
        current_os = detect_os()

        # Missing dependencies
        for dep in manifest.setup.dependencies:
            if dep.name in missing_deps:
                install_cmd = dep.install.get(current_os)
                actions.append(SetupAction(
                    action="install_dependency",
                    label=f"Install {dep.name}" + (
                        f" {dep.min_version}+" if dep.min_version else ""
                    ),
                    command=install_cmd,
                    blocking=True,
                ))

        # Agent not installed
        if not installed and manifest.setup.install_command:
            actions.append(SetupAction(
                action="install_agent",
                label=f"Install {manifest.name}",
                command=manifest.setup.install_command,
                blocking=True,
            ))

        # Missing credentials
        for key in missing_creds:
            # Find the credential from manifest
            cred_obj = None
            label = key
            for cred in manifest.setup.credentials:
                if cred.key == key:
                    cred_obj = cred
                    label = cred.label or key
                    break
            if cred_obj and cred_obj.type == "oauth_cli":
                actions.append(SetupAction(
                    action="run_cli_auth",
                    label=cred_obj.setup_label or f"Authenticate {label}",
                    command=cred_obj.setup_command or None,
                    blocking=True,
                ))
            else:
                actions.append(SetupAction(
                    action="configure_credential",
                    label=f"Set {label}",
                    command=None,
                    blocking=True,
                ))

        return actions
