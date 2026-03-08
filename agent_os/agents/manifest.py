# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Agent manifest schema, loader, and validator.

Defines the YAML manifest format that describes any agent: identity,
runtime config, dependencies, capabilities, and permissions.
"""

import re
from dataclasses import dataclass, field

import yaml


class ManifestError(Exception):
    """Raised when a manifest file is invalid or cannot be loaded."""


# ---------------------------------------------------------------------------
# Schema dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ManifestRuntime:
    adapter: str                          # "cli" | "built_in" | "api" | "browser"
    command: str | None = None            # e.g. "claude" (CLI agents only)
    args: list[str] = field(default_factory=list)
    interactive: bool = False             # needs PTY / stdin
    output_format: str = "text"           # "text" | "stream-json" | "json"
    response_field: str | None = None     # JSON field containing response text
    mode: str = "interactive"           # "pipe" or "interactive"
    prompt_flag: str = "-p"             # flag to pass prompt text
    resume_flag: str = "--resume"       # flag to resume session
    session_id_pattern: str = ""        # regex to extract session_id from output
    transport: str = "auto"             # "sdk" | "acp" | "pipe" | "pty" | "auto"
    approval_patterns: list[dict] = field(default_factory=list)
    activity_patterns: list[dict] = field(default_factory=list)


@dataclass
class ManifestDependency:
    name: str                             # "Node.js"
    check_command: str                    # "node --version"
    min_version: str | None = None        # "18.0.0"
    install: dict = field(default_factory=dict)  # {"windows": "winget install ...", ...}


@dataclass
class ManifestCredential:
    key: str                              # "ANTHROPIC_API_KEY"
    label: str                            # "Anthropic API Key"
    type: str = "secret"                  # "secret" | "text" | "oauth" | "oauth_cli"
    required: bool = True
    env_var: str = ""                     # injected as this env var at runtime
    check_command: str = ""               # e.g. "claude auth status --json"
    check_field: str = ""                 # JSON field to inspect in check_command output
    check_value: str = ""                 # expected value of check_field
    setup_command: str = ""               # e.g. "claude login"
    setup_label: str = ""                 # human-readable label for setup action


@dataclass
class ManifestSetup:
    dependencies: list[ManifestDependency] = field(default_factory=list)
    install_command: str | None = None    # "npm install -g @anthropic-ai/claude-code"
    check_command: str | None = None      # "claude --version"
    auto_detect: dict = field(default_factory=dict)  # {"windows": [...], "macos": [...]}
    credentials: list[ManifestCredential] = field(default_factory=list)


@dataclass
class ManifestCapabilities:
    skills: list[str] = field(default_factory=list)
    input_types: list[str] = field(default_factory=list)
    output_types: list[str] = field(default_factory=list)
    file_extensions: list[str] = field(default_factory=list)
    routing_hint: str = ""
    needs_shell: bool = False
    needs_network: bool = False


@dataclass
class ManifestPermissions:
    network_domains: list[str] = field(default_factory=list)
    shell: bool = False
    workspace_access: str = "read_write"  # "read_only" | "read_write" | "none"
    autonomy_recommended: str = "check_in"


@dataclass
class AgentManifest:
    manifest_version: str                 # "1"
    name: str                             # "Claude Code"
    slug: str                             # "claude-code" (unique ID, URL-safe)
    description: str
    author: str
    version: str                          # semver
    runtime: ManifestRuntime
    setup: ManifestSetup = field(default_factory=ManifestSetup)
    capabilities: ManifestCapabilities = field(default_factory=ManifestCapabilities)
    permissions: ManifestPermissions = field(default_factory=ManifestPermissions)


# ---------------------------------------------------------------------------
# Known adapter types
# ---------------------------------------------------------------------------

KNOWN_ADAPTERS = {"cli", "built_in", "api", "browser"}

# Loose semver: digits.digits.digits with optional pre-release suffix
_SEMVER_RE = re.compile(r"^\d+\.\d+\.\d+(-[\w.]+)?$")

# Slug: lowercase, digits, hyphens only (no spaces, no uppercase)
_SLUG_RE = re.compile(r"^[a-z0-9]+(-[a-z0-9]+)*$")


# ---------------------------------------------------------------------------
# Loader / Validator
# ---------------------------------------------------------------------------

class ManifestLoader:
    """Load and validate YAML agent manifest files."""

    @staticmethod
    def load(path: str) -> AgentManifest:
        """Load and validate a single YAML manifest file.

        Raises ManifestError on invalid schema or file errors.
        """
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
        except FileNotFoundError:
            raise ManifestError(f"Manifest file not found: {path}")
        except yaml.YAMLError as exc:
            raise ManifestError(f"YAML parse error in {path}: {exc}")

        if not isinstance(data, dict):
            raise ManifestError(f"Manifest must be a YAML mapping: {path}")

        errors = ManifestLoader.validate(data)
        if errors:
            raise ManifestError(
                f"Validation errors in {path}: {'; '.join(errors)}"
            )

        return ManifestLoader._build(data)

    @staticmethod
    def validate(data: dict) -> list[str]:
        """Return list of validation errors. Empty list means valid."""
        errors: list[str] = []

        # Required top-level fields
        for field_name in (
            "manifest_version", "name", "slug", "description", "author",
            "version", "runtime",
        ):
            if field_name not in data:
                errors.append(f"Missing required field: {field_name}")

        # Slug format
        slug = data.get("slug", "")
        if slug and not _SLUG_RE.match(slug):
            errors.append(
                f"Invalid slug '{slug}': must be lowercase, hyphens, no spaces"
            )

        # Adapter
        runtime = data.get("runtime", {})
        if isinstance(runtime, dict):
            adapter = runtime.get("adapter", "")
            if adapter and adapter not in KNOWN_ADAPTERS:
                errors.append(
                    f"Unknown adapter '{adapter}': must be one of {sorted(KNOWN_ADAPTERS)}"
                )

        # Version (semver-ish)
        version = data.get("version", "")
        if version and not _SEMVER_RE.match(version):
            errors.append(
                f"Invalid version '{version}': expected semver (e.g. 1.0.0)"
            )

        return errors

    @staticmethod
    def _build(data: dict) -> AgentManifest:
        """Build AgentManifest from validated dict."""
        runtime_data = data.get("runtime", {})
        runtime = ManifestRuntime(
            adapter=runtime_data.get("adapter", ""),
            command=runtime_data.get("command"),
            args=runtime_data.get("args", []),
            interactive=runtime_data.get("interactive", False),
            output_format=runtime_data.get("output_format", "text"),
            response_field=runtime_data.get("response_field"),
            mode=runtime_data.get("mode", "interactive"),
            prompt_flag=runtime_data.get("prompt_flag", "-p"),
            resume_flag=runtime_data.get("resume_flag", "--resume"),
            session_id_pattern=runtime_data.get("session_id_pattern", ""),
            transport=runtime_data.get("transport", "auto"),
            approval_patterns=runtime_data.get("approval_patterns", []),
            activity_patterns=runtime_data.get("activity_patterns", []),
        )

        # Setup
        setup_data = data.get("setup", {})
        dependencies = []
        for dep in setup_data.get("dependencies", []):
            dependencies.append(ManifestDependency(
                name=dep.get("name", ""),
                check_command=dep.get("check_command", ""),
                min_version=dep.get("min_version"),
                install=dep.get("install", {}),
            ))
        credentials = []
        for cred in setup_data.get("credentials", []):
            credentials.append(ManifestCredential(
                key=cred.get("key", ""),
                label=cred.get("label", ""),
                type=cred.get("type", "secret"),
                required=cred.get("required", True),
                env_var=cred.get("env_var", ""),
                check_command=cred.get("check_command", ""),
                check_field=cred.get("check_field", ""),
                check_value=cred.get("check_value", ""),
                setup_command=cred.get("setup_command", ""),
                setup_label=cred.get("setup_label", ""),
            ))
        setup = ManifestSetup(
            dependencies=dependencies,
            install_command=setup_data.get("install_command"),
            check_command=setup_data.get("check_command"),
            auto_detect=setup_data.get("auto_detect", {}),
            credentials=credentials,
        )

        # Capabilities
        cap_data = data.get("capabilities", {})
        capabilities = ManifestCapabilities(
            skills=cap_data.get("skills", []),
            input_types=cap_data.get("input_types", []),
            output_types=cap_data.get("output_types", []),
            file_extensions=cap_data.get("file_extensions", []),
            routing_hint=cap_data.get("routing_hint", ""),
            needs_shell=cap_data.get("needs_shell", False),
            needs_network=cap_data.get("needs_network", False),
        )

        # Permissions
        perm_data = data.get("permissions", {})
        permissions = ManifestPermissions(
            network_domains=perm_data.get("network_domains", []),
            shell=perm_data.get("shell", False),
            workspace_access=perm_data.get("workspace_access", "read_write"),
            autonomy_recommended=perm_data.get("autonomy_recommended", "check_in"),
        )

        return AgentManifest(
            manifest_version=str(data["manifest_version"]),
            name=data["name"],
            slug=data["slug"],
            description=data["description"],
            author=data["author"],
            version=data["version"],
            runtime=runtime,
            setup=setup,
            capabilities=capabilities,
            permissions=permissions,
        )
