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

from agent_os.agents.desktop_apps import parse_desktop_apps


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
    transport: str = "auto"             # "sdk" | "acp-sdk" | "pipe" | "pty" | "codex-appserver" | "pi-rpc" | "auto"
    approval_patterns: list[dict] = field(default_factory=list)
    activity_patterns: list[dict] = field(default_factory=list)
    # Filename of a composition template inside the agent's installed
    # directory (``<data_dir>/agents/<slug>/``). When set, Orbital renders a
    # per-spawn copy of it (model, permission mode, persona, session root) and
    # appends ``--config <rendered>`` to the spawn args. Empty = argv-only
    # configuration, which is every other agent today.
    config_template: str = ""
    # The flag the rendered file is handed to the CLI with: dsh 0.2 takes a
    # patch overlay on its shipped ``acp`` profile (``--patch <file>``).
    config_flag: str = "--config"
    # Rows copied verbatim from the user's own setup of this agent into the
    # rendered file (dsh: provider definitions from DSH Desktop's profile, so
    # what works in the app works here). {home_env, home_default, file, rows,
    # default_model_row, default_provider} — see agents/composition.py.
    config_mirror: dict = field(default_factory=dict)
    # A vendor's soft refusal that arrives as an ordinary reply (Antigravity
    # outside its regions answers "no valid license (#3501)" with
    # stop_reason end_turn). Each entry is {regex, reason}; the ACP transport
    # turns a turn whose WHOLE output fullmatches ``regex`` (no tool calls)
    # into an error carrying ``reason``, so the vendor quirk stays data here.
    refusal_patterns: list[dict] = field(default_factory=list)


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
    type: str = "secret"                  # "secret" | "text" | "oauth" | "oauth_cli" | "model_provider"
    required: bool = True
    env_var: str = ""                     # injected as this env var at runtime
    check_command: str = ""               # e.g. "claude auth status --json"
    check_field: str = ""                 # JSON field to inspect in check_command output
    check_value: str = ""                 # expected value of check_field
    setup_command: str = ""               # e.g. "claude login"
    setup_label: str = ""                 # human-readable label for setup action
    # ``model_provider`` only. The credential follows the agent's configured
    # ``model`` param (provider/model): each entry maps the CLI's provider id
    # to the credential-card provider (and region, for region-locked keys)
    # whose key may be injected, and the env var the CLI reads it from —
    # {provider, card_provider, card_region?, env_var}. ``check_command`` runs
    # as an argv list with a ``{model}`` token; ``check_missing_value`` is the
    # ``check_field`` value that means "definitely not configured" (any other
    # non-ready answer is reported as unknown).
    provider_keys: list[dict] = field(default_factory=list)
    check_missing_value: str = ""
    # The agent keeps this key in a store of its own that WINS over the env
    # Orbital injects (DSH Desktop's ``$DSH_HOME/.credentials.yaml``): a key
    # there means the credential is configured, and an Orbital-held one would
    # be ignored. {file, home_env?, key, label}: ``file`` is ``~``-expanded,
    # its directory replaced by ``$home_env`` when that is set; ``key`` is a
    # dotted path into the YAML document; ``label`` names the owner for the UI.
    agent_store: dict = field(default_factory=dict)
    # ``setup_command`` waits on stdin for a pasted value (``agy acp --login``
    # asks for the authorization code its browser page shows). When set, the
    # login runner keeps the child's stdin open and, on a progress line
    # containing this text (case-insensitive), asks the card for the value.
    setup_input_prompt: str = ""


@dataclass
class ManifestOrbitalInstall:
    """Declared only by agents Orbital installs itself into its data dir.

    ``platforms`` is the honest statement of where that install is supported.
    An empty list means the agent is not Orbital-installable at all — the user
    brings their own binary, which is every other agent today. An agent that
    names its platforms is not expected to carry ``auto_detect`` entries for
    the ones it left out.
    """
    platforms: list[str] = field(default_factory=list)


@dataclass
class ManifestSetup:
    dependencies: list[ManifestDependency] = field(default_factory=list)
    # "npm install -g @anthropic-ai/claude-code". The YAML may also give a
    # per-OS mapping ({macos, linux, windows}); the loader keeps this OS's.
    install_command: str | None = None
    check_command: str | None = None      # "claude --version"
    auto_detect: dict = field(default_factory=dict)  # {"windows": [...], "macos": [...]}
    credentials: list[ManifestCredential] = field(default_factory=list)
    orbital_install: ManifestOrbitalInstall = field(
        default_factory=ManifestOrbitalInstall)
    # Spec 116 D1/D2: the oldest CLI version Orbital can drive, checked
    # offline against ``check_command``'s output, and the CLI's own updater
    # (run through the login-job runner). All optional; None = no floor.
    min_version: str | None = None        # "2.1.114"
    update_command: str | None = None     # "claude update"
    update_label: str | None = None       # "Update Claude Code"
    # The npm package an npm-method install came from. When the resolved
    # binary is an npm install, the Update runs ``npm install -g
    # <package>@latest`` directly instead of ``update_command`` (whose own npm
    # step gave up after 39 s on a slow network and left the package
    # half-removed — 1018 bench item 5). The same string names the manual
    # recovery command. None = always use ``update_command``.
    npm_package: str | None = None        # "@anthropic-ai/claude-code"
    # Vendor desktop apps that bundle this agent's CLI, found by identity
    # (bundle id / uninstall registry) — see agents/desktop_apps.py.
    desktop_apps: list = field(default_factory=list)
    # Which copy to drive when detection finds more than one (PATH,
    # ``auto_detect`` entries, ``after_cli`` desktop apps): "first" takes the
    # first in that order; "newest" the one whose version check reports the
    # highest version, ties going to the earlier one. A ``before_cli``
    # desktop app and the user's chosen location still win outright.
    binary_choice: str = "first"


@dataclass
class ManifestCapabilities:
    skills: list[str] = field(default_factory=list)
    input_types: list[str] = field(default_factory=list)
    output_types: list[str] = field(default_factory=list)
    file_extensions: list[str] = field(default_factory=list)
    routing_hint: str = ""
    needs_shell: bool = False
    needs_network: bool = False
    # Whether the agent reports tool activity back to Orbital at all. False
    # means its turns are silent until the final answer — the UI says so
    # rather than showing a "working…" affordance it cannot back up.
    # Defaults True: every pre-existing agent reports something.
    emits_tool_activity: bool = True


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
    # False keeps the agent off every list (spec 171 B2: the PTY agents).
    # It stays registered, so a pin or automation that names it still runs.
    listed: bool = True


# ---------------------------------------------------------------------------
# Known adapter types
# ---------------------------------------------------------------------------

KNOWN_ADAPTERS = {"cli", "built_in", "api", "browser"}

# Loose semver: digits.digits.digits with optional pre-release suffix
_SEMVER_RE = re.compile(r"^\d+\.\d+\.\d+(-[\w.]+)?$")

# Slug: lowercase, digits, hyphens only (no spaces, no uppercase)
_SLUG_RE = re.compile(r"^[a-z0-9]+(-[a-z0-9]+)*$")


def _for_this_os(value):
    """A manifest value that may be given per OS ({macos, linux, windows}):
    this OS's entry, or None when the mapping leaves it out. A plain value
    is the same on every OS."""
    if not isinstance(value, dict):
        return value
    from agent_os.daemon_v2.models import detect_os
    return value.get(detect_os())


# ---------------------------------------------------------------------------
# Loader / Validator
# ---------------------------------------------------------------------------

BINARY_CHOICES = ("first", "newest")


def parse_binary_choice(raw) -> str:
    """``setup.binary_choice``; anything unknown reads as "first"."""
    return raw if isinstance(raw, str) and raw in BINARY_CHOICES else "first"


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
            config_template=runtime_data.get("config_template", ""),
            config_flag=runtime_data.get("config_flag", "--config"),
            config_mirror=dict(runtime_data.get("config_mirror") or {}),
            refusal_patterns=[
                dict(p) for p in (runtime_data.get("refusal_patterns") or [])
                if isinstance(p, dict) and p.get("regex") and p.get("reason")
            ],
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
                provider_keys=cred.get("provider_keys", []),
                check_missing_value=cred.get("check_missing_value", ""),
                agent_store=dict(cred.get("agent_store") or {}),
                setup_input_prompt=cred.get("setup_input_prompt", ""),
            ))
        setup = ManifestSetup(
            dependencies=dependencies,
            install_command=_for_this_os(setup_data.get("install_command")),
            check_command=setup_data.get("check_command"),
            auto_detect=setup_data.get("auto_detect", {}),
            credentials=credentials,
            orbital_install=ManifestOrbitalInstall(
                platforms=(setup_data.get("orbital_install") or {}).get(
                    "platforms", []),
            ),
            min_version=setup_data.get("min_version"),
            update_command=setup_data.get("update_command"),
            update_label=setup_data.get("update_label"),
            npm_package=setup_data.get("npm_package"),
            desktop_apps=parse_desktop_apps(setup_data.get("desktop_apps")),
            binary_choice=parse_binary_choice(setup_data.get("binary_choice")),
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
            emits_tool_activity=cap_data.get("emits_tool_activity", True),
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
            listed=data.get("listed", True) is not False,
        )
