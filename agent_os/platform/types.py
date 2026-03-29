# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

import io
from dataclasses import dataclass, field
from typing import Literal, Callable

# --- Constants ---

SANDBOX_USERNAME = "AgentOS-Worker"
CREDENTIAL_KEY_PREFIX = "AgentOS"
SANDBOX_PASSWORD_KEY = "AgentOS/sandbox_password"
DEFAULT_PROXY_HOST = "127.0.0.1"
WORKSPACE_AGENT_DIR = "orbital"
WORKSPACE_OUTPUT_DIR = "orbital-output"

# --- C1: Sandbox Account ---

@dataclass
class AccountStatus:
    exists: bool
    username: str                    # "AgentOS-Worker"
    password_valid: bool
    is_admin: bool                   # is current process elevated?
    error: str | None = None

# --- C2: Permission Manager ---

@dataclass
class PermissionResult:
    success: bool
    path: str
    error: str | None = None

@dataclass
class AccessInfo:
    has_access: bool
    mode: Literal["none", "read_only", "read_write"]
    path: str

@dataclass
class FolderInfo:
    path: str
    display_name: str               # "Desktop", "Documents"
    accessible: bool
    access_note: str | None = None

# --- C3: Process Launcher ---

@dataclass
class ProcessHandle:
    pid: int
    command: str
    stdin: io.RawIOBase | None = None
    stdout: io.RawIOBase | None = None
    stderr: io.RawIOBase | None = None
    _native_handles: dict = field(default_factory=dict, repr=False)

@dataclass
class CommandResult:
    exit_code: int
    stdout: str
    stderr: str
    timed_out: bool = False

# --- C4: Network Proxy ---

@dataclass
class NetworkRules:
    mode: Literal["allowlist", "denylist"]
    domains: list[str]
    log_blocked: bool = True

DEFAULT_ALLOWLIST_DOMAINS: list[str] = [
    "api.openai.com",
    "api.anthropic.com",
    "generativelanguage.googleapis.com",
    "api.deepseek.com",
    "api.mistral.ai",
    "pypi.org",
    "files.pythonhosted.org",
    "registry.npmjs.org",
    "github.com",
    "raw.githubusercontent.com",
    "objects.githubusercontent.com",
]

# Type alias for proxy blocked callback
# Args: (project_id, domain, method)
BlockedCallback = Callable[[str, str, str], None]

# --- C6: Setup ---

@dataclass
class SetupStatus:
    is_complete: bool
    sandbox_user_exists: bool
    sandbox_password_valid: bool
    workspace_ready: bool
    issues: list[str] = field(default_factory=list)

@dataclass
class SetupResult:
    success: bool
    error: str | None = None

# --- Provider: Platform Capabilities ---

@dataclass
class PlatformCapabilities:
    platform: str                    # "windows"
    isolation_method: str            # "sandbox_user"
    setup_complete: bool
    setup_issues: list[str]
    supports_network_restriction: bool
    supports_folder_access: bool
    sandbox_username: str | None     # "AgentOS-Worker" or None if not set up
