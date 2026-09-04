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

# --- Spec 077: the sandbox boundary, defined once for both platforms ---
#
# The invariant both mechanisms enforce: the agent can use every tool you
# installed and everything in the workspace; it cannot read your secrets; it
# cannot alter files that run outside the sandbox.
#
# macOS emits these as Seatbelt rules (``platform/macos/sandbox.py``).
# Windows never grants the credential paths (the user profile stays closed to
# the worker account) and applies deny ACEs for the control files
# (``platform/windows/permissions.py``).

#: Credential stores denied for *reading* even though reads are otherwise open
#: to the whole disk. Entries are POSIX-style paths relative to the user's home
#: directory. Deliberately narrow: this is a credential list, not an attention
#: list (attention is the prompt/shell layer's job — spec 077 §4.7).
#:
#: ``~/.npmrc`` is deliberately absent: npm needs the registry config, and a
#: token stored there is the user's call.
CREDENTIAL_READ_DENY: tuple[str, ...] = (
    ".ssh",
    ".gnupg",
    ".aws",
    ".azure",
    ".kube",
    ".docker/config.json",
    ".netrc",
    ".git-credentials",
    ".pypirc",
    # Spec writes ``.cargo/credentials*``; Seatbelt's ``subpath`` has no glob,
    # so both real spellings are listed explicitly.
    ".cargo/credentials",
    ".cargo/credentials.toml",
    ".config/gh",
    ".config/gcloud",
    ".claude/.credentials.json",
    ".codex/auth.json",
    # Legacy Orbital data dir (project keys, credential-meta.json).
    "orbital",
)

#: macOS-only additions to :data:`CREDENTIAL_READ_DENY`, same home-relative form.
CREDENTIAL_READ_DENY_MACOS: tuple[str, ...] = (
    "Library/Keychains",
    # Plaintext project keys until spec 082, credential-meta.json, and the
    # agent browser profile's cookies.
    "Library/Application Support/Orbital",
)

#: Files that must never be written *inside a writable root* — they run outside
#: the sandbox the next time the user does something ordinary. Relative to each
#: writable root. ``.git/objects``, ``.git/index`` and ``.git/refs`` are
#: deliberately absent so ``git commit`` keeps working.
PROTECTED_CONTROL_FILES: tuple[str, ...] = (
    ".git/hooks",
    ".git/config",
    ".bashrc",
    ".zshrc",
    ".profile",
    ".gitconfig",
)


#: Per-user toolchain roots the Windows worker account gets read+execute on
#: (spec 077 W1). The user's profile stays closed — a Windows profile holds
#: browsers, mail and vaults, not just dev tools — so these are granted one by
#: one instead. Anything else is one click in Settings › Folder access.
#: Credential folders are never in this list, by construction.
WINDOWS_TOOLCHAIN_ROOTS: tuple[str, ...] = (
    r"%APPDATA%\npm",
    r"%APPDATA%\nvm",
    r"%LOCALAPPDATA%\Programs",
    r"%LOCALAPPDATA%\fnm_multishells",
    r"%LOCALAPPDATA%\Microsoft\WinGet",
    r"%LOCALAPPDATA%\pnpm",
    r"%LOCALAPPDATA%\uv",
    r"%USERPROFILE%\scoop",
    r"%USERPROFILE%\.cargo",
    r"%USERPROFILE%\.rustup",
    r"%USERPROFILE%\.pyenv",
    r"%USERPROFILE%\.local\bin",
    r"%USERPROFILE%\go",
    r"%USERPROFILE%\.bun",
    r"%USERPROFILE%\.deno",
    r"%USERPROFILE%\.jdks",
    r"%USERPROFILE%\.m2",
)

#: The worker's own home under ProgramData (spec 077 W2), relative to
#: ``%ProgramData%``. Windows' answer to macOS M4: every cache the agent writes
#: lands in a directory the worker owns, so no cache allowlist is needed and
#: nothing leaks into the user's profile.
WINDOWS_WORKER_HOME_RELATIVE = r"Orbital\worker"


def windows_worker_home(environ: dict | None = None) -> str:
    """Absolute path of the sandbox worker's home (spec 077 W2).

    ``%ProgramData%`` is readable by every account and writable only where
    granted, which is what makes it the right root: the worker owns its subtree
    and the user's profile is never touched.

    Built with ``ntpath`` rather than ``os.path`` so the result is a real
    Windows path no matter which host computes it — the unit tests run on
    macOS, Linux and windows-latest.
    """
    import ntpath
    import os as _os

    env = _os.environ if environ is None else environ
    program_data = env.get("ProgramData") or env.get("PROGRAMDATA") or r"C:\ProgramData"
    return ntpath.join(program_data, WINDOWS_WORKER_HOME_RELATIVE)


def windows_worker_env(
    home: str | None = None, environ: dict | None = None
) -> dict[str, str]:
    """Environment overrides pointing a worker process at its own home (W2).

    Replaces the ``C:\\Temp`` / ``C:\\Windows\\Temp`` scheme. ``APPDATA`` and
    ``LOCALAPPDATA`` were previously *inherited from the daemon*, so every tool
    that writes a cache (npm, pip, uv, pnpm) aimed at the main user's profile
    and failed on write. ``PATH`` is deliberately NOT overridden — it is what
    makes the W1 toolchain grants discoverable.
    """
    import ntpath

    root = home or windows_worker_home(environ)
    drive, tail = ntpath.splitdrive(root)
    return {
        "USERPROFILE": root,
        "HOMEDRIVE": drive or "C:",
        "HOMEPATH": tail or root,
        "APPDATA": ntpath.join(root, "AppData", "Roaming"),
        "LOCALAPPDATA": ntpath.join(root, "AppData", "Local"),
        "TEMP": ntpath.join(root, "Temp"),
        "TMP": ntpath.join(root, "Temp"),
    }


def windows_toolchain_roots(environ: dict | None = None) -> list[str]:
    """:data:`WINDOWS_TOOLCHAIN_ROOTS` with ``%VARS%`` expanded.

    Entries whose variables are unset are dropped. Existence is the caller's
    check — a grant on an absent folder is an icacls error, not a no-op.
    """
    import os as _os
    import re as _re

    env = _os.environ if environ is None else environ
    expanded: list[str] = []
    for entry in WINDOWS_TOOLCHAIN_ROOTS:
        parts = _re.split(r"%([^%]+)%", entry)
        # re.split with one group alternates: literal, varname, literal, …
        out: list[str] = []
        missing = False
        for index, part in enumerate(parts):
            if index % 2 == 0:
                out.append(part)
                continue
            value = env.get(part) or env.get(part.upper())
            if not value:
                missing = True
                break
            out.append(value)
        if not missing:
            expanded.append("".join(out))
    return expanded


def windows_protected_control_files(root: str) -> list[str]:
    """Control-file paths inside a granted Windows root (spec 077 W3).

    Only the ``.git`` entries of :data:`PROTECTED_CONTROL_FILES` apply: shell
    rc files and ``.gitconfig`` live in the user's profile on Windows, which
    the worker never gets access to in the first place.
    """
    import ntpath

    return [
        ntpath.join(root, *entry.split("/"))
        for entry in PROTECTED_CONTROL_FILES
        if entry.startswith(".git/")
    ]


def credential_read_deny_paths(home: str, platform_name: str) -> list[str]:
    """Absolute credential-deny paths for ``platform_name`` under ``home``.

    ``platform_name`` takes ``sys.platform`` values (``"darwin"``, ``"win32"``).
    Paths are joined with the *host* separator so Windows callers get
    ``C:\\Users\\me\\.aws``; they are NOT realpath'd here (the caller decides,
    because Seatbelt needs resolved paths and icacls does not).
    """
    import os as _os

    entries = list(CREDENTIAL_READ_DENY)
    if platform_name == "darwin":
        entries.extend(CREDENTIAL_READ_DENY_MACOS)
    return [_os.path.join(home, *entry.split("/")) for entry in entries]

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
    # LLM provider APIs — every provider the product's dropdown offers
    "api.openai.com",
    "api.anthropic.com",
    "generativelanguage.googleapis.com",
    "api.deepseek.com",
    "api.mistral.ai",
    "api.moonshot.cn",       # Kimi (China)
    "api.moonshot.ai",       # Kimi (intl)
    "api.minimaxi.com",      # MiniMax (China)
    "api.minimax.io",        # MiniMax (intl)
    "api.z.ai",              # Zhipu/GLM (intl)
    "open.bigmodel.cn",      # Zhipu/GLM (China)
    "dashscope-intl.aliyuncs.com",  # Qwen (intl)
    "dashscope.aliyuncs.com",       # Qwen (China)
    "api.x.ai",
    "api.groq.com",
    "api.together.xyz",
    "openrouter.ai",
    "tokendance.space",      # TokenDance router (China, Spec 47)
    "opencode.ai",           # OpenCode Zen + Go (one host serves both tiers)
    # Python
    "pypi.org",
    "files.pythonhosted.org",
    "pypi.tuna.tsinghua.edu.cn",    # Tsinghua mirror (China)
    # Node
    "registry.npmjs.org",
    "registry.yarnpkg.com",
    "registry.npmmirror.com",       # npmmirror (China)
    "cdn.npmmirror.com",            # npmmirror binaries (China)
    # Rust
    "crates.io",
    "static.crates.io",
    "index.crates.io",
    # Go
    "proxy.golang.org",
    "sum.golang.org",
    "goproxy.cn",                   # Go proxy (China)
    # General package mirrors (China) — same trust class as the registries above
    "mirrors.aliyun.com",
    "mirrors.ustc.edu.cn",
    "mirrors.cloud.tencent.com",
    # GitHub — wildcard does NOT match the apex, so both forms are needed
    "github.com",
    "*.github.com",
    "*.githubusercontent.com",
    # Hugging Face model downloads
    "huggingface.co",
    "*.huggingface.co",
    "hf.co",
    "*.hf.co",
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
