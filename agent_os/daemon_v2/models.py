# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Data models for the daemon layer.

Owned by Component F. AgentConfig, AgentStatus, ActivityEvent, detect_os.
"""

import platform
from dataclasses import dataclass, field
from enum import Enum

from agent_os.agent.prompt_builder import Autonomy


# Default chat session identifier used when callers do not supply one.
# Existing single-loop call sites pass through ``DEFAULT_SESSION_ID`` so
# the new ``(project_id, session_id)`` keying remains backward-compatible
# with code that only knows about ``project_id``.
DEFAULT_SESSION_ID: str = "default"


# A ``SessionKey`` is the composite key used by the agent and sub-agent
# managers to address a particular chat session within a project. Phase 3c
# multi-loop work introduces per-session isolation; the alias lets us refactor
# the dict keying from ``project_id`` to ``(project_id, session_id)`` without
# spreading raw tuples through the call graph.
SessionKey = tuple[str, str]


def make_session_key(project_id: str, session_id: str | None = None) -> SessionKey:
    """Build a ``SessionKey`` from a project id + optional session id.

    Passing ``None`` (or omitting ``session_id``) yields the default-session
    key, which is the back-compat target for every existing single-loop call
    site that has not yet been multi-session aware.
    """
    return (project_id, session_id or DEFAULT_SESSION_ID)


class AgentStatus(str, Enum):
    RUNNING = "running"
    IDLE = "idle"
    PAUSED = "paused"
    STOPPED = "stopped"
    ERROR = "error"


# Default Format-1 chat session id used when an API caller does not pass one
# explicitly on /agents/start or /inject. See
# ``TASK/ACTIVE-session-and-queue-model.md`` and the F7 audit for the F1/F2
# split. Omitting ``session_id`` on the request means "the only session, use
# the default" — which mirrors the historical single-session-per-project
# behaviour and keeps existing API clients working unchanged.
DEFAULT_SESSION_ID: str = "default"


@dataclass
class ActivityEvent:
    id: str
    project_id: str
    category: str       # "file_read" | "file_write" | "command_exec" | "file_edit" | "web_search" | "tool_result"
    description: str
    tool_name: str
    source: str         # "management" | "{agent_handle}"
    timestamp: str


def detect_os() -> str:
    """Detect OS type for shell commands and prompt builder."""
    system = platform.system().lower()
    if system == "darwin":
        return "macos"
    if system == "windows":
        return "windows"
    return "linux"


@dataclass
class DaemonConfig:
    """Global daemon configuration including budget limits."""
    global_monthly_limit_usd: float | None = None
    global_monthly_action: str = "ask"  # "ask" or "stop"
    global_monthly_spent_usd: float = 0.0


@dataclass
class FallbackModelEntry:
    """Single fallback model configuration for provider rotation."""
    provider: str = "custom"
    model: str = ""
    base_url: str | None = None
    api_key: str = ""
    sdk: str = "openai"


@dataclass
class AgentConfig:
    workspace: str
    model: str
    api_key: str
    base_url: str | None = None
    max_iterations: int = 0
    token_budget: int = 100_000_000
    utility_model: str | None = None
    search_api_key: str | None = None
    autonomy: Autonomy = Autonomy.HANDS_OFF
    enabled_agents: list[str] = field(default_factory=list)
    agent_slug: str = "built-in"
    enabled_sub_agents: list[str] = field(default_factory=list)
    # Denylist of sub-agent slugs to exclude from this project. New canonical
    # source of truth as of the sub-agent settings rework. ``enabled_sub_agents``
    # remains for backward compatibility but is treated as informational only.
    disabled_sub_agents: list[str] = field(default_factory=list)
    agent_credentials: dict = field(default_factory=dict)
    network_extra_domains: list[str] = field(default_factory=list)
    project_name: str = ""
    project_instructions: str = ""
    budget_limit_usd: float | None = None
    budget_action: str = "ask"  # "ask" or "stop"
    budget_spent_usd: float = 0.0
    sdk: str = "openai"        # "openai" or "anthropic"
    provider: str = "custom"   # provider key from providers.json
    is_scratch: bool = False
    agent_name: str = ""
    global_preferences_path: str = ""
    llm_fallback_models: list[FallbackModelEntry] = field(default_factory=list)


def resolve_api_key(project_config: dict) -> str:
    """Centralize API key resolution from project config.

    Single point for future BYOK/bundled/platform key sources.
    Currently returns the project's api_key field directly.
    """
    return project_config.get("api_key", "")
