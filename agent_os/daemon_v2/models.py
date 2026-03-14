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


class AgentStatus(str, Enum):
    RUNNING = "running"
    IDLE = "idle"
    PAUSED = "paused"
    STOPPED = "stopped"
    ERROR = "error"


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
