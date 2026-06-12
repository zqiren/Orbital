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


# Seam 3 / decision D1: the "default" routing sentinel is retired. The canonical
# session identity is the uuid; the sentinel for "no session specified" is None.
# Resolution of None is CALLER-CLASS specific (read/affect → holder; inject →
# persistent chat session; sub-agent → raise), so it is decided at the call site
# via ``resolve_session_id``'s on_none policy, NOT by a global default.

# A ``SessionKey`` is the composite key used by the agent and sub-agent
# managers to address a particular chat session within a project. The second
# element may be None (no session resolved) — a key that intentionally matches
# no live handle, so callers degrade to handle-miss / no-op rather than routing
# to a phantom "default" session.
SessionKey = tuple[str, str | None]


def make_session_key(project_id: str, session_id: str | None = None) -> SessionKey:
    """Build a ``SessionKey`` from a project id + optional session id.

    Passing ``None`` yields ``(project_id, None)`` — a key that matches no live
    handle. Callers that must resolve None to a concrete session do so BEFORE
    keying, via ``resolve_session_id`` with the appropriate on_none policy.
    """
    return (project_id, session_id)


def resolve_session_id(session_id, *, on_none):
    """Shared session-id resolution (seam 3 / D1).

    The COMMON rule — identical for every caller class — is: a provided
    (non-None) ``session_id`` passes through unchanged. The ONLY per-caller
    variation is the None-policy callback ``on_none``:

      - read/affect paths pass ``on_none=lambda: current_holder_session_id(pid)``
        (None → holder, or None again → handle-miss → idle/no-op; never mints);
      - the inject path passes a policy that resolves the project's persistent
        chat session (holder if draining, else lazy-mint via the single funnel);
      - the sub-agent resolver passes a raising policy (None is a real bug).

    Centralizing the common rule here means the call sites differ ONLY in
    on_none — there is no second hand-maintained "default" rule to drift.
    """
    if session_id is not None:
        return session_id
    return on_none()


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
    # Denylist of sub-agent slugs to exclude from this project. New canonical
    # source of truth as of the sub-agent settings rework. ``enabled_sub_agents``
    # remains for backward compatibility but is treated as informational only.
    disabled_sub_agents: list[str] = field(default_factory=list)
    agent_credentials: dict = field(default_factory=dict)
    # network_extra_domains REMOVED (TASK-network-config-cleanup): persisted but
    # never plumbed into NetworkRules — the allowlist is internal/hardcoded
    # (DEFAULT_ALLOWLIST_DOMAINS + per-agent manifest network_domains).
    project_name: str = ""
    project_instructions: str = ""
    budget_limit_usd: float | None = None
    budget_action: str = "pause"  # normalized to "pause" | "stop" by guard.normalize_budget_action; not read by enforcement (the guard reads live config)
    budget_spent_usd: float = 0.0
    sdk: str = "openai"        # "openai" or "anthropic"
    provider: str = "custom"   # provider key from providers.json
    is_scratch: bool = False
    agent_name: str = ""
    global_preferences_path: str = ""
    llm_fallback_models: list[FallbackModelEntry] = field(default_factory=list)
    # Budget Piece 1, Task 4 — derived-cost window config. Additive; the legacy
    # budget fields above (budget_limit_usd / budget_action / budget_spent_usd)
    # are unchanged and stay. Codes/enums/ISO codes only — no display strings.
    budget_period: str = "daily"  # "daily" | "weekly" | "monthly" | "total"
    budget_anchor_ts: str | None = None  # ISO8601; lower bound for the "total" window
    budget_currency: str | None = None  # ISO 4217; the LIMIT owns its currency


def resolve_api_key(project_config: dict) -> str:
    """Centralize API key resolution from project config.

    Single point for future BYOK/bundled/platform key sources.
    Currently returns the project's api_key field directly.
    """
    return project_config.get("api_key", "")
