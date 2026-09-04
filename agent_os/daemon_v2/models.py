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
    """One rung of the provider rotation, RESOLVED from a credential card.

    Spec 082: what is stored (settings.json, projects.json, the API) is only
    ``{card_id}``; the config builder looks the card up and fills the rest in
    here. Keeping the resolved fields on the dataclass is what lets
    ``_build_llm_providers`` and the loop stay byte-identical — they still
    receive a fully-specified rung and never learn what a card is.
    """
    provider: str = "custom"
    model: str = ""
    base_url: str | None = None
    api_key: str = ""
    sdk: str = "openai"
    # The card this rung was resolved from ("" for a rung with no card, e.g.
    # a legacy entry or a test-constructed one).
    card_id: str = ""


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
    # Per-project connector enablement (Spec 011 §2): ids of catalog/custom
    # connectors whose remote MCP tools reflect into this project's registry.
    # Authenticate globally (Global Settings), enable per project.
    enabled_connectors: list[str] = field(default_factory=list)
    agent_credentials: dict = field(default_factory=dict)
    # TOFU network grants (Plan 2). approved_domains: bare registrable
    # domains the user approved for THIS project (wildcarded at rules-build
    # time). pending_domain_requests: asks that auto-denied in hands-off,
    # awaiting a Settings decision — entries {domain, reason, requested_at}.
    approved_domains: list[str] = field(default_factory=list)
    pending_domain_requests: list[dict] = field(default_factory=list)
    project_name: str = ""
    project_instructions: str = ""
    sub_agent_deployment_instructions: str = ""
    budget_limit_usd: float | None = None
    budget_action: str = "pause"  # normalized to "pause" | "stop" by guard.normalize_budget_action; not read by enforcement (the guard reads live config)
    budget_spent_usd: float = 0.0
    sdk: str = "openai"        # "openai" or "anthropic"
    provider: str = "custom"   # provider key from providers.json
    # Spec 082 — the credential card this config resolved from. Everything
    # above it (model/api_key/base_url/sdk/provider) is that card's content,
    # looked up once by _build_agent_config_from_project. "" means no card
    # resolved (a project on a fresh install with nothing configured).
    card_id: str = ""
    card_name: str = ""
    # Spec 072 — auth-fallback rung: a wholesale snapshot of the GLOBAL default
    # (provider/model/key/base_url/sdk), attached by the config builder only
    # when it is materially usable AND its key differs from the resolved
    # primary key. NOT part of llm_fallback_models: the loop appends it to the
    # rotation only after a 401/403 proves the project key dead, so transient
    # rotation semantics stay untouched. Never persisted — re-derived per run.
    auth_fallback: FallbackModelEntry | None = None
    is_scratch: bool = False
    agent_name: str = ""
    global_preferences_path: str = ""
    # Spec 073 — user-level memory file (~/orbital/user_memory.md sibling of
    # user_preferences.md). "" means the feature is toggled off: the config
    # builder leaves it empty when user_memory_enabled is False, which both
    # unregisters the remember_about_user tool and omits the prompt section.
    user_memory_path: str = ""
    llm_fallback_models: list[FallbackModelEntry] = field(default_factory=list)
    # Budget Piece 1, Task 4 — derived-cost window config. Additive; the legacy
    # budget fields above (budget_limit_usd / budget_action / budget_spent_usd)
    # are unchanged and stay. Codes/enums/ISO codes only — no display strings.
    budget_period: str = "daily"  # "daily" | "weekly" | "monthly" | "total"
    budget_anchor_ts: str | None = None  # ISO8601; lower bound for the "total" window
    budget_currency: str | None = None  # ISO 4217; the LIMIT owns its currency

# ``resolve_api_key`` is GONE (spec 082 §3.4). There is no longer a per-project
# key to resolve: a project references a credential card, and the card's key is
# read once by the config builder through ``SettingsStore.key_for``.
