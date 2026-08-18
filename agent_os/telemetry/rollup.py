# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Daily rollup: spool events → one aggregate ping (spec 046 §5).

The ping is the ONLY thing that ever leaves the machine. Counters, enums,
booleans, and ISO dates — no content, no paths, no per-project or per-session
identifiers. The schema is published in the README; the settings payload
viewer renders this exact JSON.
"""

from __future__ import annotations

from .identity import InstallIdentity
from .spool import Spool

SCHEMA_VERSION = 1

# Milestone names carried in every ping (lifetime booleans, §5).
MILESTONES = ("key_set", "first_project", "first_session", "first_turn")


def build_ping(
    identity: InstallIdentity,
    spool: Spool,
    day: str,
    version: str,
    os_name: str,
) -> dict:
    """Aggregate ``day``'s (UTC) spool rows into the §5 ping shape."""
    counters: dict = {
        "app_starts": 0,
        "projects_created": 0,
        "sessions": 0,
        "turns": 0,
        "errors": {},
        # Same failures as ``errors``, keyed by provider instead of by code
        # (spec 063 §7 P1). Both are published: ``errors`` is the only error
        # series with history, so it stays for continuity.
        "errors_by_provider": {},
        "tokens_by_provider": {},
        # Sub-agent setup funnel (spec 063 §12 decision 5): the CLI login jobs
        # emitted nothing, so the product's differentiating setup step was
        # invisible between "key set" and "first turn".
        "login_attempted": 0,
        "login_failed": 0,
    }
    for row in spool.read_day(day):
        event = row.get("event")
        if event == "app_start":
            counters["app_starts"] += 1
        elif event == "project_created":
            counters["projects_created"] += 1
        elif event == "session_created":
            counters["sessions"] += 1
        elif event == "turn_completed":
            # "turns" counts management-loop responses only (the error-rate
            # denominator); sub-agent responses still contribute tokens.
            if not str(row.get("source") or "management").startswith("subagent"):
                counters["turns"] += 1
            provider = str(row.get("provider") or "unknown")
            bucket = counters["tokens_by_provider"].setdefault(provider, {"in": 0, "out": 0})
            # The ledger's four disjoint categories: everything that entered the
            # context window counts as "in"; generated tokens as "out".
            bucket["in"] += int(row.get("uncached_input") or 0)
            bucket["in"] += int(row.get("cache_read") or 0)
            bucket["in"] += int(row.get("cache_write") or 0)
            bucket["out"] += int(row.get("output") or 0)
        elif event == "llm_error":
            code = str(row.get("error_code") or "unknown")
            counters["errors"][code] = counters["errors"].get(code, 0) + 1
            # Same "unknown" fallback as tokens above: the provider is
            # unresolved when the failure happens before config resolution.
            provider = str(row.get("provider") or "unknown")
            by_provider = counters["errors_by_provider"]
            by_provider[provider] = by_provider.get(provider, 0) + 1
        elif event == "login_attempted":
            counters["login_attempted"] += 1
        elif event == "login_failed":
            counters["login_failed"] += 1

    milestones = identity.milestones
    return {
        "schema": SCHEMA_VERSION,
        "install_id": identity.install_id,
        "account_id": None,  # nullable from day one (§8)
        "version": version,
        "os": os_name,
        "date": day,
        "first_seen": identity.first_seen,
        "milestones": {name: bool(milestones.get(name)) for name in MILESTONES},
        "counters": counters,
    }
