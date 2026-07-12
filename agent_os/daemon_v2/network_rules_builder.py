# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Single construction point for a project's NetworkRules.

TOFU model: DEFAULT_ALLOWLIST_DOMAINS (universally-safe tooling/provider
domains) + the project's user-approved domains (each granted as apex +
``*.`` wildcard, since the proxy matcher treats them separately) + any
sub-agent manifest ``network_domains``. Enforcement is destination-based —
there is no MITM, so read/write intent can never be an enforcement input.
"""

import ipaddress
import re
from urllib.parse import urlsplit

from agent_os.platform.types import DEFAULT_ALLOWLIST_DOMAINS, NetworkRules

_DOMAIN_RE = re.compile(
    r"^(?=.{1,253}$)([a-z0-9]([a-z0-9-]{0,61}[a-z0-9])?\.)+[a-z]{2,}$"
)


def normalize_domain(raw: str) -> str | None:
    """Reduce user/agent input to a bare lowercase registrable host.

    Rejects wildcards (grants always cover subdomains implicitly), bare
    IPs (no meaningful TOFU identity), and anything that isn't a plausible
    DNS name. Returns None on rejection.
    """
    if not raw or not raw.strip():
        return None
    candidate = raw.strip().lower()
    if "//" in candidate:
        candidate = urlsplit(candidate).hostname or ""
    else:
        # strip any path/port from schemeless input like "x.com:443/a"
        candidate = candidate.split("/")[0].split(":")[0]
    if not candidate or "*" in candidate:
        return None
    try:
        ipaddress.ip_address(candidate)
        return None  # bare IPs are not grantable
    except ValueError:
        pass
    if not _DOMAIN_RE.match(candidate):
        return None
    return candidate


def build_network_rules(
    approved_domains: list[str] | None,
    extra: list[str] | None = None,
) -> NetworkRules:
    """Defaults + wildcarded per-project grants + manifest extras, deduped."""
    domains: list[str] = list(DEFAULT_ALLOWLIST_DOMAINS)
    for d in approved_domains or []:
        for entry in (d, f"*.{d}"):
            if entry not in domains:
                domains.append(entry)
    for entry in extra or []:
        if entry not in domains:
            domains.append(entry)
    return NetworkRules(mode="allowlist", domains=domains)
