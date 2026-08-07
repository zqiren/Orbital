# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""normalize_domain + build_network_rules — the TOFU allowlist core."""

import pytest

from agent_os.daemon_v2.network_rules_builder import (
    build_network_rules,
    normalize_domain,
)
from agent_os.platform.types import DEFAULT_ALLOWLIST_DOMAINS


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("x.com", "x.com"),
        ("X.COM", "x.com"),
        ("https://x.com/some/path?q=1", "x.com"),
        ("http://api.stripe.com:443", "api.stripe.com"),
        ("  docs.python.org  ", "docs.python.org"),
    ],
)
def test_normalize_domain_accepts(raw, expected):
    assert normalize_domain(raw) == expected


@pytest.mark.parametrize(
    "raw",
    [
        "", "   ", "*.x.com", "*", "127.0.0.1", "192.168.1.10", "https://", "not a domain",
        "::1", "2001:db8::1", "[::1]:8080", "http://[::1]/x", "fe80::1",
    ],
)
def test_normalize_domain_rejects(raw):
    assert normalize_domain(raw) is None


def test_build_rules_includes_defaults_and_wildcarded_grants():
    rules = build_network_rules(["x.com"])
    assert rules.mode == "allowlist"
    for d in DEFAULT_ALLOWLIST_DOMAINS:
        assert d in rules.domains
    assert "x.com" in rules.domains
    assert "*.x.com" in rules.domains


def test_build_rules_handles_none_and_extra():
    rules = build_network_rules(None, extra=["internal.corp.example"])
    assert "internal.corp.example" in rules.domains
    assert set(DEFAULT_ALLOWLIST_DOMAINS) <= set(rules.domains)


def test_build_rules_dedupes():
    rules = build_network_rules(["pypi.org"])  # already a default
    assert rules.domains.count("pypi.org") == 1


def _registry_endpoint_hosts():
    import json
    from pathlib import Path
    from urllib.parse import urlparse

    registry = Path(__file__).resolve().parents[2] / "agent_os" / "config" / "providers.json"
    providers = json.loads(registry.read_text())["providers"]
    hosts = set()
    for entry in providers.values():
        for field in ("base_url", "china_base_url"):
            url = entry.get(field)
            if url:
                hosts.add(urlparse(url).hostname)
    return hosts


def test_defaults_cover_every_registry_provider_endpoint():
    """The allowlist claims to cover every provider the dropdown offers —
    hold it to that, both regions of every registry endpoint included."""
    missing = _registry_endpoint_hosts() - set(DEFAULT_ALLOWLIST_DOMAINS)
    assert not missing, f"provider endpoints missing from DEFAULT_ALLOWLIST_DOMAINS: {sorted(missing)}"


def test_defaults_include_china_package_mirrors():
    """Sandboxed pip/npm/go from mainland China needs the domestic mirrors."""
    for mirror in (
        "pypi.tuna.tsinghua.edu.cn",
        "registry.npmmirror.com",
        "cdn.npmmirror.com",
        "mirrors.aliyun.com",
        "mirrors.ustc.edu.cn",
        "goproxy.cn",
        "mirrors.cloud.tencent.com",
    ):
        assert mirror in DEFAULT_ALLOWLIST_DOMAINS, mirror
