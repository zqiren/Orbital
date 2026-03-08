# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Unit tests for agent_os.agent.tools.browser_safety."""

from unittest.mock import patch

import pytest

from agent_os.agent.tools.browser_safety import (
    detect_secrets,
    substitute_secrets,
    validate_url_post_navigation,
    validate_url_pre_navigation,
    wrap_untrusted_content,
)

# ---------------------------------------------------------------------------
# URL pre-navigation validation
# ---------------------------------------------------------------------------


def test_valid_http_url():
    assert validate_url_pre_navigation("https://example.com") is None


def test_valid_http_with_path():
    assert validate_url_pre_navigation("https://example.com/page?q=1") is None


def test_block_file_scheme():
    result = validate_url_pre_navigation("file:///etc/passwd")
    assert result is not None
    assert "only http/https" in result


def test_block_javascript_scheme():
    result = validate_url_pre_navigation("javascript:alert(1)")
    assert result is not None


def test_block_data_scheme():
    result = validate_url_pre_navigation("data:text/html,<h1>hi</h1>")
    assert result is not None


def test_block_blob_scheme():
    result = validate_url_pre_navigation("blob:http://example.com/abc")
    assert result is not None


def test_block_private_ip_127():
    result = validate_url_pre_navigation("http://127.0.0.1")
    assert result is not None
    assert "private" in result.lower()


def test_block_private_ip_10():
    result = validate_url_pre_navigation("http://10.0.0.1")
    assert result is not None
    assert "private" in result.lower()


def test_block_private_ip_172():
    result = validate_url_pre_navigation("http://172.16.0.1")
    assert result is not None
    assert "private" in result.lower()


def test_block_private_ip_192():
    result = validate_url_pre_navigation("http://192.168.1.1")
    assert result is not None
    assert "private" in result.lower()


def test_block_ipv6_localhost():
    result = validate_url_pre_navigation("http://[::1]")
    assert result is not None
    assert "private" in result.lower()


def test_block_dns_resolving_private():
    """Hostname that resolves to a private IP should be blocked."""
    fake_addrinfo = [(2, 1, 6, "", ("127.0.0.1", 0))]
    with patch("agent_os.agent.tools.browser_safety.socket.getaddrinfo", return_value=fake_addrinfo):
        result = validate_url_pre_navigation("http://evil.example.com")
    assert result is not None
    assert "private" in result.lower()


def test_allow_dns_resolving_public():
    """Hostname that resolves to a public IP should be allowed."""
    fake_addrinfo = [(2, 1, 6, "", ("93.184.216.34", 0))]
    with patch("agent_os.agent.tools.browser_safety.socket.getaddrinfo", return_value=fake_addrinfo):
        result = validate_url_pre_navigation("http://example.com")
    assert result is None


def test_invalid_url():
    result = validate_url_pre_navigation("not a url")
    assert result is not None


# ---------------------------------------------------------------------------
# URL post-navigation validation
# ---------------------------------------------------------------------------


def test_post_nav_blocks_redirected_private():
    result = validate_url_post_navigation("http://192.168.1.1/admin")
    assert result is not None
    assert "redirected" in result.lower()


def test_post_nav_allows_public():
    assert validate_url_post_navigation("https://example.com") is None


# ---------------------------------------------------------------------------
# Untrusted content wrapping
# ---------------------------------------------------------------------------


def test_wrap_untrusted():
    wrapped = wrap_untrusted_content("hello", "https://example.com")
    assert wrapped.startswith("[BROWSER CONTENT")
    assert "UNTRUSTED" in wrapped
    assert "https://example.com" in wrapped
    assert wrapped.endswith("[/BROWSER CONTENT]")


def test_wrap_preserves_content():
    content = "Some <b>html</b> content\nwith newlines"
    wrapped = wrap_untrusted_content(content, "https://example.com")
    assert content in wrapped


# ---------------------------------------------------------------------------
# Secret detection and substitution
# ---------------------------------------------------------------------------


def test_detect_secrets():
    text = "Login with <secret:username> and <secret:password>"
    keys = detect_secrets(text)
    assert keys == ["username", "password"]


def test_detect_secrets_dotted():
    text = "Login with <secret:gmail.email> and <secret:gmail.password>"
    keys = detect_secrets(text)
    assert keys == ["gmail.email", "gmail.password"]


def test_detect_no_secrets():
    assert detect_secrets("plain text with no secrets") == []


def test_substitute_secrets():
    text = "pw is <secret:gmail.password>"
    resolver = lambda key: "abc" if key == "gmail.password" else None
    result = substitute_secrets(text, resolver)
    assert result == "pw is abc"


def test_substitute_missing_secret():
    resolver = lambda key: None
    with pytest.raises(ValueError, match="not found in credential store"):
        substitute_secrets("<secret:missing>", resolver)


def test_substitute_multiple():
    text = "user=<secret:site.user> pass=<secret:site.pass>"
    store = {"site.user": "alice", "site.pass": "s3cret"}
    resolver = lambda key: store.get(key)
    result = substitute_secrets(text, resolver)
    assert result == "user=alice pass=s3cret"


def test_secret_pattern_edge_cases():
    # Valid pattern with underscores and numbers
    assert detect_secrets("<secret:a_b_1>") == ["a_b_1"]
    # Dotted name.field format
    assert detect_secrets("<secret:gmail.email>") == ["gmail.email"]
    # Empty key does not match
    assert detect_secrets("<secret:>") == []
    # Spaces in key do not match
    assert detect_secrets("<secret:has space>") == []
