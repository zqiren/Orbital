# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Tests for agent_os.desktop.main._is_foreign_url — the pure helper behind the
desktop shell's origin guard (task 3, Fix B). Importing agent_os.desktop.main
must not require pywebview: the module only imports `webview` lazily inside
functions, so this pure helper is reachable headless."""

from agent_os.desktop.main import _is_foreign_url

PORT = 8000


def test_app_origin_127_right_port_is_not_foreign():
    assert _is_foreign_url(f"http://127.0.0.1:{PORT}", PORT) is False


def test_app_origin_127_wrong_port_is_foreign():
    assert _is_foreign_url("http://127.0.0.1:9000", PORT) is True


def test_app_origin_localhost_right_port_is_not_foreign():
    assert _is_foreign_url(f"http://localhost:{PORT}", PORT) is False


def test_app_origin_localhost_wrong_port_is_foreign():
    assert _is_foreign_url("http://localhost:9000", PORT) is True


def test_external_http_is_foreign():
    assert _is_foreign_url("http://example.com", PORT) is True


def test_external_https_is_foreign():
    assert _is_foreign_url("https://example.com", PORT) is True


def test_about_blank_is_not_foreign():
    assert _is_foreign_url("about:blank", PORT) is False


def test_none_is_not_foreign():
    assert _is_foreign_url(None, PORT) is False


def test_empty_string_is_not_foreign():
    assert _is_foreign_url("", PORT) is False


def test_data_url_is_not_foreign():
    assert _is_foreign_url("data:text/html,<h1>hi</h1>", PORT) is False


def test_file_url_is_not_foreign():
    assert _is_foreign_url("file:///etc/hosts", PORT) is False


def test_lookalike_host_with_port_suffix_is_foreign():
    """A malicious/odd URL like http://127.0.0.1:8000.evil.com must NOT match
    a naive startswith(f"http://127.0.0.1:{port}") check — the real origin
    (via urllib.parse) has host "127.0.0.1" but an unparseable port
    ("8000.evil.com"), so it must count as foreign rather than same-app."""
    assert _is_foreign_url(f"http://127.0.0.1:{PORT}.evil.com", PORT) is True


def test_userinfo_trick_host_is_foreign():
    """http://127.0.0.1:8000@evil.com/ parses to host evil.com — the app-origin
    lookalike lives in the userinfo section, not the authority."""
    assert _is_foreign_url(f"http://127.0.0.1:{PORT}@evil.com/", PORT) is True


def test_uppercase_scheme_and_host_normalize():
    assert _is_foreign_url("HTTP://EXAMPLE.COM/x", PORT) is True
    assert _is_foreign_url(f"HTTP://127.0.0.1:{PORT}/chat", PORT) is False


def test_ipv6_loopback_is_foreign():
    """The app only ever serves on 127.0.0.1/localhost; [::1] is not treated
    as same-app (pinning current behavior: bounce it)."""
    assert _is_foreign_url(f"http://[::1]:{PORT}/", PORT) is True


def test_out_of_range_port_is_foreign():
    """urlsplit().port raises ValueError for ports > 65535 — classified
    foreign rather than crashing the guard."""
    assert _is_foreign_url("http://127.0.0.1:99999/", PORT) is True
