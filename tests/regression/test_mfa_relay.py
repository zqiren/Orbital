# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Regression tests for MFA relay: skip-result formatting.

Covers format_skip_result() output correctness for all combinations
of signals, redirect info, and attempt counts.
"""

from __future__ import annotations

import pytest

from agent_os.daemon_v2.browser_resolution import format_skip_result


# ---------------------------------------------------------------------------
# 1. URL present in output
# ---------------------------------------------------------------------------

def test_skip_result_contains_url():
    result = format_skip_result("https://accounts.google.com/login", {})
    assert "https://accounts.google.com/login" in result


# ---------------------------------------------------------------------------
# 2. Password field and captcha signals mentioned
# ---------------------------------------------------------------------------

def test_skip_result_contains_signals():
    signals = {"has_password_field": True, "has_captcha_iframe": "recaptcha"}
    result = format_skip_result("https://example.com/login", signals)
    assert "password field" in result
    assert "recaptcha" in result


# ---------------------------------------------------------------------------
# 3. Attempt count when > 0
# ---------------------------------------------------------------------------

def test_skip_result_contains_attempts():
    result = format_skip_result("https://example.com", {}, attempts=3)
    assert "3" in result
    assert "failed submissions" in result.lower() or "attempts" in result.lower()


# ---------------------------------------------------------------------------
# 4. Redirect info when applicable
# ---------------------------------------------------------------------------

def test_skip_result_contains_redirect():
    result = format_skip_result(
        "https://accounts.google.com/login",
        {},
        redirected_from="https://mail.google.com",
    )
    assert "https://mail.google.com" in result
    assert "https://accounts.google.com/login" in result


# ---------------------------------------------------------------------------
# 5. "Do NOT retry" always present
# ---------------------------------------------------------------------------

def test_skip_result_contains_do_not_retry():
    result = format_skip_result("https://example.com", {})
    assert "Do NOT retry" in result


# ---------------------------------------------------------------------------
# 6. No redirect line when not redirected
# ---------------------------------------------------------------------------

def test_skip_result_no_redirect_when_none():
    result = format_skip_result("https://example.com", {})
    assert "Redirect" not in result


# ---------------------------------------------------------------------------
# 7. Only password signal (no captcha)
# ---------------------------------------------------------------------------

def test_skip_result_password_only():
    signals = {"has_password_field": True, "has_captcha_iframe": None}
    result = format_skip_result("https://example.com/login", signals)
    assert "password field" in result
    assert "captcha" not in result.lower()


# ---------------------------------------------------------------------------
# 8. Only captcha signal (no password)
# ---------------------------------------------------------------------------

def test_skip_result_captcha_only():
    signals = {"has_password_field": False, "has_captcha_iframe": "hcaptcha"}
    result = format_skip_result("https://example.com", signals)
    assert "hcaptcha" in result
    assert "password" not in result.lower()


# ---------------------------------------------------------------------------
# 9. Zero attempts omits attempt line
# ---------------------------------------------------------------------------

def test_skip_result_zero_attempts_omitted():
    result = format_skip_result("https://example.com", {}, attempts=0)
    assert "Attempts" not in result
