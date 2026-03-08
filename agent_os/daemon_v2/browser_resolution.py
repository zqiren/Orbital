# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Browser resolution helpers — skip formatting and blocker resolution polling.

Used by the MFA relay system to:
1. Format skip results when a blocker page is abandoned
2. Poll a page for signs that a user has resolved a blocker (login, captcha, MFA)
"""

from __future__ import annotations

import asyncio


def format_skip_result(
    url: str,
    signals: dict,
    attempts: int = 0,
    redirected_from: str | None = None,
) -> str:
    """Format a skip result for agent consumption.

    Called when a blocker page (login, captcha, MFA) is skipped — either by
    user action or timeout. The agent receives this as a tool result so it
    can adjust its plan.
    """
    lines = [f"SKIPPED: {url} is inaccessible."]
    if redirected_from:
        lines.append(f"Redirect: {redirected_from} \u2192 {url}")
    signal_parts = []
    if signals.get("has_password_field"):
        signal_parts.append("password field")
    captcha = signals.get("has_captcha_iframe")
    if captcha:
        signal_parts.append(f"captcha iframe ({captcha})")
    if signal_parts:
        lines.append(f"Signals: {', '.join(signal_parts)}")
    if attempts > 0:
        lines.append(f"Attempts: {attempts} failed submissions before skip")
    lines.append("Do NOT retry this URL. Consider alternative sources if needed.")
    return "\n".join(lines)


async def poll_for_resolution(
    page,
    original_url: str,
    original_signals: dict,
    interval: float = 2.0,
    timeout: float = 120.0,
) -> bool:
    """Poll a page for signs that a blocker has been resolved.

    Returns True if the page appears to have moved past the blocker
    (URL changed, password field disappeared, or page content changed
    significantly). Returns False on timeout.

    This is a helper that can be called by agent tools — it does NOT
    auto-trigger. The agent loop or a user-action handler invokes it
    after the user signals they have completed a manual action.
    """
    elapsed = 0.0
    while elapsed < timeout:
        await asyncio.sleep(interval)
        elapsed += interval
        # Check URL change
        if page.url != original_url:
            return True
        try:
            # Check if password field disappeared
            has_pw = await page.evaluate(
                '!!document.querySelector("input[type=password]:not([hidden])")'
            )
            if original_signals.get("has_password_field") and not has_pw:
                return True
            # Check text content change
            text = await page.evaluate(
                '(document.body && document.body.innerText || "").substring(0, 200)'
            )
            if text != original_signals.get("visible_text_snippet", ""):
                return True
        except Exception:
            return True  # Page crashed/navigated = probably resolved
    return False
